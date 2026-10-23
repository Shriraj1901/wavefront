"""Streaming a reply's text to the wire.

Only text-only calls are streamed (no ``functions``): no provider's ``stream()``
surfaces tool calls, so a round that may call a tool must stay on ``generate``.

``stream_reply`` forwards the provider's chunks as ``content_delta`` events, with
three guarantees:

- chunks are coalesced (``DeltaCoalescer``), so a token-by-token provider does not
  become one SSE frame per token;
- a guardrail control chunk becomes a ``retract`` event, as in
  ``agent_stream_events.StreamingTapLLM``, and is never shown as text;
- however the generator ends (finished, timed out, failed, cancelled, closed by the
  consumer) the provider stream is closed, and nothing is yielded after a cancel.

A provider whose ``stream()`` fails before any text arrived degrades to the
non-streamed ``generate`` call, which is what the assistant did before streaming.
"""

import asyncio
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
)

from common_module.log.logger import logger

from agents_module.models.workflow_assistant_schemas import (
    content_delta,
    retract_event,
)
from agents_module.services.agent_stream_events import GUARDRAIL_CONTROL_KEY
from agents_module.utils.agent_guardrails import declare_retract_support

try:
    from flo_ai.llm.guarded_llm import GuardrailBlocked
except ImportError:
    # Guardrails is an optional extra; see agent_inference_service.
    class GuardrailBlocked(Exception):  # type: ignore[no-redef]
        retract = False


#: A delta is sent at most this often, or once this many characters are waiting
FLUSH_INTERVAL_SECONDS = 0.04
FLUSH_MAX_CHARS = 80


class DeltaCoalescer:
    """Batches small text chunks. Pure: the clock is injectable for tests.

    ``add`` returns the text to send now, or None while it is still batching. The
    first chunk is sent at once so the reply starts as early as it can.
    """

    def __init__(
        self,
        interval: float = FLUSH_INTERVAL_SECONDS,
        max_chars: int = FLUSH_MAX_CHARS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval = interval
        self.max_chars = max_chars
        self._clock = clock
        self._parts: List[str] = []
        self._size = 0
        self._last_flush: Optional[float] = None

    def add(self, text: str) -> Optional[str]:
        self._parts.append(text)
        self._size += len(text)
        now = self._clock()
        if (
            self._size >= self.max_chars
            or self._last_flush is None
            or now - self._last_flush >= self.interval
        ):
            return self._take(now)
        return None

    def flush(self) -> Optional[str]:
        """Whatever is waiting, or None when nothing is."""
        return self._take(self._clock()) if self._parts else None

    def discard(self) -> None:
        """Drop what is waiting (a retract replaces it)."""
        self._parts = []
        self._size = 0

    def seconds_until_due(self) -> Optional[float]:
        """How long until waiting text is due, or None when nothing is waiting."""
        if not self._parts:
            return None
        if self._last_flush is None:
            return 0.0
        return max(0.0, self._last_flush + self.interval - self._clock())

    def _take(self, now: float) -> str:
        text = ''.join(self._parts)
        self._parts = []
        self._size = 0
        self._last_flush = now
        return text


@dataclass
class StreamedReply:
    """What ``stream_reply`` learned, read by the caller once it is exhausted."""

    #: The reply as the client now sees it (a retract's replacement counts)
    text: str = ''
    #: Any content_delta or retract event was yielded
    emitted: bool = False
    #: The deadline passed; ``text`` is what arrived before it
    timed_out: bool = False
    #: The stream failed after some text was sent; ``text`` is what was sent
    interrupted: bool = False
    #: Set when the reply came from the non-streamed fallback (usage, finish reason)
    response: Any = None


async def _close(pending: Optional['asyncio.Future[Any]'], iterator: Any) -> None:
    """Stop the in-flight read, then close the provider's stream."""
    if pending is not None:
        if not pending.done():
            pending.cancel()
        try:
            await pending
        except BaseException:  # noqa: BLE001 - cancelled or failed; both are moot now
            pass
    aclose = getattr(iterator, 'aclose', None)
    if aclose is not None:
        try:
            await aclose()
        except Exception as exc:
            logger.debug(
                f'Workflow assistant: closing the stream failed ({type(exc).__name__})'
            )


async def _generate_reply(
    llm: Any,
    messages: List[Dict[str, Any]],
    functions: Optional[List[Dict[str, Any]]],
    reply: StreamedReply,
    deadline: float,
    scope: Callable[[], ContextManager[Any]],
) -> AsyncIterator[Dict[str, Any]]:
    """The non-streamed call: one delta carrying the whole reply."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        reply.timed_out = True
        return

    async def call() -> Any:
        with scope():
            return await llm.generate(messages, functions=functions)

    try:
        response = await asyncio.wait_for(call(), timeout=remaining)
    except asyncio.TimeoutError:
        reply.timed_out = True
        return
    text = llm.get_message_content(response) or ''
    reply.text = text
    reply.response = response
    if text:
        reply.emitted = True
        yield content_delta(text)


async def stream_reply(
    llm: Any,
    messages: List[Dict[str, Any]],
    reply: StreamedReply,
    *,
    deadline: float,
    scope: Callable[[], ContextManager[Any]] = nullcontext,
    fallback_functions: Optional[List[Dict[str, Any]]] = None,
    coalescer: Optional[DeltaCoalescer] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield the reply's wire events; ``reply`` is filled in as they go.

    ``deadline`` is a ``time.monotonic()`` value. ``scope`` is entered where each
    read from the provider is started: a guardrail run scope has to be bound in
    the task that makes the call. ``fallback_functions`` are passed to ``generate``
    only if streaming degrades to it; the stream itself never carries functions.

    A ``GuardrailBlocked`` propagates (the caller turns it into a retract and an
    error). Any other failure after text was sent ends the reply as interrupted.
    """
    if getattr(llm, 'stream', None) is None:
        async for event in _generate_reply(
            llm, messages, fallback_functions, reply, deadline, scope
        ):
            yield event
        return

    # The consumer is the assistant's own client, which handles `retract`
    declare_retract_support(llm)

    coalescer = coalescer or DeltaCoalescer()
    parts: List[str] = []
    iterator: Any = None
    pending: Optional['asyncio.Future[Any]'] = None
    failed = False

    try:
        iterator = llm.stream(messages).__aiter__()
        while True:
            if pending is None:
                with scope():
                    pending = asyncio.ensure_future(iterator.__anext__())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reply.timed_out = True
                break
            due = coalescer.seconds_until_due()
            wait = remaining if due is None else min(remaining, due)
            done, _ = await asyncio.wait({pending}, timeout=wait)
            if not done:
                # A quiet provider must not hold back text that is already due
                out = coalescer.flush()
                if out:
                    reply.emitted = True
                    yield content_delta(out)
                continue

            finished, pending = pending, None
            try:
                chunk = finished.result()
            except StopAsyncIteration:
                break

            control = chunk.get(GUARDRAIL_CONTROL_KEY) if chunk else None
            if control is not None:
                if control.get('action') != 'retract':
                    logger.warning(
                        'Ignoring unrecognised guardrail control action '
                        f'{control.get("action")!r} on the assistant stream'
                    )
                    continue
                # The replacement is the whole of what may now be shown
                replacement = control.get('replacement') or ''
                coalescer.discard()
                parts = [replacement]
                reply.emitted = True
                yield retract_event(replacement, control.get('message'))
                continue

            content = chunk.get('content') if chunk else None
            if not content:
                continue
            parts.append(content)
            out = coalescer.add(content)
            if out:
                reply.emitted = True
                yield content_delta(out)
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except GuardrailBlocked:
        raise
    except Exception as exc:
        logger.warning(f'Workflow assistant: stream failed ({type(exc).__name__})')
        failed = True
    finally:
        await _close(pending, iterator)

    out = coalescer.flush()
    if out:
        reply.emitted = True
        yield content_delta(out)
    reply.text = ''.join(parts)

    if failed:
        if reply.text or reply.emitted:
            reply.interrupted = True
        else:
            # Nothing reached the client: degrade to today's non-streamed call
            async for event in _generate_reply(
                llm, messages, fallback_functions, reply, deadline, scope
            ):
                yield event
