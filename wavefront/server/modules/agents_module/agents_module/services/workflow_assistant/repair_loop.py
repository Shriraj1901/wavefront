"""The bounded tool loop: call the model, run its tools, feed errors back.

A thin direct loop over flo_ai's ``BaseLLM`` rather than a flo_ai ``Agent``: the
Agent hides its tool-call bound, adds LLM calls of its own to every reply and
reports tools without their arguments or results. Here every bound is explicit:

- ``max_rounds``: LLM calls in the turn that may call tools. One more call is
  allowed after a clean proposal, so the model can write its reply
- ``max_repair_rounds``: proposal attempts (failed calls included) before giving
  up, and an early exit as soon as a proposal fails with exactly the introduced
  errors the previous one did. Only errors the change introduced count: problems
  the workflow already had never keep the loop going
- a proposal with no introduced errors ends the loop after one reply round
- ``max_total_tokens``: provider-reported usage, or a chars/4 estimate
- ``timeout_seconds``: wall clock for the whole turn
"""

import asyncio
import json
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
)

from common_module.log.logger import logger

from agents_module.models.workflow_assistant_schemas import (
    content_delta,
    tool_called,
    tool_failed,
    tool_result,
)
from agents_module.services.workflow_assistant.limits import AssistantLimits
from agents_module.services.workflow_assistant.prompt import fence_untrusted
from agents_module.services.workflow_assistant.streaming import (
    StreamedReply,
    stream_reply,
)
from agents_module.services.workflow_assistant.tools.base import (
    Proposal,
    ToolContext,
    ToolDispatchError,
)
from agents_module.services.workflow_assistant.tools.registry import ToolRegistry

COMPLETED = 'completed'
MAX_ROUNDS = 'max_rounds'
TOKEN_BUDGET = 'token_budget'
TIMEOUT = 'timeout'
REPAIR_EXHAUSTED = 'repair_exhausted'
REPEATED_ERRORS = 'repeated_errors'


@dataclass
class LoopStats:
    """Filled in as the loop runs; read by the service after it ends."""

    rounds: int = 0
    tokens: int = 0
    tools_called: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    stop_reason: str = COMPLETED
    #: The provider cut the last LLM response off at its output limit
    truncated: bool = False
    #: The message of the last tool call that could not run, for a plain-words
    #: explanation when nothing at all could be proposed
    last_error: Optional[str] = None
    #: The streamed reply broke off after some text had been sent
    interrupted: bool = False
    #: 'ask' or 'edit': how the turn was routed (set by the service)
    intent: str = 'edit'


class RepairTracker:
    """Decides when validate-and-repair should stop.

    Only INTRODUCED errors (see Proposal.introduced) are repaired; every proposal
    attempt counts against the cap, so the cap is exact.
    """

    def __init__(self, max_repair_rounds: int) -> None:
        self.max_repair_rounds = max_repair_rounds
        self.validations = 0
        self._previous: Optional[FrozenSet[Tuple[str, str]]] = None

    def record(self, proposal: Proposal) -> Optional[str]:
        """Feed one validated proposal; a stop reason when the loop should end.

        None for a proposal with no introduced errors too: the loop's caller ends
        the turn on those itself, after the model's reply.
        """
        self.validations += 1
        introduced = proposal.introduced
        if not introduced:
            self._previous = None
            return None
        errors = frozenset((e.path, e.message) for e in introduced)
        repeated = errors == self._previous
        self._previous = errors
        if repeated:
            return REPEATED_ERRORS
        if self.validations >= self.max_repair_rounds:
            return REPAIR_EXHAUSTED
        return None

    def record_failure(self) -> Optional[str]:
        """A proposal call that could not run (bad arguments, a crash) still uses
        up an attempt."""
        self.validations += 1
        self._previous = None
        if self.validations >= self.max_repair_rounds:
            return REPAIR_EXHAUSTED
        return None


def response_tokens(response: Any, messages: List[Dict[str, Any]], text: str) -> int:
    """Tokens one LLM call used: what the provider reported, else an estimate."""
    usage = getattr(response, 'usage', None)
    if usage is None and isinstance(response, dict):
        usage = response.get('usage')
    if usage is not None:

        def read(name: str) -> Optional[int]:
            value = (
                usage.get(name)
                if isinstance(usage, dict)
                else getattr(usage, name, None)
            )
            return value if isinstance(value, int) else None

        total = read('total_tokens')
        if total is None:
            parts = [
                read(n)
                for n in (
                    'input_tokens',
                    'output_tokens',
                    'prompt_tokens',
                    'completion_tokens',
                )
            ]
            total = sum(p for p in parts if p) or None
        if total:
            return total
    return (len(json.dumps(messages, default=str)) + len(text)) // 4


#: finish_reason / stop_reason values (OpenAI, Anthropic, Gemini and compatible
#: servers) that mean "stopped at the output token limit"
_TRUNCATION_REASONS = frozenset(
    {
        'length',
        'max_tokens',
        'max_output_tokens',
        'max_completion_tokens',
        'model_length',
    }
)


def response_truncated(response: Any) -> bool:
    """True when the provider reports the response was cut off at its token limit.

    Reads ``finish_reason`` / ``stop_reason`` when the response carries one. flo_ai's
    clients do not pass either through today (they return the message or a bare
    dict), so this is False until they do; a cut-off tool call is caught separately
    by its invalid JSON arguments.
    """
    for name in ('finish_reason', 'stop_reason'):
        value = (
            response.get(name)
            if isinstance(response, dict)
            else getattr(response, name, None)
        )
        if value is None:
            continue
        label = getattr(value, 'name', None) or str(value)
        if str(label).lower().rsplit('.', 1)[-1] in _TRUNCATION_REASONS:
            return True
    return False


FINAL_REPLY_INSTRUCTION = (
    'The workflow below was validated and accepted as your proposal; it is delivered '
    'to the user separately. Reply now with a short plain-language summary of what '
    'you changed and anything the user still has to set up. Do not call any tool and '
    'do not paste the YAML.'
)


def final_reply_messages(
    base: List[Dict[str, Any]],
    proposal: Optional[Proposal],
    nonce: str,
    limits: AssistantLimits,
) -> List[Dict[str, Any]]:
    """The transcript for the streamed reply after an accepted proposal.

    The tool-call turns are left out: a streaming call carries no tools, and
    providers do not all accept tool-call messages without them. The accepted
    proposal is handed over in the last user message instead, so the model still
    knows what it changed. ``base`` is not modified.
    """
    messages = [dict(m) for m in base]
    lines = [FINAL_REPLY_INSTRUCTION]
    if proposal is not None:
        lines.append(
            fence_untrusted(
                'ACCEPTED_PROPOSAL',
                proposal.yaml[: limits.max_tool_output_chars],
                nonce,
            )
        )
        if proposal.notes:
            lines.append('Notes you gave with it: ' + ' '.join(proposal.notes))
        if proposal.existing:
            lines.append(
                f'{len(proposal.existing)} validation issue(s) were already in the '
                'workflow before your change and remain; mention that briefly.'
            )
    suffix = '\n\n'.join(lines)
    if (
        messages
        and messages[-1].get('role') == 'user'
        and isinstance(messages[-1].get('content'), str)
    ):
        messages[-1]['content'] = f'{messages[-1]["content"]}\n\n{suffix}'
    else:
        messages.append({'role': 'user', 'content': suffix})
    return messages


TRUNCATED_CALL_MESSAGE = (
    'Your last tool call was cut off (its arguments were not valid JSON, usually '
    'because it was too long). Send a shorter call: keep new prompts brief and '
    'do not repeat long unchanged text.'
)


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolDispatchError(TRUNCATED_CALL_MESSAGE) from exc
    return raw if raw is not None else {}


async def run_loop(
    llm: Any,
    messages: List[Dict[str, Any]],
    registry: ToolRegistry,
    ctx: ToolContext,
    limits: AssistantLimits,
    nonce: str,
    stats: LoopStats,
    generate_scope: Callable[[], ContextManager[Any]] = nullcontext,
    deadline: Optional[float] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Run the turn, yielding wire events. ``stats.stop_reason`` says why it ended.

    ``generate_scope`` is entered around each LLM call, inside the task that makes
    it, which is where a guardrail run scope has to be bound. ``deadline`` is a
    ``time.monotonic()`` value; by default the turn's budget starts now.

    Rounds that may call tools use ``generate``. Only the reply after an accepted
    proposal, which cannot call anything, is streamed.
    """

    async def generate() -> Any:
        with generate_scope():
            return await llm.generate(messages, functions=functions)

    functions = registry.schemas(llm)
    tracker = RepairTracker(limits.max_repair_rounds)
    if deadline is None:
        deadline = time.monotonic() + limits.timeout_seconds
    base_messages = [dict(m) for m in messages]

    finalizing = False  # a clean proposal is in; the next call is the reply

    for round_index in range(limits.max_rounds + 1):
        if round_index >= limits.max_rounds and not finalizing:
            stats.stop_reason = MAX_ROUNDS
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stats.stop_reason = TIMEOUT
            return
        if stats.tokens >= limits.max_total_tokens:
            stats.stop_reason = TOKEN_BUDGET
            return

        already_sent = False  # the reply text is already on the wire as deltas
        if finalizing:
            reply = StreamedReply()
            final_messages = final_reply_messages(
                base_messages, ctx.last_proposal, nonce, limits
            )
            async for event in stream_reply(
                llm,
                final_messages,
                reply,
                deadline=deadline,
                scope=generate_scope,
                fallback_functions=functions,
            ):
                yield event
            if reply.response is None:
                # Streamed, or out of time before any call: no provider usage to read
                stats.rounds += 1
                stats.truncated = False
                stats.interrupted = reply.interrupted
                stats.tokens += response_tokens({}, final_messages, reply.text)
                if reply.text:
                    stats.texts.append(reply.text)
                stats.stop_reason = TIMEOUT if reply.timed_out else COMPLETED
                return
            # The provider could not stream; the non-streamed call answered, and
            # stream_reply has already sent its text
            response = reply.response
            already_sent = True
        else:
            try:
                response = await asyncio.wait_for(generate(), timeout=remaining)
            except asyncio.TimeoutError:
                stats.stop_reason = TIMEOUT
                return
        stats.rounds += 1
        stats.truncated = response_truncated(response)

        text = llm.get_message_content(response) or ''
        stats.tokens += response_tokens(response, messages, text)
        if text:
            stats.texts.append(text)
            if not already_sent:
                yield content_delta(text)

        function_call = await llm.get_function_call(response)
        if finalizing:
            # The proposal is accepted. Whatever else the model asked for, the
            # turn ends here with the text it wrote.
            stats.stop_reason = COMPLETED
            return
        if not function_call:
            stats.stop_reason = COMPLETED
            return

        tool_name = function_call.get('name') or ''
        tool_use_id = llm.get_tool_use_id(function_call)
        call_id = tool_use_id or uuid.uuid4().hex

        call_message = llm.format_tool_call_message(response, function_call)
        if call_message:
            messages.append(call_message)
        else:
            content = llm.get_assistant_message_for_tool_call(response)
            if content:
                messages.append({'role': 'assistant', 'content': content})

        tool = registry.get(tool_name)
        yield tool_called(call_id, tool_name, tool.label if tool else 'Calling a tool')
        stats.tools_called.append(tool_name)

        proposals_before = len(ctx.proposals)
        args: Any = None
        try:
            args = _parse_arguments(function_call.get('arguments'))
            _, result = await asyncio.wait_for(
                registry.dispatch(tool_name, args, ctx),
                timeout=max(deadline - time.monotonic(), 0.001),
            )
        except asyncio.TimeoutError:
            stats.stop_reason = TIMEOUT
            return
        except ToolDispatchError as exc:
            if exc.__cause__ is not None:
                logger.warning(
                    f'Workflow assistant tool {tool_name} crashed: '
                    f'{type(exc.__cause__).__name__}'
                )
            stats.last_error = str(exc)
            yield tool_failed(call_id, tool_name, str(exc))
            messages.append(
                llm.format_function_result_message(
                    tool_name, f'Error: {exc}', tool_use_id
                )
            )
            if tool is not None and tool.is_proposal:
                # Keep the YAML the model sent when the call crashed after it
                # parsed, so the turn does not end with nothing to show
                tool.capture_failed_call(args, ctx, str(exc))
                reason = tracker.record_failure()
                if reason:
                    stats.stop_reason = reason
                    return
            continue

        yield tool_result(call_id, tool_name, result.ok, result.summary)
        output = result.content[: limits.max_tool_output_chars]
        messages.append(
            llm.format_function_result_message(
                tool_name,
                fence_untrusted(f'TOOL_OUTPUT {tool_name}', output, nonce),
                tool_use_id,
            )
        )

        if len(ctx.proposals) > proposals_before:
            latest = ctx.proposals[-1]
            reason = tracker.record(latest)
            if reason:
                stats.stop_reason = reason
                return
            if not latest.introduced:
                finalizing = True

    stats.stop_reason = MAX_ROUNDS
