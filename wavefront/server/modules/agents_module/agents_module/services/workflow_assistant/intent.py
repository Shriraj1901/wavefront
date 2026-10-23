"""Routing a chat message: does the user want an answer, or a workflow change?

One small non-streaming call. An answer is streamed with no tools at all, so the
safe default for anything unclear is EDIT, which is the full tool loop.
"""

import asyncio
import re
import time
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional, Tuple

from common_module.log.logger import logger

from agents_module.services.workflow_assistant.repair_loop import response_tokens

ASK = 'ask'
EDIT = 'edit'

#: Characters of the previous assistant message shown to the classifier
PREVIOUS_REPLY_CHARS = 600
#: Characters of the user's message shown to the classifier
MESSAGE_CHARS = 2000

CLASSIFIER_PROMPT = """You route messages sent to a workflow-editing assistant. \
Reply with exactly one word and nothing else:
ASK  - the user only wants information, an explanation or advice and no change to the workflow
EDIT - the user wants the workflow created, changed, fixed or validated, or is \
agreeing to a change the assistant just offered (for example "yes do it")
If you are unsure, reply EDIT."""

_FIRST_WORD = re.compile(r'[^A-Za-z]*([A-Za-z]+)')


def parse_intent(text: Optional[str]) -> str:
    """ASK only for a reply whose first word is ASK; everything else is EDIT."""
    match = _FIRST_WORD.match(text or '')
    if match and match.group(1).upper() == 'ASK':
        return ASK
    return EDIT


def classifier_messages(user_message: str, last_assistant: Optional[str]) -> list:
    parts = []
    if last_assistant:
        parts.append(
            'Previous assistant message (truncated):\n'
            + last_assistant[:PREVIOUS_REPLY_CHARS]
        )
    parts.append('Latest user message:\n' + user_message[:MESSAGE_CHARS])
    return [
        {'role': 'system', 'content': CLASSIFIER_PROMPT},
        {'role': 'user', 'content': '\n\n'.join(parts)},
    ]


async def classify_intent(
    llm: Any,
    user_message: str,
    last_assistant: Optional[str],
    *,
    deadline: float,
    scope: Callable[[], ContextManager[Any]] = nullcontext,
) -> Tuple[str, int]:
    """(``ASK`` or ``EDIT``, tokens used). Never raises except on cancellation:
    a failed, empty, slow or unreadable classification is EDIT."""
    messages = classifier_messages(user_message, last_assistant)

    async def call() -> Any:
        with scope():
            return await llm.generate(messages)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return EDIT, 0
    try:
        response = await asyncio.wait_for(call(), timeout=remaining)
        text = llm.get_message_content(response) or ''
    except Exception as exc:
        logger.warning(
            f'Workflow assistant: intent classification failed ({type(exc).__name__})'
        )
        return EDIT, 0
    return parse_intent(text), response_tokens(response, messages, text)
