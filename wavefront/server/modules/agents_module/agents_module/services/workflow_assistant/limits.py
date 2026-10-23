"""Per-request limits for the workflow assistant, from [workflow_assistant] in config.ini."""

from dataclasses import dataclass, fields
from typing import Any, Callable

from common_module.log.logger import logger

DEFAULT_ENABLED = True
# Upper bound on LLM calls in one turn (tool rounds plus the final text round)
DEFAULT_MAX_ROUNDS = 8
# validate-and-repair attempts: propose_workflow_yaml calls before giving up
DEFAULT_MAX_REPAIR_ROUNDS = 3
DEFAULT_MAX_TOTAL_TOKENS = 60000
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_HISTORY_MESSAGES = 20
DEFAULT_MAX_MESSAGE_CHARS = 8000
DEFAULT_MAX_YAML_CHARS = 100000
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 20000
# The assistant writes whole workflow YAML in a tool call: a lower output limit on the
# selected LLM config cuts the call off mid-JSON, so its calls get at least this many.
DEFAULT_MAX_OUTPUT_TOKENS = 8192


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'not a boolean: {raw!r}')


def _coerce(name: str, raw: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    if raw is None or (isinstance(raw, str) and raw.strip() == ''):
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning(
            f'Workflow assistant: {name}={raw!r} is not usable, using {default}'
        )
        return default


@dataclass(frozen=True)
class AssistantLimits:
    """Values arrive as ini strings, or None for an undefined key; both are
    coerced, and unusable ones fall back to the default."""

    # Kill switch: the endpoint answers 503 when false.
    enabled: bool = DEFAULT_ENABLED
    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS
    # Provider-reported tokens when available, otherwise a chars/4 estimate.
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS
    max_yaml_chars: int = DEFAULT_MAX_YAML_CHARS
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    # Floor for the output tokens of each assistant LLM call (never lowers a higher config limit)
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        casts: dict[str, Callable[[Any], Any]] = {
            'enabled': _to_bool,
            'max_rounds': int,
            'max_repair_rounds': int,
            'max_total_tokens': int,
            'timeout_seconds': float,
            'max_history_messages': int,
            'max_message_chars': int,
            'max_yaml_chars': int,
            'max_tool_output_chars': int,
            'max_output_tokens': int,
        }
        for field in fields(self):
            value = _coerce(
                f'[workflow_assistant] {field.name}',
                getattr(self, field.name),
                casts[field.name],
                field.default,
            )
            object.__setattr__(self, field.name, value)
