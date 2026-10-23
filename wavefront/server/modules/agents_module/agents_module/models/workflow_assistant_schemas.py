"""Request model and wire events for the workflow assistant chat endpoint."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from agents_module.models.yaml_validation_schemas import YamlValidationError
from agents_module.services.agent_stream_events import AgentEventType, make_event

# Hard ceilings enforced by the schema. The configurable, tighter limits live in
# AssistantLimits and are applied by the service.
# History is never rejected for length: only this many of the most recent messages
# are even parsed (a safety bound on work and memory); the service then keeps its
# own, smaller max_history_messages.
MAX_HISTORY_ITEMS = 100
MAX_CONTENT_CHARS = 50000
MAX_YAML_CHARS = 500000


class AssistantHistoryMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(..., max_length=MAX_CONTENT_CHARS)


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)
    current_yaml: str = Field(
        '',
        max_length=MAX_YAML_CHARS,
        description='Workflow YAML currently on the canvas. Untrusted.',
    )
    history: List[AssistantHistoryMessage] = Field(
        default_factory=list,
        description=(
            'Earlier turns, held by the client. Untrusted. Only the most recent '
            'ones are used; older ones are dropped, not rejected.'
        ),
    )
    llm_inference_config_id: str = Field(
        ..., description='UUID of the LLM inference config the assistant runs on'
    )
    mode: Literal['auto', 'ask', 'edit'] = Field(
        'auto',
        description=(
            "'ask' answers a question with a streamed reply and never changes the "
            "workflow, 'edit' runs the tool loop, 'auto' (default) lets the server "
            'classify the message first. Omit for the default.'
        ),
    )

    @field_validator('history', mode='before')
    @classmethod
    def keep_recent_history(cls, value: Any) -> Any:
        # Before item validation, so dropped messages cost nothing and a long chat
        # is trimmed here instead of failing with a 422
        if isinstance(value, list) and len(value) > MAX_HISTORY_ITEMS:
            return value[-MAX_HISTORY_ITEMS:]
        return value


class AssistantEventType:
    """Event names the assistant adds to AgentEventType's shared vocabulary."""

    ASSISTANT_STARTED = 'assistant_started'
    PROPOSAL = 'proposal'


def assistant_started() -> Dict[str, Any]:
    return make_event(AssistantEventType.ASSISTANT_STARTED)


def content_delta(content: str) -> Dict[str, Any]:
    return make_event(AgentEventType.CONTENT_DELTA, content=content)


def retract_event(content: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Everything sent as ``content_delta`` so far is withdrawn; ``content`` is
    what to show in its place (empty when nothing may be shown)."""
    return make_event(AgentEventType.RETRACT, content=content, reason=reason)


def tool_called(tool_call_id: str, tool_name: str, label: str) -> Dict[str, Any]:
    return make_event(
        AgentEventType.TOOL_CALLED,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        label=label,
    )


def tool_result(
    tool_call_id: str, tool_name: str, ok: bool, summary: Optional[str] = None
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        'tool_call_id': tool_call_id,
        'tool_name': tool_name,
        'ok': ok,
    }
    if summary is not None:
        fields['summary'] = summary
    return make_event(AgentEventType.TOOL_RESULT, **fields)


def tool_failed(tool_call_id: str, tool_name: str, error: str) -> Dict[str, Any]:
    return make_event(
        AgentEventType.TOOL_FAILED,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        error=error,
    )


def proposal_event(
    proposal_id: str,
    yaml: str,
    valid: bool,
    errors: List[YamlValidationError],
    introduced_errors: Optional[List[YamlValidationError]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """``errors`` is every problem in the proposed YAML and ``valid`` is "no
    errors". ``introduced_errors`` is the subset the change caused (the rest were
    already in the workflow); it defaults to all of ``errors``. ``notes`` are the
    choices the assistant made on the user's behalf; the key is left off when
    there are none, so older clients see the event unchanged."""
    introduced = errors if introduced_errors is None else introduced_errors
    fields: Dict[str, Any] = {
        'proposal_id': proposal_id,
        'yaml': yaml,
        'valid': valid,
        'errors': [{'path': e.path, 'message': e.message} for e in errors],
        'introduced_errors': [
            {'path': e.path, 'message': e.message} for e in introduced
        ],
    }
    if notes:
        fields['notes'] = list(notes)
    return make_event(AssistantEventType.PROPOSAL, **fields)


def output_event(content: str, truncated: bool = False) -> Dict[str, Any]:
    """``truncated`` marks a reply the provider cut off at its output limit; the
    key is left off otherwise, so older clients see the event unchanged."""
    fields: Dict[str, Any] = {'content': content}
    if truncated:
        fields['truncated'] = True
    return make_event(AgentEventType.OUTPUT, **fields)


def error_event(message: str) -> Dict[str, Any]:
    return make_event(AgentEventType.ERROR, message=message)
