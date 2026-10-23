from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dependency_injector.wiring import inject, Provide

from common_module.common_container import CommonContainer
from common_module.response_formatter import ResponseFormatter
from agents_module.agents_container import AgentsContainer
from agents_module.models.workflow_assistant_schemas import AssistantChatRequest
from agents_module.services.workflow_assistant.service import (
    AssistantRequestError,
    WorkflowAssistantService,
)
from agents_module.utils.sse_utils import SSE_HEADERS, format_sse

workflow_assistant_router = APIRouter()


# Under the '/v1/workflow-management' prefix on purpose: require_auth gates that
# prefix to admins, so this endpoint (which spends LLM tokens and reads the app's
# resource inventory) needs no auth change of its own.
@workflow_assistant_router.post('/v1/workflow-management/assistant/chat')
@inject
async def workflow_assistant_chat(
    request: Request,
    payload: AssistantChatRequest,
    assistant_service: WorkflowAssistantService = Depends(
        Provide[AgentsContainer.workflow_assistant_service]
    ),
    response_formatter: ResponseFormatter = Depends(
        Provide[CommonContainer.response_formatter]
    ),
):
    """
    Chat with the workflow assistant (server-sent events).

    Streams `assistant_started`, then any `content_delta` (replies arrive as they
    are written, in small batches), `retract` (a guardrail withdrew text already
    sent; `content` replaces it), `tool_called`, `tool_result` and `tool_failed`
    events, then at most one `proposal` (the last
    workflow YAML the assistant put forward, with `valid` and its `errors`), and
    always ends with `output` or `error`.

    Everything that can be rejected up front (disabled, oversize input, unknown or
    unsupported LLM config) answers with a plain error status before the stream
    opens.

    `mode` is `auto` (default: the server decides whether the message is a question
    or a change request), `ask` (answer only, never changes the workflow) or `edit`.
    """
    session = getattr(request.state, 'session', None)
    user_id = getattr(session, 'user_id', None)

    # Prepared before the response, not inside the generator: an async generator
    # does not run until Starlette iterates it, which is after the 200 has gone
    # out, so a rejected request would look like a success with the error hidden
    # in the stream.
    try:
        prepared = await assistant_service.prepare(
            payload, user_id=str(user_id) if user_id is not None else None
        )
    except AssistantRequestError as e:
        return JSONResponse(
            status_code=e.status_code,
            content=response_formatter.buildErrorResponse(e.message),
        )

    async def event_stream():
        # One frame per event, yielded as it is produced: nothing here buffers
        events = assistant_service.chat(prepared)
        try:
            async for event in events:
                yield format_sse(event)
        finally:
            # A client that hangs up leaves `events` suspended; closing it here
            # is what closes the provider's stream and writes the audit record
            await events.aclose()

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers=SSE_HEADERS,
    )
