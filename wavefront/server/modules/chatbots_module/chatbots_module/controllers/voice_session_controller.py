"""Browser voice sessions over WebSocket.

    WS /floware/v1/chatbots/{chatbot_id}/voice

1. Client sends `{"type": "start", "token": "<access token>",
   "session_id"?: "<chat session>", "config"?: {voice_name, language, vad}}`
   within AUTH_TIMEOUT_SECONDS.
2. Server replies `{"type": "ready", "session_id", "input_sample_rate",
   "output_sample_rate"}`.
3. Client streams binary PCM16 mono frames at input_sample_rate; server streams
   binary PCM16 mono at output_sample_rate plus JSON event frames. See
   VoiceSessionService.bridge for the frame protocol.

The user is the one the token authenticates; nothing the client sends can
choose it. That user_id and the chat session_id are merged into every tool
call the model makes.

Close codes: 4400 bad request, 4401 unauthenticated, 4403 not a wavefront
user, 4404 chatbot or session not found, 1011 model failure.
"""

import asyncio
import uuid
from uuid import UUID

from common_module.log.logger import logger
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState
from user_management_module.authorization.websocket_auth import (
    WebSocketAuthError,
    authenticate_token,
)

from chatbots_module.chatbots_container import ChatbotsContainer
from chatbots_module.controllers.chat_session_controller import (
    _require_chattable_chatbot,
)
from chatbots_module.models.voice_schemas import VoiceStartFrame
from chatbots_module.services.chat_session_service import (
    ChatSessionNotFoundError,
    UnknownChatUserError,
)
from chatbots_module.services.chatbot_service import (
    ChatbotNotFoundError,
    ChatbotService,
)
from chatbots_module.services.voice_session_service import (
    VoiceSessionError,
    VoiceSessionService,
)

voice_session_router = APIRouter()

AUTH_TIMEOUT_SECONDS = 10

CLOSE_BAD_REQUEST = 4400
CLOSE_UNAUTHENTICATED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_NOT_FOUND = 4404
CLOSE_INTERNAL_ERROR = 1011

MODEL_FAILURE_MESSAGE = 'The voice model failed to respond'


async def _close(websocket: WebSocket, code: int, reason: str) -> None:
    if websocket.client_state == WebSocketState.DISCONNECTED:
        return
    # A close reason is capped at 123 bytes by the protocol.
    await websocket.close(
        code=code, reason=reason.encode()[:123].decode(errors='ignore')
    )


@voice_session_router.websocket('/v1/chatbots/{chatbot_id}/voice')
@inject
async def chatbot_voice_session(
    websocket: WebSocket,
    chatbot_id: UUID,
    chatbot_service: ChatbotService = Depends(
        Provide[ChatbotsContainer.chatbot_service]
    ),
    voice_session_service: VoiceSessionService = Depends(
        Provide[ChatbotsContainer.voice_session_service]
    ),
):
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), AUTH_TIMEOUT_SECONDS)
        start = VoiceStartFrame.model_validate_json(raw)
    except asyncio.TimeoutError:
        return await _close(
            websocket, CLOSE_UNAUTHENTICATED, 'Authentication timed out'
        )
    except WebSocketDisconnect:
        return
    except (ValidationError, KeyError):
        # KeyError: the first frame was binary.
        return await _close(
            websocket, CLOSE_BAD_REQUEST, 'First frame must be a start message'
        )

    try:
        user_session = await authenticate_token(start.token)
    except WebSocketAuthError as exc:
        return await _close(websocket, CLOSE_UNAUTHENTICATED, str(exc))

    try:
        user_id = uuid.UUID(str(user_session.user_id))
    except ValueError:
        return await _close(
            websocket, CLOSE_FORBIDDEN, 'Voice sessions require an end-user identity'
        )

    try:
        chatbot = await _require_chattable_chatbot(chatbot_service, chatbot_id)
        voice = await voice_session_service.open(
            chatbot, user_id, start.session_id, start.config
        )
    except (ChatbotNotFoundError, ChatSessionNotFoundError) as exc:
        return await _close(websocket, CLOSE_NOT_FOUND, str(exc))
    except UnknownChatUserError:
        return await _close(
            websocket,
            CLOSE_FORBIDDEN,
            'Your credential is not associated with a wavefront user',
        )
    except VoiceSessionError as exc:
        return await _close(websocket, CLOSE_BAD_REQUEST, str(exc))

    try:
        await voice.agent.start()
    except Exception:
        # Provider errors can carry the base_url or a key prefix; logs only.
        logger.exception(f'Voice model failed to connect for chatbot {chatbot_id}')
        return await _close(websocket, CLOSE_INTERNAL_ERROR, MODEL_FAILURE_MESSAGE)

    try:
        await websocket.send_json(
            {
                'type': 'ready',
                'session_id': str(voice.session.id),
                'input_sample_rate': voice.input_sample_rate,
                'output_sample_rate': voice.output_sample_rate,
            }
        )
        await voice_session_service.bridge(websocket, voice.agent, voice.recorder)
        await _close(websocket, 1000, 'Session ended')
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f'Voice session {voice.session.id} failed')
        await _close(websocket, CLOSE_INTERNAL_ERROR, MODEL_FAILURE_MESSAGE)
    finally:
        await voice.agent.close()
