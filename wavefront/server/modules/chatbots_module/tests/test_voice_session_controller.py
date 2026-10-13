"""The voice WebSocket: first-frame auth, close codes, and the ready frame."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from user_management_module.authorization.require_auth import UserSession
from user_management_module.authorization.websocket_auth import WebSocketAuthError

from chatbots_module.chatbots_container import ChatbotsContainer
from chatbots_module.controllers import voice_session_controller as controller
from chatbots_module.services.chat_session_service import ChatSessionNotFoundError
from chatbots_module.services.voice_session_service import VoiceSessionError

USER_ID = uuid.uuid4()
CHATBOT_ID = uuid.uuid4()
PATH = f'/floware/v1/chatbots/{CHATBOT_ID}/voice'


@pytest.fixture
def harness(monkeypatch):
    chatbot = SimpleNamespace(id=CHATBOT_ID, enabled=True)
    chatbot_service = AsyncMock()
    chatbot_service.get_chatbot.return_value = chatbot

    session = SimpleNamespace(id=uuid.uuid4())
    agent = MagicMock()
    agent.start = AsyncMock()
    agent.close = AsyncMock()
    voice = SimpleNamespace(
        agent=agent,
        session=session,
        recorder=object(),
        input_sample_rate=16000,
        output_sample_rate=24000,
    )
    voice_service = MagicMock()
    voice_service.open = AsyncMock(return_value=voice)

    async def bridge(websocket, agent, recorder):
        await websocket.send_json(
            {'type': 'transcript', 'role': 'assistant', 'text': 'hi', 'final': True}
        )

    voice_service.bridge = AsyncMock(side_effect=bridge)

    auth = AsyncMock(
        return_value=UserSession(role_id='r', user_id=str(USER_ID), session_id='s')
    )
    monkeypatch.setattr(controller, 'authenticate_token', auth)

    container = ChatbotsContainer()
    container.chatbot_service.override(providers.Object(chatbot_service))
    container.voice_session_service.override(providers.Object(voice_service))
    container.wire(modules=[controller])

    app = FastAPI()
    app.include_router(controller.voice_session_router, prefix='/floware')
    client = TestClient(app)
    yield SimpleNamespace(
        client=client,
        auth=auth,
        chatbot=chatbot,
        voice_service=voice_service,
        voice=voice,
    )
    container.unwire()


def _close_code(ws):
    with pytest.raises(WebSocketDisconnect) as exc:
        ws.receive_json()
    return exc.value.code, exc.value.reason


def test_happy_path(harness):
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json(
            {'type': 'start', 'token': 'tok', 'config': {'voice_name': 'Puck'}}
        )
        ready = ws.receive_json()
        assert ready == {
            'type': 'ready',
            'session_id': str(harness.voice.session.id),
            'input_sample_rate': 16000,
            'output_sample_rate': 24000,
        }
        assert ws.receive_json()['text'] == 'hi'
        assert _close_code(ws)[0] == 1000

    harness.auth.assert_awaited_once_with('tok')
    chatbot, user_id, session_id, overrides = harness.voice_service.open.await_args.args
    assert chatbot is harness.chatbot
    assert user_id == USER_ID
    assert session_id is None
    assert overrides.voice_name == 'Puck'
    harness.voice.agent.start.assert_awaited_once()
    harness.voice.agent.close.assert_awaited_once()


@pytest.mark.parametrize(
    'frame',
    [
        {'type': 'start'},
        {'type': 'start', 'token': 'tok', 'user_id': str(uuid.uuid4())},
        {'type': 'audio', 'token': 'tok'},
    ],
)
def test_bad_start_frame(harness, frame):
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json(frame)
        assert _close_code(ws)[0] == 4400
    harness.auth.assert_not_called()


def test_binary_first_frame(harness):
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_bytes(b'pcm')
        assert _close_code(ws)[0] == 4400


def test_bad_token(harness):
    harness.auth.side_effect = WebSocketAuthError('Invalid token')
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'bad'})
        assert _close_code(ws) == (4401, 'Invalid token')
    harness.voice_service.open.assert_not_called()


def test_service_identity_rejected(harness):
    harness.auth.return_value = UserSession(
        role_id='r', user_id='service', session_id='s'
    )
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok'})
        assert _close_code(ws)[0] == 4403


def test_disabled_chatbot_is_not_found(harness):
    harness.chatbot.enabled = False
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok'})
        assert _close_code(ws)[0] == 4404


def test_someone_elses_session_is_not_found(harness):
    harness.voice_service.open.side_effect = ChatSessionNotFoundError('nope')
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok', 'session_id': str(uuid.uuid4())})
        assert _close_code(ws)[0] == 4404


def test_voice_not_configured(harness):
    harness.voice_service.open.side_effect = VoiceSessionError(
        'This chatbot is not configured for voice'
    )
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok'})
        assert _close_code(ws) == (4400, 'This chatbot is not configured for voice')


def test_model_connect_failure_hides_provider_error(harness):
    harness.voice.agent.start.side_effect = RuntimeError(
        'https://internal-gw key=sk-123'
    )
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok'})
        assert _close_code(ws) == (1011, controller.MODEL_FAILURE_MESSAGE)


def test_bridge_failure_closes_and_releases_model(harness):
    harness.voice_service.bridge.side_effect = RuntimeError('provider down')
    with harness.client.websocket_connect(PATH) as ws:
        ws.send_json({'type': 'start', 'token': 'tok'})
        assert ws.receive_json()['type'] == 'ready'
        assert _close_code(ws)[0] == 1011
    harness.voice.agent.close.assert_awaited_once()
