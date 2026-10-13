"""Browser voice sessions: config validation, wiring of the realtime agent,
persistence of transcripts and tool captures, and the audio bridge."""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from flo_ai.llm.base_realtime_llm import (
    AudioOutEvent,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from flo_ai.llm.gemini_live_llm import GeminiLiveLLM
from flo_ai.tool.base_tool import Tool
from flo_ai.tool.partial_tool import PartialTool
from pydantic import ValidationError

from chatbots_module.models.chatbot_schemas import CreateChatbotPayload
from chatbots_module.models.voice_schemas import VoiceSessionOverrides
from chatbots_module.services import voice_session_service as vss
from chatbots_module.services.chat_inference_service import ChatInferenceService
from chatbots_module.services.voice_session_service import (
    VoiceSessionError,
    VoiceSessionService,
    VoiceTranscriptRecorder,
    event_payload,
)
from chatbots_module.utils.constants import ROLE_TOOL

USER_ID = uuid.uuid4()


def _chatbot(voice=None, **config):
    if voice is not None:
        config['voice'] = voice
    return SimpleNamespace(
        id=uuid.uuid4(),
        llm_config_id=uuid.uuid4(),
        config=config or None,
        system_prompt='You are a voice assistant',
        welcome_message=None,
    )


def _llm_config(type_='gemini'):
    return {
        'id': uuid.uuid4(),
        'llm_model': 'gemini-2.5-flash',
        'display_name': 'g',
        'api_key': 'key',
        'type': type_,
        'base_url': None,
        'parameters': None,
    }


def _session(chatbot):
    return SimpleNamespace(
        id=uuid.uuid4(),
        chatbot_id=chatbot.id,
        user_id=USER_ID,
        system_prompt_snapshot=chatbot.system_prompt,
    )


async def _lookup(order_id: str, user_id: str):
    return {'order_id': order_id, 'user_id': user_id}


def _tool_loader(tools=None, registry=None):
    tools = tools or {}
    loader = MagicMock()
    loader.load_tool.side_effect = lambda name: tools.get(name)
    loader.function_registry = registry or {}
    return loader


def _service(config=None, sessions=None, tool_loader=None):
    llm_configs = AsyncMock()
    llm_configs.get_config.return_value = (
        config if config is not None else _llm_config()
    )
    return VoiceSessionService(
        llm_inference_config_service=llm_configs,
        chat_session_service=sessions or AsyncMock(),
        tool_loader=tool_loader,
    )


class TestVoiceConfigValidation:
    def _payload(self, voice):
        return CreateChatbotPayload(
            name='bot',
            system_prompt='p',
            llm_config_id=uuid.uuid4(),
            config={'voice': voice},
        )

    def test_valid_config(self):
        payload = self._payload(
            {
                'voice_name': 'Puck',
                'language': 'en-US',
                'vad': {'silence_duration_ms': 500},
                'tools': [{'name': 'datasource_execute_query'}],
                'capture': {'datasource_id': 'd', 'table_name': 't'},
            }
        )
        assert payload.config['voice']['voice_name'] == 'Puck'

    @pytest.mark.parametrize(
        'voice',
        [
            {'vad': {'end_sensitivity': 'medium'}},
            {'tools': [{'name': 'a'}, {'name': 'a'}]},
            {'voice': 'Puck'},
            {'capture': {'datasource_id': 'd'}},
        ],
    )
    def test_invalid_config_rejected_on_save(self, voice):
        with pytest.raises(ValidationError):
            self._payload(voice)

    def test_overrides_cannot_carry_tools(self):
        with pytest.raises(ValidationError):
            VoiceSessionOverrides.model_validate({'tools': [{'name': 'x'}]})


def test_tool_rows_are_not_replayed_to_text_models():
    session = SimpleNamespace(system_prompt_snapshot='p')
    history = [
        SimpleNamespace(role='user', content='hi'),
        SimpleNamespace(role=ROLE_TOOL, content='{}'),
        SimpleNamespace(role='assistant', content='hello'),
    ]
    roles = [m['role'] for m in ChatInferenceService.build_messages(session, history)]
    assert roles == ['system', 'user', 'assistant']


class TestOpen:
    async def test_not_a_voice_chatbot(self):
        with pytest.raises(VoiceSessionError, match='not configured for voice'):
            await _service().open(_chatbot(), USER_ID)

    async def test_requires_gemini(self):
        with pytest.raises(VoiceSessionError, match='Gemini'):
            await _service(config=_llm_config('openai')).open(_chatbot({}), USER_ID)

    async def test_missing_llm_config(self):
        with pytest.raises(VoiceSessionError, match='no LLM configuration'):
            await _service(config={}).open(_chatbot({}), USER_ID)

    async def test_builds_agent_with_context_tools_and_settings(self):
        lookup = Tool(
            name='lookup_order',
            description='Find an order',
            function=_lookup,
            parameters={
                'order_id': {'type': 'string', 'description': ''},
                'user_id': {'type': 'string', 'description': ''},
            },
        )
        chatbot = _chatbot(
            {
                'model': 'gemini-live-test',
                'voice_name': 'Kore',
                'language': 'hi-IN',
                'tools': [
                    {'name': 'lookup_order'},
                    {
                        'name': 'datasource_execute_query',
                        'prefilled_params': {'datasource_id': 'ds-1'},
                    },
                ],
            },
            temperature=0.3,
        )
        session = _session(chatbot)
        sessions = AsyncMock()
        sessions.create_session.return_value = session
        query = Tool(
            name='datasource_execute_query',
            description='Run a query',
            function=AsyncMock(),
            parameters={
                'datasource_id': {'type': 'string', 'description': ''},
                'query_id': {'type': 'string', 'description': ''},
            },
        )
        loader = _tool_loader(
            {'lookup_order': lookup, 'datasource_execute_query': query}
        )

        voice = await _service(sessions=sessions, tool_loader=loader).open(
            chatbot, USER_ID, overrides=VoiceSessionOverrides(voice_name='Puck')
        )

        sessions.create_session.assert_awaited_once_with(
            chatbot=chatbot, user_id=USER_ID
        )
        agent = voice.agent
        assert agent.session_context == {
            'user_id': str(USER_ID),
            'session_id': str(session.id),
        }
        assert agent.system_prompt == chatbot.system_prompt
        assert agent.config == {'voice_name': 'Puck'}
        partial = agent.tools['datasource_execute_query']
        assert isinstance(partial, PartialTool)
        assert list(partial.parameters) == ['query_id']
        llm = agent.llm
        assert isinstance(llm, GeminiLiveLLM)
        assert (llm.model, llm.voice_name, llm.language, llm.temperature) == (
            'gemini-live-test',
            'Kore',
            'hi-IN',
            0.3,
        )
        assert voice.input_sample_rate == 16000

    async def test_unknown_tool(self, monkeypatch):
        monkeypatch.setattr(vss, 'load_api_service_tool', AsyncMock(return_value=None))
        with pytest.raises(VoiceSessionError, match='Voice tool not found: nope'):
            await _service(tool_loader=_tool_loader()).open(
                _chatbot({'tools': [{'name': 'nope'}]}), USER_ID
            )

    async def test_falls_back_to_api_service_tools(self, monkeypatch):
        api_tool = Tool(
            name='svc_api', description='', function=AsyncMock(), parameters={}
        )
        loader = AsyncMock(return_value=api_tool)
        monkeypatch.setattr(vss, 'load_api_service_tool', loader)
        chatbot = _chatbot({'tools': [{'name': 'svc_api'}]})
        sessions = AsyncMock()
        sessions.create_session.return_value = _session(chatbot)
        voice = await _service(sessions=sessions, tool_loader=_tool_loader()).open(
            chatbot, USER_ID
        )
        assert voice.agent.tools['svc_api'] is api_tool

    async def test_continues_an_owned_session(self):
        chatbot = _chatbot({})
        session = _session(chatbot)
        sessions = AsyncMock()
        sessions.get_owned_session.return_value = session
        voice = await _service(sessions=sessions).open(chatbot, USER_ID, session.id)
        sessions.get_owned_session.assert_awaited_once_with(session.id, USER_ID)
        sessions.create_session.assert_not_called()
        assert voice.session is session

    async def test_rejects_session_of_another_chatbot(self):
        chatbot = _chatbot({})
        other = _session(_chatbot({}))
        sessions = AsyncMock()
        sessions.get_owned_session.return_value = other
        with pytest.raises(VoiceSessionError, match='different chatbot'):
            await _service(sessions=sessions).open(chatbot, USER_ID, other.id)


class TestRecorder:
    def _recorder(self, capture=None, insert_rows=None):
        chatbot = _chatbot({})
        session = _session(chatbot)
        sessions = AsyncMock()
        return (
            VoiceTranscriptRecorder(sessions, session, USER_ID, capture, insert_rows),
            sessions,
            session,
        )

    async def test_final_transcripts_persisted_partials_ignored(self):
        recorder, sessions, session = self._recorder()
        await recorder.record(TranscriptEvent(role='user', text='Hel', final=False))
        await recorder.record(TranscriptEvent(role='user', text='Hello'))
        await recorder.record(TranscriptEvent(role='assistant', text='Hi there'))

        calls = sessions.add_message.await_args_list
        assert [(c.kwargs['role'], c.kwargs['content']) for c in calls] == [
            ('user', 'Hello'),
            ('assistant', 'Hi there'),
        ]
        assert calls[0].kwargs['session_id'] == session.id
        assert calls[0].kwargs['metadata'] == {'source': 'voice'}
        sessions.ensure_title.assert_awaited_once_with(session, 'Hello')

    async def test_tool_result_persisted_as_tool_row(self):
        recorder, sessions, _ = self._recorder()
        await recorder.record(
            ToolResultEvent(
                id='c1',
                name='lookup_order',
                arguments={'order_id': '7', 'user_id': str(USER_ID)},
                result={'status': 'shipped'},
            )
        )
        call = sessions.add_message.await_args
        assert call.kwargs['role'] == ROLE_TOOL
        assert call.kwargs['metadata']['arguments']['user_id'] == str(USER_ID)
        assert json.loads(call.kwargs['content'])['result'] == {'status': 'shipped'}

    async def test_capture_goes_through_datasource_tool_keyed_by_user(self):
        insert_rows = AsyncMock(return_value='Inserted 1 row(s)')
        capture = SimpleNamespace(datasource_id='ds', table_name='captures')
        recorder, _, session = self._recorder(capture, insert_rows)
        await recorder.record(TranscriptEvent(role='user', text='Book a demo'))
        kwargs = insert_rows.await_args.kwargs
        assert kwargs['datasource_id'] == 'ds'
        assert kwargs['table_name'] == 'captures'
        assert kwargs['single_row'] is True
        row = kwargs['data']
        assert row['user_id'] == str(USER_ID)
        assert row['session_id'] == str(session.id)
        assert (row['kind'], row['role'], row['content']) == (
            'transcript',
            'user',
            'Book a demo',
        )


def test_event_payload_hides_tool_data():
    assert event_payload(
        ToolResultEvent(id='1', name='t', arguments={'secret': 1}, result='rows')
    ) == {'type': 'tool_result', 'id': '1', 'name': 't', 'ok': True}
    assert event_payload(ToolCallEvent(id='1', name='t', arguments={'secret': 1})) == {
        'type': 'tool_call',
        'id': '1',
        'name': 't',
    }
    assert event_payload(AudioOutEvent(data=b'x')) is None


class FakeWebSocket:
    def __init__(self, incoming):
        self.incoming = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(message)
        self.sent_bytes = []
        self.sent_json = []

    async def receive(self):
        return await self.incoming.get()

    async def send_bytes(self, data):
        self.sent_bytes.append(data)

    async def send_json(self, data):
        self.sent_json.append(data)


class FakeAgent:
    def __init__(self, events, wait_for_audio=0):
        self._events = events
        self.audio = []
        self.texts = []
        self._wait_for_audio = wait_for_audio
        self._got_audio = asyncio.Event()

    async def send_audio(self, audio):
        self.audio.append(audio)
        if len(self.audio) >= self._wait_for_audio:
            self._got_audio.set()

    async def send_text(self, text):
        self.texts.append(text)

    async def events(self):
        if self._wait_for_audio:
            await self._got_audio.wait()
        for event in self._events:
            yield event
        await asyncio.Event().wait()


class TestBridge:
    async def test_audio_events_and_persistence(self):
        events = [
            TranscriptEvent(role='user', text='Hi'),
            AudioOutEvent(data=b'out'),
            ToolCallEvent(id='1', name='t', arguments={}),
            ToolResultEvent(id='1', name='t', arguments={}, result='ok'),
            TurnCompleteEvent(),
        ]
        agent = FakeAgent(events, wait_for_audio=1)
        websocket = FakeWebSocket(
            [
                {'type': 'websocket.receive', 'bytes': b'pcm'},
                {
                    'type': 'websocket.receive',
                    'text': json.dumps({'type': 'text', 'text': 'typed'}),
                },
                {'type': 'websocket.receive', 'text': 'garbage'},
            ]
        )
        recorder = AsyncMock()

        async def hang_up_after_events():
            while len(websocket.sent_json) < 5:
                await asyncio.sleep(0.01)
            websocket.incoming.put_nowait({'type': 'websocket.disconnect'})

        hangup = asyncio.create_task(hang_up_after_events())
        try:
            await asyncio.wait_for(
                _service().bridge(websocket, agent, recorder), timeout=2
            )
        finally:
            hangup.cancel()
            await asyncio.gather(hangup, return_exceptions=True)

        assert agent.audio == [b'pcm']
        assert agent.texts == ['typed']
        assert websocket.sent_bytes == [b'out']
        # The error answers the garbage frame, which races the model's events.
        types_sent = [frame['type'] for frame in websocket.sent_json]
        assert sorted(types_sent) == sorted(
            ['error', 'transcript', 'tool_call', 'tool_result', 'turn_complete']
        )
        model_frames = [t for t in types_sent if t != 'error']
        assert model_frames == [
            'transcript',
            'tool_call',
            'tool_result',
            'turn_complete',
        ]
        recorded = [call.args[0] for call in recorder.record.await_args_list]
        assert recorded == [events[0], events[3]]

    async def test_model_failure_propagates(self):
        class Failing(FakeAgent):
            async def events(self):
                raise RuntimeError('provider down')
                yield  # pragma: no cover

        websocket = FakeWebSocket([])
        with pytest.raises(RuntimeError, match='provider down'):
            await asyncio.wait_for(
                _service().bridge(websocket, Failing([]), AsyncMock()), timeout=2
            )

    async def test_stop_command_ends_session(self):
        websocket = FakeWebSocket(
            [{'type': 'websocket.receive', 'text': json.dumps({'type': 'stop'})}]
        )
        await asyncio.wait_for(
            _service().bridge(websocket, FakeAgent([]), AsyncMock()), timeout=2
        )
