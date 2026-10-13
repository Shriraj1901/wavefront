import pytest
from google.genai import errors, types

from flo_ai.llm.base_realtime_llm import (
    AudioOutEvent,
    InterruptedEvent,
    ToolCallEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from flo_ai.llm.gemini_live_llm import GeminiLiveLLM, build_vad_config
from flo_ai.tool.base_tool import Tool


async def _lookup(order_id: str) -> str:
    return order_id


LOOKUP = Tool(
    name='lookup_order',
    description='Look up an order',
    function=_lookup,
    parameters={'order_id': {'type': 'string', 'description': 'Order id'}},
)


def _llm(**kwargs) -> GeminiLiveLLM:
    return GeminiLiveLLM(api_key='test-key', **kwargs)


def _content(**kwargs) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(**kwargs))


def _audio_turn(data: bytes) -> types.Content:
    return types.Content(
        role='model',
        parts=[types.Part(inline_data=types.Blob(data=data, mime_type='audio/pcm'))],
    )


class FakeSession:
    def __init__(self, turns):
        self.turns = list(turns)
        self.audio = []
        self.texts = []
        self.tool_responses = []

    async def receive(self):
        if not self.turns:
            return
        for message in self.turns.pop(0):
            yield message

    async def send_realtime_input(self, audio=None, text=None):
        if audio is not None:
            self.audio.append(audio)
        if text is not None:
            self.texts.append(text)

    async def send_tool_response(self, function_responses):
        self.tool_responses.extend(function_responses)


class TestConnectConfig:
    def test_voice_language_vad_and_tools(self):
        llm = _llm(
            voice_name='Puck',
            language='en-US',
            vad={'start_sensitivity': 'low', 'silence_duration_ms': 700},
        )
        config = llm.build_connect_config('Be helpful', [LOOKUP])

        assert config.response_modalities == [types.Modality.AUDIO]
        assert config.system_instruction.parts[0].text == 'Be helpful'
        assert (
            config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Puck'
        )
        assert config.speech_config.language_code == 'en-US'
        detection = config.realtime_input_config.automatic_activity_detection
        assert detection.start_of_speech_sensitivity == (
            types.StartSensitivity.START_SENSITIVITY_LOW
        )
        assert detection.silence_duration_ms == 700
        declaration = config.tools[0].function_declarations[0]
        assert declaration.name == 'lookup_order'
        assert declaration.parameters.required == ['order_id']
        assert config.input_audio_transcription is not None
        assert config.output_audio_transcription is not None

    def test_session_config_overrides_constructor(self):
        llm = _llm(voice_name='Puck', language='en-US')
        config = llm.build_connect_config(
            'x', config={'voice_name': 'Kore', 'language': 'hi-IN'}
        )
        assert (
            config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Kore'
        )
        assert config.speech_config.language_code == 'hi-IN'

    def test_minimal_config_omits_optional_blocks(self):
        config = _llm().build_connect_config('x')
        assert config.speech_config is None
        assert config.tools is None
        assert config.realtime_input_config is None

    def test_unknown_session_setting_rejected(self):
        with pytest.raises(ValueError, match='voice'):
            _llm().build_connect_config('x', config={'voice': 'Puck'})


class TestVadConfig:
    def test_none_when_empty(self):
        assert build_vad_config(None) is None
        assert build_vad_config({}) is None

    def test_disable_and_no_interruptions(self):
        config = build_vad_config({'disabled': True, 'interruptions': False})
        assert config.automatic_activity_detection.disabled is True
        assert config.activity_handling == types.ActivityHandling.NO_INTERRUPTION

    def test_bad_sensitivity(self):
        with pytest.raises(ValueError, match='end_sensitivity'):
            build_vad_config({'end_sensitivity': 'medium'})

    def test_unknown_key(self):
        with pytest.raises(ValueError, match='silence_ms'):
            build_vad_config({'silence_ms': 100})


class TestTranslate:
    def test_audio_and_transcripts_through_a_turn(self):
        llm = _llm()
        events = []
        for message in [
            _content(input_transcription=types.Transcription(text='Where is ')),
            _content(input_transcription=types.Transcription(text='my order?')),
            _content(
                model_turn=_audio_turn(b'\x01\x02'),
                output_transcription=types.Transcription(text='Let me check.'),
            ),
            _content(turn_complete=True),
        ]:
            events.extend(llm._translate(message))

        finals = [e for e in events if isinstance(e, TranscriptEvent) and e.final]
        assert finals == [
            TranscriptEvent(role='user', text='Where is my order?'),
            TranscriptEvent(role='assistant', text='Let me check.'),
        ]
        assert AudioOutEvent(data=b'\x01\x02') in events
        assert isinstance(events[-1], TurnCompleteEvent)
        partials = [
            e.text for e in events if isinstance(e, TranscriptEvent) and not e.final
        ]
        assert partials == ['Where is ', 'my order?', 'Let me check.']

    def test_user_final_precedes_model_audio(self):
        llm = _llm()
        llm._translate(_content(input_transcription=types.Transcription(text='Hi')))
        events = llm._translate(_content(model_turn=_audio_turn(b'a')))
        assert events[0] == TranscriptEvent(role='user', text='Hi')
        assert events[1] == AudioOutEvent(data=b'a')

    def test_interrupted_flushes_partial_assistant_text(self):
        llm = _llm()
        llm._translate(
            _content(output_transcription=types.Transcription(text='Your order'))
        )
        events = llm._translate(_content(interrupted=True))
        assert events == [
            TranscriptEvent(role='assistant', text='Your order'),
            InterruptedEvent(),
        ]

    def test_tool_call(self):
        llm = _llm()
        events = llm._translate(
            types.LiveServerMessage(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            id='c1', name='lookup_order', args={'order_id': '42'}
                        )
                    ]
                )
            )
        )
        assert events == [
            ToolCallEvent(id='c1', name='lookup_order', arguments={'order_id': '42'})
        ]


class TestSession:
    async def test_events_span_turns_and_tool_result_round_trip(self):
        llm = _llm()
        session = FakeSession(
            [
                [
                    types.LiveServerMessage(
                        tool_call=types.LiveServerToolCall(
                            function_calls=[
                                types.FunctionCall(
                                    id='c1', name='lookup_order', args={}
                                )
                            ]
                        )
                    )
                ],
                [_content(turn_complete=True)],
            ]
        )
        llm._session = session

        events = [event async for event in llm.events()]
        assert [e.type for e in events] == ['tool_call', 'turn_complete']

        await llm.send_tool_result('c1', 'shipped')
        response = session.tool_responses[0]
        assert response.id == 'c1'
        assert response.name == 'lookup_order'
        assert response.response == {'result': 'shipped'}

    async def test_send_audio_and_text(self):
        llm = _llm()
        llm._session = session = FakeSession([])
        await llm.send_audio(b'pcm')
        await llm.send_text('hello')
        assert session.audio[0].data == b'pcm'
        assert session.audio[0].mime_type == 'audio/pcm;rate=16000'
        assert session.texts == ['hello']

    async def test_dict_result_passed_through_and_unserialisable_stringified(self):
        llm = _llm()
        llm._session = session = FakeSession([])
        await llm.send_tool_result('a', {'status': 'ok'})
        await llm.send_tool_result('b', object())
        assert session.tool_responses[0].response == {'status': 'ok'}
        assert set(session.tool_responses[1].response) == {'result'}

    async def test_events_end_quietly_after_close(self):
        llm = _llm()

        class Closing(FakeSession):
            async def receive(self):
                await llm.close()
                raise errors.APIError(1006, {})
                yield  # pragma: no cover

        llm._session = Closing([])
        assert [event async for event in llm.events()] == []

    async def test_send_before_connect(self):
        with pytest.raises(RuntimeError, match='not connected'):
            await _llm().send_audio(b'x')

    async def test_connect_and_close(self):
        llm = _llm()
        entered = []

        class Ctx:
            async def __aenter__(self):
                entered.append('in')
                return FakeSession([])

            async def __aexit__(self, *exc):
                entered.append('out')

        captured = {}

        def connect(model, config):
            captured.update(model=model, config=config)
            return Ctx()

        llm.client = type(
            'C',
            (),
            {
                'aio': type(
                    'A', (), {'live': type('L', (), {'connect': staticmethod(connect)})}
                )
            },
        )
        await llm.connect('Be brief', [LOOKUP])
        assert captured['model'] == llm.model
        assert (
            captured['config'].tools[0].function_declarations[0].name == 'lookup_order'
        )
        with pytest.raises(RuntimeError, match='already connected'):
            await llm.connect('again')
        await llm.close()
        await llm.close()
        assert entered == ['in', 'out']
