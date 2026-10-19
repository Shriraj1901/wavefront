"""Provider factories build real pipecat services from stored agent configs.

pipecat's Settings classes are dataclasses, so a renamed or removed field raises
TypeError at construction -- these tests catch that in CI instead of on a live
call. Constructors open no connections, so nothing here touches the network.
"""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from call_processing.serializers.smartflo_serializer import SmartfloFrameSerializer
from call_processing.services.llm_service import (
    FUNCTION_CALL_TIMEOUT_SECS,
    LLMServiceFactory,
)
from call_processing.services.stt_service import STTServiceFactory
from call_processing.services.tts_service import TTSServiceFactory

# Every pipecat name call_processing imports. A removal in a future pipecat
# release fails here rather than at the first call after deploy.
PIPECAT_IMPORTS = [
    ('pipecat.adapters.schemas.function_schema', 'FunctionSchema'),
    ('pipecat.adapters.schemas.tools_schema', 'ToolsSchema'),
    ('pipecat.audio.dtmf.types', 'KeypadEntry'),
    ('pipecat.audio.turn.smart_turn.base_smart_turn', 'SmartTurnParams'),
    ('pipecat.audio.turn.smart_turn.local_smart_turn_v3', 'LocalSmartTurnAnalyzerV3'),
    ('pipecat.audio.utils', 'create_stream_resampler'),
    ('pipecat.audio.utils', 'pcm_to_ulaw'),
    ('pipecat.audio.utils', 'ulaw_to_pcm'),
    ('pipecat.audio.vad.silero', 'SileroVADAnalyzer'),
    ('pipecat.audio.vad.vad_analyzer', 'VADParams'),
    ('pipecat.frames.frames', 'EndWorkerFrame'),
    ('pipecat.frames.frames', 'Frame'),
    ('pipecat.frames.frames', 'InterimTranscriptionFrame'),
    ('pipecat.frames.frames', 'TranscriptionFrame'),
    ('pipecat.frames.frames', 'TTSSpeakFrame'),
    ('pipecat.frames.frames', 'STTUpdateSettingsFrame'),
    ('pipecat.frames.frames', 'TTSUpdateSettingsFrame'),
    ('pipecat.observers.speaking_observer', 'SpeakingObserver'),
    ('pipecat.observers.speaking_observer', 'SpeechEvent'),
    ('pipecat.observers.speaking_observer', 'SpeechEventKind'),
    ('pipecat.observers.user_bot_latency_observer', 'LatencyBreakdown'),
    ('pipecat.observers.user_bot_latency_observer', 'UserBotLatencyObserver'),
    ('pipecat.pipeline.pipeline', 'Pipeline'),
    ('pipecat.pipeline.worker', 'PipelineParams'),
    ('pipecat.pipeline.worker', 'PipelineWorker'),
    ('pipecat.processors.aggregators.llm_context', 'LLMContext'),
    (
        'pipecat.processors.aggregators.llm_response_universal',
        'LLMContextAggregatorPair',
    ),
    (
        'pipecat.processors.aggregators.llm_response_universal',
        'LLMUserAggregatorParams',
    ),
    ('pipecat.processors.frame_processor', 'FrameDirection'),
    ('pipecat.processors.frame_processor', 'FrameProcessorSetup'),
    ('pipecat.runner.types', 'WebSocketRunnerArguments'),
    ('pipecat.serializers.base_serializer', 'FrameSerializer'),
    ('pipecat.serializers.exotel', 'ExotelFrameSerializer'),
    ('pipecat.serializers.twilio', 'TwilioFrameSerializer'),
    ('pipecat.services.llm_service', 'FunctionCallParams'),
    ('pipecat.transcriptions.language', 'Language'),
    ('pipecat.transports.base_transport', 'BaseTransport'),
    ('pipecat.transports.websocket.fastapi', 'FastAPIWebsocketParams'),
    ('pipecat.transports.websocket.fastapi', 'FastAPIWebsocketTransport'),
    ('pipecat.turns.empty_user_turn', 'EmptyUserTurnConfig'),
    ('pipecat.turns.types', 'ProcessFrameResult'),
    ('pipecat.turns.user_mute', 'BaseUserMuteStrategy'),
    ('pipecat.turns.user_mute', 'FunctionCallUserMuteStrategy'),
    ('pipecat.turns.user_mute', 'MuteUntilFirstBotCompleteUserMuteStrategy'),
    ('pipecat.turns.user_start', 'TranscriptionUserTurnStartStrategy'),
    ('pipecat.turns.user_start', 'VADUserTurnStartStrategy'),
    ('pipecat.turns.user_stop', 'TurnAnalyzerUserTurnStopStrategy'),
    ('pipecat.turns.user_turn_strategies', 'UserTurnStrategies'),
    ('pipecat.utils.tracing.setup', 'setup_tracing'),
    ('pipecat.workers.runner', 'WorkerRunner'),
]


@pytest.mark.parametrize(('module', 'name'), PIPECAT_IMPORTS)
def test_pipecat_import_still_exists(module, name):
    assert hasattr(importlib.import_module(module), name)


def test_transport_params_no_longer_carry_vad():
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    # Passing vad_analyzer here would be silently ignored, leaving no VAD at all.
    assert 'vad_analyzer' not in FastAPIWebsocketParams.model_fields


# --- LLM ---------------------------------------------------------------------


def llm_config(llm_type, **extra):
    return {
        'type': llm_type,
        'api_key': 'test-key',
        'llm_model': 'test-model',
        'parameters': {'temperature': 0.3, 'top_p': 0.9},
        **extra,
    }


@pytest.mark.parametrize(
    'config',
    [
        llm_config('openai'),
        llm_config('openai', base_url='https://llm.example.com/v1'),
        llm_config('azure_openai', base_url='https://example.openai.azure.com'),
        llm_config('gemini'),
        llm_config('groq'),
    ],
    ids=['openai', 'openai-compatible', 'azure', 'gemini', 'groq'],
)
async def test_llm_settings_reach_the_service(config):
    llm = LLMServiceFactory.create_llm_service(config)

    # Before 1.x, Groq and Azure dropped these: their constructors always pass
    # their own settings, and the base class only reads params without settings.
    assert llm.settings.model == 'test-model'
    assert llm.settings.temperature == 0.3
    assert llm.settings.top_p == 0.9
    assert llm._function_call_timeout_secs == FUNCTION_CALL_TIMEOUT_SECS


async def test_llm_ignores_keys_its_provider_has_no_setting_for():
    config = llm_config('gemini')
    config['parameters'] = {'max_tokens': 200, 'max_completion_tokens': 300}

    llm = LLMServiceFactory.create_llm_service(config)

    assert llm.settings.max_tokens == 200


@pytest.mark.parametrize(
    ('config', 'sends_developer_role'),
    [
        (llm_config('openai'), True),
        (llm_config('openai', base_url='https://llm.example.com/v1'), False),
        (llm_config('azure_openai', base_url='https://example.openai.azure.com'), True),
        (llm_config('groq'), False),
    ],
    ids=['openai', 'openai-compatible', 'azure', 'groq'],
)
async def test_developer_role_sent_only_where_supported(config, sends_developer_role):
    # Empty-turn recovery injects a developer-role message.
    llm = LLMServiceFactory.create_llm_service(config)

    assert llm.supports_developer_role is sends_developer_role


def test_pipecat_accepts_the_options_call_processing_passes():
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.services.llm_service import LLMService

    worker_params = inspect.signature(PipelineWorker.__init__).parameters
    register_params = inspect.signature(LLMService.register_function).parameters

    assert {'enable_rtvi', 'observers'} <= set(worker_params)
    assert {'cancel_on_interruption', 'timeout_secs'} <= set(register_params)


async def test_openai_service_tier_is_an_init_argument():
    config = llm_config('openai')
    config['parameters']['service_tier'] = 'flex'

    llm = LLMServiceFactory.create_llm_service(config)

    assert llm._service_tier == 'flex'


# --- STT ---------------------------------------------------------------------


def stt_config(provider, **parameters):
    return {
        'provider': provider,
        'api_key': 'test-key',
        'region': 'centralindia',
        'parameters': parameters,
    }


async def test_sarvam_stt_defaults_to_a_current_model():
    stt = STTServiceFactory.create_stt_service(stt_config('sarvam'))

    assert stt.settings.model == 'saaras:v4'


@pytest.mark.parametrize(
    ('stored_model', 'mode'),
    [('saarika:v2.5', 'transcribe'), ('saaras:v2.5', 'translate')],
)
async def test_sarvam_stt_maps_retired_models(stored_model, mode):
    # pipecat 1.x raises ValueError for these at construction.
    stt = STTServiceFactory.create_stt_service(
        stt_config('sarvam', model=stored_model, language='hi')
    )

    assert stt.settings.model == 'saaras:v4'
    assert stt._mode == mode


def test_deepgram_stt_forwards_base_url():
    with patch('call_processing.services.stt_service.DeepgramSTTService') as service:
        STTServiceFactory.create_stt_service(
            stt_config('deepgram', base_url='api.eu.deepgram.com')
        )

    assert service.call_args.kwargs['base_url'] == 'api.eu.deepgram.com'


@pytest.mark.parametrize(
    ('provider', 'model'),
    [('deepgram', 'nova-2'), ('elevenlabs', 'scribe_v2_realtime'), ('azure', None)],
)
async def test_stt_providers_construct(provider, model):
    stt = STTServiceFactory.create_stt_service(stt_config(provider, language='hi'))

    if model:
        assert stt.settings.model == model


@pytest.mark.parametrize('provider', ['deepgram', 'azure', 'sarvam', 'elevenlabs'])
def test_stt_language_update_frame(provider):
    frame = STTServiceFactory.create_language_update_frame(provider, 'hi')

    assert frame is not None and frame.delta is not None


STT_SERVICE_CLASSES = {
    'deepgram': 'DeepgramSTTService',
    'sarvam': 'SarvamSTTService',
    'elevenlabs': 'ElevenLabsRealtimeSTTService',
    'azure': 'AzureSTTService',
}


def created_stt_kwargs(provider, **parameters):
    with patch(
        f'call_processing.services.stt_service.{STT_SERVICE_CLASSES[provider]}'
    ) as service:
        STTServiceFactory.create_stt_service(stt_config(provider, **parameters))
    return service.call_args.kwargs


@pytest.mark.parametrize('provider', list(STT_SERVICE_CLASSES))
def test_stt_ttfs_p99_latency_reaches_the_service(provider):
    assert created_stt_kwargs(provider, ttfs_p99_latency=0.5)['ttfs_p99_latency'] == 0.5


@pytest.mark.parametrize('provider', list(STT_SERVICE_CLASSES))
@pytest.mark.parametrize('stored', [None, 0, -1, 'soon'])
def test_stt_ttfs_p99_latency_left_to_pipecat_unless_valid(provider, stored):
    # Omitted, each pipecat service uses its provider's measured default; a
    # non-positive value would end turns before their transcript lands.
    assert 'ttfs_p99_latency' not in created_stt_kwargs(
        provider, ttfs_p99_latency=stored
    )


def test_deepgram_stt_forwards_endpointing():
    kwargs = created_stt_kwargs('deepgram', endpointing=300, utterance_end_ms=1000)

    assert kwargs['settings'].endpointing == 300
    assert kwargs['settings'].utterance_end_ms == 1000


# --- TTS ---------------------------------------------------------------------


def tts_config(provider, voice_id='voice-1', **parameters):
    return {
        'provider': provider,
        'api_key': 'test-key',
        'region': 'centralindia',
        'voice_id': voice_id,
        'parameters': parameters,
    }


async def test_sarvam_tts_defaults_to_bulbul_v3():
    tts = TTSServiceFactory.create_tts_service(tts_config('sarvam', voice_id='priya'))

    assert tts.settings.model == 'bulbul:v3'
    assert tts.settings.voice == 'priya'


async def test_sarvam_tts_moves_a_v2_config_forward():
    # Sarvam's API rejects bulbul:v2, and none of its speakers exist on v3.
    tts = TTSServiceFactory.create_tts_service(
        tts_config('sarvam', voice_id='anushka', model='bulbul:v2')
    )

    assert tts.settings.model == 'bulbul:v3'
    assert tts.settings.voice != 'anushka'


def test_sarvam_tts_language_update_drops_a_v2_speaker():
    frame = TTSServiceFactory.create_language_update_frame('sarvam', 'hi', 'anushka')

    assert frame.delta.voice != 'anushka'


@pytest.mark.parametrize(('stored', 'requested'), [(None, 8000), (16000, 16000)])
def test_sarvam_tts_requests_telephony_rate_audio(stored, requested):
    parameters = {'sample_rate': stored} if stored else {}
    with patch('call_processing.services.tts_service.SarvamTTSService') as service:
        TTSServiceFactory.create_tts_service(tts_config('sarvam', **parameters))

    assert service.call_args.kwargs['sample_rate'] == requested


@pytest.mark.parametrize(
    ('provider', 'model'),
    [
        ('elevenlabs', 'eleven_turbo_v2_5'),
        ('deepgram', None),
        ('cartesia', 'sonic-2'),
        ('azure', None),
    ],
)
async def test_tts_providers_construct(provider, model):
    tts = TTSServiceFactory.create_tts_service(tts_config(provider, language='hi'))

    assert tts.settings.voice == 'voice-1'
    if model:
        assert tts.settings.model == model


# --- Serializer --------------------------------------------------------------


async def test_smartflo_serializer_reads_the_setup_sample_rate():
    serializer = SmartfloFrameSerializer(stream_sid='stream-1')

    await serializer.setup(SimpleNamespace(audio_in_sample_rate=16000))

    assert serializer._sample_rate == 16000
