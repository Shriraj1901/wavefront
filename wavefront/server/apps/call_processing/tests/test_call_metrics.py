"""Per-call latency and barge-in metrics, and the call.evaluation span they feed."""

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pipecat.observers.speaking_observer import (
    SpeakingObserver,
    SpeechEvent,
    SpeechEventKind,
)
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    UserBotLatencyObserver,
)

from call_processing.services import call_evaluation_service
from call_processing.services.call_evaluation_service import CallEvaluationService
from call_processing.services.call_metrics import (
    CallMetricsCollector,
    latency_percentiles,
)


def collector(tts_provider='elevenlabs'):
    return CallMetricsCollector(
        tts_provider=tts_provider, stt_provider='deepgram', llm_provider='openai'
    )


def speech(kind):
    return SpeechEvent(kind=kind, timestamp=0.0)


@pytest.mark.parametrize('samples', [[], [0.8]])
def test_percentiles_need_two_samples(samples):
    assert latency_percentiles(samples) == {}


def test_percentiles():
    result = latency_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result['p50'] == 3.0
    assert 4.0 < result['p95'] <= 5.0


def test_collector_attaches_a_latency_and_a_speaking_observer():
    observers = collector().observers

    assert [type(o) for o in observers] == [UserBotLatencyObserver, SpeakingObserver]


async def test_user_turn_during_bot_speech_is_a_barge_in():
    metrics = collector()

    await metrics._on_speech_event(None, speech(SpeechEventKind.USER_TURN_STARTED))
    await metrics._on_speech_event(None, speech(SpeechEventKind.BOT_SPEECH_STARTED))
    await metrics._on_speech_event(None, speech(SpeechEventKind.USER_TURN_STARTED))
    await metrics._on_speech_event(None, speech(SpeechEventKind.BOT_SPEECH_STOPPED))
    await metrics._on_speech_event(None, speech(SpeechEventKind.USER_TURN_STARTED))

    summary = metrics.summary()
    assert summary['interruption_count'] == 1
    assert summary['desync_risk_interruptions'] == 0


async def test_barge_in_on_a_sentence_level_tts_is_a_desync_risk():
    metrics = collector(tts_provider='sarvam')

    await metrics._on_speech_event(None, speech(SpeechEventKind.BOT_SPEECH_STARTED))
    await metrics._on_speech_event(None, speech(SpeechEventKind.USER_TURN_STARTED))

    summary = metrics.summary()
    assert summary['interruption_count'] == 1
    assert summary['desync_risk_interruptions'] == 1
    assert summary['tts_word_timestamps'] is False


async def test_summary_reports_latency_only_once_measured():
    metrics = collector()
    assert 'v2v_p50_secs' not in metrics.summary()

    for seconds in (0.9, 1.1, 1.3):
        await metrics._on_latency_measured(None, seconds)
    await metrics._on_latency_breakdown(None, LatencyBreakdown(user_turn_secs=0.2))
    await metrics._on_latency_breakdown(None, LatencyBreakdown(user_turn_secs=0.4))
    await metrics._on_first_bot_speech_latency(None, 1.5)
    metrics.record_stop_timeout()

    summary = metrics.summary()
    assert summary['v2v_samples'] == 3
    assert summary['v2v_p50_secs'] == 1.1
    assert 'v2v_p95_secs' in summary
    assert 'user_turn_p95_secs' in summary
    assert summary['first_bot_speech_secs'] == 1.5
    assert summary['stop_timeout_count'] == 1
    assert summary['llm_provider'] == 'openai'


async def test_evaluation_span_carries_the_collected_metrics():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    stats = {
        'user_turns': 2,
        'assistant_turns': 2,
        'tool_calls_count': 1,
        'language_switch_count': 0,
        **collector(tts_provider='sarvam').summary(),
    }

    with patch.object(call_evaluation_service, 'tracer', provider.get_tracer('test')):
        await CallEvaluationService.record_call_metrics(
            call_id='call-1',
            agent_config={'id': 'agent-1', 'name': 'Agent'},
            call_outcome='completed',
            transcript_log=[],
            stats=stats,
        )

    (span,) = exporter.get_finished_spans()
    assert span.attributes['call.interruption_count'] == 0
    assert span.attributes['call.tool_calls_count'] == 1
    assert span.attributes['tts.provider'] == 'sarvam'
    assert span.attributes['tts.word_timestamps'] is False
    assert span.attributes['call.v2v_samples'] == 0
    assert 'call.v2v_p50_secs' not in span.attributes
