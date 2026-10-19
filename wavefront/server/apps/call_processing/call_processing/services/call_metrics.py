"""
Per-call conversation metrics

Collects voice-to-voice latency, barge-ins and turn watchdog events from
pipecat observers, and summarizes them into the stats the call.evaluation span
records.
"""

import statistics
from typing import Any, Dict, List, Optional

from pipecat.observers.speaking_observer import (
    SpeakingObserver,
    SpeechEvent,
    SpeechEventKind,
)
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    UserBotLatencyObserver,
)

# TTS providers whose pipecat service releases text by word timestamp, so an
# interrupted reply puts only the words actually played into the LLM context.
# The others release a whole sentence when it is sent for synthesis, so a
# barge-in can leave text in context that the caller never heard.
WORD_TIMESTAMP_TTS_PROVIDERS = frozenset({'elevenlabs', 'cartesia', 'azure'})


def latency_percentiles(samples: List[float]) -> Dict[str, float]:
    """p50 and p95 of samples, or {} when there are fewer than two."""
    if len(samples) < 2:
        return {}
    cuts = statistics.quantiles(samples, n=100, method='inclusive')
    return {'p50': cuts[49], 'p95': cuts[94]}


class CallMetricsCollector:
    """Turn-taking and latency metrics for one call"""

    def __init__(self, *, tts_provider: str, stt_provider: str, llm_provider: str):
        self._tts_provider = tts_provider
        self._stt_provider = stt_provider
        self._llm_provider = llm_provider
        self._tts_word_timestamps = tts_provider in WORD_TIMESTAMP_TTS_PROVIDERS

        self._v2v_secs: List[float] = []
        self._user_turn_secs: List[float] = []
        self._first_bot_speech_secs: Optional[float] = None
        self._bot_speaking = False
        self._interruption_count = 0
        self._desync_risk_interruptions = 0
        self._stop_timeout_count = 0

        self._latency_observer = UserBotLatencyObserver()
        self._latency_observer.add_event_handler(
            'on_latency_measured', self._on_latency_measured
        )
        self._latency_observer.add_event_handler(
            'on_latency_breakdown', self._on_latency_breakdown
        )
        self._latency_observer.add_event_handler(
            'on_first_bot_speech_latency', self._on_first_bot_speech_latency
        )

        self._speaking_observer = SpeakingObserver()
        self._speaking_observer.add_event_handler(
            'on_speech_event', self._on_speech_event
        )

    @property
    def observers(self) -> list:
        """Observers to attach to the PipelineWorker"""
        return [self._latency_observer, self._speaking_observer]

    def record_stop_timeout(self) -> None:
        """Count a user turn ended by the stop watchdog instead of a transcript"""
        self._stop_timeout_count += 1

    async def _on_latency_measured(self, observer, latency_seconds: float):
        self._v2v_secs.append(latency_seconds)

    async def _on_latency_breakdown(self, observer, breakdown: LatencyBreakdown):
        if breakdown.user_turn_secs is not None:
            self._user_turn_secs.append(breakdown.user_turn_secs)

    async def _on_first_bot_speech_latency(self, observer, latency_seconds: float):
        self._first_bot_speech_secs = latency_seconds

    async def _on_speech_event(self, observer, event: SpeechEvent):
        # A barge-in is a user turn starting while bot audio is playing. Every
        # user turn start broadcasts an interruption, so that event alone can't
        # tell one apart.
        if event.kind == SpeechEventKind.BOT_SPEECH_STARTED:
            self._bot_speaking = True
        elif event.kind == SpeechEventKind.BOT_SPEECH_STOPPED:
            self._bot_speaking = False
        elif event.kind == SpeechEventKind.USER_TURN_STARTED and self._bot_speaking:
            self._interruption_count += 1
            if not self._tts_word_timestamps:
                self._desync_risk_interruptions += 1

    def summary(self) -> Dict[str, Any]:
        """Stats for CallEvaluationService; latency keys only when measured"""
        stats: Dict[str, Any] = {
            'interruption_count': self._interruption_count,
            'desync_risk_interruptions': self._desync_risk_interruptions,
            'stop_timeout_count': self._stop_timeout_count,
            'v2v_samples': len(self._v2v_secs),
            'tts_provider': self._tts_provider,
            'tts_word_timestamps': self._tts_word_timestamps,
            'stt_provider': self._stt_provider,
            'llm_provider': self._llm_provider,
        }
        v2v = latency_percentiles(self._v2v_secs)
        if v2v:
            stats['v2v_p50_secs'] = round(v2v['p50'], 3)
            stats['v2v_p95_secs'] = round(v2v['p95'], 3)
        user_turn = latency_percentiles(self._user_turn_secs)
        if user_turn:
            stats['user_turn_p95_secs'] = round(user_turn['p95'], 3)
        if self._first_bot_speech_secs is not None:
            stats['first_bot_speech_secs'] = round(self._first_bot_speech_secs, 3)
        return stats
