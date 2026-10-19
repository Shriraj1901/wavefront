"""Per-agent turn-taking config and the user aggregator it builds.

An agent with no voice_runtime config must get exactly the turn-taking stack
the service hard-coded before it was configurable, plus the transcription
fallback and empty-turn recovery. Constructing the VAD and Smart Turn analyzers
loads their bundled ONNX models; nothing here touches the network.
"""

from unittest.mock import patch

import pytest
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute import (
    BaseUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

from call_processing.services import turn_taking
from call_processing.services.turn_strategies import (
    GuardedTranscriptionUserTurnStartStrategy,
    UnmuteNotifyingUserMuteStrategy,
)
from call_processing.services.turn_taking import (
    build_user_aggregator_params,
    create_context_aggregator,
)
from call_processing.services.voice_runtime_config import (
    VoiceRuntimeConfig,
    parse_voice_runtime_config,
)


def agent_with(voice_runtime):
    return {'id': 'agent-1', 'conversation_config': {'voice_runtime': voice_runtime}}


# --- Config parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    'agent_config',
    [
        {'id': 'agent-1'},
        {'id': 'agent-1', 'conversation_config': None},
        {'id': 'agent-1', 'conversation_config': {}},
        {'id': 'agent-1', 'conversation_config': {'other': 1}},
        {'id': 'agent-1', 'conversation_config': '{"voice_runtime": {}}'},
        agent_with(None),
    ],
    ids=['absent', 'null', 'empty', 'unrelated-keys', 'unparsed-string', 'null-key'],
)
def test_missing_voice_runtime_gives_defaults(agent_config):
    assert parse_voice_runtime_config(agent_config) == VoiceRuntimeConfig()


def test_voice_runtime_values_are_read():
    cfg = parse_voice_runtime_config(
        agent_with({'vad_confidence': 0.8, 'transcription_turn_start': False})
    )

    assert cfg.vad_confidence == 0.8
    assert cfg.transcription_turn_start is False
    assert cfg.vad_min_volume == 0.6


@pytest.mark.parametrize(
    'voice_runtime',
    [{'vad_confidence': 'high'}, {'vad_confidence': 1.5}, {'vad_stop_secs': 0}, []],
    ids=['wrong-type', 'out-of-range', 'zero-secs', 'not-an-object'],
)
def test_invalid_voice_runtime_falls_back_to_defaults(voice_runtime):
    assert parse_voice_runtime_config(agent_with(voice_runtime)) == VoiceRuntimeConfig()


def test_unknown_voice_runtime_keys_are_ignored():
    cfg = parse_voice_runtime_config(
        agent_with({'not_a_knob': 1, 'vad_stop_secs': 0.3})
    )

    assert cfg.vad_stop_secs == 0.3


# --- Aggregator params -------------------------------------------------------


def test_default_params_keep_the_previous_turn_taking_stack():
    with patch.object(
        turn_taking, 'LocalSmartTurnAnalyzerV3', wraps=LocalSmartTurnAnalyzerV3
    ) as smart_turn:
        params, transcription_start = build_user_aggregator_params(VoiceRuntimeConfig())

    vad = params.vad_analyzer.params
    assert (vad.confidence, vad.start_secs, vad.stop_secs, vad.min_volume) == (
        0.7,
        0.2,
        0.2,
        0.6,
    )
    assert smart_turn.call_args.kwargs['params'].stop_secs == 3.0

    start = params.user_turn_strategies.start
    assert [type(s) for s in start] == [
        VADUserTurnStartStrategy,
        GuardedTranscriptionUserTurnStartStrategy,
    ]
    assert start[1] is transcription_start
    assert [type(s) for s in params.user_turn_strategies.stop] == [
        TurnAnalyzerUserTurnStopStrategy
    ]
    (mute,) = params.user_mute_strategies
    assert isinstance(mute, UnmuteNotifyingUserMuteStrategy)
    assert [type(s) for s in mute.strategies] == [
        MuteUntilFirstBotCompleteUserMuteStrategy,
        FunctionCallUserMuteStrategy,
    ]
    assert isinstance(params.empty_user_turn, EmptyUserTurnConfig)
    assert params.empty_user_turn.max_consecutive_recoveries == 1
    assert params.user_turn_stop_timeout == 5.0
    # Idle detection is not enabled yet.
    assert params.user_idle_timeout == 0


def test_transcription_start_and_recovery_can_be_turned_off():
    params, transcription_start = build_user_aggregator_params(
        VoiceRuntimeConfig(transcription_turn_start=False, empty_turn_recovery=False)
    )

    assert [type(s) for s in params.user_turn_strategies.start] == [
        VADUserTurnStartStrategy
    ]
    assert transcription_start is None
    assert [type(s) for s in params.user_mute_strategies] == [
        MuteUntilFirstBotCompleteUserMuteStrategy,
        FunctionCallUserMuteStrategy,
    ]
    assert params.empty_user_turn is None


def test_configured_values_reach_the_params():
    params, _ = build_user_aggregator_params(
        VoiceRuntimeConfig(vad_min_volume=0.4, user_turn_stop_timeout=3.0)
    )

    assert params.vad_analyzer.params.min_volume == 0.4
    assert params.user_turn_stop_timeout == 3.0


def test_context_aggregator_uses_the_configured_params():
    context_aggregator = create_context_aggregator(
        LLMContext(), VoiceRuntimeConfig(vad_min_volume=0.4)
    )

    assert context_aggregator.user() is not None
    assert context_aggregator.assistant() is not None


class FixedMute(BaseUserMuteStrategy):
    def __init__(self):
        super().__init__()
        self.muted = False
        self.frames = 0

    async def process_frame(self, frame):
        self.frames += 1
        return self.muted


async def test_unmute_is_reported_inline_on_the_muted_to_unmuted_edge():
    first, second = FixedMute(), FixedMute()
    unmutes = []
    mute = UnmuteNotifyingUserMuteStrategy(
        [first, second], on_unmuted=lambda: unmutes.append(1)
    )
    frame = interim()

    assert await mute.process_frame(frame) is False
    first.muted = True
    assert await mute.process_frame(frame) is True
    second.muted = True
    first.muted = False
    assert await mute.process_frame(frame) is True
    assert unmutes == []
    second.muted = False
    assert await mute.process_frame(frame) is False
    assert unmutes == [1]
    assert await mute.process_frame(frame) is False
    assert unmutes == [1]
    # Every strategy saw every frame.
    assert first.frames == second.frames == 5


# --- Post-unmute transcript guard --------------------------------------------


class StrategyEvents:
    def __init__(self, strategy):
        self.turns_started = 0
        self.resets = 0
        strategy.add_event_handler('on_user_turn_started', self._turn_started)
        strategy.add_event_handler('on_reset_aggregation', self._reset)

    async def _turn_started(self, strategy, params):
        self.turns_started += 1

    async def _reset(self, strategy):
        self.resets += 1


def final(text='hello'):
    return TranscriptionFrame(text, 'user', '2026-01-01T00:00:00Z', finalized=True)


def interim(text='hel'):
    return InterimTranscriptionFrame(text, 'user', '2026-01-01T00:00:00Z')


async def test_transcript_starts_a_turn_outside_the_grace_window():
    strategy = GuardedTranscriptionUserTurnStartStrategy(grace_secs=60)
    events = StrategyEvents(strategy)

    assert await strategy.process_frame(final()) == ProcessFrameResult.STOP
    assert events.turns_started == 1
    assert events.resets == 0


async def test_transcript_of_muted_speech_is_dropped_after_unmute():
    strategy = GuardedTranscriptionUserTurnStartStrategy(grace_secs=60)
    events = StrategyEvents(strategy)
    strategy.notify_unmuted()

    assert await strategy.process_frame(interim()) == ProcessFrameResult.CONTINUE
    assert events.resets == 0
    assert await strategy.process_frame(final()) == ProcessFrameResult.CONTINUE
    assert events.resets == 1
    assert events.turns_started == 0


async def test_transcript_after_a_new_turn_is_not_dropped():
    strategy = GuardedTranscriptionUserTurnStartStrategy(grace_secs=60)
    events = StrategyEvents(strategy)
    strategy.notify_unmuted()
    # VAD heard the user after unmute and started a turn.
    await strategy.handle_user_turn_started()

    assert await strategy.process_frame(final()) == ProcessFrameResult.STOP
    assert events.resets == 0


async def test_turn_open_across_the_mute_keeps_its_transcripts():
    strategy = GuardedTranscriptionUserTurnStartStrategy(grace_secs=60)
    events = StrategyEvents(strategy)
    await strategy.handle_user_turn_started()
    strategy.notify_unmuted()

    await strategy.process_frame(final())
    assert events.resets == 0

    # Once that turn ends, the next unmute opens the window again.
    await strategy.handle_user_turn_stopped()
    strategy.notify_unmuted()
    assert await strategy.process_frame(final()) == ProcessFrameResult.CONTINUE
    assert events.resets == 1


async def test_transcript_after_the_grace_window_starts_a_turn():
    strategy = GuardedTranscriptionUserTurnStartStrategy(grace_secs=0)
    events = StrategyEvents(strategy)
    strategy.notify_unmuted()

    assert await strategy.process_frame(final()) == ProcessFrameResult.STOP
    assert events.turns_started == 1
