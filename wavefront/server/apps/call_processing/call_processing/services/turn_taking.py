"""
User turn-taking configuration for the voice pipeline

Builds the context aggregators with the user side's VAD, turn strategies, mute
strategies and recovery settings taken from an agent's VoiceRuntimeConfig.
"""

from typing import Optional, Tuple

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.user_mute import (
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from call_processing.services.turn_strategies import (
    GuardedTranscriptionUserTurnStartStrategy,
    UnmuteNotifyingUserMuteStrategy,
)
from call_processing.services.voice_runtime_config import VoiceRuntimeConfig


def build_user_aggregator_params(
    cfg: VoiceRuntimeConfig,
) -> Tuple[
    LLMUserAggregatorParams, Optional[GuardedTranscriptionUserTurnStartStrategy]
]:
    """Build the user aggregator params for one call.

    Returns the params and the transcription start strategy, if enabled.
    """
    # Start strategies run in order and the first to fire wins. VAD stays first
    # so voice starts a turn immediately; the transcription strategy only fires
    # for speech VAD missed. A MinWords strategy listed here could not gate
    # barge-in while either of these is present.
    start_strategies = [VADUserTurnStartStrategy()]
    transcription_start = None
    if cfg.transcription_turn_start:
        transcription_start = GuardedTranscriptionUserTurnStartStrategy(
            use_interim=True,
            grace_secs=cfg.post_unmute_transcript_grace_secs,
        )
        start_strategies.append(transcription_start)

    mute_strategies = [
        MuteUntilFirstBotCompleteUserMuteStrategy(),
        FunctionCallUserMuteStrategy(),
    ]
    if transcription_start is not None:
        # Tell the transcription strategy about an unmute inline, before the
        # next frame reaches it.
        mute_strategies = [
            UnmuteNotifyingUserMuteStrategy(
                mute_strategies, on_unmuted=transcription_start.notify_unmuted
            )
        ]

    params = LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=start_strategies,
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=cfg.smart_turn_stop_secs)
                    )
                ),
            ],
        ),
        user_mute_strategies=mute_strategies,
        # VAD runs here since pipecat 1.0 removed TransportParams.vad_analyzer
        # (an unknown transport param is silently ignored). The aggregator
        # broadcasts VAD frames upstream too, so STT still receives them.
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=cfg.vad_confidence,
                start_secs=cfg.vad_start_secs,
                stop_secs=cfg.vad_stop_secs,
                min_volume=cfg.vad_min_volume,
            ),
        ),
        # An empty turn (no transcript) that interrupted the bot gets one
        # "please repeat" inference through a developer-role message. Providers
        # without that role get it as a user message (see LLMServiceFactory).
        empty_user_turn=EmptyUserTurnConfig() if cfg.empty_turn_recovery else None,
        user_turn_stop_timeout=cfg.user_turn_stop_timeout,
    )
    return params, transcription_start


def create_context_aggregator(
    context: LLMContext, cfg: VoiceRuntimeConfig
) -> LLMContextAggregatorPair:
    """Create the call's user/assistant context aggregators."""
    user_params, _ = build_user_aggregator_params(cfg)
    return LLMContextAggregatorPair(context, user_params=user_params)
