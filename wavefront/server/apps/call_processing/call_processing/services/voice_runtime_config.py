"""
Per-agent voice runtime tuning

Read from the voice agent's conversation_config column, under the
'voice_runtime' key so it cannot collide with anything else stored in that
free-form JSON. Every field defaults to the behaviour an agent gets without
the key, and parsing never fails a call: a bad value logs a warning and the
defaults apply.
"""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from call_processing.log.logger import logger

VOICE_RUNTIME_KEY = 'voice_runtime'


class VoiceRuntimeConfig(BaseModel):
    """Turn-taking knobs for one voice agent"""

    model_config = ConfigDict(extra='ignore')

    # Silero VAD. Pipecat's defaults; stop_secs stays short because Smart Turn,
    # not VAD silence, decides when the user has finished.
    vad_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    vad_start_secs: float = Field(default=0.2, gt=0.0)
    vad_stop_secs: float = Field(default=0.2, gt=0.0)
    vad_min_volume: float = Field(default=0.6, ge=0.0, le=1.0)

    # Longest an "incomplete" Smart Turn prediction may hold the floor.
    smart_turn_stop_secs: float = Field(default=3.0, gt=0.0)

    # Watchdog that ends a turn which never produced a transcript (noise, STT
    # stall). It is also how long an empty turn takes to be recovered from.
    user_turn_stop_timeout: float = Field(default=5.0, gt=0.0)

    # Let a transcript start a turn when VAD misses soft or far-field speech.
    transcription_turn_start: bool = True

    # A transcript arriving this soon after the user is unmuted, with no turn
    # started since, is speech made while muted and is dropped.
    post_unmute_transcript_grace_secs: float = Field(default=1.0, ge=0.0)

    # After a barge-in that produced no transcript (a cough, a click), ask the
    # user to repeat instead of leaving the bot silent.
    empty_turn_recovery: bool = True


def parse_voice_runtime_config(agent_config: Dict[str, Any]) -> VoiceRuntimeConfig:
    """Build the agent's VoiceRuntimeConfig; never raises."""
    agent_id = agent_config.get('id')
    conversation_config = agent_config.get('conversation_config')
    if not isinstance(conversation_config, dict):
        return VoiceRuntimeConfig()

    raw = conversation_config.get(VOICE_RUNTIME_KEY)
    if raw is None:
        return VoiceRuntimeConfig()
    if not isinstance(raw, dict):
        logger.warning(
            f"Agent {agent_id}: conversation_config['{VOICE_RUNTIME_KEY}'] is not "
            f'an object, using defaults'
        )
        return VoiceRuntimeConfig()

    unknown_keys = sorted(set(raw) - set(VoiceRuntimeConfig.model_fields))
    if unknown_keys:
        logger.warning(
            f'Agent {agent_id}: ignoring unknown {VOICE_RUNTIME_KEY} keys: {unknown_keys}'
        )

    try:
        return VoiceRuntimeConfig.model_validate(raw)
    except ValidationError as e:
        logger.warning(
            f'Agent {agent_id}: invalid {VOICE_RUNTIME_KEY} config, using defaults: '
            f'{e.errors(include_url=False)}'
        )
        return VoiceRuntimeConfig()
