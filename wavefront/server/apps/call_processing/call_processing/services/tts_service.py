"""
TTS (Text-to-Speech) service factory

Supports multiple providers: ElevenLabs, Deepgram, Cartesia, Azure, Google, AWS
"""

from typing import Dict, Any
from call_processing.log.logger import logger

# Pipecat TTS services
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
from pipecat.services.azure.tts import AzureTTSService, AzureTTSSettings

# Language for params
from pipecat.transcriptions.language import Language

# Frames
from pipecat.frames.frames import TTSUpdateSettingsFrame


class TTSServiceFactory:
    """Factory for creating TTS service instances from configuration"""

    AZURE_LANGUAGE_MAP = {
        'en': Language.EN_US,
        'hi': Language.HI_IN,
        'ta': Language.TA_IN,
        'te': Language.TE_IN,
        'kn': Language.KN_IN,
        'ml': Language.ML_IN,
        'gu': Language.GU_IN,
        'bn': Language.BN_IN,
        'mr': Language.MR_IN,
        'pa': Language.PA_IN,
        'or': Language.OR_IN,
    }

    @staticmethod
    def create_tts_service(tts_config: Dict[str, Any]):
        """
        Create TTS service from configuration

        Args:
            tts_config: {
                'provider': 'elevenlabs' | 'deepgram' | 'cartesia' | 'azure' | 'google' | 'aws',
                'api_key': 'key',
                'voice_id': 'voice_id',
                'parameters': {
                    'model': 'model_name',
                    'stability': 0.5,
                    'similarity_boost': 0.75,
                    ...
                }
            }

        Returns:
            Pipecat TTS service instance
        """
        provider = tts_config['provider']
        api_key = tts_config['api_key']
        region = tts_config.get('region')
        voice_id = tts_config['voice_id']
        parameters = tts_config.get('parameters', {})

        if parameters is None:
            parameters = {}

        logger.info(f'Creating TTS service: {provider} / voice: {voice_id}')

        if provider == 'elevenlabs':
            return TTSServiceFactory._create_elevenlabs_tts(
                api_key, voice_id, parameters
            )
        elif provider == 'deepgram':
            return TTSServiceFactory._create_deepgram_tts(api_key, voice_id, parameters)
        elif provider == 'cartesia':
            return TTSServiceFactory._create_cartesia_tts(api_key, voice_id, parameters)
        elif provider == 'sarvam':
            return TTSServiceFactory._create_sarvam_tts(api_key, voice_id, parameters)
        elif provider == 'azure':
            return TTSServiceFactory._create_azure_tts(
                api_key, voice_id, region, parameters
            )
        else:
            raise ValueError(f'Unsupported TTS provider: {provider}')

    @staticmethod
    def _create_elevenlabs_tts(api_key: str, voice_id: str, parameters: Dict[str, Any]):
        """Create ElevenLabs TTS service"""
        model = parameters.get('model', 'eleven_turbo_v2_5')

        settings_kwargs: Dict[str, Any] = {
            'voice': voice_id,
            'model': model,
        }

        if 'language' in parameters:
            lang = parameters['language']
            if isinstance(lang, str):
                try:
                    settings_kwargs['language'] = Language(lang)
                except ValueError:
                    logger.warning(f"Unknown language '{lang}', skipping")
            else:
                settings_kwargs['language'] = lang

        if 'stability' in parameters:
            settings_kwargs['stability'] = parameters['stability']
        if 'similarity_boost' in parameters:
            settings_kwargs['similarity_boost'] = parameters['similarity_boost']
        if 'style' in parameters:
            settings_kwargs['style'] = parameters['style']
        if 'use_speaker_boost' in parameters:
            settings_kwargs['use_speaker_boost'] = parameters['use_speaker_boost']
        if 'speed' in parameters:
            settings_kwargs['speed'] = parameters['speed']

        logger.info(
            f"ElevenLabs TTS config: model={model}, "
            f"stability={settings_kwargs.get('stability', 'default')}"
        )

        return ElevenLabsTTSService(
            api_key=api_key,
            settings=ElevenLabsTTSSettings(**settings_kwargs),
        )

    @staticmethod
    def _create_deepgram_tts(api_key: str, voice_id: str, parameters: Dict[str, Any]):
        """Create Deepgram TTS service"""
        kwargs: Dict[str, Any] = {'api_key': api_key}

        if 'base_url' in parameters:
            kwargs['base_url'] = parameters['base_url']
        if 'encoding' in parameters:
            kwargs['encoding'] = parameters['encoding']
        if 'sample_rate' in parameters:
            kwargs['sample_rate'] = parameters['sample_rate']

        logger.info(f'Deepgram TTS config: voice={voice_id}')
        return DeepgramTTSService(
            **kwargs,
            settings=DeepgramTTSSettings(voice=voice_id),
        )

    @staticmethod
    def _create_cartesia_tts(api_key: str, voice_id: str, parameters: Dict[str, Any]):
        """Create Cartesia TTS service"""
        model = parameters.get('model', 'sonic-2')

        settings_kwargs: Dict[str, Any] = {
            'voice': voice_id,
            'model': model,
        }

        if 'language' in parameters:
            lang = parameters['language']
            if isinstance(lang, str):
                try:
                    settings_kwargs['language'] = Language(lang)
                except ValueError:
                    logger.warning(f"Unknown language '{lang}', skipping")
            else:
                settings_kwargs['language'] = lang

        logger.info(f'Cartesia TTS config: voice={voice_id}, model={model}')

        return CartesiaTTSService(
            api_key=api_key,
            settings=CartesiaTTSSettings(**settings_kwargs),
        )

    # Mapping of short language codes to pipecat Language enum for Sarvam
    SARVAM_LANGUAGE_MAP = {
        'bn': Language.BN_IN,
        'en': Language.EN_IN,
        'gu': Language.GU_IN,
        'hi': Language.HI_IN,
        'kn': Language.KN_IN,
        'ml': Language.ML_IN,
        'mr': Language.MR_IN,
        'or': Language.OR_IN,
        'pa': Language.PA_IN,
        'ta': Language.TA_IN,
        'te': Language.TE_IN,
    }

    SARVAM_DEFAULT_MODEL = 'bulbul:v3'
    SARVAM_DEFAULT_SAMPLE_RATE = 8000

    # Sarvam's API rejects bulbul:v2 ("has been deprecated. Please use 'bulbul:v3'"),
    # so a stored v2 config is moved forward instead of failing every call.
    SARVAM_LEGACY_MODELS = {'bulbul:v2': 'bulbul:v3'}

    # bulbul:v2's speakers, none of which exist on bulbul:v3 / v3-beta. Spelled out
    # rather than read from pipecat's SarvamTTSSpeakerV2, which goes with v2 in 2.0.
    SARVAM_V2_ONLY_SPEAKERS = frozenset(
        {'anushka', 'abhilash', 'manisha', 'vidya', 'arya', 'karun', 'hitesh'}
    )

    @staticmethod
    def _sarvam_voice(voice_id: str | None) -> str | None:
        """Return voice_id, or None for a bulbul:v2-only speaker v3 cannot use."""
        if voice_id in TTSServiceFactory.SARVAM_V2_ONLY_SPEAKERS:
            logger.warning(
                f"Sarvam speaker '{voice_id}' only exists on the retired bulbul:v2 "
                f'model; using the bulbul:v3 default speaker. Pick a v3 voice for '
                f'this agent.'
            )
            return None
        return voice_id

    @staticmethod
    def _create_sarvam_tts(api_key: str, voice_id: str, parameters: Dict[str, Any]):
        """Create Sarvam TTS service (WebSocket-based streaming)"""
        model = parameters.get('model') or TTSServiceFactory.SARVAM_DEFAULT_MODEL
        if model in TTSServiceFactory.SARVAM_LEGACY_MODELS:
            legacy_model = model
            model = TTSServiceFactory.SARVAM_LEGACY_MODELS[legacy_model]
            logger.warning(
                f"Sarvam TTS model '{legacy_model}' is no longer served, using "
                f"'{model}'. Update the agent's TTS model."
            )
        # Request telephony-rate audio. Left unset, pipecat asks for the model's
        # default (24 kHz on bulbul:v3) and every chunk is resampled to 8 kHz.
        sample_rate = (
            parameters.get('sample_rate')
            or TTSServiceFactory.SARVAM_DEFAULT_SAMPLE_RATE
        )

        settings_kwargs: Dict[str, Any] = {'model': model}
        voice = TTSServiceFactory._sarvam_voice(voice_id)
        if voice:
            settings_kwargs['voice'] = voice

        if 'language' in parameters and parameters['language']:
            lang_code = parameters['language']
            lang_enum = TTSServiceFactory.SARVAM_LANGUAGE_MAP.get(lang_code)
            if lang_enum:
                settings_kwargs['language'] = lang_enum
            else:
                logger.warning(f"Unknown Sarvam language '{lang_code}', skipping")

        if 'pitch' in parameters:
            settings_kwargs['pitch'] = parameters['pitch']
        if 'pace' in parameters:
            settings_kwargs['pace'] = parameters['pace']
        if 'loudness' in parameters:
            settings_kwargs['loudness'] = parameters['loudness']
        if 'enable_preprocessing' in parameters:
            settings_kwargs['enable_preprocessing'] = parameters['enable_preprocessing']
        if 'temperature' in parameters:
            settings_kwargs['temperature'] = parameters['temperature']

        logger.info(
            f"Sarvam TTS config: voice={voice or 'default'}, model={model}, "
            f'sample_rate={sample_rate}'
        )

        return SarvamTTSService(
            api_key=api_key,
            sample_rate=sample_rate,
            settings=SarvamTTSSettings(**settings_kwargs),
        )

    @staticmethod
    def _create_azure_tts(
        api_key: str, voice_id: str, region: str, parameters: Dict[str, Any]
    ):
        """Create Azure TTS service"""
        if not region:
            raise ValueError("Azure TTS requires 'region' to be set in the TTS config")

        settings_kwargs: Dict[str, Any] = {'voice': voice_id}

        if 'language' in parameters and parameters['language']:
            lang_code = parameters['language']
            lang_enum = TTSServiceFactory.AZURE_LANGUAGE_MAP.get(lang_code)
            if lang_enum:
                settings_kwargs['language'] = lang_enum
            else:
                logger.warning(
                    f"Unknown Azure language '{lang_code}', using service default"
                )

        if 'style' in parameters and parameters['style']:
            settings_kwargs['style'] = parameters['style']
        if 'style_degree' in parameters and parameters['style_degree']:
            settings_kwargs['style_degree'] = parameters['style_degree']
        if 'role' in parameters and parameters['role']:
            settings_kwargs['role'] = parameters['role']
        if 'rate' in parameters and parameters['rate']:
            settings_kwargs['rate'] = parameters['rate']
        if 'pitch' in parameters and parameters['pitch']:
            settings_kwargs['pitch'] = parameters['pitch']
        if 'volume' in parameters and parameters['volume']:
            settings_kwargs['volume'] = parameters['volume']

        kwargs: Dict[str, Any] = {
            'api_key': api_key,
            'region': region,
        }

        if 'sample_rate' in parameters and parameters['sample_rate']:
            kwargs['sample_rate'] = parameters['sample_rate']

        logger.info(f'Azure TTS config: voice={voice_id}, region={region}')

        return AzureTTSService(**kwargs, settings=AzureTTSSettings(**settings_kwargs))

    @staticmethod
    def create_language_update_frame(
        provider: str, lang_code: str, voice_id: str = None
    ):
        """Create TTSUpdateSettingsFrame for a runtime language+voice switch, provider-aware."""
        if provider == 'elevenlabs':
            # Language is implicit from the voice model; only update voice
            if not voice_id:
                logger.warning(
                    'ElevenLabs TTS: no voice_id provided for language update'
                )
                return None
            return TTSUpdateSettingsFrame(delta=ElevenLabsTTSSettings(voice=voice_id))
        elif provider == 'deepgram':
            if not voice_id:
                logger.warning('Deepgram TTS: no voice_id provided for language update')
                return None
            return TTSUpdateSettingsFrame(delta=DeepgramTTSSettings(voice=voice_id))
        elif provider == 'cartesia':
            delta_kwargs: Dict[str, Any] = {}
            if voice_id:
                delta_kwargs['voice'] = voice_id
            try:
                delta_kwargs['language'] = Language(lang_code)
            except ValueError:
                logger.warning(
                    f"Unknown Cartesia language '{lang_code}', skipping language update"
                )
            if not delta_kwargs:
                return None
            return TTSUpdateSettingsFrame(delta=CartesiaTTSSettings(**delta_kwargs))
        elif provider == 'azure':
            delta_kwargs = {}
            if voice_id:
                delta_kwargs['voice'] = voice_id
            lang_enum = TTSServiceFactory.AZURE_LANGUAGE_MAP.get(lang_code)
            if lang_enum:
                delta_kwargs['language'] = lang_enum
            else:
                logger.warning(f"No Azure TTS language mapping for '{lang_code}'")
            if not delta_kwargs:
                return None
            return TTSUpdateSettingsFrame(delta=AzureTTSSettings(**delta_kwargs))
        elif provider == 'sarvam':
            delta_kwargs = {}
            voice = TTSServiceFactory._sarvam_voice(voice_id)
            if voice:
                delta_kwargs['voice'] = voice
            lang_enum = TTSServiceFactory.SARVAM_LANGUAGE_MAP.get(lang_code)
            if lang_enum:
                delta_kwargs['language'] = lang_enum
            else:
                logger.warning(f"No Sarvam TTS language mapping for '{lang_code}'")
            if not delta_kwargs:
                return None
            return TTSUpdateSettingsFrame(delta=SarvamTTSSettings(**delta_kwargs))
        else:
            logger.warning(
                f"TTS provider '{provider}' does not support runtime language updates"
            )
            return None
