"""Gemini Live API as a RealtimeLLM."""

import contextlib
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import errors, types

from flo_ai.llm.base_realtime_llm import (
    AudioOutEvent,
    InterruptedEvent,
    RealtimeEvent,
    RealtimeLLM,
    ToolCallEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from flo_ai.llm.gemini_llm import gemini_function_declaration
from flo_ai.tool.base_tool import Tool
from flo_ai.utils.logger import logger

_START_SENSITIVITY = {
    'high': types.StartSensitivity.START_SENSITIVITY_HIGH,
    'low': types.StartSensitivity.START_SENSITIVITY_LOW,
}
_END_SENSITIVITY = {
    'high': types.EndSensitivity.END_SENSITIVITY_HIGH,
    'low': types.EndSensitivity.END_SENSITIVITY_LOW,
}

# Session settings a per-connection `config` may override.
_SESSION_KEYS = ('voice_name', 'language', 'vad', 'temperature')


def build_vad_config(
    vad: Optional[Dict[str, Any]],
) -> Optional[types.RealtimeInputConfig]:
    """Map a provider-neutral VAD dict onto Gemini's RealtimeInputConfig.

    Keys (all optional):
        disabled: turn off server-side VAD; the client then has to signal
            activity itself
        start_sensitivity / end_sensitivity: 'high' or 'low'
        prefix_padding_ms: speech required before a start of speech is committed
        silence_duration_ms: silence required before an end of speech is committed
        interruptions: False to stop the user barging in over the model
    """
    if not vad:
        return None

    unknown = set(vad) - {
        'disabled',
        'start_sensitivity',
        'end_sensitivity',
        'prefix_padding_ms',
        'silence_duration_ms',
        'interruptions',
    }
    if unknown:
        raise ValueError(f'Unknown VAD settings: {sorted(unknown)}')

    detection: Dict[str, Any] = {}
    if 'disabled' in vad:
        detection['disabled'] = bool(vad['disabled'])
    for key, mapping, field in (
        ('start_sensitivity', _START_SENSITIVITY, 'start_of_speech_sensitivity'),
        ('end_sensitivity', _END_SENSITIVITY, 'end_of_speech_sensitivity'),
    ):
        if key in vad:
            value = str(vad[key]).lower()
            if value not in mapping:
                raise ValueError(f'VAD {key} must be one of {sorted(mapping)}')
            detection[field] = mapping[value]
    for key in ('prefix_padding_ms', 'silence_duration_ms'):
        if key in vad:
            detection[key] = int(vad[key])

    config: Dict[str, Any] = {}
    if detection:
        config['automatic_activity_detection'] = types.AutomaticActivityDetection(
            **detection
        )
    if 'interruptions' in vad:
        config['activity_handling'] = (
            types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
            if vad['interruptions']
            else types.ActivityHandling.NO_INTERRUPTION
        )
    return types.RealtimeInputConfig(**config) if config else None


def _as_response_dict(result: Any) -> Dict[str, Any]:
    """FunctionResponse.response must be a JSON object."""
    if isinstance(result, dict):
        payload = result
    else:
        payload = {'result': result}
    try:
        json.dumps(payload)
        return payload
    except (TypeError, ValueError):
        return {'result': str(result)}


class GeminiLiveLLM(RealtimeLLM):
    """Native-audio Gemini over the Live API websocket.

    Audio in is PCM16 mono 16 kHz, audio out PCM16 mono 24 kHz. Both sides are
    transcribed so the conversation can be persisted as text.
    """

    provider_name = 'gemini'
    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(
        self,
        model: str = 'gemini-2.5-flash-native-audio-preview-09-2025',
        api_key: Optional[str] = None,
        voice_name: Optional[str] = None,
        language: Optional[str] = None,
        vad: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        client: Optional[genai.Client] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, **kwargs)
        self.voice_name = voice_name
        self.language = language
        self.vad = vad
        self.temperature = temperature
        if client is not None:
            self.client = client
        elif base_url:
            self.client = genai.Client(
                api_key=api_key, http_options={'base_url': base_url}
            )
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

        self._session: Any = None
        self._exit_stack: Optional[contextlib.AsyncExitStack] = None
        self._closed = False
        self._tool_names: Dict[str, str] = {}
        self._user_text: List[str] = []
        self._assistant_text: List[str] = []

    def format_tool_for_llm(self, tool: Tool) -> Dict[str, Any]:
        return gemini_function_declaration(tool)

    def build_connect_config(
        self,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> types.LiveConnectConfig:
        overrides = dict(config or {})
        unknown = set(overrides) - set(_SESSION_KEYS)
        if unknown:
            raise ValueError(f'Unknown realtime session settings: {sorted(unknown)}')

        voice_name = overrides.get('voice_name', self.voice_name)
        language = overrides.get('language', self.language)
        vad = overrides.get('vad', self.vad)
        temperature = overrides.get('temperature', self.temperature)

        speech: Dict[str, Any] = {}
        if voice_name:
            speech['voice_config'] = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        if language:
            speech['language_code'] = language

        params: Dict[str, Any] = {
            'response_modalities': [types.Modality.AUDIO],
            'system_instruction': types.Content(parts=[types.Part(text=system_prompt)]),
            'input_audio_transcription': types.AudioTranscriptionConfig(),
            'output_audio_transcription': types.AudioTranscriptionConfig(),
        }
        if speech:
            params['speech_config'] = types.SpeechConfig(**speech)
        if tools:
            params['tools'] = [
                types.Tool(function_declarations=self.format_tools_for_llm(tools))
            ]
        realtime_input = build_vad_config(vad)
        if realtime_input is not None:
            params['realtime_input_config'] = realtime_input
        if temperature is not None:
            params['temperature'] = temperature
        return types.LiveConnectConfig(**params)

    async def connect(
        self,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._session is not None:
            raise RuntimeError('GeminiLiveLLM session is already connected')
        connect_config = self.build_connect_config(system_prompt, tools, config)
        stack = contextlib.AsyncExitStack()
        try:
            self._session = await stack.enter_async_context(
                self.client.aio.live.connect(model=self.model, config=connect_config)
            )
        except BaseException:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._closed = False
        logger.info(f'Gemini Live session connected (model={self.model})')

    def _require_session(self) -> Any:
        if self._session is None or self._closed:
            raise RuntimeError('GeminiLiveLLM session is not connected')
        return self._session

    async def send_audio(self, audio: bytes) -> None:
        await self._require_session().send_realtime_input(
            audio=types.Blob(
                data=audio, mime_type=f'audio/pcm;rate={self.input_sample_rate}'
            )
        )

    async def send_text(self, text: str) -> None:
        await self._require_session().send_realtime_input(text=text)

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        await self._require_session().send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call_id,
                    name=self._tool_names.pop(call_id, None),
                    response=_as_response_dict(result),
                )
            ]
        )

    def _flush(self, role: str) -> Optional[TranscriptEvent]:
        buffer = self._user_text if role == 'user' else self._assistant_text
        text = ''.join(buffer).strip()
        buffer.clear()
        return TranscriptEvent(role=role, text=text, final=True) if text else None

    def _translate(self, message: Any) -> List[RealtimeEvent]:
        """One LiveServerMessage as zero or more RealtimeEvents."""
        events: List[RealtimeEvent] = []

        content = message.server_content
        if content is not None:
            if content.input_transcription and content.input_transcription.text:
                text = content.input_transcription.text
                self._user_text.append(text)
                events.append(TranscriptEvent(role='user', text=text, final=False))

            model_spoke = False
            if content.model_turn and content.model_turn.parts:
                for part in content.model_turn.parts:
                    inline = part.inline_data
                    if inline is not None and inline.data:
                        model_spoke = True
                        events.append(AudioOutEvent(data=inline.data))
            if content.output_transcription and content.output_transcription.text:
                model_spoke = True
                text = content.output_transcription.text
                self._assistant_text.append(text)
                events.append(TranscriptEvent(role='assistant', text=text, final=False))

            # The user's utterance is over once the model starts answering it.
            if model_spoke:
                final = self._flush('user')
                if final:
                    events.insert(0, final)

            if content.interrupted:
                for role in ('user', 'assistant'):
                    final = self._flush(role)
                    if final:
                        events.append(final)
                events.append(InterruptedEvent())

            if content.turn_complete:
                for role in ('user', 'assistant'):
                    final = self._flush(role)
                    if final:
                        events.append(final)
                events.append(TurnCompleteEvent())

        tool_call = message.tool_call
        if tool_call is not None and tool_call.function_calls:
            final = self._flush('user')
            if final:
                events.append(final)
            for call in tool_call.function_calls:
                call_id = call.id or call.name
                self._tool_names[call_id] = call.name
                events.append(
                    ToolCallEvent(
                        id=call_id, name=call.name, arguments=dict(call.args or {})
                    )
                )

        if message.tool_call_cancellation is not None:
            logger.info(
                f'Gemini Live cancelled tool calls: {message.tool_call_cancellation.ids}'
            )
        if message.go_away is not None:
            logger.warning(
                f'Gemini Live session ending soon (time_left={message.go_away.time_left})'
            )
        return events

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        session = self._require_session()
        try:
            while not self._closed:
                received = False
                # receive() returns at the end of each model turn, so it is
                # re-entered for the next one.
                async for message in session.receive():
                    received = True
                    for event in self._translate(message):
                        yield event
                if not received:
                    break
        except errors.APIError as exc:
            if self._closed or getattr(exc, 'code', None) == 1000:
                return
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stack, self._exit_stack = self._exit_stack, None
        self._session = None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:
                logger.warning(f'Error closing Gemini Live session: {exc}')
