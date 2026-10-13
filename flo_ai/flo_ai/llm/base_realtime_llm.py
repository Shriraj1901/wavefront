"""Speech-to-speech LLM sessions.

`BaseLLM` is request/response: messages in, one completion out. A realtime
model holds a bidirectional connection open for the whole conversation, takes
raw audio, detects turns itself and answers in audio, so it gets an interface
of its own rather than being forced through `generate()`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from flo_ai.tool.base_tool import Tool


@dataclass(frozen=True)
class AudioOutEvent:
    """A chunk of model speech, raw PCM16 mono at the LLM's output_sample_rate."""

    data: bytes
    type: Literal['audio_out'] = field(default='audio_out', init=False)


@dataclass(frozen=True)
class TranscriptEvent:
    """Text of what the user or the model said.

    Providers stream transcription in fragments. Fragments arrive with
    `final=False` for live captions; once an utterance ends, its full text is
    emitted again with `final=True`, which is the one to persist.
    """

    role: Literal['user', 'assistant']
    text: str
    final: bool = True
    type: Literal['transcript'] = field(default='transcript', init=False)


@dataclass(frozen=True)
class ToolCallEvent:
    """The model wants a tool run; answer with `send_tool_result(id, ...)`."""

    id: str
    name: str
    arguments: Dict[str, Any]
    type: Literal['tool_call'] = field(default='tool_call', init=False)


@dataclass(frozen=True)
class ToolResultEvent:
    """A tool call has been executed and its result sent back to the model.

    Emitted by RealtimeVoiceAgent rather than by an LLM, so callers can
    persist what each tool captured. `arguments` are the ones the tool actually
    ran with, session context included.
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    type: Literal['tool_result'] = field(default='tool_result', init=False)


@dataclass(frozen=True)
class InterruptedEvent:
    """The user barged in; drop any model audio still queued for playback."""

    type: Literal['interrupted'] = field(default='interrupted', init=False)


@dataclass(frozen=True)
class TurnCompleteEvent:
    """The model has finished its turn."""

    type: Literal['turn_complete'] = field(default='turn_complete', init=False)


RealtimeEvent = Union[
    AudioOutEvent,
    TranscriptEvent,
    ToolCallEvent,
    ToolResultEvent,
    InterruptedEvent,
    TurnCompleteEvent,
]


class RealtimeLLM(ABC):
    """A live, bidirectional audio session with a speech-to-speech model.

    Lifecycle: `connect()` once, then feed `send_audio()` and consume
    `events()` concurrently until `close()`. Also usable as an async context
    manager, which closes on exit.
    """

    provider_name: str = ''
    # PCM16 mono sample rates the provider expects and produces.
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    async def connect(
        self,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Open the session.

        Args:
            system_prompt: Instructions for the whole conversation
            tools: Tools the model may call; formatted via format_tools_for_llm
            config: Per-session overrides of the constructor's provider settings
                (voice, language, VAD, ...)
        """

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Stream user audio: raw PCM16 mono at `input_sample_rate`."""

    async def send_text(self, text: str) -> None:
        """Send a typed user message into the live conversation."""
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support text input'
        )

    @abstractmethod
    async def send_tool_result(self, call_id: str, result: Any) -> None:
        """Answer a ToolCallEvent with the tool's result."""

    @abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield events from the model until the session ends."""

    @abstractmethod
    async def close(self) -> None:
        """Close the session. Safe to call more than once."""

    @abstractmethod
    def format_tool_for_llm(self, tool: Tool) -> Dict[str, Any]:
        """Format a tool for this provider's function declarations."""

    def format_tools_for_llm(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [self.format_tool_for_llm(tool) for tool in tools]

    async def __aenter__(self) -> 'RealtimeLLM':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
