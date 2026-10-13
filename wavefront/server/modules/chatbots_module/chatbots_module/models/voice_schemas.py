"""Voice settings for a chatbot, and the frames a browser voice session opens with.

A chatbot speaks when its `config` carries a `voice` block:

    {
      "voice": {
        "model": "gemini-2.5-flash-native-audio-preview-09-2025",
        "voice_name": "Puck",
        "language": "en-US",
        "vad": {"silence_duration_ms": 600},
        "tools": [
          {"name": "datasource_execute_query",
           "prefilled_params": {"datasource_id": "...", "query_id": "..."}}
        ],
        "capture": {"datasource_id": "...", "table_name": "voice_captures"}
      }
    }
"""

import uuid
from typing import Any, Dict, List, Literal, Optional

from flo_ai.llm.gemini_live_llm import build_vad_config
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_vad(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    build_vad_config(value)
    return value


class VoiceToolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description='Registry tool or API service tool name')
    prefilled_params: Dict[str, Any] = Field(
        default_factory=dict,
        description='Arguments fixed by the admin and hidden from the model',
    )
    description: Optional[str] = Field(
        None, description="Replaces the tool's description shown to the model"
    )


class VoiceCaptureConfig(BaseModel):
    """Datasource table that every final transcript and tool result is also
    written to, keyed by user_id, through the datasource_insert_rows tool."""

    model_config = ConfigDict(extra='forbid')

    datasource_id: str
    table_name: str


class VoiceSessionOverrides(BaseModel):
    """What the browser may choose per session. Tools, prompt and capture are
    the admin's, so they are not here."""

    model_config = ConfigDict(extra='forbid')

    voice_name: Optional[str] = None
    language: Optional[str] = None
    vad: Optional[Dict[str, Any]] = None

    @field_validator('vad')
    @classmethod
    def validate_vad(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return _validate_vad(value)


class VoiceConfig(VoiceSessionOverrides):
    model: Optional[str] = Field(
        None, description='Realtime model; defaults to the provider default'
    )
    tools: List[VoiceToolConfig] = Field(default_factory=list)
    capture: Optional[VoiceCaptureConfig] = None

    @field_validator('tools')
    @classmethod
    def unique_tool_names(cls, tools: List[VoiceToolConfig]) -> List[VoiceToolConfig]:
        names = [tool.name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f'Duplicate voice tools: {duplicates}')
        return tools


class VoiceStartFrame(BaseModel):
    """First frame of a voice WebSocket.

    The token travels here rather than in an Authorization header because
    browsers cannot set headers on a WebSocket handshake, and a query string
    would put it in access logs.
    """

    model_config = ConfigDict(extra='forbid')

    type: Literal['start']
    token: str = Field(..., min_length=1)
    session_id: Optional[uuid.UUID] = Field(
        None, description='Continue this chat session instead of opening a new one'
    )
    config: Optional[VoiceSessionOverrides] = None
