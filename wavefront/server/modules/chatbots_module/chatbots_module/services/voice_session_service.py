"""Browser voice sessions: a chatbot spoken to through a native realtime model.

Web voice runs here, speech-to-speech over the provider's realtime API.
Telephony (Twilio/Exotel/Smartflo) stays on Pipecat in apps/call_processing,
which owns the phone-specific concerns (codecs, SIP/media streams, DTMF,
call control) that a browser session does not have.
"""

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from common_module.log.logger import logger
from db_repo_module.models.chat_session import ChatSession
from db_repo_module.models.chatbot import Chatbot
from db_repo_module.models.llm_inference_config import LlmInferenceConfig
from flo_ai.agent.realtime_voice_agent import RealtimeVoiceAgent
from flo_ai.llm.base_realtime_llm import (
    AudioOutEvent,
    InterruptedEvent,
    RealtimeEvent,
    RealtimeLLM,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from flo_ai.llm.gemini_live_llm import GeminiLiveLLM
from flo_ai.tool.base_tool import Tool
from flo_ai.tool.partial_tool import PartialTool
from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect
from tools_module.utils.api_service_tool_loader import load_api_service_tool

from chatbots_module.models.voice_schemas import (
    VoiceCaptureConfig,
    VoiceConfig,
    VoiceSessionOverrides,
)
from chatbots_module.services.chat_session_service import ChatSessionService
from chatbots_module.utils.constants import ROLE_TOOL
from chatbots_module.utils.llm_factory import resolve_temperature

# How long queued transcripts may take to land after the socket closes.
PERSIST_DRAIN_SECONDS = 10.0
CAPTURE_TOOL = 'datasource_insert_rows'


class VoiceSessionError(Exception):
    """The chatbot cannot hold a voice session. The message is client-safe."""


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def event_payload(event: RealtimeEvent) -> Optional[dict]:
    """The JSON frame sent to the browser for a non-audio event.

    Tool arguments and results are not sent: they can carry rows from a
    datasource or values filled in by the admin, and the browser only needs to
    know that a tool ran.
    """
    if isinstance(event, TranscriptEvent):
        return {
            'type': 'transcript',
            'role': event.role,
            'text': event.text,
            'final': event.final,
        }
    if isinstance(event, ToolCallEvent):
        return {'type': 'tool_call', 'id': event.id, 'name': event.name}
    if isinstance(event, ToolResultEvent):
        return {
            'type': 'tool_result',
            'id': event.id,
            'name': event.name,
            'ok': event.error is None,
        }
    if isinstance(event, (InterruptedEvent, TurnCompleteEvent)):
        return {'type': event.type}
    return None


@dataclass
class VoiceSession:
    agent: RealtimeVoiceAgent
    session: ChatSession
    recorder: 'VoiceTranscriptRecorder'
    input_sample_rate: int
    output_sample_rate: int


class VoiceTranscriptRecorder:
    """Persists a voice session against its user.

    Final transcripts become user/assistant chat messages and tool results
    become `tool` messages, so the session reads back like a text thread. With
    a `capture` block configured, each is also inserted into that datasource
    table through the existing datasource_insert_rows tool, tagged with
    user_id, session_id and chatbot_id.
    """

    def __init__(
        self,
        chat_session_service: ChatSessionService,
        session: ChatSession,
        user_id: uuid.UUID,
        capture: Optional[VoiceCaptureConfig] = None,
        insert_rows: Optional[Callable] = None,
    ):
        self.chat_session_service = chat_session_service
        self.session = session
        self.user_id = user_id
        self.capture = capture if insert_rows is not None else None
        self.insert_rows = insert_rows
        self._titled = False
        if capture is not None and insert_rows is None:
            logger.warning(
                f'Voice capture configured but {CAPTURE_TOOL} is unavailable; '
                'transcripts are stored in chat messages only'
            )

    async def record(self, event: RealtimeEvent) -> None:
        if isinstance(event, TranscriptEvent):
            if not event.final or not event.text.strip():
                return
            await self.chat_session_service.add_message(
                session_id=self.session.id,
                role=event.role,
                content=event.text,
                metadata={'source': 'voice'},
            )
            if event.role == 'user' and not self._titled:
                self._titled = True
                await self.chat_session_service.ensure_title(self.session, event.text)
            await self._capture(
                {'kind': 'transcript', 'role': event.role, 'content': event.text}
            )
        elif isinstance(event, ToolResultEvent):
            details = {
                'tool_call_id': event.id,
                'name': event.name,
                'arguments': _jsonable(event.arguments),
                'result': _jsonable(event.result),
                'error': event.error,
            }
            await self.chat_session_service.add_message(
                session_id=self.session.id,
                role=ROLE_TOOL,
                content=json.dumps(details),
                metadata={'source': 'voice', **details},
            )
            await self._capture(
                {
                    'kind': 'tool',
                    'role': ROLE_TOOL,
                    'content': json.dumps(details),
                    'tool_name': event.name,
                }
            )

    async def _capture(self, row: dict) -> None:
        if self.capture is None:
            return
        outcome = await self.insert_rows(
            datasource_id=self.capture.datasource_id,
            table_name=self.capture.table_name,
            data={
                'user_id': str(self.user_id),
                'session_id': str(self.session.id),
                'chatbot_id': str(self.session.chatbot_id),
                'created_at': datetime.now(timezone.utc).isoformat(),
                **row,
            },
            single_row=True,
        )
        if not str(outcome).startswith('Inserted'):
            logger.warning(f'Voice capture insert failed: {outcome}')


class VoiceSessionService:
    def __init__(
        self,
        llm_inference_config_service,
        chat_session_service: ChatSessionService,
        tool_loader=None,
        api_services_manager=None,
    ):
        self.llm_inference_config_service = llm_inference_config_service
        self.chat_session_service = chat_session_service
        self.tool_loader = tool_loader
        self.api_services_manager = api_services_manager

    @staticmethod
    def voice_config(chatbot: Chatbot) -> VoiceConfig:
        raw = (chatbot.config or {}).get('voice')
        if raw is None:
            raise VoiceSessionError('This chatbot is not configured for voice')
        try:
            return VoiceConfig.model_validate(raw)
        except ValidationError as exc:
            # Validated on save, so only reachable for rows written before.
            logger.error(f'Invalid voice config on chatbot {chatbot.id}: {exc}')
            raise VoiceSessionError('This chatbot has an invalid voice configuration')

    async def build_llm(self, chatbot: Chatbot, voice: VoiceConfig) -> RealtimeLLM:
        config_dict = await self.llm_inference_config_service.get_config(
            chatbot.llm_config_id
        )
        if not config_dict:
            raise VoiceSessionError('The chatbot has no LLM configuration')
        llm_config = LlmInferenceConfig(**config_dict)
        if llm_config.type != 'gemini':
            raise VoiceSessionError(
                'Voice sessions need a Gemini LLM configuration on the chatbot'
            )
        kwargs: dict[str, Any] = {}
        if voice.model:
            kwargs['model'] = voice.model
        return GeminiLiveLLM(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url or None,
            voice_name=voice.voice_name,
            language=voice.language,
            vad=voice.vad,
            temperature=resolve_temperature(chatbot.config),
            **kwargs,
        )

    async def load_tools(self, voice: VoiceConfig) -> List[Tool]:
        tools: List[Tool] = []
        for spec in voice.tools:
            tool = self.tool_loader.load_tool(spec.name) if self.tool_loader else None
            if tool is None:
                tool = await load_api_service_tool(spec.name, self.api_services_manager)
            if tool is None:
                raise VoiceSessionError(f'Voice tool not found: {spec.name}')
            if spec.prefilled_params or spec.description:
                tool = PartialTool(
                    tool,
                    spec.prefilled_params,
                    name_override=spec.name,
                    description_override=spec.description or tool.description,
                )
            tools.append(tool)
        return tools

    def _insert_rows(self) -> Optional[Callable]:
        if self.tool_loader is None:
            return None
        return self.tool_loader.function_registry.get(CAPTURE_TOOL)

    async def open(
        self,
        chatbot: Chatbot,
        user_id: uuid.UUID,
        session_id: Optional[uuid.UUID] = None,
        overrides: Optional[VoiceSessionOverrides] = None,
    ) -> VoiceSession:
        """Resolve everything a voice session needs, before any audio flows.

        Raises VoiceSessionError, ChatSessionNotFoundError or
        UnknownChatUserError. The realtime connection is not opened here.
        """
        voice = self.voice_config(chatbot)
        llm = await self.build_llm(chatbot, voice)
        tools = await self.load_tools(voice)

        if session_id is not None:
            session = await self.chat_session_service.get_owned_session(
                session_id, user_id
            )
            if session.chatbot_id != chatbot.id:
                raise VoiceSessionError('The session belongs to a different chatbot')
        else:
            session = await self.chat_session_service.create_session(
                chatbot=chatbot, user_id=user_id
            )

        agent = RealtimeVoiceAgent(
            llm,
            tools,
            system_prompt=session.system_prompt_snapshot,
            session_context={'user_id': str(user_id), 'session_id': str(session.id)},
            config=overrides.model_dump(exclude_none=True) if overrides else None,
        )
        recorder = VoiceTranscriptRecorder(
            self.chat_session_service,
            session,
            user_id,
            voice.capture,
            self._insert_rows() if voice.capture else None,
        )
        return VoiceSession(
            agent=agent,
            session=session,
            recorder=recorder,
            input_sample_rate=llm.input_sample_rate,
            output_sample_rate=llm.output_sample_rate,
        )

    async def bridge(
        self, websocket: WebSocket, agent: RealtimeVoiceAgent, recorder
    ) -> None:
        """Pump browser audio into the agent and agent events back out.

        Binary frames in are PCM16 user audio; text frames are JSON commands
        (`{"type": "text", "text": ...}`, `{"type": "stop"}`). Binary frames out
        are PCM16 model audio; text frames are event_payload() JSON.

        Persistence runs on its own task so a slow insert never stalls audio.
        Returns when either side ends; raises if the model connection fails.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def browser_to_agent() -> None:
            while True:
                message = await websocket.receive()
                if message['type'] == 'websocket.disconnect':
                    return
                if message.get('bytes'):
                    await agent.send_audio(message['bytes'])
                    continue
                text = message.get('text')
                if not text:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    command = None
                kind = command.get('type') if isinstance(command, dict) else None
                if kind == 'stop':
                    return
                if kind == 'text' and isinstance(command.get('text'), str):
                    await agent.send_text(command['text'])
                else:
                    await websocket.send_json(
                        {'type': 'error', 'message': 'Unrecognised frame'}
                    )

        async def agent_to_browser() -> None:
            async for event in agent.events():
                if isinstance(event, AudioOutEvent):
                    await websocket.send_bytes(event.data)
                    continue
                payload = event_payload(event)
                if payload is not None:
                    await websocket.send_json(payload)
                if isinstance(event, (TranscriptEvent, ToolResultEvent)):
                    queue.put_nowait(event)

        async def persist() -> None:
            while True:
                event = await queue.get()
                try:
                    await recorder.record(event)
                except Exception:
                    logger.exception('Failed to persist voice session event')
                finally:
                    queue.task_done()

        persister = asyncio.create_task(persist())
        pumps = [
            asyncio.create_task(browser_to_agent()),
            asyncio.create_task(agent_to_browser()),
        ]
        try:
            done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(queue.join(), PERSIST_DRAIN_SECONDS)
            persister.cancel()
            await asyncio.gather(persister, return_exceptions=True)
