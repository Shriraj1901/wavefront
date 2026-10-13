"""A voice agent driven by a speech-to-speech RealtimeLLM."""

import asyncio
import inspect
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from flo_ai.llm.base_realtime_llm import (
    RealtimeEvent,
    RealtimeLLM,
    ToolCallEvent,
    ToolResultEvent,
)
from flo_ai.tool.base_tool import Tool
from flo_ai.tool.partial_tool import PartialTool
from flo_ai.utils.logger import logger


def _accepted_context_keys(tool: Tool, session_context: Dict[str, Any]) -> List[str]:
    """Session context keys this tool can be called with.

    A tool that declares neither the key nor **kwargs would raise TypeError if
    handed it, so context is only merged where the function takes it.
    """
    function = (
        tool.base_tool.function if isinstance(tool, PartialTool) else tool.function
    )
    try:
        params = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return [key for key in session_context if key in tool.parameters]
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return list(session_context)
    return [key for key in session_context if key in params or key in tool.parameters]


class RealtimeVoiceAgent:
    """Binds a RealtimeLLM to tools, a system prompt and per-session context.

    Tool calls are executed here and answered on the same session, so callers
    only see the event stream. `session_context` (e.g. user_id, session_id) is
    merged into every tool call's arguments and always wins over what the model
    supplied: those parameters are hidden from the model's view of the tool, so
    a value it emits for them is not one it was ever asked for.

        agent = RealtimeVoiceAgent(llm, tools, system_prompt, {'user_id': uid})
        async with agent:
            async for event in agent.events():
                ...
    """

    def __init__(
        self,
        llm: RealtimeLLM,
        tools: Optional[List[Tool]] = None,
        system_prompt: str = '',
        session_context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_timeout: Optional[float] = 30.0,
    ):
        self.llm = llm
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools or []}
        self.system_prompt = system_prompt
        self.session_context = dict(session_context or {})
        self.config = config
        self.tool_timeout = tool_timeout
        self._context_keys = {
            name: _accepted_context_keys(tool, self.session_context)
            for name, tool in self.tools.items()
        }
        self._started = False

    def _model_view(self, tool: Tool) -> Tool:
        """The tool as declared to the model, without session-context params."""
        hidden = set(self._context_keys[tool.name])
        if not hidden & set(tool.parameters):
            return tool
        return Tool(
            name=tool.name,
            description=tool.description,
            function=tool.function,
            parameters={
                name: info
                for name, info in tool.parameters.items()
                if name not in hidden
            },
        )

    async def start(self) -> None:
        if self._started:
            return
        await self.llm.connect(
            self.system_prompt,
            [self._model_view(tool) for tool in self.tools.values()],
            self.config,
        )
        self._started = True

    async def send_audio(self, audio: bytes) -> None:
        await self.llm.send_audio(audio)

    async def send_text(self, text: str) -> None:
        await self.llm.send_text(text)

    async def close(self) -> None:
        await self.llm.close()

    async def __aenter__(self) -> 'RealtimeVoiceAgent':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def build_arguments(self, call: ToolCallEvent) -> Dict[str, Any]:
        arguments = call.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments else {}
        merged = dict(arguments or {})
        for key in self._context_keys.get(call.name, []):
            merged[key] = self.session_context[key]
        return merged

    async def execute_tool(self, call: ToolCallEvent) -> ToolResultEvent:
        """Run one tool call. Failures become an error result, not an exception,
        so the model can tell the user instead of the session dying."""
        tool = self.tools.get(call.name)
        if tool is None:
            logger.warning(f'Realtime model called unknown tool: {call.name}')
            return ToolResultEvent(
                id=call.id,
                name=call.name,
                arguments=dict(call.arguments or {}),
                error=f'Unknown tool: {call.name}',
            )

        try:
            arguments = self.build_arguments(call)
        except (TypeError, ValueError) as exc:
            return ToolResultEvent(
                id=call.id,
                name=call.name,
                arguments={},
                error=f'Invalid arguments: {exc}',
            )

        try:
            result = await asyncio.wait_for(
                tool.run(inputs=[], variables=None, **arguments),
                timeout=self.tool_timeout,
            )
            return ToolResultEvent(
                id=call.id, name=call.name, arguments=arguments, result=result
            )
        except asyncio.TimeoutError:
            logger.warning(f'Tool {call.name} timed out after {self.tool_timeout}s')
            error = f'Tool {call.name} timed out'
        except Exception as exc:
            logger.error(f'Tool {call.name} failed: {exc}', exc_info=True)
            error = f'Tool {call.name} failed: {exc}'
        return ToolResultEvent(
            id=call.id, name=call.name, arguments=arguments, error=error
        )

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """The LLM's events, with each tool_call followed by its tool_result."""
        if not self._started:
            await self.start()
        async for event in self.llm.events():
            yield event
            if isinstance(event, ToolCallEvent):
                outcome = await self.execute_tool(event)
                await self.llm.send_tool_result(
                    event.id,
                    {'error': outcome.error}
                    if outcome.error is not None
                    else outcome.result,
                )
                yield outcome
