"""The set of tools the assistant exposes, and dispatch to them by name."""

from typing import Any, Collection, Dict, List, Optional

from flo_ai.tool.base_tool import Tool

from agents_module.services.workflow_assistant.tools.base import (
    AssistantTool,
    ToolContext,
    ToolDispatchError,
    ToolResult,
)


async def _unused(**_: Any) -> None:
    # flo_ai's Tool wants a callable; the registry dispatches itself, so the
    # Tool only exists to feed format_tools_for_llm.
    return None


class ToolRegistry:
    def __init__(
        self,
        tools: Optional[List[AssistantTool]] = None,
        enabled_flags: Optional[Collection[str]] = None,
    ) -> None:
        self._tools: Dict[str, AssistantTool] = {}
        self._enabled_flags = frozenset(enabled_flags or ())
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AssistantTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f'Assistant tool already registered: {tool.name}')
        self._tools[tool.name] = tool

    def _exposed(self) -> List[AssistantTool]:
        return [
            tool
            for tool in self._tools.values()
            if tool.feature_flag is None or tool.feature_flag in self._enabled_flags
        ]

    def get(self, name: str) -> Optional[AssistantTool]:
        """The tool by name, or None when unknown or switched off by its flag."""
        for tool in self._exposed():
            if tool.name == name:
                return tool
        return None

    def names(self) -> List[str]:
        return [tool.name for tool in self._exposed()]

    def schemas(self, llm: Any) -> List[Dict[str, Any]]:
        """The exposed tools in ``llm``'s own function-calling format."""
        return llm.format_tools_for_llm([_to_flo_tool(t) for t in self._exposed()])

    async def dispatch(
        self, name: str, args: Any, ctx: ToolContext
    ) -> tuple[AssistantTool, ToolResult]:
        tool = self.get(name)
        if tool is None:
            raise ToolDispatchError(f'Unknown tool: {name}')
        if not isinstance(args, dict):
            raise ToolDispatchError('Tool arguments must be a JSON object')
        missing = [
            key for key in tool.parameters.get('required', []) if args.get(key) is None
        ]
        if missing:
            raise ToolDispatchError(
                f'Missing required argument(s): {", ".join(missing)}'
            )
        try:
            return tool, await tool.run(args, ctx)
        except ToolDispatchError:
            raise
        except Exception as exc:
            # The cause is logged by the caller from the type alone; the text of
            # an exception from a provider or storage client is not shown on.
            raise ToolDispatchError(
                f'Tool {name} failed ({type(exc).__name__})'
            ) from exc


def _to_flo_tool(tool: AssistantTool) -> Tool:
    required = set(tool.parameters.get('required', []))
    return Tool(
        name=tool.name,
        description=tool.description,
        function=_unused,
        parameters={
            key: {**spec, 'required': key in required}
            for key, spec in tool.parameters.get('properties', {}).items()
        },
    )
