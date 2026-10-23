"""Read-only tool: what the app already has that a workflow can bind to."""

import json
from typing import Any, Dict

from agents_module.services.workflow_assistant.tools.base import (
    AssistantTool,
    ToolContext,
    ToolResult,
)


class ListResourcesTool(AssistantTool):
    name = 'list_resources'
    description = (
        'List the resources of this app that a workflow can use: LLM configs '
        '(each with id, display_name, provider type and model_name; use an id as '
        'model_id with provider rootflo), tool names that agents can '
        'bind, function node names, message processors and API service tools. '
        'Call this before choosing models or tools. Names and ids not in this '
        'list do not exist.'
    )
    parameters: Dict[str, Any] = {'type': 'object', 'properties': {}, 'required': []}
    read_only = True
    label = 'Reading app resources'

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        inventory = await ctx.resources.get_inventory()
        data = inventory.to_dict()
        summary = (
            f'{len(inventory.llm_configs)} models, '
            f'{len(inventory.tool_names)} tools, '
            f'{len(inventory.function_nodes)} function nodes'
        )
        return ToolResult(content=json.dumps(data), summary=summary)
