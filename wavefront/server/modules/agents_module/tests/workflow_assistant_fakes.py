"""Scripted stand-ins for the LLM and collaborators the workflow assistant uses.

No provider is ever called: the LLM replays a transcript of canned responses.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agents_module.models.yaml_validation_schemas import YamlValidationError
from agents_module.services.workflow_assistant.resources import ResourceInventory


def text(content: str, **extra: Any) -> Dict[str, Any]:
    return {'text': content, **extra}


def call(name: str, arguments: Any, content: str = '', **extra: Any) -> Dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        'text': content,
        'function_call': {
            'name': name,
            'arguments': arguments,
            'id': f'call_{name}_{len(content)}',
        },
        **extra,
    }


class FakeLLM:
    """Replays ``script`` one response per generate() call."""

    def __init__(
        self,
        script: List[Dict[str, Any]],
        delay: float = 0.0,
        chunk_size: int = 1000,
    ) -> None:
        self.script = list(script)
        self.delay = delay
        #: stream() splits a scripted reply's text into pieces of this many chars
        self.chunk_size = chunk_size
        self.calls: List[Dict[str, Any]] = []
        #: Calls made through stream(); generate() calls are in ``calls``
        self.stream_calls: List[Dict[str, Any]] = []

    async def generate(self, messages, functions=None, output_schema=None):
        import asyncio

        self.calls.append(
            {'messages': [dict(m) for m in messages], 'functions': functions}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.script:
            raise AssertionError('FakeLLM script exhausted')
        return self.script.pop(0)

    async def stream(self, messages, functions=None, output_schema=None, **kwargs):
        """Replays the next scripted response's text, as chunks. Like the real
        providers' stream(), it yields text only and drops any function call."""
        self.stream_calls.append(
            {'messages': [dict(m) for m in messages], 'functions': functions}
        )
        if not self.script:
            raise AssertionError('FakeLLM script exhausted')
        content = self.script.pop(0).get('text', '')
        for i in range(0, len(content), self.chunk_size):
            yield {'content': content[i : i + self.chunk_size]}

    def get_message_content(self, response):
        return response.get('text', '')

    async def get_function_call(self, response):
        return response.get('function_call')

    def get_tool_use_id(self, function_call):
        return function_call.get('id')

    def format_tool_call_message(self, response, function_call):
        return {'role': 'assistant', 'content': None, 'tool_call': function_call}

    def get_assistant_message_for_tool_call(self, response):
        return None

    def format_function_result_message(self, function_name, content, tool_use_id=None):
        return {
            'role': 'function',
            'name': function_name,
            'content': content,
            'tool_use_id': tool_use_id,
        }

    def format_tools_for_llm(self, tools):
        return [{'name': t.name, 'parameters': t.parameters} for t in tools]


#: A YAML ending with this is "the workflow on the canvas": the service validates
#: it once for the baseline, and that call must not consume a proposal verdict.
CANVAS_MARKER = '# canvas\n'


class FakeWorkflowCrud:
    """validate_yaml answers from ``verdicts`` in order, repeating the last.

    A canvas YAML (ending in CANVAS_MARKER) is answered with ``baseline`` instead.
    """

    def __init__(
        self,
        verdicts: Optional[List[List[YamlValidationError]]] = None,
        baseline: Optional[List[YamlValidationError]] = None,
    ):
        self.verdicts = verdicts or [[]]
        self.baseline = baseline or []
        self.seen: List[str] = []
        self.baseline_calls = 0

    async def validate_yaml(self, yaml_content: str):
        if yaml_content.endswith(CANVAS_MARKER):
            self.baseline_calls += 1
            return list(self.baseline)
        self.seen.append(yaml_content)
        index = min(len(self.seen) - 1, len(self.verdicts) - 1)
        return list(self.verdicts[index])


class FakeResources:
    def __init__(self, inventory: Optional[ResourceInventory] = None) -> None:
        self.inventory = inventory or default_inventory()

    async def get_inventory(self):
        return self.inventory


MODEL_ID = '11111111-1111-1111-1111-111111111111'


def default_inventory() -> ResourceInventory:
    return ResourceInventory(
        llm_configs=[
            {
                'id': MODEL_ID,
                'display_name': 'Main model',
                'type': 'openai',
                'model_name': 'gpt-4.1-mini',
            }
        ],
        bindable_tools=[{'name': 'web_search', 'description': 'Search the web'}],
        function_nodes=['passthrough'],
        message_processors=[{'name': 'notify', 'description': 'Send a message'}],
        api_service_tools=[{'name': 'crm_get_customer', 'description': 'Get one'}],
    )


def err(path: str, message: str) -> YamlValidationError:
    return YamlValidationError(path=path, message=message)


def workflow_yaml(model_id: str = MODEL_ID, tool: Optional[str] = None) -> str:
    tools = f'\n      tools:\n        - name: {tool}' if tool else ''
    return f"""arium:
  agents:
    - name: writer
      job: Write a draft
      model:
        provider: rootflo
        model_id: {model_id}{tools}
  workflow:
    start: writer
    edges: []
    end: [writer]
"""


def fake_agent_inference_service(llm: Any, config: Optional[Dict[str, Any]] = None):
    """Stands in for the collaborators the assistant is built from. Use
    ``assistant_kwargs`` to hand them to WorkflowAssistantService; only ``build_llm``
    is called on the AgentInferenceService itself."""
    from unittest.mock import AsyncMock

    configs = [
        {
            'id': MODEL_ID,
            'display_name': 'Main model',
            'model_type': 'llm',
            'type': 'openai',
            'llm_model': 'gpt-4.1-mini',
        }
    ]
    config_service = SimpleNamespace(
        get_config=AsyncMock(return_value=config),
        list_configs=AsyncMock(return_value=configs),
    )
    tool_loader = SimpleNamespace(
        get_available_tools=lambda: {'web_search': {'description': 'Search the web'}},
        function_node_registry={'passthrough': object()},
    )
    return SimpleNamespace(
        llm_inference_config_service=config_service,
        tool_loader=tool_loader,
        message_processor_repository=None,
        api_services_manager=None,
        guardrails_engine=None,
        cache_manager=None,
        build_llm=lambda cfg: llm,
    )


def assistant_kwargs(svc: Any) -> Dict[str, Any]:
    """The constructor arguments WorkflowAssistantService takes from the container."""
    return dict(
        agent_inference_service=svc,
        llm_inference_config_service=svc.llm_inference_config_service,
        tool_loader=svc.tool_loader,
        message_processor_repository=svc.message_processor_repository,
        api_services_manager=svc.api_services_manager,
        guardrails_engine=svc.guardrails_engine,
        cache_manager=svc.cache_manager,
    )


def llm_config_dict(**overrides: Any) -> Dict[str, Any]:
    base = {
        'id': MODEL_ID,
        'llm_model': 'gpt-4.1-mini',
        'display_name': 'Main model',
        'api_key': 'sk-secret',
        'type': 'openai',
        'model_type': 'llm',
        'base_url': None,
        'parameters': None,
        'is_deleted': False,
    }
    base.update(overrides)
    return base
