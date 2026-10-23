"""The app's resource inventory, fetched by the server for every request.

The client never supplies it: a tampered inventory would let a caller steer the
assistant toward ids and tools that do not exist, or hide ones that do.

Datasources are deliberately absent. They are reachable only through the static
``datasource_*`` tools, whose ``datasource_id`` and ``table_name`` the user
supplies, and listing them needs a datasource service the agents container does
not have.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from common_module.log.logger import logger

from agents_module.utils.cache_utils import get_assistant_api_tools_cache_key

DESCRIPTION_CHARS = 160

#: How long the API-service tool listing is reused. Short: the listing only feeds
#: the prompt, and the validator (not this inventory) decides what is accepted.
API_TOOLS_CACHE_TTL_SECONDS = 60


def _short(text: Optional[str]) -> str:
    text = ' '.join((text or '').split())
    return (
        text if len(text) <= DESCRIPTION_CHARS else text[: DESCRIPTION_CHARS - 1] + '…'
    )


@dataclass
class ResourceInventory:
    llm_configs: List[Dict[str, str]] = field(default_factory=list)
    bindable_tools: List[Dict[str, str]] = field(default_factory=list)
    function_nodes: List[str] = field(default_factory=list)
    message_processors: List[Dict[str, str]] = field(default_factory=list)
    api_service_tools: List[Dict[str, str]] = field(default_factory=list)

    @property
    def llm_config_ids(self) -> Set[str]:
        return {item['id'] for item in self.llm_configs}

    @property
    def tool_names(self) -> Set[str]:
        """Every name an agent's ``tools[].name`` can resolve to."""
        return {
            item['name']
            for group in (
                self.bindable_tools,
                self.message_processors,
                self.api_service_tools,
            )
            for item in group
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'llm_configs': self.llm_configs,
            'bindable_tools': self.bindable_tools,
            'function_nodes': self.function_nodes,
            'message_processors': self.message_processors,
            'api_service_tools': self.api_service_tools,
            'datasources': 'not listed; datasource_* tools need a datasource_id and table_name from the user',
        }


class ResourceProvider:
    """Builds a ResourceInventory once per request (cached on the instance, so
    create one per request). The API-service tool listing, the one slow part (it
    reads every service definition from storage), is also shared across requests
    through ``cache_manager`` for a short TTL."""

    def __init__(
        self,
        tool_loader: Any,
        llm_inference_config_service: Any,
        message_processor_repository: Any = None,
        api_services_manager: Any = None,
        cache_manager: Any = None,
    ) -> None:
        self.tool_loader = tool_loader
        self.llm_inference_config_service = llm_inference_config_service
        self.message_processor_repository = message_processor_repository
        self.api_services_manager = api_services_manager
        self.cache_manager = cache_manager
        self._inventory: Optional[ResourceInventory] = None

    async def get_inventory(self) -> ResourceInventory:
        if self._inventory is None:
            self._inventory = ResourceInventory(
                llm_configs=await self._llm_configs(),
                bindable_tools=self._static_tools(),
                function_nodes=sorted(self.tool_loader.function_node_registry.keys()),
                message_processors=await self._message_processors(),
                api_service_tools=await self._api_service_tools(),
            )
        return self._inventory

    async def _llm_configs(self) -> List[Dict[str, str]]:
        configs = await self.llm_inference_config_service.list_configs()
        # id, display name, provider type and model name only: api_key and base_url
        # never reach the model. type and model_name let it match an inline model
        # block (provider openai, name gpt-4o-mini) to the config that serves it.
        return [
            {
                'id': str(c['id']),
                'display_name': c.get('display_name') or '',
                'type': c.get('type') or '',
                'model_name': c.get('llm_model') or '',
            }
            for c in configs
            if c.get('model_type', 'llm') == 'llm' and not c.get('is_deleted')
        ]

    def _static_tools(self) -> List[Dict[str, str]]:
        metadata = self.tool_loader.get_available_tools()
        return [
            {'name': name, 'description': _short((meta or {}).get('description'))}
            for name, meta in metadata.items()
        ]

    async def _message_processors(self) -> List[Dict[str, str]]:
        if self.message_processor_repository is None:
            return []
        try:
            rows = await self.message_processor_repository.find()
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: message processors unavailable ({type(exc).__name__})'
            )
            return []
        return [{'name': r.name, 'description': _short(r.description)} for r in rows]

    async def _api_service_tools(self) -> List[Dict[str, str]]:
        if self.api_services_manager is None:
            return []
        cached = self._cached_api_service_tools()
        if cached is not None:
            return cached
        manager = self.api_services_manager
        try:
            services = await manager.get_all_api_services() or []
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: API services unavailable ({type(exc).__name__})'
            )
            return []
        tools: List[Dict[str, str]] = []
        for service in services:
            if not service.is_active:
                continue
            try:
                tools.extend(await asyncio.to_thread(self._service_tools, service))
            except Exception as exc:
                logger.warning(
                    f'Workflow assistant: skipping API service {service.id} '
                    f'({type(exc).__name__})'
                )
        self._store_api_service_tools(tools)
        return tools

    def _cached_api_service_tools(self) -> Optional[List[Dict[str, str]]]:
        if self.cache_manager is None:
            return None
        try:
            raw = self.cache_manager.get_str(get_assistant_api_tools_cache_key())
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: tool cache read failed ({type(exc).__name__})'
            )
            return None

    def _store_api_service_tools(self, tools: List[Dict[str, str]]) -> None:
        if self.cache_manager is None:
            return
        try:
            self.cache_manager.add(
                get_assistant_api_tools_cache_key(),
                json.dumps(tools),
                expiry=API_TOOLS_CACHE_TTL_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: tool cache write failed ({type(exc).__name__})'
            )

    def _service_tools(self, service: Any) -> List[Dict[str, str]]:
        from api_services_module.config.parser import ServiceDefinitionParser

        definition = ServiceDefinitionParser.parse_yaml_string(
            self.api_services_manager.fetch_service_def(service)
        )
        # Named exactly as ApiServiceToolDetailsProvider does (tools_module/
        # api_service/provider.py _generate_tool_name): the id declared in the
        # parsed definition, not the database row id, joined to the api id by '_'.
        # The tool loader resolves only v1 apis, so only those are bindable.
        return [
            {
                'name': f'{definition.id}_{api.id}',
                'description': _short(api.description),
            }
            for api in definition.apis
            if api.version == 'v1'
        ]


def parse_uuid(value: str) -> Optional[UUID]:
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None
