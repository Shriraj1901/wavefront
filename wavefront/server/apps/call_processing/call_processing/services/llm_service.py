"""
LLM (Language Model) service factory

Supports multiple providers: OpenAI, Anthropic, Google, etc.
"""

import math
from typing import Dict, Any, Iterable
from call_processing.log.logger import logger

# Pipecat LLM services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.azure.llm import AzureLLMService
# Add more as needed

# pipecat 0.0.x gave a function call 10s before answering the LLM with an empty
# result; 1.x defaults to no timeout, and at the deadline cancels the handler
# (CancelledError) and lets the LLM report the failure. Keep the 10s bound so a
# slow tool cannot hold the caller in silence indefinitely.
FUNCTION_CALL_TIMEOUT_SECS = 10.0

# Head-room over a tool's own HTTP timeout, so a slow request ends with the
# tool's clean timeout result instead of being cancelled mid-flight.
TOOL_TIMEOUT_MARGIN_SECS = 2.0


def function_call_timeout_secs(tool_timeout: Any) -> float:
    """Handler deadline for a tool whose own request timeout is tool_timeout.

    Never shorter than FUNCTION_CALL_TIMEOUT_SECS: HTTP timeouts apply per phase
    (connect, read, ...), so a call can legitimately run past its configured
    timeout, and only tools configured to wait longer get a longer deadline.
    """
    try:
        timeout = float(tool_timeout)
    except (TypeError, ValueError):
        return FUNCTION_CALL_TIMEOUT_SECS
    if not math.isfinite(timeout) or timeout <= 0:
        return FUNCTION_CALL_TIMEOUT_SECS
    return max(FUNCTION_CALL_TIMEOUT_SECS, timeout + TOOL_TIMEOUT_MARGIN_SECS)


# Keys each provider's Settings accepts from the stored `parameters` dict. Settings
# are dataclasses, so an unlisted key would raise TypeError at construction.
OPENAI_SETTINGS_KEYS = (
    'temperature',
    'max_completion_tokens',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'seed',
)
GOOGLE_SETTINGS_KEYS = ('temperature', 'max_tokens', 'top_p', 'top_k')


def _pick(parameters: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: parameters[key] for key in keys if key in parameters}


def _send_developer_messages_as_user(service: BaseOpenAILLMService):
    """Have the service send "developer"-role messages as "user" messages.

    For providers that may reject the developer role; pipecat injects it for
    empty-turn recovery. Set on the instance, which pipecat reads per request,
    rather than through a subclass, which would rename the service in traces.
    """
    service.supports_developer_role = False
    return service


class LLMServiceFactory:
    """Factory for creating LLM service instances from configuration"""

    @staticmethod
    def create_llm_service(llm_config: Dict[str, Any]):
        """
        Create LLM service from configuration

        Args:
            llm_config: {
                'type': 'openai' | 'anthropic' | 'gemini' | 'groq',
                'api_key': 'key',
                'llm_model': 'gpt-4',
                'parameters': {
                    'temperature': 0.7,
                    'max_tokens': 150,
                    ...
                }
            }

        Returns:
            Pipecat LLM service instance
        """
        llm_type = llm_config['type']
        api_key = llm_config['api_key']
        model = llm_config['llm_model']
        parameters = llm_config.get('parameters', {})

        if parameters is None:
            parameters = {}

        logger.info(f'Creating LLM service: {llm_type} / {model}')

        if llm_type == 'openai':
            base_url = llm_config.get('base_url')
            if not base_url or base_url == 'https://api.openai.com/v1':
                return LLMServiceFactory._create_openai_llm(api_key, model, parameters)
            else:
                return LLMServiceFactory._create_openai_compatible_llm(
                    api_key, model, parameters, base_url
                )
        elif llm_type == 'azure_openai':
            base_url = llm_config.get('base_url')
            if not base_url:
                raise ValueError(
                    'Azure OpenAI requires base_url (endpoint) to be configured'
                )
            return LLMServiceFactory._create_azure_llm(
                api_key, model, parameters, base_url
            )
        elif llm_type == 'gemini':
            return LLMServiceFactory._create_google_llm(api_key, model, parameters)
        elif llm_type == 'groq':
            return LLMServiceFactory._create_groq_llm(api_key, model, parameters)
        else:
            raise ValueError(f'Unsupported LLM type: {llm_type}')

    @staticmethod
    def _create_openai_llm(api_key: str, model: str, parameters: Dict[str, Any]):
        """Create OpenAI LLM service"""
        settings_kwargs = _pick(parameters, OPENAI_SETTINGS_KEYS)

        logger.info(
            f"OpenAI LLM config: model={model}, temp={settings_kwargs.get('temperature', 'default')}"
        )

        return OpenAILLMService(
            api_key=api_key,
            # An init argument, not a Settings field
            service_tier=parameters.get('service_tier'),
            settings=OpenAILLMService.Settings(model=model, **settings_kwargs),
            function_call_timeout_secs=FUNCTION_CALL_TIMEOUT_SECS,
        )

    @staticmethod
    def _create_openai_compatible_llm(
        api_key: str, model: str, parameters: Dict[str, Any], base_url: str
    ):
        """Create a BaseOpenAILLMService for OpenAI-compatible endpoints"""
        settings_kwargs = _pick(parameters, OPENAI_SETTINGS_KEYS)

        logger.info(
            f"OpenAI-compatible LLM config: model={model}, base_url={base_url}, temp={settings_kwargs.get('temperature', 'default')}"
        )

        # Unknown server: it may not accept the developer role.
        return _send_developer_messages_as_user(
            BaseOpenAILLMService(
                api_key=api_key,
                base_url=base_url,
                settings=BaseOpenAILLMService.Settings(model=model, **settings_kwargs),
                function_call_timeout_secs=FUNCTION_CALL_TIMEOUT_SECS,
            )
        )

    @staticmethod
    def _create_google_llm(api_key: str, model: str, parameters: Dict[str, Any]):
        """Create Google LLM service"""
        settings_kwargs = _pick(parameters, GOOGLE_SETTINGS_KEYS)

        logger.info(
            f"Google LLM config: model={model}, temp={settings_kwargs.get('temperature', 'default')}"
        )

        return GoogleLLMService(
            api_key=api_key,
            settings=GoogleLLMService.Settings(model=model, **settings_kwargs),
            function_call_timeout_secs=FUNCTION_CALL_TIMEOUT_SECS,
        )

    @staticmethod
    def _create_groq_llm(api_key: str, model: str, parameters: Dict[str, Any]):
        """Create Groq LLM service"""
        settings_kwargs = _pick(parameters, OPENAI_SETTINGS_KEYS)

        logger.info(
            f"Groq LLM config: model={model}, temp={settings_kwargs.get('temperature', 'default')}"
        )

        # pipecat's GroqLLMService keeps the OpenAI default of sending the
        # developer role unconverted.
        return _send_developer_messages_as_user(
            GroqLLMService(
                api_key=api_key,
                # An init argument, not a Settings field
                service_tier=parameters.get('service_tier'),
                settings=GroqLLMService.Settings(model=model, **settings_kwargs),
                function_call_timeout_secs=FUNCTION_CALL_TIMEOUT_SECS,
            )
        )

    @staticmethod
    def _create_azure_llm(
        api_key: str, model: str, parameters: Dict[str, Any], base_url: str = None
    ):
        """Create Azure OpenAI LLM service"""
        # api_version is deliberately not passed: it is deprecated since pipecat
        # 1.8.0. An endpoint ending in /openai/v1 selects the v1 API surface;
        # any other endpoint gets pipecat's dated default (2025-04-01-preview).
        settings_kwargs = _pick(parameters, OPENAI_SETTINGS_KEYS)

        logger.info(
            f"Azure OpenAI LLM config: model={model}, endpoint={base_url}, temp={settings_kwargs.get('temperature', 'default')}"
        )

        return AzureLLMService(
            api_key=api_key,
            endpoint=base_url,
            settings=AzureLLMService.Settings(model=model, **settings_kwargs),
            function_call_timeout_secs=FUNCTION_CALL_TIMEOUT_SECS,
        )
