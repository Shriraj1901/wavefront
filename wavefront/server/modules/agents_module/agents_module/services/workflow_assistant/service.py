"""The workflow assistant: resolve a model, run the bounded loop, emit events."""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from common_module.log.logger import logger
from common_module.middleware.request_id_middleware import get_current_request_id
from common_module.utils.guardrails import guarded_llm
from db_repo_module.models.llm_inference_config import LlmInferenceConfig

from flo_ai.helpers.generation_params import TOKEN_LIMIT_ALIASES

from agents_module.models.workflow_assistant_schemas import (
    AssistantChatRequest,
    assistant_started,
    error_event,
    output_event,
    proposal_event,
    retract_event,
)
from agents_module.services.workflow_assistant.intent import ASK, classify_intent
from agents_module.services.workflow_assistant.limits import AssistantLimits
from agents_module.services.workflow_assistant.prompt import (
    PromptBuilder,
    PromptContext,
    default_ask_prompt_builder,
    default_prompt_builder,
)
from agents_module.services.workflow_assistant.repair_loop import (
    COMPLETED,
    LoopStats,
    MAX_ROUNDS,
    REPAIR_EXHAUSTED,
    REPEATED_ERRORS,
    TIMEOUT,
    TOKEN_BUDGET,
    response_tokens,
    response_truncated,
    run_loop,
)
from agents_module.services.workflow_assistant.resources import (
    ResourceProvider,
    parse_uuid,
)
from agents_module.services.workflow_assistant.streaming import (
    GuardrailBlocked,
    StreamedReply,
    stream_reply,
)
from agents_module.services.workflow_assistant.tools.base import Proposal, ToolContext
from agents_module.services.workflow_assistant.tools.list_resources import (
    ListResourcesTool,
)
from agents_module.services.workflow_assistant.tools.propose_workflow_yaml import (
    ProposeWorkflowYamlTool,
    compute_errors,
)
from agents_module.services.workflow_assistant.tools.registry import ToolRegistry
from agents_module.utils.agent_guardrails import guardrails

#: Providers whose flo_ai client formats tool calls. ollama and vllm depend on the
#: served model, so they are allowed on a best-effort basis. bedrock is accepted by
#: the config API but the shared LLM builder has no branch for it yet.
TOOL_CAPABLE_TYPES = frozenset(
    {'openai', 'groq', 'azure_openai', 'anthropic', 'gemini', 'ollama', 'vllm'}
)

#: The assistant has no namespace of its own: no console UI can create a policy
#: for one, so it runs under the `default` namespace, the one policies can target.
GUARDRAIL_NAMESPACE = 'default'
GUARDRAIL_PRINCIPAL = 'workflow_assistant'

STOP_NOTES = {
    MAX_ROUNDS: 'I stopped because I reached the step limit for one request.',
    TOKEN_BUDGET: 'I stopped because I reached the size limit for one request.',
    TIMEOUT: 'I stopped because the request took too long.',
    REPAIR_EXHAUSTED: (
        'I could not fix all the validation errors my change introduced in the '
        'allowed attempts.'
    ),
    REPEATED_ERRORS: (
        'I stopped because the same validation errors from my change kept '
        'coming back.'
    ),
}

#: Stops that only mean "out of budget". After a proposal with no introduced
#: errors they are not a failure: the proposal is the answer.
BUDGET_STOPS = frozenset({MAX_ROUNDS, TOKEN_BUDGET, TIMEOUT})

TRUNCATED_NOTICE = (
    'This reply was cut off at the output limit. Press Continue to get the rest.'
)

INTERRUPTED_NOTICE = 'This reply was interrupted. Ask again to get the rest.'


def with_min_output_tokens(
    parameters: Optional[Dict[str, Any]], minimum: int
) -> Dict[str, Any]:
    """A COPY of an LLM config's parameters whose token limit is at least ``minimum``.

    A higher limit is kept. A lower or missing one (under any provider's spelling)
    is replaced by the canonical ``max_tokens``, which the shared LLM builder
    renames to the provider's own key. The stored config is never touched.
    """
    params = dict(parameters or {})
    limits = [
        params[key]
        for key in TOKEN_LIMIT_ALIASES
        if isinstance(params.get(key), int) and not isinstance(params.get(key), bool)
    ]
    if limits and max(limits) >= minimum:
        return params
    params = {k: v for k, v in params.items() if k not in TOKEN_LIMIT_ALIASES}
    params['max_tokens'] = minimum
    return params


def remaining_issues_message(count: int) -> str:
    noun = 'issue remains' if count == 1 else 'issues remain'
    return (
        f'I made the change but {count} validation {noun}. The workflow will not '
        'save until ' + ('it is' if count == 1 else 'they are') + ' fixed.'
    )


def no_change_message(reason: str, last_error: Optional[str]) -> str:
    """Plain words for a turn that never produced a usable proposal."""
    # Repair-cap reasons talk about validation errors; with no proposal at all the
    # calls failed outright, so the last error is the explanation
    why = (
        ''
        if reason in (REPAIR_EXHAUSTED, REPEATED_ERRORS)
        else STOP_NOTES.get(reason, '').rstrip('.')
    )
    parts = ['I could not make a change to the workflow.']
    if why:
        parts.append(f'{why}.')
    if last_error:
        parts.append(f'The last error was: {last_error}')
    parts.append('You can try again, or ask for a smaller change.')
    return ' '.join(parts)


def default_reply(proposal: Proposal) -> str:
    """The reply when the model wrote no text of its own."""
    if proposal.introduced:
        return remaining_issues_message(len(proposal.errors))
    if proposal.existing:
        n = len(proposal.existing)
        return (
            f'Proposal ready; {n} existing validation '
            f'{"issue still remains" if n == 1 else "issues still remain"} in the '
            'workflow.'
        )
    return 'Here is the proposed workflow.'


class AssistantRequestError(Exception):
    """A request rejected before the stream opens; maps to an HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class PreparedChat:
    llm: Any
    messages: List[Dict[str, Any]]
    nonce: str
    llm_config_id: str
    provider: str
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    #: Tool context built in prepare, holding the baseline taken there
    ctx: Optional[ToolContext] = None
    #: 'auto' classifies the message first; 'ask' and 'edit' skip that
    mode: str = 'auto'
    #: What the system prompts are built from (the answer-only prompt is built
    #: from it when a turn is routed to ask)
    prompt_ctx: Optional[PromptContext] = None


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ListResourcesTool())
    registry.register(ProposeWorkflowYamlTool())
    return registry


class WorkflowAssistantService:
    def __init__(
        self,
        agent_inference_service: Any,
        workflow_crud_service: Any,
        limits: AssistantLimits,
        llm_inference_config_service: Any = None,
        tool_loader: Any = None,
        message_processor_repository: Any = None,
        api_services_manager: Any = None,
        guardrails_engine: Any = None,
        cache_manager: Any = None,
        registry: Optional[ToolRegistry] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        ask_prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        # Used only to build the LLM (its public build_llm); everything else the
        # assistant needs is injected below rather than read off that service
        self.agent_inference_service = agent_inference_service
        self.llm_inference_config_service = llm_inference_config_service
        self.tool_loader = tool_loader
        self.message_processor_repository = message_processor_repository
        self.api_services_manager = api_services_manager
        self.guardrails_engine = guardrails_engine
        self.cache_manager = cache_manager
        self.workflow_crud_service = workflow_crud_service
        self.limits = limits
        self.registry = registry or build_default_registry()
        self.prompt_builder = prompt_builder or default_prompt_builder()
        self.ask_prompt_builder = ask_prompt_builder or default_ask_prompt_builder()

    # -- request validation and model resolution -------------------------------

    async def resolve_llm_config(self, config_id: str) -> LlmInferenceConfig:
        """The config the assistant will run on, or an AssistantRequestError.

        The config table has no tenant column, so "belongs to this app" means it
        exists, is not deleted, and is an LLM (not an embedding) config.
        """
        config_uuid = parse_uuid(config_id)
        if config_uuid is None:
            raise AssistantRequestError(
                400, 'llm_inference_config_id must be a valid UUID'
            )
        service = self.llm_inference_config_service
        if service is None:
            raise AssistantRequestError(503, 'LLM configuration is not available')
        config = await service.get_config(config_uuid)
        if not config or config.get('is_deleted'):
            raise AssistantRequestError(404, 'LLM inference config not found')
        if config.get('model_type', 'llm') != 'llm':
            raise AssistantRequestError(
                400, 'The selected config is not a chat model (LLM) config'
            )
        provider = config.get('type')
        if provider not in TOOL_CAPABLE_TYPES:
            raise AssistantRequestError(
                400,
                f"The assistant does not support the '{provider}' provider yet. "
                'Choose a config for one of: ' + ', '.join(sorted(TOOL_CAPABLE_TYPES)),
            )
        # Output floor on a copy: the stored config keeps its own limit
        config = {
            **config,
            'parameters': with_min_output_tokens(
                config.get('parameters'), self.limits.max_output_tokens
            ),
        }
        return LlmInferenceConfig(**config)

    def _check_sizes(self, req: AssistantChatRequest) -> None:
        if len(req.message) > self.limits.max_message_chars:
            raise AssistantRequestError(
                400, f'Message is too long (limit {self.limits.max_message_chars})'
            )
        if len(req.current_yaml) > self.limits.max_yaml_chars:
            raise AssistantRequestError(
                400,
                f'Current workflow YAML is too large (limit {self.limits.max_yaml_chars})',
            )

    def _history_messages(self, req: AssistantChatRequest) -> List[Dict[str, Any]]:
        recent = req.history[-self.limits.max_history_messages :]
        cap = self.limits.max_message_chars
        return [{'role': m.role, 'content': m.content[:cap]} for m in recent]

    async def prepare(
        self, req: AssistantChatRequest, user_id: Optional[str] = None
    ) -> PreparedChat:
        """Everything that can fail before the stream opens. Raises
        AssistantRequestError."""
        if not self.limits.enabled:
            raise AssistantRequestError(503, 'The workflow assistant is disabled')
        self._check_sizes(req)
        config = await self.resolve_llm_config(req.llm_inference_config_id)

        try:
            llm = self.agent_inference_service.build_llm(config)
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: could not build LLM ({type(exc).__name__})'
            )
            raise AssistantRequestError(
                400, 'The selected LLM config could not be used'
            ) from exc

        # Built per request, so binding this request's user into the principal is
        # safe (audit only). Unguarded (fail-open) when there is no engine.
        llm = guarded_llm(
            llm,
            self.guardrails_engine,
            GUARDRAIL_NAMESPACE,
            GUARDRAIL_PRINCIPAL,
            user_id,
        )

        ctx = self._tool_context()
        # Set first: the baseline's parse is then kept on ctx for the proposals
        ctx.baseline_yaml = req.current_yaml
        ctx.baseline_errors = await self._baseline_errors(req.current_yaml, ctx)

        inventory = await self._inventory(ctx)
        prompt_ctx = PromptContext(
            current_yaml=req.current_yaml,
            existing_issues=[(e.path, e.message) for e in ctx.baseline_errors],
            llm_configs=list(inventory.llm_configs) if inventory is not None else [],
            selected_llm_config_id=str(config.id),
            inventory=inventory,
        )
        messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': self.prompt_builder.build(prompt_ctx)},
            *self._history_messages(req),
            {'role': 'user', 'content': req.message},
        ]
        return PreparedChat(
            llm=llm,
            messages=messages,
            nonce=prompt_ctx.nonce,
            llm_config_id=str(config.id),
            provider=config.type,
            user_id=user_id,
            run_id=get_current_request_id(),
            ctx=ctx,
            mode=req.mode,
            prompt_ctx=prompt_ctx,
        )

    async def _inventory(self, ctx: ToolContext) -> Optional[Any]:
        """The app's resource inventory for the prompts; None when it cannot be
        read (the edit prompt's model can still call list_resources)."""
        try:
            return await ctx.resources.get_inventory()
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: resources unavailable ({type(exc).__name__})'
            )
            return None

    async def _baseline_errors(self, current_yaml: str, ctx: ToolContext) -> List[Any]:
        """The problems the workflow on the canvas already has, validated once per
        request. Empty when there is none, and when it cannot be checked, which
        treats every error in a proposal as introduced (the strict fallback)."""
        if not current_yaml.strip():
            return []
        try:
            return await compute_errors(current_yaml, ctx)
        except Exception as exc:
            logger.warning(
                f'Workflow assistant: baseline validation failed ({type(exc).__name__})'
            )
            return []

    # -- the turn --------------------------------------------------------------

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            resources=ResourceProvider(
                tool_loader=self.tool_loader,
                llm_inference_config_service=self.llm_inference_config_service,
                message_processor_repository=self.message_processor_repository,
                api_services_manager=self.api_services_manager,
                cache_manager=self.cache_manager,
            ),
            workflow_crud_service=self.workflow_crud_service,
            max_yaml_chars=self.limits.max_yaml_chars,
        )

    async def chat(self, prepared: PreparedChat) -> AsyncIterator[Dict[str, Any]]:
        """Yield the turn's events. Always ends with ``output`` or ``error``."""
        started = time.monotonic()
        stats = LoopStats()
        ctx = prepared.ctx or self._tool_context()
        outcome = 'error'

        def scope():
            return guardrails(run_id=prepared.run_id)

        try:
            yield assistant_started()
            deadline = started + self.limits.timeout_seconds
            intent = prepared.mode
            if intent == 'auto':
                intent, classifier_tokens = await classify_intent(
                    prepared.llm,
                    prepared.messages[-1]['content'],
                    self._previous_assistant_message(prepared.messages),
                    deadline=deadline,
                    scope=scope,
                )
                stats.tokens += classifier_tokens
            stats.intent = intent

            if intent == ASK:
                ask_messages = self._ask_messages(prepared)
                reply = StreamedReply()
                async for event in stream_reply(
                    prepared.llm,
                    ask_messages,
                    reply,
                    deadline=deadline,
                    scope=scope,
                ):
                    yield event
                terminal, outcome = self._ask_terminal(reply, ask_messages, stats)
                yield terminal
                return

            async for event in run_loop(
                prepared.llm,
                prepared.messages,
                self.registry,
                ctx,
                self.limits,
                prepared.nonce,
                stats,
                generate_scope=scope,
                deadline=deadline,
            ):
                yield event

            last = ctx.last_proposal
            reason = stats.stop_reason
            text = '\n\n'.join(stats.texts)

            if last is not None and not last.introduced and reason in BUDGET_STOPS:
                # Out of budget after a clean proposal: the proposal is the answer.
                reason = COMPLETED

            if last is None and reason != COMPLETED:
                # Nothing parseable was ever proposed: say why, in plain words
                outcome = reason
                yield error_event(no_change_message(reason, stats.last_error))
                return
            if last is None and not text:
                outcome = 'empty_reply'
                yield error_event('The model returned an empty reply.')
                return

            if last is not None:
                yield proposal_event(
                    last.proposal_id,
                    last.yaml,
                    last.valid,
                    last.errors,
                    last.introduced,
                    last.notes,
                )
            if reason != COMPLETED:
                # Never end with nothing: the best proposal is already on the wire,
                # so say plainly what is left rather than that no change was made
                if last is not None and last.errors:
                    note = remaining_issues_message(len(last.errors))
                    if reason in BUDGET_STOPS:
                        note = f'{STOP_NOTES[reason]} {note}'
                else:
                    note = STOP_NOTES[reason]
                text = f'{text}\n\n{note}'.strip()
            elif last is not None and last.introduced:
                # The model stopped on its own with errors left: say so plainly
                message = remaining_issues_message(len(last.errors))
                text = f'{text}\n\n{message}'.strip() if text else message
            elif not text and last is not None:
                text = default_reply(last)
            if stats.interrupted:
                text = f'{text}\n\n{INTERRUPTED_NOTICE}'.strip()
            if stats.truncated:
                text = f'{text}\n\n{TRUNCATED_NOTICE}'.strip()
            outcome = reason if reason != COMPLETED else 'completed'
            yield output_event(text, truncated=stats.truncated)
        except GuardrailBlocked as exc:
            outcome = 'guardrail_blocked'
            if getattr(exc, 'retract', False):
                # Text of this reply may already be on screen; an error alone
                # would leave it there, so withdraw it first
                yield retract_event('', reason=str(exc))
            yield error_event('The request was blocked by a guardrail policy.')
        except (GeneratorExit, asyncio.CancelledError):
            outcome = 'cancelled'
            raise
        except Exception as exc:
            logger.warning(f'Workflow assistant failed: {type(exc).__name__}')
            yield error_event('The assistant could not complete the request.')
        finally:
            self._audit(prepared, stats, ctx, outcome, started)

    @staticmethod
    def _previous_assistant_message(messages: List[Dict[str, Any]]) -> Optional[str]:
        """The assistant turn right before the user's latest message, if any:
        what a follow-up such as 'yes do it' refers to."""
        if len(messages) >= 3 and messages[-2].get('role') == 'assistant':
            return str(messages[-2].get('content') or '')
        return None

    def _ask_messages(self, prepared: PreparedChat) -> List[Dict[str, Any]]:
        """The conversation with the answer-only system prompt in place of the
        editing one."""
        prompt_ctx = prepared.prompt_ctx or PromptContext()
        return [
            {'role': 'system', 'content': self.ask_prompt_builder.build(prompt_ctx)},
            *prepared.messages[1:],
        ]

    @staticmethod
    def _ask_terminal(
        reply: StreamedReply, messages: List[Dict[str, Any]], stats: LoopStats
    ) -> Tuple[Dict[str, Any], str]:
        """Account for an answered turn; its closing event and audit outcome."""
        text = reply.text
        stats.rounds += 1
        stats.tokens += response_tokens(reply.response or {}, messages, text)
        stats.truncated = reply.response is not None and response_truncated(
            reply.response
        )
        if text:
            stats.texts.append(text)
        if reply.timed_out:
            note = STOP_NOTES[TIMEOUT]
            if not text:
                return error_event(note), TIMEOUT
            return output_event(f'{text}\n\n{note}'), TIMEOUT
        if not text:
            return error_event('The model returned an empty reply.'), 'empty_reply'
        if reply.interrupted:
            text = f'{text}\n\n{INTERRUPTED_NOTICE}'
        if stats.truncated:
            text = f'{text}\n\n{TRUNCATED_NOTICE}'
        return output_event(text, truncated=stats.truncated), 'completed'

    def _audit(
        self,
        prepared: PreparedChat,
        stats: LoopStats,
        ctx: ToolContext,
        outcome: str,
        started: float,
    ) -> None:
        """One structured record per request. Never carries YAML or prompts."""
        last = ctx.last_proposal
        record = {
            'user_id': prepared.user_id,
            'llm_config_id': prepared.llm_config_id,
            'provider': prepared.provider,
            'tokens': stats.tokens,
            'tools_called': stats.tools_called,
            'rounds': stats.rounds,
            'proposals': len(ctx.proposals),
            'final_proposal_valid': None if last is None else last.valid,
            'final_introduced_errors': None if last is None else len(last.introduced),
            'intent': stats.intent,
            'outcome': outcome,
            'duration_ms': int((time.monotonic() - started) * 1000),
        }
        logger.info(f'workflow_assistant_audit {json.dumps(record)}')
