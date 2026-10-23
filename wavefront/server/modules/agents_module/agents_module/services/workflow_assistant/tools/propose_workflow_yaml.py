"""Tool the model uses to put a complete workflow YAML forward for validation."""

import asyncio
import json
import re
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from agents_module.models.yaml_validation_schemas import YamlValidationError
from agents_module.services.workflow_assistant.resources import ResourceInventory
from agents_module.services.workflow_assistant.tools.base import (
    AssistantTool,
    Proposal,
    ToolContext,
    ToolDispatchError,
    ToolResult,
)


#: The C loader when libyaml is available. Still a SAFE loader (no arbitrary
#: object construction); it is several times faster on large documents.
_YAML_LOADER = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)

#: Marks "not parsed yet" for the optional pre-parsed arguments below; ``None`` is
#: a real parse result (an empty or unparseable document).
_UNPARSED: Any = object()


def parse_yaml_document(yaml_text: str) -> Any:
    """The parsed document, or None when it does not parse (the validator reports
    that). One call per document: every check below takes the result."""
    try:
        return yaml.load(yaml_text, Loader=_YAML_LOADER)
    except yaml.YAMLError:
        return None


def _document(ctx: ToolContext, yaml_text: str) -> Any:
    """``yaml_text`` parsed; the baseline's parse is kept on ``ctx`` so it happens
    once per request however many proposals are checked against it."""
    cached = ctx.baseline_parse
    if cached is not None and cached[0] == yaml_text:
        return cached[1]
    data = parse_yaml_document(yaml_text)
    if yaml_text == ctx.baseline_yaml:
        ctx.baseline_parse = (yaml_text, data)
    return data


def _iter_arium(arium: Any, path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Every agent and router mapping in an arium, nested ariums included, with
    the same path spelling the validator uses."""
    if not isinstance(arium, dict):
        return
    for key in ('agents', 'routers'):
        items = arium.get(key)
        if isinstance(items, list):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    yield f'{path}.{key}[{index}]', item
    nested = arium.get('ariums')
    if isinstance(nested, list):
        for index, node in enumerate(nested):
            yield from _iter_arium(node, f'{path}.ariums[{index}]')


def assistant_extra_errors(
    yaml_text: str,
    inventory: ResourceInventory,
    existing: List[YamlValidationError],
    data: Any = _UNPARSED,
) -> List[YamlValidationError]:
    """The check the static validator does not make: a rootflo model_id must be an
    LLM config of this app. (Tool names are the validator's: it resolves them
    against the same providers the runtime uses, so they are not re-checked here.)

    Skips a path the validator already reported, so one problem is not listed
    twice. A document that does not parse is the validator's to report. Pass
    ``data`` (the parsed document) to avoid parsing ``yaml_text`` again.
    """
    if data is _UNPARSED:
        data = parse_yaml_document(yaml_text)
    if not isinstance(data, dict):
        return []

    seen = {error.path for error in existing}
    extra: List[YamlValidationError] = []

    def add(path: str, message: str) -> None:
        if path not in seen:
            seen.add(path)
            extra.append(YamlValidationError(path=path, message=message))

    known_models = inventory.llm_config_ids
    for path, node in _iter_arium(data.get('arium'), 'arium'):
        model = node.get('model')
        if isinstance(model, dict) and model.get('provider') == 'rootflo':
            model_id = str(model.get('model_id') or '')
            if model_id not in known_models:
                add(
                    f'{path}.model.model_id',
                    f"model_id '{model_id}' is not an LLM config of this app; "
                    'use an id from list_resources',
                )
    return extra


async def compute_errors_and_document(
    yaml_text: str, ctx: ToolContext
) -> Tuple[List[YamlValidationError], Any]:
    """Every problem in ``yaml_text``: the static validator's plus the assistant's,
    and the parsed document (parsed once, off the event loop, so a large document
    does not stall other requests)."""
    errors = list(await ctx.workflow_crud_service.validate_yaml(yaml_text))
    inventory = await ctx.resources.get_inventory()

    def classify() -> Any:
        data = _document(ctx, yaml_text)
        errors.extend(assistant_extra_errors(yaml_text, inventory, errors, data))
        return data

    data = await asyncio.to_thread(classify)
    return errors, data


async def compute_errors(yaml_text: str, ctx: ToolContext) -> List[YamlValidationError]:
    """Every problem in ``yaml_text``: the static validator's plus the assistant's."""
    errors, _ = await compute_errors_and_document(yaml_text, ctx)
    return errors


_PATH_TOKEN = re.compile(r'([^.\[\]]+)|\[(\d+)\]')
_INDEX = re.compile(r'\[\d+\]')
#: List keys whose items are not workflow nodes, so their ``name`` is not a node name.
_NON_NODE_LISTS = frozenset({'tools', 'edges'})


def node_name_for_path(
    yaml_text: str, path: str, data: Any = _UNPARSED
) -> Optional[str]:
    """The name of the node (agent, router, function node) a path points into,
    or None when the path is not inside a named node or cannot be followed.
    Pass ``data`` (the parsed document) to avoid parsing ``yaml_text`` again."""
    if data is _UNPARSED:
        data = parse_yaml_document(yaml_text)
    node: Any = data
    last_key = ''
    name: Optional[str] = None
    for key, index in _PATH_TOKEN.findall(path):
        if key:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
            last_key = key
        else:
            i = int(index)
            if not isinstance(node, list) or i >= len(node):
                break
            node = node[i]
            if (
                isinstance(node, dict)
                and isinstance(node.get('name'), str)
                and last_key not in _NON_NODE_LISTS
            ):
                name = node['name']
    return name


def split_errors(
    errors: List[YamlValidationError],
    proposed_yaml: str,
    baseline: List[YamlValidationError],
    baseline_yaml: str,
    proposed_data: Any = _UNPARSED,
    baseline_data: Any = _UNPARSED,
) -> Tuple[List[YamlValidationError], List[YamlValidationError]]:
    """Split ``errors`` into (introduced, existing).

    An error inside a named node is existing when the baseline had the same
    message on the same node (same field), whatever index the node sits at now: an
    insert before it moves its path without changing the problem. An error outside
    any node is existing when the baseline had the same (path, message), or the
    same message at the same path ignoring list indexes. Each baseline error
    matches at most one proposal error, so a problem that appears more often than
    before is partly introduced.

    Each document is parsed once, here, unless the caller already has the parsed
    ``proposed_data`` / ``baseline_data``.
    """
    if not baseline:
        return list(errors), []
    if proposed_data is _UNPARSED:
        proposed_data = parse_yaml_document(proposed_yaml)
    if baseline_data is _UNPARSED:
        baseline_data = parse_yaml_document(baseline_yaml)

    node_pool: Counter = Counter()
    exact_pool: Counter = Counter()
    loose_pool: Counter = Counter()
    for e in baseline:
        node = node_name_for_path(baseline_yaml, e.path, baseline_data)
        if node is not None:
            node_pool[(node, e.message, _INDEX.sub('[]', e.path))] += 1
        else:
            exact_pool[(e.path, e.message)] += 1
            loose_pool[(_INDEX.sub('[]', e.path), e.message)] += 1

    existing_ids = set()
    unmatched: List[YamlValidationError] = []
    for e in errors:
        node = node_name_for_path(proposed_yaml, e.path, proposed_data)
        if node is not None:
            key = (node, e.message, _INDEX.sub('[]', e.path))
            if node_pool[key] > 0:
                node_pool[key] -= 1
                existing_ids.add(id(e))
            continue
        key = (e.path, e.message)
        if exact_pool[key] > 0:
            exact_pool[key] -= 1
            loose_pool[(_INDEX.sub('[]', e.path), e.message)] -= 1
            existing_ids.add(id(e))
        else:
            unmatched.append(e)
    for e in unmatched:
        key = (_INDEX.sub('[]', e.path), e.message)
        if loose_pool[key] > 0:
            loose_pool[key] -= 1
            existing_ids.add(id(e))

    introduced = [e for e in errors if id(e) not in existing_ids]
    existing = [e for e in errors if id(e) in existing_ids]
    return introduced, existing


#: Bounds on what the model can attach to a proposal, so a runaway list cannot
#: flood the client.
MAX_NOTES = 12
MAX_NOTE_CHARS = 400


def clean_notes(raw: Any) -> List[str]:
    """The model's notes as a short list of non-empty strings. Anything that is
    not a list of strings is dropped rather than failing the proposal: notes are
    a courtesy to the user, never a reason to reject a valid workflow."""
    if not isinstance(raw, list):
        return []
    notes: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = ' '.join(item.split())
        if text:
            notes.append(text[:MAX_NOTE_CHARS])
        if len(notes) >= MAX_NOTES:
            break
    return notes


class ProposeWorkflowYamlTool(AssistantTool):
    name = 'propose_workflow_yaml'
    description = (
        'Submit the COMPLETE workflow YAML (the whole document, not a diff) for '
        'validation. Change only what the request needs, plus problems that make '
        'the workflow invalid or unsavable that you can fix confidently (for '
        'example a missing or invalid model). Returns introduced_errors (problems '
        'your change caused; fix these and call again) and existing_errors '
        '(problems the workflow already had; informational). Pass `notes` to list '
        'every value you chose or changed beyond the request, with before -> '
        'after and the reason.'
    )
    parameters: Dict[str, Any] = {
        'type': 'object',
        'properties': {
            'yaml': {
                'type': 'string',
                'description': 'The complete workflow YAML document',
            },
            'notes': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': (
                    'Short plain-language list of every value you chose or changed '
                    'beyond the literal request, one note per item, each with its '
                    'reason and, for an existing value, before -> after. Required '
                    'whenever you changed an existing node or picked a model, '
                    'prompt, temperature or name. Example: "Existing agent '
                    "content_creator: model openai/gpt-4o-mini -> config 'GPT "
                    "mini' (same model; the builder only shows configured "
                    'models)." End the list with: "You can change any of these in '
                    'the node settings, or ask me to adjust."'
                ),
            },
        },
        'required': ['yaml'],
    }
    read_only = True
    label = 'Validating workflow'
    is_proposal = True

    def capture_failed_call(self, args: Any, ctx: ToolContext, error: str) -> None:
        """A call that crashed after its arguments parsed (validation itself
        failed) still leaves the user the YAML it carried, marked invalid, unless a
        real proposal already exists. Skipped when the YAML does not parse into a
        workflow document: there is nothing to show then."""
        if ctx.proposals or not isinstance(args, dict):
            return
        yaml_text = args.get('yaml')
        if (
            not isinstance(yaml_text, str)
            or not yaml_text.strip()
            or len(yaml_text) > ctx.max_yaml_chars
        ):
            return
        data = parse_yaml_document(yaml_text)
        if not isinstance(data, dict) or not isinstance(data.get('arium'), dict):
            return
        problem = YamlValidationError(
            path='', message=f'Validation could not run on this proposal: {error}'
        )
        ctx.proposals.append(
            Proposal(
                yaml=yaml_text,
                valid=False,
                errors=[problem],
                introduced_errors=[problem],
                notes=clean_notes(args.get('notes')),
            )
        )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        yaml_text = args.get('yaml')
        if not isinstance(yaml_text, str) or not yaml_text.strip():
            raise ToolDispatchError("'yaml' must be a non-empty string")
        if len(yaml_text) > ctx.max_yaml_chars:
            raise ToolDispatchError(
                f'YAML is too large (limit {ctx.max_yaml_chars} characters)'
            )

        # A repair round that omits notes keeps the ones from the proposal it replaces
        notes = clean_notes(args.get('notes'))
        if not notes and ctx.last_proposal is not None:
            notes = list(ctx.last_proposal.notes)

        errors, data = await compute_errors_and_document(yaml_text, ctx)

        def classify() -> Tuple[List[YamlValidationError], List[YamlValidationError]]:
            # CPU-bound on a large document, so off the event loop
            baseline_data = (
                _document(ctx, ctx.baseline_yaml) if ctx.baseline_errors else None
            )
            return split_errors(
                errors,
                yaml_text,
                ctx.baseline_errors,
                ctx.baseline_yaml,
                proposed_data=data,
                baseline_data=baseline_data,
            )

        introduced, existing = await asyncio.to_thread(classify)

        proposal = Proposal(
            yaml=yaml_text,
            valid=not errors,
            errors=errors,
            introduced_errors=introduced,
            notes=notes,
        )
        ctx.proposals.append(proposal)

        def listing(items: List[YamlValidationError]) -> List[Dict[str, str]]:
            return [{'path': e.path, 'message': e.message} for e in items]

        if introduced:
            # Feedback to the model, not a failure: the client shows it neutral
            summary = f'Found {len(introduced)} issue(s), refining'
            next_step = (
                'Fix every introduced error and call propose_workflow_yaml again. '
                'Do not touch existing_errors.'
            )
        else:
            summary = (
                'Valid'
                if not existing
                else f'Ready; {len(existing)} existing issue(s) remain'
            )
            next_step = (
                'The proposal is accepted; there is nothing more to validate. Do '
                'not call any tool. Reply now with a short summary of the change '
                'and anything the user must configure. If you did not pass notes, '
                'they cannot be added now, so mention your choices in the reply.'
            )
            if existing:
                next_step += (
                    ' The existing_errors were already in the workflow and are '
                    'not yours to fix; you may mention that they remain.'
                )
        return ToolResult(
            content=json.dumps(
                {
                    'valid': not errors,
                    'introduced_errors': listing(introduced),
                    'existing_errors': listing(existing),
                    'next_step': next_step,
                }
            ),
            ok=not introduced,
            summary=summary,
        )
