"""The contract between the assistant loop and the tools it can call."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents_module.models.yaml_validation_schemas import YamlValidationError


class ToolDispatchError(Exception):
    """A tool call that could not run: unknown tool, bad arguments, or a crash.

    The message is safe to show the model and the client; it never carries a
    stack trace or provider detail.
    """


@dataclass
class Proposal:
    """A workflow YAML the model put forward, with the verdict on it."""

    yaml: str
    valid: bool
    errors: List[YamlValidationError]
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: The subset of ``errors`` the change introduced; None means "all of them"
    #: (no baseline was taken). The rest were already in the current workflow.
    introduced_errors: Optional[List[YamlValidationError]] = None
    #: Choices the model says it made on the user's behalf (defaults it picked,
    #: values the user did not specify, things still to configure). Shown to the
    #: user next to the proposal; empty when the model gave none.
    notes: List[str] = field(default_factory=list)

    @property
    def introduced(self) -> List[YamlValidationError]:
        """Errors the proposal is responsible for; only these block it."""
        return self.errors if self.introduced_errors is None else self.introduced_errors

    @property
    def existing(self) -> List[YamlValidationError]:
        """Errors the current workflow already had."""
        introduced = {id(e) for e in self.introduced}
        return [e for e in self.errors if id(e) not in introduced]


@dataclass
class ToolResult:
    """What a tool hands back to the loop.

    ``content`` goes to the model as the tool result. ``ok`` is False when the
    tool ran but found problems (a proposal with validation errors); a tool that
    could not run raises ToolDispatchError instead.
    """

    content: str
    ok: bool = True
    summary: Optional[str] = None


@dataclass
class ToolContext:
    """Per-request state shared by the tools.

    ``resources`` is a ResourceProvider and ``workflow_crud_service`` a
    WorkflowCrudService; they are typed Any here only to keep this module free
    of imports that would make the tools hard to fake in tests.
    """

    resources: Any
    workflow_crud_service: Any
    max_yaml_chars: int
    proposals: List[Proposal] = field(default_factory=list)
    #: Validation problems the current workflow had before the request. Empty
    #: means no baseline, so every error in a proposal counts as introduced.
    baseline_errors: List[YamlValidationError] = field(default_factory=list)
    #: The workflow YAML the baseline was taken from; used to follow a node whose
    #: errors moved to a new path.
    baseline_yaml: str = ''
    #: ``(baseline_yaml, parsed document)``, filled the first time the baseline is
    #: parsed so it is parsed once per request, not once per proposal.
    baseline_parse: Optional[Tuple[str, Any]] = None

    @property
    def last_proposal(self) -> Optional[Proposal]:
        """The last proposal wins; it is captured from the tool call, never
        parsed out of the model's prose."""
        return self.proposals[-1] if self.proposals else None


class AssistantTool(ABC):
    """One capability the model can call.

    Adding a capability is one subclass in its own file plus one
    ``ToolRegistry.register`` call; the service and loop never special-case a
    tool by name.
    """

    name: str
    description: str
    #: Flat JSON schema: ``{'type': 'object', 'properties': {...}, 'required': [...]}``.
    #: Properties are limited to what flo_ai's tool formatting carries (``type``,
    #: ``description``, and ``items`` for arrays).
    parameters: Dict[str, Any] = {'type': 'object', 'properties': {}, 'required': []}
    #: True when running the tool changes nothing outside this request.
    read_only: bool = True
    #: When set, the tool is exposed only if the registry was built with this flag.
    feature_flag: Optional[str] = None
    #: Short human string shown in the client while the tool runs.
    label: str = 'Working'
    #: True for a tool that puts a proposal forward for validation. Every call to
    #: it, failed or not, counts against the repair-round cap.
    is_proposal: bool = False

    def capture_failed_call(self, args: Any, ctx: ToolContext, error: str) -> None:
        """Called when a call to this tool could not run, with whatever arguments
        parsed (None when they did not). A proposal tool keeps the best proposal it
        can salvage on ``ctx``. The default does nothing."""

    @abstractmethod
    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Run the tool. Raise ToolDispatchError for a call that cannot proceed."""
