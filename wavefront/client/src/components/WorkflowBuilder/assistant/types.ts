// The handoff contract between the assistant framework and the UI. The UI reads it only through useWorkflowAssistant().
// Later changes must be additive: a breaking change needs the UI plan updated too.

export type AssistantStatus = 'idle' | 'streaming' | 'error';

export interface TimelineStep {
  id: string;
  kind: 'thought' | 'tool';
  // Human-readable, already formatted
  label: string;
  status: 'running' | 'done' | 'failed';
  // Additive: 'warning' marks a finished step that is feedback rather than a failure (a validation pass that found
  // issues the assistant is now refining). Rendered amber, never as a red 'Failed'.
  tone?: 'warning';
  detail?: string;
}

export interface ProposalError {
  path: string;
  message: string;
}

// One thing that changed on an existing node, as the review card shows it. Values are short, readable strings.
export interface FieldChange {
  nodeId: string;
  // Same label vocabulary as changedFields ('prompt', 'model', 'settings', ...)
  field: string;
  before: string;
  after: string;
}

export interface DiffSummary {
  nodesAdded: string[];
  nodesModified: string[];
  nodesRemoved: string[];
  edgesAdded: number;
  edgesRemoved: number;
  // Additive: for each modified node id, short labels of what changed ('prompt', 'model', ...). Absent when nothing
  // was modified.
  changedFields?: Record<string, string[]>;
  // Additive: before -> after detail for each changed field of each modified node, in a stable order. Absent when
  // nothing was modified (or from an older build): fall back to changedFields.
  changes?: FieldChange[];
}

export interface Proposal {
  id: string;
  // No errors at all in the proposed YAML. false does not by itself block Apply: see introducedErrors.
  valid: boolean;
  // Every problem in the proposed YAML, including ones the workflow already had
  errors: ProposalError[];
  // The subset of errors the change itself introduced. Apply is NOT blocked by them (the user may fix them in the
  // editor, or ask the assistant to); the workflow will not save until every error is fixed.
  introducedErrors: ProposalError[];
  diff: DiffSummary;
  // Additive: false when the proposed YAML could not be parsed into a drawable graph, which is the only thing that
  // disables Apply and Preview. Absent means applicable.
  applicable?: boolean;
  // Additive: choices the assistant made on the user's behalf (values the user did not specify, defaults, things
  // still to configure). Absent from older servers.
  notes?: string[];
  // Additive: true when the workflow on the canvas changed after this proposal was made, so applying it would overwrite
  // those edits. A stale proposal cannot be applied or previewed; the card offers to redo the request instead.
  stale?: boolean;
  state: 'pending' | 'applied' | 'discarded';
}

export interface AssistantMessage {
  id: string;
  role: 'user' | 'assistant';
  // Markdown, may grow while streaming
  text: string;
  steps: TimelineStep[];
  proposal?: Proposal;
  // Additive: the provider cut this reply off at its output limit; the UI offers a 'Continue' button
  truncated?: boolean;
  createdAt: number;
}

export interface WorkflowAssistant {
  // false when no assistant model is configured / no permission
  available: boolean;
  messages: AssistantMessage[];
  status: AssistantStatus;
  error?: string;
  send(text: string): void;
  stop(): void;
  retry(): void;
  reset(): void;
  // The app's configured LLM configs; empty => no usable model
  models: { id: string; name: string }[];
  selectedModelId?: string;
  selectModel(id: string): void;
  // Shows/clears the canvas overlay
  previewProposal(id: string | null): void;
  applyProposal(id: string): void;
  discardProposal(id: string): void;
  canUndo: boolean;
  undoLastApply(): void;
  focusNode(nodeId: string): void;
}
