import type { AssistantMessage, AssistantStatus, DiffSummary, Proposal, ProposalError } from '../types';

// 'pending-valid': no errors from the change (pre-existing issues may remain, see hasExistingIssues).
// 'pending-invalid': the change itself introduced errors. Apply stays available: the user can fix them in the
// editor or ask the assistant to; the workflow just will not save until they are fixed.
export type CardState = 'pending-valid' | 'pending-invalid' | 'applied' | 'discarded';

// The change introduced errors of its own. This does NOT block Apply (see getApplyState).
export const isBlocked = (proposal: Proposal): boolean => proposal.introducedErrors.length > 0;

// The proposed YAML could not be read or drawn at all: the only thing that blocks Apply and Preview
export const isUnreadable = (proposal: Proposal): boolean => proposal.applicable === false;

// Every validation problem left in the proposed workflow, introduced or already there
export const getRemainingIssues = (proposal: Proposal): ProposalError[] => proposal.errors;

export const getCardState = (proposal: Proposal): CardState => {
  if (proposal.state === 'applied') return 'applied';
  if (proposal.state === 'discarded') return 'discarded';
  return isBlocked(proposal) ? 'pending-invalid' : 'pending-valid';
};

// Problems the workflow already had that the proposal does not fix: shown as a warning, never as a blocker
export const getExistingErrors = (proposal: Proposal): ProposalError[] => {
  const introduced = new Set(proposal.introducedErrors.map((e) => `${e.path}\u0000${e.message}`));
  return proposal.errors.filter((e) => !introduced.has(`${e.path}\u0000${e.message}`));
};

export const hasExistingIssues = (proposal: Proposal): boolean =>
  !isBlocked(proposal) && getExistingErrors(proposal).length > 0;

export const STALE_REASON = 'The workflow changed after this proposal was made. Applying would overwrite your edits.';
export const REDO_LABEL = 'Redo on the current workflow';

// The editor re-serializes YAML (line endings, trailing newline), so only differences that survive this count
export const normalizeYaml = (yaml: string): string => yaml.replace(/\r\n?/g, '\n').trimEnd();

// Stale: the proposal was written against a workflow that is no longer the one on the canvas. Unknown on either side
// (no editor to read, or a proposal from before bases were recorded) is never stale.
export const isStaleAgainst = (baseYaml: string | undefined, currentYaml: string | undefined): boolean =>
  baseYaml !== undefined && currentYaml !== undefined && normalizeYaml(baseYaml) !== normalizeYaml(currentYaml);

// Flags every pending proposal whose base differs from the editor's YAML now. Applied and discarded ones are history,
// not stale. Returns the same array when nothing changes, so memoized consumers do not re-render.
export const markStaleProposals = (
  messages: AssistantMessage[],
  bases: Readonly<Record<string, string>>,
  currentYaml: string | undefined
): AssistantMessage[] => {
  let changed = false;
  const next = messages.map((m) => {
    const proposal = m.proposal;
    if (!proposal) return m;
    const stale = proposal.state === 'pending' && isStaleAgainst(bases[proposal.id], currentYaml);
    if (!!proposal.stale === stale) return m;
    changed = true;
    const updated: Proposal = { ...proposal, stale: true };
    if (!stale) delete updated.stale;
    return { ...m, proposal: updated };
  });
  return changed ? next : messages;
};

// What the user asked in the turn that produced the message at `index`: the closest user message before it
export const getTurnUserText = (messages: readonly AssistantMessage[], index: number): string | undefined => {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') return messages[i].text;
  }
  return undefined;
};

// A stale proposal can be redone against the workflow as it is now, by asking again
export const canRedo = (proposal: Proposal, redoText: string | undefined): boolean =>
  proposal.state === 'pending' && proposal.stale === true && !!redoText?.trim();

export interface ApplyState {
  enabled: boolean;
  // Shown next to the disabled button
  reason?: string;
}

export const getApplyState = (proposal: Proposal, status: AssistantStatus): ApplyState => {
  if (proposal.state !== 'pending') return { enabled: false };
  if (proposal.stale) return { enabled: false, reason: STALE_REASON };
  if (isUnreadable(proposal)) {
    return { enabled: false, reason: "Can't apply: the proposed workflow could not be read." };
  }
  if (status === 'streaming') return { enabled: false, reason: 'Wait for the reply to finish.' };
  return { enabled: true };
};

// Preview needs something drawable: a proposal whose YAML parses can be previewed even with validation errors. A stale
// one is not previewed: its diff would be drawn against edits it knows nothing about.
export const canPreview = (proposal: Proposal): boolean =>
  proposal.state === 'pending' && !isUnreadable(proposal) && !proposal.stale;

// Why Preview is disabled, for its tooltip
export const getPreviewDisabledReason = (proposal: Proposal): string | undefined => {
  if (canPreview(proposal)) return undefined;
  return proposal.stale ? STALE_REASON : 'Nothing to preview: the proposed workflow could not be read';
};

export interface DiffTotals {
  added: number;
  modified: number;
  removed: number;
  edgesAdded: number;
  edgesRemoved: number;
  isEmpty: boolean;
}

export const getDiffTotals = (diff: DiffSummary): DiffTotals => {
  const totals = {
    added: diff.nodesAdded.length,
    modified: diff.nodesModified.length,
    removed: diff.nodesRemoved.length,
    edgesAdded: diff.edgesAdded,
    edgesRemoved: diff.edgesRemoved,
  };
  return { ...totals, isEmpty: Object.values(totals).every((n) => n === 0) };
};

export interface DiffNodeEntry {
  name: string;
  status: 'added' | 'modified' | 'removed';
  // Modified nodes only: short labels of what changed, e.g. 'prompt, model'
  changed?: string;
  // Modified nodes only: before -> after per changed field. Absent when the diff carries no detail (older build).
  lines?: ChangeLine[];
}

export interface ChangeLine {
  field: string;
  before: string;
  after: string;
}

// Before -> after lines for a modified node, in the diff's stable order
export const describeChangeLines = (diff: DiffSummary, name: string): ChangeLine[] =>
  (diff.changes ?? []).filter((c) => c.nodeId === name).map(({ field, before, after }) => ({ field, before, after }));

// 'prompt, model' for a modified node; undefined when the diff carries no detail for it
export const describeChangedFields = (diff: DiffSummary, name: string): string | undefined => {
  const fields = diff.changedFields?.[name];
  return fields && fields.length > 0 ? fields.join(', ') : undefined;
};

// Flat node list (added, modified, removed order) cut to `limit`; `hidden` feeds the "+N more" row
export const listDiffNodes = (diff: DiffSummary, limit: number): { items: DiffNodeEntry[]; hidden: number } => {
  const all: DiffNodeEntry[] = [
    ...diff.nodesAdded.map((name) => ({ name, status: 'added' as const })),
    ...diff.nodesModified.map(
      (name): DiffNodeEntry => ({
        name,
        status: 'modified',
        changed: describeChangedFields(diff, name),
        ...(describeChangeLines(diff, name).length > 0 ? { lines: describeChangeLines(diff, name) } : {}),
      })
    ),
    ...diff.nodesRemoved.map((name) => ({ name, status: 'removed' as const })),
  ];
  return { items: all.slice(0, limit), hidden: Math.max(0, all.length - limit) };
};

export interface DiffGroups {
  // What the user asked for: new steps
  requested: DiffNodeEntry[];
  // Existing steps the proposal touches
  alsoChanged: DiffNodeEntry[];
  removed: DiffNodeEntry[];
}

export const groupDiffEntries = (items: readonly DiffNodeEntry[]): DiffGroups => ({
  requested: items.filter((i) => i.status === 'added'),
  alsoChanged: items.filter((i) => i.status === 'modified'),
  removed: items.filter((i) => i.status === 'removed'),
});

// The assistant's notes on choices it made for the user: trimmed, blanks dropped; empty for older servers
export const getProposalNotes = (proposal: Proposal): string[] =>
  (proposal.notes ?? []).map((n) => n.trim()).filter((n) => n.length > 0);

export const summarizeDiffText = (diff: DiffSummary): string => {
  const t = getDiffTotals(diff);
  const parts = [
    t.added && `${t.added} added`,
    t.modified && `${t.modified} modified`,
    t.removed && `${t.removed} removed`,
    t.edgesAdded && `${t.edgesAdded} ${t.edgesAdded === 1 ? 'edge' : 'edges'} added`,
    t.edgesRemoved && `${t.edgesRemoved} ${t.edgesRemoved === 1 ? 'edge' : 'edges'} removed`,
  ].filter(Boolean);
  return parts.length ? parts.join(', ') : 'No changes';
};

// The timeline is open while work is in flight or something failed, and collapsed once the message settles cleanly
export const shouldExpandSteps = (steps: readonly { status: 'running' | 'done' | 'failed' }[]): boolean =>
  steps.some((s) => s.status === 'running' || s.status === 'failed');

export const summarizeSteps = (steps: readonly { status: 'running' | 'done' | 'failed' }[]): string => {
  const failed = steps.filter((s) => s.status === 'failed').length;
  const base = `${steps.length} ${steps.length === 1 ? 'step' : 'steps'}`;
  return failed ? `${base}, ${failed} failed` : base;
};
