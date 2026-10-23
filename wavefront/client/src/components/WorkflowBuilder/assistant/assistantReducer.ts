import type { AssistantEvent } from './events';
import type { AssistantMessage, AssistantStatus, DiffSummary, Proposal, TimelineStep } from './types';

export interface AssistantState {
  messages: AssistantMessage[];
  status: AssistantStatus;
  error?: string;
  // Proposed YAML by proposal id. The contract's Proposal has no YAML field: preview and apply read it from here.
  proposalYaml: Record<string, string>;
  // The editor YAML at the moment the current turn was sent, which the model worked from. Internal, not in the contract.
  turnBaseYaml?: string;
  // The turn's base YAML by proposal id: what the proposal was written against, to tell when the workflow has since
  // changed under it (stale). Absent for a proposal whose turn had no editor to read.
  proposalBase: Record<string, string>;
}

export const initialAssistantState: AssistantState = {
  messages: [],
  status: 'idle',
  proposalYaml: {},
  proposalBase: {},
};

// Injected so the reducer stays pure: the diff is computed from the current editor YAML, the clock is the caller's
export interface AssistantReducerDeps {
  computeDiff: (proposedYaml: string) => DiffSummary;
  // Whether the proposed YAML parses into a drawable graph. Optional: absent means yes.
  canApply?: (proposedYaml: string) => boolean;
  now: () => number;
}

export type AssistantAction =
  | { type: 'user_sent'; text: string; now: number; baseYaml?: string }
  | { type: 'event'; event: AssistantEvent; deps: AssistantReducerDeps }
  // The user stopped the stream
  | { type: 'stopped' }
  // Drops the failed/partial assistant reply after the last user message and streams again
  | { type: 'retry_started'; baseYaml?: string }
  | { type: 'reset' }
  | { type: 'proposal_state'; id: string; state: Proposal['state'] }
  | { type: 'set_error'; message: string };

// Allowed proposal transitions. applied -> pending is the undo of an apply.
const PROPOSAL_TRANSITIONS: Record<Proposal['state'], Proposal['state'][]> = {
  pending: ['applied', 'discarded'],
  applied: ['pending'],
  discarded: [],
};

const cleanNotes = (notes: readonly string[] | undefined): string[] =>
  (notes ?? []).map((n) => n.trim()).filter((n) => n.length > 0);

const messageId = (messages: AssistantMessage[]) => `m${messages.length + 1}`;

// The streaming assistant message is always the last one. Create it when the turn's first event arrives.
const withAssistantMessage = (state: AssistantState, now: number): AssistantMessage[] => {
  const last = state.messages[state.messages.length - 1];
  if (last?.role === 'assistant') return state.messages;
  const created: AssistantMessage = {
    id: messageId(state.messages),
    role: 'assistant',
    text: '',
    steps: [],
    createdAt: now,
  };
  return [...state.messages, created];
};

const updateLast = (messages: AssistantMessage[], update: (m: AssistantMessage) => AssistantMessage) => [
  ...messages.slice(0, -1),
  update(messages[messages.length - 1]),
];

const settleSteps = (steps: TimelineStep[], status: 'done' | 'failed'): TimelineStep[] =>
  steps.some((s) => s.status === 'running') ? steps.map((s) => (s.status === 'running' ? { ...s, status } : s)) : steps;

const upsertStep = (steps: TimelineStep[], step: TimelineStep): TimelineStep[] =>
  steps.some((s) => s.id === step.id) ? steps.map((s) => (s.id === step.id ? { ...s, ...step } : s)) : [...steps, step];

const patchStep = (steps: TimelineStep[], id: string, fallbackLabel: string, patch: Partial<TimelineStep>) =>
  steps.some((s) => s.id === id)
    ? steps.map((s) => (s.id === id ? { ...s, ...patch } : s))
    : [...steps, { id, kind: 'tool' as const, label: fallbackLabel, status: 'running' as const, ...patch }];

// Applies one wire event. Events outside the known set never reach here (see isAssistantEvent); unknown ones are
// ignored by the default branch.
export const applyAssistantEvent = (
  state: AssistantState,
  event: AssistantEvent,
  deps: AssistantReducerDeps
): AssistantState => {
  switch (event.event_type) {
    case 'assistant_started':
      return { ...state, messages: withAssistantMessage(state, deps.now()), status: 'streaming', error: undefined };

    case 'content_delta': {
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        text: m.text + event.content,
      }));
      return { ...state, messages, status: 'streaming' };
    }

    case 'retract': {
      // The replacement is the whole of what may be shown, not a delta
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({ ...m, text: event.content }));
      return { ...state, messages, status: 'streaming' };
    }

    case 'tool_called': {
      const step: TimelineStep = { id: event.tool_call_id, kind: 'tool', label: event.label, status: 'running' };
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        steps: upsertStep(m.steps, step),
      }));
      return { ...state, messages, status: 'streaming' };
    }

    case 'tool_result': {
      // ok=false means the tool ran and reported problems (a proposal with validation errors): feedback the assistant
      // acts on, not a failure. Only tool_failed (the call could not run) is a failed step.
      const patch: Partial<TimelineStep> = {
        status: 'done',
        ...(event.ok ? {} : { tone: 'warning' as const, detail: event.summary || 'Found issues, refining' }),
        ...(event.ok && event.summary ? { detail: event.summary } : {}),
      };
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        steps: patchStep(m.steps, event.tool_call_id, event.tool_name, patch),
      }));
      return { ...state, messages, status: 'streaming' };
    }

    case 'tool_failed': {
      const patch: Partial<TimelineStep> = { status: 'failed', detail: event.error };
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        steps: patchStep(m.steps, event.tool_call_id, event.tool_name, patch),
      }));
      return { ...state, messages, status: 'streaming' };
    }

    case 'proposal': {
      const proposal: Proposal = {
        id: event.proposal_id,
        valid: event.valid,
        errors: event.errors,
        // An older server sends no split: every error blocks, as before
        introducedErrors: event.introduced_errors ?? event.errors,
        diff: deps.computeDiff(event.yaml),
        ...(deps.canApply && !deps.canApply(event.yaml) ? { applicable: false } : {}),
        // Blank notes are dropped; the field is left off when there are none
        ...(cleanNotes(event.notes).length > 0 ? { notes: cleanNotes(event.notes) } : {}),
        state: 'pending',
      };
      const base = withAssistantMessage(state, deps.now());
      // A repair-loop proposal replaces the earlier one on the same message
      const replaced = base[base.length - 1].proposal?.id;
      const proposalYaml = { ...state.proposalYaml, [event.proposal_id]: event.yaml };
      const proposalBase = { ...state.proposalBase };
      if (state.turnBaseYaml !== undefined) proposalBase[event.proposal_id] = state.turnBaseYaml;
      if (replaced && replaced !== event.proposal_id) {
        delete proposalYaml[replaced];
        delete proposalBase[replaced];
      }
      return {
        ...state,
        messages: updateLast(base, (m) => ({ ...m, proposal })),
        status: 'streaming',
        proposalYaml,
        proposalBase,
      };
    }

    case 'output': {
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        text: event.content,
        steps: settleSteps(m.steps, 'done'),
        ...(event.truncated ? { truncated: true } : {}),
      }));
      return { ...state, messages, status: 'idle' };
    }

    case 'error': {
      const messages = updateLast(withAssistantMessage(state, deps.now()), (m) => ({
        ...m,
        steps: settleSteps(m.steps, 'failed'),
      }));
      return { ...state, messages, status: 'error', error: event.message };
    }

    default:
      return state;
  }
};

export const assistantReducer = (state: AssistantState, action: AssistantAction): AssistantState => {
  switch (action.type) {
    case 'user_sent': {
      const message: AssistantMessage = {
        id: messageId(state.messages),
        role: 'user',
        text: action.text,
        steps: [],
        createdAt: action.now,
      };
      return {
        ...state,
        messages: [...state.messages, message],
        status: 'streaming',
        error: undefined,
        turnBaseYaml: action.baseYaml,
      };
    }

    case 'event':
      return applyAssistantEvent(state, action.event, action.deps);

    case 'stopped': {
      if (state.status !== 'streaming') return state;
      const last = state.messages[state.messages.length - 1];
      const messages =
        last?.role === 'assistant'
          ? updateLast(state.messages, (m) => ({ ...m, steps: settleSteps(m.steps, 'failed') }))
          : state.messages;
      return { ...state, messages, status: 'idle' };
    }

    case 'retry_started': {
      let lastUser = -1;
      state.messages.forEach((m, i) => {
        if (m.role === 'user') lastUser = i;
      });
      // Nothing to retry, or a turn is already running
      if (lastUser < 0 || state.status === 'streaming') return state;
      const messages = state.messages.slice(0, lastUser + 1);
      // Proposals on the dropped messages are gone, so their YAML goes too
      const keep = new Set(messages.map((m) => m.proposal?.id).filter((id): id is string => !!id));
      const proposalYaml = Object.fromEntries(Object.entries(state.proposalYaml).filter(([id]) => keep.has(id)));
      const proposalBase = Object.fromEntries(Object.entries(state.proposalBase).filter(([id]) => keep.has(id)));
      return {
        ...state,
        messages,
        status: 'streaming',
        error: undefined,
        proposalYaml,
        proposalBase,
        turnBaseYaml: action.baseYaml,
      };
    }

    case 'reset':
      return initialAssistantState;

    case 'proposal_state': {
      let changed = false;
      const messages = state.messages.map((m) => {
        if (m.proposal?.id !== action.id || !PROPOSAL_TRANSITIONS[m.proposal.state].includes(action.state)) return m;
        changed = true;
        return { ...m, proposal: { ...m.proposal, state: action.state } };
      });
      return changed ? { ...state, messages } : state;
    }

    case 'set_error':
      return { ...state, error: action.message };

    default:
      return state;
  }
};
