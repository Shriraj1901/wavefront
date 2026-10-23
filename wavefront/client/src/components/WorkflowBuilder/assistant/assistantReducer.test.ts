import { describe, expect, it } from 'vitest';
import {
  assistantReducer,
  initialAssistantState,
  type AssistantAction,
  type AssistantReducerDeps,
  type AssistantState,
} from './assistantReducer';
import type { AssistantEvent } from './events';
import type { DiffSummary } from './types';

const DIFF: DiffSummary = { nodesAdded: ['n1'], nodesModified: [], nodesRemoved: [], edgesAdded: 1, edgesRemoved: 0 };

const deps: AssistantReducerDeps = { computeDiff: () => DIFF, now: () => 1000 };

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
  }
  return value;
};

const run = (actions: AssistantAction[], start: AssistantState = initialAssistantState): AssistantState =>
  actions.reduce((state, action) => assistantReducer(deepFreeze(state), action), start);

const ev = (event: AssistantEvent): AssistantAction => ({ type: 'event', event, deps });
const sent = (text = 'hi'): AssistantAction => ({ type: 'user_sent', text, now: 1 });
const started = ev({ event_type: 'assistant_started' });

const proposalEvent = (id: string, valid = true): AssistantEvent => ({
  event_type: 'proposal',
  proposal_id: id,
  yaml: `yaml-${id}`,
  valid,
  errors: valid ? [] : [{ path: 'arium.agents[0]', message: 'bad' }],
});

const lastMessage = (state: AssistantState) => state.messages[state.messages.length - 1];

describe('assistantReducer', () => {
  it('appends the user message and starts streaming', () => {
    const state = run([sent('build a router')]);
    expect(state.status).toBe('streaming');
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]).toMatchObject({ role: 'user', text: 'build a router', createdAt: 1 });
  });

  it('creates the assistant message on assistant_started and does not duplicate it', () => {
    const state = run([sent(), started, started]);
    expect(state.messages.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(state.status).toBe('streaming');
  });

  it('grows the assistant text with content_delta', () => {
    const state = run([
      sent(),
      started,
      ev({ event_type: 'content_delta', content: 'Hel' }),
      ev({ event_type: 'content_delta', content: 'lo' }),
    ]);
    expect(lastMessage(state).text).toBe('Hello');
    expect(state.messages).toHaveLength(2);
  });

  it('creates the assistant message from a delta when assistant_started was missed', () => {
    const state = run([sent(), ev({ event_type: 'content_delta', content: 'x' })]);
    expect(lastMessage(state)).toMatchObject({ role: 'assistant', text: 'x' });
  });

  it('maps tool_called -> tool_result to a done step with the summary as detail', () => {
    const state = run([
      sent(),
      started,
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'list_resources', label: 'Listing agents' }),
      ev({ event_type: 'tool_result', tool_call_id: 't1', tool_name: 'list_resources', ok: true, summary: '3 found' }),
    ]);
    expect(lastMessage(state).steps).toEqual([
      { id: 't1', kind: 'tool', label: 'Listing agents', status: 'done', detail: '3 found' },
    ]);
  });

  it('treats tool_result ok: false as validation feedback (done, warning tone), not a failure', () => {
    const state = run([
      sent(),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      ev({
        event_type: 'tool_result',
        tool_call_id: 't1',
        tool_name: 'x',
        ok: false,
        summary: 'Found 2 issue(s), refining',
      }),
    ]);
    expect(lastMessage(state).steps[0]).toMatchObject({
      status: 'done',
      tone: 'warning',
      label: 'Doing x',
      detail: 'Found 2 issue(s), refining',
    });
  });

  it('falls back to a generic refining detail when an ok: false result has no summary', () => {
    const state = run([
      sent(),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      ev({ event_type: 'tool_result', tool_call_id: 't1', tool_name: 'x', ok: false }),
    ]);
    expect(lastMessage(state).steps[0]).toMatchObject({
      status: 'done',
      tone: 'warning',
      detail: 'Found issues, refining',
    });
  });

  it('flags the last message truncated from an output event', () => {
    const state = run([sent(), started, ev({ event_type: 'output', content: 'cut', truncated: true })]);
    expect(lastMessage(state)).toMatchObject({ text: 'cut', truncated: true });
    expect(run([sent(), started, ev({ event_type: 'output', content: 'ok' })]).messages[1].truncated).toBeUndefined();
  });

  it('maps tool_failed to a failed step carrying the error', () => {
    const state = run([
      sent(),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      ev({ event_type: 'tool_failed', tool_call_id: 't1', tool_name: 'x', error: 'boom' }),
    ]);
    expect(lastMessage(state).steps).toEqual([
      { id: 't1', kind: 'tool', label: 'Doing x', status: 'failed', detail: 'boom' },
    ]);
    // A failed tool does not end the turn
    expect(state.status).toBe('streaming');
  });

  it('adds a step for a tool_result whose tool_called was never seen', () => {
    const state = run([
      sent(),
      ev({ event_type: 'tool_result', tool_call_id: 't9', tool_name: 'late_tool', ok: true }),
    ]);
    expect(lastMessage(state).steps).toEqual([{ id: 't9', kind: 'tool', label: 'late_tool', status: 'done' }]);
  });

  it('keeps steps in order and does not duplicate a repeated tool_called', () => {
    const called = (id: string) =>
      ev({ event_type: 'tool_called', tool_call_id: id, tool_name: 'x', label: `step ${id}` });
    const state = run([sent(), called('a'), called('b'), called('a')]);
    expect(lastMessage(state).steps.map((s) => s.id)).toEqual(['a', 'b']);
  });

  it('attaches a pending proposal with the injected diff and stores its YAML', () => {
    const state = run([sent(), started, ev(proposalEvent('p1'))]);
    expect(lastMessage(state).proposal).toEqual({
      id: 'p1',
      valid: true,
      errors: [],
      introducedErrors: [],
      diff: DIFF,
      state: 'pending',
    });
    expect(state.proposalYaml).toEqual({ p1: 'yaml-p1' });
  });

  it('marks a proposal not applicable only when deps say the YAML cannot be drawn', () => {
    const unreadable = { ...deps, canApply: () => false };
    const bad = run([sent(), { type: 'event', event: proposalEvent('p1', false), deps: unreadable }]);
    expect(lastMessage(bad).proposal?.applicable).toBe(false);
    const good = run([
      sent(),
      { type: 'event', event: proposalEvent('p2', false), deps: { ...deps, canApply: () => true } },
    ]);
    expect(lastMessage(good).proposal?.applicable).toBeUndefined();
  });

  it('keeps errors on an invalid proposal', () => {
    const state = run([sent(), ev(proposalEvent('p1', false))]);
    expect(lastMessage(state).proposal).toMatchObject({
      valid: false,
      errors: [{ path: 'arium.agents[0]', message: 'bad' }],
      state: 'pending',
    });
  });

  it('treats a missing introduced_errors as every error (older server)', () => {
    const state = run([sent(), ev(proposalEvent('p1', false))]);
    expect(lastMessage(state).proposal?.introducedErrors).toEqual([{ path: 'arium.agents[0]', message: 'bad' }]);
  });

  it('keeps pre-existing errors apart from the introduced ones', () => {
    const existing = { path: 'arium.agents[1].model', message: 'model required' };
    const event: AssistantEvent = {
      ...(proposalEvent('p1', false) as Extract<AssistantEvent, { event_type: 'proposal' }>),
      errors: [existing],
      introduced_errors: [],
    };
    const proposal = lastMessage(run([sent(), ev(event)])).proposal;
    expect(proposal).toMatchObject({ valid: false, errors: [existing], introducedErrors: [] });
  });

  it('passes the proposal YAML to computeDiff', () => {
    const seen: string[] = [];
    const spy: AssistantReducerDeps = {
      now: () => 1,
      computeDiff: (yaml) => {
        seen.push(yaml);
        return DIFF;
      },
    };
    run([sent(), { type: 'event', event: proposalEvent('p1'), deps: spy }]);
    expect(seen).toEqual(['yaml-p1']);
  });

  it('replaces the proposal on the same message when a newer one arrives', () => {
    const state = run([sent(), ev(proposalEvent('p1', false)), ev(proposalEvent('p2'))]);
    expect(lastMessage(state).proposal?.id).toBe('p2');
    expect(state.proposalYaml).toEqual({ p2: 'yaml-p2' });
  });

  it('output sets the full text, settles running steps and returns to idle', () => {
    const state = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'partial' }),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      ev({ event_type: 'output', content: 'final text' }),
    ]);
    expect(lastMessage(state).text).toBe('final text');
    expect(lastMessage(state).steps[0].status).toBe('done');
    expect(state.status).toBe('idle');
  });

  it('error sets status and message and fails running steps, keeping partial text', () => {
    const state = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'partial' }),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      ev({ event_type: 'error', message: 'model unavailable' }),
    ]);
    expect(state.status).toBe('error');
    expect(state.error).toBe('model unavailable');
    expect(lastMessage(state).text).toBe('partial');
    expect(lastMessage(state).steps[0].status).toBe('failed');
  });

  it('retract replaces the streamed text with its content and later deltas continue from it', () => {
    const state = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'unsafe ' }),
      ev({ event_type: 'content_delta', content: 'words' }),
      ev({ event_type: 'retract', content: 'redacted', reason: 'policy' }),
      ev({ event_type: 'content_delta', content: ' tail' }),
    ]);
    expect(lastMessage(state).text).toBe('redacted tail');
    expect(state.status).toBe('streaming');
  });

  it('retract with empty content clears the text and creates the message if it was missing', () => {
    const cleared = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'x' }),
      ev({ event_type: 'retract', content: '' }),
    ]);
    expect(lastMessage(cleared).text).toBe('');
    const created = run([sent(), ev({ event_type: 'retract', content: 'only this' })]);
    expect(lastMessage(created)).toMatchObject({ role: 'assistant', text: 'only this' });
  });

  it('output replaces the streamed deltas instead of appending to them', () => {
    const state = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'Looking.' }),
      ev({ event_type: 'content_delta', content: 'Added it.' }),
      ev({ event_type: 'output', content: 'Looking.\n\nAdded it.' }),
    ]);
    expect(lastMessage(state).text).toBe('Looking.\n\nAdded it.');
    expect(state.status).toBe('idle');
  });

  it('ignores unknown events', () => {
    const before = run([sent(), started]);
    const unknown = { event_type: 'something_new', payload: 1 } as unknown as AssistantEvent;
    expect(assistantReducer(before, ev(unknown))).toBe(before);
  });

  it('sending again clears the error', () => {
    const state = run([sent(), ev({ event_type: 'error', message: 'x' }), sent('again')]);
    expect(state.error).toBeUndefined();
    expect(state.status).toBe('streaming');
  });

  it('stopped ends the stream, fails running steps and keeps the text', () => {
    const state = run([
      sent(),
      ev({ event_type: 'content_delta', content: 'abc' }),
      ev({ event_type: 'tool_called', tool_call_id: 't1', tool_name: 'x', label: 'Doing x' }),
      { type: 'stopped' },
    ]);
    expect(state.status).toBe('idle');
    expect(lastMessage(state).text).toBe('abc');
    expect(lastMessage(state).steps[0].status).toBe('failed');
  });

  it('stopped is a no-op when nothing is streaming', () => {
    expect(assistantReducer(initialAssistantState, { type: 'stopped' })).toBe(initialAssistantState);
  });

  it('retry_started drops the reply after the last user message and its proposal YAML', () => {
    const failed = run([sent('q'), ev(proposalEvent('p1')), ev({ event_type: 'error', message: 'x' })]);
    const state = run([{ type: 'retry_started' }], failed);
    expect(state.messages.map((m) => m.role)).toEqual(['user']);
    expect(state.status).toBe('streaming');
    expect(state.error).toBeUndefined();
    expect(state.proposalYaml).toEqual({});
  });

  it('retry_started keeps earlier turns and their proposals', () => {
    const turn1 = run([sent('one'), ev(proposalEvent('p1')), ev({ event_type: 'output', content: 'done' })]);
    const failed = run([sent('two'), ev({ event_type: 'error', message: 'x' })], turn1);
    const state = run([{ type: 'retry_started' }], failed);
    expect(state.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(state.proposalYaml).toEqual({ p1: 'yaml-p1' });
  });

  it('retry_started is a no-op with no user message or while streaming', () => {
    expect(assistantReducer(initialAssistantState, { type: 'retry_started' })).toBe(initialAssistantState);
    const streaming = run([sent()]);
    expect(assistantReducer(streaming, { type: 'retry_started' })).toBe(streaming);
  });

  it('carries proposal notes onto the proposal, trimmed, and leaves the field off when there are none', () => {
    const withNotes = run([
      sent(),
      ev({ ...(proposalEvent('p1') as Extract<AssistantEvent, { event_type: 'proposal' }>), notes: [' a ', '', 'b'] }),
    ]);
    expect(lastMessage(withNotes).proposal?.notes).toEqual(['a', 'b']);
    const without = run([sent(), ev(proposalEvent('p1'))]);
    expect(lastMessage(without).proposal).not.toHaveProperty('notes');
    const blank = run([
      sent(),
      ev({ ...(proposalEvent('p1') as Extract<AssistantEvent, { event_type: 'proposal' }>), notes: ['  '] }),
    ]);
    expect(lastMessage(blank).proposal).not.toHaveProperty('notes');
  });

  it('reset returns the initial state', () => {
    expect(run([sent(), started, { type: 'reset' }])).toEqual(initialAssistantState);
  });

  describe('proposal_state', () => {
    const withProposal = () => run([sent(), ev(proposalEvent('p1')), ev({ event_type: 'output', content: 'ok' })]);

    it('moves pending -> applied -> pending (undo), and pending -> discarded', () => {
      const applied = run([{ type: 'proposal_state', id: 'p1', state: 'applied' }], withProposal());
      expect(lastMessage(applied).proposal?.state).toBe('applied');
      const undone = run([{ type: 'proposal_state', id: 'p1', state: 'pending' }], applied);
      expect(lastMessage(undone).proposal?.state).toBe('pending');
      const discarded = run([{ type: 'proposal_state', id: 'p1', state: 'discarded' }], withProposal());
      expect(lastMessage(discarded).proposal?.state).toBe('discarded');
    });

    it('ignores disallowed transitions and unknown ids', () => {
      const discarded = run([{ type: 'proposal_state', id: 'p1', state: 'discarded' }], withProposal());
      expect(assistantReducer(discarded, { type: 'proposal_state', id: 'p1', state: 'applied' })).toBe(discarded);
      const pending = withProposal();
      expect(assistantReducer(pending, { type: 'proposal_state', id: 'p1', state: 'pending' })).toBe(pending);
      expect(assistantReducer(pending, { type: 'proposal_state', id: 'nope', state: 'applied' })).toBe(pending);
    });
  });

  it('set_error sets only the error, not the status', () => {
    const state = run([{ type: 'set_error', message: 'cannot apply' }]);
    expect(state.error).toBe('cannot apply');
    expect(state.status).toBe('idle');
  });

  it('never mutates the previous state', () => {
    const frozen = deepFreeze(run([sent(), started]));
    expect(() => assistantReducer(frozen, ev({ event_type: 'content_delta', content: 'x' }))).not.toThrow();
    expect(() => assistantReducer(frozen, ev(proposalEvent('p1')))).not.toThrow();
  });
});

describe('proposal base YAML', () => {
  const baseSent = (baseYaml: string): AssistantAction => ({ type: 'user_sent', text: 'hi', now: 1, baseYaml });

  it('records the YAML the turn was sent against with each proposal', () => {
    const state = run([baseSent('base: 1'), started, ev(proposalEvent('p1'))]);
    expect(state.proposalBase).toEqual({ p1: 'base: 1' });
  });

  it('a repair-loop proposal replaces the earlier one and its base', () => {
    const state = run([baseSent('base: 1'), started, ev(proposalEvent('p1')), ev(proposalEvent('p2'))]);
    expect(state.proposalBase).toEqual({ p2: 'base: 1' });
  });

  it('records nothing when the turn had no editor YAML', () => {
    expect(run([sent(), started, ev(proposalEvent('p1'))]).proposalBase).toEqual({});
  });

  it('keeps each turn its own base, and a retry takes a fresh one', () => {
    const first = run([
      baseSent('base: 1'),
      started,
      ev(proposalEvent('p1')),
      ev({ event_type: 'output', content: 'ok' }),
    ]);
    const second = run(
      [baseSent('base: 2'), started, ev(proposalEvent('p2')), ev({ event_type: 'error', message: 'x' })],
      first
    );
    expect(second.proposalBase).toEqual({ p1: 'base: 1', p2: 'base: 2' });

    const retried = run([{ type: 'retry_started', baseYaml: 'base: 3' }, started, ev(proposalEvent('p3'))], second);
    expect(retried.proposalBase).toEqual({ p1: 'base: 1', p3: 'base: 3' });
  });

  it('reset clears the bases', () => {
    expect(run([baseSent('base: 1'), started, ev(proposalEvent('p1')), { type: 'reset' }]).proposalBase).toEqual({});
  });
});
