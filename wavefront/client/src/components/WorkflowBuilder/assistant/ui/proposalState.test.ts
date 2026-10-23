import { describe, expect, it } from 'vitest';
import type { AssistantMessage, Proposal } from '../types';
import {
  STALE_REASON,
  canPreview,
  canRedo,
  getPreviewDisabledReason,
  getTurnUserText,
  isStaleAgainst,
  markStaleProposals,
  normalizeYaml,
  describeChangeLines,
  describeChangedFields,
  getApplyState,
  getCardState,
  getDiffTotals,
  getExistingErrors,
  getProposalNotes,
  getRemainingIssues,
  groupDiffEntries,
  hasExistingIssues,
  listDiffNodes,
  shouldExpandSteps,
  summarizeSteps,
  summarizeDiffText,
} from './proposalState';

const base: Proposal = {
  id: 'p',
  valid: true,
  errors: [],
  introducedErrors: [],
  diff: {
    nodesAdded: ['a'],
    nodesModified: ['m1', 'm2'],
    nodesRemoved: ['r'],
    edgesAdded: 1,
    edgesRemoved: 0,
    changedFields: { m1: ['prompt', 'model'] },
  },
  state: 'pending',
};

const oldIssue = { path: 'arium.agents[1].model', message: 'model required' };
const newIssue = { path: 'arium.workflow.end', message: 'bad end' };
// Valid per the change, but the workflow already had an error
const existingOnly: Proposal = { ...base, valid: false, errors: [oldIssue], introducedErrors: [] };
// The change itself broke something (and the old error is still there)
const introduced: Proposal = { ...base, valid: false, errors: [oldIssue, newIssue], introducedErrors: [newIssue] };

describe('getCardState', () => {
  it('maps lifecycle and validity', () => {
    expect(getCardState(base)).toBe('pending-valid');
    expect(getCardState(introduced)).toBe('pending-invalid');
    expect(getCardState(existingOnly)).toBe('pending-valid');
    expect(getCardState({ ...base, state: 'applied' })).toBe('applied');
    expect(getCardState({ ...introduced, state: 'discarded' })).toBe('discarded');
  });
});

describe('getApplyState', () => {
  it('enables a valid pending proposal', () => {
    expect(getApplyState(base, 'idle')).toEqual({ enabled: true });
  });
  it('keeps Apply enabled for a proposal that introduces errors (the workflow just will not save)', () => {
    expect(getApplyState(introduced, 'idle')).toEqual({ enabled: true });
  });
  it('disables Apply with a reason only when the proposed YAML is unreadable', () => {
    const s = getApplyState({ ...introduced, applicable: false }, 'idle');
    expect(s.enabled).toBe(false);
    expect(s.reason).toMatch(/could not be read/);
    expect(getApplyState({ ...base, applicable: true }, 'idle')).toEqual({ enabled: true });
  });
  it('enables a proposal whose only errors already existed', () => {
    expect(getApplyState(existingOnly, 'idle')).toEqual({ enabled: true });
  });
  it('waits for the stream, and is disabled once settled', () => {
    expect(getApplyState(base, 'streaming').enabled).toBe(false);
    expect(getApplyState({ ...base, state: 'applied' }, 'idle')).toEqual({ enabled: false });
  });
  it('allows apply after an error status (the proposal itself is fine)', () => {
    expect(getApplyState(base, 'error').enabled).toBe(true);
  });
});

describe('canPreview', () => {
  it('needs a pending proposal whose YAML is readable, errors or not', () => {
    expect(canPreview(base)).toBe(true);
    expect(canPreview(existingOnly)).toBe(true);
    expect(canPreview(introduced)).toBe(true);
    expect(canPreview({ ...introduced, applicable: false })).toBe(false);
    expect(canPreview({ ...base, state: 'discarded' })).toBe(false);
  });
});

describe('existing issues', () => {
  it('lists errors the change did not introduce', () => {
    expect(getExistingErrors(existingOnly)).toEqual([oldIssue]);
    expect(getExistingErrors(introduced)).toEqual([oldIssue]);
    expect(getExistingErrors(base)).toEqual([]);
  });
  it('lists every remaining issue, introduced or not', () => {
    expect(getRemainingIssues(introduced)).toEqual([oldIssue, newIssue]);
    expect(getRemainingIssues(base)).toEqual([]);
  });
  it('splits existing-only warnings from introduced ones', () => {
    expect(hasExistingIssues(existingOnly)).toBe(true);
    expect(hasExistingIssues(introduced)).toBe(false);
    expect(hasExistingIssues(base)).toBe(false);
  });
});

describe('changed fields', () => {
  it('joins the labels of a modified node', () => {
    expect(describeChangedFields(base.diff, 'm1')).toBe('prompt, model');
    expect(describeChangedFields(base.diff, 'm2')).toBeUndefined();
  });
  it('rides along on the node list', () => {
    const { items } = listDiffNodes(base.diff, 10);
    expect(items.find((i) => i.name === 'm1')?.changed).toBe('prompt, model');
    expect(items.find((i) => i.name === 'a')?.changed).toBeUndefined();
  });
});

describe('diff helpers', () => {
  it('totals and text', () => {
    expect(getDiffTotals(base.diff)).toMatchObject({ added: 1, modified: 2, removed: 1, isEmpty: false });
    expect(summarizeDiffText(base.diff)).toBe('1 added, 2 modified, 1 removed, 1 edge added');
    const empty = { nodesAdded: [], nodesModified: [], nodesRemoved: [], edgesAdded: 0, edgesRemoved: 0 };
    expect(getDiffTotals(empty).isEmpty).toBe(true);
    expect(summarizeDiffText(empty)).toBe('No changes');
  });
  it('truncates the node list with a hidden count', () => {
    const { items, hidden } = listDiffNodes(base.diff, 3);
    expect(items.map((i) => i.name)).toEqual(['a', 'm1', 'm2']);
    expect(hidden).toBe(1);
    expect(listDiffNodes(base.diff, 10).hidden).toBe(0);
  });
});

describe('steps', () => {
  it('expands for running or failed steps only', () => {
    expect(shouldExpandSteps([{ status: 'done' }])).toBe(false);
    expect(shouldExpandSteps([{ status: 'done' }, { status: 'running' }])).toBe(true);
    expect(shouldExpandSteps([{ status: 'failed' }])).toBe(true);
    expect(shouldExpandSteps([])).toBe(false);
  });
  it('summarizes counts', () => {
    expect(summarizeSteps([{ status: 'done' }])).toBe('1 step');
    expect(summarizeSteps([{ status: 'done' }, { status: 'failed' }])).toBe('2 steps, 1 failed');
  });
});

describe('before -> after lines and grouping', () => {
  const detailed: Proposal = {
    ...base,
    diff: {
      ...base.diff,
      changes: [
        { nodeId: 'm1', field: 'prompt', before: 'old', after: 'new' },
        { nodeId: 'm1', field: 'model', before: 'gpt-4o-mini (openai)', after: 'Claude Sonnet' },
      ],
    },
  };

  it('returns the change lines of one node in order, and none without detail', () => {
    expect(describeChangeLines(detailed.diff, 'm1').map((l) => l.field)).toEqual(['prompt', 'model']);
    expect(describeChangeLines(detailed.diff, 'm2')).toEqual([]);
    expect(describeChangeLines(base.diff, 'm1')).toEqual([]);
  });

  it('puts the lines on modified entries only', () => {
    const { items } = listDiffNodes(detailed.diff, 10);
    expect(items.find((i) => i.name === 'm1')?.lines).toHaveLength(2);
    expect(items.find((i) => i.name === 'm2')?.lines).toBeUndefined();
    expect(items.find((i) => i.name === 'a')?.lines).toBeUndefined();
  });

  it('groups requested (added), also changed (modified) and removed', () => {
    const groups = groupDiffEntries(listDiffNodes(base.diff, 10).items);
    expect(groups.requested.map((i) => i.name)).toEqual(['a']);
    expect(groups.alsoChanged.map((i) => i.name)).toEqual(['m1', 'm2']);
    expect(groups.removed.map((i) => i.name)).toEqual(['r']);
  });

  it('cleans the notes and tolerates their absence', () => {
    expect(getProposalNotes(base)).toEqual([]);
    expect(getProposalNotes({ ...base, notes: [' keep ', '', '   '] })).toEqual(['keep']);
  });
});

describe('stale proposals', () => {
  const stale: Proposal = { ...base, stale: true };

  it('disables Apply and Preview with the same reason', () => {
    expect(getApplyState(stale, 'idle')).toEqual({ enabled: false, reason: STALE_REASON });
    expect(canPreview(stale)).toBe(false);
    expect(getPreviewDisabledReason(stale)).toBe(STALE_REASON);
    expect(STALE_REASON).toBe(
      'The workflow changed after this proposal was made. Applying would overwrite your edits.'
    );
  });

  it('leaves a fresh proposal alone', () => {
    expect(getApplyState(base, 'idle')).toEqual({ enabled: true });
    expect(canPreview(base)).toBe(true);
    expect(getPreviewDisabledReason(base)).toBeUndefined();
  });

  it('explains an unreadable proposal differently from a stale one', () => {
    expect(getPreviewDisabledReason({ ...base, applicable: false })).toMatch(/could not be read/);
  });

  it('offers a redo only for a pending stale proposal with a request to repeat', () => {
    expect(canRedo(stale, 'add an editor')).toBe(true);
    expect(canRedo(stale, undefined)).toBe(false);
    expect(canRedo(stale, '  ')).toBe(false);
    expect(canRedo(base, 'add an editor')).toBe(false);
    expect(canRedo({ ...stale, state: 'applied' }, 'add an editor')).toBe(false);
  });
});

describe('isStaleAgainst', () => {
  it('is stale only when the YAML really differs', () => {
    expect(isStaleAgainst('a: 1\n', 'a: 2\n')).toBe(true);
    expect(isStaleAgainst('a: 1\n', 'a: 1\n')).toBe(false);
  });

  it('ignores line endings and trailing whitespace the editor may re-serialize', () => {
    expect(normalizeYaml('a: 1\r\nb: 2\r\n\n')).toBe('a: 1\nb: 2');
    expect(isStaleAgainst('a: 1\r\nb: 2', 'a: 1\nb: 2\n')).toBe(false);
  });

  it('is never stale when either side is unknown', () => {
    expect(isStaleAgainst(undefined, 'a: 1')).toBe(false);
    expect(isStaleAgainst('a: 1', undefined)).toBe(false);
  });
});

describe('markStaleProposals', () => {
  const message = (id: string, proposal?: Proposal): AssistantMessage => ({
    id,
    role: proposal ? 'assistant' : 'user',
    text: id,
    steps: [],
    createdAt: 0,
    ...(proposal ? { proposal } : {}),
  });
  const withId = (id: string, over: Partial<Proposal> = {}): Proposal => ({ ...base, id, ...over });

  it('flags pending proposals whose base differs and clears the flag when it matches again', () => {
    const messages = [message('u1'), message('a1', withId('p1')), message('a2', withId('p2'))];
    const bases = { p1: 'x: 1', p2: 'x: 2' };

    const flagged = markStaleProposals(messages, bases, 'x: 2');
    expect(flagged[1].proposal?.stale).toBe(true);
    expect(flagged[2].proposal?.stale).toBeUndefined();

    // undo brought the editor back to p1's base
    const cleared = markStaleProposals(flagged, bases, 'x: 1');
    expect(cleared[1].proposal?.stale).toBeUndefined();
    expect(cleared[2].proposal?.stale).toBe(true);
    expect('stale' in (cleared[1].proposal ?? {})).toBe(false);
  });

  it('does not mark applied or discarded proposals, or ones with no recorded base', () => {
    const messages = [
      message('a1', withId('p1', { state: 'applied' })),
      message('a2', withId('p2', { state: 'discarded' })),
      message('a3', withId('p3')),
    ];
    const result = markStaleProposals(messages, { p1: 'x: 1', p2: 'x: 1' }, 'x: 9');
    expect(result.map((m) => m.proposal?.stale)).toEqual([undefined, undefined, undefined]);
  });

  it('returns the same array when nothing changes and never mutates its input', () => {
    const messages = [message('a1', withId('p1'))];
    expect(markStaleProposals(messages, { p1: 'x: 1' }, 'x: 1')).toBe(messages);
    const frozen = Object.freeze([Object.freeze(message('a1', Object.freeze(withId('p1'))))]);
    expect(() => markStaleProposals(frozen as AssistantMessage[], { p1: 'x: 1' }, 'x: 2')).not.toThrow();
  });

  it('does nothing without an editor YAML to compare', () => {
    const messages = [message('a1', withId('p1'))];
    expect(markStaleProposals(messages, { p1: 'x: 1' }, undefined)).toBe(messages);
  });
});

describe('getTurnUserText', () => {
  const msg = (role: 'user' | 'assistant', text: string): AssistantMessage => ({
    id: text,
    role,
    text,
    steps: [],
    createdAt: 0,
  });

  it('is the closest user message before the reply', () => {
    const messages = [msg('user', 'first'), msg('assistant', 'r1'), msg('user', 'second'), msg('assistant', 'r2')];
    expect(getTurnUserText(messages, 3)).toBe('second');
    expect(getTurnUserText(messages, 1)).toBe('first');
    expect(getTurnUserText(messages, 0)).toBeUndefined();
  });
});
