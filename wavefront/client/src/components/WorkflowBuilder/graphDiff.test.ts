import { describe, expect, it } from 'vitest';
import type { Node } from '@xyflow/react';
import { NODE_TYPES } from '@app/constants/workflow-builder';
import { changedNodeFields, describeModel, describeNodeChanges, diffGraphs, excerptPair } from './graphDiff';
import { makeFlowEdge, makeStartEdge, type Graph } from './graphUtils';

const agent = (id: string, data: Record<string, unknown> = {}): Node => ({
  id,
  type: NODE_TYPES.AGENT,
  position: { x: 0, y: 0 },
  data: { label: id, ...data },
});

const trigger = (): Node => ({
  id: 'trigger-start',
  type: NODE_TYPES.TRIGGER,
  position: { x: 0, y: 0 },
  data: { label: 'Workflow Trigger' },
});

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
  }
  return value;
};

const frozenCopy = (graph: Graph): Graph => deepFreeze(structuredClone(graph));

// trigger -> a -> b
const sampleGraph = (): Graph => ({
  nodes: [trigger(), agent('a', { job: 'first' }), agent('b', { job: 'second' })],
  edges: [makeStartEdge('trigger-start', 'a'), makeFlowEdge('a', 'b')],
});

describe('diffGraphs', () => {
  it('reports no changes for identical graphs', () => {
    const diff = diffGraphs(sampleGraph(), sampleGraph());
    expect(diff.nodes.size).toBe(0);
    expect(diff.edgesAdded.size).toBe(0);
    expect(diff.edgesRemoved.size).toBe(0);
    expect(diff.summary).toEqual({
      nodesAdded: [],
      nodesModified: [],
      nodesRemoved: [],
      edgesAdded: 0,
      edgesRemoved: 0,
    });
  });

  it('marks a new node and its edge as added', () => {
    const proposed = sampleGraph();
    proposed.nodes.push(agent('c'));
    proposed.edges.push(makeFlowEdge('b', 'c'));
    const diff = diffGraphs(sampleGraph(), proposed);
    expect(diff.nodes.get('c')).toBe('added');
    expect([...diff.nodes.keys()]).toEqual(['c']);
    expect([...diff.edgesAdded]).toEqual([makeFlowEdge('b', 'c').id]);
    expect(diff.summary.nodesAdded).toEqual(['c']);
    expect(diff.summary.edgesAdded).toBe(1);
    expect(diff.summary.edgesRemoved).toBe(0);
  });

  it('marks a node whose data differs as modified', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => (n.id === 'b' ? agent('b', { job: 'changed' }) : n));
    const diff = diffGraphs(sampleGraph(), proposed);
    expect(diff.nodes.get('b')).toBe('modified');
    expect(diff.summary.nodesModified).toEqual(['b']);
    expect(diff.summary.nodesAdded).toEqual([]);
  });

  it('detects nested data changes (arrays and objects)', () => {
    const current: Graph = { nodes: [agent('a', { input_filter: ['x'], model: { name: 'm1' } })], edges: [] };
    const sameNested: Graph = { nodes: [agent('a', { input_filter: ['x'], model: { name: 'm1' } })], edges: [] };
    const changedArray: Graph = { nodes: [agent('a', { input_filter: ['x', 'y'], model: { name: 'm1' } })], edges: [] };
    const changedObject: Graph = { nodes: [agent('a', { input_filter: ['x'], model: { name: 'm2' } })], edges: [] };
    expect(diffGraphs(current, sameNested).nodes.size).toBe(0);
    expect(diffGraphs(current, changedArray).nodes.get('a')).toBe('modified');
    expect(diffGraphs(current, changedObject).nodes.get('a')).toBe('modified');
  });

  it('treats an undefined data value as equal to a missing key', () => {
    const current: Graph = { nodes: [agent('a', { job: 'x', role: undefined })], edges: [] };
    const proposed: Graph = { nodes: [agent('a', { job: 'x' })], edges: [] };
    expect(diffGraphs(current, proposed).nodes.size).toBe(0);
  });

  it('ignores position differences', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => ({ ...n, position: { x: 400, y: 90 } }));
    expect(diffGraphs(sampleGraph(), proposed).nodes.size).toBe(0);
  });

  it('marks a node whose type changed as modified', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => (n.id === 'b' ? { ...n, type: NODE_TYPES.FUNCTION } : n));
    expect(diffGraphs(sampleGraph(), proposed).nodes.get('b')).toBe('modified');
  });

  it('marks nodes and edges missing from the proposal as removed', () => {
    const proposed: Graph = {
      nodes: [trigger(), agent('a', { job: 'first' })],
      edges: [makeStartEdge('trigger-start', 'a')],
    };
    const diff = diffGraphs(sampleGraph(), proposed);
    expect(diff.nodes.get('b')).toBe('removed');
    expect([...diff.edgesRemoved]).toEqual([makeFlowEdge('a', 'b').id]);
    expect(diff.summary).toEqual({
      nodesAdded: [],
      nodesModified: [],
      nodesRemoved: ['b'],
      edgesAdded: 0,
      edgesRemoved: 1,
    });
  });

  it('reports a mix of added, modified and removed nodes', () => {
    const proposed: Graph = {
      nodes: [trigger(), agent('a', { job: 'rewritten' }), agent('d')],
      edges: [makeStartEdge('trigger-start', 'a'), makeFlowEdge('a', 'd')],
    };
    const diff = diffGraphs(sampleGraph(), proposed);
    expect(diff.summary.nodesAdded).toEqual(['d']);
    expect(diff.summary.nodesModified).toEqual(['a']);
    expect(diff.summary.nodesRemoved).toEqual(['b']);
    expect(diff.summary.edgesAdded).toBe(1);
    expect(diff.summary.edgesRemoved).toBe(1);
  });

  it('diffs against an empty current graph: everything is added', () => {
    const diff = diffGraphs({ nodes: [], edges: [] }, sampleGraph());
    expect(diff.summary.nodesAdded).toEqual(['trigger-start', 'a', 'b']);
    expect(diff.summary.edgesAdded).toBe(2);
  });

  it('does not mutate its inputs', () => {
    const proposed = sampleGraph();
    proposed.nodes.push(agent('c'));
    expect(() => diffGraphs(frozenCopy(sampleGraph()), frozenCopy(proposed))).not.toThrow();
  });
});

describe('semantic comparison of node data', () => {
  const modifiedAfter = (before: Record<string, unknown>, after: Record<string, unknown>) =>
    diffGraphs({ nodes: [agent('a', before)], edges: [] }, { nodes: [agent('a', after)], edges: [] }).nodes.get('a');

  it('ignores key order, at any depth', () => {
    const before = { job: 'x', model: { provider: 'rootflo', model_id: 'm' } };
    const after = { model: { model_id: 'm', provider: 'rootflo' }, job: 'x' };
    expect(modifiedAfter(before, after)).toBeUndefined();
  });

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['an empty string', ''],
    ['a blank string', '  \n '],
  ])('treats %s as the same as a missing value', (_name, unset) => {
    expect(modifiedAfter({ job: 'x', role: unset }, { job: 'x' })).toBeUndefined();
    expect(modifiedAfter({ job: 'x' }, { job: 'x', role: unset })).toBeUndefined();
  });

  it('treats an object with nothing set as a missing one', () => {
    expect(modifiedAfter({ job: 'x', parser: { a: null, b: '' } }, { job: 'x' })).toBeUndefined();
    expect(modifiedAfter({ job: 'x', parser: {} }, { job: 'x' })).toBeUndefined();
  });

  it('ignores line endings, trailing whitespace on a line and surrounding blank space in text', () => {
    const before = { job: 'Write a draft.\nBe brief.' };
    const after = { job: '  Write a draft.  \r\nBe brief.\n\n' };
    expect(modifiedAfter(before, after)).toBeUndefined();
  });

  it('still sees a changed word, or changed spacing inside a line', () => {
    expect(modifiedAfter({ job: 'Write a draft.' }, { job: 'Write a draft!' })).toBe('modified');
    expect(modifiedAfter({ job: 'Write  a draft.' }, { job: 'Write a draft.' })).toBe('modified');
    expect(modifiedAfter({ job: 'a\nb' }, { job: 'a\n\nb' })).toBe('modified');
  });

  it('does not hide a set value becoming unset, or the reverse', () => {
    expect(modifiedAfter({ model_id: 'm' }, { model_id: '' })).toBe('modified');
    expect(modifiedAfter({ model_id: '' }, { model_id: 'm' })).toBe('modified');
    expect(modifiedAfter({ temperature: 0 }, {})).toBe('modified');
    expect(modifiedAfter({ isEndNode: false }, {})).toBe('modified');
  });

  it('keeps list order and the difference between an empty list and a missing one', () => {
    const before = { tools: [{ name: 'a' }, { name: 'b' }] };
    const after = { tools: [{ name: 'b' }, { name: 'a' }] };
    expect(modifiedAfter(before, after)).toBe('modified');
    expect(modifiedAfter({ input_filter: [] }, {})).toBe('modified');
  });

  it('never counts the label (it follows the node id)', () => {
    const current: Graph = { nodes: [{ ...agent('a'), data: { label: 'a', job: 'x' } }], edges: [] };
    const proposed: Graph = { nodes: [{ ...agent('a'), data: { label: 'A', job: 'x' } }], edges: [] };
    expect(diffGraphs(current, proposed).nodes.size).toBe(0);
  });
});

describe('changed fields', () => {
  it('names what changed in short labels, once each, in first-seen order', () => {
    const before = agent('a', { job: 'old', provider: '', model: '', model_id: '', tools: [] });
    const after = agent('a', { job: 'new', provider: 'rootflo', model: '', model_id: 'm', tools: [] });
    expect(changedNodeFields(before, after)).toEqual(['prompt', 'model']);
  });

  it('maps temperature and max_tokens to settings, and reports unknown keys as they are', () => {
    const before = agent('a', { temperature: 0.1, max_tokens: 5 });
    expect(changedNodeFields(before, agent('a', { temperature: 0.9 }))).toEqual(['settings']);
    expect(changedNodeFields(agent('a', { custom_key: 1 }), agent('a', { custom_key: 2 }))).toEqual(['custom_key']);
  });

  it('reports a node type change', () => {
    expect(changedNodeFields(agent('a'), { ...agent('a'), type: NODE_TYPES.FUNCTION })).toContain('type');
  });

  it('is empty for an unchanged node', () => {
    expect(changedNodeFields(agent('a', { job: 'x' }), agent('a', { job: 'x' }))).toEqual([]);
  });

  it('appears on the diff and its summary only for modified nodes', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => (n.id === 'b' ? agent('b', { job: 'changed' }) : n));
    proposed.nodes.push(agent('c'));
    const diff = diffGraphs(sampleGraph(), proposed);
    expect(diff.changedFields.get('b')).toEqual(['prompt']);
    expect(diff.changedFields.has('c')).toBe(false);
    expect(diff.summary.changedFields).toEqual({ b: ['prompt'] });
  });

  it('leaves the summary field out when nothing was modified', () => {
    expect(diffGraphs(sampleGraph(), sampleGraph()).summary.changedFields).toBeUndefined();
  });
});

describe('before -> after detail', () => {
  it('reads an inline model block as "name (provider)" and resolves a rootflo id to its config name', () => {
    expect(describeModel({ provider: 'openai', model: 'gpt-4o-mini', model_id: '' })).toBe('gpt-4o-mini (openai)');
    expect(describeModel({ provider: '', model: '', model_id: 'abc' }, { modelNames: { abc: 'Claude Sonnet' } })).toBe(
      'Claude Sonnet'
    );
    expect(describeModel({ model_id: 'abc' })).toBe('abc');
    expect(describeModel({})).toBe('none');
  });

  it('gives one entry per changed label with readable values, in a stable order', () => {
    const before = agent('a', { provider: 'openai', model: 'gpt-4o-mini', job: 'old', temperature: 0.2 });
    const after = agent('a', { provider: '', model: '', model_id: 'abc', job: 'new', temperature: 0.7 });
    const changes = describeNodeChanges(before, after, { modelNames: { abc: 'Claude Sonnet' } });
    expect(changes.map((c) => c.field)).toEqual(['prompt', 'model', 'settings']);
    expect(changes[0]).toEqual({ field: 'prompt', before: 'old', after: 'new' });
    expect(changes[1]).toEqual({ field: 'model', before: 'gpt-4o-mini (openai)', after: 'Claude Sonnet' });
    expect(changes[2]).toEqual({ field: 'settings', before: 'temperature 0.2', after: 'temperature 0.7' });
  });

  it('shortens long values and keeps the differing part visible', () => {
    const prefix = 'You are a careful assistant. '.repeat(6);
    const [before, after] = excerptPair(`${prefix}Answer briefly.`, `${prefix}Answer at length.`);
    expect(before).not.toBe(after);
    expect(before.length).toBeLessThanOrEqual(70);
    expect(after.length).toBeLessThanOrEqual(70);
    expect(after).toContain('at length');
  });

  it('puts changes on the summary only for modified nodes, tagged with the node id', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => (n.id === 'b' ? agent('b', { job: 'changed' }) : n));
    proposed.nodes.push(agent('c'));
    const { summary } = diffGraphs(sampleGraph(), proposed);
    expect(summary.changes).toEqual([{ nodeId: 'b', field: 'prompt', before: 'second', after: 'changed' }]);
    expect(diffGraphs(sampleGraph(), sampleGraph()).summary.changes).toBeUndefined();
  });

  it('does not mutate either graph', () => {
    const proposed = sampleGraph();
    proposed.nodes = proposed.nodes.map((n) => (n.id === 'b' ? agent('b', { job: 'changed' }) : n));
    expect(() => diffGraphs(frozenCopy(sampleGraph()), frozenCopy(proposed))).not.toThrow();
  });
});
