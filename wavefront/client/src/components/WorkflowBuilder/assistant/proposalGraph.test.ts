import { describe, expect, it } from 'vitest';
import { NODE_TYPES } from '@app/constants/workflow-builder';
import { DEFAULT_WORKFLOW_YAML } from '@app/pages/apps/[appId]/workflows/defaultWorkflowYaml';
import { getNodeSize } from '../autoLayout';
import { diffGraphs } from '../graphDiff';
import { applyGraphAction } from '../graphUtils';
import { createWorkflowNode } from '../nodeFactory';
import { serializeGraphToYaml } from '../yamlSync';
import {
  EMPTY_DIFF_SUMMARY,
  buildDisplayNodes,
  buildPreviewOverlay,
  ghostNodeIds,
  parseGraph,
  positionProposedGraph,
  previewFocusNodeIds,
  summarizeYamlDiff,
} from './proposalGraph';

// DEFAULT_WORKFLOW_YAML plus one agent wired after `editor`
const proposedYaml = (): string => {
  const graph = parseGraph(DEFAULT_WORKFLOW_YAML);
  if (!graph) throw new Error('default YAML must parse');
  const node = createWorkflowNode(NODE_TYPES.AGENT, 'fact_checker', { x: 0, y: 0 });
  const next = applyGraphAction(graph, { type: 'addNode', node, connectFrom: { source: 'editor' } });
  const result = serializeGraphToYaml(next.nodes, next.edges, DEFAULT_WORKFLOW_YAML);
  if (!result.ok) throw new Error(result.error);
  return result.yaml;
};

describe('buildPreviewOverlay / summarizeYamlDiff', () => {
  it('reports no changes for the same YAML', () => {
    expect(summarizeYamlDiff(DEFAULT_WORKFLOW_YAML, DEFAULT_WORKFLOW_YAML)).toEqual(EMPTY_DIFF_SUMMARY);
  });

  it('reports an added node and its edge from real YAML', () => {
    const diff = summarizeYamlDiff(DEFAULT_WORKFLOW_YAML, proposedYaml());
    expect(diff.nodesAdded).toEqual(['fact_checker']);
    expect(diff.nodesRemoved).toEqual([]);
    expect(diff.edgesAdded).toBeGreaterThanOrEqual(1);
  });

  it('marks everything added when the current YAML is empty', () => {
    const diff = summarizeYamlDiff('', DEFAULT_WORKFLOW_YAML);
    expect(diff.nodesAdded).toContain('content_creator');
    expect(diff.nodesRemoved).toEqual([]);
  });

  it('returns null / an empty summary when a side cannot be drawn', () => {
    expect(buildPreviewOverlay(DEFAULT_WORKFLOW_YAML, 'arium: [unclosed')).toBeNull();
    expect(buildPreviewOverlay(DEFAULT_WORKFLOW_YAML, 'not_arium: 1')).toBeNull();
    expect(summarizeYamlDiff(DEFAULT_WORKFLOW_YAML, 'not_arium: 1')).toEqual(EMPTY_DIFF_SUMMARY);
  });

  it('carries the proposed graph for the ghost preview', () => {
    const overlay = buildPreviewOverlay(DEFAULT_WORKFLOW_YAML, proposedYaml());
    expect(overlay?.diff.nodes.get('fact_checker')).toBe('added');
    expect(overlay?.proposed.nodes.some((n) => n.id === 'fact_checker')).toBe(true);
  });
});

describe('positionProposedGraph', () => {
  it('keeps the positions of nodes already on the canvas and places new ones', () => {
    const current = parseGraph(DEFAULT_WORKFLOW_YAML);
    const proposed = parseGraph(proposedYaml());
    if (!current || !proposed) throw new Error('YAML must parse');
    const onCanvas = {
      nodes: current.nodes.map((n, i) => ({ ...n, position: { x: 500 + i * 10, y: 700 + i * 10 } })),
      edges: current.edges,
    };
    const positioned = positionProposedGraph(onCanvas, proposed, 'LR');
    onCanvas.nodes.forEach((n) => {
      expect(positioned.nodes.find((p) => p.id === n.id)?.position).toEqual(n.position);
    });
    const added = positioned.nodes.find((n) => n.id === 'fact_checker');
    expect(added).toBeDefined();
    expect(Number.isFinite(added?.position.x)).toBe(true);
    expect(Number.isFinite(added?.position.y)).toBe(true);
    expect(positioned.nodes).toHaveLength(proposed.nodes.length);
  });

  it('lays the graph out when nothing is shared', () => {
    const proposed = parseGraph(DEFAULT_WORKFLOW_YAML);
    if (!proposed) throw new Error('YAML must parse');
    const positioned = positionProposedGraph({ nodes: [], edges: [] }, proposed, 'LR');
    expect(positioned.nodes).toHaveLength(proposed.nodes.length);
    expect(positioned.nodes.some((n) => n.position.x !== 0 || n.position.y !== 0)).toBe(true);
  });

  it('does not mutate the proposed graph', () => {
    const proposed = parseGraph(proposedYaml());
    if (!proposed) throw new Error('YAML must parse');
    const before = structuredClone(proposed.nodes.map((n) => n.position));
    positionProposedGraph({ nodes: [], edges: [] }, proposed, 'LR');
    expect(proposed.nodes.map((n) => n.position)).toEqual(before);
  });
});

describe('ghost positions', () => {
  const setup = () => {
    const current = parseGraph(DEFAULT_WORKFLOW_YAML);
    const proposed = parseGraph(proposedYaml());
    if (!current || !proposed) throw new Error('YAML must parse');
    // Real canvases hold laid-out positions, not the parser's all-zero ones
    const laidOut = positionProposedGraph({ nodes: [], edges: [] }, current, 'LR');
    const onCanvas = { nodes: laidOut.nodes, edges: current.edges };
    return { onCanvas, proposed };
  };

  const overlaps = (a: Node, b: Node) => {
    const sa = { w: getNodeSize(a).width, h: getNodeSize(a).height };
    const sb = { w: getNodeSize(b).width, h: getNodeSize(b).height };
    return (
      a.position.x < b.position.x + sb.w &&
      b.position.x < a.position.x + sa.w &&
      a.position.y < b.position.y + sb.h &&
      b.position.y < a.position.y + sa.h
    );
  };

  it.each(['LR', 'TB'] as const)('never overlaps a node or another ghost and is never NaN (%s)', (direction) => {
    const { onCanvas, proposed } = setup();
    const positioned = positionProposedGraph(onCanvas, proposed, direction);
    positioned.nodes.forEach((n, i) => {
      expect(Number.isFinite(n.position.x) && Number.isFinite(n.position.y)).toBe(true);
      positioned.nodes.slice(i + 1).forEach((other) => expect(overlaps(n, other)).toBe(false));
    });
  });

  it('moves a ghost that would land on an existing card, and survives a NaN position on the canvas', () => {
    const { onCanvas, proposed } = setup();
    // Every existing node sits where the new node would be laid out
    const crowded = { ...onCanvas, nodes: onCanvas.nodes.map((n) => ({ ...n, position: { x: 100, y: 100 } })) };
    const ghost = positionProposedGraph(crowded, proposed, 'LR').nodes.find((n) => n.id === 'fact_checker');
    expect(ghost && overlaps(ghost, crowded.nodes[0])).toBe(false);
    const broken = { ...onCanvas, nodes: onCanvas.nodes.map((n) => ({ ...n, position: { x: NaN, y: NaN } })) };
    const safe = positionProposedGraph(broken, proposed, 'LR').nodes.find((n) => n.id === 'fact_checker');
    expect(Number.isFinite(safe?.position.x) && Number.isFinite(safe?.position.y)).toBe(true);
  });
});

describe('preview display model', () => {
  const overlayAndPositions = () => {
    const overlay = buildPreviewOverlay(DEFAULT_WORKFLOW_YAML, proposedYaml());
    const current = parseGraph(DEFAULT_WORKFLOW_YAML);
    if (!overlay || !current) throw new Error('overlay expected');
    return { overlay, current, positioned: positionProposedGraph(current, overlay.proposed, 'LR') };
  };

  it('produces the ghost node and an edge to it, and focuses the ghost plus the node it hangs off', () => {
    const { overlay, positioned } = overlayAndPositions();
    expect(ghostNodeIds(overlay.diff)).toEqual(['fact_checker']);
    const ghostEdges = positioned.edges.filter((e) => overlay.diff.edgesAdded.has(e.id));
    expect(ghostEdges.some((e) => e.source === 'editor' && e.target === 'fact_checker')).toBe(true);
    const focus = previewFocusNodeIds(overlay.diff, positioned.edges);
    expect(focus).toEqual(expect.arrayContaining(['fact_checker', 'editor']));
  });

  it('adds ghosts after the live nodes, carrying the measured size handed back from React Flow', () => {
    const { overlay, current, positioned } = overlayAndPositions();
    const none = buildDisplayNodes(current.nodes, overlay, positioned, {}, null);
    expect(none.map((n) => n.id)).toEqual([...current.nodes.map((n) => n.id), 'fact_checker']);
    expect(none[none.length - 1].measured).toBeUndefined();
    const measured = { fact_checker: { width: 288, height: 200 } };
    const sized = buildDisplayNodes(current.nodes, overlay, positioned, measured, 'fact_checker');
    expect(sized[sized.length - 1].measured).toEqual({ width: 288, height: 200 });
    expect(sized[sized.length - 1].selected).toBe(true);
  });

  it('shows modified nodes with their proposed data without touching the live nodes', () => {
    const current = parseGraph(DEFAULT_WORKFLOW_YAML);
    const proposed = parseGraph(DEFAULT_WORKFLOW_YAML);
    if (!current || !proposed) throw new Error('YAML must parse');
    const target = current.nodes.find((n) => n.type === NODE_TYPES.AGENT);
    if (!target) throw new Error('agent expected');
    proposed.nodes = proposed.nodes.map((n) =>
      n.id === target.id ? { ...n, data: { ...n.data, job: 'PROPOSED JOB' } } : n
    );
    const overlay = { diff: diffGraphs(current, proposed), proposed };
    const live = structuredClone(current.nodes);
    const shown = buildDisplayNodes(current.nodes, overlay, positionProposedGraph(current, proposed, 'LR'), {}, null);
    expect(shown.find((n) => n.id === target.id)?.data.job).toBe('PROPOSED JOB');
    expect(current.nodes).toEqual(live);
    expect(current.nodes.find((n) => n.id === target.id)?.data.job).not.toBe('PROPOSED JOB');
  });

  it('returns the live nodes untouched when nothing is previewed', () => {
    const current = parseGraph(DEFAULT_WORKFLOW_YAML);
    if (!current) throw new Error('YAML must parse');
    expect(buildDisplayNodes(current.nodes, null, null, {}, null)).toBe(current.nodes);
  });
});
