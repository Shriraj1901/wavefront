import type { Edge, Node, XYPosition } from '@xyflow/react';
import { getNodeSize } from '../autoLayout';
import { diffGraphs, type DiffOptions, type GraphDiff } from '../graphDiff';
import { layoutGraph, normalizeGraph, type Graph } from '../graphUtils';
import type { LayoutDirection } from '../layoutDirection';
import { parseYamlToGraph } from '../yamlSync';
import type { DiffSummary } from './types';

// What the canvas needs to show a previewed proposal without touching the live nodes: the statuses, and the proposed
// graph so the nodes that do not exist yet can be drawn as ghosts.
export interface PreviewOverlay {
  diff: GraphDiff;
  proposed: Graph;
}

export const EMPTY_DIFF_SUMMARY: DiffSummary = {
  nodesAdded: [],
  nodesModified: [],
  nodesRemoved: [],
  edgesAdded: 0,
  edgesRemoved: 0,
};

// The graph the editor would hold for this YAML, or null when it cannot be drawn
export const parseGraph = (yamlText: string): Graph | null => {
  const parsed = parseYamlToGraph(yamlText);
  return parsed.error ? null : normalizeGraph(parsed);
};

// Both sides go through the same parser, so the diff only reports what the YAML actually changes
export const buildPreviewOverlay = (
  currentYaml: string,
  proposedYaml: string,
  options: DiffOptions = {}
): PreviewOverlay | null => {
  const current = parseGraph(currentYaml);
  const proposed = parseGraph(proposedYaml);
  if (!current || !proposed) return null;
  return { diff: diffGraphs(current, proposed, options), proposed };
};

// Diff summary for the Proposal contract. An undrawable side (invalid proposal) gives an empty summary.
export const summarizeYamlDiff = (currentYaml: string, proposedYaml: string, options: DiffOptions = {}): DiffSummary =>
  buildPreviewOverlay(currentYaml, proposedYaml, options)?.diff.summary ?? EMPTY_DIFF_SUMMARY;

const GHOST_GAP = 40;
const MAX_NUDGES = 60;

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const isFinitePosition = (p: XYPosition | undefined): p is XYPosition =>
  !!p && Number.isFinite(p.x) && Number.isFinite(p.y);

const rectOf = (node: Node, position: XYPosition): Rect => ({ ...position, ...getNodeSize(node) });

const rectsOverlap = (a: Rect, b: Rect): boolean =>
  a.x < b.x + b.width + GHOST_GAP &&
  b.x < a.x + a.width + GHOST_GAP &&
  a.y < b.y + b.height + GHOST_GAP &&
  b.y < a.y + a.height + GHOST_GAP;

// A spot for a new node that is finite and clear of every occupied rectangle: a bad position falls back to just
// below everything, a colliding one slides along the cross axis (down for LR, right for TB) past what it hits.
const findFreeSpot = (node: Node, wanted: XYPosition, occupied: Rect[], direction: LayoutDirection): XYPosition => {
  let position = wanted;
  if (!isFinitePosition(position)) {
    const bottom = occupied.reduce((max, r) => Math.max(max, r.y + r.height), 0);
    const left = occupied.length ? Math.min(...occupied.map((r) => r.x)) : 0;
    position = { x: left, y: occupied.length ? bottom + GHOST_GAP : 0 };
  }
  for (let i = 0; i < MAX_NUDGES; i++) {
    const rect = rectOf(node, position);
    const hit = occupied.find((r) => rectsOverlap(rect, r));
    if (!hit) break;
    position =
      direction === 'TB'
        ? { x: hit.x + hit.width + GHOST_GAP, y: position.y }
        : { x: position.x, y: hit.y + hit.height + GHOST_GAP };
  }
  return position;
};

// Positions for a proposed graph (YAML carries none): nodes already on the canvas keep their place, new ones take
// their spot from a full layout shifted so the shared nodes line up, then are moved clear of every other node so a
// ghost never hides behind (or on top of) a card. With nothing shared it is the plain layout.
// Used for both the ghost preview and the apply, so a node lands where its ghost was shown.
export const positionProposedGraph = (current: Graph, proposed: Graph, direction: LayoutDirection): Graph => {
  const laidOut = layoutGraph(proposed, direction);
  const currentById = new Map(current.nodes.map((n) => [n.id, n]));
  const shared = laidOut.nodes.filter((n) => currentById.has(n.id));

  // Mean shift between where the layout put the shared nodes and where they actually are
  let dx = 0;
  let dy = 0;
  shared.forEach((n) => {
    const existing = currentById.get(n.id);
    if (!existing) return;
    dx += existing.position.x - n.position.x;
    dy += existing.position.y - n.position.y;
  });
  if (shared.length > 0) {
    dx /= shared.length;
    dy /= shared.length;
  }
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    dx = 0;
    dy = 0;
  }

  // Rectangles already taken: every node that stays on the canvas, and the new ones as they are placed
  const occupied: Rect[] = current.nodes.filter((n) => isFinitePosition(n.position)).map((n) => rectOf(n, n.position));

  return {
    nodes: laidOut.nodes.map((n) => {
      const existing = currentById.get(n.id);
      if (existing) return { ...n, position: existing.position };
      const position = findFreeSpot(n, { x: n.position.x + dx, y: n.position.y + dy }, occupied, direction);
      occupied.push(rectOf(n, position));
      return { ...n, position };
    }),
    edges: laidOut.edges,
  };
};

export type MeasuredSizes = Record<string, { width: number; height: number }>;

// Ids of the nodes that do not exist yet (the ghosts)
export const ghostNodeIds = (diff: GraphDiff): string[] =>
  [...diff.nodes.entries()].filter(([, status]) => status === 'added').map(([id]) => id);

// The steps to bring into view when a preview starts: every ghost plus the nodes the new edges connect it to
export const previewFocusNodeIds = (diff: GraphDiff, proposedEdges: Edge[]): string[] => {
  const ids = new Set(ghostNodeIds(diff));
  proposedEdges.forEach((e) => {
    if (!diff.edgesAdded.has(e.id)) return;
    ids.add(e.source);
    ids.add(e.target);
  });
  return [...ids];
};

// What the canvas renders during a preview: the live nodes, with modified ones showing their PROPOSED data, plus a
// ghost for every added step. Built fresh from the live nodes, never written back, so nothing here can reach the YAML.
// `measured` carries the ghosts' sizes: React Flow reports them through onNodesChange, and a node that is not in the
// editor's own node list would otherwise lose them on every render and stay hidden.
export const buildDisplayNodes = (
  nodes: Node[],
  overlay: PreviewOverlay | null,
  positioned: Graph | null,
  measured: MeasuredSizes,
  selectedId: string | null
): Node[] => {
  if (!overlay || !positioned) return nodes;
  const proposedById = new Map(overlay.proposed.nodes.map((n) => [n.id, n]));
  const live = nodes.map((n) => {
    if (overlay.diff.nodes.get(n.id) !== 'modified') return n;
    const proposed = proposedById.get(n.id);
    return proposed ? { ...n, type: proposed.type, data: proposed.data } : n;
  });
  const ghosts = positioned.nodes
    .filter((n) => overlay.diff.nodes.get(n.id) === 'added')
    .map((n) => {
      const size = measured[n.id];
      return {
        ...n,
        draggable: false,
        connectable: false,
        deletable: false,
        selected: n.id === selectedId,
        ...(size ? { measured: size } : {}),
      };
    });
  return ghosts.length ? [...live, ...ghosts] : live;
};
