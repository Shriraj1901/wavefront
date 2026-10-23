import type { Edge, Node } from '@xyflow/react';
import type { DiffSummary, FieldChange } from './assistant/types';
import type { Graph } from './graphUtils';

export type NodeDiffStatus = 'added' | 'modified' | 'removed';

// What a proposed graph changes compared with the current one. Edge ids are deterministic (see makeEdgeId),
// so nodes and edges are both matched by id. Edges that exist on both sides count as unchanged.
export interface GraphDiff {
  nodes: Map<string, NodeDiffStatus>;
  edgesAdded: Set<string>;
  edgesRemoved: Set<string>;
  // For each modified node, what changed (see changedNodeFields)
  changedFields: Map<string, string[]>;
  summary: DiffSummary;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Canonical form of a node data value, so a re-emitted YAML that means the same thing compares equal: null, undefined,
// '' (and a whitespace-only string) are all "unset"; strings ignore line endings, trailing whitespace on each line and
// leading/trailing blank space; an object holding nothing set equals a missing one; key order never matters. Arrays
// keep their order and their length (an empty list is not the same as a missing one: an omitted input_filter reads
// everything). Anything else is compared as is, so a real change is never hidden.
const normalizeValue = (value: unknown): unknown => {
  if (value === null || value === undefined) return undefined;
  if (typeof value === 'string') {
    const text = value
      .replace(/\r\n?/g, '\n')
      .split('\n')
      .map((line) => line.replace(/[ \t]+$/, ''))
      .join('\n')
      .trim();
    return text === '' ? undefined : text;
  }
  if (Array.isArray(value)) return value.map(normalizeValue);
  if (isRecord(value)) {
    const entries = Object.entries(value)
      .map(([key, item]) => [key, normalizeValue(item)] as const)
      .filter(([, item]) => item !== undefined);
    return entries.length === 0 ? undefined : Object.fromEntries(entries);
  }
  return value;
};

// Structural equality for normalized values (plain JSON-like data); key order is ignored
const isDeepEqual = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b)) return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((item, i) => isDeepEqual(item, b[i]))
    );
  }
  if (!isRecord(a) || !isRecord(b)) return false;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) {
    if (!isDeepEqual(a[key], b[key])) return false;
  }
  return true;
};

// Short names for the card. Several data keys can describe one thing (provider, model and model_id are all "model").
const FIELD_LABELS: Record<string, string> = {
  job: 'prompt',
  role: 'role',
  provider: 'model',
  model: 'model',
  model_id: 'model',
  temperature: 'settings',
  max_tokens: 'settings',
  tools: 'tools',
  input_filter: 'input filter',
  isEndNode: 'end node',
  function_name: 'function',
  prefilled_params: 'parameters',
  configMethod: 'config method',
  execute_node: 'iterated node',
  forward_all_results: 'forward results',
  inherit_variables: 'variables',
};

// Which parts of a node differ, as short deduplicated labels in first-seen order. Empty when it did not change.
// `label` follows the node id, which both sides share, so it is never a change.
export const changedNodeFields = (current: Node, proposed: Node): string[] => {
  const labels: string[] = [];
  const add = (label: string) => {
    if (!labels.includes(label)) labels.push(label);
  };
  if (current.type !== proposed.type) add('type');
  const keys = new Set([...Object.keys(current.data ?? {}), ...Object.keys(proposed.data ?? {})]);
  keys.forEach((key) => {
    if (key === 'label') return;
    const before = normalizeValue((current.data as Record<string, unknown>)?.[key]);
    const after = normalizeValue((proposed.data as Record<string, unknown>)?.[key]);
    if (!isDeepEqual(before, after)) add(FIELD_LABELS[key] ?? key);
  });
  return labels;
};

// ---------------------------------------------------------------------------
// Before -> after detail for the review card
// ---------------------------------------------------------------------------

const MAX_VALUE_LENGTH = 60;
const EMPTY_VALUE = 'none';

export interface DiffOptions {
  // Rootflo model config id -> display name, so `model_id: <uuid>` reads as the name the user knows
  modelNames?: Readonly<Record<string, string>>;
}

const shorten = (text: string): string => {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > MAX_VALUE_LENGTH ? `${flat.slice(0, MAX_VALUE_LENGTH - 3).trimEnd()}...` : flat;
};

const CONTEXT_BEFORE_DIFF = 20;

// Shortens a before/after pair so the part that differs stays visible: long text that only differs near its end
// would otherwise read the same on both sides once cut to the first 60 characters.
export const excerptPair = (before: string, after: string): [string, string] => {
  const a = before.replace(/\s+/g, ' ').trim();
  const b = after.replace(/\s+/g, ' ').trim();
  if ((a.length <= MAX_VALUE_LENGTH && b.length <= MAX_VALUE_LENGTH) || a === b) return [shorten(a), shorten(b)];
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  const start = Math.max(0, i - CONTEXT_BEFORE_DIFF);
  const cut = (value: string) => {
    const body = value.slice(start, start + MAX_VALUE_LENGTH - 6).trimEnd();
    return `${start > 0 ? '...' : ''}${body}${start + MAX_VALUE_LENGTH - 6 < value.length ? '...' : ''}`;
  };
  return [cut(a), cut(b)];
};

const dataOf = (node: Node): Record<string, unknown> => (node.data ?? {}) as Record<string, unknown>;

const asText = (value: unknown): string => {
  const normalized = normalizeValue(value);
  return typeof normalized === 'string' ? normalized : '';
};

// 'gpt-4o-mini (openai)' for an inline model block; the config's display name (or id) for a rootflo model
export const describeModel = (data: Record<string, unknown>, options: DiffOptions = {}): string => {
  const name = asText(data.model);
  const provider = asText(data.provider);
  const modelId = asText(data.model_id);
  if (name) return provider ? `${name} (${provider})` : name;
  if (modelId) return options.modelNames?.[modelId] ?? modelId;
  return provider ? `${provider} (no model)` : EMPTY_VALUE;
};

const describeSettings = (data: Record<string, unknown>): string => {
  const parts = ['temperature', 'max_tokens']
    .filter((key) => normalizeValue(data[key]) !== undefined)
    .map((key) => `${key} ${String(data[key])}`);
  return parts.length ? parts.join(', ') : 'defaults';
};

const describeGeneric = (value: unknown): string => {
  const normalized = normalizeValue(value);
  if (normalized === undefined) return EMPTY_VALUE;
  if (typeof normalized === 'string') return normalized;
  if (Array.isArray(normalized)) {
    return normalized.length === 0 ? 'empty list' : normalized.map((v) => describeGeneric(v)).join(', ');
  }
  if (typeof normalized === 'object') return JSON.stringify(normalized);
  return String(normalized);
};

// How a node's value for a label reads (several data keys can make up one label)
const describeLabel = (label: string, node: Node, key: string, options: DiffOptions): string => {
  const data = dataOf(node);
  if (label === 'model') return describeModel(data, options);
  if (label === 'settings') return describeSettings(data);
  return describeGeneric(data[key]);
};

// Fixed order so the card does not reshuffle: the labels the card knows first, in the order FIELD_LABELS lists them
const LABEL_ORDER = [...new Set(Object.values(FIELD_LABELS)), 'type'];
const labelRank = (label: string): number => {
  const index = LABEL_ORDER.indexOf(label);
  return index === -1 ? LABEL_ORDER.length : index;
};

// One entry per changed label of a modified node, with short readable before and after values
export const describeNodeChanges = (
  current: Node,
  proposed: Node,
  options: DiffOptions = {}
): Omit<FieldChange, 'nodeId'>[] => {
  const before = dataOf(current);
  const after = dataOf(proposed);
  const byLabel = new Map<string, string>();
  const note = (label: string, key: string) => {
    if (!byLabel.has(label)) byLabel.set(label, key);
  };
  if (current.type !== proposed.type) note('type', 'type');
  new Set([...Object.keys(before), ...Object.keys(after)]).forEach((key) => {
    if (key === 'label') return;
    if (isDeepEqual(normalizeValue(before[key]), normalizeValue(after[key]))) return;
    note(FIELD_LABELS[key] ?? key, key);
  });
  return [...byLabel.entries()]
    .sort(([a], [b]) => labelRank(a) - labelRank(b) || a.localeCompare(b))
    .map(([field, key]) => {
      if (field === 'type') {
        return { field, before: shorten(String(current.type ?? EMPTY_VALUE)), after: shorten(String(proposed.type)) };
      }
      const [shownBefore, shownAfter] = excerptPair(
        describeLabel(field, current, key, options),
        describeLabel(field, proposed, key, options)
      );
      return { field, before: shownBefore, after: shownAfter };
    });
};

export const diffGraphs = (current: Graph, proposed: Graph, options: DiffOptions = {}): GraphDiff => {
  const statuses = new Map<string, NodeDiffStatus>();
  const currentById = new Map(current.nodes.map((n) => [n.id, n]));
  const proposedIds = new Set(proposed.nodes.map((n) => n.id));

  const nodesAdded: string[] = [];
  const nodesModified: string[] = [];
  const nodesRemoved: string[] = [];
  const changedFields = new Map<string, string[]>();
  const changes: FieldChange[] = [];

  proposed.nodes.forEach((node) => {
    const existing = currentById.get(node.id);
    if (!existing) {
      statuses.set(node.id, 'added');
      nodesAdded.push(node.id);
      return;
    }
    const fields = changedNodeFields(existing, node);
    if (fields.length > 0) {
      statuses.set(node.id, 'modified');
      nodesModified.push(node.id);
      changedFields.set(node.id, fields);
      describeNodeChanges(existing, node, options).forEach((c) => changes.push({ nodeId: node.id, ...c }));
    }
  });
  current.nodes.forEach((node) => {
    if (!proposedIds.has(node.id)) {
      statuses.set(node.id, 'removed');
      nodesRemoved.push(node.id);
    }
  });

  const ids = (edges: Edge[]) => new Set(edges.map((e) => e.id));
  const currentEdgeIds = ids(current.edges);
  const proposedEdgeIds = ids(proposed.edges);
  const edgesAdded = new Set([...proposedEdgeIds].filter((id) => !currentEdgeIds.has(id)));
  const edgesRemoved = new Set([...currentEdgeIds].filter((id) => !proposedEdgeIds.has(id)));

  return {
    nodes: statuses,
    edgesAdded,
    edgesRemoved,
    changedFields,
    summary: {
      nodesAdded,
      nodesModified,
      nodesRemoved,
      edgesAdded: edgesAdded.size,
      edgesRemoved: edgesRemoved.size,
      // Only present when something was modified, so a summary with no modifications is unchanged from before
      ...(changedFields.size > 0 ? { changedFields: Object.fromEntries(changedFields) } : {}),
      ...(changes.length > 0 ? { changes } : {}),
    },
  };
};
