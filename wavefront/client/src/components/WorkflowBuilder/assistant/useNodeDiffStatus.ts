import { createContext, useContext } from 'react';
import type { NodeDiffStatus } from '../graphDiff';

// Statuses of the previewed proposal by node id; null when nothing is previewed. Kept out of node data so the
// statuses can never reach the serialized YAML.
export const NodeDiffContext = createContext<ReadonlyMap<string, NodeDiffStatus> | null>(null);

// Safe without a provider (returns undefined), so node components can call it unconditionally
export const useNodeDiffStatus = (nodeId: string): NodeDiffStatus | undefined =>
  useContext(NodeDiffContext)?.get(nodeId);
