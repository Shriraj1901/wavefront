import React from 'react';
import type { NodeDiffStatus } from '../graphDiff';
import { NodeDiffContext } from './useNodeDiffStatus';
import { AssistantContext } from './useWorkflowAssistant';
import type { WorkflowAssistant } from './types';

interface AssistantSurfaceProps {
  assistant: WorkflowAssistant;
  nodeStatuses?: ReadonlyMap<string, NodeDiffStatus> | null;
  children: React.ReactNode;
}

// Publishes an assistant implementation and its canvas overlay to the UI. Every provider ends in this component,
// so the UI sees the same contexts whichever provider is mounted.
export const AssistantSurface = ({ assistant, nodeStatuses = null, children }: AssistantSurfaceProps) => (
  <AssistantContext.Provider value={assistant}>
    <NodeDiffContext.Provider value={nodeStatuses}>{children}</NodeDiffContext.Provider>
  </AssistantContext.Provider>
);
