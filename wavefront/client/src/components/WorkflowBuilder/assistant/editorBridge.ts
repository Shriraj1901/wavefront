import { createContext, useContext, useEffect } from 'react';
import type { GraphIssue } from '@app/types/workflow-builder';
import type { PreviewOverlay } from './proposalGraph';

export type ApplyYamlResult = { ok: true; yaml: string } | { ok: false; error: string };

// The narrow surface the editor offers the assistant. The editor registers it through context (not forwardRef: it
// sits behind React.lazy and is always mounted).
export interface EditorBridge {
  // The YAML the canvas currently corresponds to
  getYaml(): string;
  // Parse, lay out and commit the YAML once. Snapshots the previous YAML first, for undoLastApply.
  applyYaml(yaml: string): ApplyYamlResult;
  // Restores the YAML from before the last apply. false when there is nothing to undo.
  undoLastApply(): boolean;
  // true only while the editor still holds exactly what the last apply produced
  canUndo: boolean;
  issues: GraphIssue[];
  // Shows (or clears, with null) a proposal over the canvas. Never written into the live nodes.
  setPreviewDiff(overlay: PreviewOverlay | null): void;
  focusNode(nodeId: string): void;
}

export interface EditorBridgeRegistry {
  registerBridge: (bridge: EditorBridge | null) => void;
  // Always the latest registered bridge, readable from callbacks without a re-render
  getBridge: () => EditorBridge | null;
}

const noopRegistry: EditorBridgeRegistry = { registerBridge: () => {}, getBridge: () => null };

// Stable for the provider's lifetime, so the editor (which only registers) never re-renders when the bridge changes
export const EditorBridgeRegistryContext = createContext<EditorBridgeRegistry>(noopRegistry);

// Changes whenever the bridge does (canUndo, issues), for the assistant side
export const EditorBridgeStateContext = createContext<EditorBridge | null>(null);

export const useEditorBridgeRegistry = () => useContext(EditorBridgeRegistryContext);

// The currently registered bridge; re-renders the caller when it changes. null while no editor is mounted.
export const useEditorBridgeState = () => useContext(EditorBridgeStateContext);

// For the editor: keep the bridge registered while it is mounted. Pass a memoized object. A no-op without a provider.
export const useRegisterEditorBridge = (bridge: EditorBridge) => {
  const { registerBridge } = useEditorBridgeRegistry();
  useEffect(() => {
    registerBridge(bridge);
  }, [registerBridge, bridge]);
  useEffect(() => () => registerBridge(null), [registerBridge]);
};
