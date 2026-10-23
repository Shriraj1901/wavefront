import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
  EditorBridgeRegistryContext,
  EditorBridgeStateContext,
  type EditorBridge,
  type EditorBridgeRegistry,
} from './editorBridge';

export const EditorBridgeProvider = ({ children }: { children: React.ReactNode }) => {
  const bridgeRef = useRef<EditorBridge | null>(null);
  const [bridge, setBridge] = useState<EditorBridge | null>(null);

  const registerBridge = useCallback((next: EditorBridge | null) => {
    bridgeRef.current = next;
    setBridge(next);
  }, []);
  const getBridge = useCallback(() => bridgeRef.current, []);

  const registry = useMemo<EditorBridgeRegistry>(() => ({ registerBridge, getBridge }), [registerBridge, getBridge]);

  return (
    <EditorBridgeRegistryContext.Provider value={registry}>
      <EditorBridgeStateContext.Provider value={bridge}>{children}</EditorBridgeStateContext.Provider>
    </EditorBridgeRegistryContext.Provider>
  );
};
