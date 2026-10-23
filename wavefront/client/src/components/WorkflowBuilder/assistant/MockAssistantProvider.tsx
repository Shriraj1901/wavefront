import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEditorBridgeRegistry } from './editorBridge';
import { buildScript, type MockScenario } from './mockScenarios';
import { AssistantSurface } from './AssistantSurface';
import type { WorkflowAssistant } from './types';
import { useAssistantCore } from './useAssistantCore';

const MOCK_MODELS = [
  { id: 'mock-model', name: 'Mock model' },
  { id: 'mock-model-large', name: 'Mock model (large)' },
];

interface MockAssistantProviderProps {
  scenario: MockScenario;
  children: React.ReactNode;
}

// Implements WorkflowAssistant with a scripted turn per scenario and no backend, so every UI state can be built and
// reviewed offline. Proposals are derived from the live editor YAML, so preview, apply and undo run for real.
export const MockAssistantProvider = ({ scenario, children }: MockAssistantProviderProps) => {
  const { state, shared, nodeStatuses, beginTurn, feed, markStopped, beginRetry, clear } = useAssistantCore();
  const { getBridge } = useEditorBridgeRegistry();
  const [selectedModelId, setSelectedModelId] = useState<string | undefined>(MOCK_MODELS[0].id);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const proposalCounter = useRef(0);
  const available = scenario !== 'unavailable';

  const cancelScript = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  useEffect(() => cancelScript, [cancelScript]);

  const runScript = useCallback(() => {
    cancelScript();
    proposalCounter.current += 1;
    const script = buildScript(scenario, {
      currentYaml: getBridge()?.getYaml() ?? '',
      proposalId: `mock-proposal-${proposalCounter.current}`,
      now: Date.now(),
    });
    let elapsed = 0;
    script.forEach((step) => {
      elapsed += step.delayMs;
      timers.current.push(setTimeout(() => feed({ ...step.event, timestamp: Date.now() }), elapsed));
    });
  }, [scenario, getBridge, cancelScript, feed]);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!available || !trimmed || state.status === 'streaming') return;
      beginTurn(trimmed);
      runScript();
    },
    [available, state.status, beginTurn, runScript]
  );

  const stop = useCallback(() => {
    cancelScript();
    markStopped();
  }, [cancelScript, markStopped]);

  const retry = useCallback(() => {
    if (!available) return;
    const text = beginRetry();
    if (text !== undefined) runScript();
  }, [available, beginRetry, runScript]);

  const reset = useCallback(() => {
    cancelScript();
    clear();
  }, [cancelScript, clear]);

  const selectModel = useCallback((id: string) => {
    if (MOCK_MODELS.some((m) => m.id === id)) setSelectedModelId(id);
  }, []);

  const assistant = useMemo<WorkflowAssistant>(
    () => ({
      ...shared,
      available,
      models: available ? MOCK_MODELS : [],
      selectedModelId: available ? selectedModelId : undefined,
      selectModel,
      send,
      stop,
      retry,
      reset,
    }),
    [shared, available, selectedModelId, selectModel, send, stop, retry, reset]
  );

  return (
    <AssistantSurface assistant={assistant} nodeStatuses={nodeStatuses}>
      {children}
    </AssistantSurface>
  );
};
