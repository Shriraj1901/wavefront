import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router';
import { streamWorkflowAssistantChat } from '@app/api/workflow-assistant-service';
import { useGetLLMConfigs } from '@app/hooks/data/fetch-hooks';
import { AssistantSurface } from './AssistantSurface';
import { buildChatBody, pickDefaultModelId, toAssistantModels, toModelNameMap } from './assistantRequest';
import { useEditorBridgeRegistry } from './editorBridge';
import { isAssistantEvent } from './events';
import type { AssistantMessage, WorkflowAssistant } from './types';
import { useAssistantCore } from './useAssistantCore';

const LAST_MODEL_KEY = 'workflow-assistant:last-model';

const readLastModel = (): string | null => {
  try {
    return localStorage.getItem(LAST_MODEL_KEY);
  } catch {
    return null;
  }
};

const writeLastModel = (id: string) => {
  try {
    localStorage.setItem(LAST_MODEL_KEY, id);
  } catch {
    // Per-viewer convenience only
  }
};

const isAbort = (error: unknown) => error instanceof DOMException && error.name === 'AbortError';

// Streams turns from the workflow assistant endpoint into the shared core (state, proposal lifecycle, preview).
export const RealAssistantProvider = ({ children }: { children: React.ReactNode }) => {
  const { getBridge } = useEditorBridgeRegistry();
  const { app: appId } = useParams<{ app: string }>();
  const { data: configs } = useGetLLMConfigs(appId || '');
  const modelNames = useMemo(() => toModelNameMap(configs), [configs]);
  const { state, shared, nodeStatuses, beginTurn, feed, markStopped, beginRetry, clear } = useAssistantCore({
    modelNames,
  });
  const models = useMemo(() => toAssistantModels(configs), [configs]);
  const [chosenModelId, setChosenModelId] = useState<string | null>(readLastModel);
  const selectedModelId = pickDefaultModelId(models, chosenModelId);

  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef(state.messages);
  messagesRef.current = state.messages;
  const statusRef = useRef(state.status);
  statusRef.current = state.status;

  useEffect(() => () => abortRef.current?.abort(), []);

  // Streams one turn; `prior` is the conversation before the user text being sent
  const run = useCallback(
    async (text: string, prior: AssistantMessage[]) => {
      if (!selectedModelId) return;
      const controller = new AbortController();
      abortRef.current = controller;
      let finished = false;
      try {
        await streamWorkflowAssistantChat(buildChatBody(text, getBridge()?.getYaml() ?? '', prior, selectedModelId), {
          signal: controller.signal,
          onFrame: (frame) => {
            if (controller.signal.aborted || !isAssistantEvent(frame)) return;
            if (frame.event_type === 'output' || frame.event_type === 'error') finished = true;
            feed(frame);
          },
        });
        if (!finished && !controller.signal.aborted) {
          feed({ event_type: 'error', message: 'The assistant stopped before finishing. Try again.' });
        }
      } catch (error) {
        if (isAbort(error) || controller.signal.aborted) return;
        feed({
          event_type: 'error',
          message: error instanceof Error && error.message ? error.message : 'The assistant could not be reached.',
        });
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [selectedModelId, getBridge, feed]
  );

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !selectedModelId || statusRef.current === 'streaming') return;
      const prior = messagesRef.current;
      beginTurn(trimmed);
      void run(trimmed, prior);
    },
    [selectedModelId, beginTurn, run]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    markStopped();
  }, [markStopped]);

  const retry = useCallback(() => {
    // Everything before the last user message; read before beginRetry drops the failed reply
    const current = messagesRef.current;
    let lastUser = -1;
    current.forEach((m, i) => {
      if (m.role === 'user') lastUser = i;
    });
    const text = beginRetry();
    if (text === undefined) return;
    void run(text, current.slice(0, Math.max(lastUser, 0)));
  }, [beginRetry, run]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    clear();
  }, [clear]);

  const selectModel = useCallback(
    (id: string) => {
      if (!models.some((m) => m.id === id)) return;
      setChosenModelId(id);
      writeLastModel(id);
    },
    [models]
  );

  const assistant = useMemo<WorkflowAssistant>(
    () => ({
      ...shared,
      available: models.length > 0,
      models,
      selectedModelId,
      selectModel,
      send,
      stop,
      retry,
      reset,
    }),
    [shared, models, selectedModelId, selectModel, send, stop, retry, reset]
  );

  return (
    <AssistantSurface assistant={assistant} nodeStatuses={nodeStatuses}>
      {children}
    </AssistantSurface>
  );
};
