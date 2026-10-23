import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import type { NodeDiffStatus } from '../graphDiff';
import { useEditorBridgeRegistry, useEditorBridgeState } from './editorBridge';
import { createDeltaBatcher, type DeltaBatcher } from './deltaBatcher';
import type { AssistantEvent } from './events';
import { buildPreviewOverlay, EMPTY_DIFF_SUMMARY, parseGraph, summarizeYamlDiff } from './proposalGraph';
import {
  assistantReducer,
  initialAssistantState,
  type AssistantReducerDeps,
  type AssistantState,
} from './assistantReducer';
import type { WorkflowAssistant } from './types';
import { isStaleAgainst, markStaleProposals, STALE_REASON } from './ui/proposalState';

// Everything a provider shares regardless of where its events come from: state, the proposal lifecycle against the
// editor bridge, and the preview overlay. A provider adds the transport (send/stop/retry/models) on top, which is
// the seam between the mock and the real streaming provider.
export type AssistantShared = Pick<
  WorkflowAssistant,
  | 'messages'
  | 'status'
  | 'error'
  | 'previewProposal'
  | 'applyProposal'
  | 'discardProposal'
  | 'canUndo'
  | 'undoLastApply'
  | 'focusNode'
>;

export interface AssistantCore {
  state: AssistantState;
  shared: AssistantShared;
  // Statuses of the previewed proposal for the node overlay; null when nothing is previewed
  nodeStatuses: ReadonlyMap<string, NodeDiffStatus> | null;
  // Transport helpers
  beginTurn: (text: string) => void;
  feed: (event: AssistantEvent) => void;
  markStopped: () => void;
  // Drops the failed reply and returns the user text to send again; undefined when there is nothing to retry
  beginRetry: () => string | undefined;
  clear: () => void;
}

const NOT_APPLICABLE_PROPOSAL = 'This proposal can no longer be applied.';

export interface AssistantCoreOptions {
  // Rootflo model config id -> display name, for readable before -> after model changes. Optional: ids show as is.
  modelNames?: Readonly<Record<string, string>>;
}

export const useAssistantCore = ({ modelNames }: AssistantCoreOptions = {}): AssistantCore => {
  const { getBridge } = useEditorBridgeRegistry();
  const bridge = useEditorBridgeState();
  const [state, dispatch] = useReducer(assistantReducer, initialAssistantState);
  // At most one previewed proposal at a time
  const [preview, setPreview] = useState<{ id: string; statuses: ReadonlyMap<string, NodeDiffStatus> } | null>(null);

  // Callbacks read the latest state without being recreated on every event
  const modelNamesRef = useRef(modelNames);
  modelNamesRef.current = modelNames;
  const stateRef = useRef<AssistantState>(state);
  stateRef.current = state;
  // Pending proposals are flagged stale when the editor's YAML is no longer the one they were written against. The
  // bridge changes on every editor commit, so this re-evaluates as the user edits. Reads no new data: only the bridge
  // the provider already holds.
  const currentYaml = bridge?.getYaml();
  const messages = useMemo(
    () => markStaleProposals(state.messages, state.proposalBase, currentYaml),
    [state.messages, state.proposalBase, currentYaml]
  );
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const previewIdRef = useRef<string | null>(null);
  previewIdRef.current = preview?.id ?? null;
  // The proposal the editor's one-level undo currently belongs to
  const appliedIdRef = useRef<string | null>(null);

  const deps = useMemo<AssistantReducerDeps>(
    () => ({
      computeDiff: (proposedYaml) => {
        const current = getBridge();
        return current
          ? summarizeYamlDiff(current.getYaml(), proposedYaml, { modelNames: modelNamesRef.current })
          : EMPTY_DIFF_SUMMARY;
      },
      // Unparseable YAML is the one thing that blocks Apply and Preview
      canApply: (proposedYaml) => parseGraph(proposedYaml) !== null,
      now: () => Date.now(),
    }),
    [getBridge]
  );

  // content_delta text is queued and dispatched once per animation frame, so a long answer does not re-render per
  // chunk. Every other event flushes the queue first, which keeps the order the server sent.
  const depsRef = useRef(deps);
  depsRef.current = deps;
  const batcherRef = useRef<DeltaBatcher | null>(null);
  if (batcherRef.current === null) {
    batcherRef.current = createDeltaBatcher((content) =>
      dispatch({ type: 'event', event: { event_type: 'content_delta', content }, deps: depsRef.current })
    );
  }
  useEffect(() => () => batcherRef.current?.discard(), []);

  const findProposal = useCallback((id: string) => {
    const message = messagesRef.current.find((m) => m.proposal?.id === id);
    return message?.proposal;
  }, []);

  const clearPreview = useCallback(() => {
    setPreview(null);
    getBridge()?.setPreviewDiff(null);
  }, [getBridge]);

  // The overlay only makes sense for a pending proposal that still matches the workflow: drop it when that proposal is
  // gone (discard, reset, retry) or went stale
  useEffect(() => {
    const id = previewIdRef.current;
    if (!id) return;
    const proposal = findProposal(id);
    if (proposal?.state !== 'pending' || proposal.stale) clearPreview();
  }, [messages, findProposal, clearPreview]);

  const previewProposal = useCallback(
    (id: string | null) => {
      if (id === null) {
        if (previewIdRef.current) clearPreview();
        return;
      }
      const proposal = findProposal(id);
      const proposedYaml = stateRef.current.proposalYaml[id];
      const current = getBridge();
      if (!proposal || proposal.state !== 'pending' || proposedYaml === undefined || !current) return;
      // Checked against the editor right now, not the last render: the overlay would diff against edits the proposal
      // knows nothing about
      if (isStaleAgainst(stateRef.current.proposalBase[id], current.getYaml())) return;
      const overlay = buildPreviewOverlay(current.getYaml(), proposedYaml);
      // An undrawable proposal (typically an invalid one) has nothing to show on the canvas
      if (!overlay) return;
      setPreview({ id, statuses: overlay.diff.nodes });
      current.setPreviewDiff(overlay);
    },
    [findProposal, getBridge, clearPreview]
  );

  const applyProposal = useCallback(
    (id: string) => {
      const proposal = findProposal(id);
      const proposedYaml = stateRef.current.proposalYaml[id];
      if (!proposal || proposal.state !== 'pending' || proposedYaml === undefined) return;
      // Validation errors do not block Apply (the workflow just will not save until fixed); an unreadable YAML does
      if (proposal.applicable === false) {
        dispatch({ type: 'set_error', message: 'The proposed workflow could not be read, so it cannot be applied.' });
        return;
      }
      const current = getBridge();
      if (!current) {
        dispatch({ type: 'set_error', message: 'The workflow editor is not ready.' });
        return;
      }
      // Never overwrite edits made since the proposal's turn started (checked now, not at the last render)
      if (isStaleAgainst(stateRef.current.proposalBase[id], current.getYaml())) {
        dispatch({ type: 'set_error', message: STALE_REASON });
        return;
      }
      const result = current.applyYaml(proposedYaml);
      if (!result.ok) {
        dispatch({ type: 'set_error', message: result.error || NOT_APPLICABLE_PROPOSAL });
        return;
      }
      appliedIdRef.current = id;
      clearPreview();
      dispatch({ type: 'proposal_state', id, state: 'applied' });
    },
    [findProposal, getBridge, clearPreview]
  );

  const discardProposal = useCallback(
    (id: string) => {
      const proposal = findProposal(id);
      if (!proposal || proposal.state !== 'pending') return;
      if (previewIdRef.current === id) clearPreview();
      dispatch({ type: 'proposal_state', id, state: 'discarded' });
    },
    [findProposal, clearPreview]
  );

  const undoLastApply = useCallback(() => {
    const current = getBridge();
    if (!current?.canUndo || !current.undoLastApply()) return;
    const id = appliedIdRef.current;
    appliedIdRef.current = null;
    if (id) dispatch({ type: 'proposal_state', id, state: 'pending' });
  }, [getBridge]);

  const focusNode = useCallback(
    (nodeId: string) => {
      getBridge()?.focusNode(nodeId);
    },
    [getBridge]
  );

  const beginTurn = useCallback(
    (text: string) => {
      batcherRef.current?.discard();
      // The YAML the request is built from (same instant): what this turn's proposal is a change to
      dispatch({ type: 'user_sent', text, now: Date.now(), baseYaml: getBridge()?.getYaml() });
    },
    [getBridge]
  );
  const feed = useCallback(
    (event: AssistantEvent) => {
      if (event.event_type === 'content_delta') {
        batcherRef.current?.push(event.content);
        return;
      }
      // output, error, retract and the rest must see all the text sent before them
      batcherRef.current?.flush();
      dispatch({ type: 'event', event, deps });
    },
    [deps]
  );
  const markStopped = useCallback(() => {
    batcherRef.current?.flush();
    dispatch({ type: 'stopped' });
  }, []);

  const beginRetry = useCallback((): string | undefined => {
    const current = stateRef.current;
    if (current.status === 'streaming') return undefined;
    const lastUser = [...current.messages].reverse().find((m) => m.role === 'user');
    if (!lastUser) return undefined;
    batcherRef.current?.discard();
    dispatch({ type: 'retry_started', baseYaml: getBridge()?.getYaml() });
    return lastUser.text;
  }, [getBridge]);

  const clear = useCallback(() => {
    appliedIdRef.current = null;
    clearPreview();
    batcherRef.current?.discard();
    dispatch({ type: 'reset' });
  }, [clearPreview]);

  const shared = useMemo<AssistantShared>(
    () => ({
      messages,
      status: state.status,
      error: state.error,
      previewProposal,
      applyProposal,
      discardProposal,
      canUndo: bridge?.canUndo ?? false,
      undoLastApply,
      focusNode,
    }),
    [
      messages,
      state.status,
      state.error,
      previewProposal,
      applyProposal,
      discardProposal,
      bridge?.canUndo,
      undoLastApply,
      focusNode,
    ]
  );

  return {
    state,
    shared,
    nodeStatuses: preview?.statuses ?? null,
    beginTurn,
    feed,
    markStopped,
    beginRetry,
    clear,
  };
};
