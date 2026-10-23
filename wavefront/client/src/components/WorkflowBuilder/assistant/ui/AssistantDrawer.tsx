import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useWorkflowAssistant } from '../useWorkflowAssistant';
import { AssistantHeader } from './AssistantHeader';
import { useAssistantDock } from './AssistantDockContext';
import { clampWidth, DOCK_MAX_WIDTH, DOCK_MIN_WIDTH, DOCK_OVERLAY_BELOW } from './dockState';
import { Composer } from './Composer';
import type { ProposalActions } from './MessageBubble';
import { MessageList } from './MessageList';
import { UndoBanner } from './UndoBanner';

const useIsOverlay = () => {
  const query = `(max-width: ${DOCK_OVERLAY_BELOW - 1}px)`;
  const [overlay, setOverlay] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setOverlay(mql.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);
  return overlay;
};

const NotConfigured = () => (
  <div className="flex flex-1 flex-col items-start gap-1 px-4 py-6">
    <h3 className="text-sm font-semibold">No usable model</h3>
    <p className="text-muted-foreground text-xs leading-relaxed">
      The assistant needs an LLM configuration for this app. Add one under LLM configs, then reopen this panel.
    </p>
  </div>
);

// Docked at the right edge, outside the visual editor (so beside the property inspector, not inside it). At or above
// DOCK_OVERLAY_BELOW it takes space from the canvas; below it, it overlays the canvas edge instead.
export const AssistantDrawer = () => {
  const assistant = useWorkflowAssistant();
  const { open, width, setOpen, setWidth } = useAssistantDock();
  const overlay = useIsOverlay();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const { previewProposal, applyProposal, discardProposal, focusNode } = assistant;

  // Focus the composer on open and give focus back on close
  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (assistant.available) inputRef.current?.focus();
    return () => {
      returnFocusRef.current?.focus?.();
    };
    // Only on open/close transitions
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Closing the drawer (or unmounting it) always clears the canvas preview
  useEffect(() => {
    if (open) return;
    setPreviewId(null);
    previewProposal(null);
  }, [open, previewProposal]);
  useEffect(() => () => previewProposal(null), [previewProposal]);

  const togglePreview = useCallback(
    (id: string) => {
      if (previewId === id) {
        setPreviewId(null);
        previewProposal(null);
      } else {
        setPreviewId(id);
        previewProposal(id);
      }
    },
    [previewId, previewProposal]
  );
  const apply = useCallback(
    (id: string) => {
      setPreviewId(null);
      applyProposal(id);
    },
    [applyProposal]
  );
  const discard = useCallback(
    (id: string) => {
      setPreviewId(null);
      discardProposal(id);
    },
    [discardProposal]
  );

  const { send } = assistant;
  const actions = useMemo<ProposalActions>(
    () => ({
      previewId,
      onTogglePreview: togglePreview,
      onApply: apply,
      onDiscard: discard,
      onFocusNode: focusNode,
      onSend: send,
    }),
    [previewId, togglePreview, apply, discard, focusNode, send]
  );

  const startResize = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = width;
    const move = (ev: PointerEvent) => setWidth(startWidth + (startX - ev.clientX));
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const onResizeKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setWidth(width + 24);
    else if (e.key === 'ArrowRight') setWidth(width - 24);
    else return;
    e.preventDefault();
  };

  if (!open) return null;

  return (
    <aside
      aria-label="Workflow assistant"
      onKeyDown={(e) => {
        if (e.key === 'Escape' && overlay && !e.defaultPrevented) setOpen(false);
      }}
      style={{ width: clampWidth(width, window.innerWidth) }}
      className={`border-border bg-card animate-in slide-in-from-right-4 fade-in-0 z-30 flex h-full max-w-full shrink-0 flex-col border-l duration-200 motion-reduce:animate-none ${
        overlay ? 'absolute top-0 right-0 shadow-xl' : 'relative'
      }`}
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize assistant"
        aria-valuemin={DOCK_MIN_WIDTH}
        aria-valuemax={DOCK_MAX_WIDTH}
        aria-valuenow={clampWidth(width, window.innerWidth)}
        tabIndex={0}
        onPointerDown={startResize}
        onKeyDown={onResizeKey}
        className="hover:bg-primary/30 focus-visible:bg-primary/40 absolute inset-y-0 -left-1 z-10 w-2 cursor-col-resize outline-none"
      />
      <AssistantHeader
        models={assistant.models}
        selectedModelId={assistant.selectedModelId}
        onSelectModel={assistant.selectModel}
        hasMessages={assistant.messages.length > 0}
        onNewChat={assistant.reset}
        onClose={() => setOpen(false)}
      />
      {assistant.canUndo && <UndoBanner onUndo={assistant.undoLastApply} />}
      {assistant.available ? (
        <>
          <MessageList
            messages={assistant.messages}
            status={assistant.status}
            error={assistant.error}
            actions={actions}
            onSend={assistant.send}
            onRetry={assistant.retry}
          />
          <Composer
            streaming={assistant.status === 'streaming'}
            onSend={assistant.send}
            onStop={assistant.stop}
            inputRef={inputRef}
          />
        </>
      ) : (
        <NotConfigured />
      )}
      <span className="sr-only" role="status" aria-live="polite">
        {assistant.status === 'streaming' ? 'Assistant is working' : ''}
      </span>
    </aside>
  );
};
