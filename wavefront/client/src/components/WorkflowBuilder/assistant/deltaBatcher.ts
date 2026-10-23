// Batches streamed text so a long answer re-renders at most once per animation frame instead of once per chunk.
// Pure apart from the injected scheduler, so it is testable in the node env. The reducer stays unaware of it: the hook
// layer feeds it content_delta text and dispatches the combined text.

const FALLBACK_FRAME_MS = 16;

export type CancelScheduled = () => void;
export type Schedule = (callback: () => void) => CancelScheduled;

// requestAnimationFrame where it exists, a 16 ms timeout where it does not (node, workers, old webviews)
export const scheduleFrame: Schedule = (callback) => {
  if (typeof requestAnimationFrame === 'function' && typeof cancelAnimationFrame === 'function') {
    const id = requestAnimationFrame(() => callback());
    return () => cancelAnimationFrame(id);
  }
  const id = setTimeout(callback, FALLBACK_FRAME_MS);
  return () => clearTimeout(id);
};

export interface DeltaBatcher {
  // Queues text; `emit` receives everything queued by the next frame as one string
  push: (text: string) => void;
  // Emits whatever is queued now (before an event that must see the text: output, error, stop)
  flush: () => void;
  // Drops whatever is queued (a new turn, a reset, unmount)
  discard: () => void;
}

export const createDeltaBatcher = (emit: (text: string) => void, schedule: Schedule = scheduleFrame): DeltaBatcher => {
  let pending = '';
  let cancel: CancelScheduled | null = null;

  const emitPending = () => {
    if (!pending) return;
    const text = pending;
    pending = '';
    emit(text);
  };

  return {
    push: (text) => {
      if (!text) return;
      pending += text;
      if (!cancel) {
        cancel = schedule(() => {
          cancel = null;
          emitPending();
        });
      }
    },
    flush: () => {
      cancel?.();
      cancel = null;
      emitPending();
    },
    discard: () => {
      cancel?.();
      cancel = null;
      pending = '';
    },
  };
};
