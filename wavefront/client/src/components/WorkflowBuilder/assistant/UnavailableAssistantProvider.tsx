import React from 'react';
import { AssistantSurface } from './AssistantSurface';
import type { WorkflowAssistant } from './types';

const noop = () => {};

// Module-level so the value is stable: nothing re-renders because of it
const UNAVAILABLE_ASSISTANT: WorkflowAssistant = {
  available: false,
  messages: [],
  status: 'idle',
  send: noop,
  stop: noop,
  retry: noop,
  reset: noop,
  models: [],
  selectModel: noop,
  previewProposal: noop,
  applyProposal: noop,
  discardProposal: noop,
  canUndo: false,
  undoLastApply: noop,
  focusNode: noop,
};

// Used until the real streaming provider exists, and for any scenario where the assistant is off
export const UnavailableAssistantProvider = ({ children }: { children: React.ReactNode }) => (
  <AssistantSurface assistant={UNAVAILABLE_ASSISTANT}>{children}</AssistantSurface>
);
