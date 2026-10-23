import { createContext, useContext } from 'react';
import type { WorkflowAssistant } from './types';

// Set by whichever provider is mounted (mock, unavailable, later the real streaming one)
export const AssistantContext = createContext<WorkflowAssistant | null>(null);

// The only hook the UI may use to reach the assistant
export const useWorkflowAssistant = (): WorkflowAssistant => {
  const assistant = useContext(AssistantContext);
  if (!assistant) {
    throw new Error('useWorkflowAssistant must be used inside <AssistantProvider>');
  }
  return assistant;
};
