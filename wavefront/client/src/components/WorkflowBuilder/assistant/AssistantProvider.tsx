import React from 'react';
import { useSearchParams } from 'react-router';
import { EditorBridgeProvider } from './EditorBridgeProvider';
import { MockAssistantProvider } from './MockAssistantProvider';
import { parseMockScenario } from './mockScenarios';
import { RealAssistantProvider } from './RealAssistantProvider';

// Mounts the editor bridge and one WorkflowAssistant implementation above the workflow editor: the real streaming
// provider by default (it reports available=false when the app has no usable LLM config, see
// UnavailableAssistantProvider for the static off state). ?assistantMock=<scenario> mounts the scripted mock instead (dev only, for building UI without a backend).
export const AssistantProvider = ({ children }: { children: React.ReactNode }) => {
  const [searchParams] = useSearchParams();
  const scenario = parseMockScenario(searchParams.get('assistantMock'));

  return (
    <EditorBridgeProvider>
      {scenario ? (
        <MockAssistantProvider scenario={scenario}>{children}</MockAssistantProvider>
      ) : (
        <RealAssistantProvider>{children}</RealAssistantProvider>
      )}
    </EditorBridgeProvider>
  );
};
