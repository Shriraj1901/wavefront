import { appEnv } from '@app/config/env';
import { TOKEN_KEY } from '@app/lib/constants';
import { consumeSSE } from '@app/utils/sse';
import {
  describeAssistantHttpError,
  extractServerMessage,
  type AssistantChatBody,
} from '@app/components/WorkflowBuilder/assistant/assistantRequest';

export class AssistantRequestError extends Error {}

/**
 * Stream one assistant turn, calling `onFrame` with every parsed SSE payload (unvalidated: the caller runs the
 * events.ts guards).
 *
 * Uses fetch rather than the shared axios instance for the same reason as AgentService.streamInference: XHR buffers
 * the whole response. Failures before the stream opens (bad config, disabled assistant, auth) throw an
 * AssistantRequestError with a user-readable message.
 */
export async function streamWorkflowAssistantChat(
  body: AssistantChatBody,
  options: { signal?: AbortSignal; onFrame: (frame: unknown) => void }
): Promise<void> {
  const appId = window.location.pathname.split('/')[2] ?? '';
  const url = `${appEnv.baseURL}/v1/${appId}/floware/v1/workflow-management/assistant/chat`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY)}`,
      'Cache-Control': 'no-cache',
    },
    body: JSON.stringify(body),
    signal: options.signal,
  });

  if (!response.ok) {
    let serverMessage: string | undefined;
    try {
      serverMessage = extractServerMessage(await response.json());
    } catch {
      // Not JSON: fall back to the status-based message
    }
    throw new AssistantRequestError(describeAssistantHttpError(response.status, serverMessage));
  }

  await consumeSSE(response, options.onFrame);
}
