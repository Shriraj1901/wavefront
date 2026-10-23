import type { LLMInferenceConfig } from '@app/types/llm-inference-config';
import type { AssistantMessage } from './types';

export interface AssistantHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantChatBody {
  message: string;
  current_yaml: string;
  history: AssistantHistoryItem[];
  llm_inference_config_id: string;
}

export interface AssistantModel {
  id: string;
  name: string;
}

// How many prior messages go with a request. Mirrors the server default `max_history_messages` (AssistantLimits): the
// server keeps only this many anyway, so sending more only risks the request size.
export const ASSISTANT_HISTORY_LIMIT = 20;

// Prior turns as plain text, the most recent ASSISTANT_HISTORY_LIMIT. Steps and proposals are not sent: the server
// rebuilds context from current_yaml.
export const buildHistory = (messages: readonly AssistantMessage[]): AssistantHistoryItem[] =>
  messages
    .filter((m) => m.text.trim().length > 0)
    .map((m) => ({ role: m.role, content: m.text }))
    .slice(-ASSISTANT_HISTORY_LIMIT);

export const buildChatBody = (
  message: string,
  currentYaml: string,
  priorMessages: readonly AssistantMessage[],
  llmInferenceConfigId: string
): AssistantChatBody => ({
  message,
  current_yaml: currentYaml,
  history: buildHistory(priorMessages),
  llm_inference_config_id: llmInferenceConfigId,
});

// The server rejects Bedrock for the assistant (no tool-calling path yet) and non-LLM configs
export const isUsableLlmConfig = (config: LLMInferenceConfig): boolean =>
  !config.is_deleted && (config.model_type ?? 'llm') === 'llm' && config.type !== 'bedrock';

export const toAssistantModels = (configs: readonly LLMInferenceConfig[] | undefined): AssistantModel[] =>
  (configs ?? []).filter(isUsableLlmConfig).map((c) => ({ id: c.id, name: c.display_name || c.llm_model }));

// Config id -> the name the user knows it by, for every config (including ones the assistant itself cannot use)
export const toModelNameMap = (configs: readonly LLMInferenceConfig[] | undefined): Record<string, string> =>
  Object.fromEntries((configs ?? []).map((c) => [c.id, c.display_name || c.llm_model]));

// Last used model if it still exists, otherwise the first one
export const pickDefaultModelId = (models: readonly AssistantModel[], lastUsedId?: string | null): string | undefined =>
  models.find((m) => m.id === lastUsedId)?.id ?? models[0]?.id;

const MAX_VALIDATION_ISSUES_SHOWN = 2;

// FastAPI 422 bodies carry `detail: [{ loc, msg, type }]`; this turns them into "field: message" text. Returns
// undefined when `detail` is not that shape.
export const formatValidationDetail = (detail: unknown): string | undefined => {
  if (!Array.isArray(detail) || detail.length === 0) return undefined;
  const issues = detail
    .map((item) => {
      if (typeof item !== 'object' || item === null) return undefined;
      const { loc, msg } = item as { loc?: unknown; msg?: unknown };
      if (typeof msg !== 'string' || !msg) return undefined;
      const field = Array.isArray(loc)
        ? loc.filter((part) => (typeof part === 'string' || typeof part === 'number') && part !== 'body').join('.')
        : '';
      return field ? `${field}: ${msg}` : msg;
    })
    .filter((issue): issue is string => issue !== undefined);
  if (issues.length === 0) return undefined;
  const shown = issues.slice(0, MAX_VALIDATION_ISSUES_SHOWN).join('; ');
  const more = issues.length - MAX_VALIDATION_ISSUES_SHOWN;
  return more > 0 ? `${shown} (and ${more} more)` : shown;
};

// The readable message in an error response body: the app's own `meta.error`, a string `detail`, or a validation list
export const extractServerMessage = (json: unknown): string | undefined => {
  if (typeof json !== 'object' || json === null) return undefined;
  const { meta, detail } = json as { meta?: { error?: unknown }; detail?: unknown };
  if (typeof meta?.error === 'string') return meta.error;
  if (typeof detail === 'string') return detail;
  return formatValidationDetail(detail);
};

// What the user reads when the request fails before the stream opens
export const describeAssistantHttpError = (status: number, serverMessage?: string): string => {
  switch (status) {
    case 400:
      return serverMessage
        ? `${serverMessage} Pick a different model and try again.`
        : 'This model cannot be used by the assistant. Pick a different model and try again.';
    case 401:
      return 'Your session has expired. Sign in again to keep using the assistant.';
    case 403:
      return 'You do not have permission to use the assistant in this app.';
    case 404:
      return 'The selected model configuration no longer exists. Pick a different model.';
    case 422:
      return serverMessage
        ? `The assistant could not read this request (${serverMessage}).`
        : 'The assistant could not read this request. Try again with a shorter message.';
    case 503:
      return 'The assistant is turned off on this server. Ask an administrator to enable it.';
    default:
      return status >= 500
        ? 'The assistant hit a server error. Try again in a moment.'
        : serverMessage || `The assistant request failed (status ${status}).`;
  }
};
