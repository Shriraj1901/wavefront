import type { ProposalError } from './types';

// SSE `data:` frames sent by the assistant endpoint. Every frame also carries a numeric `timestamp`.
interface BaseEvent {
  timestamp?: number;
}

export interface AssistantStartedEvent extends BaseEvent {
  event_type: 'assistant_started';
}

export interface ContentDeltaEvent extends BaseEvent {
  event_type: 'content_delta';
  content: string;
}

// Everything streamed as content_delta so far is withdrawn (a guardrail revised it); `content` replaces it, and is
// empty when nothing may be shown. Additive: older servers never send it.
export interface RetractEvent extends BaseEvent {
  event_type: 'retract';
  content: string;
  reason?: string | null;
}

export interface ToolCalledEvent extends BaseEvent {
  event_type: 'tool_called';
  tool_call_id: string;
  tool_name: string;
  label: string;
}

export interface ToolResultEvent extends BaseEvent {
  event_type: 'tool_result';
  tool_call_id: string;
  tool_name: string;
  ok: boolean;
  summary?: string;
}

export interface ToolFailedEvent extends BaseEvent {
  event_type: 'tool_failed';
  tool_call_id: string;
  tool_name: string;
  error: string;
}

export interface ProposalEvent extends BaseEvent {
  event_type: 'proposal';
  proposal_id: string;
  yaml: string;
  valid: boolean;
  errors: ProposalError[];
  // Additive. Missing from older servers: treat as equal to errors.
  introduced_errors?: ProposalError[];
  // Additive. Choices the assistant made on the user's behalf. Missing from older servers.
  notes?: string[];
}

// End of turn: the full final text
export interface OutputEvent extends BaseEvent {
  event_type: 'output';
  content: string;
  // Additive. True when the provider cut the reply off at its output limit.
  truncated?: boolean;
}

export interface ErrorEvent extends BaseEvent {
  event_type: 'error';
  message: string;
}

export type AssistantEvent =
  | AssistantStartedEvent
  | ContentDeltaEvent
  | RetractEvent
  | ToolCalledEvent
  | ToolResultEvent
  | ToolFailedEvent
  | ProposalEvent
  | OutputEvent
  | ErrorEvent;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isString = (value: unknown): value is string => typeof value === 'string';

const isOptionalTimestamp = (value: unknown): boolean => value === undefined || typeof value === 'number';

const isProposalError = (value: unknown): value is ProposalError =>
  isRecord(value) && isString(value.path) && isString(value.message);

export const isAssistantStartedEvent = (value: unknown): value is AssistantStartedEvent =>
  isRecord(value) && value.event_type === 'assistant_started' && isOptionalTimestamp(value.timestamp);

export const isContentDeltaEvent = (value: unknown): value is ContentDeltaEvent =>
  isRecord(value) &&
  value.event_type === 'content_delta' &&
  isString(value.content) &&
  isOptionalTimestamp(value.timestamp);

export const isRetractEvent = (value: unknown): value is RetractEvent =>
  isRecord(value) &&
  value.event_type === 'retract' &&
  isString(value.content) &&
  (value.reason === undefined || value.reason === null || isString(value.reason)) &&
  isOptionalTimestamp(value.timestamp);

export const isToolCalledEvent = (value: unknown): value is ToolCalledEvent =>
  isRecord(value) &&
  value.event_type === 'tool_called' &&
  isString(value.tool_call_id) &&
  isString(value.tool_name) &&
  isString(value.label) &&
  isOptionalTimestamp(value.timestamp);

export const isToolResultEvent = (value: unknown): value is ToolResultEvent =>
  isRecord(value) &&
  value.event_type === 'tool_result' &&
  isString(value.tool_call_id) &&
  isString(value.tool_name) &&
  typeof value.ok === 'boolean' &&
  (value.summary === undefined || isString(value.summary)) &&
  isOptionalTimestamp(value.timestamp);

export const isToolFailedEvent = (value: unknown): value is ToolFailedEvent =>
  isRecord(value) &&
  value.event_type === 'tool_failed' &&
  isString(value.tool_call_id) &&
  isString(value.tool_name) &&
  isString(value.error) &&
  isOptionalTimestamp(value.timestamp);

export const isProposalEvent = (value: unknown): value is ProposalEvent =>
  isRecord(value) &&
  value.event_type === 'proposal' &&
  isString(value.proposal_id) &&
  isString(value.yaml) &&
  typeof value.valid === 'boolean' &&
  Array.isArray(value.errors) &&
  value.errors.every(isProposalError) &&
  (value.introduced_errors === undefined ||
    (Array.isArray(value.introduced_errors) && value.introduced_errors.every(isProposalError))) &&
  (value.notes === undefined || (Array.isArray(value.notes) && value.notes.every(isString))) &&
  isOptionalTimestamp(value.timestamp);

export const isOutputEvent = (value: unknown): value is OutputEvent =>
  isRecord(value) &&
  value.event_type === 'output' &&
  isString(value.content) &&
  (value.truncated === undefined || typeof value.truncated === 'boolean') &&
  isOptionalTimestamp(value.timestamp);

export const isErrorEvent = (value: unknown): value is ErrorEvent =>
  isRecord(value) && value.event_type === 'error' && isString(value.message) && isOptionalTimestamp(value.timestamp);

// Narrows an arbitrary parsed frame to a known, well-formed event. Unknown or malformed frames return false.
export const isAssistantEvent = (value: unknown): value is AssistantEvent =>
  isAssistantStartedEvent(value) ||
  isContentDeltaEvent(value) ||
  isRetractEvent(value) ||
  isToolCalledEvent(value) ||
  isToolResultEvent(value) ||
  isToolFailedEvent(value) ||
  isProposalEvent(value) ||
  isOutputEvent(value) ||
  isErrorEvent(value);
