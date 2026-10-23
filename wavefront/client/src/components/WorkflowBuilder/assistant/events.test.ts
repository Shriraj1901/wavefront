import { describe, expect, it } from 'vitest';
import {
  isAssistantEvent,
  isAssistantStartedEvent,
  isContentDeltaEvent,
  isErrorEvent,
  isOutputEvent,
  isProposalEvent,
  isRetractEvent,
  isToolCalledEvent,
  isToolFailedEvent,
  isToolResultEvent,
} from './events';

const frames = {
  assistant_started: { event_type: 'assistant_started', timestamp: 1 },
  content_delta: { event_type: 'content_delta', content: 'hi', timestamp: 1 },
  retract: { event_type: 'retract', content: 'safe', reason: null, timestamp: 1 },
  tool_called: { event_type: 'tool_called', tool_call_id: 't', tool_name: 'x', label: 'Doing x', timestamp: 1 },
  tool_result: { event_type: 'tool_result', tool_call_id: 't', tool_name: 'x', ok: true, summary: 's', timestamp: 1 },
  tool_failed: { event_type: 'tool_failed', tool_call_id: 't', tool_name: 'x', error: 'boom', timestamp: 1 },
  proposal: {
    event_type: 'proposal',
    proposal_id: 'p',
    yaml: 'arium: {}',
    valid: false,
    errors: [{ path: 'a', message: 'b' }],
    timestamp: 1,
  },
  output: { event_type: 'output', content: 'done', timestamp: 1 },
  error: { event_type: 'error', message: 'bad', timestamp: 1 },
};

const guards = {
  assistant_started: isAssistantStartedEvent,
  content_delta: isContentDeltaEvent,
  retract: isRetractEvent,
  tool_called: isToolCalledEvent,
  tool_result: isToolResultEvent,
  tool_failed: isToolFailedEvent,
  proposal: isProposalEvent,
  output: isOutputEvent,
  error: isErrorEvent,
};

describe('event guards', () => {
  it.each(Object.keys(frames) as (keyof typeof frames)[])('%s: only its own guard accepts the frame', (name) => {
    Object.entries(guards).forEach(([guardName, guard]) => {
      expect(guard(frames[name])).toBe(guardName === name);
    });
    expect(isAssistantEvent(frames[name])).toBe(true);
  });

  it('accepts a frame without a timestamp and a tool_result without a summary', () => {
    expect(isAssistantEvent({ event_type: 'assistant_started' })).toBe(true);
    expect(isToolResultEvent({ event_type: 'tool_result', tool_call_id: 't', tool_name: 'x', ok: false })).toBe(true);
  });

  it('rejects frames with missing or wrongly typed fields', () => {
    expect(isContentDeltaEvent({ event_type: 'content_delta' })).toBe(false);
    expect(isContentDeltaEvent({ event_type: 'content_delta', content: 1 })).toBe(false);
    expect(isRetractEvent({ event_type: 'retract' })).toBe(false);
    expect(isRetractEvent({ event_type: 'retract', content: '', reason: 5 })).toBe(false);
    expect(isToolCalledEvent({ event_type: 'tool_called', tool_call_id: 't', tool_name: 'x' })).toBe(false);
    expect(isToolResultEvent({ event_type: 'tool_result', tool_call_id: 't', tool_name: 'x', ok: 'yes' })).toBe(false);
    expect(isProposalEvent({ ...frames.proposal, errors: [{ path: 'a' }] })).toBe(false);
    expect(isProposalEvent({ ...frames.proposal, valid: undefined })).toBe(false);
    expect(isProposalEvent({ ...frames.proposal, introduced_errors: [{ path: 'a' }] })).toBe(false);
    expect(isProposalEvent({ ...frames.proposal, introduced_errors: 'x' })).toBe(false);
    expect(isOutputEvent({ ...frames.output, timestamp: 'now' })).toBe(false);
    expect(isOutputEvent({ ...frames.output, truncated: 'yes' })).toBe(false);
  });

  it('accepts an output with or without truncated', () => {
    expect(isOutputEvent({ ...frames.output, truncated: true })).toBe(true);
    expect(isOutputEvent(frames.output)).toBe(true);
  });

  it('accepts a proposal with or without introduced_errors', () => {
    expect(isProposalEvent(frames.proposal)).toBe(true);
    expect(isProposalEvent({ ...frames.proposal, introduced_errors: [] })).toBe(true);
    expect(isProposalEvent({ ...frames.proposal, introduced_errors: [{ path: 'a', message: 'b' }] })).toBe(true);
  });

  it('accepts a proposal with or without notes, and rejects malformed notes', () => {
    expect(isProposalEvent({ ...frames.proposal, notes: [] })).toBe(true);
    expect(isProposalEvent({ ...frames.proposal, notes: ['uses gpt-4o-mini'] })).toBe(true);
    expect(isProposalEvent({ ...frames.proposal, notes: 'x' })).toBe(false);
    expect(isProposalEvent({ ...frames.proposal, notes: [1] })).toBe(false);
  });

  it('rejects unknown event types and non-objects', () => {
    expect(isAssistantEvent({ event_type: 'not_a_real_event', content: 'x' })).toBe(false);
    expect(isAssistantEvent({})).toBe(false);
    expect(isAssistantEvent(null)).toBe(false);
    expect(isAssistantEvent('output')).toBe(false);
    expect(isAssistantEvent(undefined)).toBe(false);
  });
});
