import { describe, expect, it } from 'vitest';
import { DEFAULT_WORKFLOW_YAML } from '@app/pages/apps/[appId]/workflows/defaultWorkflowYaml';
import { assistantReducer, initialAssistantState, type AssistantReducerDeps } from './assistantReducer';
import { isAssistantEvent } from './events';
import { MOCK_SCENARIOS, buildMockProposalYaml, buildScript, parseMockScenario } from './mockScenarios';
import { getApplyState } from './ui/proposalState';
import { buildPreviewOverlay, parseGraph, summarizeYamlDiff } from './proposalGraph';

const ctx = (currentYaml: string) => ({ currentYaml, proposalId: 'p1', now: 5 });

const deps = (currentYaml: string): AssistantReducerDeps => ({
  computeDiff: (proposed) => summarizeYamlDiff(currentYaml, proposed),
  now: () => 5,
});

const playScript = (scenario: (typeof MOCK_SCENARIOS)[number], currentYaml: string) =>
  buildScript(scenario, ctx(currentYaml)).reduce(
    (state, step) => assistantReducer(state, { type: 'event', event: step.event, deps: deps(currentYaml) }),
    assistantReducer(initialAssistantState, { type: 'user_sent', text: 'go', now: 1 })
  );

describe('parseMockScenario', () => {
  it('returns null without the param, the scenario when known, the default otherwise', () => {
    expect(parseMockScenario(null)).toBeNull();
    expect(parseMockScenario('error')).toBe('error');
    expect(parseMockScenario('')).toBe('valid-proposal');
    expect(parseMockScenario('nope')).toBe('valid-proposal');
  });
});

describe('buildMockProposalYaml', () => {
  it('derives a drawable proposal from the current YAML: a reviewer added, the first agent modified', () => {
    const proposed = buildMockProposalYaml(DEFAULT_WORKFLOW_YAML);
    expect(proposed).not.toBeNull();
    expect(parseGraph(proposed ?? '')).not.toBeNull();
    const diff = summarizeYamlDiff(DEFAULT_WORKFLOW_YAML, proposed ?? '');
    expect(diff.nodesAdded).toEqual(['reviewer_1']);
    expect(diff.nodesModified).toContain('content_creator');
    expect(diff.nodesRemoved).toEqual([]);
    expect(diff.edgesAdded).toBeGreaterThanOrEqual(1);
  });

  it('is built on the CURRENT YAML: proposing again on the result adds the next reviewer', () => {
    const once = buildMockProposalYaml(DEFAULT_WORKFLOW_YAML) ?? '';
    const twice = buildMockProposalYaml(once) ?? '';
    expect(summarizeYamlDiff(once, twice).nodesAdded).toEqual(['reviewer_2']);
  });

  it('falls back to a starter workflow on an empty canvas', () => {
    const proposed = buildMockProposalYaml('');
    expect(proposed).not.toBeNull();
    expect(buildPreviewOverlay('', proposed ?? '')?.diff.summary.nodesAdded).toContain('reviewer_1');
  });

  it('returns null when the current YAML cannot be drawn', () => {
    expect(buildMockProposalYaml('not_arium: 1')).toBeNull();
  });
});

describe('buildScript', () => {
  it('emits only well-formed wire events in every scenario', () => {
    MOCK_SCENARIOS.forEach((scenario) => {
      buildScript(scenario, ctx(DEFAULT_WORKFLOW_YAML)).forEach((step) => {
        expect(isAssistantEvent(step.event)).toBe(true);
      });
    });
  });

  it('unavailable emits nothing', () => {
    expect(buildScript('unavailable', ctx(DEFAULT_WORKFLOW_YAML))).toEqual([]);
  });

  it.each(['plain-reply', 'streaming-steps', 'long-text', 'streaming-long-answer', 'tool-failure'] as const)(
    '%s ends idle with assistant text',
    (scenario) => {
      const state = playScript(scenario, DEFAULT_WORKFLOW_YAML);
      expect(state.status).toBe('idle');
      expect(state.messages[state.messages.length - 1].text.length).toBeGreaterThan(0);
    }
  );

  it('streaming-long-answer sends many small deltas that add up to the final text', () => {
    const script = buildScript('streaming-long-answer', ctx(DEFAULT_WORKFLOW_YAML));
    const deltas = script.flatMap((s) => (s.event.event_type === 'content_delta' ? [s.event.content] : []));
    const final = script[script.length - 1].event;
    expect(deltas.length).toBeGreaterThan(100);
    expect(Math.max(...deltas.map((d) => d.length))).toBeLessThanOrEqual(6);
    expect(final.event_type).toBe('output');
    expect(final.event_type === 'output' && final.content).toBe(deltas.join(''));
    expect(script.every((s) => s.delayMs > 0)).toBe(true);
  });

  it('streaming-steps ends with every step done', () => {
    const state = playScript('streaming-steps', DEFAULT_WORKFLOW_YAML);
    const steps = state.messages[state.messages.length - 1].steps;
    expect(steps.length).toBeGreaterThan(0);
    expect(steps.every((s) => s.status === 'done')).toBe(true);
  });

  it('tool-failure keeps the failed step and still finishes the turn', () => {
    const state = playScript('tool-failure', DEFAULT_WORKFLOW_YAML);
    const last = state.messages[state.messages.length - 1];
    expect(last.steps[0]).toMatchObject({ status: 'failed', detail: 'Timed out reading resources.' });
    expect(state.status).toBe('idle');
  });

  it('valid-proposal yields a valid pending proposal with a real diff', () => {
    const state = playScript('valid-proposal', DEFAULT_WORKFLOW_YAML);
    const proposal = state.messages[state.messages.length - 1].proposal;
    expect(proposal).toMatchObject({ id: 'p1', valid: true, errors: [], state: 'pending' });
    expect(proposal?.diff.nodesAdded).toEqual(['reviewer_1']);
    expect(state.proposalYaml.p1).toBeTruthy();
  });

  it('invalid-proposal yields an invalid proposal with errors', () => {
    const state = playScript('invalid-proposal', DEFAULT_WORKFLOW_YAML);
    const proposal = state.messages[state.messages.length - 1].proposal;
    expect(proposal?.valid).toBe(false);
    expect(proposal?.errors.length).toBeGreaterThan(0);
    expect(proposal?.introducedErrors).toEqual(proposal?.errors);
  });

  it('valid-with-existing-issues has errors but none introduced, so it can be applied', () => {
    const state = playScript('valid-with-existing-issues', DEFAULT_WORKFLOW_YAML);
    const proposal = state.messages[state.messages.length - 1].proposal;
    expect(proposal?.valid).toBe(false);
    expect(proposal?.errors.length).toBeGreaterThan(0);
    expect(proposal?.introducedErrors).toEqual([]);
    expect(proposal && getApplyState(proposal, 'idle').enabled).toBe(true);
    expect(proposal?.diff.nodesAdded).toEqual(['reviewer_1']);
  });

  it('valid-with-notes carries the assistant notes and before -> after detail for the modified agent', () => {
    const state = playScript('valid-with-notes', DEFAULT_WORKFLOW_YAML);
    const proposal = state.messages[state.messages.length - 1].proposal;
    expect(proposal?.valid).toBe(true);
    expect(proposal?.notes?.length).toBeGreaterThan(0);
    const change = proposal?.diff.changes?.find((c) => c.nodeId === 'content_creator');
    expect(change?.field).toBe('prompt');
    expect(change?.before).not.toBe(change?.after);
  });

  it('error ends in the error status with partial text kept', () => {
    const state = playScript('error', DEFAULT_WORKFLOW_YAML);
    expect(state.status).toBe('error');
    expect(state.error).toBeTruthy();
    expect(state.messages[state.messages.length - 1].text).toContain('Let me think');
  });

  it('a proposal scenario on undrawable YAML ends in an error instead of a proposal', () => {
    const state = playScript('valid-proposal', 'not_arium: 1');
    expect(state.status).toBe('error');
    expect(state.messages[state.messages.length - 1].proposal).toBeUndefined();
  });
});
