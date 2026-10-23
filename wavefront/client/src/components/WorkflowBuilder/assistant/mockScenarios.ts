import { NODE_TYPES } from '@app/constants/workflow-builder';
import { applyGraphAction, isExecutableNode, nextNodeId, normalizeGraph, type Graph } from '../graphUtils';
import { createWorkflowNode } from '../nodeFactory';
import { serializeGraphToYaml } from '../yamlSync';
import { createLinearTemplate } from '../workflowTemplates';
import type { AssistantEvent } from './events';
import { parseGraph } from './proposalGraph';
import type { ProposalError } from './types';

export const MOCK_SCENARIOS = [
  'plain-reply',
  'streaming-steps',
  'valid-proposal',
  'valid-with-notes',
  'invalid-proposal',
  'valid-with-existing-issues',
  'tool-failure',
  'long-text',
  'streaming-long-answer',
  'error',
  'unavailable',
] as const;

export type MockScenario = (typeof MOCK_SCENARIOS)[number];

const DEFAULT_SCENARIO: MockScenario = 'valid-proposal';

// Value of the ?assistantMock query param. null means the param is absent (no mock); an empty or unknown value
// picks the default scenario.
export const parseMockScenario = (value: string | null): MockScenario | null => {
  if (value === null) return null;
  return (MOCK_SCENARIOS as readonly string[]).includes(value) ? (value as MockScenario) : DEFAULT_SCENARIO;
};

export interface ScriptStep {
  // Wait before this event, after the previous one
  delayMs: number;
  event: AssistantEvent;
}

interface ScriptContext {
  currentYaml: string;
  proposalId: string;
  now: number;
}

const INVALID_PROPOSAL_ERRORS: ProposalError[] = [
  { path: 'arium.agents[0].model', message: 'Model is required: choose a model for this agent.' },
  { path: 'arium.workflow.edges[1].to', message: "Node 'missing_step' is not defined." },
];

// Problems the canvas workflow "already had" in the valid-with-existing-issues scenario
const EXISTING_ERRORS: ProposalError[] = [
  { path: 'arium.agents[0].model', message: 'Model is required: choose a model for this agent.' },
  { path: 'arium.agents[1].model', message: 'Model is required: choose a model for this agent.' },
];

// A proposal derived from the YAML the editor holds right now, so apply, diff and undo run against real data:
// the first agent's job is reworded and a new agent is chained after the last open-ended agent. An empty canvas
// gets the linear template instead. null when the current YAML cannot be drawn.
export const buildMockProposalYaml = (currentYaml: string): string | null => {
  const parsed = currentYaml.trim() ? parseGraph(currentYaml) : { nodes: [], edges: [] };
  if (!parsed) return null;
  let graph: Graph = parsed.nodes.some((n) => isExecutableNode(n)) ? parsed : normalizeGraph(createLinearTemplate());

  const agents = graph.nodes.filter((n) => n.type === NODE_TYPES.AGENT);
  const first = agents[0];
  if (first) {
    const job = typeof first.data.job === 'string' ? first.data.job : '';
    graph = applyGraphAction(graph, {
      type: 'updateNodeData',
      id: first.id,
      patch: { job: `${job} Keep answers short and cite sources.`.trim() },
    });
  }

  const hasOutgoing = new Set(graph.edges.map((e) => e.source));
  const tail = [...agents].reverse().find((n) => !hasOutgoing.has(n.id)) ?? agents[agents.length - 1];
  const node = createWorkflowNode(
    NODE_TYPES.AGENT,
    nextNodeId('reviewer', graph.nodes),
    { x: 0, y: 0 },
    { role: 'Reviewer', job: 'Reviews the previous answer for accuracy and tone.' }
  );
  graph = applyGraphAction(graph, { type: 'addNode', node, connectFrom: tail ? { source: tail.id } : undefined });

  const result = serializeGraphToYaml(graph.nodes, graph.edges, currentYaml);
  return result.ok ? result.yaml : null;
};

const LONG_TEXT = [
  '## Plan',
  'I looked at the current workflow and I would change it in three steps.',
  '- **Tighten the entry agent.** Its job is broad; narrowing it makes routing more reliable.',
  '- **Add a reviewer.** A final pass catches tone and factual slips before the answer leaves the workflow.',
  '- **Keep the data wires.** The `input_filter` entries already give each step only what it needs.',
  '## Why',
  'Every agent in a chain sees the whole conversation by default. Narrow jobs and explicit filters cost fewer ' +
    'tokens and make failures easier to trace back to one step. ' +
    'Longer chains add latency, so keep the reviewer optional if the workflow is latency sensitive.',
  '```yaml\narium:\n  workflow:\n    start: entry\n    edges:\n      - from: entry\n        to: [reviewer]\n```',
  'Let me know if you want a different shape, for example a router that skips the reviewer for short answers.',
].join('\n\n');

// An explanation long enough to stream for several seconds, with the markdown shapes that stress partial rendering
const STREAMING_TEXT = [
  '## What this workflow does',
  'It takes the **user request**, hands it to the entry agent and passes the draft along a short chain of agents. ' +
    'Each step reads only what its `input_filter` allows, so a later agent never sees raw tool output it does not need.',
  '- **Entry agent.** Reads the request and decides what the answer has to cover.',
  '- **Writer.** Produces the first draft from the entry agent notes.',
  '- **Reviewer.** Checks tone and factual slips before the answer leaves the workflow.',
  '1. The request arrives at the start node.',
  '2. The entry agent writes its notes.',
  '3. The writer drafts, then the reviewer edits and finishes.',
  LONG_TEXT,
  'In short: a linear chain, easy to trace, with one cheap review pass at the end. Ask me if you want a router or a ' +
    'second reviewer added.',
].join('\n\n');

const charDeltas = (text: string, ctx: ScriptContext, size = 4, delayMs = 15): ScriptStep[] => {
  const steps: ScriptStep[] = [];
  for (let i = 0; i < text.length; i += size) {
    steps.push({
      delayMs,
      event: { event_type: 'content_delta', content: text.slice(i, i + size), timestamp: ctx.now },
    });
  }
  return steps;
};

const words = (text: string, chunk = 3): string[] => {
  const parts = text.split(/(\s+)/);
  const out: string[] = [];
  for (let i = 0; i < parts.length; i += chunk * 2) out.push(parts.slice(i, i + chunk * 2).join(''));
  return out.filter(Boolean);
};

const deltas = (text: string, ctx: ScriptContext, delayMs = 40): ScriptStep[] =>
  words(text).map((content) => ({ delayMs, event: { event_type: 'content_delta', content, timestamp: ctx.now } }));

const started = (ctx: ScriptContext): ScriptStep => ({
  delayMs: 150,
  event: { event_type: 'assistant_started', timestamp: ctx.now },
});

const toolCall = (ctx: ScriptContext, id: string, name: string, label: string, summary: string): ScriptStep[] => [
  { delayMs: 300, event: { event_type: 'tool_called', tool_call_id: id, tool_name: name, label, timestamp: ctx.now } },
  {
    delayMs: 700,
    event: { event_type: 'tool_result', tool_call_id: id, tool_name: name, ok: true, summary, timestamp: ctx.now },
  },
];

const output = (ctx: ScriptContext, content: string): ScriptStep => ({
  delayMs: 100,
  event: { event_type: 'output', content, timestamp: ctx.now },
});

const reply = (ctx: ScriptContext, text: string): ScriptStep[] => [...deltas(text, ctx), output(ctx, text)];

type ProposalKind = 'valid' | 'valid-notes' | 'invalid' | 'existing-issues';

// What the assistant says it decided for the user in the valid-with-notes scenario
const MOCK_NOTES = [
  'New agent reviewer_1 uses the same model as the agent before it.',
  'Its prompt is a generic review prompt; edit it in the node settings.',
  'I reworded the first agent so it hands its draft to the reviewer.',
];

const proposalSteps = (ctx: ScriptContext, kind: ProposalKind): ScriptStep[] => {
  const valid = kind === 'valid' || kind === 'valid-notes';
  const yaml = buildMockProposalYaml(ctx.currentYaml);
  if (yaml === null) {
    return [
      {
        delayMs: 300,
        event: {
          event_type: 'error',
          message: 'Mock: the current YAML cannot be drawn, so no proposal was built.',
          timestamp: ctx.now,
        },
      },
    ];
  }
  const text = {
    valid: 'I reworded the first agent and added a reviewer at the end. Review the changes below, then apply them.',
    'valid-notes':
      'I added a reviewer at the end and reworded the first agent. I listed the choices I made below so you can check them.',
    invalid:
      'I drafted a change, but it has problems the validator rejected. Nothing can be applied until they are fixed.',
    'existing-issues':
      'I added a reviewer at the end. The new agent has no model yet, so choose one for it. ' +
      'The workflow still has validation issues from before; they do not block this change.',
  }[kind];
  const errors = { valid: [], 'valid-notes': [], invalid: INVALID_PROPOSAL_ERRORS, 'existing-issues': EXISTING_ERRORS }[
    kind
  ];
  // Only the invalid scenario's problems come from the change itself
  const introducedErrors = kind === 'invalid' ? INVALID_PROPOSAL_ERRORS : [];
  return [
    ...toolCall(ctx, 'tool-1', 'read_workflow', 'Reading the current workflow', 'Read the workflow'),
    ...toolCall(
      ctx,
      'tool-2',
      'propose_workflow_yaml',
      'Drafting the change',
      {
        valid: 'Valid',
        'valid-notes': 'Valid',
        invalid: 'Rejected by the validator',
        'existing-issues': 'Ready; 2 existing issues remain',
      }[kind]
    ),
    {
      delayMs: 200,
      event: {
        event_type: 'proposal',
        proposal_id: ctx.proposalId,
        yaml,
        valid,
        errors,
        introduced_errors: introducedErrors,
        ...(kind === 'valid-notes' ? { notes: MOCK_NOTES } : {}),
        timestamp: ctx.now,
      },
    },
    ...reply(ctx, text),
  ];
};

// The events a scripted turn emits, with the pause before each. 'unavailable' emits nothing: it only turns the
// assistant off.
export const buildScript = (scenario: MockScenario, ctx: ScriptContext): ScriptStep[] => {
  switch (scenario) {
    case 'plain-reply':
      return [started(ctx), ...reply(ctx, 'Sure. Describe the workflow you want and I will draft it for you.')];

    case 'streaming-steps':
      return [
        started(ctx),
        ...toolCall(ctx, 'tool-1', 'list_resources', 'Looking at your agents and functions', '4 agents, 6 functions'),
        ...toolCall(ctx, 'tool-2', 'read_workflow', 'Reading the current workflow', 'Read the workflow'),
        ...reply(ctx, 'I looked at what you have. A router in front of the writers would fit well here.'),
      ];

    case 'valid-proposal':
      return [started(ctx), ...proposalSteps(ctx, 'valid')];

    case 'valid-with-notes':
      return [started(ctx), ...proposalSteps(ctx, 'valid-notes')];

    case 'invalid-proposal':
      return [started(ctx), ...proposalSteps(ctx, 'invalid')];

    case 'valid-with-existing-issues':
      return [started(ctx), ...proposalSteps(ctx, 'existing-issues')];

    case 'tool-failure':
      return [
        started(ctx),
        {
          delayMs: 300,
          event: {
            event_type: 'tool_called',
            tool_call_id: 'tool-1',
            tool_name: 'list_resources',
            label: 'Looking at your agents and functions',
            timestamp: ctx.now,
          },
        },
        {
          delayMs: 900,
          event: {
            event_type: 'tool_failed',
            tool_call_id: 'tool-1',
            tool_name: 'list_resources',
            error: 'Timed out reading resources.',
            timestamp: ctx.now,
          },
        },
        ...reply(ctx, 'I could not read your resources, so I am working from the YAML on the canvas only.'),
      ];

    case 'long-text':
      return [started(ctx), ...reply(ctx, LONG_TEXT)];

    case 'streaming-long-answer':
      return [started(ctx), ...charDeltas(STREAMING_TEXT, ctx), output(ctx, STREAMING_TEXT)];

    case 'error':
      return [
        started(ctx),
        ...deltas('Let me think about that', ctx),
        {
          delayMs: 600,
          event: {
            event_type: 'error',
            message: 'The model request failed. Try again in a moment.',
            timestamp: ctx.now,
          },
        },
      ];

    case 'unavailable':
      return [];

    default:
      return [];
  }
};
