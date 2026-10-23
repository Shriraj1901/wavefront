import { describe, expect, it } from 'vitest';
import type { LLMInferenceConfig } from '@app/types/llm-inference-config';
import {
  ASSISTANT_HISTORY_LIMIT,
  buildChatBody,
  buildHistory,
  describeAssistantHttpError,
  extractServerMessage,
  formatValidationDetail,
  isUsableLlmConfig,
  pickDefaultModelId,
  toAssistantModels,
} from './assistantRequest';
import type { AssistantMessage } from './types';

const msg = (role: 'user' | 'assistant', text: string): AssistantMessage => ({
  id: text,
  role,
  text,
  steps: [],
  createdAt: 0,
});

const config = (over: Partial<LLMInferenceConfig>): LLMInferenceConfig => ({
  id: 'c1',
  llm_model: 'gpt-4o',
  display_name: 'GPT',
  type: 'openai',
  is_deleted: false,
  created_at: '',
  updated_at: '',
  ...over,
});

describe('buildHistory', () => {
  it('keeps text only and drops empty replies', () => {
    const history = buildHistory([msg('user', 'hi'), { ...msg('assistant', ''), steps: [] }, msg('assistant', 'yo')]);
    expect(history).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'yo' },
    ]);
  });
});

describe('buildHistory trimming', () => {
  const chat = (n: number) => Array.from({ length: n }, (_, i) => msg(i % 2 === 0 ? 'user' : 'assistant', `turn ${i}`));

  it('sends at most the last ASSISTANT_HISTORY_LIMIT messages', () => {
    expect(ASSISTANT_HISTORY_LIMIT).toBe(20);
    const history = buildHistory(chat(150));
    expect(history).toHaveLength(ASSISTANT_HISTORY_LIMIT);
    expect(history[0].content).toBe('turn 130');
    expect(history[history.length - 1].content).toBe('turn 149');
  });

  it('keeps a short chat whole', () => {
    expect(buildHistory(chat(3))).toHaveLength(3);
  });

  it('drops empty replies before counting', () => {
    const messages = [...chat(25), msg('assistant', '   ')];
    const history = buildHistory(messages);
    expect(history).toHaveLength(ASSISTANT_HISTORY_LIMIT);
    expect(history[history.length - 1].content).toBe('turn 24');
  });
});

describe('buildChatBody', () => {
  it('builds the wire body', () => {
    expect(buildChatBody('add', 'yaml', [msg('user', 'a')], 'cfg')).toEqual({
      message: 'add',
      current_yaml: 'yaml',
      history: [{ role: 'user', content: 'a' }],
      llm_inference_config_id: 'cfg',
    });
  });
});

describe('model selection', () => {
  it('filters unusable configs and names them', () => {
    const models = toAssistantModels([
      config({ id: 'a' }),
      config({ id: 'b', is_deleted: true }),
      config({ id: 'c', type: 'bedrock' }),
      config({ id: 'd', model_type: 'embedding' }),
      config({ id: 'e', display_name: '', llm_model: 'raw' }),
    ]);
    expect(models).toEqual([
      { id: 'a', name: 'GPT' },
      { id: 'e', name: 'raw' },
    ]);
    expect(isUsableLlmConfig(config({ model_type: 'llm' }))).toBe(true);
    expect(toAssistantModels(undefined)).toEqual([]);
  });

  it('prefers the last used model when it still exists', () => {
    const models = [
      { id: 'a', name: 'A' },
      { id: 'b', name: 'B' },
    ];
    expect(pickDefaultModelId(models, 'b')).toBe('b');
    expect(pickDefaultModelId(models, 'gone')).toBe('a');
    expect(pickDefaultModelId(models, null)).toBe('a');
    expect(pickDefaultModelId([], 'a')).toBeUndefined();
  });
});

describe('describeAssistantHttpError', () => {
  it('maps each status to a readable message', () => {
    expect(describeAssistantHttpError(400)).toMatch(/different model/);
    expect(describeAssistantHttpError(400, 'Bedrock is not supported.')).toMatch(/^Bedrock is not supported\./);
    expect(describeAssistantHttpError(401)).toMatch(/Sign in/);
    expect(describeAssistantHttpError(403)).toMatch(/permission/);
    expect(describeAssistantHttpError(404)).toMatch(/no longer exists/);
    expect(describeAssistantHttpError(503)).toMatch(/turned off/);
    expect(describeAssistantHttpError(500)).toMatch(/server error/);
    expect(describeAssistantHttpError(418)).toMatch(/418/);
  });
});

describe('422 validation errors', () => {
  const detail = [
    {
      loc: ['body', 'history', 3, 'content'],
      msg: 'String should have at most 50000 characters',
      type: 'string_too_long',
    },
    { loc: ['body', 'message'], msg: 'Field required', type: 'missing' },
    { loc: ['body', 'mode'], msg: 'Input should be auto, ask or edit', type: 'literal_error' },
  ];

  it('reads field and message from the detail array', () => {
    expect(formatValidationDetail(detail.slice(0, 1))).toBe(
      'history.3.content: String should have at most 50000 characters'
    );
    expect(formatValidationDetail([{ loc: [], msg: 'Bad' }])).toBe('Bad');
  });

  it('shows the first two issues and counts the rest', () => {
    expect(formatValidationDetail(detail)).toBe(
      'history.3.content: String should have at most 50000 characters; message: Field required (and 1 more)'
    );
  });

  it('ignores anything that is not a validation list', () => {
    expect(formatValidationDetail('nope')).toBeUndefined();
    expect(formatValidationDetail([])).toBeUndefined();
    expect(formatValidationDetail([{ loc: ['x'] }, null, 4])).toBeUndefined();
  });

  it('extracts a message from each response body shape', () => {
    expect(extractServerMessage({ meta: { error: 'App says no' } })).toBe('App says no');
    expect(extractServerMessage({ detail: 'Plain detail' })).toBe('Plain detail');
    expect(extractServerMessage({ detail: detail.slice(1, 2) })).toBe('message: Field required');
    expect(extractServerMessage({ meta: { error: 5 } })).toBeUndefined();
    expect(extractServerMessage(null)).toBeUndefined();
  });

  it('describes a 422 with the field instead of a bare status', () => {
    const message = describeAssistantHttpError(422, extractServerMessage({ detail: detail.slice(1, 2) }));
    expect(message).toMatch(/message: Field required/);
    expect(message).not.toMatch(/status 422/);
    expect(describeAssistantHttpError(422)).not.toMatch(/status 422/);
  });
});
