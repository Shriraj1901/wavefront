"""Routing and streaming in the workflow assistant, driven by mocked LLMs only."""

import asyncio
import json
import logging
import time

import pytest

from agents_module.models.workflow_assistant_schemas import AssistantChatRequest
from agents_module.services.agent_stream_events import GUARDRAIL_CONTROL_KEY
from agents_module.services.workflow_assistant.intent import (
    ASK,
    CLASSIFIER_PROMPT,
    EDIT,
    PREVIOUS_REPLY_CHARS,
    classify_intent,
    parse_intent,
)
from agents_module.services.workflow_assistant.limits import AssistantLimits
from agents_module.services.workflow_assistant.service import WorkflowAssistantService
from agents_module.services.workflow_assistant.streaming import (
    DeltaCoalescer,
    GuardrailBlocked,
    StreamedReply,
    stream_reply,
)

from workflow_assistant_fakes import (
    CANVAS_MARKER,
    MODEL_ID,
    FakeLLM,
    FakeWorkflowCrud,
    call,
    assistant_kwargs,
    fake_agent_inference_service,
    llm_config_dict,
    text,
    workflow_yaml,
)

LONG_ANSWER = 'This workflow has one writer agent that drafts text. ' * 6


def request(**overrides):
    base = {
        'message': 'what does this workflow do?',
        'current_yaml': workflow_yaml() + CANVAS_MARKER,
        'llm_inference_config_id': MODEL_ID,
    }
    base.update(overrides)
    return AssistantChatRequest(**base)


def make_service(llm, limits=None):
    return WorkflowAssistantService(
        **assistant_kwargs(fake_agent_inference_service(llm, llm_config_dict())),
        workflow_crud_service=FakeWorkflowCrud(),
        limits=limits or AssistantLimits(),
    )


async def run_chat(service, req):
    prepared = await service.prepare(req, user_id='user-1')
    return [e async for e in service.chat(prepared)]


def types(events):
    return [e['event_type'] for e in events]


def terminal(events):
    return [e for e in events if e['event_type'] in ('output', 'error')]


def deltas(events):
    return [e['content'] for e in events if e['event_type'] == 'content_delta']


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class TestDeltaCoalescer:
    def test_first_chunk_goes_out_at_once(self):
        assert DeltaCoalescer(clock=Clock()).add('a') == 'a'

    def test_small_chunks_inside_the_interval_are_batched(self):
        clock = Clock()
        c = DeltaCoalescer(interval=0.04, max_chars=80, clock=clock)
        assert c.add('a') == 'a'
        clock.now += 0.01
        assert c.add('b') is None
        clock.now += 0.01
        assert c.add('c') is None
        assert c.flush() == 'bc'
        assert c.flush() is None

    def test_interval_elapsed_releases_the_batch(self):
        clock = Clock()
        c = DeltaCoalescer(interval=0.04, max_chars=80, clock=clock)
        c.add('a')
        clock.now += 0.01
        assert c.add('b') is None
        clock.now += 0.05
        assert c.add('c') == 'bc'

    def test_char_threshold_releases_the_batch(self):
        clock = Clock()
        c = DeltaCoalescer(interval=10, max_chars=10, clock=clock)
        c.add('a')
        assert c.add('x' * 9) is None
        assert c.add('y') == 'x' * 9 + 'y'

    def test_discard_drops_waiting_text(self):
        clock = Clock()
        c = DeltaCoalescer(clock=clock)
        c.add('a')
        c.add('b')
        c.discard()
        assert c.flush() is None
        assert c.seconds_until_due() is None

    def test_seconds_until_due(self):
        clock = Clock()
        c = DeltaCoalescer(interval=0.04, clock=clock)
        assert c.seconds_until_due() is None
        c.add('a')
        c.add('b')
        assert c.seconds_until_due() == pytest.approx(0.04)
        clock.now += 1
        assert c.seconds_until_due() == 0.0


class TestParseIntent:
    @pytest.mark.parametrize(
        'reply, expected',
        [
            ('ASK', ASK),
            ('ask', ASK),
            ('Ask.', ASK),
            ('  **ASK**', ASK),
            ('ASK - they want an explanation', ASK),
            ('EDIT', EDIT),
            ('edit', EDIT),
            ('Edit it', EDIT),
            ('', EDIT),
            (None, EDIT),
            ('banana', EDIT),
            ('I think this is an ASK', EDIT),
            ('ASKING for a change', EDIT),
            ('12345', EDIT),
        ],
    )
    def test_only_a_leading_ask_is_ask(self, reply, expected):
        assert parse_intent(reply) == expected


def deadline_in(seconds=30.0):
    return time.monotonic() + seconds


class TestClassifyIntent:
    async def test_ask_and_edit(self):
        assert (
            await classify_intent(
                FakeLLM([text('ASK')]), 'why?', None, deadline=deadline_in()
            )
        )[0] == ASK
        assert (
            await classify_intent(
                FakeLLM([text('EDIT')]), 'add one', None, deadline=deadline_in()
            )
        )[0] == EDIT

    async def test_garbage_and_empty_are_edit(self):
        for reply in ('maybe', '', '???'):
            result, _ = await classify_intent(
                FakeLLM([text(reply)]), 'x', None, deadline=deadline_in()
            )
            assert result == EDIT

    async def test_exception_is_edit_with_no_tokens(self):
        class Crashing(FakeLLM):
            async def generate(self, *a, **k):
                raise RuntimeError('boom')

        assert await classify_intent(
            Crashing([]), 'x', None, deadline=deadline_in()
        ) == (EDIT, 0)

    async def test_out_of_time_is_edit_without_calling(self):
        llm = FakeLLM([text('ASK')])
        assert await classify_intent(llm, 'x', None, deadline=time.monotonic() - 1) == (
            EDIT,
            0,
        )
        assert llm.calls == []

    async def test_timeout_is_edit(self):
        llm = FakeLLM([text('ASK')], delay=0.5)
        result, _ = await classify_intent(llm, 'x', None, deadline=deadline_in(0.05))
        assert result == EDIT

    async def test_sends_no_functions_and_counts_tokens(self):
        llm = FakeLLM([text('ASK', usage={'total_tokens': 17})])
        result, tokens = await classify_intent(llm, 'x', None, deadline=deadline_in())
        assert (result, tokens) == (ASK, 17)
        assert llm.calls[0]['functions'] is None
        assert llm.calls[0]['messages'][0] == {
            'role': 'system',
            'content': CLASSIFIER_PROMPT,
        }

    async def test_previous_assistant_message_is_truncated_and_optional(self):
        llm = FakeLLM([text('EDIT'), text('EDIT')])
        await classify_intent(llm, 'yes do it', 'A' * 5000, deadline=deadline_in())
        body = llm.calls[0]['messages'][1]['content']
        assert 'A' * PREVIOUS_REPLY_CHARS in body
        assert 'A' * (PREVIOUS_REPLY_CHARS + 1) not in body
        assert 'yes do it' in body
        await classify_intent(llm, 'hello', None, deadline=deadline_in())
        assert (
            'Previous assistant message' not in llm.calls[1]['messages'][1]['content']
        )


class TestAskPath:
    async def test_streams_several_deltas_then_one_output_and_never_passes_functions(
        self,
    ):
        llm = FakeLLM([text('ASK'), text(LONG_ANSWER)], chunk_size=10)
        events = await run_chat(make_service(llm), request())
        assert types(events)[0] == 'assistant_started'
        assert len(deltas(events)) > 1
        assert ''.join(deltas(events)) == LONG_ANSWER
        assert len(terminal(events)) == 1 and events[-1]['event_type'] == 'output'
        assert events[-1]['content'] == LONG_ANSWER
        assert not {'proposal', 'tool_called', 'tool_result', 'tool_failed'} & set(
            types(events)
        )
        # the only generate() call is the classifier; the answer was streamed
        assert len(llm.calls) == 1 and llm.calls[0]['functions'] is None
        assert len(llm.stream_calls) == 1 and llm.stream_calls[0]['functions'] is None

    async def test_uses_the_answer_only_prompt_with_the_canvas_and_resources(self):
        llm = FakeLLM([text('ASK'), text('ok')])
        await run_chat(make_service(llm), request())
        system = llm.stream_calls[0]['messages'][0]['content']
        assert 'ANSWER QUESTIONS' in system
        assert 'never claim' in system.lower() or 'Never claim' in system
        assert 'UNTRUSTED CANVAS_YAML' in system
        assert f'- id: {MODEL_ID} | name: Main model' in system
        assert 'web_search' in system
        assert 'Scope: do what was asked' not in system
        assert 'Validation: introduced versus existing' not in system
        sent = llm.stream_calls[0]['messages']
        assert sent[-1] == {'role': 'user', 'content': 'what does this workflow do?'}

    async def test_mode_ask_skips_the_classifier(self):
        llm = FakeLLM([text('plain answer')])
        events = await run_chat(make_service(llm), request(mode='ask'))
        assert llm.calls == []
        assert events[-1]['content'] == 'plain answer'

    async def test_stream_failure_before_text_degrades_to_generate(self):
        class NoStream(FakeLLM):
            async def stream(self, *a, **k):
                raise RuntimeError('provider cannot stream')
                yield  # pragma: no cover

        llm = NoStream([text('whole answer')])
        events = await run_chat(make_service(llm), request(mode='ask'))
        assert deltas(events) == ['whole answer']
        assert events[-1]['event_type'] == 'output'
        assert len(terminal(events)) == 1

    async def test_llm_without_stream_degrades_to_generate(self):
        class Plain(FakeLLM):
            stream = None

        events = await run_chat(
            make_service(Plain([text('whole answer')])), request(mode='ask')
        )
        assert events[-1]['content'] == 'whole answer'

    async def test_stream_failure_after_text_ends_interrupted_with_one_output(self):
        class Breaks(FakeLLM):
            async def stream(self, *a, **k):
                yield {'content': 'partial '}
                raise RuntimeError('connection reset')

        events = await run_chat(make_service(Breaks([])), request(mode='ask'))
        assert len(terminal(events)) == 1
        assert events[-1]['event_type'] == 'output'
        assert events[-1]['content'].startswith('partial')
        assert 'interrupted' in events[-1]['content']

    async def test_failure_everywhere_is_one_error(self):
        class Dead(FakeLLM):
            async def stream(self, *a, **k):
                raise RuntimeError('down')
                yield  # pragma: no cover

            async def generate(self, *a, **k):
                raise RuntimeError('down')

        events = await run_chat(make_service(Dead([])), request(mode='ask'))
        assert len(terminal(events)) == 1
        assert events[-1]['event_type'] == 'error'

    async def test_empty_answer_is_an_error(self):
        events = await run_chat(make_service(FakeLLM([text('')])), request(mode='ask'))
        assert [e['event_type'] for e in terminal(events)] == ['error']

    async def test_timeout_mid_stream_keeps_the_text_and_says_why(self):
        class Slow(FakeLLM):
            async def stream(self, *a, **k):
                yield {'content': 'started '}
                await asyncio.sleep(5)
                yield {'content': 'never'}

        service = make_service(Slow([]), limits=AssistantLimits(timeout_seconds=0.2))
        events = await run_chat(service, request(mode='ask'))
        assert len(terminal(events)) == 1
        assert events[-1]['event_type'] == 'output'
        assert events[-1]['content'].startswith('started')
        assert 'took too long' in events[-1]['content']

    async def test_audit_record_has_intent_and_streamed_token_estimate(self, caplog):
        llm = FakeLLM([text('ASK'), text('x' * 400)])
        with caplog.at_level(logging.INFO):
            await run_chat(make_service(llm), request())
        records = [
            r.getMessage()
            for r in caplog.records
            if 'workflow_assistant_audit' in r.getMessage()
        ]
        assert len(records) == 1
        audit = json.loads(records[0].split(' ', 1)[1])
        assert audit['intent'] == 'ask'
        assert audit['outcome'] == 'completed'
        assert audit['rounds'] == 1
        # classifier estimate plus the streamed answer: chars/4 of prompt and text
        assert audit['tokens'] >= 100


class TestEditPath:
    async def test_mode_edit_skips_the_classifier_and_uses_generate(self):
        llm = FakeLLM([text('Which agent?')])
        events = await run_chat(make_service(llm), request(mode='edit'))
        assert len(llm.calls) == 1
        assert llm.calls[0]['functions']  # the tool schemas
        assert llm.stream_calls == []
        assert events[-1]['content'] == 'Which agent?'

    async def test_auto_classifies_edit_then_streams_only_the_reply_round(self):
        llm = FakeLLM(
            [
                text('EDIT'),
                call('list_resources', {}, 'Looking.'),
                call('propose_workflow_yaml', {'yaml': 'second: 2'}),
                text('Added it.'),
            ],
            chunk_size=3,
        )
        events = await run_chat(make_service(llm), request(message='add an editor'))
        # classifier + two tool rounds went through generate
        assert len(llm.calls) == 3
        assert llm.calls[0]['functions'] is None
        assert llm.calls[1]['functions'] and llm.calls[2]['functions']
        # exactly one streamed call, the reply, and it carries no functions
        assert len(llm.stream_calls) == 1
        assert llm.stream_calls[0]['functions'] is None
        assert types(events)[-2:] == ['proposal', 'output']
        assert events[-1]['content'] == 'Looking.\n\nAdded it.'
        assert ''.join(deltas(events)).endswith('Added it.')
        assert len(terminal(events)) == 1

    async def test_streamed_reply_gets_the_accepted_proposal_not_tool_messages(self):
        llm = FakeLLM(
            [
                call(
                    'propose_workflow_yaml',
                    {'yaml': 'second: 2', 'notes': ['Used the main model.']},
                ),
                text('Done.'),
            ]
        )
        await run_chat(make_service(llm), request(mode='edit', message='add an editor'))
        sent = llm.stream_calls[0]['messages']
        assert all(m['role'] in ('system', 'user', 'assistant') for m in sent)
        assert sent[-1]['role'] == 'user'
        assert sent[-1]['content'].startswith('add an editor')
        assert 'second: 2' in sent[-1]['content']
        assert 'UNTRUSTED ACCEPTED_PROPOSAL' in sent[-1]['content']
        assert 'Used the main model.' in sent[-1]['content']

    async def test_unclear_classification_runs_the_tool_loop(self):
        llm = FakeLLM([text('hmm not sure'), text('Which agent do you mean?')])
        events = await run_chat(make_service(llm), request(message='tweak it'))
        assert len(llm.calls) == 2 and llm.stream_calls == []
        assert events[-1]['content'] == 'Which agent do you mean?'

    async def test_classifier_exception_runs_the_tool_loop(self):
        class FirstCallFails(FakeLLM):
            async def generate(self, messages, functions=None, output_schema=None):
                if functions is None:
                    raise RuntimeError('classifier down')
                return await super().generate(messages, functions, output_schema)

        llm = FirstCallFails([text('Which agent?')])
        events = await run_chat(make_service(llm), request(message='tweak it'))
        assert events[-1]['content'] == 'Which agent?'

    async def test_stream_failure_in_the_reply_round_degrades_to_generate(self):
        class NoStream(FakeLLM):
            async def stream(self, *a, **k):
                raise RuntimeError('provider cannot stream')
                yield  # pragma: no cover

        llm = NoStream(
            [call('propose_workflow_yaml', {'yaml': 'second: 2'}), text('Done.')]
        )
        events = await run_chat(
            make_service(llm), request(mode='edit', message='add one')
        )
        assert types(events)[-2:] == ['proposal', 'output']
        assert deltas(events) == ['Done.']
        assert events[-1]['content'] == 'Done.'


class ControlChunkLLM(FakeLLM):
    def __init__(self, chunks):
        super().__init__([])
        self.chunks = chunks

    async def stream(self, *a, **k):
        for chunk in self.chunks:
            yield chunk


class TestGuardrails:
    async def test_control_chunk_becomes_retract_and_is_never_text(self):
        llm = ControlChunkLLM(
            [
                {'content': 'unsafe '},
                {
                    GUARDRAIL_CONTROL_KEY: {
                        'action': 'retract',
                        'replacement': 'safe',
                        'message': 'redacted',
                    }
                },
                {'content': ' tail'},
            ]
        )
        events = await run_chat(make_service(llm), request(mode='ask'))
        retracts = [e for e in events if e['event_type'] == 'retract']
        assert len(retracts) == 1
        assert retracts[0]['content'] == 'safe'
        assert retracts[0]['reason'] == 'redacted'
        assert 'guardrail' not in json.dumps(deltas(events))
        assert events[-1]['event_type'] == 'output'
        assert events[-1]['content'] == 'safe tail'
        assert types(events).index('retract') > types(events).index('content_delta')

    async def test_retract_discards_text_still_waiting_to_be_sent(self):
        llm = ControlChunkLLM(
            [
                {'content': 'first'},
                {'content': ' held back'},
                {GUARDRAIL_CONTROL_KEY: {'action': 'retract', 'replacement': ''}},
            ]
        )
        events = await run_chat(make_service(llm), request(mode='ask'))
        assert deltas(events) == ['first']
        assert [e['event_type'] for e in terminal(events)] == [
            'error'
        ]  # nothing left to show

    async def test_unknown_control_action_is_ignored(self):
        llm = ControlChunkLLM(
            [
                {'content': 'a'},
                {GUARDRAIL_CONTROL_KEY: {'action': 'something-new'}},
                {'content': 'b'},
            ]
        )
        events = await run_chat(make_service(llm), request(mode='ask'))
        assert 'retract' not in types(events)
        assert events[-1]['content'] == 'ab'

    async def test_block_mid_stream_retracts_then_errors_once(self):
        class Blocks(FakeLLM):
            async def stream(self, *a, **k):
                yield {'content': 'shown '}
                raise GuardrailBlocked('blocked', retract=True)

        events = await run_chat(make_service(Blocks([])), request(mode='ask'))
        assert types(events)[-2:] == ['retract', 'error']
        assert events[-2]['content'] == ''
        assert len(terminal(events)) == 1

    async def test_block_without_text_shown_is_just_an_error(self):
        class Blocks(FakeLLM):
            async def stream(self, *a, **k):
                raise GuardrailBlocked('blocked')
                yield  # pragma: no cover

        events = await run_chat(make_service(Blocks([])), request(mode='ask'))
        assert 'retract' not in types(events)
        assert events[-1]['event_type'] == 'error'


class HangingLLM(FakeLLM):
    """Yields one chunk, then waits forever; records whether its stream was closed."""

    def __init__(self):
        super().__init__([])
        self.closed = asyncio.Event()
        self.started = asyncio.Event()

    async def stream(self, *a, **k):
        try:
            yield {'content': 'hello'}
            self.started.set()
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class TestCancellation:
    async def test_cancelling_the_consumer_closes_the_provider_stream(self):
        llm = HangingLLM()
        reply = StreamedReply()
        seen = []

        async def consume():
            async for event in stream_reply(llm, [], reply, deadline=deadline_in()):
                seen.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(llm.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert llm.closed.is_set()
        assert [e['event_type'] for e in seen] == [
            'content_delta'
        ]  # nothing after the cancel

    async def test_closing_the_generator_closes_the_provider_stream(self):
        llm = HangingLLM()
        gen = stream_reply(llm, [], StreamedReply(), deadline=deadline_in())
        first = await gen.__anext__()
        assert first['content'] == 'hello'
        await gen.aclose()
        assert llm.closed.is_set()

    async def test_service_cancel_closes_the_stream_and_audits_without_a_terminal_event(
        self, caplog
    ):
        llm = HangingLLM()
        service = make_service(llm)
        prepared = await service.prepare(request(mode='ask'), user_id='u')
        events = []

        async def consume():
            async for event in service.chat(prepared):
                events.append(event)

        with caplog.at_level(logging.INFO):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(llm.started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert llm.closed.is_set()
        assert terminal(events) == []
        audits = [
            r.getMessage()
            for r in caplog.records
            if 'workflow_assistant_audit' in r.getMessage()
        ]
        assert len(audits) == 1 and '"outcome": "cancelled"' in audits[0]
        assert not [r for r in caplog.records if r.exc_info]  # no stack trace logged

    async def test_deadline_closes_the_provider_stream(self):
        llm = HangingLLM()
        reply = StreamedReply()
        events = [
            e async for e in stream_reply(llm, [], reply, deadline=deadline_in(0.1))
        ]
        assert reply.timed_out and reply.text == 'hello'
        assert llm.closed.is_set()
        assert [e['content'] for e in events] == ['hello']


class TestStreamReplyDelivery:
    async def test_chunks_are_coalesced_and_flushed_at_the_end(self):
        class Tokens(FakeLLM):
            async def stream(self, *a, **k):
                for _ in range(50):
                    yield {'content': 'ab'}

        reply = StreamedReply()
        events = [
            e async for e in stream_reply(Tokens([]), [], reply, deadline=deadline_in())
        ]
        sent = [e['content'] for e in events]
        assert len(sent) < 50
        assert ''.join(sent) == 'ab' * 50 == reply.text
        assert reply.emitted

    async def test_quiet_provider_does_not_hold_back_due_text(self):
        class Pauses(FakeLLM):
            async def stream(self, *a, **k):
                yield {'content': 'a'}
                yield {'content': 'b'}
                await asyncio.sleep(0.3)
                yield {'content': 'c'}

        arrivals = []
        started = time.monotonic()
        async for event in stream_reply(
            Pauses([]), [], StreamedReply(), deadline=deadline_in()
        ):
            arrivals.append((event['content'], time.monotonic() - started))
        assert [c for c, _ in arrivals] == ['a', 'b', 'c']
        assert arrivals[1][1] < 0.2  # 'b' left at its flush time, not with 'c'
