"""The bounded loop and the service around it, driven by mocked LLM transcripts."""

import json
import logging

import pytest

from agents_module.models.workflow_assistant_schemas import AssistantChatRequest
from agents_module.services.workflow_assistant.limits import AssistantLimits
from agents_module.services.workflow_assistant.repair_loop import (
    COMPLETED,
    MAX_ROUNDS,
    REPAIR_EXHAUSTED,
    REPEATED_ERRORS,
    TIMEOUT,
    TOKEN_BUDGET,
    LoopStats,
    RepairTracker,
    response_tokens,
    response_truncated,
    run_loop,
)
from agents_module.services.workflow_assistant.service import (
    AssistantRequestError,
    WorkflowAssistantService,
    build_default_registry,
)
from agents_module.services.workflow_assistant.tools.base import Proposal, ToolContext

from workflow_assistant_fakes import (
    FakeLLM,
    FakeResources,
    CANVAS_MARKER,
    FakeWorkflowCrud,
    MODEL_ID,
    call,
    err,
    assistant_kwargs,
    fake_agent_inference_service,
    llm_config_dict,
    text,
    workflow_yaml,
)


def propose(yaml_text, content=''):
    return call('propose_workflow_yaml', {'yaml': yaml_text}, content)


def make_ctx(crud):
    return ToolContext(
        resources=FakeResources(), workflow_crud_service=crud, max_yaml_chars=10000
    )


async def collect(llm, crud, limits=None, nonce='n'):
    stats = LoopStats()
    ctx = make_ctx(crud)
    events = [
        e
        async for e in run_loop(
            llm,
            [{'role': 'user', 'content': 'hi'}],
            build_default_registry(),
            ctx,
            limits or AssistantLimits(),
            nonce,
            stats,
        )
    ]
    return events, stats, ctx


def types(events):
    return [e['event_type'] for e in events]


class TestRepairTracker:
    def proposal(self, *errors):
        return Proposal(yaml='y', valid=not errors, errors=list(errors))

    def test_valid_never_stops(self):
        assert RepairTracker(3).record(self.proposal()) is None

    def test_stops_at_max_validations(self):
        tracker = RepairTracker(3)
        assert tracker.record(self.proposal(err('a', '1'))) is None
        assert tracker.record(self.proposal(err('a', '2'))) is None
        assert tracker.record(self.proposal(err('a', '3'))) == REPAIR_EXHAUSTED

    def test_same_errors_twice_stops_early(self):
        tracker = RepairTracker(3)
        assert tracker.record(self.proposal(err('a', '1'), err('b', '2'))) is None
        assert (
            tracker.record(self.proposal(err('b', '2'), err('a', '1')))
            == REPEATED_ERRORS
        )


class TestRepairTrackerIntroducedOnly:
    def proposal(self, errors, introduced):
        return Proposal(
            yaml='y',
            valid=not errors,
            errors=list(errors),
            introduced_errors=list(introduced),
        )

    def test_existing_errors_never_exhaust_or_repeat(self):
        tracker = RepairTracker(2)
        old = err('a', 'old')
        for _ in range(5):
            assert tracker.record(self.proposal([old], [])) is None

    def test_repeat_is_judged_on_introduced_errors_only(self):
        tracker = RepairTracker(5)
        old, new = err('a', 'old'), err('b', 'new')
        assert tracker.record(self.proposal([old, new], [new])) is None
        assert tracker.record(self.proposal([old], [])) is None
        assert tracker.record(self.proposal([old, new], [new])) is None  # not a repeat

    def test_cap_is_exact(self):
        tracker = RepairTracker(2)
        assert tracker.record(self.proposal([err('a', '1')], [err('a', '1')])) is None
        assert (
            tracker.record(self.proposal([err('a', '2')], [err('a', '2')]))
            == REPAIR_EXHAUSTED
        )
        assert tracker.validations == 2

    def test_failed_attempts_count(self):
        tracker = RepairTracker(2)
        assert tracker.record_failure() is None
        assert tracker.record_failure() == REPAIR_EXHAUSTED


class TestResponseTokens:
    def test_reads_provider_usage(self):
        assert response_tokens({'usage': {'total_tokens': 42}}, [], '') == 42
        assert (
            response_tokens({'usage': {'input_tokens': 5, 'output_tokens': 7}}, [], '')
            == 12
        )

    def test_estimates_without_usage(self):
        assert response_tokens({}, [{'role': 'user', 'content': 'x' * 400}], '') > 90


class TestResponseTruncated:
    def test_reads_finish_or_stop_reason_from_dicts_and_objects(self):
        from types import SimpleNamespace

        assert response_truncated({'finish_reason': 'length'}) is True
        assert response_truncated({'stop_reason': 'max_tokens'}) is True
        assert response_truncated(SimpleNamespace(finish_reason='LENGTH')) is True
        assert (
            response_truncated(
                SimpleNamespace(finish_reason=SimpleNamespace(name='MAX_TOKENS'))
            )
            is True
        )
        assert response_truncated({'finish_reason': 'stop'}) is False
        assert response_truncated({'text': 'x'}) is False
        assert response_truncated('plain string') is False


class TestLoop:
    async def test_plain_reply_completes_without_tools(self):
        events, stats, ctx = await collect(FakeLLM([text('Hello')]), FakeWorkflowCrud())
        assert types(events) == ['content_delta']
        assert stats.stop_reason == COMPLETED
        assert stats.rounds == 1
        assert ctx.last_proposal is None

    async def test_tool_round_trip_orders_events_and_feeds_result_back(self):
        llm = FakeLLM(
            [
                call('list_resources', {}, 'Checking.'),
                propose(workflow_yaml()),
                text('Done.'),
            ]
        )
        events, stats, ctx = await collect(llm, FakeWorkflowCrud())
        assert types(events) == [
            'content_delta',
            'tool_called',
            'tool_result',
            'tool_called',
            'tool_result',
            'content_delta',
        ]
        called = [e for e in events if e['event_type'] == 'tool_called']
        assert [e['label'] for e in called] == [
            'Reading app resources',
            'Validating workflow',
        ]
        results = [e for e in events if e['event_type'] == 'tool_result']
        assert [e['ok'] for e in results] == [True, True]
        assert called[0]['tool_call_id'] == results[0]['tool_call_id']
        assert stats.tools_called == ['list_resources', 'propose_workflow_yaml']
        # tool output reaches the model fenced as untrusted data
        last_messages = llm.calls[-1]['messages']
        function_msgs = [m for m in last_messages if m['role'] == 'function']
        assert function_msgs[-1]['content'].startswith('<<<UNTRUSTED TOOL_OUTPUT')
        assert ctx.last_proposal.valid is True

    async def test_repair_then_success(self):
        crud = FakeWorkflowCrud([[err('arium.x', 'bad')], []])
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2'), text('Fixed.')])
        events, stats, ctx = await collect(llm, crud)
        assert stats.stop_reason == COMPLETED
        assert [p.valid for p in ctx.proposals] == [False, True]
        assert ctx.last_proposal.yaml == 'v2: 2'

    async def test_exhausted_repair_rounds_leave_an_invalid_last_proposal(self):
        crud = FakeWorkflowCrud(
            [[err('a', 'one')], [err('a', 'two')], [err('a', 'three')]]
        )
        llm = FakeLLM(
            [propose('v1: 1'), propose('v2: 2'), propose('v3: 3'), text('no')]
        )
        events, stats, ctx = await collect(llm, crud)
        assert stats.stop_reason == REPAIR_EXHAUSTED
        assert len(ctx.proposals) == 3
        assert ctx.last_proposal.valid is False
        assert ctx.last_proposal.yaml == 'v3: 3'
        assert len(llm.script) == 1  # the fourth round never ran

    async def test_repeated_error_set_exits_early(self):
        crud = FakeWorkflowCrud([[err('a', 'same')]])
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2'), propose('v3: 3')])
        _, stats, ctx = await collect(llm, crud)
        assert stats.stop_reason == REPEATED_ERRORS
        assert len(ctx.proposals) == 2
        assert len(llm.script) == 1

    async def collect_with_baseline(self, llm, crud, baseline, limits=None):
        stats = LoopStats()
        ctx = make_ctx(crud)
        ctx.baseline_errors = list(baseline)
        events = [
            e
            async for e in run_loop(
                llm,
                [{'role': 'user', 'content': 'hi'}],
                build_default_registry(),
                ctx,
                limits or AssistantLimits(),
                'n',
                stats,
            )
        ]
        return events, stats, ctx

    async def test_loop_ends_after_a_proposal_with_no_introduced_errors(self):
        old = err('a', 'old')
        crud = FakeWorkflowCrud([[old]])
        llm = FakeLLM([propose('v1: 1'), text('Added it.'), propose('v2: 2')])
        events, stats, ctx = await self.collect_with_baseline(llm, crud, [old])
        assert stats.stop_reason == COMPLETED
        assert len(ctx.proposals) == 1
        assert ctx.last_proposal.valid is False  # the old error is still there
        assert ctx.last_proposal.introduced == []
        assert stats.texts == ['Added it.']
        assert len(llm.script) == 1  # no round chasing existing issues

    async def test_tool_call_in_the_reply_round_is_ignored(self):
        crud = FakeWorkflowCrud([[]])
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2', 'Tweaking.')])
        _, stats, ctx = await collect(llm, crud)
        assert stats.stop_reason == COMPLETED
        assert len(ctx.proposals) == 1
        assert stats.tools_called == ['propose_workflow_yaml']

    async def test_existing_errors_do_not_burn_repair_rounds(self):
        old = err('a', 'old')
        crud = FakeWorkflowCrud([[old, err('b', 'new1')], [old]])
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2'), text('Done.')])
        _, stats, ctx = await self.collect_with_baseline(llm, crud, [old])
        assert stats.stop_reason == COMPLETED
        assert [len(p.introduced) for p in ctx.proposals] == [1, 0]

    async def test_repair_cap_counts_introduced_errors_exactly(self):
        old = err('a', 'old')
        crud = FakeWorkflowCrud(
            [[old, err('n', '1')], [old, err('n', '2')], [old, err('n', '3')]]
        )
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2'), propose('v3: 3'), text('x')])
        _, stats, ctx = await self.collect_with_baseline(
            llm, crud, [old], AssistantLimits(max_repair_rounds=2)
        )
        assert stats.stop_reason == REPAIR_EXHAUSTED
        assert len(ctx.proposals) == 2
        assert len(llm.script) == 2

    async def test_failed_proposal_calls_count_against_the_cap(self):
        llm = FakeLLM([call('propose_workflow_yaml', '{bad') for _ in range(5)])
        _, stats, ctx = await collect(llm, FakeWorkflowCrud())
        assert stats.stop_reason == REPAIR_EXHAUSTED
        assert stats.rounds == 3  # max_repair_rounds, not max_rounds
        assert ctx.proposals == []

    async def test_valid_proposals_are_not_chased_to_the_round_cap(self):
        # the old loop let a valid proposal return without counting, so a model
        # that kept re-validating ran until max_rounds
        llm = FakeLLM([propose(f'v{i}: 1') for i in range(8)])
        _, stats, ctx = await collect(llm, FakeWorkflowCrud())
        assert stats.stop_reason == COMPLETED
        assert len(ctx.proposals) == 1

    async def test_max_rounds_bound(self):
        llm = FakeLLM([call('list_resources', {}) for _ in range(5)])
        _, stats, _ = await collect(
            llm, FakeWorkflowCrud(), AssistantLimits(max_rounds=2)
        )
        assert stats.stop_reason == MAX_ROUNDS
        assert stats.rounds == 2

    async def test_token_budget_bound(self):
        llm = FakeLLM(
            [call('list_resources', {}, usage={'total_tokens': 500}), text('late')]
        )
        _, stats, _ = await collect(
            llm, FakeWorkflowCrud(), AssistantLimits(max_total_tokens=100)
        )
        assert stats.stop_reason == TOKEN_BUDGET
        assert stats.rounds == 1

    async def test_wall_clock_timeout(self):
        llm = FakeLLM([text('slow')], delay=0.5)
        _, stats, _ = await collect(
            llm, FakeWorkflowCrud(), AssistantLimits(timeout_seconds=0.05)
        )
        assert stats.stop_reason == TIMEOUT

    async def test_unknown_tool_is_reported_and_loop_continues(self):
        llm = FakeLLM([call('rm_rf', {}), text('sorry')])
        events, stats, _ = await collect(llm, FakeWorkflowCrud())
        assert types(events) == ['tool_called', 'tool_failed', 'content_delta']
        assert 'Unknown tool' in events[1]['error']
        assert stats.stop_reason == COMPLETED

    async def test_malformed_arguments_are_a_tool_failure(self):
        llm = FakeLLM([call('propose_workflow_yaml', '{not json'), text('ok')])
        events, _, _ = await collect(llm, FakeWorkflowCrud())
        assert types(events)[:2] == ['tool_called', 'tool_failed']

    async def test_truncated_call_tells_the_model_to_shorten_and_counts_as_a_round(
        self,
    ):
        llm = FakeLLM(
            [call('propose_workflow_yaml', '{"yaml": "arium: [cut'), text('ok')]
        )
        events, stats, ctx = await collect(llm, FakeWorkflowCrud())
        assert 'cut off' in events[1]['error'] and 'shorter' in events[1]['error']
        assert stats.last_error == events[1]['error']
        fed_back = llm.calls[1]['messages'][-1]['content']
        assert 'cut off' in fed_back
        assert ctx.proposals == []

    async def test_truncated_response_is_recorded_on_stats(self):
        llm = FakeLLM([text('partial', finish_reason='length')])
        _, stats, _ = await collect(llm, FakeWorkflowCrud())
        assert stats.truncated is True

    async def test_a_crashed_proposal_call_keeps_the_yaml_as_an_invalid_proposal(self):
        class Crashing(FakeWorkflowCrud):
            async def validate_yaml(self, yaml_content):
                raise RuntimeError('boom')

        llm = FakeLLM([propose(workflow_yaml()), text('x')])
        events, stats, ctx = await collect(llm, Crashing())
        assert 'tool_failed' in types(events)
        assert len(ctx.proposals) == 1
        assert ctx.last_proposal.valid is False
        assert ctx.last_proposal.yaml == workflow_yaml()


def request(**overrides):
    base = {
        'message': 'add an editor',
        'current_yaml': workflow_yaml() + CANVAS_MARKER,
        'llm_inference_config_id': MODEL_ID,
        # these tests script the tool loop; routing has its own tests
        'mode': 'edit',
    }
    base.update(overrides)
    return AssistantChatRequest(**base)


def make_service(llm, config=None, crud=None, limits=None):
    return WorkflowAssistantService(
        **assistant_kwargs(
            fake_agent_inference_service(
                llm, llm_config_dict() if config is None else config
            )
        ),
        workflow_crud_service=crud or FakeWorkflowCrud(),
        limits=limits or AssistantLimits(),
    )


async def run_chat(service, req):
    prepared = await service.prepare(req, user_id='user-1')
    return [e async for e in service.chat(prepared)]


class TestServiceStream:
    async def test_event_order_and_final_proposal(self):
        llm = FakeLLM(
            [
                call('list_resources', {}, 'Looking.'),
                propose(workflow_yaml()),
                text('Added it.'),
            ]
        )
        events = await run_chat(make_service(llm), request())
        assert types(events) == [
            'assistant_started',
            'content_delta',
            'tool_called',
            'tool_result',
            'tool_called',
            'tool_result',
            'content_delta',
            'proposal',
            'output',
        ]
        proposal = events[-2]
        assert proposal['valid'] is True
        assert proposal['errors'] == []
        assert proposal['yaml'] == workflow_yaml()
        assert events[-1]['content'] == 'Looking.\n\nAdded it.'
        assert all(isinstance(e['timestamp'], float) for e in events)

    async def test_last_proposal_is_the_one_emitted(self):
        crud = FakeWorkflowCrud([[err('p', 'bad')], []])
        llm = FakeLLM([propose('first: 1'), propose('second: 2'), text('done')])
        events = await run_chat(make_service(llm, crud=crud), request())
        proposals = [e for e in events if e['event_type'] == 'proposal']
        assert len(proposals) == 1
        assert proposals[0]['yaml'] == 'second: 2'
        assert proposals[0]['valid'] is True

    async def test_exhausted_repairs_emit_invalid_proposal_then_output(self):
        crud = FakeWorkflowCrud([[err('a', '1')], [err('a', '2')], [err('a', '3')]])
        llm = FakeLLM([propose('v1: 1'), propose('v2: 2'), propose('v3: 3')])
        events = await run_chat(make_service(llm, crud=crud), request())
        assert types(events)[-2:] == ['proposal', 'output']
        proposal = events[-2]
        assert proposal['valid'] is False
        assert proposal['yaml'] == 'v3: 3'
        assert proposal['errors'] == [{'path': 'a', 'message': '3'}]
        assert (
            'I made the change but 1 validation issue remains' in events[-1]['content']
        )
        assert 'will not save' in events[-1]['content']

    async def test_limit_without_proposal_ends_with_error(self):
        llm = FakeLLM([call('list_resources', {}) for _ in range(3)])
        service = make_service(llm, limits=AssistantLimits(max_rounds=2))
        events = await run_chat(service, request())
        assert types(events)[0] == 'assistant_started'
        assert events[-1]['event_type'] == 'error'
        assert 'step limit' in events[-1]['message']
        assert 'could not make a change' in events[-1]['message']

    async def test_unparseable_calls_end_with_a_plain_reason_including_the_last_error(
        self,
    ):
        llm = FakeLLM([call('propose_workflow_yaml', '{bad') for _ in range(3)])
        events = await run_chat(make_service(llm), request())
        assert events[-1]['event_type'] == 'error'
        message = events[-1]['message']
        assert 'could not make a change' in message
        assert 'cut off' in message  # the last error, in plain words
        assert 'No workflow change was proposed' not in message

    async def test_a_crashed_validation_still_emits_the_proposal_with_a_plain_message(
        self,
    ):
        class Crashing(FakeWorkflowCrud):
            async def validate_yaml(self, yaml_content):
                if yaml_content.endswith(CANVAS_MARKER):
                    return []
                raise RuntimeError('boom')

        llm = FakeLLM([propose(workflow_yaml()) for _ in range(3)])
        events = await run_chat(make_service(llm, crud=Crashing()), request())
        assert types(events)[-2:] == ['proposal', 'output']
        assert events[-2]['yaml'] == workflow_yaml()
        assert events[-2]['valid'] is False
        assert (
            'I made the change but 1 validation issue remains' in events[-1]['content']
        )

    async def test_model_stopping_with_introduced_errors_gets_the_plain_message(self):
        crud = FakeWorkflowCrud([[err('b', 'new thing'), err('c', 'newer')]])
        llm = FakeLLM([propose('second: 2'), text('I give up.')])
        events = await run_chat(make_service(llm, crud=crud), request())
        assert types(events)[-2:] == ['proposal', 'output']
        content = events[-1]['content']
        assert content.startswith('I give up.')
        assert 'I made the change but 2 validation issues remain' in content

    async def test_truncated_reply_flags_the_output_event(self):
        llm = FakeLLM([text('partial answer', finish_reason='length')])
        events = await run_chat(make_service(llm), request())
        assert events[-1]['event_type'] == 'output'
        assert events[-1]['truncated'] is True
        assert 'cut off' in events[-1]['content']

    async def test_normal_output_has_no_truncated_key(self):
        events = await run_chat(make_service(FakeLLM([text('fine')])), request())
        assert 'truncated' not in events[-1]

    async def test_introduced_error_result_is_neutral_feedback_on_the_wire(self):
        crud = FakeWorkflowCrud([[err('b', 'new thing')], []])
        llm = FakeLLM([propose('first: 1'), propose('second: 2'), text('done')])
        events = await run_chat(make_service(llm, crud=crud), request())
        results = [e for e in events if e['event_type'] == 'tool_result']
        assert results[0]['ok'] is False
        assert results[0]['summary'] == 'Found 1 issue(s), refining'
        assert 'tool_failed' not in types(events)

    async def test_prompt_names_every_config_and_marks_the_selected_one(self):
        llm = FakeLLM([text('hi')])
        await run_chat(make_service(llm), request())
        system = llm.calls[0]['messages'][0]['content']
        assert (
            f'- id: {MODEL_ID} | name: Main model | provider type: openai | model: gpt-4.1-mini [SELECTED]'
            in system
        )

    async def test_baseline_is_validated_once_and_reaches_the_prompt(self):
        old = err('a', 'old thing')
        crud = FakeWorkflowCrud([[old]], baseline=[old])
        llm = FakeLLM([propose('second: 2'), text('Added it.')])
        await run_chat(make_service(llm, crud=crud), request())
        assert crud.baseline_calls == 1
        assert crud.seen == ['second: 2']
        system = llm.calls[0]['messages'][0]['content']
        assert 'UNTRUSTED EXISTING_ISSUES' in system
        assert 'a: old thing' in system

    async def test_existing_only_proposal_is_not_blocking_on_the_wire(self):
        old = err('a', 'old thing')
        crud = FakeWorkflowCrud([[old]], baseline=[old])
        llm = FakeLLM([propose('second: 2'), text('')])
        events = await run_chat(make_service(llm, crud=crud), request())
        proposal = events[-2]
        assert proposal['event_type'] == 'proposal'
        assert proposal['valid'] is False
        assert proposal['errors'] == [{'path': 'a', 'message': 'old thing'}]
        assert proposal['introduced_errors'] == []
        assert (
            'Proposal ready; 1 existing validation issue still remains'
            in events[-1]['content']
        )

    async def test_notes_ride_on_the_proposal_event(self):
        llm = FakeLLM(
            [
                call(
                    'propose_workflow_yaml',
                    {
                        'yaml': 'second: 2',
                        'notes': ['Uses the same model as the writer.'],
                    },
                ),
                text('Done.'),
            ]
        )
        events = await run_chat(make_service(llm), request())
        proposal = [e for e in events if e['event_type'] == 'proposal'][0]
        assert proposal['notes'] == ['Uses the same model as the writer.']

    async def test_proposal_event_has_no_notes_key_when_there_are_none(self):
        llm = FakeLLM([propose('second: 2'), text('Done.')])
        events = await run_chat(make_service(llm), request())
        proposal = [e for e in events if e['event_type'] == 'proposal'][0]
        assert 'notes' not in proposal

    async def test_introduced_errors_are_listed_on_the_wire(self):
        old, new = err('a', 'old thing'), err('b', 'new thing')
        crud = FakeWorkflowCrud([[old, new]], baseline=[old])
        llm = FakeLLM([propose('second: 2'), propose('third: 3'), propose('x: 4')])
        events = await run_chat(
            make_service(llm, crud=crud, limits=AssistantLimits(max_repair_rounds=1)),
            request(),
        )
        proposal = [e for e in events if e['event_type'] == 'proposal'][0]
        assert proposal['introduced_errors'] == [{'path': 'b', 'message': 'new thing'}]
        assert len(proposal['errors']) == 2

    async def test_budget_cap_after_a_clean_proposal_still_ends_normally(self):
        old = err('a', 'old thing')
        crud = FakeWorkflowCrud([[old]], baseline=[old])
        llm = FakeLLM(
            [
                call(
                    'propose_workflow_yaml',
                    {'yaml': 'second: 2'},
                    usage={'total_tokens': 500},
                )
            ]
        )
        service = make_service(
            llm, crud=crud, limits=AssistantLimits(max_total_tokens=100)
        )
        events = await run_chat(service, request())
        assert types(events)[-2:] == ['proposal', 'output']
        assert 'stopped' not in events[-1]['content']
        assert 'Proposal ready; 1 existing validation issue' in events[-1]['content']

    async def test_round_cap_with_introduced_errors_keeps_proposal_and_says_so(self):
        crud = FakeWorkflowCrud([[err('b', 'new thing')]])
        llm = FakeLLM([propose('second: 2'), text('x')])
        service = make_service(llm, crud=crud, limits=AssistantLimits(max_rounds=1))
        events = await run_chat(service, request())
        assert types(events)[-2:] == ['proposal', 'output']
        assert 'step limit' in events[-1]['content']

    async def test_llm_crash_ends_with_generic_error(self):
        class Crashing(FakeLLM):
            async def generate(self, *a, **k):
                raise RuntimeError('sk-secret-key leaked here')

        events = await run_chat(make_service(Crashing([])), request())
        assert events[-1]['event_type'] == 'error'
        assert 'sk-secret' not in json.dumps(events)

    async def test_empty_reply_is_an_error(self):
        events = await run_chat(make_service(FakeLLM([text('')])), request())
        assert events[-1]['event_type'] == 'error'

    async def test_history_is_capped_and_untrusted_canvas_is_fenced(self):
        llm = FakeLLM([text('ok')])
        history = [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'turn {i}'}
            for i in range(10)
        ]
        service = make_service(llm, limits=AssistantLimits(max_history_messages=4))
        await run_chat(service, request(history=history))
        sent = llm.calls[0]['messages']
        assert sent[0]['role'] == 'system'
        assert 'UNTRUSTED CANVAS_YAML' in sent[0]['content']
        assert [m['content'] for m in sent[1:-1]] == [f'turn {i}' for i in range(6, 10)]
        assert sent[-1] == {'role': 'user', 'content': 'add an editor'}

    async def test_audit_record_has_no_yaml_or_prompt(self, caplog):
        llm = FakeLLM([propose(workflow_yaml()), text('done')])
        with caplog.at_level(logging.INFO):
            await run_chat(make_service(llm), request())
        records = [
            r.getMessage()
            for r in caplog.records
            if 'workflow_assistant_audit' in r.getMessage()
        ]
        assert len(records) == 1
        audit = json.loads(records[0].split(' ', 1)[1])
        assert audit['user_id'] == 'user-1'
        assert audit['llm_config_id'] == MODEL_ID
        assert audit['provider'] == 'openai'
        assert audit['tools_called'] == ['propose_workflow_yaml']
        assert audit['rounds'] == 2
        assert audit['outcome'] == 'completed'
        assert 'arium' not in records[0]
        assert 'add an editor' not in records[0]


class TestGuardrails:
    def service_with_engine(self, llm, engine):
        svc = fake_agent_inference_service(llm, llm_config_dict())
        svc.guardrails_engine = engine
        return WorkflowAssistantService(
            **assistant_kwargs(svc),
            workflow_crud_service=FakeWorkflowCrud(),
            limits=AssistantLimits(),
        )

    async def test_default_namespace_engine_and_user_reach_the_wrapper(
        self, monkeypatch
    ):
        from agents_module.services.workflow_assistant import service as service_module

        llm, engine, wrapped = FakeLLM([text('x')]), object(), object()
        calls = []

        def spy(*args):
            calls.append(args)
            return wrapped

        monkeypatch.setattr(service_module, 'guarded_llm', spy)
        prepared = await self.service_with_engine(llm, engine).prepare(
            request(), user_id='user-7'
        )
        assert calls == [(llm, engine, 'default', 'workflow_assistant', 'user-7')]
        assert service_module.GUARDRAIL_NAMESPACE == 'default'
        assert prepared.llm is wrapped

    async def test_missing_user_id_is_tolerated(self, monkeypatch):
        from agents_module.services.workflow_assistant import service as service_module

        calls = []
        monkeypatch.setattr(
            service_module, 'guarded_llm', lambda *args: calls.append(args) or args[0]
        )
        await self.service_with_engine(FakeLLM([text('x')]), object()).prepare(
            request()
        )
        assert calls[0][4] is None

    async def test_no_engine_leaves_the_llm_unwrapped(self):
        llm = FakeLLM([text('x')])
        prepared = await self.service_with_engine(llm, None).prepare(
            request(), user_id='user-7'
        )
        assert prepared.llm is llm


class TestHistorySchema:
    def test_a_long_chat_is_trimmed_not_rejected(self):
        from agents_module.models.workflow_assistant_schemas import (
            AssistantChatRequest,
            MAX_HISTORY_ITEMS,
        )

        history = [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'turn {i}'}
            for i in range(MAX_HISTORY_ITEMS * 3)
        ]
        req = AssistantChatRequest(
            message='hi', llm_inference_config_id=MODEL_ID, history=history
        )
        assert len(req.history) == MAX_HISTORY_ITEMS
        assert req.history[-1].content == f'turn {MAX_HISTORY_ITEMS * 3 - 1}'

    def test_per_message_cap_still_applies(self):
        from pydantic import ValidationError

        from agents_module.models.workflow_assistant_schemas import (
            AssistantChatRequest,
            MAX_CONTENT_CHARS,
        )

        with pytest.raises(ValidationError):
            AssistantChatRequest(
                message='hi',
                llm_inference_config_id=MODEL_ID,
                history=[{'role': 'user', 'content': 'x' * (MAX_CONTENT_CHARS + 1)}],
            )

    async def test_the_service_keeps_only_its_own_limit(self):
        llm = FakeLLM([text('ok')])
        history = [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'turn {i}'}
            for i in range(150)
        ]
        await run_chat(make_service(llm), request(history=history))
        sent = llm.calls[0]['messages']
        assert len(sent) == 1 + AssistantLimits().max_history_messages + 1


class TestOutputTokenFloor:
    async def prepared_config(self, parameters, limits=None):
        seen = []
        llm = FakeLLM([text('x')])
        svc = fake_agent_inference_service(llm, llm_config_dict(parameters=parameters))
        svc.build_llm = lambda cfg: seen.append(cfg) or llm
        service = WorkflowAssistantService(
            **assistant_kwargs(svc),
            workflow_crud_service=FakeWorkflowCrud(),
            limits=limits or AssistantLimits(),
        )
        await service.prepare(request())
        return seen[0], svc

    async def test_missing_limit_gets_the_floor(self):
        cfg, _ = await self.prepared_config(None)
        assert cfg.parameters == {'max_tokens': 8192}

    async def test_lower_limit_under_any_spelling_is_raised_on_a_copy(self):
        stored = {'max_completion_tokens': 1024, 'temperature': 0.2}
        cfg, svc = await self.prepared_config(stored)
        assert cfg.parameters == {'temperature': 0.2, 'max_tokens': 8192}
        # the stored config is untouched
        assert stored == {'max_completion_tokens': 1024, 'temperature': 0.2}
        assert (
            svc.llm_inference_config_service.get_config.return_value['parameters']
            is stored
        )

    async def test_higher_limit_is_kept(self):
        cfg, _ = await self.prepared_config({'max_tokens': 16000})
        assert cfg.parameters == {'max_tokens': 16000}

    async def test_floor_follows_the_configured_limit(self):
        cfg, _ = await self.prepared_config(
            {'max_tokens': 100}, AssistantLimits(max_output_tokens=2000)
        )
        assert cfg.parameters == {'max_tokens': 2000}


class TestPrepareRejections:
    async def rejected(self, service, req=None):
        with pytest.raises(AssistantRequestError) as info:
            await service.prepare(req or request())
        return info.value

    async def test_missing_config(self):
        error = await self.rejected(make_service(FakeLLM([]), config={}))
        assert error.status_code == 404

    async def test_deleted_config(self):
        error = await self.rejected(
            make_service(FakeLLM([]), config=llm_config_dict(is_deleted=True))
        )
        assert error.status_code == 404

    async def test_non_llm_config(self):
        error = await self.rejected(
            make_service(FakeLLM([]), config=llm_config_dict(model_type='embedding'))
        )
        assert error.status_code == 400
        assert 'chat model' in error.message

    @pytest.mark.parametrize('provider', ['bedrock', 'rootflo', 'mystery'])
    async def test_unsupported_provider(self, provider):
        error = await self.rejected(
            make_service(FakeLLM([]), config=llm_config_dict(type=provider))
        )
        assert error.status_code == 400
        assert provider in error.message

    @pytest.mark.parametrize(
        'provider',
        ['openai', 'groq', 'azure_openai', 'anthropic', 'gemini', 'ollama', 'vllm'],
    )
    async def test_supported_providers_pass(self, provider):
        service = make_service(FakeLLM([]), config=llm_config_dict(type=provider))
        prepared = await service.prepare(request())
        assert prepared.provider == provider

    async def test_invalid_uuid(self):
        error = await self.rejected(
            make_service(FakeLLM([])), request(llm_inference_config_id='nope')
        )
        assert error.status_code == 400

    async def test_disabled(self):
        error = await self.rejected(
            make_service(FakeLLM([]), limits=AssistantLimits(enabled=False))
        )
        assert error.status_code == 503

    async def test_message_and_yaml_limits(self):
        service = make_service(
            FakeLLM([]), limits=AssistantLimits(max_message_chars=5, max_yaml_chars=50)
        )
        assert (
            await self.rejected(service, request(message='x' * 6))
        ).status_code == 400
        error = await self.rejected(
            service, request(message='ok', current_yaml='x' * 51)
        )
        assert error.status_code == 400
        assert 'too large' in error.message

    async def test_build_failure_is_masked(self):
        service = make_service(FakeLLM([]))

        def boom(cfg):
            raise ValueError('Unsupported LLM type: sk-secret')

        service.agent_inference_service.build_llm = boom
        error = await self.rejected(service)
        assert error.status_code == 400
        assert 'sk-secret' not in error.message
