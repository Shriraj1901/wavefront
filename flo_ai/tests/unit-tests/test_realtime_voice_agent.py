import asyncio

from flo_ai.agent.realtime_voice_agent import RealtimeVoiceAgent
from flo_ai.llm.base_realtime_llm import (
    RealtimeLLM,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from flo_ai.tool.base_tool import Tool


class FakeRealtimeLLM(RealtimeLLM):
    def __init__(self, scripted):
        super().__init__(model='fake')
        self.scripted = scripted
        self.connected_with = None
        self.tool_results = []
        self.audio = []
        self.closed = False

    async def connect(self, system_prompt, tools=None, config=None):
        self.connected_with = (system_prompt, tools, config)

    async def send_audio(self, audio):
        self.audio.append(audio)

    async def send_tool_result(self, call_id, result):
        self.tool_results.append((call_id, result))

    async def events(self):
        for event in self.scripted:
            yield event

    async def close(self):
        self.closed = True

    def format_tool_for_llm(self, tool):
        return {'name': tool.name}


def _tool(name, function, parameters):
    return Tool(name=name, description=name, function=function, parameters=parameters)


async def save_lead(name: str, user_id: str, session_id: str):
    return {'saved': name, 'user_id': user_id, 'session_id': session_id}


async def echo(text: str):
    return text


async def flexible(**kwargs):
    return kwargs


SAVE_LEAD = _tool(
    'save_lead',
    save_lead,
    {
        'name': {'type': 'string', 'description': 'Lead name'},
        'user_id': {'type': 'string', 'description': 'User'},
        'session_id': {'type': 'string', 'description': 'Session'},
    },
)
ECHO = _tool('echo', echo, {'text': {'type': 'string', 'description': 'Text'}})
CONTEXT = {'user_id': 'u-1', 'session_id': 's-1'}


async def _run(agent):
    return [event async for event in agent.events()]


async def test_session_context_merged_and_wins_over_model_args():
    call = ToolCallEvent(
        id='c1',
        name='save_lead',
        arguments={'name': 'Asha', 'user_id': 'someone-else'},
    )
    llm = FakeRealtimeLLM([call, TurnCompleteEvent()])
    agent = RealtimeVoiceAgent(llm, [SAVE_LEAD], 'prompt', CONTEXT)

    events = await _run(agent)

    expected = {'saved': 'Asha', 'user_id': 'u-1', 'session_id': 's-1'}
    assert llm.tool_results == [('c1', expected)]
    assert events[0] is call
    assert events[1] == ToolResultEvent(
        id='c1',
        name='save_lead',
        arguments={'name': 'Asha', 'user_id': 'u-1', 'session_id': 's-1'},
        result=expected,
    )
    assert isinstance(events[2], TurnCompleteEvent)


async def test_context_params_hidden_from_model():
    llm = FakeRealtimeLLM([])
    agent = RealtimeVoiceAgent(
        llm, [SAVE_LEAD, ECHO], 'prompt', CONTEXT, {'voice_name': 'Puck'}
    )
    await agent.start()

    prompt, tools, config = llm.connected_with
    assert prompt == 'prompt'
    assert config == {'voice_name': 'Puck'}
    by_name = {tool.name: tool for tool in tools}
    assert list(by_name['save_lead'].parameters) == ['name']
    assert by_name['echo'] is ECHO


async def test_context_only_passed_to_tools_that_accept_it():
    calls = [
        ToolCallEvent(id='1', name='echo', arguments={'text': 'hi'}),
        ToolCallEvent(id='2', name='flexible', arguments={'a': 1}),
    ]
    llm = FakeRealtimeLLM(calls)
    agent = RealtimeVoiceAgent(
        llm, [ECHO, _tool('flexible', flexible, {})], 'p', CONTEXT
    )
    await _run(agent)
    assert llm.tool_results == [
        ('1', 'hi'),
        ('2', {'a': 1, 'user_id': 'u-1', 'session_id': 's-1'}),
    ]


async def test_unknown_tool_reports_error_to_model():
    llm = FakeRealtimeLLM([ToolCallEvent(id='x', name='nope', arguments={})])
    events = await _run(RealtimeVoiceAgent(llm, [ECHO], 'p'))
    assert llm.tool_results == [('x', {'error': 'Unknown tool: nope'})]
    assert events[1].error == 'Unknown tool: nope'


async def test_tool_failure_does_not_end_session():
    async def boom(text: str):
        raise RuntimeError('db down')

    llm = FakeRealtimeLLM(
        [
            ToolCallEvent(id='1', name='boom', arguments={'text': 'a'}),
            TranscriptEvent(role='assistant', text='Sorry'),
        ]
    )
    agent = RealtimeVoiceAgent(
        llm, [_tool('boom', boom, {'text': {'type': 'string', 'description': ''}})], 'p'
    )
    events = await _run(agent)
    assert 'db down' in llm.tool_results[0][1]['error']
    assert events[-1] == TranscriptEvent(role='assistant', text='Sorry')


async def test_tool_timeout():
    async def slow(text: str):
        await asyncio.sleep(1)

    llm = FakeRealtimeLLM([ToolCallEvent(id='1', name='slow', arguments={'text': 'a'})])
    agent = RealtimeVoiceAgent(
        llm,
        [_tool('slow', slow, {'text': {'type': 'string', 'description': ''}})],
        'p',
        tool_timeout=0.01,
    )
    events = await _run(agent)
    assert events[1].error == 'Tool slow timed out'


async def test_json_string_arguments_parsed_and_invalid_rejected():
    llm = FakeRealtimeLLM(
        [
            ToolCallEvent(id='1', name='echo', arguments='{"text": "hey"}'),
            ToolCallEvent(id='2', name='echo', arguments='{not json'),
        ]
    )
    await _run(RealtimeVoiceAgent(llm, [ECHO], 'p'))
    assert llm.tool_results[0] == ('1', 'hey')
    assert llm.tool_results[1][1]['error'].startswith('Invalid arguments')


async def test_context_manager_connects_and_closes():
    llm = FakeRealtimeLLM([])
    async with RealtimeVoiceAgent(llm, [], 'p') as agent:
        await agent.send_audio(b'pcm')
    assert llm.connected_with is not None
    assert llm.audio == [b'pcm']
    assert llm.closed
