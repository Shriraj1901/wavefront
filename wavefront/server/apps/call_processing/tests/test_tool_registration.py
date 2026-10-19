"""Tool handlers and the deadline each is registered with.

Pipecat cancels a tool handler at its deadline, mid-request. A tool whose own
request timeout is longer than the global deadline must get a longer one;
no tool may get a shorter one than before.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from call_processing.services.llm_service import (
    FUNCTION_CALL_TIMEOUT_SECS,
    function_call_timeout_secs,
)
from call_processing.services.tool_wrapper_service import (
    ApiToolWrapper,
    ToolRegistration,
    ToolWrapperFactory,
)


@pytest.mark.parametrize(
    ('tool_timeout', 'expected'),
    [
        (None, FUNCTION_CALL_TIMEOUT_SECS),
        (6, FUNCTION_CALL_TIMEOUT_SECS),
        (8, FUNCTION_CALL_TIMEOUT_SECS),
        (15, 17.0),
        ('20', 22.0),
        (0, FUNCTION_CALL_TIMEOUT_SECS),
        (-5, FUNCTION_CALL_TIMEOUT_SECS),
        ('soon', FUNCTION_CALL_TIMEOUT_SECS),
        (math.inf, FUNCTION_CALL_TIMEOUT_SECS),
    ],
)
def test_function_call_timeout_never_shortens_the_global_deadline(
    tool_timeout, expected
):
    assert function_call_timeout_secs(tool_timeout) == expected


def api_tool(name='lookup', **config):
    return {
        'name': name,
        'description': 'A test tool',
        'tool_type': 'api',
        'config': {'url': 'https://api.example.com/x', 'method': 'GET', **config},
        'parameter_schema': {'properties': {}, 'required': []},
        'association': {'is_enabled': True},
    }


def test_registrations_carry_each_tools_request_timeout():
    overridden = api_tool('book', timeout=15)
    overridden['association']['config_overrides'] = {'timeout': 20}
    python_tool = {**api_tool('calc'), 'tool_type': 'python'}

    schemas, registrations = ToolWrapperFactory.create_all_tool_functions(
        [api_tool(), overridden, python_tool]
    )

    assert [s.name for s in schemas] == ['lookup', 'book', 'calc']
    assert all(isinstance(r, ToolRegistration) for r in registrations)
    assert [(r.name, r.request_timeout_secs) for r in registrations] == [
        ('lookup', 6),
        ('book', 20),
        ('calc', None),
    ]


async def test_handler_reports_through_the_result_callback():
    _, handler = ToolWrapperFactory.create_pipecat_function(
        api_tool(), ApiToolWrapper(url='https://api.example.com/x', method='GET')
    )
    params = SimpleNamespace(arguments={'q': 1}, result_callback=AsyncMock())

    with patch.object(ApiToolWrapper, 'execute', AsyncMock(return_value={'ok': True})):
        returned = await handler(params)

    params.result_callback.assert_awaited_once_with({'ok': True})
    assert returned is None


async def test_handler_reports_a_failure_through_the_result_callback():
    _, handler = ToolWrapperFactory.create_pipecat_function(
        api_tool(), ApiToolWrapper(url='https://api.example.com/x', method='GET')
    )
    params = SimpleNamespace(arguments={}, result_callback=AsyncMock())

    with patch.object(ApiToolWrapper, 'execute', AsyncMock(side_effect=RuntimeError)):
        await handler(params)

    result = params.result_callback.await_args.args[0]
    assert result['success'] is False
    assert result['status_code'] == 500
