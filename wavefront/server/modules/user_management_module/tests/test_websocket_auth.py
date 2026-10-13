"""WebSocket routes authenticate the token themselves, because
RequireAuthMiddleware never runs for a WebSocket handshake."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

from user_management_module.authorization.websocket_auth import (
    WebSocketAuthError,
    authenticate_token,
)

USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())


def _deps(decoded=None, decode_error=None, cached=None, db_session=None):
    token_service = MagicMock()
    token_service.token_expiry = 3600
    if decode_error is not None:
        token_service.decode_token.side_effect = decode_error
    else:
        token_service.decode_token.return_value = (
            decoded
            if decoded is not None
            else {'user_id': USER_ID, 'session_id': SESSION_ID, 'role_id': 'r'}
        )
    cache_manager = MagicMock()
    cache_manager.get_str.return_value = cached
    session_repository = AsyncMock()
    session_repository.find_one.return_value = db_session
    return {
        'token_service': token_service,
        'session_repository': session_repository,
        'cache_manager': cache_manager,
    }


async def test_valid_token_with_cached_session():
    deps = _deps(cached=json.dumps({'user_id': USER_ID}))
    session = await authenticate_token('tok', **deps)
    assert (session.user_id, session.session_id, session.role_id) == (
        USER_ID,
        SESSION_ID,
        'r',
    )
    deps['session_repository'].find_one.assert_not_called()


async def test_valid_token_falls_back_to_db_and_caches():
    db_session = SimpleNamespace(id=SESSION_ID, user_id=USER_ID, device_info=None)
    deps = _deps(db_session=db_session)
    session = await authenticate_token('tok', **deps)
    assert session.user_id == USER_ID
    deps['cache_manager'].add.assert_called_once()


@pytest.mark.parametrize(
    'deps, message',
    [
        (_deps(decode_error=jwt.ExpiredSignatureError()), 'expired'),
        (_deps(decode_error=jwt.InvalidTokenError('bad')), 'Invalid token'),
        (_deps(decoded={}), 'Invalid token'),
        (_deps(decoded={'user_id': USER_ID}), 'Invalid token'),
        (_deps(db_session=None), 'Invalid session'),
        (_deps(cached=json.dumps({'user_id': str(uuid.uuid4())})), 'Invalid session'),
        (
            _deps(
                db_session=SimpleNamespace(
                    id=SESSION_ID, user_id=uuid.uuid4(), device_info=None
                )
            ),
            'Invalid session',
        ),
    ],
)
async def test_rejected(deps, message):
    with pytest.raises(WebSocketAuthError, match=message):
        await authenticate_token('tok', **deps)


async def test_empty_token():
    with pytest.raises(WebSocketAuthError, match='missing'):
        await authenticate_token('', **_deps())
