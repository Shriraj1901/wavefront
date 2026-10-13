"""Bearer-token authentication for WebSocket routes.

RequireAuthMiddleware is a BaseHTTPMiddleware, which Starlette only runs for
HTTP requests: a WebSocket handshake reaches its route unauthenticated. Routes
that accept WebSockets call authenticate_token themselves, with the token taken
from the first frame (browsers cannot set an Authorization header on a
WebSocket).

Mirrors the middleware's user-token branch: signature check, then the session
the token names must exist and belong to the token's user. Service credentials
(HMAC, mTLS, passthrough) are not accepted here.
"""

import json

import jwt
from auth_module.auth_container import AuthContainer
from auth_module.services.token_service import TokenService
from common_module.log.logger import logger
from db_repo_module.cache.cache_manager import CacheManager
from db_repo_module.models.session import Session
from db_repo_module.repositories.sql_alchemy_repository import SQLAlchemyRepository
from dependency_injector.wiring import Provide, inject

from user_management_module.authorization.require_auth import UserSession
from user_management_module.constants.cache import get_session_cache_key
from user_management_module.user_container import UserContainer


class WebSocketAuthError(Exception):
    """The token was missing, invalid, expired or names a dead session.

    The message is safe to send to the client.
    """


@inject
async def authenticate_token(
    token: str,
    token_service: TokenService = Provide[AuthContainer.token_service],
    session_repository: SQLAlchemyRepository[Session] = Provide[
        UserContainer.session_repository
    ],
    cache_manager: CacheManager = Provide[UserContainer.cache_manager],
) -> UserSession:
    if not token:
        raise WebSocketAuthError('Token missing')

    try:
        decoded = token_service.decode_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise WebSocketAuthError('Token has expired. Please log in again.') from exc
    except Exception as exc:
        logger.warning(f'WebSocket token rejected: {exc}')
        raise WebSocketAuthError('Invalid token') from exc

    if not decoded or 'session_id' not in decoded or 'user_id' not in decoded:
        raise WebSocketAuthError('Invalid token')

    session_cache_key = get_session_cache_key(decoded['session_id'])
    session_user_id = None
    cached_session = cache_manager.get_str(session_cache_key)
    if cached_session:
        try:
            session_user_id = json.loads(cached_session).get('user_id')
        except json.JSONDecodeError:
            logger.error('Failed to decode cached session data, fetching from DB')

    if session_user_id is None:
        session = await session_repository.find_one(id=decoded['session_id'])
        if not session:
            raise WebSocketAuthError('Invalid session')
        session_user_id = str(session.user_id)
        cache_manager.add(
            session_cache_key,
            json.dumps(
                {
                    'id': str(session.id),
                    'user_id': session_user_id,
                    'device_info': session.device_info,
                }
            ),
            token_service.token_expiry,
        )

    if session_user_id != decoded['user_id']:
        raise WebSocketAuthError('Invalid session')

    return UserSession(
        role_id=decoded.get('role_id'),
        user_id=decoded['user_id'],
        session_id=decoded['session_id'],
    )
