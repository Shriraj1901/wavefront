"""The worker follows floware's API-service update channel."""

import json
import threading
from unittest.mock import MagicMock

import celery_worker.services as services
from api_services_module.utils.api_change_publisher import (
    REDIS_API_SERVICE_UPDATES_CHANNEL,
)


def test_updates_evict_the_service_so_it_reloads_on_next_use():
    messages = [
        {'type': 'subscribe', 'data': 1},
        {
            'type': 'message',
            'data': json.dumps({'service_id': 'osm-nominatim', 'operation': 'update'}),
        },
        {'type': 'message', 'data': 'not json'},
        {
            'type': 'message',
            'data': json.dumps({'service_id': 'old-service', 'operation': 'delete'}),
        },
    ]
    done = threading.Event()

    def listen():
        yield from messages
        done.set()

    pubsub = MagicMock()
    pubsub.listen.side_effect = listen
    cache = MagicMock()
    cache.subscribe.return_value = pubsub
    proxy = MagicMock()

    services._listen_for_api_service_updates(cache, proxy)

    assert done.wait(timeout=5)
    cache.subscribe.assert_called_once_with(
        channels=[REDIS_API_SERVICE_UPDATES_CHANNEL]
    )
    assert [c.args[0] for c in proxy.remove_service.call_args_list] == [
        'osm-nominatim',
        'old-service',
    ]
