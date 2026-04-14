from collections.abc import Callable

import redis

from app.config import get_settings


RedisFactory = Callable[[], redis.Redis]


def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )
