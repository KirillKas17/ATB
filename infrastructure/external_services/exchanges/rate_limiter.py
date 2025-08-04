"""
Ограничитель запросов - Production Ready
"""

import asyncio
import time
from asyncio import Lock
from typing import List

from domain.type_definitions.external_service_types import RateLimit, RateLimitWindow


class RateLimiter:
    """Ограничитель запросов."""

    def __init__(self, rate_limit: RateLimit, window: RateLimitWindow = RateLimitWindow(60)):
        self.rate_limit = rate_limit
        self.window = window
        self.requests: List[float] = []
        self.lock = Lock()

    async def acquire(self) -> None:
        """Получить разрешение на запрос."""
        async with self.lock:
            now = time.time()
            # Удаляем старые запросы
            self.requests = [req for req in self.requests if now - req < self.window]

            if len(self.requests) >= self.rate_limit:
                sleep_time = self.window - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.requests.append(time.time())

    async def get_remaining_requests(self) -> int:
        """Получить количество оставшихся запросов."""
        async with self.lock:
            now = time.time()
            self.requests = [req for req in self.requests if now - req < self.window]
            return max(0, int(self.rate_limit) - len(self.requests))

    async def get_reset_time(self) -> float:
        """Получить время сброса лимита."""
        async with self.lock:
            if not self.requests:
                return 0.0

            now = time.time()
            oldest_request = min(self.requests)
            return max(0.0, float(self.window) - (now - oldest_request))

    async def reset(self) -> None:
        """Сбросить лимит."""
        async with self.lock:
            self.requests.clear()
