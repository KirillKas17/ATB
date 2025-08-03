"""
Кэш для данных биржи - Production Ready
"""

import time
from asyncio import Lock
from typing import Any, Dict, List, Optional, Tuple


class ExchangeCache:
    """Кэш для данных биржи."""

    def __init__(self, max_size: int = 1000, ttl: int = 60):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        async with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        """Установить значение в кэш."""
        async with self.lock:
            if len(self.cache) >= self.max_size:
                # Удаляем самый старый элемент
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (value, time.time())

    async def clear(self) -> None:
        """Очистить кэш."""
        async with self.lock:
            self.cache.clear()

    async def remove(self, key: str) -> bool:
        """Удалить ключ из кэша."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def get_size(self) -> int:
        """Получить размер кэша."""
        async with self.lock:
            return len(self.cache)

    async def get_keys(self) -> List[str]:
        """Получить все ключи кэша."""
        async with self.lock:
            return list(self.cache.keys())
