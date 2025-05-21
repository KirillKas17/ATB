from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class LRUCache:
    """LRU (Least Recently Used) кэш с TTL"""

    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        """
        Инициализация кэша.

        Args:
            capacity: Максимальное количество элементов
            ttl_seconds: Время жизни элемента в секундах
        """
        self.capacity = capacity
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.

        Args:
            key: Ключ

        Returns:
            Optional[Any]: Значение или None, если ключ не найден или устарел
        """
        if key not in self.cache:
            return None

        # Проверка TTL
        if datetime.now() - self.timestamps[key] > self.ttl:
            self._remove(key)
            return None

        # Обновление порядка
        value = self.cache.pop(key)
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

        return value

    def put(self, key: str, value: Any):
        """
        Добавление значения в кэш.

        Args:
            key: Ключ
            value: Значение
        """
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Удаление самого старого элемента
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)

        self.cache[key] = value
        self.timestamps[key] = datetime.now()

    def _remove(self, key: str):
        """Удаление элемента из кэша"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.timestamps.clear()

    def get_size(self) -> int:
        """Получение текущего размера кэша"""
        return len(self.cache)

    def is_full(self) -> bool:
        """Проверка заполненности кэша"""
        return len(self.cache) >= self.capacity
