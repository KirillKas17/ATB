# -*- coding: utf-8 -*-
"""
Промышленный кэш для инфраструктуры торговых сессий.
"""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    value: V
    expires_at: float


class SessionCache(Generic[K, V]):
    """
    Кэш с поддержкой TTL, LRU и потокобезопасностью для сессионных данных.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time():
                del self._store[key]
                return None
            # LRU: перемещаем в конец
            self._store.move_to_end(key)
            return entry.value

    def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        with self._lock:
            expires = time() + (ttl if ttl is not None else self._ttl)
            self._store[key] = CacheEntry(value, expires)
            self._store.move_to_end(key)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def delete(self, key: K) -> None:
        with self._lock:
            if key in self._store:
                del self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def exists(self, key: K) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.expires_at < time():
                return False
            return True

    def size(self) -> int:
        with self._lock:
            return len(self._store)
