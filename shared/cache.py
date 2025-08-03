"""
Cache management module for shared components.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cache."""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


class CacheEvictionStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Configuration for cache."""

    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000
    ttl_seconds: int = 300
    eviction_strategy: CacheEvictionStrategy = CacheEvictionStrategy.LRU
    enable_compression: bool = False
    enable_encryption: bool = False


@dataclass
class CacheEntry:
    """Cache entry."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def update_access(self) -> None:
        """Update access information."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheProtocol:
    """Protocol for cache implementations."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value by key."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all cache."""
        raise NotImplementedError

    async def size(self) -> int:
        """Get cache size."""
        raise NotImplementedError


class MemoryCache(CacheProtocol):
    """In-memory cache implementation."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None

            entry.update_access()
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value by key."""
        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.config.max_size:
                await self._evict_entries()

            entry = CacheEntry(key=key, value=value, ttl=ttl or self.config.ttl_seconds)
            self._cache[key] = entry
            return True

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False

            return True

    async def clear(self) -> None:
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        """Get cache size."""
        async with self._lock:
            # Remove expired entries
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]

            return len(self._cache)

    async def _evict_entries(self) -> None:
        """Evict entries based on strategy."""
        if not self._cache:
            return

        if self.config.eviction_strategy == CacheEvictionStrategy.LRU:
            # Remove least recently used
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].accessed_at
            )
            del self._cache[oldest_key]

        elif self.config.eviction_strategy == CacheEvictionStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].access_count
            )
            del self._cache[least_used_key]

        elif self.config.eviction_strategy == CacheEvictionStrategy.FIFO:
            # Remove first in
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]


class CacheManager:
    """Centralized cache manager."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._caches: Dict[str, CacheProtocol] = {}
        self._default_cache: Optional[CacheProtocol] = None

    async def initialize(self) -> None:
        """Initialize cache manager."""
        if self.config.cache_type == CacheType.MEMORY:
            self._default_cache = MemoryCache(self.config)
        elif self.config.cache_type == CacheType.REDIS:
            raise NotImplementedError(
                "Redis cache type is not implemented yet. Please use MEMORY or extend CacheManager."
            )
        elif self.config.cache_type == CacheType.DISK:
            raise NotImplementedError(
                "Disk cache type is not implemented yet. Please use MEMORY or extend CacheManager."
            )
        elif self.config.cache_type == CacheType.HYBRID:
            raise NotImplementedError(
                "Hybrid cache type is not implemented yet. Please use MEMORY or extend CacheManager."
            )
        else:
            raise NotImplementedError(f"Unknown cache type: {self.config.cache_type}")

    async def get_cache(self, name: str = "default") -> CacheProtocol:
        """Get cache by name."""
        if name not in self._caches:
            self._caches[name] = MemoryCache(self.config)
        return self._caches[name]

    async def get_default_cache(self) -> Optional[CacheProtocol]:
        """Get default cache."""
        if self._default_cache is None:
            await self.initialize()
        return self._default_cache

    async def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            await cache.clear()
        if self._default_cache:
            await self._default_cache.clear()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        config = config or CacheConfig()
        _cache_manager = CacheManager(config)
        await _cache_manager.initialize()
    return _cache_manager


def cache_decorator(ttl: Optional[int] = None, cache_name: str = "default"):
    """Decorator for caching function results."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_manager = await get_cache_manager()
            cache = await cache_manager.get_cache(cache_name)

            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator
