"""
Модуль кэширования для infrastructure/evolution слоя.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, cast
from uuid import UUID

from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, StrategyCandidate
from infrastructure.evolution.exceptions import CacheError
from infrastructure.evolution.types import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    EvolutionCacheProtocol,
)

logger = logging.getLogger(__name__)


class EvolutionCache(EvolutionCacheProtocol):
    """
    Кэш для эволюционных стратегий.
    Реализует LRU (Least Recently Used) стратегию кэширования
    с поддержкой TTL (Time To Live).
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """
        Инициализировать кэш.
        Args:
            config: Конфигурация кэша
        """
        if config is None:
            config = {
                "cache_size": 1000,
                "cache_ttl": 300,
                "cache_strategy": "lru",
                "enable_persistence": False,
                "persistence_path": "",
                "enable_compression": False,
                "compression_level": 6,
            }
        # Валидация конфига
        if (
            config.get("cache_size", 0) <= 0
            or config.get("cache_ttl", 0) <= 0
            or config.get("cache_strategy", "lru") not in ("lru", "fifo")
        ):
            raise CacheError("Некорректная конфигурация кэша", "invalid_config")
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}  # Изменяем тип на str вместо CacheKey
        self.access_order: list[str] = []  # Изменяем тип на str
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0, "deletes": 0}
        logger.info(
            f"Кэш эволюции инициализирован: размер={config.get('cache_size', 1000)}, TTL={config.get('cache_ttl', 300)}"
        )

    @property
    def max_size(self) -> int:
        return self.config.get("cache_size", 1000)

    @property
    def ttl(self) -> int:
        return self.config.get("cache_ttl", 3600)

    @property
    def strategy(self) -> str:
        return self.config.get("cache_strategy", "lru")

    @property
    def _cache(self) -> Dict[Any, Any]:
        result = {}
        for k, v in self.cache.items():
            if (
                str(k).startswith("candidate:")
                or str(k).startswith("evaluation:")
                or str(k).startswith("context:")
            ):
                try:
                    uuid = UUID(str(k).split(":", 1)[1])
                    val = v["value"]
                    if isinstance(val, dict) and "data" in val:
                        result[str(uuid)] = dict(val)  # Используем str(uuid) вместо uuid
                    else:
                        result[str(uuid)] = dict(v)  # Используем str(uuid) вместо uuid
                except Exception:
                    result[k] = dict(v)
            else:
                result[k] = dict(v)
        return result

    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Получить значение из кэша.
        Args:
            key: Ключ кэша
        Returns:
            Значение или None, если не найдено или истекло
        Raises:
            CacheError: При ошибке получения данных
        """
        try:
            if key not in self.cache:
                self.stats["misses"] += 1
                logger.debug(f"Кэш-промах: {key}")
                return None
            entry = self.cache[key]
            # Проверить TTL
            expires_at = entry["expires_at"]
            if expires_at:
                if not isinstance(expires_at, str):
                    expires_at = str(expires_at)
                if datetime.fromisoformat(expires_at) < datetime.now():
                    self.delete(key)
                    self.stats["misses"] += 1
                    logger.debug(f"Кэш истек: {key}")
                    return None
            # Обновить статистику доступа
            entry["access_count"] += 1
            entry["last_accessed"] = datetime.now().isoformat()
            # Обновить порядок доступа для LRU
            if self.config.get("cache_strategy", "lru") == "lru":
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            self.stats["hits"] += 1
            logger.debug(f"Кэш-попадание: {key}")
            return entry["value"]
        except Exception as e:
            logger.error(f"Ошибка получения из кэша: {e}")
            raise CacheError(f"Не удалось получить значение из кэша: {e}", "get_error")

    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """
        Установить значение в кэш.
        Args:
            key: Ключ кэша
            value: Значение для кэширования
            ttl: Время жизни в секундах (None = использовать значение по умолчанию)
        Raises:
            CacheError: При ошибке установки данных
        """
        if not isinstance(key, str) or not key:
            raise CacheError("Ключ кэша не может быть пустым", "invalid_key")
        if value is None:
            raise CacheError("Значение кэша не может быть пустым", "invalid_value")
        try:
            if ttl is None:
                ttl = self.config.get("cache_ttl", 300)
            if len(self.cache) >= self.config.get("cache_size", 1000):
                if self.config.get("cache_strategy", "lru") == "fifo":
                    self.evict_fifo()
                else:
                    self._evict_oldest()
            expires_at = None
            if ttl > 0:
                expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
            entry_value = value
            if hasattr(value, "to_dict"):
                entry_value = {
                    "data": value.to_dict(),
                    "__type__": type(value).__name__,
                }
            entry: CacheEntry = {
                "key": key,
                "value": entry_value,
                "created_at": datetime.now().isoformat(),
                "expires_at": str(expires_at) if expires_at is not None else None,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": (
                    len(json.dumps(entry_value))
                    if isinstance(entry_value, (dict, list))
                    else 0
                ),
            }
            self.cache[key] = entry
            if key not in self.access_order:
                self.access_order.append(key)
            self.stats["sets"] += 1
            logger.debug(f"Значение установлено в кэш: {key}")
        except Exception as e:
            logger.error(f"Ошибка установки в кэш: {e}")
            raise CacheError(f"Не удалось установить значение в кэш: {e}", "set_error")

    def delete(self, key: CacheKey) -> bool:
        """
        Удалить значение из кэша.
        Args:
            key: Ключ кэша
        Returns:
            True, если значение было удалено, False, если не найдено
        Raises:
            CacheError: При ошибке удаления
        """
        try:
            if key not in self.cache:
                logger.debug(f"Попытка удаления несуществующего ключа: {key}")
                return False
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.stats["deletes"] += 1
            logger.debug(f"Значение удалено из кэша: {key}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления из кэша: {e}")
            raise CacheError(
                f"Не удалось удалить значение из кэша: {e}", "delete_error"
            )

    def clear(self) -> None:
        """
        Очистить кэш.
        Raises:
            CacheError: При ошибке очистки
        """
        try:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Кэш очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
            raise CacheError(f"Не удалось очистить кэш: {e}", "clear_error")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.
        Returns:
            Статистика кэша
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "max_size": self.config["cache_size"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "total_requests": total_requests,
            "memory_usage_bytes": sum(
                entry["size_bytes"] for entry in self.cache.values()
            ),
        }

    def _evict_oldest(self) -> None:
        """Удалить самый старый элемент из кэша (LRU)."""
        if not self.access_order:
            return
        oldest_key = self.access_order[0]
        self.delete(cast(CacheKey, oldest_key))
        self.stats["evictions"] += 1
        logger.debug(f"Вытеснен старый элемент: {oldest_key}")

    def evict_fifo(self) -> None:
        """Удалить первый добавленный элемент (FIFO)."""
        if not self.access_order:
            return
        fifo_key = self.access_order.pop(0)
        self.delete(cast(CacheKey, fifo_key))
        self.stats["evictions"] += 1
        logger.debug(f"FIFO-вытеснен элемент: {fifo_key}")

    def cleanup_expired(self) -> int:
        """
        Очистить истекшие элементы.
        Returns:
            Количество удаленных элементов
        """
        expired_keys = []
        now = datetime.now()
        for key, entry in self.cache.items():
            expires_at = entry["expires_at"]
            if expires_at:
                if not isinstance(expires_at, str):
                    expires_at = str(expires_at)
                if datetime.fromisoformat(expires_at) < now:
                    expired_keys.append(key)
        for key in expired_keys:
            self.delete(cast(CacheKey, key))
        if expired_keys:
            logger.info(f"Удалено истекших элементов: {len(expired_keys)}")
        return len(expired_keys)

    def set_candidate(self, candidate_id: UUID, candidate: Any) -> None:
        if not isinstance(candidate_id, UUID):
            raise CacheError("Некорректный ключ кандидата", "invalid_key")
        self.set(cast(CacheKey, f"candidate:{candidate_id}"), candidate)

    def get_candidate(self, candidate_id: UUID) -> Optional[Any]:
        if not isinstance(candidate_id, UUID):
            raise CacheError("Некорректный ключ кандидата", "invalid_key")
        key = cast(CacheKey, f"candidate:{candidate_id}")
        entry = self.cache.get(key)
        if entry:
            expires_at = entry.get("expires_at")
            if expires_at:
                if not isinstance(expires_at, str):
                    expires_at = str(expires_at)
                if datetime.fromisoformat(expires_at) < datetime.now():
                    self.delete(key)
                    return None
        value = entry["value"] if entry else None
        if isinstance(value, dict) and "data" in value and "__type__" in value:
            if value["__type__"] == "StrategyCandidate":
                return StrategyCandidate.from_dict(value["data"])
        return value

    def delete_candidate(self, candidate_id: UUID) -> bool:
        if not isinstance(candidate_id, UUID):
            raise CacheError("Некорректный ключ кандидата", "invalid_key")
        return self.delete(cast(CacheKey, f"candidate:{candidate_id}"))

    def set_evaluation(self, evaluation_id: UUID, evaluation: Any) -> None:
        if not isinstance(evaluation_id, UUID):
            raise CacheError("Некорректный ключ оценки", "invalid_key")
        self.set(cast(CacheKey, f"evaluation:{evaluation_id}"), evaluation)

    def get_evaluation(self, evaluation_id: UUID) -> Optional[Any]:
        if not isinstance(evaluation_id, UUID):
            raise CacheError("Некорректный ключ оценки", "invalid_key")
        key = cast(CacheKey, f"evaluation:{evaluation_id}")
        entry = self.cache.get(key)
        if entry:
            expires_at = entry.get("expires_at")
            if expires_at:
                if not isinstance(expires_at, str):
                    expires_at = str(expires_at)
                if datetime.fromisoformat(expires_at) < datetime.now():
                    self.delete(key)
                    return None
        value = entry["value"] if entry else None
        if isinstance(value, dict) and "data" in value and "__type__" in value:
            if value["__type__"] == "StrategyEvaluationResult":
                return StrategyEvaluationResult.from_dict(value["data"])
        return value

    def delete_evaluation(self, evaluation_id: UUID) -> bool:
        if not isinstance(evaluation_id, UUID):
            raise CacheError("Некорректный ключ оценки", "invalid_key")
        return self.delete(cast(CacheKey, f"evaluation:{evaluation_id}"))

    def set_context(self, context_id: UUID, context: Any) -> None:
        if not isinstance(context_id, UUID):
            raise CacheError("Некорректный ключ контекста", "invalid_key")
        self.set(cast(CacheKey, f"context:{context_id}"), context)

    def get_context(self, context_id: UUID) -> Optional[Any]:
        if not isinstance(context_id, UUID):
            raise CacheError("Некорректный ключ контекста", "invalid_key")
        key = cast(CacheKey, f"context:{context_id}")
        entry = self.cache.get(key)
        if entry:
            expires_at = entry.get("expires_at")
            if expires_at:
                if not isinstance(expires_at, str):
                    expires_at = str(expires_at)
                if datetime.fromisoformat(expires_at) < datetime.now():
                    self.delete(key)
                    return None
        value = entry["value"] if entry else None
        if isinstance(value, dict) and "data" in value and "__type__" in value:
            if value["__type__"] == "EvolutionContext":
                return EvolutionContext.from_dict(value["data"])
        return value

    def delete_context(self, context_id: UUID) -> bool:
        if not isinstance(context_id, UUID):
            raise CacheError("Некорректный ключ контекста", "invalid_key")
        return self.delete(cast(CacheKey, f"context:{context_id}"))

    def get_stats(self) -> Dict[str, Any]:
        stats = self.get_statistics()
        stats["total_items"] = len(self.cache)
        total_requests = stats["total_requests"]
        stats["miss_rate"] = (
            (stats["misses"] / total_requests) if total_requests > 0 else 0.0
        )
        stats["candidates"] = sum(
            1 for k in self.cache if str(k).startswith("candidate:")
        )
        stats["evaluations"] = sum(
            1 for k in self.cache if str(k).startswith("evaluation:")
        )
        stats["contexts"] = sum(1 for k in self.cache if str(k).startswith("context:"))
        stats["custom_data"] = (
            stats["total_items"]
            - stats["candidates"]
            - stats["evaluations"]
            - stats["contexts"]
        )
        return stats

    _cleanup_expired = cleanup_expired
