"""
Кэширование для торгового репозитория.
"""

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

T = TypeVar("T")


class CacheEntry(Generic[T]):
    """Запись в кэше."""

    def __init__(self, key: str, value: T, ttl: Optional[timedelta] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        self._is_expired = False

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        if self._is_expired:
            return True
        if self.ttl is None:
            return False
        now = datetime.now(timezone.utc)
        return (now - self.created_at) > self.ttl

    def access(self) -> None:
        """Отметить доступ к записи."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def mark_expired(self) -> None:
        """Пометить запись как истекшую."""
        self._is_expired = True


class LRUCache:
    """LRU кэш с поддержкой TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[timedelta] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.logger = logging.getLogger(__name__)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Запуск фоновой очистки кэша."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Остановка фоновой очистки кэша."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        async with self._lock:
            if key not in self.cache:
                return None
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return None
            entry.access()
            # Перемещаем в конец (LRU)
            self.cache.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Установка значения в кэш."""
        async with self._lock:
            # Удаляем старую запись если существует
            if key in self.cache:
                del self.cache[key]
            # Проверяем размер кэша
            if len(self.cache) >= self.max_size:
                # Удаляем самую старую запись
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            # Создаем новую запись
            entry_ttl = ttl or self.default_ttl
            entry = CacheEntry(key, value, entry_ttl)
            self.cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Очистка всего кэша."""
        async with self._lock:
            self.cache.clear()

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа в кэше."""
        async with self._lock:
            if key not in self.cache:
                return False
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return False
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        async with self._lock:
            total_entries = len(self.cache)
            expired_entries = sum(
                1 for entry in self.cache.values() if entry.is_expired()
            )
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            avg_accesses = total_accesses / total_entries if total_entries > 0 else 0
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "max_size": self.max_size,
                "usage_percent": (total_entries / self.max_size) * 100,
                "total_accesses": total_accesses,
                "average_accesses": avg_accesses,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _cleanup_loop(self) -> None:
        """Фоновая очистка истекших записей."""
        while True:
            try:
                await asyncio.sleep(60)  # Проверяем каждую минуту
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired(self) -> None:
        """Очистка истекших записей."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                self.logger.debug(
                    f"Cleaned up {len(expired_keys)} expired cache entries"
                )


class TradingRepositoryCache:
    """Кэш для торгового репозитория."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # Кэши для разных типов данных
        self.orders_cache = LRUCache(max_size=1000, default_ttl=timedelta(minutes=5))
        self.positions_cache = LRUCache(max_size=500, default_ttl=timedelta(minutes=5))
        self.trading_pairs_cache = LRUCache(
            max_size=100, default_ttl=timedelta(hours=1)
        )
        self.accounts_cache = LRUCache(max_size=50, default_ttl=timedelta(minutes=30))
        self.metrics_cache = LRUCache(max_size=10, default_ttl=timedelta(minutes=2))
        # Кэш для списков (с более коротким TTL)
        self.order_lists_cache = LRUCache(
            max_size=100, default_ttl=timedelta(minutes=1)
        )
        self.position_lists_cache = LRUCache(
            max_size=50, default_ttl=timedelta(minutes=1)
        )
        # Кэш для паттернов и анализа
        self.patterns_cache = LRUCache(max_size=200, default_ttl=timedelta(minutes=10))
        self.liquidity_cache = LRUCache(max_size=100, default_ttl=timedelta(minutes=5))

    async def start(self) -> None:
        """Запуск всех кэшей."""
        await asyncio.gather(
            self.orders_cache.start(),
            self.positions_cache.start(),
            self.trading_pairs_cache.start(),
            self.accounts_cache.start(),
            self.metrics_cache.start(),
            self.order_lists_cache.start(),
            self.position_lists_cache.start(),
            self.patterns_cache.start(),
            self.liquidity_cache.start(),
        )

    async def stop(self) -> None:
        """Остановка всех кэшей."""
        await asyncio.gather(
            self.orders_cache.stop(),
            self.positions_cache.stop(),
            self.trading_pairs_cache.stop(),
            self.accounts_cache.stop(),
            self.metrics_cache.stop(),
            self.order_lists_cache.stop(),
            self.position_lists_cache.stop(),
            self.patterns_cache.stop(),
            self.liquidity_cache.stop(),
        )

    async def get_order(self, order_id: Union[str, UUID]) -> Optional[Any]:
        """Получение ордера из кэша."""
        key = f"order:{str(order_id)}"
        return await self.orders_cache.get(key)

    async def set_order(
        self, order_id: Union[str, UUID], order: Any, ttl: Optional[timedelta] = None
    ) -> None:
        """Сохранение ордера в кэш."""
        key = f"order:{str(order_id)}"
        await self.orders_cache.set(key, order, ttl)

    async def delete_order(self, order_id: Union[str, UUID]) -> None:
        """Удаление ордера из кэша."""
        key = f"order:{str(order_id)}"
        await self.orders_cache.delete(key)
        # Инвалидируем связанные кэши
        await self._invalidate_order_related_caches(order_id)

    async def get_position(self, position_id: Union[str, UUID]) -> Optional[Any]:
        """Получение позиции из кэша."""
        key = f"position:{str(position_id)}"
        return await self.positions_cache.get(key)

    async def set_position(
        self,
        position_id: Union[str, UUID],
        position: Any,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Сохранение позиции в кэш."""
        key = f"position:{str(position_id)}"
        await self.positions_cache.set(key, position, ttl)

    async def delete_position(self, position_id: Union[str, UUID]) -> None:
        """Удаление позиции из кэша."""
        key = f"position:{str(position_id)}"
        await self.positions_cache.delete(key)
        # Инвалидируем связанные кэши
        await self._invalidate_position_related_caches(position_id)

    async def get_order_list(self, filters: Dict[str, Any]) -> Optional[List[Any]]:
        """Получение списка ордеров из кэша."""
        key = self._generate_list_key("orders", filters)
        return await self.order_lists_cache.get(key)

    async def set_order_list(
        self,
        filters: Dict[str, Any],
        orders: List[Any],
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Сохранение списка ордеров в кэш."""
        key = self._generate_list_key("orders", filters)
        await self.order_lists_cache.set(key, orders, ttl)

    async def get_position_list(self, filters: Dict[str, Any]) -> Optional[List[Any]]:
        """Получение списка позиций из кэша."""
        key = self._generate_list_key("positions", filters)
        return await self.position_lists_cache.get(key)

    async def set_position_list(
        self,
        filters: Dict[str, Any],
        positions: List[Any],
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Сохранение списка позиций в кэш."""
        key = self._generate_list_key("positions", filters)
        await self.position_lists_cache.set(key, positions, ttl)

    async def get_metrics(
        self, account_id: Optional[Union[str, UUID]] = None
    ) -> Optional[Dict[str, Any]]:
        """Получение метрик из кэша."""
        key = f"metrics:{str(account_id) if account_id else 'global'}"
        return await self.metrics_cache.get(key)

    async def set_metrics(
        self,
        account_id: Optional[Union[str, UUID]],
        metrics: Dict[str, Any],
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Сохранение метрик в кэш."""
        key = f"metrics:{str(account_id) if account_id else 'global'}"
        await self.metrics_cache.set(key, metrics, ttl)

    async def get_pattern(self, pattern_id: str) -> Optional[Any]:
        """Получение паттерна из кэша."""
        key = f"pattern:{pattern_id}"
        return await self.patterns_cache.get(key)

    async def set_pattern(
        self, pattern_id: str, pattern: Any, ttl: Optional[timedelta] = None
    ) -> None:
        """Сохранение паттерна в кэш."""
        key = f"pattern:{pattern_id}"
        await self.patterns_cache.set(key, pattern, ttl)

    async def get_liquidity_analysis(
        self, trading_pair_id: Union[str, UUID]
    ) -> Optional[Any]:
        """Получение анализа ликвидности из кэша."""
        key = f"liquidity:{str(trading_pair_id)}"
        return await self.liquidity_cache.get(key)

    async def set_liquidity_analysis(
        self,
        trading_pair_id: Union[str, UUID],
        analysis: Any,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Сохранение анализа ликвидности в кэш."""
        key = f"liquidity:{str(trading_pair_id)}"
        await self.liquidity_cache.set(key, analysis, ttl)

    async def clear_all(self) -> None:
        """Очистка всех кэшей."""
        await asyncio.gather(
            self.orders_cache.clear(),
            self.positions_cache.clear(),
            self.trading_pairs_cache.clear(),
            self.accounts_cache.clear(),
            self.metrics_cache.clear(),
            self.order_lists_cache.clear(),
            self.position_lists_cache.clear(),
            self.patterns_cache.clear(),
            self.liquidity_cache.clear(),
        )

    async def get_all_stats(self) -> Dict[str, Any]:
        """Получение статистики всех кэшей."""
        stats = await asyncio.gather(
            self.orders_cache.get_stats(),
            self.positions_cache.get_stats(),
            self.trading_pairs_cache.get_stats(),
            self.accounts_cache.get_stats(),
            self.metrics_cache.get_stats(),
            self.order_lists_cache.get_stats(),
            self.position_lists_cache.get_stats(),
            self.patterns_cache.get_stats(),
            self.liquidity_cache.get_stats(),
        )
        return {
            "orders": stats[0],
            "positions": stats[1],
            "trading_pairs": stats[2],
            "accounts": stats[3],
            "metrics": stats[4],
            "order_lists": stats[5],
            "position_lists": stats[6],
            "patterns": stats[7],
            "liquidity": stats[8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_list_key(self, prefix: str, filters: Dict[str, Any]) -> str:
        """Генерация ключа для списков."""
        # Сортируем фильтры для консистентности ключей
        sorted_filters = sorted(filters.items())
        filter_str = "_".join(f"{k}:{v}" for k, v in sorted_filters)
        return f"{prefix}_list:{filter_str}"

    async def _invalidate_order_related_caches(
        self, order_id: Union[str, UUID]
    ) -> None:
        """Инвалидация кэшей, связанных с ордером."""
        # Очищаем кэши списков ордеров
        await self.order_lists_cache.clear()
        # Очищаем метрики
        await self.metrics_cache.clear()

    async def _invalidate_position_related_caches(
        self, position_id: Union[str, UUID]
    ) -> None:
        """Инвалидация кэшей, связанных с позицией."""
        # Очищаем кэши списков позиций
        await self.position_lists_cache.clear()
        # Очищаем метрики
        await self.metrics_cache.clear()
