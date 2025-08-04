"""
Реализация репозитория рисков.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager, AbstractAsyncContextManager, _AsyncGeneratorContextManager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Coroutine, Sequence, cast, AsyncContextManager
from uuid import UUID

import asyncpg

from domain.entities.risk import RiskLevel, RiskManager, RiskProfile
from domain.repositories.base_repository import (
    BulkOperationResult,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    TransactionProtocol,
)
from domain.protocols.repository_protocol import RepositoryState
from domain.type_definitions.protocol_types import HealthCheckDict, PerformanceMetricsDict
from domain.type_definitions import RiskProfileId, PortfolioId
from domain.protocols.repository_protocol import RepositoryProtocol

class RiskRepositoryProtocol(RepositoryProtocol[RiskProfile]):
    """Протокол репозитория рисков."""
    pass




class InMemoryRiskRepository(RiskRepositoryProtocol):
    """
    In-memory реализация репозитория рисков.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._risk_profiles: Dict[Union[UUID, str], RiskProfile] = {}
        self.risk_managers: Dict[str, RiskManager] = {}
        self._risk_limits: Dict[Union[UUID, str], Dict[str, Any]] = {}
        self._risk_metrics: Dict[PortfolioId, List[Dict[str, Any]]] = defaultdict(list)
        self._cache: Dict[Union[UUID, str], RiskProfile] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._metrics: Dict[str, Union[int, datetime]] = {
            "total_profiles": 0,
            "total_managers": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": datetime.now(),
        }
        self._state = RepositoryState.CONNECTED
        self._startup_time = datetime.now()
        self.logger.info("InMemoryRiskRepository initialized")

    async def save_risk_profile(self, risk_profile: RiskProfile) -> RiskProfile:
        """Сохранить профиль риска."""
        try:
            profile_id = str(risk_profile.id)
            self._risk_profiles[profile_id] = risk_profile
            self._metrics["total_profiles"] = len(self._risk_profiles)
            await self.invalidate_cache(profile_id)
            return risk_profile
        except Exception as e:
            self.logger.error(f"Failed to save risk profile {risk_profile.id}: {e}")
            raise

    async def get_risk_profile(self, profile_id: RiskProfileId) -> Optional[RiskProfile]:
        """Получить профиль риска."""
        profile_id_str = str(profile_id)
        cache_key = profile_id_str
        cached = await self.get_from_cache(cache_key)
        if cached:
            cache_hits = self._metrics.get("cache_hits", 0)
            if isinstance(cache_hits, int):
                self._metrics["cache_hits"] = cache_hits + 1
            return cached
        cache_misses = self._metrics.get("cache_misses", 0)
        if isinstance(cache_misses, int):
            self._metrics["cache_misses"] = cache_misses + 1
        profile = self._risk_profiles.get(profile_id_str)
        if profile:
            await self.set_cache(cache_key, profile)
        return profile

    async def get_default_risk_profile(self) -> Optional[RiskProfile]:
        """Получить профиль риска по умолчанию."""
        # Возвращаем первый профиль или None
        cache_key = "default_risk_profile"
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        profiles = list(self._risk_profiles.values())
        if profiles:
            default_profile = profiles[0]
            await self.set_cache(cache_key, default_profile)
            return default_profile
        return None

    async def save(self, entity: RiskProfile) -> RiskProfile:
        """Сохранить сущность."""
        return await self.save_risk_profile(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[RiskProfile]:
        """Получить сущность по ID."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        return await self.get_risk_profile(RiskProfileId(entity_id))

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[RiskProfile]:
        """Получить все сущности."""
        return list(self._risk_profiles.values())

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить сущность."""
        try:
            profile_id = str(entity_id)
            if profile_id in self._risk_profiles:
                del self._risk_profiles[profile_id]
                await self.invalidate_cache(profile_id)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete risk profile {entity_id}: {e}")
            return False

    async def get_risk_manager(self, manager_id: str) -> Optional[RiskManager]:
        """Получить менеджера рисков."""
        return self.risk_managers.get(manager_id)

    async def update_risk_limits(self, profile_id: RiskProfileId, risk_limits: Dict[str, Any]) -> bool:
        """Обновить лимиты риска."""
        try:
            profile_id_str = str(profile_id)
            self._risk_limits[profile_id_str] = risk_limits
            return True
        except Exception as e:
            self.logger.error(f"Failed to update risk limits for {profile_id}: {e}")
            return False

    async def get_risk_metrics(
        self, 
        portfolio_id: PortfolioId, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Получить метрики риска."""
        try:
            metrics = self._risk_metrics.get(portfolio_id, [])
            if start_date and end_date:
                metrics = [
                    m for m in metrics 
                    if isinstance(m.get("timestamp"), datetime) and start_date <= m.get("timestamp", datetime.min) <= end_date
                ]
            
            if not metrics:
                return {"total_risk": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
            
            total_risk = sum(m.get("risk_score", 0.0) for m in metrics)
            max_drawdown = max(m.get("drawdown", 0.0) for m in metrics)
            sharpe_ratio = sum(m.get("sharpe", 0.0) for m in metrics) / len(metrics)
            
            return {
                "total_risk": total_risk,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio
            }
        except Exception as e:
            self.logger.error(f"Failed to get risk metrics for {portfolio_id}: {e}")
            return {"total_risk": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    async def save_risk_manager(self, risk_manager: RiskManager) -> bool:
        """Сохранить менеджера рисков."""
        try:
            self.risk_managers[risk_manager.id] = risk_manager
            self._metrics["total_managers"] = len(self.risk_managers)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save risk manager {risk_manager.id}: {e}")
            return False

    async def update(self, entity: RiskProfile) -> RiskProfile:
        """Обновить сущность."""
        return await self.save_risk_profile(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление сущности."""
        # В in-memory реализации просто удаляем
        return await self.delete(entity_id)

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить сущность."""
        # В in-memory реализации нечего восстанавливать
        return False

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[RiskProfile]:
        """Поиск по фильтрам."""
        # Простая реализация поиска
        results = list(self._risk_profiles.values())
        # Здесь можно добавить логику фильтрации
        return results

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[RiskProfile]:
        """Поиск одной сущности по фильтрам."""
        results = await self.find_by(filters)
        return results[0] if results else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование сущности."""
        return str(entity_id) in self._risk_profiles

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество сущностей."""
        return len(self._risk_profiles)

    async def stream(self, options: Optional[QueryOptions] = None, batch_size: int = 100) -> AsyncIterator[RiskProfile]:
        """Потоковая передача сущностей."""
        profiles = list(self._risk_profiles.values())
        
        if options:
            if options.filters:
                profiles = self._apply_filters(profiles, options.filters)
            if options.sort_orders:
                profiles = self._apply_sort(profiles, options.sort_orders)
            if options.pagination:
                profiles = self._apply_pagination(profiles, options.pagination)
        
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            for profile in batch:
                yield profile

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        class MockTransaction(TransactionProtocol):
            async def __aenter__(self) -> "MockTransaction":
                return self

            async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
                pass

            async def commit(self) -> None:
                pass

            async def rollback(self) -> None:
                pass

            async def is_active(self) -> bool:
                return True

        transaction = MockTransaction()
        yield transaction

    async def execute_in_transaction(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as transaction:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[RiskProfile]) -> BulkOperationResult:
        """Пакетное сохранение сущностей."""
        try:
            saved_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            
            for entity in entities:
                try:
                    await self.save_risk_profile(entity)
                    saved_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    self.logger.error(f"Failed to save entity {entity.id}: {e}")
                    failed_count += 1
            
            return BulkOperationResult(
                success_count=saved_count,
                error_count=failed_count,
                processed_ids=list(processed_ids),
                errors=[]
            )
        except Exception as e:
            self.logger.error(f"Bulk save failed: {e}")
            return BulkOperationResult(
                success_count=0,
                error_count=len(entities),
                processed_ids=[],
                errors=[{"error": str(e)}]
            )

    async def bulk_update(self, entities: List[RiskProfile]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        try:
            updated_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            
            for entity in entities:
                try:
                    await self.update(entity)
                    updated_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    self.logger.error(f"Failed to update entity {entity.id}: {e}")
                    failed_count += 1
            
            return BulkOperationResult(
                success_count=updated_count,
                error_count=failed_count,
                processed_ids=list(processed_ids),
                errors=[]
            )
        except Exception as e:
            self.logger.error(f"Bulk update failed: {e}")
            return BulkOperationResult(
                success_count=0,
                error_count=len(entities),
                processed_ids=[],
                errors=[{"error": str(e)}]
            )

    async def bulk_delete(self, entity_ids: List[Union[UUID, str]]) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        try:
            deleted_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            
            for entity_id in entity_ids:
                try:
                    if await self.delete(entity_id):
                        deleted_count += 1
                        processed_ids.append(entity_id)
                    else:
                        failed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete entity {entity_id}: {e}")
                    failed_count += 1
            
            return BulkOperationResult(
                success_count=deleted_count,
                error_count=failed_count,
                processed_ids=processed_ids,
                errors=[]
            )
        except Exception as e:
            self.logger.error(f"Bulk delete failed: {e}")
            return BulkOperationResult(
                success_count=0,
                error_count=len(entity_ids),
                processed_ids=[],
                errors=[{"error": str(e)}]
            )

    async def bulk_upsert(self, entities: List[RiskProfile], conflict_fields: List[str]) -> BulkOperationResult:
        """Пакетное обновление или вставка сущностей."""
        return await self.bulk_save(entities)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[RiskProfile]:
        """Получить сущность из кэша."""
        key_str = str(key)
        if key_str in self._cache:
            ttl = self._cache_ttl.get(key_str)
            if ttl and datetime.now() < ttl:
                return self._cache[key_str]
            else:
                # Удаляем просроченный кэш
                del self._cache[key_str]
                if key_str in self._cache_ttl:
                    del self._cache_ttl[key_str]
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: RiskProfile, ttl: Optional[int] = None
    ) -> None:
        """Установить значение в кэш."""
        if len(self._cache) >= self._cache_max_size:
            await self._evict_cache()
        
        cache_key = str(key)
        self._cache[cache_key] = entity
        
        if ttl:
            self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl)
        else:
            self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self._cache_ttl_seconds)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        key_str = str(key)
        if key_str in self._cache:
            del self._cache[key_str]
        if key_str in self._cache_ttl:
            del self._cache_ttl[key_str]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._cache_ttl.clear()

    async def _evict_cache(self) -> None:
        """Удаление устаревших записей из кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, ttl in self._cache_ttl.items() 
            if isinstance(ttl, datetime) and current_time > ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        if len(self._cache) > self._cache_max_size:
            # Удаляем самые старые записи
            sorted_keys = sorted(
                self._cache_ttl.keys(), 
                key=lambda k: self._cache_ttl[k] if isinstance(self._cache_ttl[k], datetime) else datetime.min
            )
            keys_to_remove = sorted_keys[:len(self._cache) - self._cache_max_size]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_ttl[key]

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(3600)  # Каждый час
                await self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {str(e)}")

    def _get_cache_hit_rate(self) -> float:
        """Получить процент попаданий в кэш."""
        cache_hits_raw = self._metrics.get("cache_hits", 0) or 0
        cache_misses_raw = self._metrics.get("cache_misses", 0) or 0
        cache_hits = int(cache_hits_raw) if isinstance(cache_hits_raw, (int, float, str)) else 0
        cache_misses = int(cache_misses_raw) if isinstance(cache_misses_raw, (int, float, str)) else 0
        total_requests = cache_hits + cache_misses
        
        if total_requests == 0:
            return 0.0
        
        return float(cache_hits) / float(total_requests)

    def _apply_filters(self, profiles: List[RiskProfile], filters: List[QueryFilter]) -> List[RiskProfile]:
        """Применить фильтры к списку профилей."""
        filtered_profiles = profiles
        
        for filter_item in filters:
            if filter_item.field == "risk_level":
                filtered_profiles = [
                    p for p in filtered_profiles 
                    if p.risk_level.value == filter_item.value
                ]
            elif filter_item.field == "name":
                filtered_profiles = [
                    p for p in filtered_profiles 
                    if filter_item.value.lower() in p.name.lower()
                ]
        
        return filtered_profiles

    def _apply_sort(self, profiles: List[RiskProfile], sort_orders: List[Any]) -> List[RiskProfile]:
        """Применить сортировку к списку профилей."""
        for sort_order in sort_orders:
            reverse = sort_order.direction == "desc"
            
            if sort_order.field == "name":
                profiles.sort(key=lambda p: p.name, reverse=reverse)
            elif sort_order.field == "risk_level":
                profiles.sort(key=lambda p: p.risk_level.value, reverse=reverse)
            elif sort_order.field == "created_at":
                profiles.sort(key=lambda p: p.created_at, reverse=reverse)
            elif sort_order.field == "updated_at":
                profiles.sort(key=lambda p: p.updated_at, reverse=reverse)
        
        return profiles

    def _apply_pagination(self, profiles: List[RiskProfile], pagination: Any) -> List[RiskProfile]:
        """Применить пагинацию к списку профилей."""
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        return profiles[start:end]


class PostgresRiskRepository(RiskRepositoryProtocol):
    """PostgreSQL реализация репозитория рисков."""

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        super().__init__()
        self.connection_string = connection_string
        self.cache_service = cache_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pool: Optional[Any] = None
        self._state = RepositoryState.DISCONNECTED

    async def _get_pool(self) -> Any:
        """Получить пул соединений."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.connection_string)
            self._state = RepositoryState.CONNECTED
        return self._pool

    async def _execute_with_retry(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнить операцию с повторными попытками."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pool = await self._get_pool()
                result = await operation(pool, *args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Retry {attempt + 1} for operation: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))
        return None

    async def save_risk_profile(self, risk_profile: RiskProfile) -> RiskProfile:
        """Сохранить профиль риска."""
        async def _save_operation(conn: Any) -> RiskProfile:
            query = """
                INSERT INTO risk_profiles (id, name, risk_level, max_daily_loss, stop_loss_method, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    risk_level = EXCLUDED.risk_level,
                    max_daily_loss = EXCLUDED.max_daily_loss,
                    stop_loss_method = EXCLUDED.stop_loss_method,
                    updated_at = EXCLUDED.updated_at
            """
            await conn.execute(
                query,
                str(risk_profile.id),
                risk_profile.name,
                risk_profile.risk_level.value,
                risk_profile.max_daily_loss,
                risk_profile.stop_loss_method,
                risk_profile.created_at,
                risk_profile.updated_at
            )
            return risk_profile
        
        result = await self._execute_with_retry(_save_operation)
        return cast(RiskProfile, result)

    async def get_risk_profile(self, profile_id: RiskProfileId) -> Optional[RiskProfile]:
        """Получить профиль риска."""
        async def _get_operation(conn: Any) -> Optional[RiskProfile]:
            query = "SELECT * FROM risk_profiles WHERE id = $1"
            row = await conn.fetchrow(query, str(profile_id))
            return self._row_to_risk_profile(row) if row else None
        
        result = await self._execute_with_retry(_get_operation)
        return cast(Optional[RiskProfile], result)

    async def get_default_risk_profile(self) -> Optional[RiskProfile]:
        """Получить профиль риска по умолчанию."""
        async def _get_operation(conn: Any) -> Optional[RiskProfile]:
            query = "SELECT * FROM risk_profiles WHERE is_default = true LIMIT 1"
            row = await conn.fetchrow(query)
            return self._row_to_risk_profile(row) if row else None
        
        result = await self._execute_with_retry(_get_operation)
        return cast(Optional[RiskProfile], result)

    async def save(self, entity: RiskProfile) -> RiskProfile:
        """Сохранить сущность."""
        return await self.save_risk_profile(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[RiskProfile]:
        """Получить сущность по ID."""
        profile_id = RiskProfileId(UUID(str(entity_id)))
        return await self.get_risk_profile(profile_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[RiskProfile]:
        """Получить все сущности."""
        async def _get_all_operation(conn: Any) -> List[RiskProfile]:
            query = "SELECT * FROM risk_profiles"
            rows = await conn.fetch(query)
            return [self._row_to_risk_profile(row) for row in rows]
        
        result = await self._execute_with_retry(_get_all_operation)
        return cast(List[RiskProfile], result)

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить сущность."""
        async def _delete_operation(conn: Any) -> bool:
            # Преобразуем UUID в RiskProfileId если нужно
            profile_id = RiskProfileId(UUID(str(entity_id)))
            query = "DELETE FROM risk_profiles WHERE id = $1"
            result = await conn.execute(query, str(profile_id))
            return str(result) == "DELETE 1"
        result = await self._execute_with_retry(_delete_operation)
        return bool(result) if isinstance(result, (bool, str, int)) else False

    async def get_risk_manager(self, manager_id: str) -> Optional[RiskManager]:
        """Получить менеджера рисков."""
        async def _get_operation(conn: Any) -> Optional[RiskManager]:
            query = "SELECT * FROM risk_managers WHERE id = $1"
            row = await conn.fetchrow(query, manager_id)
            return self._row_to_risk_manager(row) if row else None
        
        result = await self._execute_with_retry(_get_operation)
        return cast(Optional[RiskManager], result)

    async def save_risk_manager(self, risk_manager: RiskManager) -> bool:
        """Сохранить менеджера рисков."""
        async def _save_operation(conn: Any) -> bool:
            # Сначала сохраняем профиль риска
            await self.save_risk_profile(risk_manager.risk_profile)
            
            query = """
                INSERT INTO risk_managers (id, name, risk_profile_id, permissions)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    risk_profile_id = EXCLUDED.risk_profile_id,
                    permissions = EXCLUDED.permissions
            """
            await conn.execute(
                query,
                risk_manager.id,
                risk_manager.name,
                str(risk_manager.risk_profile.id),
                json.dumps(risk_manager.permissions)
            )
            return True
        
        result: Optional[bool] = await self._execute_with_retry(_save_operation)
        return bool(result)

    async def update_risk_limits(self, profile_id: RiskProfileId, risk_limits: Dict[str, Any]) -> bool:
        """Обновить лимиты риска."""
        async def _update_operation(conn: Any) -> bool:
            query = """
                UPDATE risk_profiles 
                SET max_daily_loss = $2, stop_loss_method = $3, take_profit_method = $4, updated_at = NOW()
                WHERE id = $1
            """
            result = await conn.execute(
                query,
                str(profile_id),
                risk_limits.get("max_daily_loss", 0.0),
                risk_limits.get("stop_loss_method", "atr"),
                risk_limits.get("take_profit_method", "atr")
            )
            return str(result) == "UPDATE 1"
        result = await self._execute_with_retry(_update_operation)
        return bool(result) if isinstance(result, (bool, str, int)) else False

    async def get_risk_metrics(
        self, 
        portfolio_id: PortfolioId, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Получить метрики риска."""
        async def _get_operation(conn: Any) -> Dict[str, float]:
            query = """
                SELECT 
                    AVG(daily_pnl) as avg_daily_pnl,
                    MAX(daily_pnl) as max_daily_pnl,
                    MIN(daily_pnl) as min_daily_pnl,
                    STDDEV(daily_pnl) as pnl_volatility,
                    COUNT(*) as total_days
                FROM portfolio_metrics 
                WHERE portfolio_id = $1
            """
            params: List[Any] = [str(portfolio_id)]
            
            if start_date:
                query += " AND date >= $2"
                params.append(start_date)
            if end_date:
                query += " AND date <= $3"
                params.append(end_date)
            
            row = await conn.fetchrow(query, *params)
            if row:
                return {
                    "avg_daily_pnl": float(row["avg_daily_pnl"]) if row["avg_daily_pnl"] else 0.0,
                    "max_daily_pnl": float(row["max_daily_pnl"]) if row["max_daily_pnl"] else 0.0,
                    "min_daily_pnl": float(row["min_daily_pnl"]) if row["min_daily_pnl"] else 0.0,
                    "pnl_volatility": float(row["pnl_volatility"]) if row["pnl_volatility"] else 0.0,
                    "total_days": int(row["total_days"]) if row["total_days"] else 0
                }
            return {}
        
        result = await self._execute_with_retry(_get_operation)
        return cast(Dict[str, float], result)

    async def update(self, entity: RiskProfile) -> RiskProfile:
        """Обновить сущность."""
        return await self.save_risk_profile(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление сущности."""
        async def _soft_delete_operation(conn: Any) -> bool:
            # Преобразуем UUID в RiskProfileId если нужно
            profile_id = RiskProfileId(UUID(str(entity_id)))
            query = "UPDATE risk_profiles SET deleted_at = NOW() WHERE id = $1"
            result = await conn.execute(query, str(profile_id))
            return str(result) == "UPDATE 1"
        result = await self._execute_with_retry(_soft_delete_operation)
        return bool(result) if isinstance(result, (bool, str, int)) else False

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить сущность."""
        async def _restore_operation(conn: Any) -> bool:
            # Преобразуем UUID в RiskProfileId если нужно
            profile_id = RiskProfileId(UUID(str(entity_id)))
            query = "UPDATE risk_profiles SET deleted_at = NULL WHERE id = $1"
            result = await conn.execute(query, str(profile_id))
            return str(result) == "UPDATE 1"
        result = await self._execute_with_retry(_restore_operation)
        return bool(result) if isinstance(result, (bool, str, int)) else False

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[RiskProfile]:
        """Найти сущности по фильтрам."""
        async def _find_operation(conn: Any) -> List[RiskProfile]:
            query = "SELECT * FROM risk_profiles WHERE deleted_at IS NULL"
            params: List[Any] = []
            
            if filters:
                for i, filter_item in enumerate(filters, start=len(params) + 1):
                    query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                    params.append(filter_item.value)
            
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                sort_clauses = []
                for sort_order in options.sort_orders:
                    direction = "DESC" if sort_order.direction == "desc" else "ASC"
                    sort_clauses.append(f"{sort_order.field} {direction}")
                query += f" ORDER BY {', '.join(sort_clauses)}"
            else:
                query += " ORDER BY created_at DESC"
            
            if options and options.pagination:
                offset = (options.pagination.page - 1) * options.pagination.page_size
                query += f" LIMIT {options.pagination.page_size} OFFSET {offset}"
            
            rows = await conn.fetch(query, *params)
            return [self._row_to_risk_profile(row) for row in rows]
        
        result = await self._execute_with_retry(_find_operation)
        return cast(List[RiskProfile], result)

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[RiskProfile]:
        """Найти одну сущность по фильтрам."""
        profiles = await self.find_by(filters)
        return profiles[0] if profiles else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование сущности."""
        async def _exists_operation(conn: Any) -> bool:
            profile_id = RiskProfileId(UUID(str(entity_id)))
            query = "SELECT 1 FROM risk_profiles WHERE id = $1 AND deleted_at IS NULL"
            row = await conn.fetchrow(query, str(profile_id))
            return row is not None
        
        result = await self._execute_with_retry(_exists_operation)
        return True if result else False

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество сущностей."""
        async def _count_operation(conn: Any) -> int:
            query = "SELECT COUNT(*) FROM risk_profiles WHERE deleted_at IS NULL"
            params: List[Any] = []
            
            if filters:
                for i, filter_item in enumerate(filters, start=len(params) + 1):
                    query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                    params.append(filter_item.value)
            
            result = await conn.fetchval(query, *params)
            return result if result else 0
        
        result = await self._execute_with_retry(_count_operation)
        return int(result) if result is not None else 0

    async def stream(self, options: Optional[QueryOptions] = None, batch_size: int = 100) -> AsyncIterator[RiskProfile]:
        """Потоковое чтение сущностей."""
        async def _stream_operation(conn: Any) -> AsyncIterator[RiskProfile]:
            query = "SELECT * FROM risk_profiles WHERE deleted_at IS NULL"
            params: List[Any] = []
            
            if options and options.filters:
                for i, filter_item in enumerate(options.filters, start=len(params) + 1):
                    query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                    params.append(filter_item.value)
            
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                sort_clauses = []
                for sort_order in options.sort_orders:
                    direction = "DESC" if sort_order.direction == "desc" else "ASC"
                    sort_clauses.append(f"{sort_order.field} {direction}")
                query += f" ORDER BY {', '.join(sort_clauses)}"
            else:
                query += " ORDER BY created_at DESC"
            
            async with conn.transaction():
                async for record in conn.cursor(query, *params):
                    yield self._row_to_risk_profile(record)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async for profile in _stream_operation(conn):
                yield profile

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        class PostgresTransaction(TransactionProtocol):
            def __init__(self, connection: Any) -> None:
                self.connection = connection
                self._active = True
            
            async def __aenter__(self) -> "PostgresTransaction":
                return self
            
            async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
                self._active = False
            
            async def commit(self) -> None:
                await self.connection.commit()
            
            async def rollback(self) -> None:
                await self.connection.rollback()
            
            async def is_active(self) -> bool:
                return self._active
        
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield PostgresTransaction(connection)

    async def execute_in_transaction(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async def _transaction_operation(conn: Any) -> Any:
            async with conn.transaction():
                return await operation(*args, **kwargs)
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await _transaction_operation(conn)

    async def bulk_save(self, entities: List[RiskProfile]) -> BulkOperationResult:
        """Пакетное сохранение сущностей."""
        async def _bulk_save_operation(conn: Any) -> BulkOperationResult:
            saved_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity in entities:
                try:
                    await self.save_risk_profile(entity)
                    saved_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    failed_count += 1
                    errors.append({"entity_id": str(entity.id), "error": str(e)})
            
            return BulkOperationResult(
                success_count=saved_count,
                error_count=failed_count,
                processed_ids=processed_ids,
                errors=errors
            )
        
        result = await self._execute_with_retry(_bulk_save_operation)
        return cast(BulkOperationResult, result)

    async def bulk_update(self, entities: List[RiskProfile]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        async def _bulk_update_operation(conn: Any) -> BulkOperationResult:
            updated_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity in entities:
                try:
                    await self.update(entity)
                    updated_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    failed_count += 1
                    errors.append({"entity_id": str(entity.id), "error": str(e)})
            
            return BulkOperationResult(
                success_count=updated_count,
                error_count=failed_count,
                processed_ids=processed_ids,
                errors=errors
            )
        
        result = await self._execute_with_retry(_bulk_update_operation)
        return cast(BulkOperationResult, result)

    async def bulk_delete(self, entity_ids: List[Union[UUID, str]]) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        async def _bulk_delete_operation(conn: Any) -> BulkOperationResult:
            deleted_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity_id in entity_ids:
                try:
                    success = await self.delete(entity_id)
                    if success:
                        deleted_count += 1
                        processed_ids.append(entity_id)
                    else:
                        failed_count += 1
                        errors.append({"entity_id": str(entity_id), "error": "Entity not found"})
                except Exception as e:
                    failed_count += 1
                    errors.append({"entity_id": str(entity_id), "error": str(e)})
            
            return BulkOperationResult(
                success_count=deleted_count,
                error_count=failed_count,
                processed_ids=processed_ids,
                errors=errors
            )
        
        result = await self._execute_with_retry(_bulk_delete_operation)
        return cast(BulkOperationResult, result)

    async def bulk_upsert(self, entities: List[RiskProfile], conflict_fields: List[str]) -> BulkOperationResult:
        """Пакетное upsert сущностей."""
        # Простая реализация через bulk_save
        return await self.bulk_save(entities)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[RiskProfile]:
        """Получить из кэша."""
        if self.cache_service:
            result = await self.cache_service.get(str(key))
            return cast(Optional[RiskProfile], result)
        return None

    async def set_cache(self, key: Union[UUID, str], entity: RiskProfile, ttl: Optional[int] = None) -> None:
        """Установить в кэш."""
        if self.cache_service:
            await self.cache_service.set(str(key), entity, ttl)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        if self.cache_service:
            await self.cache_service.delete(str(key))

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        if self.cache_service:
            await self.cache_service.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        async def _stats_operation(conn: Any) -> RepositoryResponse:
            # Получаем базовую статистику
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(CASE WHEN deleted_at IS NOT NULL THEN 1 END) as deleted_entities
                FROM risk_profiles
            """)
            
            return RepositoryResponse(
                success=True,
                data={
                    "total_entities": stats["total_entities"] if stats else 0,
                    "deleted_entities": stats["deleted_entities"] if stats else 0,
                    "active_entities": (stats["total_entities"] - stats["deleted_entities"]) if stats else 0,
                    "repository_type": "postgres",
                    "connection_status": self._state
                },
                total_count=stats["total_entities"] if stats else 0
            )
        
        result = await self._execute_with_retry(_stats_operation)
        return cast(RepositoryResponse, result)

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        async def _metrics_operation(conn: Any) -> PerformanceMetricsDict:
            # Получаем статистику из БД
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_entities,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_query_time
                FROM risk_profiles 
                WHERE deleted_at IS NULL
            """)
            
            return PerformanceMetricsDict(
                total_trades=stats["total_entities"] if stats else 0,
                winning_trades=0,
                losing_trades=0,
                win_rate="0.0",
                profit_factor="0.0",
                sharpe_ratio="0.0",
                max_drawdown="0.0",
                total_return="0.0",
                average_trade="0.0",
                calmar_ratio="0.0",
                sortino_ratio="0.0",
                var_95="0.0",
                cvar_95="0.0"
            )
        
        result = await self._execute_with_retry(_metrics_operation)
        return cast(PerformanceMetricsDict, result)

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        return RepositoryResponse(
            success=True,
            data={
                "cache_size": 0,  # Будет заполнено отдельно
                "cache_hit_rate": 0.0,
                "cache_misses": 0,
                "cache_hits": 0,
                "total_operations": 0,
            }
        )

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        async def _health_operation(conn: Any) -> HealthCheckDict:
            # Проверяем соединение с БД
            await conn.execute("SELECT 1")
            
            return HealthCheckDict(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                response_time=0.0,
                error_count=0,
                last_error=None,
                uptime=0.0
            )
        
        try:
            result = await self._execute_with_retry(_health_operation)
            return cast(HealthCheckDict, result)
        except Exception as e:
            return HealthCheckDict(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                response_time=0.0,
                error_count=1,
                last_error=str(e),
                uptime=0.0
            )

    def _row_to_risk_profile(self, row: Any) -> RiskProfile:
        """Преобразовать строку БД в объект RiskProfile."""
        return RiskProfile(
            id=row["id"],
            name=row["name"],
            risk_level=RiskLevel(row["risk_level"]),
            max_daily_loss=Decimal(str(row["max_daily_loss"])) if row["max_daily_loss"] else Decimal("0"),
            stop_loss_method=row["stop_loss_method"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_risk_manager(self, row: Any) -> RiskManager:
        """Преобразовать строку БД в объект RiskManager."""
        # Получаем профиль риска
        risk_profile = RiskProfile(
            id=row["risk_profile_id"],
            name="",  # Будет заполнено отдельно
            risk_level=RiskLevel.LOW,  # Будет заполнено отдельно
            max_daily_loss=Decimal("0"),
            stop_loss_method="atr",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return RiskManager(
            risk_profile=risk_profile,
            name=row["name"]
        )

    async def close(self) -> None:
        """Закрыть соединения."""
        if self._pool:
            await self._pool.close()
            self._state = RepositoryState.DISCONNECTED
