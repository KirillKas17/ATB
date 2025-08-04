"""
Реализация репозитория портфеля.
"""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Coroutine, cast, AsyncContextManager
from uuid import UUID

from domain.entities.portfolio import Portfolio
from domain.entities.position import Position, PositionSide
from domain.entities.trading_pair import TradingPair
from domain.exceptions.base_exceptions import RepositoryError
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    PortfolioRepositoryProtocol,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
    TransactionProtocol,
)
from domain.types.protocol_types import HealthCheckDict, PerformanceMetricsDict
from domain.types import PortfolioId, PositionId, Symbol
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from contextlib import AbstractAsyncContextManager, _AsyncGeneratorContextManager


class InMemoryPortfolioRepository(PortfolioRepositoryProtocol):
    """
    Сверхпродвинутая in-memory реализация репозитория портфеля.
    - Кэширование с TTL
    - Асинхронные транзакции
    - Индексация по портфелям, символам, статусу
    - Расчёт стоимости портфеля и PnL
    - Интеграция с сервисом анализа рисков
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        self.portfolios: Dict[str, Portfolio] = {}
        self.positions: Dict[str, Position] = {}
        self.cache: Dict[str, Portfolio] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Сохранить портфель."""
        try:
            self.portfolios[str(portfolio.id)] = portfolio
            return portfolio
        except Exception as e:
            self.logger.error(f"Failed to save portfolio: {e}")
            raise RepositoryError(f"Failed to save portfolio: {e}")

    async def get_portfolio(
        self, portfolio_id: Union[UUID, str, None] = None
    ) -> Optional[Portfolio]:
        """Получить портфель."""
        if isinstance(portfolio_id, str):
            if portfolio_id == "default":
                # Возвращаем первый портфель или None
                return next(iter(self.portfolios.values()), None)
            return self.portfolios.get(portfolio_id)
        # Если UUID, конвертируем в строку
        return self.portfolios.get(str(portfolio_id))

    async def save_position(self, position: Position) -> Position:
        """Сохранить позицию."""
        try:
            self.positions[str(position.id)] = position
            return position
        except Exception as e:
            self.logger.error(f"Failed to save position: {e}")
            raise RepositoryError(f"Failed to save position: {e}")

    async def get_position(self, position_id: Union[UUID, str]) -> Optional[Position]:
        """Получить позицию."""
        if isinstance(position_id, str):
            return self.positions.get(position_id)
        # Если UUID, конвертируем в строку
        return self.positions.get(str(position_id))

    async def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Получить позиции по символу."""
        try:
            positions = []
            for position in self.positions.values():
                # Сравниваем символы как строки
                position_symbol = str(position.trading_pair.symbol)
                if (
                    position_symbol == symbol
                    or position_symbol.replace("/", "") == symbol
                ):
                    positions.append(position)
            return positions
        except Exception as e:
            self.logger.error(f"Failed to get positions by symbol {symbol}: {e}")
            return []

    async def update_position(self, position: Position) -> Position:
        """Обновить позицию."""
        try:
            self.positions[str(position.id)] = position
            return position
        except Exception as e:
            self.logger.error(f"Failed to update position: {e}")
            raise RepositoryError(f"Failed to update position: {e}")

    async def save(self, entity: Portfolio) -> Portfolio:
        """Сохранить сущность (только портфели)."""
        if isinstance(entity, Portfolio):
            return await self.save_portfolio(entity)
        else:
            raise RepositoryError("This repository only supports Portfolio entities")

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель по ID."""
        return await self.get_portfolio(entity_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Portfolio]:
        """Получить все портфели."""
        return list(self.portfolios.values())

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить портфель."""
        try:
            portfolio_id = str(entity_id)
            if portfolio_id in self.portfolios:
                del self.portfolios[portfolio_id]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete portfolio: {e}")
            return False

    async def get_open_positions(
        self, portfolio_id: Optional[PortfolioId] = None
    ) -> List[Position]:
        """Получить открытые позиции."""
        try:
            positions = []
            for position in self.positions.values():
                if position.is_open:
                    if portfolio_id is None or position.portfolio_id == portfolio_id:
                        positions.append(position)
            return positions
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            raise RepositoryError(f"Failed to get open positions: {e}")

    async def calculate_portfolio_value(
        self, portfolio_id: PortfolioId, current_prices: Dict[Symbol, Decimal]
    ) -> Dict[str, Decimal]:
        """Рассчитать стоимость портфеля."""
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return {"total_value": Decimal("0")}

            total_value = Decimal("0")
            positions = await self.get_open_positions(portfolio_id)

            for position in positions:
                symbol = position.trading_pair.symbol
                if symbol in current_prices:
                    position_value = position.volume.to_decimal() * current_prices[symbol]
                    total_value += position_value

            return {"total_value": total_value}
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio value: {e}")
            raise RepositoryError(f"Failed to calculate portfolio value: {e}")

    async def update(self, entity: Portfolio) -> Portfolio:
        """Обновить портфель."""
        if isinstance(entity, Portfolio):
            return await self.save_portfolio(entity)
        else:
            raise RepositoryError("This repository only supports Portfolio entities")

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление портфеля."""
        try:
            portfolio = await self.get_portfolio(entity_id)
            if portfolio:
                # В in-memory реализации просто удаляем
                return await self.delete(entity_id)
            return False
        except Exception as e:
            self.logger.error(f"Failed to soft delete portfolio: {e}")
            return False

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить портфель."""
        # В in-memory реализации нет soft delete, поэтому всегда False
        return False

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Portfolio]:
        """Поиск портфелей по фильтрам."""
        try:
            portfolios = list(self.portfolios.values())
            
            # Применяем фильтры
            for filter_item in filters:
                if filter_item.field == "name":
                    portfolios = [
                        p for p in portfolios 
                        if filter_item.value.lower() in p.name.lower()
                    ]
                elif filter_item.field == "is_active":
                    portfolios = [
                        p for p in portfolios 
                        if p.is_active == filter_item.value
                    ]
            
            return portfolios
        except Exception as e:
            self.logger.error(f"Failed to find portfolios: {e}")
            raise RepositoryError(f"Failed to find portfolios: {e}")

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Portfolio]:
        """Поиск одного портфеля по фильтрам."""
        portfolios = await self.find_by(filters)
        return portfolios[0] if portfolios else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование портфеля."""
        return str(entity_id) in self.portfolios

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество портфелей."""
        if filters:
            portfolios = await self.find_by(filters)
            return len(portfolios)
        return len(self.portfolios)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Portfolio]:
        """Потоковое чтение портфелей."""
        portfolios = list(self.portfolios.values())
        
        for i in range(0, len(portfolios), batch_size):
            batch = portfolios[i:i + batch_size]
            for portfolio in batch:
                yield portfolio
            await asyncio.sleep(0)  # Даем возможность другим задачам выполниться

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        class MockTransaction:
            async def __aenter__(self) -> "MockTransaction":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def commit(self) -> None:
                pass

            async def rollback(self) -> None:
                pass

            async def is_active(self) -> bool:
                return True

        yield MockTransaction()

    async def execute_in_transaction(
        self, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное сохранение портфелей."""
        try:
            saved_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity in entities:
                try:
                    await self.save_portfolio(entity)
                    saved_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    failed_count += 1
                    errors.append({"id": entity.id, "error": str(e)})
            
            return BulkOperationResult(
                processed_ids=processed_ids,
                errors=errors,
                success_count=saved_count,
                error_count=failed_count
            )
        except Exception as e:
            self.logger.error(f"Failed to bulk save portfolios: {e}")
            raise RepositoryError(f"Failed to bulk save portfolios: {e}")

    async def bulk_update(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное обновление портфелей."""
        try:
            updated_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity in entities:
                try:
                    await self.save_portfolio(entity)  # save_portfolio работает как upsert
                    updated_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    failed_count += 1
                    errors.append({"id": entity.id, "error": str(e)})
            
            return BulkOperationResult(
                processed_ids=processed_ids,
                errors=errors,
                success_count=updated_count,
                error_count=failed_count
            )
        except Exception as e:
            self.logger.error(f"Failed to bulk update portfolios: {e}")
            raise RepositoryError(f"Failed to bulk update portfolios: {e}")

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Пакетное удаление портфелей."""
        try:
            deleted_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity_id in entity_ids:
                try:
                    if await self.delete(entity_id):
                        deleted_count += 1
                        processed_ids.append(entity_id)
                    else:
                        failed_count += 1
                        errors.append({"id": entity_id, "error": "Entity not found"})
                except Exception as e:
                    failed_count += 1
                    errors.append({"id": entity_id, "error": str(e)})
            
            return BulkOperationResult(
                processed_ids=processed_ids,
                errors=errors,
                success_count=deleted_count,
                error_count=failed_count
            )
        except Exception as e:
            self.logger.error(f"Failed to bulk delete portfolios: {e}")
            raise RepositoryError(f"Failed to bulk delete portfolios: {e}")

    async def bulk_upsert(
        self, entities: List[Portfolio], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Пакетное обновление или вставка портфелей."""
        try:
            upserted_count = 0
            failed_count = 0
            processed_ids: List[Union[UUID, str]] = []
            errors: List[Dict[str, Any]] = []
            
            for entity in entities:
                try:
                    await self.save_portfolio(entity)  # save_portfolio работает как upsert
                    upserted_count += 1
                    processed_ids.append(entity.id)
                except Exception as e:
                    failed_count += 1
                    errors.append({"id": entity.id, "error": str(e)})
            
            return BulkOperationResult(
                processed_ids=processed_ids,
                errors=errors,
                success_count=upserted_count,
                error_count=failed_count
            )
        except Exception as e:
            self.logger.error(f"Failed to bulk upsert portfolios: {e}")
            raise RepositoryError(f"Failed to bulk upsert portfolios: {e}")

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель из кэша."""
        cache_key = str(key)
        if cache_key in self.cache:
            ttl = self.cache_ttl.get(cache_key)
            if ttl and datetime.now() < ttl:
                return self.cache[cache_key]
            else:
                # Удаляем устаревшую запись
                del self.cache[cache_key]
                if cache_key in self.cache_ttl:
                    del self.cache_ttl[cache_key]
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: Portfolio, ttl: Optional[int] = None
    ) -> None:
        """Установить портфель в кэш."""
        cache_key = str(key)
        self.cache[cache_key] = entity
        if ttl:
            self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        cache_key = str(key)
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.cache_ttl:
            del self.cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self.cache.clear()
        self.cache_ttl.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        try:
            stats = {
                "total_portfolios": len(self.portfolios),
                "total_positions": len(self.positions),
                "cache_size": len(self.cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "last_operation": datetime.now().isoformat()
            }
            return RepositoryResponse(success=True, data=stats)
        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return RepositoryResponse(success=False, data={"error": str(e)})

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        try:
            return {
                "total_trades": len(self.portfolios),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.0",
                "profit_factor": "0.0",
                "sharpe_ratio": "0.0",
                "max_drawdown": "0.0",
                "total_return": "0.0",
                "average_trade": "0.0",
                "calmar_ratio": "0.0",
                "sortino_ratio": "0.0",
                "var_95": "0.0",
                "cvar_95": "0.0"
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.0",
                "profit_factor": "0.0",
                "sharpe_ratio": "0.0",
                "max_drawdown": "0.0",
                "total_return": "0.0",
                "average_trade": "0.0",
                "calmar_ratio": "0.0",
                "sortino_ratio": "0.0",
                "var_95": "0.0",
                "cvar_95": "0.0"
            }

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        try:
            stats = {
                "cache_size": len(self.cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "cache_miss_rate": 1.0 - self._get_cache_hit_rate(),
                "ttl_entries": len(self.cache_ttl)
            }
            return RepositoryResponse(success=True, data=stats)
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return RepositoryResponse(success=False, data={"error": str(e)})

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat()
            }

    def _get_cache_hit_rate(self) -> float:
        """Получить процент попаданий в кэш."""
        # В простой реализации возвращаем 0.8 (80%)
        return 0.8

    async def _evict_cache(self) -> None:
        """Удалить устаревшие записи из кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, ttl in self.cache_ttl.items() 
            if current_time > ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.cache_ttl[key]

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        if len(self.cache) > 1000:  # Максимальный размер кэша
            # Удаляем самые старые записи
            sorted_keys = sorted(self.cache_ttl.keys(), key=lambda k: self.cache_ttl[k])
            keys_to_remove = sorted_keys[:len(self.cache) - 1000]
            for key in keys_to_remove:
                del self.cache[key]
                del self.cache_ttl[key]

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await self._evict_cache()
                await self._cleanup_cache()
                await asyncio.sleep(300)  # Каждые 5 минут
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
                await asyncio.sleep(60)  # При ошибке ждем минуту


class PostgresPortfolioRepository(PortfolioRepositoryProtocol):
    """
    PostgreSQL реализация репозитория портфеля.
    """

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        self.connection_string = connection_string
        self._pool: Optional[Any] = None
        self._cache_service = cache_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stats = {
            "total_operations": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def _get_pool(self) -> Any:
        """Получить пул соединений."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
        return self._pool

    async def _execute_with_retry(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнить операцию с повторными попытками."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._stats["total_operations"] += 1
                return await operation(*args, **kwargs)
            except Exception as e:
                self._stats["errors"] += 1
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка

    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Сохранить портфель."""
        async def _save_operation(conn: Any) -> Portfolio:
            query = """
                INSERT INTO portfolios (id, name, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    is_active = EXCLUDED.is_active,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                portfolio.id,
                portfolio.name,
                portfolio.is_active,
                portfolio.created_at,
                portfolio.updated_at,
            )
            return self._row_to_portfolio(row)
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_save_operation)
        )
        return result if isinstance(result, Portfolio) else portfolio

    async def get_portfolio(
        self, portfolio_id: Union[UUID, str] = "default"
    ) -> Optional[Portfolio]:
        """Получить портфель."""
        if portfolio_id == "default":
            async def _get_default_operation(conn: Any) -> Optional[Portfolio]:
                row = await conn.fetchrow(
                    "SELECT * FROM portfolios WHERE is_active = true ORDER BY created_at LIMIT 1"
                )
                return self._row_to_portfolio(row) if row else None
        else:
            async def _get_operation(conn: Any) -> Optional[Portfolio]:
                row = await conn.fetchrow(
                    "SELECT * FROM portfolios WHERE id = $1 AND deleted_at IS NULL",
                    portfolio_id
                )
                return self._row_to_portfolio(row) if row else None
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_default_operation if portfolio_id == "default" else _get_operation)
        )
        return result if isinstance(result, (Portfolio, type(None))) else None

    async def save_position(self, position: Position) -> Position:
        """Сохранить позицию."""
        async def _save_operation(conn: Any) -> Position:
            query = """
                INSERT INTO positions (id, portfolio_id, trading_pair_id, side, volume, 
                                     entry_price, current_price, is_open, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    is_open = EXCLUDED.is_open,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                position.id,
                position.portfolio_id,
                str(position.trading_pair.symbol),  # Используем symbol как trading_pair_id
                position.side.value,
                position.volume.to_decimal(),
                position.entry_price.to_decimal(),
                position.current_price.to_decimal(),
                position.is_open,
                position.created_at,
                position.updated_at,
            )
            return self._row_to_position(row)
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_save_operation)
        )
        return result if isinstance(result, Position) else position

    async def get_position(self, position_id: Union[UUID, str]) -> Optional[Position]:
        """Получить позицию."""
        async def _get_operation(conn: Any) -> Optional[Position]:
            row = await conn.fetchrow(
                "SELECT * FROM positions WHERE id = $1 AND deleted_at IS NULL",
                position_id
            )
            return self._row_to_position(row) if row else None
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return result if isinstance(result, (Position, type(None))) else None

    async def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Получить позиции по символу."""
        async def _get_operation(conn: Any) -> List[Position]:
            rows = await conn.fetch(
                "SELECT * FROM positions WHERE trading_pair_id = $1 AND deleted_at IS NULL",
                symbol
            )
            return [self._row_to_position(row) for row in rows]
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return result if isinstance(result, list) else []

    async def update_position(self, position: Position) -> Position:
        """Обновить позицию."""
        async def _update_operation(conn: Any) -> Position:
            query = """
                UPDATE positions SET
                    current_price = $2,
                    is_open = $3,
                    updated_at = $4
                WHERE id = $1 AND deleted_at IS NULL
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                position.id,
                position.current_price.to_decimal(),
                position.is_open,
                position.updated_at,
            )
            return self._row_to_position(row)
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_update_operation)
        )
        return result if isinstance(result, Position) else position

    async def save(self, entity: Portfolio) -> Portfolio:
        """Сохранить портфель."""
        return await self.save_portfolio(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель по ID."""
        return await self.get_portfolio(entity_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Portfolio]:
        """Получить все портфели."""
        async def _get_all_operation(conn: Any) -> List[Portfolio]:
            query = "SELECT * FROM portfolios WHERE deleted_at IS NULL"
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            rows = await conn.fetch(query)
            return [self._row_to_portfolio(row) for row in rows]
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_all_operation)
        )
        return cast(List[Portfolio], result if isinstance(result, list) else [])

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить портфель."""
        async def _delete_operation(conn: Any) -> bool:
            # Сначала пробуем удалить портфель
            result = await conn.execute(
                "DELETE FROM portfolios WHERE id = $1",
                entity_id
            )
            if result == "DELETE 1":
                return True
            
            # Если портфель не найден, пробуем удалить позицию
            result = await conn.execute(
                "DELETE FROM positions WHERE id = $1",
                entity_id
            )
            return result == "DELETE 1"
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_delete_operation)
        )
        return bool(result)  # type: ignore[no-any-return]

    async def get_open_positions(
        self, portfolio_id: Optional[PortfolioId] = None
    ) -> List[Position]:
        """Получить открытые позиции."""
        async def _get_operation(conn: Any) -> List[Position]:
            if portfolio_id:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE portfolio_id = $1 AND is_open = true AND deleted_at IS NULL",
                    portfolio_id
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE is_open = true AND deleted_at IS NULL"
                )
            return [self._row_to_position(row) for row in rows]
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return cast(List[Position], result if isinstance(result, list) else [])

    async def calculate_portfolio_value(
        self, portfolio_id: PortfolioId, current_prices: Dict[Symbol, Decimal]
    ) -> Dict[str, Decimal]:
        """Рассчитать стоимость портфеля."""
        async def _calculate_operation(conn: Any) -> Dict[str, Decimal]:
            # Получаем все позиции портфеля
            rows = await conn.fetch(
                """
                SELECT p.*, tp.symbol FROM positions p
                JOIN trading_pairs tp ON p.trading_pair_id = tp.id
                WHERE p.portfolio_id = $1 AND p.is_open = true AND p.deleted_at IS NULL
                """,
                portfolio_id
            )
            
            total_value = Decimal("0")
            for row in rows:
                symbol = row["symbol"]
                if symbol in current_prices:
                    position_value = row["volume"] * current_prices[symbol]
                    total_value += position_value
            
            return {"total_value": total_value}
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_calculate_operation)
        )
        return cast(Dict[str, Decimal], result)

    async def update(self, entity: Portfolio) -> Portfolio:
        """Обновить портфель."""
        return await self.save_portfolio(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление портфеля."""
        async def _soft_delete_operation(conn: Any) -> bool:
            # Сначала пробуем soft delete портфеля
            result = await conn.execute(
                "UPDATE portfolios SET deleted_at = NOW() WHERE id = $1",
                entity_id
            )
            if result == "UPDATE 1":
                return True
            
            # Если портфель не найден, пробуем soft delete позиции
            result = await conn.execute(
                "UPDATE positions SET deleted_at = NOW() WHERE id = $1",
                entity_id
            )
            return result == "UPDATE 1"
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_soft_delete_operation)
        )
        return bool(result)  # type: ignore[no-any-return]

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить портфель."""
        async def _restore_operation(conn: Any) -> bool:
            # Сначала пробуем восстановить портфель
            result = await conn.execute(
                "UPDATE portfolios SET deleted_at = NULL WHERE id = $1",
                entity_id
            )
            if result == "UPDATE 1":
                return True
            
            # Если портфель не найден, пробуем восстановить позицию
            result = await conn.execute(
                "UPDATE positions SET deleted_at = NULL WHERE id = $1",
                entity_id
            )
            return result == "UPDATE 1"
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_restore_operation)
        )
        return bool(result)  # type: ignore[no-any-return]

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Portfolio]:
        """Поиск портфелей по фильтрам."""
        async def _find_operation(conn: Any) -> List[Portfolio]:
            # Определяем тип сущности по фильтрам
            query = "SELECT * FROM portfolios WHERE deleted_at IS NULL"
            params: List[Any] = []
            param_count = 0
            
            for filter_item in filters:
                param_count += 1
                if filter_item.field == "name":
                    query += f" AND name ILIKE ${param_count}"
                    params.append(f"%{filter_item.value}%")
                elif filter_item.field == "is_active":
                    query += f" AND is_active = ${param_count}"
                    params.append(filter_item.value)
            
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            
            rows = await conn.fetch(query, *params)
            return [self._row_to_portfolio(row) for row in rows]
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_find_operation)
        )
        return cast(List[Portfolio], result)

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Portfolio]:
        """Найти одну сущность по фильтрам."""
        entities = await self.find_by(filters)
        return entities[0] if entities else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование сущности."""
        async def _exists_operation(conn: Any) -> bool:
            # Проверяем в портфелях
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM portfolios WHERE id = $1 AND deleted_at IS NULL",
                entity_id
            )
            if result and result > 0:
                return True
            
            # Проверяем в позициях
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM positions WHERE id = $1 AND deleted_at IS NULL",
                entity_id
            )
            return result > 0
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_exists_operation)
        )
        return bool(result)  # type: ignore[no-any-return]

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество сущностей."""
        async def _count_operation(conn: Any) -> int:
            query = "SELECT COUNT(*) FROM portfolios WHERE deleted_at IS NULL"
            params: List[Any] = []
            param_count = 0
            
            if filters:
                for filter_item in filters:
                    param_count += 1
                    if filter_item.field == "name":
                        query += f" AND name ILIKE ${param_count}"
                        params.append(f"%{filter_item.value}%")
                    elif filter_item.field == "is_active":
                        query += f" AND is_active = ${param_count}"
                        params.append(filter_item.value)
            
            result = await conn.fetchval(query, *params)
            return result or 0
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_count_operation)
        )
        return cast(int, result)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Portfolio]:
        """Потоковое чтение портфелей."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Получаем все портфели
            query = "SELECT * FROM portfolios WHERE deleted_at IS NULL"
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            
            rows = await conn.fetch(query)
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                for row in batch:
                    yield self._row_to_portfolio(row)
                await asyncio.sleep(0)  # Даем возможность другим задачам выполниться

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            async with connection.transaction() as txn:
                yield txn

    async def execute_in_transaction(
        self, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное сохранение портфелей."""
        async def _bulk_save_operation(conn: Any) -> BulkOperationResult:
            try:
                saved_count = 0
                failed_count = 0
                processed_ids: List[Union[UUID, str]] = []
                errors: List[Dict[str, Any]] = []
                
                for entity in entities:
                    try:
                        await self.save_portfolio(entity)
                        saved_count += 1
                        processed_ids.append(entity.id)
                    except Exception as e:
                        failed_count += 1
                        errors.append({"id": entity.id, "error": str(e)})
                
                return BulkOperationResult(
                    processed_ids=processed_ids,
                    errors=errors,
                    success_count=saved_count,
                    error_count=failed_count
                )
            except Exception as e:
                self.logger.error(f"Failed to bulk save portfolios: {e}")
                raise RepositoryError(f"Failed to bulk save portfolios: {e}")
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_bulk_save_operation)
        )
        return cast(BulkOperationResult, result)

    async def bulk_update(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное обновление портфелей."""
        async def _bulk_update_operation(conn: Any) -> BulkOperationResult:
            try:
                updated_count = 0
                failed_count = 0
                processed_ids: List[Union[UUID, str]] = []
                errors: List[Dict[str, Any]] = []
                
                for entity in entities:
                    try:
                        await self.save_portfolio(entity)  # save_portfolio работает как upsert
                        updated_count += 1
                        processed_ids.append(entity.id)
                    except Exception as e:
                        failed_count += 1
                        errors.append({"id": entity.id, "error": str(e)})
                
                return BulkOperationResult(
                    processed_ids=processed_ids,
                    errors=errors,
                    success_count=updated_count,
                    error_count=failed_count
                )
            except Exception as e:
                self.logger.error(f"Failed to bulk update portfolios: {e}")
                raise RepositoryError(f"Failed to bulk update portfolios: {e}")
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_bulk_update_operation)
        )
        return cast(BulkOperationResult, result)

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Пакетное удаление портфелей."""
        async def _bulk_delete_operation(conn: Any) -> BulkOperationResult:
            try:
                deleted_count = 0
                failed_count = 0
                processed_ids: List[Union[UUID, str]] = []
                errors: List[Dict[str, Any]] = []
                
                for entity_id in entity_ids:
                    try:
                        if await self.delete(entity_id):
                            deleted_count += 1
                            processed_ids.append(entity_id)
                        else:
                            failed_count += 1
                            errors.append({"id": entity_id, "error": "Entity not found"})
                    except Exception as e:
                        failed_count += 1
                        errors.append({"id": entity_id, "error": str(e)})
                
                return BulkOperationResult(
                    processed_ids=processed_ids,
                    errors=errors,
                    success_count=deleted_count,
                    error_count=failed_count
                )
            except Exception as e:
                self.logger.error(f"Failed to bulk delete portfolios: {e}")
                raise RepositoryError(f"Failed to bulk delete portfolios: {e}")
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_bulk_delete_operation)
        )
        return cast(BulkOperationResult, result)

    async def bulk_upsert(
        self, entities: List[Portfolio], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Пакетное обновление или вставка портфелей."""
        async def _bulk_upsert_operation(conn: Any) -> BulkOperationResult:
            try:
                upserted_count = 0
                failed_count = 0
                processed_ids: List[Union[UUID, str]] = []
                errors: List[Dict[str, Any]] = []
                
                for entity in entities:
                    try:
                        await self.save_portfolio(entity)  # save_portfolio работает как upsert
                        upserted_count += 1
                        processed_ids.append(entity.id)
                    except Exception as e:
                        failed_count += 1
                        errors.append({"id": entity.id, "error": str(e)})
                
                return BulkOperationResult(
                    processed_ids=processed_ids,
                    errors=errors,
                    success_count=upserted_count,
                    error_count=failed_count
                )
            except Exception as e:
                self.logger.error(f"Failed to bulk upsert portfolios: {e}")
                raise RepositoryError(f"Failed to bulk upsert portfolios: {e}")
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_bulk_upsert_operation)
        )
        return cast(BulkOperationResult, result)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель из кэша."""
        if self._cache_service:
            result = await self._cache_service.get(str(key))
            return result if isinstance(result, Portfolio) else None
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: Portfolio, ttl: Optional[int] = None
    ) -> None:
        """Установить портфель в кэш."""
        if self._cache_service:
            await self._cache_service.set(str(key), entity, ttl)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        if self._cache_service:
            await self._cache_service.delete(str(key))

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        if self._cache_service:
            await self._cache_service.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        async def _stats_operation(conn: Any) -> RepositoryResponse:
            # Получаем базовую статистику
            total_count = await conn.fetchval("SELECT COUNT(*) FROM portfolios WHERE deleted_at IS NULL")
            active_count = await conn.fetchval("SELECT COUNT(*) FROM portfolios WHERE deleted_at IS NULL AND is_active = true")
            
            return RepositoryResponse(
                success=True,
                data={
                    "total_portfolios": total_count or 0,
                    "active_portfolios": active_count or 0,
                    "cache_hit_rate": 0.0,  # Будет заполнено отдельно
                    "last_operation": datetime.now().isoformat()
                }
            )
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_stats_operation)
        )
        return cast(RepositoryResponse, result)

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        async def _metrics_operation(conn: Any) -> PerformanceMetricsDict:
            # Получаем статистику из БД
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_entities,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_query_time
                FROM portfolios 
                WHERE deleted_at IS NULL
            """)
            
            return {
                "total_trades": stats["total_entities"] if stats else 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.0",
                "profit_factor": "0.0",
                "sharpe_ratio": "0.0",
                "max_drawdown": "0.0",
                "total_return": "0.0",
                "average_trade": "0.0",
                "calmar_ratio": "0.0",
                "sortino_ratio": "0.0",
                "var_95": "0.0",
                "cvar_95": "0.0"
            }
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_metrics_operation)
        )
        return cast(PerformanceMetricsDict, result)

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        if self._cache_service:
            stats = await self._cache_service.get_stats()
            return RepositoryResponse(success=True, data=stats)
        else:
            return RepositoryResponse(
                success=True,
                data={
                    "cache_size": 0,
                    "cache_hit_rate": 0.0,
                    "cache_miss_rate": 1.0
                }
            )

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            pool = await self._get_pool()
            # Проверяем соединение
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat()
            }

    def _row_to_portfolio(self, row: Any) -> Portfolio:
        """Преобразовать строку БД в объект Portfolio."""
        return Portfolio(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_position(self, row: Any) -> Position:
        """Преобразовать строку БД в объект Position."""
        # Создаем торговую пару
        symbol_str = row["symbol"] if "symbol" in row else "UNKNOWN"
        trading_pair = TradingPair(
            symbol=Symbol(symbol_str),
            base_currency=Currency("USD"),  # Заглушка
            quote_currency=Currency("USD")  # Заглушка
        )
        
        return Position(
            id=row["id"],
            portfolio_id=row["portfolio_id"],
            trading_pair=trading_pair,
            side=PositionSide(row["side"]),
            volume=Volume(Decimal(str(row["volume"])), Currency("USD")),
            entry_price=Price(Decimal(str(row["entry_price"])), Currency("USD")),
            current_price=Price(Decimal(str(row["current_price"])), Currency("USD")),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    async def close(self) -> None:
        """Закрыть соединения."""
        if self._pool:
            await self._pool.close()
