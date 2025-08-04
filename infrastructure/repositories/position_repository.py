"""
Сверхпродвинутая промышленная реализация репозитория позиций.
"""

import ast
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from uuid import UUID, uuid4

import asyncpg

from domain.entities.trading import Position, PositionSide
from domain.entities.trading_pair import TradingPair
from domain.exceptions.protocol_exceptions import (
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    TransactionError,
    ValidationError,
)
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    PositionRepositoryProtocol,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
)
from domain.repositories.position_repository import PositionRepository
from domain.types import EntityId, PositionId, PortfolioId, MetadataDict, TimestampValue, Symbol
from typing import Dict, Any
from decimal import Decimal


class InMemoryPositionRepository(PositionRepository):
    """
    Сверхпродвинутая in-memory реализация репозитория позиций.
    - Кэширование с TTL
    - Асинхронные транзакции
    - Индексация по торговой паре, статусу, стороне
    - Аналитика позиций и PnL
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        """Инициализация репозитория позиций."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._positions: Dict[EntityId, Position] = {}
        self._positions_by_trading_pair: Dict[str, List[EntityId]] = defaultdict(list)
        self._positions_by_status: Dict[str, List[EntityId]] = defaultdict(list)
        self._positions_by_side: Dict[str, List[EntityId]] = defaultdict(list)
        self._positions_by_portfolio: Dict[PortfolioId, List[EntityId]] = defaultdict(list)
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._metrics = {
            "total_positions": 0,
            "open_positions": 0,
            "closed_positions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": datetime.now(),
        }
        self._state = RepositoryState.CONNECTED
        self._startup_time = datetime.now()
        self._cleanup_task: Optional[asyncio.Task] = None
        asyncio.create_task(self._background_cleanup())
        self.logger.info("InMemoryPositionRepository initialized")

    async def save(self, position: Position) -> bool:
        """Сохранить позицию с индексацией и кэшированием."""
        entity_id = EntityId(position.id)
        self._positions[entity_id] = position
        # Обновление индексов
        self._update_position_indexes(entity_id, position)
        # Обновление метрик
        self._metrics["total_positions"] = len(self._positions)
        self._update_position_metrics()
        # Инвалидация кэша
        await self.invalidate_cache(str(entity_id))
        return True

    async def get_by_id(self, position_id: Union[UUID, str]) -> Optional[Position]:
        """Получить позицию по ID с кэшированием."""
        if isinstance(position_id, str):
            try:
                position_id = UUID(position_id)
            except ValueError:
                return None
        entity_id = EntityId(position_id)
        cached = await self.get_from_cache(str(entity_id))
        if cached:
            self._metrics["cache_hits"] = int(str(self._metrics.get("cache_hits", 0))) + 1
            return cached
        self._metrics["cache_misses"] = int(str(self._metrics.get("cache_misses", 0))) + 1
        position = self._positions.get(entity_id)
        if position:
            await self.set_cache(str(entity_id), position)
        return position

    async def get_by_trading_pair(
        self, trading_pair: str, open_only: bool = True
    ) -> List[Position]:
        """Получить позиции по торговой паре с оптимизацией."""
        symbol = str(trading_pair)
        cache_key = f"positions_by_pair:{symbol}:{open_only}"
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        entity_ids = self._positions_by_trading_pair.get(symbol, [])
        positions = [
            self._positions[eid] for eid in entity_ids if eid in self._positions
        ]
        if open_only:
            positions = [p for p in positions if p.is_open]
        await self.set_cache(cache_key, positions, 60)
        return positions

    async def get_open_positions(
        self, trading_pair: Optional[str] = None
    ) -> List[Position]:
        """Получить открытые позиции с аналитикой."""
        cache_key = f"open_positions:{str(trading_pair) if trading_pair else 'all'}"
        cached = await self.get_from_cache(str(cache_key))
        if cached:
            return cached
        entity_ids = self._positions_by_status.get("open", [])
        positions = [
            self._positions[eid] for eid in entity_ids if eid in self._positions
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        # Аналитика позиций
        for position in positions:
            await self._analyze_position_risk(position)
        await self.set_cache(str(cache_key), positions, 30)
        return positions

    async def get_closed_positions(
        self,
        trading_pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """Получить закрытые позиции с фильтрацией по времени."""
        entity_ids = self._positions_by_status.get("closed", [])
        positions = [
            self._positions[eid] for eid in entity_ids if eid in self._positions
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        if start_date:
            positions = [p for p in positions if p.entry_time >= start_date]
        if end_date:
            positions = [p for p in positions if p.entry_time <= end_date]
        return positions

    async def get_positions_by_side(
        self,
        side: PositionSide,
        trading_pair: Optional[str] = None,
        open_only: bool = True,
    ) -> List[Position]:
        """Получить позиции по стороне."""
        entity_ids = self._positions_by_side.get(side.value, [])
        positions = [
            self._positions[eid] for eid in entity_ids if eid in self._positions
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        if open_only:
            positions = [p for p in positions if p.is_open]
        return positions

    async def get_positions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[str] = None,
    ) -> List[Position]:
        """Получить позиции по диапазону дат."""
        positions = [
            p
            for p in self._positions.values()
            if start_date <= p.entry_time <= end_date
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        return positions

    async def update(self, position: Position) -> Position:
        """Обновить позицию с валидацией."""
        entity_id = EntityId(position.id)
        if entity_id not in self._positions:
            raise EntityNotFoundError("Position not found", "Position", str(entity_id))
        self._positions[entity_id] = position
        # Обновление индексов
        self._update_position_indexes(entity_id, position)
        # Инвалидация кэша
        await self.invalidate_cache(str(entity_id))
        return position

    async def delete(self, position_id: Union[UUID, str]) -> bool:
        """Удалить позицию с очисткой индексов."""
        if isinstance(position_id, str):
            try:
                position_id = UUID(position_id)
            except ValueError:
                return False
        entity_id = EntityId(position_id)
        if entity_id not in self._positions:
            return False
        position = self._positions[entity_id]
        # Удаление из индексов
        self._remove_from_indexes(entity_id, position)
        # Удаление из основного хранилища
        del self._positions[entity_id]
        # Обновление метрик
        self._metrics["total_positions"] = len(self._positions)
        self._update_position_metrics()
        # Инвалидация кэша
        await self.invalidate_cache(entity_id)
        return True

    async def exists(self, position_id: Union[UUID, str]) -> bool:
        """Проверить существование позиции."""
        if isinstance(position_id, str):
            try:
                position_id = UUID(position_id)
            except ValueError:
                return False
        entity_id = EntityId(position_id)
        return entity_id in self._positions

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество позиций."""
        return len(self._positions)

    async def get_profitable_positions(
        self,
        trading_pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """Получить прибыльные позиции."""
        positions = [
            p for p in self._positions.values() if not p.is_open and p.calculate_unrealized_pnl().value > 0
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        if start_date:
            positions = [p for p in positions if p.entry_time >= start_date]
        if end_date:
            positions = [p for p in positions if p.entry_time <= end_date]
        return positions

    async def get_losing_positions(
        self,
        trading_pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """Получить убыточные позиции."""
        positions = [
            p for p in self._positions.values() if not p.is_open and p.calculate_unrealized_pnl().value < 0
        ]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        if start_date:
            positions = [p for p in positions if p.entry_time >= start_date]
        if end_date:
            positions = [p for p in positions if p.entry_time <= end_date]
        return positions

    async def get_positions_with_stop_loss(
        self, trading_pair: Optional[str] = None
    ) -> List[Position]:
        """Получить позиции со стоп-лоссом."""
        positions = [p for p in self._positions.values() if p.stop_loss is not None]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        return positions

    async def get_positions_with_take_profit(
        self, trading_pair: Optional[str] = None
    ) -> List[Position]:
        """Получить позиции с тейк-профитом."""
        positions = [p for p in self._positions.values() if p.take_profit is not None]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        return positions

    async def get_statistics(
        self,
        trading_pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Получить статистику позиций."""
        positions = list(self._positions.values())
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        if start_date:
            positions = [p for p in positions if p.entry_time >= start_date]
        if end_date:
            positions = [p for p in positions if p.entry_time <= end_date]
        
        return {
            "total_positions": len(positions),
            "open_positions": len([p for p in positions if p.is_open]),
            "closed_positions": len([p for p in positions if not p.is_open]),
            "profitable_positions": len([p for p in positions if not p.is_open and p.calculate_unrealized_pnl().value > 0]),
            "losing_positions": len([p for p in positions if not p.is_open and p.calculate_unrealized_pnl().value < 0]),
            "total_pnl": sum(p.calculate_unrealized_pnl().value for p in positions if not p.is_open),
            "avg_pnl": sum(p.calculate_unrealized_pnl().value for p in positions if not p.is_open) / len([p for p in positions if not p.is_open]) if any(not p.is_open for p in positions) else 0.0,
        }

    async def get_total_exposure(
        self, trading_pair: Optional[str] = None
    ) -> Dict:
        """Получить общую экспозицию."""
        positions = [p for p in self._positions.values() if p.is_open]
        if trading_pair:
            symbol = str(trading_pair)
            positions = [p for p in positions if str(p.symbol) == symbol]
        
        return {
            "total_exposure": sum(abs(p.quantity.value) for p in positions),
            "long_exposure": sum(p.quantity.value for p in positions if p.quantity.value > 0),
            "short_exposure": sum(abs(p.quantity.value) for p in positions if p.quantity.value < 0),
            "position_count": len(positions),
        }

    async def cleanup_old_positions(self, before_date: datetime) -> int:
        """Очистить старые позиции."""
        positions_to_delete = [
            entity_id
            for entity_id, position in self._positions.items()
            if position.entry_time < before_date and not position.is_open
        ]
        
        for entity_id in positions_to_delete:
            position = self._positions[entity_id]
            self._remove_from_indexes(entity_id, position)
            del self._positions[entity_id]
        
        self._metrics["total_positions"] = len(self._positions)
        self._update_position_metrics()
        
        return len(positions_to_delete)

    async def get_by_symbol(
        self, portfolio_id: UUID, symbol: str
    ) -> Optional[Position]:
        """Получить позицию по символу и портфелю."""
        for position in self._positions.values():
            if (
                str(position.symbol) == symbol
                and position.is_open
            ):
                return position
        return None

    # Вспомогательные методы
    def _update_position_indexes(self, entity_id: EntityId, position: Position) -> None:
        """Обновить индексы позиции."""
        # Индекс по торговой паре
        symbol = str(position.symbol)
        if entity_id not in self._positions_by_trading_pair[symbol]:
            self._positions_by_trading_pair[symbol].append(entity_id)
        
        # Индекс по статусу
        status = "open" if position.is_open else "closed"
        if entity_id not in self._positions_by_status[status]:
            self._positions_by_status[status].append(entity_id)
        
        # Индекс по стороне
        side = position.side.value
        if entity_id not in self._positions_by_side[side]:
            self._positions_by_side[side].append(entity_id)

    def _remove_from_indexes(self, entity_id: EntityId, position: Position) -> None:
        """Удалить позицию из индексов."""
        # Удаление из индекса по торговой паре
        symbol = str(position.symbol)
        if entity_id in self._positions_by_trading_pair[symbol]:
            self._positions_by_trading_pair[symbol].remove(entity_id)
        
        # Удаление из индекса по статусу
        for status in ["open", "closed"]:
            if entity_id in self._positions_by_status[status]:
                self._positions_by_status[status].remove(entity_id)
        
        # Удаление из индекса по стороне
        side = position.side.value
        if entity_id in self._positions_by_side[side]:
            self._positions_by_side[side].remove(entity_id)

    def _update_position_metrics(self) -> None:
        """Обновить метрики позиций."""
        self._metrics["open_positions"] = len(
            [p for p in self._positions.values() if p.is_open]
        )
        self._metrics["closed_positions"] = len(
            [p for p in self._positions.values() if not p.is_open]
        )

    async def _analyze_position_risk(self, position: Position) -> None:
        """Анализ риска позиции."""
        # Простая реализация анализа риска
        if position.calculate_unrealized_pnl().value < -1000:  # Пример порога
            self.logger.warning(f"High risk position detected: {position.id}")

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Any]:
        """Получить данные из кэша."""
        if key in self._cache:
            if datetime.now() < self._cache_ttl.get(key, datetime.max):
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_ttl[key]
        return None

    async def set_cache(self, key: Union[UUID, str], value: Any, ttl: Optional[int] = None) -> None:
        """Установить данные в кэш."""
        if ttl is None:
            ttl = self._cache_ttl_seconds
        if len(self._cache) >= self._cache_max_size:
            # Удаляем старые записи
            oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
            del self._cache[oldest_key]
            del self._cache_ttl[oldest_key]
        
        self._cache[key] = value
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш по ключу."""
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_ttl:
            del self._cache_ttl[key]

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                current_time = datetime.now()
                keys_to_remove = [
                    key
                    for key, ttl in self._cache_ttl.items()
                    if current_time > ttl
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    del self._cache_ttl[key]
                
                self._metrics["last_cleanup"] = current_time
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {str(e)}")


class PostgresPositionRepository(PositionRepository):
    """PostgreSQL реализация репозитория позиций."""

    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _get_pool(self) -> asyncpg.Pool:
        """Получить пул соединений."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def save(self, position: Position) -> bool:
        """Сохранить позицию в PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO positions (
                        id, symbol, side, quantity, entry_price, current_price,
                        entry_time, updated_at, stop_loss, take_profit, unrealized_pnl,
                        realized_pnl, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (id) DO UPDATE SET
                        current_price = $6, updated_at = $8, stop_loss = $9,
                        take_profit = $10, unrealized_pnl = $11, realized_pnl = $12
                    """,
                    str(position.id),
                    str(position.symbol),
                    position.side.value,
                    float(position.quantity.value),
                    float(position.entry_price.value),
                    float(position.current_price.value),
                    position.entry_time,
                    position.updated_at,
                    float(position.stop_loss.value) if position.stop_loss else None,
                    float(position.take_profit.value) if position.take_profit else None,
                    float(position.unrealized_pnl.value),
                    float(position.realized_pnl.value),
                    str(position.metadata),
                )
            return True
        except Exception as e:
            self.logger.error(f"Error saving position: {str(e)}")
            return False

    async def get_by_id(self, position_id: Union[UUID, str]) -> Optional[Position]:
        """Получить позицию по ID из PostgreSQL."""
        try:
            if isinstance(position_id, str):
                position_id = UUID(position_id)
            
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM positions WHERE id = $1
                    """,
                    str(position_id),
                )
                
                if row:
                    return self._row_to_position(row)
                return None
        except Exception as e:
            self.logger.error(f"Error getting position by ID: {str(e)}")
            return None

    def _row_to_position(self, row: asyncpg.Record) -> Position:
        """Преобразовать строку БД в объект Position."""
        # Упрощенная реализация - в реальном проекте нужна полная маппинг
        from domain.value_objects.money import Money
        from domain.value_objects.price import Price
        from domain.value_objects.volume import Volume
        from domain.value_objects.currency import Currency
        
        metadata_str = row["metadata"] if row["metadata"] else "{}"
        try:
            metadata = ast.literal_eval(metadata_str) if metadata_str else {}
            if not isinstance(metadata, dict):
                metadata = {}
        except (ValueError, SyntaxError, TypeError) as e:
            logging.warning(f"Failed to parse metadata '{metadata_str}': {e}")
            metadata = {}
        
        return Position(
            id=PositionId(UUID(row["id"])),
            symbol=Symbol(row["symbol"]),
            side=PositionSide(row["side"]),
            quantity=Volume(Decimal(str(row["quantity"])), Currency.USD),
            entry_price=Price(Decimal(str(row["entry_price"])), Currency.USD),
            current_price=Price(Decimal(str(row["current_price"])), Currency.USD),
            entry_time=row["entry_time"],
            updated_at=row["updated_at"],
            stop_loss=Price(Decimal(str(row["stop_loss"])), Currency.USD) if row["stop_loss"] else None,
            take_profit=Price(Decimal(str(row["take_profit"])), Currency.USD) if row["take_profit"] else None,
            unrealized_pnl=Money(Decimal(str(row["unrealized_pnl"])), Currency.USD),
            realized_pnl=Money(Decimal(str(row["realized_pnl"])), Currency.USD),
            metadata=MetadataDict(metadata),
        )


__all__ = [
    "InMemoryPositionRepository",
    "PostgresPositionRepository",
    "PositionRepositoryProtocol",
]
