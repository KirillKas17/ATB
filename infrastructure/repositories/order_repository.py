"""
Сверхпродвинутая промышленная реализация репозитория ордеров.
"""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager, AbstractAsyncContextManager, _AsyncGeneratorContextManager
from typing import AsyncGenerator, Coroutine, AsyncContextManager
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable, cast
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trading_pair import TradingPair
from domain.exceptions.protocol_exceptions import (
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    TransactionError,
    ValidationError,
)
from domain.exceptions.base_exceptions import RepositoryError
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    HealthCheckDict,
    OrderRepositoryProtocol,
    PerformanceMetricsDict,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
    TransactionProtocol,
)
from domain.repositories.base_repository_impl import BaseRepositoryImpl
from domain.type_definitions import OrderId, PortfolioId, Symbol, StrategyId, SignalId, VolumeValue, TradingPair as DomainTradingPair
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import TimestampValue, Timestamp
from domain.type_definitions import MetadataDict


class InMemoryOrderRepository(
    BaseRepositoryImpl[Order],
    OrderRepositoryProtocol,
):
    """
    Сверхпродвинутая in-memory реализация репозитория ордеров.
    - Кэширование с TTL
    - Асинхронные транзакции
    - Индексация по торговой паре, статусу, типу, стороне
    - Аналитика ордеров и исполнения
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        super().__init__()
        self._orders: Dict[OrderId, Order] = {}
        self._orders_by_trading_pair: Dict[str, List[OrderId]] = defaultdict(list)
        self._orders_by_status: Dict[str, List[OrderId]] = defaultdict(list)
        self._orders_by_type: Dict[str, List[OrderId]] = defaultdict(list)
        self._orders_by_side: Dict[str, List[OrderId]] = defaultdict(list)
        self._orders_by_portfolio: Dict[UUID, List[OrderId]] = defaultdict(list)
        self._startup_time: datetime = datetime.now()
        asyncio.create_task(self._background_cleanup())
        self.logger.info("InMemoryOrderRepository initialized")

    async def _save_entity_impl(self, order: Order) -> Order:
        """Реализация сохранения ордера с индексацией."""
        order_id: OrderId = OrderId(UUID(str(order.id)))
        self._orders[order_id] = order
        # Обновление индексов
        self._update_order_indexes(order_id, order)
        return order

    async def _get_entity_by_id_impl(self, order_id: Union[UUID, str]) -> Optional[Order]:
        """Реализация получения ордера по ID."""
        return self._orders.get(OrderId(UUID(str(order_id))))

    async def _update_entity_impl(self, order: Order) -> Order:
        """Реализация обновления ордера."""
        order_id: OrderId = OrderId(UUID(str(order.id)))
        if order_id not in self._orders:
            raise EntityNotFoundError(
                message="Order not found", entity_type="Order", entity_id=str(order.id)
            )
        # Удаление из старых индексов
        old_order: Order = self._orders[order_id]
        self._remove_from_indexes(order_id, old_order)
        # Обновление ордера
        self._orders[order_id] = order
        # Обновление индексов
        self._update_order_indexes(order_id, order)
        return order

    async def _delete_entity_impl(self, order_id: Union[UUID, str]) -> bool:
        """Реализация удаления ордера."""
        order_id_obj = OrderId(UUID(str(order_id)))
        if order_id_obj not in self._orders:
            return False
        order: Order = self._orders[order_id_obj]
        self._remove_from_indexes(order_id_obj, order)
        del self._orders[order_id_obj]
        return True

    def _get_entity_id(self, order: Order) -> OrderId:
        """Получение ID ордера."""
        return OrderId(UUID(str(order.id)))

    async def get_by_trading_pair(
        self, trading_pair: TradingPair, status: Optional[OrderStatus] = None
    ) -> List[Order]:
        """Получить ордеры по торговой паре с фильтрацией по статусу."""
        symbol: str = str(trading_pair)
        cache_key: str = f"orders_by_pair:{symbol}:{status.value if status else 'all'}"
        cached: Optional[List[Order]] = await self.get_from_cache(cache_key)
        if cached:
            return cached
        order_ids: List[OrderId] = self._orders_by_trading_pair.get(symbol, [])
        orders: List[Order] = [
            self._orders[eid] for eid in order_ids if eid in self._orders
        ]
        if status:
            orders = [o for o in orders if o.status == status]
        await self.set_cache(cache_key, orders, ttl=60)
        return orders

    async def get_active_orders(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить активные ордеры с аналитикой."""
        cache_key: str = f"active_orders:{str(trading_pair) if trading_pair else 'all'}"
        cached: Optional[List[Order]] = await self.get_from_cache(cache_key)
        if cached:
            return cached
        active_statuses: List[OrderStatus] = [
            OrderStatus.PENDING,
            OrderStatus.PARTIALLY_FILLED,
        ]
        orders: List[Order] = []
        for status in active_statuses:
            order_ids: List[OrderId] = self._orders_by_status.get(status.value, [])
            orders.extend(
                [self._orders[eid] for eid in order_ids if eid in self._orders]
            )
        if trading_pair:
            symbol: str = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        # Аналитика исполнения
        for order in orders:
            await self._analyze_order_execution(order)
        await self.set_cache(cache_key, orders, ttl=30)
        return orders

    async def get_orders_by_status(self, status: str) -> List[Order]:
        """Получить ордеры по статусу."""
        order_ids: List[OrderId] = self._orders_by_status.get(status, [])
        orders: List[Order] = [
            self._orders[eid] for eid in order_ids if eid in self._orders
        ]
        return orders

    async def get_orders_by_side(
        self, side: OrderSide, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить ордеры по стороне."""
        order_ids: List[OrderId] = self._orders_by_side.get(side.value, [])
        orders: List[Order] = [
            self._orders[eid] for eid in order_ids if eid in self._orders
        ]
        if trading_pair:
            symbol: str = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        return orders

    async def get_orders_by_type(
        self, order_type: OrderType, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить ордеры по типу."""
        order_ids: List[OrderId] = self._orders_by_type.get(order_type.value, [])
        orders: List[Order] = [
            self._orders[eid] for eid in order_ids if eid in self._orders
        ]
        if trading_pair:
            symbol: str = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        return orders

    async def get_orders_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[Order]:
        """Получить ордеры по диапазону дат."""
        orders: List[Order] = [
            o for o in self._orders.values() if start_date <= o.created_at.value <= end_date
        ]
        if trading_pair:
            symbol: str = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        return orders

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество ордеров."""
        if not filters:
            return len(self._orders)
        # Применяем фильтры
        orders: List[Order] = list(self._orders.values())
        for filter_obj in filters:
            orders = self._apply_filter(orders, filter_obj)
        return len(orders)

    def _apply_filter(
        self, orders: List[Order], filter_obj: QueryFilter
    ) -> List[Order]:
        """Применение фильтра к списку ордеров."""
        filtered = []
        for order in orders:
            if self._matches_filter(order, filter_obj):
                filtered.append(order)
        return filtered

    def _matches_filter(self, order: Order, filter_obj: QueryFilter) -> bool:
        """Проверка соответствия ордера фильтру."""
        if filter_obj.field == "status":
            return bool(order.status.value == filter_obj.value)
        elif filter_obj.field == "symbol":
            return bool(str(order.symbol) == filter_obj.value)
        elif filter_obj.field == "side":
            return bool(order.side.value == filter_obj.value)
        elif filter_obj.field == "order_type":
            return bool(order.order_type.value == filter_obj.value)
        elif filter_obj.field == "portfolio_id":
            return bool(str(order.portfolio_id) == filter_obj.value)
        return True

    async def get_filled_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить исполненные ордеры."""
        order_ids = self._orders_by_status.get(OrderStatus.FILLED.value, [])
        orders = [self._orders[eid] for eid in order_ids if eid in self._orders]
        if trading_pair:
            symbol = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        if start_date:
            orders = [o for o in orders if o.created_at.value >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at.value <= end_date]
        return orders

    async def get_cancelled_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить отменённые ордеры."""
        order_ids = self._orders_by_status.get(OrderStatus.CANCELLED.value, [])
        orders = [self._orders[eid] for eid in order_ids if eid in self._orders]
        if trading_pair:
            symbol = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        if start_date:
            orders = [o for o in orders if o.created_at.value >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at.value <= end_date]
        return orders

    async def get_rejected_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить отклонённые ордеры."""
        order_ids = self._orders_by_status.get(OrderStatus.REJECTED.value, [])
        orders = [self._orders[eid] for eid in order_ids if eid in self._orders]
        if trading_pair:
            symbol = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        if start_date:
            orders = [o for o in orders if o.created_at.value >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at.value <= end_date]
        return orders

    async def get_statistics(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Получить статистику ордеров."""
        orders = list(self._orders.values())
        if trading_pair:
            symbol: str = str(trading_pair)
            orders = [o for o in orders if str(o.trading_pair) == symbol]
        if start_date:
            orders = [o for o in orders if o.created_at.value >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at.value <= end_date]
        return {
            "total": len(orders),
            "filled": len([o for o in orders if o.status == OrderStatus.FILLED]),
            "cancelled": len([o for o in orders if o.status == OrderStatus.CANCELLED]),
            "rejected": len([o for o in orders if o.status == OrderStatus.REJECTED]),
            "pending": len([o for o in orders if o.status == OrderStatus.PENDING]),
            "partially_filled": len(
                [o for o in orders if o.status == OrderStatus.PARTIALLY_FILLED]
            ),
        }

    async def cleanup_expired_orders(self, before_date: datetime) -> int:
        """Очистить просроченные ордеры."""
        to_delete = [
            eid for eid, o in self._orders.items() if o.created_at.value < before_date
        ]
        for eid in to_delete:
            order = self._orders[eid]
            self._remove_from_indexes(eid, order)
            del self._orders[eid]
        # Обновление метрик
        self._update_order_metrics()
        return len(to_delete)

    async def create(self, order: Order) -> Order:
        """Создать новый ордер."""
        return await self.save(order)

    async def get_by_portfolio_id(self, portfolio_id: OrderId) -> List[Order]:
        """Получить ордеры по ID портфеля."""
        order_ids = self._orders_by_portfolio.get(portfolio_id, [])
        return [self._orders[eid] for eid in order_ids if eid in self._orders]

    async def count_by_portfolio_id(self, portfolio_id: OrderId) -> int:
        """Подсчитать ордеры по ID портфеля."""
        return len(await self.get_by_portfolio_id(portfolio_id))

    async def find_by_id(self, id: OrderId) -> Optional[Order]:
        """Найти ордер по ID."""
        return await self.get_by_id(id)

    async def find_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Order]:
        """Найти все ордеры с пагинацией."""
        orders = list(self._orders.values())
        if offset:
            orders = orders[offset:]
        if limit:
            orders = orders[:limit]
        return orders

    async def find_by_criteria(
        self, criteria: Dict, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Order]:
        """Найти ордеры по критериям."""
        result = list(self._orders.values())
        for key, value in criteria.items():
            result = [o for o in result if getattr(o, key, None) == value]
        if offset:
            result = result[offset:]
        if limit:
            result = result[:limit]
        return result

    # Кэширование
    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Any]:
        """Получить из кэша."""
        key_str = str(key)
        if key_str in self._cache:
            ttl = self._cache_ttl.get(key_str)
            if ttl and datetime.now() > ttl:
                await self.invalidate_cache(key)
                return None
            return self._cache[key_str]
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: Any, ttl: Optional[int] = None
    ) -> None:
        """Установить кэш."""
        cache_key = str(key)
        ttl_seconds = ttl or self._cache_ttl_seconds
        if len(self._cache) >= self._cache_max_size:
            self._evict_cache()
        self._cache[cache_key] = entity
        self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        cache_key = str(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_ttl:
            del self._cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистить весь кэш."""
        self._cache.clear()
        self._cache_ttl.clear()

    def _evict_cache(self) -> None:
        """Удаление устаревших записей из кэша."""
        current_time = datetime.now()
        expired_keys = []
        for key, cache_entry in self._cache.items():
            if isinstance(cache_entry, tuple) and len(cache_entry) >= 2:
                _, timestamp = cache_entry
                if isinstance(timestamp, datetime) and current_time - timestamp > timedelta(minutes=5):
                    expired_keys.append(key)
        for key in expired_keys:
            del self._cache[key]

    # Вспомогательные методы
    def _update_order_indexes(self, entity_id: OrderId, order: Order) -> None:
        """Обновить индексы ордера."""
        # Индекс по торговой паре
        symbol = str(order.trading_pair)
        if entity_id not in self._orders_by_trading_pair[symbol]:
            self._orders_by_trading_pair[symbol].append(entity_id)
        
        # Индекс по статусу
        status = order.status.value
        if entity_id not in self._orders_by_status[status]:
            self._orders_by_status[status].append(entity_id)
        
        # Индекс по стороне
        side = order.side.value
        if entity_id not in self._orders_by_side[side]:
            self._orders_by_side[side].append(entity_id)
        
        # Индекс по типу
        order_type = order.order_type.value
        if entity_id not in self._orders_by_type[order_type]:
            self._orders_by_type[order_type].append(entity_id)

    def _remove_from_indexes(self, entity_id: OrderId, order: Order) -> None:
        """Удалить ордер из индексов."""
        # Удаление из индекса по торговой паре
        symbol = str(order.trading_pair)
        if entity_id in self._orders_by_trading_pair[symbol]:
            self._orders_by_trading_pair[symbol].remove(entity_id)
        
        # Удаление из индекса по статусу
        status = order.status.value
        if entity_id in self._orders_by_status[status]:
            self._orders_by_status[status].remove(entity_id)
        
        # Удаление из индекса по стороне
        side = order.side.value
        if entity_id in self._orders_by_side[side]:
            self._orders_by_side[side].remove(entity_id)
        
        # Удаление из индекса по типу
        order_type = order.order_type.value
        if entity_id in self._orders_by_type[order_type]:
            self._orders_by_type[order_type].remove(entity_id)

    def _update_order_metrics(self) -> None:
        """Обновление метрик ордеров."""
        # Обновляем метрики через атрибуты объекта RepositoryMetrics
        self._metrics.operation_count = len(self._orders)
        # Остальные метрики сохраняем в custom_metrics
        self._custom_metrics = {
            "active_orders": len([
                o for o in self._orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
            ]),
            "filled_orders": len([
                o for o in self._orders.values() if o.status == OrderStatus.FILLED
            ]),
            "cancelled_orders": len([
                o for o in self._orders.values() if o.status == OrderStatus.CANCELLED
            ]),
            "rejected_orders": len([
                o for o in self._orders.values() if o.status == OrderStatus.REJECTED
            ]),
        }

    async def _analyze_order_execution(self, order: Order) -> None:
        """Анализ исполнения ордера."""
        # Простая реализация анализа исполнения
        if order.status == OrderStatus.FILLED:
            self.logger.info(f"Order {order.id} fully executed")
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            self.logger.info(f"Order {order.id} partially executed")

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                self._evict_cache()
                self._update_order_metrics()
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {str(e)}")


class PostgresOrderRepository(OrderRepositoryProtocol):
    """
    Production-ready PostgreSQL репозиторий ордеров.
    - Асинхронные операции с asyncpg
    - Транзакции и rollback
    - Кэширование с Redis
    - Connection pooling
    - Fault tolerance и retry logic
    - Метрики и мониторинг
    """

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None):
        self.connection_string = connection_string
        self.cache_service = cache_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pool = None
        self._metrics = {
            "queries_executed": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_orders": 0
        }
        self._state = RepositoryState.DISCONNECTED
        self._retry_attempts = 3
        self._retry_delay = 1.0

    async def _get_pool(self) -> Any:
        """Получить connection pool с lazy initialization."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=5,
                    max_size=20,
                    command_timeout=30,
                    server_settings={"application_name": "ATB_OrderRepository"},
                )
                self._state = RepositoryState.CONNECTED
                self.logger.info("PostgreSQL connection pool established")
            except Exception as e:
                self._state = RepositoryState.ERROR
                self.logger.error(f"Failed to create connection pool: {e}")
                raise RepositoryError(f"Database connection failed: {e}")
        return self._pool

    async def _execute_with_retry(self, operation: Callable, *args: Any, **kwargs: Any) -> Any:
        """Выполнить операцию с retry logic."""
        last_exception = None
        for attempt in range(self._retry_attempts):
            try:
                pool = await self._get_pool()
                async with pool.acquire() as conn:
                    result = await operation(conn, *args, **kwargs)
                    self._metrics["queries_executed"] += 1
                    return result
            except Exception as e:
                last_exception = e
                self._metrics["errors"] += 1
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self._retry_attempts - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
        raise RepositoryError(
            f"Operation failed after {self._retry_attempts} attempts: {last_exception}"
        )

    async def _execute_order_operation(self, operation: Callable[..., Order]) -> Order:
        """Типизированная версия для операций с Order."""
        result = await self._execute_with_retry(operation)
        if isinstance(result, Order):
            return result
        raise RepositoryError(f"Expected Order, got {type(result)}")

    async def _execute_order_list_operation(self, operation: Callable[..., List[Order]]) -> List[Order]:
        """Типизированная версия для операций со списком Order."""
        result = await self._execute_with_retry(operation)
        if isinstance(result, list):
            return result
        raise RepositoryError(f"Expected List[Order], got {type(result)}")

    async def _execute_optional_order_operation(self, operation: Callable[..., Optional[Order]]) -> Optional[Order]:
        """Типизированная версия для операций с Optional[Order]."""
        result = await self._execute_with_retry(operation)
        if result is None or isinstance(result, Order):
            return result
        raise RepositoryError(f"Expected Optional[Order], got {type(result)}")

    async def _execute_int_operation(self, operation: Callable[..., int]) -> int:
        """Типизированная версия для операций с int."""
        result = await self._execute_with_retry(operation)
        if isinstance(result, int):
            return result
        raise RepositoryError(f"Expected int, got {type(result)}")

    async def save(self, order: Order) -> Order:
        """Сохранить ордер в PostgreSQL."""

        async def _save_operation(conn: Any) -> Order:
            async with conn.transaction():
                query = """
                INSERT INTO orders (
                    id, portfolio_id, trading_pair, side, order_type, status,
                    price, quantity, filled_quantity, remaining_quantity,
                    created_at, updated_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    filled_quantity = EXCLUDED.filled_quantity,
                    remaining_quantity = EXCLUDED.remaining_quantity,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                RETURNING *
                """
                result = await conn.fetchrow(
                    query,
                    str(order.id),
                    str(order.portfolio_id),
                    str(order.trading_pair),
                    order.side.value,
                    order.order_type.value,
                    order.status.value,
                    float(order.price.amount) if order.price else 0.0,
                    float(order.quantity),
                    float(order.filled_quantity),
                    float(order.remaining_quantity),
                    order.created_at.value,
                    order.updated_at.value,
                    order.metadata,
                )
                # Инвалидация кэша
                if self.cache_service:
                    await self.cache_service.invalidate(f"order:{order.id}")
                return self._row_to_order(result)

        result = await self._execute_with_retry(_save_operation, order)
        if isinstance(result, Order):
            return result
        raise RepositoryError(f"Expected Order from save operation, got {type(result)}")

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Order]:
        """Получить ордер по ID с кэшированием."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        cache_key = f"order:{entity_id}"
        # Попытка получить из кэша
        if self.cache_service:
            cached = await self.cache_service.get(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                return cached
            self._metrics["cache_misses"] += 1

        async def _get_operation(conn: Any) -> Optional[Order]:
            query = "SELECT * FROM orders WHERE id = $1"
            result = await conn.fetchrow(query, str(entity_id))
            if result:
                order = self._row_to_order(result)
                # Сохранение в кэш
                if self.cache_service:
                    await self.cache_service.set(cache_key, order, ttl=300)
                return order
            return None

        result = await self._execute_with_retry(_get_operation)
        if result is None or isinstance(result, Order):
            return result
        raise RepositoryError(f"Expected Optional[Order] from get operation, got {type(result)}")

    async def get_by_trading_pair(
        self, trading_pair: TradingPair, status: Optional[OrderStatus] = None
    ) -> List[Order]:
        """Получить ордеры по торговой паре."""

        async def _get_operation(conn: Any) -> List[Order]:
            if status:
                query = "SELECT * FROM orders WHERE trading_pair = $1 AND status = $2 ORDER BY created_at DESC"
                results = await conn.fetch(query, str(trading_pair), status.value)
            else:
                query = "SELECT * FROM orders WHERE trading_pair = $1 ORDER BY created_at DESC"
                results = await conn.fetch(query, str(trading_pair))
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_active_orders(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить активные ордеры."""

        async def _get_operation(conn: Any) -> List[Order]:
            active_statuses = [
                OrderStatus.PENDING.value,
                OrderStatus.PARTIALLY_FILLED.value,
            ]
            if trading_pair:
                query = """
                SELECT * FROM orders 
                WHERE status = ANY($1) AND trading_pair = $2 
                ORDER BY created_at DESC
                """
                results = await conn.fetch(query, active_statuses, str(trading_pair))
            else:
                query = """
                SELECT * FROM orders 
                WHERE status = ANY($1) 
                ORDER BY created_at DESC
                """
                results = await conn.fetch(query, active_statuses)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_orders_by_status(self, status: str) -> List[Order]:
        """Получить ордеры по статусу."""

        async def _get_operation(conn: Any) -> List[Order]:
            query = "SELECT * FROM orders WHERE status = $1 ORDER BY created_at DESC"
            results = await conn.fetch(query, status)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_orders_by_side(
        self, side: OrderSide, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить ордеры по стороне."""

        async def _get_operation(conn: Any) -> List[Order]:
            if trading_pair:
                query = "SELECT * FROM orders WHERE side = $1 AND trading_pair = $2 ORDER BY created_at DESC"
                results = await conn.fetch(query, side.value, str(trading_pair))
            else:
                query = "SELECT * FROM orders WHERE side = $1 ORDER BY created_at DESC"
                results = await conn.fetch(query, side.value)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_orders_by_type(
        self, order_type: OrderType, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """Получить ордеры по типу."""

        async def _get_operation(conn: Any) -> List[Order]:
            if trading_pair:
                query = "SELECT * FROM orders WHERE order_type = $1 AND trading_pair = $2 ORDER BY created_at DESC"
                results = await conn.fetch(query, order_type.value, str(trading_pair))
            else:
                query = "SELECT * FROM orders WHERE order_type = $1 ORDER BY created_at DESC"
                results = await conn.fetch(query, order_type.value)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_orders_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[Order]:
        """Получить ордеры по диапазону дат."""

        async def _get_operation(conn: Any) -> List[Order]:
            if trading_pair:
                query = """
                SELECT * FROM orders 
                WHERE created_at BETWEEN $1 AND $2 AND trading_pair = $3 
                ORDER BY created_at DESC
                """
                results = await conn.fetch(
                    query, start_date, end_date, str(trading_pair)
                )
            else:
                query = """
                SELECT * FROM orders 
                WHERE created_at BETWEEN $1 AND $2 
                ORDER BY created_at DESC
                """
                results = await conn.fetch(query, start_date, end_date)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def update(self, order: Order) -> Order:
        """Обновить ордер."""

        async def _update_operation(conn: Any) -> Order:
            async with conn.transaction():
                query = """
                UPDATE orders SET
                    status = $2,
                    filled_quantity = $3,
                    remaining_quantity = $4,
                    updated_at = $5,
                    metadata = $6
                WHERE id = $1
                RETURNING *
                """
                result = await conn.fetchrow(
                    query,
                    str(order.id),
                    order.status.value,
                    float(order.filled_quantity),
                    float(order.remaining_quantity),
                    order.updated_at.value,
                    order.metadata,
                )
                if not result:
                    raise EntityNotFoundError(
                        message="Order not found",
                        entity_type="Order",
                        entity_id=str(order.id),
                    )
                # Инвалидация кэша
                if self.cache_service:
                    await self.cache_service.invalidate(f"order:{order.id}")
                return self._row_to_order(result)

        return await self._execute_with_retry(_update_operation)

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить ордер."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        async def _delete_operation(conn: Any) -> bool:
            async with conn.transaction():
                query = "DELETE FROM orders WHERE id = $1 RETURNING id"
                result = await conn.fetchrow(query, str(entity_id))
                if result:
                    # Инвалидация кэша
                    if self.cache_service:
                        await self.cache_service.invalidate(f"order:{entity_id}")
                    return True
                return False

        result = await self._execute_with_retry(_delete_operation)
        return bool(result)

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование ордера."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        async def _exists_operation(conn: Any) -> bool:
            query = "SELECT 1 FROM orders WHERE id = $1"
            result = await conn.fetchval(query, str(entity_id))
            return result is not None

        result = await self._execute_with_retry(_exists_operation)
        return bool(result)

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество ордеров."""

        async def _count_operation(conn: Any) -> int:
            if filters:
                conditions = []
                params = []
                param_count = 1
                for filter_obj in filters:
                    if filter_obj.field == "status":
                        conditions.append(f"status = ${param_count}")
                        params.append(filter_obj.value)
                    elif filter_obj.field == "symbol":
                        conditions.append(f"symbol = ${param_count}")
                        params.append(filter_obj.value)
                    elif filter_obj.field == "side":
                        conditions.append(f"side = ${param_count}")
                        params.append(filter_obj.value)
                    elif filter_obj.field == "order_type":
                        conditions.append(f"order_type = ${param_count}")
                        params.append(filter_obj.value)
                    elif filter_obj.field == "portfolio_id":
                        conditions.append(f"portfolio_id = ${param_count}")
                        params.append(filter_obj.value)
                    param_count += 1
                where_clause = " AND ".join(conditions)
                query = f"SELECT COUNT(*) FROM orders WHERE {where_clause}"
                result = await conn.fetchval(query, *params)
                return result or 0
            else:
                query = "SELECT COUNT(*) FROM orders"
                result = await conn.fetchval(query)
                return result or 0

        return await self._execute_with_retry(_count_operation)

    async def get_filled_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить исполненные ордеры."""

        async def _get_operation(conn: Any) -> List[Order]:
            conditions = ["status = 'FILLED'"]
            params: List[Any] = []
            param_count = 1
            if trading_pair:
                conditions.append(f"trading_pair = ${param_count}")
                params.append(str(trading_pair))
                param_count += 1
            if start_date:
                conditions.append(f"created_at >= ${param_count}")
                params.append(start_date)
                param_count += 1
            if end_date:
                conditions.append(f"created_at <= ${param_count}")
                params.append(end_date)
            where_clause = " AND ".join(conditions)
            query = (
                f"SELECT * FROM orders WHERE {where_clause} ORDER BY created_at DESC"
            )
            results = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_cancelled_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить отменённые ордеры."""

        async def _get_operation(conn: Any) -> List[Order]:
            conditions = ["status = 'CANCELLED'"]
            params: List[Any] = []
            param_count = 1
            if trading_pair:
                conditions.append(f"trading_pair = ${param_count}")
                params.append(str(trading_pair))
                param_count += 1
            if start_date:
                conditions.append(f"created_at >= ${param_count}")
                params.append(start_date)
                param_count += 1
            if end_date:
                conditions.append(f"created_at <= ${param_count}")
                params.append(end_date)
            where_clause = " AND ".join(conditions)
            query = (
                f"SELECT * FROM orders WHERE {where_clause} ORDER BY created_at DESC"
            )
            results = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_rejected_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить отклонённые ордеры."""

        async def _get_operation(conn: Any) -> List[Order]:
            conditions = ["status = 'REJECTED'"]
            params: List[Any] = []
            param_count = 1
            if trading_pair:
                conditions.append(f"trading_pair = ${param_count}")
                params.append(str(trading_pair))
                param_count += 1
            if start_date:
                conditions.append(f"created_at >= ${param_count}")
                params.append(start_date)
                param_count += 1
            if end_date:
                conditions.append(f"created_at <= ${param_count}")
                params.append(end_date)
            where_clause = " AND ".join(conditions)
            query = (
                f"SELECT * FROM orders WHERE {where_clause} ORDER BY created_at DESC"
            )
            results = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def get_statistics(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Получить статистику ордеров."""

        async def _get_operation(conn: Any) -> Dict[str, Any]:
            conditions = []
            params: List[Any] = []
            param_count = 1
            if trading_pair:
                conditions.append(f"trading_pair = ${param_count}")
                params.append(str(trading_pair))
                param_count += 1
            if start_date:
                conditions.append(f"created_at >= ${param_count}")
                params.append(start_date)
                param_count += 1
            if end_date:
                conditions.append(f"created_at <= ${param_count}")
                params.append(end_date)
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
            SELECT 
                COUNT(*) as total_orders,
                COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as filled_orders,
                COUNT(CASE WHEN status = 'CANCELLED' THEN 1 END) as cancelled_orders,
                COUNT(CASE WHEN status = 'REJECTED' THEN 1 END) as rejected_orders,
                COUNT(CASE WHEN status IN ('PENDING', 'PARTIALLY_FILLED') THEN 1 END) as active_orders,
                SUM(CASE WHEN status = 'FILLED' THEN price * filled_quantity ELSE 0 END) as total_volume,
                AVG(CASE WHEN status = 'FILLED' THEN price ELSE NULL END) as avg_price
            FROM orders 
            WHERE {where_clause}
            """
            result = await conn.fetchrow(query, *params)
            return dict(result) if result else {}

        return await self._execute_with_retry(_get_operation)

    async def cleanup_expired_orders(self, before_date: datetime) -> int:
        """Очистить устаревшие ордеры."""

        async def _cleanup_operation(conn: Any) -> int:
            async with conn.transaction():
                query = "DELETE FROM orders WHERE created_at < $1 AND status IN ('FILLED', 'CANCELLED', 'REJECTED')"
                result = await conn.execute(query, before_date)
                return int(result.split()[-1]) if result else 0

        return await self._execute_with_retry(_cleanup_operation)

    async def create(self, order: Order) -> Order:
        """Создать новый ордер."""
        return await self.save(order)

    async def get_by_portfolio_id(self, portfolio_id: OrderId) -> List[Order]:
        """Получить ордеры по ID портфеля."""

        async def _get_operation(conn: Any) -> List[Order]:
            query = (
                "SELECT * FROM orders WHERE portfolio_id = $1 ORDER BY created_at DESC"
            )
            results = await conn.fetch(query, str(portfolio_id))
            return [self._row_to_order(row) for row in results]

        return await self._execute_with_retry(_get_operation)

    async def count_by_portfolio_id(self, portfolio_id: OrderId) -> int:
        """Подсчитать ордеры по ID портфеля."""

        async def _count_operation(conn: Any) -> int:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM orders WHERE portfolio_id = $1", str(portfolio_id)
            )
            return result or 0

        return await self._execute_with_retry(_count_operation)

    async def find_by_id(self, id: OrderId) -> Optional[Order]:
        """Найти ордер по ID."""
        return await self.get_by_id(id)

    async def find_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Order]:
        """Найти все ордеры."""
        async def _find_operation(conn: Any) -> List[Order]:
            query = "SELECT * FROM orders"
            params: List[Any] = []
            if limit:
                query += f" LIMIT ${len(params) + 1}"
                params.append(limit)
            if offset:
                query += f" OFFSET ${len(params) + 1}"
                params.append(offset)
            result = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in result]

        return await self._execute_with_retry(_find_operation)

    async def find_by_criteria(
        self, criteria: Dict, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Order]:
        """Найти ордеры по критериям."""
        async def _find_operation(conn: Any) -> List[Order]:
            query = "SELECT * FROM orders WHERE 1=1"
            params: List[Any] = []
            for key, value in criteria.items():
                query += f" AND {key} = ${len(params) + 1}"
                params.append(value)
            if limit:
                query += f" LIMIT ${len(params) + 1}"
                params.append(limit)
            if offset:
                query += f" OFFSET ${len(params) + 1}"
                params.append(offset)
            result = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in result]

        return await self._execute_with_retry(_find_operation)

    def _row_to_order(self, row: Any) -> Order:
        """Преобразование строки БД в объект Order."""
        return Order(
            id=OrderId(UUID(row['id'])),
            portfolio_id=PortfolioId(UUID(row['portfolio_id'])),
            strategy_id=StrategyId(UUID(row['strategy_id'])),
            signal_id=SignalId(UUID(row['signal_id'])) if row.get('signal_id') else None,
            exchange_order_id=row.get('exchange_order_id'),
            symbol=Symbol(row['symbol']),
            trading_pair=DomainTradingPair(row['trading_pair']),
            order_type=OrderType(row['order_type']),
            side=OrderSide(row['side']),
            amount=Volume(Decimal(str(row['amount'])), Currency.USD),
            quantity=VolumeValue(Decimal(str(row['quantity']))),
            price=Price(Decimal(str(row['price'])), Currency.USD) if row.get('price') else None,
            stop_price=Price(Decimal(str(row['stop_price'])), Currency.USD) if row.get('stop_price') else None,
            status=OrderStatus(row['status']),
            filled_amount=Volume(Decimal(str(row['filled_amount'])), Currency.USD),
            filled_quantity=VolumeValue(Decimal(str(row['filled_quantity']))),
            average_price=Price(Decimal(str(row['average_price'])), Currency.USD) if row.get('average_price') else None,
            commission=Price(Decimal(str(row['commission'])), Currency.USD) if row.get('commission') else None,
            created_at=Timestamp.from_iso(row['created_at']),
            updated_at=Timestamp.from_iso(row['updated_at']),
            filled_at=Timestamp.from_iso(row['filled_at']) if row.get('filled_at') else None,
            metadata={} if not row.get('metadata') else row['metadata'],
        )

    async def get_health_status(self) -> RepositoryState:
        """Получить статус здоровья репозитория."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                self._metrics["last_health_check"] = int(datetime.now().timestamp())
                return RepositoryState.CONNECTED
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._state = RepositoryState.ERROR
            return self._state

    async def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики репозитория."""
        return {
            **self._metrics,
            "state": self._state.value,
            "pool_size": self._pool.get_size() if self._pool else 0,
            "free_size": self._pool.get_free_size() if self._pool else 0,
        }

    async def close(self) -> None:
        """Закрытие соединения с БД."""
        if hasattr(self, '_pool') and self._pool:
            await self._pool.close()  # type: ignore[unreachable]

    # Реализация методов OrderRepositoryProtocol
    async def save_order(self, order: Order) -> bool:
        """Сохранение ордера."""
        try:
            await self.save(order)
            return True
        except Exception:
            return False

    async def get_order(self, order_id: OrderId) -> Optional[Order]:
        """Получение ордера по ID."""
        return await self.get_by_id(order_id)

    async def get_orders_by_symbol(self, symbol: Symbol) -> List[Order]:
        """Получение ордеров по символу."""
        # Реализация через find_by_criteria
        criteria = {"symbol": str(symbol)}
        return await self.find_by_criteria(criteria)

    # Реализация недостающих методов RepositoryProtocol
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Order]:
        """Получить все ордера."""
        limit = None
        offset = None
        if options and options.pagination:
            limit = options.pagination.limit
            offset = options.pagination.offset
        return await self.find_all(limit=limit, offset=offset)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление ордера."""
        # Реализация через обновление статуса
        order = await self.get_by_id(entity_id)
        if order:
            order.status = OrderStatus.CANCELLED
            update_result = await self.update(order)
            return bool(update_result)
        return False

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановление ордера."""
        # Для ордеров восстановление не применимо
        return False

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Order]:
        """Поиск ордеров по фильтрам."""
        # Преобразование QueryFilter в criteria
        criteria = {}
        for filter_obj in filters:
            criteria[filter_obj.field] = filter_obj.value
        limit = None
        offset = None
        if options and options.pagination:
            limit = options.pagination.limit
            offset = options.pagination.offset
        return await self.find_by_criteria(criteria, limit=limit, offset=offset)

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Order]:
        """Поиск одного ордера по фильтрам."""
        results = await self.find_by(filters)
        return results[0] if results else None

    async def check_exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверка существования ордера."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        result = await self.exists(OrderId(entity_id))
        return bool(result)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Order]:
        """Потоковая обработка ордеров."""
        offset = 0
        while True:
            orders = await self.find_all(limit=batch_size, offset=offset)
            if not orders:
                break
            for order in orders:
                yield order
            offset += batch_size

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")  # type: ignore[unreachable]
        
        async with self._pool.acquire() as connection:
                async with connection.transaction() as transaction:
                    class PostgresOrderTransaction(TransactionProtocol):
                        def __init__(self, transaction: Any) -> None:
                            self.transaction = transaction
                            self._active = True
                        async def __aenter__(self) -> "PostgresOrderTransaction":
                            return self
                        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                            if self._active:
                                if exc_type is not None:
                                    await self.transaction.rollback()
                                else:
                                    await self.transaction.commit()
                                self._active = False
                        async def commit(self) -> None:
                            await self.transaction.commit()
                        async def rollback(self) -> None:
                            await self.transaction.rollback()
                        async def is_active(self) -> bool:
                            return self._active
                    yield PostgresOrderTransaction(transaction)

    async def execute_in_transaction(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[Order]) -> BulkOperationResult:
        """Массовое сохранение ордеров."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                await self.save(entity)
                processed_ids.append(entity.id)
            except Exception as e:
                errors.append({"entity_id": str(entity.id), "error": str(e)})
        
        return BulkOperationResult(
            success_count=len(processed_ids),
            error_count=len(errors),
            processed_ids=processed_ids,
            errors=errors
        )

    async def bulk_update(self, entities: List[Order]) -> BulkOperationResult:
        """Массовое обновление ордеров."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                await self.update(entity)
                processed_ids.append(entity.id)
            except Exception as e:
                errors.append({"entity_id": str(entity.id), "error": str(e)})
        
        return BulkOperationResult(
            success_count=len(processed_ids),
            error_count=len(errors),
            processed_ids=processed_ids,
            errors=errors
        )

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Массовое удаление ордеров."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity_id in entity_ids:
            try:
                if await self.delete(entity_id):
                    processed_ids.append(entity_id)
                else:
                    errors.append({"entity_id": str(entity_id), "error": "Entity not found"})
            except Exception as e:
                errors.append({"entity_id": str(entity_id), "error": str(e)})
        
        return BulkOperationResult(
            success_count=len(processed_ids),
            error_count=len(errors),
            processed_ids=processed_ids,
            errors=errors
        )

    async def bulk_upsert(
        self, entities: List[Order], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Массовое upsert ордеров."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                if await self.exists(entity.id):
                    await self.update(entity)
                else:
                    await self.save(entity)
                processed_ids.append(entity.id)
            except Exception as e:
                errors.append({"entity_id": str(entity.id), "error": str(e)})
        
        return BulkOperationResult(
            success_count=len(processed_ids),
            error_count=len(errors),
            processed_ids=processed_ids,
            errors=errors
        )

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Order]:
        """Получение из кэша."""
        if self.cache_service:
            result = await self.cache_service.get(str(key))
            return cast(Optional[Order], result)
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: Order, ttl: Optional[int] = None
    ) -> None:
        """Сохранение в кэш."""
        if self.cache_service:
            await self.cache_service.set(str(key), entity, ttl=ttl or 300)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидация кэша."""
        if self.cache_service:
            await self.cache_service.invalidate(str(key))

    async def clear_cache(self) -> None:
        """Очистка кэша."""
        if self.cache_service:
            await self.cache_service.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получение статистики репозитория."""
        return RepositoryResponse(
            success=True,
            data={"total_entities": await self.count()},
            total_count=await self.count(),
            execution_time=0.0,
            cache_hit=False,
        )

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получение метрик производительности."""
        return {
            # Только допустимые ключи PerformanceMetricsDict
            # Например, total_queries, avg_query_time, cache_hit_rate, error_rate — не входят
        }

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получение статистики кэша."""
        return RepositoryResponse(
            success=True,
            data={"cache_stats": {
                "hits": self._metrics.get("cache_hits", 0),
                "misses": self._metrics.get("cache_misses", 0),
                "size": len(self.cache_service._cache) if self.cache_service else 0,
            }},
            execution_time=0.0,
            cache_hit=False,
        )

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "error_count": 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error_count": 1,
                "last_error": str(e),
            }
