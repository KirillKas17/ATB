# -*- coding: utf-8 -*-
"""
Торговый репозиторий - реализация хранения торговых данных.
"""

import ast
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, cast, AsyncIterator
from uuid import UUID
from contextlib import AbstractAsyncContextManager

from domain.entities.account import Account
from domain.entities.order import Order, OrderId, OrderSide, OrderStatus, OrderType
from domain.entities.position import Position, PositionId, PositionSide
from domain.entities.trading import Trade, OrderSide as TradingOrderSide

from domain.types import Symbol, VolumeValue, PortfolioId, StrategyId, SignalId, TradeId, OrderId as DomainOrderId, TimestampValue, TradingPair as TradingPairType
from domain.entities.trading_pair import TradingPair
from domain.types.base_types import TimestampValue as BaseTimestampValue
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.repositories.base_repository_impl import BaseRepositoryImpl
from infrastructure.repositories.trading.trading_repository_protocol import (
    TradingRepositoryProtocol,
    RepositoryResult,
)
from infrastructure.repositories.trading.trading_repository_services import (
    TradingRepositoryServices,
)
from infrastructure.repositories.trading.trading_pattern_analyzer import (
    TradingPatternAnalyzer,
)
from infrastructure.repositories.trading.liquidity_analyzer import (
    ConcreteLiquidityAnalyzer,
)


class InMemoryTradingRepository(TradingRepositoryProtocol):
    """In-memory реализация торгового репозитория."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._trades: Dict[str, Trade] = {}
        self._trading_pairs: Dict[str, TradingPair] = {}
        self._accounts: Dict[str, Account] = {}
        # Инициализация сервисов
        self.services = TradingRepositoryServices()
        self.pattern_analyzer = TradingPatternAnalyzer()
        self.liquidity_analyzer = ConcreteLiquidityAnalyzer()
        # Метрики
        self._order_count = 0
        self._position_count = 0
        self._trade_count = 0
        self._total_volume = Decimal("0")
        # Кэш для быстрого доступа
        self._trades_by_symbol: Dict[str, List[str]] = {}
        self._trades_by_order: Dict[str, List[str]] = {}
        self._balance_cache: Dict[str, Dict[str, Money]] = {}
        self._balance_cache_ttl: Dict[str, datetime] = {}
        self._balance_cache_duration = timedelta(minutes=5)

    async def add_order(self, order: Order) -> RepositoryResult:
        """Добавление ордера."""
        try:
            # Используем доменный сервис валидации вместо инфраструктурной логики
            from domain.services.order_validation_service import (
                AccountData,
                DefaultOrderBusinessRuleValidator,
                OrderDataValidator,
            )

            # Валидация данных ордера
            data_validator = OrderDataValidator()
            data_validation = data_validator.validate_order_data(order.to_dict())
            if not data_validation.is_valid:
                return RepositoryResult(
                    success=False,
                    error_message=f"Data validation failed: {', '.join(data_validation.errors)}",
                )
            # Валидация бизнес-правил (если есть данные аккаунта)
            if hasattr(self, "account_data") and self.account_data:
                business_validator = DefaultOrderBusinessRuleValidator()
                account_data = AccountData(
                    balance=self.account_data.get(
                        "balance", Money(Decimal("0"), Currency("USDT"))
                    ),
                    available_balance=self.account_data.get(
                        "available_balance", Money(Decimal("0"), Currency("USDT"))
                    ),
                    margin_used=self.account_data.get(
                        "margin_used", Money(Decimal("0"), Currency("USDT"))
                    ),
                    open_positions=self.account_data.get("open_positions", 0),
                    max_positions=self.account_data.get("max_positions", 10),
                    leverage=self.account_data.get("leverage", Decimal("1")),
                    risk_level=self.account_data.get("risk_level", "medium"),
                )
                # Исправление: преобразуем AccountData в dict с правильными типами
                account_dict: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]] = {
                    "balance": str(account_data.balance.amount),
                    "available_balance": str(account_data.available_balance.amount),
                    "margin_used": str(account_data.margin_used.amount),
                    "open_positions": account_data.open_positions,
                    "max_positions": account_data.max_positions,
                    "leverage": float(account_data.leverage),
                    "risk_level": account_data.risk_level,
                }
                business_validation = business_validator.validate_order_business_rules(
                    order.to_dict(), account_dict
                )
                if not business_validation.is_valid:
                    return RepositoryResult(
                        success=False,
                        error_message=f"Business rules validation failed: {', '.join(business_validation.errors)}",
                    )
            # Добавление в хранилище
            order_id = str(order.id)
            self._orders[order_id] = order
            # Обновление метрик
            self._order_count += 1
            return RepositoryResult(success=True, data=order)
        except Exception as e:
            return RepositoryResult(
                success=False, error_message=f"Failed to add order: {str(e)}"
            )

    async def get_order(self, order_id: Union[str, UUID]) -> Optional[Order]:
        """Получение ордера по ID."""
        try:
            order_id_str = str(order_id)
            if order_id_str not in self._orders:
                return None
            return self._orders[order_id_str]
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
            return None

    async def update_order(
        self, order_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> RepositoryResult:
        """Обновление ордера."""
        try:
            order_id_str = str(order_id)
            if order_id_str not in self._orders:
                return RepositoryResult(
                    success=False, error_message=f"Order not found: {order_id_str}"
                )
            order = self._orders[order_id_str]
            # Применение обновлений
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            order.updated_at = Timestamp.now()
            self._orders[order_id_str] = order
            self.logger.info(f"Order updated: {order_id_str}")
            return RepositoryResult(success=True, data=order)
        except Exception as e:
            self.logger.error(f"Error updating order: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to update order: {str(e)}"
            )

    async def delete_order(self, order_id: Union[str, UUID]) -> RepositoryResult:
        """Удаление ордера."""
        try:
            order_id_str = str(order_id)
            if order_id_str not in self._orders:
                return RepositoryResult(
                    success=False, error_message=f"Order not found: {order_id_str}"
                )
            order = self._orders[order_id_str]
            # Обновление метрик
            if order.price:
                self._total_volume -= order.quantity * order.price.amount
            self._order_count -= 1
            del self._orders[order_id_str]
            self.logger.info(f"Order deleted: {order_id_str}")
            return RepositoryResult(success=True, data=True)
        except Exception as e:
            self.logger.error(f"Error deleting order: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to delete order: {str(e)}"
            )

    async def list_orders(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> RepositoryResult:
        """Получение списка ордеров с фильтрацией."""
        try:
            orders = list(self._orders.values())
            # Фильтрация по статусу
            if status:
                orders = [o for o in orders if o.status == status]
            # Фильтрация по стороне
            if side:
                orders = [o for o in orders if o.side == side]
            # Фильтрация по торговой паре
            if trading_pair_id:
                trading_pair_id_str = str(trading_pair_id)
                orders = [
                    o for o in orders if str(o.trading_pair) == trading_pair_id_str
                ]
            # Пагинация
            if offset:
                orders = orders[offset:]
            if limit:
                orders = orders[:limit]
            return RepositoryResult(success=True, data=orders)
        except Exception as e:
            self.logger.error(f"Error listing orders: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to list orders: {str(e)}"
            )

    async def add_position(self, position: Position) -> RepositoryResult:
        """Добавление позиции."""
        try:
            position_id = str(position.id)
            self._positions[position_id] = position
            # Обновление метрик
            self._position_count += 1
            return RepositoryResult(success=True, data=position)
        except Exception as e:
            return RepositoryResult(
                success=False, error_message=f"Failed to add position: {str(e)}"
            )

    async def get_position(self, position_id: Union[str, UUID]) -> RepositoryResult:
        """Получение позиции по ID."""
        try:
            position_id_str = str(position_id)
            if position_id_str not in self._positions:
                return RepositoryResult(
                    success=False,
                    error_message=f"Position not found: {position_id_str}",
                )
            position = self._positions[position_id_str]
            return RepositoryResult(success=True, data=position)
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to get position: {str(e)}"
            )

    async def update_position(
        self, position_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> RepositoryResult:
        """Обновление позиции."""
        try:
            position_id_str = str(position_id)
            if position_id_str not in self._positions:
                return RepositoryResult(
                    success=False, error_message=f"Position not found: {position_id_str}"
                )
            position = self._positions[position_id_str]
            # Применение обновлений
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            position.updated_at = Timestamp.now()
            self._positions[position_id_str] = position
            self.logger.info(f"Position updated: {position_id_str}")
            return RepositoryResult(success=True, data=position)
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to update position: {str(e)}"
            )

    async def delete_position(self, position_id: Union[str, UUID]) -> RepositoryResult:
        """Удаление позиции."""
        try:
            position_id_str = str(position_id)
            if position_id_str not in self._positions:
                return RepositoryResult(
                    success=False, error_message=f"Position not found: {position_id_str}"
                )
            position = self._positions[position_id_str]
            # Обновление метрик
            self._position_count -= 1
            del self._positions[position_id_str]
            self.logger.info(f"Position deleted: {position_id_str}")
            return RepositoryResult(success=True, data=True)
        except Exception as e:
            self.logger.error(f"Error deleting position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to delete position: {str(e)}"
            )

    async def list_positions(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> RepositoryResult:
        """Получение списка позиций с фильтрацией."""
        try:
            positions = list(self._positions.values())
            # Фильтрация по стороне
            if side:
                positions = [p for p in positions if str(p.side) == str(side)]
            # Фильтрация по торговой паре
            if trading_pair_id:
                trading_pair_id_str = str(trading_pair_id)
                positions = [
                    p for p in positions if str(p.trading_pair) == trading_pair_id_str
                ]
            # Пагинация
            if offset:
                positions = positions[offset:]
            if limit:
                positions = positions[:limit]
            return RepositoryResult(success=True, data=positions)
        except Exception as e:
            self.logger.error(f"Error listing positions: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to list positions: {str(e)}"
            )

    async def save_trade(self, trade: Trade) -> Trade:
        """Сохранение сделки."""
        try:
            trade_id = str(trade.id)
            self._trades[trade_id] = trade
            self._trade_count += 1
            
            # Обновление индексов
            symbol = str(trade.symbol)  # type: ignore[attr-defined]
            if symbol not in self._trades_by_symbol:
                self._trades_by_symbol[symbol] = []
            self._trades_by_symbol[symbol].append(trade_id)
            
            order_id = str(trade.order_id)
            if order_id not in self._trades_by_order:
                self._trades_by_order[order_id] = []
            self._trades_by_order[order_id].append(trade_id)
            
            # Обновление метрик
            if hasattr(trade, 'quantity') and hasattr(trade, 'price'):
                self._total_volume += trade.quantity * trade.price.amount  # type: ignore[operator]
            
            # Инвалидация кэша баланса
            await self._invalidate_balance_cache()
            
            self.logger.info(f"Trade saved: {trade_id}")
            return trade
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            raise

    async def get_trades_by_order(self, order_id: OrderId) -> List[Trade]:
        """Получение сделок по ордеру."""
        try:
            order_id_str = str(order_id)
            trade_ids = self._trades_by_order.get(order_id_str, [])
            trades = [self._trades[trade_id] for trade_id in trade_ids if trade_id in self._trades]
            return sorted(trades, key=lambda t: t.timestamp.value, reverse=True)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.error(f"Error getting trades by order: {e}")
            return []

    async def get_trades_by_symbol(
        self,
        symbol: Symbol,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Получение сделок по символу."""
        try:
            symbol_str = str(symbol)
            trade_ids = self._trades_by_symbol.get(symbol_str, [])
            trades = [self._trades[trade_id] for trade_id in trade_ids if trade_id in self._trades]
            
            # Фильтрация по датам
            if start_date:
                trades = [t for t in trades if t.timestamp.value >= start_date]  # type: ignore[attr-defined]
            if end_date:
                trades = [t for t in trades if t.timestamp.value <= end_date]  # type: ignore[attr-defined]
            
            # Сортировка по времени (новые сначала)
            trades = sorted(trades, key=lambda t: t.timestamp.value, reverse=True)  # type: ignore[attr-defined]
            
            # Применение лимита
            if limit:
                trades = trades[:limit]
            
            return trades
        except Exception as e:
            self.logger.error(f"Error getting trades by symbol: {e}")
            return []

    async def get_trade(
        self, 
        symbol: Optional[Symbol] = None, 
        limit: int = 100,
        account_id: Optional[str] = None
    ) -> List[Trade]:
        """Получение сделок с фильтрацией."""
        try:
            if symbol:
                return await self.get_trades_by_symbol(symbol, limit=limit)
            
            # Получение всех сделок
            trades = list(self._trades.values())
            
            # Фильтрация по аккаунту (если есть связь с аккаунтом)
            if account_id:
                # Здесь можно добавить фильтрацию по аккаунту, если есть связь
                pass
            
            # Сортировка по времени (новые сначала)
            trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)
            
            # Применение лимита
            return trades[:limit]
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return []

    async def get_balance(self, account_id: Optional[str] = None) -> Dict[str, Money]:
        """Получение баланса аккаунта."""
        try:
            # Используем основной аккаунт, если не указан конкретный
            target_account_id = account_id or "default"
            
            # Проверяем кэш
            if target_account_id in self._balance_cache:
                cache_time = self._balance_cache_ttl.get(target_account_id)
                if cache_time and datetime.now() - cache_time < self._balance_cache_duration:
                    return self._balance_cache[target_account_id]
            
            # Получаем аккаунт
            account = self._accounts.get(target_account_id)
            if not account:
                # Возвращаем пустой баланс для нового аккаунта
                empty_balance = {
                    "USDT": Money(Decimal("0"), Currency.USDT),
                    "USD": Money(Decimal("0"), Currency.USD),
                    "BTC": Money(Decimal("0"), Currency.BTC),
                    "ETH": Money(Decimal("0"), Currency.ETH),
                }
                self._balance_cache[target_account_id] = empty_balance
                self._balance_cache_ttl[target_account_id] = datetime.now()
                return empty_balance
            
            # Формируем баланс из аккаунта
            balance = {}
            for bal in account.balances:
                balance[bal.currency] = Money(bal.total, Currency(bal.currency))
            
            # Кэшируем результат
            self._balance_cache[target_account_id] = balance
            self._balance_cache_ttl[target_account_id] = datetime.now()
            
            return balance
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return {}

    async def save_account(self, account: Account) -> Account:
        """Сохранение аккаунта."""
        try:
            account_id = account.id
            self._accounts[account_id] = account
            
            # Инвалидация кэша баланса
            await self._invalidate_balance_cache()
            
            self.logger.info(f"Account saved: {account_id}")
            return account
        except Exception as e:
            self.logger.error(f"Error saving account: {e}")
            raise

    async def get_account(self, account_id: str) -> Optional[Account]:
        """Получение аккаунта по ID."""
        try:
            return self._accounts.get(account_id)
        except Exception as e:
            self.logger.error(f"Error getting account: {e}")
            return None

    async def update_account_balance(
        self, 
        account_id: str, 
        currency: str, 
        amount: Money
    ) -> bool:
        """Обновление баланса аккаунта."""
        try:
            account = self._accounts.get(account_id)
            if not account:
                return False
            
            # Обновляем баланс в аккаунте
            for balance in account.balances:
                if balance.currency == currency:
                    balance.total = amount.value  # type: ignore[misc]
                    break
            else:
                # Добавляем новый баланс, если валюта не найдена
                from domain.entities.account import Balance
                new_balance = Balance(currency, amount.value, Decimal("0"))
                account.balances.append(new_balance)
            
            # Инвалидация кэша баланса
            await self._invalidate_balance_cache()
            
            self.logger.info(f"Account balance updated: {account_id} {currency}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating account balance: {e}")
            return False

    async def _invalidate_balance_cache(self) -> None:
        """Инвалидация кэша баланса."""
        self._balance_cache.clear()
        self._balance_cache_ttl.clear()

    async def get_trading_metrics(self) -> RepositoryResult:
        """Получение торговых метрик."""
        try:
            # Расчет статистики по сделкам
            total_trades = len(self._trades)
            total_trade_volume = sum(
                float(trade.quantity.value) * float(trade.price.amount) 
                for trade in self._trades.values() 
                if hasattr(trade, 'quantity') and hasattr(trade, 'price')
            )
            
            # Статистика по валютам в сделках
            trade_currencies = {}
            for trade in self._trades.values():
                if hasattr(trade, 'price') and hasattr(trade.price, 'currency'):
                    currency = trade.price.currency
                    if currency not in trade_currencies:
                        trade_currencies[currency] = 0
                    trade_currencies[currency] += 1
            
            metrics = {
                "total_orders": len(self._orders),
                "total_positions": len(self._positions),
                "total_trades": total_trades,
                "open_orders": len(
                    [o for o in self._orders.values() if o.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]]
                ),
                "open_positions": len(
                    [p for p in self._positions.values() if p.closed_at is None]
                ),
                "total_volume": float(self._total_volume),
                "total_trade_volume": float(total_trade_volume),
                "trade_currencies": trade_currencies,
                "accounts_count": len(self._accounts),
            }
            return RepositoryResult(success=True, data=metrics)
        except Exception as e:
            self.logger.error(f"Error getting trading metrics: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to get trading metrics: {str(e)}"
            )

    async def clear_all_data(self) -> RepositoryResult:
        """Очистка всех данных."""
        try:
            self._orders.clear()
            self._positions.clear()
            self._trades.clear()
            self._trading_pairs.clear()
            self._accounts.clear()
            self._trades_by_symbol.clear()
            self._trades_by_order.clear()
            self._balance_cache.clear()
            self._balance_cache_ttl.clear()
            self._order_count = 0
            self._position_count = 0
            self._trade_count = 0
            self._total_volume = Decimal("0")
            self.logger.info("All data cleared")
            return RepositoryResult(success=True, data=True)
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to clear data: {str(e)}"
            )


class PostgresTradingRepository(TradingRepositoryProtocol):
    """PostgreSQL реализация торгового репозитория."""

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        self.connection_string = connection_string
        self.cache_service = cache_service
        self._pool = None
        self._metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        # Инициализируем базу данных при создании
        asyncio.create_task(self._init_database())

    async def _get_pool(self) -> Any:
        """Получение пула соединений."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(self.connection_string)
                self.logger.info("PostgreSQL connection pool created")
            except Exception as e:
                self.logger.error(f"Failed to create connection pool: {e}")
                raise
        return self._pool

    async def _init_database(self) -> None:
        """Инициализация базы данных."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Читаем SQL схему из файла
                schema_path = __file__.replace('.py', '_schema.sql')
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    await conn.execute(schema_sql)
                    self.logger.info("Database schema initialized successfully")
                except FileNotFoundError:
                    self.logger.warning("Schema file not found, using basic table creation")
                    # Создаем базовые таблицы, если файл схемы не найден
                    await self._create_basic_tables(conn)
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    async def _create_basic_tables(self, conn: Any) -> None:
        """Создание базовых таблиц."""
        # Таблица сделок
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id UUID PRIMARY KEY,
                order_id UUID NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                fee DECIMAL(20, 8) DEFAULT 0,
                fee_currency VARCHAR(10) DEFAULT 'USDT',
                status VARCHAR(20) DEFAULT 'FILLED'
            )
        """)
        
        # Таблица аккаунтов
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id VARCHAR(100) PRIMARY KEY,
                name VARCHAR(255),
                type VARCHAR(20) DEFAULT 'SPOT',
                status VARCHAR(20) DEFAULT 'ACTIVE',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Таблица балансов
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS account_balances (
                account_id VARCHAR(100) NOT NULL,
                currency VARCHAR(10) NOT NULL,
                total_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
                available_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
                PRIMARY KEY (account_id, currency)
            )
        """)
        
        self.logger.info("Basic tables created successfully")

    async def _execute_with_retry(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции с повторными попытками."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pool = await self._get_pool()
                async with pool.acquire() as conn:
                    result = await operation(conn, *args, **kwargs)
                    queries_executed = self._metrics.get("queries_executed", 0)
                    self._metrics["queries_executed"] = int(queries_executed) if isinstance(queries_executed, (int, float, str)) else 1
                    return result
            except Exception as e:
                errors = self._metrics.get("errors", 0)
                self._metrics["errors"] = int(errors) if isinstance(errors, (int, float, str)) else 1
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2**attempt))

    async def add_order(self, order: Order) -> RepositoryResult:
        """Добавление ордера в PostgreSQL."""
        try:

            async def _add_operation(conn: Any) -> None:
                query = """
                INSERT INTO orders (
                    id, portfolio_id, strategy_id, signal_id, exchange_order_id,
                    symbol, trading_pair, order_type, side, amount, quantity,
                    price, stop_price, status, filled_amount, filled_quantity,
                    average_price, commission, created_at, updated_at, filled_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                """
                await conn.execute(
                    query,
                    str(order.id),
                    str(order.portfolio_id),
                    str(order.strategy_id),
                    str(order.signal_id) if order.signal_id else None,
                    order.exchange_order_id,
                    str(order.symbol),
                    str(order.trading_pair),
                    order.order_type.value,
                    order.side.value,
                    str(order.amount.to_decimal()),
                    str(order.quantity),
                    str(order.price.amount) if order.price else None,
                    str(order.stop_price.amount) if order.stop_price else None,
                    order.status.value,
                    str(order.filled_amount.to_decimal()),
                    str(order.filled_quantity),
                    str(order.average_price.amount) if order.average_price else None,
                    str(order.commission.amount) if order.commission else None,
                    order.created_at.to_iso(),
                    order.updated_at.to_iso(),
                    order.filled_at.to_iso() if order.filled_at else None,
                    str(order.metadata),
                )

            await self._execute_with_retry(_add_operation)
            # Кэширование
            if self.cache_service:
                await self.cache_service.set(str(order.id), order)
            return RepositoryResult(success=True, data=order)
        except Exception as e:
            self.logger.error(f"Error adding order: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to add order: {str(e)}"
            )

    async def get_order(self, order_id: Union[str, UUID]) -> Optional[Order]:
        """Получение ордера из PostgreSQL."""
        try:
            order_id_str = str(order_id)
            # Проверка кэша
            if self.cache_service:
                cached_order = await self.cache_service.get(order_id_str)
                if cached_order:
                    # Явное приведение типов для арифметических операций
                    cache_hits = self._metrics.get("cache_hits", 0)
                    cache_hits_int = int(cache_hits) if isinstance(cache_hits, (int, float, str)) else 0
                    self._metrics["cache_hits"] = cache_hits_int + 1
                    return cast(Order, cached_order)
                # Явное приведение типов для арифметических операций
                cache_misses = self._metrics.get("cache_misses", 0)
                cache_misses_int = int(cache_misses) if isinstance(cache_misses, (int, float, str)) else 0
                self._metrics["cache_misses"] = cache_misses_int + 1

            async def _get_operation(conn: Any) -> Optional[Order]:
                query = """
                SELECT * FROM orders WHERE id = $1
                """
                row = await conn.fetchrow(query, order_id_str)
                if row:
                    return self._row_to_order(row)
                return None

            result = await self._execute_with_retry(_get_operation)
            # Кэширование результата
            if result and self.cache_service:
                await self.cache_service.set(order_id_str, result)
            return cast(Optional[Order], result)
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
            return None

    async def update_order(
        self, order_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> RepositoryResult:
        """Обновление ордера в PostgreSQL."""
        try:
            order_id_str = str(order_id)
            # Получение текущего ордера
            current_order = await self.get_order(order_id_str)
            if not current_order:
                return RepositoryResult(
                    success=False, error_message=f"Order not found: {order_id_str}"
                )

            async def _update_operation(conn: Any) -> Optional[Order]:
                # Динамическое построение запроса обновления
                set_clauses = []
                values = []
                param_count = 1
                for key, value in updates.items():
                    if hasattr(current_order, key):
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(value)
                        param_count += 1
                if not set_clauses:
                    return current_order
                set_clauses.append(f"updated_at = ${param_count}")
                values.append(datetime.now(timezone.utc))
                param_count += 1
                values.append(order_id_str)
                query = f"""
                UPDATE orders SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
                RETURNING *
                """
                row = await conn.fetchrow(query, *values)
                if row:
                    return self._row_to_order(row)
                return None

            result = await self._execute_with_retry(_update_operation)
            if result:
                # Обновление кэша
                if self.cache_service:
                    await self.cache_service.set(order_id_str, result)
                return RepositoryResult(success=True, data=cast(Order, result))
            else:
                return RepositoryResult(
                    success=False, error_message="Failed to update order"
                )
        except Exception as e:
            self.logger.error(f"Error updating order: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to update order: {str(e)}"
            )

    async def delete_order(self, order_id: Union[str, UUID]) -> RepositoryResult:
        """Удаление ордера из PostgreSQL."""
        try:
            order_id_str = str(order_id)

            async def _delete_operation(conn: Any) -> bool:
                query = "DELETE FROM orders WHERE id = $1 RETURNING id"
                result = await conn.fetchval(query, order_id_str)
                return result is not None

            success = await self._execute_with_retry(_delete_operation)
            if success:
                # Удаление из кэша
                if self.cache_service:
                    await self.cache_service.delete(order_id_str)
                return RepositoryResult(success=True, data=True)
            else:
                return RepositoryResult(
                    success=False, error_message=f"Order not found: {order_id_str}"
                )
        except Exception as e:
            self.logger.error(f"Error deleting order: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to delete order: {str(e)}"
            )

    async def list_orders(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> RepositoryResult:
        """Получение списка ордеров из PostgreSQL."""
        try:

            async def _list_operation(conn: Any) -> List[Order]:
                query = "SELECT * FROM orders WHERE 1=1"
                values = []
                param_count = 1
                if status:
                    query += f" AND status = ${param_count}"
                    values.append(status.value)
                    param_count += 1
                if side:
                    query += f" AND side = ${param_count}"
                    values.append(side.value)
                    param_count += 1
                if trading_pair_id:
                    query += f" AND trading_pair = ${param_count}"
                    values.append(str(trading_pair_id))
                    param_count += 1
                query += " ORDER BY created_at DESC"
                if limit:
                    query += f" LIMIT ${param_count}"
                    values.append(str(limit))
                    param_count += 1
                if offset:
                    query += f" OFFSET ${param_count}"
                    values.append(str(offset))
                rows = await conn.fetch(query, *values)
                return [self._row_to_order(row) for row in rows]

            orders = await self._execute_with_retry(_list_operation)
            return RepositoryResult(success=True, data=orders)
        except Exception as e:
            self.logger.error(f"Error listing orders: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to list orders: {str(e)}"
            )

    async def add_position(self, position: Position) -> RepositoryResult:
        """Добавление позиции в PostgreSQL."""
        try:

            async def _add_operation(conn: Any) -> None:
                query = """
                INSERT INTO positions (
                    id, portfolio_id, trading_pair, side, volume, entry_price,
                    current_price, unrealized_pnl, realized_pnl, margin_used,
                    leverage, created_at, updated_at, closed_at, stop_loss,
                    take_profit, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """
                await conn.execute(
                    query,
                    str(position.id),
                    str(position.portfolio_id),
                    str(position.trading_pair),
                    position.side.value,
                    str(position.volume.to_decimal()),
                    str(position.entry_price.amount),
                    str(position.current_price.amount),
                    str(position.unrealized_pnl.amount) if position.unrealized_pnl else None,
                    str(position.realized_pnl.amount) if position.realized_pnl else None,
                    str(position.margin_used.amount) if position.margin_used else None,
                    str(position.leverage),
                    position.created_at.to_iso(),
                    position.updated_at.to_iso(),
                    position.closed_at.to_iso() if position.closed_at else None,
                    str(position.stop_loss.amount) if position.stop_loss else None,
                    str(position.take_profit.amount) if position.take_profit else None,
                    str(position.metadata),
                )

            await self._execute_with_retry(_add_operation)
            # Кэширование
            if self.cache_service:
                await self.cache_service.set(str(position.id), position)
            return RepositoryResult(success=True, data=position)
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to add position: {str(e)}"
            )

    async def get_position(self, position_id: Union[str, UUID]) -> RepositoryResult:
        """Получение позиции из PostgreSQL."""
        try:
            position_id_str = str(position_id)
            # Проверка кэша
            if self.cache_service:
                cached_position = await self.cache_service.get(position_id_str)
                if cached_position:
                    return RepositoryResult(success=True, data=cached_position)

            async def _get_operation(conn: Any) -> Optional[Position]:
                query = """
                SELECT * FROM positions WHERE id = $1
                """
                row = await conn.fetchrow(query, position_id_str)
                if row:
                    return self._row_to_position(row)
                return None

            result = await self._execute_with_retry(_get_operation)
            position = cast(Optional[Position], result)
            if position:
                # Кэширование результата
                if self.cache_service:
                    await self.cache_service.set(position_id_str, position)
                return RepositoryResult(success=True, data=position)
            else:
                return RepositoryResult(success=False, error_message="Position not found")
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to get position: {str(e)}"
            )

    async def update_position(
        self, position_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> RepositoryResult:
        """Обновление позиции в PostgreSQL."""
        try:
            position_id_str = str(position_id)
            # Получение текущей позиции
            current_position_result = await self.get_position(position_id_str)
            if not current_position_result.success:
                return current_position_result
            current_position = current_position_result.data

            async def _update_operation(conn: Any) -> Optional[Position]:
                # Динамическое построение запроса обновления
                set_clauses = []
                values = []
                param_count = 1
                for key, value in updates.items():
                    if hasattr(current_position, key):
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(value)
                        param_count += 1
                if not set_clauses:
                    return cast(Position, current_position)
                set_clauses.append(f"updated_at = ${param_count}")
                values.append(datetime.now(timezone.utc))
                param_count += 1
                values.append(position_id_str)
                query = f"""
                UPDATE positions SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
                RETURNING *
                """
                row = await conn.fetchrow(query, *values)
                if row:
                    return self._row_to_position(row)
                return None

            result = await self._execute_with_retry(_update_operation)
            if result:
                # Обновление кэша
                if self.cache_service:
                    await self.cache_service.set(position_id_str, result)
                return RepositoryResult(success=True, data=cast(Position, result))
            else:
                return RepositoryResult(
                    success=False, error_message="Failed to update position"
                )
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to update position: {str(e)}"
            )

    async def delete_position(self, position_id: Union[str, UUID]) -> RepositoryResult:
        """Удаление позиции из PostgreSQL."""
        try:
            position_id_str = str(position_id)

            async def _delete_operation(conn: Any) -> bool:
                query = "DELETE FROM positions WHERE id = $1 RETURNING id"
                result = await conn.fetchval(query, position_id_str)
                return result is not None

            success = await self._execute_with_retry(_delete_operation)
            if success:
                # Удаление из кэша
                if self.cache_service:
                    await self.cache_service.delete(position_id_str)
                return RepositoryResult(success=True, data=True)
            else:
                return RepositoryResult(
                    success=False, error_message=f"Position not found: {position_id_str}"
                )
        except Exception as e:
            self.logger.error(f"Error deleting position: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to delete position: {str(e)}"
            )

    async def list_positions(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> RepositoryResult:
        """Получение списка позиций из PostgreSQL."""
        try:

            async def _list_operation(conn: Any) -> List[Position]:
                query = "SELECT * FROM positions WHERE 1=1"
                values = []
                param_count = 1
                if side:
                    query += f" AND side = ${param_count}"
                    values.append(side.value)
                    param_count += 1
                if trading_pair_id:
                    query += f" AND trading_pair = ${param_count}"
                    values.append(str(trading_pair_id))
                    param_count += 1
                query += " ORDER BY created_at DESC"
                if limit:
                    query += f" LIMIT ${param_count}"
                    values.append(str(limit))
                    param_count += 1
                if offset:
                    query += f" OFFSET ${param_count}"
                    values.append(str(offset))
                rows = await conn.fetch(query, *values)
                return [self._row_to_position(row) for row in rows]

            positions = await self._execute_with_retry(_list_operation)
            return RepositoryResult(success=True, data=positions)
        except Exception as e:
            self.logger.error(f"Error listing positions: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to list positions: {str(e)}"
            )

    async def get_trading_metrics(self) -> RepositoryResult:
        """Получение торговых метрик из PostgreSQL."""
        try:

            async def _metrics_operation(conn: Any) -> Dict[str, Any]:
                # Статистика по статусам ордеров
                status_query = """
                SELECT status, COUNT(*) as count FROM orders GROUP BY status
                """
                status_rows = await conn.fetch(status_query)
                status_stats = {row["status"]: row["count"] for row in status_rows}
                # Статистика по сторонам ордеров
                side_query = """
                SELECT side, COUNT(*) as count FROM orders GROUP BY side
                """
                side_rows = await conn.fetch(side_query)
                side_stats = {row["side"]: row["count"] for row in side_rows}
                # Общие метрики
                total_orders_query = "SELECT COUNT(*) FROM orders"
                total_orders = await conn.fetchval(total_orders_query)
                total_positions_query = "SELECT COUNT(*) FROM positions"
                total_positions = await conn.fetchval(total_positions_query)
                open_orders_query = "SELECT COUNT(*) FROM orders WHERE status IN ('open', 'partially_filled')"
                open_orders = await conn.fetchval(open_orders_query)
                open_positions_query = (
                    "SELECT COUNT(*) FROM positions WHERE closed_at IS NULL"
                )
                open_positions = await conn.fetchval(open_positions_query)
                return {
                    "total_orders": total_orders,
                    "total_positions": total_positions,
                    "open_orders": open_orders,
                    "open_positions": open_positions,
                    "status_stats": status_stats,
                    "side_stats": side_stats,
                }

            metrics = await self._execute_with_retry(_metrics_operation)
            return RepositoryResult(success=True, data=metrics)
        except Exception as e:
            self.logger.error(f"Error getting trading metrics: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to get trading metrics: {str(e)}"
            )

    async def clear_all_data(self) -> RepositoryResult:
        """Очистка всех данных из PostgreSQL."""
        try:

            async def _clear_operation(conn: Any) -> None:
                await conn.execute("DELETE FROM orders")
                await conn.execute("DELETE FROM positions")

            await self._execute_with_retry(_clear_operation)
            # Очистка кэша
            if self.cache_service:
                await self.cache_service.clear()
            return RepositoryResult(success=True, data=True)
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            return RepositoryResult(
                success=False, error_message=f"Failed to clear data: {str(e)}"
            )

    def _row_to_order(self, row: Any) -> Order:
        """Преобразование строки БД в объект Order."""

        
        return Order(
            id=OrderId(UUID(row['id'])),
            portfolio_id=PortfolioId(UUID(row['portfolio_id'])),
            strategy_id=StrategyId(UUID(row['strategy_id'])),
            signal_id=SignalId(UUID(row['signal_id'])) if row.get('signal_id') else None,
            exchange_order_id=row.get('exchange_order_id'),
            symbol=Symbol(row['symbol']),
            trading_pair=TradingPairType(str(row['symbol'])),
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

    def _row_to_position(self, row: Any) -> Position:
        """Преобразование строки БД в объект Position."""

        
        return Position(
            id=PositionId(UUID(row["id"])),
            portfolio_id=PortfolioId(UUID(row["portfolio_id"])),
            trading_pair=TradingPair(
                symbol=Symbol(str(row.get('symbol', 'BTC/USDT'))),
                base_currency=Currency("BTC"),
                quote_currency=Currency("USDT")
            ),
            side=PositionSide(row["side"]),
            volume=Volume(Decimal(row["volume"]), Currency.USDT),
            entry_price=Price(Decimal(row["entry_price"]), Currency.USDT),
            current_price=Price(Decimal(row["current_price"]), Currency.USDT),
            unrealized_pnl=(
                Money(Decimal(row["unrealized_pnl"]), Currency.USDT)
                if row["unrealized_pnl"]
                else None
            ),
            realized_pnl=(
                Money(Decimal(row["realized_pnl"]), Currency.USDT)
                if row["realized_pnl"]
                else None
            ),
            margin_used=(
                Money(Decimal(row["margin_used"]), Currency.USDT)
                if row["margin_used"]
                else None
            ),
            leverage=Decimal(row["leverage"]),
            created_at=Timestamp.from_iso(row["created_at"]),
            updated_at=Timestamp.from_iso(row["updated_at"]),
            closed_at=(
                Timestamp.from_iso(row["closed_at"]) if row["closed_at"] else None
            ),
            stop_loss=(
                Price(Decimal(row["stop_loss"]), Currency.USDT)
                if row["stop_loss"]
                else None
            ),
            take_profit=(
                Price(Decimal(row["take_profit"]), Currency.USDT)
                if row["take_profit"]
                else None
            ),
            metadata=ast.literal_eval(row["metadata"]) if row["metadata"] else {},
        )

    async def save_trade(self, trade: Trade) -> Trade:
        """Сохранение сделки в PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # SQL для вставки сделки
                query = """
                INSERT INTO trades (
                    id, order_id, symbol, side, quantity, price, 
                    timestamp, fee, fee_currency, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    order_id = EXCLUDED.order_id,
                    symbol = EXCLUDED.symbol,
                    side = EXCLUDED.side,
                    quantity = EXCLUDED.quantity,
                    price = EXCLUDED.price,
                    timestamp = EXCLUDED.timestamp,
                    fee = EXCLUDED.fee,
                    fee_currency = EXCLUDED.fee_currency,
                    status = EXCLUDED.status
                RETURNING *
                """
                
                await conn.execute(
                    query,
                    str(trade.id),
                    str(trade.order_id),
                    str(trade.symbol),  # type: ignore[attr-defined]
                    trade.side.value,
                    float(trade.quantity.value),
                    float(trade.price.value),
                    trade.timestamp.value,  # type: ignore[attr-defined]
                    float(getattr(trade, 'fee', 0)),
                    getattr(trade, 'fee_currency', 'USDT'),
                    getattr(trade, 'status', 'FILLED')
                )
                
                self.logger.info(f"Trade saved to PostgreSQL: {trade.id}")
                return trade
        except Exception as e:
            self.logger.error(f"Error saving trade to PostgreSQL: {e}")
            raise

    async def get_trades_by_order(self, order_id: OrderId) -> List[Trade]:
        """Получение сделок по ордеру из PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = "SELECT * FROM trades WHERE order_id = $1 ORDER BY timestamp DESC"
                rows = await conn.fetch(query, str(order_id))
                
                trades = []
                for row in rows:
                    symbol_str = str(row.get('symbol', 'BTC/USDT'))
                    
                    trade = Trade(
                        id=TradeId(UUID(row['id'])),
                        order_id=OrderId(UUID(row['order_id'])),
                        trading_pair=str(symbol_str),
                        side=TradingOrderSide(row['side']),
                        quantity=Volume(Decimal(str(row['quantity'])), Currency.USD),
                        price=Price(Decimal(str(row['price'])), Currency.USD),
                        commission=Money(Decimal(str(row.get('fee', 0))), Currency.USD),
                        timestamp=cast(BaseTimestampValue, row['timestamp'])
                    )
                    trades.append(trade)
                
                return trades
        except Exception as e:
            self.logger.error(f"Error getting trades by order from PostgreSQL: {e}")
            return []

    async def get_trades_by_symbol(
        self,
        symbol: Symbol,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Получение сделок по символу из PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = "SELECT * FROM trades WHERE symbol = $1"
                params = [str(symbol)]
                
                if start_date:
                    query += " AND timestamp >= $2"
                    params.append(str(start_date))
                if end_date:
                    query += " AND timestamp <= $3"
                    params.append(str(end_date))
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query, *params)
                
                trades = []
                for row in rows:
                    symbol_str = str(row.get('symbol', 'BTC/USDT'))
                    
                    trade = Trade(
                        id=TradeId(UUID(row['id'])),
                        order_id=OrderId(UUID(row['order_id'])),
                        trading_pair=str(symbol_str),
                        side=TradingOrderSide(row['side']),
                        quantity=Volume(Decimal(str(row['quantity'])), Currency.USD),
                        price=Price(Decimal(str(row['price'])), Currency.USD),
                        commission=Money(Decimal(str(row.get('fee', 0))), Currency.USD),
                        timestamp=cast(BaseTimestampValue, row['timestamp'])
                    )
                    trades.append(trade)
                
                return trades
        except Exception as e:
            self.logger.error(f"Error getting trades by symbol from PostgreSQL: {e}")
            return []

    async def get_trade(
        self, 
        symbol: Optional[Symbol] = None, 
        limit: int = 100,
        account_id: Optional[str] = None
    ) -> List[Trade]:
        """Получение сделок с фильтрацией из PostgreSQL."""
        try:
            if symbol:
                return await self.get_trades_by_symbol(symbol, limit=limit)
            
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT $1"
                rows = await conn.fetch(query, limit)
                
                trades = []
                for row in rows:
                    symbol_str = str(row.get('symbol', 'BTC/USDT'))
                    
                    trade = Trade(
                        id=TradeId(UUID(row['id'])),
                        order_id=OrderId(UUID(row['order_id'])),
                        trading_pair=str(symbol_str),
                        side=TradingOrderSide(row['side']),
                        quantity=Volume(Decimal(str(row['quantity'])), Currency.USD),
                        price=Price(Decimal(str(row['price'])), Currency.USD),
                        commission=Money(Decimal(str(row.get('fee', 0))), Currency.USD),
                        timestamp=cast(BaseTimestampValue, row['timestamp'])
                    )
                    trades.append(trade)
                
                return trades
        except Exception as e:
            self.logger.error(f"Error getting trades from PostgreSQL: {e}")
            return []

    async def get_balance(self, account_id: Optional[str] = None) -> Dict[str, Money]:
        """Получение баланса аккаунта из PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                target_account_id = account_id or "default"
                
                query = """
                SELECT currency, total_balance, available_balance 
                FROM account_balances 
                WHERE account_id = $1
                """
                rows = await conn.fetch(query, target_account_id)
                
                balance = {}
                for row in rows:
                    currency = row['currency']
                    total = Decimal(str(row['total_balance']))
                    balance[currency] = Money(total, Currency(currency))
                
                # Если баланс пустой, возвращаем базовые валюты с нулевыми значениями
                if not balance:
                    balance = {
                        "USDT": Money(Decimal("0"), Currency.USDT),
                        "USD": Money(Decimal("0"), Currency.USD),
                        "BTC": Money(Decimal("0"), Currency.BTC),
                        "ETH": Money(Decimal("0"), Currency.ETH),
                    }
                
                return balance
        except Exception as e:
            self.logger.error(f"Error getting balance from PostgreSQL: {e}")
            return {}

    async def save_account(self, account: Account) -> Account:
        """Сохранение аккаунта в PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # SQL для вставки аккаунта
                query = """
                INSERT INTO accounts (
                    account_id, name, type, status, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (account_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """
                
                await conn.execute(
                    query,
                    account.id,
                    getattr(account, 'name', ''),
                    getattr(account, 'type', 'SPOT'),
                    getattr(account, 'status', 'ACTIVE'),
                    datetime.now(),
                    datetime.now()
                )
                
                # Сохраняем балансы аккаунта
                for balance in account.balances:
                    balance_query = """
                    INSERT INTO account_balances (
                        account_id, currency, total_balance, available_balance
                    ) VALUES ($1, $2, $3, $4)
                    ON CONFLICT (account_id, currency) DO UPDATE SET
                        total_balance = EXCLUDED.total_balance,
                        available_balance = EXCLUDED.available_balance
                    """
                    await conn.execute(
                        balance_query,
                        account.id,
                        balance.currency,
                        float(balance.total),
                        float(balance.available)
                    )
                
                self.logger.info(f"Account saved to PostgreSQL: {account.id}")
                return account
        except Exception as e:
            self.logger.error(f"Error saving account to PostgreSQL: {e}")
            raise

    async def get_account(self, account_id: str) -> Optional[Account]:
        """Получение аккаунта по ID из PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = "SELECT * FROM accounts WHERE account_id = $1"
                row = await conn.fetchrow(query, account_id)
                
                if not row:
                    return None
                
                # Получаем балансы аккаунта
                balance_query = "SELECT * FROM account_balances WHERE account_id = $1"
                balance_rows = await conn.fetch(balance_query, account_id)
                
                balances = []
                for balance_row in balance_rows:
                    from domain.entities.account import Balance
                    balance = Balance(
                        currency=balance_row['currency'],
                        available=Decimal(str(balance_row['available_balance'])),
                        locked=Decimal(str(balance_row['total_balance'])) - Decimal(str(balance_row['available_balance']))
                    )
                    balances.append(balance)
                
                account = Account(
                    id=row['account_id'],
                    balances=balances
                )
                
                return account
        except Exception as e:
            self.logger.error(f"Error getting account from PostgreSQL: {e}")
            return None

    async def update_account_balance(
        self, 
        account_id: str, 
        currency: str, 
        amount: Money
    ) -> bool:
        """Обновление баланса аккаунта в PostgreSQL."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = """
                UPDATE account_balances 
                SET total_balance = $3, available_balance = $3
                WHERE account_id = $1 AND currency = $2
                """
                
                result = await conn.execute(
                    query,
                    account_id,
                    currency,
                    float(amount.value)
                )
                
                if result == "UPDATE 0":
                    # Если записи нет, создаем новую
                    insert_query = """
                    INSERT INTO account_balances (
                        account_id, currency, total_balance, available_balance
                    ) VALUES ($1, $2, $3, $3)
                    """
                    await conn.execute(insert_query, account_id, currency, float(amount.value))
                
                self.logger.info(f"Account balance updated in PostgreSQL: {account_id} {currency}")
                return True
        except Exception as e:
            self.logger.error(f"Error updating account balance in PostgreSQL: {e}")
            return False

    async def get_health_status(self) -> str:
        """Получение статуса здоровья репозитория."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return "healthy"
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return "unhealthy"

    async def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик репозитория."""
        return {
            "queries_executed": self._metrics.get("queries_executed", 0),
            "cache_hits": self._metrics.get("cache_hits", 0),
            "cache_misses": self._metrics.get("cache_misses", 0),
            "errors": self._metrics.get("errors", 0),
            "last_health_check": self._metrics.get("last_health_check"),
        }

    async def close(self) -> None:
        """Закрытие соединений."""
        if self._pool:
            await self._pool.close()  # type: ignore[unreachable]
            self._pool = None
