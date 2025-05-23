import asyncio
from asyncio import Lock
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import backoff
import numpy as np
import pandas as pd
from loguru import logger

from .account_manager import AccountManager, AccountMetrics, RiskConfig
from .bybit_client import BybitClient
from .market_data import DataCache, MarketData
from .order_manager import Order as OrderManagerOrder
from .order_manager import OrderConfig, OrderManager


@dataclass
class Order:
    """Структура ордера"""

    symbol: str
    side: str  # 'buy' или 'sell'
    type: str  # 'market' или 'limit'
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = datetime.now()
    status: str = "new"  # 'new', 'filled', 'cancelled', 'rejected'
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0


@dataclass
class Position:
    """Структура позиции"""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = datetime.now()


@dataclass
class ExchangeConfig:
    """Конфигурация биржи"""

    api_key: str
    api_secret: str
    testnet: bool = False
    symbols: List[str] = field(default_factory=list)
    intervals: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: int = 1
    cache_ttl: int = 60
    update_interval: int = 1
    health_check_interval: int = 60
    risk_config: Optional[RiskConfig] = None
    order_config: Optional[OrderConfig] = None


class Exchange:
    """Класс для работы с биржей"""

    def __init__(self, config: ExchangeConfig):
        """Инициализация биржи"""
        self.config = config

        # Клиенты
        self.client = BybitClient(
            api_key=config.api_key, api_secret=config.api_secret, testnet=config.testnet
        )

        # Менеджеры
        self.account_manager = AccountManager(
            client=self.client,
            risk_config=config.risk_config,
            order_manager=None,  # Будет установлен позже
        )
        self.order_manager = OrderManager(
            client=self.client, config=config.order_config or OrderConfig()
        )
        self.account_manager.order_manager = self.order_manager

        # Кэш данных
        self.data_cache = DataCache(max_size=1000)

        # Состояние
        self.is_running = False
        self.lock = Lock()
        self.last_update = None

        # Задачи
        self.tasks: Set[asyncio.Task] = set()

        # Метрики
        self.metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_volume": 0.0,
            "total_commission": 0.0,
            "total_pnl": 0.0,
            "average_execution_time": 0.0,
            "uptime": 0.0,
            "last_error": None,
            "error_count": 0,
        }

        # События
        self.on_trade = None
        self.on_error = None
        self.on_metrics_update = None

        # Инициализация атрибутов
        self.orders: Dict[str, OrderManagerOrder] = {}
        self.positions: Dict[str, Position] = {}
        self.balance: Dict[str, float] = {}

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def start(self):
        """Запуск биржи с повторными попытками"""
        try:
            # Запуск менеджеров
            await self.account_manager.start()

            # Запуск задач
            self.tasks.add(asyncio.create_task(self._update_market_data()))
            self.tasks.add(asyncio.create_task(self._monitor_trades()))
            self.tasks.add(asyncio.create_task(self._health_check()))

            self.is_running = True
            self.metrics["uptime"] = datetime.now().timestamp()

            logger.info("Exchange started")
        except Exception as e:
            logger.error(f"Error starting exchange: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def stop(self):
        """Корректная остановка биржи"""
        try:
            self.is_running = False

            # Отмена задач
            for task in self.tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Остановка менеджеров
            await self.account_manager.stop()

            # Сохранение метрик
            await self._save_metrics()

            logger.info("Exchange stopped")
        except Exception as e:
            logger.error(f"Error stopping exchange: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def get_market_data(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[MarketData]:
        """Получение рыночных данных с кэшированием"""
        try:
            # Проверка кэша
            cache_key = f"{symbol}_{interval}"
            cached_data = self.data_cache.get(cache_key)

            if (
                cached_data
                and (datetime.now() - cached_data.timestamp).total_seconds()
                < self.config.cache_ttl
            ):
                return [cached_data]

            # Получение новых данных
            klines = await self.client.get_klines(symbol, interval, limit)
            if not klines:
                return []

            # Создание объекта данных
            market_data = MarketData(symbol, interval)
            await market_data.update(klines)

            # Обновление кэша
            self.data_cache.add(cache_key, market_data)

            return [market_data]

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            return []

    async def get_account_metrics(self) -> AccountMetrics:
        """Получение метрик аккаунта"""
        try:
            return await self.account_manager.get_metrics()
        except Exception as e:
            logger.error(f"Error getting account metrics: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        leverage: Optional[float] = None,
        tags: Optional[Set[str]] = None,
    ) -> Order:
        """Создание ордера с расширенной функциональностью"""
        try:
            # Проверка доступной маржи
            available_margin = await self.account_manager.get_available_margin(symbol)
            required_margin = amount * price / leverage if leverage else amount * price

            if required_margin > available_margin:
                raise ValueError("Insufficient margin")

            # Создание ордера
            order = await self.order_manager.create_entry_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                leverage=leverage,
                tags=tags,
            )

            # Обновление метрик
            self.metrics["total_trades"] += 1

            # Вызов обработчика
            if self.on_trade:
                await self.on_trade(order)

            return order

        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            self.metrics["failed_trades"] += 1

            if self.on_error:
                await self.on_error(str(e))

            raise

    async def cancel_order(self, order_id: str):
        """Отмена ордера"""
        try:
            await self.order_manager.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def _update_market_data(self):
        """Обновление рыночных данных"""
        try:
            while self.is_running:
                try:
                    for symbol in self.config.symbols:
                        for interval in self.config.intervals:
                            await self.get_market_data(symbol, interval)

                    await asyncio.sleep(self.config.update_interval)

                except Exception as e:
                    logger.error(f"Error updating market data: {str(e)}")
                    self.metrics["last_error"] = str(e)
                    self.metrics["error_count"] += 1
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in market data update: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def _monitor_trades(self):
        """Мониторинг сделок"""
        try:
            while self.is_running:
                try:
                    # Обновление метрик
                    metrics = await self.get_account_metrics()

                    # Обновление статистики
                    self.metrics["total_volume"] = metrics.total_volume
                    self.metrics["total_commission"] = metrics.total_commission
                    self.metrics["total_pnl"] = metrics.total_pnl

                    # Вызов обработчика
                    if self.on_metrics_update:
                        await self.on_metrics_update(self.metrics)

                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error monitoring trades: {str(e)}")
                    self.metrics["last_error"] = str(e)
                    self.metrics["error_count"] += 1
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in trade monitor: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def _health_check(self):
        """Проверка здоровья системы"""
        try:
            while self.is_running:
                try:
                    # Проверка соединения
                    await self.client.ping()

                    # Проверка API ключей
                    await self.client.validate_api_keys()

                    # Проверка менеджеров
                    await self.account_manager._health_check()

                    await asyncio.sleep(self.config.health_check_interval)

                except Exception as e:
                    logger.error(f"Error in health check: {str(e)}")
                    self.metrics["last_error"] = str(e)
                    self.metrics["error_count"] += 1
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    async def _save_metrics(self):
        """Сохранение метрик"""
        try:
            # Здесь можно добавить логику сохранения в базу данных
            pass
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            raise

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Получение исторических данных.

        Args:
            symbol: Торговая пара
            interval: Интервал ('1m', '5m', '1h', '1d' и т.д.)
            start_time: Время начала
            end_time: Время окончания
            limit: Лимит свечей

        Returns:
            DataFrame: Исторические данные
        """
        try:
            # TODO: Реализовать получение данных с биржи
            # Временная заглушка
            dates = pd.date_range(
                start=start_time or datetime.now(), periods=limit, freq=interval
            )
            data = pd.DataFrame(
                {
                    "open": np.random.normal(100, 1, limit),
                    "high": np.random.normal(101, 1, limit),
                    "low": np.random.normal(99, 1, limit),
                    "close": np.random.normal(100, 1, limit),
                    "volume": np.random.normal(1000, 100, limit),
                },
                index=dates,
            )
            return data

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()

    def place_order(self, order: Order) -> bool:
        """
        Размещение ордера.

        Args:
            order: Ордер

        Returns:
            bool: Успешность размещения
        """
        try:
            # TODO: Реализовать размещение ордера на бирже
            # Временная заглушка
            order_id = f"{order.symbol}_{order.side}_{order.timestamp.timestamp()}"
            self.orders[order_id] = order
            return True

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Получение статуса ордера.

        Args:
            order_id: ID ордера

        Returns:
            Optional[Order]: Ордер или None
        """
        return self.orders.get(order_id)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Получение позиции.

        Args:
            symbol: Торговая пара

        Returns:
            Optional[Position]: Позиция или None
        """
        return self.positions.get(symbol)

    def get_balance(self, asset: str) -> float:
        """
        Получение баланса.

        Args:
            asset: Актив

        Returns:
            float: Баланс
        """
        return self.balance.get(asset, 0.0)

    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """
        Обновление рыночных данных.

        Args:
            symbol: Торговая пара
            data: Рыночные данные
        """
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = data["close"].iloc[-1]
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.quantity

        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    def _process_response(self, data: Optional[Dict]) -> Optional[Dict]:
        """Обработка ответа от API"""
        if not data:
            return None
        return data if isinstance(data, dict) else None

    def _get_value(self, result: Optional[Dict], key: str) -> Optional[Any]:
        """Безопасное получение значения из словаря"""
        if not result or not isinstance(result, dict):
            return None
        return result.get(key)

    async def get_account(self) -> Optional[Dict]:
        """Получение информации об аккаунте"""
        if not self.client:
            return None
        try:
            return await self.client.get_account()
        except Exception as e:
            logger.error(f"Error getting account: {str(e)}")
            return None
