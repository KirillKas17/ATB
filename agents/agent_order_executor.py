import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.agent_market_regime import MarketRegime, MarketRegimeAgent
from agents.agent_risk import RiskAgent
from exchange.bybit_client import BybitClient as ExchangeBybitClient
from utils.indicators import calculate_fractals, calculate_liquidity_zones
from utils.logger import setup_logger
from utils.math_utils import calculate_fibonacci_levels, calculate_support_resistance

logger = setup_logger(__name__)


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class OrderParams:
    """Параметры ордера"""

    symbol: str
    direction: str  # 'buy' или 'sell'
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    order_type: OrderType
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    reason: str = ""


class IBrokerClient(ABC):
    @abstractmethod
    async def place_order(self, params: OrderParams) -> Optional[str]:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        pass


class BybitClient(ExchangeBybitClient):
    def get_balance(self) -> float:
        """Получение баланса аккаунта."""
        try:
            return float(super().get_balance())
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0.0

    def get_ticker(self, symbol: str) -> float:
        """Получение текущей цены."""
        try:
            return float(super().get_ticker(symbol))
        except Exception as e:
            logger.error(f"Error getting ticker: {str(e)}")
            return 0.0


class BybitAsyncClient(IBrokerClient):
    def __init__(self):
        self.client = BybitClient()

    async def place_order(self, params: OrderParams) -> Optional[str]:
        # ... асинхронная обёртка ...
        return None

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        # ... асинхронная обёртка ...
        return {}

    async def cancel_order(self, order_id: str) -> bool:
        # ... асинхронная обёртка ...
        return False

    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.client.get_market_data, symbol)


class OrderHistoryService:
    def __init__(self):
        self.history: List[Dict] = []

    def add(self, record: Dict):
        self.history.append(record)

    def get(self) -> List[Dict]:
        return self.history

    def clear(self):
        self.history.clear()


class OrderExecutorAgent:
    """
    Агент исполнения ордеров: асинхронное размещение, мониторинг, история, симуляция, VaR/ES.
    TODO: Вынести работу с брокером, расчёты и кэширование в отдельные классы/модули (SRP).
    """

    config: Dict[str, Any]
    broker: IBrokerClient
    risk_agent: RiskAgent
    market_regime_agent: MarketRegimeAgent
    active_orders: Dict[str, OrderParams]
    order_history: OrderHistoryService
    _needs_recalculation: bool
    client: BybitClient

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента исполнения ордеров.
        :param config: словарь параметров
        """
        self.config = config or {
            "max_retries": 3,
            "retry_delay": 1.0,
            "order_timeout": 30.0,
            "min_spread": 0.001,
            "max_slippage": 0.002,
            "liquidity_threshold": 100000,
            "update_interval": 1.0,
        }
        self.broker = BybitAsyncClient()
        self.risk_agent = RiskAgent()
        self.market_regime_agent = MarketRegimeAgent()
        self.active_orders: Dict[str, OrderParams] = {}
        self.order_history = OrderHistoryService()
        self._needs_recalculation = False
        self.client = self.broker.client

    async def place_limit_order(
        self, pair: str, direction: str, amount: float, entry: float, stop: float, take: float
    ) -> Optional[str]:
        """
        Размещение лимитного ордера с учетом рыночных условий.
        :param pair: тикер пары
        :param direction: 'buy' или 'sell'
        :param amount: объем
        :param entry: цена входа
        :param stop: стоп-лосс
        :param take: тейк-профит
        :return: ID ордера или None
        """
        try:
            market_data = await self.broker.get_market_data(pair)
            regime, confidence = self.market_regime_agent.detect_regime(market_data)
            entry, stop, take = self._calculate_optimal_levels(
                pair, direction, entry, stop, take, market_data, regime
            )
            leverage = self.risk_agent.get_leverage_score(pair, confidence, market_data)
            order_params = OrderParams(
                symbol=pair,
                direction=direction,
                amount=amount,
                entry_price=entry,
                stop_loss=stop,
                take_profit=take,
                leverage=leverage,
                order_type=OrderType.LIMIT,
                time_in_force="GTC",
            )
            if not self._validate_order_conditions(order_params, market_data):
                logger.warning(f"Order conditions not met for {pair}")
                return None
            order_id = await self._place_order_with_retry(order_params)
            if order_id:
                self.active_orders[order_id] = order_params
                logger.info(f"Successfully placed order {order_id} for {pair}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing order for {pair}: {str(e)}")
            return None

    async def monitor_orders(self) -> None:
        """
        Мониторинг и обновление статусов активных ордеров.
        """
        try:
            for order_id, params in list(self.active_orders.items()):
                status = await self.broker.get_order_status(order_id)
                if status["status"] in ["filled", "canceled", "rejected"]:
                    await self._handle_completed_order(order_id, status)
                elif status["status"] == "open":
                    if await self._should_update_order(order_id, params):
                        await self._update_order(order_id, params)
        except Exception as e:
            logger.error(f"Error monitoring orders: {str(e)}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Отмена ордера.
        :param order_id: ID ордера
        :return: True если отмена успешна
        """
        try:
            if order_id in self.active_orders:
                success = await self.broker.cancel_order(order_id)
                if success:
                    del self.active_orders[order_id]
                    logger.info(f"Successfully canceled order {order_id}")
                return success
            return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False

    async def recalculate_if_conditions_changed(self, order_id: str) -> bool:
        """
        Пересчет параметров ордера при изменении рыночных условий.
        :param order_id: ID ордера
        :return: True если пересчет выполнен
        """
        try:
            if order_id not in self.active_orders:
                return False
            params = self.active_orders[order_id]
            market_data = await self.broker.get_market_data(params.symbol)
            if not self._needs_recalculation:
                return False
            new_entry, new_stop, new_take = self._calculate_optimal_levels(
                params.symbol,
                params.direction,
                params.entry_price,
                params.stop_loss,
                params.take_profit,
                market_data,
                self.market_regime_agent.detect_regime(market_data)[0],
            )
            if await self._should_update_order(order_id, params):
                await self._update_order(order_id, params)
                return True
            return False
        except Exception as e:
            logger.error(f"Error recalculating order {order_id}: {str(e)}")
            return False

    def _calculate_optimal_levels(
        self,
        pair: str,
        direction: str,
        entry: float,
        stop: float,
        take: float,
        market_data: pd.DataFrame,
        regime: MarketRegime,
    ) -> Tuple[float, float, float]:
        """Расчет оптимальных уровней для ордера"""
        try:
            # Получение уровней Фибоначчи
            fib_levels = calculate_fibonacci_levels(market_data, low=market_data['low'])

            # Получение уровней поддержки и сопротивления
            sr_levels = calculate_support_resistance(market_data['close'].tolist())

            # Получение зон ликвидности
            liquidity_zones = calculate_liquidity_zones(market_data)

            # Получение фракталов
            fractals = calculate_fractals(market_data)

            # Корректировка уровней в зависимости от режима рынка
            if regime == MarketRegime.MANIPULATION:
                # Использование зон ликвидности для манипуляций
                entry = self._adjust_to_liquidity_zone(entry, dict_to_list_of_dicts(liquidity_zones))
                stop = self._adjust_to_liquidity_zone(stop, dict_to_list_of_dicts(liquidity_zones))
                take = self._adjust_to_liquidity_zone(take, dict_to_list_of_dicts(liquidity_zones))
            else:
                # Использование технических уровней
                entry = self._adjust_to_technical_levels(entry, fib_levels, sr_levels)
                stop = self._adjust_to_technical_levels(stop, fib_levels, sr_levels)
                take = self._adjust_to_technical_levels(take, fib_levels, sr_levels)

            return entry, stop, take

        except Exception as e:
            logger.error(f"Error calculating optimal levels: {str(e)}")
            return entry, stop, take

    def _adjust_to_liquidity_zone(self, price: float, zones: List[Dict]) -> float:
        """Корректировка цены к ближайшей зоне ликвидности"""
        if not zones:
            return price

        closest_zone = min(zones, key=lambda x: abs(x["price"] - price))
        return closest_zone["price"]

    def _adjust_to_technical_levels(
        self, price: float, fib_levels: Dict[str, float], sr_levels: Dict[str, List[float]]
    ) -> float:
        """Корректировка цены к ближайшему техническому уровню"""
        all_levels = list(fib_levels.values()) + sr_levels["support"] + sr_levels["resistance"]
        if not all_levels:
            return price

        closest_level = min(all_levels, key=lambda x: abs(x - price))
        return closest_level

    def _validate_order_conditions(self, params: OrderParams, market_data: pd.DataFrame) -> bool:
        """Проверка условий для размещения ордера"""
        try:
            # Проверка спреда
            current_spread = abs(market_data["ask"].iloc[-1] - market_data["bid"].iloc[-1])
            if current_spread > self.config["min_spread"]:
                return False

            # Проверка ликвидности
            if market_data["volume"].iloc[-1] < self.config["liquidity_threshold"]:
                return False

            # Проверка проскальзывания
            expected_slippage = (
                abs(params.entry_price - market_data["close"].iloc[-1])
                / market_data["close"].iloc[-1]
            )
            if expected_slippage > self.config["max_slippage"]:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating order conditions: {str(e)}")
            return False

    async def _place_order_with_retry(self, params: OrderParams) -> Optional[str]:
        """Размещение ордера с повторными попытками"""
        for attempt in range(self.config["max_retries"]):
            try:
                order_id = await self.broker.place_order(params)
                return order_id

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config["max_retries"] - 1:
                    await asyncio.sleep(self.config["retry_delay"])

        return None

    async def _should_update_order(self, order_id: str, params: OrderParams) -> bool:
        """Проверка необходимости обновления ордера"""
        try:
            market_data = await self.broker.get_market_data(params.symbol)
            current_price = market_data["close"].iloc[-1]

            # Проверка отклонения от текущей цены
            price_deviation = abs(params.entry_price - current_price) / current_price

            # Проверка изменения волатильности
            volatility_change = abs(
                market_data["close"].pct_change().std()
                - await self._get_historical_volatility(params.symbol)
            )

            return (
                price_deviation > self.config["max_slippage"] or volatility_change > 0.1
            )  # 10% изменение волатильности

        except Exception as e:
            logger.error(f"Error checking order update: {str(e)}")
            return False

    async def _update_order(self, order_id: str, params: OrderParams):
        """Обновление параметров ордера"""
        try:
            # Отмена текущего ордера
            if await self.cancel_order(order_id):
                # Размещение нового ордера
                new_order_id = await self.place_limit_order(
                    params.symbol,
                    params.direction,
                    params.amount,
                    params.entry_price,
                    params.stop_loss,
                    params.take_profit,
                )

                if new_order_id:
                    self.active_orders[new_order_id] = params
                    logger.info(f"Successfully updated order {order_id} to {new_order_id}")

        except Exception as e:
            logger.error(f"Error updating order {order_id}: {str(e)}")

    async def _handle_completed_order(self, order_id: str, status: Dict):
        """Обработка завершенного ордера"""
        try:
            params = self.active_orders[order_id]

            # Добавление в историю
            self.order_history.add(
                {
                    "order_id": order_id,
                    "symbol": params.symbol,
                    "direction": params.direction,
                    "amount": params.amount,
                    "entry_price": params.entry_price,
                    "exit_price": status.get("filled_price", 0.0),
                    "status": status["status"],
                    "timestamp": status.get("timestamp", pd.Timestamp.now()),
                    "reason": params.reason,
                }
            )

            # Удаление из активных ордеров
            del self.active_orders[order_id]

            logger.info(f"Order {order_id} completed with status {status['status']}")

        except Exception as e:
            logger.error(f"Error handling completed order {order_id}: {str(e)}")

    async def _get_historical_volatility(self, symbol: str) -> float:
        """Получение исторической волатильности"""
        try:
            market_data = await self.broker.get_market_data(symbol)
            return market_data["close"].pct_change().std()
        except Exception as e:
            logger.error(f"Error getting historical volatility: {str(e)}")
            return 0.0

    def get_active_orders(self) -> Dict[str, OrderParams]:
        """Получение активных ордеров"""
        return self.active_orders

    def get_order_history(self) -> List[Dict]:
        """Получение истории ордеров"""
        return self.order_history.get()

    def calculate_position_size(self, price: float, risk_per_trade: float) -> float:
        """Расчет размера позиции."""
        try:
            account_balance = float(self.client.get_balance())
            position_size = (account_balance * risk_per_trade) / price
            return float(position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def calculate_stop_loss(self, entry_price: float, atr: float, low: float) -> float:
        """Расчет уровня стоп-лосса."""
        try:
            return float(min(entry_price - atr * 2, low))
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return 0.0

    def calculate_take_profit(self, entry_price: float, atr: float, high: float) -> float:
        """Расчет уровня тейк-профита."""
        try:
            return float(max(entry_price + atr * 3, high))
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return 0.0

    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены."""
        try:
            value = self.client.get_ticker(symbol)
            return float(value) if isinstance(value, (float, int)) else float(value.item())
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return 0.0

def dict_to_list_of_dicts(d: dict) -> list:
    keys = list(d.keys())
    length = len(next(iter(d.values())))
    return [{k: d[k][i] for k in keys} for i in range(length)]
