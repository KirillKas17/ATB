"""Модуль утилит для работы с ордерами.
Предоставляет функции для валидации, создания, управления и анализа ордеров.
Включает проверки безопасности, оптимизацию исполнения и мониторинг.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from decimal import Decimal, InvalidOperation

from loguru import logger


class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """Стороны ордера."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Статусы ордера."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderValidationResult:
    """Результат валидации ордера."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class OrderExecutionMetrics:
    """Метрики исполнения ордера."""

    order_id: str
    symbol: str
    side: str
    type: str
    requested_price: float
    requested_amount: float
    executed_price: float
    executed_amount: float
    remaining_amount: float
    execution_time: float
    slippage: float
    fees: float
    timestamp: datetime = field(default_factory=datetime.now)


class OrderUtils:
    """
    Продвинутые утилиты для работы с ордерами.
    Обеспечивает валидацию, оптимизацию и мониторинг ордеров
    с учетом рыночных условий и рисков.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация утилит ордеров."""
        self.config = config or {}
        self.min_order_size = self.config.get("min_order_size", 0.001)
        self.max_order_size = self.config.get("max_order_size", 1000.0)
        self.max_slippage = self.config.get("max_slippage", 0.02)  # 2%
        self.order_timeout = self.config.get("order_timeout", 300)  # 5 минут
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        logger.info("Order Utils initialized")

    def validate_order(self, order: Dict[str, Any]) -> OrderValidationResult:
        """
        Продвинутая валидация ордера.
        Args:
            order: Ордер для валидации
        Returns:
            OrderValidationResult: Результат валидации
        """
        result = OrderValidationResult(is_valid=True)
        try:
            # Проверка обязательных полей
            required_fields = ["symbol", "side", "type", "amount"]
            for field in required_fields:
                if field not in order:
                    result.errors.append(f"Missing required field: {field}")
                    result.is_valid = False
            if not result.is_valid:
                return result
            # Валидация символа
            if not self._validate_symbol(order["symbol"]):
                result.errors.append("Invalid symbol format")
                result.is_valid = False
            # Валидация стороны
            if order["side"] not in [side.value for side in OrderSide]:
                result.errors.append(f"Invalid side: {order['side']}")
                result.is_valid = False
            # Валидация типа
            if order["type"] not in [type.value for type in OrderType]:
                result.errors.append(f"Invalid order type: {order['type']}")
                result.is_valid = False
            # Валидация количества
            amount_validation = self._validate_amount(order["amount"])
            if not amount_validation["is_valid"]:
                result.errors.extend(amount_validation["errors"])
                result.is_valid = False
            # Валидация цены для лимитных ордеров
            if order["type"] in ["limit", "stop_limit"]:
                if "price" not in order:
                    result.errors.append("Price required for limit orders")
                    result.is_valid = False
                elif not self._validate_price(order["price"]):
                    result.errors.append("Invalid price")
                    result.is_valid = False
            # Валидация стоп-цены
            if order["type"] in ["stop", "stop_limit"]:
                if "stop_price" not in order:
                    result.errors.append("Stop price required for stop orders")
                    result.is_valid = False
                elif not self._validate_price(order["stop_price"]):
                    result.errors.append("Invalid stop price")
                    result.is_valid = False
            # Проверка логики цен
            if self._has_price_logic_errors(order):
                result.errors.append("Price logic error")
                result.is_valid = False
            # Предупреждения
            warnings = self._generate_warnings(order)
            result.warnings.extend(warnings)
            # Предложения
            suggestions = self._generate_suggestions(order)
            result.suggestions.extend(suggestions)
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.is_valid = False
        return result

    def _validate_symbol(self, symbol: str) -> bool:
        """Валидация символа."""
        try:
            # Проверка формата символа (например, BTC/USDT)
            if not isinstance(symbol, str) or len(symbol) < 3:
                return False
            # Проверка на наличие разделителя
            if "/" not in symbol and "-" not in symbol:
                return False
            return True
        except Exception:
            return False

    def _validate_amount(self, amount: Union[float, str]) -> Dict[str, Any]:
        """Валидация количества."""
        result: Dict[str, Any] = {"is_valid": True, "errors": []}
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                errors_list = result.get("errors", [])
                if not isinstance(errors_list, list):
                    errors_list = []
                errors_list.append("Amount must be positive")
                result["errors"] = errors_list
                result["is_valid"] = False
            if amount_float < self.min_order_size:
                errors_list = result.get("errors", [])
                if not isinstance(errors_list, list):
                    errors_list = []
                errors_list.append(
                    f"Amount too small: {amount_float} < {self.min_order_size}"
                )
                result["errors"] = errors_list
                result["is_valid"] = False
            if amount_float > self.max_order_size:
                errors_list = result.get("errors", [])
                if not isinstance(errors_list, list):
                    errors_list = []
                errors_list.append(
                    f"Amount too large: {amount_float} > {self.max_order_size}"
                )
                result["errors"] = errors_list
                result["is_valid"] = False
            # Проверка на разумность (не слишком много знаков после запятой)
            decimal_places = (
                len(str(amount_float).split(".")[-1]) if "." in str(amount_float) else 0
            )
            if decimal_places > 8:
                result["warnings"] = ["Amount has too many decimal places"]
        except (ValueError, TypeError):
            errors_list = result.get("errors", [])
            if not isinstance(errors_list, list):
                errors_list = []
            errors_list.append("Invalid amount format")
            result["errors"] = errors_list
            result["is_valid"] = False
        return result

    def _validate_price(self, price: Union[float, str]) -> bool:
        """Валидация цены."""
        try:
            price_float = float(price)
            return price_float > 0
        except (ValueError, TypeError):
            return False

    def _has_price_logic_errors(self, order: Dict[str, Any]) -> bool:
        """Проверка логических ошибок в ценах."""
        try:
            if order["type"] == "stop_limit":
                if "price" in order and "stop_price" in order:
                    if order["side"] == "buy":
                        # Для покупки: stop_price >= price
                        return float(order["stop_price"]) < float(order["price"])
                    else:
                        # Для продажи: stop_price <= price
                        return float(order["stop_price"]) > float(order["price"])
            return False
        except Exception:
            return True

    def _generate_warnings(self, order: Dict[str, Any]) -> List[str]:
        """Генерация предупреждений."""
        warnings = []
        try:
            # Предупреждение о больших ордерах
            if "amount" in order and float(order["amount"]) > self.max_order_size * 0.8:
                warnings.append("Large order size detected")
            # Предупреждение о рыночных ордерах
            if order.get("type") == "market":
                warnings.append("Market orders may have high slippage")
            # Предупреждение о стоп-ордерах
            if order.get("type") in ["stop", "stop_limit"]:
                warnings.append("Stop orders may not execute in volatile markets")
        except Exception as e:
            logger.warning(f"Failed to generate warnings for order {order.get('id')}: {e}")
        return warnings

    def _generate_suggestions(self, order: Dict[str, Any]) -> List[str]:
        """Генерация предложений по улучшению."""
        suggestions = []
        try:
            # Предложение использовать лимитные ордера
            if order.get("type") == "market":
                suggestions.append("Consider using limit orders to reduce slippage")
            # Предложение добавить стоп-лосс
            if "stop_loss" not in order:
                suggestions.append("Consider adding stop-loss for risk management")
            # Предложение использовать OCO ордера
            if order.get("type") in ["limit", "stop"]:
                suggestions.append("Consider using OCO orders for better execution")
        except Exception as e:
            logger.warning(f"Failed to generate suggestions for order {order.get('id')}: {e}")
        return suggestions

    async def create_optimized_order(
        self,
        exchange: Any,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Создание оптимизированного ордера.
        Args:
            exchange: Объект биржи
            symbol: Символ
            side: Сторона (buy/sell)
            amount: Количество
            order_type: Тип ордера
            **kwargs: Дополнительные параметры
        Returns:
            Dict: Созданный ордер
        """
        try:
            # Получение рыночных данных
            ticker = await exchange.get_ticker(symbol)
            orderbook = await exchange.get_order_book(symbol)
            # Расчет оптимальной цены
            optimal_price = self._calculate_optimal_price(
                side, order_type, ticker, orderbook, **kwargs
            )
            # Создание ордера
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": amount,
                "price": optimal_price,
                **kwargs,
            }
            # Валидация
            validation = self.validate_order(order_params)
            if not validation.is_valid:
                raise ValueError(f"Invalid order: {validation.errors}")
            # Создание ордера с повторными попытками
            for attempt in range(self.retry_attempts):
                try:
                    order = await exchange.create_order(**order_params)
                    logger.info(f"Order created: {order['id']}")
                    return order
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    logger.warning(f"Order creation attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.retry_delay)
        except Exception as e:
            logger.error(f"Error creating optimized order: {e}")
            raise
        # Явный возврат пустого словаря на случай, если ни один return не сработал
        return {}

    def _calculate_optimal_price(
        self,
        side: str,
        order_type: str,
        ticker: Dict[str, Any],
        orderbook: Dict[str, Any],
        **kwargs: Any,
    ) -> float:
        """Расчет оптимальной цены."""
        try:
            current_price = float(ticker["last"])
            if order_type == "market":
                return current_price
            elif order_type == "limit":
                # Расчет цены на основе спреда
                spread = float(ticker["ask"]) - float(ticker["bid"])
                mid_price = (float(ticker["ask"]) + float(ticker["bid"])) / 2
                if side == "buy":
                    # Покупаем чуть выше bid
                    return mid_price + spread * 0.1
                else:
                    # Продаем чуть ниже ask
                    return mid_price - spread * 0.1
            elif order_type == "stop":
                # Стоп-цена на основе волатильности
                volatility = kwargs.get("volatility", 0.02)
                if side == "buy":
                    return current_price * (1 + volatility)
                else:
                    return current_price * (1 - volatility)
            else:
                return current_price
        except Exception as e:
            logger.error(f"Error calculating optimal price: {e}")
            return float(ticker.get("last", 0.0))

    async def monitor_order_execution(
        self, exchange: Any, order_id: str, timeout: Optional[int] = None
    ) -> OrderExecutionMetrics:
        """
        Мониторинг исполнения ордера.
        Args:
            exchange: Объект биржи
            order_id: ID ордера
            timeout: Таймаут в секундах
        Returns:
            OrderExecutionMetrics: Метрики исполнения
        """
        try:
            timeout = timeout or self.order_timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Получение статуса ордера
                order = await exchange.get_order(order_id)
                if order["status"] in ["filled", "partially_filled"]:
                    # Расчет метрик исполнения
                    metrics = self._calculate_execution_metrics(order, start_time)
                    logger.info(f"Order {order_id} executed: {metrics}")
                    return metrics
                elif order["status"] in ["cancelled", "rejected", "expired"]:
                    raise Exception(f"Order {order_id} failed: {order['status']}")
                # Ожидание
                await asyncio.sleep(1)
            # Таймаут
            raise TimeoutError(f"Order {order_id} execution timeout")
        except Exception as e:
            logger.error(f"Error monitoring order execution: {e}")
            raise

    def _calculate_execution_metrics(
        self, order: Dict[str, Any], start_time: float
    ) -> OrderExecutionMetrics:
        """Расчет метрик исполнения."""
        try:
            execution_time = time.time() - start_time
            # Расчет проскальзывания
            requested_price = float(order.get("price", 0))
            executed_price = float(order.get("average", 0))
            slippage = (
                abs(executed_price - requested_price) / requested_price
                if requested_price > 0
                else 0
            )
            # Расчет комиссий
            fees = float(order.get("fee", {}).get("cost", 0))
            return OrderExecutionMetrics(
                order_id=order["id"],
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                requested_price=requested_price,
                requested_amount=float(order["amount"]),
                executed_price=executed_price,
                executed_amount=float(order["filled"]),
                remaining_amount=float(order["remaining"]),
                execution_time=execution_time,
                slippage=slippage,
                fees=fees,
            )
        except Exception as e:
            logger.error(f"Error calculating execution metrics: {e}")
            raise

    async def cancel_invalid_orders(
        self, exchange: Any, trading_pairs: List[str]
    ) -> List[str]:
        """
        Отмена невалидных ордеров.
        Args:
            exchange: Объект биржи
            trading_pairs: Список торговых пар
        Returns:
            List[str]: Список отмененных ордеров
        """
        try:
            cancelled_orders = []
            # Получение всех открытых ордеров
            open_orders = await exchange.get_open_orders()
            for order in open_orders:
                # Проверка валидности
                validation = self.validate_order(order)
                # Проверка торговой пары
                if order["symbol"] not in trading_pairs:
                    validation.is_valid = False
                    validation.errors.append("Symbol not in trading pairs")
                # Отмена невалидных ордеров
                if not validation.is_valid:
                    try:
                        await exchange.cancel_order(order["id"])
                        cancelled_orders.append(order["id"])
                        logger.info(
                            f"Cancelled invalid order {order['id']}: {validation.errors}"
                        )
                    except Exception as e:
                        logger.error(f"Error cancelling order {order['id']}: {e}")
            logger.info(f"Cancelled {len(cancelled_orders)} invalid orders")
            return cancelled_orders
        except Exception as e:
            logger.error(f"Error cancelling invalid orders: {e}")
            raise

    def calculate_order_sizing(
        self,
        account_balance: float,
        risk_per_trade: float,
        stop_loss_pct: float,
        current_price: float,
    ) -> float:
        """
        Расчет размера ордера на основе управления рисками.
        Args:
            account_balance: Баланс аккаунта
            risk_per_trade: Риск на сделку (в процентах)
            stop_loss_pct: Процент стоп-лосса
            current_price: Текущая цена
        Returns:
            float: Размер ордера
        """
        try:
            # Расчет риска в денежном выражении
            risk_amount = account_balance * (risk_per_trade / 100)
            # Расчет размера позиции
            position_size = risk_amount / (stop_loss_pct / 100)
            # Расчет количества единиц
            order_size = position_size / current_price
            # Ограничение размера
            order_size = max(self.min_order_size, min(order_size, self.max_order_size))
            return order_size
        except Exception as e:
            logger.error(f"Error calculating order sizing: {e}")
            return float(self.min_order_size)

    def create_oco_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        limit_price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Создание OCO (One-Cancels-Other) ордера.
        Args:
            symbol: Символ
            side: Сторона
            amount: Количество
            limit_price: Лимитная цена
            stop_price: Стоп-цена
            stop_limit_price: Стоп-лимитная цена
        Returns:
            Dict: OCO ордер
        """
        try:
            if stop_limit_price is None:
                stop_limit_price = stop_price
            oco_order = {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "type": "oco",
                "orders": [
                    {"type": "limit", "price": limit_price, "amount": amount},
                    {
                        "type": "stop_limit",
                        "stop_price": stop_price,
                        "price": stop_limit_price,
                        "amount": amount,
                    },
                ],
            }
            # Валидация
            validation = self.validate_order(oco_order)
            if not validation.is_valid:
                raise ValueError(f"Invalid OCO order: {validation.errors}")
            return oco_order
        except Exception as e:
            logger.error(f"Error creating OCO order: {e}")
            raise

    def get_order_statistics(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Получение статистики по ордерам.
        Args:
            orders: Список ордеров
        Returns:
            Dict: Статистика
        """
        try:
            if not orders:
                return {}
            stats = {
                "total_orders": len(orders),
                "filled_orders": 0,
                "cancelled_orders": 0,
                "rejected_orders": 0,
                "total_volume": 0.0,
                "total_fees": 0.0,
                "average_execution_time": 0.0,
                "average_slippage": 0.0,
            }
            execution_times = []
            slippages = []
            for order in orders:
                if order["status"] == "filled":
                    stats["filled_orders"] += 1
                    stats["total_volume"] += float(order.get("filled", 0))
                    stats["total_fees"] += float(order.get("fee", {}).get("cost", 0))
                    # Расчет времени исполнения
                    if "timestamp" in order and "datetime" in order:
                        execution_time = (
                            order["datetime"] - order["timestamp"]
                        ).total_seconds()
                        execution_times.append(execution_time)
                    # Расчет проскальзывания
                    if "price" in order and "average" in order:
                        slippage = abs(
                            float(order["average"]) - float(order["price"])
                        ) / float(order["price"])
                        slippages.append(slippage)
                elif order["status"] == "cancelled":
                    stats["cancelled_orders"] += 1
                elif order["status"] == "rejected":
                    stats["rejected_orders"] += 1
            # Средние значения
            if execution_times:
                stats["average_execution_time"] = sum(execution_times) / len(
                    execution_times
                )
            if slippages:
                stats["average_slippage"] = sum(slippages) / len(slippages)
            return stats
        except Exception as e:
            logger.error(f"Error calculating order statistics: {e}")
            return {}


# Функции для обратной совместимости
def is_valid_order(order: Dict[str, Any]) -> bool:
    """Проверка валидности ордера (для обратной совместимости)."""
    utils = OrderUtils()
    result = utils.validate_order(order)
    return result.is_valid


async def clear_invalid_orders(exchange: Any, trading_pairs: Dict[str, Any]) -> None:
    """Очистка невалидных ордеров (для обратной совместимости)."""
    utils = OrderUtils()
    await utils.cancel_invalid_orders(exchange, list(trading_pairs.keys()))
