"""
Модуль создания и управления ордерами.
Содержит промышленные функции для создания и управления ордерами,
включая smart order routing, TWAP, VWAP, iceberg orders.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from domain.entities.order import OrderSide, OrderStatus, OrderType
from domain.types import TimeInForceType

__all__ = [
    "create_market_order",
    "create_limit_order",
    "create_stop_order",
    "create_stop_limit_order",
    "create_twap_order",
    "create_vwap_order",
    "create_iceberg_order",
    "create_bracket_order",
    "calculate_optimal_order_size",
    "calculate_order_timing",
    "validate_order_parameters",
    "calculate_slippage_estimate",
    "optimize_order_execution",
    "create_smart_order",
]


def validate_order_parameters(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Optional[Decimal] = None,
    order_type: OrderType = OrderType.MARKET,
) -> bool:
    """Валидация параметров ордера."""
    if not symbol or not symbol.strip():
        return False
    if side not in [OrderSide.BUY, OrderSide.SELL]:
        return False
    if quantity <= 0:
        return False
    if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
        return False
    if price is not None and price <= 0:
        return False
    return True


def create_market_order(
    symbol: str, side: OrderSide, quantity: Decimal, time_in_force: TimeInForceType = "IOC"
) -> Dict[str, Any]:
    """Создание рыночного ордера."""
    if not validate_order_parameters(symbol, side, quantity):
        raise ValueError("Invalid order parameters")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.MARKET,
        "quantity": quantity,
        "time_in_force": time_in_force,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": calculate_slippage_estimate(symbol, side, quantity),
        "execution_priority": "high",
    }


def create_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    time_in_force: TimeInForceType = "GTC",
) -> Dict[str, Any]:
    """Создание лимитного ордера."""
    if not validate_order_parameters(symbol, side, quantity, price, OrderType.LIMIT):
        raise ValueError("Invalid order parameters")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.LIMIT,
        "quantity": quantity,
        "price": price,
        "time_in_force": time_in_force,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": Decimal("0"),  # Лимитные ордера не имеют проскальзывания
        "execution_priority": "medium",
    }


def create_stop_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    stop_price: Decimal,
    time_in_force: TimeInForceType = "GTC",
) -> Dict[str, Any]:
    """Создание стоп-ордера."""
    if not validate_order_parameters(symbol, side, quantity, stop_price):
        raise ValueError("Invalid order parameters")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.STOP,
        "quantity": quantity,
        "stop_price": stop_price,
        "time_in_force": time_in_force,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": calculate_slippage_estimate(symbol, side, quantity),
        "execution_priority": "high",
    }


def create_stop_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    stop_price: Decimal,
    limit_price: Decimal,
    time_in_force: TimeInForceType = "GTC",
) -> Dict[str, Any]:
    """Создание стоп-лимитного ордера."""
    if not validate_order_parameters(
        symbol, side, quantity, limit_price, OrderType.STOP_LIMIT
    ):
        raise ValueError("Invalid order parameters")
    if stop_price <= 0 or limit_price <= 0:
        raise ValueError("Invalid price parameters")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.STOP_LIMIT,
        "quantity": quantity,
        "stop_price": stop_price,
        "limit_price": limit_price,
        "time_in_force": time_in_force,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": Decimal("0"),
        "execution_priority": "medium",
    }


def create_twap_order(
    symbol: str,
    side: OrderSide,
    total_quantity: Decimal,
    duration_minutes: int,
    start_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Создание TWAP ордера (Time-Weighted Average Price)."""
    if not validate_order_parameters(symbol, side, total_quantity):
        raise ValueError("Invalid order parameters")
    if duration_minutes <= 0:
        raise ValueError("Duration must be positive")
    start_time = start_time or datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    # Разбиваем на подордера
    num_slices = max(1, duration_minutes // 5)  # Один срез каждые 5 минут
    slice_quantity = total_quantity / num_slices
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.TWAP,
        "total_quantity": total_quantity,
        "slice_quantity": slice_quantity,
        "num_slices": num_slices,
        "duration_minutes": duration_minutes,
        "start_time": start_time,
        "end_time": end_time,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": calculate_slippage_estimate(symbol, side, total_quantity)
        * Decimal("0.5"),
        "execution_priority": "medium",
        "twap_parameters": {
            "slice_interval_minutes": duration_minutes // num_slices,
            "volume_profile": "uniform",
            "price_improvement": True,
        },
    }


def create_vwap_order(
    symbol: str,
    side: OrderSide,
    total_quantity: Decimal,
    target_vwap: Optional[Decimal] = None,
    max_deviation: Optional[Decimal] = None,
) -> Dict[str, Any]:
    """Создание VWAP ордера (Volume-Weighted Average Price)."""
    if not validate_order_parameters(symbol, side, total_quantity):
        raise ValueError("Invalid order parameters")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.VWAP,
        "total_quantity": total_quantity,
        "target_vwap": target_vwap,
        "max_deviation": max_deviation or Decimal("0.01"),  # 1% по умолчанию
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": calculate_slippage_estimate(symbol, side, total_quantity)
        * Decimal("0.3"),
        "execution_priority": "medium",
        "vwap_parameters": {
            "volume_profile": "market",
            "price_improvement": True,
            "adaptive_sizing": True,
        },
    }


def create_iceberg_order(
    symbol: str,
    side: OrderSide,
    total_quantity: Decimal,
    visible_quantity: Decimal,
    price: Decimal,
    time_in_force: TimeInForceType = "GTC",
) -> Dict[str, Any]:
    """Создание айсберг-ордера."""
    if not validate_order_parameters(
        symbol, side, total_quantity, price, OrderType.LIMIT
    ):
        raise ValueError("Invalid order parameters")
    if visible_quantity <= 0 or visible_quantity > total_quantity:
        raise ValueError("Invalid visible quantity")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.ICEBERG,
        "total_quantity": total_quantity,
        "visible_quantity": visible_quantity,
        "price": price,
        "time_in_force": time_in_force,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": Decimal("0"),
        "execution_priority": "low",
        "iceberg_parameters": {
            "refill_strategy": "immediate",
            "min_refill_size": visible_quantity * Decimal("0.5"),
            "max_refill_delay": 30,  # секунды
        },
    }


def create_bracket_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    entry_price: Decimal,
    take_profit_price: Decimal,
    stop_loss_price: Decimal,
) -> Dict[str, Any]:
    """Создание брекет-ордера."""
    if not validate_order_parameters(symbol, side, quantity, entry_price):
        raise ValueError("Invalid order parameters")
    if take_profit_price <= entry_price or stop_loss_price >= entry_price:
        raise ValueError("Invalid take profit or stop loss prices")
    return {
        "symbol": symbol,
        "side": side,
        "order_type": OrderType.BRACKET,
        "quantity": quantity,
        "entry_price": entry_price,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "estimated_slippage": calculate_slippage_estimate(symbol, side, quantity),
        "execution_priority": "high",
        "bracket_parameters": {
            "entry_order": create_limit_order(symbol, side, quantity, entry_price),
            "take_profit_order": create_limit_order(
                symbol, side, quantity, take_profit_price
            ),
            "stop_loss_order": create_stop_order(
                symbol, side, quantity, stop_loss_price
            ),
        },
    }


def calculate_optimal_order_size(
    symbol: str,
    side: OrderSide,
    total_quantity: Decimal,
    market_data: pd.DataFrame,
    risk_limits: Optional[Dict[str, Decimal]] = None,
) -> Tuple[Decimal, int]:
    """Расчёт оптимального размера ордера."""
    if total_quantity <= 0:
        return Decimal("0"), 0
    # Базовые параметры
    base_size = total_quantity
    num_slices = 1
    # Анализируем рыночные данные
    if not market_data.empty and "volume" in market_data.columns:
        avg_volume = market_data["volume"].mean()
        current_volume = market_data["volume"].iloc[-1]
        # Корректируем размер на основе объёма
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio < 0.5:  # Низкий объём
            base_size *= Decimal("0.5")
            num_slices = 3
        elif volume_ratio > 2.0:  # Высокий объём
            base_size *= Decimal("1.5")
            num_slices = 1
        else:
            num_slices = 2
    # Применяем риск-лимиты
    if risk_limits:
        max_order_size = risk_limits.get("max_order_size", total_quantity)
        max_position_size = risk_limits.get("max_position_size", total_quantity)
        base_size = min(base_size, max_order_size, max_position_size)
    # Округляем до разумного размера
    optimal_size = base_size / num_slices
    return optimal_size, num_slices


def calculate_order_timing(
    symbol: str, market_data: pd.DataFrame, order_type: OrderType
) -> Dict[str, Any]:
    """Расчёт оптимального времени исполнения ордера."""
    timing: Dict[str, Any] = {
        "immediate": True,
        "delay_seconds": 0,
        "best_time_windows": [],
        "avoid_time_windows": [],
    }
    if market_data.empty:
        return timing
    # Анализируем волатильность
    if "close" in market_data.columns:
        returns = market_data["close"].pct_change().dropna()
        volatility = returns.std()
        # При высокой волатильности исполняем немедленно
        if volatility > 0.02:  # 2% волатильность
            timing["immediate"] = True
        else:
            timing["immediate"] = False
            timing["delay_seconds"] = 30
    # Анализируем объём
    if "volume" in market_data.columns:
        avg_volume = market_data["volume"].mean()
        current_volume = market_data["volume"].iloc[-1]
        # При низком объёме ждём
        if current_volume < avg_volume * 0.5:
            timing["immediate"] = False
            timing["delay_seconds"] = 60
    # Определяем лучшие временные окна
    current_hour = datetime.now().hour
    # Избегаем открытия/закрытия рынка
    if current_hour in [9, 10, 15, 16]:  # Примерные часы открытия/закрытия
                        timing["avoid_time_windows"].append(f"{current_hour}:00-{current_hour+1}:00")
    # Лучшие часы для торговли
    if 11 <= current_hour <= 14:
                        timing["best_time_windows"].append(f"{current_hour}:00-{current_hour+1}:00")
    return timing


def calculate_slippage_estimate(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    market_data: Optional[pd.DataFrame] = None,
) -> Decimal:
    """Расчёт ожидаемого проскальзывания."""
    base_slippage = Decimal("0.001")  # 0.1% базовое проскальзывание
    # Корректируем на основе размера ордера
    if quantity > Decimal("1000"):
        base_slippage *= Decimal("2")
    elif quantity > Decimal("100"):
        base_slippage *= Decimal("1.5")
    # Корректируем на основе рыночных данных
    if market_data is not None and not market_data.empty:
        if "volume" in market_data.columns:
            avg_volume = market_data["volume"].mean()
            current_volume = market_data["volume"].iloc[-1]
            # При низком объёме проскальзывание выше
            if current_volume < avg_volume * 0.5:
                base_slippage *= Decimal("2")
        if "close" in market_data.columns:
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std()
            # При высокой волатильности проскальзывание выше
            if volatility > 0.02:
                base_slippage *= Decimal("1.5")
    return base_slippage


def optimize_order_execution(
    order: Dict[str, Any],
    market_data: pd.DataFrame,
    execution_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Оптимизация исполнения ордера."""
    optimized_order = order.copy()
    if execution_parameters is None:
        execution_parameters = {}
    # Оптимизируем размер ордера
    if "total_quantity" in order:
        optimal_size, num_slices = calculate_optimal_order_size(
            order["symbol"],
            order["side"],
            order["total_quantity"],
            market_data,
            execution_parameters.get("risk_limits"),
        )
        optimized_order["optimal_size"] = optimal_size
        optimized_order["num_slices"] = num_slices
    # Оптимизируем время исполнения
    timing = calculate_order_timing(order["symbol"], market_data, order["order_type"])
    optimized_order["execution_timing"] = timing
    # Корректируем проскальзывание
    optimized_order["estimated_slippage"] = calculate_slippage_estimate(
        order["symbol"],
        order["side"],
        order.get("quantity", order.get("total_quantity", Decimal("0"))),
        market_data,
    )
    # Добавляем параметры оптимизации
    optimized_order["optimization_parameters"] = {
        "market_impact_model": "linear",
        "execution_urgency": execution_parameters.get("urgency", "normal"),
        "price_improvement": execution_parameters.get("price_improvement", True),
        "adaptive_sizing": execution_parameters.get("adaptive_sizing", True),
    }
    return optimized_order


def create_smart_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    order_type: OrderType,
    market_data: pd.DataFrame,
    execution_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Создание умного ордера с автоматической оптимизацией."""
    # Создаём базовый ордер
    if order_type == OrderType.MARKET:
        base_order = create_market_order(symbol, side, quantity)
    elif order_type == OrderType.LIMIT:
        price = execution_parameters.get("price") if execution_parameters else None
        if price is None:
            raise ValueError("Price required for limit order")
        base_order = create_limit_order(symbol, side, quantity, price)
    elif order_type == OrderType.TWAP:
        duration = (
            execution_parameters.get("duration_minutes", 30)
            if execution_parameters
            else 30
        )
        base_order = create_twap_order(symbol, side, quantity, duration)
    elif order_type == OrderType.VWAP:
        base_order = create_vwap_order(symbol, side, quantity)
    else:
        raise ValueError(f"Unsupported order type: {order_type}")
    # Оптимизируем исполнение
    optimized_order = optimize_order_execution(
        base_order, market_data, execution_parameters
    )
    # Добавляем метаданные
    optimized_order["smart_order"] = True
    optimized_order["optimization_timestamp"] = datetime.now()
    optimized_order["market_conditions"] = {
        "volatility": (
            float(market_data["close"].pct_change().std())
            if not market_data.empty
            else 0.0
        ),
        "volume_profile": "normal",
        "liquidity_score": calculate_liquidity_score(symbol, market_data),
    }
    return optimized_order


def calculate_liquidity_score(symbol: str, market_data: pd.DataFrame) -> float:
    """Расчёт оценки ликвидности."""
    if market_data.empty or "volume" not in market_data.columns:
        return 0.5
    # Нормализуем объём
    avg_volume = market_data["volume"].mean()
    current_volume = market_data["volume"].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    # Рассчитываем спред (если есть данные)
    spread_score = 0.5
    if "high" in market_data.columns and "low" in market_data.columns:
        avg_spread = (
            (market_data["high"] - market_data["low"]) / market_data["close"]
        ).mean()
        spread_score = max(0.1, min(1.0, 1.0 - avg_spread * 100))
    # Комбинируем метрики
    liquidity_score = volume_ratio * 0.7 + spread_score * 0.3
    return float(max(0.0, min(1.0, liquidity_score)))
