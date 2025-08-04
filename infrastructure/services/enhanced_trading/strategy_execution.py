"""
Модуль исполнения торговых стратегий.
Содержит промышленные функции для исполнения торговых стратегий,
включая position sizing, risk management, execution algorithms.
"""

import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Type aliases for better mypy support
Series = pd.Series
DataFrame = pd.DataFrame

# Типы для стратегий и алгоритмов
StrategyType = Literal["mean_reversion", "momentum", "arbitrage", "scalping"]
ExecutionAlgorithm = Literal["twap", "vwap", "pov", "iceberg"]
RiskLevel = Literal["low", "medium", "high"]

# Константы для совместимости
class StrategyTypes:
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"

STRATEGY_TYPES = {
    "mean_reversion": StrategyTypes.MEAN_REVERSION,
    "momentum": StrategyTypes.MOMENTUM,
    "arbitrage": StrategyTypes.ARBITRAGE,
    "scalping": StrategyTypes.SCALPING,
}
EXECUTION_ALGORITHMS = {
    "twap": "twap",
    "vwap": "vwap",
    "pov": "pov",
    "iceberg": "iceberg",
}
RISK_LEVELS = {"low": "low", "medium": "medium", "high": "high"}
__all__ = [
    "calculate_position_size",
    "calculate_risk_metrics",
    "apply_risk_management",
    "execute_algorithm",
    "monitor_execution",
    "calculate_performance_metrics",
    "optimize_strategy_parameters",
    "validate_strategy_parameters",
    "create_execution_plan",
]


def validate_strategy_parameters(
    strategy_type: StrategyType, parameters: Dict[str, Any]
) -> bool:
    """Валидация параметров стратегии."""
    if strategy_type not in ["mean_reversion", "momentum", "arbitrage", "scalping"]:
        return False
    required_params = {
        "mean_reversion": ["lookback_period", "entry_threshold", "exit_threshold"],
        "momentum": ["momentum_period", "entry_threshold", "stop_loss"],
        "arbitrage": ["spread_threshold", "max_holding_time"],
        "scalping": ["profit_target", "stop_loss", "max_holding_time"],
    }
    required = required_params.get(strategy_type, [])
    return all(param in parameters for param in required)


def calculate_position_size(
    capital: Decimal,
    risk_per_trade: Decimal,
    entry_price: Decimal,
    stop_loss_price: Optional[Decimal] = None,
    volatility: Optional[Decimal] = None,
    risk_level: RiskLevel = "medium",
) -> Decimal:
    """Расчёт размера позиции на основе риска."""
    if capital <= 0 or risk_per_trade <= 0 or entry_price <= 0:
        return Decimal("0")
    # Базовый размер позиции
    risk_amount = capital * risk_per_trade
    # Корректируем на основе уровня риска
    risk_multipliers = {
        "low": Decimal("0.5"),
        "medium": Decimal("1.0"),
        "high": Decimal("1.5"),
    }
    risk_multiplier = risk_multipliers.get(risk_level, Decimal("1.0"))
    adjusted_risk_amount = risk_amount * risk_multiplier
    # Если есть стоп-лосс, рассчитываем размер на основе риска на сделку
    if stop_loss_price and stop_loss_price > 0:
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        if price_risk > 0:
            position_size = adjusted_risk_amount / price_risk
        else:
            position_size = adjusted_risk_amount / Decimal("0.02")  # 2% по умолчанию
    else:
        # Используем волатильность для расчёта
        if volatility and volatility > 0:
            position_size = adjusted_risk_amount / volatility
        else:
            position_size = adjusted_risk_amount / Decimal("0.02")  # 2% по умолчанию
    # Ограничиваем размер позиции
    max_position_size = capital * Decimal("0.1")  # Максимум 10% капитала
    position_size = min(position_size, max_position_size)
    return position_size


def calculate_risk_metrics(
    positions: List[Dict[str, Any]], market_data: pd.DataFrame, capital: Decimal
) -> Dict[str, Decimal]:
    """Расчёт метрик риска для портфеля."""
    if not positions:
        return {
            "total_exposure": Decimal("0"),
            "var_95": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "sharpe_ratio": Decimal("0"),
            "correlation": Decimal("0"),
        }
    # Общая экспозиция
    total_exposure = sum(pos.get("market_value", Decimal("0")) for pos in positions)
    # Value at Risk (упрощённо)
    position_values = [pos.get("market_value", Decimal("0")) for pos in positions]
    var_95 = sum(position_values) * Decimal("0.02")  # 2% VaR
    # Максимальная просадка (упрощённо)
    max_drawdown = total_exposure * Decimal("0.05")  # 5% max drawdown
    # Коэффициент Шарпа (упрощённо)
    if (
        market_data is not None
        and not market_data.empty
        and "close" in market_data.columns
    ):
        returns = market_data["close"].pct_change().dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            if std_return > 0:
                sharpe_ratio = Decimal(str(mean_return / std_return))
            else:
                sharpe_ratio = Decimal("0")
        else:
            sharpe_ratio = Decimal("0")
    else:
        sharpe_ratio = Decimal("0")
    # Корреляция позиций (упрощённо)
    correlation = Decimal("0.3")  # Средняя корреляция
    return {
        "total_exposure": total_exposure,
        "var_95": var_95,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "correlation": correlation,
    }


def apply_risk_management(
    positions: List[Dict[str, Any]],
    risk_limits: Dict[str, Decimal],
    market_data: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Применение риск-менеджмента к позициям."""
    managed_positions = positions.copy()
    # Проверяем лимиты экспозиции
    total_exposure = sum(
        pos.get("market_value", Decimal("0")) for pos in managed_positions
    )
    max_exposure = risk_limits.get("max_exposure", Decimal("1.0"))
    if total_exposure > max_exposure:
        # Уменьшаем позиции пропорционально
        reduction_factor = max_exposure / total_exposure
        for pos in managed_positions:
            if "quantity" in pos:
                pos["quantity"] *= reduction_factor
                pos["market_value"] *= reduction_factor
    # Проверяем лимиты по символам
    symbol_exposure: Dict[str, Decimal] = {}
    for pos in managed_positions:
        symbol = pos.get("symbol", "")
        if symbol:
            symbol_exposure[symbol] = symbol_exposure.get(
                symbol, Decimal("0")
            ) + pos.get("market_value", Decimal("0"))
    max_symbol_exposure = risk_limits.get("max_symbol_exposure", Decimal("0.2"))
    for pos in managed_positions:
        symbol = pos.get("symbol", "")
        if symbol and symbol_exposure.get(symbol, Decimal("0")) > max_symbol_exposure:
            # Уменьшаем позицию
            reduction_factor = max_symbol_exposure / symbol_exposure[symbol]
            if "quantity" in pos:
                pos["quantity"] *= reduction_factor
                pos["market_value"] *= reduction_factor
    # Применяем стоп-лоссы
    for pos in managed_positions:
        if "stop_loss" in pos and "current_price" in pos:
            if pos["current_price"] <= pos["stop_loss"]:
                pos["action"] = "close"
                pos["reason"] = "stop_loss_hit"
    # Применяем тейк-профиты
    for pos in managed_positions:
        if "take_profit" in pos and "current_price" in pos:
            if pos["current_price"] >= pos["take_profit"]:
                pos["action"] = "close"
                pos["reason"] = "take_profit_hit"
    return managed_positions


def execute_algorithm(
    algorithm: ExecutionAlgorithm, market_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Исполнение торгового алгоритма."""
    if algorithm == "twap":
        return execute_twap_algorithm(market_data, parameters)
    elif algorithm == "vwap":
        return execute_vwap_algorithm(market_data, parameters)
    elif algorithm == "pov":
        return execute_pov_algorithm(market_data, parameters)
    elif algorithm == "iceberg":
        return execute_iceberg_algorithm(market_data, parameters)
    else:
        raise ValueError(f"Unsupported execution algorithm: {algorithm}")


def execute_twap_algorithm(
    market_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Исполнение TWAP алгоритма."""
    total_quantity = parameters.get("total_quantity", Decimal("0"))
    duration_minutes = parameters.get("duration_minutes", 30)
    start_time = parameters.get("start_time", datetime.now())
    # Разбиваем на равные части
    num_slices = max(1, duration_minutes // 5)
    slice_quantity = total_quantity / num_slices
    execution_plan = {
        "algorithm": "twap",
        "total_quantity": total_quantity,
        "slice_quantity": slice_quantity,
        "num_slices": num_slices,
        "slices": [],
        "execution_status": "pending",
    }
    # Создаём срезы
    for i in range(num_slices):
        slice_time = start_time + timedelta(
            minutes=i * (duration_minutes // num_slices)
        )
        execution_plan["slices"].append(
            {
                "slice_id": i + 1,
                "quantity": slice_quantity,
                "execution_time": slice_time,
                "status": "pending",
            }
        )
    return execution_plan


def execute_vwap_algorithm(
    market_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Исполнение VWAP алгоритма."""
    total_quantity = parameters.get("total_quantity", Decimal("0"))
    target_vwap = parameters.get("target_vwap")
    max_deviation = parameters.get("max_deviation", Decimal("0.01"))
    if market_data.empty or "volume" not in market_data.columns:
        return {"error": "Insufficient market data for VWAP"}
    # Рассчитываем VWAP
    typical_price = (
        market_data["high"] + market_data["low"] + market_data["close"]
    ) / 3
    vwap = (typical_price * market_data["volume"]).sum() / market_data["volume"].sum()
    execution_plan = {
        "algorithm": "vwap",
        "total_quantity": total_quantity,
        "target_vwap": target_vwap or vwap,
        "current_vwap": vwap,
        "max_deviation": max_deviation,
        "execution_status": "pending",
        "volume_profile": calculate_volume_profile(market_data),
    }
    return execution_plan


def execute_pov_algorithm(
    market_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Исполнение POV (Percentage of Volume) алгоритма."""
    total_quantity = parameters.get("total_quantity", Decimal("0"))
    pov_percentage = parameters.get(
        "pov_percentage", Decimal("0.1")
    )  # 10% по умолчанию
    if market_data.empty or "volume" not in market_data.columns:
        return {"error": "Insufficient market data for POV"}
    avg_volume = market_data["volume"].mean()
    target_volume_per_slice = avg_volume * pov_percentage
    execution_plan = {
        "algorithm": "pov",
        "total_quantity": total_quantity,
        "pov_percentage": pov_percentage,
        "target_volume_per_slice": target_volume_per_slice,
        "execution_status": "pending",
        "slices": [],
    }
    return execution_plan


def execute_iceberg_algorithm(
    market_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Исполнение Iceberg алгоритма."""
    total_quantity = parameters.get("total_quantity", Decimal("0"))
    visible_quantity = parameters.get(
        "visible_quantity", total_quantity * Decimal("0.1")
    )
    price = parameters.get("price")
    if not price:
        return {"error": "Price required for Iceberg algorithm"}
    execution_plan = {
        "algorithm": "iceberg",
        "total_quantity": total_quantity,
        "visible_quantity": visible_quantity,
        "hidden_quantity": total_quantity - visible_quantity,
        "price": price,
        "execution_status": "pending",
        "refill_strategy": "immediate",
    }
    return execution_plan


def calculate_volume_profile(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Расчёт профиля объёма по часам."""
    if market_data.empty or "volume" not in market_data.columns:
        return {}
    
    # Создаем копию для безопасного изменения
    if hasattr(market_data, 'copy'):
        data_copy = market_data.copy()
    else:
        data_copy = market_data
    
    # Группируем по часам
    # Добавляем временные признаки
    if hasattr(data_copy.index, 'hour'):
        data_copy["hour"] = data_copy.index.hour
    else:
        # Безопасное извлечение часа из индекса
        datetime_index = pd.to_datetime(data_copy.index)
        if hasattr(datetime_index, 'hour'):
            data_copy["hour"] = datetime_index.hour
        else:
            # Альтернативный способ для DatetimeIndex
            if hasattr(datetime_index, '__iter__'):
                data_copy["hour"] = [dt.hour if hasattr(dt, 'hour') else 0 for dt in datetime_index]
            else:
                # Если это одиночный Timestamp
                data_copy["hour"] = [datetime_index.hour if hasattr(datetime_index, 'hour') else 0]
    
    hourly_volume = data_copy.groupby("hour")["volume"].mean()
    
    # Нормализуем
    total_volume = hourly_volume.sum()
    if total_volume > 0:
        volume_profile = (hourly_volume / total_volume).to_dict()
    else:
        volume_profile = {hour: 1.0 / 24 for hour in range(24)}
    
    return {
        "hourly_profile": volume_profile,
        "peak_hours": [
            hour for hour, vol in volume_profile.items() if vol > 1.0 / 24 * 1.5
        ],
        "low_volume_hours": [
            hour for hour, vol in volume_profile.items() if vol < 1.0 / 24 * 0.5
        ],
    }


def monitor_execution(
    execution_plan: Dict[str, Any],
    market_data: pd.DataFrame,
    current_positions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Мониторинг исполнения."""
    monitoring_result = {
        "execution_status": execution_plan.get("execution_status", "unknown"),
        "progress": Decimal("0"),
        "performance_metrics": {},
        "alerts": [],
        "recommendations": [],
    }
    # Рассчитываем прогресс
    if "slices" in execution_plan:
        completed_slices = sum(
            1
            for slice_data in execution_plan["slices"]
            if slice_data.get("status") == "completed"
        )
        total_slices = len(execution_plan["slices"])
        if total_slices > 0:
            monitoring_result["progress"] = Decimal(
                str(completed_slices / total_slices)
            )
    # Рассчитываем метрики производительности
    if market_data is not None and not market_data.empty:
        monitoring_result["performance_metrics"] = calculate_performance_metrics(
            execution_plan, market_data, current_positions
        )
    # Проверяем алерты
    alerts = check_execution_alerts(execution_plan, market_data)
    monitoring_result["alerts"] = alerts
    # Генерируем рекомендации
    recommendations = generate_execution_recommendations(
        execution_plan, market_data, alerts
    )
    monitoring_result["recommendations"] = recommendations
    return monitoring_result


def calculate_performance_metrics(
    execution_plan: Dict[str, Any],
    market_data: pd.DataFrame,
    current_positions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Расчёт метрик производительности исполнения."""
    metrics = {
        "execution_quality": Decimal("0"),
        "market_impact": Decimal("0"),
        "slippage": Decimal("0"),
        "fill_rate": Decimal("0"),
        "cost_savings": Decimal("0"),
    }
    # Упрощённый расчёт метрик
    if market_data is not None and not market_data.empty:
        # Качество исполнения (на основе отклонения от VWAP)
        if "close" in market_data.columns and "volume" in market_data.columns:
            typical_price = (
                market_data["high"] + market_data["low"] + market_data["close"]
            ) / 3
            vwap = (typical_price * market_data["volume"]).sum() / market_data[
                "volume"
            ].sum()
            current_price = market_data["close"].iloc[-1]
            if vwap > 0:
                execution_quality = 1 - abs(current_price - vwap) / vwap
                metrics["execution_quality"] = Decimal(str(max(0, execution_quality)))
        # Рыночное воздействие (упрощённо)
        if "volume" in market_data.columns:
            avg_volume = market_data["volume"].mean()
            current_volume = market_data["volume"].iloc[-1]
            if avg_volume > 0:
                market_impact = min(1.0, current_volume / avg_volume)
                metrics["market_impact"] = Decimal(str(market_impact))
    return metrics


def check_execution_alerts(
    execution_plan: Dict[str, Any], market_data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Проверка алертов исполнения."""
    alerts = []
    # Проверяем отклонение от плана
    if "progress" in execution_plan:
        expected_progress = calculate_expected_progress(execution_plan)
        actual_progress = execution_plan["progress"]
        if abs(actual_progress - expected_progress) > 0.2:  # 20% отклонение
            alerts.append(
                {
                    "type": "execution_delay",
                    "severity": "medium",
                    "message": f"Execution progress {actual_progress:.1%} vs expected {expected_progress:.1%}",
                    "timestamp": datetime.now(),
                }
            )
    # Проверяем рыночные условия
    if market_data is not None and not market_data.empty:
        if "volume" in market_data.columns:
            current_volume = market_data["volume"].iloc[-1]
            avg_volume = market_data["volume"].mean()
            if current_volume < avg_volume * 0.5:
                alerts.append(
                    {
                        "type": "low_liquidity",
                        "severity": "high",
                        "message": "Low market liquidity detected",
                        "timestamp": datetime.now(),
                    }
                )
        if "close" in market_data.columns:
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std()
            if volatility > 0.03:  # 3% волатильность
                alerts.append(
                    {
                        "type": "high_volatility",
                        "severity": "medium",
                        "message": f"High volatility detected: {volatility:.2%}",
                        "timestamp": datetime.now(),
                    }
                )
    return alerts


def calculate_expected_progress(execution_plan: Dict[str, Any]) -> float:
    """Расчёт ожидаемого прогресса исполнения."""
    if "start_time" in execution_plan and "duration_minutes" in execution_plan:
        start_time = execution_plan["start_time"]
        duration = execution_plan["duration_minutes"]
        if isinstance(start_time, datetime) and isinstance(duration, (int, float)):
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            return min(1.0, elapsed / duration)
    return 0.0


def generate_execution_recommendations(
    execution_plan: Dict[str, Any],
    market_data: pd.DataFrame,
    alerts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Генерация рекомендаций по исполнению."""
    recommendations = []
    # Рекомендации на основе алертов
    for alert in alerts:
        if alert["type"] == "low_liquidity":
            recommendations.append(
                {
                    "action": "reduce_order_size",
                    "reason": "Low market liquidity",
                    "priority": "high",
                }
            )
        elif alert["type"] == "high_volatility":
            recommendations.append(
                {
                    "action": "increase_slippage_tolerance",
                    "reason": "High market volatility",
                    "priority": "medium",
                }
            )
        elif alert["type"] == "execution_delay":
            recommendations.append(
                {
                    "action": "accelerate_execution",
                    "reason": "Execution behind schedule",
                    "priority": "medium",
                }
            )
    # Рекомендации на основе рыночных данных
    if market_data is not None and not market_data.empty:
        if "volume" in market_data.columns:
            current_volume = market_data["volume"].iloc[-1]
            avg_volume = market_data["volume"].mean()
            if current_volume > avg_volume * 2:
                recommendations.append(
                    {
                        "action": "increase_order_size",
                        "reason": "High market volume",
                        "priority": "low",
                    }
                )
    return recommendations


def optimize_strategy_parameters(
    strategy_type: StrategyType,
    historical_data: pd.DataFrame,
    performance_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Оптимизация параметров стратегии."""
    optimized_params = {}
    if strategy_type == StrategyTypes.MEAN_REVERSION:
        # Оптимизируем параметры mean reversion
        if not historical_data.empty and "close" in historical_data.columns:
            returns = historical_data["close"].pct_change().dropna()
            # Оптимальный период lookback на основе автокорреляции
            autocorr = [returns.autocorr(lag=i) for i in range(1, 21)]
            optimal_lookback = autocorr.index(max(autocorr)) + 1 if autocorr else 10
            optimized_params = {
                "lookback_period": optimal_lookback,
                "entry_threshold": Decimal("0.02"),  # 2%
                "exit_threshold": Decimal("0.01"),  # 1%
            }
    elif strategy_type == StrategyTypes.MOMENTUM:
        # Оптимизируем параметры momentum
        if not historical_data.empty and "close" in historical_data.columns:
            returns = historical_data["close"].pct_change().dropna()
            # Оптимальный период momentum на основе волатильности
            volatility = returns.std()
            optimal_period = (
                max(5, min(50, int(20 / float(volatility)))) if volatility > 0 else 20
            )
            optimized_params = {
                "momentum_period": optimal_period,
                "entry_threshold": Decimal("0.01"),  # 1%
                "stop_loss": Decimal("0.02"),  # 2%
            }
    return optimized_params


def create_execution_plan(
    strategy_type: StrategyType,
    parameters: Dict[str, Any],
    market_data: pd.DataFrame,
    capital: Decimal,
) -> Dict[str, Any]:
    """Создание плана исполнения стратегии."""
    if not validate_strategy_parameters(strategy_type, parameters):
        raise ValueError("Invalid strategy parameters")
    execution_plan: Dict[str, Any] = {
        "strategy_type": strategy_type,
        "parameters": parameters,
        "execution_steps": [],
        "risk_limits": {
            "max_exposure": Decimal("0.8"),
            "max_symbol_exposure": Decimal("0.2"),
            "max_drawdown": Decimal("0.05"),
        },
        "performance_targets": {
            "target_return": Decimal("0.1"),
            "max_volatility": Decimal("0.15"),
            "sharpe_ratio": Decimal("1.0"),
        },
        "created_at": datetime.now(),
        "status": "pending",
    }
    # Создаём шаги исполнения в зависимости от типа стратегии
    if strategy_type == StrategyTypes.MEAN_REVERSION:
        execution_plan["execution_steps"] = create_mean_reversion_steps(
            parameters, market_data
        )
    elif strategy_type == StrategyTypes.MOMENTUM:
        execution_plan["execution_steps"] = create_momentum_steps(
            parameters, market_data
        )
    elif strategy_type == StrategyTypes.SCALPING:
        execution_plan["execution_steps"] = create_scalping_steps(
            parameters, market_data
        )
    return execution_plan


def create_mean_reversion_steps(
    parameters: Dict[str, Any], market_data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Создание шагов исполнения для mean reversion стратегии."""
    steps = []
    lookback_period = parameters.get("lookback_period", 20)
    entry_threshold = parameters.get("entry_threshold", Decimal("0.02"))
    exit_threshold = parameters.get("exit_threshold", Decimal("0.01"))
    # Шаг 1: Анализ отклонения от среднего
    steps.append(
        {
            "step_id": 1,
            "action": "analyze_deviation",
            "parameters": {
                "lookback_period": lookback_period,
                "entry_threshold": entry_threshold,
            },
            "status": "pending",
        }
    )
    # Шаг 2: Открытие позиции
    steps.append(
        {
            "step_id": 2,
            "action": "open_position",
            "parameters": {
                "order_type": "limit",
                "side": "buy",  # или 'sell' в зависимости от сигнала
                "quantity": "calculated",
                "price": "market",
            },
            "status": "pending",
        }
    )
    # Шаг 3: Мониторинг и выход
    steps.append(
        {
            "step_id": 3,
            "action": "monitor_and_exit",
            "parameters": {
                "exit_threshold": exit_threshold,
                "max_holding_time": 24,  # часы
            },
            "status": "pending",
        }
    )
    return steps


def create_momentum_steps(
    parameters: Dict[str, Any], market_data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Создание шагов исполнения для momentum стратегии."""
    steps = []
    momentum_period = parameters.get("momentum_period", 10)
    entry_threshold = parameters.get("entry_threshold", Decimal("0.01"))
    stop_loss = parameters.get("stop_loss", Decimal("0.02"))
    # Шаг 1: Расчёт momentum
    steps.append(
        {
            "step_id": 1,
            "action": "calculate_momentum",
            "parameters": {
                "period": momentum_period,
                "entry_threshold": entry_threshold,
            },
            "status": "pending",
        }
    )
    # Шаг 2: Открытие позиции
    steps.append(
        {
            "step_id": 2,
            "action": "open_position",
            "parameters": {
                "order_type": "market",
                "side": "buy",  # или 'sell' в зависимости от сигнала
                "quantity": "calculated",
            },
            "status": "pending",
        }
    )
    # Шаг 3: Управление риском
    steps.append(
        {
            "step_id": 3,
            "action": "risk_management",
            "parameters": {"stop_loss": stop_loss, "trailing_stop": True},
            "status": "pending",
        }
    )
    return steps


def create_scalping_steps(
    parameters: Dict[str, Any], market_data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Создание шагов исполнения для scalping стратегии."""
    steps = []
    profit_target = parameters.get("profit_target", Decimal("0.005"))
    stop_loss = parameters.get("stop_loss", Decimal("0.003"))
    max_holding_time = parameters.get("max_holding_time", 1)  # минуты
    # Шаг 1: Быстрый анализ
    steps.append(
        {
            "step_id": 1,
            "action": "quick_analysis",
            "parameters": {"timeframe": "1m", "indicators": ["rsi", "macd"]},
            "status": "pending",
        }
    )
    # Шаг 2: Быстрое исполнение
    steps.append(
        {
            "step_id": 2,
            "action": "fast_execution",
            "parameters": {
                "order_type": "market",
                "execution_algorithm": "immediate",
                "quantity": "calculated",
            },
            "status": "pending",
        }
    )
    # Шаг 3: Быстрый выход
    steps.append(
        {
            "step_id": 3,
            "action": "quick_exit",
            "parameters": {
                "profit_target": profit_target,
                "stop_loss": stop_loss,
                "max_holding_time": max_holding_time,
            },
            "status": "pending",
        }
    )
    return steps
