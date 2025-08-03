"""
Валидаторы для стратегий.
"""

import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from domain.entities.market import MarketData, MarketState
from domain.entities.strategy import Signal, StrategyStatus, StrategyType
from domain.strategies.exceptions import (
    StrategyConfigurationError,
    StrategyParameterError,
    StrategyValidationError,
)
from domain.types import RiskLevel

logger = logging.getLogger(__name__)


class StrategyValidator:
    """Валидатор стратегий."""

    @staticmethod
    def validate_strategy_config(config: Dict[str, Any]) -> List[str]:
        """
        Валидировать конфигурацию стратегии.
        Args:
            config: Конфигурация стратегии
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка обязательных полей
        required_fields = ["name", "strategy_type", "trading_pairs"]
        for field in required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")
        # Валидация имени стратегии
        if "name" in config:
            name_errors = StrategyValidator._validate_strategy_name(config["name"])
            errors.extend(name_errors)
        # Валидация типа стратегии
        if "strategy_type" in config:
            type_errors = StrategyValidator._validate_strategy_type(
                config["strategy_type"]
            )
            errors.extend(type_errors)
        # Валидация торговых пар
        if "trading_pairs" in config:
            pairs_errors = StrategyValidator._validate_trading_pairs(
                config["trading_pairs"]
            )
            errors.extend(pairs_errors)
        # Валидация параметров
        if "parameters" in config:
            params_errors = StrategyValidator.validate_parameters(config["parameters"])
            errors.extend(params_errors)
        # Валидация метаданных
        if "metadata" in config:
            metadata_errors = StrategyValidator._validate_metadata(config["metadata"])
            errors.extend(metadata_errors)
        return errors

    @staticmethod
    def validate_market_data(market_data: MarketData) -> List[str]:
        """
        Валидировать рыночные данные.
        Args:
            market_data: Рыночные данные
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка обязательных полей
        if not market_data.symbol:
            errors.append("Market data must have a symbol")
        if not market_data.timestamp:
            errors.append("Market data must have a timestamp")
        # Проверка цен
        if market_data.open.value <= 0:
            errors.append("Open price must be positive")
        if market_data.high.value <= 0:
            errors.append("High price must be positive")
        if market_data.low.value <= 0:
            errors.append("Low price must be positive")
        if market_data.close.value <= 0:
            errors.append("Close price must be positive")
        # Проверка логики цен
        if market_data.high.value < market_data.low.value:
            errors.append("High price cannot be less than low price")
        if market_data.high.value < market_data.open.value:
            errors.append("High price cannot be less than open price")
        if market_data.high.value < market_data.close.value:
            errors.append("High price cannot be less than close price")
        if market_data.low.value > market_data.open.value:
            errors.append("Low price cannot be greater than open price")
        if market_data.low.value > market_data.close.value:
            errors.append("Low price cannot be greater than close price")
        return errors

    @staticmethod
    def validate_signal(signal: Signal) -> List[str]:
        """
        Валидировать торговый сигнал.
        Args:
            signal: Торговый сигнал
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка обязательных полей
        if not signal.trading_pair:
            errors.append("Signal must have a trading pair")
        if not signal.timestamp:
            errors.append("Signal must have a timestamp")
        if not signal.signal_type:
            errors.append("Signal must have a signal type")
        # Проверка цены
        if signal.price is not None and signal.price.value <= 0:
            errors.append("Signal price must be positive")
        # Проверка количества
        if signal.quantity is not None and signal.quantity <= 0:
            errors.append("Signal quantity must be positive")
        # Проверка уверенности
        if signal.confidence < 0 or signal.confidence > 1:
            errors.append("Signal confidence must be between 0 and 1")
        return errors

    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> List[str]:
        errors = []
        if not isinstance(parameters, dict):
            errors.append("Parameters must be a dictionary")
            return errors
        else:
            # Проверка обязательных параметров
            required_params = ["risk_level", "time_horizon"]
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            # Валидация уровня риска
            if "risk_level" in parameters:
                risk_errors = StrategyValidator.validate_risk_level(parameters["risk_level"])
                errors.extend(risk_errors)
            # Валидация временного горизонта
            if "time_horizon" in parameters:
                horizon_errors = StrategyValidator.validate_time_horizon(parameters["time_horizon"])
                errors.extend(horizon_errors)
            # Валидация рыночных условий
            if "market_condition" in parameters:
                condition_errors = StrategyValidator.validate_market_condition(parameters["market_condition"])
                errors.extend(condition_errors)
            return errors

    @staticmethod
    def validate_trading_pair(trading_pair: str) -> List[str]:
        errors = []
        if not isinstance(trading_pair, str):
            errors.append("Trading pair must be a string")
            return errors
        else:
            if not trading_pair:
                errors.append("Trading pair cannot be empty")
                return errors
            else:
                # Проверка формата (например, BTCUSDT)
                if not re.match(r'^[A-Z0-9]+$', trading_pair):
                    errors.append("Trading pair must contain only uppercase letters and numbers")
                if len(trading_pair) < 6:
                    errors.append("Trading pair is too short")
                if len(trading_pair) > 20:
                    errors.append("Trading pair is too long")
                return errors

    @staticmethod
    def validate_risk_level(risk_level: Union[str, Decimal]) -> List[str]:
        errors = []
        if isinstance(risk_level, str):
            try:
                rl_value = Decimal(risk_level)
                if rl_value < 0 or rl_value > 1:
                    errors.append("Risk level must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append(f"Invalid risk level: {risk_level}")
        elif isinstance(risk_level, Decimal):
            if risk_level < 0 or risk_level > 1:
                errors.append("Risk level must be between 0 and 1")
        else:
            errors.append("Risk level must be a string or Decimal")
        return errors

    @staticmethod
    def validate_time_horizon(time_horizon: Union[str, Enum]) -> List[str]:
        errors = []
        if isinstance(time_horizon, str):
            valid_horizons = ["short", "medium", "long"]
            if time_horizon not in valid_horizons:
                errors.append(
                    f"Invalid time horizon: {time_horizon}. Valid horizons: {valid_horizons}"
                )
        elif isinstance(time_horizon, Enum):
            pass
        else:
            errors.append("Time horizon must be a string or Enum")
        return errors

    @staticmethod
    def validate_market_condition(condition: Union[str, Enum]) -> List[str]:
        errors = []
        if isinstance(condition, str):
            valid_conditions = ["bull", "bear", "sideways", "volatile"]
            if condition not in valid_conditions:
                errors.append(
                    f"Invalid market condition: {condition}. Valid conditions: {valid_conditions}"
                )
        elif isinstance(condition, Enum):
            pass
        else:
            errors.append("Market condition must be a string or Enum")
        return errors

    @staticmethod
    def _validate_strategy_name(name: str) -> List[str]:
        errors = []
        if not name:
            errors.append("Strategy name cannot be empty")
        elif len(name) < 3:
            errors.append("Strategy name must be at least 3 characters long")
        elif len(name) > 100:
            errors.append("Strategy name is too long (max 100 characters)")
        elif re.search(r'[<>:"/\\|?*]', name):
            errors.append("Strategy name contains invalid characters")
        return errors

    @staticmethod
    def _validate_strategy_type(strategy_type: str) -> List[str]:
        errors = []
        valid_types = [st.value for st in StrategyType]
        if strategy_type not in valid_types:
            errors.append(
                f"Invalid strategy type: {strategy_type}. Valid types: {valid_types}"
            )
        return errors

    @staticmethod
    def _validate_trading_pairs(trading_pairs: List[str]) -> List[str]:
        errors = []
        if not isinstance(trading_pairs, list):
            errors.append("Trading pairs must be a list")
            return errors
        elif not trading_pairs:
            errors.append("At least one trading pair is required")
            return errors
        else:
            if len(trading_pairs) > 50:
                errors.append("Too many trading pairs (max 50)")
            for pair in trading_pairs:
                pair_errors = StrategyValidator.validate_trading_pair(pair)
                errors.extend(pair_errors)
            if len(trading_pairs) != len(set(trading_pairs)):
                errors.append("Duplicate trading pairs are not allowed")
            return errors

    @staticmethod
    def _validate_metadata(metadata: Dict[str, Any]) -> List[str]:
        errors = []
        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
            return errors
        elif len(metadata) > 100:
            errors.append("Too many metadata entries (max 100)")
            return errors
        else:
            for key, value in metadata.items():
                if not isinstance(key, str):
                    errors.append("Metadata keys must be strings")
                if len(key) > 50:
                    errors.append("Metadata key is too long (max 50 characters)")
                if not isinstance(value, (str, int, float, bool)):
                    errors.append("Metadata values must be primitive types")
            return errors


class ParameterValidator:
    """Валидатор параметров стратегий."""

    @staticmethod
    def validate_trend_following_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры тренд-следующей стратегии."""
        errors = []
        required_params = ["ma_period", "stop_loss", "take_profit"]
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        # Валидация периода MA
        if "ma_period" in params:
            ma_period = params["ma_period"]
            if not isinstance(ma_period, int) or ma_period <= 0:
                errors.append("MA period must be a positive integer")
        # Валидация stop loss
        if "stop_loss" in params:
            stop_loss = params["stop_loss"]
            if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                errors.append("Stop loss must be a positive number")
        # Валидация take profit
        if "take_profit" in params:
            take_profit = params["take_profit"]
            if not isinstance(take_profit, (int, float)) or take_profit <= 0:
                errors.append("Take profit must be a positive number")
        return errors

    @staticmethod
    def validate_mean_reversion_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры стратегии возврата к среднему."""
        errors = []
        required_params = ["bollinger_period", "std_dev", "entry_threshold"]
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        # Валидация периода Bollinger
        if "bollinger_period" in params:
            period = params["bollinger_period"]
            if not isinstance(period, int) or period <= 0:
                errors.append("Bollinger period must be a positive integer")
        # Валидация стандартного отклонения
        if "std_dev" in params:
            std_dev = params["std_dev"]
            if not isinstance(std_dev, (int, float)) or std_dev <= 0:
                errors.append("Standard deviation must be a positive number")
        return errors

    @staticmethod
    def validate_breakout_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры стратегии пробоя."""
        errors = []
        required_params = ["breakout_period", "volume_threshold", "confirmation_period"]
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        # Валидация периода пробоя
        if "breakout_period" in params:
            period = params["breakout_period"]
            if not isinstance(period, int) or period <= 0:
                errors.append("Breakout period must be a positive integer")
        # Валидация порога объема
        if "volume_threshold" in params:
            threshold = params["volume_threshold"]
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                errors.append("Volume threshold must be a positive number")
        return errors

    @staticmethod
    def validate_scalping_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры скальпинг стратегии."""
        errors = []
        required_params = ["timeframe", "profit_target", "max_holding_time"]
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        # Валидация таймфрейма
        if "timeframe" in params:
            timeframe = params["timeframe"]
            if not isinstance(timeframe, str) or timeframe not in ["1m", "5m", "15m"]:
                errors.append("Timeframe must be one of: 1m, 5m, 15m")
        # Валидация целевой прибыли
        if "profit_target" in params:
            target = params["profit_target"]
            if not isinstance(target, (int, float)) or target <= 0:
                errors.append("Profit target must be a positive number")
        return errors

    @staticmethod
    def validate_arbitrage_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры арбитражной стратегии."""
        errors = []
        required_params = ["min_spread", "max_execution_time", "exchange_pairs"]
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        # Валидация минимального спреда
        if "min_spread" in params:
            spread = params["min_spread"]
            if not isinstance(spread, (int, float)) or spread <= 0:
                errors.append("Minimum spread must be a positive number")
        # Валидация времени исполнения
        if "max_execution_time" in params:
            time = params["max_execution_time"]
            if not isinstance(time, (int, float)) or time <= 0:
                errors.append("Maximum execution time must be a positive number")
        return errors


class PerformanceValidator:
    """Валидатор производительности стратегий."""

    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, Any]) -> List[str]:
        """Валидировать метрики производительности."""
        errors = []
        required_metrics = ["total_return", "sharpe_ratio", "max_drawdown"]
        for metric in required_metrics:
            if metric not in metrics:
                errors.append(f"Missing required metric: {metric}")
        # Валидация общей доходности
        if "total_return" in metrics:
            total_return = metrics["total_return"]
            if not isinstance(total_return, (int, float)):
                errors.append("Total return must be a number")
        # Валидация коэффициента Шарпа
        if "sharpe_ratio" in metrics:
            sharpe = metrics["sharpe_ratio"]
            if not isinstance(sharpe, (int, float)):
                errors.append("Sharpe ratio must be a number")
        # Валидация максимальной просадки
        if "max_drawdown" in metrics:
            drawdown = metrics["max_drawdown"]
            if not isinstance(drawdown, (int, float)) or drawdown < 0:
                errors.append("Maximum drawdown must be a non-negative number")
        # Валидация количества сделок
        if "total_trades" in metrics:
            trades = metrics["total_trades"]
            if not isinstance(trades, int) or trades < 0:
                errors.append("Total trades must be a non-negative integer")
        # Валидация процента прибыльных сделок
        if "win_rate" in metrics:
            win_rate = metrics["win_rate"]
            if not isinstance(win_rate, (int, float)) or win_rate < 0 or win_rate > 1:
                errors.append("Win rate must be between 0 and 1")
        return errors

    @staticmethod
    def validate_risk_metrics(metrics: Dict[str, Any]) -> List[str]:
        """Валидировать метрики риска."""
        errors = []
        required_metrics = ["var_95", "cvar_95", "volatility"]
        for metric in required_metrics:
            if metric not in metrics:
                errors.append(f"Missing required metric: {metric}")
        # Валидация VaR
        if "var_95" in metrics:
            var = metrics["var_95"]
            if not isinstance(var, (int, float)):
                errors.append("VaR must be a number")
        # Валидация CVaR
        if "cvar_95" in metrics:
            cvar = metrics["cvar_95"]
            if not isinstance(cvar, (int, float)):
                errors.append("CVaR must be a number")
        # Валидация волатильности
        if "volatility" in metrics:
            vol = metrics["volatility"]
            if not isinstance(vol, (int, float)) or vol < 0:
                errors.append("Volatility must be a non-negative number")
        # Валидация беты
        if "beta" in metrics:
            beta = metrics["beta"]
            if not isinstance(beta, (int, float)):
                errors.append("Beta must be a number")
        # Валидация корреляции
        if "correlation" in metrics:
            corr = metrics["correlation"]
            if not isinstance(corr, (int, float)) or corr < -1 or corr > 1:
                errors.append("Correlation must be between -1 and 1")
        return errors


def get_strategy_validator() -> StrategyValidator:
    """Получить экземпляр валидатора стратегий."""
    return StrategyValidator() 