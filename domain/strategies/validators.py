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
        # Проверка объема
        if market_data.volume.value < 0:
            errors.append("Volume cannot be negative")
        # Проверка временной метки
        if market_data.timestamp > datetime.now() + timedelta(hours=1):
            errors.append("Market data timestamp cannot be in the future")
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
        if not signal.strategy_id:
            errors.append("Signal must have a strategy ID")
        if not signal.trading_pair:
            errors.append("Signal must have a trading pair")
        # Валидация уверенности
        if signal.confidence < 0 or signal.confidence > 1:
            errors.append("Confidence must be between 0 and 1")
        # Валидация цены
        if signal.price and signal.price.value <= 0:
            errors.append("Signal price must be positive")
        # Валидация количества
        if signal.quantity and signal.quantity <= 0:
            errors.append("Signal quantity must be positive")
        # Валидация stop loss
        if signal.stop_loss and signal.stop_loss.value <= 0:
            errors.append("Stop loss must be positive")
        # Валидация take profit
        if signal.take_profit and signal.take_profit.value <= 0:
            errors.append("Take profit must be positive")
        # Валидация временной метки
        if signal.timestamp > datetime.now() + timedelta(minutes=5):
            errors.append("Signal timestamp cannot be in the future")
        return errors

    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> List[str]:
        """
        Валидировать параметры стратегии.
        Args:
            parameters: Параметры стратегии
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка stop_loss
        if "stop_loss" in parameters:
            stop_loss = parameters["stop_loss"]
            if not isinstance(stop_loss, (int, float, Decimal)):
                errors.append("Stop loss must be a number")
            elif stop_loss <= 0:
                errors.append("Stop loss must be positive")
            elif stop_loss > 1:
                errors.append("Stop loss cannot exceed 100%")
        # Проверка take_profit
        if "take_profit" in parameters:
            take_profit = parameters["take_profit"]
            if not isinstance(take_profit, (int, float, Decimal)):
                errors.append("Take profit must be a number")
            elif take_profit <= 0:
                errors.append("Take profit must be positive")
            elif take_profit > 10:
                errors.append("Take profit cannot exceed 1000%")
        # Проверка position_size
        if "position_size" in parameters:
            position_size = parameters["position_size"]
            if not isinstance(position_size, (int, float, Decimal)):
                errors.append("Position size must be a number")
            elif position_size <= 0:
                errors.append("Position size must be positive")
            elif position_size > 1:
                errors.append("Position size cannot exceed 100%")
        # Проверка confidence_threshold
        if "confidence_threshold" in parameters:
            confidence_threshold = parameters["confidence_threshold"]
            if not isinstance(confidence_threshold, (int, float, Decimal)):
                errors.append("Confidence threshold must be a number")
            elif confidence_threshold < 0 or confidence_threshold > 1:
                errors.append("Confidence threshold must be between 0 and 1")
        # Проверка max_signals
        if "max_signals" in parameters:
            max_signals = parameters["max_signals"]
            if not isinstance(max_signals, int):
                errors.append("Max signals must be an integer")
            elif max_signals <= 0:
                errors.append("Max signals must be positive")
            elif max_signals > 100:
                errors.append("Max signals cannot exceed 100")
        # Проверка signal_cooldown
        if "signal_cooldown" in parameters:
            signal_cooldown = parameters["signal_cooldown"]
            if not isinstance(signal_cooldown, int):
                errors.append("Signal cooldown must be an integer")
            elif signal_cooldown < 0:
                errors.append("Signal cooldown cannot be negative")
            elif signal_cooldown > 86400:  # 24 часа
                errors.append("Signal cooldown cannot exceed 24 hours")
        return errors

    @staticmethod
    def validate_trading_pair(trading_pair: str) -> List[str]:
        """
        Валидировать торговую пару.
        Args:
            trading_pair: Торговая пара
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        if not trading_pair:
            errors.append("Trading pair cannot be empty")
            return errors
        # Проверка формата
        if not re.match(r"^[A-Z0-9]+/[A-Z0-9]+$", trading_pair):
            errors.append("Trading pair must be in format BASE/QUOTE (e.g., BTC/USDT)")
        # Проверка длины
        if len(trading_pair) > 20:
            errors.append("Trading pair is too long")
        # Проверка на специальные символы
        if any(char in trading_pair for char in ["<", ">", "&", '"', "'"]):
            errors.append("Trading pair contains invalid characters")
        return errors

    @staticmethod
    def validate_risk_level(risk_level: Union[str, Decimal]) -> List[str]:
        errors = []
        if isinstance(risk_level, str):
            try:
                # Преобразуем строку в Decimal для RiskLevel
                rl_value = Decimal(risk_level)
                if rl_value < 0 or rl_value > 1:
                    errors.append("Risk level must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append(f"Invalid risk level: {risk_level}")
        elif isinstance(risk_level, Decimal):
            # RiskLevel это NewType(Decimal), поэтому проверяем как Decimal
            if risk_level < 0 or risk_level > 1:
                errors.append("Risk level must be between 0 and 1")
        else:
            errors.append("Risk level must be a string or Decimal")
        return errors

    @staticmethod
    def validate_time_horizon(time_horizon: Union[str, Enum]) -> List[str]:
        errors = []
        if isinstance(time_horizon, str):
            # Временное решение - проверяем только строку
            valid_horizons = ["short", "medium", "long"]
            if time_horizon not in valid_horizons:
                errors.append(
                    f"Invalid time horizon: {time_horizon}. Valid horizons: {valid_horizons}"
                )
        elif isinstance(time_horizon, Enum):
            pass  # Енумы всегда валидны
        else:
            errors.append("Time horizon must be a string or Enum")
        return errors

    @staticmethod
    def validate_market_condition(condition: Union[str, Enum]) -> List[str]:
        errors = []
        if isinstance(condition, str):
            # Временное решение - проверяем только строку
            valid_conditions = ["bull", "bear", "sideways", "volatile"]
            if condition not in valid_conditions:
                errors.append(
                    f"Invalid market condition: {condition}. Valid conditions: {valid_conditions}"
                )
        elif isinstance(condition, Enum):
            pass  # Енумы всегда валидны
        else:
            errors.append("Market condition must be a string or Enum")
        return errors

    @staticmethod
    def _validate_strategy_name(name: str) -> List[str]:
        """Валидировать имя стратегии."""
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
        """Валидировать тип стратегии."""
        errors = []
        valid_types = [st.value for st in StrategyType]
        if strategy_type not in valid_types:
            errors.append(
                f"Invalid strategy type: {strategy_type}. Valid types: {valid_types}"
            )
        return errors

    @staticmethod
    def _validate_trading_pairs(trading_pairs: List[str]) -> List[str]:
        """Валидировать список торговых пар."""
        errors = []
        if not isinstance(trading_pairs, list):
            errors.append("Trading pairs must be a list")
            return errors
        elif not trading_pairs:
            errors.append("At least one trading pair is required")
            return errors  # Енумы всегда валидны
        else:
            if len(trading_pairs) > 50:
                errors.append("Too many trading pairs (max 50)")
            for pair in trading_pairs:
                pair_errors = StrategyValidator.validate_trading_pair(pair)
                errors.extend(pair_errors)
            # Проверка на дубликаты
            if len(trading_pairs) != len(set(trading_pairs)):
                errors.append("Duplicate trading pairs are not allowed")
            return errors

    @staticmethod
    def _validate_metadata(metadata: Dict[str, Any]) -> List[str]:
        """Валидировать метаданные."""
        errors = []
        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
            return errors
        elif len(metadata) > 100:
            errors.append("Too many metadata entries (max 100)")
            return errors  # Енумы всегда валидны
        else:
            for key, value in metadata.items():
                if not isinstance(key, str):
                    errors.append("Metadata keys must be strings")
                elif len(key) > 50:
                    errors.append("Metadata key is too long (max 50 characters)")
                elif not isinstance(value, (str, int, float, bool, list, dict)):
                    errors.append("Metadata values must be basic types")
            return errors  # Енумы всегда валидны


class ParameterValidator:
    """Валидатор параметров стратегий."""

    @staticmethod
    def validate_trend_following_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры трендовой стратегии."""
        errors = []
        # Проверка периодов
        if "short_period" in params:
            short_period = params["short_period"]
            if not isinstance(short_period, int) or short_period <= 0:
                errors.append("Short period must be a positive integer")
            elif short_period > 100:
                errors.append("Short period cannot exceed 100")
        if "long_period" in params:
            long_period = params["long_period"]
            if not isinstance(long_period, int) or long_period <= 0:
                errors.append("Long period must be a positive integer")
            elif long_period > 200:
                errors.append("Long period cannot exceed 200")
        # Проверка порога силы тренда
        if "trend_strength_threshold" in params:
            threshold = params["trend_strength_threshold"]
            if not isinstance(threshold, (int, float, Decimal)):
                errors.append("Trend strength threshold must be a number")
            elif threshold < 0 or threshold > 1:
                errors.append("Trend strength threshold must be between 0 and 1")
        return errors

    @staticmethod
    def validate_mean_reversion_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры стратегии возврата к среднему."""
        errors = []
        # Проверка периода анализа
        if "lookback_period" in params:
            lookback_period = params["lookback_period"]
            if not isinstance(lookback_period, int) or lookback_period <= 0:
                errors.append("Lookback period must be a positive integer")
            elif lookback_period > 1000:
                errors.append("Lookback period cannot exceed 1000")
        # Проверка порога отклонения
        if "deviation_threshold" in params:
            threshold = params["deviation_threshold"]
            if not isinstance(threshold, (int, float, Decimal)):
                errors.append("Deviation threshold must be a number")
            elif threshold <= 0:
                errors.append("Deviation threshold must be positive")
            elif threshold > 10:
                errors.append("Deviation threshold cannot exceed 10")
        return errors

    @staticmethod
    def validate_breakout_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры стратегии пробоя."""
        errors = []
        # Проверка порога пробоя
        if "breakout_threshold" in params:
            threshold = params["breakout_threshold"]
            if not isinstance(threshold, (int, float, Decimal)):
                errors.append("Breakout threshold must be a number")
            elif threshold <= 0:
                errors.append("Breakout threshold must be positive")
            elif threshold > 5:
                errors.append("Breakout threshold cannot exceed 5")
        # Проверка множителя объема
        if "volume_multiplier" in params:
            multiplier = params["volume_multiplier"]
            if not isinstance(multiplier, (int, float, Decimal)):
                errors.append("Volume multiplier must be a number")
            elif multiplier <= 0:
                errors.append("Volume multiplier must be positive")
            elif multiplier > 10:
                errors.append("Volume multiplier cannot exceed 10")
        return errors

    @staticmethod
    def validate_scalping_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры скальпинг стратегии."""
        errors = []
        # Проверка порога скальпинга
        if "scalping_threshold" in params:
            threshold = params["scalping_threshold"]
            if not isinstance(threshold, (int, float, Decimal)):
                errors.append("Scalping threshold must be a number")
            elif threshold <= 0:
                errors.append("Scalping threshold must be positive")
            elif threshold > 1:
                errors.append("Scalping threshold cannot exceed 1")
        # Проверка времени удержания
        if "max_hold_time" in params:
            hold_time = params["max_hold_time"]
            if not isinstance(hold_time, int) or hold_time <= 0:
                errors.append("Max hold time must be a positive integer")
            elif hold_time > 3600:  # 1 час
                errors.append("Max hold time cannot exceed 1 hour")
        return errors

    @staticmethod
    def validate_arbitrage_params(params: Dict[str, Any]) -> List[str]:
        """Валидировать параметры арбитражной стратегии."""
        errors = []
        # Проверка порога арбитража
        if "arbitrage_threshold" in params:
            threshold = params["arbitrage_threshold"]
            if not isinstance(threshold, (int, float, Decimal)):
                errors.append("Arbitrage threshold must be a number")
            elif threshold <= 0:
                errors.append("Arbitrage threshold must be positive")
            elif threshold > 5:
                errors.append("Arbitrage threshold cannot exceed 5")
        # Проверка максимального проскальзывания
        if "max_slippage" in params:
            slippage = params["max_slippage"]
            if not isinstance(slippage, (int, float, Decimal)):
                errors.append("Max slippage must be a number")
            elif slippage <= 0:
                errors.append("Max slippage must be positive")
            elif slippage > 1:
                errors.append("Max slippage cannot exceed 100%")
        return errors


class PerformanceValidator:
    """Валидатор производительности стратегий."""

    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, Any]) -> List[str]:
        """
        Валидировать метрики производительности.
        Args:
            metrics: Метрики производительности
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка общего количества сделок
        if "total_trades" in metrics:
            total_trades = metrics["total_trades"]
            if not isinstance(total_trades, int) or total_trades < 0:
                errors.append("Total trades must be a non-negative integer")
        # Проверка выигрышных сделок
        if "winning_trades" in metrics:
            winning_trades = metrics["winning_trades"]
            if not isinstance(winning_trades, int) or winning_trades < 0:
                errors.append("Winning trades must be a non-negative integer")
        # Проверка проигрышных сделок
        if "losing_trades" in metrics:
            losing_trades = metrics["losing_trades"]
            if not isinstance(losing_trades, int) or losing_trades < 0:
                errors.append("Losing trades must be a non-negative integer")
        # Проверка процента выигрышных сделок
        if "win_rate" in metrics:
            win_rate = metrics["win_rate"]
            if not isinstance(win_rate, (int, float, Decimal)):
                errors.append("Win rate must be a number")
            elif win_rate < 0 or win_rate > 100:
                errors.append("Win rate must be between 0 and 100")
        # Проверка коэффициента прибыли
        if "profit_factor" in metrics:
            profit_factor = metrics["profit_factor"]
            if not isinstance(profit_factor, (int, float, Decimal)):
                errors.append("Profit factor must be a number")
            elif profit_factor < 0:
                errors.append("Profit factor cannot be negative")
        # Проверка коэффициента Шарпа
        if "sharpe_ratio" in metrics:
            sharpe_ratio = metrics["sharpe_ratio"]
            if not isinstance(sharpe_ratio, (int, float, Decimal)):
                errors.append("Sharpe ratio must be a number")
        # Проверка максимальной просадки
        if "max_drawdown" in metrics:
            max_drawdown = metrics["max_drawdown"]
            if not isinstance(max_drawdown, (int, float, Decimal)):
                errors.append("Max drawdown must be a number")
            elif max_drawdown < 0:
                errors.append("Max drawdown cannot be negative")
            elif max_drawdown > 100:
                errors.append("Max drawdown cannot exceed 100%")
        return errors

    @staticmethod
    def validate_risk_metrics(metrics: Dict[str, Any]) -> List[str]:
        """
        Валидировать метрики риска.
        Args:
            metrics: Метрики риска
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        # Проверка волатильности
        if "volatility" in metrics:
            volatility = metrics["volatility"]
            if not isinstance(volatility, (int, float, Decimal)):
                errors.append("Volatility must be a number")
            elif volatility < 0:
                errors.append("Volatility cannot be negative")
        # Проверка VaR
        if "var_95" in metrics:
            var_95 = metrics["var_95"]
            if not isinstance(var_95, (int, float, Decimal)):
                errors.append("VaR 95% must be a number")
            elif var_95 < 0:
                errors.append("VaR 95% cannot be negative")
            elif var_95 > 100:
                errors.append("VaR 95% cannot exceed 100%")
        # Проверка CVaR
        if "cvar_95" in metrics:
            cvar_95 = metrics["cvar_95"]
            if not isinstance(cvar_95, (int, float, Decimal)):
                errors.append("CVaR 95% must be a number")
            elif cvar_95 < 0:
                errors.append("CVaR 95% cannot be negative")
            elif cvar_95 > 100:
                errors.append("CVaR 95% cannot exceed 100%")
        # Проверка беты
        if "beta" in metrics:
            beta = metrics["beta"]
            if not isinstance(beta, (int, float, Decimal)):
                errors.append("Beta must be a number")
        # Проверка корреляции
        if "correlation" in metrics:
            correlation = metrics["correlation"]
            if not isinstance(correlation, (int, float, Decimal)):
                errors.append("Correlation must be a number")
            elif correlation < -1 or correlation > 1:
                errors.append("Correlation must be between -1 and 1")
        return errors


# Глобальный экземпляр валидатора
_strategy_validator_instance: Optional[StrategyValidator] = None


def get_strategy_validator() -> StrategyValidator:
    """
    Получить глобальный экземпляр валидатора стратегий.
    Returns:
        StrategyValidator: Экземпляр валидатора
    """
    global _strategy_validator_instance
    if _strategy_validator_instance is None:
        _strategy_validator_instance = StrategyValidator()
    return _strategy_validator_instance
