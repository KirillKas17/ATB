"""
Доменные сущности стратегий - основной файл.
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Union, List, Optional, Protocol, cast
from uuid import UUID, uuid4

# Импорты из декомпозированных модулей
from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy_parameters import (
    ParameterDict,
    ParameterValue,
    StrategyParameters,
)
from domain.entities.strategy_performance import StrategyPerformance
from domain.exceptions import StrategyExecutionError

logger = logging.getLogger(__name__)

# Строгие типы для типизации
RiskMetricsDict = Dict[str, Union[float, Decimal, str]]
PerformanceMetricsDict = Dict[str, Union[float, Decimal, str, int]]
MarketDataDict = Dict[str, Union[str, float, int, bool, List[float]]]

# Расширенный тип для параметров, которые могут быть разными типами
ExtendedParameterValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ExtendedParameterDict = Dict[str, ExtendedParameterValue]


class StrategyType(Enum):
    """Типы стратегий."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    MARTINGALE = "martingale"
    HEDGING = "hedging"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


class StrategyStatus(Enum):
    """Статусы стратегии."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    INACTIVE = "inactive"


@dataclass
class Strategy:
    """Стратегия - основной агрегат"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    strategy_type: StrategyType = StrategyType.TREND_FOLLOWING
    status: StrategyStatus = StrategyStatus.ACTIVE
    trading_pairs: List[str] = field(default_factory=list)
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    signals: List[Signal] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, ExtendedParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameters.id == uuid4():
            self.parameters.strategy_id = self.id
        if self.performance.id == uuid4():
            self.performance.strategy_id = self.id

    def add_signal(self, signal: Signal) -> None:
        """Добавить сигнал"""
        signal.strategy_id = self.id
        self.signals.append(signal)
        self.updated_at = datetime.now()

    def get_latest_signal(self, trading_pair: str = "") -> Optional[Signal]:
        """Получить последний сигнал"""
        if not self.signals:
            return None
        if trading_pair:
            filtered_signals = [
                s for s in self.signals if s.trading_pair == trading_pair
            ]
            if not filtered_signals:
                return None
            return max(filtered_signals, key=lambda x: x.timestamp)
        return max(self.signals, key=lambda x: x.timestamp)

    def get_signals_by_type(self, signal_type: SignalType) -> List[Signal]:
        """Получить сигналы по типу"""
        return [s for s in self.signals if s.signal_type == signal_type]

    def update_status(self, status: StrategyStatus) -> None:
        """Обновить статус"""
        self.status = status
        self.updated_at = datetime.now()

    def add_trading_pair(self, trading_pair: str) -> None:
        """Добавить торговую пару"""
        if trading_pair not in self.trading_pairs:
            self.trading_pairs.append(trading_pair)
            self.updated_at = datetime.now()

    def remove_trading_pair(self, trading_pair: str) -> None:
        """Удалить торговую пару"""
        if trading_pair in self.trading_pairs:
            self.trading_pairs.remove(trading_pair)
            self.updated_at = datetime.now()

    def update_parameter(self, key: str, value: ParameterValue) -> None:
        """Обновить параметр"""
        self.parameters.set_parameter(key, value)
        self.updated_at = datetime.now()

    def get_parameter(
        self, key: str, default: Optional[ParameterValue] = None
    ) -> Optional[ParameterValue]:
        """Получить параметр"""
        return self.parameters.get_parameter(key, default)

    def calculate_signal(self, market_data: MarketDataDict) -> Optional[Signal]:
        """Рассчитать сигнал на основе рыночных данных"""
        # Базовая реализация - должна быть переопределена в конкретных стратегиях
        return None

    def should_execute_signal(self, signal: Signal) -> bool:
        """Проверить, следует ли исполнять сигнал"""
        if not self.is_active or self.status != StrategyStatus.ACTIVE:
            return False
        if not signal.is_actionable:
            return False
        # Проверить, что торговая пара поддерживается
        if signal.trading_pair and signal.trading_pair not in self.trading_pairs:
            return False
        return True

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            str,
            int,
            float,
            Decimal,
            bool,
            List[str],
            Dict[str, Union[str, int, float, Decimal, bool]],
            None,
        ],
    ]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "trading_pairs": self.trading_pairs,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parameters": cast(
                Dict[str, Union[str, int, float, Decimal, bool]],
                self.parameters.parameters,
            ),
            "performance": cast(
                Dict[str, Union[str, int, float, Decimal, bool]],
                self.performance.to_dict(),
            ),
            "metadata": cast(
                Dict[str, Union[str, int, float, Decimal, bool]], self.metadata
            ),
        }

    def generate_signals(
        self,
        symbol: str,
        amount: Optional[Decimal] = None,
        risk_level: Optional[str] = None,
    ) -> List[Signal]:
        """
        Генерация торговых сигналов для указанного символа.
        Args:
            symbol: Торговая пара (например, "BTC/USD")
            amount: Размер позиции (опционально)
            risk_level: Уровень риска (опционально)
        Returns:
            List[Signal]: Список сгенерированных сигналов
        Raises:
            ValueError: Если символ не поддерживается стратегией
            StrategyExecutionError: Если стратегия неактивна
        """
        # Проверяем, что стратегия активна
        if not self.is_active or self.status != StrategyStatus.ACTIVE:
            raise StrategyExecutionError(f"Strategy {self.id} is not active")
        # Проверяем, что торговая пара поддерживается
        if symbol not in self.trading_pairs:
            raise ValueError(f"Symbol {symbol} is not supported by strategy {self.id}")
        # Получаем параметры стратегии с безопасными преобразованиями
        confidence_threshold = self._safe_get_parameter(
            "confidence_threshold", Decimal("0.6")
        )
        max_signals = self._safe_get_parameter("max_signals", 5)
        signal_cooldown = self._safe_get_parameter("signal_cooldown", 300)  # 5 минут
        # Проверяем cooldown для последних сигналов
        current_time = datetime.now()
        recent_signals = [
            signal
            for signal in self.signals
            if signal.trading_pair == symbol
            and (current_time - signal.timestamp).total_seconds() < signal_cooldown
        ]
        if len(recent_signals) >= max_signals:
            return []  # Слишком много недавних сигналов
        # Генерируем базовый сигнал на основе типа стратегии
        signals = []
        if self.strategy_type == StrategyType.TREND_FOLLOWING:
            signals.extend(
                self._generate_trend_following_signals(symbol, amount, risk_level)
            )
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            signals.extend(
                self._generate_mean_reversion_signals(symbol, amount, risk_level)
            )
        elif self.strategy_type == StrategyType.BREAKOUT:
            signals.extend(self._generate_breakout_signals(symbol, amount, risk_level))
        elif self.strategy_type == StrategyType.SCALPING:
            signals.extend(self._generate_scalping_signals(symbol, amount, risk_level))
        elif self.strategy_type == StrategyType.ARBITRAGE:
            signals.extend(self._generate_arbitrage_signals(symbol, amount, risk_level))
        elif self.strategy_type == StrategyType.GRID:
            signals.extend(self._generate_grid_signals(symbol, amount, risk_level))
        elif self.strategy_type == StrategyType.MOMENTUM:
            signals.extend(self._generate_momentum_signals(symbol, amount, risk_level))
        elif self.strategy_type == StrategyType.VOLATILITY:
            signals.extend(
                self._generate_volatility_signals(symbol, amount, risk_level)
            )
        else:
            # Базовая реализация для неизвестных типов стратегий
            signals.extend(self._generate_default_signals(symbol, amount, risk_level))
        # Фильтруем сигналы по уровню уверенности
        filtered_signals = [
            signal for signal in signals if signal.confidence >= confidence_threshold
        ]
        # Ограничиваем количество сигналов
        if len(filtered_signals) > max_signals:
            filtered_signals = filtered_signals[: int(max_signals)]
        # Добавляем сигналы к стратегии
        for signal in filtered_signals:
            self.add_signal(signal)
        return filtered_signals

    def _safe_get_parameter(
        self, key: str, default: Union[Decimal, int]
    ) -> Union[Decimal, int]:
        """Безопасное получение параметра с преобразованием типов"""
        value = self.get_parameter(key, default)
        if isinstance(value, (int, float, str, Decimal)):
            if isinstance(default, Decimal):
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return default
            elif isinstance(default, int):
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return default
        return default

    def _generate_trend_following_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для трендовой стратегии."""
        trend_strength = self._safe_get_parameter("trend_strength", Decimal("0.7"))
        trend_period = self._safe_get_parameter("trend_period", 20)
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=(
                SignalStrength.STRONG
                if trend_strength > Decimal("0.8")
                else SignalStrength.MEDIUM
            ),
            confidence=Decimal(str(trend_strength)),
            quantity=amount,
            metadata={
                "strategy_type": "trend_following",
                "trend_period": trend_period,
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_mean_reversion_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для стратегии возврата к среднему."""
        mean_reversion_threshold = self._safe_get_parameter(
            "mean_reversion_threshold", Decimal("2.0")
        )
        lookback_period = self._safe_get_parameter("lookback_period", 50)
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=Decimal("0.65"),
            quantity=amount,
            metadata={
                "strategy_type": "mean_reversion",
                "threshold": str(mean_reversion_threshold),
                "lookback_period": lookback_period,
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_breakout_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для стратегии пробоя."""
        breakout_threshold = self._safe_get_parameter(
            "breakout_threshold", Decimal("1.5")
        )
        volume_multiplier = self._safe_get_parameter(
            "volume_multiplier", Decimal("2.0")
        )
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=SignalStrength.VERY_STRONG,
            confidence=Decimal("0.8"),
            quantity=amount,
            metadata={
                "strategy_type": "breakout",
                "threshold": str(breakout_threshold),
                "volume_multiplier": str(volume_multiplier),
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_scalping_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для скальпинга."""
        scalping_threshold = self._safe_get_parameter(
            "scalping_threshold", Decimal("0.1")
        )
        max_hold_time = self._safe_get_parameter("max_hold_time", 300)
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=Decimal("0.55"),
            quantity=amount,
            metadata={
                "strategy_type": "scalping",
                "threshold": str(scalping_threshold),
                "max_hold_time": max_hold_time,
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_arbitrage_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для арбитража."""
        arbitrage_threshold = self._safe_get_parameter(
            "arbitrage_threshold", Decimal("0.5")
        )
        max_slippage = self._safe_get_parameter("max_slippage", Decimal("0.1"))
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.9"),
            quantity=amount,
            metadata={
                "strategy_type": "arbitrage",
                "threshold": str(arbitrage_threshold),
                "max_slippage": str(max_slippage),
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_grid_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для сеточной стратегии."""
        grid_levels = self._safe_get_parameter("grid_levels", 10)
        if isinstance(grid_levels, Decimal):
            grid_levels = int(grid_levels)
        grid_spacing = self._safe_get_parameter("grid_spacing", Decimal("0.02"))
        signals = []
        for level in range(int(grid_levels)):
            signal = Signal(
                strategy_id=self.id,
                trading_pair=symbol,
                signal_type=SignalType.BUY if level % 2 == 0 else SignalType.SELL,
                strength=SignalStrength.MEDIUM,
                confidence=Decimal("0.6"),
                quantity=(
                    Decimal(str(float(amount) / int(grid_levels)))
                    if amount is not None
                    else None
                ),
                metadata={
                    "strategy_type": "grid",
                    "level": level,
                    "spacing": str(grid_spacing),
                    "risk_level": str(risk_level) if risk_level is not None else "",
                },
            )
            signals.append(signal)
        return signals

    def _generate_momentum_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для стратегии импульса."""
        momentum_period = self._safe_get_parameter("momentum_period", 14)
        momentum_threshold = self._safe_get_parameter(
            "momentum_threshold", Decimal("0.5")
        )
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.75"),
            quantity=amount,
            metadata={
                "strategy_type": "momentum",
                "period": momentum_period,
                "threshold": str(momentum_threshold),
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_volatility_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов для волатильной стратегии."""
        volatility_period = self._safe_get_parameter("volatility_period", 20)
        volatility_threshold = self._safe_get_parameter(
            "volatility_threshold", Decimal("0.03")
        )
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=Decimal("0.7"),
            quantity=amount,
            metadata={
                "strategy_type": "volatility",
                "period": volatility_period,
                "threshold": str(volatility_threshold),
                "risk_level": str(risk_level) if risk_level is not None else "",
            },
        )
        return [signal]

    def _generate_default_signals(
        self, symbol: str, amount: Optional[Decimal], risk_level: Optional[str]
    ) -> List[Signal]:
        """Генерация сигналов по умолчанию для неизвестных типов стратегий."""
        signal = Signal(
            strategy_id=self.id,
            trading_pair=symbol,
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=Decimal("0.5"),
            quantity=amount,
            metadata={
                "strategy_type": "default",
                "risk_level": str(risk_level) if risk_level is not None else "",
                "note": "Default signal for unknown strategy type",
            },
        )
        return [signal]

    def validate_config(self, config: Dict[str, ExtendedParameterValue]) -> List[str]:
        """
        Валидация конфигурации стратегии.
        Args:
            config: Конфигурация для валидации
        Returns:
            List[str]: Список ошибок валидации (пустой список если валидация прошла успешно)
        """
        errors = []
        # Проверка обязательных полей
        if not config.get("name"):
            errors.append("Strategy name is required")
        if not config.get("trading_pairs"):
            errors.append("At least one trading pair is required")
        # Проверка параметров стратегии
        parameters = config.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        # Проверка stop_loss
        stop_loss = parameters.get("stop_loss", 0)
        if isinstance(stop_loss, (int, float, str, Decimal)):
            stop_loss_val = float(stop_loss)
            if stop_loss_val <= 0:
                errors.append("Stop loss must be positive")
            elif stop_loss_val > 1:
                errors.append("Stop loss cannot exceed 100%")
        # Проверка take_profit
        take_profit = parameters.get("take_profit", 0)
        if isinstance(take_profit, (int, float, str, Decimal)):
            take_profit_val = float(take_profit)
            if take_profit_val <= 0:
                errors.append("Take profit must be positive")
            elif take_profit_val > 10:
                errors.append("Take profit cannot exceed 1000%")
        # Проверка position_size
        position_size = parameters.get("position_size", 0)
        if isinstance(position_size, (int, float, str, Decimal)):
            position_size_val = float(position_size)
            if position_size_val <= 0:
                errors.append("Position size must be positive")
            elif position_size_val > 1:
                errors.append("Position size cannot exceed 100%")
        # Проверка confidence_threshold
        confidence_threshold = parameters.get("confidence_threshold", 0)
        if isinstance(confidence_threshold, (int, float, str, Decimal)):
            conf_val = float(confidence_threshold)
            if conf_val < 0 or conf_val > 1:
                errors.append("Confidence threshold must be between 0 and 1")
        # Проверка max_signals
        max_signals = parameters.get("max_signals", 0)
        if isinstance(max_signals, (int, float, str, Decimal)):
            max_sig_val = int(float(max_signals))
            if max_sig_val <= 0:
                errors.append("Max signals must be positive")
            elif max_sig_val > 100:
                errors.append("Max signals cannot exceed 100")
        # Проверка signal_cooldown
        signal_cooldown = parameters.get("signal_cooldown", 0)
        if isinstance(signal_cooldown, (int, float, str, Decimal)):
            cooldown_val = float(signal_cooldown)
            if cooldown_val < 0:
                errors.append("Signal cooldown cannot be negative")
            elif cooldown_val > 86400:  # 24 часа
                errors.append("Signal cooldown cannot exceed 24 hours")
        return errors

    async def calculate_risk_metrics(self, data: pd.DataFrame) -> RiskMetricsDict:
        """Расчет метрик риска с использованием централизованного сервиса."""
        try:
            from domain.services.risk_analysis import DefaultRiskAnalysisService

            # Создаем сервис анализа рисков
            risk_service = DefaultRiskAnalysisService()
            # Рассчитываем доходности
            if "close" not in data.columns:
                raise ValueError("Data must contain 'close' column")
            returns = data["close"].pct_change().dropna()
            if len(returns) < 2:
                return {
                    "volatility": 0.0,
                    "var_95": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                }
            # Используем централизованный сервис
            risk_metrics = await risk_service.calculate_portfolio_risk(returns)
            return {
                "volatility": float(risk_metrics.volatility),
                "var_95": float(risk_metrics.daily_var),  # Using daily_var instead of var_95.value
                "max_drawdown": float(getattr(risk_metrics, 'max_drawdown', 0.0)),  # Safe access
                "sharpe_ratio": float(getattr(risk_metrics, 'sharpe_ratio', 0.0)),  # Safe access
                "sortino_ratio": float(getattr(risk_metrics, 'sortino_ratio', 0.0)),  # Safe access
                "cvar_95": float(risk_metrics.expected_shortfall),  # Using expected_shortfall instead of var_95.amount
                "avg_correlation": float(risk_metrics.correlation_risk),  # Using correlation_risk
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "cvar_95": 0.0,
                "avg_correlation": 0.0,
            }

    def is_ready_for_trading(self) -> bool:
        """
        Проверка готовности стратегии к торговле.
        Проверяет все необходимые условия для начала торговли:
        - Наличие исторических данных
        - Корректность параметров
        - Состояние модели (если используется ML)
        - Доступность рыночных данных
        Returns:
            bool: True если стратегия готова к торговле, False в противном случае
        """
        return (
            self.is_active
            and self.status == StrategyStatus.ACTIVE
            and len(self.trading_pairs) > 0
            and self.parameters.is_active
        )

    def reset(self) -> None:
        """
        Сброс состояния стратегии к начальным значениям.
        Очищает все внутренние состояния, историю сделок,
        метрики и возвращает стратегию к исходному состоянию.
        """
        self.signals.clear()
        self.performance = StrategyPerformance()
        self.updated_at = datetime.now()

    def save_state(self, filepath: str) -> bool:
        """
        Сохранение состояния стратегии в файл.
        Args:
            filepath: Путь к файлу для сохранения
        Returns:
            bool: True если состояние успешно сохранено, False в противном случае
        """
        try:
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(self.to_dict(), f)
            return True
        except Exception as e:
            logger.error(f"Error saving strategy state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Загрузка состояния стратегии из файла.
        Args:
            filepath: Путь к файлу с сохраненным состоянием
        Returns:
            bool: True если состояние успешно загружено, False в противном случае
        """
        try:
            import pickle

            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Восстанавливаем состояние из словаря
            if isinstance(data, dict):
                # Восстанавливаем имя
                if "name" in data:
                    self.name = data["name"]
                # Восстанавливаем параметры
                if "parameters" in data and isinstance(data["parameters"], dict):
                    for key, value in data["parameters"].items():
                        self.parameters.set_parameter(key, value)
                # Восстанавливаем торговые пары
                if "trading_pairs" in data and isinstance(data["trading_pairs"], list):
                    self.trading_pairs = data["trading_pairs"]
                # Восстанавливаем статус
                if "status" in data:
                    try:
                        self.status = StrategyStatus(data["status"])
                    except ValueError:
                        pass
                # Восстанавливаем метаданные
                if "metadata" in data and isinstance(data["metadata"], dict):
                    self.metadata = data["metadata"]
                
                self.updated_at = datetime.now()
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
            return False


class StrategyProtocol(Protocol):
    """Протокол для стратегий."""

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Генерация сигнала."""
        ...

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Валидация данных."""
        ...


class AbstractStrategy(ABC):
    """Абстрактная стратегия."""

    def __init__(
        self,
        id: UUID,
        name: str,
        strategy_type: StrategyType,
        config: Dict[str, ExtendedParameterValue],
        status: StrategyStatus = StrategyStatus.INACTIVE,
    ):
        """Инициализация абстрактной стратегии."""
        self.id = id
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
        self.status = status
        self.performance = StrategyPerformance()

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> MarketDataDict:
        """Анализ данных."""
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Генерация сигнала."""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Валидация данных."""
        pass

    def is_active(self) -> bool:
        """Проверка активности стратегии."""
        return self.status == StrategyStatus.ACTIVE

    def activate(self) -> None:
        """Активация стратегии."""
        self.status = StrategyStatus.ACTIVE

    def deactivate(self) -> None:
        """Деактивация стратегии."""
        self.status = StrategyStatus.INACTIVE

    def pause(self) -> None:
        """Приостановка стратегии."""
        self.status = StrategyStatus.PAUSED

    def update_metrics(self, metrics: StrategyPerformance) -> None:
        """Обновление метрик."""
        self.performance = metrics

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            str,
            int,
            float,
            Decimal,
            bool,
            List[str],
            Dict[str, Union[str, int, float, Decimal, bool]],
            None,
        ],
    ]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "config": cast(
                Dict[str, Union[str, int, float, Decimal, bool]], self.config
            ),
            "performance": cast(
                Dict[str, Union[str, int, float, Decimal, bool]],
                self.performance.to_dict(),
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, ExtendedParameterValue]) -> "AbstractStrategy":
        """Создание из словаря."""
        # Это абстрактный метод, реализация зависит от конкретной стратегии
        raise NotImplementedError("Subclasses must implement from_dict method")
