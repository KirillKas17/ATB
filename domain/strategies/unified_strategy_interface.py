"""
Единый интерфейс стратегий для объединения domain и infrastructure слоёв.
Этот модуль создаёт мост между доменной логикой и инфраструктурными реализациями,
обеспечивая единообразный интерфейс для всех стратегий в системе.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
from uuid import UUID, uuid4

from domain.entities.market import MarketData, MarketState
from domain.entities.signal import Signal
from domain.entities.strategy import StrategyStatus, StrategyType
from domain.entities.strategy_performance import StrategyPerformance
from domain.memory.pattern_memory import PatternMemory
from domain.types import (
    ConfidenceLevel,
    MetadataDict,
    PerformanceScore,
    RiskLevel,
    SignalId,
    StrategyConfig,
    StrategyId,
    TradingPair,
)


class StrategyExecutionMode(Enum):
    """Режимы выполнения стратегии."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


@dataclass
class StrategyContext:
    """Контекст выполнения стратегии."""

    mode: StrategyExecutionMode
    timestamp: datetime
    market_data: MarketData
    portfolio_state: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, Any]] = None
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    liquidity_regime: Optional[str] = None
    sentiment_data: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    pattern_memory: Optional[List[PatternMemory]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyAnalysisResult:
    """Результат анализа стратегии."""

    confidence_score: float
    trend_direction: str
    trend_strength: float
    volatility_level: float
    volume_analysis: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    market_regime: str
    risk_assessment: Dict[str, Any]
    support_resistance: Tuple[Optional[float], Optional[float]]
    momentum_indicators: Dict[str, float]
    pattern_recognition: List[str]
    market_sentiment: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyMetrics:
    """Метрики стратегии."""

    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    volatility: float = 0.0
    mar_ratio: float = 0.0
    ulcer_index: float = 0.0
    omega_ratio: float = 0.0
    gini_coefficient: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    drawdown_duration: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    median_trade: float = 0.0
    median_duration: float = 0.0
    profit_streak: int = 0
    loss_streak: int = 0
    stability: float = 0.0
    additional: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """Протокол для поставщика рыночных данных."""

    def get_market_data(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[MarketData]:
        """Получить рыночные данные."""
        ...

    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Получить состояние рынка."""
        ...


@runtime_checkable
class SignalGeneratorProtocol(Protocol):
    """Протокол для генератора сигналов."""

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Генерировать сигнал."""
        ...

    def validate_signal(self, signal: Signal) -> bool:
        """Валидировать сигнал."""
        ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Протокол для управления рисками."""

    def calculate_position_size(
        self, signal: Signal, account_balance: Decimal
    ) -> Decimal:
        """Рассчитать размер позиции."""
        ...

    def validate_risk_limits(self, signal: Signal) -> bool:
        """Проверить лимиты риска."""
        ...


class UnifiedStrategyInterface(ABC):
    """
    Единый интерфейс стратегии - абстрактный базовый класс для всех стратегий.
    Объединяет доменную логику и инфраструктурные реализации, обеспечивая
    единообразный интерфейс для всех стратегий в системе.
    """

    def __init__(
        self,
        strategy_id: StrategyId,
        name: str,
        strategy_type: StrategyType,
        trading_pairs: List[str],
        parameters: Dict[str, Any],
        risk_level: RiskLevel = RiskLevel(Decimal("0.5")),
        confidence_threshold: ConfidenceLevel = ConfidenceLevel(Decimal("0.6")),
        execution_mode: StrategyExecutionMode = StrategyExecutionMode.PAPER,
    ):
        """
        Инициализация единого интерфейса стратегии.
        Args:
            strategy_id: Уникальный идентификатор стратегии
            name: Название стратегии
            strategy_type: Тип стратегии
            trading_pairs: Список торговых пар
            parameters: Параметры стратегии
            risk_level: Уровень риска
            confidence_threshold: Порог уверенности
            execution_mode: Режим выполнения
        """
        self._strategy_id = strategy_id
        self._name = name
        self._strategy_type = strategy_type
        self._trading_pairs = [TradingPair(pair) for pair in trading_pairs]
        self._parameters = parameters
        self._risk_level = risk_level
        self._confidence_threshold = confidence_threshold
        self._execution_mode = execution_mode
        self._status = StrategyStatus.INACTIVE
        self._performance = StrategyPerformance()
        self._metrics = StrategyMetrics()
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._metadata: Dict[str, Any] = {}
        self._last_analysis: Optional[StrategyAnalysisResult] = None
        self._signal_history: List[Signal] = []
        self._execution_count = 0
        self._success_count = 0
        self._error_count = 0

    def get_strategy_id(self) -> StrategyId:
        """Получить ID стратегии."""
        return self._strategy_id

    def get_strategy_type(self) -> StrategyType:
        """Получить тип стратегии."""
        return self._strategy_type

    def get_execution_mode(self) -> StrategyExecutionMode:
        """Получить режим выполнения."""
        return self._execution_mode

    def set_execution_mode(self, mode: StrategyExecutionMode) -> None:
        """Установить режим выполнения."""
        self._execution_mode = mode
        self._updated_at = datetime.now()

    def analyze_market(self, context: StrategyContext) -> StrategyAnalysisResult:
        """
        Анализировать рынок и возвращать результаты анализа.
        Args:
            context: Контекст выполнения стратегии
        Returns:
            StrategyAnalysisResult: Результаты анализа рынка
        Raises:
            ValueError: Если данные некорректны
            RuntimeError: Если анализ не может быть выполнен
        """
        if not context.market_data:
            raise ValueError("Market data cannot be None")
        if not self._is_trading_pair_supported(context.market_data.symbol):
            raise ValueError(
                f"Trading pair {context.market_data.symbol} not supported by strategy"
            )
        # Выполняем специализированный анализ
        analysis_result = self._perform_market_analysis(context)
        # Сохраняем результат анализа
        self._last_analysis = analysis_result
        return analysis_result

    def generate_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Генерировать торговый сигнал на основе контекста.
        Args:
            context: Контекст выполнения стратегии
        Returns:
            Optional[Signal]: Сгенерированный сигнал или None
        """
        if not context.market_data:
            raise ValueError("Market data cannot be None")
        if not self._is_trading_pair_supported(context.market_data.symbol):
            raise ValueError(
                f"Trading pair {context.market_data.symbol} not supported by strategy"
            )
        # Проверка активности стратегии
        if not self.is_active():
            return None
        # Анализируем рынок
        analysis = self.analyze_market(context)
        # Генерируем сигнал на основе анализа
        signal = self._generate_signal_by_type(context, analysis)
        if signal:
            # Валидируем сигнал
            if not self.validate_signal(signal, context):
                return None
            # Добавляем в историю
            self._signal_history.append(signal)
            self._execution_count += 1
        return signal

    def validate_signal(self, signal: Signal, context: StrategyContext) -> bool:
        """
        Валидировать торговый сигнал.
        Args:
            signal: Сигнал для валидации
            context: Контекст выполнения
        Returns:
            bool: True если сигнал валиден
        """
        if not signal:
            return False
        # Проверяем базовые параметры
        if not hasattr(signal, "direction") or not signal.direction:
            return False
        if not hasattr(signal, "entry_price") or signal.entry_price <= 0:
            return False
        # Проверяем уверенность
        if hasattr(signal, "confidence") and signal.confidence < float(
            self._confidence_threshold
        ):
            return False
        # Проверяем риск-лимиты
        if context.risk_limits:
            if not self._validate_risk_limits(signal, context.risk_limits):
                return False
        return True

    def update_performance(self, signal: Signal, result: Dict[str, Any]) -> None:
        """
        Обновить производительность стратегии.
        Args:
            signal: Обработанный сигнал
            result: Результат обработки
        """
        if not signal or not result:
            return
        # Обновляем метрики
        if result.get("success", False):
            self._success_count += 1
            profit = result.get("profit", 0.0)
            if profit > 0:
                self._metrics.avg_profit = (
                    self._metrics.avg_profit * (self._metrics.successful_signals - 1)
                    + profit
                ) / self._metrics.successful_signals
        else:
            self._metrics.failed_signals += 1
            loss = abs(result.get("loss", 0.0))
            if loss > 0:
                self._metrics.avg_loss = (
                    self._metrics.avg_loss * (self._metrics.failed_signals - 1) + loss
                ) / self._metrics.failed_signals
        # Обновляем общие метрики
        self._metrics.total_signals = self._success_count + self._metrics.failed_signals
        if self._metrics.total_signals > 0:
            self._metrics.win_rate = self._success_count / self._metrics.total_signals
        # Обновляем время
        self._updated_at = datetime.now()

    def get_metrics(self) -> StrategyMetrics:
        """Получить метрики стратегии."""
        return self._metrics

    def get_performance(self) -> StrategyPerformance:
        """Получить производительность стратегии."""
        return self._performance

    def is_active(self) -> bool:
        """Проверить активность стратегии."""
        return self._status == StrategyStatus.ACTIVE

    def activate(self) -> None:
        """Активировать стратегию."""
        self._status = StrategyStatus.ACTIVE
        self._updated_at = datetime.now()

    def deactivate(self) -> None:
        """Деактивировать стратегию."""
        self._status = StrategyStatus.INACTIVE
        self._updated_at = datetime.now()

    def pause(self) -> None:
        """Приостановить стратегию."""
        self._status = StrategyStatus.PAUSED
        self._updated_at = datetime.now()

    def get_parameters(self) -> Dict[str, Any]:
        """Получить параметры стратегии."""
        return self._parameters.copy()

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Обновить параметры стратегии."""
        if self.validate_parameters(parameters):
            self._parameters.update(parameters)
            self._updated_at = datetime.now()

    def get_trading_pairs(self) -> List[TradingPair]:
        """Получить торговые пары."""
        return self._trading_pairs.copy()

    def add_trading_pair(self, trading_pair: str) -> None:
        """Добавить торговую пару."""
        if trading_pair not in [str(pair) for pair in self._trading_pairs]:
            self._trading_pairs.append(TradingPair(trading_pair))
            self._updated_at = datetime.now()

    def remove_trading_pair(self, trading_pair: str) -> None:
        """Удалить торговую пару."""
        self._trading_pairs = [
            pair for pair in self._trading_pairs if str(pair) != trading_pair
        ]
        self._updated_at = datetime.now()

    def get_metadata(self) -> Dict[str, Any]:
        """Получить метаданные."""
        return self._metadata.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """Установить метаданные."""
        self._metadata[key] = value
        self._updated_at = datetime.now()

    def get_execution_stats(self) -> Dict[str, Any]:
        """Получить статистику выполнения."""
        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": self._success_count / max(self._execution_count, 1),
            "last_analysis": (
                self._last_analysis.timestamp if self._last_analysis else None
            ),
            "signal_history_length": len(self._signal_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "strategy_id": str(self._strategy_id),
            "name": self._name,
            "strategy_type": self._strategy_type.value,
            "trading_pairs": [str(pair) for pair in self._trading_pairs],
            "parameters": self._parameters,
            "risk_level": float(self._risk_level),
            "confidence_threshold": float(self._confidence_threshold),
            "execution_mode": self._execution_mode.value,
            "status": self._status.value,
            "performance": self._performance.to_dict(),
            "metrics": self._metrics.__dict__,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "metadata": self._metadata,
            "execution_stats": self.get_execution_stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedStrategyInterface":
        """Десериализация из словаря."""
        strategy_id = StrategyId(UUID(data["strategy_id"]))
        strategy_type = StrategyType(data["strategy_type"])
        risk_level = RiskLevel(Decimal(str(data["risk_level"])))
        confidence_threshold = ConfidenceLevel(
            Decimal(str(data["confidence_threshold"]))
        )
        execution_mode = StrategyExecutionMode(data["execution_mode"])
        instance = cls(
            strategy_id=strategy_id,
            name=data["name"],
            strategy_type=strategy_type,
            trading_pairs=data["trading_pairs"],
            parameters=data["parameters"],
            risk_level=risk_level,
            confidence_threshold=confidence_threshold,
            execution_mode=execution_mode,
        )
        # Восстанавливаем состояние
        instance._status = StrategyStatus(data["status"])
        instance._performance = StrategyPerformance.from_dict(data["performance"])
        instance._metrics = StrategyMetrics(**data["metrics"])
        instance._created_at = datetime.fromisoformat(data["created_at"])
        instance._updated_at = datetime.fromisoformat(data["updated_at"])
        instance._metadata = data["metadata"]
        return instance

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Валидировать параметры стратегии.
        Args:
            parameters: Параметры для валидации
        Returns:
            bool: True если параметры валидны
        """
        if not parameters:
            return False
        required_params = self._get_required_parameters()
        for param in required_params:
            if param not in parameters:
                return False
        return self._validate_parameter_types_and_ranges(parameters)

    @abstractmethod
    def _perform_market_analysis(
        self, context: StrategyContext
    ) -> StrategyAnalysisResult:
        """
        Выполнить анализ рынка.
        Args:
            context: Контекст выполнения
        Returns:
            StrategyAnalysisResult: Результаты анализа
        """
        pass

    @abstractmethod
    def _generate_signal_by_type(
        self, context: StrategyContext, analysis: StrategyAnalysisResult
    ) -> Optional[Signal]:
        """
        Генерировать сигнал по типу стратегии.
        Args:
            context: Контекст выполнения
            analysis: Результаты анализа
        Returns:
            Optional[Signal]: Сгенерированный сигнал
        """
        pass

    def _is_trading_pair_supported(self, symbol: str) -> bool:
        """Проверить поддержку торговой пары."""
        return symbol in [str(pair) for pair in self._trading_pairs]

    def _validate_risk_limits(
        self, signal: Signal, risk_limits: Dict[str, Any]
    ) -> bool:
        """Валидировать риск-лимиты."""
        # Базовая реализация - всегда True
        # Подклассы могут переопределить для специфичной логики
        return True

    def _get_required_parameters(self) -> List[str]:
        """Получить список обязательных параметров."""
        return []

    def _validate_parameter_types_and_ranges(self, parameters: Dict[str, Any]) -> bool:
        """Валидировать типы и диапазоны параметров."""
        # Базовая реализация - всегда True
        # Подклассы могут переопределить для специфичной логики
        return True
