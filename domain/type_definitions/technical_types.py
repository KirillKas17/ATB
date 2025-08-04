"""
Промышленные типы для технического анализа.
Содержит строго типизированные структуры данных для технического анализа,
индикаторов и рыночной структуры.
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)


class MarketStructure(Enum):
    """Структура рынка."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


class TrendStrength(Enum):
    """Сила тренда."""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalType(Enum):
    """Типы сигналов."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalStrength(Enum):
    """Сила сигнала."""

    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TrendDirection(Enum):
    """Направление тренда."""

    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"


class PatternType(Enum):
    """Типы паттернов."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class IndicatorType(Enum):
    """Типы индикаторов."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"


class IndicatorCategory(Enum):
    """Категории индикаторов."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass(frozen=True)
class TradingSignal:
    """Торговый сигнал."""

    signal_type: SignalType
    strength: SignalStrength
    indicator: str
    price: Decimal
    timestamp: datetime
    description: str
    confidence: float


@dataclass(frozen=True)
class SupportResistanceLevel:
    """Уровень поддержки/сопротивления."""

    price: float
    level_type: str  # "support" или "resistance"
    strength: float
    touches: int
    timestamp: datetime


@dataclass(frozen=True)
class BollingerBandsResult:
    """Результат расчета полос Боллинджера."""

    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series
    # Параметры расчета
    period: int = field(default=20)
    std_dev: float = field(default=2.0)
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        if self.std_dev <= 0:
            raise ValueError("Standard deviation must be positive")


@dataclass(frozen=True)
class MACDResult:
    """Результат расчета MACD."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series
    divergence: Optional[str] = field(default=None)
    # Параметры расчета
    fast_period: int = field(default=12)
    slow_period: int = field(default=26)
    signal_period: int = field(default=9)
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")


@dataclass(frozen=True)
class VolumeProfileResult:
    """Результат расчета профиля объема."""

    poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_by_price: Dict[float, float]
    histogram: List[float]
    price_levels: List[float]
    # Параметры расчета
    bins: int = field(default=24)
    value_area_percentage: float = field(default=0.68)
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.bins <= 0:
            raise ValueError("Number of bins must be positive")
        if not 0 < self.value_area_percentage < 1:
            raise ValueError("Value area percentage must be between 0 and 1")


@dataclass(frozen=True)
class MarketStructureResult:
    """Результат анализа структуры рынка."""

    structure: MarketStructure
    trend_strength: TrendStrength
    volatility: Decimal
    adx: Decimal
    rsi: Decimal
    confidence: Decimal
    # Дополнительные метрики
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    key_levels: List[float] = field(default_factory=list)
    # Метаданные
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    analysis_period: str = field(default="daily")
    data_points: int = field(default=0)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")


@dataclass(frozen=True)
class TechnicalIndicatorResult:
    """Результат расчета технических индикаторов."""

    indicators: Dict[str, pd.Series]
    market_structure: MarketStructureResult
    volume_profile: VolumeProfileResult
    support_levels: List[float]
    resistance_levels: List[float]
    # Дополнительные данные
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)
    pivot_points: Dict[str, float] = field(default_factory=dict)
    candlestick_patterns: List[str] = field(default_factory=list)
    # Метаданные
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    symbols: List[str] = field(default_factory=list)
    timeframe: str = field(default="1d")


@dataclass(frozen=True)
class SignalResult:
    """Результат генерации сигналов."""

    signal_type: SignalType
    strength: Decimal
    confidence: Decimal
    entry_price: Optional[float] = field(default=None)
    stop_loss: Optional[float] = field(default=None)
    take_profit: Optional[float] = field(default=None)
    # Дополнительные данные
    indicators_used: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    risk_reward_ratio: Optional[Decimal] = field(default=None)
    # Метаданные
    signal_timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = field(default="")
    timeframe: str = field(default="1d")

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not 0 <= self.strength <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Signal confidence must be between 0 and 1")


@dataclass(frozen=True)
class TechnicalAnalysisReport:
    """Полный отчет технического анализа."""

    indicator_results: TechnicalIndicatorResult
    signals: List[SignalResult]
    market_structure: MarketStructureResult
    # Рекомендации
    trading_recommendations: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)
    key_levels: List[float] = field(default_factory=list)
    # Метаданные
    report_timestamp: datetime = field(default_factory=datetime.now)
    analysis_period: str = field(default="daily")
    symbols_analyzed: List[str] = field(default_factory=list)


# Protocol для сервиса технического анализа
@runtime_checkable
class TechnicalAnalysisServiceProtocol(Protocol):
    """Протокол для сервиса технического анализа."""

    def calculate_indicators(self, data: pd.DataFrame) -> TechnicalIndicatorResult: ...
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series: ...
    def calculate_macd(
        self,
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> MACDResult: ...
    def calculate_bollinger_bands(
        self, data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> BollingerBandsResult: ...
    def calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series: ...
    def calculate_volume_profile(
        self, data: pd.DataFrame, bins: int = 24
    ) -> VolumeProfileResult: ...
    def calculate_market_structure(
        self, data: pd.DataFrame
    ) -> MarketStructureResult: ...
    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List[SignalResult]: ...
    def detect_support_resistance(
        self, data: pd.DataFrame, window: int = 20
    ) -> Dict[str, List[float]]: ...
    def calculate_fibonacci_retracements(
        self, high: float, low: float
    ) -> Dict[str, float]: ...
    def generate_technical_report(
        self, data: pd.DataFrame, symbols: List[str]
    ) -> TechnicalAnalysisReport: ...


# Устаревшие типы для обратной совместимости
class LegacyTechnicalIndicatorResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    indicators: Dict[str, List[float]]


class LegacyBollingerBandsResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    upper: List[float]
    middle: List[float]
    lower: List[float]


class LegacyMACDResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    macd: List[float]
    signal: List[float]
    histogram: List[float]


class LegacyVolumeProfileResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    poc: float
    value_area: List[float]
    histogram: List[float]
    bins: List[float]


class LegacyMarketStructureResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    structure: str
    trend_strength: float
    volatility: float
    adx: float
    rsi: float
