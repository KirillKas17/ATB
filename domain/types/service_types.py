"""
Типы для доменных сервисов.
Определяет строгую типизацию для всех сервисов в domain/services,
обеспечивая типобезопасность и соответствие DDD принципам.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
    TypeAlias,
)

import pandas as pd

from domain.entities.signal import Signal
from domain.entities.strategy import Strategy
from domain.types import (
    ConfidenceLevel,
    MetadataDict,
    PerformanceScore,
    PriceValue,
    RiskLevel,
    SignalId,
    StrategyId,
    Symbol,
    TimestampValue,
    VolumeValue,
)

# ============================================================================
# БАЗОВЫЕ ТИПЫ ДЛЯ СЕРВИСОВ
# ============================================================================
# Типы для рыночных данных
MarketDataFrame: TypeAlias = pd.DataFrame
OrderBookData = Dict[str, Any]
HistoricalData: TypeAlias = pd.DataFrame
FeatureVector = List[float]
# Типы для метрик
VolatilityValue = NewType("VolatilityValue", Decimal)
TrendStrengthValue = NewType("TrendStrengthValue", Decimal)
CorrelationValue = NewType("CorrelationValue", Decimal)
LiquidityScore = NewType("LiquidityScore", Decimal)
SpreadValue = NewType("SpreadValue", Decimal)
# Типы для конфигураций
ServiceConfig = NewType("ServiceConfig", Dict[str, Any])
AnalysisConfig = NewType("AnalysisConfig", Dict[str, Any])
ModelConfig = NewType("ModelConfig", Dict[str, Any])


# ============================================================================
# ENUMS
# ============================================================================
class TrendDirection(str, Enum):
    """Направления тренда."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class VolatilityTrend(str, Enum):
    """Тренды волатильности."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class VolumeTrend(str, Enum):
    """Тренды объема."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class CorrelationStrength(str, Enum):
    """Сила корреляции."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class MarketRegime(str, Enum):
    """Рыночные режимы."""

    EFFICIENT = "efficient"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"


class LiquidityZoneType(str, Enum):
    """Типы зон ликвидности."""

    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


class SweepType(str, Enum):
    """Типы sweep'ов."""

    SWEEP_HIGH = "sweep_high"
    SWEEP_LOW = "sweep_low"


class PatternType(str, Enum):
    """Типы паттернов."""

    CANDLE = "candle"
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    COMBINED = "combined"


class IndicatorType(str, Enum):
    """Типы индикаторов."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"


# ============================================================================
# TYPEDDICT ДЛЯ РЕЗУЛЬТАТОВ АНАЛИЗА
# ============================================================================
class VolatilityMetrics(TypedDict, total=False):
    """Метрики волатильности."""

    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    volatility_trend: VolatilityTrend
    garch_volatility: Optional[float]
    realized_volatility: Optional[float]
    implied_volatility: Optional[float]


class TrendMetrics(TypedDict, total=False):
    """Метрики тренда."""

    trend_direction: TrendDirection
    trend_strength: float
    trend_duration: int
    support_resistance: Dict[str, float]
    adx_value: Optional[float]
    trend_quality: Optional[float]
    momentum_score: Optional[float]


class VolumeMetrics(TypedDict, total=False):
    """Метрики объема."""

    volume_ma: float
    volume_ratio: float
    volume_trend: VolumeTrend
    volume_profile: Dict[str, Any]
    volume_delta: Optional[float]
    volume_imbalance: Optional[float]
    vwap: Optional[float]


class CorrelationMetrics(TypedDict, total=False):
    """Метрики корреляции."""

    correlation_coefficient: float
    correlation_strength: CorrelationStrength
    correlation_trend: str
    lag_value: Optional[int]
    rolling_correlation: Optional[float]
    cointegration_score: Optional[float]


class MarketEfficiencyMetrics(TypedDict, total=False):
    """Метрики эффективности рынка."""

    efficiency_ratio: float
    hurst_exponent: float
    market_regime: MarketRegime
    fractal_dimension: Optional[float]
    entropy_value: Optional[float]


class LiquidityMetrics(TypedDict, total=False):
    """Метрики ликвидности."""

    liquidity_score: float
    bid_ask_spread: float
    order_book_depth: float
    volume_imbalance: float
    market_impact: Optional[float]
    slippage_estimate: Optional[float]


class MomentumMetrics(TypedDict, total=False):
    """Метрики моментума."""

    rsi_value: float
    macd_value: float
    stochastic_value: float
    williams_r: float
    momentum_score: float
    divergence_detected: bool


class MarketStressMetrics(TypedDict, total=False):
    """Метрики рыночного стресса."""

    stress_index: float
    fear_greed_index: float
    volatility_regime: str
    liquidity_crisis: bool
    flash_crash_risk: float


# ============================================================================
# DATACLASS ДЛЯ СТРУКТУРИРОВАННЫХ ДАННЫХ
# ============================================================================
@dataclass(frozen=True)
class MarketMetricsResult:
    """Результат расчета рыночных метрик."""

    volatility: VolatilityMetrics
    trend: TrendMetrics
    volume: VolumeMetrics
    momentum: MomentumMetrics
    liquidity: LiquidityMetrics
    stress: MarketStressMetrics
    timestamp: TimestampValue
    symbol: Symbol
    confidence: ConfidenceLevel
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class LiquidityZone:
    """Зона ликвидности."""

    price: PriceValue
    zone_type: LiquidityZoneType
    strength: float
    volume: VolumeValue
    touches: int
    timestamp: TimestampValue
    confidence: ConfidenceLevel
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class LiquiditySweep:
    """Sweep ликвидности."""

    timestamp: TimestampValue
    price: PriceValue
    sweep_type: SweepType
    confidence: ConfidenceLevel
    volume: VolumeValue
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class LiquidityAnalysisResult:
    """Результат анализа ликвидности."""

    liquidity_score: LiquidityScore
    confidence: ConfidenceLevel
    volume_score: float
    order_book_score: float
    volatility_score: float
    zones: List[LiquidityZone]
    sweeps: List[LiquiditySweep]
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class SpreadAnalysisResult:
    """Результат анализа спреда."""

    spread: SpreadValue
    imbalance: float
    confidence: ConfidenceLevel
    best_bid: PriceValue
    best_ask: PriceValue
    depth_imbalance: float
    spread_trend: str
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class SpreadMovementPrediction:
    """Предсказание движения спреда."""

    prediction: float
    confidence: ConfidenceLevel
    ma_short: float
    ma_long: float
    volatility: float
    direction: str
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class PredictionResult:
    """Результат ML предсказания."""

    predicted_spread: float
    confidence: ConfidenceLevel
    model_accuracy: float
    feature_importance: Dict[str, float]
    prediction_interval: Optional[tuple[float, float]]
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class LiquidityPredictionResult:
    """Результат предсказания ликвидности."""

    predicted_class: str
    confidence: ConfidenceLevel
    probabilities: Dict[str, float]
    model_accuracy: float
    feature_importance: Dict[str, float]
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(frozen=True)
class MLModelPerformance:
    """Производительность ML моделей."""

    spread_accuracy: float
    liquidity_accuracy: float
    spread_loss: float
    liquidity_loss: float
    overall_performance: float
    training_samples: int
    validation_samples: int
    last_training: TimestampValue
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


# ============================================================================
# ПРОТОКОЛЫ ДЛЯ СЕРВИСОВ
# ============================================================================
@runtime_checkable
class MarketAnalysisProtocol(Protocol):
    """Протокол для анализа рынка."""

    async def analyze_market_data(
        self, data: MarketDataFrame, order_book: Optional[OrderBookData] = None
    ) -> MarketMetricsResult: ...
    async def calculate_volatility_metrics(
        self, data: MarketDataFrame
    ) -> VolatilityMetrics: ...
    async def calculate_trend_metrics(self, data: MarketDataFrame) -> TrendMetrics: ...
@runtime_checkable
class PatternAnalysisProtocol(Protocol):
    """Протокол для анализа паттернов."""

    async def discover_patterns(
        self, data: MarketDataFrame, config: AnalysisConfig
    ) -> List[Any]: ...
    async def validate_pattern(self, pattern: Any, data: MarketDataFrame) -> float: ...
@runtime_checkable
class RiskAnalysisProtocol(Protocol):
    """Протокол для анализа рисков."""

    async def calculate_risk_metrics(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> Any: ...
    async def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float: ...
@runtime_checkable
class LiquidityAnalysisProtocol(Protocol):
    """Протокол для анализа ликвидности."""

    async def analyze_liquidity(
        self, market_data: MarketDataFrame, order_book: OrderBookData
    ) -> LiquidityAnalysisResult: ...
    async def identify_liquidity_zones(
        self, market_data: MarketDataFrame
    ) -> List[LiquidityZone]: ...
@runtime_checkable
class SpreadAnalysisProtocol(Protocol):
    """Протокол для анализа спредов."""

    async def analyze_spread(
        self, order_book: OrderBookData
    ) -> SpreadAnalysisResult: ...
    async def predict_spread_movement(
        self, historical_data: HistoricalData
    ) -> SpreadMovementPrediction: ...
@runtime_checkable
class MLPredictionProtocol(Protocol):
    """Протокол для ML предсказаний."""

    async def predict_spread(self, features: FeatureVector) -> PredictionResult: ...
    async def predict_liquidity(
        self, features: FeatureVector
    ) -> LiquidityPredictionResult: ...
    async def train_models(self, training_data: HistoricalData) -> bool: ...


# ============================================================================
# ТИПЫ ДЛЯ СИГНАЛОВ
# ============================================================================
SignalGenerator = Callable[[Strategy, MarketDataFrame], List[Signal]]
AsyncSignalGenerator = Callable[
    [Strategy, MarketDataFrame], Coroutine[Any, Any, List[Signal]]
]


@dataclass(frozen=True)
class AggregationRules:
    """Правила агрегации сигналов."""

    confidence_threshold: float = 0.5
    max_signals: int = 10
    time_window_hours: int = 24
    weight_by_confidence: bool = True
    weight_by_recency: bool = True
    weight_by_volume: bool = False


# ============================================================================
# КОНСТАНТЫ
# ============================================================================
DEFAULT_VOLATILITY_WINDOW: Final[int] = 20
DEFAULT_TREND_WINDOW: Final[int] = 30
DEFAULT_VOLUME_WINDOW: Final[int] = 20
DEFAULT_CORRELATION_WINDOW: Final[int] = 30
DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_MACD_FAST: Final[int] = 12
DEFAULT_MACD_SLOW: Final[int] = 26
DEFAULT_MACD_SIGNAL: Final[int] = 9
DEFAULT_BB_PERIOD: Final[int] = 20
DEFAULT_BB_STD: Final[float] = 2.0
DEFAULT_ATR_PERIOD: Final[int] = 14
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.3
MAX_CONFIDENCE_THRESHOLD: Final[float] = 1.0
DEFAULT_RISK_FREE_RATE: Final[float] = 0.02
DEFAULT_VAR_CONFIDENCE: Final[float] = 0.95
# ============================================================================
# ЭКСПОРТ
# ============================================================================
__all__ = [
    # Базовые типы
    "MarketDataFrame",
    "OrderBookData",
    "HistoricalData",
    "FeatureVector",
    "VolatilityValue",
    "TrendStrengthValue",
    "CorrelationValue",
    "LiquidityScore",
    "SpreadValue",
    "ServiceConfig",
    "AnalysisConfig",
    "ModelConfig",
    # Enums
    "TrendDirection",
    "VolatilityTrend",
    "VolumeTrend",
    "CorrelationStrength",
    "MarketRegime",
    "LiquidityZoneType",
    "SweepType",
    "PatternType",
    "IndicatorType",
    # TypedDict
    "VolatilityMetrics",
    "TrendMetrics",
    "VolumeMetrics",
    "CorrelationMetrics",
    "MarketEfficiencyMetrics",
    "LiquidityMetrics",
    "MomentumMetrics",
    "MarketStressMetrics",
    # Dataclasses
    "MarketMetricsResult",
    "LiquidityZone",
    "LiquiditySweep",
    "LiquidityAnalysisResult",
    "SpreadAnalysisResult",
    "SpreadMovementPrediction",
    "PredictionResult",
    "LiquidityPredictionResult",
    "MLModelPerformance",
    "AggregationRules",
    # Protocols
    "MarketAnalysisProtocol",
    "PatternAnalysisProtocol",
    "RiskAnalysisProtocol",
    "LiquidityAnalysisProtocol",
    "SpreadAnalysisProtocol",
    "MLPredictionProtocol",
    # Signal types
    "SignalGenerator",
    "AsyncSignalGenerator",
    # Константы
    "DEFAULT_VOLATILITY_WINDOW",
    "DEFAULT_TREND_WINDOW",
    "DEFAULT_VOLUME_WINDOW",
    "DEFAULT_CORRELATION_WINDOW",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_MACD_FAST",
    "DEFAULT_MACD_SLOW",
    "DEFAULT_MACD_SIGNAL",
    "DEFAULT_BB_PERIOD",
    "DEFAULT_BB_STD",
    "DEFAULT_ATR_PERIOD",
    "MIN_CONFIDENCE_THRESHOLD",
    "MAX_CONFIDENCE_THRESHOLD",
    "DEFAULT_RISK_FREE_RATE",
    "DEFAULT_VAR_CONFIDENCE",
]
