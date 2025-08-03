# -*- coding: utf-8 -*-
"""Типы для модуля прогнозирования разворотов."""
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)
from uuid import UUID

import pandas as pd

from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp

# Новые типы для строгой типизации
PredictionId = NewType("PredictionId", UUID)
ReversalSignalId = NewType("ReversalSignalId", UUID)
DivergenceId = NewType("DivergenceId", UUID)
PatternId = NewType("PatternId", UUID)
PivotId = NewType("PivotId", UUID)
# Типы для метрик
ConfidenceScore = NewType("ConfidenceScore", float)
SignalStrengthScore = NewType("SignalStrengthScore", float)
RiskScore = NewType("RiskScore", float)
MomentumScore = NewType("MomentumScore", float)
VolumeScore = NewType("VolumeScore", float)
# Типы для технических индикаторов
RSIValue = NewType("RSIValue", float)
MACDValue = NewType("MACDValue", float)
MACDSignalValue = NewType("MACDSignalValue", float)
MACDHistogramValue = NewType("MACDHistogramValue", float)
ATRValue = NewType("ATRValue", float)
BBUpperValue = NewType("BBUpperValue", float)
BBMiddleValue = NewType("BBMiddleValue", float)
BBLowerValue = NewType("BBLowerValue", float)
# Типы для рыночных данных
OHLCVData = NewType("OHLCVData", Any)  # pd.DataFrame не может быть subclass
OrderBookData = Dict[str, Any]
MarketDataWindow = NewType("MarketDataWindow", Any)  # pd.DataFrame не может быть subclass
# Константы
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.0
MAX_CONFIDENCE_THRESHOLD: Final[float] = 1.0
MIN_SIGNAL_STRENGTH: Final[float] = 0.0
MAX_SIGNAL_STRENGTH: Final[float] = 1.0
DEFAULT_LOOKBACK_PERIOD: Final[int] = 100
DEFAULT_PREDICTION_HORIZON: Final[timedelta] = timedelta(hours=4)


# Перечисления
class ReversalDirection(Enum):
    """Направление разворота."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Сила сигнала."""

    WEAK = "weak"  # 0.0 - 0.3
    MODERATE = "moderate"  # 0.3 - 0.6
    STRONG = "strong"  # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0


class DivergenceType(Enum):
    """Типы дивергенций."""

    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    TRIPLE_BULLISH = "triple_bullish"
    TRIPLE_BEARISH = "triple_bearish"


class CandlestickPatternType(Enum):
    """Типы свечных паттернов."""

    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HANGING_MAN = "hanging_man"
    INVERSION_HAMMER = "inversion_hammer"


class MarketPhase(Enum):
    """Фазы рынка."""

    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    TRANSITION = "transition"


class RiskLevel(Enum):
    """Уровни риска."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# TypedDict для конфигураций
class PredictionConfig(TypedDict, total=False):
    """Конфигурация прогнозирования разворотов."""

    lookback_period: int
    min_confidence: float
    min_signal_strength: float
    prediction_horizon: timedelta
    rsi_oversold: float
    rsi_overbought: float
    macd_signal_threshold: float
    volume_threshold: float
    divergence_lookback: int
    min_divergence_strength: float
    pattern_confidence_threshold: float
    volume_confirmation_required: bool
    momentum_window: int
    momentum_threshold: float
    mean_reversion_window: int
    deviation_threshold: float


# Protocol интерфейсы
@runtime_checkable
class ReversalPredictorProtocol(Protocol):
    """Протокол для прогнозатора разворотов."""

    def predict_reversal(
        self,
        symbol: str,
        market_data: OHLCVData,
        order_book: Optional[OrderBookData] = None,
    ) -> Optional["ReversalSignalProtocol"]:
        """Прогнозирование разворота."""
        ...

    def validate_market_data(self, data: OHLCVData) -> bool:
        """Валидация рыночных данных."""
        ...

    def get_prediction_config(self) -> PredictionConfig:
        """Получение конфигурации прогнозирования."""
        ...

    def update_prediction_config(self, config: PredictionConfig) -> None:
        """Обновление конфигурации прогнозирования."""
        ...


@runtime_checkable
class ReversalSignalProtocol(Protocol):
    """Протокол для сигнала разворота."""

    @property
    def symbol(self) -> str:
        """Символ торговой пары."""
        ...

    @property
    def direction(self) -> ReversalDirection:
        """Направление разворота."""
        ...

    @property
    def pivot_price(self) -> Price:
        """Цена разворота."""
        ...

    @property
    def confidence(self) -> ConfidenceScore:
        """Уверенность в сигнале."""
        ...

    @property
    def signal_strength(self) -> SignalStrengthScore:
        """Сила сигнала."""
        ...

    @property
    def timestamp(self) -> Timestamp:
        """Временная метка."""
        ...

    @property
    def horizon(self) -> timedelta:
        """Горизонт прогнозирования."""
        ...

    @property
    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        ...

    @property
    def strength_category(self) -> SignalStrength:
        """Категория силы сигнала."""
        ...

    @property
    def risk_level(self) -> RiskLevel:
        """Уровень риска."""
        ...

    def enhance_confidence(self, factor: float) -> None:
        """Усиление уверенности."""
        ...

    def reduce_confidence(self, factor: float) -> None:
        """Снижение уверенности."""
        ...

    def mark_controversial(
        self, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Пометка как спорного."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        ...


# ============================================================================
# TYPED DICTS ДЛЯ ПРЕДИКТИВНОГО АНАЛИЗА
# ============================================================================
class PivotPointData(TypedDict, total=False):
    """Данные точки разворота."""

    price: float
    timestamp: datetime
    volume: float
    pivot_type: Literal["high", "low"]
    strength: float
    confirmation_levels: list[float]
    volume_cluster: Optional[float]
    fibonacci_levels: list[float]


class CandlestickMetadata(TypedDict, total=False):
    """Метаданные свечного паттерна."""

    pattern_type: str
    confirmation_count: int
    volume_ratio: float
    body_ratio: float
    shadow_ratio: float
    trend_alignment: bool
    support_resistance_levels: list[float]
    fibonacci_retracements: list[float]


class ControversyDetails(TypedDict, total=False):
    """Детали спорности сигнала."""

    reason: str
    timestamp: str
    conflicting_indicators: list[str]
    confidence_delta: float
    strength_delta: float
    market_context: str
    technical_analysis: dict[str, float]


class AnalysisMetadata(TypedDict, total=False):
    """Метаданные анализа."""

    analysis_version: str
    indicators_used: list[str]
    timeframes_analyzed: list[str]
    data_quality_score: float
    market_regime: str
    volatility_regime: str
    trend_strength: float
    volume_profile_analysis: bool
    order_book_analysis: bool
    sentiment_analysis: bool


class RiskMetrics(TypedDict, total=False):
    """Метрики риска."""

    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    beta: float
    correlation: float


class OrderBookSnapshot(TypedDict, total=False):
    """Снимок ордербука."""

    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]  # (price, volume)
    asks: list[tuple[float, float]]  # (price, volume)
    spread: float
    depth: float
    imbalance: float
    liquidity_score: float


class MomentumMetrics(TypedDict, total=False):
    """Метрики импульса."""

    momentum_loss: float
    velocity_change: float
    acceleration: float
    volume_momentum: float
    price_momentum: float
    momentum_divergence: Optional[float]
    momentum_strength: float
    momentum_direction: str


class VolumeProfileData(TypedDict, total=False):
    """Данные профиля объема."""

    price_level: float
    volume_density: float
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_nodes: list[dict[str, float]]
    imbalance_ratio: float
    volume_clusters: list[dict[str, float]]


class LiquidityClusterData(TypedDict, total=False):
    """Данные кластера ликвидности."""

    price: float
    volume: float
    side: Literal["bid", "ask"]
    cluster_size: int
    strength: float
    timestamp: datetime
    order_count: int
    average_order_size: float


class DivergenceAnalysis(TypedDict, total=False):
    """Анализ дивергенций."""

    type: str
    indicator: str
    price_highs: list[float]
    price_lows: list[float]
    indicator_highs: list[float]
    indicator_lows: list[float]
    strength: float
    confidence: float
    timestamp: datetime
    confirmation_count: int
    false_signal_count: int


class MeanReversionData(TypedDict, total=False):
    """Данные возврата к среднему."""

    upper_band: float
    lower_band: float
    middle_line: float
    deviation: float
    band_width: float
    current_position: float
    timestamp: datetime
    mean_reversion_strength: float
    breakout_probability: float


class PredictionResult(TypedDict, total=False):
    """Результат прогнозирования."""

    symbol: str
    direction: str
    pivot_price: float
    confidence: float
    signal_strength: float
    timestamp: datetime
    horizon: timedelta
    risk_level: str
    is_controversial: bool
    agreement_score: float
    analysis_metadata: AnalysisMetadata
    risk_metrics: RiskMetrics
