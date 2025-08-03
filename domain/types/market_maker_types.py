"""
Типы для модуля market maker.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Tuple,
    Union,
    runtime_checkable,
)

from domain.types import Symbol

# ============================================================================
# NEW TYPES
# ============================================================================
PatternId = NewType("PatternId", str)
# Symbol импортируется из domain.types
Confidence = NewType("Confidence", float)
SimilarityScore = NewType("SimilarityScore", float)
SignalStrength = NewType("SignalStrength", float)
BookPressure = NewType("BookPressure", float)
VolumeDelta = NewType("VolumeDelta", float)
PriceReaction = NewType("PriceReaction", float)
SpreadChange = NewType("SpreadChange", float)
OrderImbalance = NewType("OrderImbalance", float)
LiquidityDepth = NewType("LiquidityDepth", float)
TimeDuration = NewType("TimeDuration", int)
VolumeConcentration = NewType("VolumeConcentration", float)
PriceVolatility = NewType("PriceVolatility", float)
Accuracy = NewType("Accuracy", float)
AverageReturn = NewType("AverageReturn", float)
SuccessCount = NewType("SuccessCount", int)
TotalCount = NewType("TotalCount", int)


# ============================================================================
# ENUMS
# ============================================================================
class MarketMakerPatternType(Enum):
    """Типы паттернов маркет-мейкера."""

    ACCUMULATION = "accumulation"
    ABSORPTION = "absorption"
    SPOOFING = "spoofing"
    EXIT = "exit"
    PRESSURE_ZONE = "pressure_zone"
    LIQUIDITY_GRAB = "liquidity_grab"
    STOP_HUNT = "stop_hunt"
    FAKE_BREAKOUT = "fake_breakout"
    WASH_TRADING = "wash_trading"
    PUMP_AND_DUMP = "pump_and_dump"


class PatternOutcome(Enum):
    """Результаты паттернов."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    PARTIAL_SUCCESS = "partial_success"


class PatternConfidence(Enum):
    """Уровни уверенности в паттерне."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketPhase(Enum):
    """Рыночные фазы."""

    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    TRANSITION = "transition"


class OrderSide(Enum):
    """Стороны ордера."""

    BUY = "buy"
    SELL = "sell"


class TradeType(Enum):
    """Типы сделок."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# ============================================================================
# TYPED DICTS
# ============================================================================
class OrderBookLevel(TypedDict):
    """Уровень стакана заявок."""

    price: float
    size: float
    orders_count: int


class TradeData(TypedDict):
    """Данные сделки."""

    price: float
    size: float
    side: str
    time: datetime
    trade_id: str
    maker: bool


class MarketMicrostructure(TypedDict, total=False):
    """Микроструктура рынка."""

    avg_trade_size: float
    trade_size_std: float
    avg_interval: float
    interval_std: float
    buy_sell_ratio: float
    maker_taker_ratio: float
    large_trades_ratio: float
    price_impact: float
    order_flow_imbalance: float
    liquidity_consumption: float


class PatternContext(TypedDict, total=False):
    """Контекст паттерна."""

    symbol: str
    timestamp: str
    last_price: float
    volume_24h: float
    price_change_24h: float
    trades_count: int
    order_book_depth: int
    market_phase: str
    volatility_regime: str
    liquidity_regime: str
    external_factors: Dict[str, Any]


class PatternFeaturesDict(TypedDict):
    """Словарь признаков паттерна."""

    book_pressure: float
    volume_delta: float
    price_reaction: float
    spread_change: float
    order_imbalance: float
    liquidity_depth: float
    time_duration: int
    volume_concentration: float
    price_volatility: float
    market_microstructure: MarketMicrostructure


class PatternResultDict(TypedDict):
    """Словарь результата паттерна."""

    outcome: str
    price_change_5min: float
    price_change_15min: float
    price_change_30min: float
    volume_change: float
    volatility_change: float
    market_context: PatternContext


class PatternMemoryDict(TypedDict):
    """Словарь памяти паттерна."""

    pattern: Dict[str, Any]
    result: Optional[PatternResultDict]
    accuracy: float
    avg_return: float
    success_count: int
    total_count: int
    last_seen: Optional[str]


class MatchedPatternDict(TypedDict):
    """Словарь совпадения паттерна."""

    pattern_memory: PatternMemoryDict
    similarity_score: float
    confidence_boost: float
    expected_outcome: PatternResultDict
    signal_strength: float


# ============================================================================
# PROTOCOLS
# ============================================================================
@runtime_checkable
class OrderBookProtocol(Protocol):
    """Протокол для ордербука."""

    timestamp: datetime
    symbol: Symbol
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_price: float
    volume_24h: float
    price_change_24h: float

    def get_bid_volume(self, levels: int = 5) -> float: ...
    def get_ask_volume(self, levels: int = 5) -> float: ...
    def get_mid_price(self) -> float: ...
    def get_spread(self) -> float: ...
    def get_spread_percentage(self) -> float: ...
    def get_order_imbalance(self) -> OrderImbalance: ...
    def get_liquidity_depth(self) -> LiquidityDepth: ...
@runtime_checkable
class TradeSnapshotProtocol(Protocol):
    """Протокол для снимка сделок."""

    timestamp: datetime
    symbol: Symbol
    trades: List[TradeData]

    def get_total_volume(self) -> float: ...
    def get_buy_volume(self) -> float: ...
    def get_sell_volume(self) -> float: ...
    def get_volume_delta(self, window: int = 10) -> VolumeDelta: ...
    def get_price_reaction(self) -> PriceReaction: ...
    def get_volume_concentration(self) -> VolumeConcentration: ...
    def get_price_volatility(self) -> PriceVolatility: ...
@runtime_checkable
class PatternFeaturesProtocol(Protocol):
    """Протокол для признаков паттерна."""

    book_pressure: BookPressure
    volume_delta: VolumeDelta
    price_reaction: PriceReaction
    spread_change: SpreadChange
    order_imbalance: OrderImbalance
    liquidity_depth: LiquidityDepth
    time_duration: TimeDuration
    volume_concentration: VolumeConcentration
    price_volatility: PriceVolatility
    market_microstructure: MarketMicrostructure

    def to_dict(self) -> PatternFeaturesDict: ...
    @classmethod
    def from_dict(cls, data: PatternFeaturesDict) -> "PatternFeaturesProtocol": ...
@runtime_checkable
class PatternResultProtocol(Protocol):
    """Протокол для результата паттерна."""

    outcome: PatternOutcome
    price_change_5min: float
    price_change_15min: float
    price_change_30min: float
    volume_change: float
    volatility_change: float
    market_context: PatternContext

    def to_dict(self) -> PatternResultDict: ...
    @classmethod
    def from_dict(cls, data: PatternResultDict) -> "PatternResultProtocol": ...
@runtime_checkable
class MarketMakerPatternProtocol(Protocol):
    """Протокол для паттерна маркет-мейкера."""

    pattern_type: MarketMakerPatternType
    symbol: Symbol
    timestamp: datetime
    features: PatternFeaturesProtocol
    confidence: Confidence
    context: PatternContext

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketMakerPatternProtocol": ...
@runtime_checkable
class PatternMemoryProtocol(Protocol):
    """Протокол для памяти паттерна."""

    pattern: MarketMakerPatternProtocol
    result: Optional[PatternResultProtocol]
    accuracy: Accuracy
    avg_return: AverageReturn
    success_count: SuccessCount
    total_count: TotalCount
    last_seen: Optional[datetime]

    def update_result(self, result: PatternResultProtocol) -> None: ...
    def to_dict(self) -> PatternMemoryDict: ...
    @classmethod
    def from_dict(cls, data: PatternMemoryDict) -> "PatternMemoryProtocol": ...
@runtime_checkable
class MatchedPatternProtocol(Protocol):
    """Протокол для совпадения паттерна."""

    pattern_memory: PatternMemoryProtocol
    similarity_score: SimilarityScore
    confidence_boost: Confidence
    expected_outcome: PatternResultProtocol
    signal_strength: SignalStrength

    def to_dict(self) -> MatchedPatternDict: ...
    @classmethod
    def from_dict(cls, data: MatchedPatternDict) -> "MatchedPatternProtocol": ...


# ============================================================================
# CONFIGURATION TYPES
# ============================================================================
@dataclass(frozen=True)
class PatternClassifierConfig:
    """Конфигурация классификатора паттернов."""

    min_confidence: Confidence = Confidence(0.6)
    volume_threshold: float = 1000.0
    spread_threshold: float = 0.001
    imbalance_threshold: float = 0.3
    pressure_threshold: float = 0.2
    time_window: int = 300  # 5 минут
    min_trades: int = 10
    similarity_threshold: SimilarityScore = SimilarityScore(0.8)
    accuracy_threshold: Accuracy = Accuracy(0.7)
    max_history_size: int = 1000


@dataclass(frozen=True)
class PatternMemoryConfig:
    """Конфигурация памяти паттернов."""

    base_path: str = "market_profiles"
    cleanup_days: int = 30
    min_accuracy_for_cleanup: Accuracy = Accuracy(0.5)
    max_patterns_per_symbol: int = 1000
    backup_enabled: bool = True
    compression_enabled: bool = True


# ============================================================================
# UTILITY TYPES
# ============================================================================
@dataclass(frozen=True)
class PatternScore:
    """Оценка паттерна."""

    pattern_type: MarketMakerPatternType
    score: float
    confidence: Confidence
    factors: Dict[str, float]


@dataclass(frozen=True)
class PatternPrediction:
    """Предсказание паттерна."""

    pattern_type: MarketMakerPatternType
    probability: float
    expected_duration: TimeDuration
    expected_outcome: PatternOutcome
    risk_level: str
    confidence_interval: Tuple[float, float]


@dataclass(frozen=True)
class MarketMakerSignal:
    """Сигнал маркет-мейкера."""

    timestamp: datetime
    symbol: Symbol
    signal_type: MarketMakerPatternType
    confidence: Confidence
    signal_strength: SignalStrength
    expected_price_change: float
    time_horizon: TimeDuration
    risk_reward_ratio: float
    context: PatternContext


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_confidence(value: float) -> Confidence:
    """Валидация значения уверенности."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {value}")
    return Confidence(value)


def validate_similarity_score(value: float) -> SimilarityScore:
    """Валидация значения схожести."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Similarity score must be between 0.0 and 1.0, got {value}")
    return SimilarityScore(value)


def validate_signal_strength(value: float) -> SignalStrength:
    """Валидация силы сигнала."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"Signal strength must be between -1.0 and 1.0, got {value}")
    return SignalStrength(value)


def validate_book_pressure(value: float) -> BookPressure:
    """Валидация давления в стакане."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"Book pressure must be between -1.0 and 1.0, got {value}")
    return BookPressure(value)


def validate_order_imbalance(value: float) -> OrderImbalance:
    """Валидация дисбаланса ордеров."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"Order imbalance must be between -1.0 and 1.0, got {value}")
    return OrderImbalance(value)


def validate_accuracy(value: float) -> Accuracy:
    """Валидация точности."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Accuracy must be between 0.0 and 1.0, got {value}")
    return Accuracy(value)


def validate_positive_int(value: int, name: str) -> int:
    """Валидация положительного целого числа."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative_float(value: float, name: str) -> float:
    """Валидация неотрицательного float."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
