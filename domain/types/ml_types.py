from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)


class LiquidityZoneType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


class PredictionResult(TypedDict, total=False):
    predicted_spread: float
    confidence: float
    model_accuracy: float


class LiquidityPredictionResult(TypedDict, total=False):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_accuracy: float


class SpreadAnalysisResult(TypedDict, total=False):
    spread: float
    imbalance: float
    confidence: float
    best_bid: float
    best_ask: float


class SpreadMovementPrediction(TypedDict, total=False):
    prediction: float
    confidence: float
    ma_short: float
    ma_long: float


class MLModelPerformance(TypedDict, total=False):
    spread_accuracy: float
    liquidity_accuracy: float
    spread_loss: float
    liquidity_loss: float
    overall_performance: float


class LiquidityZone(TypedDict, total=False):
    price: float
    type: str  # support, resistance, neutral
    strength: float
    volume: float
    touches: int


class LiquiditySweep(TypedDict, total=False):
    timestamp: Any
    price: float
    type: Literal["sweep_high", "sweep_low"]
    confidence: float


class LiquidityAnalysisResult(TypedDict, total=False):
    liquidity_score: float
    confidence: float
    volume_score: float
    order_book_score: float
    volatility_score: float


# Дополнительные типы для тестов
class CorrelationResult(TypedDict, total=False):
    correlation: float
    lag: int
    strength: float


class CorrelationChainResult(TypedDict, total=False):
    chains: List[List[str]]
    correlations: List[float]
    summary: Dict[str, float]


class ModelPerformance(TypedDict, total=False):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float


class FeatureImportance(TypedDict, total=False):
    feature_name: str
    importance: float
    rank: int


class SignalResult(TypedDict, total=False):
    signal_type: str
    strength: float
    confidence: float
    timestamp: Any


class SignalStrength(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class StrategyResult(TypedDict, total=False):
    strategy_name: str
    performance: float
    risk_score: float
    status: str


class StrategyPerformance(TypedDict, total=False):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


class StrategyConfig(TypedDict, total=False):
    name: str
    type: str
    parameters: Dict[str, Any]
    risk_level: str


# Дополнительные типы для pattern_discovery
class PatternResult(TypedDict, total=False):
    pattern_type: str
    confidence: float
    support: float
    metadata: Dict[str, Any]


class PatternType(str, Enum):
    CANDLE = "candle"
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"


class PatternConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@runtime_checkable
class MLPredictorProtocol(Protocol):
    async def predict_spread(self, features: List[float]) -> PredictionResult: ...
    async def predict_liquidity(
        self, features: List[float]
    ) -> LiquidityPredictionResult: ...
    async def train_models(self, training_data: Any) -> bool: ...
    async def get_model_performance(self) -> MLModelPerformance: ...


@runtime_checkable
class SpreadAnalyzerProtocol(Protocol):
    async def analyze_spread(
        self, order_book: Dict[str, Any]
    ) -> SpreadAnalysisResult: ...
    async def calculate_imbalance(self, order_book: Dict[str, Any]) -> float: ...
    async def predict_spread_movement(
        self, historical_data: Any
    ) -> SpreadMovementPrediction: ...


@runtime_checkable
class ILiquidityAnalyzerProtocol(Protocol):
    async def analyze_liquidity(
        self, market_data: Any, order_book: Dict[str, Any]
    ) -> LiquidityAnalysisResult: ...
    async def identify_liquidity_zones(
        self, market_data: Any
    ) -> List[LiquidityZone]: ...
    async def detect_liquidity_sweeps(
        self, market_data: Any
    ) -> List[LiquiditySweep]: ...


# Дополнительные типы для market_metrics
class MarketMetricsResult(TypedDict, total=False):
    symbol: str
    timeframe: str
    current_price: str
    price_change_percent: str
    high_price: str
    low_price: str
    average_volume: str
    data_points: int
    last_update: str


class VolatilityMetrics(TypedDict, total=False):
    symbol: str
    volatility: float
    period: int


class TrendMetrics(TypedDict, total=False):
    symbol: str
    trend_strength: float
    trend_direction: str
    period: int


class ActionType(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    HOLD = "hold"


class DirectionType(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class SignalSource:
    name: str
    weight: float
    confidence: float
    signal_type: SignalType
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class AggregatedSignal:
    action: ActionType
    confidence: float
    risk_score: float
    sources: List[SignalSource]
    timestamp: datetime
    explanation: str


@dataclass(frozen=True)
class DecisionConfig:
    min_confidence: float = 0.7
    max_risk: float = 0.3
    min_samples: int = 100
    max_samples: int = 1000
    update_interval: int = 1
    cache_size: int = 1000
    compression: bool = True
    metrics_window: int = 24
    ensemble_size: int = 3
    feature_importance_threshold: float = 0.1
    decision_timeout: int = 5
    online_learning_rate: float = 0.01
    drift_detection_threshold: float = 0.1
    signal_aggregation_method: str = "weighted_voting"


@dataclass(frozen=True)
class DecisionMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confidence: float
    risk_score: float
    last_update: datetime
    samples_count: int
    decision_count: int
    error_count: int
    feature_importance: Dict[str, float]
    signal_quality: Dict[str, float]
    drift_score: float


@dataclass(frozen=True)
class DecisionReport:
    signal_type: str
    confidence: float
    timestamp: datetime
    features_importance: Dict[str, float]
    technical_indicators: Dict[str, float]
    market_context: Dict[str, Any]
    explanation: str
    visualization_path: str
    aggregated_signals: List[SignalSource]


@dataclass(frozen=True)
class TradeDecision:
    symbol: str
    action: ActionType
    direction: DirectionType
    volume: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
