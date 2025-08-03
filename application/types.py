"""
Промышленные типы для application слоя с полной интеграцией domain типов.
Обеспечивает строгую типизацию для всех use cases, сервисов и протоколов.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4

from domain.entities.market import MarketData

# Domain entities
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.position import Position
from domain.entities.signal import Signal
from domain.entities.trading import Trade

# Domain types
from domain.types import (
    ConfidenceLevel,
    MetadataDict,
    OrderId,
    PortfolioId,
    PositionId,
    PriceValue,
    SignalId,
    StrategyId,
    Symbol,
    TimestampValue,
    TradeId,
    VolumeValue,
)
from domain.entities.trading_pair import TradingPair
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage
from domain.value_objects.price import Price

# Domain value objects
from domain.value_objects.volume import Volume

# ============================================================================
# STRICT TYPING DEFINITIONS
# ============================================================================

# Базовые типы для совместимости с тестами
class MarketSummary:
    """Сводка рынка."""
    pass

class PriceLevel:
    """Уровень цены."""
    pass

class VolumeLevel:
    """Уровень объема."""
    pass

class MoneyAmount:
    """Денежная сумма."""
    pass

class Timestamp:
    """Временная метка."""
    pass

# Новые типы для строгой типизации
ParameterValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ParameterDict = Dict[str, ParameterValue]
RiskMetricsDict = Dict[str, Union[float, Decimal, str]]
PerformanceMetricsDict = Dict[str, Union[float, Decimal, str, int]]
TechnicalIndicatorsDict = Dict[str, Union[float, Decimal, Optional[float]]]
VolumeProfileDict = Dict[str, Union[List[Dict[str, float]], float, str]]
OrderBookLevel = TypedDict(
    "OrderBookLevel", {"price": float, "quantity": float, "total": float}
)
OrderBookData = TypedDict(
    "OrderBookData",
    {"bids": List[OrderBookLevel], "asks": List[OrderBookLevel], "timestamp": str},
)
MarketAnalysisData = TypedDict(
    "MarketAnalysisData",
    {
        "trend": str,
        "support_levels": List[float],
        "resistance_levels": List[float],
        "volatility": float,
        "volume_profile": VolumeProfileDict,
        "technical_indicators": TechnicalIndicatorsDict,
        "sentiment_score": float,
    },
)
NotificationData = TypedDict(
    "NotificationData",
    {
        "title": str,
        "message": str,
        "level": str,
        "type": str,
        "priority": str,
        "channels": List[str],
        "recipients": List[str],
    },
)


# ============================================================================
# APPLICATION LAYER ENUMS
# ============================================================================
class MarketPhase(str, Enum):
    """Фазы рынка для application слоя."""

    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"


class SignalStrength(str, Enum):
    """Сила сигнала для application слоя."""

    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREME = "extreme"


class SignalType(str, Enum):
    """Типы сигналов для application слоя."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    REVERSE = "reverse"


class RiskLevel(str, Enum):
    """Уровни риска для application слоя."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    CRITICAL = "critical"


class PositionStatus(str, Enum):
    """Статусы позиций для application слоя."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    PENDING = "pending"
    LIQUIDATED = "liquidated"


class NotificationLevel(str, Enum):
    """Уровни уведомлений."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"


class NotificationType(str, Enum):
    """Типы уведомлений."""

    SYSTEM = "system"
    TRADE = "trade"
    ORDER = "order"
    POSITION = "position"
    RISK = "risk"
    ALERT = "alert"
    PERFORMANCE = "performance"


class NotificationPriority(str, Enum):
    """Приоритеты уведомлений."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class NotificationChannel(str, Enum):
    """Каналы уведомлений."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"


class SessionStatus(str, Enum):
    """Статусы торговых сессий."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class StrategyExecutionStatus(str, Enum):
    """Статусы выполнения стратегий."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# BASE REQUEST/RESPONSE TYPES
# ============================================================================
@dataclass
class BaseRequest:
    """Базовый класс для всех запросов application слоя."""

    request_id: UUID = field(default_factory=uuid4)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass
class BaseResponse:
    """Базовый класс для всех ответов application слоя."""

    success: bool = True
    message: str = ""
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass
class PaginatedRequest(BaseRequest):
    """Базовый класс для пагинированных запросов."""

    limit: int = 100
    offset: int = 0
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "desc"


@dataclass
class PaginatedResponse(BaseResponse):
    """Базовый класс для пагинированных ответов."""

    total_count: int = 0
    has_more: bool = False
    page_info: Dict[str, Union[int, bool, str]] = field(default_factory=dict)


# ============================================================================
# ORDER MANAGEMENT TYPES
# ============================================================================
@dataclass(kw_only=True)
class CreateOrderRequest(BaseRequest):
    """Запрос на создание ордера."""

    portfolio_id: PortfolioId
    trading_pair: TradingPair
    order_type: OrderType
    side: OrderSide
    volume: Volume
    strategy_id: Optional[StrategyId] = None
    signal_id: Optional[SignalId] = None
    price: Optional[Price] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None


@dataclass(kw_only=True)
class CreateOrderResponse(BaseResponse):
    """Ответ с созданным ордером."""

    order_id: Optional[OrderId] = None
    exchange_order_id: Optional[str] = None
    order: Optional[Order] = None
    estimated_cost: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


@dataclass(kw_only=True)
class CancelOrderRequest(BaseRequest):
    """Запрос на отмену ордера."""

    order_id: OrderId
    portfolio_id: PortfolioId
    reason: Optional[str] = None


@dataclass(kw_only=True)
class CancelOrderResponse(BaseResponse):
    """Ответ на отмену ордера."""

    order: Optional[Order] = None
    cancellation_time: Optional[TimestampValue] = None
    cancelled: bool = False


@dataclass(kw_only=True)
class GetOrdersRequest(PaginatedRequest):
    """Запрос на получение ордеров."""

    portfolio_id: PortfolioId
    trading_pair: Optional[TradingPair] = None
    status: Optional[OrderStatus] = None
    side: Optional[OrderSide] = None
    order_type: Optional[OrderType] = None
    start_time: Optional[TimestampValue] = None
    end_time: Optional[TimestampValue] = None


@dataclass(kw_only=True)
class GetOrdersResponse(PaginatedResponse):
    """Ответ с ордерами."""

    orders: List[Order] = field(default_factory=list)
    total_value: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


# ============================================================================
# POSITION MANAGEMENT TYPES
# ============================================================================
@dataclass(kw_only=True)
class CreatePositionRequest(BaseRequest):
    """Запрос на создание позиции."""

    portfolio_id: PortfolioId
    trading_pair: TradingPair
    volume: Volume
    entry_price: Price
    side: OrderSide
    strategy_id: Optional[StrategyId] = None


@dataclass(kw_only=True)
class CreatePositionResponse(BaseResponse):
    """Ответ с созданной позицией."""

    position_id: Optional[PositionId] = None
    position: Optional[Position] = None


@dataclass(kw_only=True)
class UpdatePositionRequest(BaseRequest):
    """Запрос на обновление позиции."""

    position_id: PositionId
    portfolio_id: PortfolioId
    volume: Optional[Volume] = None
    price: Optional[Price] = None


@dataclass(kw_only=True)
class UpdatePositionResponse(BaseResponse):
    """Ответ на обновление позиции."""

    position: Optional[Position] = None
    changes: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    updated: bool = False


@dataclass(kw_only=True)
class ClosePositionRequest(BaseRequest):
    """Запрос на закрытие позиции."""

    position_id: PositionId
    portfolio_id: PortfolioId
    volume: Optional[Volume] = None
    price: Optional[Price] = None
    reason: Optional[str] = None


@dataclass(kw_only=True)
class ClosePositionResponse(BaseResponse):
    """Ответ на закрытие позиции."""

    close_price: Optional[Price] = None
    closed: bool = False
    closed_volume: Volume = field(
        default_factory=lambda: Volume(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


@dataclass(kw_only=True)
class GetPositionsRequest(PaginatedRequest):
    """Запрос на получение позиций."""

    portfolio_id: PortfolioId
    trading_pair: Optional[TradingPair] = None
    min_size: Optional[VolumeValue] = None
    status: Optional[PositionStatus] = None
    side: Optional[OrderSide] = None


@dataclass(kw_only=True)
class GetPositionsResponse(PaginatedResponse):
    """Ответ с позициями."""

    positions: List[Position] = field(default_factory=list)
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


@dataclass(kw_only=True)
class PositionMetrics:
    """Метрики позиции."""

    position_id: PositionId
    trading_pair: TradingPair
    side: OrderSide
    volume: Volume
    entry_price: PriceValue
    pnl_percentage: Decimal
    notional_value: Money
    current_price: Optional[PriceValue] = None
    unrealized_pnl: Optional[Money] = None
    realized_pnl: Optional[Money] = None
    total_pnl: Optional[Money] = None
    is_open: bool = True
    created_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    updated_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    days_held: int = 0
    avg_entry_price: Optional[PriceValue] = None
    current_value: Optional[Money] = None
    margin_used: Optional[Money] = None
    leverage: Decimal = Decimal("1")
    liquidation_price: Optional[PriceValue] = None
    risk_metrics: RiskMetricsDict = field(default_factory=dict)


# ============================================================================
# RISK MANAGEMENT TYPES
# ============================================================================
@dataclass(kw_only=True)
class RiskAssessmentRequest(BaseRequest):
    """Запрос на оценку риска."""

    portfolio_id: PortfolioId
    positions: List[Position]
    market_data: Dict[Symbol, MarketData]
    risk_tolerance: Decimal = Decimal("0.05")
    time_horizon: str = "1d"
    confidence_level: Decimal = Decimal("0.95")


@dataclass(kw_only=True)
class RiskAssessmentResponse(BaseResponse):
    """Ответ с оценкой риска."""

    portfolio_risk: RiskMetricsDict = field(default_factory=dict)
    position_risks: List[RiskMetricsDict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: Decimal = Decimal("0")
    is_acceptable: bool = True
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    var_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    max_drawdown: Decimal = Decimal("0")


@dataclass(kw_only=True)
class RiskLimitRequest(BaseRequest):
    """Запрос на установку лимитов риска."""

    portfolio_id: PortfolioId
    max_var_95: Decimal
    max_var_99: Decimal
    max_drawdown: Decimal
    max_position_size: Decimal
    max_correlation: Decimal
    max_leverage: Decimal = Decimal("10")


@dataclass(kw_only=True)
class RiskLimitResponse(BaseResponse):
    """Ответ с лимитами риска."""

    current_risk: RiskMetricsDict = field(default_factory=dict)
    risk_limits: RiskMetricsDict = field(default_factory=dict)
    limits_set: bool = False


# ============================================================================
# TRADING PAIR TYPES
# ============================================================================
@dataclass(kw_only=True)
class CreateTradingPairRequest(BaseRequest):
    """Запрос на создание торговой пары."""

    base_currency: str = ""
    quote_currency: str = ""
    exchange: str = ""
    min_amount: Optional[VolumeValue] = None
    max_amount: Optional[VolumeValue] = None
    tick_size: Optional[PriceValue] = None
    step_size: Optional[VolumeValue] = None
    commission_rate: Optional[Decimal] = None
    is_active: bool = True


@dataclass(kw_only=True)
class CreateTradingPairResponse(BaseResponse):
    """Ответ с созданной торговой парой."""

    trading_pair: Optional[TradingPair] = None


@dataclass(kw_only=True)
class UpdateTradingPairRequest(BaseRequest):
    """Запрос на обновление торговой пары."""

    pair_id: str = ""
    min_amount: Optional[VolumeValue] = None
    max_amount: Optional[VolumeValue] = None
    tick_size: Optional[PriceValue] = None
    step_size: Optional[VolumeValue] = None
    commission_rate: Optional[Decimal] = None
    is_active: Optional[bool] = None


@dataclass(kw_only=True)
class UpdateTradingPairResponse(BaseResponse):
    """Ответ на обновление торговой пары."""

    trading_pair: Optional[TradingPair] = None
    updated: bool = False


@dataclass(kw_only=True)
class GetTradingPairsRequest(PaginatedRequest):
    """Запрос на получение торговых пар."""

    exchange: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None


@dataclass(kw_only=True)
class GetTradingPairsResponse(PaginatedResponse):
    """Ответ с торговыми парами."""

    trading_pairs: List[TradingPair] = field(default_factory=list)


@dataclass(kw_only=True)
class TradingPairMetrics:
    """Метрики торговой пары."""

    volume_24h: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))
    price_change_24h: PriceValue = field(
        default_factory=lambda: PriceValue(Decimal("0"))
    )
    price_change_percent_24h: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
    high_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    low_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    last_price: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    bid: Optional[PriceValue] = None
    ask: Optional[PriceValue] = None
    spread: Optional[PriceValue] = None
    volatility: Decimal = Decimal("0")
    market_cap: Optional[Money] = None
    circulating_supply: Optional[VolumeValue] = None


# ============================================================================
# STRATEGY EXECUTION TYPES
# ============================================================================
@dataclass(kw_only=True)
class ExecuteStrategyRequest(BaseRequest):
    """Запрос на выполнение стратегии."""

    strategy_id: StrategyId
    portfolio_id: PortfolioId
    symbol: Symbol
    amount: Optional[VolumeValue] = None
    risk_level: Optional[RiskLevel] = None
    use_sentiment_analysis: bool = True
    parameters: ParameterDict = field(default_factory=dict)


@dataclass(kw_only=True)
class ExecuteStrategyResponse(BaseResponse):
    """Ответ на выполнение стратегии."""

    orders_created: List[Order] = field(default_factory=list)
    signals_generated: List[Signal] = field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    execution_time_ms: float = 0.0
    executed: bool = False


@dataclass(kw_only=True)
class ProcessSignalRequest(BaseRequest):
    """Запрос на обработку сигнала."""

    signal: Signal
    portfolio_id: PortfolioId
    auto_execute: bool = True
    use_sentiment_analysis: bool = True
    risk_validation: bool = True


@dataclass(kw_only=True)
class ProcessSignalResponse(BaseResponse):
    """Ответ на обработку сигнала."""

    orders_created: List[Order] = field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    risk_assessment: Optional[RiskMetricsDict] = None
    processed: bool = False


# ============================================================================
# PORTFOLIO MANAGEMENT TYPES
# ============================================================================
@dataclass(kw_only=True)
class PortfolioRebalanceRequest(BaseRequest):
    """Запрос на ребалансировку портфеля."""

    portfolio_id: PortfolioId
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    tolerance: Decimal = Decimal("0.05")
    use_sentiment_analysis: bool = True
    rebalance_strategy: str = "optimal"


@dataclass(kw_only=True)
class PortfolioRebalanceResponse(BaseResponse):
    """Ответ на ребалансировку портфеля."""

    orders_created: List[Order] = field(default_factory=list)
    current_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    rebalance_cost: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    rebalanced: bool = False


# ============================================================================
# SESSION MANAGEMENT TYPES
# ============================================================================
@dataclass(kw_only=True)
class TradingSession:
    """Сессия торговли."""

    session_id: str = ""
    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: Optional[TimestampValue] = None
    orders_created: List[Order] = field(default_factory=list)
    trades_executed: List[Trade] = field(default_factory=list)
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    status: SessionStatus = SessionStatus.ACTIVE
    strategy_id: Optional[StrategyId] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class SessionMetrics:
    """Метрики торговой сессии."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    avg_trade_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    max_drawdown: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    sharpe_ratio: Decimal = Decimal("0")
    total_volume: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))


# ============================================================================
# CONFIGURATION TYPES
# ============================================================================
@dataclass(kw_only=True)
class ServiceConfig:
    """Базовая конфигурация сервиса."""

    enabled: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    log_level: str = "INFO"
    max_workers: int = 8
    cache_enabled: bool = True


@dataclass(kw_only=True)
class CacheConfig:
    """Конфигурация кэша."""

    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 1000
    cleanup_interval: int = 60
    eviction_policy: str = "lru"


@dataclass(kw_only=True)
class PerformanceConfig:
    """Конфигурация производительности."""

    enable_monitoring: bool = True
    enable_profiling: bool = False
    metrics_interval: int = 60
    max_workers: int = 8
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80


# ============================================================================
# MARKET ANALYSIS TYPES
# ============================================================================
@dataclass(kw_only=True)
class MarketAnalysis:
    """Результат анализа рынка."""

    symbol: Symbol
    phase: MarketPhase
    trend: str
    support_levels: List[PriceValue]
    resistance_levels: List[PriceValue]
    volatility: Decimal
    volume_profile: VolumeProfileDict
    technical_indicators: TechnicalIndicatorsDict
    sentiment_score: Decimal
    confidence: ConfidenceLevel
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )


@dataclass(kw_only=True)
class TradingSignal:
    """Торговый сигнал."""

    signal_id: SignalId
    symbol: Symbol
    signal_type: SignalType
    strength: SignalStrength
    price: PriceValue
    timestamp: TimestampValue
    confidence: ConfidenceLevel
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class PortfolioMetrics:
    """Метрики портфеля."""

    total_value: Money
    total_pnl: Money
    pnl_percentage: Percentage
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    var_95: Decimal
    var_99: Decimal
    beta: Decimal
    alpha: Decimal
    correlation_matrix: Dict[Symbol, Dict[Symbol, Decimal]]
    sector_allocation: Dict[str, Money]
    risk_metrics: RiskMetricsDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )


@dataclass(kw_only=True)
class TechnicalIndicators:
    """Технические индикаторы."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_middle: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class VolumeProfile:
    """Профиль объема."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    poc_price: Optional[PriceValue] = None  # Point of Control
    value_areas: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    volume_nodes: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class SupportResistanceLevels:
    """Уровни поддержки и сопротивления."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
    strength_scores: Dict[str, Decimal] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class MarketRegime:
    """Рыночный режим."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    regime_type: str = ""  # "trending", "ranging", "volatile", "quiet"
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    volatility: Decimal = Decimal("0")
    trend_strength: Decimal = Decimal("0")
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class OrderBookSnapshot:
    """Снимок ордербука."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    spread: Optional[PriceValue] = None
    depth: Optional[Dict[str, Union[float, int]]] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


# ============================================================================
# PORTFOLIO SUMMARY TYPES
# ============================================================================
@dataclass(kw_only=True)
class PortfolioSummary:
    """Сводка портфеля."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    available_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    total_equity: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_used: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_level: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    open_positions: int = 0
    open_orders: int = 0
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class RiskMetrics:
    """Метрики риска."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    beta: Decimal = Decimal("0")
    correlation: Decimal = Decimal("0")
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class PerformanceMetrics:
    """Метрики производительности."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    annualized_return: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
    volatility: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    average_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class StrategyPerformance:
    """Производительность стратегии."""

    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


# ============================================================================
# ML AND AI TYPES
# ============================================================================
@dataclass(kw_only=True)
class MLPrediction:
    """ML предсказание."""

    model_id: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    prediction_type: str = ""
    predicted_value: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    features: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class NewsItem:
    """Новостной элемент."""

    news_id: str = ""
    title: str = ""
    content: str = ""
    source: str = ""
    published_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    sentiment_score: Decimal = Decimal("0")
    relevance_score: Decimal = Decimal("0")
    url: Optional[str] = None
    symbols: List[Symbol] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class SocialSentiment:
    """Социальный сентимент."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    sentiment_score: Decimal = Decimal("0")
    fear_greed_index: Decimal = Decimal("0")
    posts_count: int = 0
    positive_posts: int = 0
    negative_posts: int = 0
    neutral_posts: int = 0
    trending_topics: List[str] = field(default_factory=list)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class PatternDetection:
    """Обнаружение паттерна."""

    pattern_id: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    pattern_type: str = ""
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    price_levels: Dict[str, PriceValue] = field(default_factory=dict)
    volume_profile: Optional[VolumeProfileDict] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class EntanglementResult:
    """Результат анализа запутанности."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    exchange_pair: str = ""
    is_entangled: bool = False
    correlation_score: Decimal = Decimal("0")
    lag_ms: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class MirrorSignal:
    """Зеркальный сигнал."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    mirror_symbol: Symbol = field(default_factory=lambda: Symbol(""))
    correlation: Decimal = Decimal("0")
    lag: int = 0
    signal_strength: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class SessionInfluence:
    """Влияние сессии."""

    session_type: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    influence_score: Decimal = Decimal("0")
    volatility_impact: Decimal = Decimal("0")
    volume_impact: Decimal = Decimal("0")
    momentum_impact: Decimal = Decimal("0")
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class LiquidityGravityResult:
    """Результат анализа гравитации ликвидности."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    gravity_score: Decimal = Decimal("0")
    liquidity_score: Decimal = Decimal("0")
    volatility_score: Decimal = Decimal("0")
    anomaly_detected: bool = False
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class EvolutionResult:
    """Результат эволюции стратегии."""

    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    generation: int = 0
    fitness_score: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    parameters: ParameterDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class SymbolSelectionResult:
    """Результат выбора символов."""

    selected_symbols: List[Symbol]
    opportunity_scores: Dict[Symbol, Decimal]
    confidence_scores: Dict[Symbol, ConfidenceLevel]
    market_phases: Dict[Symbol, MarketPhase]
    processing_time_ms: float = 0.0
    total_symbols_analyzed: int = 0
    cache_hit_rate: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    detailed_profiles: Dict[Symbol, Dict[str, Union[str, float, int, bool]]] = field(
        default_factory=dict
    )
    rejection_reasons: Dict[Symbol, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)


# ============================================================================
# NOTIFICATION TYPES
# ============================================================================
@dataclass(kw_only=True)
class NotificationTemplate:
    """Шаблон уведомления."""

    name: str
    title: str
    message: str
    level: NotificationLevel
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
class NotificationConfig:
    """Конфигурация уведомлений."""

    email_enabled: bool = True
    email_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    sms_enabled: bool = False
    sms_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    push_enabled: bool = False
    push_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    telegram_enabled: bool = False
    telegram_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    default_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_queue_size: int = 1000
    batch_size: int = 10
    batch_timeout: float = 5.0
