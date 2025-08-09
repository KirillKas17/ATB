"""
Типы для доменных сущностей.
"""

from datetime import datetime
from decimal import Decimal
from typing import (
    Any,
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
from uuid import UUID

# Новые типы для строгой типизации
StrategyId = NewType("StrategyId", UUID)
PortfolioId = NewType("PortfolioId", UUID)
OrderId = NewType("OrderId", UUID)
PositionId = NewType("PositionId", UUID)
SignalId = NewType("SignalId", UUID)
TradeId = NewType("TradeId", UUID)
AccountId = NewType("AccountId", UUID)
MarketId = NewType("MarketId", UUID)
ModelId = NewType("ModelId", UUID)
PredictionId = NewType("PredictionId", UUID)
RiskProfileId = NewType("RiskProfileId", UUID)
EntityId = NewType("EntityId", UUID)  # Универсальный тип для всех ID
# Типы для торговых пар
Symbol = NewType("Symbol", str)
TradingPair = NewType("TradingPair", str)
MarketName = NewType("MarketName", str)
ExchangeName = NewType("ExchangeName", str)
# Типы для цен и объемов
PriceValue = NewType("PriceValue", Decimal)
VolumeValue = NewType("VolumeValue", Decimal)
AmountValue = NewType("AmountValue", Decimal)
MoneyValue = NewType("MoneyValue", Decimal)
# Типы для метрик
ConfidenceLevel = NewType("ConfidenceLevel", Decimal)
RiskLevel = NewType("RiskLevel", Decimal)
PerformanceScore = NewType("PerformanceScore", Decimal)
VolatilityValue = NewType("VolatilityValue", Decimal)
# Типы для рыночных данных
TimeframeValue = NewType("TimeframeValue", str)
MarketRegimeValue = NewType("MarketRegimeValue", str)
TrendStrengthValue = NewType("TrendStrengthValue", Decimal)
VolumeTrendValue = NewType("VolumeTrendValue", Decimal)
PriceMomentumValue = NewType("PriceMomentumValue", Decimal)
RSIMetric = NewType("RSIMetric", Decimal)
MACDMetric = NewType("MACDMetric", Decimal)
ATRMetric = NewType("ATRMetric", Decimal)
TimestampValue = NewType("TimestampValue", datetime)
MetadataDict = NewType("MetadataDict", Dict[str, Any])
PricePrecision = NewType("PricePrecision", int)
VolumePrecision = NewType("VolumePrecision", int)
# Типы для статусов
OrderStatusType = Literal[
    "pending", "open", "partially_filled", "filled", "cancelled", "rejected", "expired"
]
PositionStatusType = Literal["open", "closed", "partial"]
StrategyStatusType = Literal["active", "paused", "stopped", "error", "inactive"]
SignalTypeType = Literal["buy", "sell", "hold", "close", "strong_buy", "strong_sell"]
TradingSessionStatusType = Literal["active", "paused", "stopped", "completed"]
# Типы для торговых операций
OrderSideType = Literal["buy", "sell"]
OrderTypeType = Literal[
    "market", "limit", "stop", "stop_limit", "take_profit", "stop_loss"
]
TimeInForceType = Literal["GTC", "IOC", "FOK", "GTX"]


# Типизированные словари
class StrategyConfig(TypedDict, total=False):
    """Конфигурация стратегии."""

    name: str
    description: str
    strategy_type: str
    trading_pairs: List[str]
    parameters: Dict[str, Union[str, int, float, bool]]
    risk_level: str
    max_position_size: float
    stop_loss: float
    take_profit: float
    confidence_threshold: float
    max_signals: int
    signal_cooldown: int


class MarketDataConfig(TypedDict, total=False):
    """Конфигурация рыночных данных."""

    symbol: str
    timeframe: str
    limit: int
    include_volume: bool
    include_trades: bool


class OHLCVData(TypedDict):
    """Данные OHLCV (Open, High, Low, Close, Volume)."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


class OrderRequest(TypedDict, total=False):
    """Запрос на создание ордера."""

    symbol: str
    side: str
    order_type: str
    quantity: str
    price: str
    stop_price: str
    time_in_force: str
    post_only: bool
    reduce_only: bool


class PositionUpdate(TypedDict, total=False):
    """Обновление позиции."""

    current_price: str
    unrealized_pnl: str
    margin_used: str
    leverage: str


class SignalMetadata(TypedDict, total=False):
    """Метаданные сигнала."""

    strategy_type: str
    confidence: str
    risk_level: str
    market_conditions: Dict[str, str]
    technical_indicators: Dict[str, str]
    fundamental_factors: Dict[str, str]


class PerformanceMetrics(TypedDict, total=False):
    """Метрики производительности."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: str
    profit_factor: str
    sharpe_ratio: str
    max_drawdown: str
    total_pnl: str
    average_trade: str


class RiskMetrics(TypedDict, total=False):
    """Метрики риска."""

    volatility: str
    var_95: str
    cvar_95: str
    max_drawdown: str
    beta: str
    correlation: str
    exposure: str


class TradingSessionConfig(TypedDict, total=False):
    """Конфигурация торговой сессии."""

    session_id: str
    portfolio_id: str
    strategy_id: str
    start_time: str
    end_time: Optional[str]
    status: TradingSessionStatusType
    max_orders: int
    max_positions: int
    risk_limits: Dict[str, float]


# Протоколы для интерфейсов
@runtime_checkable
class StrategyProtocol(Protocol):
    """Протокол для стратегий."""

    def generate_signals(
        self, market_data: "MarketDataProtocol"
    ) -> List["SignalProtocol"]:
        """Генерация сигналов."""
        ...

    def validate_data(self, data: "MarketDataProtocol") -> bool:
        """Валидация данных."""
        ...

    def get_parameters(self) -> StrategyConfig:
        """Получение параметров."""
        ...

    def update_parameters(self, parameters: StrategyConfig) -> None:
        """Обновление параметров."""
        ...

    def is_active(self) -> bool:
        """Проверка активности."""
        ...


@runtime_checkable
class MarketDataProtocol(Protocol):
    """Протокол для рыночных данных."""

    def get_price(self) -> PriceValue:
        """Получение цены."""
        ...

    def get_volume(self) -> VolumeValue:
        """Получение объема."""
        ...

    def get_timestamp(self) -> TimestampValue:
        """Получение временной метки."""
        ...


@runtime_checkable
class SignalProtocol(Protocol):
    """Протокол для сигналов."""

    def get_signal_type(self) -> SignalTypeType:
        """Получение типа сигнала."""
        ...

    def get_confidence(self) -> ConfidenceLevel:
        """Получение уверенности."""
        ...

    def get_price(self) -> PriceValue:
        """Получение цены."""
        ...


@runtime_checkable
class OrderProtocol(Protocol):
    """Протокол для ордеров."""

    def get_status(self) -> OrderStatusType:
        """Получение статуса."""
        ...

    def get_quantity(self) -> VolumeValue:
        """Получение количества."""
        ...

    def get_price(self) -> PriceValue:
        """Получение цены."""
        ...


@runtime_checkable
class PositionProtocol(Protocol):
    """Протокол для позиций."""

    def get_side(self) -> OrderSideType:
        """Получение стороны."""
        ...

    def get_volume(self) -> VolumeValue:
        """Получение объема."""
        ...

    def get_pnl(self) -> MoneyValue:
        """Получение P&L."""
        ...


@runtime_checkable
class TradingSessionProtocol(Protocol):
    """Протокол для торговых сессий."""

    def get_status(self) -> TradingSessionStatusType:
        """Получение статуса сессии."""
        ...

    def get_portfolio_id(self) -> PortfolioId:
        """Получение ID портфеля."""
        ...

    def get_strategy_id(self) -> StrategyId:
        """Получение ID стратегии."""
        ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Протокол для риск-менеджера."""

    def validate_order(
        self,
        portfolio_id: PortfolioId,
        symbol: Symbol,
        side: OrderSideType,
        quantity: VolumeValue,
        price: Optional[PriceValue],
    ) -> "RiskValidationResult":
        """Валидация ордера."""
        ...

    def calculate_position_risk(self, position: PositionProtocol) -> RiskMetrics:
        """Расчет риска позиции."""
        ...

    def get_portfolio_risk(self, portfolio_id: PortfolioId) -> RiskMetrics:
        """Получение риска портфеля."""
        ...


class RiskValidationResult(TypedDict):
    """Результат валидации риска."""

    is_valid: bool
    reason: Optional[str]
    risk_score: float
    recommendations: List[str]


# Утилитарные функции для работы с типами
def create_entity_id(uuid_value: UUID) -> EntityId:
    """Создание EntityId из UUID."""
    return EntityId(uuid_value)


def create_portfolio_id(uuid_value: UUID) -> PortfolioId:
    """Создание PortfolioId из UUID."""
    return PortfolioId(uuid_value)


def create_order_id(uuid_value: UUID) -> OrderId:
    """Создание OrderId из UUID."""
    return OrderId(uuid_value)


def create_trade_id(uuid_value: UUID) -> TradeId:
    """Создание TradeId из UUID."""
    return TradeId(uuid_value)


def create_strategy_id(uuid_value: UUID) -> StrategyId:
    """Создание StrategyId из UUID."""
    return StrategyId(uuid_value)


def create_symbol(symbol_str: str) -> Symbol:
    """Создание Symbol из строки."""
    return Symbol(symbol_str.upper())


def create_trading_pair(pair_str: str) -> TradingPair:
    """Создание TradingPair из строки."""
    return TradingPair(pair_str.upper())


def create_price_value(decimal_value: Decimal) -> PriceValue:
    """Создание PriceValue из Decimal."""
    return PriceValue(decimal_value)


def create_volume_value(decimal_value: Decimal) -> VolumeValue:
    """Создание VolumeValue из Decimal."""
    return VolumeValue(decimal_value)


def create_timestamp_value(datetime_value: datetime) -> TimestampValue:
    """Создание TimestampValue из datetime."""
    return TimestampValue(datetime_value)


# Экспорты типов из evolution_types.py
from .evolution_types import (
    EvolutionConfig,  # NewType; Literal; Final constants; TypedDict; Protocol; Enum; Dataclass
)
from .evolution_types import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITE_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    MAX_DRAWDOWN_THRESHOLD,
    MIN_ACCURACY_THRESHOLD,
    MIN_PROFITABILITY_THRESHOLD,
    MIN_SHARPE_THRESHOLD,
    AccuracyScore,
    ComplexityScore,
    ConsistencyScore,
    CrossoverStrategy,
    CrossoverType,
    DiversityScore,
    EntryCondition,
    EvaluationStatus,
    EvolutionMetrics,
    EvolutionOrchestratorProtocol,
    EvolutionPhase,
    ExitCondition,
    FilterParameters,
    FitnessComponent,
    FitnessEvaluatorProtocol,
    FitnessScore,
    FitnessWeights,
    IndicatorParameters,
    MutationStrategy,
    MutationType,
    OptimizationMethod,
    OptimizationResult,
    ProfitabilityScore,
    RiskScore,
    SelectionMethod,
    SelectionStatistics,
    SelectionStrategy,
    StrategyGeneratorProtocol,
    StrategyOptimizerProtocol,
    StrategyPerformance,
    StrategySelectorProtocol,
    TradePosition,
)

# Экспорты типов из symbol_types.py
from .symbol_types import (  # NewType; Типы данных; Enum; TypedDict; Protocol; Dataclass; Exceptions
    ATRValue,
    ConfidenceValue,
    ConfigurationError,
    DataInsufficientError,
    EntropyValue,
    MarketDataError,
    MarketDataFrame,
    MarketDataValidator,
    MarketPhase,
    MarketPhaseClassifierProtocol,
    MarketPhaseResult,
    MomentumValue,
    OpportunityScoreCalculatorProtocol,
    OpportunityScoreConfig,
    OpportunityScoreConfigData,
    OpportunityScoreResult,
    OpportunityScoreValue,
    OrderBookData,
    OrderBookError,
    OrderBookMetrics,
    OrderBookSymmetry,
    PatternConfidenceValue,
    PatternMemoryData,
    PatternMetrics,
    PhaseDetectionConfig,
    PhaseDetectionConfigData,
    PriceStructure,
    PriceStructureMetrics,
    SessionAlignmentValue,
    SessionData,
    SessionMetrics,
    SpreadValue,
    SymbolAnalysisEvent,
    SymbolAnalysisMetrics,
    SymbolId,
    SymbolProfileCache,
    SymbolProfileProtocol,
    ValidationError,
    VolumeProfileMetrics,
    VolumeTrend,
    VWAPValue,
)

from .infrastructure_types import PositionSide

# Экспорт типов из messaging_types
from .messaging_types import (
    Event,
    EventPriority,
    EventType,
    Message,
    MessagePriority,
    EventMetadata,
    HandlerConfig,
    EventHandlerInfo,
    MessageHandlerInfo,
    WebSocketMessage,
    WebSocketCommand,
    WebSocketResponse,
    EventBusProtocol,
    MessageQueueProtocol,
    WebSocketServiceProtocol,
    EventBusConfig,
    MessageQueueConfig,
    WebSocketServiceConfig,
    EventBusMetrics,
    MessageQueueMetrics,
    WebSocketServiceMetrics,
    MessagingError,
    EventBusError,
    MessageQueueError,
    WebSocketServiceError,
    HandlerError,
    ConnectionError,
    TimeoutError,
    RetryExhaustedError,
    create_event,
    create_message,
    create_handler_config,
)

# Экспорт типов из monitoring_types
from .monitoring_types import (
    MetricType,
    AlertSeverity,
    Metric,
    Alert,
    TraceSpan,
    LogLevel,
    LogContext,
    LogEntry,
    MetricCollectorProtocol,
    AlertHandlerProtocol,
    TraceProtocol,
    LoggerProtocol,
)

# Экспорт уникальных типов из infrastructure_types (без дублирующихся)
from .infrastructure_types import (
    SystemMetrics,
    PortfolioMetrics,
    StrategyMetrics,
    SystemState,
    MarketState,
    Event as InfrastructureEvent,
    EventType as InfrastructureEventType,
    EventPriority as InfrastructureEventPriority,
    EvolvableProtocol,
    MonitorProtocol,
    ControllerProtocol,
    StrategyError,
    PortfolioError,
    RiskError,
    SystemError,
    DatabaseError,
    ExchangeError,
    validate_market_data,
    validate_strategy_config,
)
# Если нужна альтернативная версия RiskMetrics:
# from .infrastructure_types import RiskMetrics as InfraRiskMetrics
# Если нужна альтернативная версия StrategyProtocol:
# from .infrastructure_types import StrategyProtocol as InfraStrategyProtocol
# Если нужна альтернативная версия EvolutionError:
# from .infrastructure_types import EvolutionError as InfraEvolutionError

# Экспорт типов из common_types (без дублирующихся)
from .common_types import (
    HealthStatus,
    ValidationResult,
    OperationResult,
    MetricsData,
    LogEntry as CommonLogEntry,
    MarketData as CommonMarketData,
    OrderBook,
    TradeData as CommonTradeData,
    OrderData,
    PositionData,
    BalanceData,
    TechnicalIndicator,
    PatternData,
    SignalData,
    PredictionData,
    DatabaseConfig,
    CacheConfig,
    ExchangeConfig,
    StrategyConfig as CommonStrategyConfig,
    ErrorInfo,
    QueryFilter,
    QueryOptions,
    BulkOperationResult,
    AgentContext,
    AgentResponse,
    ValidationRule,
    ValidationContext,
    CacheEntry,
    CacheStats,
)

# Экспорт типов из entity_system_types
from .entity_system_types import (
    EntityState,
    CodeStructure,
    AnalysisResult,
    Hypothesis,
    Experiment,
    Improvement,
    MemorySnapshot,
    EvolutionStats,
    EntitySystemConfig,
    OperationMode,
    OptimizationLevel,
    SystemPhase,
    ImprovementCategory,
    EntityControllerProtocol,
    CodeScannerProtocol,
    CodeAnalyzerProtocol,
    ExperimentRunnerProtocol,
    ImprovementApplierProtocol,
    MemoryManagerProtocol,
    AIEnhancementProtocol,
    EvolutionEngineProtocol,
    EntitySystemError,
    CodeAnalysisError,
    ExperimentError,
    ImprovementError,
    MemoryError,
    AIEnhancementError,
    EvolutionError,
)

__all__ = [
    # Базовые типы
    "StrategyId",
    "PortfolioId",
    "OrderId",
    "PositionId",
    "SignalId",
    "TradeId",
    "AccountId",
    "MarketId",
    "Symbol",
    "TradingPair",
    "MarketName",
    "PriceValue",
    "VolumeValue",
    "AmountValue",
    "ConfidenceLevel",
    "RiskLevel",
    "PerformanceScore",
    "TimeframeValue",
    "MarketRegimeValue",
    "VolatilityValue",
    "TrendStrengthValue",
    "VolumeTrendValue",
    "PriceMomentumValue",
    "RSIMetric",
    "MACDMetric",
    "ATRMetric",
    "TimestampValue",
    "MetadataDict",
    "PricePrecision",
    "VolumePrecision",
    # Статусы
    "OrderStatusType",
    "PositionStatusType",
    "StrategyStatusType",
    "SignalTypeType",
    "TradingSessionStatusType",
    # Типы операций
    "OrderSideType",
    "OrderTypeType",
    "TimeInForceType",
    # TypedDict
    "StrategyConfig",
    "MarketDataConfig",
    "OHLCVData",
    "OrderRequest",
    "PositionUpdate",
    "SignalMetadata",
    "PerformanceMetrics",
    "RiskMetrics",
    "TradingSessionConfig",
    # Протоколы
    "StrategyProtocol",
    "MarketDataProtocol",
    "SignalProtocol",
    "OrderProtocol",
    "PositionProtocol",
    "TradingSessionProtocol",
    "RiskManagerProtocol",
    # Evolution types
    "FitnessScore",
    "AccuracyScore",
    "ProfitabilityScore",
    "RiskScore",
    "ConsistencyScore",
    "DiversityScore",
    "ComplexityScore",
    "OptimizationMethod",
    "SelectionMethod",
    "MutationType",
    "CrossoverType",
    "EvaluationStatus",
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_GENERATIONS",
    "DEFAULT_MUTATION_RATE",
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_ELITE_SIZE",
    "MIN_ACCURACY_THRESHOLD",
    "MIN_PROFITABILITY_THRESHOLD",
    "MAX_DRAWDOWN_THRESHOLD",
    "MIN_SHARPE_THRESHOLD",
    "IndicatorParameters",
    "FilterParameters",
    "EntryCondition",
    "ExitCondition",
    "TradePosition",
    "OptimizationResult",
    "SelectionStatistics",
    "EvolutionMetrics",
    "StrategyPerformance",
    "EvolutionConfigDict",
    "FitnessEvaluatorProtocol",
    "StrategyGeneratorProtocol",
    "StrategyOptimizerProtocol",
    "StrategySelectorProtocol",
    "EvolutionOrchestratorProtocol",
    "EvolutionPhase",
    "FitnessComponent",
    "MutationStrategy",
    "CrossoverStrategy",
    "SelectionStrategy",
    "EvolutionConfig",
    "FitnessWeights",
    # Symbol types
    "SymbolId",
    "OpportunityScoreValue",
    "ConfidenceValue",
    "ATRValue",
    "VWAPValue",
    "SpreadValue",
    "EntropyValue",
    "MomentumValue",
    "PatternConfidenceValue",
    "SessionAlignmentValue",
    "MarketDataFrame",
    "OrderBookData",
    "PatternMemoryData",
    "SessionData",
    "MarketPhase",
    "VolumeTrend",
    "PriceStructure",
    "OrderBookSymmetry",
    "PhaseDetectionConfig",
    "OpportunityScoreConfig",
    "VolumeProfileMetrics",
    "PriceStructureMetrics",
    "OrderBookMetrics",
    "PatternMetrics",
    "SessionMetrics",
    "MarketPhaseResult",
    "OpportunityScoreResult",
    "SymbolAnalysisMetrics",
    "SymbolAnalysisEvent",
    "MarketPhaseClassifierProtocol",
    "OpportunityScoreCalculatorProtocol",
    "SymbolProfileProtocol",
    "MarketDataValidator",
    "SymbolProfileCache",
    "PhaseDetectionConfigData",
    "OpportunityScoreConfigData",
    "ValidationError",
    "ConfigurationError",
    "DataInsufficientError",
    "MarketDataError",
    "OrderBookError",
    # Messaging types
    "Event",
    "EventPriority",
    "EventType",
    "Message",
    "MessagePriority",
    "EventMetadata",
    "HandlerConfig",
    "EventHandlerInfo",
    "MessageHandlerInfo",
    "WebSocketMessage",
    "WebSocketCommand",
    "WebSocketResponse",
    "EventBusProtocol",
    "MessageQueueProtocol",
    "WebSocketServiceProtocol",
    "EventBusConfig",
    "MessageQueueConfig",
    "WebSocketServiceConfig",
    "EventBusMetrics",
    "MessageQueueMetrics",
    "WebSocketServiceMetrics",
    "MessagingError",
    "EventBusError",
    "MessageQueueError",
    "WebSocketServiceError",
    "HandlerError",
    "ConnectionError",
    "TimeoutError",
    "RetryExhaustedError",
    "create_event",
    "create_message",
    "create_handler_config",
    # Monitoring types
    "MetricType",
    "AlertSeverity",
    "Metric",
    "Alert",
    "TraceSpan",
    "LogLevel",
    "LogContext",
    "LogEntry",
    "MetricCollectorProtocol",
    "AlertHandlerProtocol",
    "TraceProtocol",
    "LoggerProtocol",
    # Infrastructure types
    "SystemMetrics",
    "PortfolioMetrics",
    "StrategyMetrics",
    "SystemState",
    "MarketState",
    "Event as InfrastructureEvent",
    "EventType as InfrastructureEventType",
    "EventPriority as InfrastructureEventPriority",
    "EvolvableProtocol",
    "MonitorProtocol",
    "ControllerProtocol",
    "StrategyError",
    "PortfolioError",
    "RiskError",
    "SystemError",
    "DatabaseError",
    "ExchangeError",
    "validate_market_data",
    "validate_strategy_config",
    # Common types
    "HealthStatus",
    "ValidationResult",
    "OperationResult",
    "MetricsData",
    "LogEntry as CommonLogEntry",
    "MarketData as CommonMarketData",
    "OrderBook",
    "TradeData as CommonTradeData",
    "OrderData",
    "PositionData",
    "BalanceData",
    "TechnicalIndicator",
    "PatternData",
    "SignalData",
    "PredictionData",
    "DatabaseConfig",
    "CacheConfig",
    "ExchangeConfig",
    "StrategyConfig as CommonStrategyConfig",
    "ErrorInfo",
    "QueryFilter",
    "QueryOptions",
    "BulkOperationResult",
    "AgentContext",
    "AgentResponse",
    "ValidationRule",
    "ValidationContext",
    "CacheEntry",
    "CacheStats",
    # Entity system types
    "EntityState",
    "CodeStructure",
    "AnalysisResult",
    "Hypothesis",
    "Experiment",
    "Improvement",
    "MemorySnapshot",
    "EvolutionStats",
    "EntitySystemConfig",
    "OperationMode",
    "OptimizationLevel",
    "SystemPhase",
    "ImprovementCategory",
    "EntityControllerProtocol",
    "CodeScannerProtocol",
    "CodeAnalyzerProtocol",
    "ExperimentRunnerProtocol",
    "ImprovementApplierProtocol",
    "MemoryManagerProtocol",
    "AIEnhancementProtocol",
    "EvolutionEngineProtocol",
    "EntitySystemError",
    "CodeAnalysisError",
    "ExperimentError",
    "ImprovementError",
    "MemoryError",
    "AIEnhancementError",
    "EvolutionError",
]
