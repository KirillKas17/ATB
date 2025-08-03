"""
Типы для промышленных протоколов домена.
Обеспечивают строгую типизацию для всех протоколов и интерфейсов.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)
from uuid import UUID

from domain.types import (
    ConfidenceLevel,
    MetadataDict,
    ModelId,
    OrderId,
    PortfolioId,
    PositionId,
    PredictionId,
    PriceValue,
    RiskProfileId,
    SignalId,
    StrategyId,
    Symbol,
    TimestampValue,
    TradeId,
    TradingPair,
    VolumeValue,
)

# ============================================================================
# ENUMS
# ============================================================================


class ExchangeType(Enum):
    """Типы бирж."""

    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    KUCOIN = "kucoin"
    GATE = "gate"
    MEXC = "mexc"
    HUOBI = "huobi"
    BITGET = "bitget"
    BINGX = "bingx"
    COINBASE = "coinbase"


class ConnectionStatus(Enum):
    """Статусы соединения."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class OrderBookDepth(Enum):
    """Глубина ордербука."""

    LEVEL_1 = 1
    LEVEL_5 = 5
    LEVEL_10 = 10
    LEVEL_20 = 20
    LEVEL_50 = 50
    LEVEL_100 = 100


class TimeframeType(Enum):
    """Типы временных интервалов."""

    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class ModelStatusType(Enum):
    """Статусы ML моделей."""

    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class StrategyStatusType(Enum):
    """Статусы стратегий."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    OPTIMIZING = "optimizing"


class SignalQualityType(Enum):
    """Качество сигналов."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ============================================================================
# TYPED DICTS
# ============================================================================


class ExchangeConfig(TypedDict):
    """Конфигурация биржи."""

    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str]  # Для некоторых бирж
    sandbox: Optional[bool]
    timeout: Optional[int]
    rate_limit: Optional[int]


class MarketDataRequest(TypedDict):
    """Запрос рыночных данных."""

    symbol: Symbol
    timeframe: TimeframeType
    limit: Optional[int]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class OrderRequest(TypedDict):
    """Запрос на создание ордера."""

    symbol: Symbol
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit", "take_profit"]
    quantity: VolumeValue
    price: Optional[PriceValue]
    stop_price: Optional[PriceValue]
    time_in_force: Optional[Literal["GTC", "IOC", "FOK"]]
    client_order_id: Optional[str]


class ModelTrainingRequest(TypedDict):
    """Запрос на обучение модели."""

    model_id: ModelId
    training_data_path: str
    validation_data_path: Optional[str]
    hyperparameters: Dict[str, Any]
    callbacks: Optional[List[str]]
    epochs: Optional[int]
    batch_size: Optional[int]


class StrategyExecutionRequest(TypedDict):
    """Запрос на исполнение стратегии."""

    strategy_id: StrategyId
    signal: Dict[str, Any]
    execution_params: Optional[Dict[str, Any]]
    risk_limits: Optional[Dict[str, Any]]


class RepositoryQuery(TypedDict):
    """Запрос к репозиторию."""

    filters: Optional[Dict[str, Any]]
    sort_by: Optional[str]
    sort_order: Optional[Literal["asc", "desc"]]
    limit: Optional[int]
    offset: Optional[int]
    include_deleted: Optional[bool]


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class ExchangeConnectionInfo:
    """Информация о соединении с биржей."""

    exchange_type: ExchangeType
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    latency_ms: Optional[float] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


@dataclass
class MarketDataSnapshot:
    """Снимок рыночных данных."""

    symbol: Symbol
    timestamp: TimestampValue
    open: PriceValue
    high: PriceValue
    low: PriceValue
    close: PriceValue
    volume: VolumeValue
    quote_volume: VolumeValue
    trade_count: int
    taker_buy_volume: VolumeValue
    taker_buy_quote_volume: VolumeValue


@dataclass
class OrderBookSnapshot:
    """Снимок ордербука."""

    symbol: Symbol
    timestamp: TimestampValue
    bids: List[Tuple[PriceValue, VolumeValue]]
    asks: List[Tuple[PriceValue, VolumeValue]]
    sequence_id: Optional[int] = None

    @property
    def best_bid(self) -> Optional[Tuple[PriceValue, VolumeValue]]:
        """Лучшая цена покупки."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Tuple[PriceValue, VolumeValue]]:
        """Лучшая цена продажи."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[PriceValue]:
        """Спред."""
        if self.best_bid and self.best_ask:
            return PriceValue(self.best_ask[0] - self.best_bid[0])
        return None


@dataclass
class ModelMetrics:
    """Метрики ML модели."""

    model_id: ModelId
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyMetrics:
    """Метрики стратегии."""

    strategy_id: StrategyId
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: Optional[float] = None
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RiskMetrics:
    """Метрики риска."""

    portfolio_id: PortfolioId
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    volatility: float
    beta: float
    correlation: float
    calculated_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# RESPONSE TYPES
# ============================================================================


class ExchangeResponse(TypedDict):
    """Ответ от биржи."""

    success: bool
    data: Any
    error_code: Optional[str]
    error_message: Optional[str]
    timestamp: datetime


class ModelTrainingResponse(TypedDict):
    """Ответ на обучение модели."""

    success: bool
    model_id: ModelId
    training_time: float
    metrics: ModelMetrics
    error_message: Optional[str]


class StrategyExecutionResponse(TypedDict):
    """Ответ на исполнение стратегии."""

    success: bool
    strategy_id: StrategyId
    order_id: Optional[OrderId]
    execution_time: float
    error_message: Optional[str]


class RepositoryResponse(TypedDict):
    """Ответ от репозитория."""

    success: bool
    data: Any
    total_count: Optional[int]
    error_message: Optional[str]


# ============================================================================
# ERROR TYPES
# ============================================================================


class ExchangeError(TypedDict):
    """Ошибка биржи."""

    error_code: str
    error_message: str
    exchange_type: ExchangeType
    timestamp: datetime
    retry_after: Optional[int]


class ModelError(TypedDict):
    """Ошибка модели."""

    error_code: str
    error_message: str
    model_id: ModelId
    timestamp: datetime
    context: Optional[Dict[str, Any]]


class StrategyError(TypedDict):
    """Ошибка стратегии."""

    error_code: str
    error_message: str
    strategy_id: StrategyId
    timestamp: datetime
    signal_data: Optional[Dict[str, Any]]


class RepositoryError(TypedDict):
    """Ошибка репозитория."""

    error_code: str
    error_message: str
    entity_type: str
    entity_id: Optional[str]
    timestamp: datetime


# ============================================================================
# CONFIGURATION TYPES
# ============================================================================


class ProtocolConfig(TypedDict):
    """Конфигурация протоколов."""

    exchange: ExchangeConfig
    retry_attempts: Optional[int]
    retry_delay: Optional[float]
    timeout: Optional[int]
    cache_ttl: Optional[int]
    batch_size: Optional[int]


class MonitoringConfig(TypedDict):
    """Конфигурация мониторинга."""

    enable_metrics: bool
    enable_logging: bool
    enable_alerts: bool
    metrics_interval: Optional[int]
    log_level: Optional[str]
    alert_thresholds: Optional[Dict[str, float]]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ExchangeType",
    "ConnectionStatus",
    "OrderBookDepth",
    "TimeframeType",
    "ModelStatusType",
    "StrategyStatusType",
    "SignalQualityType",
    # TypedDicts
    "ExchangeConfig",
    "MarketDataRequest",
    "OrderRequest",
    "ModelTrainingRequest",
    "StrategyExecutionRequest",
    "RepositoryQuery",
    # Dataclasses
    "ExchangeConnectionInfo",
    "MarketDataSnapshot",
    "OrderBookSnapshot",
    "ModelMetrics",
    "StrategyMetrics",
    "RiskMetrics",
    # Response types
    "ExchangeResponse",
    "ModelTrainingResponse",
    "StrategyExecutionResponse",
    "RepositoryResponse",
    # Error types
    "ExchangeError",
    "ModelError",
    "StrategyError",
    "RepositoryError",
    # Configuration types
    "ProtocolConfig",
    "MonitoringConfig",
]


# ============================================================================
# ОСНОВНЫЕ ТИПЫ ПРОТОКОЛОВ
# ============================================================================

# ID для протоколов
ProtocolId = NewType("ProtocolId", UUID)
ConnectionId = NewType("ConnectionId", UUID)
SessionId = NewType("SessionId", UUID)
RequestId = NewType("RequestId", UUID)
ResponseId = NewType("ResponseId", UUID)

# Типы для конфигурации
ConfigValue = Union[str, int, float, bool, Dict[str, Any]]
ConfigDict = Dict[str, ConfigValue]
EnvironmentType = str  # "development", "staging", "production"

# Типы для состояний
StateValue = str
StatusValue = str
HealthStatus = str  # "healthy", "degraded", "unhealthy"

# Типы для метрик
MetricValue = Union[int, float, Decimal]
MetricName = str
MetricUnit = str
MetricTimestamp = datetime

# ============================================================================
# ТИПЫ ДЛЯ БИРЖЕВЫХ ПРОТОКОЛОВ
# ============================================================================

# Типы для бирж
ExchangeName = NewType("ExchangeName", str)
ExchangeVersion = NewType("ExchangeVersion", str)
APIKey = NewType("APIKey", str)
APISecret = NewType("APISecret", str)
APIPassphrase = NewType("APIPassphrase", str)

# Типы для соединений
ConnectionType = str  # "websocket", "rest", "grpc"
ConnectionTimeout = int
ReconnectAttempts = int

# Типы для запросов
RequestTimeout = float
RequestPriority = int  # 1-10
RequestMethod = str  # "GET", "POST", "PUT", "DELETE"

# Типы для ответов
ResponseCode = int
ResponseStatus = str  # "success", "error", "timeout"
ResponseTime = float

# Типы для лимитов
RateLimit = int
RateLimitWindow = int
RateLimitRemaining = int
RateLimitReset = datetime

# ============================================================================
# ТИПЫ ДЛЯ ML ПРОТОКОЛОВ
# ============================================================================

# Типы для моделей
ModelName = str
ModelVersion = str
ModelPath = str
ModelSize = int  # в байтах
ModelFormat = str  # "pickle", "joblib", "onnx", "tensorflow"

# Типы для признаков
FeatureName = str
FeatureValue = Union[int, float, str, bool]
FeatureType = str  # "numerical", "categorical", "text"
FeatureImportance = float

# Типы для предсказаний
PredictionValue = Union[int, float, str]
PredictionProbability = float
PredictionTimestamp = datetime
PredictionMetadata = Dict[str, Any]

# Типы для обучения
TrainingEpoch = int
TrainingLoss = float
TrainingAccuracy = float
LearningRate = float
BatchSize = int

# Типы для валидации
ValidationScore = float
ValidationMetric = str
CrossValidationFold = int

# ============================================================================
# ТИПЫ ДЛЯ СТРАТЕГИЙ
# ============================================================================

# Типы для стратегий
StrategyName = str
StrategyVersion = str
StrategyAuthor = str
StrategyDescription = str

# Типы для сигналов
SignalValue = Union[int, float, str]
SignalStrength = float  # 0.0 - 1.0
SignalDirection = str  # "buy", "sell", "hold"
SignalSource = str  # "technical", "fundamental", "sentiment"

# Типы для исполнения
ExecutionStatus = str  # "pending", "executing", "completed", "failed"
ExecutionTime = datetime
ExecutionLatency = float  # в миллисекундах

# Типы для рисков
RiskScore = float  # 0.0 - 1.0
RiskMetric = str  # "var", "cvar", "volatility", "drawdown"

# ============================================================================
# ТИПЫ ДЛЯ РЕПОЗИТОРИЕВ
# ============================================================================

# Типы для запросов к БД
QueryString = str
QueryParams = Dict[str, Any]
QueryResult = List[Dict[str, Any]]
QueryCount = int

# Типы для фильтров
FilterField = str
FilterOperator = str  # "eq", "ne", "gt", "lt", "gte", "lte", "in", "like"
FilterValue = Any

# Типы для сортировки
SortField = str
SortDirection = str  # "asc", "desc"

# Типы для пагинации
PageNumber = int
PageSize = int
TotalPages = int
TotalRecords = int

# Типы для транзакций
TransactionId = UUID
TransactionStatus = str  # "active", "committed", "rolled_back"
TransactionTimestamp = datetime

# Типы для кэша
CacheKey = str
CacheValue = Any
CacheTTL = int  # в секундах
CacheHitRate = float

# ============================================================================
# ТИПИЗИРОВАННЫЕ СЛОВАРИ
# ============================================================================


class ExchangeConfigDict(TypedDict, total=False):
    """Конфигурация биржи."""

    exchange_name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str]
    sandbox: bool
    timeout: int
    retry_attempts: int
    rate_limit_delay: float
    websocket_enabled: bool
    heartbeat_interval: int


class ModelConfigDict(TypedDict, total=False):
    """Конфигурация ML модели."""

    name: str
    model_type: str
    trading_pair: str
    prediction_type: str
    hyperparameters: Dict[str, Any]
    features: List[str]
    target: str
    description: str
    version: str
    author: str
    tags: List[str]


class StrategyConfigDict(TypedDict, total=False):
    """Конфигурация стратегии."""

    name: str
    strategy_type: str
    trading_pairs: List[str]
    parameters: Dict[str, Any]
    risk_level: str
    max_position_size: str
    stop_loss: str
    take_profit: str
    confidence_threshold: str
    max_signals: int
    signal_cooldown: int
    description: str
    version: str
    author: str
    tags: List[str]


class RepositoryConfigDict(TypedDict, total=False):
    """Конфигурация репозитория."""

    connection_string: str
    pool_size: int
    timeout: float
    retry_attempts: int
    cache_enabled: bool
    cache_ttl: int
    cache_max_size: int


class QueryFilterDict(TypedDict, total=False):
    """Фильтр запроса."""

    field: str
    operator: str
    value: Any


class SortOrderDict(TypedDict, total=False):
    """Порядок сортировки."""

    field: str
    direction: str


class PaginationDict(TypedDict, total=False):
    """Пагинация."""

    page: int
    page_size: int
    offset: Optional[int]
    limit: Optional[int]


class PerformanceMetricsDict(TypedDict, total=False):
    """Метрики производительности."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: str
    profit_factor: str
    sharpe_ratio: str
    max_drawdown: str
    total_return: str
    average_trade: str
    calmar_ratio: str
    sortino_ratio: str
    var_95: str
    cvar_95: str


class RiskMetricsDict(TypedDict, total=False):
    """Метрики риска."""

    volatility: str
    var_95: str
    cvar_95: str
    max_drawdown: str
    beta: str
    correlation: str
    exposure: str


class HealthCheckDict(TypedDict, total=False):
    """Проверка здоровья."""

    status: str
    timestamp: str
    response_time: float
    error_count: int
    last_error: Optional[str]
    uptime: float


class ErrorInfoDict(TypedDict, total=False):
    """Информация об ошибке."""

    error_code: str
    error_message: str
    error_type: str
    timestamp: str
    context: Dict[str, Any]
    stack_trace: Optional[str]


# ============================================================================
# ПРОТОКОЛЫ ДЛЯ ИНТЕРФЕЙСОВ
# ============================================================================


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Протокол для конфигурируемых объектов."""

    def get_config(self) -> ConfigDict: ...
    def update_config(self, config: ConfigDict) -> None: ...
    def validate_config(self, config: ConfigDict) -> bool: ...


@runtime_checkable
class StatefulProtocol(Protocol):
    """Протокол для объектов с состоянием."""

    def get_state(self) -> StateValue: ...
    def set_state(self, state: StateValue) -> None: ...
    def is_healthy(self) -> bool: ...


@runtime_checkable
class MetricCollectorProtocol(Protocol):
    """Протокол для сбора метрик."""

    def collect_metrics(self) -> Dict[MetricName, MetricValue]: ...
    def get_metric(self, name: MetricName) -> Optional[MetricValue]: ...
    def reset_metrics(self) -> None: ...


@runtime_checkable
class ErrorHandlerProtocol(Protocol):
    """Протокол для обработки ошибок."""

    def handle_error(self, error: Exception) -> ErrorInfoDict: ...
    def get_error_history(self) -> List[ErrorInfoDict]: ...
    def clear_errors(self) -> None: ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Протокол для проверки здоровья."""

    def health_check(self) -> HealthCheckDict: ...
    def is_healthy(self) -> bool: ...
    def get_health_status(self) -> HealthStatus: ...


@runtime_checkable
class CacheableProtocol(Protocol):
    """Протокол для кэшируемых объектов."""

    def get_cache_key(self) -> CacheKey: ...
    def get_cache_value(self) -> CacheValue: ...
    def get_cache_ttl(self) -> CacheTTL: ...
    def is_cache_valid(self) -> bool: ...


@runtime_checkable
class TransactionalProtocol(Protocol):
    """Протокол для транзакционных объектов."""

    async def begin_transaction(self) -> TransactionId: ...
    async def commit_transaction(self, transaction_id: TransactionId) -> bool: ...
    async def rollback_transaction(self, transaction_id: TransactionId) -> bool: ...
    async def get_transaction_status(
        self, transaction_id: TransactionId
    ) -> TransactionStatus: ...


@runtime_checkable
class QueryableProtocol(Protocol):
    """Протокол для запрашиваемых объектов."""

    async def query(
        self, filters: List[QueryFilterDict], options: Optional[Dict[str, Any]] = None
    ) -> QueryResult: ...
    async def count(
        self, filters: Optional[List[QueryFilterDict]] = None
    ) -> QueryCount: ...
    async def exists(self, filters: List[QueryFilterDict]) -> bool: ...


@runtime_checkable
class PaginatableProtocol(Protocol):
    """Протокол для пагинируемых объектов."""

    async def get_page(self, pagination: PaginationDict) -> QueryResult: ...
    async def get_total_pages(self, page_size: PageSize) -> TotalPages: ...
    async def get_total_records(self) -> TotalRecords: ...


@runtime_checkable
class SortableProtocol(Protocol):
    """Протокол для сортируемых объектов."""

    async def sort(self, sort_orders: List[SortOrderDict]) -> QueryResult: ...
    async def get_sortable_fields(self) -> List[SortField]: ...


@runtime_checkable
class FilterableProtocol(Protocol):
    """Протокол для фильтруемых объектов."""

    async def filter(self, filters: List[QueryFilterDict]) -> QueryResult: ...
    async def get_filterable_fields(self) -> List[FilterField]: ...
    async def get_filter_operators(
        self, field: FilterField
    ) -> List[FilterOperator]: ...


# ============================================================================
# TYPED DICTS ДЛЯ ПРОТОКОЛОВ СТРАТЕГИЙ
# ============================================================================


class MarketAnalysisResult(TypedDict, total=False):
    """Результат комплексного анализа рынка."""

    indicators: Dict[str, float]
    patterns: List["PatternDetectionResult"]
    regime: str
    volatility: float
    support_levels: List[float]
    resistance_levels: List[float]
    momentum: Dict[str, float]
    meta: MetadataDict


class PatternDetectionResult(TypedDict, total=False):
    """Результат обнаружения рыночного паттерна."""

    pattern_type: str
    start_index: int
    end_index: int
    confidence: float
    direction: str
    meta: MetadataDict


class SignalFilterDict(TypedDict, total=False):
    """Фильтр для сигналов."""

    min_confidence: float
    max_risk: float
    allowed_types: List[str]
    custom_rules: MetadataDict


class StrategyAdaptationRules(TypedDict, total=False):
    """Правила адаптации стратегии к рыночному режиму."""

    regime: str
    parameter_overrides: Dict[str, float]
    risk_adjustments: Dict[str, float]
    notes: str


class StrategyErrorContext(TypedDict, total=False):
    """Контекст ошибки стратегии."""

    error_code: str
    error_message: str
    signal_id: Optional[SignalId]
    strategy_id: Optional[StrategyId]
    market_data_snapshot: Optional[Dict[str, float]]
    extra: MetadataDict


# ============================================================================
# TYPED DICTS ДЛЯ ML ПРОТОКОЛОВ
# ============================================================================


class ModelHyperparameters(TypedDict, total=False):
    """Гиперпараметры ML модели."""

    learning_rate: float
    batch_size: int
    epochs: int
    hidden_layers: list[int]
    dropout_rate: float
    regularization: float
    optimizer: str
    loss_function: str
    activation_function: str
    custom_params: dict[str, float]


class ModelFeatures(TypedDict, total=False):
    """Признаки для ML модели."""

    technical_indicators: dict[str, float]
    market_data: dict[str, float]
    sentiment_data: dict[str, float]
    external_factors: dict[str, float]
    metadata: MetadataDict


class ModelPrediction(TypedDict, total=False):
    """Предсказание ML модели."""

    prediction_value: float
    confidence: float
    prediction_type: str
    timestamp: str
    model_id: str
    features_used: list[str]
    metadata: MetadataDict


class ModelTrainingResult(TypedDict, total=False):
    """Результат обучения ML модели."""

    model_id: str
    training_time: float
    epochs_completed: int
    final_loss: float
    validation_metrics: dict[str, float]
    training_metrics: dict[str, float]
    model_path: str
    metadata: MetadataDict


class ModelOptimizationResult(TypedDict, total=False):
    """Результат оптимизации ML модели."""

    best_hyperparameters: ModelHyperparameters
    best_score: float
    optimization_time: float
    trials_completed: int
    convergence_rate: float
    improvement_history: list[float]
    metadata: MetadataDict


class ModelDriftResult(TypedDict, total=False):
    """Результат мониторинга дрифта ML модели."""

    drift_detected: bool
    drift_score: float
    drift_type: str
    affected_features: list[str]
    confidence: float
    recommendations: list[str]
    metadata: MetadataDict


class ModelAdaptationRules(TypedDict, total=False):
    """Правила адаптации ML модели."""

    market_regime: str
    adaptation_type: str
    parameter_adjustments: dict[str, float]
    retraining_threshold: float
    adaptation_strategy: str
    metadata: MetadataDict


class ModelPerformanceHistory(TypedDict, total=False):
    """История производительности ML модели."""

    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    metadata: MetadataDict
