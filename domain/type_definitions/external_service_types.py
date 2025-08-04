"""
Типы для внешних сервисов - Production Ready
Обеспечивает строгую типизацию для всех внешних сервисов и их интеграций.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)

from domain.type_definitions import (
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
    """Типы поддерживаемых бирж."""

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
    """Статусы подключения к внешним сервисам."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class OrderStatus(Enum):
    """Статусы ордеров."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Стороны ордеров."""

    BUY = "buy"
    SELL = "sell"


class TimeFrame(Enum):
    """Временные интервалы."""

    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
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


class MLModelType(Enum):
    """Типы ML моделей."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class PredictionType(Enum):
    """Типы предсказаний."""

    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    TREND = "trend"
    REVERSAL = "reversal"
    SUPPORT_RESISTANCE = "support_resistance"
    LIQUIDITY = "liquidity"
    SPREAD = "spread"


class RiskLevel(Enum):
    """Уровни риска."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class MarketRegime(Enum):
    """Рыночные режимы."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


# ============================================================================
# NEW TYPES
# ============================================================================
# Базовые типы
ExchangeName = NewType("ExchangeName", str)
APIKey = NewType("APIKey", str)
APISecret = NewType("APISecret", str)
APIPassphrase = NewType("APIPassphrase", str)
WebSocketURL = NewType("WebSocketURL", str)
RESTURL = NewType("RESTURL", str)
# Типы для соединений
ConnectionTimeout = NewType("ConnectionTimeout", float)
ReconnectAttempts = NewType("ReconnectAttempts", int)
RateLimit = NewType("RateLimit", int)
RateLimitWindow = NewType("RateLimitWindow", int)
# Типы для запросов
RequestTimeout = NewType("RequestTimeout", float)
RequestPriority = NewType("RequestPriority", int)
RequestMethod = NewType("RequestMethod", str)
# Типы для ответов
ResponseCode = NewType("ResponseCode", int)
ResponseStatus = NewType("ResponseStatus", str)
ResponseTime = NewType("ResponseTime", float)
# Типы для ML
ModelName = NewType("ModelName", str)
ModelVersion = NewType("ModelVersion", str)
ModelPath = NewType("ModelPath", str)
ModelSize = NewType("ModelSize", int)
FeatureName = NewType("FeatureName", str)
TargetName = NewType("TargetName", str)
# Типы для метрик
MetricName = NewType("MetricName", str)
MetricUnit = NewType("MetricUnit", str)
MetricTimestamp = NewType("MetricTimestamp", datetime)
# Типы для кэша
CacheKey = NewType("CacheKey", str)
CacheTTL = NewType("CacheTTL", int)
CacheSize = NewType("CacheSize", int)


# ============================================================================
# TYPED DICTS
# ============================================================================
class ExchangeConfig(TypedDict):
    """Конфигурация биржи."""

    exchange_name: ExchangeName
    api_key: APIKey
    api_secret: APISecret
    api_passphrase: Optional[APIPassphrase]
    testnet: bool
    sandbox: bool
    rate_limit: RateLimit
    timeout: ConnectionTimeout
    max_retries: ReconnectAttempts
    websocket_url: Optional[WebSocketURL]
    rest_url: Optional[RESTURL]


class MarketDataConfig(TypedDict):
    """Конфигурация рыночных данных."""

    symbol: Symbol
    timeframe: TimeFrame
    limit: int
    update_interval: int
    cache_ttl: CacheTTL
    websocket_enabled: bool
    rest_enabled: bool


class OrderConfig(TypedDict):
    """Конфигурация ордера."""

    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: VolumeValue
    price: Optional[PriceValue]
    stop_price: Optional[PriceValue]
    time_in_force: Literal["GTC", "IOC", "FOK"]
    post_only: bool
    reduce_only: bool
    close_on_trigger: bool


class MLModelConfig(TypedDict):
    """Конфигурация ML модели."""

    name: ModelName
    model_type: MLModelType
    trading_pair: Symbol
    prediction_type: PredictionType
    features: List[FeatureName]
    target: TargetName
    hyperparameters: Dict[str, Any]
    description: str


class RiskConfig(TypedDict):
    """Конфигурация рисков."""

    max_leverage: float
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    stop_loss_percentage: float
    take_profit_percentage: float
    risk_per_trade: float
    max_open_positions: int


# ============================================================================
# DATACLASSES
# ============================================================================
@dataclass(frozen=True)
class ExchangeCredentials:
    """Учетные данные биржи."""

    api_key: APIKey
    api_secret: APISecret
    api_passphrase: Optional[APIPassphrase] = None
    testnet: bool = False
    sandbox: bool = False


@dataclass(frozen=True)
class ConnectionConfig:
    """Конфигурация соединения."""

    timeout: ConnectionTimeout = ConnectionTimeout(30.0)
    max_retries: ReconnectAttempts = ReconnectAttempts(3)
    retry_delay: float = 1.0
    rate_limit: RateLimit = RateLimit(10)
    rate_limit_window: RateLimitWindow = RateLimitWindow(60)
    websocket_enabled: bool = True
    rest_enabled: bool = True


@dataclass(frozen=True)
class MarketDataRequest:
    """Запрос рыночных данных."""

    symbol: Symbol
    timeframe: TimeFrame
    limit: int = 100
    since: Optional[datetime] = None
    until: Optional[datetime] = None


@dataclass(frozen=True)
class OrderRequest:
    """Запрос на создание ордера."""

    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: VolumeValue
    price: Optional[PriceValue] = None
    stop_price: Optional[PriceValue] = None
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"
    post_only: bool = False
    reduce_only: bool = False
    close_on_trigger: bool = False
    portfolio_id: Optional[PortfolioId] = None
    strategy_id: Optional[StrategyId] = None
    signal_id: Optional[SignalId] = None


@dataclass(frozen=True)
class MLPredictionRequest:
    """Запрос на предсказание."""

    model_id: ModelId
    features: Dict[FeatureName, float]
    confidence_threshold: float = 0.7
    return_confidence: bool = True
    return_explanation: bool = False


@dataclass(frozen=True)
class RiskMetrics:
    """Метрики риска."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Метрики производительности."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# PROTOCOLS
# ============================================================================
@runtime_checkable
class ExchangeServiceProtocol(Protocol):
    """Протокол для сервисов бирж."""

    @property
    def exchange_name(self) -> ExchangeName:
        """Название биржи."""
        ...

    @property
    def connection_status(self) -> ConnectionStatus:
        """Статус подключения."""
        ...

    async def connect(self, credentials: ExchangeCredentials) -> bool:
        """Подключение к бирже."""
        ...

    async def disconnect(self) -> None:
        """Отключение от биржи."""
        ...

    async def get_market_data(self, request: MarketDataRequest) -> List[Dict[str, Any]]:
        """Получение рыночных данных."""
        ...

    async def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Размещение ордера."""
        ...

    async def cancel_order(self, order_id: OrderId) -> bool:
        """Отмена ордера."""
        ...

    async def get_order_status(self, order_id: OrderId) -> Dict[str, Any]:
        """Получение статуса ордера."""
        ...

    async def get_balance(self) -> Dict[str, float]:
        """Получение баланса."""
        ...

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение позиций."""
        ...


@runtime_checkable
class MLServiceProtocol(Protocol):
    """Протокол для ML сервисов."""

    async def train_model(
        self, config: MLModelConfig, training_data: Dict[str, Any]
    ) -> ModelId:
        """Обучение модели."""
        ...

    async def predict(self, request: MLPredictionRequest) -> Dict[str, Any]:
        """Выполнение предсказания."""
        ...

    async def evaluate_model(
        self, model_id: ModelId, test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Оценка модели."""
        ...

    async def save_model(self, model_id: ModelId, path: ModelPath) -> bool:
        """Сохранение модели."""
        ...

    async def load_model(self, model_id: ModelId, path: ModelPath) -> bool:
        """Загрузка модели."""
        ...

    async def get_model_status(self, model_id: ModelId) -> Dict[str, Any]:
        """Получение статуса модели."""
        ...


@runtime_checkable
class MetricsServiceProtocol(Protocol):
    """Протокол для сервисов метрик."""

    async def record_counter(
        self, name: MetricName, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Запись счетчика."""
        ...

    async def record_gauge(
        self, name: MetricName, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Запись gauge."""
        ...

    async def record_histogram(
        self, name: MetricName, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Запись гистограммы."""
        ...

    async def get_metric(
        self, name: MetricName, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Получение метрики."""
        ...


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Enums
    "ExchangeType",
    "ConnectionStatus",
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeFrame",
    "MLModelType",
    "PredictionType",
    "RiskLevel",
    "MarketRegime",
    # New Types
    "ExchangeName",
    "APIKey",
    "APISecret",
    "APIPassphrase",
    "WebSocketURL",
    "RESTURL",
    "ConnectionTimeout",
    "ReconnectAttempts",
    "RateLimit",
    "RateLimitWindow",
    "RequestTimeout",
    "RequestPriority",
    "RequestMethod",
    "ResponseCode",
    "ResponseStatus",
    "ResponseTime",
    "ModelName",
    "ModelVersion",
    "ModelPath",
    "ModelSize",
    "FeatureName",
    "TargetName",
    "MetricName",
    "MetricUnit",
    "MetricTimestamp",
    "CacheKey",
    "CacheTTL",
    "CacheSize",
    # TypedDicts
    "ExchangeConfig",
    "MarketDataConfig",
    "OrderConfig",
    "MLModelConfig",
    "RiskConfig",
    # Dataclasses
    "ExchangeCredentials",
    "ConnectionConfig",
    "MarketDataRequest",
    "OrderRequest",
    "MLPredictionRequest",
    "RiskMetrics",
    "PerformanceMetrics",
    # Protocols
    "ExchangeServiceProtocol",
    "MLServiceProtocol",
    "MetricsServiceProtocol",
]
