"""
Улучшенные типы для замены избыточного использования Any.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, Union
from uuid import UUID

# ============================================================================
# БАЗОВЫЕ ТИПЫ
# ============================================================================

# Типы для значений, которые могут быть любыми
FlexibleValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
"""Гибкий тип для значений, которые могут быть разными типами."""

JsonValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
"""Тип для JSON значений."""

ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
"""Тип для значений конфигурации."""

CacheValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
"""Тип для значений кэша."""

# Типы для идентификаторов
EntityId = Union[UUID, str]
"""Тип для ID сущности."""

ResourceId = str
"""Тип для ID ресурса."""

TaskId = str
"""Тип для ID задачи."""

# Типы для временных меток
Timestamp = Union[datetime, float, int]
"""Тип для временных меток."""

# Типы для денежных сумм
Amount = Union[Decimal, float, int]
"""Тип для денежных сумм."""

Percentage = float
"""Тип для процентов (0.0 - 1.0)."""

# ============================================================================
# ТИПЫ ДЛЯ РЫНОЧНЫХ ДАННЫХ
# ============================================================================


class MarketDataType(Protocol):
    """Протокол для рыночных данных."""

    symbol: str
    price: Amount
    volume: Amount
    timestamp: Timestamp


class OrderBookDataType(Protocol):
    """Протокол для данных ордербука."""

    symbol: str
    timestamp: Timestamp
    bids: List[Dict[str, Amount]]
    asks: List[Dict[str, Amount]]


class TradeDataType(Protocol):
    """Протокол для данных сделки."""

    id: str
    symbol: str
    price: Amount
    quantity: Amount
    side: str
    timestamp: Timestamp


# ============================================================================
# ТИПЫ ДЛЯ ТОРГОВЫХ ОПЕРАЦИЙ
# ============================================================================


class OrderDataType(Protocol):
    """Протокол для данных ордера."""

    id: str
    symbol: str
    side: str
    type: str
    quantity: Amount
    price: Optional[Amount]
    status: str
    timestamp: Timestamp


class PositionDataType(Protocol):
    """Протокол для данных позиции."""

    symbol: str
    side: str
    quantity: Amount
    entry_price: Amount
    current_price: Amount
    unrealized_pnl: Amount
    realized_pnl: Amount


class BalanceDataType(Protocol):
    """Протокол для данных баланса."""

    currency: str
    available: Amount
    locked: Amount
    total: Amount


# ============================================================================
# ТИПЫ ДЛЯ АНАЛИТИКИ
# ============================================================================


class TechnicalIndicatorType(Protocol):
    """Протокол для технического индикатора."""

    name: str
    value: float
    timestamp: Timestamp
    parameters: Dict[str, FlexibleValue]


class PatternDataType(Protocol):
    """Протокол для данных паттерна."""

    name: str
    confidence: Percentage
    start_time: Timestamp
    end_time: Timestamp
    parameters: Dict[str, FlexibleValue]


class SignalDataType(Protocol):
    """Протокол для данных сигнала."""

    type: str
    strength: Percentage
    direction: str
    timestamp: Timestamp
    metadata: Dict[str, FlexibleValue]


class PredictionDataType(Protocol):
    """Протокол для данных предсказания."""

    model_id: str
    target: str
    value: float
    confidence: Percentage
    timestamp: Timestamp
    features: Dict[str, float]


# ============================================================================
# ТИПЫ ДЛЯ КОНФИГУРАЦИИ
# ============================================================================


class DatabaseConfigType(Protocol):
    """Протокол для конфигурации базы данных."""

    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    max_overflow: int


class CacheConfigType(Protocol):
    """Протокол для конфигурации кэша."""

    type: str
    host: Optional[str]
    port: Optional[int]
    ttl: int
    max_size: int
    eviction_policy: str


class ExchangeConfigType(Protocol):
    """Протокол для конфигурации биржи."""

    name: str
    api_key: str
    api_secret: str
    testnet: bool
    rate_limit: int


class StrategyConfigType(Protocol):
    """Протокол для конфигурации стратегии."""

    name: str
    enabled: bool
    parameters: Dict[str, FlexibleValue]
    risk_limits: Dict[str, float]


# ============================================================================
# ТИПЫ ДЛЯ СИСТЕМНЫХ ОПЕРАЦИЙ
# ============================================================================


class HealthStatusType(Protocol):
    """Протокол для статуса здоровья."""

    status: str
    timestamp: Timestamp
    components: Dict[str, str]
    metrics: Dict[str, float]


class ErrorInfoType(Protocol):
    """Протокол для информации об ошибке."""

    code: str
    message: str
    details: Optional[str]
    timestamp: Timestamp
    context: Dict[str, FlexibleValue]


class PerformanceMetricsType(Protocol):
    """Протокол для метрик производительности."""

    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: Timestamp


# ============================================================================
# ТИПЫ ДЛЯ РЕПОЗИТОРИЕВ
# ============================================================================


class QueryFilterType(Protocol):
    """Протокол для фильтра запроса."""

    field: str
    operator: str
    value: FlexibleValue


class QueryOptionsType(Protocol):
    """Протокол для опций запроса."""

    limit: Optional[int]
    offset: Optional[int]
    order_by: Optional[str]
    order_direction: Optional[str]
    filters: Optional[List[QueryFilterType]]


class BulkOperationResultType(Protocol):
    """Протокол для результата массовой операции."""

    success_count: int
    error_count: int
    errors: List[Dict[str, FlexibleValue]]
    processed_ids: List[str]


# ============================================================================
# ТИПЫ ДЛЯ АГЕНТОВ
# ============================================================================


class AgentContextType(Protocol):
    """Протокол для контекста агента."""

    agent_id: str
    session_id: str
    timestamp: Timestamp
    data: Dict[str, FlexibleValue]
    state: Dict[str, FlexibleValue]


class AgentResponseType(Protocol):
    """Протокол для ответа агента."""

    success: bool
    data: Optional[FlexibleValue]
    error: Optional[str]
    metadata: Dict[str, FlexibleValue]


# ============================================================================
# ТИПЫ ДЛЯ ВАЛИДАЦИИ
# ============================================================================


class ValidationRuleType(Protocol):
    """Протокол для правила валидации."""

    field: str
    rule_type: str
    parameters: Dict[str, FlexibleValue]
    message: str


class ValidationContextType(Protocol):
    """Протокол для контекста валидации."""

    entity_type: str
    operation: str
    data: Dict[str, FlexibleValue]
    user_id: Optional[str]


# ============================================================================
# ТИПЫ ДЛЯ КЭШИРОВАНИЯ
# ============================================================================


class CacheEntryType(Protocol):
    """Протокол для записи кэша."""

    key: str
    value: CacheValue
    ttl: int
    created_at: Timestamp
    accessed_at: Timestamp


class CacheStatsType(Protocol):
    """Протокол для статистики кэша."""

    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int


# ============================================================================
# ТИПЫ ДЛЯ СОБЫТИЙ
# ============================================================================


class EventDataType(Protocol):
    """Протокол для данных события."""

    event_type: str
    timestamp: Timestamp
    source: str
    data: Dict[str, FlexibleValue]
    metadata: Dict[str, FlexibleValue]


class EventHandlerType(Protocol):
    """Протокол для обработчика событий."""

    async def handle(self, event: EventDataType) -> None: ...


# ============================================================================
# ТИПЫ ДЛЯ СТРАТЕГИЙ
# ============================================================================


class StrategyDataType(Protocol):
    """Протокол для данных стратегии."""

    id: EntityId
    name: str
    type: str
    parameters: Dict[str, FlexibleValue]
    status: str
    performance_metrics: Dict[str, float]


class StrategyResultType(Protocol):
    """Протокол для результата стратегии."""

    strategy_id: EntityId
    timestamp: Timestamp
    action: str
    confidence: Percentage
    metadata: Dict[str, FlexibleValue]


# ============================================================================
# ТИПЫ ДЛЯ МАШИННОГО ОБУЧЕНИЯ
# ============================================================================


class ModelDataType(Protocol):
    """Протокол для данных модели."""

    id: EntityId
    name: str
    type: str
    version: str
    parameters: Dict[str, FlexibleValue]
    performance: Dict[str, float]


class TrainingDataType(Protocol):
    """Протокол для данных обучения."""

    model_id: EntityId
    features: List[float]
    target: float
    timestamp: Timestamp
    metadata: Dict[str, FlexibleValue]


class PredictionResultType(Protocol):
    """Протокол для результата предсказания."""

    model_id: EntityId
    prediction: float
    confidence: Percentage
    features: Dict[str, float]
    timestamp: Timestamp


# ============================================================================
# ТИПЫ ДЛЯ МОНИТОРИНГА
# ============================================================================


class MetricDataType(Protocol):
    """Протокол для данных метрики."""

    name: str
    value: float
    unit: str
    timestamp: Timestamp
    tags: Dict[str, str]


class AlertDataType(Protocol):
    """Протокол для данных алерта."""

    id: EntityId
    type: str
    severity: str
    message: str
    timestamp: Timestamp
    metadata: Dict[str, FlexibleValue]


class LogEntryType(Protocol):
    """Протокол для записи лога."""

    timestamp: Timestamp
    level: str
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, FlexibleValue]


# ============================================================================
# ТИПЫ ДЛЯ БЕЗОПАСНОСТИ
# ============================================================================


class SecurityEventType(Protocol):
    """Протокол для события безопасности."""

    event_type: str
    severity: str
    source_ip: str
    user_id: Optional[str]
    timestamp: Timestamp
    details: Dict[str, FlexibleValue]


class AuthenticationDataType(Protocol):
    """Протокол для данных аутентификации."""

    user_id: str
    method: str
    success: bool
    timestamp: Timestamp
    metadata: Dict[str, FlexibleValue]


# ============================================================================
# УТИЛИТАРНЫЕ ТИПЫ
# ============================================================================

# Типы для функций
AsyncFunction = TypeVar("AsyncFunction", bound=Callable[..., Any])
"""Тип для асинхронных функций."""

SyncFunction = TypeVar("SyncFunction", bound=Callable[..., Any])
"""Тип для синхронных функций."""

# Типы для итераторов
AsyncIteratorType = TypeVar("AsyncIteratorType")
"""Тип для асинхронных итераторов."""

SyncIteratorType = TypeVar("SyncIteratorType")
"""Тип для синхронных итераторов."""

# Типы для контекстных менеджеров
AsyncContextManager = TypeVar("AsyncContextManager")
"""Тип для асинхронных контекстных менеджеров."""

SyncContextManager = TypeVar("SyncContextManager")
"""Тип для синхронных контекстных менеджеров."""

# ============================================================================
# ТИПЫ ДЛЯ РЕЗУЛЬТАТОВ ОПЕРАЦИЙ
# ============================================================================


@dataclass
class OperationResult:
    """Результат операции."""

    success: bool
    message: str
    data: Optional[FlexibleValue] = None
    error: Optional[str] = None
    timestamp: Timestamp = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Результат валидации."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: Timestamp = field(default_factory=datetime.now)


@dataclass
class ProcessingResult:
    """Результат обработки."""

    success: bool
    processed_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    data: Optional[FlexibleValue] = None
    timestamp: Timestamp = field(default_factory=datetime.now)
