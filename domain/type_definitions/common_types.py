"""
Общие типы для устранения избыточного использования Any и Dict[str, Any].
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union
from uuid import UUID

# ============================================================================
# БАЗОВЫЕ ТИПЫ ДАННЫХ
# ============================================================================


class CacheValue(Protocol):
    """Протокол для значений кэша."""

    pass


ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
"""Тип для значений конфигурации."""


class ValidationResult(TypedDict):
    """Результат валидации."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


class OperationResult(TypedDict):
    """Результат операции."""

    success: bool
    message: str
    data: Optional[Any]
    error: Optional[str]


class MetricsData(TypedDict):
    """Данные метрик."""

    timestamp: datetime
    value: float
    unit: str
    tags: Dict[str, str]


class LogEntry(TypedDict):
    """Запись лога."""

    timestamp: datetime
    level: str
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, Any]


# ============================================================================
# РЫНОЧНЫЕ ДАННЫЕ
# ============================================================================


class MarketData(TypedDict):
    """Рыночные данные."""

    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]


class OrderBookLevel(TypedDict):
    """Уровень ордербука."""

    price: Decimal
    quantity: Decimal
    side: str


class OrderBook(TypedDict):
    """Ордербук."""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: Decimal
    depth: int


class TradeData(TypedDict):
    """Данные сделки."""

    id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str
    timestamp: datetime
    maker_order_id: Optional[str]
    taker_order_id: Optional[str]


# ============================================================================
# ТОРГОВЫЕ ТИПЫ
# ============================================================================


class OrderData(TypedDict):
    """Данные ордера."""

    id: str
    symbol: str
    side: str
    type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    timestamp: datetime
    client_order_id: Optional[str]


class PositionData(TypedDict):
    """Данные позиции."""

    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin: Decimal
    leverage: int


class BalanceData(TypedDict):
    """Данные баланса."""

    currency: str
    available: Decimal
    locked: Decimal
    total: Decimal


# ============================================================================
# АНАЛИТИЧЕСКИЕ ТИПЫ
# ============================================================================


class TechnicalIndicator(TypedDict):
    """Технический индикатор."""

    name: str
    value: float
    timestamp: datetime
    parameters: Dict[str, Any]


class PatternData(TypedDict):
    """Данные паттерна."""

    name: str
    confidence: float
    start_time: datetime
    end_time: datetime
    parameters: Dict[str, Any]


class SignalData(TypedDict):
    """Данные сигнала."""

    type: str
    strength: float
    direction: str
    timestamp: datetime
    metadata: Dict[str, Any]


class PredictionData(TypedDict):
    """Данные предсказания."""

    model_id: str
    target: str
    value: float
    confidence: float
    timestamp: datetime
    features: Dict[str, float]


# ============================================================================
# КОНФИГУРАЦИОННЫЕ ТИПЫ
# ============================================================================


class DatabaseConfig(TypedDict):
    """Конфигурация базы данных."""

    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    max_overflow: int


class CacheConfig(TypedDict):
    """Конфигурация кэша."""

    type: str
    host: Optional[str]
    port: Optional[int]
    ttl: int
    max_size: int
    eviction_policy: str


class ExchangeConfig(TypedDict):
    """Конфигурация биржи."""

    name: str
    api_key: str
    api_secret: str
    testnet: bool
    rate_limit: int


class StrategyConfig(TypedDict):
    """Конфигурация стратегии."""

    name: str
    enabled: bool
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]


# ============================================================================
# СИСТЕМНЫЕ ТИПЫ
# ============================================================================


class HealthStatus(TypedDict):
    """Статус здоровья."""

    status: str
    timestamp: datetime
    components: Dict[str, str]
    metrics: Dict[str, float]


class ErrorInfo(TypedDict):
    """Информация об ошибке."""

    code: str
    message: str
    details: Optional[str]
    timestamp: datetime
    context: Dict[str, Any]


class PerformanceMetrics(TypedDict):
    """Метрики производительности."""

    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: datetime


# ============================================================================
# ТИПЫ ДЛЯ РЕПОЗИТОРИЕВ
# ============================================================================


class QueryFilter(TypedDict):
    """Фильтр запроса."""

    field: str
    operator: str
    value: Any


class QueryOptions(TypedDict):
    """Опции запроса."""

    limit: Optional[int]
    offset: Optional[int]
    order_by: Optional[str]
    order_direction: Optional[str]
    filters: Optional[List[QueryFilter]]


class BulkOperationResult(TypedDict):
    """Результат массовой операции."""

    success_count: int
    error_count: int
    errors: List[Dict[str, Any]]
    processed_ids: List[str]


# ============================================================================
# ТИПЫ ДЛЯ АГЕНТОВ
# ============================================================================


class AgentContext(TypedDict):
    """Контекст агента."""

    agent_id: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    state: Dict[str, Any]


class AgentResponse(TypedDict):
    """Ответ агента."""

    success: bool
    data: Optional[Any]
    error: Optional[str]
    metadata: Dict[str, Any]


# ============================================================================
# УТИЛИТАРНЫЕ ТИПЫ
# ============================================================================

JsonData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
"""Тип для JSON данных."""

ConfigData = Dict[str, ConfigValue]
"""Тип для данных конфигурации."""

EntityId = Union[UUID, str]
"""Тип для ID сущности."""

Timestamp = Union[datetime, float, int]
"""Тип для временных меток."""

Amount = Union[Decimal, float, int]
"""Тип для денежных сумм."""

Percentage = float
"""Тип для процентов (0.0 - 1.0)."""

# ============================================================================
# ТИПЫ ДЛЯ ВАЛИДАЦИИ
# ============================================================================


class ValidationRule(TypedDict):
    """Правило валидации."""

    field: str
    rule_type: str
    parameters: Dict[str, Any]
    message: str


class ValidationContext(TypedDict):
    """Контекст валидации."""

    entity_type: str
    operation: str
    data: Dict[str, Any]
    user_id: Optional[str]


# ============================================================================
# ТИПЫ ДЛЯ КЭШИРОВАНИЯ
# ============================================================================


class CacheEntry(TypedDict):
    """Запись кэша."""

    key: str
    value: Any
    ttl: int
    created_at: datetime
    accessed_at: datetime


class CacheStats(TypedDict):
    """Статистика кэша."""

    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int
