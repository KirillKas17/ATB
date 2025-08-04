"""
Типы для мессенджинга системы.
Включает:
- Типы событий и сообщений
- Протоколы для EventBus, MessageQueue, WebSocketService
- Типы для обработчиков и подписчиков
- Метаданные и конфигурации
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Set,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field, validator

# ============================================================================
# Базовые типы
# ============================================================================
EventName = NewType("EventName", str)
TopicName = NewType("TopicName", str)
ConnectionID = NewType("ConnectionID", str)
CorrelationID = NewType("CorrelationID", str)
HandlerID = NewType("HandlerID", str)
MessageID = NewType("MessageID", str)


# ============================================================================
# Приоритеты
# ============================================================================
class EventPriority(Enum):
    """Приоритеты событий."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: 'EventPriority') -> bool:
        return self.value < other.value

    def __le__(self, other: 'EventPriority') -> bool:
        return self.value <= other.value

    def __gt__(self, other: 'EventPriority') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'EventPriority') -> bool:
        return self.value >= other.value


class MessagePriority(Enum):
    """Приоритеты сообщений."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

    def __lt__(self, other: 'MessagePriority') -> bool:
        return self.value < other.value

    def __le__(self, other: 'MessagePriority') -> bool:
        return self.value <= other.value

    def __gt__(self, other: 'MessagePriority') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'MessagePriority') -> bool:
        return self.value >= other.value


# ============================================================================
# Типы событий
# ============================================================================
class EventType(Enum):
    """Типы событий системы."""

    # Системные события
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    # Торговые события
    TRADE_EXECUTED = "trade.executed"
    ORDER_PLACED = "order.placed"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    ORDER_UPDATED = "order.updated"
    # События стратегий
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    STRATEGY_SIGNAL = "strategy.signal"
    STRATEGY_ERROR = "strategy.error"
    STRATEGY_PARAMETERS_UPDATED = "strategy.parameters_updated"
    STRATEGY_PERFORMANCE_UPDATED = "strategy.performance_updated"
    # События риска
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    RISK_ALERT = "risk.alert"
    RISK_EMERGENCY_STOP = "risk.emergency_stop"
    RISK_METRICS_UPDATED = "risk.metrics_updated"
    # События эволюции
    EVOLUTION_STARTED = "evolution.started"
    EVOLUTION_COMPLETED = "evolution.completed"
    EVOLUTION_ERROR = "evolution.error"
    EVOLUTION_ADAPTATION = "evolution.adaptation"
    EVOLUTION_LEARNING = "evolution.learning"
    # События портфеля
    PORTFOLIO_UPDATED = "portfolio.updated"
    PORTFOLIO_REBALANCED = "portfolio.rebalanced"
    PORTFOLIO_ALERT = "portfolio.alert"
    PORTFOLIO_POSITION_CHANGED = "portfolio.position_changed"
    # События рынка
    MARKET_DATA_UPDATED = "market.data_updated"
    MARKET_REGIME_CHANGED = "market.regime_changed"
    MARKET_VOLATILITY_CHANGED = "market.volatility_changed"
    MARKET_LIQUIDITY_CHANGED = "market.liquidity_changed"
    # События агентов
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_DECISION = "agent.decision"
    AGENT_ERROR = "agent.error"
    # События ML
    ML_MODEL_TRAINED = "ml.model_trained"
    ML_PREDICTION_MADE = "ml.prediction_made"
    ML_MODEL_UPDATED = "ml.model_updated"
    ML_ERROR = "ml.error"


# ============================================================================
# Базовые модели данных
# ============================================================================
@dataclass
class EventMetadata:
    """Метаданные события."""

    source: str
    correlation_id: Optional[CorrelationID] = None
    causation_id: Optional[MessageID] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


def _default_event_metadata() -> EventMetadata:
    """Создать метаданные события по умолчанию."""
    return EventMetadata(source="system")


@dataclass
class Event:
    """Событие системы."""

    name: EventName
    type: EventType
    data: Dict[str, Any]
    id: MessageID = field(default_factory=lambda: MessageID(str(uuid.uuid4())))
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: EventMetadata = field(default_factory=_default_event_metadata)
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[float] = None  # Time to live в секундах

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            self.metadata = EventMetadata(**self.metadata)

    def is_expired(self) -> bool:
        """Проверить, истекло ли событие."""
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": {
                "source": self.metadata.source,
                "correlation_id": self.metadata.correlation_id,
                "causation_id": self.metadata.causation_id,
                "user_id": self.metadata.user_id,
                "session_id": self.metadata.session_id,
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "properties": self.metadata.properties,
            },
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Создать из словаря."""
        return cls(
            id=MessageID(data["id"]),
            name=EventName(data["name"]),
            type=EventType(data["type"]),
            data=data["data"],
            priority=EventPriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata(**data["metadata"]),
            retry_count=data["retry_count"],
            max_retries=data["max_retries"],
            ttl=data.get("ttl"),
        )


@dataclass
class Message:
    """Сообщение для очереди."""

    topic: TopicName
    data: Any
    id: MessageID = field(default_factory=lambda: MessageID(str(uuid.uuid4())))
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: EventMetadata = field(default_factory=_default_event_metadata)
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            self.metadata = EventMetadata(**self.metadata)

    def is_expired(self) -> bool:
        """Проверить, истекло ли сообщение."""
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": self.id,
            "topic": self.topic,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": {
                "source": self.metadata.source,
                "correlation_id": self.metadata.correlation_id,
                "causation_id": self.metadata.causation_id,
                "user_id": self.metadata.user_id,
                "session_id": self.metadata.session_id,
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "properties": self.metadata.properties,
            },
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Создать из словаря."""
        return cls(
            id=MessageID(data["id"]),
            topic=TopicName(data["topic"]),
            data=data["data"],
            priority=MessagePriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata(**data["metadata"]),
            retry_count=data["retry_count"],
            max_retries=data["max_retries"],
            ttl=data.get("ttl"),
        )


# ============================================================================
# Обработчики
# ============================================================================
EventHandler = Callable[[Event], Union[None, Awaitable[None]]]
MessageHandler = Callable[[Message], Union[None, Awaitable[None]]]
ErrorHandler = Callable[
    [Exception, Union[Event, Message]], Union[None, Awaitable[None]]
]


@dataclass
class HandlerConfig:
    """Конфигурация обработчика."""

    name: str
    id: HandlerID = field(default_factory=lambda: HandlerID(str(uuid.uuid4())))
    priority: EventPriority = EventPriority.NORMAL
    timeout: float = 30.0
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    error_handler: Optional[ErrorHandler] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandlerInfo:
    """Информация об обработчике событий."""

    config: HandlerConfig
    handler: EventHandler
    is_async: bool
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_execution: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class MessageHandlerInfo:
    """Информация об обработчике сообщений."""

    config: HandlerConfig
    handler: MessageHandler
    is_async: bool
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_execution: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0


# ============================================================================
# WebSocket типы
# ============================================================================
class WebSocketMessage(TypedDict):
    """WebSocket сообщение."""

    type: Literal["event", "message", "command", "response", "error"]
    topic: Optional[str]
    data: Any
    timestamp: str
    id: Optional[str]
    correlation_id: Optional[str]


class WebSocketCommand(TypedDict):
    """WebSocket команда."""

    command: str
    parameters: Dict[str, Any]
    id: str
    timestamp: str


class WebSocketResponse(TypedDict):
    """WebSocket ответ."""

    success: bool
    data: Optional[Any]
    error: Optional[str]
    command_id: str
    timestamp: str


# ============================================================================
# Протоколы (интерфейсы)
# ============================================================================
class EventBusProtocol(Protocol):
    """Протокол для EventBus."""

    async def start(self) -> None:
        """Запустить EventBus."""
        ...

    async def stop(self) -> None:
        """Остановить EventBus."""
        ...

    def subscribe(
        self,
        event_name: EventName,
        handler: EventHandler,
        config: Optional[HandlerConfig] = None,
    ) -> HandlerID:
        """Подписаться на событие."""
        ...

    def unsubscribe(self, event_name: EventName, handler_id: HandlerID) -> bool:
        """Отписаться от события."""
        ...

    async def publish(self, event: Event) -> bool:
        """Опубликовать событие."""
        ...

    def publish_sync(self, event: Event) -> bool:
        """Опубликовать событие синхронно."""
        ...

    def get_event_history(
        self, event_name: Optional[EventName] = None, limit: int = 100
    ) -> List[Event]:
        """Получить историю событий."""
        ...

    def clear_history(self) -> None:
        """Очистить историю событий."""
        ...

    def get_handlers(self, event_name: EventName) -> List[EventHandlerInfo]:
        """Получить обработчики события."""
        ...

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получить метрики производительности."""
        ...


class MessageQueueProtocol(Protocol):
    """Протокол для MessageQueue."""

    async def start(self) -> None:
        """Запустить MessageQueue."""
        ...

    async def stop(self) -> None:
        """Остановить MessageQueue."""
        ...

    async def publish(
        self,
        topic: TopicName,
        data: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[EventMetadata] = None,
    ) -> bool:
        """Опубликовать сообщение."""
        ...

    async def subscribe(
        self,
        topic: TopicName,
        handler: MessageHandler,
        config: Optional[HandlerConfig] = None,
    ) -> HandlerID:
        """Подписаться на тему."""
        ...

    async def unsubscribe(self, topic: TopicName, handler_id: HandlerID) -> bool:
        """Отписаться от темы."""
        ...

    async def get_message(self, topic: TopicName) -> Optional[Message]:
        """Получить сообщение из очереди."""
        ...

    async def peek_message(self, topic: TopicName) -> Optional[Message]:
        """Посмотреть сообщение без извлечения."""
        ...

    def get_queue_size(self, topic: TopicName) -> int:
        """Получить размер очереди."""
        ...

    def get_all_topics(self) -> List[TopicName]:
        """Получить все темы."""
        ...

    def get_handlers(self, topic: TopicName) -> List[MessageHandlerInfo]:
        """Получить обработчики темы."""
        ...

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получить метрики производительности."""
        ...


class WebSocketServiceProtocol(Protocol):
    """Протокол для WebSocketService."""

    async def start(self) -> None:
        """Запустить WebSocketService."""
        ...

    async def stop(self) -> None:
        """Остановить WebSocketService."""
        ...

    async def add_connection(self, connection_id: ConnectionID, websocket: Any) -> bool:
        """Добавить соединение."""
        ...

    async def remove_connection(self, connection_id: ConnectionID) -> bool:
        """Удалить соединение."""
        ...

    async def subscribe(self, connection_id: ConnectionID, topic: TopicName) -> bool:
        """Подписаться на тему."""
        ...

    async def unsubscribe(self, connection_id: ConnectionID, topic: TopicName) -> bool:
        """Отписаться от темы."""
        ...

    async def broadcast(
        self,
        topic: TopicName,
        message: Any,
        exclude_connection: Optional[ConnectionID] = None,
    ) -> int:
        """Отправить сообщение всем подписчикам темы."""
        ...

    async def send_to_connection(
        self, connection_id: ConnectionID, message: Any
    ) -> bool:
        """Отправить сообщение конкретному соединению."""
        ...

    def get_connection_count(self) -> int:
        """Получить количество соединений."""
        ...

    def get_subscriber_count(self, topic: TopicName) -> int:
        """Получить количество подписчиков темы."""
        ...

    def get_all_topics(self) -> List[TopicName]:
        """Получить все темы."""
        ...

    def get_connection_info(
        self, connection_id: ConnectionID
    ) -> Optional[Dict[str, Any]]:
        """Получить информацию о соединении."""
        return None


# ============================================================================
# Конфигурации
# ============================================================================
class EventBusConfig(BaseModel):
    """Конфигурация EventBus."""

    max_workers: int = Field(default=10, ge=1, le=100)
    queue_size: int = Field(default=1000, ge=100, le=100000)
    max_history: int = Field(default=1000, ge=100, le=10000)
    enable_metrics: bool = True
    enable_retry: bool = True
    default_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    default_retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    enable_priority_queues: bool = True
    enable_async_processing: bool = True


class MessageQueueConfig(BaseModel):
    """Конфигурация MessageQueue."""

    max_size: int = Field(default=10000, ge=1000, le=1000000)
    max_workers: int = Field(default=10, ge=1, le=100)
    enable_priority_queues: bool = True
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    enable_metrics: bool = True
    default_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    enable_retry: bool = True
    default_retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)


class WebSocketServiceConfig(BaseModel):
    """Конфигурация WebSocketService."""

    max_connections: int = Field(default=1000, ge=10, le=10000)
    heartbeat_interval: float = Field(default=30.0, ge=5.0, le=300.0)
    connection_timeout: float = Field(default=300.0, ge=60.0, le=3600.0)
    idle_timeout: float = Field(default=600.0, ge=60.0, le=3600.0)
    monitor_interval: float = Field(default=30.0, ge=5.0, le=300.0)
    max_message_size: int = Field(default=1024 * 1024, ge=1024, le=10 * 1024 * 1024)
    enable_compression: bool = True
    enable_metrics: bool = True
    enable_authentication: bool = False
    allowed_origins: List[str] = Field(default_factory=list)


# ============================================================================
# Метрики
# ============================================================================
class EventBusMetrics(TypedDict):
    """Метрики EventBus."""

    total_events_published: int
    total_events_processed: int
    total_errors: int
    events_per_second: float
    average_processing_time: float
    queue_size: int
    active_handlers: int
    total_handlers: int
    uptime_seconds: float


class MessageQueueMetrics(TypedDict):
    """Метрики MessageQueue."""

    total_messages_published: int
    total_messages_processed: int
    total_errors: int
    messages_per_second: float
    average_processing_time: float
    total_queue_size: int
    active_handlers: int
    total_handlers: int
    topics_count: int
    uptime_seconds: float


class WebSocketServiceMetrics(TypedDict):
    """Метрики WebSocketService."""

    total_connections: int
    active_connections: int
    total_messages_sent: int
    total_messages_received: int
    total_errors: int
    messages_per_second: float
    topics_count: int
    uptime_seconds: float


# ============================================================================
# Исключения
# ============================================================================
class MessagingError(Exception):
    """Базовое исключение для мессенджинга."""

    pass


class EventBusError(MessagingError):
    """Ошибка EventBus."""

    pass


class MessageQueueError(MessagingError):
    """Ошибка MessageQueue."""

    pass


class WebSocketServiceError(MessagingError):
    """Ошибка WebSocketService."""

    pass


class HandlerError(MessagingError):
    """Ошибка обработчика."""

    pass


class ConnectionError(MessagingError):
    """Ошибка соединения."""

    pass


class TimeoutError(MessagingError):
    """Ошибка таймаута."""

    pass


class RetryExhaustedError(MessagingError):
    """Исчерпаны попытки повтора."""

    pass


# ============================================================================
# Утилиты
# ============================================================================
def create_event(
    name: EventName,
    data: Dict[str, Any],
    event_type: EventType,
    priority: EventPriority = EventPriority.NORMAL,
    source: str = "system",
    correlation_id: Optional[CorrelationID] = None,
    **kwargs: Any
) -> Event:
    """Создать событие."""
    metadata = EventMetadata(source=source, correlation_id=correlation_id, **kwargs)
    return Event(
        name=name, type=event_type, data=data, priority=priority, metadata=metadata
    )


def create_message(
    topic: TopicName,
    data: Any,
    priority: MessagePriority = MessagePriority.NORMAL,
    source: str = "system",
    correlation_id: Optional[CorrelationID] = None,
    **kwargs: Any
) -> Message:
    """Создать сообщение."""
    metadata = EventMetadata(source=source, correlation_id=correlation_id, **kwargs)
    return Message(topic=topic, data=data, priority=priority, metadata=metadata)


def create_handler_config(
    name: str,
    priority: EventPriority = EventPriority.NORMAL,
    timeout: float = 30.0,
    retry_on_error: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    error_handler: Optional[ErrorHandler] = None,
    **kwargs: Any
) -> HandlerConfig:
    """Создать конфигурацию обработчика."""
    return HandlerConfig(
        name=name,
        priority=priority,
        timeout=timeout,
        retry_on_error=retry_on_error,
        max_retries=max_retries,
        retry_delay=retry_delay,
        error_handler=error_handler,
        metadata=kwargs,
    )
