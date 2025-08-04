"""
Профессиональная система мессенджинга.
Включает:
- EventBus - асинхронная обработка событий с приоритетными очередями
- MessageQueue - очередь сообщений с поддержкой тем и приоритетов
- WebSocketService - управление WebSocket соединениями
- Строгая типизация и соответствие протоколам
- Мониторинг производительности и метрики
- Обработка ошибок и retry механизмы
"""

# Экспорт типов из domain
from domain.type_definitions.messaging_types import (  # Базовые типы; Приоритеты; Типы событий; Модели данных; Обработчики; WebSocket типы; Протоколы; Конфигурации; Метрики; Исключения; Утилиты
    ConnectionError,
    ConnectionID,
    CorrelationID,
    ErrorHandler,
    Event,
    EventBusConfig,
    EventBusError,
    EventBusMetrics,
    EventBusProtocol,
    EventHandler,
    EventHandlerInfo,
    EventMetadata,
    EventName,
    EventPriority,
    EventType,
    HandlerConfig,
    HandlerError,
    HandlerID,
    Message,
    MessageHandler,
    MessageHandlerInfo,
    MessageID,
    MessagePriority,
    MessageQueueConfig,
    MessageQueueError,
    MessageQueueMetrics,
    MessageQueueProtocol,
    MessagingError,
    RetryExhaustedError,
    TimeoutError,
    TopicName,
    WebSocketCommand,
    WebSocketMessage,
    WebSocketResponse,
    WebSocketServiceConfig,
    WebSocketServiceError,
    WebSocketServiceMetrics,
    WebSocketServiceProtocol,
    create_event,
    create_handler_config,
    create_message,
)

from .event_bus import EventBus
from .message_queue import MessageQueue
from .websocket_service import WebSocketService
from .websocket_service import WebSocketConnection

__all__ = [
    # Основные классы
    "EventBus",
    "MessageQueue",
    "WebSocketService",
    "WebSocketConnection",
    # Базовые типы
    "EventName",
    "TopicName",
    "ConnectionID",
    "CorrelationID",
    "HandlerID",
    "MessageID",
    # Приоритеты
    "EventPriority",
    "MessagePriority",
    # Типы событий
    "EventType",
    # Модели данных
    "Event",
    "Message",
    "EventMetadata",
    # Обработчики
    "EventHandler",
    "MessageHandler",
    "ErrorHandler",
    "HandlerConfig",
    "EventHandlerInfo",
    "MessageHandlerInfo",
    # WebSocket типы
    "WebSocketMessage",
    "WebSocketCommand",
    "WebSocketResponse",
    # Протоколы
    "EventBusProtocol",
    "MessageQueueProtocol",
    "WebSocketServiceProtocol",
    # Конфигурации
    "EventBusConfig",
    "MessageQueueConfig",
    "WebSocketServiceConfig",
    # Метрики
    "EventBusMetrics",
    "MessageQueueMetrics",
    "WebSocketServiceMetrics",
    # Исключения
    "MessagingError",
    "EventBusError",
    "MessageQueueError",
    "WebSocketServiceError",
    "HandlerError",
    "ConnectionError",
    "TimeoutError",
    "RetryExhaustedError",
    # Утилиты
    "create_event",
    "create_message",
    "create_handler_config",
]
