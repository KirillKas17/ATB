"""
Примеры использования профессиональной системы мессенджинга.
Демонстрирует:
- Создание и настройку EventBus, MessageQueue, WebSocketService
- Подписку на события и сообщения
- Публикацию событий и сообщений
- Обработку ошибок и retry механизмы
- Мониторинг производительности
- Асинхронную обработку
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

from loguru import logger

from .event_bus import EventBus
from .message_queue import MessageQueue
from .websocket_service import WebSocketService
from domain.types.messaging_types import (
    ErrorHandler,
    Event,
    EventBusConfig,
    EventHandler,
    EventMetadata,
    EventPriority,
    EventType,
    HandlerConfig,
    Message,
    MessageHandler,
    MessagePriority,
    MessageQueueConfig,
    WebSocketServiceConfig,
    EventName,
    TopicName,
    create_event,
    create_handler_config,
    create_message,
    CorrelationID,
)


# ============================================================================
# Примеры обработчиков
# ============================================================================
async def async_event_handler(event: Event) -> None:
    """Асинхронный обработчик событий."""
    logger.info(f"Async event handler: {event.name} - {event.data}")
    await asyncio.sleep(0.1)  # Имитация асинхронной работы


def sync_event_handler(event: Event) -> None:
    """Синхронный обработчик событий."""
    logger.info(f"Sync event handler: {event.name} - {event.data}")


async def async_message_handler(message: Message) -> None:
    """Асинхронный обработчик сообщений."""
    logger.info(f"Async message handler: {message.topic} - {message.data}")
    await asyncio.sleep(0.1)  # Имитация асинхронной работы


def sync_message_handler(message: Message) -> None:
    """Синхронный обработчик сообщений."""
    logger.info(f"Sync message handler: {message.topic} - {message.data}")


def error_handler(error: Exception, event_or_message: Any) -> None:
    """Обработчик ошибок."""
    logger.error(f"Error in handler: {error} for {event_or_message}")


# ============================================================================
# Пример использования EventBus
# ============================================================================
async def event_bus_example() -> None:
    """Пример использования EventBus."""
    logger.info("=== EventBus Example ===")
    # Создаем конфигурацию
    config = EventBusConfig(
        max_workers=5,
        queue_size=1000,
        max_history=500,
        enable_metrics=True,
        enable_retry=True,
        default_timeout=10.0,
    )
    # Создаем EventBus
    event_bus = EventBus(config)
    try:
        # Запускаем EventBus
        await event_bus.start()
        # Создаем конфигурации обработчиков
        async_config = create_handler_config(
            name="async_handler",
            priority=EventPriority.HIGH,
            timeout=5.0,
            retry_on_error=True,
            max_retries=3,
            error_handler=error_handler,
        )
        sync_config = create_handler_config(
            name="sync_handler",
            priority=EventPriority.NORMAL,
            timeout=3.0,
            retry_on_error=True,
            max_retries=2,
            error_handler=error_handler,
        )
        # Подписываемся на события
        async_handler_id = event_bus.subscribe(
            EventName("test.event"), async_event_handler, async_config
        )
        sync_handler_id = event_bus.subscribe(
            EventName("test.event"), sync_event_handler, sync_config
        )
        # Создаем и публикуем события
        for i in range(5):
            event = create_event(
                name=EventName("test.event"),
                data={"message": f"Test event {i}", "timestamp": datetime.utcnow()},
                event_type=EventType.SYSTEM_HEALTH_CHECK,
                priority=EventPriority.HIGH if i % 2 == 0 else EventPriority.NORMAL,
                source="example",
                correlation_id=CorrelationID(f"corr_{i}"),
            )
            success = await event_bus.publish(event)
            logger.info(f"Event {i} published: {success}")
        # Ждем обработки событий
        await asyncio.sleep(2)
        # Получаем метрики
        metrics = event_bus.get_performance_metrics()
        logger.info(f"EventBus metrics: {metrics}")
        # Получаем историю событий
        history = event_bus.get_event_history(EventName("test.event"), limit=10)
        logger.info(f"Event history: {len(history)} events")
        # Получаем информацию об обработчиках
        handlers = event_bus.get_handlers(EventName("test.event"))
        logger.info(f"Handlers: {len(handlers)}")
    finally:
        # Останавливаем EventBus
        await event_bus.stop()


# ============================================================================
# Пример использования MessageQueue
# ============================================================================
async def message_queue_example() -> None:
    """Пример использования MessageQueue."""
    logger.info("=== MessageQueue Example ===")
    # Создаем конфигурацию
    config = MessageQueueConfig(
        max_size=5000,
        max_workers=5,
        enable_priority_queues=True,
        enable_metrics=True,
        default_timeout=10.0,
    )
    # Создаем MessageQueue
    message_queue = MessageQueue(config)
    try:
        # Запускаем MessageQueue
        await message_queue.start()
        # Создаем конфигурации обработчиков
        async_config = create_handler_config(
            name="async_message_handler",
            priority=EventPriority.HIGH,
            timeout=5.0,
            retry_on_error=True,
            max_retries=3,
            error_handler=error_handler,
        )
        sync_config = create_handler_config(
            name="sync_message_handler",
            priority=EventPriority.NORMAL,
            timeout=3.0,
            retry_on_error=True,
            max_retries=2,
            error_handler=error_handler,
        )
        # Подписываемся на темы
        async_handler_id = await message_queue.subscribe(
            TopicName("test.topic"), async_message_handler, async_config
        )
        sync_handler_id = await message_queue.subscribe(
            TopicName("test.topic"), sync_message_handler, sync_config
        )
        # Создаем и публикуем сообщения
        for i in range(5):
            message = create_message(
                topic=TopicName("test.topic"),
                data={"message": f"Test message {i}", "timestamp": datetime.utcnow()},
                priority=MessagePriority.HIGH if i % 2 == 0 else MessagePriority.NORMAL,
                source="example",
                correlation_id=CorrelationID(f"corr_{i}"),
            )
            success = await message_queue.publish(TopicName("test.topic"), message.data)
            logger.info(f"Message {i} published: {success}")
        # Ждем обработки сообщений
        await asyncio.sleep(2)
        # Получаем метрики
        metrics = message_queue.get_performance_metrics()
        logger.info(f"MessageQueue metrics: {metrics}")
        # Получаем размер очереди
        queue_size = message_queue.get_queue_size(TopicName("test.topic"))
        logger.info(f"Queue size: {queue_size}")
        # Получаем информацию об обработчиках
        handlers = message_queue.get_handlers(TopicName("test.topic"))
        logger.info(f"Handlers: {len(handlers)}")
    finally:
        # Останавливаем MessageQueue
        await message_queue.stop()


# ============================================================================
# Пример использования WebSocketService
# ============================================================================
async def websocket_service_example() -> None:
    """Пример использования WebSocketService."""
    logger.info("=== WebSocketService Example ===")
    # Создаем конфигурацию
    config = WebSocketConfig(
        host="localhost",
        port=8080,
        max_connections=100,
        enable_metrics=True,
        enable_heartbeat=True,
        heartbeat_interval=30.0,
    )
    # Создаем WebSocketService
    websocket_service = WebSocketService(config)
    try:
        # Запускаем WebSocketService
        await websocket_service.start()
        # Создаем обработчики
        async def message_handler(websocket: Any, message: str) -> None:
            logger.info(f"Received message: {message}")
            await websocket.send(f"Echo: {message}")

        async def connection_handler(websocket: Any, path: str) -> None:
            logger.info(f"New connection from {path}")
            await websocket.send("Welcome!")

        # Ждем подключений
        await asyncio.sleep(5)
        # Получаем метрики
        metrics = websocket_service.metrics
        logger.info(f"WebSocket metrics: {metrics}")
    finally:
        # Останавливаем WebSocketService
        await websocket_service.stop()


# ============================================================================
# Пример интеграции всех сервисов
# ============================================================================
async def integration_example() -> None:
    """Пример интеграции всех сервисов."""
    logger.info("=== Integration Example ===")
    # Создаем все сервисы
    event_bus = EventBus(EventBusConfig())
    message_queue = MessageQueue(MessageQueueConfig())
    websocket_service = WebSocketService(WebSocketServiceConfig())
    try:
        # Запускаем все сервисы
        await event_bus.start()
        await message_queue.start()
        await websocket_service.start()
        # Создаем мост между EventBus и MessageQueue
        async def bridge_handler(event: Event) -> None:
            await message_queue.publish(
                TopicName(event.name.value),
                event.data,
                priority=MessagePriority.HIGH if event.priority == EventPriority.HIGH else MessagePriority.NORMAL
            )

        # Создаем мост между MessageQueue и WebSocket
        async def websocket_bridge_handler(message: Message) -> None:
            # Отправляем сообщение через WebSocket
            await websocket_service.broadcast(TopicName("bridge.message"), f"Message: {message.data}")
        # Подписываемся на события
        event_bus.subscribe(EventName("bridge.event"), bridge_handler)
        await message_queue.subscribe(TopicName("bridge.message"), websocket_bridge_handler)
        # Публикуем тестовые события
        for i in range(3):
            event = create_event(
                name=EventName("bridge.event"),
                data={"message": f"Bridge test {i}"},
                event_type=EventType.SYSTEM_HEALTH_CHECK,
                priority=EventPriority.NORMAL,
                source="integration_test",
                correlation_id=CorrelationID(f"bridge_{i}"),
            )
            await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(3)
    finally:
        # Останавливаем все сервисы
        await event_bus.stop()
        await message_queue.stop()
        await websocket_service.stop()


# ============================================================================
# Пример обработки ошибок
# ============================================================================
async def error_handling_example() -> None:
    """Пример обработки ошибок."""
    logger.info("=== Error Handling Example ===")
    event_bus = EventBus(EventBusConfig())
    try:
        await event_bus.start()
        # Создаем обработчик, который может вызвать ошибку
        async def error_prone_handler(event: Event) -> None:
            if "error" in event.data:
                raise ValueError("Simulated error")
            logger.info(f"Processed event: {event.data}")
        # Создаем кастомный обработчик ошибок
        async def custom_error_handler(error: Exception, event: Event) -> None:
            logger.error(f"Custom error handler: {error} for event {event.name}")
        # Создаем конфигурацию с обработчиком ошибок
        config = create_handler_config(
            name="error_prone_handler",
            priority=EventPriority.NORMAL,
            timeout=5.0,
            retry_on_error=True,
            max_retries=2,
            error_handler=custom_error_handler,
        )
        # Подписываемся на события
        event_bus.subscribe(EventName("error.test"), error_prone_handler, config)
        # Публикуем события, включая те, которые вызовут ошибки
        events = [
            create_event(
                name=EventName("error.test"),
                data={"message": "Normal event"},
                event_type=EventType.SYSTEM_HEALTH_CHECK,
                priority=EventPriority.NORMAL,
                source="error_test",
            ),
            create_event(
                name=EventName("error.test"),
                data={"message": "Error event", "error": True},
                event_type=EventType.SYSTEM_HEALTH_CHECK,
                priority=EventPriority.NORMAL,
                source="error_test",
            ),
            create_event(
                name=EventName("error.test"),
                data={"message": "Another normal event"},
                event_type=EventType.SYSTEM_HEALTH_CHECK,
                priority=EventPriority.NORMAL,
                source="error_test",
            ),
        ]
        for event in events:
            await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(2)
    finally:
        await event_bus.stop()


# ============================================================================
# Запуск всех примеров
# ============================================================================
async def run_all_examples() -> None:
    """Запуск всех примеров."""
    logger.info("Starting all examples...")
    try:
        await event_bus_example()
        await message_queue_example()
        await websocket_service_example()
        await integration_example()
        await error_handling_example()
        logger.info("All examples completed successfully!")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Запускаем примеры
    asyncio.run(run_all_examples())
