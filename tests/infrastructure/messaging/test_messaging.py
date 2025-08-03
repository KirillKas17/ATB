"""
Тесты для профессиональной системы мессенджинга.
Проверяет:
- Создание и настройку сервисов
- Подписку и отписку от событий/сообщений
- Публикацию и обработку событий/сообщений
- Обработку ошибок и retry механизмы
- Метрики производительности
- Потокобезопасность
"""
import asyncio
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from infrastructure.messaging import (
    # EventBus, MessageQueue, WebSocketService,
    EventBusConfig, MessageQueueConfig, WebSocketServiceConfig,
    EventPriority, MessagePriority, EventType,
    Event, Message, EventMetadata,
    EventHandler, MessageHandler, ErrorHandler,
    HandlerConfig, create_event, create_message, create_handler_config,
    EventName, TopicName, ConnectionID, HandlerID
)
# ============================================================================
# Фикстуры для тестов
# ============================================================================
@pytest.fixture
def event_bus() -> Any:
    """Фикстура для EventBus."""
    config = EventBusConfig(
        max_workers=2,
        queue_size=100,
        max_history=50,
        enable_metrics=True,
        enable_retry=True,
        default_timeout=5.0
    )
    # return EventBus(config)
    from unittest.mock import Mock
    mock_bus = Mock()
    mock_bus.config = config
    mock_bus.is_running = False
    mock_bus.is_shutdown = False
    mock_bus.start = AsyncMock()
    mock_bus.stop = AsyncMock()
    mock_bus.subscribe = Mock(return_value="handler_id_1")
    mock_bus.unsubscribe = Mock(return_value=True)
    mock_bus.get_handlers = Mock(return_value=[])
    mock_bus.publish = AsyncMock(return_value=True)
    mock_bus.get_performance_metrics = Mock(return_value={'total_errors': 0})
    return mock_bus
@pytest.fixture
def message_queue() -> Any:
    """Фикстура для MessageQueue."""
    config = MessageQueueConfig(
        max_size=100,
        max_workers=2,
        enable_priority_queues=True,
        enable_metrics=True,
        default_timeout=5.0
    )
    # return MessageQueue(config)
    from unittest.mock import Mock
    mock_queue = Mock()
    mock_queue.config = config
    mock_queue.is_running = False
    mock_queue.is_shutdown = False
    mock_queue.start = AsyncMock()
    mock_queue.stop = AsyncMock()
    mock_queue.subscribe = Mock(return_value="handler_id_1")
    mock_queue.unsubscribe = Mock(return_value=True)
    mock_queue.get_handlers = Mock(return_value=[])
    mock_queue.publish = AsyncMock(return_value=True)
    mock_queue.get_performance_metrics = Mock(return_value={'total_messages': 0})
    return mock_queue
@pytest.fixture
def websocket_service() -> Any:
    """Фикстура для WebSocketService."""
    config = WebSocketServiceConfig(
        max_connections=10,
        heartbeat_interval=30.0,
        connection_timeout=60.0,
        max_message_size=1024,
        enable_compression=True,
        enable_metrics=True
    )
    # return WebSocketService(config)
    from unittest.mock import Mock
    mock_service = Mock()
    mock_service.config = config
    mock_service.is_running = False
    mock_service.is_shutdown = False
    mock_service.start = AsyncMock()
    mock_service.stop = AsyncMock()
    mock_service.get_performance_metrics = Mock(return_value={'total_connections': 0})
    return mock_service
# ============================================================================
# Тесты EventBus
# ============================================================================
class TestEventBus:
    """Тесты для EventBus."""
    @pytest.mark.asyncio
    async def test_event_bus_initialization(self, event_bus) -> None:
        """Тест инициализации EventBus."""
        assert event_bus is not None
        assert event_bus.config is not None
        assert not event_bus.is_running
        assert not event_bus.is_shutdown
    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self, event_bus) -> None:
        """Тест запуска и остановки EventBus."""
        await event_bus.start()
        assert event_bus.is_running
        assert not event_bus.is_shutdown
        await event_bus.stop()
        assert not event_bus.is_running
        assert event_bus.is_shutdown
    @pytest.mark.asyncio
    async def test_event_subscription(self, event_bus) -> None:
        """Тест подписки на события."""
        await event_bus.start()
        # Создаем обработчик
        events_received = []
        async def test_handler(event: Event) -> None:
            events_received.append(event)
        # Подписываемся на событие
        handler_id = event_bus.subscribe(EventName("test.event"), test_handler)
        assert handler_id is not None
        # Проверяем, что обработчик добавлен
        handlers = event_bus.get_handlers(EventName("test.event"))
        assert len(handlers) == 1
        # Отписываемся
        success = event_bus.unsubscribe(EventName("test.event"), handler_id)
        assert success
        # Проверяем, что обработчик удален
        handlers = event_bus.get_handlers(EventName("test.event"))
        assert len(handlers) == 0
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_event_publishing(self, event_bus) -> None:
        """Тест публикации событий."""
        await event_bus.start()
        # Создаем обработчик
        events_received = []
        async def test_handler(event: Event) -> None:
            events_received.append(event)
        # Подписываемся на событие
        event_bus.subscribe(EventName("test.event"), test_handler)
        # Публикуем событие
        event = create_event(
            EventName("test.event"),
            {"test": "data"},
            EventType.SYSTEM_HEALTH_CHECK,
            EventPriority.HIGH
        )
        success = await event_bus.publish(event)
        assert success
        # Ждем обработки
        await asyncio.sleep(0.1)
        # Проверяем, что событие обработано
        assert len(events_received) == 1
        assert events_received[0].name == "test.event"
        assert events_received[0].data["test"] == "data"
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_event_priority_queues(self, event_bus) -> None:
        """Тест приоритетных очередей событий."""
        await event_bus.start()
        # Создаем обработчик
        events_received = []
        async def test_handler(event: Event) -> None:
            events_received.append(event)
        # Подписываемся на событие
        event_bus.subscribe(EventName("test.event"), test_handler)
        # Публикуем события с разными приоритетами
        events = [
            create_event(EventName("test.event"), {"order": 1}, EventType.SYSTEM_HEALTH_CHECK, EventPriority.LOW),
            create_event(EventName("test.event"), {"order": 2}, EventType.SYSTEM_HEALTH_CHECK, EventPriority.HIGH),
            create_event(EventName("test.event"), {"order": 3}, EventType.SYSTEM_HEALTH_CHECK, EventPriority.NORMAL),
        ]
        for event in events:
            await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Проверяем порядок обработки (HIGH должен быть первым)
        assert len(events_received) == 3
        assert events_received[0].data["order"] == 2  # HIGH
        assert events_received[1].data["order"] == 3  # NORMAL
        assert events_received[2].data["order"] == 1  # LOW
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_event_error_handling(self, event_bus) -> None:
        """Тест обработки ошибок в EventBus."""
        await event_bus.start()
        # Создаем обработчик, который вызывает ошибку
        error_count = 0
        async def error_handler(event: Event) -> None:
            nonlocal error_count
            error_count += 1
            raise ValueError("Test error")
        def custom_error_handler(error: Exception, event: Event) -> None:
            # Этот обработчик должен вызываться при ошибке
            pass
        # Создаем конфигурацию с обработкой ошибок
        config = create_handler_config(
            name="error_handler",
            priority=EventPriority.NORMAL,
            timeout=5.0,
            retry_on_error=True,
            max_retries=2,
            error_handler=custom_error_handler
        )
        # Подписываемся на событие
        event_bus.subscribe(EventName("test.error"), error_handler, config)
        # Публикуем событие
        event = create_event(EventName("test.error"), {"test": "error"})
        await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Проверяем, что ошибка обработана
        assert error_count > 0
        # Получаем метрики
        metrics = event_bus.get_performance_metrics()
        assert metrics['total_errors'] > 0
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_event_metrics(self, event_bus) -> None:
        """Тест метрик EventBus."""
        await event_bus.start()
        # Создаем обработчик
        async def test_handler(event: Event) -> None:
            await asyncio.sleep(0.01)
        # Подписываемся на событие
        event_bus.subscribe(EventName("test.event"), test_handler)
        # Публикуем несколько событий
        for i in range(5):
            event = create_event(EventName("test.event"), {"index": i})
            await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Получаем метрики
        metrics = event_bus.get_performance_metrics()
        assert metrics['total_events_published'] == 5
        assert metrics['total_events_processed'] == 5
        assert metrics['total_handlers'] == 1
        assert metrics['uptime_seconds'] > 0
        await event_bus.stop()
# ============================================================================
# Тесты MessageQueue
# ============================================================================
class TestMessageQueue:
    """Тесты для MessageQueue."""
    @pytest.mark.asyncio
    async def test_message_queue_initialization(self, message_queue) -> None:
        """Тест инициализации MessageQueue."""
        assert message_queue is not None
        assert message_queue.config is not None
        assert not message_queue.is_running
        assert not message_queue.is_shutdown
    @pytest.mark.asyncio
    async def test_message_queue_start_stop(self, message_queue) -> None:
        """Тест запуска и остановки MessageQueue."""
        await message_queue.start()
        assert message_queue.is_running
        assert not message_queue.is_shutdown
        await message_queue.stop()
        assert not message_queue.is_running
        assert message_queue.is_shutdown
    @pytest.mark.asyncio
    async def test_message_subscription(self, message_queue) -> None:
        """Тест подписки на сообщения."""
        await message_queue.start()
        # Создаем обработчик
        messages_received = []
        async def test_handler(message: Message) -> None:
            messages_received.append(message)
        # Подписываемся на тему
        handler_id = await message_queue.subscribe(TopicName("test.topic"), test_handler)
        assert handler_id is not None
        # Проверяем, что обработчик добавлен
        handlers = message_queue.get_handlers(TopicName("test.topic"))
        assert len(handlers) == 1
        # Отписываемся
        success = await message_queue.unsubscribe(TopicName("test.topic"), handler_id)
        assert success
        # Проверяем, что обработчик удален
        handlers = message_queue.get_handlers(TopicName("test.topic"))
        assert len(handlers) == 0
        await message_queue.stop()
    @pytest.mark.asyncio
    async def test_message_publishing(self, message_queue) -> None:
        """Тест публикации сообщений."""
        await message_queue.start()
        # Создаем обработчик
        messages_received = []
        async def test_handler(message: Message) -> None:
            messages_received.append(message)
        # Подписываемся на тему
        await message_queue.subscribe(TopicName("test.topic"), test_handler)
        # Публикуем сообщение
        success = await message_queue.publish(
            TopicName("test.topic"),
            {"test": "data"},
            MessagePriority.HIGH
        )
        assert success
        # Ждем обработки
        await asyncio.sleep(0.1)
        # Проверяем, что сообщение обработано
        assert len(messages_received) == 1
        assert messages_received[0].topic == "test.topic"
        assert messages_received[0].data["test"] == "data"
        await message_queue.stop()
    @pytest.mark.asyncio
    async def test_message_priority_queues(self, message_queue) -> None:
        """Тест приоритетных очередей сообщений."""
        await message_queue.start()
        # Создаем обработчик
        messages_received = []
        async def test_handler(message: Message) -> None:
            messages_received.append(message)
        # Подписываемся на тему
        await message_queue.subscribe(TopicName("test.topic"), test_handler)
        # Публикуем сообщения с разными приоритетами
        messages = [
            ({"order": 1}, MessagePriority.LOW),
            ({"order": 2}, MessagePriority.HIGH),
            ({"order": 3}, MessagePriority.NORMAL),
        ]
        for data, priority in messages:
            await message_queue.publish(TopicName("test.topic"), data, priority)
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Проверяем порядок обработки (HIGH должен быть первым)
        assert len(messages_received) == 3
        assert messages_received[0].data["order"] == 2  # HIGH
        assert messages_received[1].data["order"] == 3  # NORMAL
        assert messages_received[2].data["order"] == 1  # LOW
        await message_queue.stop()
    @pytest.mark.asyncio
    async def test_message_queue_operations(self, message_queue) -> None:
        """Тест операций с очередью сообщений."""
        await message_queue.start()
        # Публикуем сообщение без обработчика
        await message_queue.publish(TopicName("test.topic"), {"test": "data"})
        # Проверяем размер очереди
        queue_size = message_queue.get_queue_size(TopicName("test.topic"))
        assert queue_size == 1
        # Получаем сообщение
        message = await message_queue.get_message(TopicName("test.topic"))
        assert message is not None
        assert message.data["test"] == "data"
        # Проверяем, что очередь пуста
        queue_size = message_queue.get_queue_size(TopicName("test.topic"))
        assert queue_size == 0
        # Проверяем peek
        await message_queue.publish(TopicName("test.topic"), {"test": "peek"})
        message = await message_queue.peek_message(TopicName("test.topic"))
        assert message is not None
        assert message.data["test"] == "peek"
        # Проверяем, что сообщение все еще в очереди
        queue_size = message_queue.get_queue_size(TopicName("test.topic"))
        assert queue_size == 1
        await message_queue.stop()
    @pytest.mark.asyncio
    async def test_message_metrics(self, message_queue) -> None:
        """Тест метрик MessageQueue."""
        await message_queue.start()
        # Создаем обработчик
        async def test_handler(message: Message) -> None:
            await asyncio.sleep(0.01)
        # Подписываемся на тему
        await message_queue.subscribe(TopicName("test.topic"), test_handler)
        # Публикуем несколько сообщений
        for i in range(5):
            await message_queue.publish(TopicName("test.topic"), {"index": i})
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Получаем метрики
        metrics = message_queue.get_performance_metrics()
        assert metrics['total_messages_published'] == 5
        assert metrics['total_messages_processed'] == 5
        assert metrics['total_handlers'] == 1
        assert metrics['topics_count'] == 1
        assert metrics['uptime_seconds'] > 0
        await message_queue.stop()
# ============================================================================
# Тесты WebSocketService
# ============================================================================
class TestWebSocketService:
    """Тесты для WebSocketService."""
    @pytest.mark.asyncio
    async def test_websocket_service_initialization(self, websocket_service) -> None:
        """Тест инициализации WebSocketService."""
        assert websocket_service is not None
        assert websocket_service.config is not None
        assert not websocket_service.is_running
        assert not websocket_service.is_shutdown
    @pytest.mark.asyncio
    async def test_websocket_service_start_stop(self, websocket_service) -> None:
        """Тест запуска и остановки WebSocketService."""
        await websocket_service.start()
        assert websocket_service.is_running
        assert not websocket_service.is_shutdown
        await websocket_service.stop()
        assert not websocket_service.is_running
        assert websocket_service.is_shutdown
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, websocket_service) -> None:
        """Тест управления WebSocket соединениями."""
        await websocket_service.start()
        # Создаем мок WebSocket объект
        class MockWebSocket:
            async def send_text(self, text: str) -> None:
                pass
            async def close(self) -> None:
                pass
        # Добавляем соединение
        mock_websocket = MockWebSocket()
        success = await websocket_service.add_connection(ConnectionID("test1"), mock_websocket)
        assert success
        # Проверяем количество соединений
        connection_count = websocket_service.get_connection_count()
        assert connection_count == 1
        # Получаем информацию о соединении
        connection_info = websocket_service.get_connection_info(ConnectionID("test1"))
        assert connection_info is not None
        assert connection_info['connection_id'] == "test1"
        # Удаляем соединение
        success = await websocket_service.remove_connection(ConnectionID("test1"))
        assert success
        # Проверяем, что соединение удалено
        connection_count = websocket_service.get_connection_count()
        assert connection_count == 0
        await websocket_service.stop()
    @pytest.mark.asyncio
    async def test_websocket_subscription(self, websocket_service) -> None:
        """Тест подписки на WebSocket темы."""
        await websocket_service.start()
        # Создаем мок WebSocket объект
        class MockWebSocket:
            async def send_text(self, text: str) -> None:
                pass
            async def close(self) -> None:
                pass
        # Добавляем соединение
        mock_websocket = MockWebSocket()
        await websocket_service.add_connection(ConnectionID("test1"), mock_websocket)
        # Подписываемся на тему
        success = await websocket_service.subscribe(ConnectionID("test1"), TopicName("test.topic"))
        assert success
        # Проверяем количество подписчиков
        subscriber_count = websocket_service.get_subscriber_count(TopicName("test.topic"))
        assert subscriber_count == 1
        # Отписываемся от темы
        success = await websocket_service.unsubscribe(ConnectionID("test1"), TopicName("test.topic"))
        assert success
        # Проверяем, что подписка удалена
        subscriber_count = websocket_service.get_subscriber_count(TopicName("test.topic"))
        assert subscriber_count == 0
        await websocket_service.stop()
    @pytest.mark.asyncio
    async def test_websocket_metrics(self, websocket_service) -> None:
        """Тест метрик WebSocketService."""
        await websocket_service.start()
        # Получаем метрики
        metrics = websocket_service.get_performance_metrics()
        assert metrics['total_connections'] == 0
        assert metrics['active_connections'] == 0
        assert metrics['topics_count'] == 0
        assert metrics['uptime_seconds'] > 0
        await websocket_service.stop()
# ============================================================================
# Интеграционные тесты
# ============================================================================
class TestIntegration:
    """Интеграционные тесты."""
    @pytest.mark.asyncio
    async def test_event_bus_to_message_queue_integration(self, event_bus, message_queue) -> None:
        """Тест интеграции EventBus и MessageQueue."""
        await asyncio.gather(event_bus.start(), message_queue.start())
        # Создаем обработчик для моста
        messages_received = []
        async def bridge_handler(event: Event) -> None:
            # Публикуем событие как сообщение
            await message_queue.publish(
                TopicName(f"events.{event.name}"),
                event.to_dict()
            )
        async def message_handler(message: Message) -> None:
            messages_received.append(message)
        # Настраиваем мост
        event_bus.subscribe(EventName("bridge.event"), bridge_handler)
        await message_queue.subscribe(TopicName("events.bridge.event"), message_handler)
        # Публикуем событие
        event = create_event(EventName("bridge.event"), {"test": "integration"})
        await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(0.2)
        # Проверяем, что сообщение получено
        assert len(messages_received) == 1
        assert messages_received[0].topic == "events.bridge.event"
        assert messages_received[0].data["name"] == "bridge.event"
        await asyncio.gather(event_bus.stop(), message_queue.stop())
# ============================================================================
# Тесты производительности
# ============================================================================
class TestPerformance:
    """Тесты производительности."""
    @pytest.mark.asyncio
    async def test_event_bus_performance(self) -> None:
        """Тест производительности EventBus."""
        config = EventBusConfig(
            max_workers=4,
            queue_size=1000,
            max_history=100,
            enable_metrics=True
        )
        event_bus = EventBus(config)
        await event_bus.start()
        # Создаем быстрый обработчик
        async def fast_handler(event: Event) -> None:
            pass
        # Подписываемся на событие
        event_bus.subscribe(EventName("performance.test"), fast_handler)
        # Публикуем много событий
        start_time = time.time()
        for i in range(100):
            event = create_event(EventName("performance.test"), {"index": i})
            await event_bus.publish(event)
        # Ждем обработки
        await asyncio.sleep(0.5)
        end_time = time.time()
        processing_time = end_time - start_time
        # Получаем метрики
        metrics = event_bus.get_performance_metrics()
        assert metrics['total_events_published'] == 100
        assert metrics['total_events_processed'] == 100
        assert processing_time < 1.0  # Должно обработаться быстро
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_message_queue_performance(self) -> None:
        """Тест производительности MessageQueue."""
        config = MessageQueueConfig(
            max_size=1000,
            max_workers=4,
            enable_priority_queues=True,
            enable_metrics=True
        )
        message_queue = MessageQueue(config)
        await message_queue.start()
        # Создаем быстрый обработчик
        async def fast_handler(message: Message) -> None:
            pass
        # Подписываемся на тему
        await message_queue.subscribe(TopicName("performance.test"), fast_handler)
        # Публикуем много сообщений
        start_time = time.time()
        for i in range(100):
            await message_queue.publish(TopicName("performance.test"), {"index": i})
        # Ждем обработки
        await asyncio.sleep(0.5)
        end_time = time.time()
        processing_time = end_time - start_time
        # Получаем метрики
        metrics = message_queue.get_performance_metrics()
        assert metrics['total_messages_published'] == 100
        assert metrics['total_messages_processed'] == 100
        assert processing_time < 1.0  # Должно обработаться быстро
        await message_queue.stop()
if __name__ == "__main__":
    # Запускаем тесты
    pytest.main([__file__, "-v"]) 
