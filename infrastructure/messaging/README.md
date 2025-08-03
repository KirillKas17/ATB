# Профессиональная система мессенджинга

Профессиональная система мессенджинга для алготрейдинга с поддержкой событий, сообщений и WebSocket соединений.

## Особенности

- **Строгая типизация** - полная поддержка типов Python с использованием Protocol
- **Асинхронная обработка** - высокопроизводительная асинхронная обработка
- **Приоритетные очереди** - поддержка приоритетов для событий и сообщений
- **Мониторинг производительности** - встроенные метрики и мониторинг
- **Обработка ошибок** - retry механизмы и обработчики ошибок
- **Потокобезопасность** - безопасная работа в многопоточной среде
- **Соответствие DDD и SOLID** - архитектурные принципы

## Компоненты

### EventBus

Асинхронная система обработки событий с приоритетными очередями.

```python
from infrastructure.messaging import EventBus, EventBusConfig, EventPriority, create_event

# Создание EventBus
config = EventBusConfig(
    max_workers=5,
    queue_size=1000,
    max_history=500,
    enable_metrics=True,
    enable_retry=True,
    default_timeout=10.0
)

event_bus = EventBus(config)

# Запуск
await event_bus.start()

# Подписка на события
async def event_handler(event):
    print(f"Received event: {event.name}")

handler_id = event_bus.subscribe("market.tick", event_handler)

# Публикация событий
event = create_event(
    name="market.tick",
    data={"price": 50000, "volume": 100},
    priority=EventPriority.HIGH
)

await event_bus.publish(event)

# Остановка
await event_bus.stop()
```

### MessageQueue

Очередь сообщений с поддержкой тем и приоритетов.

```python
from infrastructure.messaging import MessageQueue, MessageQueueConfig, MessagePriority

# Создание MessageQueue
config = MessageQueueConfig(
    max_size=5000,
    max_workers=5,
    enable_priority_queues=True,
    enable_metrics=True
)

message_queue = MessageQueue(config)

# Запуск
await message_queue.start()

# Подписка на темы
async def message_handler(message):
    print(f"Received message: {message.topic}")

handler_id = await message_queue.subscribe("trading.signals", message_handler)

# Публикация сообщений
await message_queue.publish(
    topic="trading.signals",
    data={"signal": "BUY", "confidence": 0.8},
    priority=MessagePriority.HIGH
)

# Остановка
await message_queue.stop()
```

### WebSocketService

Управление WebSocket соединениями для real-time коммуникации.

```python
from infrastructure.messaging import WebSocketService, WebSocketServiceConfig

# Создание WebSocketService
config = WebSocketServiceConfig(
    max_connections=100,
    heartbeat_interval=30.0,
    connection_timeout=300.0,
    enable_metrics=True
)

websocket_service = WebSocketService(config)

# Запуск
await websocket_service.start()

# Добавление соединения (в реальном приложении через WebSocket сервер)
await websocket_service.add_connection("client1", websocket)

# Подписка на темы
await websocket_service.subscribe("client1", "market.data")

# Broadcast сообщений
await websocket_service.broadcast(
    topic="market.data",
    message={"price": 50000}
)

# Остановка
await websocket_service.stop()
```

## Конфигурация

### EventBusConfig

```python
@dataclass
class EventBusConfig:
    max_workers: int = 10
    queue_size: int = 10000
    max_history: int = 1000
    enable_metrics: bool = True
    enable_retry: bool = True
    default_timeout: float = 30.0
    enable_async_processing: bool = True
```

### MessageQueueConfig

```python
@dataclass
class MessageQueueConfig:
    max_size: int = 10000
    max_workers: int = 10
    enable_priority_queues: bool = True
    enable_metrics: bool = True
    default_timeout: float = 30.0
```

### WebSocketServiceConfig

```python
@dataclass
class WebSocketServiceConfig:
    max_connections: int = 1000
    heartbeat_interval: float = 30.0
    connection_timeout: float = 300.0
    max_message_size: int = 1024 * 1024
    enable_compression: bool = True
    enable_metrics: bool = True
```

## Обработчики

### Создание обработчиков

```python
from infrastructure.messaging import HandlerConfig, create_handler_config

# Конфигурация обработчика
config = create_handler_config(
    name="trading_handler",
    priority=EventPriority.HIGH,
    timeout=5.0,
    retry_on_error=True,
    max_retries=3,
    retry_delay=1.0,
    error_handler=custom_error_handler
)

# Асинхронный обработчик
async def async_handler(event):
    # Асинхронная обработка
    await process_event(event)

# Синхронный обработчик
def sync_handler(event):
    # Синхронная обработка
    process_event_sync(event)
```

### Обработка ошибок

```python
def error_handler(error: Exception, event_or_message):
    logger.error(f"Error processing {event_or_message}: {error}")
    
    # Можно отправить уведомление, записать в лог и т.д.
    send_alert(error, event_or_message)

# Использование в конфигурации
config = create_handler_config(
    name="handler_with_error_handling",
    error_handler=error_handler,
    retry_on_error=True,
    max_retries=3
)
```

## Приоритеты

### EventPriority

```python
class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
```

### MessagePriority

```python
class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
```

## Метрики

### Получение метрик

```python
# EventBus метрики
event_metrics = event_bus.get_performance_metrics()
print(f"Events processed: {event_metrics['total_events_processed']}")
print(f"Events per second: {event_metrics['events_per_second']}")

# MessageQueue метрики
message_metrics = message_queue.get_performance_metrics()
print(f"Messages processed: {message_metrics['total_messages_processed']}")
print(f"Queue size: {message_metrics['total_queue_size']}")

# WebSocketService метрики
websocket_metrics = websocket_service.get_performance_metrics()
print(f"Active connections: {websocket_metrics['active_connections']}")
print(f"Topics count: {websocket_metrics['topics_count']}")
```

### Доступные метрики

#### EventBus
- `total_events_published` - общее количество опубликованных событий
- `total_events_processed` - общее количество обработанных событий
- `total_errors` - общее количество ошибок
- `events_per_second` - события в секунду
- `average_processing_time` - среднее время обработки
- `queue_size` - размер очереди
- `active_handlers` - активные обработчики
- `total_handlers` - общее количество обработчиков
- `uptime_seconds` - время работы

#### MessageQueue
- `total_messages_published` - общее количество опубликованных сообщений
- `total_messages_processed` - общее количество обработанных сообщений
- `total_errors` - общее количество ошибок
- `messages_per_second` - сообщения в секунду
- `average_processing_time` - среднее время обработки
- `total_queue_size` - общий размер очереди
- `active_handlers` - активные обработчики
- `total_handlers` - общее количество обработчиков
- `topics_count` - количество тем
- `uptime_seconds` - время работы

#### WebSocketService
- `total_connections` - общее количество соединений
- `active_connections` - активные соединения
- `total_messages_sent` - общее количество отправленных сообщений
- `total_messages_received` - общее количество полученных сообщений
- `total_errors` - общее количество ошибок
- `messages_per_second` - сообщения в секунду
- `topics_count` - количество тем
- `uptime_seconds` - время работы

## Интеграция

### EventBus + MessageQueue

```python
# Создаем мост между EventBus и MessageQueue
async def bridge_handler(event):
    # Публикуем событие как сообщение
    await message_queue.publish(
        topic=f"events.{event.name}",
        data=event.to_dict(),
        priority=MessagePriority.HIGH if event.priority == EventPriority.HIGH else MessagePriority.NORMAL
    )

# Подписываемся на события
event_bus.subscribe("market.tick", bridge_handler)

# Обрабатываем сообщения
async def message_handler(message):
    # Обработка сообщения
    process_message(message)

await message_queue.subscribe("events.market.tick", message_handler)
```

### MessageQueue + WebSocketService

```python
# Создаем мост между MessageQueue и WebSocketService
async def websocket_bridge_handler(message):
    # Отправляем сообщение через WebSocket
    await websocket_service.broadcast(
        topic=message.topic,
        message=message.data
    )

# Подписываемся на сообщения
await message_queue.subscribe("market.data", websocket_bridge_handler)
```

## Лучшие практики

### 1. Правильное использование приоритетов

```python
# Критические события (ошибки, алерты)
event = create_event("system.error", data, priority=EventPriority.CRITICAL)

# Высокий приоритет (торговые сигналы)
event = create_event("trading.signal", data, priority=EventPriority.HIGH)

# Обычный приоритет (логи, метрики)
event = create_event("system.log", data, priority=EventPriority.NORMAL)

# Низкий приоритет (аналитика, отчеты)
event = create_event("analytics.report", data, priority=EventPriority.LOW)
```

### 2. Обработка ошибок

```python
def robust_error_handler(error: Exception, event_or_message):
    # Логируем ошибку
    logger.error(f"Handler error: {error}")
    
    # Отправляем алерт для критических ошибок
    if isinstance(error, CriticalError):
        send_critical_alert(error, event_or_message)
    
    # Записываем в метрики
    record_error_metric(error)

# Используем в конфигурации
config = create_handler_config(
    name="robust_handler",
    error_handler=robust_error_handler,
    retry_on_error=True,
    max_retries=3,
    retry_delay=2.0
)
```

### 3. Мониторинг производительности

```python
async def monitor_performance():
    while True:
        # Получаем метрики
        event_metrics = event_bus.get_performance_metrics()
        message_metrics = message_queue.get_performance_metrics()
        
        # Проверяем производительность
        if event_metrics['events_per_second'] < 100:
            logger.warning("Low event processing rate")
        
        if message_metrics['total_queue_size'] > 1000:
            logger.warning("Large message queue")
        
        # Отправляем метрики в систему мониторинга
        send_metrics_to_monitoring(event_metrics, message_metrics)
        
        await asyncio.sleep(60)  # Проверяем каждую минуту

# Запускаем мониторинг
asyncio.create_task(monitor_performance())
```

### 4. Правильное закрытие ресурсов

```python
async def main():
    # Создаем сервисы
    event_bus = EventBus()
    message_queue = MessageQueue()
    websocket_service = WebSocketService()
    
    try:
        # Запускаем сервисы
        await asyncio.gather(
            event_bus.start(),
            message_queue.start(),
            websocket_service.start()
        )
        
        # Основная логика
        await run_trading_system()
        
    finally:
        # Правильно останавливаем сервисы
        await asyncio.gather(
            event_bus.stop(),
            message_queue.stop(),
            websocket_service.stop()
        )

# Использование контекстных менеджеров
async def with_context_managers():
    async with EventBus() as event_bus:
        async with MessageQueue() as message_queue:
            async with WebSocketService() as websocket_service:
                # Работа с сервисами
                await run_trading_system()
```

## Примеры использования

### Торговая система

```python
async def trading_system():
    # Создаем сервисы
    event_bus = EventBus()
    message_queue = MessageQueue()
    
    await asyncio.gather(event_bus.start(), message_queue.start())
    
    # Обработчик рыночных данных
    async def market_data_handler(event):
        # Анализируем рыночные данные
        analysis = analyze_market_data(event.data)
        
        # Публикуем результат анализа
        await message_queue.publish(
            "market.analysis",
            analysis,
            MessagePriority.HIGH
        )
    
    # Обработчик торговых сигналов
    async def trading_signal_handler(message):
        # Генерируем торговые сигналы
        signal = generate_trading_signal(message.data)
        
        if signal:
            # Публикуем сигнал как событие
            event = create_event(
                "trading.signal",
                signal,
                priority=EventPriority.CRITICAL
            )
            await event_bus.publish(event)
    
    # Настраиваем обработчики
    event_bus.subscribe("market.tick", market_data_handler)
    await message_queue.subscribe("market.analysis", trading_signal_handler)
    
    # Запускаем получение рыночных данных
    await start_market_data_feed(event_bus)
    
    # Ждем завершения
    await asyncio.Event().wait()
```

### Система мониторинга

```python
async def monitoring_system():
    event_bus = EventBus()
    websocket_service = WebSocketService()
    
    await asyncio.gather(event_bus.start(), websocket_service.start())
    
    # Обработчик системных событий
    async def system_event_handler(event):
        # Отправляем событие через WebSocket
        await websocket_service.broadcast(
            "system.events",
            {
                "type": event.name,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "priority": event.priority.value
            }
        )
    
    # Обработчик алертов
    async def alert_handler(event):
        # Отправляем алерт
        await websocket_service.broadcast(
            "system.alerts",
            {
                "alert": event.data,
                "severity": event.priority.value,
                "timestamp": event.timestamp.isoformat()
            }
        )
    
    # Настраиваем обработчики
    event_bus.subscribe("system.*", system_event_handler)
    event_bus.subscribe("system.alert", alert_handler)
    
    # Запускаем мониторинг
    await start_system_monitoring(event_bus)
```

## Заключение

Профессиональная система мессенджинга предоставляет мощные инструменты для построения масштабируемых и надежных алготрейдинговых систем. Строгая типизация, асинхронная обработка и встроенный мониторинг делают её идеальным выбором для продакшен-среды. 