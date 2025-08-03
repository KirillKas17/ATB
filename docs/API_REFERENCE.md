# API Reference - Advanced Trading Bot

## Содержание

1. [EventBus](#eventbus)
2. [OptimizedCache](#optimizedcache)
3. [SystemMonitor](#systemmonitor)
4. [PerformanceMonitor](#performancemonitor)
5. [AlertManager](#alertmanager)
6. [CacheManager](#cachemanager)

---

## EventBus

Высокопроизводительная система событий с оптимизациями.

### Классы

#### EventPriority
```python
class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
```

#### Event
```python
@dataclass
class Event:
    event_type: str
    data: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
```

#### EventBus
```python
class EventBus:
    def __init__(self, max_queue_size: int = 10000, batch_size: int = 100)
```

**Параметры:**
- `max_queue_size` (int): Максимальный размер очереди событий
- `batch_size` (int): Размер батча для обработки событий

**Методы:**

##### start()
```python
async def start(self) -> None
```
Запуск EventBus с асинхронной обработкой событий.

##### stop()
```python
async def stop(self) -> None
```
Остановка EventBus и обработка оставшихся событий.

##### subscribe()
```python
def subscribe(self, event_type: str, handler: Callable) -> None
```
Подписка на события определенного типа.

**Параметры:**
- `event_type` (str): Тип события
- `handler` (Callable): Обработчик события

##### unsubscribe()
```python
def unsubscribe(self, event_type: str, handler: Callable) -> None
```
Отписка от событий.

##### publish()
```python
async def publish(self, event: Event) -> bool
```
Публикация события в очередь.

**Возвращает:** True если событие добавлено, False если очередь переполнена

##### get_metrics()
```python
def get_metrics(self) -> Dict[str, Any]
```
Получение метрик производительности EventBus.

**Возвращает:** Словарь с метриками

### Пример использования

```python
import asyncio
from utils.event_bus import EventBus, Event, EventPriority

async def main():
    # Создание EventBus
    event_bus = EventBus(max_queue_size=5000, batch_size=50)
    
    # Подписка на события
    def handle_trade(trade_event):
        print(f"Trade: {trade_event.data}")
    
    event_bus.subscribe("trade", handle_trade)
    
    # Запуск EventBus
    await event_bus.start()
    
    # Публикация события
    trade_event = Event(
        event_type="trade",
        data={"symbol": "BTC/USD", "price": 50000},
        priority=EventPriority.HIGH
    )
    
    await event_bus.publish(trade_event)
    
    # Остановка
    await event_bus.stop()

asyncio.run(main())
```

---

## OptimizedCache

Высокопроизводительный кэш с LRU, TTL и сжатием данных.

### Классы

#### CacheStrategy
```python
class CacheStrategy(Enum):
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
```

#### CompressionType
```python
class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
```

#### CacheEntry
```python
@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    compression_type: CompressionType = CompressionType.NONE
    compressed_data: Optional[bytes] = None
```

#### OptimizedCache
```python
class OptimizedCache:
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.MEMORY,
        enable_compression: bool = True,
        compression_threshold: int = 1024,
        cleanup_interval: float = 60.0,
    )
```

**Параметры:**
- `max_size` (int): Максимальное количество записей
- `ttl_seconds` (Optional[int]): Время жизни записей в секундах
- `strategy` (CacheStrategy): Стратегия кэширования
- `enable_compression` (bool): Включение сжатия данных
- `compression_threshold` (int): Порог для сжатия в байтах
- `cleanup_interval` (float): Интервал очистки в секундах

**Методы:**

##### start()
```python
async def start(self) -> None
```
Запуск кэша с асинхронной очисткой.

##### stop()
```python
async def stop(self) -> None
```
Остановка кэша и финальная очистка.

##### get()
```python
async def get(self, key: str, default: Any = None) -> Any
```
Получение значения из кэша.

**Параметры:**
- `key` (str): Ключ записи
- `default` (Any): Значение по умолчанию

**Возвращает:** Значение из кэша или default

##### set()
```python
async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool
```
Установка значения в кэш.

**Параметры:**
- `key` (str): Ключ записи
- `value` (Any): Значение для кэширования
- `ttl` (Optional[int]): Время жизни записи

**Возвращает:** True если успешно, False при ошибке

##### delete()
```python
async def delete(self, key: str) -> bool
```
Удаление записи из кэша.

##### clear()
```python
async def clear(self) -> None
```
Очистка всего кэша.

##### exists()
```python
async def exists(self, key: str) -> bool
```
Проверка существования ключа.

##### get_stats()
```python
async def get_stats(self, key: str) -> Optional[Dict[str, Any]]
```
Получение статистики по ключу.

##### get_metrics()
```python
def get_metrics(self) -> Dict[str, Any]
```
Получение метрик производительности кэша.

### Пример использования

```python
import asyncio
from utils.cache import OptimizedCache, CacheStrategy

async def main():
    # Создание кэша
    cache = OptimizedCache(
        max_size=1000,
        ttl_seconds=3600,  # 1 час
        strategy=CacheStrategy.MEMORY,
        enable_compression=True
    )
    
    # Запуск кэша
    await cache.start()
    
    # Установка значения
    await cache.set("user:123", {"name": "John", "balance": 1000})
    
    # Получение значения
    user_data = await cache.get("user:123")
    print(user_data)  # {"name": "John", "balance": 1000}
    
    # Проверка существования
    exists = await cache.exists("user:123")
    print(exists)  # True
    
    # Получение метрик
    metrics = cache.get_metrics()
    print(metrics)
    
    # Остановка кэша
    await cache.stop()

asyncio.run(main())
```

---

## SystemMonitor

Основная система мониторинга с алертами и метриками.

### Классы

#### AlertLevel
```python
class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

#### MetricType
```python
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
```

#### Alert
```python
@dataclass
class Alert:
    id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
```

#### Metric
```python
@dataclass
class Metric:
    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
```

#### SystemMonitor
```python
class SystemMonitor:
    def __init__(self)
```

**Методы:**

##### start()
```python
async def start(self) -> None
```
Запуск системы мониторинга.

##### stop()
```python
async def stop(self) -> None
```
Остановка системы мониторинга.

##### add_integration()
```python
def add_integration(self, name: str, integration: MonitoringIntegration) -> None
```
Добавление интеграции мониторинга.

##### get_status()
```python
def get_status(self) -> Dict[str, Any]
```
Получение статуса системы мониторинга.

### Пример использования

```python
import asyncio
from utils.monitoring import SystemMonitor

async def main():
    # Создание монитора
    monitor = SystemMonitor()
    
    # Запуск мониторинга
    await monitor.start()
    
    # Получение статуса
    status = monitor.get_status()
    print(f"Monitoring status: {status}")
    
    # Работа системы...
    await asyncio.sleep(60)
    
    # Остановка мониторинга
    await monitor.stop()

asyncio.run(main())
```

---

## PerformanceMonitor

Монитор производительности системы.

### Классы

#### PerformanceMonitor
```python
class PerformanceMonitor:
    def __init__(self, sample_interval: float = 1.0, max_samples: int = 3600)
```

**Параметры:**
- `sample_interval` (float): Интервал сбора метрик в секундах
- `max_samples` (int): Максимальное количество образцов

**Методы:**

##### register_metric()
```python
def register_metric(self, name: str, metric_type: MetricType, description: Optional[str] = None) -> None
```
Регистрация новой метрики.

##### record_metric()
```python
def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None
```
Запись значения метрики.

##### start_monitoring()
```python
async def start_monitoring(self) -> None
```
Запуск мониторинга производительности.

##### stop_monitoring()
```python
async def stop_monitoring(self) -> None
```
Остановка мониторинга.

##### get_metrics()
```python
def get_metrics(self, name: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]
```
Получение метрик.

### Пример использования

```python
import asyncio
from utils.monitoring import PerformanceMonitor, MetricType

async def main():
    # Создание монитора
    monitor = PerformanceMonitor(sample_interval=2.0, max_samples=1800)
    
    # Регистрация кастомной метрики
    monitor.register_metric("trading.orders_per_second", MetricType.GAUGE, "Orders per second")
    
    # Запуск мониторинга
    await monitor.start_monitoring()
    
    # Запись кастомной метрики
    monitor.record_metric("trading.orders_per_second", 15.5, {"exchange": "binance"})
    
    # Получение метрик
    metrics = monitor.get_metrics(limit=100)
    print(metrics)
    
    # Остановка мониторинга
    await monitor.stop_monitoring()

asyncio.run(main())
```

---

## AlertManager

Менеджер алертов с правилами и обработчиками.

### Классы

#### AlertManager
```python
class AlertManager:
    def __init__(self, max_alerts: int = 1000)
```

**Параметры:**
- `max_alerts` (int): Максимальное количество алертов

**Методы:**

##### add_alert_handler()
```python
def add_alert_handler(self, level: AlertLevel, handler: Callable) -> None
```
Добавление обработчика алертов.

##### add_alert_rule()
```python
def add_alert_rule(self, rule: AlertRule) -> None
```
Добавление правила алерта.

##### create_alert()
```python
def create_alert(self, level: AlertLevel, message: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert
```
Создание нового алерта.

##### start_evaluation()
```python
async def start_evaluation(self) -> None
```
Запуск автоматической оценки алертов.

##### stop_evaluation()
```python
async def stop_evaluation(self) -> None
```
Остановка оценки алертов.

##### get_alerts()
```python
def get_alerts(self, level: Optional[AlertLevel] = None, acknowledged: Optional[bool] = None, limit: Optional[int] = None) -> List[Alert]
```
Получение алертов с фильтрацией.

##### acknowledge_alert()
```python
def acknowledge_alert(self, alert_id: str) -> bool
```
Подтверждение алерта.

##### resolve_alert()
```python
def resolve_alert(self, alert_id: str) -> bool
```
Разрешение алерта.

#### AlertRule
```python
class AlertRule:
    def __init__(self, name: str, condition: Callable, level: AlertLevel, message: str, source: str)
```

**Параметры:**
- `name` (str): Название правила
- `condition` (Callable): Функция условия
- `level` (AlertLevel): Уровень алерта
- `message` (str): Сообщение алерта
- `source` (str): Источник алерта

### Пример использования

```python
import asyncio
from utils.monitoring import AlertManager, AlertLevel, AlertRule

async def main():
    # Создание менеджера алертов
    alert_manager = AlertManager(max_alerts=500)
    
    # Обработчик алертов
    def handle_critical_alert(alert):
        print(f"CRITICAL ALERT: {alert.message}")
        # Отправка уведомления
    
    alert_manager.add_alert_handler(AlertLevel.CRITICAL, handle_critical_alert)
    
    # Правило алерта
    def check_memory_usage():
        import psutil
        return psutil.virtual_memory().percent > 90
    
    memory_rule = AlertRule(
        name="high_memory_usage",
        condition=check_memory_usage,
        level=AlertLevel.CRITICAL,
        message="Memory usage is above 90%",
        source="system"
    )
    
    alert_manager.add_alert_rule(memory_rule)
    
    # Запуск оценки
    await alert_manager.start_evaluation()
    
    # Создание алерта вручную
    alert_manager.create_alert(
        level=AlertLevel.WARNING,
        message="Trading volume is low",
        source="trading",
        metadata={"volume": 1000}
    )
    
    # Получение алертов
    critical_alerts = alert_manager.get_alerts(level=AlertLevel.CRITICAL)
    print(f"Critical alerts: {len(critical_alerts)}")
    
    # Остановка оценки
    await alert_manager.stop_evaluation()

asyncio.run(main())
```

---

## CacheManager

Менеджер для управления несколькими кэшами.

### Классы

#### CacheManager
```python
class CacheManager:
    def __init__(self)
```

**Методы:**

##### create_cache()
```python
def create_cache(self, name: str, **config) -> OptimizedCache
```
Создание нового кэша.

**Параметры:**
- `name` (str): Название кэша
- `**config`: Конфигурация кэша

**Возвращает:** Экземпляр OptimizedCache

##### get_cache()
```python
def get_cache(self, name: str) -> Optional[OptimizedCache]
```
Получение кэша по имени.

##### start_all()
```python
async def start_all(self) -> None
```
Запуск всех кэшей.

##### stop_all()
```python
async def stop_all(self) -> None
```
Остановка всех кэшей.

##### get_all_metrics()
```python
def get_all_metrics(self) -> Dict[str, Dict[str, Any]]
```
Получение метрик всех кэшей.

### Пример использования

```python
import asyncio
from utils.cache import CacheManager, CacheStrategy

async def main():
    # Создание менеджера кэшей
    cache_manager = CacheManager()
    
    # Создание кэшей
    user_cache = cache_manager.create_cache(
        "users",
        max_size=1000,
        ttl_seconds=1800,
        strategy=CacheStrategy.MEMORY
    )
    
    trading_cache = cache_manager.create_cache(
        "trading",
        max_size=5000,
        ttl_seconds=300,
        strategy=CacheStrategy.MEMORY,
        enable_compression=True
    )
    
    # Запуск всех кэшей
    await cache_manager.start_all()
    
    # Работа с кэшами
    await user_cache.set("user:123", {"name": "John"})
    await trading_cache.set("price:BTC", 50000)
    
    # Получение метрик всех кэшей
    all_metrics = cache_manager.get_all_metrics()
    print(all_metrics)
    
    # Остановка всех кэшей
    await cache_manager.stop_all()

asyncio.run(main())
```

---

## Интеграция с основным приложением

### Инициализация в main.py

```python
import asyncio
from utils.event_bus import EventBus
from utils.cache import CacheManager
from utils.monitoring import SystemMonitor

async def main():
    # Инициализация компонентов
    event_bus = EventBus(max_queue_size=10000, batch_size=100)
    cache_manager = CacheManager()
    system_monitor = SystemMonitor()
    
    # Создание кэшей
    market_data_cache = cache_manager.create_cache("market_data", ttl_seconds=60)
    strategy_cache = cache_manager.create_cache("strategy", ttl_seconds=300)
    
    # Запуск всех систем
    await event_bus.start()
    await cache_manager.start_all()
    await system_monitor.start()
    
    try:
        # Основная логика приложения
        await run_trading_bot(event_bus, cache_manager, system_monitor)
    finally:
        # Остановка всех систем
        await event_bus.stop()
        await cache_manager.stop_all()
        await system_monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Конфигурация в config.yaml

```yaml
# EventBus конфигурация
event_bus:
  max_queue_size: 10000
  batch_size: 100
  enable_metrics: true

# Кэш конфигурация
cache:
  market_data:
    max_size: 5000
    ttl_seconds: 60
    enable_compression: true
  strategy:
    max_size: 1000
    ttl_seconds: 300
    enable_compression: false

# Мониторинг конфигурация
monitoring:
  sample_interval: 1.0
  max_samples: 3600
  alerts:
    max_alerts: 1000
    cleanup_interval: 300
```

---

## Метрики и мониторинг

### Доступные метрики

#### EventBus метрики:
- `events_processed`: Количество обработанных событий
- `events_dropped`: Количество отброшенных событий
- `avg_processing_time`: Среднее время обработки
- `queue_sizes`: Размеры очередей по приоритетам
- `handler_cache_hits`: Попадания в кэш обработчиков
- `handler_cache_misses`: Промахи кэша обработчиков

#### Cache метрики:
- `hits`: Количество попаданий в кэш
- `misses`: Количество промахов кэша
- `sets`: Количество установок значений
- `deletes`: Количество удалений
- `evictions`: Количество вытеснений (LRU)
- `compressions`: Количество сжатий
- `decompressions`: Количество распаковок
- `total_size_bytes`: Общий размер кэша
- `avg_access_time`: Среднее время доступа

#### System метрики:
- `system.cpu_percent`: Использование CPU
- `system.memory_percent`: Использование памяти
- `system.disk_usage`: Использование диска
- `system.network_io`: Сетевая активность
- `system.process_count`: Количество процессов
- `system.thread_count`: Количество потоков

### Алерты по умолчанию

- **High CPU Usage**: CPU > 80% (WARNING)
- **High Memory Usage**: Memory > 85% (WARNING)
- **Low Disk Space**: Disk > 90% (ERROR)

---

## Лучшие практики

### EventBus
1. Используйте приоритеты для критических событий
2. Обрабатывайте исключения в обработчиках событий
3. Мониторьте размеры очередей
4. Используйте батчинг для повышения производительности

### Cache
1. Устанавливайте разумные TTL для разных типов данных
2. Используйте сжатие для больших объектов
3. Мониторьте hit rate кэша
4. Настройте размер кэша в зависимости от доступной памяти

### Monitoring
1. Настройте алерты для критических метрик
2. Регулярно проверяйте логи мониторинга
3. Используйте интеграции для внешних систем
4. Настройте автоматическое восстановление

### Производительность
1. Используйте асинхронные операции
2. Кэшируйте часто используемые данные
3. Мониторьте производительность в реальном времени
4. Оптимизируйте на основе метрик 