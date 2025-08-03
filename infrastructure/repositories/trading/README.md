# Trading Repository Module

Промышленный модуль торгового репозитория с полной декомпозицией и строгой типизацией.

## Архитектура

Модуль разделен на логические компоненты согласно принципам DDD и SOLID:

### Основные компоненты

- **`trading_repository.py`** - Основные реализации репозиториев (InMemory, PostgreSQL)
- **`analyzers.py`** - Анализаторы паттернов и ликвидности
- **`services.py`** - Сервисы для расчета метрик и валидации
- **`models.py`** - Модели данных для хранения
- **`converters.py`** - Конвертеры между доменными объектами и моделями
- **`cache.py`** - Система кэширования с LRU и TTL
- **`validators.py`** - Валидаторы данных и бизнес-правил
- **`events.py`** - Система событий и шина событий
- **`config.py`** - Конфигурация и менеджер конфигурации

## Использование

### Базовое использование

```python
from infrastructure.repositories.trading import (
    InMemoryTradingRepository,
    TradingRepositoryServices,
    TradingEventBus
)

# Создание репозитория
repo = InMemoryTradingRepository()

# Создание сервисов
services = TradingRepositoryServices()

# Создание шины событий
event_bus = TradingEventBus()
await event_bus.start()

# Добавление ордера
result = await repo.add_order(order)
if result.is_success():
    order = result.data
    print(f"Order added: {order.id}")
```

### С кэшированием

```python
from infrastructure.repositories.trading import (
    InMemoryTradingRepository,
    TradingRepositoryCache
)

# Создание кэша
cache = TradingRepositoryCache()
await cache.start()

# Получение ордера с кэшированием
cached_order = await cache.get_order(order_id)
if cached_order:
    return cached_order

# Если нет в кэше, получаем из репозитория
order_result = await repo.get_order(order_id)
if order_result.is_success():
    await cache.set_order(order_id, order_result.data)
    return order_result.data
```

### С валидацией

```python
from infrastructure.repositories.trading import (
    TradingDataValidator,
    TradingBusinessRuleValidator
)

# Создание валидаторов
data_validator = TradingDataValidator()
business_validator = TradingBusinessRuleValidator()

# Валидация данных ордера
is_valid, errors = data_validator.validate_order_data(order_data)
if not is_valid:
    print(f"Validation errors: {errors}")
    return

# Валидация бизнес-правил
is_valid, errors = business_validator.validate_order_business_rules(
    order_data, account_data
)
if not is_valid:
    print(f"Business rule violations: {errors}")
    return
```

### С событиями

```python
from infrastructure.repositories.trading import (
    TradingEventBus,
    TradingEventFactory,
    EventType
)

# Создание шины событий
event_bus = TradingEventBus()
await event_bus.start()

# Создание фабрики событий
event_factory = TradingEventFactory()

# Подписка на события
async def on_order_created(event):
    print(f"Order created: {event.entity_id}")

event_bus.subscribe(EventType.ORDER_CREATED, on_order_created)

# Публикация события
order_event = event_factory.create_order_created_event(order)
await event_bus.publish(order_event)
```

### С конфигурацией

```python
from infrastructure.repositories.trading import TradingRepositoryConfigManager

# Создание менеджера конфигурации
config_manager = TradingRepositoryConfigManager("config.yaml")

# Загрузка конфигурации
config_manager.load_from_file("config.yaml")

# Получение конфигурации
config = config_manager.get_config()

# Валидация конфигурации
errors = config_manager.validate_config()
if errors:
    print(f"Configuration errors: {errors}")

# Создание репозитория с конфигурацией
repo = InMemoryTradingRepository()
if config.cache.enabled:
    cache = TradingRepositoryCache()
    await cache.start()
```

## Конфигурация

Пример файла конфигурации `config.yaml`:

```yaml
name: "TradingRepository"
version: "1.0.0"
environment: "production"

cache:
  enabled: true
  max_size: 1000
  default_ttl_minutes: 5
  cleanup_interval_seconds: 60

validation:
  enabled: true
  strict_mode: false
  validate_business_rules: true
  min_quantity: "0.00000001"
  max_quantity: "999999999.99999999"

events:
  enabled: true
  max_history_size: 10000
  publish_async: true
  log_events: true

database:
  type: "postgres"
  connection_string: "postgresql://user:pass@localhost/trading"
  pool_size: 10
  max_overflow: 20

logging:
  level: "INFO"
  file_path: "/var/log/trading_repository.log"
  max_file_size_mb: 100
  backup_count: 5

performance:
  enable_metrics: true
  batch_size: 100
  max_concurrent_operations: 50

security:
  enable_audit_log: true
  encrypt_sensitive_data: false
  rate_limit_enabled: true
  max_requests_per_minute: 1000
```

## Особенности реализации

### Строгая типизация

Все методы имеют строгую типизацию с использованием `typing`:

```python
async def add_order(self, order: Order) -> RepositoryResult[Order]:
    """Добавление ордера с полной типизацией."""
    pass
```

### Обработка ошибок

Используется система `RepositoryResult` для обработки ошибок:

```python
result = await repo.add_order(order)
if result.is_success():
    order = result.data
else:
    error = result.error
    print(f"Error: {error.message}")
```

### Кэширование

Реализована многоуровневая система кэширования:

- LRU кэш с TTL
- Раздельные кэши для разных типов данных
- Автоматическая очистка истекших записей
- Инвалидация связанных кэшей

### Валидация

Двухуровневая система валидации:

1. **Валидация данных** - проверка форматов и типов
2. **Валидация бизнес-правил** - проверка бизнес-логики

### События

Полная система событий:

- Типизированные события для всех операций
- Асинхронная публикация
- История событий
- Подписка на конкретные типы событий

### Производительность

Оптимизации для высокой производительности:

- Пакетные операции
- Соединения с пулом
- Асинхронная обработка
- Метрики производительности

## Тестирование

```python
import pytest
from infrastructure.repositories.trading import InMemoryTradingRepository

@pytest.fixture
def trading_repo():
    return InMemoryTradingRepository()

async def test_add_order(trading_repo):
    # Создание тестового ордера
    order = create_test_order()
    
    # Добавление ордера
    result = await trading_repo.add_order(order)
    
    # Проверка результата
    assert result.is_success()
    assert result.data.id == order.id
```

## Мониторинг

```python
# Получение статистики кэша
cache_stats = await cache.get_all_stats()
print(f"Cache hit rate: {cache_stats['orders']['usage_percent']}%")

# Получение статистики событий
event_stats = event_bus.get_stats()
print(f"Total events: {event_stats['total_events']}")

# Получение метрик репозитория
metrics = await repo.get_trading_metrics()
print(f"Total orders: {metrics['total_orders']}")
```

## Безопасность

- Валидация всех входных данных
- Аудит всех операций
- Маскирование персональных данных
- Ограничение скорости запросов
- Шифрование чувствительных данных

## Расширение

Модуль легко расширяется:

1. Добавление новых типов репозиториев
2. Создание новых валидаторов
3. Добавление новых типов событий
4. Расширение конфигурации

Все компоненты следуют принципам SOLID и легко тестируются. 