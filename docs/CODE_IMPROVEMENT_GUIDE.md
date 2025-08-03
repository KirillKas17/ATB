# Руководство по улучшению кода ATB

## Обзор

Данное руководство содержит рекомендации по устранению выявленных проблем в проекте ATB и улучшению качества кода.

## 1. Устранение дублирования кода

### 1.1 Репозитории

**Проблема:** 15+ классов репозиториев с дублирующейся логикой.

**Решение:** Использовать базовый класс `BaseRepositoryImpl`:

```python
from domain.repositories.base_repository_impl import BaseRepositoryImpl
from domain.types.common_types import EntityId

class UserRepository(BaseRepositoryImpl[User]):
    async def _save_entity_impl(self, entity: User) -> User:
        # Специфичная реализация сохранения
        pass
    
    async def _get_entity_by_id_impl(self, entity_id: EntityId) -> Optional[User]:
        # Специфичная реализация получения
        pass
    
    async def _update_entity_impl(self, entity: User) -> User:
        # Специфичная реализация обновления
        pass
    
    async def _delete_entity_impl(self, entity_id: EntityId) -> bool:
        # Специфичная реализация удаления
        pass
    
    def _get_entity_id(self, entity: User) -> EntityId:
        return entity.id
```

**Преимущества:**
- Единая логика кэширования, валидации, метрик
- Устранение дублирования CRUD операций
- Централизованная обработка ошибок

### 1.2 Сервисы

**Проблема:** Повторяющиеся паттерны в infrastructure/core/.

**Решение:** Использовать базовый класс `BaseServiceImpl`:

```python
from domain.services.base_service_impl import BaseServiceImpl
from domain.types.common_types import OperationResult

class MarketAnalysisService(BaseServiceImpl[MarketData]):
    async def _process_impl(self, data: MarketData) -> AnalysisResult:
        # Специфичная логика анализа
        pass
    
    async def _validate_input_specific(self, data: MarketData, context: str) -> tuple[List[str], List[str]]:
        # Специфичная валидация
        return [], []
```

### 1.3 Валидация

**Проблема:** Дублирующиеся проверки в разных слоях.

**Решение:** Использовать утилиты валидации:

```python
from shared.validation_utils import ValidationUtils
from domain.types.common_types import ValidationRule

class OrderValidator:
    def __init__(self):
        self.rules = [
            ValidationUtils.create_required_rule('symbol'),
            ValidationUtils.create_required_rule('quantity'),
            ValidationUtils.create_numeric_rule('quantity', min_value=0),
            ValidationUtils.create_rule('symbol', 'symbol', 'Invalid trading symbol')
        ]
    
    async def validate_order(self, order: Order) -> ValidationResult:
        return ValidationUtils.validate_entity(order, self.rules)
```

## 2. Исправление логических ошибок

### 2.1 Сравнения с None

**Проблема:** `if x == None` вместо `if x is None`

**Решение:** Всегда использовать `is` для сравнения с None:

```python
# ❌ Неправильно
if value == None:
    pass

# ✅ Правильно
if value is None:
    pass

# ✅ Также правильно для проверки на не None
if value is not None:
    pass
```

### 2.2 Обработка исключений

**Проблема:** Множество try/except без специфичной обработки.

**Решение:** Использовать утилиты обработки исключений:

```python
from shared.exception_utils import ExceptionUtils

# Декоратор для обработки исключений
@ExceptionUtils.handle_exceptions(context="market_data", reraise=True)
async def fetch_market_data(symbol: str) -> MarketData:
    # Логика получения данных
    pass

# Контекстный менеджер
async with ExceptionUtils.async_exception_context(context="order_processing"):
    await process_order(order)

# Специфичные обработчики
@ExceptionUtils.handle_validation_errors
async def validate_order(order: Order) -> ValidationResult:
    pass

@ExceptionUtils.handle_service_errors
async def process_payment(payment: Payment) -> PaymentResult:
    pass
```

### 2.3 Race Conditions

**Проблема:** Потенциальные race conditions в асинхронном коде.

**Решение:** Использовать утилиты для асинхронного кода:

```python
from shared.async_utils import resource_manager, task_manager, async_cache

# Блокировка ресурса
async with resource_manager.resource_lock("order_processing", owner="trading_system"):
    await process_order(order)

# Семафор для ограничения параллельных операций
async with resource_manager.resource_semaphore("api_calls", max_concurrent=5):
    await call_external_api()

# Кэширование с предотвращением race conditions
result = await async_cache.get("market_data")
if result is None:
    result = await fetch_market_data()
    await async_cache.set("market_data", result)
```

## 3. Улучшение типизации

### 3.1 Замена Any

**Проблема:** Избыточное использование Any (200+ вхождений).

**Решение:** Использовать специфичные типы:

```python
# ❌ Неправильно
def process_data(data: Any) -> Any:
    pass

# ✅ Правильно
from domain.types.enhanced_types import MarketDataType, OperationResult

def process_data(data: MarketDataType) -> OperationResult:
    pass

# Для гибких значений использовать FlexibleValue
from domain.types.enhanced_types import FlexibleValue

def process_config(config: Dict[str, FlexibleValue]) -> None:
    pass
```

### 3.2 Замена Dict[str, Any]

**Проблема:** Множество `Dict[str, Any]`.

**Решение:** Использовать TypedDict и протоколы:

```python
# ❌ Неправильно
def process_order(order: Dict[str, Any]) -> Dict[str, Any]:
    pass

# ✅ Правильно
from domain.types.enhanced_types import OrderDataType, OrderResultType

def process_order(order: OrderDataType) -> OrderResultType:
    pass

# Или использовать TypedDict
from typing import TypedDict

class OrderData(TypedDict):
    id: str
    symbol: str
    quantity: Amount
    price: Optional[Amount]
    side: str
    timestamp: Timestamp

def process_order(order: OrderData) -> OrderResult:
    pass
```

### 3.3 Обработка None

**Проблема:** Отсутствие проверок на None в критических местах.

**Решение:** Использовать Optional и проверки:

```python
from typing import Optional
from domain.types.common_types import EntityId

async def get_entity(entity_id: EntityId) -> Optional[Entity]:
    # Всегда проверяем на None
    if entity_id is None:
        return None
    
    entity = await repository.get_by_id(entity_id)
    if entity is None:
        logger.warning(f"Entity not found: {entity_id}")
        return None
    
    return entity

# Использование в коде
entity = await get_entity(entity_id)
if entity is None:
    raise EntityNotFoundError(f"Entity not found: {entity_id}")

# Обработка с значением по умолчанию
entity = await get_entity(entity_id) or create_default_entity()
```

## 4. Архитектурные улучшения

### 4.1 Следование DDD

**Принципы:**
- Разделение на слои (domain, application, infrastructure, interfaces)
- Использование доменных сущностей и value objects
- Инверсия зависимостей через протоколы

```python
# Domain layer
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.entities.order import Order

# Application layer
class OrderService:
    def __init__(self, order_repository: RepositoryProtocol[Order]):
        self.order_repository = order_repository

# Infrastructure layer
class PostgresOrderRepository(BaseRepositoryImpl[Order], RepositoryProtocol[Order]):
    pass
```

### 4.2 Использование протоколов

```python
from typing import Protocol

class MarketDataProvider(Protocol):
    async def get_market_data(self, symbol: str) -> MarketDataType: ...
    async def get_order_book(self, symbol: str) -> OrderBookDataType: ...

class TradingEngine(Protocol):
    async def place_order(self, order: OrderDataType) -> OperationResult: ...
    async def cancel_order(self, order_id: str) -> OperationResult: ...
```

### 4.3 Dependency Injection

```python
from application.di_container_refactored import DIContainer

# Регистрация зависимостей
container = DIContainer()
container.register(MarketDataProvider, BinanceMarketDataProvider)
container.register(TradingEngine, BinanceTradingEngine)
container.register(OrderService, OrderService)

# Использование
order_service = container.resolve(OrderService)
```

## 5. Рекомендации по производительности

### 5.1 Кэширование

```python
from shared.async_utils import async_cache

# Кэширование результатов
@async_cache.cache_result(ttl=300)
async def get_market_data(symbol: str) -> MarketDataType:
    pass
```

### 5.2 Асинхронная обработка

```python
import asyncio

# Параллельная обработка
async def process_multiple_orders(orders: List[OrderDataType]) -> List[OperationResult]:
    tasks = [process_order(order) for order in orders]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 5.3 Ограничение ресурсов

```python
from shared.async_utils import resource_manager

# Ограничение параллельных API вызовов
async with resource_manager.resource_semaphore("api_calls", max_concurrent=10):
    await call_external_api()
```

## 6. Тестирование

### 6.1 Unit тесты

```python
import pytest
from unittest.mock import Mock

class TestOrderService:
    def test_create_order_success(self):
        # Arrange
        mock_repository = Mock()
        service = OrderService(mock_repository)
        order_data = OrderData(
            id="order123",
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            side="BUY",
            timestamp=datetime.now()
        )
        
        # Act
        result = await service.create_order(order_data)
        
        # Assert
        assert result.success is True
        mock_repository.save.assert_called_once()
```

### 6.2 Integration тесты

```python
class TestTradingSystemIntegration:
    async def test_complete_trading_cycle(self):
        # Arrange
        system = ImprovedTradingSystem(config)
        
        # Act
        result = await system.execute_trading_cycle()
        
        # Assert
        assert result.success is True
        assert result.data is not None
```

## 7. Мониторинг и логирование

### 7.1 Метрики

```python
from shared.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

@monitor.track_operation("order_processing")
async def process_order(order: OrderDataType) -> OperationResult:
    pass
```

### 7.2 Логирование

```python
from loguru import logger

logger = logger.bind(module="OrderService", operation="create_order")

async def create_order(order_data: OrderDataType) -> OperationResult:
    logger.info(f"Creating order: {order_data['symbol']}")
    try:
        result = await repository.save(order_data)
        logger.success(f"Order created successfully: {result['id']}")
        return result
    except Exception as e:
        logger.error(f"Failed to create order: {e}")
        raise
```

## 8. Безопасность

### 8.1 Валидация входных данных

```python
from shared.validation_utils import ValidationUtils

class SecurityValidator:
    def __init__(self):
        self.rules = [
            ValidationUtils.create_rule('api_key', 'pattern', r'^[A-Za-z0-9]{32}$'),
            ValidationUtils.create_rule('amount', 'positive'),
            ValidationUtils.create_rule('symbol', 'symbol')
        ]
    
    async def validate_request(self, request: Dict[str, FlexibleValue]) -> ValidationResult:
        return ValidationUtils.validate_entity(request, self.rules)
```

### 8.2 Шифрование чувствительных данных

```python
from shared.security import SecurityUtils

# Шифрование API ключей
encrypted_key = SecurityUtils.encrypt(api_key, master_key)

# Хеширование паролей
hashed_password = SecurityUtils.hash_password(password)
```

## 9. Заключение

Следование этим рекомендациям поможет:

1. **Устранить дублирование кода** - использование базовых классов и утилит
2. **Исправить логические ошибки** - правильная обработка исключений и сравнений
3. **Улучшить типизацию** - замена Any на специфичные типы
4. **Повысить производительность** - кэширование и асинхронная обработка
5. **Улучшить безопасность** - валидация и шифрование
6. **Упростить тестирование** - четкое разделение ответственности

Регулярно проводите code review и используйте статические анализаторы кода для поддержания качества. 