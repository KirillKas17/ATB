# Протоколы домена - Полная документация

## Обзор

Протоколы домена представляют собой промышленный уровень абстракций для взаимодействия с внешними системами в торговой платформе ATB. Система построена на принципах Domain-Driven Design (DDD) и обеспечивает надежное, безопасное и производительное взаимодействие с биржами, ML-моделями, стратегиями и репозиториями.

## Архитектура

### Слои протоколов

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaces Layer                         │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                    Domain Protocols Layer                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   Exchange  │ │     ML      │ │  Strategy   │ │Repository│ │
│  │  Protocol   │ │  Protocol   │ │  Protocol   │ │Protocol │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                     │
└─────────────────────────────────────────────────────────────┘
```

### Основные компоненты

1. **ExchangeProtocol** - взаимодействие с биржами
2. **MLProtocol** - работа с ML-моделями
3. **StrategyProtocol** - управление торговыми стратегиями
4. **RepositoryProtocol** - работа с данными
5. **Декораторы** - функциональные улучшения
6. **Валидаторы** - проверка данных
7. **Мониторинг** - отслеживание состояния
8. **Производительность** - оптимизация
9. **Безопасность** - защита данных

## Детальное описание протоколов

### 1. ExchangeProtocol

Протокол для взаимодействия с криптовалютными биржами.

#### Основные интерфейсы:

```python
class ExchangeProtocol(Protocol):
    async def connect(self, config: Dict[str, Any]) -> bool: ...
    async def disconnect(self) -> bool: ...
    async def is_connected(self) -> bool: ...
    async def get_market_data(self, symbol: str) -> Dict[str, Any]: ...
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_order_status(self, order_id: str) -> Dict[str, Any]: ...
    async def get_positions(self) -> List[Dict[str, Any]]: ...
    async def get_balance(self) -> Dict[str, Any]: ...
```

#### Пример использования:

```python
from domain.protocols import ExchangeProtocol, ExampleExchangeProtocol

# Создание экземпляра протокола
exchange = ExampleExchangeProtocol()

# Подключение к бирже
config = {
    "exchange_name": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "sandbox": True
}

connected = await exchange.connect(config)
if connected:
    # Получение рыночных данных
    market_data = await exchange.get_market_data("BTC/USDT")
    
    # Создание ордера
    order = await exchange.create_order({
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.001,
        "price": 50000.0
    })
```

### 2. MLProtocol

Протокол для работы с машинным обучением.

#### Основные интерфейсы:

```python
class MLProtocol(Protocol):
    async def train_model(self, config: Dict[str, Any]) -> Dict[str, Any]: ...
    async def load_model(self, model_id: str) -> bool: ...
    async def predict(self, model_id: str, features: List[float]) -> Dict[str, Any]: ...
    async def evaluate_model(self, model_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    async def save_model(self, model_id: str, path: str) -> bool: ...
    async def delete_model(self, model_id: str) -> bool: ...
```

#### Пример использования:

```python
from domain.protocols import MLProtocol, ExampleMLProtocol

# Создание ML протокола
ml = ExampleMLProtocol()

# Обучение модели
model_config = {
    "name": "price_predictor",
    "model_type": "regression",
    "trading_pair": "BTC/USDT",
    "prediction_type": "price",
    "features": ["price", "volume", "rsi", "macd"],
    "target": "next_price"
}

model = await ml.train_model(model_config)

# Предсказание
features = [50000.0, 1000.0, 0.65, 0.2]
prediction = await ml.predict(model["id"], features)
print(f"Predicted price: {prediction['prediction']}")
```

### 3. StrategyProtocol

Протокол для управления торговыми стратегиями.

#### Основные интерфейсы:

```python
class StrategyProtocol(Protocol):
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    async def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]: ...
    async def calculate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]: ...
    async def optimize_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]: ...
```

#### Пример использования:

```python
from domain.protocols import StrategyProtocol

class TrendFollowingStrategy(StrategyProtocol):
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Анализ тренда
        prices = market_data["prices"]
        trend = "bullish" if prices[-1] > prices[-20] else "bearish"
        
        return {
            "trend": trend,
            "strength": 0.8,
            "confidence": 0.75
        }
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals = []
        if analysis["trend"] == "bullish" and analysis["confidence"] > 0.7:
            signals.append({
                "type": "buy",
                "strength": analysis["strength"],
                "confidence": analysis["confidence"]
            })
        
        return signals
```

### 4. RepositoryProtocol

Протокол для работы с данными.

#### Основные интерфейсы:

```python
class RepositoryProtocol(Protocol):
    async def create(self, collection: str, data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def read(self, collection: str, record_id: str) -> Optional[Dict[str, Any]]: ...
    async def update(self, collection: str, record_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    async def delete(self, collection: str, record_id: str) -> bool: ...
    async def query(self, collection: str, filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
    async def begin_transaction(self) -> str: ...
    async def commit_transaction(self, transaction_id: str) -> bool: ...
    async def rollback_transaction(self, transaction_id: str) -> bool: ...
```

## Декораторы и утилиты

### Декораторы производительности

```python
from domain.protocols import (
    retry, timeout, cache, metrics, 
    circuit_breaker, rate_limit, log_operation
)

@retry(max_attempts=3, base_delay=1.0)
@timeout(timeout=30.0)
@cache(ttl=300)
@metrics(enabled=True)
@circuit_breaker(failure_threshold=5)
@rate_limit(max_calls=100, time_window=60)
@log_operation(level=logging.INFO)
async def fetch_market_data(symbol: str) -> Dict[str, Any]:
    # Реализация получения рыночных данных
    pass
```

### Валидаторы

```python
from domain.protocols import (
    validate_exchange_config, validate_model_config,
    validate_strategy_config, validate_repository_config
)

# Валидация конфигурации биржи
exchange_config = {
    "exchange_name": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret"
}

result = validate_exchange_config(exchange_config)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")
```

### Мониторинг

```python
from domain.protocols import ProtocolMonitor

# Создание монитора
monitor = ProtocolMonitor()

# Мониторинг здоровья протокола
health = await monitor.monitor_protocol_health("exchange_protocol")

# Запись метрик
await monitor.record_protocol_metric(
    "exchange_protocol", 
    "order_creation_time", 
    0.15, 
    metric_type=MetricType.HISTOGRAM
)

# Создание алерта
alert = await monitor.create_protocol_alert(
    "exchange_protocol",
    AlertLevel.WARNING,
    "High latency detected"
)
```

## Безопасность

### Шифрование данных

```python
from domain.protocols import SecurityManager

security = SecurityManager()

# Шифрование чувствительных данных
sensitive_data = {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret"
}

encrypted_data = await security.encrypt_sensitive_data(sensitive_data)
decrypted_data = await security.decrypt_sensitive_data(encrypted_data)
```

### Аутентификация и авторизация

```python
# Регистрация пользователя
await security.auth_manager.register_user(
    username="trader1",
    password="secure_password",
    email="trader@example.com",
    permissions={Permission.READ, Permission.WRITE}
)

# Аутентификация
context = await security.authenticate_user(
    username="trader1",
    password="secure_password",
    ip_address="192.168.1.100",
    user_agent="TradingBot/1.0"
)

# Проверка разрешений
await security.authz_manager.require_permission(
    context, "market_data", Permission.READ
)
```

## Производительность

### Профилирование

```python
from domain.protocols import profile_performance, benchmark_performance

@profile_performance("order_creation")
@benchmark_performance("order_creation", iterations=1000)
async def create_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    # Реализация создания ордера
    pass
```

### Оптимизация

```python
from domain.protocols import optimize_performance

@optimize_performance("data_processing")
async def process_market_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Обработка рыночных данных
    pass
```

## Интеграционные тесты

### Полная торговая система

```python
from domain.protocols import TradingSystemIntegration

async def test_complete_trading_system():
    # Создание системы
    system = TradingSystemIntegration()
    
    # Инициализация
    await system.initialize()
    
    # Запуск
    await system.start()
    
    # Торговый цикл
    result = await system.run_trading_cycle()
    
    # Проверка результатов
    assert result["status"] == "success"
    assert "market_data" in result
    assert "signals" in result
    
    # Остановка
    await system.stop()
```

## Конфигурация

### Пример конфигурации

```yaml
# config/protocols.yaml
exchange:
  name: "binance"
  api_key: "${BINANCE_API_KEY}"
  api_secret: "${BINANCE_API_SECRET}"
  sandbox: true
  timeout: 30
  retry_attempts: 3

ml:
  model_storage_path: "./models"
  training_timeout: 3600
  prediction_cache_ttl: 300

strategy:
  risk_level: "medium"
  max_position_size: "0.01"
  stop_loss: "2%"
  take_profit: "5%"

repository:
  connection_string: "postgresql://user:pass@localhost/trading"
  pool_size: 10
  cache_enabled: true
  cache_ttl: 300

monitoring:
  metrics_enabled: true
  health_check_interval: 60
  alert_levels:
    - "error"
    - "warning"

security:
  encryption_enabled: true
  session_timeout: 86400
  max_failed_attempts: 5
```

## Лучшие практики

### 1. Обработка ошибок

```python
from domain.exceptions.protocol_exceptions import (
    ProtocolError, ProtocolConnectionError, ProtocolTimeoutError
)

async def safe_protocol_operation():
    try:
        result = await protocol.operation()
        return result
    except ProtocolConnectionError:
        # Логика переподключения
        await reconnect()
    except ProtocolTimeoutError:
        # Логика повторной попытки
        return await retry_operation()
    except ProtocolError as e:
        # Общая обработка ошибок протокола
        logging.error(f"Protocol error: {e}")
        raise
```

### 2. Использование декораторов

```python
@retry(max_attempts=3)
@timeout(timeout=30.0)
@cache(ttl=300)
@metrics(enabled=True)
@require_authentication("market_data", Permission.READ)
@encrypt_sensitive_fields(["api_key", "api_secret"])
@audit_security_events(AuditEvent.DATA_ACCESS, "market_data", "read")
async def fetch_market_data(symbol: str, security_context: SecurityContext) -> Dict[str, Any]:
    # Реализация
    pass
```

### 3. Мониторинг и алерты

```python
async def monitor_protocol_health():
    monitor = ProtocolMonitor()
    
    # Проверка здоровья всех протоколов
    health = await monitor.get_overall_health()
    
    if health == HealthState.UNHEALTHY:
        await monitor.create_protocol_alert(
            "system",
            AlertLevel.CRITICAL,
            "System health is unhealthy"
        )
    
    # Получение метрик
    metrics = await monitor.get_protocol_metrics("exchange_protocol")
    
    # Анализ производительности
    if metrics.get("avg_response_time", 0) > 1.0:
        await monitor.create_protocol_alert(
            "exchange_protocol",
            AlertLevel.WARNING,
            "High response time detected"
        )
```

### 4. Безопасность

```python
async def secure_protocol_operation():
    security = SecurityManager()
    
    # Аутентификация
    context = await security.authenticate_user(
        username="user",
        password="password",
        ip_address="192.168.1.100",
        user_agent="TradingBot/1.0"
    )
    
    # Проверка разрешений
    await security.authz_manager.require_permission(
        context, "trading", Permission.WRITE
    )
    
    # Шифрование данных
    sensitive_data = {"api_key": "secret_key"}
    encrypted_data = await security.encrypt_sensitive_data(sensitive_data)
    
    # Аудит
    await security.audit_manager.log_event(
        AuditEvent.DATA_ACCESS,
        context,
        "trading_data",
        "read",
        "success"
    )
```

## Заключение

Система протоколов домена обеспечивает:

1. **Надежность** - через retry механизмы, circuit breaker и обработку ошибок
2. **Производительность** - через кэширование, профилирование и оптимизацию
3. **Безопасность** - через шифрование, аутентификацию и аудит
4. **Мониторинг** - через метрики, health checks и алерты
5. **Валидацию** - через проверку данных и конфигураций
6. **Масштабируемость** - через модульную архитектуру и DDD принципы

Все протоколы готовы для промышленного использования и обеспечивают высокий уровень надежности и производительности для торговых операций. 