# Промышленные протоколы домена

## Обзор

Промышленные протоколы домена обеспечивают типобезопасное взаимодействие между слоями архитектуры алготрейдинговой системы. Они определяют контракты для всех внешних зависимостей и обеспечивают единообразный интерфейс для работы с биржами, ML моделями, стратегиями и репозиториями.

## Архитектура

```
domain/protocols/
├── __init__.py              # Экспорты всех протоколов
├── exchange_protocol.py     # Протоколы для работы с биржами
├── ml_protocol.py          # Протоколы для машинного обучения
├── strategy_protocol.py    # Протоколы для торговых стратегий
├── repository_protocol.py  # Протоколы для репозиториев
├── utils.py                # Утилиты для протоколов
├── examples.py             # Примеры использования
└── exceptions.py           # Специализированные исключения
```

## Основные протоколы

### 1. ExchangeProtocol

Протокол для работы с криптовалютными биржами.

#### Основные возможности:
- Управление соединением с биржей
- Получение рыночных данных (OHLCV, ордербук, тикеры)
- Создание и управление ордерами
- Получение баланса и позиций
- Обработка ошибок и восстановление

#### Пример использования:

```python
from domain.protocols import ExchangeProtocol

class BinanceExchange(ExchangeProtocol):
    async def initialize(self, config: Dict[str, Any]) -> bool:
        # Инициализация подключения к Binance
        pass
    
    async def get_market_data(self, symbol: Symbol, timeframe: str, limit: int) -> List[MarketData]:
        # Получение рыночных данных
        pass
    
    async def create_order(self, symbol: Symbol, side: OrderSide, order_type: OrderType, 
                          quantity: VolumeValue, price: Optional[PriceValue] = None) -> Order:
        # Создание ордера
        pass
```

### 2. MLProtocol

Протокол для машинного обучения в алготрейдинге.

#### Основные возможности:
- Создание и обучение ML моделей
- Выполнение предсказаний
- Оценка производительности моделей
- Оптимизация гиперпараметров
- Управление жизненным циклом моделей

#### Пример использования:

```python
from domain.protocols import MLProtocol

class MLService(MLProtocol):
    async def create_model(self, name: str, model_type: ModelType, 
                          trading_pair: Symbol, prediction_type: PredictionType,
                          hyperparameters: Dict[str, Any], features: List[str], 
                          target: str) -> Model:
        # Создание модели
        pass
    
    async def train_model(self, model_id: ModelId, training_data: pd.DataFrame) -> Model:
        # Обучение модели
        pass
    
    async def predict(self, model_id: ModelId, features: Dict[str, Any]) -> Optional[Prediction]:
        # Выполнение предсказания
        pass
```

### 3. StrategyProtocol

Протокол для торговых стратегий.

#### Основные возможности:
- Анализ рыночных данных
- Генерация торговых сигналов
- Исполнение стратегий
- Управление рисками
- Мониторинг производительности

#### Пример использования:

```python
from domain.protocols import StrategyProtocol

class TrendFollowingStrategy(StrategyProtocol):
    async def analyze_market(self, market_data: pd.DataFrame, 
                           strategy_type: StrategyType) -> Dict[str, Any]:
        # Анализ рынка
        pass
    
    async def generate_signal(self, strategy_id: StrategyId, 
                            market_data: pd.DataFrame) -> Optional[Signal]:
        # Генерация сигнала
        pass
    
    async def execute_strategy(self, strategy_id: StrategyId, signal: Signal) -> bool:
        # Исполнение стратегии
        pass
```

### 4. RepositoryProtocol

Базовый протокол для репозиториев.

#### Основные возможности:
- CRUD операции с доменными сущностями
- Транзакционность
- Кэширование
- Пакетные операции

#### Пример использования:

```python
from domain.protocols import RepositoryProtocol

class OrderRepository(RepositoryProtocol[Order]):
    async def save(self, entity: Order) -> Order:
        # Сохранение ордера
        pass
    
    async def get_by_id(self, entity_id: OrderId) -> Optional[Order]:
        # Получение ордера по ID
        pass
    
    async def get_all(self, limit: Optional[int] = None, 
                     filters: Optional[Dict[str, Any]] = None) -> List[Order]:
        # Получение всех ордеров
        pass
```

## Специализированные репозитории

### TradingRepositoryProtocol
- Управление ордерами и сделками
- Отслеживание исполнения
- Анализ торговой активности

### PortfolioRepositoryProtocol
- Управление портфелями
- Отслеживание позиций
- Расчет P&L

### StrategyRepositoryProtocol
- Конфигурация стратегий
- Отслеживание производительности
- Управление жизненным циклом

### MarketRepositoryProtocol
- Хранение рыночных данных
- OHLCV данные
- Ордербуки

### RiskRepositoryProtocol
- Управление профилями риска
- Настройка лимитов
- Мониторинг рисков

### MLRepositoryProtocol
- Версионирование моделей
- Отслеживание производительности
- Управление предсказаниями

## Утилиты

### Декораторы

#### @retry_on_error
Повторные попытки при ошибках.

```python
@retry_on_error(max_retries=3, delay=1.0, backoff_factor=2.0)
async def risky_operation():
    # Операция, которая может завершиться ошибкой
    pass
```

#### @timeout
Установка таймаута для операций.

```python
@timeout(30.0)
async def long_operation():
    # Операция с таймаутом 30 секунд
    pass
```

### Валидация

#### validate_symbol
Валидация торгового символа.

```python
symbol = validate_symbol("BTCUSDT")
assert isinstance(symbol, Symbol)
```

#### validate_entity_id
Валидация ID сущности.

```python
entity_id = validate_entity_id("123e4567-e89b-12d3-a456-426614174000")
assert isinstance(entity_id, UUID)
```

### Кэширование

#### ProtocolCache
Кэш для протоколов с TTL.

```python
cache = ProtocolCache(ttl_seconds=300)
cache.set("key", "value")
value = cache.get("key")
```

### Метрики

#### ProtocolMetrics
Сбор метрик производительности протоколов.

```python
metrics = ProtocolMetrics()
metrics.record_operation("get_market_data", 1.5, success=True)
stats = metrics.get_operation_stats("get_market_data")
```

## Обработка ошибок

### Специализированные исключения

#### ExchangeError
Ошибки биржи (подключение, аутентификация, лимиты).

#### ModelError
Ошибки ML моделей (не найдена, не готова, ошибка обучения).

#### StrategyError
Ошибки стратегий (не найдена, ошибка исполнения, недостаток капитала).

#### RepositoryError
Ошибки репозиториев (сущность не найдена, ошибка сохранения).

### Логирование

#### log_operation
Логирование операций протоколов.

```python
log_operation("create_order", "order", order_id, {"symbol": "BTCUSDT"})
```

#### log_error
Логирование ошибок протоколов.

```python
log_error(error, "create_order", "order", order_id)
```

## Типизация

### Основные типы

```python
from domain.types import (
    OrderId, TradeId, PositionId, StrategyId, ModelId, PredictionId,
    PortfolioId, RiskProfileId, Symbol, TradingPair, PriceValue, VolumeValue
)
```

### TypedDict

```python
class ExchangeConfig(TypedDict):
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    sandbox: NotRequired[bool]
```

### Dataclasses

```python
@dataclass
class MarketDataSnapshot:
    symbol: Symbol
    timestamp: TimestampValue
    open: PriceValue
    high: PriceValue
    low: PriceValue
    close: PriceValue
    volume: VolumeValue
```

## Примеры использования

### Полный цикл торговли

```python
async def trading_cycle():
    # 1. Инициализация биржи
    exchange = BinanceExchange()
    await exchange.initialize(config)
    
    # 2. Получение рыночных данных
    market_data = await exchange.get_market_data(Symbol("BTCUSDT"), "1h", 100)
    
    # 3. Анализ стратегией
    strategy = TrendFollowingStrategy()
    analysis = await strategy.analyze_market(market_data, StrategyType.TREND_FOLLOWING)
    
    # 4. Генерация сигнала
    signal = await strategy.generate_signal(strategy_id, market_data)
    
    if signal and signal.is_actionable:
        # 5. Создание ордера
        order = await exchange.create_order(
            symbol=signal.trading_pair,
            side=signal.signal_type,
            order_type=OrderType.MARKET,
            quantity=signal.quantity
        )
        
        # 6. Размещение ордера
        placed_order = await exchange.place_order(order)
        
        # 7. Сохранение в репозиторий
        repository = OrderRepository()
        await repository.save(placed_order)
```

### ML пайплайн

```python
async def ml_pipeline():
    # 1. Создание модели
    ml_service = MLService()
    model = await ml_service.create_model(
        name="BTC Predictor",
        model_type=ModelType.RANDOM_FOREST,
        trading_pair=Symbol("BTCUSDT"),
        prediction_type=PredictionType.PRICE,
        hyperparameters={"n_estimators": 100},
        features=["price", "volume", "rsi"],
        target="next_price"
    )
    
    # 2. Подготовка данных
    data_service = DataService()
    training_data = await data_service.prepare_training_data(
        symbol=Symbol("BTCUSDT"),
        features=["price", "volume", "rsi"],
        target="next_price"
    )
    
    # 3. Обучение модели
    trained_model = await ml_service.train_model(model.id, training_data)
    
    # 4. Оценка модели
    test_data = await data_service.prepare_test_data(symbol=Symbol("BTCUSDT"))
    metrics = await ml_service.evaluate_model(trained_model.id, test_data)
    
    # 5. Активация модели
    if metrics["accuracy"] > 0.7:
        await ml_service.activate_model(trained_model.id)
        
        # 6. Предсказание
        current_features = {"price": 50000, "volume": 1000, "rsi": 50}
        prediction = await ml_service.predict(trained_model.id, current_features)
        
        if prediction and prediction.confidence > 0.8:
            # Использование предсказания для торговли
            pass
```

## Лучшие практики

### 1. Использование декораторов
Всегда используйте декораторы для обработки ошибок и таймаутов:

```python
@retry_on_error(max_retries=3, delay=1.0)
@timeout(30.0)
async def exchange_operation():
    # Операция с биржей
    pass
```

### 2. Валидация входных данных
Всегда валидируйте входные данные:

```python
async def create_order(self, symbol: Symbol, quantity: VolumeValue):
    validate_symbol(symbol)
    if quantity <= 0:
        raise ValidationError("Quantity must be positive")
```

### 3. Логирование операций
Логируйте все важные операции:

```python
log_operation("create_order", "order", order_id, {"symbol": str(symbol)})
```

### 4. Обработка ошибок
Используйте специализированные исключения:

```python
try:
    await exchange.create_order(order)
except InsufficientBalanceError as e:
    logger.error(f"Insufficient balance: {e}")
    # Обработка ошибки
except ExchangeRateLimitError as e:
    logger.warning(f"Rate limit exceeded, retry after {e.retry_after} seconds")
    # Ожидание и повторная попытка
```

### 5. Кэширование
Используйте кэширование для часто запрашиваемых данных:

```python
cache_key = f"market_data_{symbol}_{timeframe}"
cached_data = self.cache.get(cache_key)
if cached_data:
    return cached_data

data = await self.fetch_market_data(symbol, timeframe)
self.cache.set(cache_key, data)
return data
```

### 6. Метрики
Собирайте метрики производительности:

```python
start_time = datetime.now()
try:
    result = await operation()
    duration = (datetime.now() - start_time).total_seconds()
    self.metrics.record_operation("operation_name", duration, success=True)
    return result
except Exception as e:
    duration = (datetime.now() - start_time).total_seconds()
    self.metrics.record_operation("operation_name", duration, success=False, error_type=type(e).__name__)
    raise
```

## Тестирование

### Unit тесты
Создавайте моки для протоколов в unit тестах:

```python
class MockExchangeProtocol(ExchangeProtocol):
    async def get_market_data(self, symbol, timeframe, limit):
        return [MarketData(...)]
    
    async def create_order(self, symbol, side, order_type, quantity, price):
        return Order(...)
```

### Integration тесты
Тестируйте интеграцию между протоколами:

```python
async def test_exchange_ml_integration():
    exchange = MockExchangeProtocol()
    ml_service = MockMLProtocol()
    
    # Тест полного цикла
    market_data = await exchange.get_market_data(...)
    prediction = await ml_service.predict(...)
    order = await exchange.create_order(...)
```

### Performance тесты
Тестируйте производительность протоколов:

```python
async def test_cache_performance():
    cache = ProtocolCache()
    start_time = datetime.now()
    
    for i in range(1000):
        cache.set(f"key_{i}", f"value_{i}")
    
    duration = (datetime.now() - start_time).total_seconds()
    assert duration < 1.0  # Менее 1 секунды
```

## Заключение

Промышленные протоколы домена обеспечивают:

1. **Типобезопасность** - строгая типизация всех операций
2. **Надежность** - обработка ошибок и восстановление
3. **Производительность** - кэширование и оптимизация
4. **Мониторинг** - сбор метрик и логирование
5. **Тестируемость** - легкость создания моков и тестов
6. **Расширяемость** - простота добавления новых реализаций

Использование этих протоколов обеспечивает создание надежной, масштабируемой и поддерживаемой алготрейдинговой системы. 