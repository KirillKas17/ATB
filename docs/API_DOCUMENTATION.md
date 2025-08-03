# API Documentation - ATB Trading System

## Обзор

ATB (Advanced Trading Bot) - это комплексная система алгоритмической торговли, построенная на принципах Domain-Driven Design (DDD) и микросервисной архитектуры.

## Архитектура системы

```
ATB/
├── domain/           # Бизнес-логика и доменные модели
├── application/      # Use cases и оркестрация
├── infrastructure/   # Внешние адаптеры и сервисы
├── interfaces/       # API и пользовательские интерфейсы
└── shared/          # Общие утилиты и конфигурация
```

## Основные компоненты

### 1. Система стратегий

#### UnifiedStrategyInterface

Унифицированный интерфейс для всех торговых стратегий.

```python
from domain.strategies.unified_strategy_interface import UnifiedStrategyInterface

class MyStrategy(UnifiedStrategyInterface):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "MyStrategy"
        
    async def analyze_market(self, market_data: MarketData) -> AnalysisResult:
        # Анализ рынка
        pass
        
    async def generate_signals(self, analysis: AnalysisResult) -> List[Signal]:
        # Генерация сигналов
        pass
        
    async def execute_trades(self, signals: List[Signal]) -> List[TradeResult]:
        # Выполнение сделок
        pass
```

#### StrategyAdapter

Адаптер для интеграции стратегий из разных слоёв.

```python
from domain.strategies.strategy_adapter import StrategyAdapter

# Создание адаптера
adapter = StrategyAdapter()

# Регистрация стратегий
adapter.register_strategy("domain", DomainStrategy())
adapter.register_strategy("infrastructure", InfrastructureStrategy())

# Получение стратегии
strategy = adapter.get_strategy("domain")
```

### 2. Система обработки исключений

#### ExceptionHandler

Централизованная система обработки исключений.

```python
from shared.exception_handler import ExceptionHandler, ExceptionCategory

# Создание обработчика
handler = ExceptionHandler()

# Регистрация обработчиков
@handler.register_handler(ExceptionCategory.CONFIGURATION)
def handle_config_error(exception: Exception, context: Dict[str, Any]):
    logger.error(f"Configuration error: {exception}")
    # Логика восстановления

@handler.register_handler(ExceptionCategory.TRADING)
def handle_trading_error(exception: Exception, context: Dict[str, Any]):
    logger.error(f"Trading error: {exception}")
    # Отмена ордеров, закрытие позиций

# Обработка исключения
try:
    # Код, который может вызвать исключение
    pass
except Exception as e:
    handler.handle_exception(e, {"component": "trading_engine"})
```

### 3. Система мониторинга производительности

#### PerformanceMonitor

Мониторинг производительности компонентов.

```python
from shared.performance_monitor import performance_monitor, monitor_performance

# Автоматический мониторинг функций
@monitor_performance("trading_engine", "order_processing")
def process_order(order: Order) -> OrderResult:
    # Обработка ордера
    pass

# Ручная запись метрик
performance_monitor.record_metric(
    "order_processing_time", 
    duration, 
    MetricType.TIMER, 
    "trading_engine"
)

# Установка порогов
performance_monitor.set_threshold("cpu_usage", "warning", 70.0)
performance_monitor.set_threshold("memory_usage", "error", 90.0)

# Обработчик алертов
def alert_handler(alert: Alert):
    if alert.level == AlertLevel.CRITICAL:
        # Отправка уведомления
        send_notification(f"Critical alert: {alert.message}")

performance_monitor.add_alert_handler(alert_handler)
```

### 4. Система валидации конфигурации

#### ConfigValidator

Валидация конфигурационных файлов.

```python
from shared.config_validator import config_validator, validate_config_file

# Валидация файла конфигурации
is_valid = validate_config_file("config/application.yaml")

if not is_valid:
    issues = config_validator.get_issues()
    for issue in issues:
        print(f"{issue.severity}: {issue.message}")

# Автоматическое исправление
fixed_config = config_validator.auto_fix_issues(config)

# Экспорт проблем
issues_json = config_validator.export_issues("json")
```

### 5. Система анализа метрик

#### MetricsAnalyzer

Анализ производительности и выявление аномалий.

```python
from shared.metrics_analyzer import metrics_analyzer, analyze_metric_performance

# Добавление метрик
metrics_analyzer.add_metric_point(
    "response_time", 
    datetime.now(), 
    150.0, 
    "api_gateway"
)

# Анализ метрики
summary = metrics_analyzer.analyze_metric(
    "response_time",
    start_time=datetime.now() - timedelta(hours=1)
)

print(f"Average response time: {summary.mean_value:.2f}ms")
print(f"Anomalies detected: {len(summary.anomalies)}")

# Сравнение метрик
comparison = metrics_analyzer.compare_metrics(
    "api_response_time", 
    "database_query_time"
)

print(f"Correlation: {comparison['correlation']:.2f}")

# Генерация отчёта
report = metrics_analyzer.generate_report([
    "response_time", 
    "throughput", 
    "error_rate"
])
```

### 6. Автоматизированное тестирование

#### AutomatedTestRunner

Автоматическое выполнение тестов.

```python
from tests.automated_test_suite import test_runner, TestSuite, TestType

# Создание набора тестов
suite = TestSuite(
    name="trading_tests",
    description="Trading functionality tests",
    tests=["test_order_execution", "test_position_management"],
    test_type=TestType.INTEGRATION,
    priority=TestPriority.HIGH
)

test_runner.register_test_suite(suite)

# Регистрация тестового исполнителя
@test_runner.register_test_executor("test_order_execution")
def test_order_execution():
    # Тест выполнения ордеров
    pass

# Запуск тестов
results = test_runner.run_test_suite("trading_tests")

# Нагрузочное тестирование
performance_report = test_runner.run_performance_test(
    "test_order_processing",
    iterations=1000,
    concurrent_users=10
)

# Генерация отчёта
report = test_runner.generate_test_report("test_results.json")
```

## Конфигурация

### Структура конфигурационного файла

```yaml
# config/application.yaml
application:
  app_name: "ATB Trading System"
  version: "1.0.0"
  environment: "production"
  debug: false
  logging_level: "INFO"
  max_workers: 8
  timeout: 30.0
  retry_attempts: 3
  encryption_enabled: true
  monitoring_enabled: true
  metrics_interval: 30
  alert_threshold: 80.0

trading:
  trading_enabled: true
  max_position_size: 10000.0
  min_position_size: 100.0
  leverage: 1.0
  max_drawdown: 20.0
  stop_loss_percent: 5.0
  take_profit_percent: 10.0
  strategy_timeout: 300
  max_concurrent_strategies: 5
  order_timeout: 60
  max_retries: 3

database:
  database_type: "postgresql"
  host: "localhost"
  port: 5432
  database_name: "atb_trading"
  username: "atb_user"
  password: "secure_password"
  pool_size: 10
  ssl_enabled: true

exchange:
  exchange_name: "binance"
  api_key: "your_api_key"
  secret_key: "your_secret_key"
  sandbox: false
  timeout: 30
  rate_limit: 100
  trading_pairs:
    - "BTC/USDT"
    - "ETH/USDT"
  min_order_size:
    "BTC/USDT": 0.001
    "ETH/USDT": 0.01
  fees:
    "BTC/USDT": 0.001
    "ETH/USDT": 0.001
```

## Примеры использования

### 1. Создание торговой стратегии

```python
from domain.strategies.unified_strategy_interface import UnifiedStrategyInterface
from domain.entities.market import MarketData
from domain.entities.signal import Signal
from domain.entities.trade import TradeResult

class MovingAverageStrategy(UnifiedStrategyInterface):
    def __init__(self, short_period: int = 10, long_period: int = 20):
        self.short_period = short_period
        self.long_period = long_period
        self.name = "MovingAverageStrategy"
        
    async def analyze_market(self, market_data: MarketData) -> AnalysisResult:
        # Вычисление скользящих средних
        short_ma = self._calculate_ma(market_data.prices, self.short_period)
        long_ma = self._calculate_ma(market_data.prices, self.long_period)
        
        return AnalysisResult(
            short_ma=short_ma,
            long_ma=long_ma,
            trend="bullish" if short_ma > long_ma else "bearish"
        )
        
    async def generate_signals(self, analysis: AnalysisResult) -> List[Signal]:
        signals = []
        
        if analysis.short_ma > analysis.long_ma:
            signals.append(Signal(
                type="BUY",
                symbol="BTC/USDT",
                price=analysis.current_price,
                confidence=0.8
            ))
        elif analysis.short_ma < analysis.long_ma:
            signals.append(Signal(
                type="SELL",
                symbol="BTC/USDT",
                price=analysis.current_price,
                confidence=0.8
            ))
            
        return signals
        
    async def execute_trades(self, signals: List[Signal]) -> List[TradeResult]:
        results = []
        
        for signal in signals:
            try:
                # Выполнение торговой операции
                trade = await self._execute_trade(signal)
                results.append(TradeResult(
                    success=True,
                    trade_id=trade.id,
                    executed_price=trade.price,
                    quantity=trade.quantity
                ))
            except Exception as e:
                results.append(TradeResult(
                    success=False,
                    error=str(e)
                ))
                
        return results
```

### 2. Интеграция с биржей

```python
from infrastructure.exchange.bybit_adapter import BybitAdapter
from domain.entities.order import Order, OrderType, OrderSide

# Создание адаптера биржи
exchange = BybitAdapter(
    api_key="your_api_key",
    secret_key="your_secret_key",
    sandbox=False
)

# Размещение ордера
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    type=OrderType.LIMIT,
    quantity=0.001,
    price=50000.0
)

try:
    result = await exchange.place_order(order)
    print(f"Order placed: {result.order_id}")
except Exception as e:
    print(f"Order failed: {e}")
```

### 3. Мониторинг системы

```python
from shared.performance_monitor import performance_monitor
from shared.metrics_analyzer import metrics_analyzer

# Настройка мониторинга
performance_monitor.start_monitoring()

# Установка порогов
performance_monitor.set_threshold("cpu_usage", "warning", 70.0)
performance_monitor.set_threshold("memory_usage", "error", 90.0)
performance_monitor.set_threshold("response_time", "warning", 1000.0)

# Обработчик алертов
def handle_alert(alert):
    if alert.level == AlertLevel.CRITICAL:
        # Отправка уведомления
        send_slack_notification(f"🚨 Critical alert: {alert.message}")
    elif alert.level == AlertLevel.WARNING:
        # Логирование предупреждения
        logger.warning(f"Performance warning: {alert.message}")

performance_monitor.add_alert_handler(handle_alert)

# Анализ метрик
summary = metrics_analyzer.analyze_metric(
    "trading_performance",
    start_time=datetime.now() - timedelta(hours=24)
)

print(f"Trading performance analysis:")
print(f"  Average execution time: {summary.mean_value:.2f}ms")
print(f"  Anomalies detected: {len(summary.anomalies)}")
print(f"  Trend: {summary.trend.direction.value}")
```

### 4. Автоматическое тестирование

```python
from tests.automated_test_suite import test_runner, TestSuite, TestType

# Создание тестового набора
trading_suite = TestSuite(
    name="trading_integration",
    description="Integration tests for trading functionality",
    tests=[
        "test_order_placement",
        "test_position_management", 
        "test_risk_management"
    ],
    test_type=TestType.INTEGRATION,
    priority=TestPriority.HIGH,
    timeout=300,
    parallel=True
)

test_runner.register_test_suite(trading_suite)

# Регистрация тестов
@test_runner.register_test_executor("test_order_placement")
def test_order_placement():
    # Тест размещения ордеров
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001)
    result = exchange.place_order(order)
    assert result.success
    assert result.order_id is not None

# Запуск тестов
results = test_runner.run_test_suite("trading_integration")

# Нагрузочное тестирование
performance_report = test_runner.run_performance_test(
    "test_order_processing",
    iterations=1000,
    concurrent_users=50
)

print(f"Performance test results:")
print(f"  Success rate: {performance_report['success_rate']:.2%}")
print(f"  Throughput: {performance_report['throughput']:.2f} ops/sec")
print(f"  Average response time: {performance_report['avg_response_time']:.2f}ms")
```

## Обработка ошибок

### Типы исключений

```python
from domain.exceptions import (
    TradingException,
    ConfigurationException,
    NetworkException,
    ValidationException
)

try:
    # Торговая операция
    result = await strategy.execute_trades(signals)
except TradingException as e:
    logger.error(f"Trading error: {e}")
    # Логика восстановления
except ConfigurationException as e:
    logger.error(f"Configuration error: {e}")
    # Перезагрузка конфигурации
except NetworkException as e:
    logger.error(f"Network error: {e}")
    # Повторная попытка
except ValidationException as e:
    logger.error(f"Validation error: {e}")
    # Исправление данных
```

### Централизованная обработка

```python
from shared.exception_handler import ExceptionHandler

handler = ExceptionHandler()

@handler.register_handler(ExceptionCategory.TRADING)
def handle_trading_error(exception: Exception, context: Dict[str, Any]):
    logger.error(f"Trading error in {context.get('component')}: {exception}")
    
    # Отмена активных ордеров
    if context.get('cancel_orders', True):
        cancel_all_orders()
    
    # Уведомление администратора
    send_admin_notification(f"Trading error: {exception}")

@handler.register_handler(ExceptionCategory.CONFIGURATION)
def handle_config_error(exception: Exception, context: Dict[str, Any]):
    logger.error(f"Configuration error: {exception}")
    
    # Перезагрузка конфигурации
    reload_configuration()
    
    # Перезапуск компонентов
    restart_components()

# Использование
try:
    # Код, который может вызвать исключение
    pass
except Exception as e:
    handler.handle_exception(e, {
        "component": "trading_engine",
        "cancel_orders": True
    })
```

## Лучшие практики

### 1. Архитектурные принципы

- **DDD**: Разделение на домены, агрегаты и сервисы
- **SOLID**: Следование принципам SOLID
- **DRY**: Избежание дублирования кода
- **KISS**: Простота и читаемость кода

### 2. Безопасность

- Шифрование конфиденциальных данных
- Валидация входных данных
- Ограничение доступа к API
- Логирование всех операций

### 3. Производительность

- Мониторинг ключевых метрик
- Кэширование часто используемых данных
- Асинхронная обработка
- Оптимизация запросов к базе данных

### 4. Надёжность

- Обработка исключений
- Автоматическое восстановление
- Резервное копирование данных
- Тестирование всех компонентов

### 5. Масштабируемость

- Микросервисная архитектура
- Горизонтальное масштабирование
- Балансировка нагрузки
- Мониторинг ресурсов

## Заключение

ATB Trading System предоставляет мощную и гибкую платформу для алгоритмической торговли. Система построена на современных принципах разработки и обеспечивает высокую производительность, надёжность и масштабируемость.

Для получения дополнительной информации обратитесь к документации конкретных модулей или свяжитесь с командой разработки. 