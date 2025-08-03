# Доменные стратегии

## Обзор

Модуль `domain/strategies` предоставляет полную архитектуру для создания, управления и исполнения торговых стратегий в соответствии с принципами Domain-Driven Design (DDD) и SOLID.

## Архитектура

### Основные компоненты

```
domain/strategies/
├── __init__.py              # Экспорт всех компонентов
├── strategy_interface.py    # Базовый интерфейс стратегии
├── strategy_factory.py      # Фабрика для создания стратегий
├── strategy_registry.py     # Реестр стратегий
├── base_strategies.py       # Базовые реализации стратегий
├── strategy_types.py        # Типы и конфигурации
├── exceptions.py           # Исключения стратегий
├── utils.py               # Утилиты для стратегий
├── validators.py          # Валидаторы
├── examples.py            # Примеры использования
├── tests.py              # Тесты
└── README.md             # Документация
```

### Принципы проектирования

- **DDD (Domain-Driven Design)**: Стратегии являются доменными сущностями
- **SOLID**: Соблюдение всех принципов SOLID
- **Типобезопасность**: Полная типизация с использованием `typing`
- **Расширяемость**: Легкое добавление новых типов стратегий
- **Тестируемость**: Полное покрытие тестами

## Основные интерфейсы

### StrategyInterface

Базовый абстрактный класс для всех стратегий:

```python
from domain.strategies import StrategyInterface
from domain.types import StrategyId, TradingPair, ConfidenceLevel, RiskLevel

class MyStrategy(StrategyInterface):
    def __init__(self, strategy_id: StrategyId, name: str, ...):
        super().__init__(strategy_id, name, strategy_type, trading_pairs, parameters)
    
    def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        # Реализация анализа рынка
        pass
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # Реализация генерации сигналов
        pass
```

### StrategyFactory

Фабрика для создания стратегий:

```python
from domain.strategies import StrategyFactory

factory = StrategyFactory()

# Регистрация создателей стратегий
factory.register_creator("trend_following", TrendFollowingStrategy)
factory.register_creator("mean_reversion", MeanReversionStrategy)

# Создание стратегии
config = {
    "name": "My Strategy",
    "strategy_type": "trend_following",
    "trading_pairs": ["BTC/USDT"],
    "parameters": {
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "position_size": 0.1
    }
}

strategy = factory.create_strategy(config)
```

### StrategyRegistry

Реестр для управления стратегиями:

```python
from domain.strategies import StrategyRegistry

registry = StrategyRegistry()

# Регистрация стратегии
registry.register_strategy(strategy)

# Получение стратегии
retrieved_strategy = registry.get_strategy(strategy_id)

# Фильтрация стратегий
active_strategies = registry.get_active_strategies()
trend_strategies = registry.get_strategies_by_type(StrategyType.TREND_FOLLOWING)
```

## Базовые стратегии

### TrendFollowingStrategy

Стратегия следования за трендом:

```python
from domain.strategies import TrendFollowingStrategy

strategy = TrendFollowingStrategy(
    strategy_id=StrategyId(uuid4()),
    name="BTC Trend Strategy",
    trading_pairs=["BTC/USDT"],
    parameters={
        "short_period": 10,
        "long_period": 20,
        "trend_strength_threshold": 0.7,
        "volume_confirmation": True
    }
)
```

### MeanReversionStrategy

Стратегия возврата к среднему:

```python
from domain.strategies import MeanReversionStrategy

strategy = MeanReversionStrategy(
    strategy_id=StrategyId(uuid4()),
    name="ETH Mean Reversion",
    trading_pairs=["ETH/USDT"],
    parameters={
        "lookback_period": 50,
        "deviation_threshold": 2.0,
        "bollinger_period": 20,
        "bollinger_std": 2
    }
)
```

### BreakoutStrategy

Стратегия пробоя:

```python
from domain.strategies import BreakoutStrategy

strategy = BreakoutStrategy(
    strategy_id=StrategyId(uuid4()),
    name="ADA Breakout",
    trading_pairs=["ADA/USDT"],
    parameters={
        "breakout_threshold": 1.5,
        "volume_multiplier": 2.0,
        "support_resistance_period": 20
    }
)
```

### ScalpingStrategy

Скальпинг стратегия:

```python
from domain.strategies import ScalpingStrategy

strategy = ScalpingStrategy(
    strategy_id=StrategyId(uuid4()),
    name="SOL Scalping",
    trading_pairs=["SOL/USDT"],
    parameters={
        "scalping_threshold": 0.1,
        "max_hold_time": 300,
        "min_spread": 0.001
    }
)
```

### ArbitrageStrategy

Арбитражная стратегия:

```python
from domain.strategies import ArbitrageStrategy

strategy = ArbitrageStrategy(
    strategy_id=StrategyId(uuid4()),
    name="Multi-Exchange Arbitrage",
    trading_pairs=["BTC/USDT", "ETH/USDT"],
    parameters={
        "arbitrage_threshold": 0.5,
        "max_slippage": 0.1,
        "execution_timeout": 30
    }
)
```

## Типы и конфигурации

### StrategyCategory

Категории стратегий:

```python
from domain.strategies import StrategyCategory

categories = [
    StrategyCategory.TREND_FOLLOWING,
    StrategyCategory.MEAN_REVERSION,
    StrategyCategory.BREAKOUT,
    StrategyCategory.SCALPING,
    StrategyCategory.ARBITRAGE,
    StrategyCategory.GRID,
    StrategyCategory.MOMENTUM,
    StrategyCategory.VOLATILITY,
    StrategyCategory.MACHINE_LEARNING,
    StrategyCategory.QUANTITATIVE
]
```

### RiskProfile

Профили риска:

```python
from domain.strategies import RiskProfile

risk_profiles = [
    RiskProfile.CONSERVATIVE,    # Консервативный
    RiskProfile.MODERATE,        # Умеренный
    RiskProfile.AGGRESSIVE,      # Агрессивный
    RiskProfile.VERY_AGGRESSIVE  # Очень агрессивный
]
```

### TimeHorizon

Временные горизонты:

```python
from domain.strategies import TimeHorizon

horizons = [
    TimeHorizon.ULTRA_SHORT,  # Секунды-минуты
    TimeHorizon.SHORT,        # Минуты-часы
    TimeHorizon.MEDIUM,       # Часы-дни
    TimeHorizon.LONG,         # Дни-недели
    TimeHorizon.VERY_LONG     # Недели-месяцы
]
```

### StrategyConfiguration

Полная конфигурация стратегии:

```python
from domain.strategies import StrategyConfiguration, StrategyParameters

config = StrategyConfiguration(
    strategy_id=StrategyId(uuid4()),
    name="My Strategy",
    description="Описание стратегии",
    category=StrategyCategory.TREND_FOLLOWING,
    risk_profile=RiskProfile.MODERATE,
    time_horizon=TimeHorizon.MEDIUM,
    trading_pairs=[TradingPair("BTC/USDT")],
    base_parameters=StrategyParameters(
        stop_loss=Decimal("0.02"),
        take_profit=Decimal("0.04"),
        position_size=Decimal("0.1")
    ),
    specific_parameters=TrendFollowingParameters(
        short_period=10,
        long_period=20,
        trend_strength_threshold=Decimal("0.7")
    )
)
```

## Утилиты

### StrategyUtils

Утилиты для работы со стратегиями:

```python
from domain.strategies import StrategyUtils

# Расчет технических индикаторов
sma = StrategyUtils.calculate_moving_average(prices, 20, "sma")
ema = StrategyUtils.calculate_moving_average(prices, 20, "ema")
rsi = StrategyUtils.calculate_rsi(prices, 14)
upper, middle, lower = StrategyUtils.calculate_bollinger_bands(prices, 20, 2)
macd, signal, histogram = StrategyUtils.calculate_macd(prices, 12, 26, 9)

# Расчет волатильности и импульса
volatility = StrategyUtils.calculate_volatility(prices, 20)
momentum = StrategyUtils.calculate_momentum(prices, 10)
roc = StrategyUtils.calculate_rate_of_change(prices, 10)

# Сглаживание и нормализация
smoothed = StrategyUtils.smooth_data(prices, 5, "savgol")
normalized = StrategyUtils.normalize_data(prices)

# Обнаружение выбросов
outliers = StrategyUtils.detect_outliers(prices, "z_score", 2.0)
```

## Валидация

### StrategyValidator

Валидация стратегий:

```python
from domain.strategies import StrategyValidator

# Валидация конфигурации
config = {...}
errors = StrategyValidator.validate_strategy_config(config)

# Валидация параметров
parameters = {...}
errors = StrategyValidator.validate_parameters(parameters)

# Валидация торговых пар
errors = StrategyValidator.validate_trading_pair("BTC/USDT")

# Валидация рыночных данных
errors = StrategyValidator.validate_market_data(market_data)

# Валидация сигналов
errors = StrategyValidator.validate_signal(signal)
```

### ParameterValidator

Валидация параметров стратегий:

```python
from domain.strategies import ParameterValidator

# Валидация параметров трендовой стратегии
trend_params = {...}
errors = ParameterValidator.validate_trend_following_params(trend_params)

# Валидация параметров стратегии возврата к среднему
mean_reversion_params = {...}
errors = ParameterValidator.validate_mean_reversion_params(mean_reversion_params)

# Валидация параметров стратегии пробоя
breakout_params = {...}
errors = ParameterValidator.validate_breakout_params(breakout_params)
```

## Исключения

Система исключений для стратегий:

```python
from domain.strategies import (
    StrategyException,
    StrategyCreationError,
    StrategyExecutionError,
    StrategyValidationError,
    StrategyNotFoundError,
    StrategyRegistrationError
)

try:
    strategy = factory.create_strategy(config)
except StrategyCreationError as e:
    print(f"Ошибка создания стратегии: {e}")
except StrategyValidationError as e:
    print(f"Ошибка валидации: {e}")
```

## Примеры использования

### Создание и использование стратегии

```python
from domain.strategies import (
    TrendFollowingStrategy,
    StrategyFactory,
    StrategyRegistry
)
from domain.entities.market import MarketData

# Создание стратегии
strategy = TrendFollowingStrategy(
    strategy_id=StrategyId(uuid4()),
    name="BTC Trend Strategy",
    trading_pairs=["BTC/USDT"],
    parameters={
        "short_period": 10,
        "long_period": 20,
        "trend_strength_threshold": 0.7,
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "position_size": 0.1
    }
)

# Активация стратегии
strategy.activate()

# Анализ рынка
market_data = MarketData(...)
analysis = strategy.analyze_market(market_data)

# Генерация сигнала
signal = strategy.generate_signal(market_data)

if signal:
    print(f"Сгенерирован сигнал: {signal.signal_type.value}")
    print(f"Уверенность: {signal.confidence}")
    print(f"Цена: {signal.price}")
```

### Использование фабрики и реестра

```python
from domain.strategies import StrategyFactory, StrategyRegistry

# Создание фабрики и реестра
factory = StrategyFactory()
registry = StrategyRegistry()

# Регистрация создателей стратегий
factory.register_creator("trend_following", TrendFollowingStrategy)
factory.register_creator("mean_reversion", MeanReversionStrategy)

# Создание стратегий
configs = [
    {
        "name": "BTC Trend",
        "strategy_type": "trend_following",
        "trading_pairs": ["BTC/USDT"],
        "parameters": {...}
    },
    {
        "name": "ETH Mean Reversion",
        "strategy_type": "mean_reversion",
        "trading_pairs": ["ETH/USDT"],
        "parameters": {...}
    }
]

strategies = []
for config in configs:
    strategy = factory.create_strategy(config)
    registry.register_strategy(strategy)
    strategies.append(strategy)

# Получение статистики
factory_stats = factory.get_statistics()
registry_stats = registry.get_statistics()

print(f"Создано стратегий: {factory_stats['total_strategies']}")
print(f"Активных стратегий: {registry_stats['active_strategies']}")
```

### Использование утилит

```python
from domain.strategies import StrategyUtils

# Создание тестовых данных
prices = [Decimal(str(100 + i * 0.5)) for i in range(100)]
volumes = [Decimal(str(1000 + i * 10)) for i in range(100)]

# Расчет индикаторов
sma_20 = StrategyUtils.calculate_moving_average(prices, 20, "sma")
rsi_14 = StrategyUtils.calculate_rsi(prices, 14)
bb_upper, bb_middle, bb_lower = StrategyUtils.calculate_bollinger_bands(prices, 20, 2)
volatility = StrategyUtils.calculate_volatility(prices, 20)

# Анализ данных
smoothed_prices = StrategyUtils.smooth_data(prices, 5, "savgol")
normalized_prices = StrategyUtils.normalize_data(prices)
outliers = StrategyUtils.detect_outliers(prices, "z_score", 2.0)

print(f"RSI последнее значение: {rsi_14[-1]}")
print(f"Волатильность: {volatility[-1]}")
print(f"Выбросов обнаружено: {sum(outliers)}")
```

## Тестирование

### Запуск тестов

```python
from domain.strategies.tests import run_tests

# Запуск всех тестов
result = run_tests()

if result.wasSuccessful():
    print("✅ Все тесты прошли успешно!")
else:
    print(f"❌ Тесты завершились с ошибками")
```

### Структура тестов

- `TestStrategyInterface`: Тесты базового интерфейса
- `TestStrategyFactory`: Тесты фабрики стратегий
- `TestStrategyRegistry`: Тесты реестра стратегий
- `TestStrategyUtils`: Тесты утилит
- `TestStrategyValidators`: Тесты валидаторов
- `TestStrategyTypes`: Тесты типов и конфигураций

## Лучшие практики

### Создание новых стратегий

1. **Наследование от StrategyInterface**:
```python
class MyCustomStrategy(StrategyInterface):
    def __init__(self, strategy_id: StrategyId, name: str, ...):
        super().__init__(strategy_id, name, StrategyType.CUSTOM, ...)
    
    def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        # Реализация анализа
        pass
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # Реализация генерации сигналов
        pass
```

2. **Регистрация в фабрике**:
```python
factory.register_creator("custom", MyCustomStrategy)
```

3. **Создание параметров**:
```python
@dataclass
class MyCustomParameters:
    custom_param1: int = 10
    custom_param2: Decimal = Decimal("0.5")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "custom_param1": self.custom_param1,
            "custom_param2": str(self.custom_param2)
        }
```

### Валидация данных

```python
# Всегда валидируйте входные данные
errors = StrategyValidator.validate_strategy_config(config)
if errors:
    raise StrategyValidationError(f"Invalid config: {'; '.join(errors)}")

# Валидируйте рыночные данные
errors = StrategyValidator.validate_market_data(market_data)
if errors:
    raise StrategyValidationError(f"Invalid market data: {'; '.join(errors)}")
```

### Обработка ошибок

```python
try:
    strategy = factory.create_strategy(config)
    signal = strategy.generate_signal(market_data)
except StrategyCreationError as e:
    logger.error(f"Failed to create strategy: {e}")
except StrategyExecutionError as e:
    logger.error(f"Strategy execution failed: {e}")
except StrategyValidationError as e:
    logger.error(f"Validation error: {e}")
```

### Производительность

```python
# Используйте кэширование для тяжелых вычислений
@lru_cache(maxsize=128)
def calculate_heavy_indicator(prices: Tuple[Decimal, ...]) -> Decimal:
    # Тяжелые вычисления
    pass

# Оптимизируйте циклы
for i in range(len(prices)):
    # Избегайте повторных вычислений
    pass
```

## Расширение функциональности

### Добавление новых индикаторов

```python
class StrategyUtils:
    @staticmethod
    def calculate_custom_indicator(prices: List[Decimal], period: int) -> List[Decimal]:
        """Расчет пользовательского индикатора."""
        # Реализация
        pass
```

### Добавление новых типов стратегий

```python
class StrategyType(Enum):
    CUSTOM = "custom"

class CustomStrategy(StrategyInterface):
    # Реализация
    pass
```

### Добавление новых валидаторов

```python
class ParameterValidator:
    @staticmethod
    def validate_custom_params(params: Dict[str, Any]) -> List[str]:
        """Валидация параметров пользовательской стратегии."""
        errors = []
        # Реализация валидации
        return errors
```

## Заключение

Модуль `domain/strategies` предоставляет полную, типобезопасную и расширяемую архитектуру для создания торговых стратегий. Соблюдение принципов DDD и SOLID обеспечивает высокое качество кода и легкую поддержку.

Для получения дополнительной информации обратитесь к примерам в `examples.py` и тестам в `tests.py`. 