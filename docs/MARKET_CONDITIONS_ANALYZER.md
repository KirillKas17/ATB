# MarketConditionsAnalyzer - Анализатор рыночных условий

## Обзор

`MarketConditionsAnalyzer` - это промышленный сервис для комплексного анализа рыночных условий, обеспечивающий принятие обоснованных решений о миграции агентов в системе автоматической торговли.

## Архитектура

### Основные компоненты

1. **MarketConditionsAnalyzer** - основной класс анализатора
2. **MarketConditionsConfig** - конфигурация параметров анализа
3. **MarketConditionScore** - результат анализа рыночных условий
4. **MarketConditionType** - типы рыночных условий

### Зависимости

- `MarketRepository` - для получения рыночных данных
- `TechnicalAnalysisService` - для технического анализа
- `MarketRegimeDetector` - для определения режима рынка

## Функциональность

### Комплексный анализ рыночных условий

Сервис выполняет анализ по следующим направлениям:

#### 1. Анализ волатильности
- **Текущая волатильность** - стандартное отклонение доходностей за последние периоды
- **Историческая волатильность** - долгосрочная волатильность для сравнения
- **Волатильность волатильности** - стабильность волатильности
- **Нормализованная волатильность** - отношение текущей к исторической

#### 2. Анализ тренда
- **Линейная регрессия** - определение направления и силы тренда
- **R-squared** - уверенность в тренде
- **Сила тренда** - нормализованная величина изменения цены

#### 3. Анализ объема
- **Относительный объем** - сравнение с историческими значениями
- **Тренд объема** - направление изменения объема
- **Профиль объема** - классификация (высокий/нормальный/низкий)

#### 4. Анализ моментума
- **Краткосрочный моментум** - изменение за короткий период
- **Среднесрочный моментум** - изменение за средний период
- **RSI анализ** - корректировка на основе перекупленности/перепроданности

#### 5. Анализ режима рынка
- **Технический анализ** - использование существующих индикаторов
- **Структура рынка** - определение типа структуры
- **Стабильность режима** - консистентность трендов

### Типы рыночных условий

```python
class MarketConditionType(Enum):
    BULL_TRENDING = "bull_trending"           # Бычий тренд
    BEAR_TRENDING = "bear_trending"           # Медвежий тренд
    SIDEWAYS_VOLATILE = "sideways_volatile"   # Боковой волатильный
    SIDEWAYS_STABLE = "sideways_stable"       # Боковой стабильный
    BREAKOUT_UP = "breakout_up"               # Пробой вверх
    BREAKOUT_DOWN = "breakout_down"           # Пробой вниз
    CONSOLIDATION = "consolidation"           # Консолидация
    DISTRIBUTION = "distribution"             # Распределение
    ACCUMULATION = "accumulation"             # Накопление
    EXHAUSTION = "exhaustion"                 # Истощение
    NO_STRUCTURE = "no_structure"             # Без структуры
```

## Конфигурация

### MarketConditionsConfig

```python
@dataclass
class MarketConditionsConfig:
    # Временные окна
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 200
    volatility_window: int = 30
    trend_window: int = 100
    
    # Пороги для классификации
    volatility_threshold_high: float = 0.03
    volatility_threshold_low: float = 0.01
    trend_strength_threshold: float = 0.6
    volume_threshold_high: float = 1.5
    volume_threshold_low: float = 0.7
    
    # Веса для расчета скора
    volatility_weight: float = 0.25
    trend_weight: float = 0.25
    volume_weight: float = 0.20
    momentum_weight: float = 0.15
    regime_weight: float = 0.15
    
    # Настройки кэширования
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
```

## Использование

### Базовое использование

```python
from infrastructure.services.market_conditions_analyzer import (
    MarketConditionsAnalyzer,
    MarketConditionsConfig
)

# Создание анализатора
config = MarketConditionsConfig()
analyzer = MarketConditionsAnalyzer(
    market_repository=market_repository,
    technical_analysis_service=technical_analysis_service,
    config=config
)

# Расчет скора рыночных условий
score = await analyzer.calculate_market_score(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback_periods=100
)

print(f"Overall score: {score.overall_score:.3f}")
print(f"Condition type: {score.condition_type.value}")
print(f"Confidence: {score.confidence:.3f}")
```

### Интеграция с AutoMigrationManager

```python
# В AutoMigrationManager
async def _calculate_market_score(self) -> float:
    if self.market_conditions_analyzer is None:
        return 0.7  # Fallback
    
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    total_score = 0.0
    valid_symbols = 0
    
    for symbol in symbols:
        try:
            market_score = await self.market_conditions_analyzer.calculate_market_score(
                symbol=symbol,
                timeframe="1h",
                lookback_periods=100
            )
            total_score += market_score.overall_score
            valid_symbols += 1
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            continue
    
    return total_score / valid_symbols if valid_symbols > 0 else 0.7
```

## Кэширование

Сервис использует двухуровневое кэширование:

1. **Кэш результатов анализа** - кэширует `MarketConditionScore`
2. **Кэш рыночных данных** - кэширует `DataFrame` с данными

### Настройки кэша

- **TTL**: 300 секунд (5 минут)
- **Максимальный размер**: 1000 записей
- **LRU политика**: автоматическое удаление старых записей

## Мониторинг и статистика

### Статистика анализа

```python
stats = analyzer.get_analysis_stats()
print(f"Total analyses: {stats['total_analyses']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Avg processing time: {stats['avg_processing_time']:.3f}s")
```

### Логирование

Сервис использует структурированное логирование:

```python
logger.debug(f"Market score calculated for {symbol}: {overall_score:.3f}")
logger.warning(f"No market data available for {symbol}")
logger.error(f"Error calculating market score: {e}")
```

## Производительность

### Оптимизации

1. **Асинхронная обработка** - все операции асинхронные
2. **Кэширование** - избежание повторных вычислений
3. **Векторизованные вычисления** - использование NumPy/Pandas
4. **Ленивая загрузка** - данные загружаются только при необходимости

### Метрики производительности

- **Время анализа**: ~50-200ms на символ
- **Использование памяти**: ~10-50MB в зависимости от объема данных
- **Кэш-хит**: >80% при нормальной работе

## Тестирование

### Юнит-тесты

```bash
pytest tests/unit/test_market_conditions_analyzer.py -v
```

### Покрытие тестами

- Инициализация и конфигурация
- Анализ с валидными данными
- Обработка пустых данных
- Анализ отдельных компонентов
- Функциональность кэширования
- Статистика и мониторинг

## Интеграция с DI контейнером

### Регистрация в DI

```python
# В application/di_container_refactored.py
market_conditions_analyzer = providers.Singleton(
    MarketConditionsAnalyzer,
    market_repository=market_repository,
    technical_analysis_service=technical_analysis_service
)
```

### Получение из DI

```python
# Через ServiceLocator
analyzer = service_locator.get_external_service(MarketConditionsAnalyzer)

# Или напрямую из контейнера
analyzer = container.market_conditions_analyzer()
```

## Расширение функциональности

### Добавление новых типов анализа

1. Создать новый метод анализа в `MarketConditionsAnalyzer`
2. Добавить соответствующий вес в `MarketConditionsConfig`
3. Интегрировать в `_calculate_overall_score`
4. Обновить `_determine_condition_type` при необходимости

### Кастомизация конфигурации

```python
config = MarketConditionsConfig(
    volatility_weight=0.3,  # Увеличить вес волатильности
    trend_weight=0.2,       # Уменьшить вес тренда
    cache_ttl_seconds=600   # Увеличить время жизни кэша
)
```

## Безопасность и надежность

### Обработка ошибок

- Все методы обернуты в try-catch блоки
- Fallback значения при ошибках
- Детальное логирование ошибок
- Graceful degradation при недоступности зависимостей

### Валидация данных

- Проверка на пустые данные
- Валидация временных рядов
- Проверка корректности цен и объемов
- Обработка NaN значений

## Заключение

`MarketConditionsAnalyzer` предоставляет промышленное решение для анализа рыночных условий, обеспечивая:

- **Высокую точность** - комплексный анализ множества факторов
- **Производительность** - асинхронная обработка и кэширование
- **Надежность** - обработка ошибок и fallback механизмы
- **Расширяемость** - модульная архитектура
- **Мониторинг** - детальная статистика и логирование

Сервис интегрирован в систему автоматической миграции агентов и обеспечивает принятие обоснованных решений на основе текущих рыночных условий. 