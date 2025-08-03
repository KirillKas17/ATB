# Модуль памяти паттернов (domain/memory)

## Обзор

Модуль `domain/memory` предоставляет систему памяти паттернов для рыночной аналитики. Он позволяет сохранять, анализировать и прогнозировать рыночные паттерны на основе исторических данных.

## Архитектура

### Основные компоненты

1. **PatternMemory** - основной сервис для работы с памятью паттернов
2. **PatternMatcher** - компонент для сопоставления паттернов
3. **PatternPredictor** - компонент для прогнозирования на основе паттернов
4. **SQLitePatternMemoryRepository** - репозиторий для хранения данных

### Типы данных

- **MarketFeatures** - рыночные характеристики
- **PatternSnapshot** - снимок паттерна
- **PatternOutcome** - результат развития паттерна
- **PredictionResult** - результат прогнозирования

## Использование

### Базовое использование

```python
from domain.memory.pattern_memory import PatternMemory
from domain.memory.types import MarketFeatures, PatternMemoryConfig
from domain.intelligence.market_pattern_recognizer import PatternType

# Создание конфигурации
config = PatternMemoryConfig(
    db_path="data/pattern_memory.db",
    similarity_threshold=0.9,
    max_similar_cases=10
)

# Инициализация системы памяти
memory = PatternMemory(config)

# Создание рыночных характеристик
features = MarketFeatures(
    price=100.0,
    price_change_1m=0.01,
    price_change_5m=0.02,
    price_change_15m=0.03,
    volatility=0.05,
    volume=1000000.0,
    volume_change_1m=0.1,
    volume_change_5m=0.2,
    volume_sma_ratio=1.0,
    spread=0.001,
    spread_change=0.0001,
    bid_volume=500000.0,
    ask_volume=500000.0,
    order_book_imbalance=0.0,
    depth_absorption=0.5,
    entropy=0.7,
    gravity=0.3,
    latency=10.0,
    correlation=0.8,
    whale_signal=0.2,
    mm_signal=0.1,
    external_sync=True
)

# Поиск похожих паттернов
prediction = memory.match_snapshot(features, "BTCUSDT", PatternType.WHALE_ABSORPTION)

if prediction:
    print(f"Прогноз: {prediction.predicted_direction}")
    print(f"Уверенность: {prediction.confidence:.2f}")
    print(f"Ожидаемая доходность: {prediction.predicted_return_percent:.2f}%")
```

### Сохранение паттернов

```python
from domain.memory.entities import PatternSnapshot, PatternOutcome
from domain.memory.types import OutcomeType
from domain.value_objects.timestamp import Timestamp

# Создание снимка паттерна
snapshot = PatternSnapshot(
    pattern_id="unique_pattern_id",
    timestamp=Timestamp.from_datetime(datetime.now()),
    symbol="BTCUSDT",
    pattern_type=PatternType.WHALE_ABSORPTION,
    confidence=0.8,
    strength=0.7,
    direction="up",
    features=features,
    metadata={"source": "market_analyzer"}
)

# Сохранение снимка
success = memory.save_pattern_data("unique_pattern_id", snapshot)

# Создание исхода паттерна
outcome = PatternOutcome(
    pattern_id="unique_pattern_id",
    symbol="BTCUSDT",
    outcome_type=OutcomeType.PROFITABLE,
    timestamp=Timestamp.from_datetime(datetime.now()),
    price_change_percent=2.0,
    volume_change_percent=10.0,
    duration_minutes=30,
    max_profit_percent=3.0,
    max_loss_percent=-1.0,
    final_return_percent=2.0,
    volatility_during=0.05,
    volume_profile="increasing",
    market_regime="trending",
    metadata={"analysis": "completed"}
)

# Обновление исхода
success = memory.update_pattern_outcome("unique_pattern_id", outcome)
```

### Получение статистики

```python
# Статистика по символу
symbol_stats = memory.get_pattern_statistics("BTCUSDT")
print(f"Всего паттернов: {symbol_stats['total_patterns']}")
print(f"Средняя уверенность: {symbol_stats['avg_confidence']:.2f}")

# Общая статистика
memory_stats = memory.get_memory_statistics()
print(f"Всего снимков: {memory_stats.total_snapshots}")
print(f"Всего исходов: {memory_stats.total_outcomes}")
print(f"Средняя успешность: {memory_stats.avg_success_rate:.2f}")
```

## Конфигурация

### PatternMemoryConfig

```python
config = PatternMemoryConfig(
    # Пути к данным
    db_path="data/pattern_memory.db",
    
    # Пороги сходства
    similarity_threshold=0.9,
    max_similar_cases=10,
    min_similar_cases=3,
    
    # Параметры очистки
    days_to_keep=30,
    max_snapshots_per_symbol=1000,
    
    # Параметры прогнозирования
    confidence_threshold=0.7,
    min_success_rate=0.6,
    min_accuracy=0.5,
    
    # Параметры производительности
    cache_size=1000,
    batch_size=100,
    connection_timeout=30.0
)
```

## Алгоритмы

### Сопоставление паттернов

Система использует комбинированный подход для вычисления сходства:

1. **Косинусное сходство** - для учета направления векторов
2. **Евклидово расстояние** - для учета абсолютных значений
3. **Нормализация** - для приведения к единому масштабу

### Прогнозирование

Прогнозы генерируются на основе:

1. **Взвешенного среднего** - по сходству паттернов
2. **Исторической успешности** - статистика исходов
3. **Консистентности результатов** - стабильность прогнозов

## Производительность

### Оптимизации

- **Индексы БД** - для быстрого поиска по символам и типам
- **Кэширование** - для часто используемых данных
- **Батчевая обработка** - для массовых операций
- **Очистка старых данных** - для поддержания производительности

### Мониторинг

```python
# Получение статистики производительности
stats = memory.get_memory_statistics()
print(f"Размер базы данных: {stats.total_snapshots} снимков")
print(f"Эффективность: {stats.avg_success_rate:.2f}")
```

## Тестирование

```bash
# Запуск тестов
pytest tests/domain/memory/ -v

# Запуск с покрытием
pytest tests/domain/memory/ --cov=domain.memory --cov-report=html
```

## Интеграция

### С другими модулями

```python
from domain.intelligence.market_pattern_recognizer import MarketPatternRecognizer
from application.prediction.pattern_predictor import PatternPredictor

# Интеграция с распознавателем паттернов
recognizer = MarketPatternRecognizer()
pattern = recognizer.detect_pattern(symbol, market_data, order_book)

if pattern:
    # Сохранение в память
    memory.save_pattern_data(pattern.id, pattern.to_snapshot())
    
    # Прогнозирование
    prediction = memory.match_snapshot(pattern.features, symbol)
```

## Безопасность

- **Валидация данных** - все входные данные проверяются
- **Обработка ошибок** - graceful handling исключений
- **Логирование** - детальное логирование операций
- **Изоляция данных** - каждый символ в отдельной таблице

## Расширение

### Добавление новых типов паттернов

1. Добавить новый тип в `PatternType` enum
2. Обновить логику в `PatternMatcher`
3. Добавить тесты
4. Обновить документацию

### Кастомные алгоритмы

```python
class CustomPatternMatcher(IPatternMatcher):
    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        # Ваша логика
        pass
```

## Поддержка

Для вопросов и предложений обращайтесь к команде разработки или создавайте issues в репозитории. 