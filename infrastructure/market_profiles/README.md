# Market Profiles Module

Промышленный модуль для работы с профилями рынка и паттернами маркет-мейкера.

## Архитектура

Модуль построен по принципам DDD и SOLID с разделением на слои:

```
infrastructure/market_profiles/
├── __init__.py                 # Основной экспорт модуля
├── models/                     # Модели данных
│   ├── __init__.py
│   ├── storage_config.py      # Конфигурация хранения
│   ├── analysis_config.py     # Конфигурация анализа
│   └── storage_models.py      # Модели данных
├── interfaces/                 # Интерфейсы
│   ├── __init__.py
│   └── storage_interfaces.py  # Интерфейсы хранилища
├── storage/                   # Хранилище данных
│   ├── __init__.py
│   ├── market_maker_storage.py        # Основное хранилище
│   ├── pattern_memory_repository.py   # Репозиторий памяти паттернов
│   └── behavior_history_repository.py # Репозиторий истории поведения
├── analysis/                  # Анализ данных
│   ├── __init__.py
│   ├── pattern_analyzer.py    # Анализатор паттернов
│   ├── similarity_calculator.py       # Калькулятор схожести
│   └── success_rate_analyzer.py       # Анализатор успешности
└── README.md                  # Документация
```

## Основные компоненты

### 1. Хранилище (Storage)

#### MarketMakerStorage
Основное хранилище паттернов маркет-мейкера с поддержкой:
- Асинхронных операций
- Кэширования в памяти
- Сжатия данных
- Резервного копирования
- Валидации целостности
- Многопоточности

#### PatternMemoryRepository
Специализированный репозиторий для работы с памятью паттернов:
- CRUD операции для паттернов
- Поиск похожих паттернов
- Анализ успешности
- Управление метаданными

#### BehaviorHistoryRepository
Репозиторий для истории поведения маркет-мейкера:
- Сохранение записей поведения
- Анализ статистики
- Очистка старых данных

### 2. Анализ (Analysis)

#### PatternAnalyzer
Промышленный анализатор паттернов:
- Анализ схожести паттернов
- Расчет уверенности
- Предсказание исходов
- Анализ рыночного контекста
- Генерация рекомендаций

#### SimilarityCalculator
Калькулятор схожести паттернов:
- Многомерный анализ схожести
- Взвешенные метрики
- Кэширование результатов
- Оптимизация производительности

#### SuccessRateAnalyzer
Анализатор успешности паттернов:
- Статистический анализ
- Прогнозирование трендов
- Адаптивные пороги
- Анализ рыночных условий

### 3. Модели (Models)

#### StorageConfig
Конфигурация для хранения:
- Пути к файлам
- Параметры кэширования
- Настройки сжатия
- Параметры производительности

#### AnalysisConfig
Конфигурация для анализа:
- Пороги уверенности
- Веса признаков
- Параметры схожести
- Факторы корректировки

#### StorageModels
Модели данных:
- StorageStatistics - статистика хранилища
- PatternMetadata - метаданные паттернов
- BehaviorRecord - записи поведения
- SuccessMapEntry - записи карты успешности

### 4. Интерфейсы (Interfaces)

#### IPatternStorage
Интерфейс для хранения паттернов:
- Сохранение и обновление паттернов
- Поиск и фильтрация
- Управление метаданными
- Резервное копирование

#### IBehaviorHistoryStorage
Интерфейс для истории поведения:
- Сохранение записей поведения
- Получение статистики
- Очистка данных

#### IPatternAnalyzer
Интерфейс для анализа паттернов:
- Анализ схожести
- Расчет уверенности
- Предсказание исходов
- Генерация рекомендаций

## Использование

### Базовое использование

```python
from infrastructure.market_profiles import (
    MarketMakerStorage,
    PatternAnalyzer,
    StorageConfig,
    AnalysisConfig
)

# Создание конфигураций
storage_config = StorageConfig(base_path="market_data")
analysis_config = AnalysisConfig()

# Инициализация компонентов
storage = MarketMakerStorage(storage_config)
analyzer = PatternAnalyzer(analysis_config)

# Сохранение паттерна
success = await storage.save_pattern("BTCUSDT", pattern)

# Получение паттернов
patterns = await storage.get_patterns_by_symbol("BTCUSDT", limit=100)

# Анализ схожести
similarity = await analyzer.analyze_pattern_similarity(pattern1, pattern2)

# Получение рекомендаций
recommendations = await analyzer.get_pattern_recommendations("BTCUSDT", current_patterns)
```

### Расширенное использование

```python
from infrastructure.market_profiles import (
    PatternMemoryRepository,
    BehaviorHistoryRepository,
    SimilarityCalculator,
    SuccessRateAnalyzer
)

# Специализированные репозитории
pattern_repo = PatternMemoryRepository()
behavior_repo = BehaviorHistoryRepository()

# Анализаторы
similarity_calc = SimilarityCalculator(analysis_config)
success_analyzer = SuccessRateAnalyzer(analysis_config)

# Анализ успешности паттернов
success_analysis = await success_analyzer.analyze_pattern_success(
    pattern_type=MarketMakerPatternType.ACCUMULATION,
    historical_patterns=patterns,
    time_window_days=30
)

# Получение статистики поведения
behavior_stats = await behavior_repo.get_behavior_statistics("BTCUSDT", days=30)
```

## Особенности реализации

### 1. Строгая типизация
- Все параметры и возвращаемые значения типизированы
- Использование NewType для доменных типов
- Protocol интерфейсы для контрактов

### 2. Асинхронность
- Все операции ввода-вывода асинхронные
- Использование asyncio.Lock для синхронизации
- ThreadPoolExecutor для блокирующих операций

### 3. Кэширование
- LRU кэш для часто используемых данных
- Автоматическая инвалидация кэша
- Метрики производительности кэша

### 4. Производительность
- Оптимизированные SQL запросы
- Индексы для быстрого поиска
- Сжатие данных для экономии места
- Пакетные операции

### 5. Надежность
- Валидация всех входных данных
- Обработка исключений
- Логирование операций
- Проверка целостности данных

### 6. Масштабируемость
- Модульная архитектура
- Разделение ответственности
- Конфигурируемые параметры
- Поддержка различных бэкендов

## Конфигурация

### StorageConfig
```python
storage_config = StorageConfig(
    base_path=Path("market_data"),
    max_patterns_per_symbol=1000,
    cleanup_days=30,
    min_accuracy_for_cleanup=0.5,
    backup_enabled=True,
    compression_enabled=True,
    cache_size=1000,
    max_workers=4,
    retry_attempts=3
)
```

### AnalysisConfig
```python
analysis_config = AnalysisConfig(
    min_confidence=0.6,
    similarity_threshold=0.8,
    accuracy_threshold=0.7,
    volume_threshold=1000.0,
    time_window_seconds=300,
    feature_weights={
        "book_pressure": 0.25,
        "volume_delta": 0.20,
        "price_reaction": 0.15,
        "spread_change": 0.10,
        "order_imbalance": 0.15,
        "liquidity_depth": 0.10,
        "volume_concentration": 0.05
    }
)
```

## Метрики и мониторинг

Модуль предоставляет встроенные метрики:

```python
# Статистика хранилища
storage_stats = await storage.get_storage_statistics()
print(f"Total patterns: {storage_stats.total_patterns}")
print(f"Cache hit ratio: {storage_stats.cache_hit_ratio}")

# Статистика анализа
analysis_stats = similarity_calc.get_similarity_statistics()
print(f"Total calculations: {analysis_stats['total_calculations']}")
print(f"Average similarity: {analysis_stats['avg_similarity']}")
```

## Тестирование

Модуль готов к тестированию с использованием:
- Unit тестов для каждого компонента
- Integration тестов для взаимодействия компонентов
- Performance тестов для производительности
- Mock объектов для изоляции

## Развертывание

Модуль готов к продакшену с поддержкой:
- Docker контейнеризации
- Kubernetes развертывания
- Мониторинга и алертинга
- Автоматического масштабирования
- Резервного копирования

## Лицензия

Модуль является частью ATB торговой системы и подчиняется соответствующим лицензионным соглашениям. 