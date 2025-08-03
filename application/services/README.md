# Сервисы Application слоя

## Архитектура

Сервисы разделены по принципу единственной ответственности (SRP) для лучшей поддерживаемости и тестируемости.

## Структура сервисов

### 1. MarketService (Основной сервис)
- **Назначение**: Координация между специализированными сервисами
- **Размер**: ~150 строк (было 400+)
- **Ответственность**: 
  - Делегирование задач специализированным сервисам
  - Интеграция различных типов анализа
  - Предоставление высокоуровневого API

### 2. MarketDataService (Сервис данных)
- **Назначение**: Работа с рыночными данными
- **Размер**: ~120 строк
- **Ответственность**:
  - CRUD операции с рыночными данными
  - Получение истории цен
  - Управление состоянием рынка
  - Подписка на обновления

### 3. TechnicalAnalysisService (Технический анализ)
- **Назначение**: Расчет технических индикаторов
- **Размер**: ~100 строк
- **Ответственность**:
  - RSI, MACD, Bollinger Bands, ATR
  - Другие технические индикаторы
  - Математические расчеты

### 4. MarketAnalysisService (Анализ рынка)
- **Назначение**: Анализ рыночных условий
- **Размер**: ~150 строк
- **Ответственность**:
  - Сводка рынка
  - Профиль объема
  - Анализ рыночного режима
  - Уровни поддержки/сопротивления

### 5. MarketServiceFactory (Фабрика)
- **Назначение**: Создание и управление сервисами
- **Размер**: ~50 строк
- **Ответственность**:
  - Создание экземпляров сервисов
  - Кэширование сервисов
  - Управление зависимостями

## Преимущества новой архитектуры

### 1. Разделение ответственности
- Каждый сервис отвечает за одну область
- Легче тестировать и поддерживать
- Меньше конфликтов при разработке

### 2. Модульность
- Можно использовать сервисы независимо
- Легко добавлять новые функции
- Простое переиспользование кода

### 3. Производительность
- Кэширование сервисов через фабрику
- Параллельное выполнение операций
- Оптимизированные алгоритмы

### 4. Масштабируемость
- Легко добавлять новые типы анализа
- Возможность горизонтального масштабирования
- Микросервисная архитектура

## Примеры использования

### Базовое использование
```python
from application.services import MarketServiceFactory

# Создание фабрики
factory = MarketServiceFactory(market_repository)

# Получение основного сервиса
market_service = factory.create_market_service()

# Получение специализированных сервисов
data_service = factory.create_market_data_service()
analysis_service = factory.create_market_analysis_service()
```

### Комплексный анализ
```python
# Получение комплексного анализа
analysis = await market_service.get_comprehensive_market_analysis("BTC/USD")

# Получение настроений рынка
sentiment = await market_service.get_market_sentiment("BTC/USD")

# Получение алертов
alerts = await market_service.get_market_alerts("BTC/USD")
```

### Специализированные операции
```python
# Только технический анализ
technical_service = factory.create_technical_analysis_service()
indicators = await technical_service.get_technical_indicators("BTC/USD", ["rsi", "macd"])

# Только анализ рынка
analysis_service = factory.create_market_analysis_service()
volume_profile = await analysis_service.get_volume_profile("BTC/USD")
```

## Миграция с старой архитектуры

### Старый код:
```python
market_service = MarketService(market_repository)
summary = await market_service.get_market_summary("BTC/USD")
```

### Новый код:
```python
# Вариант 1: Через основной сервис (рекомендуется)
market_service = factory.create_market_service()
summary = await market_service.get_market_summary("BTC/USD")

# Вариант 2: Через специализированный сервис
analysis_service = factory.create_market_analysis_service()
summary = await analysis_service.get_market_summary("BTC/USD")
```

## Тестирование

Каждый сервис можно тестировать независимо:

```python
# Тест технического анализа
def test_technical_analysis():
    service = TechnicalAnalysisService(mock_repository)
    indicators = service._calculate_rsi(prices)
    assert len(indicators) > 0

# Тест анализа рынка
def test_market_analysis():
    service = MarketAnalysisService(mock_repository)
    summary = service._calculate_enhanced_market_summary(market_data, "BTC/USD", "1h")
    assert "last_price" in summary
```

## Будущие улучшения

1. **Асинхронные операции**: Параллельное выполнение независимых анализов
2. **Кэширование результатов**: Redis для кэширования вычислений
3. **Плагинная архитектура**: Легкое добавление новых индикаторов
4. **Метрики производительности**: Мониторинг времени выполнения
5. **Конфигурация**: Настройка параметров через конфиг файлы 