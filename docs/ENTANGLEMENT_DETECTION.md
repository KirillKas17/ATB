# Quantum Order Entanglement Detection

## Обзор

Система Quantum Order Entanglement Detection предназначена для обнаружения квантовой запутанности в ордербуках между различными криптовалютными биржами. Это позволяет выявлять синхронизированные изменения цен, которые могут указывать на координацию между биржами или арбитражные возможности.

## Архитектура

Система построена по принципам Domain-Driven Design (DDD) и состоит из следующих слоев:

### Domain Layer (`domain/intelligence/`)

#### EntanglementDetector
Основной класс для обнаружения запутанности:

```python
from domain.intelligence.entanglement_detector import EntanglementDetector

detector = EntanglementDetector(
    max_lag_ms=3.0,           # Максимальный lag между биржами
    correlation_threshold=0.95, # Порог корреляции
    window_size=100,          # Размер окна анализа
    min_data_points=50        # Минимум точек для анализа
)
```

**Методы:**
- `detect_entanglement(exchange1, exchange2, symbol)` - обнаружение запутанности между двумя биржами
- `process_order_book_updates(updates)` - обработка обновлений ордербуков
- `get_buffer_status()` - получение статуса буферов

#### OrderBookUpdate
Модель обновления ордербука:

```python
@dataclass
class OrderBookUpdate:
    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
```

#### EntanglementResult
Результат обнаружения запутанности:

```python
@dataclass
class EntanglementResult:
    is_entangled: bool
    lag_ms: float
    correlation_score: float
    exchange_pair: Tuple[str, str]
    symbol: str
    timestamp: Timestamp
    confidence: float
    metadata: Dict[str, Any]
```

### Infrastructure Layer (`infrastructure/market_data/`)

#### BaseExchangeConnector
Базовый класс для коннекторов бирж:

```python
class BaseExchangeConnector(ABC):
    async def stream_order_book(symbol: str) -> AsyncGenerator[OrderBookUpdate, None]
```

#### Коннекторы бирж
- `BinanceConnector` - подключение к Binance
- `CoinbaseConnector` - подключение к Coinbase Pro
- `KrakenConnector` - подключение к Kraken

### Application Layer (`application/analysis/`)

#### EntanglementMonitor
Координационный сервис для мониторинга:

```python
from application.analysis.entanglement_monitor import EntanglementMonitor

monitor = EntanglementMonitor(
    log_file_path="logs/entanglement_events.json",
    detection_interval=1.0,
    max_lag_ms=3.0,
    correlation_threshold=0.95
)
```

## Алгоритм обнаружения

1. **Сбор данных**: Получение обновлений ордербуков с разных бирж
2. **Выравнивание по времени**: Синхронизация временных меток
3. **Нормализация**: Приведение изменений цен к единому масштабу
4. **Кросс-корреляция**: Вычисление корреляции между временными рядами
5. **Анализ lag**: Определение задержки между биржами
6. **Принятие решения**: Сравнение с порогами и определение запутанности

## Использование

### Базовое использование

```python
import asyncio
from application.analysis.entanglement_monitor import EntanglementMonitor

async def main():
    # Создаем мониторинг
    monitor = EntanglementMonitor()
    
    # Запускаем мониторинг
    await monitor.start_monitoring()

# Запуск
asyncio.run(main())
```

### Настройка параметров

```python
monitor = EntanglementMonitor(
    log_file_path="custom_log.json",
    detection_interval=0.5,    # Проверка каждые 500мс
    max_lag_ms=5.0,           # Максимальный lag 5мс
    correlation_threshold=0.9  # Порог корреляции 90%
)
```

### Добавление новых пар бирж

```python
# Добавление новой пары для мониторинга
monitor.add_exchange_pair("binance", "kucoin", "BTCUSDT")

# Удаление пары
monitor.remove_exchange_pair("binance", "coinbase", "BTCUSDT")
```

### Получение статистики

```python
# Статус мониторинга
status = monitor.get_status()
print(f"Total detections: {status['stats']['total_detections']}")
print(f"Entangled detections: {status['stats']['entangled_detections']}")

# История обнаружений
history = monitor.get_entanglement_history(limit=100)
for event in history:
    print(f"Entanglement: {event['data']['exchange_pair']}")
```

## Конфигурация

### Параметры EntanglementDetector

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `max_lag_ms` | Максимальный lag между биржами (мс) | 3.0 |
| `correlation_threshold` | Порог корреляции для обнаружения | 0.95 |
| `window_size` | Размер окна для анализа | 100 |
| `min_data_points` | Минимум точек для анализа | 50 |
| `price_change_threshold` | Порог изменения цены | 0.0001 |

### Параметры EntanglementMonitor

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `log_file_path` | Путь к файлу логов | "logs/entanglement_events.json" |
| `detection_interval` | Интервал проверки (сек) | 1.0 |
| `max_lag_ms` | Максимальный lag | 3.0 |
| `correlation_threshold` | Порог корреляции | 0.95 |

## Логирование

Система ведет подробные логи всех событий:

```json
{
  "timestamp": 1640995200.0,
  "event_type": "entanglement_detected",
  "data": {
    "is_entangled": true,
    "lag_ms": 2.5,
    "correlation_score": 0.97,
    "exchange_pair": ["binance", "coinbase"],
    "symbol": "BTCUSDT",
    "confidence": 0.92,
    "metadata": {
      "data_points_1": 100,
      "data_points_2": 98,
      "price_volatility_1": 0.15,
      "price_volatility_2": 0.14
    }
  }
}
```

## Мониторинг и метрики

### Статистика производительности

- Общее количество проверок
- Количество обнаруженных запутанностей
- Среднее время обработки
- Размеры буферов данных

### Метрики качества

- Уровень уверенности в обнаружениях
- Распределение lag между биржами
- Корреляционные коэффициенты
- Волатильность цен

## Расширение системы

### Добавление новой биржи

1. Создать новый коннектор в `infrastructure/market_data/`
2. Наследоваться от `BaseExchangeConnector`
3. Реализовать методы:
   - `get_websocket_url()`
   - `get_subscription_message()`
   - `parse_order_book_update()`

### Кастомные стратегии обнаружения

Можно создать собственные стратегии, наследуясь от `CircuitBreakerStrategy`:

```python
class CustomEntanglementStrategy(CircuitBreakerStrategy):
    def should_open(self, metrics, thresholds):
        # Кастомная логика
        pass
```

## Примеры

Полные примеры использования доступны в `examples/entanglement_detection_example.py`.

## Требования

- Python 3.8+
- numpy
- scipy
- websockets
- loguru
- asyncio

## Безопасность

- Все API ключи хранятся в переменных окружения
- WebSocket соединения защищены TLS
- Логи не содержат чувствительной информации
- Ограничения на частоту запросов

## Производительность

- Обработка до 1000 обновлений в секунду
- Задержка обнаружения < 10мс
- Потребление памяти < 100MB
- Поддержка до 10 одновременных бирж 