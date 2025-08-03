# Neural Noise Divergence Analysis

## Обзор

Система анализа нейронного шума ордербука (Neural Noise Divergence) предназначена для обнаружения синтетических паттернов в рыночных данных, которые могут указывать на манипуляции или искусственные торговые стратегии.

## Архитектура

### Domain Layer

#### `NoiseAnalyzer` (`domain/intelligence/noise_analyzer.py`)

Основной класс для анализа нейронного шума ордербука.

**Ключевые методы:**

- `compute_fractal_dimension(order_book: OrderBookSnapshot) -> float`
  - Вычисляет фрактальную размерность временного ряда цен
  - Использует библиотеку `antropy.fractal_dimension`
  - Fallback на метод Higuchi при ошибках

- `compute_entropy(order_book: OrderBookSnapshot) -> float`
  - Вычисляет энтропию нормализованных объемов
  - Использует `scipy.stats.entropy`
  - Нормализует результат в диапазоне [0, 1]

- `is_synthetic_noise(fd: float, entropy: float) -> bool`
  - Определяет синтетический шум на основе порогов
  - Условия: `1.2 < fd < 1.4` и `entropy < threshold`

- `analyze_noise(order_book: OrderBookSnapshot) -> NoiseAnalysisResult`
  - Полный анализ ордербука
  - Возвращает структурированный результат

**Модели данных:**

```python
@dataclass
class OrderBookSnapshot:
    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: Dict[str, Any] = None

@dataclass
class NoiseAnalysisResult:
    fractal_dimension: float
    entropy: float
    is_synthetic_noise: bool
    confidence: float
    metadata: Dict[str, Any]
    timestamp: Timestamp
```

### Application Layer

#### `OrderBookPreFilter` (`application/filters/orderbook_filter.py`)

Координационный сервис для фильтрации ордербуков.

**Основные возможности:**

- Прием потока `OrderBookSnapshot`
- Применение `NoiseAnalyzer`
- Добавление флага `synthetic_noise` в метаданные
- Статистика и мониторинг

**Конфигурация:**

```python
@dataclass
class FilterConfig:
    enabled: bool = True
    fractal_dimension_lower: float = 1.2
    fractal_dimension_upper: float = 1.4
    entropy_threshold: float = 0.7
    min_data_points: int = 50
    window_size: int = 100
    confidence_threshold: float = 0.8
    log_filtered: bool = True
    log_analysis: bool = False
```

## Алгоритм анализа

### 1. Фрактальная размерность

Фрактальная размерность измеряет сложность временного ряда:

- **Низкая размерность (1.0-1.2)**: Простые, линейные паттерны
- **Средняя размерность (1.2-1.4)**: Синтетические паттерны
- **Высокая размерность (1.4+)**: Естественные рыночные данные

### 2. Энтропия объемов

Энтропия измеряет неопределенность распределения объемов:

- **Низкая энтропия (< 0.7)**: Регулярные, предсказуемые объемы
- **Высокая энтропия (≥ 0.7)**: Случайные, естественные объемы

### 3. Комбинированный анализ

Синтетический шум определяется как:
```
is_synthetic = (1.2 ≤ fd ≤ 1.4) AND (entropy < threshold)
```

## Использование

### Базовый анализ

```python
from domain.intelligence.noise_analyzer import NoiseAnalyzer, OrderBookSnapshot
from domain.value_objects import Price, Volume, Timestamp

# Создание анализатора
analyzer = NoiseAnalyzer(
    fractal_dimension_lower=1.2,
    fractal_dimension_upper=1.4,
    entropy_threshold=0.7
)

# Создание ордербука
order_book = OrderBookSnapshot(
    exchange="binance",
    symbol="BTCUSDT",
    bids=[(Price(50000.0), Volume(1.0))],
    asks=[(Price(50010.0), Volume(1.0))],
    timestamp=Timestamp(time.time())
)

# Анализ
result = analyzer.analyze_noise(order_book)
print(f"Synthetic noise: {result.is_synthetic_noise}")
print(f"Confidence: {result.confidence:.3f}")
```

### Фильтрация потока данных

```python
from application.filters.orderbook_filter import OrderBookPreFilter, FilterConfig

# Создание фильтра
config = FilterConfig(
    enabled=True,
    fractal_dimension_lower=1.2,
    fractal_dimension_upper=1.4,
    entropy_threshold=0.7
)
filter_obj = OrderBookPreFilter(config)

# Фильтрация ордербука
filtered_ob = filter_obj.filter_order_book(
    exchange="binance",
    symbol="BTCUSDT",
    bids=[(50000.0, 1.0)],
    asks=[(50010.0, 1.0)],
    timestamp=time.time()
)

# Проверка результата
if filtered_ob.meta.get('synthetic_noise'):
    print("Synthetic noise detected!")
```

### Асинхронная обработка

```python
import asyncio

async def process_order_book_stream():
    # Создание потока данных
    async def order_book_stream():
        for i in range(100):
            yield {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'bids': [(50000.0, 1.0)],
                'asks': [(50010.0, 1.0)],
                'timestamp': time.time()
            }
            await asyncio.sleep(0.1)
    
    # Фильтрация потока
    filter_obj = OrderBookPreFilter()
    async for filtered_ob in filter_obj.filter_order_book_stream(order_book_stream()):
        if filtered_ob.meta.get('synthetic_noise'):
            print(f"Synthetic noise detected at {filtered_ob.timestamp}")

# Запуск
asyncio.run(process_order_book_stream())
```

## Конфигурация

### Параметры анализатора

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `fractal_dimension_lower` | Нижняя граница фрактальной размерности | 1.2 |
| `fractal_dimension_upper` | Верхняя граница фрактальной размерности | 1.4 |
| `entropy_threshold` | Порог энтропии для определения синтетического шума | 0.7 |
| `min_data_points` | Минимальное количество точек для анализа | 50 |
| `window_size` | Размер окна для исторических данных | 100 |
| `confidence_threshold` | Порог уверенности | 0.8 |

### Настройка фильтра

```python
config = FilterConfig(
    enabled=True,
    fractal_dimension_lower=1.2,
    fractal_dimension_upper=1.4,
    entropy_threshold=0.7,
    min_data_points=30,
    window_size=80,
    confidence_threshold=0.9,
    log_filtered=True,
    log_analysis=False
)
```

## API Reference

### NoiseAnalyzer

#### `__init__(fractal_dimension_lower=1.2, fractal_dimension_upper=1.4, entropy_threshold=0.7, min_data_points=50, window_size=100, confidence_threshold=0.8)`

Инициализация анализатора.

#### `compute_fractal_dimension(order_book: OrderBookSnapshot) -> float`

Вычисление фрактальной размерности.

#### `compute_entropy(order_book: OrderBookSnapshot) -> float`

Вычисление энтропии объемов.

#### `is_synthetic_noise(fd: float, entropy: float) -> bool`

Определение синтетического шума.

#### `analyze_noise(order_book: OrderBookSnapshot) -> NoiseAnalysisResult`

Полный анализ ордербука.

#### `get_analysis_statistics() -> Dict[str, Any]`

Получение статистики анализа.

#### `reset_history() -> None`

Сброс исторических данных.

### OrderBookPreFilter

#### `__init__(config: Optional[FilterConfig] = None)`

Инициализация фильтра.

#### `filter_order_book(exchange: str, symbol: str, bids: List[tuple], asks: List[tuple], timestamp: float, sequence_id: Optional[int] = None) -> OrderBookSnapshot`

Синхронная фильтрация ордербука.

#### `filter_order_book_async(exchange: str, symbol: str, bids: List[tuple], asks: List[tuple], timestamp: float, sequence_id: Optional[int] = None) -> OrderBookSnapshot`

Асинхронная фильтрация ордербука.

#### `filter_order_book_stream(order_book_stream: AsyncGenerator[Dict[str, Any], None]) -> AsyncGenerator[OrderBookSnapshot, None]`

Фильтрация потока ордербуков.

#### `get_filter_statistics() -> Dict[str, Any]`

Получение статистики фильтрации.

#### `reset_statistics() -> None`

Сброс статистики.

#### `update_config(new_config: FilterConfig) -> None`

Обновление конфигурации.

## Примеры

### Пример 1: Обнаружение синтетического шума

```python
from domain.intelligence.noise_analyzer import NoiseAnalyzer
import numpy as np

# Создание синтетических данных с регулярными паттернами
def create_synthetic_data():
    base_price = 50000.0
    prices = []
    for i in range(100):
        # Добавляем периодический шум
        noise = np.sin(i * 0.1) * 5  # Регулярная синусоида
        price = base_price + noise
        prices.append(price)
    return prices

# Анализ
analyzer = NoiseAnalyzer()
prices = create_synthetic_data()

# Создание ордербука
order_book = OrderBookSnapshot(
    exchange="test",
    symbol="BTCUSDT",
    bids=[(Price(prices[i]), Volume(1.0)) for i in range(10)],
    asks=[(Price(prices[i+10]), Volume(1.0)) for i in range(10)],
    timestamp=Timestamp(time.time())
)

# Анализ нескольких раз для накопления истории
for i in range(30):
    result = analyzer.analyze_noise(order_book)

print(f"Fractal dimension: {result.fractal_dimension:.3f}")
print(f"Entropy: {result.entropy:.3f}")
print(f"Synthetic noise: {result.is_synthetic_noise}")
```

### Пример 2: Интеграция с торговой системой

```python
from application.filters.orderbook_filter import OrderBookPreFilter
from infrastructure.market_data.binance_connector import BinanceConnector

class TradingSystem:
    def __init__(self):
        self.filter = OrderBookPreFilter()
        self.connector = BinanceConnector()
    
    async def process_market_data(self):
        async for order_book_data in self.connector.stream_order_book("BTCUSDT"):
            # Фильтрация
            filtered_ob = await self.filter.filter_order_book_async(
                exchange="binance",
                symbol="BTCUSDT",
                bids=order_book_data['bids'],
                asks=order_book_data['asks'],
                timestamp=order_book_data['timestamp']
            )
            
            # Проверка на синтетический шум
            if filtered_ob.meta.get('synthetic_noise'):
                print("Warning: Synthetic noise detected!")
                # Пропускаем торговые сигналы
                continue
            
            # Обычная обработка торговых сигналов
            self.process_trading_signals(filtered_ob)
    
    def process_trading_signals(self, order_book):
        # Торговая логика
        pass
```

### Пример 3: Мониторинг и статистика

```python
from application.filters.orderbook_filter import OrderBookPreFilter
import time

# Создание фильтра с логированием
config = FilterConfig(
    enabled=True,
    log_filtered=True,
    log_analysis=True
)
filter_obj = OrderBookPreFilter(config)

# Обработка данных
for i in range(1000):
    # Симуляция данных ордербука
    bids = [(50000.0 - i, 1.0) for i in range(10)]
    asks = [(50010.0 + i, 1.0) for i in range(10)]
    
    filtered_ob = filter_obj.filter_order_book(
        exchange="test",
        symbol="BTCUSDT",
        bids=bids,
        asks=asks,
        timestamp=time.time()
    )
    
    # Периодический вывод статистики
    if i % 100 == 0:
        stats = filter_obj.get_filter_statistics()
        print(f"Processed: {stats['total_processed']}")
        print(f"Filtered: {stats['filtered_out']}")
        print(f"Filter rate: {stats['filter_rate']:.3f}")
```

## Тестирование

### Запуск тестов

```bash
# Запуск всех тестов
pytest tests/test_neural_noise_analysis.py -v

# Запуск конкретного теста
pytest tests/test_neural_noise_analysis.py::TestNoiseAnalyzer::test_compute_fractal_dimension -v

# Запуск с покрытием
pytest tests/test_neural_noise_analysis.py --cov=domain.intelligence.noise_analyzer --cov=application.filters.orderbook_filter
```

### Тестовые сценарии

1. **Синтетические данные**: Проверка обнаружения искусственных паттернов
2. **Естественные данные**: Проверка корректной обработки рыночных данных
3. **Граничные случаи**: Тестирование с недостаточными данными
4. **Производительность**: Тестирование с большими объемами данных
5. **Конфигурация**: Проверка различных настроек параметров

## Расширение системы

### Добавление новых метрик

```python
class ExtendedNoiseAnalyzer(NoiseAnalyzer):
    def compute_spectral_density(self, order_book: OrderBookSnapshot) -> float:
        """Вычисление спектральной плотности."""
        # Реализация
        pass
    
    def analyze_noise(self, order_book: OrderBookSnapshot) -> NoiseAnalysisResult:
        """Расширенный анализ с новыми метриками."""
        # Базовый анализ
        result = super().analyze_noise(order_book)
        
        # Добавление новых метрик
        spectral_density = self.compute_spectral_density(order_book)
        result.metadata['spectral_density'] = spectral_density
        
        return result
```

### Интеграция с другими системами

```python
class NoiseAnalysisIntegration:
    def __init__(self):
        self.noise_analyzer = NoiseAnalyzer()
        self.entanglement_detector = EntanglementDetector()
    
    def comprehensive_analysis(self, order_book: OrderBookSnapshot):
        """Комплексный анализ ордербука."""
        # Анализ нейронного шума
        noise_result = self.noise_analyzer.analyze_noise(order_book)
        
        # Анализ запутанности (если есть данные с другой биржи)
        # entanglement_result = self.entanglement_detector.detect_entanglement(...)
        
        return {
            'noise_analysis': noise_result,
            # 'entanglement_analysis': entanglement_result
        }
```

## Производительность

### Оптимизации

1. **Кэширование**: Результаты анализа кэшируются для повторных вычислений
2. **Пакетная обработка**: Возможность обработки нескольких ордербуков одновременно
3. **Асинхронность**: Неблокирующая обработка потоков данных
4. **Параллелизм**: Использование многопоточности для тяжелых вычислений

### Мониторинг

```python
# Статистика производительности
stats = filter_obj.get_filter_statistics()
print(f"Average processing time: {stats.get('avg_processing_time', 0):.3f}ms")
print(f"Memory usage: {stats.get('memory_usage', 0):.2f}MB")
print(f"CPU usage: {stats.get('cpu_usage', 0):.1f}%")
```

## Безопасность

### Валидация данных

- Проверка корректности входных данных
- Обработка исключений и ошибок
- Логирование подозрительной активности

### Конфиденциальность

- Не сохраняются персональные данные
- Анонимизация статистики
- Безопасное хранение конфигурации

## Заключение

Система анализа нейронного шума предоставляет мощные инструменты для обнаружения синтетических паттернов в рыночных данных. Интеграция с существующей архитектурой ATB обеспечивает бесшовную работу с торговыми системами без нарушения основной логики.

Система легко расширяется и настраивается под конкретные требования, предоставляя детальную статистику и мониторинг для принятия обоснованных торговых решений. 