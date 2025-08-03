# Mirror Neuron Signal Detection

## Обзор

Система Mirror Neuron Signal Detection предназначена для обнаружения повторяющихся движений между активами на основе корреляционного анализа с учетом временных лагов. Система строит карту зависимостей между активами и предоставляет инструменты для интеграции с торговыми стратегиями.

## Архитектура

### Domain Layer

#### `MirrorDetector` (`domain/intelligence/mirror_detector.py`)

Основной класс для обнаружения зеркальных нейронных сигналов.

**Ключевые методы:**

- `detect_lagged_correlation(asset1: pd.Series, asset2: pd.Series, max_lag: int = 5) -> Tuple[int, float]`
  - Обнаруживает корреляцию с лагом между двумя активами
  - Возвращает `(best_lag, correlation)`
  - Использует различные методы корреляции (Pearson, Spearman, Kendall)

- `detect_mirror_signal(asset1: str, asset2: str, series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> Optional[MirrorSignal]`
  - Полный анализ зеркального сигнала между активами
  - Возвращает структурированный результат с уверенностью и метаданными

- `build_correlation_matrix(assets: List[str], price_data: Dict[str, pd.Series], max_lag: int = 5) -> CorrelationMatrix`
  - Строит матрицу корреляций для всех активов
  - Включает матрицы лагов, p-values и уверенности

**Модели данных:**

```python
@dataclass
class MirrorSignal:
    asset1: str
    asset2: str
    best_lag: int
    correlation: float
    p_value: float
    confidence: float
    signal_strength: float
    timestamp: Timestamp
    metadata: Dict[str, Any]

@dataclass
class CorrelationMatrix:
    assets: List[str]
    correlation_matrix: np.ndarray
    lag_matrix: np.ndarray
    p_value_matrix: np.ndarray
    confidence_matrix: np.ndarray
    timestamp: Timestamp
```

### Application Layer

#### `MirrorMapBuilder` (`application/strategy_advisor/mirror_map_builder.py`)

Сервис для построения карты зеркальных зависимостей между активами.

**Основные возможности:**

- Прием списка активов и данных о ценах
- Построение матрицы корреляций с лагами
- Создание карты зависимостей `mirror_map: Dict[str, List[str]]`
- Анализ кластеров активов
- Интеграция с торговыми стратегиями

**Конфигурация:**

```python
@dataclass
class MirrorMapConfig:
    min_correlation: float = 0.3
    max_p_value: float = 0.05
    min_confidence: float = 0.7
    max_lag: int = 5
    correlation_method: str = 'pearson'
    normalize_data: bool = True
    remove_trend: bool = True
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    update_interval: int = 3600
    parallel_processing: bool = True
    max_workers: int = 4
```

## Алгоритм анализа

### 1. Предобработка данных

```python
def _preprocess_series(self, series: pd.Series) -> pd.Series:
    # Удаление NaN значений
    clean_series = series.dropna()
    
    # Удаление тренда (если включено)
    if self.remove_trend:
        clean_series = clean_series.diff().dropna()
    
    # Нормализация (если включено)
    if self.normalize_data:
        mean_val = clean_series.mean()
        std_val = clean_series.std()
        if std_val > 0:
            clean_series = (clean_series - mean_val) / std_val
    
    return clean_series
```

### 2. Обнаружение корреляции с лагом

```python
def detect_lagged_correlation(self, asset1: pd.Series, asset2: pd.Series, max_lag: int = 5) -> Tuple[int, float]:
    # Предобработка данных
    series1 = self._preprocess_series(asset1)
    series2 = self._preprocess_series(asset2)
    
    # Поиск лучшего лага
    best_lag = 0
    best_correlation = 0.0
    
    for lag in range(-max_lag, max_lag + 1):
        correlation, p_value = self._compute_correlation_with_lag(series1, series2, lag)
        
        if abs(correlation) > abs(best_correlation) and p_value < self.max_p_value:
            best_lag = lag
            best_correlation = correlation
    
    return best_lag, best_correlation
```

### 3. Построение матрицы корреляций

```python
def build_correlation_matrix(self, assets: List[str], price_data: Dict[str, pd.Series], max_lag: int = 5) -> CorrelationMatrix:
    n = len(assets)
    correlation_matrix = np.zeros((n, n))
    lag_matrix = np.zeros((n, n), dtype=int)
    p_value_matrix = np.ones((n, n))
    confidence_matrix = np.zeros((n, n))
    
    # Вычисляем корреляции для всех пар
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:
                signal = self.detect_mirror_signal(asset1, asset2, price_data[asset1], price_data[asset2], max_lag)
                if signal:
                    correlation_matrix[i, j] = signal.correlation
                    correlation_matrix[j, i] = signal.correlation
                    lag_matrix[i, j] = signal.best_lag
                    lag_matrix[j, i] = -signal.best_lag
                    p_value_matrix[i, j] = signal.p_value
                    p_value_matrix[j, i] = signal.p_value
                    confidence_matrix[i, j] = signal.confidence
                    confidence_matrix[j, i] = signal.confidence
    
    return CorrelationMatrix(assets, correlation_matrix, lag_matrix, p_value_matrix, confidence_matrix)
```

## Использование

### Базовое обнаружение корреляции

```python
from domain.intelligence.mirror_detector import MirrorDetector
import pandas as pd

# Создание детектора
detector = MirrorDetector(
    min_correlation=0.3,
    max_p_value=0.05,
    min_confidence=0.7,
    max_lag=5
)

# Создание тестовых данных
btc_prices = pd.Series([50000, 50100, 50200, 50300, 50400])
eth_prices = pd.Series([3000, 3005, 3010, 3015, 3020])

# Обнаружение корреляции с лагом
best_lag, correlation = detector.detect_lagged_correlation(btc_prices, eth_prices, max_lag=5)
print(f"Best lag: {best_lag}, Correlation: {correlation:.3f}")
```

### Построение карты зеркальных зависимостей

```python
from application.strategy_advisor.mirror_map_builder import MirrorMapBuilder, MirrorMapConfig

# Создание конфигурации
config = MirrorMapConfig(
    min_correlation=0.3,
    max_p_value=0.05,
    max_lag=5,
    parallel_processing=True
)

# Создание построителя
builder = MirrorMapBuilder(config)

# Подготовка данных
assets = ['BTC', 'ETH', 'ADA', 'DOT']
price_data = {
    'BTC': pd.Series([50000, 50100, 50200, 50300, 50400]),
    'ETH': pd.Series([3000, 3005, 3010, 3015, 3020]),
    'ADA': pd.Series([0.5, 0.51, 0.52, 0.53, 0.54]),
    'DOT': pd.Series([7.0, 7.05, 7.10, 7.15, 7.20])
}

# Построение карты
mirror_map = builder.build_mirror_map(assets, price_data, force_rebuild=True)

# Анализ результатов
print(f"Assets with dependencies: {len(mirror_map.mirror_map)}")
print(f"Total clusters: {len(mirror_map.clusters)}")

# Получение зеркальных активов для стратегии
mirror_assets = builder.get_mirror_assets_for_strategy(mirror_map, 'BTC', min_correlation=0.3)
for asset, correlation, lag in mirror_assets:
    print(f"BTC -> {asset}: corr={correlation:.3f}, lag={lag}")
```

### Асинхронное построение карты

```python
import asyncio

async def build_mirror_map_async():
    builder = MirrorMapBuilder()
    assets = ['BTC', 'ETH', 'ADA']
    price_data = {
        'BTC': pd.Series([50000, 50100, 50200, 50300, 50400]),
        'ETH': pd.Series([3000, 3005, 3010, 3015, 3020]),
        'ADA': pd.Series([0.5, 0.51, 0.52, 0.53, 0.54])
    }
    
    mirror_map = await builder.build_mirror_map_async(assets, price_data, force_rebuild=True)
    return mirror_map

# Запуск
mirror_map = asyncio.run(build_mirror_map_async())
```

## Конфигурация

### Параметры детектора

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `min_correlation` | Минимальная корреляция для обнаружения | 0.3 |
| `max_p_value` | Максимальное p-value для значимости | 0.05 |
| `min_confidence` | Минимальная уверенность | 0.7 |
| `max_lag` | Максимальный лаг для поиска | 5 |
| `correlation_method` | Метод корреляции (pearson/spearman/kendall) | 'pearson' |
| `normalize_data` | Нормализация данных | True |
| `remove_trend` | Удаление тренда | True |

### Параметры построителя карты

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `min_cluster_size` | Минимальный размер кластера | 2 |
| `max_cluster_size` | Максимальный размер кластера | 10 |
| `update_interval` | Интервал обновления кэша (секунды) | 3600 |
| `parallel_processing` | Параллельная обработка | True |
| `max_workers` | Максимальное количество потоков | 4 |

## API Reference

### MirrorDetector

#### `__init__(min_correlation=0.3, max_p_value=0.05, min_confidence=0.7, max_lag=5, correlation_method='pearson', normalize_data=True, remove_trend=True)`

Инициализация детектора.

#### `detect_lagged_correlation(asset1: pd.Series, asset2: pd.Series, max_lag: int = 5) -> Tuple[int, float]`

Обнаружение корреляции с лагом между двумя активами.

#### `detect_mirror_signal(asset1: str, asset2: str, series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> Optional[MirrorSignal]`

Полный анализ зеркального сигнала.

#### `build_correlation_matrix(assets: List[str], price_data: Dict[str, pd.Series], max_lag: int = 5) -> CorrelationMatrix`

Построение матрицы корреляций.

#### `find_mirror_clusters(correlation_matrix: CorrelationMatrix, min_correlation: Optional[float] = None) -> List[List[str]]`

Поиск кластеров зеркальных активов.

### MirrorMapBuilder

#### `__init__(config: Optional[MirrorMapConfig] = None)`

Инициализация построителя карты.

#### `build_mirror_map(assets: List[str], price_data: Dict[str, pd.Series], force_rebuild: bool = False) -> MirrorMap`

Синхронное построение карты зеркальных зависимостей.

#### `build_mirror_map_async(assets: List[str], price_data: Dict[str, pd.Series], force_rebuild: bool = False) -> MirrorMap`

Асинхронное построение карты.

#### `get_mirror_assets_for_strategy(mirror_map: MirrorMap, base_asset: str, min_correlation: Optional[float] = None) -> List[Tuple[str, float, int]]`

Получение зеркальных активов для торговой стратегии.

#### `analyze_mirror_clusters(mirror_map: MirrorMap) -> Dict[str, Any]`

Анализ кластеров зеркальных активов.

## Примеры

### Пример 1: Обнаружение зеркальных сигналов

```python
from domain.intelligence.mirror_detector import MirrorDetector
import numpy as np

# Создание коррелированных данных
np.random.seed(42)
base_trend = np.linspace(100, 110, 100)
btc_prices = pd.Series(base_trend + np.random.normal(0, 1, 100))
eth_prices = pd.Series(base_trend + np.random.normal(0, 1, 100))

# Добавляем корреляцию с лагом
eth_prices.iloc[2:] += btc_prices.iloc[:-2] * 0.3

# Обнаружение сигнала
detector = MirrorDetector()
signal = detector.detect_mirror_signal('BTC', 'ETH', btc_prices, eth_prices, max_lag=5)

if signal:
    print(f"Mirror signal detected:")
    print(f"  Best lag: {signal.best_lag}")
    print(f"  Correlation: {signal.correlation:.3f}")
    print(f"  P-value: {signal.p_value:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
```

### Пример 2: Построение карты зависимостей

```python
from application.strategy_advisor.mirror_map_builder import MirrorMapBuilder

# Создание данных
assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
price_data = {}

for i, asset in enumerate(assets):
    base_price = 100 + i * 10
    prices = pd.Series([base_price + j + np.random.normal(0, 1) for j in range(100)])
    price_data[asset] = prices

# Построение карты
builder = MirrorMapBuilder()
mirror_map = builder.build_mirror_map(assets, price_data)

# Анализ результатов
print(f"Mirror dependencies:")
for asset, mirrors in mirror_map.mirror_map.items():
    print(f"  {asset} -> {mirrors}")

print(f"\nClusters:")
for i, cluster in enumerate(mirror_map.clusters):
    print(f"  Cluster {i+1}: {cluster}")
```

### Пример 3: Интеграция с торговой стратегией

```python
class MirrorTradingStrategy:
    def __init__(self, mirror_map: MirrorMap):
        self.mirror_map = mirror_map
    
    def analyze_mirror_signals(self, base_asset: str, price_change: float):
        """Анализ зеркальных сигналов для торгового решения."""
        mirror_assets = self.mirror_map.get_mirror_assets(base_asset)
        
        signals = []
        for mirror_asset in mirror_assets:
            correlation = self.mirror_map.get_correlation(base_asset, mirror_asset)
            lag = self.mirror_map.get_lag(base_asset, mirror_asset)
            
            if abs(correlation) > 0.5:
                predicted_change = price_change * correlation
                signal = {
                    'asset': mirror_asset,
                    'predicted_change': predicted_change,
                    'confidence': abs(correlation),
                    'lag_periods': lag
                }
                signals.append(signal)
        
        return signals

# Использование
strategy = MirrorTradingStrategy(mirror_map)
signals = strategy.analyze_mirror_signals('BTC', 0.05)  # 5% рост BTC

for signal in signals:
    print(f"Signal for {signal['asset']}: {signal['predicted_change']:.1%} change expected")
```

## Тестирование

### Запуск тестов

```bash
# Запуск всех тестов
pytest tests/test_mirror_neuron_signal.py -v

# Запуск интеграционных тестов
pytest tests/test_mirror_neuron_integration.py -v

# Запуск с покрытием
pytest tests/test_mirror_neuron_signal.py --cov=domain.intelligence.mirror_detector --cov=application.strategy_advisor.mirror_map_builder
```

### Примеры

```bash
# Запуск примера
python examples/mirror_neuron_signal_example.py
```

## Производительность

### Оптимизации

1. **Параллельная обработка**: Использование ThreadPoolExecutor для одновременного анализа пар активов
2. **Кэширование**: Результаты кэшируются для повторного использования
3. **Векторизация**: Использование numpy для быстрых вычислений
4. **Асинхронность**: Неблокирующая обработка для больших объемов данных

### Мониторинг

```python
# Статистика производительности
stats = builder.get_mirror_map_statistics()
print(f"Cache info: {stats['cache_info']}")
print(f"Mirror map info: {stats['mirror_map_info']}")
```

## Интеграция с ATB

### Совместимость

- **Domain Layer**: `MirrorDetector` и модели данных
- **Application Layer**: `MirrorMapBuilder` для координации
- **Infrastructure Layer**: Готов к интеграции с коннекторами бирж
- **Interfaces Layer**: Может быть использован в API и CLI

### Интеграция с AgentContext

```python
class MirrorSignalIntegration:
    def __init__(self, mirror_map: MirrorMap):
        self.mirror_map = mirror_map
    
    def update_agent_context(self, agent_context: dict, base_asset: str, price_change: float):
        """Обновление контекста агента зеркальными сигналами."""
        mirror_signals = self.mirror_map.get_mirror_assets(base_asset)
        
        agent_context['mirror_signals'] = {
            'base_asset': base_asset,
            'price_change': price_change,
            'mirror_assets': mirror_signals,
            'predictions': []
        }
        
        for asset in mirror_signals:
            correlation = self.mirror_map.get_correlation(base_asset, asset)
            lag = self.mirror_map.get_lag(base_asset, asset)
            
            prediction = {
                'asset': asset,
                'predicted_change': price_change * correlation,
                'confidence': abs(correlation),
                'lag_periods': lag
            }
            agent_context['mirror_signals']['predictions'].append(prediction)
        
        return agent_context
```

## Расширение системы

### Добавление новых методов корреляции

```python
class ExtendedMirrorDetector(MirrorDetector):
    def compute_spectral_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Вычисление спектральной корреляции."""
        # Реализация спектрального анализа
        pass
    
    def detect_mirror_signal(self, asset1: str, asset2: str, series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> Optional[MirrorSignal]:
        """Расширенный анализ с новыми метриками."""
        # Базовый анализ
        signal = super().detect_mirror_signal(asset1, asset2, series1, series2, max_lag)
        
        if signal:
            # Добавление новых метрик
            spectral_corr = self.compute_spectral_correlation(series1, series2)
            signal.metadata['spectral_correlation'] = spectral_corr
        
        return signal
```

### Интеграция с другими системами

```python
class ComprehensiveAnalysis:
    def __init__(self):
        self.mirror_detector = MirrorDetector()
        self.noise_analyzer = NoiseAnalyzer()
        self.entanglement_detector = EntanglementDetector()
    
    def analyze_asset_relationships(self, assets: List[str], price_data: Dict[str, pd.Series]):
        """Комплексный анализ отношений между активами."""
        # Анализ зеркальных сигналов
        mirror_signals = self.mirror_detector.build_correlation_matrix(assets, price_data)
        
        # Анализ нейронного шума
        noise_results = {}
        for asset in assets:
            order_book = self._create_order_book_from_prices(price_data[asset])
            noise_results[asset] = self.noise_analyzer.analyze_noise(order_book)
        
        # Анализ запутанности (если есть данные с разных бирж)
        # entanglement_results = self.entanglement_detector.detect_entanglement(...)
        
        return {
            'mirror_signals': mirror_signals,
            'noise_analysis': noise_results,
            # 'entanglement_analysis': entanglement_results
        }
```

## Заключение

Система Mirror Neuron Signal Detection предоставляет мощные инструменты для анализа корреляций между активами с учетом временных лагов. Интеграция с архитектурой ATB обеспечивает бесшовную работу с торговыми системами без прямого влияния на стратегии.

Система легко расширяется и настраивается, предоставляя детальную статистику и инструменты для принятия обоснованных торговых решений на основе зеркальных зависимостей между активами. 