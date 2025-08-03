# Liquidity Gravity Field Model

## Обзор

Система Liquidity Gravity Field Model реализована по принципам Domain-Driven Design (DDD) и предназначена для анализа ликвидности ордербука на основе физической модели гравитации. Система вычисляет силу притяжения между бидами и асками и использует эту информацию для оценки рисков и корректировки торговых стратегий.

## Архитектура

### Domain Layer

#### `LiquidityGravityModel` (`domain/market/liquidity_gravity.py`)

Основной класс для анализа гравитации ликвидности.

**Ключевые методы:**

- `compute_liquidity_gravity(order_book: OrderBookSnapshot) -> float`
  - Вычисляет силу гравитации ликвидности
  - Использует формулу: F = G * v₁ * v₂ / (Δp)²
  - Возвращает нормализованную метрику искривления

- `analyze_liquidity_gravity(order_book: OrderBookSnapshot) -> LiquidityGravityResult`
  - Полный анализ гравитации ликвидности
  - Включает распределение сил, уровень риска, метаданные

- `compute_gravity_gradient(order_book: OrderBookSnapshot) -> Dict[str, float]`
  - Вычисляет градиент гравитации по уровням цен

**Модели данных:**

```python
@dataclass
class LiquidityGravityConfig:
    gravitational_constant: float = 1e-6
    min_volume_threshold: float = 0.001
    max_price_distance: float = 0.1
    volume_weight: float = 1.0
    price_weight: float = 1.0
    decay_factor: float = 0.95
    normalization_factor: float = 1e6

@dataclass
class LiquidityGravityResult:
    total_gravity: float
    bid_ask_forces: List[Tuple[float, float, float]]
    gravity_distribution: Dict[str, float]
    risk_level: str
    timestamp: Timestamp
    metadata: Dict[str, Any]

@dataclass
class OrderBookSnapshot:
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    timestamp: Timestamp
    symbol: str
```

### Application Layer

#### `LiquidityRiskAssessor` (`application/risk/liquidity_gravity_monitor.py`)

Сервис для оценки рисков на основе гравитации ликвидности.

**Основные возможности:**

- Оценка риска ликвидности для агентов
- Корректировка агрессивности агентов
- Управление состоянием риска агентов
- Мониторинг и статистика

**Модели данных:**

```python
@dataclass
class RiskThresholds:
    low_risk: float = 0.1
    medium_risk: float = 0.5
    high_risk: float = 1.0
    extreme_risk: float = 2.0
    reduce_aggression_threshold: float = 0.8
    stop_trading_threshold: float = 2.0

@dataclass
class AgentRiskState:
    current_aggression: float = 1.0
    base_aggression: float = 1.0
    risk_level: str = 'low'
    gravity_history: deque
    last_update: float
    risk_start_time: Optional[float] = None
    recovery_start_time: Optional[float] = None

@dataclass
class RiskAssessmentResult:
    risk_level: str
    gravity_value: float
    agent_aggression: float
    recommended_action: str
    confidence: float
    timestamp: Timestamp
    metadata: Dict[str, Any]
```

#### `LiquidityGravityFilter`

Фильтр для принятия торговых решений на основе гравитации ликвидности.

**Основные методы:**

- `should_proceed_with_trade()` - проверка возможности торговли
- `get_adjusted_aggression()` - получение скорректированной агрессивности

## Алгоритм гравитации ликвидности

### 1. Физическая модель

Система использует физическую модель гравитации для анализа ликвидности:

```
F = G * v₁ * v₂ / (Δp)²
```

Где:
- `F` - сила гравитации между бидом и аском
- `G` - гравитационная постоянная (1e-6 по умолчанию)
- `v₁, v₂` - объемы бида и аска
- `Δp` - расстояние между ценами

### 2. Предобработка данных

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

### 3. Вычисление гравитации

```python
def compute_liquidity_gravity(self, order_book: OrderBookSnapshot) -> float:
    total_gravity = 0.0
    mid_price = order_book.get_mid_price()
    
    # Вычисляем силу гравитации между всеми парами bid-ask
    for bid_price, bid_volume in order_book.bids:
        for ask_price, ask_volume in order_book.asks:
            # Пропускаем слишком малые объемы
            if bid_volume < self.config.min_volume_threshold or ask_volume < self.config.min_volume_threshold:
                continue
            
            # Вычисляем расстояние между ценами
            price_distance = abs(ask_price - bid_price)
            
            # Пропускаем слишком большие расстояния
            if price_distance > mid_price * self.config.max_price_distance:
                continue
            
            # Вычисляем силу гравитации: F = G * v₁ * v₂ / (Δp)²
            force = self._compute_gravitational_force(bid_volume, ask_volume, price_distance)
            total_gravity += force
    
    # Нормализация результата
    normalized_gravity = total_gravity / self.config.normalization_factor
    
    return normalized_gravity
```

### 4. Оценка риска

```python
def _evaluate_risk(self, agent_state: AgentRiskState, gravity_value: float, current_time: float) -> Dict[str, Any]:
    # Определяем уровень риска
    if gravity_value < self.risk_thresholds.low_risk:
        risk_level = 'low'
    elif gravity_value < self.risk_thresholds.medium_risk:
        risk_level = 'medium'
    elif gravity_value < self.risk_thresholds.high_risk:
        risk_level = 'high'
    elif gravity_value < self.risk_thresholds.extreme_risk:
        risk_level = 'extreme'
    else:
        risk_level = 'critical'
    
    # Корректируем агрессивность агента
    if gravity_value > self.risk_thresholds.stop_trading_threshold:
        agent_state.current_aggression = 0.0
    elif gravity_value > self.risk_thresholds.reduce_aggression_threshold:
        reduction_factor = 1.0 - (gravity_value - self.risk_thresholds.reduce_aggression_threshold) / (self.risk_thresholds.stop_trading_threshold - self.risk_thresholds.reduce_aggression_threshold)
        agent_state.current_aggression = max(0.1, agent_state.base_aggression * reduction_factor)
    else:
        # Восстанавливаем агрессивность
        recovery_rate = 0.01
        agent_state.current_aggression = min(agent_state.base_aggression, agent_state.current_aggression + recovery_rate)
    
    return {
        'risk_level': risk_level,
        'aggression_change': agent_state.current_aggression - old_aggression,
        'time_in_risk': current_time - agent_state.risk_start_time if agent_state.risk_start_time else 0.0
    }
```

## Использование

### Базовое вычисление гравитации

```python
from domain.market.liquidity_gravity import LiquidityGravityModel, OrderBookSnapshot
from domain.value_objects.timestamp import Timestamp

# Создание модели
model = LiquidityGravityModel()

# Создание ордербука
order_book = OrderBookSnapshot(
    bids=[(50000, 1.0), (49999, 1.0)],
    asks=[(50001, 1.0), (50002, 1.0)],
    timestamp=Timestamp(time.time()),
    symbol="BTC/USDT"
)

# Вычисление гравитации
gravity = model.compute_liquidity_gravity(order_book)
print(f"Liquidity gravity: {gravity:.6f}")
```

### Полный анализ гравитации

```python
# Полный анализ
result = model.analyze_liquidity_gravity(order_book)

print(f"Total gravity: {result.total_gravity:.6f}")
print(f"Risk level: {result.risk_level}")
print(f"Gravity distribution: {result.gravity_distribution}")
print(f"Bid-ask forces: {len(result.bid_ask_forces)} pairs")
```

### Оценка рисков

```python
from application.risk.liquidity_gravity_monitor import LiquidityRiskAssessor, RiskThresholds

# Создание оценщика рисков
risk_assessor = LiquidityRiskAssessor()

# Установка базовой агрессивности агента
risk_assessor.set_agent_base_aggression("market_maker_1", 0.8)

# Оценка риска
risk_result = risk_assessor.assess_liquidity_risk(order_book, "market_maker_1")

print(f"Risk level: {risk_result.risk_level}")
print(f"Agent aggression: {risk_result.agent_aggression:.3f}")
print(f"Recommended action: {risk_result.recommended_action}")
print(f"Confidence: {risk_result.confidence:.3f}")
```

### Использование фильтра

```python
from application.risk.liquidity_gravity_monitor import LiquidityGravityFilter

# Создание фильтра
gravity_filter = LiquidityGravityFilter(risk_assessor)

# Проверка возможности торговли
should_proceed, metadata = gravity_filter.should_proceed_with_trade(
    order_book, "market_maker_1", trade_aggression=0.8
)

if should_proceed:
    print("Proceed with trade")
else:
    print(f"Stop trading: {metadata['decision']['reason']}")

# Получение скорректированной агрессивности
adjusted_aggression, metadata = gravity_filter.get_adjusted_aggression(
    order_book, "market_maker_1", base_aggression=0.8
)

print(f"Adjusted aggression: {adjusted_aggression:.3f}")
```

## Конфигурация

### Параметры гравитации

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `gravitational_constant` | Гравитационная постоянная | 1e-6 |
| `min_volume_threshold` | Минимальный объем для анализа | 0.001 |
| `max_price_distance` | Максимальное расстояние цен (в % от средней цены) | 0.1 |
| `volume_weight` | Вес объема в формуле | 1.0 |
| `price_weight` | Вес цены в формуле | 1.0 |
| `decay_factor` | Фактор затухания | 0.95 |
| `normalization_factor` | Фактор нормализации | 1e6 |

### Параметры риска

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `low_risk` | Порог низкого риска | 0.1 |
| `medium_risk` | Порог среднего риска | 0.5 |
| `high_risk` | Порог высокого риска | 1.0 |
| `extreme_risk` | Порог экстремального риска | 2.0 |
| `reduce_aggression_threshold` | Порог снижения агрессивности | 0.8 |
| `stop_trading_threshold` | Порог остановки торговли | 2.0 |
| `risk_duration_threshold` | Порог длительности риска (секунды) | 300 |
| `recovery_threshold` | Порог восстановления (секунды) | 600 |

## Тестирование

### Запуск тестов

```bash
# Запуск всех тестов
pytest tests/test_liquidity_gravity.py -v

# Запуск с покрытием
pytest tests/test_liquidity_gravity.py --cov=domain.market.liquidity_gravity --cov=application.risk.liquidity_gravity_monitor
```

### Примеры

```bash
# Запуск примера
python examples/liquidity_gravity_example.py
```

## Производительность

### Оптимизации

1. **Эффективные вычисления**: Использование numpy для быстрых вычислений
2. **Фильтрация данных**: Пропуск незначимых объемов и больших расстояний
3. **Кэширование**: Кэширование результатов анализа
4. **Векторизация**: Векторизованные операции с массивами

### Мониторинг

```python
# Статистика модели
stats = model.get_model_statistics()
print(f"Model statistics: {stats}")

# Статистика рисков
risk_stats = risk_assessor.get_risk_statistics()
print(f"Risk statistics: {risk_stats}")
```

## Интеграция с ATB

### Совместимость

- **Domain Layer**: `LiquidityGravityModel` и модели данных
- **Application Layer**: `LiquidityRiskAssessor` для координации
- **Infrastructure Layer**: Готов к интеграции с коннекторами бирж
- **Interfaces Layer**: Может быть использован в API и CLI

### Интеграция с MarketMakerModelAgent

```python
class MarketMakerModelAgent:
    def __init__(self):
        self.risk_assessor = LiquidityRiskAssessor()
        self.gravity_filter = LiquidityGravityFilter(self.risk_assessor)
        self.risk_assessor.set_agent_base_aggression("market_maker", 0.8)
    
    def make_trading_decision(self, order_book: OrderBookSnapshot):
        """Принятие торгового решения с учетом гравитации ликвидности."""
        
        # Проверяем, следует ли торговать
        should_proceed, trade_metadata = self.gravity_filter.should_proceed_with_trade(
            order_book, "market_maker", trade_aggression=0.8
        )
        
        if not should_proceed:
            return {
                'action': 'hold',
                'reason': trade_metadata['decision']['reason'],
                'metadata': trade_metadata
            }
        
        # Получаем скорректированную агрессивность
        adjusted_aggression, aggression_metadata = self.gravity_filter.get_adjusted_aggression(
            order_book, "market_maker", base_aggression=0.8
        )
        
        # Принимаем торговое решение с учетом скорректированной агрессивности
        decision = self._make_decision_with_aggression(order_book, adjusted_aggression)
        
        return {
            'action': decision['action'],
            'aggression': adjusted_aggression,
            'metadata': {
                'trade_metadata': trade_metadata,
                'aggression_metadata': aggression_metadata,
                'decision_metadata': decision['metadata']
            }
        }
    
    def _make_decision_with_aggression(self, order_book: OrderBookSnapshot, aggression: float):
        """Принятие решения с учетом агрессивности."""
        # Здесь реализуется логика принятия торгового решения
        # с учетом скорректированной агрессивности
        pass
```

## Расширение системы

### Добавление новых метрик

```python
class ExtendedLiquidityGravityModel(LiquidityGravityModel):
    def compute_liquidity_pressure(self, order_book: OrderBookSnapshot) -> float:
        """Вычисление давления ликвидности."""
        # Реализация новой метрики
        pass
    
    def analyze_liquidity_gravity(self, order_book: OrderBookSnapshot) -> LiquidityGravityResult:
        """Расширенный анализ с новыми метриками."""
        # Базовый анализ
        result = super().analyze_liquidity_gravity(order_book)
        
        # Добавление новых метрик
        pressure = self.compute_liquidity_pressure(order_book)
        result.metadata['liquidity_pressure'] = pressure
        
        return result
```

### Интеграция с другими системами

```python
class ComprehensiveLiquidityAnalysis:
    def __init__(self):
        self.gravity_model = LiquidityGravityModel()
        self.noise_analyzer = NoiseAnalyzer()
        self.mirror_detector = MirrorDetector()
    
    def analyze_liquidity_comprehensive(self, order_book: OrderBookSnapshot):
        """Комплексный анализ ликвидности."""
        # Анализ гравитации ликвидности
        gravity_result = self.gravity_model.analyze_liquidity_gravity(order_book)
        
        # Анализ нейронного шума
        noise_result = self.noise_analyzer.analyze_noise(order_book)
        
        # Анализ зеркальных сигналов (если есть данные о ценах)
        # mirror_result = self.mirror_detector.detect_mirror_signal(...)
        
        return {
            'gravity_analysis': gravity_result.to_dict(),
            'noise_analysis': noise_result.to_dict(),
            # 'mirror_analysis': mirror_result.to_dict()
        }
```

## Заключение

Система Liquidity Gravity Field Model предоставляет мощные инструменты для анализа ликвидности ордербука на основе физической модели гравитации. Интеграция с архитектурой ATB обеспечивает бесшовную работу с торговыми системами и позволяет корректировать агрессивность агентов на основе рисков ликвидности.

Система легко расширяется и настраивается, предоставляя детальную статистику и инструменты для принятия обоснованных торговых решений на основе анализа гравитации ликвидности. 