# Система Прогнозирования Разворотов

## Обзор

Система прогнозирования разворотов (Reversal Prediction System) представляет собой продвинутый модуль для выявления потенциальных разворотов цены на финансовых рынках. Система использует комплексный подход, объединяющий технический анализ, анализ объёмов, кластеризацию ликвидности и машинное обучение.

## Архитектура

### Domain Layer

#### `ReversalSignal` - Основная модель сигнала

```python
@dataclass
class ReversalSignal:
    symbol: str
    direction: ReversalDirection
    pivot_price: Price
    confidence: float
    horizon: timedelta
    signal_strength: float
    timestamp: Timestamp
    
    # Компоненты анализа
    pivot_points: List[PivotPoint]
    fibonacci_levels: List[FibonacciLevel]
    volume_profile: Optional[VolumeProfile]
    liquidity_clusters: List[LiquidityCluster]
    
    # Технические сигналы
    divergence_signals: List[DivergenceSignal]
    candlestick_patterns: List[CandlestickPattern]
    momentum_analysis: Optional[MomentumAnalysis]
    mean_reversion_band: Optional[MeanReversionBand]
    
    # Интеграционные флаги
    is_controversial: bool
    agreement_score: float
    controversy_reasons: List[Dict[str, Any]]
```

**Ключевые свойства:**
- `strength_category`: Категория силы сигнала (WEAK, MODERATE, STRONG, VERY_STRONG)
- `risk_level`: Уровень риска (low, medium, high, extreme)
- `is_expired`: Проверка истечения срока действия
- `time_to_expiry`: Время до истечения

#### Вспомогательные модели

- **PivotPoint**: Точки разворота с силой и уровнями подтверждения
- **FibonacciLevel**: Уровни Фибоначчи с кластерами объёма
- **VolumeProfile**: Профиль объёма с Point of Control
- **LiquidityCluster**: Кластеры ликвидности из ордербука
- **DivergenceSignal**: Сигналы дивергенции RSI/MACD
- **CandlestickPattern**: Свечные паттерны с подтверждением
- **MomentumAnalysis**: Анализ импульса цены и объёма
- **MeanReversionBand**: Полосы возврата к среднему

### Infrastructure Layer

#### `PricePatternExtractor` - Извлекатель паттернов

**Основные методы:**
- `extract_pivot_points()`: Извлечение точек разворота
- `calculate_fibonacci_levels()`: Вычисление уровней Фибоначчи
- `extract_volume_profile()`: Извлечение профиля объёма
- `extract_liquidity_clusters()`: Извлечение кластеров ликвидности

**Продвинутые алгоритмы:**
- Адаптивный поиск пиков с проминентностью
- Кластеризация ценовых уровней DBSCAN
- Взвешенный анализ объёмных профилей
- Анализ дисбаланса bid/ask

#### `ReversalPredictor` - Основной прогнозатор

**Конфигурация:**
```python
@dataclass
class PredictionConfig:
    lookback_period: int = 100
    min_confidence: float = 0.3
    min_signal_strength: float = 0.4
    prediction_horizon: timedelta = timedelta(hours=4)
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    divergence_lookback: int = 20
    momentum_window: int = 14
    mean_reversion_window: int = 20
```

**Алгоритмы анализа:**
- Дивергенции RSI и MACD (регулярные, скрытые, тройные)
- Свечные паттерны (Doji, Hammer, Shooting Star)
- Анализ импульса с ускорением и замедлением
- Mean reversion с полосами Боллинджера

### Application Layer

#### `ReversalController` - Контроллер интеграции

**Конфигурация:**
```python
@dataclass
class ControllerConfig:
    update_interval: float = 30.0
    max_signals_per_symbol: int = 5
    signal_lifetime: timedelta = timedelta(hours=2)
    min_agreement_score: float = 0.3
    max_controversy_threshold: float = 0.7
    confidence_boost_factor: float = 0.2
    confidence_reduction_factor: float = 0.15
```

**Интеграционные функции:**
- Асинхронный мониторинг рыночных данных
- Связь с AgentContext и глобальными прогнозами
- Анализ согласованности и спорности
- Динамическое управление уверенностью
- Очистка устаревших сигналов

## Алгоритмы и Методы

### 1. Извлечение Пивотов

**Продвинутый поиск экстремумов:**
```python
def _find_advanced_peaks(self, data: np.ndarray, peak_type: str) -> List[int]:
    if peak_type == 'high':
        peaks, properties = find_peaks(
            data, 
            prominence=data.std() * prominence_factor,
            distance=self.pivot_window
        )
    else:
        peaks, properties = find_peaks(
            -data, 
            prominence=data.std() * prominence_factor,
            distance=self.pivot_window
        )
```

**Вычисление силы пивота:**
- Ценовая компонента (отклонение от окружающих экстремумов)
- Объёмная компонента (отношение к среднему объёму)
- Временная компонента (близость к текущему моменту)
- Волатильность компонента (ширина диапазона)

### 2. Анализ Дивергенций

**RSI Дивергенции:**
- Регулярные: цена растёт, RSI падает (медвежья)
- Скрытые: цена падает, RSI растёт (бычья)
- Тройные: множественные подтверждения

**MACD Дивергенции:**
- Анализ расхождений между ценой и MACD
- Учёт сигнальной линии и гистограммы
- Фильтрация по силе сигнала

### 3. Свечные Паттерны

**Doji:**
```python
def _is_doji(self, candle: pd.Series) -> bool:
    body_size = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    return body_size <= total_range * 0.1
```

**Hammer/Shooting Star:**
- Анализ соотношения тела и теней
- Подтверждение объёмом
- Учёт контекста тренда

### 4. Анализ Импульса

**Компоненты импульса:**
- Скорость изменения цены (velocity)
- Ускорение/замедление (acceleration)
- Импульс объёма
- Потеря импульса

### 5. Mean Reversion

**Полосы Боллинджера:**
- Скользящее среднее как центральная линия
- Стандартные отклонения как границы
- Позиция цены в полосе
- Сила отклонения от среднего

## Интеграция с Системой

### Связь с AgentContext

```python
async def _get_market_data(self, symbol: str) -> Optional[Any]:
    market_service = self.agent_context.get_market_service()
    data = await market_service.get_ohlcv_data(symbol, '1h', limit=200)
    return data

async def _get_order_book(self, symbol: str) -> Optional[Dict]:
    market_service = self.agent_context.get_market_service()
    order_book = await market_service.get_order_book(symbol, depth=20)
    return order_book
```

### Интеграция с Глобальными Прогнозами

**Анализ согласованности:**
- Сравнение направлений
- Сравнение ценовых уровней
- Сравнение временных горизонтов
- Учёт уверенности глобального прогноза

**Управление уверенностью:**
```python
if agreement_score > 0.7:
    signal.enhance_confidence(self.config.confidence_boost_factor)
elif agreement_score < 0.3:
    signal.reduce_confidence(self.config.confidence_reduction_factor)
```

### Обнаружение Спорных Сигналов

**Критерии спорности:**
- Конфликт с глобальным прогнозом
- Несоответствие уверенности
- Множественные противоположные сигналы
- Короткое время до истечения
- Низкая сила сигнала

## Использование

### Базовое Использование

```python
# Создание прогнозатора
config = PredictionConfig(
    lookback_period=100,
    min_confidence=0.3,
    min_signal_strength=0.4
)
predictor = ReversalPredictor(config)

# Прогнозирование разворота
signal = predictor.predict_reversal('BTCUSDT', market_data, order_book)

if signal:
    print(f"Направление: {signal.direction.value}")
    print(f"Уровень: {signal.pivot_price.value}")
    print(f"Уверенность: {signal.confidence:.3f}")
    print(f"Сила: {signal.signal_strength:.3f}")
```

### Интеграция с Контроллером

```python
# Создание контроллера
controller_config = ControllerConfig(
    update_interval=30.0,
    max_signals_per_symbol=5
)
controller = ReversalController(agent_context, global_predictor, controller_config)

# Запуск мониторинга
await controller.start_monitoring()

# Получение активных сигналов
active_signals = await controller.get_active_signals('BTCUSDT')

# Получение статистики
stats = await controller.get_signal_statistics()
```

### Настройка Параметров

**Для агрессивной торговли:**
```python
config = PredictionConfig(
    min_confidence=0.2,
    min_signal_strength=0.3,
    prediction_horizon=timedelta(hours=2),
    rsi_oversold=25.0,
    rsi_overbought=75.0
)
```

**Для консервативной торговли:**
```python
config = PredictionConfig(
    min_confidence=0.5,
    min_signal_strength=0.6,
    prediction_horizon=timedelta(hours=6),
    rsi_oversold=35.0,
    rsi_overbought=65.0
)
```

## Мониторинг и Отладка

### Логирование

Система использует структурированное логирование с различными уровнями:

```python
logger.info(f"Signal integrated: {signal}")
logger.warning(f"Controversy detected for {symbol}: {reasons}")
logger.debug(f"Enhanced confidence for {symbol}: {confidence:.3f}")
```

### Статистика

```python
stats = await controller.get_signal_statistics()
print(f"Активных сигналов: {stats['active_signals_count']}")
print(f"Символов с сигналами: {stats['symbols_with_signals']}")
print(f"Размер истории: {stats['history_size']}")
print(f"Распределение по направлениям: {stats['direction_distribution']}")
print(f"Распределение по силе: {stats['strength_distribution']}")
```

### Метрики Производительности

- Время обработки сигнала
- Количество ложных срабатываний
- Точность прогнозов
- Задержка интеграции

## Тестирование

### Интеграционные Тесты

```python
def test_end_to_end_prediction_pipeline():
    # Извлечение паттернов
    extractor = PricePatternExtractor()
    high_pivots, low_pivots = extractor.extract_pivot_points(market_data)
    
    # Прогнозирование
    predictor = ReversalPredictor()
    signal = predictor.predict_reversal('BTCUSDT', market_data)
    
    # Проверка результата
    assert signal is not None
    assert signal.symbol == 'BTCUSDT'
    assert len(signal.pivot_points) > 0
```

### Unit Тесты

```python
def test_signal_confidence_management():
    signal = ReversalSignal(...)
    initial_confidence = signal.confidence
    
    signal.enhance_confidence(0.2)
    assert signal.confidence > initial_confidence
    
    signal.reduce_confidence(0.1)
    assert signal.confidence < signal.confidence
```

## Производительность

### Оптимизации

1. **Кэширование данных**: Кэширование рыночных данных и ордербука
2. **Асинхронная обработка**: Параллельная обработка множественных символов
3. **Фильтрация на ранней стадии**: Отклонение слабых сигналов до полного анализа
4. **Очистка памяти**: Автоматическая очистка устаревших данных

### Ограничения

- Минимальный размер данных: 100 свечей
- Максимальное количество сигналов на символ: 5
- Время жизни сигнала: 2 часа
- Интервал обновления: 30 секунд

## Расширение Системы

### Добавление Новых Индикаторов

```python
def _analyze_custom_indicator(self, data: pd.DataFrame) -> CustomSignal:
    # Реализация нового индикатора
    pass

def _integrate_custom_signal(self, signal: ReversalSignal, custom: CustomSignal):
    # Интеграция с основным сигналом
    pass
```

### Добавление Новых Паттернов

```python
def _detect_custom_pattern(self, data: pd.DataFrame) -> CandlestickPattern:
    # Реализация нового паттерна
    pass
```

### Кастомизация Фильтров

```python
def _custom_signal_filter(self, signal: ReversalSignal) -> bool:
    # Пользовательская логика фильтрации
    pass
```

## Заключение

Система прогнозирования разворотов представляет собой мощный инструмент для анализа финансовых рынков, объединяющий традиционные методы технического анализа с современными алгоритмами машинного обучения и кластеризации. Система обеспечивает высокую точность прогнозов при сохранении гибкости и расширяемости. 