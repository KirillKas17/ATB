"""
Типы данных для машинного обучения в ATB Trading System.
Обеспечивает строгую типизацию ML компонентов.
"""

import logging
from enum import Enum
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

# Настройка безопасного логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ActionType(Enum):
    """Типы торговых действий для ML алгоритмов."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # Для совместимости
    OPEN = "open"    # Для совместимости

class SignalType(Enum):
    """Типы торговых сигналов."""
    TREND = "trend"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"

class SignalSource(Enum):
    """Источники торговых сигналов."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    NEWS = "news"
    VOLUME = "volume"
    ORDERBOOK = "orderbook"

class ModelType(Enum):
    """Типы ML моделей."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    SVM = "svm"

@dataclass
class TradingSignal:
    """Торговый сигнал от ML модели."""
    action: ActionType
    confidence: float
    timestamp: datetime
    symbol: str
    price: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    signal_type: Optional[SignalType] = None
    source: Optional[SignalSource] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация данных сигнала."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        
        if self.volume is not None and self.volume <= 0:
            raise ValueError(f"Volume must be positive, got {self.volume}")

@dataclass
class AggregatedSignal:
    """Агрегированный сигнал из нескольких источников."""
    action: ActionType
    confidence: float
    timestamp: datetime
    symbol: str
    component_signals: List[TradingSignal] = field(default_factory=list)
    weights: Dict[SignalSource, float] = field(default_factory=dict)
    consensus_score: float = 0.0
    
    def __post_init__(self) -> None:
        """Валидация агрегированного сигнала."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if not 0.0 <= self.consensus_score <= 1.0:
            raise ValueError(f"Consensus score must be between 0.0 and 1.0, got {self.consensus_score}")

@dataclass
class ModelPrediction:
    """Предсказание ML модели."""
    model_type: ModelType
    prediction: Union[float, ActionType]
    confidence: float
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0"
    
    def __post_init__(self) -> None:
        """Валидация предсказания."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class FeatureVector:
    """Вектор признаков для ML модели."""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    target: Optional[float] = None
    label: Optional[ActionType] = None
    
    def __post_init__(self) -> None:
        """Валидация вектора признаков."""
        if not self.features:
            raise ValueError("Features dictionary cannot be empty")
        
        for name, value in self.features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {name} must be numeric, got {type(value)}")

@dataclass
class ModelMetrics:
    """Метрики производительности ML модели."""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    timestamp: datetime
    validation_samples: int = 0
    
    def __post_init__(self) -> None:
        """Валидация метрик."""
        metrics = [self.accuracy, self.precision, self.recall, self.f1_score, self.auc_roc]
        for i, metric in enumerate(metrics):
            if not 0.0 <= metric <= 1.0:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
                raise ValueError(f"{metric_names[i]} must be between 0.0 and 1.0, got {metric}")

@dataclass
class TrainingData:
    """Данные для обучения ML модели."""
    features: List[FeatureVector]
    labels: List[ActionType]
    timestamps: List[datetime]
    symbols: List[str]
    
    def __post_init__(self) -> None:
        """Валидация обучающих данных."""
        lengths = [len(self.features), len(self.labels), len(self.timestamps), len(self.symbols)]
        if len(set(lengths)) > 1:
            raise ValueError(f"All data arrays must have same length, got {lengths}")
        
        if not self.features:
            raise ValueError("Training data cannot be empty")

@dataclass
class PredictionResult:
    """Результат предсказания ML модели."""
    prediction: Union[float, ActionType]
    confidence: float
    timestamp: datetime
    model_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация результата предсказания."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class ModelPerformance:
    """Производительность ML модели."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime
    model_version: str = "1.0"
    
    def __post_init__(self) -> None:
        """Валидация метрик производительности."""
        metrics = [self.accuracy, self.precision, self.recall, self.f1_score]
        for i, metric in enumerate(metrics):
            if not 0.0 <= metric <= 1.0:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
                raise ValueError(f"{metric_names[i]} must be between 0.0 and 1.0, got {metric}")

@dataclass
class FeatureImportance:
    """Важность признаков в ML модели."""
    feature_name: str
    importance: float
    rank: int
    timestamp: datetime
    
    def __post_init__(self) -> None:
        """Валидация важности признаков."""
        if self.importance < 0:
            raise ValueError(f"Feature importance must be non-negative, got {self.importance}")
        if self.rank < 1:
            raise ValueError(f"Feature rank must be positive, got {self.rank}")

class PatternType(Enum):
    """Типы паттернов для распознавания."""
    TREND = "trend"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    HEAD_SHOULDERS = "head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"

@dataclass
class PatternResult:
    """Результат распознавания паттерна."""
    pattern_type: PatternType
    confidence: float
    timestamp: datetime
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация результата паттерна."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class PatternConfidence:
    """Уверенность в паттерне."""
    value: float
    level: str = "medium"
    
    def __post_init__(self) -> None:
        """Валидация уверенности."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence value must be between 0.0 and 1.0, got {self.value}")

class SignalStrength(Enum):
    """Сила торгового сигнала."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class SignalResult:
    """Результат анализа торгового сигнала."""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    symbol: str
    price: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация результата сигнала."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class SpreadAnalysisResult:
    """Результат анализа спреда."""
    symbol: str
    current_spread: float
    average_spread: float
    spread_volatility: float
    liquidity_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация результата анализа спреда."""
        if self.current_spread < 0:
            raise ValueError("Current spread must be non-negative")
        if self.average_spread < 0:
            raise ValueError("Average spread must be non-negative")

@dataclass
class SpreadMovementPrediction:
    """Предсказание движения спреда."""
    symbol: str
    predicted_direction: str  # "widening", "narrowing", "stable"
    confidence: float
    time_horizon: timedelta
    timestamp: datetime
    
    def __post_init__(self) -> None:
        """Валидация предсказания движения спреда."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.predicted_direction not in ["widening", "narrowing", "stable"]:
            raise ValueError("Predicted direction must be 'widening', 'narrowing', or 'stable'")

@dataclass
class StrategyResult:
    """Результат выполнения стратегии."""
    strategy_id: str
    action: ActionType
    confidence: float
    timestamp: datetime
    symbol: str
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Валидация результата стратегии."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class StrategyPerformance:
    """Производительность стратегии."""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    timestamp: datetime
    
    def __post_init__(self) -> None:
        """Валидация производительности стратегии."""
        if self.total_trades < 0:
            raise ValueError("Total trades must be non-negative")
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValueError("Winning + losing trades cannot exceed total trades")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError("Win rate must be between 0.0 and 1.0")

@dataclass
class StrategyConfig:
    """Конфигурация стратегии."""
    strategy_id: str
    name: str
    parameters: Dict[str, Any]
    risk_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    max_position_size: Optional[Decimal] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Валидация конфигурации стратегии."""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
        if self.stop_loss_pct is not None and (self.stop_loss_pct < 0 or self.stop_loss_pct > 1):
            raise ValueError("Stop loss percentage must be between 0 and 1")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("Take profit percentage must be positive")

# Функции преобразования для обратной совместимости
def action_to_legacy(action: ActionType) -> ActionType:
    """Преобразование нового ActionType в legacy формат."""
    mapping = {
        ActionType.BUY: ActionType.OPEN,
        ActionType.SELL: ActionType.CLOSE,
        ActionType.HOLD: ActionType.HOLD
    }
    return mapping.get(action, action)

def legacy_to_action(legacy_action: ActionType) -> ActionType:
    """Преобразование legacy ActionType в новый формат."""
    mapping = {
        ActionType.OPEN: ActionType.BUY,
        ActionType.CLOSE: ActionType.SELL,
        ActionType.HOLD: ActionType.HOLD
    }
    return mapping.get(legacy_action, legacy_action)

# Экспорт типов
__all__ = [
    'ActionType', 'SignalType', 'SignalSource', 'ModelType', 'PatternType', 'SignalStrength',
    'TradingSignal', 'AggregatedSignal', 'ModelPrediction',
    'FeatureVector', 'ModelMetrics', 'TrainingData',
    'PredictionResult', 'ModelPerformance', 'FeatureImportance',
    'PatternResult', 'PatternConfidence', 'SignalResult',
    'SpreadAnalysisResult', 'SpreadMovementPrediction',
    'StrategyResult', 'StrategyPerformance', 'StrategyConfig',
    'action_to_legacy', 'legacy_to_action'
]
