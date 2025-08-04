"""
Типы данных для машинного обучения в ATB Trading System.
Обеспечивает строгую типизацию ML компонентов.
"""

import logging
from enum import Enum
from decimal import Decimal
from datetime import datetime
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
    
    def __post_init__(self):
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
    
    def __post_init__(self):
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
    
    def __post_init__(self):
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
    
    def __post_init__(self):
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
    
    def __post_init__(self):
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
    
    def __post_init__(self):
        """Валидация обучающих данных."""
        lengths = [len(self.features), len(self.labels), len(self.timestamps), len(self.symbols)]
        if len(set(lengths)) > 1:
            raise ValueError(f"All data arrays must have same length, got {lengths}")
        
        if not self.features:
            raise ValueError("Training data cannot be empty")

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
    'ActionType', 'SignalType', 'SignalSource', 'ModelType',
    'TradingSignal', 'AggregatedSignal', 'ModelPrediction',
    'FeatureVector', 'ModelMetrics', 'TrainingData',
    'action_to_legacy', 'legacy_to_action'
]
