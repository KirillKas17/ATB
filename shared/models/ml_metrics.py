from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class ModelMetrics:
    win_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    last_update: datetime
    training_time: float
    inference_time: float


@dataclass(frozen=True)
class PatternMetrics:
    total_patterns: int
    pattern_lengths: Dict[int, int]
    pattern_frequencies: Dict[str, int]
    pattern_returns: Dict[str, float]
    pattern_win_rates: Dict[str, float]
    pattern_sharpe: Dict[str, float]
    pattern_drawdown: Dict[str, float]
    last_update: datetime
    confidence: float


@dataclass(frozen=True)
class TransformerMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float
    last_update: datetime
    samples_count: int
    epoch_count: int
    error_count: int
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
