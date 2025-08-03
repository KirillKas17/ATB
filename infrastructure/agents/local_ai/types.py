"""
Типы данных для локального AI контроллера.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AITaskType(Enum):
    """Типы задач AI."""

    PREDICTION = "prediction"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AIStatus(Enum):
    """Статусы AI контроллера."""

    IDLE = "idle"
    PROCESSING = "processing"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AITask:
    """Задача для AI контроллера."""

    task_id: str
    task_type: AITaskType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float = 30.0  # секунды


@dataclass
class AIResult:
    """Результат выполнения AI задачи."""

    task_id: str
    status: AIStatus
    data: Dict[str, Any]
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AIConfig:
    """Конфигурация AI контроллера."""

    model_path: str = ""
    max_concurrent_tasks: int = 5
    default_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # секунды
    log_predictions: bool = True
    enable_auto_retrain: bool = False
    retrain_interval: int = 86400  # 24 часа
