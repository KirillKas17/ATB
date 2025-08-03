"""
Типы данных для интеграции запутанности.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class EntanglementType(Enum):
    """Типы запутанности."""

    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    CORRELATION = "correlation"


class EntanglementLevel(Enum):
    """Уровни запутанности."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EntanglementEvent:
    """Событие запутанности."""

    event_id: str
    entanglement_type: EntanglementType
    symbols: List[str]
    correlation_score: float
    level: EntanglementLevel
    timestamp: datetime
    duration: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntanglementAnalysis:
    """Анализ запутанности."""

    analysis_id: str
    symbols: List[str]
    entanglement_type: EntanglementType
    correlation_matrix: Dict[str, Dict[str, float]]
    overall_correlation: float
    level: EntanglementLevel
    confidence: float
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EntanglementConfig:
    """Конфигурация интеграции запутанности."""

    enable_entanglement_detection: bool = True
    correlation_threshold: float = 0.8
    detection_interval: float = 1.0
    max_lag_ms: float = 3.0
    enable_logging: bool = True
    alert_threshold: float = 0.95
