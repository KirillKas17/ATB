"""
Типы для risk agent.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskMetrics:
    value: float = 0.0
    details: "Optional[Dict[str, Any]]" = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


@dataclass
class RiskConfig:
    threshold: float = 0.1


@dataclass
class RiskLimits:
    max_loss: float = 0.2
