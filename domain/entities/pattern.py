"""
Доменные сущности для работы с паттернами торговли.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict
from uuid import UUID, uuid4

from domain.type_definitions.pattern_types import PatternType as BasePatternType


class PatternConfidence(Enum):
    """Уровни уверенности в паттерне."""
    
    UNKNOWN = 0.0
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


@dataclass
class Pattern:
    """Сущность паттерна торговли."""
    
    id: UUID = field(default_factory=uuid4)
    pattern_type: BasePatternType = BasePatternType.UNKNOWN
    characteristics: Dict[str, Any] = field(default_factory=dict)
    confidence: PatternConfidence = PatternConfidence.UNKNOWN
    trading_pair_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict) 