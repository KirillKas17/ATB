"""
Типы данных для интеграции рыночной памяти.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Типы памяти."""

    PATTERN = "pattern"
    EVENT = "event"
    DECISION = "decision"
    OUTCOME = "outcome"


class MemoryPriority(Enum):
    """Приоритеты памяти."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MarketMemory:
    """Запись рыночной памяти."""

    memory_id: str
    memory_type: MemoryType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: MemoryPriority = MemoryPriority.MEDIUM
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQuery:
    """Запрос к памяти."""

    symbol: str
    memory_type: Optional[MemoryType] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    limit: int = 100
    min_confidence: float = 0.0


@dataclass
class MemoryResult:
    """Результат запроса к памяти."""

    memories: List[MarketMemory]
    total_count: int
    query_time: float
    relevance_score: float = 0.0


@dataclass
class MarketMemoryConfig:
    """Конфигурация рыночной памяти."""

    max_memories: int = 10000
    cleanup_interval: int = 3600  # секунды
    memory_ttl: int = 86400 * 7  # 7 дней
    enable_compression: bool = True
    enable_indexing: bool = True
    log_memory_operations: bool = False
