"""
Типы данных для интеграции памяти китов.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WhaleActivityType(Enum):
    """Типы активности китов."""

    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MANIPULATION = "manipulation"
    PUMP = "pump"
    DUMP = "dump"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"


class WhaleSize(Enum):
    """Размеры китов."""

    SMALL = "small"  # < 100 BTC
    MEDIUM = "medium"  # 100-1000 BTC
    LARGE = "large"  # 1000-10000 BTC
    WHALE = "whale"  # > 10000 BTC


@dataclass
class WhaleActivity:
    """Активность кита."""

    activity_id: str
    whale_address: str
    symbol: str
    activity_type: WhaleActivityType
    size: WhaleSize
    amount: float
    price: float
    timestamp: datetime
    confidence: float = 0.0
    impact_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhalePattern:
    """Паттерн активности китов."""

    pattern_id: str
    symbol: str
    pattern_type: str
    whale_addresses: List[str]
    total_volume: float
    start_time: datetime
    end_time: datetime
    confidence: float = 0.0
    prediction: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhaleMemory:
    """Запись памяти о китах."""

    memory_id: str
    whale_address: str
    symbol: str
    activities: List[WhaleActivity]
    patterns: List[WhalePattern]
    last_activity: datetime
    total_volume: float = 0.0
    behavior_score: float = 0.0
    risk_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhaleQuery:
    """Запрос к памяти китов."""

    symbol: str
    whale_address: Optional[str] = None
    activity_type: Optional[WhaleActivityType] = None
    size: Optional[WhaleSize] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_confidence: float = 0.0
    limit: int = 100


@dataclass
class WhaleMemoryConfig:
    """Конфигурация памяти китов."""

    max_whales: int = 1000
    max_activities_per_whale: int = 1000
    cleanup_interval: int = 3600  # секунды
    memory_ttl: int = 86400 * 30  # 30 дней
    enable_tracking: bool = True
    enable_prediction: bool = True
    log_whale_activities: bool = True
