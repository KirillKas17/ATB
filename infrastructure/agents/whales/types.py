import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class WhaleActivityType(Enum):
    ORDER_BOOK = "order_book"
    VOLUME = "volume"
    IMPULSE = "impulse"
    DOMINANCE = "dominance"


@dataclass
class WhaleActivity:
    """Активность кита."""

    timestamp: pd.Timestamp
    volume: float
    price: float
    direction: str  # 'buy' или 'sell'
    confidence: float
    impact_score: float
    details: Optional[Dict[str, Any]] = None
    impact: float = 0.0
    pair: str = ""
    activity_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}
        if self.impact == 0.0:
            self.impact = self.impact_score


@dataclass
class WhaleAnalysis:
    """Анализ поведения китов."""

    whale_activities: List[WhaleActivity]
    total_volume: float
    net_direction: str
    impact_score: float
    confidence: float
    whale_count: int
    large_transactions: int
