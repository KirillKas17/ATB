import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class SignalType(Enum):
    SPREAD = "spread"
    LIQUIDITY = "liquidity"
    FAKEOUT = "fakeout"
    MM_PATTERN = "mm_pattern"  # Новый тип сигнала для паттернов ММ


class LiquidityZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


@dataclass
class MarketMakerSignal:
    """Класс для хранения сигналов маркет-мейкера"""

    timestamp: pd.Timestamp
    pair: str
    signal_type: SignalType
    confidence: float
    details: Dict[str, Any]
    priority: int = 0
