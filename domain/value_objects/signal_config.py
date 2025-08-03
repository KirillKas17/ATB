from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Set


class SignalStrength(Enum):
    """Сила сигнала."""

    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9
    EXTREME = 1.0


class SignalDirection(Enum):
    """Направление сигнала."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class SignalType(Enum):
    """Тип сигнала."""

    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    PATTERN = "PATTERN"
    MOMENTUM = "MOMENTUM"
    REVERSAL = "REVERSAL"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"


@dataclass(frozen=True)
class SignalConfig:
    """Конфигурация для Signal value object."""

    min_confidence: Decimal = Decimal("0.1")
    max_confidence: Decimal = Decimal("1.0")
    min_strength: Decimal = Decimal("0.1")
    max_strength: Decimal = Decimal("1.0")
    default_expiry_hours: int = 24
    max_expiry_hours: int = 168  # 7 days
    min_expiry_hours: int = 1
    strength_thresholds: Optional[Dict[str, Decimal]] = None
    confidence_thresholds: Optional[Dict[str, Decimal]] = None

    def __post_init__(self) -> None:
        if self.strength_thresholds is None:
            object.__setattr__(
                self,
                "strength_thresholds",
                {
                    "VERY_WEAK": Decimal("0.1"),
                    "WEAK": Decimal("0.3"),
                    "MODERATE": Decimal("0.5"),
                    "STRONG": Decimal("0.7"),
                    "VERY_STRONG": Decimal("0.9"),
                    "EXTREME": Decimal("1.0"),
                },
            )
        if self.confidence_thresholds is None:
            object.__setattr__(
                self,
                "confidence_thresholds",
                {
                    "LOW": Decimal("0.3"),
                    "MEDIUM": Decimal("0.6"),
                    "HIGH": Decimal("0.8"),
                    "VERY_HIGH": Decimal("0.95"),
                },
            )
