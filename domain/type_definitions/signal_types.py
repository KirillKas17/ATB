from enum import Enum
from typing import Any, Dict, List, Protocol, TypedDict, runtime_checkable


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

    @classmethod
    def from_string(cls, value: str) -> "SignalType":
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower or member.name == value.upper():
                return member
        raise ValueError(f"Unknown SignalType: {value}")


class SignalStrength(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class SignalValidationResult(TypedDict, total=False):
    errors: List[str]
    is_valid: bool
    warnings: List[str]


class SignalAggregationResult(TypedDict, total=False):
    aggregated_signal: Any
    confidence: float
    details: Dict[str, Any]
    strength: Any
    metadata: Dict[str, Any]


class SignalAnalysisResult(TypedDict, total=False):
    stats: Dict[str, Any]
    recommendations: List[str]
    distribution: Dict[str, int]
    avg_confidence: float
    total_signals: int
    recent_signals: int
    signal_distribution: Dict[str, int]
    strength_distribution: Dict[str, int]
    period_analysis: Dict[str, Any]


@runtime_checkable
class SignalServiceProtocol(Protocol):
    async def generate_signals(self, strategy: Any, market_data: Any) -> List[Any]: ...
    async def validate_signal(self, signal: Any) -> SignalValidationResult: ...
    async def aggregate_signals(
        self, signals: List[Any]
    ) -> SignalAggregationResult: ...
    async def analyze_signals(
        self, signals: List[Any], period: Any
    ) -> SignalAnalysisResult: ...
