from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .market_protocols import MarketStateProtocol
from .market_types import MarketMetadataDict, MarketRegime


@dataclass
class MarketState(MarketStateProtocol):
    id: str = ""
    symbol: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_trend: float = 0.0
    price_momentum: float = 0.0
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    pivot_point: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None
    atr: Optional[float] = None
    metadata: MarketMetadataDict = field(default_factory=lambda: {"source": "", "exchange": "", "extra": {}})

    def is_trending(self) -> bool:
        return self.regime in {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN}

    def is_sideways(self) -> bool:
        return self.regime == MarketRegime.SIDEWAYS

    def is_volatile(self) -> bool:
        return self.regime == MarketRegime.VOLATILE

    def is_breakout(self) -> bool:
        return self.regime == MarketRegime.BREAKOUT

    def get_trend_direction(self) -> Optional[str]:
        if self.regime == MarketRegime.TRENDING_UP:
            return "up"
        elif self.regime == MarketRegime.TRENDING_DOWN:
            return "down"
        return None

    def get_price_position(self, current_price: float) -> Optional[str]:
        if self.support_level and current_price < self.support_level:
            return "below_support"
        if self.resistance_level and current_price > self.resistance_level:
            return "above_resistance"
        return "inside_range"

    def is_overbought(self) -> bool:
        return self.rsi is not None and self.rsi > 70

    def is_oversold(self) -> bool:
        return self.rsi is not None and self.rsi < 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "volume_trend": self.volume_trend,
            "price_momentum": self.price_momentum,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "pivot_point": self.pivot_point,
            "rsi": self.rsi,
            "macd": self.macd,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_middle": self.bollinger_middle,
            "atr": self.atr,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        return cls(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            regime=MarketRegime(data["regime"]),
            volatility=float(data["volatility"]),
            trend_strength=float(data["trend_strength"]),
            volume_trend=float(data["volume_trend"]),
            price_momentum=float(data["price_momentum"]),
            support_level=(
                float(data["support_level"])
                if data.get("support_level") is not None
                else None
            ),
            resistance_level=(
                float(data["resistance_level"])
                if data.get("resistance_level") is not None
                else None
            ),
            pivot_point=(
                float(data["pivot_point"])
                if data.get("pivot_point") is not None
                else None
            ),
            rsi=float(data["rsi"]) if data.get("rsi") is not None else None,
            macd=float(data["macd"]) if data.get("macd") is not None else None,
            bollinger_upper=(
                float(data["bollinger_upper"])
                if data.get("bollinger_upper") is not None
                else None
            ),
            bollinger_lower=(
                float(data["bollinger_lower"])
                if data.get("bollinger_lower") is not None
                else None
            ),
            bollinger_middle=(
                float(data["bollinger_middle"])
                if data.get("bollinger_middle") is not None
                else None
            ),
            atr=float(data["atr"]) if data.get("atr") is not None else None,
            metadata=data.get("metadata", {"source": "", "exchange": "", "extra": {}}),
        )
