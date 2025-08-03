from typing import Any, Dict
from .signals import MarketMakerSignal, SignalType, LiquidityZoneType

# Заглушки для отсутствующих классов
class FakeoutDetection:
    """Заглушка для FakeoutDetection"""
    pass

class LiquidityAnalysis:
    """Заглушка для LiquidityAnalysis"""
    pass

class SpreadAnalysis:
    """Заглушка для SpreadAnalysis"""
    pass

class MarketMakerConfig:
    """
    Конфиг для эволюционного маркет-мейкера.
    """

    def __init__(
        self, spread: float = 0.01, min_volume: float = 1000.0, max_orders: int = 10
    ) -> None:
        self.spread = spread
        self.min_volume = min_volume
        self.max_orders = max_orders
        self.extra: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread": self.spread,
            "min_volume": self.min_volume,
            "max_orders": self.max_orders,
            **self.extra,
        }


__all__ = [
    "MarketMakerConfig", 
    "MarketMakerSignal",
    "SignalType",
    "LiquidityZoneType",
    "FakeoutDetection",
    "LiquidityAnalysis",
    "SpreadAnalysis"
]
