from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


@dataclass(frozen=True)
class OrderBookUpdate:
    """
    Унифицированное обновление ордербука для инфраструктуры и домена.
    bids/asks — список пар (Price, Volume).
    """

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def get_mid_price(self) -> Optional[Price]:
        if self.bids and self.asks:
            mid = (self.bids[0][0].value + self.asks[0][0].value) / 2
            return Price(amount=mid, currency=self.bids[0][0].currency)
        return None

    def get_spread(self) -> Optional[Decimal]:
        if self.bids and self.asks:
            return self.asks[0][0].value - self.bids[0][0].value
        return None


@dataclass(frozen=True)
class OrderBookSnapshot:
    """
    Снимок ордербука (глубокое состояние стакана).
    """

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
