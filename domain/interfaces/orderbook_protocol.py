"""
Протокол для OrderBook данных.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class OrderBookLevel(Protocol):
    """Протокол для уровня стакана."""

    price: Decimal
    quantity: Decimal
    side: str
    timestamp: datetime


@runtime_checkable
class OrderbookProtocol(Protocol):
    """Протокол для OrderBook данных."""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_update_id: int
    event_time: datetime

    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Получение лучшей цены покупки."""
        ...

    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Получение лучшей цены продажи."""
        ...

    def get_spread(self) -> Decimal:
        """Получение спреда."""
        ...

    def get_mid_price(self) -> Decimal:
        """Получение средней цены."""
        ...

    def get_total_bid_volume(self) -> Decimal:
        """Получение общего объема покупок."""
        ...

    def get_total_ask_volume(self) -> Decimal:
        """Получение общего объема продаж."""
        ...

    def get_imbalance(self) -> Decimal:
        """Получение дисбаланса."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        ...


@runtime_checkable
class OrderBookUpdate(Protocol):
    """Протокол для обновления стакана."""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_update_id: int
    event_time: datetime

    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Получение лучшей цены покупки."""
        ...

    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Получение лучшей цены продажи."""
        ...

    def get_spread(self) -> Decimal:
        """Получение спреда."""
        ...

    def get_mid_price(self) -> Decimal:
        """Получение средней цены."""
        ...

    def get_total_bid_volume(self) -> Decimal:
        """Получение общего объема покупок."""
        ...

    def get_total_ask_volume(self) -> Decimal:
        """Получение общего объема продаж."""
        ...

    def get_imbalance(self) -> Decimal:
        """Получение дисбаланса."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        ...
