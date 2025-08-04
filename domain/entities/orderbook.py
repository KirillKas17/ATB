"""
Доменная сущность OrderBookSnapshot.
Представляет снимок ордербука с метаданными.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from domain.types import MetadataDict
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


@dataclass
class OrderBookSnapshot:
    """
    Снимок ордербука.
    Представляет состояние ордербука в определенный момент времени
    с полной информацией о ценах и объемах.
    """

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def __post_init__(self) -> None:
        """Инициализация после создания объекта."""
        # Сортируем bids по убыванию цены (лучшие цены сверху)
        self.bids.sort(key=lambda x: x[0].value, reverse=True)
        # Сортируем asks по возрастанию цены (лучшие цены сверху)
        self.asks.sort(key=lambda x: x[0].value)

    @property
    def best_bid(self) -> Optional[Tuple[Price, Volume]]:
        """Лучшая цена покупки."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Tuple[Price, Volume]]:
        """Лучшая цена продажи."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Price]:
        """Средняя цена между лучшими bid и ask."""
        if self.best_bid and self.best_ask:
            mid_amount = (self.best_bid[0].value + self.best_ask[0].value) / 2
            return Price(mid_amount, self.best_bid[0].currency)
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Спред между лучшими bid и ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask[0].value - self.best_bid[0].value
        return None

    @property
    def spread_percentage(self) -> Optional[Decimal]:
        """Спред в процентах от средней цены."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price.value) * 100
        return None

    @property
    def total_bid_volume(self) -> Volume:
        """Общий объем покупок."""
        total = sum(Decimal(str(bid[1])) for bid in self.bids)
        return Volume(
            total, Currency.USD
        )

    @property
    def total_ask_volume(self) -> Volume:
        """Общий объем продаж."""
        total = sum(Decimal(str(ask[1])) for ask in self.asks)
        return Volume(
            total, Currency.USD
        )

    @property
    def volume_imbalance(self) -> Decimal:
        """Дисбаланс объемов (bid_volume - ask_volume)."""
        bid_vol = self.total_bid_volume.value
        ask_vol = self.total_ask_volume.value
        return bid_vol - ask_vol

    @property
    def volume_imbalance_ratio(self) -> Optional[Decimal]:
        """Отношение дисбаланса объемов к общему объему."""
        total_volume = self.total_bid_volume.value + self.total_ask_volume.value
        if total_volume > 0:
            return self.volume_imbalance / total_volume
        return None

    def get_bid_volume_at_price(self, price: Price) -> Volume:
        """Получение объема покупок по цене."""
        for bid_price, bid_volume in self.bids:
            if bid_price.value >= price.value:
                return bid_volume
        return Volume(Decimal("0"), price.currency)

    def get_ask_volume_at_price(self, price: Price) -> Volume:
        """Получение объема продаж по цене."""
        for ask_price, ask_volume in self.asks:
            if ask_price.value <= price.value:
                return ask_volume
        return Volume(Decimal("0"), price.currency)

    def get_depth_at_price(self, price: Price) -> Tuple[Volume, Volume]:
        """Получение глубины рынка по цене (bid_volume, ask_volume)."""
        bid_vol = self.get_bid_volume_at_price(price)
        ask_vol = self.get_ask_volume_at_price(price)
        return bid_vol, ask_vol

    def is_filtered(self) -> bool:
        """Проверка, был ли ордербук отфильтрован."""
        filtered = self.meta.get("filtered", False)
        return bool(filtered)

    def get_noise_analysis(self) -> Optional[Dict[str, Any]]:
        """Получение результатов анализа шума."""
        return self.meta.get("noise_analysis")

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.
        Returns:
            Dict[str, Any]: Словарь с данными ордербука
        """
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "bids": [(str(bid[0].value), str(bid[1].value)) for bid in self.bids],
            "asks": [(str(ask[0].value), str(ask[1].value)) for ask in self.asks],
            "timestamp": self.timestamp.to_dict(),
            "sequence_id": self.sequence_id,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBookSnapshot":
        """
        Создание из словаря.
        Args:
            data: Словарь с данными ордербука
        Returns:
            OrderBookSnapshot: Объект снимка ордербука
        """
        # Преобразуем bids
        bids = [
            (
                Price(Decimal(price), Currency.USD),
                Volume(Decimal(volume), Currency.USD),
            )
            for price, volume in data["bids"]
        ]
        # Преобразуем asks
        asks = [
            (
                Price(Decimal(price), Currency.USD),
                Volume(Decimal(volume), Currency.USD),
            )
            for price, volume in data["asks"]
        ]
        return cls(
            exchange=data["exchange"],
            symbol=data["symbol"],
            bids=bids,
            asks=asks,
            timestamp=Timestamp.from_dict(data["timestamp"]),
            sequence_id=data.get("sequence_id"),
            meta=MetadataDict(data.get("meta", {})),
        )

    def __str__(self) -> str:
        """Строковое представление ордербука."""
        return f"OrderBookSnapshot({self.exchange}:{self.symbol}, {len(self.bids)} bids, {len(self.asks)} asks)"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return (
            f"OrderBookSnapshot(exchange='{self.exchange}', symbol='{self.symbol}', "
            f"bids={len(self.bids)}, asks={len(self.asks)}, "
            f"timestamp={self.timestamp.value})"
        )
