"""
Доменная модель сделки.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4

from domain.type_definitions import MetadataDict, SignalTypeType, Symbol, TimestampValue, TradeId

from ..value_objects.currency import Currency
from ..value_objects.money import Money
from ..value_objects.price import Price
from ..value_objects.timestamp import Timestamp
from ..value_objects.volume import Volume


@runtime_checkable
class TradeProtocol(Protocol):
    """Протокол для сделок."""

    id: TradeId
    symbol: Symbol
    side: SignalTypeType
    price: Price
    volume: Volume
    executed_at: TimestampValue
    fee: Money
    realized_pnl: Optional[Money]
    metadata: MetadataDict

    @property
    def notional_value(self) -> Money: ...
    @property
    def is_buy(self) -> bool: ...
    @property
    def is_sell(self) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeProtocol": ...


@dataclass
class Trade:
    """Сделка."""

    id: TradeId = field(default_factory=lambda: TradeId(uuid4()))
    symbol: Symbol = field(default=Symbol(""))
    side: SignalTypeType = field(default="buy")  # "buy" или "sell"
    price: Price = field(default_factory=lambda: Price(Decimal("0"), Currency.USDT))
    volume: Volume = field(default_factory=lambda: Volume(Decimal("0"), Currency.USDT))
    executed_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(Timestamp.now().value)
    )
    fee: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    realized_pnl: Optional[Money] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.id:
            raise ValueError("Trade ID cannot be empty")

        if not self.symbol:
            raise ValueError("Trade symbol cannot be empty")

        if self.side not in ["buy", "sell"]:
            raise ValueError("Trade side must be 'buy' or 'sell'")

        if self.price.value == 0:
            raise ValueError("Trade price cannot be zero")

        if self.volume.value == 0:
            raise ValueError("Trade volume cannot be zero")

    @property
    def notional_value(self) -> Money:
        """Номинальная стоимость сделки."""
        # Для криптовалют используем USDT как базовую валюту
        value = self.price.value * self.volume.to_decimal()
        return Money(value, Currency.USDT)

    @property
    def is_buy(self) -> bool:
        """Проверка, что это покупка."""
        return self.side == "buy"

    @property
    def is_sell(self) -> bool:
        """Проверка, что это продажа."""
        return self.side == "sell"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "symbol": str(self.symbol),
            "side": self.side,
            "price": self.price.to_dict(),
            "volume": self.volume.to_dict(),
            "executed_at": self.executed_at.isoformat(),
            "fee": self.fee.to_dict(),
            "realized_pnl": self.realized_pnl.to_dict() if self.realized_pnl else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Создание из словаря."""
        return cls(
            id=TradeId(uuid4() if data["id"] == "" else data["id"]),
            symbol=Symbol(data["symbol"]),
            side=data["side"],
            price=Price.from_dict(data["price"]),
            volume=Volume.from_dict(data["volume"]),
            executed_at=TimestampValue(Timestamp.from_dict(data["executed_at"]).value),
            fee=Money.from_dict(data["fee"]),
            realized_pnl=(
                Money.from_dict(data["realized_pnl"])
                if data.get("realized_pnl")
                else None
            ),
            metadata=data.get("metadata", {}),
        )
