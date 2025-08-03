"""
Доменная сущность Position с промышленной типизацией и бизнес-валидацией.
"""

import ast
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import UUID

from domain.types import AmountValue, PortfolioId, PositionId, PriceValue, Symbol
from domain.types import TradingPair as TradingPairType
from domain.types import VolumeValue
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.exceptions import BusinessRuleError

from .trading_pair import TradingPair


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@runtime_checkable
class PositionProtocol(Protocol):
    def get_side(self) -> str: ...
    def get_volume(self) -> VolumeValue: ...
    def get_pnl(self) -> AmountValue: ...


@dataclass
class Position:
    id: PositionId
    portfolio_id: PortfolioId
    trading_pair: TradingPair
    side: PositionSide
    volume: Volume
    entry_price: Price
    current_price: Price
    unrealized_pnl: Optional[Money] = None
    realized_pnl: Optional[Money] = None
    margin_used: Optional[Money] = None
    leverage: Decimal = Decimal("1")
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)
    closed_at: Optional[Timestamp] = None
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Position ID cannot be empty")
        if not self.portfolio_id:
            raise ValueError("Portfolio ID cannot be empty")
        if not self.trading_pair:
            raise ValueError("Trading pair cannot be empty")
        if self.volume.value == 0:
            raise ValueError("Position volume cannot be zero")
        if not self.trading_pair.validate_volume(self.volume):
            raise ValueError("Invalid volume for this trading pair")
        if not self.trading_pair.validate_price(self.entry_price):
            raise ValueError("Invalid entry price for this trading pair")
        if self.stop_loss and not self.trading_pair.validate_price(self.stop_loss):
            raise ValueError("Invalid stop loss price for this trading pair")
        if self.take_profit and not self.trading_pair.validate_price(self.take_profit):
            raise ValueError("Invalid take profit price for this trading pair")
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")

    def get_side(self) -> str:
        return self.side.value

    def get_volume(self) -> VolumeValue:
        return VolumeValue(self.volume.to_decimal())

    def get_pnl(self) -> AmountValue:
        return AmountValue(self.total_pnl.amount)

    @property
    def size(self) -> Volume:
        return self.volume

    @property
    def avg_entry_price(self) -> Price:
        return self.entry_price

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    @property
    def notional_value(self) -> Money:
        value = self.current_price.amount * self.volume.to_decimal()
        return Money(value, Currency.USDT)

    @property
    def current_notional_value(self) -> Optional[Money]:
        if not self.current_price:
            return None
        value = self.current_price.amount * self.volume.to_decimal()
        return Money(value, Currency.USDT)

    def update_price(self, new_price: Price) -> None:
        if not self.trading_pair.validate_price(new_price):
            raise ValueError("Invalid price for this trading pair")
        self.current_price = new_price
        self._calculate_unrealized_pnl()
        self.updated_at = Timestamp.now()

    def _calculate_unrealized_pnl(self) -> None:
        if self.volume.value == 0:
            self.unrealized_pnl = Money.zero(Currency.USDT)
            return
        if self.is_long:
            pnl_amount = (
                self.current_price.amount - self.entry_price.amount
            ) * self.volume.to_decimal()
        else:
            pnl_amount = (
                self.entry_price.amount - self.current_price.amount
            ) * self.volume.to_decimal()
        self.unrealized_pnl = Money(pnl_amount, Currency.USDT)

    @property
    def total_pnl(self) -> Money:
        total = Decimal("0")
        if self.realized_pnl:
            total += self.realized_pnl.amount
        if self.unrealized_pnl:
            total += self.unrealized_pnl.amount
        return Money(total, Currency.USDT)

    def add_realized_pnl(self, pnl: Money) -> None:
        if self.realized_pnl and pnl.currency != self.realized_pnl.currency:
            raise ValueError("P&L currency must match position currency")
        if self.realized_pnl:
            self.realized_pnl = self.realized_pnl + pnl
        else:
            self.realized_pnl = pnl
        self.updated_at = Timestamp.now()

    def close(self, close_price: Price, close_volume: Optional[Volume] = None) -> Money:
        if not self.is_open:
            raise BusinessRuleError("Cannot close already closed position")
        if not self.trading_pair.validate_price(close_price):
            raise ValueError("Invalid close price for this trading pair")
        close_vol = close_volume or self.volume
        if not self.trading_pair.validate_volume(close_vol):
            raise ValueError("Invalid close volume for this trading pair")
        if close_vol.to_decimal() > self.volume.to_decimal():
            raise BusinessRuleError("Close volume cannot exceed position volume")
        if self.is_long:
            realized_pnl_amount = (
                close_price.amount - self.entry_price.amount
            ) * close_vol.to_decimal()
        else:
            realized_pnl_amount = (
                self.entry_price.amount - close_price.amount
            ) * close_vol.to_decimal()
        realized_pnl = Money(realized_pnl_amount, Currency.USDT)
        self.add_realized_pnl(realized_pnl)
        
        # Обновляем объем позиции при частичном закрытии
        if close_vol.to_decimal() < self.volume.to_decimal():
            remaining_volume = self.volume.to_decimal() - close_vol.to_decimal()
            self.volume = Volume(remaining_volume, self.volume.currency)
        else:
            # Полное закрытие
            self.closed_at = Timestamp.now()
            
        return realized_pnl

    def set_stop_loss(self, stop_loss: Price) -> None:
        if not self.trading_pair.validate_price(stop_loss):
            raise ValueError("Invalid stop loss price for this trading pair")
        self.stop_loss = stop_loss
        self.updated_at = Timestamp.now()

    def set_take_profit(self, take_profit: Price) -> None:
        if not self.trading_pair.validate_price(take_profit):
            raise ValueError("Invalid take profit price for this trading pair")
        self.take_profit = take_profit
        self.updated_at = Timestamp.now()

    def is_stop_loss_hit(self) -> bool:
        if not self.stop_loss or not self.current_price:
            return False
        if self.is_long:
            return self.current_price.amount <= self.stop_loss.amount
        else:
            return self.current_price.amount >= self.stop_loss.amount

    def is_take_profit_hit(self) -> bool:
        if not self.take_profit or not self.current_price:
            return False
        if self.is_long:
            return self.current_price.amount >= self.take_profit.amount
        else:
            return self.current_price.amount <= self.take_profit.amount

    def get_risk_reward_ratio(self) -> Optional[float]:
        if not self.stop_loss or not self.take_profit:
            return None
        if self.is_long:
            risk = self.entry_price.amount - self.stop_loss.amount
            reward = self.take_profit.amount - self.entry_price.amount
        else:
            risk = self.stop_loss.amount - self.entry_price.amount
            reward = self.entry_price.amount - self.take_profit.amount
        if risk <= 0:
            return None
        return float(reward / risk)

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": str(self.id),
            "portfolio_id": str(self.portfolio_id),
            "trading_pair": str(self.trading_pair.symbol),
            "side": self.side.value,
            "volume": str(self.volume.to_decimal()),
            "entry_price": str(self.entry_price.amount),
            "current_price": str(self.current_price.amount),
            "unrealized_pnl": (
                str(self.unrealized_pnl.amount) if self.unrealized_pnl else ""
            ),
            "realized_pnl": str(self.realized_pnl.amount) if self.realized_pnl else "",
            "total_pnl": str(self.total_pnl.amount),
            "notional_value": str(self.notional_value.amount),
            "current_notional_value": (
                str(self.current_notional_value.amount)
                if self.current_notional_value
                else ""
            ),
            "is_open": str(self.is_open),
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "closed_at": str(self.closed_at) if self.closed_at else "",
            "stop_loss": str(self.stop_loss.amount) if self.stop_loss else "",
            "take_profit": (
                str(self.take_profit.amount) if self.take_profit else ""
            ),
            "risk_reward_ratio": (
                str(self.get_risk_reward_ratio())
                if self.get_risk_reward_ratio()
                else ""
            ),
            "margin_used": str(self.margin_used.amount) if self.margin_used else "",
            "leverage": str(self.leverage),
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Position":
        return cls(
            id=PositionId(UUID(data["id"])),
            portfolio_id=PortfolioId(UUID(data["portfolio_id"])),
            trading_pair=TradingPair.from_dict({"symbol": data["trading_pair"]}),
            side=PositionSide(data["side"]),
            volume=Volume(Decimal(data["volume"]), Currency.USDT),
            entry_price=Price(Decimal(data["entry_price"]), Currency.USDT),
            current_price=Price(Decimal(data["current_price"]), Currency.USDT),
            unrealized_pnl=(
                Money(Decimal(data["unrealized_pnl"]), Currency.USDT)
                if data.get("unrealized_pnl")
                else None
            ),
            realized_pnl=(
                Money(Decimal(data["realized_pnl"]), Currency.USDT)
                if data.get("realized_pnl")
                else None
            ),
            margin_used=(
                Money(Decimal(data["margin_used"]), Currency.USDT)
                if data.get("margin_used")
                else None
            ),
            leverage=Decimal(data["leverage"]),
            created_at=Timestamp.from_iso(data["created_at"]),
            updated_at=Timestamp.from_iso(data["updated_at"]),
            closed_at=(
                Timestamp.from_iso(data["closed_at"]) if data.get("closed_at") else None
            ),
            stop_loss=(
                Price(Decimal(data["stop_loss"]), Currency.USDT)
                if data.get("stop_loss")
                else None
            ),
            take_profit=(
                Price(Decimal(data["take_profit"]), Currency.USDT)
                if data.get("take_profit")
                else None
            ),
            metadata=ast.literal_eval(data.get("metadata", "{}")),
        )

    def __str__(self) -> str:
        return f"{self.side.value.upper()} {self.volume} {self.trading_pair.symbol} @ {self.entry_price}"

    def __repr__(self) -> str:
        return f"Position(id='{self.id}', {self.side.value} {self.volume} {self.trading_pair.symbol})"

    def __eq__(self, other: Any) -> bool:
        """Сравнение позиций по бизнес-логике."""
        if not isinstance(other, Position):
            return False
        return (
            self.id == other.id
            and self.portfolio_id == other.portfolio_id
            and self.trading_pair == other.trading_pair
            and self.side == other.side
            and self.volume == other.volume
            and self.entry_price == other.entry_price
            and self.current_price == other.current_price
            and self.leverage == other.leverage
            and self.created_at == other.created_at
            and self.updated_at == other.updated_at
        )

    def __hash__(self) -> int:
        """Хеш-код позиции."""
        return hash((
            self.id,
            self.portfolio_id,
            self.trading_pair,
            self.side,
            self.volume,
            self.entry_price,
            self.current_price,
            self.leverage,
            self.created_at,
            self.updated_at,
        ))
