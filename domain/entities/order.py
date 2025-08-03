"""
Доменная сущность ордера с промышленной типизацией и бизнес-валидацией.
"""

import ast
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from domain.types import (
    OrderId,
    OrderStatusType,
    PortfolioId,
    PriceValue,
    SignalId,
    StrategyId,
    Symbol,
    TradingPair,
    VolumeValue,
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.exceptions import OrderError
from domain.value_objects.volume import Volume


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    BRACKET = "bracket"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@runtime_checkable
class OrderProtocol(Protocol):
    def get_status(self) -> OrderStatusType: ...
    def get_quantity(self) -> VolumeValue: ...
    def get_price(self) -> PriceValue: ...
@dataclass
class Order:
    id: OrderId = field(default_factory=lambda: OrderId(uuid4()))
    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    signal_id: Optional[SignalId] = None
    exchange_order_id: Optional[str] = None
    symbol: Symbol = Symbol("")
    trading_pair: TradingPair = TradingPair("")
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    amount: Volume = field(default_factory=lambda: Volume(Decimal("0"), Currency.USD))
    quantity: VolumeValue = VolumeValue(Decimal("0"))
    price: Optional[Price] = None
    stop_price: Optional[Price] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: Volume = field(
        default_factory=lambda: Volume(Decimal("0"), Currency.USD)
    )
    filled_quantity: VolumeValue = VolumeValue(Decimal("0"))
    average_price: Optional[Price] = None
    commission: Optional[Price] = None
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)
    filled_at: Optional[Timestamp] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.trading_pair:
            raise ValueError("Trading pair cannot be empty")

    def get_status(self) -> OrderStatusType:
        return self.status.value

    def get_quantity(self) -> VolumeValue:
        return self.quantity

    def get_price(self) -> PriceValue:
        return PriceValue(self.price.amount if self.price else Decimal("0"))

    @property
    def is_open(self) -> bool:
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        return self.status in [
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def fill_percentage(self) -> Decimal:
        if Decimal(str(self.quantity)) == 0:
            return Decimal("0")
        return (Decimal(str(self.filled_quantity)) / Decimal(str(self.quantity))) * 100

    @property
    def remaining_quantity(self) -> VolumeValue:
        return VolumeValue(Decimal(str(self.quantity)) - Decimal(str(self.filled_quantity)))

    @property
    def total_value(self) -> Optional[Price]:
        if self.price and Decimal(str(self.quantity)):
            return Price(self.price.amount * Decimal(str(self.quantity)), self.price.currency)
        return None

    @property
    def filled_value(self) -> Optional[Price]:
        if self.average_price and Decimal(str(self.filled_quantity)):
            return Price(
                self.average_price.amount * Decimal(str(self.filled_quantity)),
                self.average_price.currency,
            )
        return None

    def update_status(self, status: OrderStatus) -> None:
        self.status = status
        self.updated_at = Timestamp.now()
        if status == OrderStatus.FILLED:
            self.filled_at = Timestamp.now()

    def update_fill(
        self, filled_qty: VolumeValue, price: Price, commission: Optional[Price] = None
    ) -> None:
        self.filled_quantity = VolumeValue(Decimal(str(self.filled_quantity)) + Decimal(str(filled_qty)))
        if self.average_price:
            # Правильный расчет средней цены
            old_filled_qty = Decimal(str(self.filled_quantity)) - Decimal(str(filled_qty))
            old_total_value = self.average_price.amount * old_filled_qty
            new_total_value = price.amount * Decimal(str(filled_qty))
            total_value = old_total_value + new_total_value
            self.average_price = Price(
                total_value / Decimal(str(self.filled_quantity)), self.average_price.currency
            )
        else:
            self.average_price = price
        if commission:
            if self.commission:
                self.commission = Price(
                    self.commission.amount + commission.amount, self.commission.currency
                )
            else:
                self.commission = commission
        self.updated_at = Timestamp.now()
        if Decimal(str(self.filled_quantity)) >= Decimal(str(self.quantity)):
            self.status = OrderStatus.FILLED
            self.filled_at = Timestamp.now()
        elif Decimal(str(self.filled_quantity)) > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

    def fill(self, filled_qty: Volume, price: Price) -> None:
        """Заполнить ордер."""
        if self.is_cancelled:
            raise OrderError("Cannot fill cancelled order")
        if self.is_filled:
            raise OrderError("Cannot fill already filled order")
        if filled_qty.to_decimal() > self.remaining_quantity:
            raise OrderError("Cannot fill more than order quantity")
        
        self.update_fill(VolumeValue(filled_qty.to_decimal()), price)

    def cancel(self) -> None:
        """Отменить ордер."""
        if self.is_filled:
            raise OrderError("Cannot cancel filled order")
        if self.is_cancelled:
            raise OrderError("Order is already cancelled")
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = Timestamp.now()

    def update_price(self, new_price: Price) -> None:
        """Обновить цену ордера."""
        if self.order_type == OrderType.MARKET:
            raise OrderError("Cannot update price for market order")
        if self.is_filled:
            raise OrderError("Cannot update price for filled order")
        if self.is_cancelled:
            raise OrderError("Cannot update price for cancelled order")
        
        self.price = new_price
        self.updated_at = Timestamp.now()

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": str(self.id),
            "portfolio_id": str(self.portfolio_id),
            "strategy_id": str(self.strategy_id),
            "signal_id": str(self.signal_id) if self.signal_id else "",
            "exchange_order_id": self.exchange_order_id or "",
            "symbol": str(self.symbol),
            "trading_pair": str(self.trading_pair),
            "order_type": self.order_type.value,
            "side": self.side.value,
            "amount": str(self.amount.to_decimal()),
            "quantity": str(self.quantity),
            "price": str(self.price.amount) if self.price else "",
            "stop_price": str(self.stop_price.amount) if self.stop_price else "",
            "status": self.status.value,
            "filled_amount": str(self.filled_amount.to_decimal()),
            "filled_quantity": str(self.filled_quantity),
            "average_price": (
                str(self.average_price.amount) if self.average_price else ""
            ),
            "commission": str(self.commission.amount) if self.commission else "",
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "filled_at": str(self.filled_at) if self.filled_at else "",
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Order":
        return cls(
            id=OrderId(UUID(data["id"])) if data.get("id") else OrderId(uuid4()),
            portfolio_id=(
                PortfolioId(UUID(data["portfolio_id"]))
                if data.get("portfolio_id")
                else PortfolioId(uuid4())
            ),
            strategy_id=(
                StrategyId(UUID(data["strategy_id"]))
                if data.get("strategy_id")
                else StrategyId(uuid4())
            ),
            signal_id=(
                SignalId(UUID(data["signal_id"])) if data.get("signal_id") else None
            ),
            exchange_order_id=data.get("exchange_order_id") or None,
            symbol=Symbol(data.get("symbol", "")),
            trading_pair=TradingPair(data.get("trading_pair", "")),
            order_type=OrderType(data["order_type"]),
            side=OrderSide(data["side"]),
            amount=Volume(Decimal(data["amount"]), Currency.USD),
            quantity=VolumeValue(Decimal(data["quantity"])),
            price=(
                Price(Decimal(data["price"]), Currency.USD)
                if data.get("price")
                else None
            ),
            stop_price=(
                Price(Decimal(data["stop_price"]), Currency.USD)
                if data.get("stop_price")
                else None
            ),
            status=OrderStatus(data["status"]),
            filled_amount=Volume(Decimal(data["filled_amount"]), Currency.USD),
            filled_quantity=VolumeValue(Decimal(data["filled_quantity"])),
            average_price=(
                Price(Decimal(data["average_price"]), Currency.USD)
                if data.get("average_price")
                else None
            ),
            commission=(
                Price(Decimal(data["commission"]), Currency.USD)
                if data.get("commission")
                else None
            ),
            created_at=Timestamp.from_iso(data["created_at"]),
            updated_at=Timestamp.from_iso(data["updated_at"]),
            filled_at=(
                Timestamp.from_iso(data["filled_at"]) if data.get("filled_at") else None
            ),
            metadata=(
                ast.literal_eval(data.get("metadata", "{}"))
                if data.get("metadata")
                else {}
            ),
        )

    def __str__(self) -> str:
        price_str = f" @ {self.price.amount}" if self.price else ""
        return f"Order({self.side.value} {self.order_type.value} {self.quantity} {self.trading_pair}{price_str}, status={self.status.value})"

    def __eq__(self, other: Any) -> bool:
        """Сравнение ордеров по бизнес-логике."""
        if not isinstance(other, Order):
            return False
        return (
            self.trading_pair == other.trading_pair
            and self.side == other.side
            and self.order_type == other.order_type
            and self.quantity == other.quantity
            and self.price == other.price
            and self.status == other.status
        )

    def __hash__(self) -> int:
        """Хеш-код ордера."""
        return hash((
            self.trading_pair,
            self.side,
            self.order_type,
            self.quantity,
            self.price,
            self.status,
        ))

    def __repr__(self) -> str:
        return (
            f"Order(id={self.id}, portfolio_id={self.portfolio_id}, strategy_id={self.strategy_id}, "
            f"trading_pair='{self.trading_pair}', order_type={self.order_type.value}, "
            f"side={self.side.value}, quantity={self.quantity}, status={self.status.value})"
        )
