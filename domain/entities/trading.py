# -*- coding: utf-8 -*-
"""
Доменные сущности для торговли.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Union
from uuid import UUID, uuid4

from domain.type_definitions import MetadataDict
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume

# Типы для идентификаторов
OrderId = UUID
TradeId = UUID
PositionId = UUID
Symbol = str
TradingPair = str


class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Стороны ордера."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Статусы ордера."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Стороны позиции."""

    LONG = "long"
    SHORT = "short"


class SignalType(Enum):
    """Типы сигналов."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Торговый сигнал."""

    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    strength: float = 0.0
    confidence: float = 0.0
    price: Optional[Price] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def __post_init__(self) -> None:
        if self.price is not None:
            self.price = Price(amount=Decimal(str(self.price)), currency=Currency.USD)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "price": str(self.price.value) if self.price else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Создание из словаря."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            signal_type=SignalType(data["signal_type"]),
            strength=data["strength"],
            confidence=data["confidence"],
            price=(
                Price(amount=Decimal(data["price"]), currency=Currency.USD)
                if data.get("price")
                else None
            ),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", MetadataDict({})),
        )


@runtime_checkable
class OrderProtocol(Protocol):
    """Протокол для ордеров."""

    id: OrderId
    trading_pair: TradingPair
    side: OrderSide
    order_type: OrderType
    quantity: Volume
    price: Optional[Price]
    stop_price: Optional[Price]
    status: OrderStatus
    filled_quantity: Volume
    average_price: Optional[Price]
    commission: Money
    created_at: datetime
    updated_at: datetime
    metadata: MetadataDict

    def fill(self, quantity: Volume, price: Price) -> None: ...
    def cancel(self) -> None: ...
    @property
    def remaining_quantity(self) -> Volume: ...
    @property
    def is_active(self) -> bool: ...
    def is_filled(self) -> bool: ...
    def is_cancelled(self) -> bool: ...
    def get_remaining_quantity(self) -> Volume: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderProtocol": ...


@runtime_checkable
class TradeProtocol(Protocol):
    """Протокол для сделок."""

    id: TradeId
    order_id: OrderId
    trading_pair: TradingPair
    side: OrderSide
    quantity: Volume
    price: Price
    commission: Money
    timestamp: Timestamp

    @property
    def total_value(self) -> Money: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeProtocol": ...


@runtime_checkable
class PositionProtocol(Protocol):
    """Протокол для позиций."""

    id: PositionId
    symbol: Symbol
    side: PositionSide
    quantity: Volume
    entry_price: Price
    current_price: Price
    entry_time: datetime
    updated_at: datetime
    stop_loss: Optional[Price]
    take_profit: Optional[Price]
    unrealized_pnl: Money
    realized_pnl: Money
    metadata: MetadataDict

    def get_market_value(self) -> Money: ...
    def get_entry_value(self) -> Money: ...
    def calculate_unrealized_pnl(self) -> Money: ...
    def is_profitable(self) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionProtocol": ...


@dataclass
class Trade:
    """Сделка - результат исполнения ордера"""

    id: TradeId = field(default_factory=lambda: TradeId(str(uuid4())))
    order_id: OrderId = field(default_factory=lambda: OrderId(str(uuid4())))
    trading_pair: TradingPair = field(default=TradingPair(""))
    side: OrderSide = OrderSide.BUY
    quantity: Volume = field(default_factory=lambda: Volume(amount=Decimal("0"), currency=Currency.USD))
    price: Price = field(default_factory=lambda: Price(amount=Decimal("0"), currency=Currency.USD))
    commission: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    timestamp: Timestamp = field(
        default_factory=lambda: Timestamp(datetime.now())
    )

    # Removed __post_init__ to avoid mypy type checking issues

    @property
    def total_value(self) -> Money:
        """Общая стоимость сделки"""
        return Money(amount=self.price.value * self.quantity.value, currency=Currency.USD)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "order_id": str(self.order_id),
            "trading_pair": str(self.trading_pair),
            "side": self.side.value,
            "quantity": str(self.quantity.value),
            "price": str(self.price.value),
            "commission": str(self.commission.value),
            "timestamp": self.timestamp.value.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Создание из словаря."""
        return cls(
            id=TradeId(str(data["id"])),
            order_id=OrderId(str(data["order_id"])),
            trading_pair=TradingPair(data["trading_pair"]),
            side=OrderSide(data["side"]),
            quantity=Volume(amount=Decimal(data["quantity"]), currency=Currency.USD),
            price=Price(amount=Decimal(data["price"]), currency=Currency.USD),
            commission=Money(amount=Decimal(data["commission"]), currency=Currency.USD),
            timestamp=Timestamp(datetime.fromisoformat(data["timestamp"])),
        )


@dataclass
class Position:
    """Позиция."""

    id: PositionId = field(default_factory=lambda: PositionId(str(uuid4())))
    symbol: Symbol = field(default=Symbol(""))
    side: PositionSide = PositionSide.LONG
    quantity: Volume = field(default_factory=lambda: Volume(amount=Decimal("0"), currency=Currency.USD))
    entry_price: Price = field(
        default_factory=lambda: Price(amount=Decimal("0"), currency=Currency.USD)
    )
    current_price: Price = field(
        default_factory=lambda: Price(amount=Decimal("0"), currency=Currency.USD)
    )
    entry_time: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    # Removed __post_init__ to avoid mypy type checking issues

    def get_market_value(self) -> Money:
        """Получение рыночной стоимости позиции."""
        return Money(amount=self.current_price.value * self.quantity.value, currency=Currency.USD)

    def get_entry_value(self) -> Money:
        """Получение стоимости входа в позицию."""
        return Money(amount=self.entry_price.value * self.quantity.value, currency=Currency.USD)

    def calculate_unrealized_pnl(self) -> Money:
        """Расчет нереализованного P&L."""
        if self.side == PositionSide.LONG:
            pnl = (
                self.current_price.value - self.entry_price.value
            ) * self.quantity.value
        else:
            pnl = (
                self.entry_price.value - self.current_price.value
            ) * self.quantity.value
        return Money(amount=pnl, currency=Currency.USD)

    def is_profitable(self) -> bool:
        """Проверка прибыльности позиции."""
        return self.calculate_unrealized_pnl().value > 0

    @property
    def is_open(self) -> bool:
        """Проверка, открыта ли позиция."""
        # Позиция считается открытой, если количество больше 0
        return self.quantity.value > 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "symbol": str(self.symbol),
            "side": self.side.value,
            "quantity": str(self.quantity.value),
            "entry_price": str(self.entry_price.value),
            "current_price": str(self.current_price.value),
            "entry_time": self.entry_time.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stop_loss": str(self.stop_loss.value) if self.stop_loss else None,
            "take_profit": str(self.take_profit.value) if self.take_profit else None,
            "unrealized_pnl": str(self.unrealized_pnl.value),
            "realized_pnl": str(self.realized_pnl.value),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Создание из словаря."""
        return cls(
            id=PositionId(str(data["id"])),
            symbol=Symbol(data["symbol"]),
            side=PositionSide(data["side"]),
            quantity=Volume(amount=Decimal(data["quantity"]), currency=Currency.USD),
            entry_price=Price(amount=Decimal(data["entry_price"]), currency=Currency.USD),
            current_price=Price(amount=Decimal(data["current_price"]), currency=Currency.USD),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            stop_loss=(
                Price(amount=Decimal(data["stop_loss"]), currency=Currency.USD)
                if data.get("stop_loss")
                else None
            ),
            take_profit=(
                Price(amount=Decimal(data["take_profit"]), currency=Currency.USD)
                if data.get("take_profit")
                else None
            ),
            unrealized_pnl=Money(amount=Decimal(data["unrealized_pnl"]), currency=Currency.USD),
            realized_pnl=Money(amount=Decimal(data["realized_pnl"]), currency=Currency.USD),
            metadata=data.get("metadata", MetadataDict({})),
        )


@dataclass
class TradingSession:
    """Торговая сессия"""

    id: OrderId = field(default_factory=lambda: OrderId(str(uuid4())))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    trading_pairs: List[TradingPair] = field(default_factory=list)
    total_trades: int = 0
    total_volume: Volume = field(default_factory=lambda: Volume(amount=Decimal("0"), currency=Currency.USD))
    total_commission: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    pnl: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))

    # Removed __post_init__ to avoid mypy type checking issues

    def add_trade(self, trade: Trade) -> None:
        """Добавить сделку в сессию"""
        self.total_trades += 1
        self.total_volume += trade.quantity
        self.total_commission += trade.commission

    def close(self) -> None:
        """Закрыть сессию"""
        self.end_time = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Длительность сессии в секундах"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "trading_pairs": [str(pair) for pair in self.trading_pairs],
            "total_trades": self.total_trades,
            "total_volume": str(self.total_volume.value),
            "total_commission": str(self.total_commission.value),
            "pnl": str(self.pnl.value),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSession":
        """Создание из словаря."""
        return cls(
            id=OrderId(str(data["id"])),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            trading_pairs=[TradingPair(pair) for pair in data.get("trading_pairs", [])],
            total_trades=data["total_trades"],
            total_volume=Volume(amount=Decimal(data["total_volume"]), currency=Currency.USD),
            total_commission=Money(amount=Decimal(data["total_commission"]), currency=Currency.USD),
            pnl=Money(amount=Decimal(data["pnl"]), currency=Currency.USD),
        )
