# -*- coding: utf-8 -*-
"""
Доменные сущности для торговли.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from domain.types import MetadataDict
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import TimestampValue
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
            self.price = Price(Decimal(str(self.price)), Currency.USD)

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
                Price(Decimal(data["price"]), Currency.USD)
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
    timestamp: TimestampValue

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
    quantity: Volume = field(default_factory=lambda: Volume(Decimal("0")))
    price: Price = field(default_factory=lambda: Price(Decimal("0"), Currency.USD))
    commission: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )

    def __post_init__(self) -> None:
        self.quantity = Volume(Decimal(str(self.quantity)))
        self.price = Price(Decimal(str(self.price)), Currency.USD)
        self.commission = Money(Decimal(str(self.commission)), Currency.USD)

    @property
    def total_value(self) -> Money:
        """Общая стоимость сделки"""
        return Money(self.price.value * self.quantity.value, Currency.USD)

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
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Создание из словаря."""
        return cls(
            id=TradeId(str(data["id"])),
            order_id=OrderId(str(data["order_id"])),
            trading_pair=TradingPair(data["trading_pair"]),
            side=OrderSide(data["side"]),
            quantity=Volume(Decimal(data["quantity"])),
            price=Price(Decimal(data["price"]), Currency.USD),
            commission=Money(Decimal(data["commission"]), Currency.USD),
            timestamp=TimestampValue(datetime.fromisoformat(data["timestamp"])),
        )


@dataclass
class Position:
    """Позиция."""

    id: PositionId = field(default_factory=lambda: PositionId(str(uuid4())))
    symbol: Symbol = field(default=Symbol(""))
    side: PositionSide = PositionSide.LONG
    quantity: Volume = field(default_factory=lambda: Volume(Decimal("0")))
    entry_price: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USD)
    )
    current_price: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USD)
    )
    entry_time: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def __post_init__(self) -> None:
        self.quantity = Volume(Decimal(str(self.quantity)))
        self.entry_price = Price(Decimal(str(self.entry_price)), Currency.USD)
        self.current_price = Price(Decimal(str(self.current_price)), Currency.USD)
        self.unrealized_pnl = Money(Decimal(str(self.unrealized_pnl)), Currency.USD)
        self.realized_pnl = Money(Decimal(str(self.realized_pnl)), Currency.USD)

    def get_market_value(self) -> Money:
        """Получение рыночной стоимости позиции."""
        return Money(self.current_price.value * self.quantity.value, Currency.USD)

    def get_entry_value(self) -> Money:
        """Получение стоимости входа в позицию."""
        return Money(self.entry_price.value * self.quantity.value, Currency.USD)

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
        return Money(pnl, Currency.USD)

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
            quantity=Volume(Decimal(data["quantity"])),
            entry_price=Price(Decimal(data["entry_price"]), Currency.USD),
            current_price=Price(Decimal(data["current_price"]), Currency.USD),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            stop_loss=(
                Price(Decimal(data["stop_loss"]), Currency.USD)
                if data.get("stop_loss")
                else None
            ),
            take_profit=(
                Price(Decimal(data["take_profit"]), Currency.USD)
                if data.get("take_profit")
                else None
            ),
            unrealized_pnl=Money(Decimal(data["unrealized_pnl"]), Currency.USD),
            realized_pnl=Money(Decimal(data["realized_pnl"]), Currency.USD),
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
    total_volume: Volume = field(default_factory=lambda: Volume(Decimal("0")))
    total_commission: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))

    def __post_init__(self) -> None:
        self.total_volume = Volume(Decimal(str(self.total_volume)))
        self.total_commission = Money(Decimal(str(self.total_commission)), Currency.USD)
        self.pnl = Money(Decimal(str(self.pnl)), Currency.USD)

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
            total_volume=Volume(Decimal(data["total_volume"])),
            total_commission=Money(Decimal(data["total_commission"]), Currency.USD),
            pnl=Money(Decimal(data["pnl"]), Currency.USD),
        )
