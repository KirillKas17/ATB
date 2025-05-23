from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


@dataclass
class MarketData:
    """Модель рыночных данных."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    pair: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """
        Создание объекта из словаря.

        Args:
            data: Словарь с данными

        Returns:
            MarketData: Объект с рыночными данными
        """
        timestamp = data.get("timestamp")
        if timestamp is not None:
            if isinstance(timestamp, str):
                parsed = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                parsed = pd.to_datetime(timestamp, unit="ms")
            else:
                parsed = datetime.now()
        else:
            parsed = datetime.now()

        return cls(
            timestamp=parsed,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            pair=str(data["pair"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "pair": self.pair,
        }


@dataclass
class Position:
    """Позиция."""

    pair: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "pair": self.pair,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "pnl": self.pnl,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """
        Создание из словаря.

        Args:
            data: Словарь с данными

        Returns:
            Position: Позиция
        """
        return cls(
            pair=data["pair"],
            side=data["side"],
            size=float(data["size"]),
            entry_price=float(data["entry_price"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            current_price=float(data["current_price"]),
            pnl=float(data["pnl"]) if data.get("pnl") is not None else 0.0,
            stop_loss=(
                float(data["stop_loss"]) if data.get("stop_loss") is not None else None
            ),
            take_profit=(
                float(data["take_profit"])
                if data.get("take_profit") is not None
                else None
            ),
            unrealized_pnl=(
                float(data["unrealized_pnl"])
                if data.get("unrealized_pnl") is not None
                else 0.0
            ),
            realized_pnl=(
                float(data["realized_pnl"])
                if data.get("realized_pnl") is not None
                else 0.0
            ),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now()
            ),
        )


@dataclass
class Order:
    """Модель ордера."""

    id: str
    pair: str
    type: str  # market, limit
    side: str  # buy, sell
    price: float
    size: float
    status: str  # new, filled, canceled, rejected
    timestamp: datetime
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    filled_timestamp: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """
        Создание объекта из словаря.

        Args:
            data: Словарь с данными

        Returns:
            Order: Объект ордера
        """
        return cls(
            id=str(data["id"]),
            pair=str(data["symbol"]),
            type=str(data["type"]),
            side=str(data["side"]),
            price=float(data["price"] or 0.0),
            size=float(data["amount"]),
            status=str(data["status"]),
            timestamp=pd.to_datetime(data["timestamp"]),
            filled_price=float(data["average"]) if data.get("average") else None,
            filled_size=float(data["filled"]) if data.get("filled") else None,
            filled_timestamp=(
                pd.to_datetime(data["lastTradeTimestamp"])
                if data.get("lastTradeTimestamp")
                else None
            ),
        )

    @classmethod
    def from_exchange_data(cls, data: Dict[str, Any]) -> "Order":
        """
        Создание объекта из данных биржи.

        Args:
            data: Данные от биржи

        Returns:
            Order: Объект ордера
        """
        timestamp = data.get("timestamp")
        if timestamp is not None:
            if isinstance(timestamp, str):
                parsed_timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                parsed_timestamp = pd.to_datetime(timestamp, unit="ms")
            else:
                parsed_timestamp = datetime.now()
        else:
            parsed_timestamp = datetime.now()

        filled_timestamp = data.get("lastTradeTimestamp")
        if filled_timestamp is not None:
            if isinstance(filled_timestamp, str):
                parsed_filled_timestamp = pd.to_datetime(filled_timestamp)
            elif isinstance(filled_timestamp, (int, float)):
                parsed_filled_timestamp = pd.to_datetime(filled_timestamp, unit="ms")
            else:
                parsed_filled_timestamp = None
        else:
            parsed_filled_timestamp = None

        return cls(
            id=str(data.get("id", "")),
            pair=str(data.get("symbol", "")),
            type=str(data.get("type", "market")),
            side=str(data.get("side", "buy")),
            price=float(data.get("price", 0.0)),
            size=float(data.get("amount", 0.0)),
            status=str(data.get("status", "new")),
            timestamp=parsed_timestamp,
            filled_price=(
                float(data.get("average", 0.0)) if data.get("average") else None
            ),
            filled_size=float(data.get("filled", 0.0)) if data.get("filled") else None,
            filled_timestamp=parsed_filled_timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "id": self.id,
            "pair": self.pair,
            "type": self.type,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "filled_price": self.filled_price,
            "filled_size": self.filled_size,
            "filled_timestamp": (
                self.filled_timestamp.isoformat() if self.filled_timestamp else None
            ),
        }


@dataclass
class Trade:
    """Модель сделки."""

    id: str
    pair: str
    side: str  # buy, sell
    size: float
    price: float
    timestamp: datetime
    fee: float
    pnl: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """
        Создание объекта из словаря.

        Args:
            data: Словарь с данными

        Returns:
            Trade: Объект сделки
        """
        return cls(
            id=str(data["id"]),
            pair=str(data["pair"]),
            side=str(data["side"]),
            size=float(data["size"]),
            price=float(data["price"]),
            timestamp=pd.to_datetime(data["timestamp"]),
            fee=float(data["fee"]),
            pnl=float(data["pnl"]) if "pnl" in data else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "id": self.id,
            "pair": self.pair,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "fee": self.fee,
            "pnl": self.pnl,
        }


@dataclass
class Account:
    """Модель аккаунта."""

    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    timestamp: datetime
    positions: List[Position] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        """
        Создание объекта из словаря.

        Args:
            data: Словарь с данными

        Returns:
            Account: Объект аккаунта
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, (int, float)):
            timestamp = pd.to_datetime(timestamp, unit="ms")
        else:
            timestamp = datetime.now()

        return cls(
            balance=float(data["balance"]),
            equity=float(data["equity"]),
            margin=float(data["margin"]),
            free_margin=float(data["free_margin"]),
            margin_level=float(data["margin_level"]),
            timestamp=timestamp,
            positions=[Position.from_dict(p) for p in data.get("positions", [])],
            orders=[Order.from_dict(o) for o in data.get("orders", [])],
            trades=[Trade.from_dict(t) for t in data.get("trades", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "free_margin": self.free_margin,
            "margin_level": self.margin_level,
            "timestamp": self.timestamp.isoformat(),
            "positions": [p.to_dict() for p in self.positions],
            "orders": [o.to_dict() for o in self.orders],
            "trades": [t.to_dict() for t in self.trades],
        }


class SystemState(BaseModel):
    """Состояние системы."""

    is_running: bool = False
    is_trading: bool = False
    last_update: datetime = Field(default_factory=datetime.now)
    active_pairs: List[str] = []
    active_positions: Dict[str, Any] = {}
    active_orders: Dict[str, Any] = {}
    market_data: Dict[str, Any] = {}
    risk_metrics: Dict[str, float] = {}
    performance_metrics: Dict[str, float] = {}
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    last_warning: Optional[str] = None
