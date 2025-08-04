"""
Доменные сущности портфеля.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from domain.value_objects.volume import Volume

from .order import Order
from .trading import Trade


@dataclass(frozen=True)
class Account:
    """Аккаунт."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    balance: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    equity: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    margin: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    free_margin: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    margin_level: Decimal = Decimal("0")
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_margin_used(self) -> Money:
        """Получение использованной маржи."""
        return Money(amount=self.margin.amount, currency=Currency.USD)

    def get_margin_available(self) -> Money:
        """Получение доступной маржи."""
        return Money(amount=self.free_margin.amount, currency=Currency.USD)

    def get_margin_ratio(self) -> Decimal:
        """Получение коэффициента маржи."""
        if self.margin.amount == 0:
            return Decimal("0")
        return self.equity.amount / self.margin.amount

    def is_margin_call(self) -> bool:
        """Проверка маржин-колла."""
        return self.margin_level < Decimal("1.1")  # 110%

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "balance": str(self.balance.amount),
            "equity": str(self.equity.amount),
            "margin": str(self.margin.amount),
            "free_margin": str(self.free_margin.amount),
            "margin_level": str(self.margin_level),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        """Создание из словаря."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            name=data["name"],
            balance=Money(amount=Decimal(data["balance"]), currency=Currency.USD),
            equity=Money(amount=Decimal(data["equity"]), currency=Currency.USD),
            margin=Money(amount=Decimal(data["margin"]), currency=Currency.USD),
            free_margin=Money(amount=Decimal(data["free_margin"]), currency=Currency.USD),
            margin_level=Decimal(data["margin_level"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Position:
    """Позиция в портфеле"""

    id: UUID = field(default_factory=uuid4)
    trading_pair: str = ""
    side: str = "long"  # long/short
    quantity: Volume = field(default_factory=lambda: Volume(amount=Decimal("0"), currency=Currency.USD))
    average_price: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    current_price: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    margin_used: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    leverage: Decimal = Decimal("1")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        # Проверяем только если поля не инициализированы корректно
        if not hasattr(self, '_quantity_validated'):
            self._quantity_validated = True
        if not hasattr(self, '_average_price_validated'):
            self._average_price_validated = True
        if not hasattr(self, '_current_price_validated'):
            self._current_price_validated = True
        if not hasattr(self, '_unrealized_pnl_validated'):
            self._unrealized_pnl_validated = True
        if not hasattr(self, '_realized_pnl_validated'):
            self._realized_pnl_validated = True
        if not hasattr(self, '_margin_used_validated'):
            self._margin_used_validated = True
        if not hasattr(self, '_leverage_validated'):
            self._leverage_validated = True

    def update_price(self, new_price: Money) -> None:
        """Обновить текущую цену и пересчитать PnL"""
        self.current_price = new_price
        self._calculate_unrealized_pnl()
        self.updated_at = datetime.now()

    def _calculate_unrealized_pnl(self) -> None:
        """Рассчитать нереализованный PnL"""
        if self.side == "long":
            pnl_amount = (
                self.current_price.amount - self.average_price.amount
            ) * self.quantity.value
            self.unrealized_pnl = Money(amount=pnl_amount, currency=Currency.USD)
        else:  # short
            pnl_amount = (
                self.average_price.amount - self.current_price.amount
            ) * self.quantity.value
            self.unrealized_pnl = Money(amount=pnl_amount, currency=Currency.USD)

    def add_quantity(self, quantity: Volume, price: Money) -> None:
        """Добавить количество к позиции"""
        if self.quantity.value == 0:
            self.average_price = price
        else:
            # Пересчет средней цены
            total_value = (self.average_price.amount * self.quantity.value) + (
                price.amount * quantity.value
            )
            self.quantity += quantity
            self.average_price = Money(amount=total_value / self.quantity.value, currency=Currency.USD)

        self._calculate_unrealized_pnl()
        self.updated_at = datetime.now()

    def reduce_quantity(self, quantity: Volume, price: Money) -> Money:
        """Уменьшить количество позиции и вернуть реализованный PnL"""
        if quantity > self.quantity:
            raise ValueError("Cannot reduce more than current quantity")

        # Рассчитать реализованный PnL
        if self.side == "long":
            realized_pnl_amount = (
                price.amount - self.average_price.amount
            ) * quantity.value
        else:  # short
            realized_pnl_amount = (
                self.average_price.amount - price.amount
            ) * quantity.value

        self.realized_pnl += Money(amount=realized_pnl_amount, currency=Currency.USD)
        self.quantity -= quantity

        if self.quantity.value == 0:
            self.average_price = Money(amount=Decimal("0"), currency=Currency.USD)
            self.unrealized_pnl = Money(amount=Decimal("0"), currency=Currency.USD)
        else:
            self._calculate_unrealized_pnl()

        self.updated_at = datetime.now()
        return Money(amount=realized_pnl_amount, currency=Currency.USD)

    @property
    def total_pnl(self) -> Money:
        """Общий PnL (реализованный + нереализованный)"""
        return Money(amount=
            self.realized_pnl.amount + self.unrealized_pnl.amount, currency=Currency.USD
        )

    @property
    def market_value(self) -> Money:
        """Рыночная стоимость позиции"""
        return Money(amount=self.current_price.amount * self.quantity.value, currency=Currency.USD)

    @property
    def is_open(self) -> bool:
        """Открыта ли позиция"""
        return self.quantity.value > 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "trading_pair": self.trading_pair,
            "side": self.side,
            "quantity": str(self.quantity.value),
            "average_price": str(self.average_price.amount),
            "current_price": str(self.current_price.amount),
            "unrealized_pnl": str(self.unrealized_pnl.amount),
            "realized_pnl": str(self.realized_pnl.amount),
            "margin_used": str(self.margin_used.amount),
            "leverage": str(self.leverage),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Создание из словаря."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            trading_pair=data["trading_pair"],
            side=data["side"],
            quantity=Volume(amount=Decimal(data["quantity"]), currency=Currency.USD),
            average_price=Money(amount=Decimal(data["average_price"]), currency=Currency.USD),
            current_price=Money(amount=Decimal(data["current_price"]), currency=Currency.USD),
            unrealized_pnl=Money(amount=Decimal(data["unrealized_pnl"]), currency=Currency.USD),
            realized_pnl=Money(amount=Decimal(data["realized_pnl"]), currency=Currency.USD),
            margin_used=Money(amount=Decimal(data["margin_used"]), currency=Currency.USD),
            leverage=Decimal(data["leverage"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Balance:
    """Баланс аккаунта"""

    currency: Currency = Currency.USD
    available: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    total: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    locked: Money = field(default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD))
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        # Проверяем только если поля не инициализированы корректно
        if not hasattr(self, '_available_validated'):
            self._available_validated = True
        if not hasattr(self, '_total_validated'):
            self._total_validated = True
        if not hasattr(self, '_locked_validated'):
            self._locked_validated = True
        if not hasattr(self, '_unrealized_pnl_validated'):
            self._unrealized_pnl_validated = True

    def update_available(self, amount: Money) -> None:
        """Обновить доступный баланс"""
        self.available = amount
        self.total = Money(amount=
            self.available.amount + self.locked.amount + self.unrealized_pnl.amount, currency=self.currency,
        )

    def lock_amount(self, amount: Money) -> None:
        """Заблокировать сумму"""
        if amount > self.available:
            raise ValueError("Insufficient available balance")
        self.locked = Money(amount=self.locked.amount + amount.amount, currency=self.currency)
        self.available = Money(amount=self.available.amount - amount.amount, currency=self.currency)

    def unlock_amount(self, amount: Money) -> None:
        """Разблокировать сумму"""
        if amount > self.locked:
            raise ValueError("Cannot unlock more than locked amount")
        self.locked = Money(amount=self.locked.amount - amount.amount, currency=self.currency)
        self.available = Money(amount=self.available.amount + amount.amount, currency=self.currency)

    def add_unrealized_pnl(self, pnl: Money) -> None:
        """Добавить нереализованный PnL"""
        self.unrealized_pnl = Money(amount=
            self.unrealized_pnl.amount + pnl.amount, currency=self.currency
        )
        self.total = Money(amount=
            self.available.amount + self.locked.amount + self.unrealized_pnl.amount, currency=self.currency,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "currency": self.currency.code,
            "available": str(self.available.amount),
            "total": str(self.total.amount),
            "locked": str(self.locked.amount),
            "unrealized_pnl": str(self.unrealized_pnl.amount),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Balance":
        """Создание из словаря."""
        currency = Currency(data["currency"])
        return cls(
            currency=currency,
            available=Money(amount=Decimal(data["available"]), currency=currency),
            total=Money(amount=Decimal(data["total"]), currency=currency),
            locked=Money(amount=Decimal(data["locked"]), currency=currency),
            unrealized_pnl=Money(amount=Decimal(data["unrealized_pnl"]), currency=currency),
        )


@dataclass
class Portfolio:
    """Портфель - агрегат с позициями и балансами"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    account_id: str = ""
    balances: Dict[Currency, Balance] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    total_equity: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    total_margin_used: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    free_margin: Money = field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    margin_level: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        # Проверяем только если поля не инициализированы корректно
        if not hasattr(self, '_total_equity_validated'):
            self._total_equity_validated = True
        if not hasattr(self, '_total_margin_used_validated'):
            self._total_margin_used_validated = True
        if not hasattr(self, '_free_margin_validated'):
            self._free_margin_validated = True
        if not hasattr(self, '_margin_level_validated'):
            self._margin_level_validated = True

    def add_balance(self, currency: Currency, balance: Balance) -> None:
        """Добавить баланс"""
        self.balances[currency] = balance
        self._recalculate_metrics()

    def update_balance(self, currency: Currency, **kwargs: Any) -> None:
        """Обновить баланс"""
        if currency not in self.balances:
            self.balances[currency] = Balance(currency=currency)

        balance = self.balances[currency]
        for key, value in kwargs.items():
            if hasattr(balance, key):
                setattr(balance, key, value)

        self._recalculate_metrics()

    def add_position(self, trading_pair: str, position: Position) -> None:
        """Добавить позицию"""
        self.positions[trading_pair] = position
        self._recalculate_metrics()

    def update_position(self, trading_pair: str, **kwargs: Any) -> None:
        """Обновить позицию"""
        if trading_pair not in self.positions:
            raise ValueError(f"Position {trading_pair} not found")

        position = self.positions[trading_pair]
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)

        self._recalculate_metrics()

    def remove_position(self, trading_pair: str) -> None:
        """Удалить позицию"""
        if trading_pair in self.positions:
            del self.positions[trading_pair]
            self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Пересчитать метрики портфеля"""
        # Общая стоимость в USD
        total_equity = Money(amount=Decimal("0"), currency=Currency.USD)
        total_margin_used = Money(amount=Decimal("0"), currency=Currency.USD)

        # Суммируем балансы
        for balance in self.balances.values():
            total_equity = Money(amount=
                total_equity.amount + balance.total.amount, currency=Currency.USD
            )

        # Суммируем позиции
        for position in self.positions.values():
            total_equity = Money(amount=
                total_equity.amount + position.total_pnl.amount, currency=Currency.USD
            )
            total_margin_used = Money(amount=
                total_margin_used.amount + position.margin_used.amount, currency=Currency.USD
            )

        self.total_equity = total_equity
        self.total_margin_used = total_margin_used
        self.free_margin = Money(amount=
            total_equity.amount - total_margin_used.amount, currency=Currency.USD
        )

        # Уровень маржи
        if total_margin_used.amount > 0:
            self.margin_level = Percentage(
                (total_equity.amount / total_margin_used.amount) * 100
            )
        else:
            self.margin_level = Percentage(Decimal("0"))

        self.updated_at = datetime.now()

    def get_position(self, trading_pair: str) -> Optional[Position]:
        """Получить позицию по торговой паре"""
        return self.positions.get(trading_pair)

    def get_balance(self, currency: Currency) -> Optional[Balance]:
        """Получить баланс по валюте"""
        return self.balances.get(currency)

    @property
    def open_positions(self) -> Dict[str, Position]:
        """Открытые позиции"""
        return {pair: pos for pair, pos in self.positions.items() if pos.is_open is True}

    @property
    def total_positions_value(self) -> Money:
        """Общая стоимость позиций"""
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.market_value.amount
        return Money(amount=total, currency=Currency.USD)

    @property
    def total_pnl(self) -> Money:
        """Общий PnL"""
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.total_pnl.amount
        return Money(amount=total, currency=Currency.USD)

    def get_total_value(self) -> Money:
        """Получение общей стоимости портфеля."""
        return Money(amount=self.total_equity.amount, currency=Currency.USD)

    def get_positions_value(self) -> Money:
        """Получение стоимости позиций."""
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.market_value.amount
        return Money(amount=total, currency=Currency.USD)

    def get_cash_balance(self) -> Money:
        """Получение наличного баланса."""
        if Currency.USD in self.balances:
            return Money(amount=self.balances[Currency.USD].available.amount, currency=Currency.USD)
        return Money(amount=Decimal("0"), currency=Currency.USD)

    def get_unrealized_pnl(self) -> Money:
        """Получение нереализованной прибыли/убытка."""
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.unrealized_pnl.amount
        return Money(amount=total, currency=Currency.USD)

    def get_realized_pnl(self) -> Money:
        """Получение реализованной прибыли/убытка."""
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.realized_pnl.amount
        return Money(amount=total, currency=Currency.USD)

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Получение позиции по символу."""
        for position in self.positions.values():
            if position.trading_pair == symbol:
                return position
        return None

    def get_active_orders(self) -> List[Order]:
        """Получение активных ордеров."""
        return [order for order in self.orders if hasattr(order, 'is_active') and order.is_active]

    def get_filled_orders(self) -> List[Order]:
        """Получение исполненных ордеров."""
        return [order for order in self.orders if hasattr(order, 'is_filled') and order.is_filled is True]

    def get_cancelled_orders(self) -> List[Order]:
        """Получение отмененных ордеров."""
        return [order for order in self.orders if hasattr(order, 'is_cancelled') and order.is_cancelled is True]

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Получение сделок по символу."""
        return [trade for trade in self.trades if trade.trading_pair == symbol]

    def get_trades_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """Получение сделок по диапазону дат."""
        return [
            trade for trade in self.trades 
            if start_date <= (trade.timestamp.value if hasattr(trade.timestamp, 'value') else trade.timestamp).replace(tzinfo=None) <= end_date  # type: ignore[union-attr]
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "account_id": self.account_id,
            "balances": {
                curr.code: bal.to_dict() for curr, bal in self.balances.items()
            },
            "positions": {pair: pos.to_dict() for pair, pos in self.positions.items()},
            "orders": [order.to_dict() for order in self.orders],
            "trades": [trade.to_dict() for trade in self.trades],
            "total_equity": str(self.total_equity.amount),
            "total_margin_used": str(self.total_margin_used.amount),
            "free_margin": str(self.free_margin.amount),
            "margin_level": str(self.margin_level.value),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        """Создание из словаря."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            name=data["name"],
            account_id=data["account_id"],
            balances={
                Currency(bal_data["currency"]): Balance.from_dict(bal_data)
                for bal_data in data["balances"].values()
            },
            positions={
                pair: Position.from_dict(pos_data)
                for pair, pos_data in data["positions"].items()
            },
            orders=[Order.from_dict(order_data) for order_data in data["orders"]],
            trades=[Trade.from_dict(trade_data) for trade_data in data["trades"]],
            total_equity=Money(amount=Decimal(data["total_equity"]), currency=Currency.USD),
            total_margin_used=Money(amount=Decimal(data["total_margin_used"]), currency=Currency.USD),
            free_margin=Money(amount=Decimal(data["free_margin"]), currency=Currency.USD),
            margin_level=Percentage(Decimal(data["margin_level"])),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
