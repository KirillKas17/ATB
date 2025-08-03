"""
Доменная сущность Portfolio с промышленной типизацией и бизнес-валидацией.
"""

import ast
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from domain.types import AmountValue, PortfolioId
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.timestamp import Timestamp


class PortfolioStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@runtime_checkable
class PortfolioProtocol(Protocol):
    def get_equity(self) -> AmountValue: ...
    def get_margin_ratio(self) -> Decimal: ...
    def is_active(self) -> bool: ...


@dataclass
class Portfolio:
    id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    name: str = ""
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    total_equity: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    free_margin: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    used_margin: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    risk_profile: RiskProfile = RiskProfile.MODERATE
    max_leverage: Decimal = Decimal("10")
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация и инициализация портфеля после создания."""
        if self.total_equity.amount < 0:
            raise ValueError("Total equity не может быть отрицательным")
        if self.available_balance.amount < 0:
            raise ValueError("Available balance не может быть отрицательным")
        if self.used_margin.amount < 0:
            raise ValueError("Used margin не может быть отрицательным")
        if self.max_leverage <= 0:
            raise ValueError("Max leverage должен быть положительным")
        if self.available_balance.amount > self.total_equity.amount:
            raise ValueError("Available balance не может превышать total equity")
        
        # Обновляем время последнего изменения
        self.updated_at = Timestamp.now()

    def get_equity(self) -> AmountValue:
        return AmountValue(self.total_equity.amount)

    def get_margin_ratio(self) -> Decimal:
        """
        Расчёт коэффициента использования маржи в процентах.
        
        Returns:
            Decimal: Процент использования маржи (0-100)
        """
        if self.total_equity.amount <= 0:
            return Decimal("100")  # Если капитал нулевой/отрицательный - критическая ситуация
        return (self.used_margin.amount / self.total_equity.amount) * 100

    @property
    def balance(self) -> Money:
        return self.total_equity

    @property
    def available_balance(self) -> Money:
        return self.free_margin

    @property
    def total_balance(self) -> Money:
        return self.total_equity

    @property
    def locked_balance(self) -> Money:
        return self.used_margin

    @property
    def unrealized_pnl(self) -> Money:
        return Money(Decimal("0"), self.total_equity.currency)

    @property
    def realized_pnl(self) -> Money:
        return Money(Decimal("0"), self.total_equity.currency)

    @property
    def total_pnl(self) -> Money:
        return Money(Decimal("0"), self.total_equity.currency)

    @property
    def margin_balance(self) -> Money:
        return self.total_equity

    @property
    def risk_level(self) -> str:
        return self.risk_profile.value

    @property
    def leverage(self) -> Decimal:
        return self.max_leverage

    @property
    def is_active(self) -> bool:
        return self.status == PortfolioStatus.ACTIVE

    @property
    def is_suspended(self) -> bool:
        return self.status == PortfolioStatus.SUSPENDED

    @property
    def is_closed(self) -> bool:
        return self.status == PortfolioStatus.CLOSED

    @property
    def available_margin(self) -> Money:
        return Money(self.free_margin.amount, self.free_margin.currency)

    def update_equity(self, new_equity: Money) -> None:
        self.total_equity = new_equity
        self.updated_at = Timestamp.now()

    def update_margin(self, free_margin: Money, used_margin: Money) -> None:
        self.free_margin = free_margin
        self.used_margin = used_margin
        self.updated_at = Timestamp.now()

    def suspend(self) -> None:
        self.status = PortfolioStatus.SUSPENDED
        self.updated_at = Timestamp.now()

    def activate(self) -> None:
        self.status = PortfolioStatus.ACTIVE
        self.updated_at = Timestamp.now()

    def close(self) -> None:
        self.status = PortfolioStatus.CLOSED
        self.updated_at = Timestamp.now()

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status.value,
            "total_equity": str(self.total_equity.amount),
            "free_margin": str(self.free_margin.amount),
            "used_margin": str(self.used_margin.amount),
            "risk_profile": self.risk_profile.value,
            "max_leverage": str(self.max_leverage),
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Portfolio":
        return cls(
            id=(
                PortfolioId(UUID(data["id"]))
                if data.get("id")
                else PortfolioId(uuid4())
            ),
            name=data.get("name", ""),
            status=PortfolioStatus(data["status"]),
            total_equity=Money(Decimal(data["total_equity"]), Currency.USD),
            free_margin=Money(Decimal(data["free_margin"]), Currency.USD),
            used_margin=Money(Decimal(data["used_margin"]), Currency.USD),
            risk_profile=RiskProfile(data["risk_profile"]),
            max_leverage=Decimal(data["max_leverage"]),
            created_at=Timestamp.from_iso(data["created_at"]),
            updated_at=Timestamp.from_iso(data["updated_at"]),
            metadata=ast.literal_eval(data.get("metadata", "{}")),
        )

    def __str__(self) -> str:
        return f"Portfolio({self.name}, equity={self.total_equity})"

    def __repr__(self) -> str:
        return (
            f"Portfolio(id={self.id}, name='{self.name}', "
            f"status={self.status.value}, equity={self.total_equity})"
        )
