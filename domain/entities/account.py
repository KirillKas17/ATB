"""
Доменные сущности аккаунта и баланса.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from domain.types import AccountId


@runtime_checkable
class AccountProtocol(Protocol):
    """Протокол для аккаунта."""

    id: AccountId
    exchange_name: str
    balances: List["Balance"]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

    def get_balance(self, currency: str) -> Optional["Balance"]: ...
    def has_sufficient_balance(self, currency: str, amount: Decimal) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountProtocol": ...
@runtime_checkable
class BalanceProtocol(Protocol):
    """Протокол для баланса."""

    currency: str
    available: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BalanceProtocol": ...
@dataclass
class Account:
    """Аккаунт на бирже."""

    account_id: str = field(default_factory=lambda: str(uuid4()))
    exchange_name: str = ""
    balances: List["Balance"] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_balance(self, currency: str) -> Optional["Balance"]:
        """Получение баланса по валюте."""
        for balance in self.balances:
            if balance.currency == currency:
                return balance
        return None

    def has_sufficient_balance(self, currency: str, amount: Decimal) -> bool:
        """Проверка достаточности баланса."""
        balance = self.get_balance(currency)
        if balance is None:
            return False
        return balance.available >= amount

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "account_id": self.account_id,
            "exchange_name": self.exchange_name,
            "balances": [balance.to_dict() for balance in self.balances],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        """Создание из словаря."""
        return cls(
            account_id=data["account_id"],
            exchange_name=data["exchange_name"],
            balances=[Balance.from_dict(b) for b in data.get("balances", [])],
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Balance:
    """Баланс по валюте."""

    currency: str
    available: Decimal = Decimal("0")
    locked: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        """Общий баланс."""
        return self.available + self.locked

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "currency": self.currency,
            "available": str(self.available),
            "locked": str(self.locked),
            "total": str(self.total),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Balance":
        """Создание из словаря."""
        return cls(
            currency=data["currency"],
            available=Decimal(data["available"]),
            locked=Decimal(data["locked"]),
        )
