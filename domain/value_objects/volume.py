"""
Промышленная реализация Value Object для объемов с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union
from dataclasses import dataclass, field

from domain.type_definitions.base_types import (
    VolumeAmount,
    NumericType,
)

from domain.type_definitions.value_object_types import (
    MAX_VOLUME,
    MIN_VOLUME,
    VOLUME_PRECISION,
    CurrencyValueObject,
    NumericValueObject,
    TradingValueObject,
    ValidationResult,
    generate_cache_key,
    round_decimal,
    validate_numeric,
    ValueObject,
    ValueObjectDict,
)

from .currency import Currency
from .volume_config import VolumeConfig


@dataclass(frozen=True)
class Volume:
    """
    Промышленная реализация Value Object для объемов.

    Поддерживает:
    - Точные объемные операции с Decimal
    - Анализ ликвидности и рыночной активности
    - Торговые метрики и риск-менеджмент
    - Сериализацию/десериализацию
    """

    amount: Decimal
    currency: Currency

    MAX_VOLUME: Decimal = Decimal("999999999999.99999999")
    MIN_VOLUME: Decimal = Decimal("0")

    # Константы для анализа ликвидности
    HIGH_LIQUIDITY_THRESHOLD: Decimal = Decimal("1000000")  # 1M
    MEDIUM_LIQUIDITY_THRESHOLD: Decimal = Decimal("100000")  # 100K
    LOW_LIQUIDITY_THRESHOLD: Decimal = Decimal("10000")  # 10K

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.amount < 0:
            raise ValueError("Volume cannot be negative")
        if self.amount > self.MAX_VOLUME:
            raise ValueError(f"Volume cannot exceed {self.MAX_VOLUME}")

    def to_decimal(self) -> Decimal:
        """Возвращает объем как Decimal."""
        return self.amount

    def to_float(self) -> float:
        """Возвращает объем как float."""
        return float(self.amount)

    @property
    def value(self) -> Decimal:
        """Возвращает числовое значение объема."""
        return self.amount

    def __add__(self, other: "Volume") -> "Volume":
        """Сложение объемов."""
        if not isinstance(other, Volume):
            raise TypeError("Can only add Volume with Volume")
        if self.currency != other.currency:
            raise ValueError("Cannot add volumes with different currencies")
        return Volume(
            amount=self.amount + other.amount,
            currency=self.currency
        )

    def __sub__(self, other: "Volume") -> "Volume":
        """Вычитание объемов."""
        if not isinstance(other, Volume):
            raise TypeError("Can only subtract Volume from Volume")
        if self.currency != other.currency:
            raise ValueError("Cannot subtract volumes with different currencies")
        result_amount = self.amount - other.amount
        if result_amount < 0:
            raise ValueError("Resulting volume cannot be negative")
        return Volume(
            amount=result_amount,
            currency=self.currency
        )

    def __mul__(self, factor: Union[int, float, Decimal]) -> "Volume":
        """Умножение объема на число."""
        return Volume(
            amount=self.amount * Decimal(str(factor)),
            currency=self.currency
        )

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> "Volume":
        """Деление объема на число."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return Volume(
            amount=self.amount / Decimal(str(divisor)),
            currency=self.currency
        )

    def __eq__(self, other: Any) -> bool:
        """Сравнение объемов на равенство."""
        if not isinstance(other, Volume):
            return False
        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other: "Volume") -> bool:
        """Сравнение объемов."""
        if not isinstance(other, Volume):
            raise TypeError("Can only compare Volume with Volume")
        if self.currency != other.currency:
            raise ValueError("Cannot compare volumes with different currencies")
        return self.amount < other.amount

    def __le__(self, other: "Volume") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Volume") -> bool:
        return not self <= other

    def __ge__(self, other: "Volume") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.amount, self.currency))

    def __str__(self) -> str:
        return f"{self.amount} {self.currency.code}"

    def __repr__(self) -> str:
        return f"Volume({self.amount}, {self.currency.code})"

    @classmethod
    def zero(cls, currency: Currency) -> "Volume":
        """Создает нулевой объем."""
        return cls(amount=Decimal("0"), currency=currency)

    def is_zero(self) -> bool:
        """Проверяет, является ли объем нулевым."""
        return self.amount == Decimal("0")

    def is_positive(self) -> bool:
        """Проверяет, является ли объем положительным."""
        return self.amount > Decimal("0")

    def abs(self) -> "Volume":
        """Возвращает абсолютное значение объема."""
        return Volume(
            amount=abs(self.amount),
            currency=self.currency
        )

    def round(self, precision: int = 8) -> "Volume":
        """Округляет объем до заданной точности."""
        return Volume(
            amount=round(self.amount, precision),
            currency=self.currency
        )

    def to_dict(self) -> dict[str, str]:
        """Сериализует объем в словарь."""
        return {
            "amount": str(self.amount),
            "currency": self.currency.code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Volume":
        """Создает объем из словаря."""
        currency = Currency.from_string(data["currency"])
        if not currency:
            raise ValueError(f"Invalid currency: {data['currency']}")
        return cls(
            amount=Decimal(data["amount"]),
            currency=currency,
        )
