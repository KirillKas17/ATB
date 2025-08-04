# -*- coding: utf-8 -*-
"""Промышленная реализация Value Object для цен."""

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

from .currency import Currency
from domain.type_definitions.base_types import (
    CurrencyCode,
    PriceLevel,
    NumericType,
)

from domain.type_definitions.value_object_types import (
    MAX_PRICE,
    MIN_PRICE,
    PRICE_PRECISION,
    CurrencyValueObject,
    NumericValueObject,
    TradingValueObject,
    ValidationResult,
    generate_cache_key,
    round_decimal,
    validate_numeric,
)


@dataclass(frozen=True)
class Price:
    """Value Object для цен."""

    amount: Decimal
    currency: Currency
    quote_currency: Optional[Currency] = None

    def __post_init__(self) -> None:
        if self.quote_currency is None:
            object.__setattr__(self, "quote_currency", Currency.USD)

        if self.amount < 0:
            raise ValueError("Price cannot be negative")

    def __add__(self, other: "Price") -> "Price":
        if not isinstance(other, Price):
            raise TypeError("Can only add Price with Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot add prices with different currencies")
        return Price(
            amount=self.amount + other.amount, 
            currency=self.currency, 
            quote_currency=self.quote_currency
        )

    def __sub__(self, other: "Price") -> "Price":
        if not isinstance(other, Price):
            raise TypeError("Can only subtract Price from Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot subtract prices with different currencies")
        return Price(
            amount=self.amount - other.amount,
            currency=self.currency,
            quote_currency=self.quote_currency
        )

    def __mul__(self, factor: Union[int, float, Decimal]) -> "Price":
        return Price(
            amount=self.amount * Decimal(str(factor)),
            currency=self.currency,
            quote_currency=self.quote_currency
        )

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> "Price":
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return Price(
            amount=self.amount / Decimal(str(divisor)),
            currency=self.currency,
            quote_currency=self.quote_currency
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Price):
            return False
        return (
            self.amount == other.amount
            and self.currency == other.currency
            and self.quote_currency == other.quote_currency
        )

    def __lt__(self, other: "Price") -> bool:
        if not isinstance(other, Price):
            raise TypeError("Can only compare Price with Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot compare prices with different currencies")
        return self.amount < other.amount

    def __le__(self, other: "Price") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Price") -> bool:
        return not self <= other

    def __ge__(self, other: "Price") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.amount, self.currency, self.quote_currency))

    def __str__(self) -> str:
        return f"{self.amount} {self.currency.code}"

    def __repr__(self) -> str:
        return f"Price({self.amount}, {self.currency.code})"

    @property
    def value(self) -> Decimal:
        """Возвращает числовое значение цены."""
        return self.amount

    def to_decimal(self) -> Decimal:
        """Возвращает цену как Decimal."""
        return self.amount

    def to_float(self) -> float:
        """Возвращает цену как float."""
        return float(self.amount)

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Сериализует цену в словарь."""
        return {
            "amount": str(self.amount),
            "currency": self.currency.code,
            "quote_currency": self.quote_currency.code if self.quote_currency else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Price":
        """Создает цену из словаря."""
        currency = Currency.from_string(data["currency"])
        quote_currency = None
        if data.get("quote_currency"):
            quote_currency = Currency.from_string(data["quote_currency"])
        
        if not currency:
            raise ValueError(f"Invalid currency: {data['currency']}")
            
        return cls(
            amount=Decimal(data["amount"]),
            currency=currency,
            quote_currency=quote_currency,
        )

    @classmethod
    def zero(cls, currency: Currency) -> "Price":
        """Создает нулевую цену."""
        return cls(amount=Decimal("0"), currency=currency)

    def is_zero(self) -> bool:
        """Проверяет, является ли цена нулевой."""
        return self.amount == Decimal("0")

    def is_positive(self) -> bool:
        """Проверяет, является ли цена положительной."""
        return self.amount > Decimal("0")

    def is_negative(self) -> bool:
        """Проверяет, является ли цена отрицательной."""
        return self.amount < Decimal("0")

    def abs(self) -> "Price":
        """Возвращает абсолютное значение цены."""
        return Price(
            amount=abs(self.amount),
            currency=self.currency,
            quote_currency=self.quote_currency
        )

    def round(self, precision: int = 8) -> "Price":
        """Округляет цену до заданной точности."""
        return Price(
            amount=round(self.amount, precision),
            currency=self.currency,
            quote_currency=self.quote_currency
        )

    def format(self, precision: Optional[int] = None) -> str:
        """Форматирует цену для отображения."""
        if precision is None:
            precision = 8
        return f"{self.amount:.{precision}f} {self.currency.code}"

    def convert_to(self, target_currency: Currency, rate: Decimal) -> "Price":
        """Конвертирует цену в другую валюту по курсу."""
        converted_amount = self.amount * rate
        return Price(
            amount=converted_amount,
            currency=target_currency,
            quote_currency=self.currency
        )

    def calculate_difference(self, other: "Price") -> "Price":
        """Вычисляет разность между ценами."""
        return self - other

    def calculate_percentage_change(self, other: "Price") -> Decimal:
        """Вычисляет процентное изменение относительно другой цены."""
        if other.amount == 0:
            raise ValueError("Cannot calculate percentage change from zero price")
        return ((self.amount - other.amount) / other.amount) * Decimal("100")

    def with_currency(self, currency: Currency) -> "Price":
        """Создает новую цену с той же суммой, но другой валютой."""
        return Price(
            amount=self.amount,
            currency=currency,
            quote_currency=self.quote_currency
        )

    def with_amount(self, amount: Decimal) -> "Price":
        """Создает новую цену с другой суммой, но той же валютой."""
        return Price(
            amount=amount,
            currency=self.currency,
            quote_currency=self.quote_currency
        )
