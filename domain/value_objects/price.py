# -*- coding: utf-8 -*-
"""Промышленная реализация Value Object для цен."""

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

from .currency import Currency
from domain.types.base_types import (
    CurrencyCode,
    PriceLevel,
    NumericType,
)

from domain.types.value_object_types import (
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

    value: Decimal
    currency: Currency
    quote_currency: Optional[Currency] = None

    def __post_init__(self) -> None:
        if self.quote_currency is None:
            object.__setattr__(self, "quote_currency", Currency.USD)

        if self.value < 0:
            raise ValueError("Price cannot be negative")

    def __add__(self, other: "Price") -> "Price":
        if not isinstance(other, Price):
            raise TypeError("Can only add Price with Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot add prices with different currencies")
        return Price(self.value + other.value, self.currency, self.quote_currency)

    def __sub__(self, other: "Price") -> "Price":
        if not isinstance(other, Price):
            raise TypeError("Can only subtract Price from Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot subtract prices with different currencies")
        return Price(self.value - other.value, self.currency, self.quote_currency)

    def __mul__(self, other: Union[Decimal, float, int]) -> "Price":
        if not isinstance(other, (Decimal, float, int)):
            raise TypeError("Can only multiply Price by numeric value")
        return Price(
            self.value * Decimal(str(other)), self.currency, self.quote_currency
        )

    def __truediv__(self, other: Union[Decimal, float, int]) -> "Price":
        if not isinstance(other, (Decimal, float, int)):
            raise TypeError("Can only divide Price by numeric value")
        if Decimal(str(other)) == 0:
            raise ValueError("Cannot divide by zero")
        return Price(
            self.value / Decimal(str(other)), self.currency, self.quote_currency
        )

    def __lt__(self, other: "Price") -> bool:
        if not isinstance(other, Price):
            raise TypeError("Can only compare Price with Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot compare prices with different currencies")
        return self.value < other.value

    def __le__(self, other: "Price") -> bool:
        return self < other or self == other

    def __gt__(self, other: "Price") -> bool:
        if not isinstance(other, Price):
            raise TypeError("Can only compare Price with Price")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Cannot compare prices with different currencies")
        return self.value > other.value

    def __ge__(self, other: "Price") -> bool:
        return self > other or self == other

    def __str__(self) -> str:
        quote_code = (
            self.quote_currency.code if self.quote_currency is not None else "?"
        )
        return f"{self.value} {self.currency.code}/{quote_code}"

    def __repr__(self) -> str:
        quote_code = (
            self.quote_currency.code if self.quote_currency is not None else "?"
        )
        return f"Price({self.value}, {self.currency.code}, {quote_code})"

    def percentage_change(self, other: "Price") -> Decimal:
        """Вычисляет процентное изменение относительно другой цены."""
        if not isinstance(other, Price):
            raise TypeError("Other must be Price instance")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Prices must have same currency pair")
        if other.value == 0:
            return Decimal("0")
        return ((self.value - other.value) / other.value) * 100

    def spread(self, other: "Price") -> Decimal:
        """Вычисляет спред относительно другой цены."""
        if not isinstance(other, Price):
            raise TypeError("Other must be Price instance")
        if (
            self.currency != other.currency
            or self.quote_currency != other.quote_currency
        ):
            raise ValueError("Prices must have same currency pair")
        return abs(self.value - other.value)

    def slippage(self, target_price: "Price") -> Decimal:
        """Вычисляет проскальзывание относительно целевой цены."""
        if not isinstance(target_price, Price):
            raise TypeError("Target price must be Price instance")
        if (
            self.currency != target_price.currency
            or self.quote_currency != target_price.quote_currency
        ):
            raise ValueError("Prices must have same currency pair")
        if Decimal(target_price.value) == Decimal("0"):
            return Decimal("0")
        return abs(self.value - target_price.value) / Decimal(target_price.value)

    @property
    def amount(self) -> Decimal:
        """Возвращает значение цены."""
        return self.value

    def to_decimal(self) -> Decimal:
        """Возвращает значение цены как Decimal."""
        return self.value

    def percentage_change_from(self, other: "Price") -> Decimal:
        """Вычисляет процентное изменение относительно другой цены."""
        return self.percentage_change(other)

    def spread_with(self, other: "Price") -> "Price":
        """Вычисляет спред с другой ценой и возвращает Price объект."""
        spread_value = self.spread(other)
        return Price(spread_value, self.currency, self.quote_currency)

    def apply_slippage(self, slippage_percent: Decimal) -> Tuple["Price", "Price"]:
        """Применяет проскальзывание и возвращает bid/ask цены."""
        slippage_decimal = slippage_percent / 100
        bid_price = Price(
            self.value * (1 - slippage_decimal), self.currency, self.quote_currency
        )
        ask_price = Price(
            self.value * (1 + slippage_decimal), self.currency, self.quote_currency
        )
        return bid_price, ask_price

    def to_dict(self) -> Dict[str, Any]:
        """Создает словарь для сериализации."""
        return {
            "value": str(self.value),
            "currency": self.currency.code,
            "quote_currency": (
                self.quote_currency.code if self.quote_currency is not None else None
            ),
            "type": "Price",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Price":
        """Создает из словаря."""
        value = Decimal(data["value"])
        currency = Currency.from_string(data["currency"])
        if currency is None:
            raise ValueError(f"Unknown currency: {data['currency']}")
        quote_currency = Currency.from_string(data["quote_currency"])
        if quote_currency is None:
            quote_currency = Currency.USD
        return cls(value, currency, quote_currency)

    @property
    def hash(self) -> str:
        """Возвращает хеш цены для кэширования."""
        data = f"{self.value}:{self.currency.code}:{self.quote_currency.code if self.quote_currency else 'USD'}"
        return hashlib.md5(data.encode()).hexdigest()

    def validate(self) -> bool:
        """Валидирует цену."""
        return self.value >= 0 and self.currency is not None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Price):
            return (
                self.value == other.value
                and self.currency == other.currency
                and self.quote_currency == other.quote_currency
            )
        return False

    def __hash__(self) -> int:
        return hash((self.value, self.currency, self.quote_currency))
