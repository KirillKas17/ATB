"""
Промышленная реализация Value Object для объемов с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union

from domain.types.base_types import (
    VolumeAmount,
    NumericType,
)

from domain.types.value_object_types import (
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


class Volume(ValueObject):
    """
    Промышленная реализация Value Object для объемов.

    Поддерживает:
    - Точные объемные операции с Decimal
    - Анализ ликвидности и рыночной активности
    - Торговые метрики и риск-менеджмент
    - Сериализацию/десериализацию
    """

    MAX_VOLUME: Decimal = Decimal("999999999999.99999999")
    MIN_VOLUME: Decimal = Decimal("0")

    # Константы для анализа ликвидности
    HIGH_LIQUIDITY_THRESHOLD: Decimal = Decimal("1000000")  # 1M
    MEDIUM_LIQUIDITY_THRESHOLD: Decimal = Decimal("100000")  # 100K
    LOW_LIQUIDITY_THRESHOLD: Decimal = Decimal("10000")  # 10K

    # Константы для торговых операций
    MIN_TRADABLE_VOLUME: Decimal = Decimal("0.001")
    MAX_TRADABLE_VOLUME: Decimal = Decimal("1000000")

    def __init__(
        self,
        value: NumericType,
        currency: Optional[Currency] = None,
        config: Optional[VolumeConfig] = None,
    ) -> None:
        """
        Инициализация объема.

        Args:
            value: Объем
            currency: Валюта (опционально)
            config: Конфигурация объема (опционально)
        """
        self._config = config or VolumeConfig()
        self._currency = currency
        self._value: Decimal = self._normalize_value(value)
        self._validate_volume()

    def _normalize_value(self, value: NumericType) -> Decimal:
        """
        Нормализация значения объема.

        Args:
            value: Значение объема

        Returns:
            Нормализованное значение объема
        """
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, Decimal):
            return value
        else:
            raise ValueError(f"Invalid value type: {type(value)}")

    def _validate_volume(self) -> None:
        """Валидация объема."""
        if self._value.is_nan():
            raise ValueError("Volume cannot be NaN")
        if self._value.is_infinite():
            raise ValueError("Volume cannot be infinite")
        if self._value < 0:
            raise ValueError("Volume cannot be negative")
        if self._value > self.MAX_VOLUME:
            raise ValueError(f"Volume cannot exceed {self.MAX_VOLUME}")
        if self._value < self.MIN_VOLUME:
            raise ValueError(f"Volume cannot be less than {self.MIN_VOLUME}")

    @property
    def value(self) -> VolumeAmount:
        """Получение значения объема."""
        return VolumeAmount(self._value)

    @property
    def amount(self) -> Decimal:
        """Получение значения объема (алиас для совместимости)."""
        return self._value

    @property
    def hash(self) -> str:
        """Получение хеша объема."""
        return self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Вычисление хеша объема."""
        currency_str = str(self._currency.code) if self._currency else "None"
        data = f"{self._value}:{currency_str}"
        return hashlib.md5(data.encode()).hexdigest()

    def validate(self) -> bool:
        """Валидирует объем и возвращает True если валиден."""
        if self._value.is_nan():
            return False
        if self._value.is_infinite():
            return False
        if self._value < 0:
            return False
        if self._value > self.MAX_VOLUME:
            return False
        if self._value < self.MIN_VOLUME:
            return False
        return True

    def __eq__(self, other: Any) -> bool:
        """Проверка на равенство."""
        if not isinstance(other, Volume):
            return False
        return self._value == other._value and self._currency == other._currency

    def __hash__(self) -> int:
        """Хеш-код объема."""
        return hash((self._value, self._currency))

    @property
    def currency(self) -> Optional[Currency]:
        """Получение валюты."""
        return self._currency

    def __add__(self, other: Union["Volume", NumericType]) -> "Volume":
        """Сложение объемов."""
        if isinstance(other, Volume):
            other_value = other._value
            currency = self._currency or other._currency
        else:
            other_value = Decimal(str(other))
            currency = self._currency
        return Volume(self._value + other_value, currency, self._config)

    def __sub__(self, other: Union["Volume", NumericType]) -> "Volume":
        """Вычитание объемов."""
        if isinstance(other, Volume):
            other_value = other._value
        else:
            other_value = Decimal(str(other))
        result = self._value - other_value
        if result < 0:
            raise ValueError("Volume difference cannot be negative")
        return Volume(result, self._currency, self._config)

    def __mul__(self, multiplier: NumericType) -> "Volume":
        """Умножение объема."""
        return Volume(
            self._value * Decimal(str(multiplier)), self._currency, self._config
        )

    def __truediv__(
        self, divisor: Union["Volume", NumericType]
    ) -> Union["Volume", Decimal]:
        """Деление объема."""
        if isinstance(divisor, Volume):
            if divisor._value == 0:
                raise ValueError("Cannot divide by zero volume")
            return self._value / divisor._value

        divisor_decimal = Decimal(str(divisor))
        if divisor_decimal == 0:
            raise ValueError("Cannot divide by zero")

        return Volume(self._value / divisor_decimal, self._currency, self._config)

    def __lt__(self, other: "Volume") -> bool:
        """Сравнение меньше."""
        if not isinstance(other, Volume):
            raise TypeError("Can only compare Volume with Volume")
        return self._value < other._value

    def __le__(self, other: "Volume") -> bool:
        """Сравнение меньше или равно."""
        return self < other or self == other

    def __gt__(self, other: "Volume") -> bool:
        """Сравнение больше."""
        return not self <= other

    def __ge__(self, other: "Volume") -> bool:
        """Сравнение больше или равно."""
        return not self < other

    def round(self, places: int = VOLUME_PRECISION) -> "Volume":
        """Округление до указанного количества знаков."""
        rounded_value = self._value.quantize(
            Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP
        )
        return Volume(rounded_value, self._currency, self._config)

    def percentage_of(self, total: "Volume") -> Decimal:
        """Расчет процента от общего объема."""
        if total._value == 0:
            raise ValueError("Cannot calculate percentage of zero volume")

        return (self._value / total._value) * 100

    def min(self, other: "Volume") -> "Volume":
        """Минимальное значение из двух объемов."""
        return self if self <= other else other

    def max(self, other: "Volume") -> "Volume":
        """Максимальное значение из двух объемов."""
        return self if self >= other else other

    def is_within_range(self, min_volume: "Volume", max_volume: "Volume") -> bool:
        """Проверка, что объем в заданном диапазоне."""
        return min_volume <= self <= max_volume

    def apply_percentage(self, percentage: Decimal) -> "Volume":
        """Применение процента к объему."""
        return Volume(self._value * (percentage / 100), self._currency, self._config)

    def increase_by_percentage(self, percentage: Decimal) -> "Volume":
        """Увеличение объема на процент."""
        return Volume(
            self._value * (1 + percentage / 100), self._currency, self._config
        )

    def decrease_by_percentage(self, percentage: Decimal) -> "Volume":
        """Уменьшение объема на процент."""
        return Volume(
            self._value * (1 - percentage / 100), self._currency, self._config
        )

    # Методы анализа ликвидности
    def get_liquidity_level(self) -> str:
        """
        Получение уровня ликвидности.

        Returns:
            Уровень ликвидности: "HIGH", "MEDIUM", "LOW", "VERY_LOW"
        """
        if self._value >= self.HIGH_LIQUIDITY_THRESHOLD:
            return "HIGH"
        elif self._value >= self.MEDIUM_LIQUIDITY_THRESHOLD:
            return "MEDIUM"
        elif self._value >= self.LOW_LIQUIDITY_THRESHOLD:
            return "LOW"
        else:
            return "VERY_LOW"

    def is_liquid(self) -> bool:
        """
        Проверка ликвидности объема.

        Returns:
            True, если объем ликвиден.
        """
        return self._value >= self.LOW_LIQUIDITY_THRESHOLD

    def get_liquidity_score(self) -> float:
        """
        Получение скора ликвидности (0.0 - 1.0).

        Returns:
            Скор ликвидности.
        """
        if self._value >= self.HIGH_LIQUIDITY_THRESHOLD:
            return 1.0
        elif self._value >= self.MEDIUM_LIQUIDITY_THRESHOLD:
            return 0.7
        elif self._value >= self.LOW_LIQUIDITY_THRESHOLD:
            return 0.4
        else:
            return 0.1

    # Торговые методы
    def is_tradable(self) -> bool:
        """
        Проверка, что объем подходит для торговли.

        Returns:
            True, если объем подходит для торговли.
        """
        return self.MIN_TRADABLE_VOLUME <= self._value <= self.MAX_TRADABLE_VOLUME

    def get_trading_recommendation(self) -> str:
        """
        Получение торговой рекомендации на основе объема.

        Returns:
            Торговая рекомендация.
        """
        if not self.is_tradable():
            return "NOT_TRADABLE"

        liquidity_level = self.get_liquidity_level()

        if liquidity_level == "HIGH":
            return "SAFE_TO_TRADE"
        elif liquidity_level == "MEDIUM":
            return "CAUTION_ADVISED"
        elif liquidity_level == "LOW":
            return "HIGH_RISK"
        else:
            return "AVOID_TRADING"

    def calculate_market_impact(self, total_market_volume: "Volume") -> Decimal:
        """
        Расчет влияния на рынок.

        Args:
            total_market_volume: Общий объем рынка

        Returns:
            Процент влияния на рынок.
        """
        if total_market_volume._value == 0:
            raise ValueError("Total market volume cannot be zero")

        return (self._value / total_market_volume._value) * 100

    def is_significant_volume(
        self, threshold_percentage: Decimal = Decimal("0.01")
    ) -> bool:
        """
        Проверка, является ли объем значительным.

        Args:
            threshold_percentage: Пороговый процент

        Returns:
            True, если объем значительный.
        """
        return self._value > 0 and self._value >= threshold_percentage

    def to_dict(self) -> ValueObjectDict:
        """Преобразование в словарь."""
        data = {
            "value": str(self._value),
            "type": "Volume",
            "config": {
                "precision": self._config.precision,
                "rounding_mode": self._config.rounding_mode,
                "allow_negative": self._config.allow_negative,
                "validate_limits": self._config.validate_limits,
                "max_volume": str(self._config.max_volume),
                "min_volume": str(self._config.min_volume),
            },
        }
        if self._currency:
            data["currency"] = str(self._currency.code)
        return ValueObjectDict(**data)

    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "Volume":
        """Создание из словаря."""
        value = Decimal(data["value"])
        currency = None
        if "currency" in data:
            currency = Currency(data["currency"])
        config_data = data.get("config", {})
        config = VolumeConfig(
            precision=config_data.get("precision", VOLUME_PRECISION),
            rounding_mode=config_data.get("rounding_mode", "ROUND_HALF_UP"),
            allow_negative=config_data.get("allow_negative", False),
            validate_limits=config_data.get("validate_limits", True),
            max_volume=Decimal(config_data.get("max_volume", "999999999999.99999999")),
            min_volume=Decimal(config_data.get("min_volume", "0")),
        )
        return cls(value, currency, config)

    @classmethod
    def zero(cls, currency: Optional[Currency] = None) -> "Volume":
        """Создание нулевого объема."""
        return cls(Decimal("0"), currency)

    @classmethod
    def from_string(cls, value: str, currency: Optional[Currency] = None) -> "Volume":
        """Создание из строки."""
        try:
            clean_value = value.replace(",", "").strip()
            return cls(Decimal(clean_value), currency)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid volume string: {value}") from e

    @classmethod
    def from_float(cls, value: float, currency: Optional[Currency] = None) -> "Volume":
        """Создание из float."""
        return cls(Decimal(str(value)), currency)

    @classmethod
    def from_int(cls, value: int, currency: Optional[Currency] = None) -> "Volume":
        """Создание из int."""
        return cls(Decimal(str(value)), currency)

    def to_float(self) -> float:
        """Преобразование в float."""
        return float(self._value)

    def to_decimal(self) -> Decimal:
        """Преобразование в Decimal."""
        return self._value

    def copy(self) -> "Volume":
        """Создание копии."""
        return Volume(self._value, self._currency, self._config)

    def __str__(self) -> str:
        """Строковое представление."""
        currency_str = f" {self._currency.code}" if self._currency else ""
        return f"{self._value:.8f}{currency_str}"

    def __repr__(self) -> str:
        """Представление для отладки."""
        currency_str = f", {self._currency}" if self._currency else ""
        return f"Volume({self._value}{currency_str})"
