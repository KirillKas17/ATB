"""
Промышленная реализация Value Object для процентов с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union

from domain.types.base_types import (
    PercentageValue,
    NumericType,
)

from domain.types.value_object_types import (
    MAX_PERCENTAGE,
    MIN_PERCENTAGE,
    PERCENTAGE_PRECISION,
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

from .percentage_config import PercentageConfig


class Percentage(ValueObject):
    """
    Промышленная реализация Value Object для процентов.

    Поддерживает:
    - Точные процентные операции с Decimal
    - Расчет доходности и убытков
    - Анализ рисков и торговые метрики
    - Сериализацию/десериализацию
    """

    # Константы для валидации
    MAX_PERCENTAGE: Decimal = Decimal("10000")  # 10000%
    MIN_PERCENTAGE: Decimal = Decimal("-10000")  # -10000%

    # Константы для торгового анализа
    HIGH_RISK_THRESHOLD: Decimal = Decimal("50")  # 50%
    MEDIUM_RISK_THRESHOLD: Decimal = Decimal("20")  # 20%
    LOW_RISK_THRESHOLD: Decimal = Decimal("5")  # 5%

    # Константы для доходности
    EXCELLENT_RETURN: Decimal = Decimal("20")  # 20%
    GOOD_RETURN: Decimal = Decimal("10")  # 10%
    MODERATE_RETURN: Decimal = Decimal("5")  # 5%

    def __init__(
        self, value: NumericType, config: Optional[PercentageConfig] = None
    ) -> None:
        """
        Инициализация процента.

        Args:
            value: Процентное значение
            config: Конфигурация процента
        """
        self._config = config or PercentageConfig()
        self._value: Decimal = self._normalize_value(value)
        self._validate_percentage()

    def _normalize_value(self, value: NumericType) -> Decimal:
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, Decimal):
            return value
        else:
            raise ValueError(f"Invalid value type: {type(value)}")

    def _validate_percentage(self) -> None:
        """Валидация процента."""
        if self._value.is_nan():
            raise ValueError("Percentage cannot be NaN")
        if self._value.is_infinite():
            raise ValueError("Percentage cannot be infinite")
        if self._value > self._config.max_percentage:
            raise ValueError(f"Percentage cannot exceed {self._config.max_percentage}%")
        if self._value < self._config.min_percentage:
            raise ValueError(
                f"Percentage cannot be less than {self._config.min_percentage}%"
            )

    @property
    def value(self) -> PercentageValue:
        return PercentageValue(self._value)

    @property
    def hash(self) -> str:
        return self._calculate_hash()

    def _calculate_hash(self) -> str:
        data = f"{self._value}:{self._config.precision}"
        return hashlib.md5(data.encode()).hexdigest()

    def validate(self) -> bool:
        """Валидирует процент и возвращает True если валиден."""
        if self._value.is_nan():
            return False
        if self._value.is_infinite():
            return False
        if self._value > self._config.max_percentage:
            return False
        if self._value < self._config.min_percentage:
            return False
        return True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Percentage):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __str__(self) -> str:
        """Строковое представление."""
        return f"{self._value:.2f}%"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"Percentage({self._value})"

    def __add__(self, other: Union["Percentage", NumericType]) -> "Percentage":
        """Сложение процентов."""
        if isinstance(other, Percentage):
            other_value = other._value
        else:
            other_value = Decimal(str(other))

        return Percentage(self._value + other_value, self._config)

    def __sub__(self, other: Union["Percentage", NumericType]) -> "Percentage":
        """Вычитание процентов."""
        if isinstance(other, Percentage):
            other_value = other._value
        else:
            other_value = Decimal(str(other))

        return Percentage(self._value - other_value, self._config)

    def __mul__(self, multiplier: NumericType) -> "Percentage":
        """Умножение процента."""
        return Percentage(self._value * Decimal(str(multiplier)), self._config)

    def __truediv__(
        self, divisor: Union["Percentage", NumericType]
    ) -> Union["Percentage", Decimal]:
        """Деление процента."""
        if isinstance(divisor, Percentage):
            if divisor._value == 0:
                raise ValueError("Cannot divide by zero percentage")
            return self._value / divisor._value

        divisor_decimal = Decimal(str(divisor))
        if divisor_decimal == 0:
            raise ValueError("Cannot divide by zero")

        return Percentage(self._value / divisor_decimal, self._config)

    def __lt__(self, other: "Percentage") -> bool:
        """Сравнение меньше."""
        if not isinstance(other, Percentage):
            raise TypeError("Can only compare Percentage with Percentage")
        return self._value < other._value

    def __le__(self, other: "Percentage") -> bool:
        """Сравнение меньше или равно."""
        return self < other or self == other

    def __gt__(self, other: "Percentage") -> bool:
        """Сравнение больше."""
        return not self <= other

    def __ge__(self, other: "Percentage") -> bool:
        """Сравнение больше или равно."""
        return not self < other

    def round(self, places: int = PERCENTAGE_PRECISION) -> "Percentage":
        """Округление до указанного количества знаков."""
        rounded_value = self._value.quantize(
            Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP
        )
        return Percentage(rounded_value, self._config)

    def to_float(self) -> float:
        """Преобразование в float."""
        return float(self._value)

    def to_decimal(self) -> Decimal:
        """Преобразование в Decimal."""
        return self._value

    def to_fraction(self) -> Decimal:
        """Преобразование в долю (деление на 100)."""
        return self._value / 100

    def apply_to(self, value: NumericType) -> Decimal:
        """Применение процента к значению."""
        return Decimal(str(value)) * self.to_fraction()

    def increase_by(self, value: NumericType) -> Decimal:
        """Увеличение значения на процент."""
        base_value = Decimal(str(value))
        return base_value * (1 + self.to_fraction())

    def decrease_by(self, value: NumericType) -> Decimal:
        """Уменьшение значения на процент."""
        base_value = Decimal(str(value))
        return base_value * (1 - self.to_fraction())

    def is_within_range(
        self,
        min_value: NumericType,
        max_value: NumericType,
    ) -> bool:
        """Проверка, что процент в заданном диапазоне."""
        min_decimal = Decimal(str(min_value))
        max_decimal = Decimal(str(max_value))
        return min_decimal <= self._value <= max_decimal

    def min(self, other: "Percentage") -> "Percentage":
        """Минимальное значение из двух процентов."""
        return self if self <= other else other

    def max(self, other: "Percentage") -> "Percentage":
        """Максимальное значение из двух процентов."""
        return self if self >= other else other

    def compound_with(self, other: "Percentage") -> "Percentage":
        """Сложный процент с другим процентом."""
        compound_factor = (1 + self.to_fraction()) * (1 + other.to_fraction()) - 1
        return Percentage(compound_factor * 100, self._config)

    def annualize(self, period_days: int) -> "Percentage":
        """Годовой процент (365 дней)."""
        if period_days <= 0:
            raise ValueError("Period days must be positive")

        # Используем float для возведения в степень, затем конвертируем обратно
        annual_factor = (1 + float(self.to_fraction())) ** (365 / period_days) - 1
        return Percentage(Decimal(str(annual_factor)) * 100, self._config)

    # Методы анализа рисков
    def get_risk_level(self) -> str:
        """
        Получение уровня риска на основе процента.

        Returns:
            Уровень риска: "LOW", "MEDIUM", "HIGH", "VERY_HIGH"
        """
        abs_value = abs(self._value)

        if abs_value <= self.LOW_RISK_THRESHOLD:
            return "LOW"
        elif abs_value <= self.MEDIUM_RISK_THRESHOLD:
            return "MEDIUM"
        elif abs_value <= self.HIGH_RISK_THRESHOLD:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def is_high_risk(self) -> bool:
        """
        Проверка высокого риска.

        Returns:
            True, если процент указывает на высокий риск.
        """
        return abs(self._value) > self.HIGH_RISK_THRESHOLD

    def is_acceptable_risk(self, max_risk: Decimal = Decimal("20")) -> bool:
        """
        Проверка приемлемого риска.

        Args:
            max_risk: Максимальный приемлемый риск в процентах

        Returns:
            True, если риск приемлем.
        """
        return abs(self._value) <= max_risk

    # Методы анализа доходности
    def get_return_rating(self) -> str:
        """
        Получение рейтинга доходности.

        Returns:
            Рейтинг доходности: "EXCELLENT", "GOOD", "MODERATE", "POOR", "LOSS"
        """
        if self._value >= self.EXCELLENT_RETURN:
            return "EXCELLENT"
        elif self._value >= self.GOOD_RETURN:
            return "GOOD"
        elif self._value >= self.MODERATE_RETURN:
            return "MODERATE"
        elif self._value >= 0:
            return "POOR"
        else:
            return "LOSS"

    def is_profitable(self) -> bool:
        """
        Проверка прибыльности.

        Returns:
            True, если процент положительный.
        """
        return self._value > 0

    def is_significant_return(self, threshold: Decimal = Decimal("5")) -> bool:
        """
        Проверка значительной доходности.

        Args:
            threshold: Порог значимости в процентах

        Returns:
            True, если доходность значительна.
        """
        return self._value >= threshold

    # Методы для торгового анализа
    def calculate_compound_growth(self, periods: int) -> "Percentage":
        """
        Расчет сложного роста за несколько периодов.

        Args:
            periods: Количество периодов

        Returns:
            Общий рост за все периоды.
        """
        if periods <= 0:
            raise ValueError("Periods must be positive")

        growth_factor = (1 + self.to_fraction()) ** periods - 1
        return Percentage(Decimal(str(growth_factor)) * 100, self._config)

    def calculate_break_even_periods(self, target_return: "Percentage") -> int:
        """
        Расчет количества периодов для достижения целевой доходности.

        Args:
            target_return: Целевая доходность

        Returns:
            Количество периодов.
        """
        if self._value <= 0:
            raise ValueError("Growth rate must be positive")

        if target_return._value <= 0:
            return 0

        # Используем логарифм для расчета
        import math

        periods = math.log(1 + target_return.to_fraction()) / math.log(
            1 + self.to_fraction()
        )
        return max(1, int(periods))

    def get_trading_signal_strength(self) -> str:
        """
        Получение силы торгового сигнала на основе процента.

        Returns:
            Сила сигнала: "STRONG", "MODERATE", "WEAK"
        """
        abs_value = abs(self._value)

        if abs_value >= self.HIGH_RISK_THRESHOLD:
            return "STRONG"
        elif abs_value >= self.MEDIUM_RISK_THRESHOLD:
            return "MODERATE"
        else:
            return "WEAK"

    def to_dict(self) -> ValueObjectDict:
        """Преобразование в словарь."""
        return ValueObjectDict(
            value=str(self._value),
            type="Percentage",
            config={
                "precision": self._config.precision,
                "rounding_mode": self._config.rounding_mode,
                "allow_negative": self._config.allow_negative,
                "validate_limits": self._config.validate_limits,
                "max_percentage": str(self._config.max_percentage),
                "min_percentage": str(self._config.min_percentage),
            },
        )

    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "Percentage":
        """Создание из словаря."""
        value = Decimal(data["value"])
        config_data = data.get("config", {})
        config = PercentageConfig(
            precision=config_data.get("precision", PERCENTAGE_PRECISION),
            rounding_mode=config_data.get("rounding_mode", "ROUND_HALF_UP"),
            allow_negative=config_data.get("allow_negative", True),
            validate_limits=config_data.get("validate_limits", True),
            max_percentage=Decimal(config_data.get("max_percentage", "10000")),
            min_percentage=Decimal(config_data.get("min_percentage", "-10000")),
        )
        return cls(value, config)

    @classmethod
    def zero(cls) -> "Percentage":
        """Создание нулевого процента."""
        return cls(Decimal("0"))

    @classmethod
    def from_decimal(cls, decimal_value: Decimal) -> "Percentage":
        """Создание из Decimal (в долях)."""
        return cls(decimal_value)

    @classmethod
    def from_float(cls, float_value: float) -> "Percentage":
        """Создание из float (в долях)."""
        return cls(Decimal(str(float_value)))

    @classmethod
    def from_string(cls, value: str) -> "Percentage":
        """Создание из строки."""
        try:
            # Убираем символ % если есть
            clean_value = value.replace("%", "").strip()
            return cls(Decimal(clean_value))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid percentage string: {value}") from e

    @classmethod
    def from_fraction(cls, fraction: Decimal) -> "Percentage":
        """Создание из дроби (0.05 -> 5%)."""
        return cls(fraction * 100)

    def copy(self) -> "Percentage":
        """Создание копии."""
        return Percentage(self._value, self._config)
