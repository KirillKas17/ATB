"""
Промышленная реализация Value Object для денежных сумм с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from decimal import Decimal
from typing import Any, ClassVar, Dict, Optional, Union, cast

from domain.types.base_types import (
    CurrencyCode,
    MoneyAmount,
    NumericType,
)

from domain.types.value_object_types import (
    MAX_MONEY_AMOUNT,
    MIN_MONEY_AMOUNT,
    MONEY_PRECISION,
    CurrencyValueObject,
    NumericValueObject,
    TradingValueObject,
    ValidationResult,
    generate_cache_key,
    round_decimal,
    validate_numeric,
)

from .currency import Currency
from .money_cache import MoneyCache
from .money_config import MoneyConfig
from .price import Price


class Money(CurrencyValueObject, NumericValueObject, TradingValueObject):
    """
    Промышленная реализация Value Object для денежных сумм.
    Поддерживает:
    - Точные арифметические операции с Decimal
    - Валютные конвертации и валидацию
    - Торговые расчеты и риск-менеджмент
    - Сериализацию/десериализацию
    - Кэширование и оптимизацию производительности
    """

    # Константы для валидации
    MAX_AMOUNT: Decimal = MAX_MONEY_AMOUNT
    MIN_AMOUNT: Decimal = MIN_MONEY_AMOUNT
    DEFAULT_SLIPPAGE: Decimal = Decimal("0.001")  # 0.1%
    MAX_SLIPPAGE: Decimal = Decimal("0.05")  # 5%
    # Константы для риск-менеджмента
    MAX_POSITION_SIZE: Decimal = Decimal("1000000")  # 1M
    MIN_POSITION_SIZE: Decimal = Decimal("0.01")  # 0.01
    _config: MoneyConfig

    def __init__(
        self,
        amount: NumericType,
        currency: Union[Currency, CurrencyCode],
        config: Optional[MoneyConfig] = None,
    ) -> None:
        """
        Инициализация денежной суммы.
        Args:
            amount: Сумма
            currency: Валюта
            config: Конфигурация
        """
        self._config = config or MoneyConfig()
        # Нормализация валюты
        if isinstance(currency, str):
            currency = Currency(CurrencyCode(currency))
        elif hasattr(currency, '__class__') and currency.__class__.__name__ == 'CurrencyCode':
            currency = Currency(currency)
        self._currency: Currency = currency
        # Нормализация и валидация суммы
        self._amount = self._normalize_amount(amount)
        # Валидация
        if not self.validate():
            raise ValueError("Invalid money")
        # Кэширование
        if self._config.cache_enabled:
            MoneyCache.set_cached(self)
        self._hash = self._calculate_hash()

    @property
    def amount(self) -> Decimal:
        """Возвращает сумму."""
        return self._amount

    @property
    def currency(self) -> CurrencyCode:
        """Возвращает валюту."""
        return CurrencyCode(self._currency.code)

    @property
    def value(self) -> MoneyAmount:
        """Возвращает значение value object."""
        return MoneyAmount(self._amount)

    @property
    def hash(self) -> str:
        """Возвращает хеш для кэширования."""
        return self._hash

    @property
    def code(self) -> CurrencyCode:
        """Возвращает код валюты."""
        return CurrencyCode(self._currency.code)

    def _normalize_amount(self, amount: NumericType) -> Decimal:
        """Нормализует сумму."""
        if isinstance(amount, (int, float)):
            amount = Decimal(str(amount))
        elif not isinstance(amount, Decimal):
            raise ValueError(f"Invalid amount type: {type(amount)}")
        return round_decimal(amount, self._config.precision, self._config.rounding_mode)

    def validate(self) -> bool:
        """Валидирует денежную сумму."""
        errors = []
        # Валидация суммы
        amount_validation = validate_numeric(
            self._amount,
            min_value=self.MIN_AMOUNT if self._config.validate_limits else None,
            max_value=self.MAX_AMOUNT if self._config.validate_limits else None,
        )
        errors.extend(amount_validation.errors)
        # Валидация валюты
        if not self._currency.is_active:
            errors.append(f"Currency {self._currency.code} is not active")
        # Дополнительные проверки
        if self._amount < 0 and not self._config.allow_negative:
            errors.append("Negative amounts not allowed")
        return len(errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует в словарь."""
        return {
            "type": "Money",
            "amount": str(self._amount),
            "currency": str(self._currency.code),
            "config": {
                "precision": self._config.precision,
                "rounding_mode": self._config.rounding_mode,
                "allow_negative": self._config.allow_negative,
                "validate_limits": self._config.validate_limits,
                "cache_enabled": self._config.cache_enabled,
            },
        }

    def __eq__(self, other: Any) -> bool:
        """Сравнение денежных сумм."""
        if not isinstance(other, Money):
            return False
        return self._amount == other._amount and self._currency == other._currency

    def __hash__(self) -> int:
        """Хеширование."""
        return hash((self._amount, self._currency))

    def __str__(self) -> str:
        """Строковое представление."""
        # Форматируем Decimal без лишних нулей
        amount_str = (
            str(self._amount).rstrip("0").rstrip(".")
            if "." in str(self._amount)
            else str(self._amount)
        )
        return f"{amount_str} {self._currency.code}"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"Money(amount={self._amount}, currency='{self._currency.code}')"

    def _calculate_hash(self) -> str:
        """Вычисляет хеш для кэширования."""
        data = f"{self._amount}:{self._currency.code}"
        return hashlib.md5(data.encode()).hexdigest()

    # Арифметические операции
    def __add__(self, other: "NumericValueObject") -> "Money":
        """Сложение денежных сумм."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot add Money and {type(other)}")
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot add different currencies: {self._currency.code} and {other._currency.code}"
            )
        return Money(self._amount + other._amount, self._currency, self._config)

    def __sub__(self, other: "NumericValueObject") -> "Money":
        """Вычитание денежных сумм."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot subtract {type(other)} from Money")
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot subtract different currencies: {self._currency.code} and {other._currency.code}"
            )
        return Money(self._amount - other._amount, self._currency, self._config)

    def __mul__(self, other: Union[Decimal, float, int]) -> "Money":
        """Умножение на число."""
        if not isinstance(other, (Decimal, float, int)):
            raise TypeError(f"Cannot multiply Money by {type(other)}")
        multiplier = Decimal(str(other))
        return Money(self._amount * multiplier, self._currency, self._config)

    def __truediv__(self, other: Union[Decimal, float, int]) -> "Money":
        """Деление на число."""
        if not isinstance(other, (Decimal, float, int)):
            raise TypeError(f"Cannot divide Money by {type(other)}")
        divisor = Decimal(str(other))
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Money(self._amount / divisor, self._currency, self._config)

    def __lt__(self, other: "NumericValueObject") -> bool:
        """Меньше."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money with {type(other)}")
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare different currencies: {self._currency.code} and {other._currency.code}"
            )
        return self._amount < other._amount

    def __le__(self, other: "NumericValueObject") -> bool:
        """Меньше или равно."""
        return self < other or self == other

    def __gt__(self, other: "NumericValueObject") -> bool:
        """Больше."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money with {type(other)}")
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare different currencies: {self._currency.code} and {other._currency.code}"
            )
        return self._amount > other._amount

    def __ge__(self, other: "NumericValueObject") -> bool:
        """Больше или равно."""
        return self > other or self == other

    # Валютные операции
    def convert_to(self, target_currency: CurrencyCode, rate: Decimal) -> "Money":
        """Конвертирует в другую валюту."""
        if rate <= 0:
            raise ValueError(f"Invalid exchange rate: {rate}")
        converted_amount = self._amount * rate
        target_currency_obj = Currency(target_currency)
        return Money(converted_amount, target_currency_obj, self._config)

    def get_exchange_rate(self, target_currency: CurrencyCode) -> Optional[Decimal]:
        """Возвращает курс обмена (если доступен)."""
        # Здесь можно добавить логику получения курса из внешних источников
        if str(self._currency.code) == str(target_currency):
            return Decimal("1.0")
        # Простые конвертации для стейблкоинов
        stablecoin_pairs = {
            (str(Currency.USDT.code), str(Currency.USDC.code)): Decimal("1.0"),
            (str(Currency.USDT.code), str(Currency.BUSD.code)): Decimal("1.0"),
            (str(Currency.USDC.code), str(Currency.BUSD.code)): Decimal("1.0"),
        }
        pair = (str(self._currency.code), str(target_currency))
        reverse_pair = (str(target_currency), str(self._currency.code))
        if pair in stablecoin_pairs:
            return stablecoin_pairs[pair]
        elif reverse_pair in stablecoin_pairs:
            return Decimal("1.0") / stablecoin_pairs[reverse_pair]
        return None

    # Торговые операции
    def calculate_slippage(self, target_price: "CurrencyValueObject") -> Decimal:
        """Рассчитывает проскальзывание относительно целевой цены."""
        if not hasattr(target_price, 'amount'):
            raise TypeError("Target price must have amount attribute")
        # Простое вычисление проскальзывания как разность цен
        target_amount = target_price.amount
        if not isinstance(target_amount, (int, float, Decimal)):
            raise TypeError("Target price amount must be numeric")
        target_amount_decimal = Decimal(str(target_amount))
        price_diff = abs(self._amount - target_amount_decimal)
        if target_amount_decimal == 0:
            return Decimal("0")
        return Decimal(str(price_diff / target_amount_decimal))

    def calculate_fee(self, fee_rate: Decimal) -> "Money":
        """Рассчитывает комиссию."""
        if fee_rate < 0 or fee_rate > 1:
            raise ValueError(f"Invalid fee rate: {fee_rate}")
        fee_amount = self._amount * fee_rate
        return Money(fee_amount, self._currency, self._config)

    def is_valid_for_trading(self) -> bool:
        """Проверяет пригодность для торговли."""
        if not self._currency.is_active:
            return False
        if self._amount < self.MIN_POSITION_SIZE:
            return False
        if self._amount > self.MAX_POSITION_SIZE:
            return False
        return True

    # Риск-менеджмент
    def get_position_risk_score(self) -> int:
        """Возвращает оценку риска позиции."""
        risk_score = 0
        # Базовый риск в зависимости от размера позиции
        if self._amount > self.MAX_POSITION_SIZE * Decimal("0.8"):
            risk_score += 3
        elif self._amount > self.MAX_POSITION_SIZE * Decimal("0.5"):
            risk_score += 2
        elif self._amount > self.MAX_POSITION_SIZE * Decimal("0.2"):
            risk_score += 1
        # Риск в зависимости от валюты
        if self._currency.is_stablecoin:
            risk_score -= 1
        elif self._currency.is_major_crypto:
            risk_score += 0
        else:
            risk_score += 2
        return max(0, min(5, risk_score))

    def get_margin_requirement(self, leverage: Decimal = Decimal("1")) -> "Money":
        """Возвращает требование к марже."""
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        margin_amount = self._amount / leverage
        return Money(margin_amount, self._currency, self._config)

    def get_stop_loss_amount(self, stop_loss_percentage: Decimal) -> "Money":
        """Рассчитывает сумму стоп-лосса."""
        if stop_loss_percentage <= 0 or stop_loss_percentage >= 1:
            raise ValueError(f"Invalid stop loss percentage: {stop_loss_percentage}")
        stop_loss_amount = self._amount * stop_loss_percentage
        return Money(stop_loss_amount, self._currency, self._config)

    def get_take_profit_amount(self, take_profit_percentage: Decimal) -> "Money":
        """Рассчитывает сумму тейк-профита."""
        if take_profit_percentage <= 0:
            raise ValueError(
                f"Invalid take profit percentage: {take_profit_percentage}"
            )
        take_profit_amount = self._amount * take_profit_percentage
        return Money(take_profit_amount, self._currency, self._config)

    def get_liquidity_score(self) -> float:
        """Возвращает оценку ликвидности."""
        # Простая оценка на основе валюты
        if self._currency.is_stablecoin:
            return 1.0
        elif self._currency.is_major_crypto:
            return 0.9
        elif self._currency.is_fiat:
            return 0.8
        else:
            return 0.5

    # Утилиты
    def is_zero(self) -> bool:
        """Проверяет, равна ли сумма нулю."""
        return self._amount == 0

    def is_positive(self) -> bool:
        """Проверяет, положительная ли сумма."""
        return self._amount > 0

    def is_negative(self) -> bool:
        """Проверяет, отрицательная ли сумма."""
        return self._amount < 0

    def abs(self) -> "Money":
        """Возвращает абсолютное значение."""
        return Money(abs(self._amount), self._currency, self._config)

    def round_to(self, precision: int) -> "Money":
        """Округляет до заданной точности."""
        rounded_amount = round_decimal(
            self._amount, precision, self._config.rounding_mode
        )
        return Money(rounded_amount, self._currency, self._config)

    def apply_percentage(self, percentage: Decimal) -> "Money":
        """Применяет процент к сумме."""
        if not isinstance(percentage, (Decimal, float, int)):
            raise TypeError("Percentage must be numeric")
        percentage_decimal = Decimal(str(percentage))
        new_amount = self._amount * (percentage_decimal / 100)
        return Money(new_amount, self._currency, self._config)

    def increase_by_percentage(self, percentage: Decimal) -> "Money":
        """Увеличивает сумму на указанный процент."""
        if not isinstance(percentage, (Decimal, float, int)):
            raise TypeError("Percentage must be numeric")
        percentage_decimal = Decimal(str(percentage))
        increase_amount = self._amount * (percentage_decimal / 100)
        new_amount = self._amount + increase_amount
        return Money(new_amount, self._currency, self._config)

    def decrease_by_percentage(self, percentage: Decimal) -> "Money":
        """Уменьшает сумму на указанный процент."""
        if not isinstance(percentage, (Decimal, float, int)):
            raise TypeError("Percentage must be numeric")
        percentage_decimal = Decimal(str(percentage))
        decrease_amount = self._amount * (percentage_decimal / 100)
        new_amount = self._amount - decrease_amount
        return Money(new_amount, self._currency, self._config)

    def percentage_of(self, total: "Money") -> Decimal:
        """Вычисляет процент от общей суммы."""
        if not isinstance(total, Money):
            raise TypeError("Total must be Money instance")
        if self._currency != total._currency:
            raise ValueError("Cannot calculate percentage for different currencies")
        if total._amount == 0:
            return Decimal("0")
        return (self._amount / total._amount) * 100

    # Статические методы
    @classmethod
    def zero(cls, currency: Union[Currency, CurrencyCode]) -> "Money":
        """Создает нулевую сумму."""
        return cls(Decimal("0"), currency)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Money":
        """Создает из словаря."""
        amount = Decimal(data["amount"])
        currency_code = data["currency"]
        currency = Currency.from_string(currency_code)
        if currency is None:
            raise ValueError(f"Unknown currency: {currency_code}")
        config_data = data.get("config", {})
        config = MoneyConfig(**config_data) if config_data else None
        return cls(amount, currency, config)

    @classmethod
    def get_cached(
        cls, amount: NumericType, currency: Union[Currency, CurrencyCode]
    ) -> Optional["Money"]:
        """Возвращает из кэша."""
        currency_obj = (
            currency if isinstance(currency, Currency) else Currency(currency)
        )
        return MoneyCache.get_cached(amount, currency_obj)

    @classmethod
    def clear_cache(cls) -> None:
        """Очищает кэш."""
        MoneyCache.clear_cache()

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Возвращает статистику кэша."""
        return MoneyCache.get_cache_stats()


# Фабричные функции
def create_money(
    amount: NumericType, currency: Union[str, Currency, CurrencyCode], **kwargs: Any
) -> Money:
    """Фабричная функция для создания Money."""
    if isinstance(currency, str):
        currency_obj = Currency.from_string(currency)
        if currency_obj is None:
            raise ValueError(f"Unknown currency: {currency}")
    elif isinstance(currency, Currency):
        currency_obj = currency
    else:
        currency_obj = Currency.from_string(str(currency))  # type: ignore[unreachable]
        if currency_obj is None:
            raise ValueError(f"Unknown currency: {currency}")
    config: Optional[MoneyConfig] = MoneyConfig(**kwargs) if kwargs else None
    money_obj = Money(amount, currency_obj, config)
    return money_obj


# Экспорт
__all__ = ["Money", "MoneyConfig", "create_money"]
