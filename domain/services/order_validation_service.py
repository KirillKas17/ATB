"""
Сервис валидации ордеров.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from domain.entities.order import OrderType, OrderSide
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


@dataclass
class ValidationResult:
    """Результат валидации."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]


@dataclass
class AccountData:
    """Данные аккаунта для валидации."""

    balance: Money
    available_balance: Money
    margin_used: Money
    open_positions: int
    max_positions: int
    leverage: Decimal
    risk_level: str


class OrderBusinessRuleValidator(ABC):
    """Абстрактный валидатор бизнес-правил ордеров."""

    @abstractmethod
    def validate_order_business_rules(
        self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]], account_data: AccountData
    ) -> ValidationResult:
        """Валидация бизнес-правил ордера."""


class DefaultOrderBusinessRuleValidator(OrderBusinessRuleValidator):
    """Реализация валидатора бизнес-правил ордеров."""

    def __init__(self) -> None:
        self.min_order_amount = Money(Decimal("10"), Currency.USDT)
        self.max_order_amount = Money(Decimal("100000"), Currency.USDT)
        self.max_positions_per_symbol = 5
        self.max_daily_orders = 100

    def validate_order_business_rules(
        self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]], account_data: AccountData
    ) -> ValidationResult:
        """Валидация бизнес-правил ордера."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Union[str, float, int, Dict[str, Union[str, int, float]]]] = {}
        try:
            # Валидация размера ордера
            order_amount = self._extract_order_amount(order_data)
            if order_amount:
                amount_validation = self._validate_order_amount(
                    order_amount, account_data
                )
                errors.extend(amount_validation.errors)
                warnings.extend(amount_validation.warnings)
                metadata.update(amount_validation.metadata)
            # Валидация лимитов позиций
            position_validation = self._validate_position_limits(
                order_data, account_data
            )
            errors.extend(position_validation.errors)
            warnings.extend(position_validation.warnings)
            metadata.update(position_validation.metadata)
            # Валидация рисков
            risk_validation = self._validate_risk_limits(order_data, account_data)
            errors.extend(risk_validation.errors)
            warnings.extend(risk_validation.warnings)
            metadata.update(risk_validation.metadata)
            # Валидация торговых часов
            time_validation = self._validate_trading_hours(order_data)
            errors.extend(time_validation.errors)
            warnings.extend(time_validation.warnings)
            metadata.update(time_validation.metadata)
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def _extract_order_amount(self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]) -> Optional[Money]:
        """Извлечение суммы ордера."""
        try:
            if "amount" in order_data:
                amount_data = order_data["amount"]
                if isinstance(amount_data, dict):
                    return Money(
                        Decimal(str(amount_data.get("value", 0))),
                        Currency.USDT,
                    )
                elif isinstance(amount_data, (int, float, str)):
                    return Money(Decimal(str(amount_data)), Currency.USDT)
            return None
        except Exception:
            return None

    def _validate_order_amount(
        self, amount: Money, account_data: AccountData
    ) -> ValidationResult:
        """Валидация размера ордера."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Union[str, float, int, Dict[str, Union[str, int, float]]]] = {}
        # Проверка минимального размера
        if amount.value < self.min_order_amount.value:
            errors.append(
                f"Order amount {amount} is below minimum {self.min_order_amount}"
            )
        # Проверка максимального размера
        if amount.value > self.max_order_amount.value:
            errors.append(
                f"Order amount {amount} exceeds maximum {self.max_order_amount}"
            )
        # Проверка доступного баланса
        if amount.value > account_data.available_balance.value:
            errors.append(
                f"Insufficient balance. Required: {amount}, Available: {account_data.available_balance}"
            )
        # Проверка использования маржи
        margin_usage = (account_data.margin_used.value / account_data.balance.value) * 100 if account_data.balance.value != 0 else 0.0
        if margin_usage > 80:
            warnings.append(f"High margin usage: {margin_usage:.1f}%")
        metadata["amount_validation"] = {
            "amount": str(amount),
            "min_amount": str(self.min_order_amount),
            "max_amount": str(self.max_order_amount),
            "available_balance": str(account_data.available_balance),
            "margin_usage_percent": float(margin_usage),
        }
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_position_limits(
        self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]], account_data: AccountData
    ) -> ValidationResult:
        """Валидация лимитов позиций."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Any] = {}
        # Проверка максимального количества позиций
        if account_data.open_positions >= account_data.max_positions:
            errors.append(
                f"Maximum positions limit reached: {account_data.open_positions}/{account_data.max_positions}"
            )
        # Проверка позиций по символу (если есть информация о символе)
        symbol = order_data.get("symbol", "")
        if symbol:
            # Здесь должна быть логика проверки позиций по конкретному символу
            # Пока оставляем заглушку
            pass
        metadata["position_validation"] = {
            "open_positions": account_data.open_positions,
            "max_positions": account_data.max_positions,
            "symbol": symbol,
        }
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_risk_limits(
        self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]], account_data: AccountData
    ) -> ValidationResult:
        """Валидация лимитов риска."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Union[str, float, int, Dict[str, Union[str, int, float]]]] = {}
        # Проверка уровня риска аккаунта
        if account_data.risk_level == "low":
            # Для низкого риска - дополнительные ограничения
            pass
        metadata["risk_validation"] = {
            "risk_level": account_data.risk_level,
        }
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_trading_hours(self, order_data: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]) -> ValidationResult:
        """Валидация торговых часов."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Union[str, float, int, Dict[str, Union[str, int, float]]]] = {}
        # Здесь может быть логика проверки торговых часов
        metadata["trading_hours_validation"] = {
            "status": "ok"
        }
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


class OrderDataValidator:
    """Валидатор данных ордера."""

    def __init__(self) -> None:
        self.required_fields = ["symbol", "side", "order_type", "quantity"]
        self.valid_order_types = [ot.value for ot in OrderType]
        self.valid_sides = [os.value for os in OrderSide]

    def validate_order_data(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Валидация данных ордера."""
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Any] = {}
        try:
            # Проверка обязательных полей
            for field in self.required_fields:
                if field not in order_data or order_data[field] is None:
                    errors.append(f"Required field '{field}' is missing or null")
            # Валидация типа ордера
            if "order_type" in order_data:
                order_type = order_data["order_type"]
                if order_type not in self.valid_order_types:
                    errors.append(
                        f"Invalid order type: {order_type}. Valid types: {self.valid_order_types}"
                    )
            # Валидация стороны ордера
            if "side" in order_data:
                side = order_data["side"]
                if side not in self.valid_sides:
                    errors.append(
                        f"Invalid order side: {side}. Valid sides: {self.valid_sides}"
                    )
            # Валидация количества
            if "quantity" in order_data:
                quantity = order_data["quantity"]
                if not self._is_valid_quantity(quantity):
                    errors.append(
                        f"Invalid quantity: {quantity}. Must be positive number"
                    )
            # Валидация цены для лимитных ордеров
            if (
                "order_type" in order_data
                and order_data["order_type"] == OrderType.LIMIT.value
            ):
                if "price" not in order_data or order_data["price"] is None:
                    errors.append("Price is required for limit orders")
                elif not self._is_valid_price(order_data["price"]):
                    errors.append(
                        f"Invalid price: {order_data['price']}. Must be positive number"
                    )
            # Валидация стоп-цены для стоп-ордеров
            if "order_type" in order_data and order_data["order_type"] in [
                OrderType.STOP.value,
                OrderType.STOP_LIMIT.value,
            ]:
                if "stop_price" not in order_data or order_data["stop_price"] is None:
                    errors.append("Stop price is required for stop orders")
                elif not self._is_valid_price(order_data["stop_price"]):
                    errors.append(
                        f"Invalid stop price: {order_data['stop_price']}. Must be positive number"
                    )
            metadata["data_validation"] = {
                "fields_checked": len(self.required_fields),
                "order_type_valid": "order_type" in order_data
                and order_data["order_type"] in self.valid_order_types,
                "side_valid": "side" in order_data
                and order_data["side"] in self.valid_sides,
            }
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Data validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def _is_valid_quantity(self, quantity: Any) -> bool:
        """Проверка валидности количества."""
        try:
            qty = Decimal(str(quantity))
            return qty > 0
        except (ValueError, TypeError):
            return False

    def _is_valid_price(self, price: Any) -> bool:
        """Проверка валидности цены."""
        try:
            prc = Decimal(str(price))
            return prc > 0
        except (ValueError, TypeError):
            return False
