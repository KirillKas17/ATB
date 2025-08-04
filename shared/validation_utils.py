"""
Утилиты валидации для устранения дублирования проверок.
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

from domain.type_definitions.common_types import (
    ValidationContext,
    ValidationResult,
    ValidationRule,
)


class ValidationUtils:
    """Утилиты для валидации данных."""

    # ============================================================================
    # БАЗОВЫЕ ПРОВЕРКИ
    # ============================================================================

    @staticmethod
    def is_not_none(value: Any) -> bool:
        """Проверка, что значение не None."""
        return value is not None

    @staticmethod
    def is_not_empty(value: Any) -> bool:
        """Проверка, что значение не пустое."""
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) > 0
        return True

    @staticmethod
    def is_valid_uuid(value: Any) -> bool:
        """Проверка валидности UUID."""
        if not isinstance(value, str):
            return False
        try:
            UUID(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_email(value: Any) -> bool:
        """Проверка валидности email."""
        if not isinstance(value, str):
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def is_valid_decimal(value: Any) -> bool:
        """Проверка валидности десятичного числа."""
        if isinstance(value, (int, float, Decimal)):
            return True
        if isinstance(value, str):
            try:
                Decimal(value)
                return True
            except (InvalidOperation, ValueError):
                return False
        return False

    @staticmethod
    def is_valid_date(value: Any) -> bool:
        """Проверка валидности даты."""
        if isinstance(value, (datetime, date)):
            return True
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                return True
            except ValueError:
                return False
        return False

    # ============================================================================
    # ПРОВЕРКИ ЧИСЕЛ
    # ============================================================================

    @staticmethod
    def is_positive(value: Any) -> bool:
        """Проверка, что число положительное."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            decimal_value = Decimal(str(value))
            return decimal_value > 0
        except (InvalidOperation, ValueError):
            return False

    @staticmethod
    def is_non_negative(value: Any) -> bool:
        """Проверка, что число неотрицательное."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            decimal_value = Decimal(str(value))
            return decimal_value >= 0
        except (InvalidOperation, ValueError):
            return False

    @staticmethod
    def is_in_range(
        value: Any,
        min_value: Optional[Union[int, float, Decimal]],
        max_value: Optional[Union[int, float, Decimal]],
    ) -> bool:
        """Проверка, что число в диапазоне."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            decimal_value = Decimal(str(value))
            
            # Проверка минимального значения
            if min_value is not None:
                min_decimal = Decimal(str(min_value))
                if decimal_value < min_decimal:
                    return False
            
            # Проверка максимального значения
            if max_value is not None:
                max_decimal = Decimal(str(max_value))
                if decimal_value > max_decimal:
                    return False
            
            return True
        except (InvalidOperation, ValueError):
            return False

    # ============================================================================
    # ПРОВЕРКИ СТРОК
    # ============================================================================

    @staticmethod
    def has_min_length(value: Any, min_length: int) -> bool:
        """Проверка минимальной длины строки."""
        if not isinstance(value, str):
            return False
        return len(value) >= min_length

    @staticmethod
    def has_max_length(value: Any, max_length: int) -> bool:
        """Проверка максимальной длины строки."""
        if not isinstance(value, str):
            return False
        return len(value) <= max_length

    @staticmethod
    def matches_pattern(value: Any, pattern: str) -> bool:
        """Проверка соответствия строки паттерну."""
        if not isinstance(value, str):
            return False
        return bool(re.match(pattern, value))

    @staticmethod
    def is_valid_symbol(value: Any) -> bool:
        """Проверка валидности торгового символа."""
        if not isinstance(value, str):
            return False
        # Паттерн для торговых пар: BTC/USD, ETH-USDT, etc.
        pattern = r"^[A-Z0-9]+[/\-][A-Z0-9]+$"
        return bool(re.match(pattern, value))

    # ============================================================================
    # ПРОВЕРКИ СПИСКОВ
    # ============================================================================

    @staticmethod
    def has_min_items(value: Any, min_items: int) -> bool:
        """Проверка минимального количества элементов."""
        if not isinstance(value, (list, tuple)):
            return False
        return len(value) >= min_items

    @staticmethod
    def has_max_items(value: Any, max_items: int) -> bool:
        """Проверка максимального количества элементов."""
        if not isinstance(value, (list, tuple)):
            return False
        return len(value) <= max_items

    @staticmethod
    def all_items_valid(value: Any, validator: Callable[[Any], bool]) -> bool:
        """Проверка, что все элементы списка валидны."""
        if not isinstance(value, (list, tuple)):
            return False
        return all(validator(item) for item in value)

    # ============================================================================
    # ПРОВЕРКИ СЛОВАРЕЙ
    # ============================================================================

    @staticmethod
    def has_required_keys(value: Any, required_keys: List[str]) -> bool:
        """Проверка наличия обязательных ключей."""
        if not isinstance(value, dict):
            return False
        return all(key in value for key in required_keys)

    @staticmethod
    def has_only_allowed_keys(value: Any, allowed_keys: List[str]) -> bool:
        """Проверка, что словарь содержит только разрешенные ключи."""
        if not isinstance(value, dict):
            return False
        return all(key in allowed_keys for key in value.keys())

    # ============================================================================
    # СПЕЦИФИЧНЫЕ ПРОВЕРКИ ДЛЯ ТОРГОВЛИ
    # ============================================================================

    @staticmethod
    def is_valid_order_side(value: Any) -> bool:
        """Проверка валидности стороны ордера."""
        if not isinstance(value, str):
            return False
        return value.upper() in ["BUY", "SELL", "LONG", "SHORT"]

    @staticmethod
    def is_valid_order_type(value: Any) -> bool:
        """Проверка валидности типа ордера."""
        if not isinstance(value, str):
            return False
        valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "TAKE_PROFIT"]
        return value.upper() in valid_types

    @staticmethod
    def is_valid_leverage(value: Any) -> bool:
        """Проверка валидности кредитного плеча."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            leverage = Decimal(str(value))
            return 1 <= leverage <= 1000  # Обычные лимиты
        except (InvalidOperation, ValueError):
            return False

    @staticmethod
    def is_valid_price(value: Any) -> bool:
        """Проверка валидности цены."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            price = Decimal(str(value))
            return price > 0
        except (InvalidOperation, ValueError):
            return False

    @staticmethod
    def is_valid_quantity(value: Any) -> bool:
        """Проверка валидности количества."""
        if not ValidationUtils.is_valid_decimal(value):
            return False
        try:
            quantity = Decimal(str(value))
            return quantity > 0
        except (InvalidOperation, ValueError):
            return False

    # ============================================================================
    # КОМПОЗИЦИОННЫЕ ПРОВЕРКИ
    # ============================================================================

    @staticmethod
    def validate_entity(entity: Any, rules: List[ValidationRule]) -> ValidationResult:
        """Валидация сущности по правилам."""
        errors: List[str] = []
        warnings: List[str] = []

        for rule in rules:
            field = rule["field"]
            rule_type = rule["rule_type"]
            parameters = rule["parameters"]
            message = rule["message"]

            # Получение значения поля
            if hasattr(entity, field):
                value = getattr(entity, field)
            elif isinstance(entity, dict) and field in entity:
                value = entity[field]
            else:
                errors.append(f"Field '{field}' not found")
                continue

            # Применение правила
            is_valid = ValidationUtils._apply_rule(value, rule_type, parameters)

            if not is_valid:
                errors.append(message)

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @staticmethod
    def _apply_rule(value: Any, rule_type: str, parameters: Dict[str, Any]) -> bool:
        """Применение правила валидации."""
        rule_map = {
            "not_none": ValidationUtils.is_not_none,
            "not_empty": ValidationUtils.is_not_empty,
            "uuid": ValidationUtils.is_valid_uuid,
            "email": ValidationUtils.is_valid_email,
            "decimal": ValidationUtils.is_valid_decimal,
            "date": ValidationUtils.is_valid_date,
            "positive": ValidationUtils.is_positive,
            "non_negative": ValidationUtils.is_non_negative,
            "in_range": ValidationUtils._apply_in_range_rule,
            "min_length": ValidationUtils._apply_min_length_rule,
            "max_length": ValidationUtils._apply_max_length_rule,
            "pattern": ValidationUtils._apply_pattern_rule,
            "symbol": ValidationUtils.is_valid_symbol,
            "min_items": ValidationUtils._apply_min_items_rule,
            "max_items": ValidationUtils._apply_max_items_rule,
            "required_keys": ValidationUtils._apply_required_keys_rule,
            "allowed_keys": ValidationUtils._apply_allowed_keys_rule,
            "order_side": ValidationUtils.is_valid_order_side,
            "order_type": ValidationUtils.is_valid_order_type,
            "leverage": ValidationUtils.is_valid_leverage,
            "price": ValidationUtils.is_valid_price,
            "quantity": ValidationUtils.is_valid_quantity,
        }

        validator = rule_map.get(rule_type)
        if validator is None:
            return False

        try:
            if rule_type in ["in_range", "min_length", "max_length", "pattern", "min_items", "max_items", "required_keys", "allowed_keys"]:
                if callable(validator):
                    result = validator(value, parameters)
                    return bool(result)
                return False
            else:
                if callable(validator):
                    result = validator(value)
                    return bool(result)
                return False
        except Exception:
            return False

    @staticmethod
    def _apply_in_range_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила диапазона."""
        min_val = parameters.get("min")
        max_val = parameters.get("max")
        return ValidationUtils.is_in_range(
            value,
            min_val if isinstance(min_val, (int, float, Decimal)) else None,
            max_val if isinstance(max_val, (int, float, Decimal)) else None
        )

    @staticmethod
    def _apply_min_length_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила минимальной длины."""
        min_length = parameters.get("min_length")
        return ValidationUtils.has_min_length(
            value, min_length if isinstance(min_length, int) else 0
        )

    @staticmethod
    def _apply_max_length_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила максимальной длины."""
        max_length = parameters.get("max_length")
        return ValidationUtils.has_max_length(
            value, max_length if isinstance(max_length, int) else 0
        )

    @staticmethod
    def _apply_pattern_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила паттерна."""
        pattern = parameters.get("pattern")
        return ValidationUtils.matches_pattern(
            value, pattern if isinstance(pattern, str) else ""
        )

    @staticmethod
    def _apply_min_items_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила минимального количества элементов."""
        min_items = parameters.get("min_items")
        return ValidationUtils.has_min_items(
            value, min_items if isinstance(min_items, int) else 0
        )

    @staticmethod
    def _apply_max_items_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила максимального количества элементов."""
        max_items = parameters.get("max_items")
        return ValidationUtils.has_max_items(
            value, max_items if isinstance(max_items, int) else 0
        )

    @staticmethod
    def _apply_required_keys_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила обязательных ключей."""
        keys = parameters.get("keys")
        if isinstance(keys, list):
            return ValidationUtils.has_required_keys(value, keys)
        return False

    @staticmethod
    def _apply_allowed_keys_rule(value: Any, parameters: Dict[str, Any]) -> bool:
        """Применение правила разрешенных ключей."""
        keys = parameters.get("keys")
        if isinstance(keys, list):
            return ValidationUtils.has_only_allowed_keys(value, keys)
        return False

    # ============================================================================
    # УТИЛИТЫ ДЛЯ СОЗДАНИЯ ПРАВИЛ
    # ============================================================================

    @staticmethod
    def create_rule(
        field: str,
        rule_type: str,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ValidationRule:
        """Создание правила валидации."""
        return ValidationRule(
            field=field,
            rule_type=rule_type,
            parameters=parameters or {},
            message=message,
        )

    @staticmethod
    def create_required_rule(field: str) -> ValidationRule:
        """Создание правила для обязательного поля."""
        return ValidationUtils.create_rule(
            field=field, rule_type="not_none", message=f"Field '{field}' is required"
        )

    @staticmethod
    def create_string_rule(
        field: str, min_length: int = 1, max_length: Optional[int] = None
    ) -> List[ValidationRule]:
        """Создание правил для строкового поля."""
        rules = [
            ValidationUtils.create_rule(
                field=field,
                rule_type="not_empty",
                message=f"Field '{field}' cannot be empty",
            ),
            ValidationUtils.create_rule(
                field=field,
                rule_type="min_length",
                message=f"Field '{field}' must be at least {min_length} characters long",
                parameters={"min_length": min_length},
            ),
        ]

        if max_length is not None:
            rules.append(
                ValidationUtils.create_rule(
                    field=field,
                    rule_type="max_length",
                    message=f"Field '{field}' must be at most {max_length} characters long",
                    parameters={"max_length": max_length},
                )
            )

        return rules

    @staticmethod
    def create_numeric_rule(
        field: str,
        min_value: Optional[Union[int, float, Decimal]] = None,
        max_value: Optional[Union[int, float, Decimal]] = None,
        allow_zero: bool = True,
    ) -> List[ValidationRule]:
        """Создание правил для числового поля."""
        rules = [
            ValidationUtils.create_rule(
                field=field,
                rule_type="decimal",
                message=f"Field '{field}' must be a valid number",
            )
        ]

        if min_value is not None:
            rule_type = "non_negative" if min_value == 0 and allow_zero else "positive"
            message = f"Field '{field}' must be {'non-negative' if min_value == 0 and allow_zero else 'positive'}"
            rules.append(
                ValidationUtils.create_rule(
                    field=field, rule_type=rule_type, message=message
                )
            )

        if min_value is not None and max_value is not None:
            rules.append(
                ValidationUtils.create_rule(
                    field=field,
                    rule_type="in_range",
                    message=f"Field '{field}' must be between {min_value} and {max_value}",
                    parameters={"min": min_value, "max": max_value},
                )
            )

        return rules
