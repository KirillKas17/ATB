"""
Базовый класс для use case'ов.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Generic, List, Optional, TypeVar

from shared.exceptions import ValidationError

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


@dataclass
class UseCaseResult(Generic[ResponseT]):
    """Результат выполнения use case."""

    success: bool
    data: Optional[ResponseT] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def success_result(
        cls,
        data: ResponseT,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UseCaseResult[ResponseT]":
        """Создание успешного результата."""
        return cls(
            success=True, data=data, warnings=warnings or [], metadata=metadata or {}
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UseCaseResult[ResponseT]":
        """Создание результата с ошибкой."""
        return cls(
            success=False, error=error, warnings=warnings or [], metadata=metadata or {}
        )


class BaseUseCase(ABC, Generic[RequestT, ResponseT]):
    """Базовый класс для всех use case'ов."""

    def __init__(self) -> None:
        self._validation_rules: Dict[str, Any] = {}
        self._business_rules: Dict[str, Any] = {}
        self._audit_log: List[Dict[str, Any]] = []

    @abstractmethod
    def execute(self, request: RequestT) -> UseCaseResult[ResponseT]:
        """Выполнение use case."""
        pass

    def validate_request(self, request: RequestT) -> List[str]:
        """Валидация запроса."""
        errors = []
        # Базовая валидация
        basic_errors = self._validate_basic_request(request)
        errors.extend(basic_errors)
        # Валидация по правилам
        rule_errors = self._validate_by_rules(request)
        errors.extend(rule_errors)
        # Бизнес-валидация
        business_errors = self._validate_business_rules(request)
        errors.extend(business_errors)
        return errors

    def add_validation_rule(self, field: str, rule: Any) -> None:
        """Добавление правила валидации."""
        self._validation_rules[field] = rule

    def add_business_rule(self, rule_name: str, rule: Any) -> None:
        """Добавление бизнес-правила."""
        self._business_rules[rule_name] = rule

    def log_audit_event(
        self, event_type: str, details: Dict[str, Any], user_id: Optional[str] = None
    ) -> None:
        """Логирование аудиторского события."""
        audit_event = {
            "timestamp": self._get_current_timestamp(),
            "event_type": event_type,
            "details": details,
            "user_id": user_id,
            "use_case": self.__class__.__name__,
        }
        self._audit_log.append(audit_event)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Получение аудиторского лога."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Очистка аудиторского лога."""
        self._audit_log.clear()

    def execute_with_validation(
        self, request: RequestT, user_id: Optional[str] = None
    ) -> UseCaseResult[ResponseT]:
        """Выполнение с валидацией."""
        try:
            # Валидация
            validation_errors = self.validate_request(request)
            if validation_errors:
                return UseCaseResult.error_result(
                    f"Validation failed: {', '.join(validation_errors)}"
                )
            # Логирование начала выполнения
            self.log_audit_event(
                "use_case_started",
                {"request": self._serialize_request(request)},
                user_id,
            )
            # Выполнение
            result = self.execute(request)
            # Логирование результата
            self.log_audit_event(
                "use_case_completed",
                {
                    "success": result.success,
                    "has_data": result.data is not None,
                    "error": result.error,
                },
                user_id,
            )
            return result
        except Exception as e:
            # Логирование ошибки
            self.log_audit_event(
                "use_case_error",
                {"error": str(e), "error_type": type(e).__name__},
                user_id,
            )
            return UseCaseResult.error_result(f"Use case execution failed: {str(e)}")

    def execute_batch(
        self, requests: List[RequestT], user_id: Optional[str] = None
    ) -> List[UseCaseResult[ResponseT]]:
        """Пакетное выполнение use case."""
        results = []
        for i, request in enumerate(requests):
            try:
                result = self.execute_with_validation(request, user_id)
                results.append(result)
            except Exception as e:
                results.append(
                    UseCaseResult.error_result(f"Batch item {i+1} failed: {str(e)}")
                )
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики use case."""
        total_executions = len(self._audit_log)
        successful_executions = sum(
            1
            for event in self._audit_log
            if event["event_type"] == "use_case_completed"
            and event["details"]["success"]
        )
        failed_executions = total_executions - successful_executions
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (
                (successful_executions / total_executions * 100)
                if total_executions > 0
                else 0
            ),
            "use_case_name": self.__class__.__name__,
        }

    def _validate_basic_request(self, request: RequestT) -> List[str]:
        """Базовая валидация запроса."""
        errors = []
        if request is None:
            errors.append("Request cannot be null")
            return errors
        # Проверка обязательных полей
        if hasattr(request, "__dataclass_fields__"):
            for field_name, field_info in request.__dataclass_fields__.items():
                if not field_info.default and not field_info.default_factory:
                    field_value = getattr(request, field_name, None)
                    if field_value is None:
                        errors.append(f"Required field '{field_name}' is missing")
        return errors

    def _validate_by_rules(self, request: RequestT) -> List[str]:
        """Валидация по правилам."""
        errors = []
        for field, rule in self._validation_rules.items():
            if hasattr(request, field):
                field_value = getattr(request, field)
                try:
                    if not rule(field_value):
                        errors.append(f"Field '{field}' failed validation rule")
                except Exception as e:
                    errors.append(
                        f"Validation rule for field '{field}' failed: {str(e)}"
                    )
        return errors

    def _validate_business_rules(self, request: RequestT) -> List[str]:
        """Валидация бизнес-правил."""
        errors = []
        for rule_name, rule in self._business_rules.items():
            try:
                if not rule(request):
                    errors.append(f"Business rule '{rule_name}' failed")
            except Exception as e:
                errors.append(f"Business rule '{rule_name}' failed: {str(e)}")
        return errors

    def _serialize_request(self, request: RequestT) -> Dict[str, Any]:
        """Сериализация запроса для логирования."""
        if hasattr(request, "__dict__"):
            return request.__dict__.copy()
        elif hasattr(request, "to_dict"):
            return request.to_dict()
        else:
            return {"request_type": type(request).__name__}

    def _get_current_timestamp(self) -> str:
        """Получение текущего времени."""
        from datetime import datetime

        return datetime.utcnow().isoformat()

    def _validate_decimal_range(
        self, value: Decimal, min_value: Decimal, max_value: Decimal, field_name: str
    ) -> bool:
        """Валидация диапазона Decimal значений."""
        if value < min_value or value > max_value:
            raise ValidationError(
                f"{field_name} must be between {min_value} and {max_value}, got {value}",
                field_name=field_name,
                field_value=value,
                validation_rules={"min": min_value, "max": max_value}
            )
        return True

    def _validate_string_length(
        self, value: str, min_length: int, max_length: int, field_name: str
    ) -> bool:
        """Валидация длины строки."""
        if len(value) < min_length or len(value) > max_length:
            raise ValidationError(
                f"{field_name} length must be between {min_length} and {max_length}, got {len(value)}",
                field_name=field_name,
                field_value=value,
                validation_rules={"min_length": min_length, "max_length": max_length}
            )
        return True

    def _validate_list_not_empty(self, value: List[Any], field_name: str) -> bool:
        """Валидация непустого списка."""
        if not value:
            raise ValidationError(
                f"{field_name} cannot be empty",
                field_name=field_name,
                field_value=value,
                validation_rules={"required": True}
            )
        return True

    def _safe_divide(
        self, numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
    ) -> Decimal:
        """Безопасное деление с дефолтным значением."""
        if denominator == 0:
            return default
        return numerator / denominator

    def _calculate_percentage(self, part: Decimal, total: Decimal) -> Decimal:
        """Расчет процента."""
        if total == 0:
            return Decimal("0")
        return (part / total) * Decimal("100")

    def _round_decimal(self, value: Decimal, places: int = 8) -> Decimal:
        """Округление Decimal с заданной точностью."""
        return round(value, places)

    def _format_money(self, amount: Decimal, currency: str = "USDT") -> str:
        """Форматирование денежной суммы."""
        return f"{amount:.8f} {currency}"

    def _generate_id(self) -> str:
        """Генерация уникального ID."""
        from uuid import uuid4

        return str(uuid4())

    def _is_valid_uuid(self, value: str) -> bool:
        """Проверка валидности UUID."""
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_pattern.match(value))
