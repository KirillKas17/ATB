"""
Unit тесты для base_exceptions.py.

Покрывает:
- BaseDomainException - базовое исключение домена
- ValidationError - ошибки валидации
- BusinessRuleViolationError - нарушения бизнес-правил
- EntityNotFoundError - сущности не найдены
- ConfigurationError - ошибки конфигурации
- Вспомогательные функции
"""

import pytest
from datetime import datetime
from typing import Dict, Any, Optional

from domain.exceptions.base_exceptions import (
    BaseDomainException,
    ValidationError,
    BusinessRuleViolationError,
    EntityNotFoundError,
    ConfigurationError,
    _format_error_message,
    _get_error_context,
)


class TestBaseDomainException:
    """Тесты для BaseDomainException."""

    def test_base_exception_creation(self):
        """Тест создания базового исключения."""
        message = "Тестовое сообщение об ошибке"
        exception = BaseDomainException(message)

        assert str(exception) == message
        assert exception.message == message
        assert exception.timestamp is not None
        assert isinstance(exception.timestamp, datetime)

    def test_base_exception_with_context(self):
        """Тест создания исключения с контекстом."""
        message = "Ошибка валидации"
        context = {"field": "email", "value": "invalid@", "rule": "email_format"}

        exception = BaseDomainException(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert "field" in exception.context
        assert exception.context["field"] == "email"

    def test_base_exception_with_cause(self):
        """Тест создания исключения с причиной."""
        original_error = ValueError("Оригинальная ошибка")
        message = "Ошибка обработки данных"

        exception = BaseDomainException(message, cause=original_error)

        assert exception.message == message
        assert exception.cause == original_error
        assert str(original_error) in str(exception)

    def test_base_exception_repr(self):
        """Тест строкового представления исключения."""
        message = "Тестовое исключение"
        exception = BaseDomainException(message)

        repr_str = repr(exception)
        assert "BaseDomainException" in repr_str
        assert message in repr_str

    def test_base_exception_to_dict(self):
        """Тест преобразования исключения в словарь."""
        message = "Ошибка конфигурации"
        context = {"config_file": "settings.json", "section": "database"}
        cause = FileNotFoundError("Файл не найден")

        exception = BaseDomainException(message, context=context, cause=cause)
        exception_dict = exception.to_dict()

        assert exception_dict["message"] == message
        assert exception_dict["context"] == context
        assert exception_dict["cause"] == str(cause)
        assert "timestamp" in exception_dict
        assert "exception_type" in exception_dict

    def test_base_exception_inheritance(self):
        """Тест наследования от Exception."""
        exception = BaseDomainException("Тест")

        assert isinstance(exception, Exception)
        assert isinstance(exception, BaseDomainException)


class TestValidationError:
    """Тесты для ValidationError."""

    def test_validation_error_creation(self):
        """Тест создания ошибки валидации."""
        field = "email"
        value = "invalid_email"
        rule = "email_format"

        error = ValidationError(field, value, rule)

        assert error.field == field
        assert error.value == value
        assert error.rule == rule
        assert "email" in error.message.lower()

    def test_validation_error_with_custom_message(self):
        """Тест создания ошибки валидации с кастомным сообщением."""
        field = "password"
        value = "123"
        rule = "min_length"
        custom_message = "Пароль должен содержать минимум 8 символов"

        error = ValidationError(field, value, rule, message=custom_message)

        assert error.message == custom_message
        assert error.field == field
        assert error.value == value
        assert error.rule == rule

    def test_validation_error_context(self):
        """Тест контекста ошибки валидации."""
        field = "age"
        value = -5
        rule = "positive_integer"

        error = ValidationError(field, value, rule)

        assert error.context["field"] == field
        assert error.context["value"] == value
        assert error.context["rule"] == rule

    def test_validation_error_to_dict(self):
        """Тест преобразования ошибки валидации в словарь."""
        field = "username"
        value = ""
        rule = "non_empty"

        error = ValidationError(field, value, rule)
        error_dict = error.to_dict()

        assert error_dict["field"] == field
        assert error_dict["value"] == value
        assert error_dict["rule"] == rule
        assert error_dict["exception_type"] == "ValidationError"


class TestBusinessRuleViolationError:
    """Тесты для BusinessRuleViolationError."""

    def test_business_rule_violation_creation(self):
        """Тест создания ошибки нарушения бизнес-правила."""
        rule_name = "insufficient_balance"
        details = "Недостаточно средств для совершения операции"

        error = BusinessRuleViolationError(rule_name, details)

        assert error.rule_name == rule_name
        assert error.details == details
        assert rule_name in error.message

    def test_business_rule_violation_with_context(self):
        """Тест создания ошибки с контекстом."""
        rule_name = "max_position_size"
        details = "Превышен максимальный размер позиции"
        context = {"current_size": 1000, "max_size": 500, "symbol": "BTC/USDT"}

        error = BusinessRuleViolationError(rule_name, details, context=context)

        assert error.context == context
        assert error.context["current_size"] == 1000
        assert error.context["max_size"] == 500

    def test_business_rule_violation_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        rule_name = "trading_hours"
        details = "Торговля недоступна в данное время"

        error = BusinessRuleViolationError(rule_name, details)
        error_dict = error.to_dict()

        assert error_dict["rule_name"] == rule_name
        assert error_dict["details"] == details
        assert error_dict["exception_type"] == "BusinessRuleViolationError"


class TestEntityNotFoundError:
    """Тесты для EntityNotFoundError."""

    def test_entity_not_found_creation(self):
        """Тест создания ошибки 'сущность не найдена'."""
        entity_type = "Account"
        entity_id = "acc_123"

        error = EntityNotFoundError(entity_type, entity_id)

        assert error.entity_type == entity_type
        assert error.entity_id == entity_id
        assert entity_type in error.message
        assert entity_id in error.message

    def test_entity_not_found_with_criteria(self):
        """Тест создания ошибки с критериями поиска."""
        entity_type = "Order"
        criteria = {"symbol": "BTC/USDT", "status": "pending"}

        error = EntityNotFoundError(entity_type, criteria=criteria)

        assert error.entity_type == entity_type
        assert error.criteria == criteria
        assert "BTC/USDT" in error.message

    def test_entity_not_found_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        entity_type = "Strategy"
        entity_id = "strat_456"

        error = EntityNotFoundError(entity_type, entity_id)
        error_dict = error.to_dict()

        assert error_dict["entity_type"] == entity_type
        assert error_dict["entity_id"] == entity_id
        assert error_dict["exception_type"] == "EntityNotFoundError"


class TestConfigurationError:
    """Тесты для ConfigurationError."""

    def test_configuration_error_creation(self):
        """Тест создания ошибки конфигурации."""
        config_key = "database.url"
        message = "Неверный формат URL базы данных"

        error = ConfigurationError(config_key, message)

        assert error.config_key == config_key
        assert error.message == message
        assert config_key in str(error)

    def test_configuration_error_with_value(self):
        """Тест создания ошибки с некорректным значением."""
        config_key = "api.timeout"
        value = "invalid_timeout"
        expected_type = "integer"

        error = ConfigurationError(config_key, f"Ожидается {expected_type}, получено {value}")

        assert error.config_key == config_key
        assert expected_type in error.message
        assert value in error.message

    def test_configuration_error_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        config_key = "redis.host"
        message = "Хост Redis не указан"

        error = ConfigurationError(config_key, message)
        error_dict = error.to_dict()

        assert error_dict["config_key"] == config_key
        assert error_dict["message"] == message
        assert error_dict["exception_type"] == "ConfigurationError"


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    def test_format_error_message(self):
        """Тест форматирования сообщения об ошибке."""
        base_message = "Ошибка валидации"
        context = {"field": "email", "value": "test"}

        formatted = _format_error_message(base_message, context)

        assert base_message in formatted
        assert "field" in formatted
        assert "email" in formatted

    def test_format_error_message_without_context(self):
        """Тест форматирования сообщения без контекста."""
        base_message = "Простая ошибка"

        formatted = _format_error_message(base_message)

        assert formatted == base_message

    def test_get_error_context(self):
        """Тест получения контекста ошибки."""
        context = {"user_id": 123, "action": "create_order"}

        error_context = _get_error_context(context)

        assert error_context == context
        assert "user_id" in error_context
        assert error_context["action"] == "create_order"

    def test_get_error_context_none(self):
        """Тест получения контекста при None."""
        error_context = _get_error_context(None)

        assert error_context == {}

    def test_get_error_context_empty(self):
        """Тест получения пустого контекста."""
        error_context = _get_error_context({})

        assert error_context == {}


class TestExceptionIntegration:
    """Интеграционные тесты исключений."""

    def test_exception_hierarchy(self):
        """Тест иерархии исключений."""
        # Проверяем, что все исключения наследуются от BaseDomainException
        validation_error = ValidationError("field", "value", "rule")
        business_error = BusinessRuleViolationError("rule", "details")
        not_found_error = EntityNotFoundError("Entity", "id")
        config_error = ConfigurationError("key", "message")

        assert isinstance(validation_error, BaseDomainException)
        assert isinstance(business_error, BaseDomainException)
        assert isinstance(not_found_error, BaseDomainException)
        assert isinstance(config_error, BaseDomainException)

    def test_exception_serialization(self):
        """Тест сериализации исключений."""
        exceptions = [
            ValidationError("email", "invalid", "format"),
            BusinessRuleViolationError("balance", "Недостаточно средств"),
            EntityNotFoundError("Account", "acc_123"),
            ConfigurationError("database.url", "Неверный URL"),
        ]

        for exception in exceptions:
            exception_dict = exception.to_dict()

            assert "message" in exception_dict
            assert "timestamp" in exception_dict
            assert "exception_type" in exception_dict
            assert exception_dict["exception_type"] in [
                "ValidationError",
                "BusinessRuleViolationError",
                "EntityNotFoundError",
                "ConfigurationError",
            ]

    def test_exception_chaining(self):
        """Тест цепочки исключений."""
        original_error = ValueError("Оригинальная ошибка")

        domain_error = BaseDomainException("Ошибка домена", cause=original_error, context={"operation": "validation"})

        assert domain_error.cause == original_error
        assert "Оригинальная ошибка" in str(domain_error)
        assert domain_error.context["operation"] == "validation"
