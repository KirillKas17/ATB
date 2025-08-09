"""
Unit тесты для protocol_exceptions.py.

Покрывает:
- ProtocolViolationError - нарушения протоколов
- ExchangeConnectionError - ошибки подключения к бирже
- DataIntegrityError - ошибки целостности данных
- RateLimitError - ошибки лимитов запросов
- AuthenticationError - ошибки аутентификации
- Вспомогательные функции
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from domain.exceptions.protocol_exceptions import (
    ProtocolViolationError,
    ExchangeConnectionError,
    DataIntegrityError,
    RateLimitError,
    AuthenticationError,
    _format_protocol_message,
    _get_retry_after_time,
)


class TestProtocolViolationError:
    """Тесты для ProtocolViolationError."""

    def test_protocol_violation_creation(self):
        """Тест создания ошибки нарушения протокола."""
        protocol_name = "REST API"
        violation_type = "invalid_request_format"
        details = "Неверный формат JSON в запросе"

        error = ProtocolViolationError(protocol_name, violation_type, details)

        assert error.protocol_name == protocol_name
        assert error.violation_type == violation_type
        assert error.details == details
        assert protocol_name in error.message

    def test_protocol_violation_with_context(self):
        """Тест создания ошибки с контекстом."""
        protocol_name = "WebSocket"
        violation_type = "message_sequence_error"
        details = "Нарушена последовательность сообщений"
        context = {"expected_seq": 5, "received_seq": 7, "last_message": "order_update"}

        error = ProtocolViolationError(protocol_name, violation_type, details, context=context)

        assert error.context == context
        assert error.context["expected_seq"] == 5
        assert error.context["received_seq"] == 7

    def test_protocol_violation_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        protocol_name = "FIX"
        violation_type = "tag_missing"
        details = "Отсутствует обязательный тег 35"

        error = ProtocolViolationError(protocol_name, violation_type, details)
        error_dict = error.to_dict()

        assert error_dict["protocol_name"] == protocol_name
        assert error_dict["violation_type"] == violation_type
        assert error_dict["details"] == details
        assert error_dict["exception_type"] == "ProtocolViolationError"

    def test_protocol_violation_severity(self):
        """Тест определения серьезности нарушения."""
        # Критическое нарушение
        critical_error = ProtocolViolationError("REST API", "authentication_failure", "Неверные учетные данные")

        # Некритическое нарушение
        minor_error = ProtocolViolationError("WebSocket", "heartbeat_missed", "Пропущен heartbeat")

        assert "authentication" in critical_error.message.lower()
        assert "heartbeat" in minor_error.message.lower()


class TestExchangeConnectionError:
    """Тесты для ExchangeConnectionError."""

    def test_exchange_connection_creation(self):
        """Тест создания ошибки подключения к бирже."""
        exchange_name = "Binance"
        connection_type = "REST"
        reason = "timeout"

        error = ExchangeConnectionError(exchange_name, connection_type, reason)

        assert error.exchange_name == exchange_name
        assert error.connection_type == connection_type
        assert error.reason == reason
        assert exchange_name in error.message

    def test_exchange_connection_with_retry_info(self):
        """Тест создания ошибки с информацией о повторах."""
        exchange_name = "Coinbase"
        connection_type = "WebSocket"
        reason = "connection_lost"
        retry_after = timedelta(seconds=30)
        max_retries = 3

        error = ExchangeConnectionError(
            exchange_name, connection_type, reason, retry_after=retry_after, max_retries=max_retries
        )

        assert error.retry_after == retry_after
        assert error.max_retries == max_retries
        assert "30" in str(error)

    def test_exchange_connection_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        exchange_name = "Kraken"
        connection_type = "FIX"
        reason = "network_error"

        error = ExchangeConnectionError(exchange_name, connection_type, reason)
        error_dict = error.to_dict()

        assert error_dict["exchange_name"] == exchange_name
        assert error_dict["connection_type"] == connection_type
        assert error_dict["reason"] == reason
        assert error_dict["exception_type"] == "ExchangeConnectionError"

    def test_exchange_connection_retryable(self):
        """Тест определения возможности повтора."""
        # Повторяемая ошибка
        retryable_error = ExchangeConnectionError("Binance", "REST", "timeout", retry_after=timedelta(seconds=5))

        # Неповторяемая ошибка
        non_retryable_error = ExchangeConnectionError("Coinbase", "REST", "authentication_failed")

        assert retryable_error.retry_after is not None
        assert non_retryable_error.retry_after is None


class TestDataIntegrityError:
    """Тесты для DataIntegrityError."""

    def test_data_integrity_creation(self):
        """Тест создания ошибки целостности данных."""
        data_type = "order_book"
        integrity_check = "bid_ask_spread"
        expected_value = 0.001
        actual_value = 0.005

        error = DataIntegrityError(data_type, integrity_check, expected_value, actual_value)

        assert error.data_type == data_type
        assert error.integrity_check == integrity_check
        assert error.expected_value == expected_value
        assert error.actual_value == actual_value
        assert data_type in error.message

    def test_data_integrity_with_threshold(self):
        """Тест создания ошибки с порогом."""
        data_type = "trade_data"
        integrity_check = "price_consistency"
        expected_value = 50000.0
        actual_value = 51000.0
        threshold = 0.02  # 2%

        error = DataIntegrityError(data_type, integrity_check, expected_value, actual_value, threshold=threshold)

        assert error.threshold == threshold
        assert "2%" in str(error) or "0.02" in str(error)

    def test_data_integrity_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        data_type = "market_data"
        integrity_check = "volume_consistency"
        expected_value = 1000
        actual_value = 950

        error = DataIntegrityError(data_type, integrity_check, expected_value, actual_value)
        error_dict = error.to_dict()

        assert error_dict["data_type"] == data_type
        assert error_dict["integrity_check"] == integrity_check
        assert error_dict["expected_value"] == expected_value
        assert error_dict["actual_value"] == actual_value
        assert error_dict["exception_type"] == "DataIntegrityError"

    def test_data_integrity_severity_calculation(self):
        """Тест расчета серьезности нарушения."""
        # Низкая серьезность
        minor_error = DataIntegrityError("order_book", "depth_consistency", 100, 99, threshold=0.05)

        # Высокая серьезность
        major_error = DataIntegrityError("trade_data", "price_consistency", 100, 50, threshold=0.01)

        # Проверяем, что ошибки созданы корректно
        assert minor_error.threshold == 0.05
        assert major_error.threshold == 0.01


class TestRateLimitError:
    """Тесты для RateLimitError."""

    def test_rate_limit_creation(self):
        """Тест создания ошибки лимита запросов."""
        endpoint = "/api/v3/order"
        limit_type = "requests_per_minute"
        current_usage = 1200
        limit = 1000

        error = RateLimitError(endpoint, limit_type, current_usage, limit)

        assert error.endpoint == endpoint
        assert error.limit_type == limit_type
        assert error.current_usage == current_usage
        assert error.limit == limit
        assert endpoint in error.message

    def test_rate_limit_with_reset_time(self):
        """Тест создания ошибки с временем сброса."""
        endpoint = "/api/v3/ticker"
        limit_type = "requests_per_second"
        current_usage = 10
        limit = 10
        reset_time = datetime.now() + timedelta(seconds=60)

        error = RateLimitError(endpoint, limit_type, current_usage, limit, reset_time=reset_time)

        assert error.reset_time == reset_time
        assert "60" in str(error) or "минуту" in str(error)

    def test_rate_limit_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        endpoint = "/api/v3/klines"
        limit_type = "requests_per_hour"
        current_usage = 5000
        limit = 5000

        error = RateLimitError(endpoint, limit_type, current_usage, limit)
        error_dict = error.to_dict()

        assert error_dict["endpoint"] == endpoint
        assert error_dict["limit_type"] == limit_type
        assert error_dict["current_usage"] == current_usage
        assert error_dict["limit"] == limit
        assert error_dict["exception_type"] == "RateLimitError"

    def test_rate_limit_retry_calculation(self):
        """Тест расчета времени до повтора."""
        reset_time = datetime.now() + timedelta(seconds=30)

        error = RateLimitError("/api/v3/order", "requests_per_minute", 1000, 1000, reset_time=reset_time)

        # Проверяем, что время сброса установлено
        assert error.reset_time == reset_time


class TestAuthenticationError:
    """Тесты для AuthenticationError."""

    def test_authentication_creation(self):
        """Тест создания ошибки аутентификации."""
        auth_method = "API_KEY"
        reason = "invalid_signature"
        details = "Неверная подпись запроса"

        error = AuthenticationError(auth_method, reason, details)

        assert error.auth_method == auth_method
        assert error.reason == reason
        assert error.details == details
        assert auth_method in error.message

    def test_authentication_with_expiry(self):
        """Тест создания ошибки с истечением срока."""
        auth_method = "JWT_TOKEN"
        reason = "token_expired"
        details = "Токен истек"
        expiry_time = datetime.now() - timedelta(minutes=5)

        error = AuthenticationError(auth_method, reason, details, expiry_time=expiry_time)

        assert error.expiry_time == expiry_time
        assert "истек" in error.message.lower()

    def test_authentication_to_dict(self):
        """Тест преобразования ошибки в словарь."""
        auth_method = "OAUTH2"
        reason = "invalid_scope"
        details = "Недостаточно прав доступа"

        error = AuthenticationError(auth_method, reason, details)
        error_dict = error.to_dict()

        assert error_dict["auth_method"] == auth_method
        assert error_dict["reason"] == reason
        assert error_dict["details"] == details
        assert error_dict["exception_type"] == "AuthenticationError"

    def test_authentication_severity(self):
        """Тест определения серьезности ошибки аутентификации."""
        # Критическая ошибка
        critical_error = AuthenticationError("API_KEY", "key_revoked", "Ключ API отозван")

        # Некритическая ошибка
        minor_error = AuthenticationError("JWT_TOKEN", "token_refresh_needed", "Требуется обновление токена")

        assert "отозван" in critical_error.message.lower()
        assert "обновление" in minor_error.message.lower()


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    def test_format_protocol_message(self):
        """Тест форматирования сообщения протокола."""
        protocol_name = "REST API"
        violation_type = "rate_limit_exceeded"
        details = "Превышен лимит запросов"

        formatted = _format_protocol_message(protocol_name, violation_type, details)

        assert protocol_name in formatted
        assert violation_type in formatted
        assert details in formatted

    def test_format_protocol_message_with_context(self):
        """Тест форматирования сообщения с контекстом."""
        protocol_name = "WebSocket"
        violation_type = "connection_lost"
        details = "Соединение потеряно"
        context = {"last_heartbeat": "2023-01-01T12:00:00Z"}

        formatted = _format_protocol_message(protocol_name, violation_type, details, context)

        assert protocol_name in formatted
        assert "2023-01-01" in formatted

    def test_get_retry_after_time(self):
        """Тест получения времени до повтора."""
        reset_time = datetime.now() + timedelta(seconds=60)

        retry_after = _get_retry_after_time(reset_time)

        assert retry_after > 0
        assert retry_after <= 60

    def test_get_retry_after_time_past(self):
        """Тест получения времени до повтора для прошедшего времени."""
        reset_time = datetime.now() - timedelta(seconds=60)

        retry_after = _get_retry_after_time(reset_time)

        assert retry_after == 0

    def test_get_retry_after_time_none(self):
        """Тест получения времени до повтора для None."""
        retry_after = _get_retry_after_time(None)

        assert retry_after == 0


class TestProtocolExceptionIntegration:
    """Интеграционные тесты протокольных исключений."""

    def test_exception_hierarchy(self):
        """Тест иерархии протокольных исключений."""
        from domain.exceptions.base_exceptions import BaseDomainException

        # Проверяем, что все протокольные исключения наследуются от BaseDomainException
        protocol_error = ProtocolViolationError("REST", "invalid_format", "details")
        connection_error = ExchangeConnectionError("Binance", "REST", "timeout")
        integrity_error = DataIntegrityError("order_book", "spread_check", 0.001, 0.002)
        rate_limit_error = RateLimitError("/api/order", "per_minute", 1000, 1000)
        auth_error = AuthenticationError("API_KEY", "invalid_key", "details")

        assert isinstance(protocol_error, BaseDomainException)
        assert isinstance(connection_error, BaseDomainException)
        assert isinstance(integrity_error, BaseDomainException)
        assert isinstance(rate_limit_error, BaseDomainException)
        assert isinstance(auth_error, BaseDomainException)

    def test_exception_serialization(self):
        """Тест сериализации протокольных исключений."""
        exceptions = [
            ProtocolViolationError("WebSocket", "message_error", "Invalid message"),
            ExchangeConnectionError("Coinbase", "REST", "network_error"),
            DataIntegrityError("trade_data", "price_check", 100, 101),
            RateLimitError("/api/ticker", "per_second", 10, 10),
            AuthenticationError("JWT", "expired", "Token expired"),
        ]

        for exception in exceptions:
            exception_dict = exception.to_dict()

            assert "message" in exception_dict
            assert "timestamp" in exception_dict
            assert "exception_type" in exception_dict
            assert exception_dict["exception_type"] in [
                "ProtocolViolationError",
                "ExchangeConnectionError",
                "DataIntegrityError",
                "RateLimitError",
                "AuthenticationError",
            ]

    def test_exception_context_preservation(self):
        """Тест сохранения контекста в исключениях."""
        context = {"user_id": 123, "request_id": "req_456", "timestamp": "2023-01-01T12:00:00Z"}

        protocol_error = ProtocolViolationError("REST", "validation_error", "Invalid data", context=context)

        assert protocol_error.context == context
        assert protocol_error.context["user_id"] == 123
        assert protocol_error.context["request_id"] == "req_456"

    def test_exception_chaining(self):
        """Тест цепочки протокольных исключений."""
        original_error = ConnectionError("Network timeout")

        protocol_error = ProtocolViolationError(
            "REST API", "connection_failed", "Failed to connect to exchange", cause=original_error
        )

        assert protocol_error.cause == original_error
        assert "Network timeout" in str(protocol_error)
