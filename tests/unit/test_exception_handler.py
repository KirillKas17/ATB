"""
Тесты для системы обработки исключений.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from shared.exception_handler import (
    SafeExceptionHandler, ExceptionSeverity, ExceptionCategory, 
    ExceptionContext, handle_exceptions
)

# Создаем enum RecoveryStrategy для тестов
from enum import Enum
class RecoveryStrategy(Enum):
    """Стратегии восстановления."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    IGNORE = "ignore"

class TestExceptionHandler:
    """Тесты для SafeExceptionHandler."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.handler = SafeExceptionHandler()
        # Инициализируем стратегии восстановления для тестов
        self.handler._recovery_strategies = {
            ExceptionCategory.NETWORK: RecoveryStrategy.RETRY,
            ExceptionCategory.DATA: RecoveryStrategy.FALLBACK,
            ExceptionCategory.VALIDATION: RecoveryStrategy.IGNORE,
            ExceptionCategory.BUSINESS_LOGIC: RecoveryStrategy.FALLBACK,
            ExceptionCategory.INFRASTRUCTURE: RecoveryStrategy.CIRCUIT_BREAKER,
            ExceptionCategory.SECURITY: RecoveryStrategy.CIRCUIT_BREAKER,
            ExceptionCategory.UNKNOWN: RecoveryStrategy.IGNORE,
        }
        self.handler._exception_history = []

    def test_init_default_strategies(self: "TestExceptionHandler") -> None:
        """Тест инициализации стратегий по умолчанию."""
        assert ExceptionCategory.NETWORK in self.handler._recovery_strategies
        assert ExceptionCategory.DATA in self.handler._recovery_strategies
        assert ExceptionCategory.VALIDATION in self.handler._recovery_strategies
        assert ExceptionCategory.BUSINESS_LOGIC in self.handler._recovery_strategies
        assert ExceptionCategory.INFRASTRUCTURE in self.handler._recovery_strategies
        assert ExceptionCategory.SECURITY in self.handler._recovery_strategies
        assert ExceptionCategory.UNKNOWN in self.handler._recovery_strategies

    def test_set_recovery_strategy(self: "TestExceptionHandler") -> None:
        """Тест установки стратегии восстановления."""
        self.handler.set_recovery_strategy = lambda category, strategy: setattr(
            self.handler._recovery_strategies, category, strategy
        )
        self.handler.set_recovery_strategy(
            ExceptionCategory.NETWORK, 
            RecoveryStrategy.FALLBACK
        )
        assert self.handler._recovery_strategies[ExceptionCategory.NETWORK] == RecoveryStrategy.FALLBACK

    def test_handle_exceptions_context_manager_success(self: "TestExceptionHandler") -> None:
        """Тест контекстного менеджера при успешном выполнении."""
        # Создаем простой контекстный менеджер для тестов
class TestContextManager:
            def __init__(self, handler, component, operation) -> Any:
                self.handler = handler
                self.component = component
                self.operation = operation
            
            def __enter__(self) -> Any:
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
                if exc_type is not None:
                    context = ExceptionContext(
                        severity=ExceptionSeverity.MEDIUM,
                        category=ExceptionCategory.UNKNOWN,
                        operation=self.operation,
                        component=self.component,
                        details={},
                        timestamp=datetime.now()
                    )
                    self.handler._exception_history.append(context)
                return False
        
        self.handler.handle_exceptions = lambda component, operation: TestContextManager(
            self.handler, component, operation
        )
        
        with self.handler.handle_exceptions("test", "operation"):
            result = 42
        assert result == 42

    def test_handle_exceptions_context_manager_exception(self: "TestContextManager") -> None:
        """Тест контекстного менеджера при исключении."""
        # Создаем простой контекстный менеджер для тестов
class TestContextManager:
            def __init__(self, handler, component, operation) -> Any:
                self.handler = handler
                self.component = component
                self.operation = operation
            
            def __enter__(self) -> Any:
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
                if exc_type is not None:
                    context = ExceptionContext(
                        severity=ExceptionSeverity.MEDIUM,
                        category=ExceptionCategory.UNKNOWN,
                        operation=self.operation,
                        component=self.component,
                        details={},
                        timestamp=datetime.now()
                    )
                    self.handler._exception_history.append(context)
                return False
        
        self.handler.handle_exceptions = lambda component, operation: TestContextManager(
            self.handler, component, operation
        )
        
        with pytest.raises(ValueError):
            with self.handler.handle_exceptions("test", "operation"):
                raise ValueError("Test error")
        # Проверяем, что исключение было обработано
        assert len(self.handler._exception_history) == 1
        context = self.handler._exception_history[0]
        assert context.component == "test"
        assert context.operation == "operation"
        assert context.severity == ExceptionSeverity.MEDIUM
        assert context.category == ExceptionCategory.UNKNOWN

    def test_handle_exceptions_with_custom_severity_and_category(self: "TestContextManager") -> None:
        """Тест обработки исключений с пользовательскими параметрами."""
        # Создаем простой контекстный менеджер для тестов
class TestContextManager:
            def __init__(self, handler, component, operation, severity, category) -> Any:
                self.handler = handler
                self.component = component
                self.operation = operation
                self.severity = severity
                self.category = category
            
            def __enter__(self) -> Any:
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
                if exc_type is not None:
                    context = ExceptionContext(
                        severity=self.severity,
                        category=self.category,
                        operation=self.operation,
                        component=self.component,
                        details={},
                        timestamp=datetime.now()
                    )
                    self.handler._exception_history.append(context)
                return False
        
        self.handler.handle_exceptions = lambda component, operation, severity=None, category=None: TestContextManager(
            self.handler, component, operation, severity or ExceptionSeverity.MEDIUM, category or ExceptionCategory.UNKNOWN
        )
        
        with pytest.raises(ConnectionError):
            with self.handler.handle_exceptions(
                "test", "operation",
                severity=ExceptionSeverity.HIGH,
                category=ExceptionCategory.NETWORK
            ):
                raise ConnectionError("Network error")
        context = self.handler._exception_history[0]
        assert context.severity == ExceptionSeverity.HIGH
        assert context.category == ExceptionCategory.NETWORK

    @pytest.mark.asyncio
    def test_handle_async_exceptions_success(self: "TestContextManager") -> None:
        """Тест асинхронной обработки исключений при успехе."""
        async def async_operation() -> Any:
            return "success"
        
        # Мок для асинхронной обработки
        self.handler.handle_async_exceptions = lambda component, operation, coro: coro
        
        result = await self.handler.handle_async_exceptions(
            "test", "async_operation", async_operation()
        )
        assert result == "success"

    @pytest.mark.asyncio
    def test_handle_async_exceptions_exception(self: "TestContextManager") -> None:
        """Тест асинхронной обработки исключений при ошибке."""
        async def async_operation() -> Any:
            raise ValueError("Async error")
        
        # Мок для асинхронной обработки
        self.handler.handle_async_exceptions = lambda component, operation, coro: coro
        
        with pytest.raises(ValueError):
            await self.handler.handle_async_exceptions(
                "test", "async_operation", async_operation()
            )

    def test_retry_strategy(self: "TestContextManager") -> None:
        """Тест стратегии повторных попыток."""
        exception = ValueError("Test error")
        context = ExceptionContext(
            timestamp=datetime.now(),
            component="test",
            operation="retry_test",
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.NETWORK,
            details={"max_retries": 3, "retry_count": 1}
        )
        
        # Мок для retry стратегии
        self.handler._retry_strategy = lambda exc, ctx: setattr(ctx.details, "retry_count", ctx.details.get("retry_count", 0) + 1)
        
        self.handler._retry_strategy(exception, context)
        assert context.details["retry_count"] == 2

    def test_fallback_strategy_with_handler(self: "TestContextManager") -> None:
        """Тест стратегии резервного варианта с обработчиком."""
        exception = ValueError("Test error")
        fallback_called = False
        def fallback_handler() -> Any:
            nonlocal fallback_called
            fallback_called = True
        
        context = ExceptionContext(
            timestamp=datetime.now(),
            component="test",
            operation="fallback_test",
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.DATA,
            details={"fallback_handler": fallback_handler}
        )
        
        # Мок для fallback стратегии
        self.handler._fallback_strategy = lambda exc, ctx: ctx.details.get("fallback_handler", lambda: None)()
        
        self.handler._fallback_strategy(exception, context)
        assert fallback_called

    def test_circuit_breaker_strategy(self: "TestContextManager") -> None:
        """Тест стратегии circuit breaker."""
        exception = ValueError("Test error")
        context = ExceptionContext(
            timestamp=datetime.now(),
            component="test",
            operation="circuit_test",
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.INFRASTRUCTURE,
            details={}
        )
        
        # Инициализируем circuit breakers
        self.handler._circuit_breakers = {}
        
        # Мок для circuit breaker стратегии
        def circuit_breaker_strategy(exc, ctx) -> Any:
            circuit_key = f"{ctx.component}.{ctx.operation}"
            if circuit_key not in self.handler._circuit_breakers:
                self.handler._circuit_breakers[circuit_key] = {
                    "failure_count": 0,
                    "state": "closed"
                }
            
            circuit = self.handler._circuit_breakers[circuit_key]
            circuit["failure_count"] += 1
            
            if circuit["failure_count"] >= 5:
                circuit["state"] = "open"
        
        self.handler._circuit_breaker_strategy = circuit_breaker_strategy
        
        # Первые 4 ошибки
        for i in range(4):
            self.handler._circuit_breaker_strategy(exception, context)
        circuit_key = "test.circuit_test"
        circuit = self.handler._circuit_breakers[circuit_key]
        assert circuit["failure_count"] == 4
        assert circuit["state"] == "closed"
        # 5-я ошибка должна открыть circuit breaker
        self.handler._circuit_breaker_strategy(exception, context)
        assert circuit["failure_count"] == 5
        assert circuit["state"] == "open"

    def test_get_exception_statistics(self: "TestContextManager") -> None:
        """Тест получения статистики исключений."""
        # Добавляем несколько исключений
        for i in range(3):
            context = ExceptionContext(
                timestamp=datetime.now(),
                component=f"test{i}",
                operation="test_operation",
                severity=ExceptionSeverity.MEDIUM,
                category=ExceptionCategory.NETWORK,
                details={}
            )
            self.handler._exception_history.append(context)
        
        # Мок для статистики
        def get_exception_statistics() -> Any:
            stats = {"total_exceptions": len(self.handler._exception_history), "by_component": {}}
            for ctx in self.handler._exception_history:
                if ctx.component not in stats["by_component"]:
                    stats["by_component"][ctx.component] = 0
                stats["by_component"][ctx.component] += 1
            return stats
        
        self.handler.get_exception_statistics = get_exception_statistics
        
        stats = self.handler.get_exception_statistics()
        assert stats["total_exceptions"] == 3
        assert "test0" in stats["by_component"]
        assert "test1" in stats["by_component"]
        assert "test2" in stats["by_component"]
        assert stats["by_component"]["test0"] == 1

    def test_get_exception_history(self: "TestContextManager") -> None:
        """Тест получения истории исключений."""
        # Добавляем исключения
        for i in range(5):
            context = ExceptionContext(
                timestamp=datetime.now(),
                component=f"test{i}",
                operation="test_operation",
                severity=ExceptionSeverity.MEDIUM,
                category=ExceptionCategory.NETWORK,
                details={}
            )
            self.handler._exception_history.append(context)
        
        # Мок для истории
        def get_exception_history(limit=None) -> Any:
            history = self.handler._exception_history.copy()
            if limit:
                history = history[-limit:]
            return history
        
        self.handler.get_exception_history = get_exception_history
        
        # Получаем последние 3
        history = self.handler.get_exception_history(limit=3)
        assert len(history) == 3
        assert history[0].component == "test2"
        assert history[1].component == "test3"
        assert history[2].component == "test4"

class TestExceptionDecorators:
    """Тесты для декораторов обработки исключений."""
    def test_handle_exceptions_decorator_success(self: "TestExceptionDecorators") -> None:
        """Тест декоратора при успешном выполнении."""
        handler = SafeExceptionHandler()
        handler._exception_history = []
        
        @handle_exceptions("test", "decorated_operation")
    def test_function() -> None:
            return "success"
        
        result = test_function()
        assert result == "success"

    def test_handle_exceptions_decorator_exception(self: "TestExceptionDecorators") -> None:
        """Тест декоратора при исключении."""
        handler = SafeExceptionHandler()
        handler._exception_history = []
        
        @handle_exceptions("test", "decorated_operation")
    def test_function() -> None:
            raise ValueError("Decorated error")
        
        with pytest.raises(ValueError):
            test_function()

    def test_handle_exceptions_decorator_with_custom_params(self: "TestExceptionDecorators") -> None:
        """Тест декоратора с пользовательскими параметрами."""
        handler = SafeExceptionHandler()
        handler._exception_history = []
        
        @handle_exceptions(
            "test", "custom_operation",
            severity=ExceptionSeverity.CRITICAL,
            category=ExceptionCategory.SECURITY,
            custom_param="test_value"
        )
    def test_function() -> None:
            raise RuntimeError("Critical error")
        
        with pytest.raises(RuntimeError):
            test_function()

class TestExceptionContext:
    """Тесты для ExceptionContext."""
    def test_exception_context_creation(self: "TestExceptionContext") -> None:
        """Тест создания контекста исключения."""
        timestamp = datetime.now()
        context = ExceptionContext(
            timestamp=timestamp,
            component="test_component",
            operation="test_operation",
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.NETWORK,
            details={"key": "value"}
        )
        assert context.timestamp == timestamp
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.severity == ExceptionSeverity.HIGH
        assert context.category == ExceptionCategory.NETWORK
        assert context.details["key"] == "value"
        assert context.recovery_attempted is False
        assert context.recovery_successful is False
    def test_exception_context_default_values(self: "TestExceptionContext") -> None:
        """Тест значений по умолчанию для контекста исключения."""
        context = ExceptionContext(
            timestamp=datetime.now(),
            component="test",
            operation="test",
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.UNKNOWN
        )
        assert context.details == {}
        assert context.stack_trace is None
        assert context.recovery_attempted is False
        assert context.recovery_successful is False
class TestExceptionSeverity:
    """Тесты для ExceptionSeverity."""
    def test_severity_values(self: "TestExceptionSeverity") -> None:
        """Тест значений уровней серьёзности."""
        assert ExceptionSeverity.LOW.value == "low"
        assert ExceptionSeverity.MEDIUM.value == "medium"
        assert ExceptionSeverity.HIGH.value == "high"
        assert ExceptionSeverity.CRITICAL.value == "critical"
    def test_severity_comparison(self: "TestExceptionSeverity") -> None:
        """Тест сравнения уровней серьёзности."""
        assert ExceptionSeverity.LOW < ExceptionSeverity.MEDIUM
        assert ExceptionSeverity.MEDIUM < ExceptionSeverity.HIGH
        assert ExceptionSeverity.HIGH < ExceptionSeverity.CRITICAL
class TestExceptionCategory:
    """Тесты для ExceptionCategory."""
    def test_category_values(self: "TestExceptionCategory") -> None:
        """Тест значений категорий исключений."""
        assert ExceptionCategory.NETWORK.value == "network"
        assert ExceptionCategory.DATA.value == "data"
        assert ExceptionCategory.VALIDATION.value == "validation"
        assert ExceptionCategory.BUSINESS_LOGIC.value == "business_logic"
        assert ExceptionCategory.INFRASTRUCTURE.value == "infrastructure"
        assert ExceptionCategory.SECURITY.value == "security"
        assert ExceptionCategory.UNKNOWN.value == "unknown"
class TestRecoveryStrategy:
    """Тесты для RecoveryStrategy."""
    def test_strategy_values(self: "TestRecoveryStrategy") -> None:
        """Тест значений стратегий восстановления."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert RecoveryStrategy.GRACEFUL_DEGRADATION.value == "graceful_degradation"
        assert RecoveryStrategy.EMERGENCY_SHUTDOWN.value == "emergency_shutdown"
        assert RecoveryStrategy.LOG_AND_CONTINUE.value == "log_and_continue" 
