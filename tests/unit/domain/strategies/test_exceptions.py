from decimal import Decimal
from uuid import uuid4
from domain.strategies.exceptions import (
    StrategyError,
    StrategyCreationError,
    StrategyValidationError,
    StrategyExecutionError,
    StrategyNotFoundError,
    StrategyDuplicateError,
    StrategyRegistryError,
    StrategyFactoryError,
    StrategyConfigurationError,
    StrategyParameterError,
    StrategyTypeError,
    StrategyStateError
)
class TestStrategyExceptions:
    def test_strategy_error_base_class(self: "TestStrategyExceptions") -> None:
        """Тест базового класса исключений стратегий."""
        error = StrategyError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.context == {}
    def test_strategy_error_with_context(self: "TestStrategyExceptions") -> None:
        """Тест исключения с контекстом."""
        context = {
            "strategy_id": "test_id",
            "parameter": "sma_period",
            "value": 20
        }
        error = StrategyError("Test error", context=context)
        assert error.message == "Test error"
        assert error.context == context
        assert error.context["strategy_id"] == "test_id"
    def test_strategy_creation_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки создания стратегии."""
        context = {
            "strategy_name": "invalid_strategy",
            "available_strategies": ["trend_following", "mean_reversion"]
        }
        error = StrategyCreationError("Failed to create strategy", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Failed to create strategy"
        assert error.context["strategy_name"] == "invalid_strategy"
        assert "trend_following" in error.context["available_strategies"]
    def test_strategy_validation_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки валидации стратегии."""
        validation_errors = [
            "Parameter 'sma_period' must be positive",
            "Trading pairs list cannot be empty",
            "Confidence threshold must be between 0 and 1"
        ]
        context = {
            "validation_errors": validation_errors,
            "strategy_type": "trend_following"
        }
        error = StrategyValidationError("Strategy validation failed", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Strategy validation failed"
        assert len(error.context["validation_errors"]) == 3
        assert error.context["strategy_type"] == "trend_following"
    def test_strategy_execution_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки выполнения стратегии."""
        context = {
            "strategy_id": str(uuid4()),
            "execution_step": "signal_generation",
            "market_data_timestamp": "2023-01-01T10:00:00Z"
        }
        error = StrategyExecutionError("Strategy execution failed", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Strategy execution failed"
        assert "strategy_id" in error.context
        assert error.context["execution_step"] == "signal_generation"
    def test_strategy_not_found_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки поиска стратегии."""
        strategy_id = str(uuid4())
        context = {
            "strategy_id": strategy_id,
            "search_criteria": {"name": "test_strategy"}
        }
        error = StrategyNotFoundError(f"Strategy {strategy_id} not found", context=context)
        assert isinstance(error, StrategyError)
        assert strategy_id in error.message
        assert error.context["strategy_id"] == strategy_id
    def test_strategy_duplicate_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки дублирования стратегии."""
        strategy_name = "duplicate_strategy"
        context = {
            "strategy_name": strategy_name,
            "existing_strategy_id": str(uuid4())
        }
        error = StrategyDuplicateError(f"Strategy {strategy_name} already exists", context=context)
        assert isinstance(error, StrategyError)
        assert strategy_name in error.message
        assert error.context["strategy_name"] == strategy_name
    def test_strategy_registry_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки реестра стратегий."""
        context = {
            "operation": "register",
            "registry_state": "full",
            "max_strategies": 1000
        }
        error = StrategyRegistryError("Registry operation failed", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Registry operation failed"
        assert error.context["operation"] == "register"
        assert error.context["max_strategies"] == 1000
    def test_strategy_factory_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки фабрики стратегий."""
        context = {
            "factory_state": "initializing",
            "missing_dependencies": ["config_service", "validator"]
        }
        error = StrategyFactoryError("Factory initialization failed", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Factory initialization failed"
        assert error.context["factory_state"] == "initializing"
        assert len(error.context["missing_dependencies"]) == 2
    def test_strategy_configuration_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки конфигурации стратегии."""
        context = {
            "config_file": "strategy_config.json",
            "missing_sections": ["risk_management", "execution"]
        }
        error = StrategyConfigurationError("Invalid strategy configuration", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Invalid strategy configuration"
        assert error.context["config_file"] == "strategy_config.json"
    def test_strategy_parameter_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки параметров стратегии."""
        context = {
            "parameter_name": "sma_period",
            "parameter_value": -5,
            "valid_range": "1-100"
        }
        error = StrategyParameterError("Invalid parameter value", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Invalid parameter value"
        assert error.context["parameter_name"] == "sma_period"
        assert error.context["parameter_value"] == -5
    def test_strategy_type_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки типа стратегии."""
        context = {
            "expected_type": "trend_following",
            "actual_type": "mean_reversion",
            "strategy_id": str(uuid4())
        }
        error = StrategyTypeError("Strategy type mismatch", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Strategy type mismatch"
        assert error.context["expected_type"] == "trend_following"
        assert error.context["actual_type"] == "mean_reversion"
    def test_strategy_state_error(self: "TestStrategyExceptions") -> None:
        """Тест ошибки состояния стратегии."""
        context = {
            "current_state": "inactive",
            "required_state": "active",
            "strategy_id": str(uuid4())
        }
        error = StrategyStateError("Invalid strategy state", context=context)
        assert isinstance(error, StrategyError)
        assert error.message == "Invalid strategy state"
        assert error.context["current_state"] == "inactive"
        assert error.context["required_state"] == "active"
    def test_exception_inheritance_hierarchy(self: "TestStrategyExceptions") -> None:
        """Тест иерархии наследования исключений."""
        # Проверяем, что все исключения наследуются от StrategyError
        exceptions = [
            StrategyCreationError,
            StrategyValidationError,
            StrategyExecutionError,
            StrategyNotFoundError,
            StrategyDuplicateError,
            StrategyRegistryError,
            StrategyFactoryError,
            StrategyConfigurationError,
            StrategyParameterError,
            StrategyTypeError,
            StrategyStateError
        ]
        for exception_class in exceptions:
            assert issubclass(exception_class, StrategyError)
    def test_exception_with_nested_context(self: "TestStrategyExceptions") -> None:
        """Тест исключения с вложенным контекстом."""
        nested_context = {
            "strategy": {
                "id": str(uuid4()),
                "name": "test_strategy",
                "parameters": {
                    "sma_period": 20,
                    "ema_period": 12
                }
            },
            "validation": {
                "errors": ["Error 1", "Error 2"],
                "warnings": ["Warning 1"]
            }
        }
        error = StrategyValidationError("Complex validation error", context=nested_context)
        assert error.context["strategy"]["name"] == "test_strategy"
        assert error.context["strategy"]["parameters"]["sma_period"] == 20
        assert len(error.context["validation"]["errors"]) == 2
        assert len(error.context["validation"]["warnings"]) == 1
    def test_exception_serialization(self: "TestStrategyExceptions") -> None:
        """Тест сериализации исключения."""
        context = {
            "strategy_id": str(uuid4()),
            "timestamp": "2023-01-01T10:00:00Z",
            "numeric_value": 42,
            "decimal_value": Decimal("3.14")
        }
        error = StrategyExecutionError("Serialization test", context=context)
        # Проверяем, что исключение можно преобразовать в строку
        error_str = str(error)
        assert "Serialization test" in error_str
        # Проверяем, что контекст доступен
        assert error.context["strategy_id"] == context["strategy_id"]
        assert error.context["numeric_value"] == 42
        assert error.context["decimal_value"] == Decimal("3.14") 
