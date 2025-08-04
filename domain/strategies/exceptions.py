"""
Исключения стратегий - промышленная обработка ошибок.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Type, Union


class StrategyError(Exception):
    """Базовое исключение для всех ошибок стратегий."""

    def __init__(
        self,
        message: str,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализация исключения.

        Args:
            message: Сообщение об ошибке
            strategy_id: ID стратегии
            strategy_name: Имя стратегии
            details: Дополнительные детали ошибки
        """
        super().__init__(message)
        self.message = message
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.details = details or {}
        self.timestamp = None  # Будет установлено при логировании

    def __str__(self) -> str:
        """Строковое представление исключения."""
        parts = [self.message]

        if self.strategy_id:
            parts.append(f"Strategy ID: {self.strategy_id}")

        if self.strategy_name:
            parts.append(f"Strategy Name: {self.strategy_name}")

        if self.details:
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            parts.append(f"Details: {details_str}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать исключение в словарь."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class StrategyCreationError(StrategyError):
    """Ошибка создания стратегии."""

    def __init__(
        self,
        message: str,
        strategy_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения создания стратегии.

        Args:
            message: Сообщение об ошибке
            strategy_type: Тип стратегии
            parameters: Параметры создания
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.strategy_type = strategy_type
        self.parameters = parameters or {}

        if strategy_type:
            self.details["strategy_type"] = strategy_type
        if parameters:
            self.details["parameters"] = parameters


class StrategyValidationError(StrategyError):
    """Ошибка валидации стратегии."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_errors: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения валидации.

        Args:
            message: Сообщение об ошибке
            validation_errors: Список ошибок валидации
            field_errors: Ошибки по полям
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        self.field_errors = field_errors or {}

        if validation_errors:
            self.details["validation_errors"] = validation_errors
        if field_errors:
            self.details["field_errors"] = field_errors

    def add_validation_error(self, error: str) -> None:
        """Добавить ошибку валидации."""
        self.validation_errors.append(error)
        if "validation_errors" not in self.details:
            self.details["validation_errors"] = []
        self.details["validation_errors"].append(error)

    def add_field_error(self, field: str, error: str) -> None:
        """Добавить ошибку поля."""
        self.field_errors[field] = error
        if "field_errors" not in self.details:
            self.details["field_errors"] = {}
        self.details["field_errors"][field] = error


class StrategyExecutionError(StrategyError):
    """Ошибка выполнения стратегии."""

    def __init__(
        self,
        message: str,
        execution_step: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        signal_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения выполнения.

        Args:
            message: Сообщение об ошибке
            execution_step: Шаг выполнения
            market_data: Данные рынка
            signal_data: Данные сигнала
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.execution_step = execution_step
        self.market_data = market_data or {}
        self.signal_data = signal_data or {}

        if execution_step:
            self.details["execution_step"] = execution_step
        if market_data:
            self.details["market_data"] = market_data
        if signal_data:
            self.details["signal_data"] = signal_data


class StrategyConfigurationError(StrategyError):
    """Ошибка конфигурации стратегии."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения конфигурации.

        Args:
            message: Сообщение об ошибке
            config_section: Секция конфигурации
            config_value: Значение конфигурации
            expected_type: Ожидаемый тип
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_value = config_value
        self.expected_type = expected_type

        if config_section:
            self.details["config_section"] = config_section
        if config_value is not None:
            self.details["config_value"] = config_value
        if expected_type:
            self.details["expected_type"] = expected_type


class StrategyStateError(StrategyError):
    """Ошибка состояния стратегии."""

    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        allowed_states: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения состояния.

        Args:
            message: Сообщение об ошибке
            current_state: Текущее состояние
            expected_state: Ожидаемое состояние
            allowed_states: Разрешенные состояния
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.current_state = current_state
        self.expected_state = expected_state
        self.allowed_states = allowed_states or []

        if current_state:
            self.details["current_state"] = current_state
        if expected_state:
            self.details["expected_state"] = expected_state
        if allowed_states:
            self.details["allowed_states"] = allowed_states


class StrategyNotFoundError(StrategyError):
    """Ошибка поиска стратегии."""

    def __init__(
        self, message: str, search_criteria: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Инициализация исключения поиска.

        Args:
            message: Сообщение об ошибке
            search_criteria: Критерии поиска
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.search_criteria = search_criteria or {}

        if search_criteria:
            self.details["search_criteria"] = search_criteria


class StrategyDuplicateError(StrategyError):
    """Ошибка дублирования стратегии."""

    def __init__(
        self,
        message: str,
        existing_strategy_id: Optional[str] = None,
        duplicate_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения дублирования.

        Args:
            message: Сообщение об ошибке
            existing_strategy_id: ID существующей стратегии
            duplicate_fields: Дублирующиеся поля
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.existing_strategy_id = existing_strategy_id
        self.duplicate_fields = duplicate_fields or []

        if existing_strategy_id:
            self.details["existing_strategy_id"] = existing_strategy_id
        if duplicate_fields:
            self.details["duplicate_fields"] = duplicate_fields


class StrategyPerformanceError(StrategyError):
    """Ошибка производительности стратегии."""

    def __init__(
        self,
        message: str,
        performance_metric: Optional[str] = None,
        current_value: Optional[Union[float, Decimal]] = None,
        threshold_value: Optional[Union[float, Decimal]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения производительности.

        Args:
            message: Сообщение об ошибке
            performance_metric: Метрика производительности
            current_value: Текущее значение
            threshold_value: Пороговое значение
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.performance_metric = performance_metric
        self.current_value = current_value
        self.threshold_value = threshold_value

        if performance_metric:
            self.details["performance_metric"] = performance_metric
        if current_value is not None:
            self.details["current_value"] = current_value
        if threshold_value is not None:
            self.details["threshold_value"] = threshold_value


class StrategyRiskError(StrategyError):
    """Ошибка риска стратегии."""

    def __init__(
        self,
        message: str,
        risk_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        risk_value: Optional[Union[float, Decimal]] = None,
        max_allowed: Optional[Union[float, Decimal]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения риска.

        Args:
            message: Сообщение об ошибке
            risk_type: Тип риска
            risk_level: Уровень риска
            risk_value: Значение риска
            max_allowed: Максимально допустимое значение
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.risk_type = risk_type
        self.risk_level = risk_level
        self.risk_value = risk_value
        self.max_allowed = max_allowed

        if risk_type:
            self.details["risk_type"] = risk_type
        if risk_level:
            self.details["risk_level"] = risk_level
        if risk_value is not None:
            self.details["risk_value"] = risk_value
        if max_allowed is not None:
            self.details["max_allowed"] = max_allowed


class StrategyMarketError(StrategyError):
    """Ошибка рыночных данных стратегии."""

    def __init__(
        self,
        message: str,
        market_pair: Optional[str] = None,
        data_type: Optional[str] = None,
        data_timestamp: Optional[str] = None,
        market_condition: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения рыночных данных.

        Args:
            message: Сообщение об ошибке
            market_pair: Торговая пара
            data_type: Тип данных
            data_timestamp: Временная метка данных
            market_condition: Состояние рынка
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.market_pair = market_pair
        self.data_type = data_type
        self.data_timestamp = data_timestamp
        self.market_condition = market_condition

        if market_pair:
            self.details["market_pair"] = market_pair
        if data_type:
            self.details["data_type"] = data_type
        if data_timestamp:
            self.details["data_timestamp"] = data_timestamp
        if market_condition:
            self.details["market_condition"] = market_condition


class StrategySignalError(StrategyError):
    """Ошибка сигналов стратегии."""

    def __init__(
        self,
        message: str,
        signal_type: Optional[str] = None,
        signal_strength: Optional[Union[float, Decimal]] = None,
        signal_confidence: Optional[Union[float, Decimal]] = None,
        signal_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения сигнала.

        Args:
            message: Сообщение об ошибке
            signal_type: Тип сигнала
            signal_strength: Сила сигнала
            signal_confidence: Уверенность сигнала
            signal_data: Данные сигнала
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.signal_type = signal_type
        self.signal_strength = signal_strength
        self.signal_confidence = signal_confidence
        self.signal_data = signal_data or {}

        if signal_type:
            self.details["signal_type"] = signal_type
        if signal_strength is not None:
            self.details["signal_strength"] = signal_strength
        if signal_confidence is not None:
            self.details["signal_confidence"] = signal_confidence
        if signal_data:
            self.details["signal_data"] = signal_data


class StrategyFactoryError(StrategyError):
    """Ошибка фабрики стратегий."""

    def __init__(
        self,
        message: str,
        factory_operation: Optional[str] = None,
        strategy_name: Optional[str] = None,
        creator_function: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения фабрики.

        Args:
            message: Сообщение об ошибке
            factory_operation: Операция фабрики
            strategy_name: Имя стратегии
            creator_function: Функция создания
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.factory_operation = factory_operation
        self.strategy_name = strategy_name
        self.creator_function = creator_function

        if factory_operation:
            self.details["factory_operation"] = factory_operation
        if strategy_name:
            self.details["strategy_name"] = strategy_name
        if creator_function:
            self.details["creator_function"] = creator_function


class StrategyRegistryError(StrategyError):
    """Ошибка реестра стратегий."""

    def __init__(
        self,
        message: str,
        registry_operation: Optional[str] = None,
        registry_state: Optional[str] = None,
        affected_strategies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения реестра.

        Args:
            message: Сообщение об ошибке
            registry_operation: Операция реестра
            registry_state: Состояние реестра
            affected_strategies: Затронутые стратегии
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.registry_operation = registry_operation
        self.registry_state = registry_state
        self.affected_strategies = affected_strategies or []

        if registry_operation:
            self.details["registry_operation"] = registry_operation
        if registry_state:
            self.details["registry_state"] = registry_state
        if affected_strategies:
            self.details["affected_strategies"] = affected_strategies


class StrategyTimeoutError(StrategyError):
    """Ошибка таймаута стратегии."""

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения таймаута.

        Args:
            message: Сообщение об ошибке
            timeout_duration: Длительность таймаута
            operation_type: Тип операции
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation_type = operation_type

        if timeout_duration is not None:
            self.details["timeout_duration"] = timeout_duration
        if operation_type:
            self.details["operation_type"] = operation_type


class StrategyResourceError(StrategyError):
    """Ошибка ресурсов стратегии."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_name: Optional[str] = None,
        resource_limit: Optional[Union[int, float]] = None,
        current_usage: Optional[Union[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения ресурсов.

        Args:
            message: Сообщение об ошибке
            resource_type: Тип ресурса
            resource_name: Имя ресурса
            resource_limit: Лимит ресурса
            current_usage: Текущее использование
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.resource_limit = resource_limit
        self.current_usage = current_usage

        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_name:
            self.details["resource_name"] = resource_name
        if resource_limit is not None:
            self.details["resource_limit"] = resource_limit
        if current_usage is not None:
            self.details["current_usage"] = current_usage


class StrategyRegistrationError(StrategyError):
    """Ошибка регистрации стратегии."""

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        registration_type: Optional[str] = None,
        existing_registration: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения регистрации.

        Args:
            message: Сообщение об ошибке
            strategy_name: Имя стратегии
            registration_type: Тип регистрации
            existing_registration: Существующая регистрация
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.strategy_name = strategy_name
        self.registration_type = registration_type
        self.existing_registration = existing_registration or {}

        if strategy_name:
            self.details["strategy_name"] = strategy_name
        if registration_type:
            self.details["registration_type"] = registration_type
        if existing_registration:
            self.details["existing_registration"] = existing_registration


class StrategyTypeError(StrategyError):
    """Ошибка типа стратегии."""

    def __init__(
        self,
        message: str,
        strategy_type: Optional[str] = None,
        expected_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения типа.

        Args:
            message: Сообщение об ошибке
            strategy_type: Тип стратегии
            expected_types: Ожидаемые типы
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.strategy_type = strategy_type
        self.expected_types = expected_types or []

        if strategy_type:
            self.details["strategy_type"] = strategy_type
        if expected_types:
            self.details["expected_types"] = expected_types


class StrategyParameterError(StrategyError):
    """Ошибка параметра стратегии."""

    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        valid_range: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Инициализация исключения параметра.

        Args:
            message: Сообщение об ошибке
            parameter_name: Имя параметра
            parameter_value: Значение параметра
            expected_type: Ожидаемый тип
            valid_range: Валидный диапазон
            **kwargs: Дополнительные параметры
        """
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
        self.valid_range = valid_range or {}

        if parameter_name:
            self.details["parameter_name"] = parameter_name
        if parameter_value is not None:
            self.details["parameter_value"] = parameter_value
        if expected_type:
            self.details["expected_type"] = expected_type
        if valid_range:
            self.details["valid_range"] = valid_range


# Функции-помощники для создания исключений
def create_strategy_error(error_type: str, message: str, **kwargs: Any) -> StrategyError:
    """Создать исключение стратегии по типу."""
    error_map: Dict[str, Type[StrategyError]] = {
        "creation": StrategyCreationError,
        "validation": StrategyValidationError,
        "execution": StrategyExecutionError,
        "configuration": StrategyConfigurationError,
        "state": StrategyStateError,
        "not_found": StrategyNotFoundError,
        "duplicate": StrategyDuplicateError,
        "performance": StrategyPerformanceError,
        "risk": StrategyRiskError,
        "market": StrategyMarketError,
        "signal": StrategySignalError,
        "factory": StrategyFactoryError,
        "registry": StrategyRegistryError,
        "timeout": StrategyTimeoutError,
        "resource": StrategyResourceError,
        "registration": StrategyRegistrationError,
        "type": StrategyTypeError,
        "parameter": StrategyParameterError,
    }

    error_class = error_map.get(error_type, StrategyError)
    return error_class(message, **kwargs)


def is_strategy_error(exception: Exception) -> bool:
    """
    Проверить, является ли исключение ошибкой стратегии.

    Args:
        exception: Исключение для проверки

    Returns:
        bool: True если это ошибка стратегии
    """
    return isinstance(exception, StrategyError)


def get_strategy_error_details(exception: Exception) -> Optional[Dict[str, Any]]:
    """
    Получить детали ошибки стратегии.

    Args:
        exception: Исключение

    Returns:
        Optional[Dict[str, Any]]: Детали ошибки или None
    """
    if isinstance(exception, StrategyError):
        return exception.to_dict()
    return None
