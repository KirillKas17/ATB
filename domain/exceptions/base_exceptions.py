"""
Базовые исключения домена.
"""

from typing import Optional, Any, Dict


# Алиас для совместимости с тестами
BaseDomainException = type("DomainException", (Exception,), {})


def _format_error_message(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Форматирование сообщения об ошибке с контекстом."""
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        return f"{message} (context: {context_str})"
    return message


def _get_error_context(error: Exception) -> Dict[str, Any]:
    """Получение контекста ошибки."""
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    # Добавляем дополнительные атрибуты ошибки если они есть
    for attr in ["field", "value", "entity_type", "entity_id"]:
        if hasattr(error, attr):
            context[attr] = getattr(error, attr)
    
    return context


class DomainException(Exception):
    """Базовое исключение домена."""

    pass


class DomainError(DomainException):
    """Общая ошибка домена."""

    pass


class ExchangeError(DomainException):
    """Базовое исключение для ошибок биржи."""

    pass


class OrderError(DomainException):
    """Базовое исключение для ошибок ордеров."""

    pass


class PositionError(DomainException):
    """Базовое исключение для ошибок позиций."""

    pass


class StrategyError(DomainException):
    """Базовое исключение для ошибок стратегий."""

    pass


class TradingError(DomainException):
    """Базовое исключение для ошибок торговли."""

    pass


class PortfolioError(DomainException):
    """Базовое исключение для ошибок портфеля."""

    pass


class MarketError(DomainException):
    """Базовое исключение для ошибок рынка."""

    pass


class MLModelError(DomainException):
    """Базовое исключение для ошибок ML моделей."""

    pass


class NetworkError(DomainException):
    """Базовое исключение для сетевых ошибок."""

    pass


class ConnectionError(DomainException):
    """Базовое исключение для ошибок подключения."""

    pass


class UseCaseError(DomainException):
    """Базовое исключение для ошибок use case."""

    pass


class RepositoryError(DomainException):
    """Базовое исключение для ошибок репозитория."""

    pass


class DuplicateEntityError(RepositoryError):
    """Исключение для дублирования сущности."""

    pass


class RepositoryConnectionError(RepositoryError):
    """Исключение для ошибок подключения к репозиторию."""

    pass


class BusinessRuleError(DomainException):
    """Исключение для нарушений бизнес-правил."""

    pass


# Алиас для совместимости с тестами
BusinessRuleViolationError = BusinessRuleError


class ConfigurationError(DomainException):
    """Исключение для ошибок конфигурации."""
    
    def __init__(self, message: str = "Configuration error", config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(f"{message}: {config_key}" if config_key else message)


class InsufficientFundsError(DomainException):
    """Исключение для недостаточных средств."""

    pass


class InvalidTradingPairError(DomainException):
    """Исключение для неверной торговой пары."""

    pass


class RiskManagementError(DomainException):
    """Исключение для ошибок управления рисками."""

    pass


class OrderManagementError(DomainException):
    """Исключение для ошибок управления ордерами."""

    pass


class PositionManagementError(DomainException):
    """Исключение для ошибок управления позициями."""

    pass


class TradingPairManagementError(DomainException):
    """Исключение для ошибок управления торговыми парами."""

    pass


class InsufficientPositionError(DomainException):
    """Исключение для недостаточной позиции."""

    pass


class InvalidPortfolioError(DomainException):
    """Исключение для неверного портфеля."""

    pass


class PortfolioNotFoundError(DomainException):
    """Исключение для ненайденного портфеля."""

    pass


class MarketDataError(DomainException):
    """Исключение для ошибок рыночных данных."""

    pass


class TechnicalAnalysisError(DomainException):
    """Исключение для ошибок технического анализа."""

    pass


class RiskAnalysisError(DomainException):
    """Исключение для ошибок анализа рисков."""

    pass


class InvalidPositionError(DomainException):
    """Исключение для неверной позиции."""

    pass


class PositionNotFoundError(DomainException):
    """Исключение для ненайденной позиции."""

    pass


class InvalidModelTypeError(DomainException):
    """Исключение для неверного типа модели."""

    pass


class PredictionError(DomainException):
    """Исключение для ошибок предсказаний."""

    pass


class AuthenticationError(DomainException):
    """Исключение для ошибок аутентификации."""

    pass


class EvaluationError(DomainException):
    """Исключение для ошибок оценки."""

    pass


class ModelSaveError(DomainException):
    """Исключение для ошибок сохранения модели."""

    pass


class EntityDeleteError(DomainException):
    """Исключение для ошибок удаления сущности."""

    pass


class TradingOrchestrationError(DomainException):
    """Исключение для ошибок оркестрации торговых стратегий."""

    pass


class StrategyExecutionError(DomainException):
    """Исключение для ошибок исполнения стратегии."""

    pass


class EntityNotFoundError(RepositoryError):
    """Исключение для ненайденной сущности в репозитории."""
    def __init__(self, message: str = "Entity not found", entity_type: Optional[str] = None, entity_id: Optional[str] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{message}: {entity_type} [{entity_id}]")


class EntitySaveError(RepositoryError):
    """Исключение для ошибок сохранения сущности в репозитории."""
    def __init__(self, message: str = "Entity save error", entity_type: Optional[str] = None, entity_id: Optional[str] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{message}: {entity_type} [{entity_id}]")


class ValidationError(DomainException):
    """Исключение для ошибок валидации."""
    def __init__(self, message: str = "Validation error", field: Optional[str] = None, value: Optional[Any] = None, rule: Optional[str] = None):
        self.field = field
        self.value = value
        self.rule = rule
        detail = f": {field} = {value}" if field else ""
        if rule:
            detail += f" (rule: {rule})"
        super().__init__(f"{message}{detail}")
