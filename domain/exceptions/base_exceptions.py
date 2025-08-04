"""
Базовые исключения домена.
"""

from typing import Optional, Any, Dict
from datetime import datetime


def _format_error_message(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Форматирует сообщение об ошибке с контекстом."""
    if not context:
        return message
    
    context_str = ", ".join(f"{key}={value}" for key, value in context.items())
    return f"{message} ({context_str})"


def _get_error_context(context: Optional[Dict[str, Any]] = None, field: Optional[str] = None, 
                      value: Optional[Any] = None, rule: Optional[str] = None, 
                      **kwargs: Any) -> Dict[str, Any]:
    """Создает контекст ошибки."""
    if context is not None:
        return context
    
    result = {}
    if field is not None:
        result["field"] = field
    if value is not None:
        result["value"] = value
    if rule is not None:
        result["rule"] = rule
    result.update(kwargs)
    return result


class BaseDomainException(Exception):
    """Базовое исключение домена."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()
        self.cause = cause
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует исключение в словарь."""
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


# Совместимость: DomainException = BaseDomainException  
class DomainException(BaseDomainException):
    """Базовое исключение домена (совместимость)."""
    pass


class DomainError(BaseDomainException):
    """Общая ошибка домена."""
    pass


class ConfigurationError(BaseDomainException):
    """Исключение для ошибок конфигурации."""
    
    def __init__(self, config_key: str, message: str, context: Optional[Dict[str, Any]] = None):
        self.config_key = config_key
        
        full_message = f"Configuration error for '{config_key}': {message}"
        context = context or {}
        context.update({"config_key": config_key})
        super().__init__(full_message, context)


class BusinessRuleError(BaseDomainException):
    """Исключение для нарушений бизнес-правил."""
    pass


class BusinessRuleViolationError(BusinessRuleError):
    """Исключение для нарушений бизнес-правил (совместимость)."""
    
    def __init__(self, rule_name: str, details: str, context: Optional[Dict[str, Any]] = None):
        self.rule_name = rule_name
        self.details = details
        
        message = f"Business rule violation: {rule_name} - {details}"
        context = context or {}
        context.update({"rule_name": rule_name, "details": details})
        super().__init__(message, context)


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
    
    def __init__(self, entity_type: str, entity_id: Optional[str] = None, 
                 criteria: Optional[Dict[str, Any]] = None, message: Optional[str] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.criteria = criteria
        
        if message is None:
            if entity_id:
                message = f"Entity '{entity_type}' with id '{entity_id}' not found"
            elif criteria:
                criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
                message = f"Entity '{entity_type}' not found with criteria: {criteria_str}"
            else:
                message = f"Entity '{entity_type}' not found"
        
        context = _get_error_context(entity_type=entity_type, entity_id=entity_id, criteria=criteria)
        super().__init__(message, context)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует исключение в словарь."""
        result = super().to_dict()
        result.update({
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "criteria": self.criteria
        })
        return result


class EntitySaveError(RepositoryError):
    """Исключение для ошибок сохранения сущности в репозитории."""
    def __init__(self, message: str = "Entity save error", entity_type: Optional[str] = None, entity_id: Optional[str] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{message}: {entity_type} [{entity_id}]")


class ValidationError(BaseDomainException):
    """Исключение для ошибок валидации."""
    
    def __init__(self, field: str, value: Any, rule: str, message: Optional[str] = None):
        self.field = field
        self.value = value
        self.rule = rule
        
        if message is None:
            message = f"Validation failed for field '{field}' with value '{value}' (rule: {rule})"
        
        context = _get_error_context(field=field, value=value, rule=rule)
        super().__init__(message, context)
