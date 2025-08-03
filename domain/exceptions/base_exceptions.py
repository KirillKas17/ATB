"""
Базовые исключения домена.
"""

from typing import Optional, Any


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
    def __init__(self, message: str = "Validation error", field: Optional[str] = None, value: Optional[Any] = None):
        self.field = field
        self.value = value
        super().__init__(f"{message}: {field} = {value}")
