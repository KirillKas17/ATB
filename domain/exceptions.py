from typing import Any, Dict, List, Optional, Union
"""
Доменные исключения.
"""


class DomainError(Exception):
    """Базовое исключение доменного слоя."""


class DomainException(DomainError):
    """Алиас для DomainError для совместимости."""


class UseCaseError(DomainError):
    """Ошибка use case."""


class ValidationError(DomainError):
    """Ошибка валидации."""


class BusinessRuleError(DomainError):
    """Ошибка бизнес-правила."""


class RepositoryError(DomainError):
    """Ошибка репозитория."""


class TradingError(DomainError):
    """Ошибка торговых операций."""


class InsufficientFundsError(TradingError):
    """Недостаточно средств."""


class InvalidOrderError(TradingError):
    """Некорректный ордер."""


class OrderNotFoundError(TradingError):
    """Ордер не найден."""


class StrategyError(DomainError):
    """Ошибка стратегии."""


class MLModelError(DomainError):
    """Ошибка ML модели."""


class NetworkError(DomainError):
    """Ошибка сети."""


class ExchangeError(DomainError):
    """Ошибка биржи."""


class ExchangeConnectionError(ExchangeError):
    """Ошибка подключения к бирже."""


class AuthenticationError(ExchangeError):
    """Ошибка аутентификации на бирже."""


class NotFoundError(DomainError):
    """Сущность не найдена."""


class EntityNotFoundError(NotFoundError):
    """Алиас для NotFoundError для совместимости."""


class TradingPairManagementError(DomainError):
    """Ошибка управления торговыми парами."""


class InvalidTradingPairError(DomainError):
    """Ошибка некорректной торговой пары."""


class OrderManagementError(DomainError):
    """Ошибка управления ордерами."""


class PositionManagementError(DomainError):
    """Ошибка управления позициями."""


class InsufficientPositionError(DomainError):
    """Недостаточно позиции."""


class RiskAnalysisError(DomainError):
    """Ошибка анализа рисков."""


class RiskManagementError(DomainError):
    """Ошибка управления рисками."""


class TradingOrchestrationError(DomainError):
    """Ошибка оркестрации торговых стратегий."""


class StrategyExecutionError(DomainError):
    """Ошибка исполнения стратегии."""
