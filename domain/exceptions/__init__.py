"""
Исключения домена.
"""

from typing import Optional

from .base_exceptions import (
    AuthenticationError,
    BusinessRuleError,
    ConnectionError,
    DomainError,
    DomainException,
    DuplicateEntityError,
    EntityDeleteError,
    EvaluationError,
    ExchangeError,
    InsufficientFundsError,
    InsufficientPositionError,
    InvalidModelTypeError,
    InvalidPortfolioError,
    InvalidPositionError,
    InvalidTradingPairError,
    MarketDataError,
    MarketError,
    MLModelError,
    ModelSaveError,
    NetworkError,
    OrderError,
    OrderManagementError,
    PortfolioError,
    PortfolioNotFoundError,
    PositionError,
    PositionManagementError,
    PositionNotFoundError,
    PredictionError,
    RepositoryError,
    RiskAnalysisError,
    RiskManagementError,
    StrategyError,
    StrategyExecutionError,
    TechnicalAnalysisError,
    TradingError,
    TradingOrchestrationError,
    TradingPairManagementError,
    UseCaseError,
)
from .protocol_exceptions import (
    ConfigurationError,
    DatabaseConnectionError,
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeRateLimitError,
    InsufficientBalanceError,
    InsufficientCapitalError,
    InsufficientDataError,
    InvalidFeaturesError,
    InvalidOrderError,
    ModelLoadError,
    ModelNotFoundError,
    ModelNotReadyError,
    OrderAlreadyFilledError,
    OrderNotFoundError,
    ProtocolError,
    RetryExhaustedError,
    SignalGenerationError,
    StrategyNotFoundError,
    SymbolNotFoundError,
    TimeoutError,
    TrainingError,
    TransactionError,
    ValidationError,
)


class MarketAnalysisError(DomainError):
    """Exception raised for market analysis errors."""

    def __init__(
        self,
        message: str,
        analysis_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        super().__init__(message)
        self.analysis_type = analysis_type
        self.symbol = symbol


class CurrencyNetworkError(DomainError):
    """Exception raised for currency network errors."""

    def __init__(self, message: str, currency: Optional[str] = None):
        super().__init__(message)
        self.currency = currency


class OrderBookError(DomainError):
    """Exception raised for order book errors."""

    def __init__(self, message: str, symbol: Optional[str] = None):
        super().__init__(message)
        self.symbol = symbol


__all__ = [
    # Базовые исключения
    "DomainException",
    "DomainError",
    "ExchangeError",
    "OrderError",
    "PositionError",
    "StrategyError",
    "TradingError",
    "PortfolioError",
    "MarketError",
    "MLModelError",
    "NetworkError",
    "UseCaseError",
    "RepositoryError",
    "DuplicateEntityError",
    "ConnectionError",
    "BusinessRuleError",
    "InsufficientFundsError",
    "InvalidTradingPairError",
    "RiskManagementError",
    "OrderManagementError",
    "PositionManagementError",
    "TradingPairManagementError",
    "InsufficientPositionError",
    "InvalidPortfolioError",
    "PortfolioNotFoundError",
    "MarketDataError",
    "TechnicalAnalysisError",
    "RiskAnalysisError",
    "InvalidPositionError",
    "PositionNotFoundError",
    "InvalidModelTypeError",
    "PredictionError",
    "AuthenticationError",
    "EvaluationError",
    "ModelSaveError",
    "EntityDeleteError",
    "TradingOrchestrationError",
    "StrategyExecutionError",
    # Исключения протоколов
    "ProtocolError",
    "ExchangeConnectionError",
    "ExchangeAuthenticationError",
    "ExchangeRateLimitError",
    "SymbolNotFoundError",
    "InvalidOrderError",
    "InsufficientBalanceError",
    "OrderNotFoundError",
    "OrderAlreadyFilledError",
    "ModelNotFoundError",
    "ModelNotReadyError",
    "TrainingError",
    "InsufficientDataError",
    "InvalidFeaturesError",
    "ModelLoadError",
    "StrategyNotFoundError",
    "SignalGenerationError",
    "InsufficientCapitalError",
    "EntitySaveError",
    "EntityUpdateError",
    "DatabaseConnectionError",
    "TransactionError",
    "TimeoutError",
    "RetryExhaustedError",
    "ConfigurationError",
    "ValidationError",
    "EntityNotFoundError",
    "MarketAnalysisError",
    "CurrencyNetworkError",
    "OrderBookError",
]
