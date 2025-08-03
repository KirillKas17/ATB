"""
Use cases приложения.
Модуль предоставляет все основные use cases для управления торговыми операциями,
позициями, рисками, торговыми парами и оркестрации торговли.
"""

from .manage_positions import (
    DefaultPositionManagementUseCase,
    PositionManagementUseCase,
)
from .manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
    TradingPairManagementUseCase,
)
from .trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    TradingOrchestratorUseCase,
)
# from .trading_orchestrator.dtos import (
#     ExecuteStrategyRequest,
#     ExecuteStrategyResponse,
#     PortfolioRebalanceRequest,
#     PortfolioRebalanceResponse,
#     ProcessSignalRequest,
#     ProcessSignalResponse,
#     TradingSession,
# )

__all__ = [
    # Order Management
    "OrderManagementUseCase",
    "DefaultOrderManagementUseCase",
    # Position Management
    "PositionManagementUseCase",
    "DefaultPositionManagementUseCase",
    # Risk Management
    "RiskManagementUseCase",
    "DefaultRiskManagementUseCase",
    # Trading Pair Management
    "TradingPairManagementUseCase",
    "DefaultTradingPairManagementUseCase",
    # Trading Orchestrator
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    # "ExecuteStrategyRequest",
    # "ExecuteStrategyResponse", 
    # "ProcessSignalRequest",
    # "ProcessSignalResponse",
    # "PortfolioRebalanceRequest",
    # "PortfolioRebalanceResponse",
    # "TradingSession",
]
