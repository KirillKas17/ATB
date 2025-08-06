"""
Use case для оркестрации торговли.
"""

from typing import List, Type, Union

# Импортируем основной класс из core.py
from .trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    TradingOrchestratorUseCase,
)

# Импортируем DTO из нового модуля
from .trading_orchestrator.dtos import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
    ProcessSignalResponse,
    TradingSession,
)

# Экспортируем все публичные интерфейсы
__all__: List[str] = [
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
    "PortfolioRebalanceRequest",
    "PortfolioRebalanceResponse",
    "TradingSession",
]
