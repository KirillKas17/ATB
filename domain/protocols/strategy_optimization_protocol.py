"""
Протокол оптимизации стратегий.
Обеспечивает оптимизацию и адаптацию торговых стратегий.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from domain.type_definitions import StrategyId
from domain.type_definitions.protocol_types import StrategyAdaptationRules


@runtime_checkable
class StrategyOptimizationProtocol(Protocol):
    """Протокол оптимизации стратегий."""

    async def update_strategy_parameters(
        self,
        strategy_id: StrategyId,
        parameters: Dict[str, float],
        validation_period: Optional[int] = None,
    ) -> bool: ...
    async def optimize_strategy_parameters(
        self,
        strategy_id: StrategyId,
        historical_data: pd.DataFrame,
        optimization_target: str = "sharpe_ratio",
        param_ranges: Optional[Dict[str, List[float]]] = None,
        optimization_method: str = "genetic_algorithm",
    ) -> Dict[str, float]: ...
    async def adapt_strategy_to_market(
        self,
        strategy_id: StrategyId,
        market_regime: str,
        adaptation_rules: StrategyAdaptationRules,
    ) -> bool: ...
    async def backtest_strategy(
        self,
        strategy_id: StrategyId,
        historical_data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
        transaction_cost: Decimal = Decimal("0.001"),
    ) -> Dict[str, float]: ...


class StrategyOptimizationProtocolImpl(ABC):
    """Реализация протокола оптимизации стратегий."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def update_strategy_parameters(
        self,
        strategy_id: StrategyId,
        parameters: Dict[str, float],
        validation_period: Optional[int] = None,
    ) -> bool:
        """Обновление параметров стратегии."""
        pass

    @abstractmethod
    async def optimize_strategy_parameters(
        self,
        strategy_id: StrategyId,
        historical_data: pd.DataFrame,
        optimization_target: str = "sharpe_ratio",
        param_ranges: Optional[Dict[str, List[float]]] = None,
        optimization_method: str = "genetic_algorithm",
    ) -> Dict[str, float]:
        """Оптимизация параметров стратегии."""
        pass

    @abstractmethod
    async def adapt_strategy_to_market(
        self,
        strategy_id: StrategyId,
        market_regime: str,
        adaptation_rules: StrategyAdaptationRules,
    ) -> bool:
        """Адаптация стратегии к рынку."""
        pass

    @abstractmethod
    async def backtest_strategy(
        self,
        strategy_id: StrategyId,
        historical_data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
        transaction_cost: Decimal = Decimal("0.001"),
    ) -> Dict[str, float]:
        """Бэктестинг стратегии."""
        pass 