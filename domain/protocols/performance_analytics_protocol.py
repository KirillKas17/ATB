"""
Протокол аналитики производительности.
Обеспечивает анализ и мониторинг производительности торговых стратегий.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from domain.entities.trade import Trade
from domain.types import StrategyId


@runtime_checkable
class PerformanceAnalyticsProtocol(Protocol):
    """Протокол аналитики производительности."""

    async def get_strategy_performance(
        self,
        strategy_id: StrategyId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]: ...
    async def calculate_performance_metrics(
        self, trades: List[Trade], initial_capital: Decimal
    ) -> Dict[str, Any]: ...
    async def monitor_strategy_health(
        self, strategy_id: StrategyId, market_data: pd.DataFrame
    ) -> Dict[str, float]: ...
    async def detect_strategy_drift(
        self,
        strategy_id: StrategyId,
        recent_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, float]: ...


class PerformanceAnalyticsProtocolImpl(ABC):
    """Реализация протокола аналитики производительности."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def get_strategy_performance(
        self,
        strategy_id: StrategyId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Получение производительности стратегии."""
        pass

    @abstractmethod
    async def calculate_performance_metrics(
        self, trades: List[Trade], initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Расчет метрик производительности."""
        pass

    @abstractmethod
    async def monitor_strategy_health(
        self, strategy_id: StrategyId, market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Мониторинг здоровья стратегии."""
        pass

    @abstractmethod
    async def detect_strategy_drift(
        self,
        strategy_id: StrategyId,
        recent_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Обнаружение дрифта стратегии."""
        pass 