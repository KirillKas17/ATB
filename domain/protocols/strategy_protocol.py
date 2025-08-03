"""
Базовый протокол для торговых стратегий.
Объединяет все специализированные протоколы в единый интерфейс.
"""

import logging
from abc import ABC
from typing import Dict, Optional, TypeVar, Any

from domain.types import StrategyId
from domain.entities.strategy import Strategy
from domain.entities.signal import Signal
from domain.types import PerformanceMetrics
from domain.protocols.market_analysis_protocol import MarketAnalysisProtocolImpl
from domain.protocols.signal_generation_protocol import SignalGenerationProtocolImpl
from domain.protocols.strategy_execution_protocol import StrategyExecutionProtocolImpl
from domain.protocols.risk_management_protocol import RiskManagementProtocolImpl
from domain.protocols.performance_analytics_protocol import PerformanceAnalyticsProtocolImpl
from domain.protocols.strategy_optimization_protocol import StrategyOptimizationProtocolImpl
from domain.protocols.lifecycle_management_protocol import LifecycleManagementProtocolImpl
from domain.protocols.error_handling_protocol import ErrorHandlingProtocolImpl
from domain.protocols.strategy_utilities_protocol import StrategyUtilitiesProtocolImpl


T = TypeVar("T", bound=Strategy)


class StrategyProtocol(
    MarketAnalysisProtocolImpl,
    SignalGenerationProtocolImpl,
    StrategyExecutionProtocolImpl,
    RiskManagementProtocolImpl,
    PerformanceAnalyticsProtocolImpl,
    StrategyOptimizationProtocolImpl,
    LifecycleManagementProtocolImpl,
    ErrorHandlingProtocolImpl,
    StrategyUtilitiesProtocolImpl,
    ABC
):
    """
    Базовый протокол для торговых стратегий.
    
    Объединяет все специализированные протоколы в единый интерфейс,
    обеспечивая полный жизненный цикл стратегии от анализа рынка
    до исполнения и мониторинга.
    """

    def __init__(self) -> None:
        """Инициализация протокола стратегий."""
        self.logger = logging.getLogger(__name__)
        self._strategies_cache: Dict[StrategyId, Strategy] = {}
        self._signals_cache: Dict[str, "Signal"] = {}
        self._performance_history: Dict[StrategyId, list["PerformanceMetrics"]] = {}
        self._risk_monitors: Dict[StrategyId, Dict[str, Any]] = {}

    async def _cache_strategy(self, strategy: Strategy) -> None:
        """Кэширование стратегии."""
        self._strategies_cache[StrategyId(strategy.id)] = strategy

    async def _get_cached_strategy(self, strategy_id: StrategyId) -> Optional[Strategy]:
        """Получение стратегии из кэша."""
        return self._strategies_cache.get(strategy_id)

    async def _clear_strategy_cache(
        self, strategy_id: Optional[StrategyId] = None
    ) -> None:
        """Очистка кэша стратегий."""
        if strategy_id:
            self._strategies_cache.pop(strategy_id, None)
        else:
            self._strategies_cache.clear()
