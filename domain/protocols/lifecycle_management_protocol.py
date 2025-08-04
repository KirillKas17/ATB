"""
Протокол управления жизненным циклом стратегий.
Обеспечивает управление состоянием и жизненным циклом торговых стратегий.
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from domain.type_definitions import StrategyId


@runtime_checkable
class LifecycleManagementProtocol(Protocol):
    """Протокол управления жизненным циклом."""

    async def activate_strategy(self, strategy_id: StrategyId) -> bool: ...
    async def deactivate_strategy(self, strategy_id: StrategyId) -> bool: ...
    async def pause_strategy(self, strategy_id: StrategyId) -> bool: ...
    async def resume_strategy(self, strategy_id: StrategyId) -> bool: ...
    async def emergency_stop(
        self, strategy_id: StrategyId, reason: str = "emergency_stop"
    ) -> bool: ...


class LifecycleManagementProtocolImpl(ABC):
    """Реализация протокола управления жизненным циклом."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def activate_strategy(self, strategy_id: StrategyId) -> bool:
        """Активация стратегии."""
        pass

    @abstractmethod
    async def deactivate_strategy(self, strategy_id: StrategyId) -> bool:
        """Деактивация стратегии."""
        pass

    @abstractmethod
    async def pause_strategy(self, strategy_id: StrategyId) -> bool:
        """Приостановка стратегии."""
        pass

    @abstractmethod
    async def resume_strategy(self, strategy_id: StrategyId) -> bool:
        """Возобновление стратегии."""
        pass

    @abstractmethod
    async def emergency_stop(
        self, strategy_id: StrategyId, reason: str = "emergency_stop"
    ) -> bool:
        """Экстренная остановка стратегии."""
        pass 