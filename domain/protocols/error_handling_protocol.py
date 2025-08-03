"""
Протокол обработки ошибок стратегий.
Обеспечивает обработку ошибок и восстановление состояния стратегий.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable, Optional

from domain.types import StrategyId
from domain.types.protocol_types import StrategyErrorContext


@runtime_checkable
class ErrorHandlingProtocol(Protocol):
    """Протокол обработки ошибок."""

    async def handle_strategy_error(
        self,
        strategy_id: StrategyId,
        error: Exception,
        error_context: StrategyErrorContext,
    ) -> bool: ...
    async def recover_strategy_state(
        self, strategy_id: StrategyId, recovery_point: Optional[datetime] = None
    ) -> bool: ...
    async def validate_strategy_integrity(self, strategy_id: StrategyId) -> bool: ...


class ErrorHandlingProtocolImpl(ABC):
    """Реализация протокола обработки ошибок."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def handle_strategy_error(
        self,
        strategy_id: StrategyId,
        error: Exception,
        error_context: StrategyErrorContext,
    ) -> bool:
        """Обработка ошибки стратегии."""
        pass

    @abstractmethod
    async def recover_strategy_state(
        self, strategy_id: StrategyId, recovery_point: Optional[datetime] = None
    ) -> bool:
        """Восстановление состояния стратегии."""
        pass

    @abstractmethod
    async def validate_strategy_integrity(self, strategy_id: StrategyId) -> bool:
        """Валидация целостности стратегии."""
        pass 