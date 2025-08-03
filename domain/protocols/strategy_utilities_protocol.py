"""
Протокол утилит стратегий.
Обеспечивает вспомогательные функции для торговых стратегий.
"""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Protocol, runtime_checkable
from typing import List

import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from domain.entities.position import Position


@runtime_checkable
class StrategyUtilitiesProtocol(Protocol):
    """Протокол утилит стратегий."""

    async def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> Decimal: ...
    async def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Decimal: ...
    async def _calculate_win_rate(self, positions: List["Position"]) -> Decimal: ...
    async def _calculate_profit_factor(self, positions: List["Position"]) -> Decimal: ...
    async def _validate_market_data(self, data: pd.DataFrame) -> bool: ...


class StrategyUtilitiesProtocolImpl(ABC):
    """Реализация протокола утилит стратегий."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> Decimal:
        """Расчет коэффициента Шарпа."""
        pass

    @abstractmethod
    async def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Decimal:
        """Расчет максимальной просадки."""
        pass

    @abstractmethod
    async def _calculate_win_rate(self, positions: List["Position"]) -> Decimal:
        """Рассчитать процент выигрышных сделок."""
        pass

    @abstractmethod
    async def _calculate_profit_factor(self, positions: List["Position"]) -> Decimal:
        """Рассчитать фактор прибыли."""
        pass

    @abstractmethod
    async def _validate_market_data(self, data: pd.DataFrame) -> bool:
        """Валидация рыночных данных."""
        pass 