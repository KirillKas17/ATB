"""
Протокол управления рисками.
Обеспечивает комплексное управление рисками торговых стратегий.
"""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from domain.entities.position import Position
from domain.entities.signal import Signal


@runtime_checkable
class RiskManagementProtocol(Protocol):
    """Протокол управления рисками."""

    async def validate_risk_limits(
        self,
        signal: Signal,
        current_positions: List[Position],
        risk_limits: Dict[str, float],
    ) -> bool: ...
    async def calculate_portfolio_risk(
        self, positions: List[Position], market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]: ...
    async def apply_risk_filters(
        self, signal: Signal, market_conditions: Dict[str, float]
    ) -> bool: ...
    async def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> Decimal: ...
    async def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> Decimal: ...


class RiskManagementProtocolImpl(ABC):
    """Реализация протокола управления рисками."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def validate_risk_limits(
        self,
        signal: Signal,
        current_positions: List[Position],
        risk_limits: Dict[str, float],
    ) -> bool:
        """Валидация лимитов риска."""
        pass

    @abstractmethod
    async def calculate_portfolio_risk(
        self, positions: List[Position], market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Расчет риска портфеля."""
        pass

    @abstractmethod
    async def apply_risk_filters(
        self, signal: Signal, market_conditions: Dict[str, float]
    ) -> bool:
        """Применение фильтров риска."""
        pass

    @abstractmethod
    async def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> Decimal:
        """Расчет Value at Risk."""
        pass

    @abstractmethod
    async def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> Decimal:
        """Расчет Conditional Value at Risk."""
        pass 