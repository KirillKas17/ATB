"""
Протокол исполнения стратегий.
Обеспечивает исполнение торговых сигналов, управление позициями и мониторинг.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

from domain.entities.market import MarketData
from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.signal import Signal
from domain.type_definitions import PriceValue, VolumeValue


@runtime_checkable
class StrategyExecutionProtocol(Protocol):
    """Протокол исполнения стратегии."""

    async def validate_execution_conditions(self, signal: Signal) -> bool: ...
    async def calculate_position_size(
        self, signal: Signal, capital: Decimal
    ) -> VolumeValue: ...
    async def determine_entry_price(
        self, signal: Signal, market_data: MarketData
    ) -> PriceValue: ...
    async def set_stop_loss_take_profit(
        self, signal: Signal
    ) -> Tuple[PriceValue, PriceValue]: ...
    async def execute_order(self, order: Order) -> bool: ...
    async def monitor_position(self, position: Position) -> bool: ...


class StrategyExecutionProtocolImpl(ABC):
    """Реализация протокола исполнения стратегий."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def execute_strategy(
        self,
        strategy_id: str,
        signal: Signal,
        execution_params: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Исполнение стратегии."""
        pass

    @abstractmethod
    async def create_order_from_signal(
        self, signal: Signal, account_balance: Decimal, risk_params: Dict[str, float]
    ) -> Order:
        """Создание ордера из сигнала."""
        pass

    @abstractmethod
    async def calculate_position_size(
        self,
        signal: Signal,
        account_balance: Decimal,
        risk_per_trade: Decimal = Decimal("0.02"),
    ) -> VolumeValue:
        """Расчет размера позиции."""
        pass

    @abstractmethod
    async def set_stop_loss_take_profit(
        self, signal: Signal, entry_price: PriceValue, atr_multiplier: float = 2.0
    ) -> Tuple[PriceValue, PriceValue]:
        """Установка стоп-лосса и тейк-профита."""
        pass

    @abstractmethod
    async def monitor_position(
        self, position: Position, market_data: MarketData
    ) -> Dict[str, float]:
        """Мониторинг позиции."""
        pass 