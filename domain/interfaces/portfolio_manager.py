"""
Протокол для управления портфелем.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from decimal import Decimal
from datetime import datetime

class PortfolioManagerProtocol(Protocol):
    """Протокол для управления портфелем."""
    
    async def get_balance(self, asset: str) -> Decimal:
        """Получение баланса актива."""
        ...
    
    async def get_total_balance(self) -> Dict[str, Decimal]:
        """Получение общего баланса."""
        ...
    
    async def get_available_balance(self, asset: str) -> Decimal:
        """Получение доступного баланса."""
        ...
    
    async def get_locked_balance(self, asset: str) -> Decimal:
        """Получение заблокированного баланса."""
        ...
    
    async def get_portfolio_value(self, base_currency: str = "USDT") -> Decimal:
        """Получение общей стоимости портфеля."""
        ...
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение текущих позиций."""
        ...
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение позиции по символу."""
        ...
    
    async def calculate_pnl(self, symbol: str) -> Decimal:
        """Расчет PnL по позиции."""
        ...
    
    async def get_allocation(self) -> Dict[str, float]:
        """Получение распределения активов."""
        ...
    
    async def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> None:
        """Ребалансировка портфеля."""
        ...
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Получение метрик риска."""
        ...

class BasePortfolioManager(ABC):
    """Базовый класс для менеджера портфеля."""
    
    def __init__(self):
        self._balances = {}
        self._positions = {}
        self._last_update = None
    
    @abstractmethod
    async def update_balances(self) -> None:
        """Обновление балансов."""
        pass
    
    @abstractmethod
    async def update_positions(self) -> None:
        """Обновление позиций."""
        pass
    
    @property
    def last_update(self) -> Optional[datetime]:
        """Время последнего обновления."""
        return self._last_update