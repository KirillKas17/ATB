"""
Протокол для управления рисками.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from decimal import Decimal
from datetime import datetime
from enum import Enum

class RiskLevel(Enum):
    """Уровни риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskManagerProtocol(Protocol):
    """Протокол для управления рисками."""
    
    async def assess_trade_risk(self, symbol: str, quantity: Decimal, 
                               side: str, price: Optional[Decimal] = None) -> Dict[str, Any]:
        """Оценка риска сделки."""
        ...
    
    async def check_position_limits(self, symbol: str, quantity: Decimal) -> bool:
        """Проверка лимитов позиции."""
        ...
    
    async def check_daily_loss_limit(self) -> bool:
        """Проверка дневного лимита потерь."""
        ...
    
    async def calculate_position_size(self, symbol: str, risk_percentage: float) -> Decimal:
        """Расчет размера позиции на основе риска."""
        ...
    
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        """Получение портфельного риска."""
        ...
    
    async def get_var(self, confidence_level: float = 0.95, 
                     time_horizon: int = 1) -> Decimal:
        """Расчет Value at Risk."""
        ...
    
    async def get_max_drawdown(self) -> Decimal:
        """Получение максимальной просадки."""
        ...
    
    async def check_correlation_risk(self, symbols: List[str]) -> Dict[str, float]:
        """Проверка корреляционного риска."""
        ...
    
    async def set_stop_loss(self, symbol: str, price: Decimal) -> bool:
        """Установка стоп-лосса."""
        ...
    
    async def set_take_profit(self, symbol: str, price: Decimal) -> bool:
        """Установка тейк-профита."""
        ...
    
    async def emergency_close_all(self) -> bool:
        """Экстренное закрытие всех позиций."""
        ...
    
    async def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Получение предупреждений о рисках."""
        ...

class BaseRiskManager(ABC):
    """Базовый класс для менеджера рисков."""
    
    def __init__(self):
        self._max_position_size = Decimal('0.1')  # 10% от портфеля
        self._max_daily_loss = Decimal('0.02')    # 2% дневная потеря
        self._max_drawdown = Decimal('0.05')      # 5% максимальная просадка
        self._risk_alerts = []
        self._emergency_mode = False
    
    @abstractmethod
    async def update_risk_metrics(self) -> None:
        """Обновление метрик риска."""
        pass
    
    @abstractmethod
    async def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """Расчет портфельного риска."""
        pass
    
    @property
    def emergency_mode(self) -> bool:
        """Режим экстренной остановки."""
        return self._emergency_mode
    
    @property
    def max_position_size(self) -> Decimal:
        """Максимальный размер позиции."""
        return self._max_position_size
    
    @property
    def max_daily_loss(self) -> Decimal:
        """Максимальная дневная потеря."""
        return self._max_daily_loss
    
    def activate_emergency_mode(self) -> None:
        """Активация экстренного режима."""
        self._emergency_mode = True
    
    def deactivate_emergency_mode(self) -> None:
        """Деактивация экстренного режима."""
        self._emergency_mode = False