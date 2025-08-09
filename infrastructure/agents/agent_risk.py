"""
Агент управления рисками - основной файл.
Импортирует функциональность из модульной архитектуры.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal

from domain.type_definitions import PortfolioId, Symbol
from domain.value_objects.money import Money


@dataclass
class RiskMetrics:
    """Метрики риска."""
    
    portfolio_id: PortfolioId
    total_value: Money
    var_95: Money
    var_99: Money
    max_drawdown: Decimal
    volatility: Decimal
    beta: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "portfolio_id": str(self.portfolio_id),
            "total_value": {
                "amount": str(self.total_value.amount),
                "currency": self.total_value.currency
            },
            "var_95": {
                "amount": str(self.var_95.amount),
                "currency": self.var_95.currency
            },
            "var_99": {
                "amount": str(self.var_99.amount),
                "currency": self.var_99.currency
            },
            "max_drawdown": str(self.max_drawdown),
            "volatility": str(self.volatility),
            "beta": str(self.beta),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "timestamp": self.timestamp.isoformat()
        }


class RiskLevel:
    """Уровни риска."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskConfig:
    """Конфигурация риска."""
    
    max_position_size: Decimal
    max_portfolio_concentration: Decimal
    max_daily_loss: Money
    var_threshold: Decimal
    volatility_threshold: Decimal
    drawdown_threshold: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "max_position_size": str(self.max_position_size),
            "max_portfolio_concentration": str(self.max_portfolio_concentration),
            "max_daily_loss": {
                "amount": str(self.max_daily_loss.amount),
                "currency": self.max_daily_loss.currency
            },
            "var_threshold": str(self.var_threshold),
            "volatility_threshold": str(self.volatility_threshold),
            "drawdown_threshold": str(self.drawdown_threshold)
        }


@dataclass
class RiskLimits:
    """Лимиты риска."""
    
    position_limits: Dict[Symbol, Decimal]
    portfolio_limits: Dict[str, Decimal]
    trading_limits: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "position_limits": {str(k): str(v) for k, v in self.position_limits.items()},
            "portfolio_limits": {k: str(v) for k, v in self.portfolio_limits.items()},
            "trading_limits": self.trading_limits
        }


class RiskAgent:
    """Агент управления рисками."""
    
    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig(
            max_position_size=Decimal("0.1"),
            max_portfolio_concentration=Decimal("0.2"),
            max_daily_loss=Money(Decimal("1000"), "USDT"),
            var_threshold=Decimal("0.05"),
            volatility_threshold=Decimal("0.3"),
            drawdown_threshold=Decimal("0.1")
        )
        self.risk_metrics: Dict[PortfolioId, RiskMetrics] = {}
    
    async def calculate_risk_metrics(self, portfolio_id: PortfolioId) -> RiskMetrics:
        """Расчет метрик риска."""
        # Заглушка для демонстрации
        return RiskMetrics(
            portfolio_id=portfolio_id,
            total_value=Money(Decimal("10000"), "USDT"),
            var_95=Money(Decimal("500"), "USDT"),
            var_99=Money(Decimal("800"), "USDT"),
            max_drawdown=Decimal("0.05"),
            volatility=Decimal("0.2"),
            beta=Decimal("1.0"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("1.8"),
            calmar_ratio=Decimal("2.0"),
            timestamp=datetime.now()
        )
    
    async def validate_risk_limits(self, portfolio_id: PortfolioId, order_data: Dict[str, Any]) -> bool:
        """Валидация лимитов риска."""
        return True
    
    async def get_risk_level(self, portfolio_id: PortfolioId) -> str:
        """Получение уровня риска."""
        return RiskLevel.MEDIUM


__all__ = [
    "RiskAgent",
    "RiskMetrics",
    "RiskLevel",
    "RiskConfig",
    "RiskLimits",
    "IRiskCalculator",
    "DefaultRiskCalculator",
    "RiskMetricsCalculator",
    "RiskMonitoringService",
    "RiskAlertService",
]
