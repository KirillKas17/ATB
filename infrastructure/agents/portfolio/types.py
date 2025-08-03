"""
Типы для portfolio agent.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict


class PortfolioStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


@dataclass
class PortfolioMetrics:
    """Метрики портфеля."""

    total_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    pnl_percentage: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sector_allocation: Dict[str, Decimal] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Конфигурация портфеля."""

    max_position_size: float = 0.1
    max_sector_allocation: float = 0.3
    target_volatility: float = 0.15
    rebalancing_threshold: float = 0.05


@dataclass
class PortfolioLimits:
    """Лимиты портфеля."""

    max_loss: float = 0.2
    max_drawdown: float = 0.15
    max_leverage: float = 2.0
