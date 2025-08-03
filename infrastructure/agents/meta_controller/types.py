from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MetaControllerConfig:
    """Конфигурация мета-контроллера."""

    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.1
    performance_threshold: float = 0.05
    rebalance_interval: int = 3600
    strategy_timeout: int = 300
    enable_auto_rebalance: bool = True
    enable_risk_control: bool = True
    enable_performance_monitoring: bool = True


@dataclass
class ControllerSignal:
    """Сигнал управления от мета-контроллера."""

    type: str
    action: str
    priority: str  # "low", "medium", "high", "critical"
    data: Dict[str, Any]
    timestamp: "Optional[datetime]" = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyStatus:
    """Статус торговой стратегии."""

    strategy_id: str
    name: str
    status: str  # "active", "inactive", "error", "stopped"
    performance: float
    risk_level: float
    last_update: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.last_update is None:
            self.last_update = datetime.now()


@dataclass
class PortfolioState:
    """Состояние портфеля."""

    total_value: float
    cash_balance: float
    positions: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: "Optional[datetime]" = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RiskMetrics:
    """Метрики риска портфеля."""

    portfolio_risk: float
    position_risk: float
    correlation_risk: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    timestamp: "Optional[datetime]" = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceMetrics:
    """Метрики производительности."""

    overall_performance: float
    strategy_performance: Dict[str, float]
    win_rate: float
    profit_factor: float
    average_trade: float
    timestamp: "Optional[datetime]" = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ControllerDecision:
    """Решение мета-контроллера."""

    decision_type: str
    action: str
    priority: str
    reason: str
    data: Dict[str, Any]
    timestamp: "Optional[datetime]" = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


# Экспорт типов
__all__ = [
    "MetaControllerConfig",
    "ControllerSignal",
    "StrategyStatus",
    "PortfolioState",
    "RiskMetrics",
    "PerformanceMetrics",
    "ControllerDecision",
]
