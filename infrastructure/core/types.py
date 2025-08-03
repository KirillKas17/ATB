"""
Модуль типов данных.

Содержит определения основных типов данных для торговой системы,
включая режимы торговли, торговые пары, сигналы и метрики.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TradingMode(Enum):
    """Режимы торговли."""

    PAUSED = "paused"
    TRADING = "trading"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


@dataclass
class TradingPair:
    """Торговая пара."""

    symbol: str
    base: str
    quote: str
    active: bool = True
    precision: Dict[str, int] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Торговый сигнал."""

    pair: str
    action: str  # buy, sell
    price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Метрики стратегии."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeDecision:
    """Решение о сделке."""

    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
