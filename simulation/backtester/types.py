from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TradeAction(Enum):
    """Типы торговых действий"""

    OPEN = auto()
    CLOSE = auto()
    HOLD = auto()
    MODIFY = auto()
    CANCEL = auto()


class TradeDirection(Enum):
    """Направления сделок"""

    LONG = auto()
    SHORT = auto()
    NONE = auto()
    HEDGE = auto()


class TradeStatus(Enum):
    """Статус сделки"""

    PENDING = auto()
    ACTIVE = auto()
    CLOSED = auto()
    CANCELLED = auto()
    ERROR = auto()


class SignalType(Enum):
    """Тип сигнала"""

    ENTRY = auto()
    EXIT = auto()
    STOP = auto()
    TAKE_PROFIT = auto()
    ALERT = auto()


@dataclass
class Tag:
    """Тег для сделок, сигналов, событий"""

    name: str
    value: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None


@dataclass
class RiskInfo:
    """Информация о рисках"""

    risk_score: float = 0.0
    max_drawdown: Optional[float] = None
    leverage: Optional[float] = None
    margin_usage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class TradeEvent:
    """Событие, связанное с торговлей"""

    event_type: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[Tag] = field(default_factory=list)


@dataclass
class TradeError:
    """Ошибка, связанная с торговлей"""

    code: int
    message: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """Информация о сделке"""

    symbol: str
    action: TradeAction
    direction: TradeDirection
    volume: float
    price: float
    commission: float
    pnl: float
    timestamp: datetime
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeStatus = TradeStatus.PENDING
    tags: List[Tag] = field(default_factory=list)
    events: List[TradeEvent] = field(default_factory=list)
    errors: List[TradeError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeframe: Optional[str] = None
    order_id: Optional[str] = None
    parent_id: Optional[str] = None
    risk: Optional[RiskInfo] = None


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""

    initial_balance: float
    commission: float
    slippage: float
    max_position_size: float
    risk_per_trade: float
    confidence_threshold: float = 0.7
    max_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    min_trades: Optional[int] = None
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    random_seed: Optional[int] = None
    tags: List[Tag] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class BacktestResult:
    """Результаты бэктеста"""

    trades: List[Trade]
    equity_curve: List[float]
    metrics: Dict[str, float]
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    risk_reward_ratio: float
    kelly_criterion: float
    tags: List[Tag] = field(default_factory=list)
    errors: List[TradeError] = field(default_factory=list)
    events: List[TradeEvent] = field(default_factory=list)
    risk: Optional[RiskInfo] = None


@dataclass
class MarketData:
    """Рыночные данные"""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Торговый сигнал"""

    symbol: str
    direction: TradeDirection
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    volume: Optional[float]
    leverage: float = 1.0
    signal_type: SignalType = SignalType.ENTRY
    tags: List[Tag] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeframe: Optional[str] = None
    risk: Optional[RiskInfo] = None
    generated_by: Optional[str] = None
