"""
Типы данных для infrastructure/core модулей.
Содержит все типы, используемые в core компонентах системы.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict, Union


# ============================================================================
# Enums
# ============================================================================
class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Стороны ордера."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Статусы ордера."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Стороны позиции."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Статусы позиции."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


# ============================================================================
# Базовые типы данных
# ============================================================================
class MarketData(TypedDict):
    """Структура рыночных данных."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    metadata: Dict[str, Any]


class OrderBookData(TypedDict):
    """Данные ордербука."""

    symbol: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [price, quantity]
    asks: List[Dict[str, float]]  # [price, quantity]
    last_update_id: int


class TradeData(TypedDict):
    """Данные сделки."""

    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: Literal["buy", "sell"]
    trade_id: str


# ============================================================================
# Конфигурации
# ============================================================================
class StrategyConfig(TypedDict):
    """Конфигурация стратегии."""

    name: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]
    enabled: bool
    priority: int


class EvolutionConfig(TypedDict):
    """Конфигурация эволюции."""

    fast_adaptation_interval: int
    learning_interval: int
    evolution_interval: int
    full_evolution_interval: int
    performance_threshold: float
    drift_threshold: float
    evolution_trigger_threshold: float
    efficiency_improvement_threshold: float
    confirmation_period: int
    min_test_samples: int
    statistical_significance_level: float
    rollback_on_degradation: bool
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    max_trials: int
    timeout: int
    evolution_log_path: str
    models_backup_path: str
    performance_history_path: str
    max_workers: int
    use_gpu: bool


class SystemConfig(TypedDict):
    """Конфигурация системы."""

    trading: Dict[str, Any]
    risk: Dict[str, Any]
    portfolio: Dict[str, Any]
    database: Dict[str, Any]
    exchange: Dict[str, Any]
    monitoring: Dict[str, Any]


# ============================================================================
# Метрики
# ============================================================================
class SystemMetrics(TypedDict):
    """Метрики системы."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_connections: int
    active_strategies: int
    total_trades: int
    system_uptime: float
    error_rate: float
    performance_score: float


class RiskMetrics(TypedDict):
    """Метрики риска."""

    portfolio_risk: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: float
    leverage_ratio: float
    margin_usage: float


class PortfolioMetrics(TypedDict):
    """Метрики портфеля."""

    total_value: float
    total_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    total_return: float
    daily_return: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    positions: Dict[str, Dict[str, Any]]
    cash_balance: float
    margin_balance: float


class StrategyMetrics(TypedDict):
    """Метрики стратегии."""

    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_profit: float
    average_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_return: float
    daily_return: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    start_time: datetime
    end_time: Optional[datetime]
    parameters: Dict[str, Any]
    status: Literal["active", "paused", "stopped", "error"]


# ============================================================================
# Состояния
# ============================================================================
class SystemState(TypedDict):
    """Состояние системы."""

    is_healthy: bool
    current_regime: str
    active_strategies: List[str]
    performance_metrics: SystemMetrics
    risk_metrics: RiskMetrics
    portfolio_metrics: PortfolioMetrics
    market_conditions: Dict[str, Any]
    last_update: datetime


class MarketState(TypedDict):
    """Состояние рынка."""

    symbol: str
    regime: Literal["trending", "ranging", "volatile", "unknown"]
    volatility: float
    trend_strength: float
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: List[float]
    last_update: datetime


# ============================================================================
# События
# ============================================================================
class EventType(Enum):
    """Типы событий."""

    # Системные события
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH_CHECK = "system.health_check"
    # Торговые события
    TRADE_EXECUTED = "trade.executed"
    ORDER_PLACED = "order.placed"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FILLED = "order.filled"
    # События стратегий
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    STRATEGY_SIGNAL = "strategy.signal"
    STRATEGY_ERROR = "strategy.error"
    # События риска
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    RISK_ALERT = "risk.alert"
    RISK_EMERGENCY_STOP = "risk.emergency_stop"
    # События эволюции
    EVOLUTION_STARTED = "evolution.started"
    EVOLUTION_COMPLETED = "evolution.completed"
    EVOLUTION_ERROR = "evolution.error"
    # События портфеля
    PORTFOLIO_UPDATED = "portfolio.updated"
    PORTFOLIO_REBALANCED = "portfolio.rebalanced"
    PORTFOLIO_ALERT = "portfolio.alert"


class EventPriority(Enum):
    """Приоритеты событий."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Событие системы."""

    event_type: EventType
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None


# ============================================================================
# Протоколы (интерфейсы)
# ============================================================================
class StrategyProtocol(Protocol):
    """Протокол для стратегий."""

    def generate_signals(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов."""
        ...

    def update(self, market_data: MarketData) -> None:
        """Обновление состояния стратегии."""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Получение параметров стратегии."""
        ...

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Установка параметров стратегии."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Получение состояния стратегии."""
        ...

    def save_state(self, filename: str) -> None:
        """Сохранение состояния стратегии."""
        ...

    def load_state(self, filename: str) -> None:
        """Загрузка состояния стратегии."""
        ...


class EvolvableProtocol(Protocol):
    """Протокол для эволюционирующих компонентов."""

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным."""
        ...

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных."""
        ...

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента."""
        ...

    def get_performance(self) -> float:
        """Получение текущей производительности."""
        ...

    def get_confidence(self) -> float:
        """Получение уверенности в решениях."""
        ...

    def save_state(self, path: str) -> bool:
        """Сохранение состояния."""
        ...

    def load_state(self, path: str) -> bool:
        """Загрузка состояния."""
        ...


class MonitorProtocol(Protocol):
    """Протокол для мониторинга."""

    async def check_health(self) -> Dict[str, Any]:
        """Проверка здоровья системы."""
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик."""
        ...

    async def start_monitoring(self) -> None:
        """Запуск мониторинга."""
        ...

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        ...


class ControllerProtocol(Protocol):
    """Протокол для контроллеров."""

    async def start(self) -> None:
        """Запуск контроллера."""
        ...

    async def stop(self) -> None:
        """Остановка контроллера."""
        ...

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса контроллера."""
        ...

    async def handle_event(self, event: Event) -> None:
        """Обработка события."""
        ...


# ============================================================================
# Исключения
# ============================================================================
class StrategyError(Exception):
    """Ошибка стратегии."""

    pass


class EvolutionError(Exception):
    """Ошибка эволюции."""

    pass


class PortfolioError(Exception):
    """Ошибка портфеля."""

    pass


class RiskError(Exception):
    """Ошибка риск-менеджмента."""

    pass


class SystemError(Exception):
    """Ошибка системы."""

    pass


class DatabaseError(Exception):
    """Ошибка базы данных."""

    pass


class ExchangeError(Exception):
    """Ошибка биржи."""

    pass


# ============================================================================
# Утилиты
# ============================================================================
def validate_market_data(data: Dict[str, Any]) -> MarketData:
    """Валидация рыночных данных."""
    required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    return MarketData(
        symbol=str(data["symbol"]),
        timestamp=(
            data["timestamp"]
            if isinstance(data["timestamp"], datetime)
            else datetime.fromisoformat(data["timestamp"])
        ),
        open=float(data["open"]),
        high=float(data["high"]),
        low=float(data["low"]),
        close=float(data["close"]),
        volume=float(data["volume"]),
        bid=data.get("bid"),
        ask=data.get("ask"),
        spread=data.get("spread"),
        metadata=data.get("metadata", {}),
    )


def validate_strategy_config(config: Dict[str, Any]) -> StrategyConfig:
    """Валидация конфигурации стратегии."""
    required_fields = ["name", "symbol", "timeframe"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    return StrategyConfig(
        name=str(config["name"]),
        symbol=str(config["symbol"]),
        timeframe=str(config["timeframe"]),
        parameters=config.get("parameters", {}),
        risk_limits=config.get("risk_limits", {}),
        enabled=config.get("enabled", True),
        priority=config.get("priority", 1),
    )
