"""
Централизованные типы для модуля simulation.
Полная строгая типизация для промышленного уровня.
"""

from shared.numpy_utils import np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)
from uuid import uuid4

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trading import Signal as DomainSignal, Trade as DomainTrade
from domain.entities.market import MarketData as DomainMarketData
from domain.type_definitions import (
    OrderId,
    PortfolioId,
    PriceValue,
    SignalId,
    StrategyId,
    Symbol,
    TimestampValue,
    TradeId,
    TradingPair,
    VolumeValue,
)
from domain.value_objects import Currency, Money, Percentage, Price, Volume


# ============================================================================
# Новые типы для строгой типизации
# ============================================================================
class SimulationId(NewType):
    """Уникальный идентификатор симуляции."""

    pass


class BacktestId(NewType):
    """Уникальный идентификатор бэктеста."""

    pass


class MarketRegimeType(str, Enum):
    """Типы рыночных режимов."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


class SignalStrengthType(str, Enum):
    """Сила сигнала."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class RiskLevelType(str, Enum):
    """Уровни риска."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ExecutionType(str, Enum):
    """Типы исполнения."""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    PARTIAL = "partial"
    REJECTED = "rejected"


class MarketImpactType(str, Enum):
    """Типы влияния на рынок."""

    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTREME = "extreme"


# ============================================================================
# TypedDict для строгой типизации словарей
# ============================================================================
class MarketMetricsDict(TypedDict):
    """Метрики рынка."""

    volatility: float
    trend_strength: float
    volume_trend: float
    momentum: float
    regime_score: float
    liquidity_score: float
    sentiment_score: float


class TradeMetricsDict(TypedDict):
    """Метрики сделки."""

    pnl: float
    duration: float
    slippage: float
    commission: float
    market_impact: float
    execution_quality: float


class BacktestMetricsDict(TypedDict):
    """Метрики бэктеста."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    risk_reward_ratio: float
    kelly_criterion: float


class SimulationConfigDict(TypedDict):
    """Конфигурация симуляции."""

    start_date: str
    end_date: str
    initial_balance: float
    commission: float
    slippage: float
    max_position_size: float
    risk_per_trade: float
    confidence_threshold: float
    symbols: List[str]
    timeframes: List[str]
    random_seed: Optional[int]


# ============================================================================
# Базовые модели данных
# ============================================================================
@dataclass(frozen=True)
class SimulationTimestamp:
    """Временная метка симуляции."""

    value: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime):
            raise ValueError("Timestamp must be datetime object")

    def to_iso(self) -> str:
        return self.value.isoformat()

    @classmethod
    def from_iso(cls, iso_str: str) -> 'SimulationTimestamp':
        return cls(datetime.fromisoformat(iso_str))


@dataclass(frozen=True)
class SimulationPrice:
    """Цена в симуляции."""

    value: Decimal = Decimal("0")
    currency: Currency = Currency.USDT

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, "value", Decimal(str(self.value)))
        if self.value < 0:
            raise ValueError("Price cannot be negative")
        # Устанавливаем флаг валидации
        object.__setattr__(self, '_value_validated', True)

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return f"{self.value} {self.currency.value}"

    def to_domain_price(self) -> Price:
        """Преобразование в доменную цену."""
        return Price(self.value, self.currency)

    def to_domain_money(self) -> Money:
        """Преобразование в доменные деньги."""
        return Money(self.value, self.currency)


@dataclass(frozen=True)
class SimulationVolume:
    """Объем в симуляции."""

    value: Decimal = Decimal("0")
    currency: Currency = Currency.USDT

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        # Проверяем только если поля не инициализированы корректно
        if not hasattr(self, '_value_validated'):
            self._value_validated = True
        if self.value < 0:
            raise ValueError("Volume cannot be negative")

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return f"{self.value} {self.currency.value}"

    def to_domain_volume(self) -> Volume:
        """Преобразование в доменный объем."""
        return Volume(self.value, self.currency)


@dataclass(frozen=True)
class SimulationMoney:
    """Деньги в симуляции."""

    value: Decimal = Decimal("0")
    currency: Currency = Currency.USDT

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        # Проверяем только если поля не инициализированы корректно
        if not hasattr(self, '_value_validated'):
            self._value_validated = True

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return f"{self.value} {self.currency.value}"

    def __add__(self, other: "SimulationMoney") -> "SimulationMoney":
        if self.currency != other.currency:
            raise ValueError("Cannot add money with different currencies")
        return SimulationMoney(self.value + other.value, self.currency)

    def __sub__(self, other: "SimulationMoney") -> "SimulationMoney":
        if self.currency != other.currency:
            raise ValueError("Cannot subtract money with different currencies")
        return SimulationMoney(self.value - other.value, self.currency)

    def to_domain_money(self) -> Money:
        """Преобразование в доменные деньги."""
        return Money(self.value, self.currency)


@dataclass
class SimulationMarketData:
    """Рыночные данные для симуляции."""

    symbol: Symbol
    timestamp: SimulationTimestamp
    open: SimulationPrice
    high: SimulationPrice
    low: SimulationPrice
    close: SimulationPrice
    volume: SimulationVolume
    regime: MarketRegimeType = MarketRegimeType.UNKNOWN
    volatility: float = 0.0
    trend_strength: float = 0.0
    momentum: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.high) < float(self.low):
            raise ValueError("High price cannot be less than low price")
        if float(self.high) < float(self.open) or float(self.high) < float(self.close):
            raise ValueError("High price must be highest")
        if float(self.low) > float(self.open) or float(self.low) > float(self.close):
            raise ValueError("Low price must be lowest")

    @property
    def price_range(self) -> float:
        return float(self.high) - float(self.low)

    @property
    def price_change(self) -> float:
        return float(self.close) - float(self.open)

    @property
    def price_change_percent(self) -> float:
        if float(self.open) == 0:
            return 0.0
        return (float(self.close) - float(self.open)) / float(self.open) * 100

    def to_domain_market_data(self) -> DomainMarketData:
        """Преобразование в доменную модель."""
        from domain.value_objects.price import Price
        from domain.value_objects.volume import Volume
        from domain.type_definitions import MetadataDict, TimestampValue
        return DomainMarketData(
            symbol=self.symbol,
            timestamp=TimestampValue(self.timestamp.value),
            open=Price(Decimal(str(self.open.value)), self.open.currency, self.open.currency),
            high=Price(Decimal(str(self.high.value)), self.high.currency, self.high.currency),
            low=Price(Decimal(str(self.low.value)), self.low.currency, self.low.currency),
            close=Price(Decimal(str(self.close.value)), self.close.currency, self.close.currency),
            volume=Volume(Decimal(str(self.volume.value)), self.volume.currency),
            metadata=MetadataDict(self.metadata),
        )


@dataclass
class SimulationSignal:
    """Торговый сигнал для симуляции."""

    id: SignalId = field(default_factory=lambda: SignalId(uuid4()))
    symbol: Symbol = Symbol("")
    timestamp: SimulationTimestamp = field(default_factory=SimulationTimestamp)
    signal_type: Literal["buy", "sell", "hold"] = "hold"
    strength: SignalStrengthType = SignalStrengthType.MODERATE
    confidence: float = 0.5
    price: Optional[SimulationPrice] = None
    volume: Optional[SimulationVolume] = None
    stop_loss: Optional[SimulationPrice] = None
    take_profit: Optional[SimulationPrice] = None
    strategy_id: Optional[StrategyId] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    def to_domain_signal(self) -> DomainSignal:
        """Преобразование в доменную модель."""
        from domain.entities.trading import Signal, SignalType
        from domain.type_definitions import MetadataDict
        # Преобразование типа сигнала
        if self.signal_type == "buy":
            signal_type = SignalType.BUY
        elif self.signal_type == "sell":
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        return Signal(
            id=str(self.id),
            signal_type=signal_type,
            confidence=float(self.confidence),
            price=self.price.to_domain_price() if self.price else None,
            timestamp=self.timestamp.value,
            metadata=MetadataDict(self.metadata),
        )


@dataclass
class SimulationOrder:
    """Ордер для симуляции."""

    id: OrderId = field(default_factory=lambda: OrderId(uuid4()))
    symbol: Symbol = Symbol("")
    timestamp: SimulationTimestamp = field(default_factory=SimulationTimestamp)
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: SimulationVolume = field(default_factory=lambda: SimulationVolume())
    price: Optional[SimulationPrice] = None
    stop_price: Optional[SimulationPrice] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: SimulationVolume = field(
        default_factory=lambda: SimulationVolume()
    )
    filled_price: Optional[SimulationPrice] = None
    commission: Optional[SimulationMoney] = None
    slippage: Optional[SimulationMoney] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_partially_filled(self) -> bool:
        return self.status == OrderStatus.PARTIALLY_FILLED

    @property
    def fill_ratio(self) -> float:
        if float(self.quantity) == 0:
            return 0.0
        return float(self.filled_quantity) / float(self.quantity)

    def to_domain_order(self) -> Order:
        """Преобразование в доменную модель."""
        from domain.entities.order import Order
        from domain.value_objects.volume import Volume
        from domain.value_objects.price import Price
        from domain.type_definitions import MetadataDict, VolumeValue
        return Order(
            id=self.id,
            symbol=self.symbol,
            order_type=self.order_type,
            side=self.side,
            quantity=VolumeValue(self.quantity.value),
            price=self.price.to_domain_price() if self.price else None,
            stop_price=self.stop_price.to_domain_price() if self.stop_price else None,
            status=self.status,
            filled_quantity=VolumeValue(self.filled_quantity.value),
            commission=Price(self.commission.value, self.commission.currency, self.commission.currency) if self.commission else None,
            metadata=MetadataDict(self.metadata),
        )


@dataclass
class SimulationTrade:
    """Сделка для симуляции."""

    id: TradeId = field(default_factory=lambda: TradeId(uuid4()))
    order_id: OrderId = field(default_factory=lambda: OrderId(uuid4()))
    symbol: Symbol = Symbol("")
    timestamp: SimulationTimestamp = field(default_factory=SimulationTimestamp)
    side: OrderSide = OrderSide.BUY
    quantity: SimulationVolume = field(default_factory=lambda: SimulationVolume())
    price: SimulationPrice = field(default_factory=lambda: SimulationPrice())
    commission: SimulationMoney = field(default_factory=lambda: SimulationMoney())
    slippage: SimulationMoney = field(default_factory=lambda: SimulationMoney())
    pnl: SimulationMoney = field(default_factory=lambda: SimulationMoney())
    market_impact: float = 0.0
    execution_quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.execution_quality <= 1.0:
            raise ValueError("Execution quality must be between 0 and 1")
        if not 0.0 <= self.market_impact <= 1.0:
            raise ValueError("Market impact must be between 0 and 1")

    @property
    def total_cost(self) -> SimulationMoney:
        """Общая стоимость сделки."""
        trade_value = SimulationMoney(
            Decimal(str(self.quantity.value * self.price.value)), self.price.currency
        )
        return trade_value + self.commission + self.slippage

    def to_domain_trade(self) -> DomainTrade:
        """Преобразование в доменную модель."""
        from domain.entities.trading import Trade
        from domain.value_objects.timestamp import TimestampValue
        from domain.type_definitions import MetadataDict, TradingPair
        from domain.entities.trading import OrderSide as DomainOrderSide
        from domain.type_definitions import TimestampValue as DomainTimestampValue
        
        # Преобразование OrderSide
        if self.side == OrderSide.BUY:
            domain_side = DomainOrderSide.BUY
        else:
            domain_side = DomainOrderSide.SELL
            
        return Trade(
            id=self.id,
            order_id=self.order_id,
            trading_pair=TradingPair(str(self.symbol)),
            side=domain_side,
            quantity=self.quantity.to_domain_volume(),
            price=self.price.to_domain_price(),
            commission=self.commission.to_domain_money(),
            timestamp=DomainTimestampValue(self.timestamp.value),
        )


@dataclass
class SimulationConfig:
    """Конфигурация симуляции."""

    # Временные параметры
    start_date: datetime
    end_date: datetime
    # Финансовые параметры
    initial_balance: SimulationMoney = field(
        default_factory=lambda: SimulationMoney(Decimal("10000"))
    )
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    # Риск-менеджмент
    max_position_size: float = 0.1  # 10% от баланса
    risk_per_trade: float = 0.02  # 2% от баланса
    max_drawdown: float = 0.2  # 20% максимальная просадка
    # Торговые параметры
    confidence_threshold: float = 0.7
    min_trade_size: SimulationMoney = field(
        default_factory=lambda: SimulationMoney(Decimal("10"))
    )
    max_trade_size: SimulationMoney = field(
        default_factory=lambda: SimulationMoney(Decimal("1000"))
    )
    # Рыночные параметры
    symbols: List[Symbol] = field(default_factory=list)
    timeframes: List[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    # Технические параметры
    random_seed: Optional[int] = 42
    cache_size: int = 1000
    max_workers: int = 4
    # Пути
    data_dir: Path = Path("data/simulation")
    results_dir: Path = Path("results/simulation")
    logs_dir: Path = Path("logs/simulation")

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if not 0.0 <= self.commission_rate <= 1.0:
            raise ValueError("Commission rate must be between 0 and 1")
        if not 0.0 <= self.slippage_rate <= 1.0:
            raise ValueError("Slippage rate must be between 0 and 1")
        if not 0.0 <= self.max_position_size <= 1.0:
            raise ValueError("Max position size must be between 0 and 1")
        if not 0.0 <= self.risk_per_trade <= 1.0:
            raise ValueError("Risk per trade must be between 0 and 1")
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("Max drawdown must be between 0 and 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        # Создание директорий
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig(SimulationConfig):
    """Конфигурация бэктеста."""

    # Параметры бэктеста
    use_realistic_slippage: bool = True
    use_market_impact: bool = True
    use_latency: bool = True
    use_partial_fills: bool = True
    # Анализ результатов
    calculate_metrics: bool = True
    generate_plots: bool = True
    save_trades: bool = True
    save_equity_curve: bool = True
    # Валидация
    min_trades: int = 10
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.1


@dataclass
class MarketSimulationConfig(SimulationConfig):
    """Конфигурация симуляции рынка."""

    # Параметры генерации данных
    initial_price: SimulationPrice = field(
        default_factory=lambda: SimulationPrice(Decimal("50000"))
    )
    volatility: float = 0.02
    trend_strength: float = 0.1
    mean_reversion: float = 0.05
    noise_level: float = 0.01
    # Рыночная микроструктура
    volume_scale: float = 1.0
    market_impact: float = 0.0001
    liquidity_factor: float = 1.0
    # Режимы рынка
    regime_switching: bool = True
    regime_probability: float = 0.1
    regime_duration: int = 100
    regime_transition_weights: Dict[MarketRegimeType, float] = field(
        default_factory=lambda: {
            MarketRegimeType.TRENDING_UP: 0.2,
            MarketRegimeType.TRENDING_DOWN: 0.2,
            MarketRegimeType.SIDEWAYS: 0.3,
            MarketRegimeType.VOLATILE: 0.15,
            MarketRegimeType.BREAKOUT: 0.1,
            MarketRegimeType.ACCUMULATION: 0.025,
            MarketRegimeType.DISTRIBUTION: 0.025,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.volatility <= 1.0:
            raise ValueError("Volatility must be between 0 and 1")
        if not 0.0 <= self.trend_strength <= 1.0:
            raise ValueError("Trend strength must be between 0 and 1")
        if not 0.0 <= self.mean_reversion <= 1.0:
            raise ValueError("Mean reversion must be between 0 and 1")
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("Noise level must be between 0 and 1")
        if not 0.0 <= self.regime_probability <= 1.0:
            raise ValueError("Regime probability must be between 0 and 1")
        if self.regime_duration <= 0:
            raise ValueError("Regime duration must be positive")
        # Проверка весов переходов режимов
        total_weight = sum(self.regime_transition_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError("Regime transition weights must sum to 1.0")


# ============================================================================
# Протоколы для интерфейсов
# ============================================================================
@runtime_checkable
class MarketDataProvider(Protocol):
    """Протокол для поставщика рыночных данных."""

    async def get_market_data(
        self,
        symbol: Symbol,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = "1m",
    ) -> List[SimulationMarketData]: ...
    async def get_latest_data(
        self, symbol: Symbol
    ) -> Optional[SimulationMarketData]: ...
    async def subscribe_to_updates(
        self, symbol: Symbol, callback: Callable
    ) -> None: ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Протокол для генератора сигналов."""

    async def generate_signal(
        self, market_data: SimulationMarketData, context: Dict[str, Any]
    ) -> Optional[SimulationSignal]: ...
    async def validate_signal(self, signal: SimulationSignal) -> bool: ...
    async def get_signal_confidence(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> float: ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Протокол для исполнителя ордеров."""

    async def execute_order(
        self, order: SimulationOrder, market_data: SimulationMarketData
    ) -> SimulationTrade: ...
    async def calculate_slippage(
        self, order: SimulationOrder, market_data: SimulationMarketData
    ) -> SimulationMoney: ...
    async def calculate_commission(self, order: SimulationOrder) -> SimulationMoney: ...


@runtime_checkable
class RiskManager(Protocol):
    """Протокол для риск-менеджера."""

    async def check_risk_limits(
        self,
        signal: SimulationSignal,
        current_balance: SimulationMoney,
        current_positions: List[SimulationTrade],
    ) -> bool: ...
    async def calculate_position_size(
        self, signal: SimulationSignal, balance: SimulationMoney, risk_per_trade: float
    ) -> SimulationVolume: ...
    async def calculate_stop_loss(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> Optional[SimulationPrice]: ...


@runtime_checkable
class MetricsCalculator(Protocol):
    """Протокол для калькулятора метрик."""

    async def calculate_trade_metrics(
        self, trade: SimulationTrade
    ) -> TradeMetricsDict: ...
    async def calculate_backtest_metrics(
        self, trades: List[SimulationTrade], equity_curve: List[float]
    ) -> BacktestMetricsDict: ...
    async def calculate_portfolio_metrics(
        self, positions: List[SimulationTrade], balance: SimulationMoney
    ) -> Dict[str, float]: ...


# ============================================================================
# Базовые классы
# ============================================================================
class BaseSimulationComponent(ABC):
    """Базовый класс для компонентов симуляции."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.logger = self._setup_logger()

    @abstractmethod
    def _setup_logger(self) -> Any:
        """Настройка логгера."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Инициализация компонента."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        ...


class BaseMarketSimulator(BaseSimulationComponent):
    """Базовый класс для симулятора рынка."""

    def __init__(self, config: MarketSimulationConfig) -> None:
        super().__init__(config)

    @abstractmethod
    async def generate_market_data(
        self, symbol: Symbol, start_time: datetime, end_time: datetime
    ) -> List[SimulationMarketData]:
        """Генерация рыночных данных."""
        ...

    @abstractmethod
    async def update_market_state(
        self, symbol: Symbol, market_data: SimulationMarketData
    ) -> None:
        """Обновление состояния рынка."""
        ...

    @abstractmethod
    async def get_market_regime(self, symbol: Symbol) -> MarketRegimeType:
        """Получение режима рынка."""
        ...


class BaseBacktester(BaseSimulationComponent):
    """Базовый класс для бэктестера."""

    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config)
        self.trades: List[SimulationTrade] = []
        self.equity_curve: List[float] = []

    @abstractmethod
    async def run_backtest(
        self, strategy: SignalGenerator, market_data: List[SimulationMarketData]
    ) -> "BacktestResult":
        """Запуск бэктеста."""
        ...

    @abstractmethod
    async def calculate_metrics(self) -> BacktestMetricsDict:
        """Расчет метрик."""
        ...

    @abstractmethod
    async def save_results(self, path: Path) -> None:
        """Сохранение результатов."""
        ...


# ============================================================================
# Результаты
# ============================================================================
@dataclass
class BacktestResult:
    """Результаты бэктеста."""

    config: BacktestConfig
    trades: List[SimulationTrade]
    equity_curve: List[float]
    metrics: BacktestMetricsDict
    start_time: datetime
    end_time: datetime
    initial_balance: SimulationMoney
    final_balance: SimulationMoney
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        """Общая доходность."""
        if float(self.initial_balance) == 0:
            return 0.0
        return (float(self.final_balance) - float(self.initial_balance)) / float(
            self.initial_balance
        )

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if float(t.pnl) > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if float(t.pnl) < 0])

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_balance": float(self.config.initial_balance),
            },
            "trades_count": len(self.trades),
            "equity_curve_length": len(self.equity_curve),
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_balance": float(self.initial_balance),
            "final_balance": float(self.final_balance),
            "total_return": self.total_return,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class SimulationResult:
    """Результаты симуляции."""

    config: SimulationConfig
    market_data: Dict[Symbol, List[SimulationMarketData]]
    backtest_results: Dict[str, BacktestResult]
    start_time: datetime
    end_time: datetime
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def total_symbols(self) -> int:
        return len(self.market_data)

    @property
    def total_backtests(self) -> int:
        return len(self.backtest_results)


# ============================================================================
# Утилиты
# ============================================================================
class SimulationUtils:
    """Утилиты для симуляции."""

    @staticmethod
    def calculate_volatility(prices: List[float], window: int = 20) -> float:
        """Расчет волатильности."""
        if len(prices) < window:
            return 0.0
        returns = np.diff(np.log(prices[-window:]))
        return float(np.std(returns) * np.sqrt(252))

    @staticmethod
    def calculate_trend_strength(prices: List[float], window: int = 20) -> float:
        """Расчет силы тренда."""
        if len(prices) < window:
            return 0.0
        recent_prices = prices[-window:]
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        return float(slope / np.mean(recent_prices))

    @staticmethod
    def calculate_momentum(prices: List[float], window: int = 10) -> float:
        """Расчет моментума."""
        if len(prices) < window:
            return 0.0
        return float((prices[-1] - prices[-window]) / prices[-window])

    @staticmethod
    def detect_market_regime(
        prices: List[float], volatility: float, trend_strength: float
    ) -> MarketRegimeType:
        """Определение режима рынка."""
        if volatility > 0.03:  # Высокая волатильность
            return MarketRegimeType.VOLATILE
        elif abs(trend_strength) > 0.01:  # Сильный тренд
            if trend_strength > 0:
                return MarketRegimeType.TRENDING_UP
            else:
                return MarketRegimeType.TRENDING_DOWN
        else:
            return MarketRegimeType.SIDEWAYS

    @staticmethod
    def calculate_slippage(
        order_size: float, market_volume: float, volatility: float
    ) -> float:
        """Расчет проскальзывания."""
        if market_volume == 0:
            return 0.0
        impact = (order_size / market_volume) * volatility
        return min(impact, 0.01)  # Максимум 1%

    @staticmethod
    def calculate_market_impact(
        order_size: float, market_volume: float, volatility: float
    ) -> float:
        """Расчет влияния на рынок."""
        if market_volume == 0:
            return 0.0
        # Базовая формула влияния на рынок
        impact = (order_size / market_volume) * float(np.sqrt(volatility))
        # Ограничиваем максимальное влияние
        return min(impact, 0.05)  # Максимум 5%


class SimulationConstants:
    """Константы для симуляции."""

    # Временные константы
    DEFAULT_TIMEFRAME = "1m"
    MAX_TIMEFRAME = "1d"
    MIN_TIMEFRAME = "1s"
    # Финансовые константы
    MIN_PRICE = Decimal("0.000001")
    MAX_PRICE = Decimal("1000000")
    MIN_VOLUME = Decimal("0.000001")
    MAX_VOLUME = Decimal("1000000")
    # Риск-менеджмент
    MAX_LEVERAGE = 100.0
    MIN_STOP_LOSS = 0.001  # 0.1%
    MAX_STOP_LOSS = 0.5  # 50%
    # Технические константы
    MAX_CACHE_SIZE = 10000
    MAX_WORKERS = 16
    DEFAULT_TIMEOUT = 30.0
    # Метрики
    MIN_TRADES_FOR_METRICS = 10
    MIN_WIN_RATE = 0.3
    MIN_PROFIT_FACTOR = 1.0
    # Рыночные константы
    DEFAULT_VOLATILITY = 0.02
    DEFAULT_TREND_STRENGTH = 0.1
    DEFAULT_MEAN_REVERSION = 0.05
    DEFAULT_NOISE_LEVEL = 0.01
