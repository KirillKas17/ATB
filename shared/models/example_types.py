"""
Общие типы для примеров ATB системы.
Обеспечивает строгую типизацию и согласованность между модулями.
"""

import pandas as pd
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union
from uuid import UUID

from pydantic import BaseModel, Field

from domain.entities.trading import OrderId, Position, Trade
from domain.type_definitions import Symbol
from domain.entities.market import MarketData


class ExampleMode(str, Enum):
    """Режимы работы примеров."""

    DEMO = "demo"
    BACKTEST = "backtest"
    LIVE = "live"
    SIMULATION = "simulation"


class ExampleConfig(BaseModel):
    """Конфигурация для примеров."""

    mode: ExampleMode = ExampleMode.DEMO
    duration_seconds: int = 60
    symbols: List[Symbol] = Field(default_factory=list)
    risk_level: Decimal = Decimal("0.02")
    max_positions: int = 5
    enable_logging: bool = True
    save_results: bool = True


@dataclass(frozen=True)
class ExampleResult:
    """Результат выполнения примера."""

    success: bool
    duration_seconds: float
    trades_executed: int
    total_pnl: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Протокол для провайдера рыночных данных."""

    async def get_market_data(
        self, symbol: Symbol, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> MarketData:
        """Получить рыночные данные."""
        ...

    async def get_order_book(self, symbol: Symbol) -> Dict[str, Any]:
        """Получить ордербук."""
        ...

    async def get_recent_trades(self, symbol: Symbol, limit: int = 100) -> List[Trade]:
        """Получить последние сделки."""
        ...


class StrategyExecutor(Protocol):
    """Протокол для исполнителя стратегий."""

    async def execute_strategy(
        self, strategy_id: UUID, symbol: Symbol, market_data: MarketData
    ) -> List[OrderId]:
        """Выполнить стратегию."""
        ...

    async def evaluate_strategy(
        self, strategy_id: UUID, historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Оценить стратегию."""
        ...


class RiskManager(Protocol):
    """Протокол для управления рисками."""

    def calculate_position_size(
        self, symbol: Symbol, price: Decimal, risk_per_trade: Decimal
    ) -> Decimal:
        """Рассчитать размер позиции."""
        ...

    def validate_order(self, order: OrderId, current_positions: List[Position]) -> bool:
        """Проверить валидность ордера."""
        ...

    def calculate_portfolio_risk(self, positions: List[Position]) -> Decimal:
        """Рассчитать риск портфеля."""
        ...


@dataclass
class SentimentData:
    """Данные сентимента."""

    symbol: Symbol
    timestamp: pd.Timestamp
    sentiment_score: float
    confidence: float
    source: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsEvent:
    """Новостное событие."""

    id: UUID
    timestamp: pd.Timestamp
    symbol: Optional[Symbol]
    title: str
    content: str
    impact: str  # "high", "medium", "low"
    sentiment: float
    source: str
    url: Optional[str] = None


@dataclass
class MirrorSignal:
    """Зеркальный сигнал."""

    base_asset: Symbol
    mirror_asset: Symbol
    correlation: float
    lag_periods: int
    strength: float
    direction: str  # "buy", "sell"
    confidence: float
    timestamp: pd.Timestamp


@dataclass
class EntanglementSignal:
    """Сигнал запутанности."""

    symbol: Symbol
    timestamp: pd.Timestamp
    entanglement_score: float
    pattern_type: str
    confidence: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquidityGravitySignal:
    """Сигнал гравитации ликвидности."""

    symbol: Symbol
    timestamp: pd.Timestamp
    gravity_score: float
    risk_level: str  # "low", "medium", "high", "extreme"
    spread_impact: float
    volume_impact: float
    recommendation: str


@dataclass
class MarketMemoryPattern:
    """Паттерн рыночной памяти."""

    pattern_id: UUID
    symbol: Symbol
    pattern_type: str
    confidence: float
    timestamp: pd.Timestamp
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExampleRunner(Protocol):
    """Протокол для запуска примеров."""

    async def run(self, config: ExampleConfig) -> ExampleResult:
        """Запустить пример."""
        ...

    async def setup(self) -> None:
        """Настройка примера."""
        ...

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        ...


class StrategyEvolutionConfig(BaseModel):
    """Конфигурация эволюции стратегий."""

    population_size: int = 20
    generations: int = 10
    mutation_rate: Decimal = Decimal("0.15")
    crossover_rate: Decimal = Decimal("0.8")
    elite_size: int = 3
    min_accuracy: Decimal = Decimal("0.75")
    min_profitability: Decimal = Decimal("0.02")
    max_drawdown: Decimal = Decimal("0.20")
    min_sharpe: Decimal = Decimal("0.8")
    max_complexity: int = 50
    fitness_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "profitability": 0.4,
            "risk": 0.3,
            "consistency": 0.2,
            "simplicity": 0.1,
        }
    )


class SentimentAnalysisConfig(BaseModel):
    """Конфигурация анализа сентимента."""

    sources: List[str] = Field(default_factory=lambda: ["news", "social", "technical"])
    update_interval: int = 300  # секунды
    sentiment_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_history_days: int = 30
    language: str = "en"
    enable_cache: bool = True


class MirrorMapConfig(BaseModel):
    """Конфигурация карты зеркальных зависимостей."""

    min_correlation: float = 0.3
    max_p_value: float = 0.05
    max_lag: int = 5
    parallel_processing: bool = True
    max_workers: int = 4
    update_interval: int = 3600  # секунды
    symbols: List[Symbol] = Field(default_factory=list)


class EntanglementConfig(BaseModel):
    """Конфигурация детекции запутанности."""

    detection_threshold: float = 0.7
    window_size: int = 100
    update_interval: int = 60
    symbols: List[Symbol] = Field(default_factory=list)
    min_confidence: float = 0.8
    max_patterns: int = 10
    enable_ml: bool = True


class LiquidityGravityConfig(BaseModel):
    """Конфигурация гравитации ликвидности."""

    gravity_threshold: float = 0.5
    spread_multiplier: float = 2.0
    volume_threshold: float = 1000.0
    update_frequency: int = 30
    symbols: List[Symbol] = Field(default_factory=list)
    risk_levels: Dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "extreme": 1.0,
        }
    )


class OrderBookSnapshot(TypedDict):
    """Снимок ордербука."""

    symbol: str
    timestamp: pd.Timestamp
    bids: List[Tuple[Decimal, Decimal]]  # (price, volume)
    asks: List[Tuple[Decimal, Decimal]]  # (price, volume)
    spread: Decimal
    mid_price: Decimal
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    depth: int


class MarketState(TypedDict):
    """Состояние рынка."""

    symbol: Symbol
    timestamp: pd.Timestamp
    price: Decimal
    volume: Decimal
    volatility: float
    trend: str  # "up", "down", "sideways"
    momentum: float
    support_level: Optional[Decimal]
    resistance_level: Optional[Decimal]


class TradingSignal(TypedDict):
    """Торговый сигнал."""

    symbol: Symbol
    timestamp: pd.Timestamp
    action: str  # "buy", "sell", "hold"
    confidence: float
    price: Optional[Decimal]
    quantity: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    metadata: Dict[str, Any]


class PerformanceMetrics(TypedDict):
    """Метрики производительности."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    total_trades: int
    avg_trade_duration: float
    volatility: float


class RiskMetrics(TypedDict):
    """Метрики риска."""

    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    downside_deviation: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float


class MarketRegime(TypedDict):
    """Рыночный режим."""

    regime_type: str  # "trending", "ranging", "volatile", "calm"
    confidence: float
    duration: int  # в минутах
    volatility_level: str  # "low", "medium", "high"
    trend_strength: float
    support_resistance: Dict[str, Decimal]


class SentimentMetrics(TypedDict):
    """Метрики сентимента."""

    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    technical_sentiment: float
    confidence: float
    sources_count: int
    last_update: pd.Timestamp


class MirrorCorrelation(TypedDict):
    """Корреляция между активами."""

    base_asset: Symbol
    mirror_asset: Symbol
    correlation: float
    p_value: float
    lag: int
    confidence: float
    last_update: pd.Timestamp
    sample_size: int


class EntanglementPattern(TypedDict):
    """Паттерн запутанности."""

    pattern_id: str
    pattern_type: str
    confidence: float
    strength: float
    duration: int
    symbols: List[Symbol]
    metadata: Dict[str, Any]
    timestamp: pd.Timestamp


class LiquidityProfile(TypedDict):
    """Профиль ликвидности."""

    symbol: Symbol
    timestamp: pd.Timestamp
    bid_depth: List[Tuple[Decimal, Decimal]]
    ask_depth: List[Tuple[Decimal, Decimal]]
    spread: Decimal
    mid_price: Decimal
    liquidity_score: float
    risk_level: str
    volume_profile: Dict[str, float]


class MarketMemory(TypedDict):
    """Рыночная память."""

    symbol: Symbol
    timestamp: pd.Timestamp
    memory_type: str  # "short", "medium", "long"
    strength: float
    decay_rate: float
    patterns: List[Dict[str, Any]]
    confidence: float


class EvolutionResult(TypedDict):
    """Результат эволюции стратегии."""

    generation: int
    best_fitness: float
    avg_fitness: float
    population_size: int
    mutations: int
    crossovers: int
    elite_size: int
    best_strategy: Dict[str, Any]
    convergence_rate: float


class BacktestResult(TypedDict):
    """Результат бэктестинга."""

    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series
    drawdown_curve: pd.Series


class LiveTradingResult(TypedDict):
    """Результат живой торговли."""

    session_id: UUID
    start_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    total_trades: int
    successful_trades: int
    total_pnl: Decimal
    current_positions: List[Position]
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    status: str  # "active", "paused", "stopped"


class SimulationResult(TypedDict):
    """Результат симуляции."""

    simulation_id: UUID
    config: ExampleConfig
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_seconds: float
    total_events: int
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    market_conditions: Dict[str, Any]
    system_performance: Dict[str, float]
