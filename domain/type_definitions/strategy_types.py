from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
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
from uuid import UUID
import pandas as pd

# Типы идентификаторов для стратегий
StrategyId = NewType("StrategyId", UUID)


class MarketCondition(str, Enum):
    """Рыночные условия"""

    TRENDING = "trending"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    VOLATILE = "volatile"


class StrategyDirection(str, Enum):
    """Направления торговых сигналов"""

    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    CLOSE = "close"


class StrategyType(str, Enum):
    """Типы стратегий"""

    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    GRID = "grid"
    MARTINGALE = "martingale"
    HEDGING = "hedging"
    MANIPULATION = "manipulation"
    REVERSAL = "reversal"
    SIDEWAYS = "sideways"
    ADAPTIVE = "adaptive"
    EVOLVABLE = "evolvable"
    DEEP_LEARNING = "deep_learning"
    RANDOM_FOREST = "random_forest"
    REGIME_ADAPTIVE = "regime_adaptive"


class MarketRegime(str, Enum):
    """Рыночные режимы"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    MANIPULATION = "manipulation"


@dataclass
class StrategyMetrics:
    """Метрики производительности стратегии"""

    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    volatility: float = 0.0
    mar_ratio: float = 0.0
    ulcer_index: float = 0.0
    omega_ratio: float = 0.0
    gini_coefficient: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    drawdown_duration: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    median_trade: float = 0.0
    median_duration: float = 0.0
    profit_streak: int = 0
    loss_streak: int = 0
    stability: float = 0.0
    calmar_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    kappa_ratio: float = 0.0
    gain_loss_ratio: float = 0.0
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Торговый сигнал"""

    direction: StrategyDirection
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_type: Optional[StrategyType] = None
    market_regime: Optional[MarketRegime] = None
    risk_score: float = 0.0
    expected_return: float = 0.0
    holding_period: Optional[int] = None
    position_size: Optional[float] = None


@dataclass
class StrategyConfig:
    """Конфигурация стратегии"""

    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    confidence_threshold: float = 0.7
    use_stop_loss: bool = True
    use_take_profit: bool = True
    trailing_stop: bool = False
    trailing_stop_activation: float = 0.02
    trailing_stop_distance: float = 0.01
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    symbols: List[str] = field(default_factory=list)
    market_regime_filter: List[MarketRegime] = field(default_factory=list)
    enabled: bool = True
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyAnalysis:
    """Результат анализа стратегии"""

    strategy_id: str
    timestamp: datetime
    market_data: pd.DataFrame
    indicators: Dict[str, pd.Series]
    signals: List[Signal]
    metrics: StrategyMetrics
    market_regime: MarketRegime
    confidence: float
    risk_assessment: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyValidationResult(TypedDict, total=False):
    """Результат валидации стратегии"""

    errors: List[str]
    warnings: List[str]
    is_valid: bool
    validation_score: float
    recommendations: List[str]


class StrategyOptimizationResult(TypedDict, total=False):
    """Результат оптимизации стратегии"""

    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    improvement_expected: bool
    optimization_method: str
    performance_improvement: float
    risk_adjustment: float
    confidence_interval: Tuple[float, float]


class StrategyPerformanceResult(TypedDict, total=False):
    """Результат анализа производительности"""

    analysis: Dict[str, Any]
    metrics: StrategyMetrics
    backtest_results: Dict[str, Any]
    risk_metrics: Dict[str, float]
    comparison_benchmark: Dict[str, float]


@runtime_checkable
class StrategyServiceProtocol(Protocol):
    """Протокол сервиса стратегий"""

    async def create_strategy(self, config: StrategyConfig) -> Any: ...
    async def validate_strategy(self, strategy: Any) -> StrategyValidationResult: ...
    async def optimize_strategy(
        self, strategy: Any, historical_data: pd.DataFrame
    ) -> StrategyOptimizationResult: ...
    async def analyze_performance(
        self, strategy: Any, period: Any
    ) -> StrategyPerformanceResult: ...
    async def backtest_strategy(
        self, strategy: Any, data: pd.DataFrame
    ) -> Dict[str, Any]: ...
    async def get_strategy_metrics(self, strategy_id: str) -> StrategyMetrics: ...
    async def update_strategy_config(
        self, strategy_id: str, config: StrategyConfig
    ) -> bool: ...
@runtime_checkable
class StrategyProtocol(Protocol):
    """Протокол стратегии"""

    def analyze(self, data: pd.DataFrame) -> StrategyAnalysis: ...
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]: ...
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]: ...
    def calculate_position_size(
        self, signal: Signal, account_balance: float
    ) -> float: ...
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]: ...
    def update_metrics(self, signal: Signal, result: Dict[str, Any]) -> None: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def save_state(self) -> None: ...
    def load_state(self) -> None: ...
class StrategyFactoryProtocol(Protocol):
    """Протокол фабрики стратегий"""

    def create_strategy(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> StrategyProtocol: ...
    def get_available_strategies(self) -> List[StrategyType]: ...
    def validate_strategy_config(
        self, config: StrategyConfig
    ) -> StrategyValidationResult: ...


# Типы для эволюционных стратегий
@dataclass
class EvolutionConfig:
    """Конфигурация эволюции стратегии"""

    learning_rate: float = 1e-3
    adaptation_rate: float = 0.01
    evolution_threshold: float = 0.5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5


@dataclass
class EvolutionMetrics:
    """Метрики эволюции"""

    generation: int
    best_fitness: float
    avg_fitness: float
    diversity: float
    convergence_rate: float
    improvement_rate: float
    adaptation_success: float


# Типы для адаптивных стратегий
@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""

    adaptation_threshold: float = 0.7
    learning_rate: float = 0.01
    memory_size: int = 1000
    adaptation_frequency: int = 100
    regime_detection_sensitivity: float = 0.8


@dataclass
class MarketContext:
    """Контекст рынка для адаптации"""

    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: Dict[str, float]
    liquidity_conditions: Dict[str, float]
    market_sentiment: float
    correlation_matrix: pd.DataFrame
    timestamp: datetime = field(default_factory=datetime.now)


# Типы для ML стратегий
@dataclass
class MLModelConfig:
    """Конфигурация ML модели"""

    model_type: str
    input_features: List[str]
    output_features: List[str]
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    model_path: Optional[str] = None


@dataclass
class MLPrediction:
    """Предсказание ML модели"""

    model_id: str
    timestamp: datetime
    input_features: Dict[str, float]
    predictions: Dict[str, float]
    confidence: float
    uncertainty: float
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Типы для риск-менеджмента
@dataclass
class RiskConfig:
    """Конфигурация риск-менеджмента"""

    stress_test_scenarios: List[Dict[str, Any]]
    risk_budget: Dict[str, float]
    max_drawdown: float = 0.2
    max_position_size: float = 0.1
    max_correlation: float = 0.7
    var_confidence: float = 0.95
    stop_loss_multiplier: float = 2.0
    take_profit_multiplier: float = 3.0


@dataclass
class RiskAssessment:
    """Оценка риска"""

    portfolio_var: float
    position_var: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    model_risk: float
    total_risk_score: float
    risk_decomposition: Dict[str, float]
    stress_test_results: Dict[str, float]
    recommendations: List[str]


@dataclass
class TrendFollowingParams:
    """Параметры стратегии следования за трендом"""
    short_period: int = 10
    long_period: int = 20
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    trend_strength_threshold: float = 0.7
    trend_confirmation_period: int = 3
    volume_confirmation: bool = True
    volume_threshold: float = 1.5

@dataclass
class MeanReversionParams:
    """Параметры стратегии возврата к среднему"""
    lookback_period: int = 20
    std_dev_multiplier: float = 2.0
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    deviation_threshold: float = 2.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    min_reversion_probability: float = 0.6
    max_holding_period: int = 24

@dataclass
class BreakoutParams:
    """Параметры стратегии пробоя"""
    breakout_period: int = 20
    volume_multiplier: float = 1.5
    atr_period: int = 14
    atr_multiplier: float = 2.0
    breakout_threshold: float = 1.5
    volume_confirmation_period: int = 3
    support_resistance_period: int = 20
    level_tolerance: float = 0.001
    confirmation_period: int = 2
    false_breakout_filter: bool = True
    min_volatility: float = 0.01
    max_volatility: float = 0.1

@dataclass
class ScalpingParams:
    """Параметры скальпинг стратегии"""
    entry_threshold: float = 0.001
    exit_threshold: float = 0.002
    max_holding_time: int = 300
    volume_threshold: float = 1000.0
    profit_threshold: float = 0.001
    stop_loss: float = 0.0005
    max_hold_time: int = 300
    min_hold_time: int = 10
    min_volatility: float = 0.0001
    max_volatility: float = 0.01
    min_volume: float = 1000.0
    volume_spike_threshold: float = 2.0
    execution_timeout: int = 5
    max_slippage: float = 0.0001

@dataclass
class ArbitrageParams:
    """Параметры арбитражной стратегии"""
    min_spread: float = 0.001
    max_execution_time: int = 60
    correlation_threshold: float = 0.8
    max_slippage: float = 0.0005
    max_hold_time: int = 60
    min_liquidity: float = 10000.0
    max_order_size: float = 1000.0
    max_total_fees: float = 0.002
    include_fees_in_calculation: bool = True
    exchange_monitoring_interval: int = 1
    price_update_frequency: int = 1

@dataclass
class StrategyParameters:
    """Базовые параметры стратегии"""
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    confidence_threshold: float = 0.7
    use_stop_loss: bool = True
    use_take_profit: bool = True
    trailing_stop: bool = False
    trailing_stop_activation: float = 0.02
    trailing_stop_distance: float = 0.01

class StrategyCategory(str, Enum):
    """Категории стратегий"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    GRID = "grid"
    MARTINGALE = "martingale"
    HEDGING = "hedging"
    MANIPULATION = "manipulation"
    REVERSAL = "reversal"
    SIDEWAYS = "sideways"
    ADAPTIVE = "adaptive"
    EVOLVABLE = "evolvable"
    DEEP_LEARNING = "deep_learning"
    RANDOM_FOREST = "random_forest"
    REGIME_ADAPTIVE = "regime_adaptive"

class Timeframe(str, Enum):
    """Временные интервалы"""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class TimeHorizon(str, Enum):
    """Временной горизонт"""
    ULTRA_SHORT = "ultra_short"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    ULTRA_LONG = "ultra_long"


class RiskProfile(str, Enum):
    """Профиль риска"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
