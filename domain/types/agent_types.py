"""
Типы для агентов торговой системы.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd

# Базовые типы для агентов
AgentId = NewType("AgentId", str)
AgentType = Literal[
    "market_maker",
    "whale_analyzer",
    "risk_manager",
    "portfolio_optimizer",
    "order_executor",
    "news_analyzer",
    "market_regime",
    "strategy_agent",
    "meta_controller",
    "evolutionary_agent",
    "social_media",
    "entanglement_detector",
]
# Типы для сигналов агентов
SignalConfidence = NewType("SignalConfidence", Decimal)
SignalPriority = NewType("SignalPriority", int)
SignalStrength = NewType("SignalStrength", Decimal)
# Типы для рыночных данных агентов
OrderBookDepth = NewType("OrderBookDepth", int)
LiquidityScore = NewType("LiquidityScore", Decimal)
SpreadValue = NewType("SpreadValue", Decimal)
VolumeImbalance = NewType("VolumeImbalance", Decimal)
# Типы для анализа китов
WhaleActivityLevel = Literal["low", "medium", "high", "extreme"]
WhaleIntentType = Literal[
    "accumulation", "distribution", "manipulation", "liquidity_hunting"
]
WhalePatternType = Literal["iceberg", "spoofing", "layering", "momentum_break"]
# Типы для анализа рисков
RiskMetricType = Literal["var", "cvar", "volatility", "drawdown", "correlation", "beta"]
RiskLevelType = Literal["low", "medium", "high", "extreme"]
# Типы для портфельного анализа
PortfolioWeight = NewType("PortfolioWeight", Decimal)
DiversificationScore = NewType("DiversificationScore", Decimal)
CorrelationMatrix = NewType("CorrelationMatrix", np.ndarray)
# Типы для исполнения ордеров
ExecutionSpeed = NewType("ExecutionSpeed", int)  # milliseconds
SlippageTolerance = NewType("SlippageTolerance", Decimal)
OrderFillRate = NewType("OrderFillRate", Decimal)
# Типы для новостного анализа
NewsSentimentType = Literal["positive", "negative", "neutral", "mixed"]
NewsImpactLevel = Literal["low", "medium", "high", "critical"]
NewsSourceType = Literal["official", "social_media", "analyst", "rumor"]
# Типы для рыночных режимов
MarketRegimeType = Literal[
    "trending", "ranging", "volatile", "breakout", "consolidation"
]
RegimeConfidence = NewType("RegimeConfidence", Decimal)
# Новые типы для строгой типизации
Symbol = NewType("Symbol", str)
Confidence = NewType("Confidence", float)
RiskScore = NewType("RiskScore", float)
PerformanceScore = NewType("PerformanceScore", float)
# Константы
MAX_CONFIDENCE: Final[float] = 1.0
MIN_CONFIDENCE: Final[float] = 0.0
MAX_RISK_SCORE: Final[float] = 1.0
MIN_RISK_SCORE: Final[float] = 0.0


class AgentStatus(Enum):
    """Статус агента."""

    INITIALIZING = auto()
    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    EVOLVING = auto()
    LEARNING = auto()


class RiskLevel(Enum):
    """Уровень риска."""

    MINIMAL = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class MarketPhase(Enum):
    """Фаза рынка."""

    ACCUMULATION = auto()
    MARKUP = auto()
    DISTRIBUTION = auto()
    MARKDOWN = auto()
    BREAKOUT_SETUP = auto()
    BREAKOUT_ACTIVE = auto()
    EXHAUSTION = auto()
    REVERSION_POTENTIAL = auto()


@dataclass(frozen=True)
class AgentMetrics:
    """Метрики агента."""

    name: str
    agent_type: AgentType
    status: AgentStatus
    performance_score: float = 0.0
    confidence: float = 0.0
    risk_score: float = 0.0
    total_processed: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_error: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.success_count + self.error_count != self.total_processed:
            raise ValueError("Total processed must equal success + error count")
        if self.success_count > self.total_processed:
            raise ValueError("Success count cannot exceed total processed")
        if not 0.0 <= self.performance_score <= 1.0:
            raise ValueError("Performance score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("Risk score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class AgentState:
    """Состояние агента."""

    agent_id: AgentId
    status: AgentStatus
    is_running: bool
    is_healthy: bool
    performance_score: float
    confidence: float
    risk_score: float
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentConfig(TypedDict, total=False):
    """Конфигурация агента."""

    name: str
    agent_type: AgentType
    max_position_size: float
    max_portfolio_risk: float
    max_risk_per_trade: float
    confidence_threshold: float
    risk_threshold: float
    performance_threshold: float
    rebalance_interval: int
    processing_timeout_ms: int
    retry_attempts: int
    enable_evolution: bool
    enable_learning: bool
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Результат обработки данных агентом."""

    success: bool
    data: Dict[str, Any]
    confidence: float
    risk_score: float
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AgentEvolutionResult:
    """Результат эволюции агента."""

    success: bool
    performance_improvement: float
    confidence_improvement: float
    risk_reduction: float
    evolution_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """Результат обучения агента."""

    success: bool
    loss_reduction: float
    accuracy_improvement: float
    learning_rate: float
    epochs_completed: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IAgent(Protocol):
    """Протокол для всех агентов."""

    @property
    def agent_id(self) -> AgentId:
        """Уникальный идентификатор агента."""
        ...

    @property
    def agent_type(self) -> AgentType:
        """Тип агента."""
        ...

    @property
    def name(self) -> str:
        """Имя агента."""
        ...

    @property
    def state(self) -> AgentState:
        """Текущее состояние агента."""
        ...

    @property
    def metrics(self) -> AgentMetrics:
        """Метрики агента."""
        ...

    async def initialize(self) -> bool:
        """Инициализация агента."""
        ...

    async def start(self) -> bool:
        """Запуск агента."""
        ...

    async def stop(self) -> None:
        """Остановка агента."""
        ...

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных."""
        ...

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        ...

    def is_healthy(self) -> bool:
        """Проверка здоровья агента."""
        ...

    def get_performance(self) -> float:
        """Получение показателя производительности."""
        ...

    def get_confidence(self) -> float:
        """Получение уверенности агента."""
        ...

    def get_risk_score(self) -> float:
        """Получение оценки риска."""
        ...


class IEvolvableAgent(IAgent, Protocol):
    """Протокол для эволюционных агентов."""

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным."""
        ...

    async def learn(self, data: Any) -> LearningResult:
        """Обучение на новых данных."""
        ...

    async def evolve(self, data: Any) -> AgentEvolutionResult:
        """Полная эволюция компонента."""
        ...

    def save_state(self, path: str) -> bool:
        """Сохранение состояния."""
        ...

    def load_state(self, path: str) -> bool:
        """Загрузка состояния."""
        ...


class IRiskAgent(IAgent, Protocol):
    """Протокол для агентов управления рисками."""

    async def calculate_risk_metrics(
        self, market_data: pd.DataFrame, positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Расчет метрик риска."""
        ...

    async def validate_position(
        self, symbol: Symbol, size: float, price: float
    ) -> bool:
        """Валидация позиции."""
        ...

    async def get_risk_level(self) -> RiskLevel:
        """Получение текущего уровня риска."""
        ...

    async def get_position_size_recommendation(
        self, symbol: Symbol, confidence: float
    ) -> float:
        """Получение рекомендации по размеру позиции."""
        ...


class INewsAgent(IAgent, Protocol):
    """Протокол для новостных агентов."""

    async def analyze_sentiment(self, text: str) -> float:
        """Анализ настроений в тексте."""
        ...

    async def get_news_impact(self, symbol: Symbol) -> float:
        """Получение влияния новостей на символ."""
        ...

    async def filter_relevant_news(self, symbol: Symbol) -> List[Dict[str, Any]]:
        """Фильтрация релевантных новостей."""
        ...


class IMarketMakerAgent(IAgent, Protocol):
    """Протокол для агентов маркет-мейкинга."""

    async def detect_patterns(
        self, orderbook_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Детекция паттернов в ордербуке."""
        ...

    async def generate_signals(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов."""
        ...

    async def calculate_spread(self, bid: float, ask: float) -> float:
        """Расчет спреда."""
        ...


class AgentFactory(Protocol):
    """Фабрика для создания агентов."""

    def create_agent(self, agent_type: AgentType, config: AgentConfig) -> IAgent:
        """Создание агента указанного типа."""
        ...

    def create_evolvable_agent(
        self, agent_type: AgentType, config: AgentConfig
    ) -> IEvolvableAgent:
        """Создание эволюционного агента."""
        ...


class AgentRegistry(Protocol):
    """Реестр агентов."""

    def register_agent(self, agent: IAgent) -> None:
        """Регистрация агента."""
        ...

    def unregister_agent(self, agent_id: AgentId) -> None:
        """Отмена регистрации агента."""
        ...

    def get_agent(self, agent_id: AgentId) -> Optional[IAgent]:
        """Получение агента по ID."""
        ...

    def get_agents_by_type(self, agent_type: AgentType) -> List[IAgent]:
        """Получение агентов по типу."""
        ...

    def get_all_agents(self) -> List[IAgent]:
        """Получение всех агентов."""
        ...


# Валидаторы для типов
def validate_confidence(value: float) -> float:
    """Валидация значения уверенности."""
    if not MIN_CONFIDENCE <= value <= MAX_CONFIDENCE:
        raise ValueError(
            f"Confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"
        )
    return value


def validate_risk_score(value: float) -> float:
    """Валидация значения риска."""
    if not MIN_RISK_SCORE <= value <= MAX_RISK_SCORE:
        raise ValueError(
            f"Risk score must be between {MIN_RISK_SCORE} and {MAX_RISK_SCORE}"
        )
    return value


def validate_performance_score(value: float) -> float:
    """Валидация показателя производительности."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("Performance score must be between 0.0 and 1.0")
    return value


# Типизированные словари для агентов
class MarketMakerConfig(TypedDict, total=False):
    """Конфигурация маркет-мейкера."""

    spread_threshold: float
    volume_threshold: float
    fakeout_threshold: float
    liquidity_zone_size: float
    lookback_period: int
    confidence_threshold: float
    analytics_enabled: bool
    entanglement_enabled: bool
    noise_enabled: bool
    mirror_enabled: bool
    gravity_enabled: bool
    mm_pattern_recognition_enabled: bool
    mm_pattern_following_enabled: bool
    mm_pattern_confidence_threshold: float
    mm_pattern_similarity_threshold: float
    mm_pattern_memory_enabled: bool


class WhaleAnalysisConfig(TypedDict, total=False):
    """Конфигурация анализа китов."""

    volume_threshold: float
    cluster_eps: float
    cluster_min_samples: int
    correlation_threshold: float
    momentum_threshold: float
    spoofing_detection_enabled: bool
    iceberg_detection_enabled: bool
    layering_detection_enabled: bool


class RiskAnalysisConfig(TypedDict, total=False):
    """Конфигурация анализа рисков."""

    var_confidence_level: float
    cvar_confidence_level: float
    volatility_window: int
    correlation_window: int
    max_drawdown_threshold: float
    position_limit: float
    leverage_limit: float
    ml_risk_enabled: bool


class PortfolioConfig(TypedDict, total=False):
    """Конфигурация портфельного анализа."""

    max_positions: int
    min_diversification: float
    rebalance_threshold: float
    correlation_threshold: float
    volatility_target: float
    sharpe_target: float
    max_allocation: float


class OrderExecutorConfig(TypedDict, total=False):
    """Конфигурация исполнения ордеров."""

    max_slippage: float
    execution_timeout_ms: int
    retry_attempts: int
    smart_routing_enabled: bool
    iceberg_enabled: bool
    twap_enabled: bool
    vwap_enabled: bool


class NewsAnalysisConfig(TypedDict, total=False):
    """Конфигурация анализа новостей."""

    sentiment_threshold: float
    impact_threshold: float
    source_weights: Dict[str, float]
    keyword_filters: List[str]
    time_decay_factor: float
    correlation_enabled: bool
    ml_sentiment_enabled: bool


class MarketRegimeConfig(TypedDict, total=False):
    """Конфигурация анализа рыночных режимов."""

    regime_window: int
    volatility_threshold: float
    trend_strength_threshold: float
    regime_confidence_threshold: float
    regime_transition_threshold: float
    ml_regime_enabled: bool


# Результаты анализа агентов
class MarketMakerSignal(TypedDict, total=False):
    """Сигнал маркет-мейкера."""

    timestamp: datetime
    pair: str
    signal_type: Literal["spread", "liquidity", "fakeout", "mm_pattern"]
    confidence: float
    details: Dict[str, Any]
    priority: int
    spread: float
    imbalance: float
    liquidity_score: float
    fakeout_detected: bool
    pattern_type: Optional[str]
    expected_direction: Optional[str]
    expected_return: float


class WhaleAnalysisResult(TypedDict, total=False):
    """Результат анализа китов."""

    timestamp: datetime
    pair: str
    whale_activity_level: WhaleActivityLevel
    whale_intent: WhaleIntentType
    whale_pattern: WhalePatternType
    confidence: float
    volume_clusters: List[Dict[str, Any]]
    correlation_score: float
    momentum_score: float
    manipulation_detected: bool
    spoofing_detected: bool
    iceberg_detected: bool
    layering_detected: bool
    expected_impact: str
    risk_level: RiskLevelType


class RiskAnalysisResult(TypedDict, total=False):
    """Результат анализа рисков."""

    timestamp: datetime
    pair: str
    var_95: float
    cvar_95: float
    volatility: float
    max_drawdown: float
    correlation_risk: float
    beta: float
    exposure: float
    risk_level: RiskLevelType
    confidence: float
    recommendations: List[str]
    position_limits: Dict[str, float]
    leverage_recommendations: Dict[str, float]


class PortfolioAnalysisResult(TypedDict, total=False):
    """Результат портфельного анализа."""

    timestamp: datetime
    portfolio_id: str
    total_value: float
    total_pnl: float
    sharpe_ratio: float
    diversification_score: float
    correlation_matrix: CorrelationMatrix
    position_weights: Dict[str, PortfolioWeight]
    rebalance_recommendations: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


class OrderExecutionResult(TypedDict, total=False):
    """Результат исполнения ордера."""

    timestamp: datetime
    order_id: str
    pair: str
    side: str
    quantity: float
    price: float
    executed_quantity: float
    average_price: float
    slippage: float
    execution_time_ms: int
    fill_rate: float
    status: str
    fees: float
    success: bool


class NewsAnalysisResult(TypedDict, total=False):
    """Результат анализа новостей."""

    timestamp: datetime
    pair: str
    sentiment: NewsSentimentType
    impact_level: NewsImpactLevel
    source: NewsSourceType
    confidence: float
    keywords: List[str]
    entities: List[str]
    correlation_with_price: float
    expected_impact: str
    trading_recommendations: List[str]


class MarketRegimeResult(TypedDict, total=False):
    """Результат анализа рыночного режима."""

    timestamp: datetime
    pair: str
    regime_type: MarketRegimeType
    confidence: float
    volatility: float
    trend_strength: float
    regime_duration: int
    transition_probability: float
    indicators: Dict[str, float]
    trading_recommendations: List[str]


# Протоколы для агентов
@runtime_checkable
class AgentProtocol(Protocol):
    """Базовый протокол для всех агентов."""

    def get_agent_id(self) -> AgentId:
        """Получение ID агента."""
        ...

    def get_agent_type(self) -> AgentType:
        """Получение типа агента."""
        ...

    def is_enabled(self) -> bool:
        """Проверка активности агента."""
        ...

    def get_confidence(self) -> SignalConfidence:
        """Получение уверенности агента."""
        ...

    def get_performance(self) -> float:
        """Получение производительности агента."""
        ...

    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Основной метод анализа."""
        ...

    async def adapt(self, data: Any) -> bool:
        """Адаптация агента."""
        ...

    async def learn(self, data: Any) -> bool:
        """Обучение агента."""
        ...

    async def evolve(self, data: Any) -> bool:
        """Эволюция агента."""
        ...

    def save_state(self, path: str) -> bool:
        """Сохранение состояния."""
        ...

    def load_state(self, path: str) -> bool:
        """Загрузка состояния."""
        ...


@runtime_checkable
class MarketMakerProtocol(AgentProtocol, Protocol):
    """Протокол для маркет-мейкера."""

    async def analyze_spread(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ спреда."""
        ...

    async def analyze_liquidity(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Dict[str, float]:
        """Анализ ликвидности."""
        ...

    async def detect_fakeouts(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Обнаружение фейкаутов."""
        ...

    async def analyze_mm_patterns(
        self, symbol: str, order_book: Dict[str, Any], trades: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Анализ паттернов маркет-мейкера."""
        ...


@runtime_checkable
class WhaleAnalyzerProtocol(AgentProtocol, Protocol):
    """Протокол для анализа китов."""

    async def detect_whale_activity(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> WhaleAnalysisResult:
        """Обнаружение активности китов."""
        ...

    async def analyze_volume_clusters(
        self, order_book: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Анализ кластеров объема."""
        ...

    async def detect_manipulation(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Обнаружение манипуляций."""
        ...


@runtime_checkable
class RiskManagerProtocol(AgentProtocol, Protocol):
    """Протокол для управления рисками."""

    async def calculate_risk_metrics(
        self, positions: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> RiskAnalysisResult:
        """Расчет метрик риска."""
        ...

    async def validate_position(self, position: Dict[str, Any]) -> bool:
        """Валидация позиции."""
        ...

    async def get_position_limits(self, symbol: str) -> Dict[str, float]:
        """Получение лимитов позиций."""
        ...


@runtime_checkable
class PortfolioOptimizerProtocol(AgentProtocol, Protocol):
    """Протокол для оптимизации портфеля."""

    async def optimize_portfolio(
        self, positions: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> PortfolioAnalysisResult:
        """Оптимизация портфеля."""
        ...

    async def calculate_weights(
        self, assets: List[str], constraints: Dict[str, Any]
    ) -> Dict[str, PortfolioWeight]:
        """Расчет весов активов."""
        ...

    async def rebalance_portfolio(
        self,
        current_weights: Dict[str, PortfolioWeight],
        target_weights: Dict[str, PortfolioWeight],
    ) -> List[Dict[str, Any]]:
        """Ребалансировка портфеля."""
        ...


@runtime_checkable
class OrderExecutorProtocol(AgentProtocol, Protocol):
    """Протокол для исполнения ордеров."""

    async def execute_order(self, order: Dict[str, Any]) -> OrderExecutionResult:
        """Исполнение ордера."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        ...

    async def get_order_status(self, order_id: str) -> str:
        """Получение статуса ордера."""
        ...

    async def calculate_slippage(
        self, order: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """Расчет проскальзывания."""
        ...


@runtime_checkable
class NewsAnalyzerProtocol(AgentProtocol, Protocol):
    """Протокол для анализа новостей."""

    async def analyze_sentiment(
        self, news_data: List[Dict[str, Any]]
    ) -> NewsAnalysisResult:
        """Анализ настроений."""
        ...

    async def extract_entities(self, text: str) -> List[str]:
        """Извлечение сущностей."""
        ...

    async def calculate_impact(
        self, news_data: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """Расчет влияния новости."""
        ...


@runtime_checkable
class MarketRegimeProtocol(AgentProtocol, Protocol):
    """Протокол для анализа рыночных режимов."""

    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegimeResult:
        """Обнаружение режима."""
        ...

    async def calculate_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Расчет индикаторов."""
        ...

    async def predict_regime_transition(self, market_data: pd.DataFrame) -> float:
        """Предсказание смены режима."""
        ...


# Типы для эволюционных агентов
class EvolutionaryConfig(TypedDict, total=False):
    """Конфигурация эволюционного агента."""

    exploration_rate: float
    learning_rate: float
    adaptation_rate: float
    max_strategies: int
    evolution_threshold: float
    mutation_rate: float
    crossover_rate: float
    population_size: int


class EvolutionResult(TypedDict, total=False):
    """Результат эволюции."""

    timestamp: datetime
    agent_id: str
    evolution_type: Literal["architecture", "parameters", "strategy"]
    success: bool
    performance_improvement: float
    confidence_improvement: float
    new_config: Dict[str, Any]
    metadata: Dict[str, Any]


# Типы для интеграции агентов
class AgentIntegrationConfig(TypedDict, total=False):
    """Конфигурация интеграции агентов."""

    enabled_agents: List[AgentType]
    integration_mode: Literal["sequential", "parallel", "hierarchical"]
    priority_order: List[AgentType]
    conflict_resolution: Literal[
        "highest_confidence", "weighted_average", "majority_vote"
    ]
    aggregation_method: Literal["simple_average", "weighted_average", "ensemble"]


class AgentIntegrationResult(TypedDict, total=False):
    """Результат интеграции агентов."""

    timestamp: datetime
    symbol: str
    aggregated_signal: Dict[str, Any]
    agent_contributions: Dict[AgentType, Dict[str, Any]]
    confidence: float
    consensus_level: float
    conflicts_resolved: List[str]
    execution_recommendations: List[Dict[str, Any]]
