"""
Рефакторенный DI контейнер с разделением ответственностей согласно SOLID принципам.
Разделен на:
- RepositoryRegistry: регистрация репозиториев
- ServiceRegistry: регистрация сервисов
- AgentRegistry: регистрация агентов
- UseCaseRegistry: регистрация use cases
- ConfigurationManager: управление конфигурацией
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from dependency_injector import containers, providers

from application.evolution.evolution_orchestrator import EvolutionOrchestrator
from application.services.market_service import MarketService
from application.services.ml_service import MLService
from application.services.order_validator import OrderValidator
from application.services.portfolio_service import PortfolioService
from application.services.risk_service import RiskService

# Application Services
from application.services.service_factory import get_service_factory
# from application.services.strategy_service import StrategyService
from application.services.trading_service import TradingService
from application.symbol_selection.opportunity_selector import (
    DynamicOpportunityAwareSymbolSelector,
)
from application.use_cases.manage_orders import (
    CreateOrderRequest,
    CreateOrderResponse,
    DefaultOrderManagementUseCase,
    OrderManagementUseCase,
)
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
)

# Use Cases
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
)

# Импорты модуля evolution
from domain.evolution import (
    EvolutionContext,
    StrategyCandidate,
    StrategyFitnessEvaluator,
    StrategyGenerator,
    StrategySelector,
)
from domain.evolution.strategy_optimizer import StrategyOptimizer

# Добавляем недостающие импорты для анализаторов
from domain.services.spread_analysis import SpreadAnalyzer
from domain.services.liquidity_analysis import LiquidityAnalyzer
from domain.services.ml_predictor import MLPredictor
from infrastructure.external_services.risk_analysis_adapter import RiskAnalysisServiceAdapter
from infrastructure.external_services.technical_analysis_adapter import TechnicalAnalysisServiceAdapter
from infrastructure.services.market_conditions_analyzer import MarketConditionsAnalyzer
from infrastructure.repositories.in_memory_repositories import (
    InMemoryMarketRepository,
    InMemoryMLRepository,
    InMemoryPortfolioRepository,
    InMemoryRiskRepository,
    InMemoryStrategyRepository,
    InMemoryTradingRepository,
)
from domain.protocols.repository_protocol import (
    MarketRepositoryProtocol,
    MLRepositoryProtocol,
    OrderRepositoryProtocol,
    PortfolioRepositoryProtocol,
    PositionRepositoryProtocol,
    RiskRepositoryProtocol,
    StrategyRepositoryProtocol,
    TradingPairRepositoryProtocol,
    TradingRepositoryProtocol,
)
from domain.protocols.service_protocols import (
    CacheServiceProtocol,
    ConfigurationServiceProtocol,
    DatabaseServiceProtocol,
    LoggingServiceProtocol,
    MarketDataServiceProtocol,
    MetricsServiceProtocol,
    MLServiceProtocol,
    NotificationServiceProtocol,
    RiskServiceProtocol,
    # StrategyServiceProtocol,
    TradingServiceProtocol,
)
from domain.services.correlation_chain import DefaultCorrelationChain
from domain.services.market_metrics import MarketMetricsService
from domain.services.pattern_discovery import PatternConfig, PatternDiscovery
from domain.services.risk_analysis import (
    DefaultRiskAnalysisService,
    RiskAnalysisService,
)
from domain.services.signal_service import DefaultSignalService, SignalService

# Domain Services
# from domain.services.strategy_service import DefaultStrategyService, StrategyService
from domain.services.technical_analysis import (
    DefaultTechnicalAnalysisService,
    ITechnicalAnalysisService,
)
from domain.symbols.market_phase_classifier import MarketPhaseClassifier
from domain.symbols.opportunity_score import OpportunityScoreCalculator
from domain.symbols.symbol_validator import SymbolValidator
from domain.symbols.cache import SymbolCacheManager
from domain.sessions.factories import SessionRepositoryConfig
from domain.sessions.implementations import SessionService
from infrastructure.agents.market_maker.model_agent import MarketMakerModelAgent
from infrastructure.strategies.trend_strategy import TrendStrategy
from infrastructure.strategies.sideways_strategy import SidewaysStrategy
from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
from infrastructure.strategies.evolvable_base_strategy import EvolvableBaseStrategy
from infrastructure.strategies.manipulation_strategy import ManipulationStrategy
from infrastructure.strategies.volatility_strategy import VolatilityStrategy
from infrastructure.strategies.pairs_trading_strategy import PairsTradingStrategy
from infrastructure.external_services.bybit_client import BybitClient
from infrastructure.external_services.account_manager import AccountManager

# Session Services
from domain.sessions.services import SessionService

# Новые компоненты domain/strategies
from domain.strategies import (
    StrategyFactory,
    StrategyRegistry,
    StrategyValidator,
    get_strategy_factory,
    get_strategy_registry,
    get_strategy_validator,
)

# Импорты модулей domain/symbols
from domain.symbols import (
    MarketPhase,
    MarketPhaseClassifier,
    OpportunityScoreCalculator,
    SymbolCacheManager,
    SymbolProfile,
    SymbolValidator,
)

# Infrastructure
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
from infrastructure.evolution import (
    EvolutionBackup,
    EvolutionCache,
    EvolutionMigration,
    StrategyStorage,
)
from infrastructure.external_services.account_manager import AccountManager

# External Services
from infrastructure.external_services.bybit_client import BybitClient

# Market Profiles Module
# from infrastructure.market_profiles import (
#     AnalysisConfig,
#     BehaviorHistoryRepository,
#     MarketMakerStorage,
#     PatternAnalyzer,
#     PatternMemoryRepository,
#     SimilarityCalculator,
#     StorageConfig,
#     SuccessRateAnalyzer,
# )
from infrastructure.market_profiles.interfaces.storage_interfaces import (
    IBehaviorHistoryStorage,
    IPatternAnalyzer,
    IPatternStorage,
)
from infrastructure.repositories.market_repository import InMemoryMarketRepository
from infrastructure.repositories.ml_repository import InMemoryMLRepository
from infrastructure.repositories.portfolio_repository import InMemoryPortfolioRepository
from infrastructure.repositories.risk_repository import InMemoryRiskRepository
from infrastructure.repositories.strategy_repository import InMemoryStrategyRepository
from infrastructure.repositories.trading.trading_repository import (
    InMemoryTradingRepository,
)
# from infrastructure.sessions import (
#     SessionAnalytics,
#     SessionCache,
#     SessionMetricsCalculator,
#     SessionMonitor,
#     SessionOptimizer,
#     SessionPatternRecognizer,
#     SessionPredictor,
#     SessionRepository,
#     SessionRiskAnalyzer,
#     SessionTransitionManager,
#     SessionValidator,
# )
from infrastructure.sessions.session_repository import SessionRepositoryConfig
from infrastructure.strategies.adaptive.adaptive_strategy_generator import (
    AdaptiveStrategyGenerator,
)
from infrastructure.strategies.evolution.evolvable_base_strategy import (
    EvolvableBaseStrategy,
)
from infrastructure.strategies.manipulation_strategies import ManipulationStrategy
from infrastructure.strategies.pairs_trading_strategy import PairsTradingStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy

# Импорты стратегий из infrastructure/strategies
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.volatility_strategy import VolatilityStrategy
from shared.abstractions.base_service import BaseService


@dataclass
class ContainerConfig:
    """Конфигурация контейнера."""

    # Основные сервисы
    cache_enabled: bool = True
    risk_management_enabled: bool = True
    technical_analysis_enabled: bool = True
    signal_processing_enabled: bool = True
    strategy_management_enabled: bool = True
    pattern_discovery_enabled: bool = True
    # Модули domain/symbols
    symbols_analysis_enabled: bool = True
    doass_enabled: bool = True
    # Агенты
    agent_whales_enabled: bool = True
    agent_risk_enabled: bool = True
    agent_portfolio_enabled: bool = True
    agent_meta_controller_enabled: bool = True
    # ML и AI
    evolution_enabled: bool = True
    meta_learning_enabled: bool = True
    # Интеграции
    news_integration_enabled: bool = True
    liquidity_monitoring_enabled: bool = True
    entanglement_monitoring_enabled: bool = True


class Registry(ABC):
    """Абстрактный базовый класс для реестров (оставлен только для динамики, без ручных фабрик)."""

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}

    def register_instance(self, name: str, instance: Any) -> None:
        self._instances[name] = instance

    def get(self, name: str) -> Any:
        return self._instances.get(name)

    def has(self, name: str) -> bool:
        return name in self._instances


class RepositoryRegistry(Registry):
    pass


class ServiceRegistry(Registry):
    pass


class AgentRegistry(Registry):
    pass


class UseCaseRegistry(Registry):
    pass


class ConfigurationManager:
    """Менеджер конфигурации."""

    def __init__(self, config: ContainerConfig) -> None:
        self.config = config

    def is_enabled(self, feature: str) -> bool:
        """Проверка включения функции."""
        return getattr(self.config, feature, False)

    def get_config(self) -> ContainerConfig:
        """Получение конфигурации."""
        return self.config


class Container(containers.DeclarativeContainer):
    """Контейнер зависимостей с улучшенной архитектурой"""

    # Конфигурация
    config = providers.Configuration()
    # Repositories (перемещаем выше для зависимостей)
    market_repository = providers.Singleton(InMemoryMarketRepository)
    ml_repository = providers.Singleton(InMemoryMLRepository)
    portfolio_repository = providers.Singleton(InMemoryPortfolioRepository)
    risk_repository = providers.Singleton(InMemoryRiskRepository)
    strategy_repository = providers.Singleton(InMemoryStrategyRepository)
    trading_repository = providers.Singleton(InMemoryTradingRepository)
    
    # Domain Services
    spread_analyzer: providers.Provider[SpreadAnalyzer] = providers.Singleton(SpreadAnalyzer, config=config.spread_analyzer)
    liquidity_analyzer: providers.Provider[LiquidityAnalyzer] = providers.Singleton(
        LiquidityAnalyzer, config=config.liquidity_analyzer
    )
    ml_predictor: providers.Provider[MLPredictor] = providers.Singleton(MLPredictor, config=config.ml_predictor)
    technical_analysis_service = providers.Singleton(DefaultTechnicalAnalysisService)
    market_metrics_service = providers.Singleton(MarketMetricsService)
    market_conditions_analyzer = providers.Singleton(
        MarketConditionsAnalyzer,
        market_repository=market_repository,
        technical_analysis_service=technical_analysis_service
    )
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
    # strategy_service = providers.Singleton(DefaultStrategyService)
    pattern_discovery = providers.Singleton(
        PatternDiscovery,
        config=providers.Callable(
            PatternConfig,
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.6,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=["candle", "price", "volume"],
            feature_columns=["open", "high", "low", "close", "volume"],
            window_sizes=[5, 10, 20],
            similarity_threshold=0.8,
        ),
    )
    risk_analysis_service = providers.Singleton(DefaultRiskAnalysisService)
    # Новые компоненты domain/strategies
    strategy_factory = providers.Singleton(get_strategy_factory)
    strategy_registry = providers.Singleton(get_strategy_registry)
    strategy_validator = providers.Singleton(get_strategy_validator)
    # Стратегии из infrastructure/strategies
    trend_strategy = providers.Singleton(TrendStrategy)
    sideways_strategy = providers.Singleton(SidewaysStrategy)
    adaptive_strategy_generator = providers.Singleton(AdaptiveStrategyGenerator)
    evolvable_base_strategy = providers.Singleton(EvolvableBaseStrategy)
    manipulation_strategy = providers.Singleton(ManipulationStrategy)
    volatility_strategy = providers.Singleton(VolatilityStrategy)
    pairs_trading_strategy = providers.Singleton(PairsTradingStrategy)
    # Модули domain/symbols
    market_phase_classifier = providers.Singleton(MarketPhaseClassifier)
    opportunity_score_calculator = providers.Singleton(OpportunityScoreCalculator)
    symbol_validator = providers.Singleton(SymbolValidator)
    symbol_cache = providers.Singleton(SymbolCacheManager)
    doass_selector = providers.Singleton(
        DynamicOpportunityAwareSymbolSelector,
        market_phase_classifier=market_phase_classifier,
        opportunity_calculator=opportunity_score_calculator,
    )
    # Session Infrastructure (перемещаем выше)
    session_repository_config: providers.Provider[SessionRepositoryConfig] = providers.Singleton(SessionRepositoryConfig)
    session_repository: providers.Provider[Any] = providers.Singleton(
        Any, config=session_repository_config
    )
    session_cache: providers.Provider[Any] = providers.Singleton(Any, max_size=1000, ttl_seconds=300)
    session_validator: providers.Provider[Any] = providers.Singleton(Any)
    session_metrics_calculator: providers.Provider[Any] = providers.Singleton(Any)
    session_pattern_recognizer: providers.Provider[Any] = providers.Singleton(Any)
    session_transition_manager: providers.Provider[Any] = providers.Singleton(Any)
    session_predictor: providers.Provider[Any] = providers.Singleton(Any)
    session_optimizer: providers.Provider[Any] = providers.Singleton(Any)
    session_monitor: providers.Provider[Any] = providers.Singleton(Any)
    session_analytics: providers.Provider[Any] = providers.Singleton(Any)
    session_risk_analyzer: providers.Provider[Any] = providers.Singleton(Any)
    # Session Services
    session_service = providers.Singleton(
        SessionService,
        registry=providers.Dependency(),
        session_marker=providers.Dependency(),
        influence_analyzer=providers.Dependency(),
        transition_manager=providers.Dependency(),
        cache=providers.Dependency(),
        validator=providers.Dependency(),
        data_repository=providers.Dependency(),
        repository=session_repository,
        session_cache=session_cache,
        session_validator=session_validator,
        metrics_calculator=session_metrics_calculator,
        pattern_recognizer=session_pattern_recognizer,
        transition_manager_new=session_transition_manager,
        predictor=session_predictor,
        optimizer=session_optimizer,
        monitor=session_monitor,
        analytics=session_analytics,
        risk_analyzer=session_risk_analyzer,
    )
    # Service Factory
    service_factory = providers.Singleton(get_service_factory, config=config)
    # Application Services (через фабрику)
    market_service = providers.Singleton(
        lambda factory: factory.create_market_service(), service_factory=service_factory
    )
    ml_service = providers.Singleton(
        lambda factory: factory.create_ml_service(), service_factory=service_factory
    )
    portfolio_service = providers.Singleton(
        lambda factory: factory.create_portfolio_service(),
        service_factory=service_factory,
    )
    risk_service = providers.Singleton(
        lambda factory: factory.create_risk_service(), service_factory=service_factory
    )
    strategy_service_app = providers.Singleton(
        lambda factory: factory.create_strategy_service(),
        service_factory=service_factory,
    )
    trading_service = providers.Singleton(
        lambda factory: factory.create_trading_service(),
        service_factory=service_factory,
    )
    cache_service = providers.Singleton(
        lambda factory: factory.create_cache_service(), service_factory=service_factory
    )
    notification_service = providers.Singleton(
        lambda factory: factory.create_notification_service(),
        service_factory=service_factory,
    )
    order_validator = providers.Singleton(
        OrderValidator, market_service=market_service, risk_service=risk_service
    )
    # Use Cases
    manage_orders_use_case = providers.Singleton(
        DefaultOrderManagementUseCase,
        order_validator=order_validator,
        trading_service=trading_service,
        market_service=market_service,
    )
    manage_positions_use_case = providers.Singleton(
        DefaultPositionManagementUseCase,
        portfolio_service=portfolio_service,
        risk_service=risk_service,
        market_service=market_service,
    )
    manage_risk_use_case = providers.Singleton(
        DefaultRiskManagementUseCase,
        risk_service=risk_service,
        market_service=market_service,
        risk_analysis=risk_analysis_service,
    )
    manage_trading_pairs_use_case = providers.Singleton(
        DefaultTradingPairManagementUseCase,
        market_service=market_service,
        strategy_service=strategy_service_app,
    )
    trading_orchestrator_use_case = providers.Singleton(
        DefaultTradingOrchestratorUseCase,
        session_service=session_service,
        trading_service=trading_service,
        strategy_factory=strategy_factory,
        strategy_registry=strategy_registry,
        strategy_validator=strategy_validator,
    )
    # Infrastructure
    market_maker_agent = providers.Singleton(
        MarketMakerModelAgent, config=config.market_maker_agent
    )
    # Market Profiles Components
    market_maker_storage: providers.Provider[Any] = providers.Singleton(
        Any, config=providers.Callable(Any)
    )
    pattern_memory_repository: providers.Provider[Any] = providers.Singleton(Any)
    behavior_history_repository: providers.Provider[Any] = providers.Singleton(Any)
    pattern_analyzer: providers.Provider[Any] = providers.Singleton(
        Any, config=providers.Callable(Any)
    )
    similarity_calculator: providers.Provider[Any] = providers.Singleton(Any)
    success_rate_analyzer: providers.Provider[Any] = providers.Singleton(Any)
    # Evolution Components
    strategy_storage: providers.Provider[Any] = providers.Singleton(Any)
    evolution_cache: providers.Provider[Any] = providers.Singleton(Any)
    evolution_backup: providers.Provider[Any] = providers.Singleton(Any)
    evolution_migration: providers.Provider[Any] = providers.Singleton(Any)
    strategy_fitness_evaluator: providers.Provider[Any] = providers.Singleton(Any)
    strategy_generator: providers.Provider[Any] = providers.Singleton(Any)
    strategy_optimizer: providers.Provider[Any] = providers.Singleton(Any)
    strategy_selector: providers.Provider[Any] = providers.Singleton(Any)
    evolution_orchestrator: providers.Provider[Any] = providers.Singleton(Any)
    # External Services
    bybit_client: providers.Provider[Any] = providers.Singleton(
        Any,
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=config.bybit.testnet,
    )
    account_manager: providers.Provider[Any] = providers.Singleton(Any, bybit_client=bybit_client)
    # Risk and Technical Analysis Services
    risk_analysis_service_external: providers.Provider[Any] = providers.Singleton(Any)
    technical_analysis_service_external: providers.Provider[Any] = providers.Singleton(
        Any
    )


class ServiceLocator:
    """Локатор сервисов для упрощенного доступа."""

    def __init__(self, container: Container) -> None:
        """Инициализация контейнера зависимостей."""
        self.container = container
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._config: Dict[str, Any] = {}

    def get_service(self, service_type: Type[BaseService]) -> BaseService:
        """Получение сервиса по типу."""
        # Маппинг типов на имена сервисов
        service_mapping = {
            MarketService: "market_service",
            MLService: "ml_service",
            PortfolioService: "portfolio_service",
            RiskService: "risk_service",
            # StrategyService: "strategy_service_app",
            TradingService: "trading_service",
        }
        service_name = service_mapping.get(service_type)
        if service_name:
            return self.container.providers[service_name]()  # type: ignore[no-any-return]
        raise ValueError(f"Unknown service type: {service_type}")

    def get_repository(self, repository_type: Type) -> Any:
        """Получение репозитория по типу."""
        # Маппинг типов на имена репозиториев
        repository_mapping = {
            "market": "market_repository",
            "ml": "ml_repository",
            "portfolio": "portfolio_repository",
            "risk": "risk_repository",
            "strategy": "strategy_repository",
            "trading": "trading_repository",
        }
        repository_name = repository_mapping.get(
            repository_type.__name__.lower().replace("repository", "")
        )
        if repository_name:
            return self.container.providers[repository_name]()
        raise ValueError(f"Unknown repository type: {repository_type}")

    def get_use_case(self, use_case_type: Type) -> Any:
        """Получение use case по типу."""
        # Маппинг типов на имена use cases
        use_case_mapping = {
            "order_management": "manage_orders_use_case",
            "position_management": "manage_positions_use_case",
            "risk_management": "manage_risk_use_case",
            "trading_pair_management": "manage_trading_pairs_use_case",
            "trading_orchestrator": "trading_orchestrator_use_case",
        }
        use_case_name = use_case_mapping.get(
            use_case_type.__name__.lower().replace("usecase", "")
        )
        if use_case_name:
            return self.container.providers[use_case_name]()
        raise ValueError(f"Unknown use case type: {use_case_type}")

    def get_agent(self) -> MarketMakerModelAgent:
        """Получение агента."""
        return self.container.providers["market_maker_agent"]()

    def get_external_service(self, service_type: Type) -> Any:
        """Получение внешнего сервиса по типу."""
        # Маппинг типов на имена внешних сервисов
        external_service_mapping = {
            BybitClient: "bybit_client",
            AccountManager: "account_manager",
            RiskAnalysisServiceAdapter: "risk_analysis_service_external",
            TechnicalAnalysisServiceAdapter: "technical_analysis_service_external",
            MarketConditionsAnalyzer: "market_conditions_analyzer",
        }
        service_name = external_service_mapping.get(service_type)
        if service_name:
            return self.container.providers[service_name]()
        raise ValueError(f"Unknown external service type: {service_type}")


def configure_container(config: Dict[str, Any]) -> None:
    """Конфигурация контейнера."""
    container = Container()
    # Устанавливаем конфигурацию
    container.config.from_dict(config)
    # Инициализируем контейнер
    container.wire(
        modules=[
            "application.services",
            "application.use_cases",
            "domain.services",
            "infrastructure.repositories",
        ]
    )


def get_service_locator() -> ServiceLocator:
    """Получение глобального локатора сервисов."""
    container = Container()
    return ServiceLocator(container)
