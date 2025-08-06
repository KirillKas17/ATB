"""
Рефакторенный DI контейнер с улучшенной архитектурой согласно SOLID принципам.
Устранены проблемы дублирования, типизации и сложности.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, Union

from dependency_injector import containers, providers

# Application Services
from application.services.market_service import MarketService
from application.services.ml_service import MLService
from application.services.order_validator import OrderValidator
from application.services.portfolio_service import PortfolioService
from application.services.risk_service import RiskService
from application.services.service_factory import get_service_factory
from application.services.trading_service import TradingService

# Use Cases
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
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
)

# Domain Services
from domain.evolution import (
    EvolutionContext,
    StrategyCandidate,
    StrategyFitnessEvaluator,
    StrategyGenerator,
    StrategySelector,
)
from domain.evolution.strategy_optimizer import StrategyOptimizer
from domain.services.spread_analysis import SpreadAnalyzer
from domain.services.liquidity_analysis import LiquidityAnalyzer
from domain.services.ml_predictor import MLPredictor
from domain.services.correlation_chain import DefaultCorrelationChain
from domain.services.market_metrics import MarketMetricsService
from domain.services.pattern_discovery import PatternConfig, PatternDiscovery
from domain.services.risk_analysis import (
    DefaultRiskAnalysisService,
    RiskAnalysisService,
)
from domain.services.signal_service import DefaultSignalService, SignalService
from domain.services.technical_analysis import (
    DefaultTechnicalAnalysisService,
    ITechnicalAnalysisService,
)
from domain.services.order_validation_service import DefaultOrderValidationService

# Domain Strategies
from domain.strategies import (
    StrategyFactory,
    StrategyRegistry,
    StrategyValidator,
    get_strategy_factory,
    get_strategy_registry,
    get_strategy_validator,
)

# Domain Symbols
from domain.symbols.market_phase_classifier import MarketPhaseClassifier
from domain.symbols.opportunity_score import OpportunityScoreCalculator
from domain.symbols.validators import SymbolValidator
from domain.symbols.cache import SymbolCacheManager

# Infrastructure
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
from infrastructure.evolution import (
    EvolutionBackup,
    EvolutionCache,
    EvolutionMigration,
    StrategyStorage,
)
from infrastructure.external_services.bybit_client import BybitClient
from infrastructure.external_services.account_manager import AccountManager
from infrastructure.external_services.risk_analysis_adapter import RiskAnalysisServiceAdapter
from infrastructure.external_services.technical_analysis_adapter import TechnicalAnalysisServiceAdapter
from infrastructure.services.market_conditions_analyzer import MarketConditionsAnalyzer
from infrastructure.repositories.market_repository import InMemoryMarketRepository
from infrastructure.repositories.ml_repository import InMemoryMLRepository
from infrastructure.repositories.portfolio_repository import InMemoryPortfolioRepository
from infrastructure.repositories.risk_repository import InMemoryRiskRepository
from infrastructure.repositories.strategy_repository import InMemoryStrategyRepository
from infrastructure.repositories.trading.trading_repository import (
    InMemoryTradingRepository,
)

# Strategies
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy
from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
from infrastructure.strategies.evolution.evolvable_base_strategy import EvolvableBaseStrategy
from infrastructure.strategies.manipulation_strategies import ManipulationStrategy
from infrastructure.strategies.volatility_strategy import VolatilityStrategy
from infrastructure.strategies.pairs_trading_strategy import PairsTradingStrategy

# Shared
from shared.abstractions.base_service import BaseService


@dataclass
class ContainerConfig:
    """Конфигурация контейнера с улучшенной типизацией."""
    
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


class ServiceRegistry:
    """Упрощенный реестр сервисов без дублирования."""
    
    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
    
    def register(self, name: str, service: Any) -> None:
        """Регистрация сервиса."""
        self._services[name] = service
    
    def get(self, name: str) -> Optional[Any]:
        """Получение сервиса."""
        return self._services.get(name)
    
    def has(self, name: str) -> bool:
        """Проверка наличия сервиса."""
        return name in self._services


class Container(containers.DeclarativeContainer):
    """Улучшенный контейнер зависимостей с упрощенной архитектурой."""
    
    # Конфигурация
    config = providers.Configuration()
    
    # Repositories
    market_repository = providers.Singleton(InMemoryMarketRepository)
    ml_repository = providers.Singleton(InMemoryMLRepository)
    portfolio_repository = providers.Singleton(InMemoryPortfolioRepository)
    risk_repository = providers.Singleton(InMemoryRiskRepository)
    strategy_repository = providers.Singleton(InMemoryStrategyRepository)
    trading_repository = providers.Singleton(InMemoryTradingRepository)
    
    # Domain Services
    spread_analyzer = providers.Singleton(SpreadAnalyzer)
    liquidity_analyzer = providers.Singleton(LiquidityAnalyzer)
    ml_predictor = providers.Singleton(MLPredictor)
    technical_analysis_service = providers.Singleton(DefaultTechnicalAnalysisService)
    market_metrics_service = providers.Singleton(MarketMetricsService)
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
    risk_analysis_service = providers.Singleton(DefaultRiskAnalysisService)
    order_validation_service = providers.Singleton(DefaultOrderValidationService)
    
    # Pattern Discovery с упрощенной конфигурацией
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
    
    # Market Conditions Analyzer
    market_conditions_analyzer = providers.Singleton(
        MarketConditionsAnalyzer,
        market_repository=market_repository,
        technical_analysis_service=technical_analysis_service
    )
    
    # Domain Strategies
    strategy_factory = providers.Singleton(get_strategy_factory)
    strategy_registry = providers.Singleton(get_strategy_registry)
    strategy_validator = providers.Singleton(get_strategy_validator)
    
    # Infrastructure Strategies
    trend_strategy = providers.Singleton(TrendStrategy)
    sideways_strategy = providers.Singleton(SidewaysStrategy)
    adaptive_strategy_generator = providers.Singleton(AdaptiveStrategyGenerator)
    evolvable_base_strategy = providers.Singleton(EvolvableBaseStrategy)
    manipulation_strategy = providers.Singleton(ManipulationStrategy)
    volatility_strategy = providers.Singleton(VolatilityStrategy)
    pairs_trading_strategy = providers.Singleton(PairsTradingStrategy)
    
    # Domain Symbols
    market_phase_classifier = providers.Singleton(MarketPhaseClassifier)
    opportunity_score_calculator = providers.Singleton(OpportunityScoreCalculator)
    symbol_validator = providers.Singleton(SymbolValidator)
    symbol_cache = providers.Singleton(SymbolCacheManager)
    
    # Service Factory
    service_factory = providers.Singleton(get_service_factory, config=config)
    
    # Application Services через фабрику
    market_service = providers.Singleton(
        lambda factory: factory.create_market_service(), 
        service_factory=service_factory
    )
    ml_service = providers.Singleton(
        lambda factory: factory.create_ml_service(), 
        service_factory=service_factory
    )
    portfolio_service = providers.Singleton(
        lambda factory: factory.create_portfolio_service(),
        service_factory=service_factory,
    )
    risk_service = providers.Singleton(
        lambda factory: factory.create_risk_service(), 
        service_factory=service_factory
    )
    trading_service = providers.Singleton(
        lambda factory: factory.create_trading_service(),
        service_factory=service_factory,
    )
    
    # Order Validator
    order_validator = providers.Singleton(
        OrderValidator, 
        market_service=market_service, 
        risk_service=risk_service,
        order_validation_service=order_validation_service
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
        strategy_service=strategy_factory,  # Используем factory вместо service
    )
    trading_orchestrator_use_case = providers.Singleton(
        DefaultTradingOrchestratorUseCase,
        order_repository=trading_repository,
        position_repository=portfolio_repository,
        portfolio_repository=portfolio_repository,
        trading_repository=trading_repository,
        strategy_repository=strategy_repository,
        enhanced_trading_service=trading_service,
    )
    
    # Infrastructure
    market_maker_agent = providers.Singleton(MarketMakerModelAgent)
    
    # External Services
    bybit_client = providers.Singleton(
        BybitClient,
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=config.bybit.testnet,
    )
    account_manager = providers.Singleton(AccountManager, bybit_client=bybit_client)
    
    # External Analysis Services
    risk_analysis_service_external = providers.Singleton(RiskAnalysisServiceAdapter)
    technical_analysis_service_external = providers.Singleton(TechnicalAnalysisServiceAdapter)


class ServiceLocator:
    """Улучшенный локатор сервисов с упрощенной архитектурой."""
    
    def __init__(self, container: Container) -> None:
        """Инициализация локатора сервисов."""
        self.container = container
        self._registry = ServiceRegistry()
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Инициализация реестра сервисов."""
        # Регистрируем основные сервисы
        service_mapping = {
            'market_service': self.container.market_service,
            'ml_service': self.container.ml_service,
            'portfolio_service': self.container.portfolio_service,
            'risk_service': self.container.risk_service,
            'trading_service': self.container.trading_service,
            'order_validator': self.container.order_validator,
        }
        
        for name, provider in service_mapping.items():
            self._registry.register(name, provider)
    
    def get_service(self, service_type: Type[BaseService]) -> BaseService:
        """Получение сервиса по типу."""
        service_mapping = {
            MarketService: 'market_service',
            MLService: 'ml_service',
            PortfolioService: 'portfolio_service',
            RiskService: 'risk_service',
            TradingService: 'trading_service',
        }
        
        service_name = service_mapping.get(service_type)
        if service_name and self._registry.has(service_name):
            return self._registry.get(service_name)()  # type: ignore[no-any-return]
        
        raise ValueError(f"Unknown service type: {service_type}")
    
    def get_repository(self, repository_type: str) -> Any:
        """Получение репозитория по типу."""
        repository_mapping = {
            'market': self.container.market_repository,
            'ml': self.container.ml_repository,
            'portfolio': self.container.portfolio_repository,
            'risk': self.container.risk_repository,
            'strategy': self.container.strategy_repository,
            'trading': self.container.trading_repository,
        }
        
        provider = repository_mapping.get(repository_type.lower())
        if provider:
            return provider()
        
        raise ValueError(f"Unknown repository type: {repository_type}")
    
    def get_use_case(self, use_case_type: str) -> Any:
        """Получение use case по типу."""
        use_case_mapping = {
            'order_management': self.container.manage_orders_use_case,
            'position_management': self.container.manage_positions_use_case,
            'risk_management': self.container.manage_risk_use_case,
            'trading_pair_management': self.container.manage_trading_pairs_use_case,
            'trading_orchestrator': self.container.trading_orchestrator_use_case,
        }
        
        provider = use_case_mapping.get(use_case_type.lower())
        if provider:
            return provider()
        
        raise ValueError(f"Unknown use case type: {use_case_type}")
    
    def get_agent(self) -> MarketMakerModelAgent:
        """Получение агента."""
        return self.container.market_maker_agent()
    
    def get_external_service(self, service_type: str) -> Any:
        """Получение внешнего сервиса по типу."""
        external_service_mapping = {
            'bybit_client': self.container.bybit_client,
            'account_manager': self.container.account_manager,
            'risk_analysis_service_external': self.container.risk_analysis_service_external,
            'technical_analysis_service_external': self.container.technical_analysis_service_external,
            'market_conditions_analyzer': self.container.market_conditions_analyzer,
        }
        
        provider = external_service_mapping.get(service_type.lower())
        if provider:
            return provider()
        
        raise ValueError(f"Unknown external service type: {service_type}")


def configure_container(config: Dict[str, Any]) -> None:
    """Конфигурация контейнера."""
    container = Container()
    container.config.from_dict(config)
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
