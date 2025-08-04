"""
Основной модуль торгового оркестратора.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from application.types import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
    ProcessSignalResponse,
    TradingSession,
)
from domain.entities.strategy import Signal, SignalType, Strategy
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trade import Trade
from domain.exceptions import StrategyExecutionError, TradingOrchestrationError
from domain.intelligence.entanglement_detector import EntanglementDetector
from domain.intelligence.market_pattern_recognizer import MarketPatternRecognizer
from domain.intelligence.mirror_detector import MirrorDetector
from domain.intelligence.noise_analyzer import NoiseAnalyzer

# Временно закомментированные импорты агентов для обхода ошибок
# from infrastructure.agents.agent_whales import WhalesAgent
# from infrastructure.agents.agent_risk import RiskAgent
# from infrastructure.agents.agent_portfolio import PortfolioAgent
# from infrastructure.agents.agent_meta_controller import MetaControllerAgent as AgentMetaController
# from infrastructure.entity_system.evolution import GeneticOptimizer
# from infrastructure.agents.evolvable_news_agent import EvolvableNewsAgent
# from domain.strategies.evolvable_market_regime import EvolvableMarketRegimeAgent
# from infrastructure.agents.evolvable_strategy_agent import EvolvableStrategyAgent
# from infrastructure.agents.evolvable_risk_agent import EvolvableRiskAgent
# from domain.strategies.evolvable_portfolio_agent import EvolvablePortfolioAgent
# from domain.strategies.evolvable_order_executor import EvolvableOrderExecutor
# from domain.strategies.evolvable_meta_controller import EvolvableMetaController
# from infrastructure.agents.evolvable_decision_reasoner import EvolvableDecisionReasoner
# from infrastructure.ml_services.regime_discovery import RegimeDiscovery
# from infrastructure.agents.evolvable_market_maker import EvolvableMarketMakerAgent
# from infrastructure.agents.market_memory_integration import MarketMemoryIntegration
# from infrastructure.ml_services.sandbox_trainer import SandboxTrainer
# Импорты модуля прогнозирования разворотов
from domain.prediction.reversal_predictor import ReversalPredictor
from domain.repositories.order_repository import OrderRepository
from domain.repositories.strategy_repository import StrategyRepository
from domain.repositories.trading_repository import TradingRepository
from domain.services.pattern_discovery import PatternDiscovery
from domain.sessions.services import SessionService

# Импорты новых компонентов domain/strategies
from domain.strategies import (
    StrategyFactory,
    StrategyRegistry,
    StrategyValidator,
    get_strategy_factory,
    get_strategy_registry,
    get_strategy_validator,
)
from domain.strategies.exceptions import (
    StrategyCreationError,
    StrategyNotFoundError,
    StrategyValidationError,
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
from infrastructure.ml_services.decision_reasoner import DecisionReasoner
from infrastructure.ml_services.live_adaptation import LiveAdaptation
from infrastructure.ml_services.meta_learning import MetaLearning
# from infrastructure.ml_services.transformer_predictor import EvolutionaryTransformer
from infrastructure.services.enhanced_trading_service import EnhancedTradingService
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

from .modifiers import Modifiers
from .update_handlers import UpdateHandlers
from domain.strategies.base_strategy import BaseStrategy
from domain.repositories.position_repository import PositionRepository
from domain.repositories.portfolio_repository import PortfolioRepository
from domain.type_definitions import EntityId
from domain.type_definitions import Symbol
from domain.value_objects.money import Money, Currency
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp


class TradingOrchestratorUseCase(ABC):
    """Абстрактный use case для оркестрации торговли."""

    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""

    @abstractmethod
    async def process_signal(
        self, request: ProcessSignalRequest
    ) -> ProcessSignalResponse:
        """Обработка торгового сигнала."""

    @abstractmethod
    async def rebalance_portfolio(
        self, request: PortfolioRebalanceRequest
    ) -> PortfolioRebalanceResponse:
        """Ребалансировка портфеля."""

    @abstractmethod
    async def start_trading_session(
        self, portfolio_id: str, strategy_id: str
    ) -> TradingSession:
        """Запуск торговой сессии."""

    @abstractmethod
    async def stop_trading_session(self, session_id: str) -> bool:
        """Остановка торговой сессии."""

    @abstractmethod
    async def get_trading_session(self, session_id: str) -> Optional[TradingSession]:
        """Получение торговой сессии."""

    @abstractmethod
    async def calculate_portfolio_weights(
        self, portfolio_id: str
    ) -> Dict[str, Decimal]:
        """Расчет текущих весов портфеля."""

    @abstractmethod
    async def validate_trading_conditions(
        self, portfolio_id: str, symbol: str
    ) -> Tuple[bool, List[str]]:
        """Проверка торговых условий."""


class DefaultTradingOrchestratorUseCase(TradingOrchestratorUseCase):
    """Реализация use case для оркестрации торговли."""

    def __init__(
        self,
        order_repository: OrderRepository,
        position_repository: PositionRepository,
        portfolio_repository: PortfolioRepository,
        trading_repository: TradingRepository,
        strategy_repository: StrategyRepository,
        enhanced_trading_service: EnhancedTradingService,
        mirror_map_builder: Optional[Any] = None,
        noise_analyzer: Optional[NoiseAnalyzer] = None,
        market_pattern_recognizer: Optional[MarketPatternRecognizer] = None,
        entanglement_detector: Optional[EntanglementDetector] = None,
        mirror_detector: Optional[MirrorDetector] = None,
        session_service: Optional[SessionService] = None,
        # deprecated:
        session_influence_analyzer: Optional[Any] = None,
        session_marker: Optional[Any] = None,
        live_adaptation_model: Optional[LiveAdaptation] = None,
        decision_reasoner: Optional[DecisionReasoner] = None,
        # evolutionary_transformer: Optional[EvolutionaryTransformer] = None,
        pattern_discovery: Optional[PatternDiscovery] = None,
        meta_learning: Optional[MetaLearning] = None,
        # Временно закомментированные агенты - заменены на Any
        agent_risk: Optional[Any] = None,
        agent_portfolio: Optional[Any] = None,
        agent_meta_controller: Optional[Any] = None,
        genetic_optimizer: Optional[Any] = None,
        evolvable_news_agent: Optional[Any] = None,
        evolvable_market_regime: Optional[Any] = None,
        evolvable_strategy_agent: Optional[Any] = None,
        evolvable_risk_agent: Optional[Any] = None,
        evolvable_portfolio_agent: Optional[Any] = None,
        evolvable_order_executor: Optional[Any] = None,
        evolvable_meta_controller: Optional[Any] = None,
        evolvable_market_maker: Optional[Any] = None,
        model_selector: Optional[Any] = None,
        advanced_price_predictor: Optional[Any] = None,
        window_optimizer: Optional[Any] = None,
        state_manager: Optional[Any] = None,
        dataset_manager: Optional[Any] = None,
        evolvable_decision_reasoner: Optional[Any] = None,
        regime_discovery: Optional[Any] = None,
        advanced_market_maker: Optional[Any] = None,
        market_memory_integration: Optional[Any] = None,
        market_memory_whale_integration: Optional[Any] = None,
        local_ai_controller: Optional[Any] = None,
        analytical_integration: Optional[Any] = None,
        entanglement_integration: Optional[Any] = None,
        agent_order_executor: Optional[Any] = None,
        agent_market_regime: Optional[Any] = None,
        agent_market_maker_model: Optional[Any] = None,
        sandbox_trainer: Optional[Any] = None,
        # Добавляем эволюционный оркестратор
        evolution_orchestrator: Optional[Any] = None,
        # Добавляем модуль прогнозирования разворотов
        reversal_predictor: Optional[ReversalPredictor] = None,
        reversal_controller: Optional[Any] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        strategy_registry: Optional[StrategyRegistry] = None,
        strategy_validator: Optional[StrategyValidator] = None,
        # Добавляем модули domain/symbols
        market_phase_classifier: Optional[MarketPhaseClassifier] = None,
        opportunity_score_calculator: Optional[OpportunityScoreCalculator] = None,
        symbol_validator: Optional[SymbolValidator] = None,
        symbol_cache: Optional[SymbolCacheManager] = None,
        doass_selector: Optional[Any] = None,
    ):
        """Инициализация оркестратора."""
        self.order_repository = order_repository
        self.position_repository = position_repository
        self.portfolio_repository = portfolio_repository
        self.trading_repository = trading_repository
        self.strategy_repository = strategy_repository
        self.enhanced_trading_service = enhanced_trading_service
        self.session_service = session_service

        # Инициализация модификаторов и обработчиков
        self.modifiers = Modifiers(self)
        self.update_handlers = UpdateHandlers(self)

        # Сохраняем все опциональные зависимости
        self._dependencies = {
            "mirror_map_builder": mirror_map_builder,
            "noise_analyzer": noise_analyzer,
            "market_pattern_recognizer": market_pattern_recognizer,
            "entanglement_detector": entanglement_detector,
            "mirror_detector": mirror_detector,
            "session_influence_analyzer": session_influence_analyzer,
            "session_marker": session_marker,
            "live_adaptation_model": live_adaptation_model,
            "decision_reasoner": decision_reasoner,
            "evolutionary_transformer": evolutionary_transformer,
            "pattern_discovery": pattern_discovery,
            "meta_learning": meta_learning,
            "agent_risk": agent_risk,
            "agent_portfolio": agent_portfolio,
            "agent_meta_controller": agent_meta_controller,
            "genetic_optimizer": genetic_optimizer,
            "evolvable_news_agent": evolvable_news_agent,
            "evolvable_market_regime": evolvable_market_regime,
            "evolvable_strategy_agent": evolvable_strategy_agent,
            "evolvable_risk_agent": evolvable_risk_agent,
            "evolvable_portfolio_agent": evolvable_portfolio_agent,
            "evolvable_order_executor": evolvable_order_executor,
            "evolvable_meta_controller": evolvable_meta_controller,
            "evolvable_market_maker": evolvable_market_maker,
            "model_selector": model_selector,
            "advanced_price_predictor": advanced_price_predictor,
            "window_optimizer": window_optimizer,
            "state_manager": state_manager,
            "dataset_manager": dataset_manager,
            "evolvable_decision_reasoner": evolvable_decision_reasoner,
            "regime_discovery": regime_discovery,
            "advanced_market_maker": advanced_market_maker,
            "market_memory_integration": market_memory_integration,
            "market_memory_whale_integration": market_memory_whale_integration,
            "local_ai_controller": local_ai_controller,
            "analytical_integration": analytical_integration,
            "entanglement_integration": entanglement_integration,
            "agent_order_executor": agent_order_executor,
            "agent_market_regime": agent_market_regime,
            "agent_market_maker_model": agent_market_maker_model,
            "sandbox_trainer": sandbox_trainer,
            "evolution_orchestrator": evolution_orchestrator,
            "reversal_predictor": reversal_predictor,
            "reversal_controller": reversal_controller,
            "strategy_factory": strategy_factory,
            "strategy_registry": strategy_registry,
            "strategy_validator": strategy_validator,
            "market_phase_classifier": market_phase_classifier,
            "opportunity_score_calculator": opportunity_score_calculator,
            "symbol_validator": symbol_validator,
            "symbol_cache": symbol_cache,
            "doass_selector": doass_selector,
        }

    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        try:
            start_time = datetime.now()
            
            # Получаем стратегию
            strategy = await self._get_strategy(str(request.strategy_id))
            if not strategy:
                return ExecuteStrategyResponse(
                    success=False,
                    errors=["Strategy not found"],
                    execution_time_ms=0,
                )

            # Создаем MarketData из Symbol для generate_signal
            from domain.entities.market_data import MarketData
            market_data = MarketData(
                symbol=request.symbol,
                timestamp=Timestamp.now(),
                open_price=Price(Decimal('0'), Currency.USD),
                high_price=Price(Decimal('0'), Currency.USD),
                low_price=Price(Decimal('0'), Currency.USD),
                close_price=Price(Decimal('0'), Currency.USD),
                volume=Volume(Decimal('0')),
            )

            # Генерируем сигналы - убираем await
            signals = strategy.generate_signal(market_data)
            
            # Создаем ордера на основе сигналов
            orders = []
            signal_list: List[Signal] = []
            if signals:
                if isinstance(signals, list):
                    signal_list = signals
                else:
                    signal_list = [signals]
                
                for signal in signal_list:
                    order = await self._create_order_from_signal(signal, str(request.portfolio_id))
                    if order:
                        orders.append(order)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ExecuteStrategyResponse(
                success=True,
                orders_created=orders,
                signals_generated=signal_list if signals else [],
                execution_time_ms=int(execution_time),
                executed=True,
            )

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return ExecuteStrategyResponse(
                success=False,
                errors=[str(e)],
                execution_time_ms=0,
            )

    async def _get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Получение стратегии."""
        try:
            # Пытаемся получить стратегию из репозитория
            strategy = await self.strategy_repository.get_by_id(EntityId(strategy_id))
            if strategy:
                return strategy if isinstance(strategy, BaseStrategy) else None

            # Если не найдена, пытаемся создать из infrastructure
            return await self._create_strategy_from_infrastructure(strategy_id)
        except Exception as e:
            logger.error(f"Error getting strategy {strategy_id}: {e}")
            return None

    async def _create_strategy_from_infrastructure(
        self, strategy_id: str
    ) -> Optional[BaseStrategy]:
        """Создание стратегии из infrastructure."""
        try:
            # Создаем стратегию на основе ID
            if "adaptive" in strategy_id:
                from infrastructure.strategies.adaptive_strategy_generator import AdaptiveStrategyGenerator
                return AdaptiveStrategyGenerator(
                    market_regime_agent=None,
                    meta_learner=None,
                    backtest_results={},
                    base_strategies=[],
                    config={}
                )
            elif "manipulation" in strategy_id:
                from infrastructure.strategies.manipulation_strategy import ManipulationStrategy
                return ManipulationStrategy()
            elif "sideways" in strategy_id:
                from infrastructure.strategies.sideways_strategy import SidewaysStrategy
                return SidewaysStrategy()
            elif "trend" in strategy_id:
                from infrastructure.strategies.trend_strategies import TrendStrategy
                return TrendStrategy()
            elif "volatility" in strategy_id:
                from infrastructure.strategies.volatility_strategy import VolatilityStrategy
                return VolatilityStrategy()
            else:
                logger.warning(f"Unknown strategy type: {strategy_id}")
                return None
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_id}: {e}")
            return None

    async def process_signal(
        self, request: ProcessSignalRequest
    ) -> ProcessSignalResponse:
        """Обработка торгового сигнала."""
        try:
            start_time = datetime.now()
            
            # Создаем ордер на основе сигнала
            order = await self._create_order_from_signal(request.signal, str(request.portfolio_id))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if order:
                return ProcessSignalResponse(
                    success=True,
                    orders_created=[order],
                    processed=True,
                )
            else:
                return ProcessSignalResponse(
                    success=False,
                    errors=["Failed to create order from signal"],
                    processed=False,
                )

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return ProcessSignalResponse(
                success=False,
                errors=[str(e)],
                processed=False,
            )

    async def rebalance_portfolio(
        self, request: PortfolioRebalanceRequest
    ) -> PortfolioRebalanceResponse:
        """Ребалансировка портфеля."""
        try:
            start_time = datetime.now()
            
            # Получаем текущие веса портфеля
            current_weights = await self.calculate_portfolio_weights(str(request.portfolio_id))
            
            # Рассчитываем необходимые изменения
            target_weights = request.target_weights
            weight_diffs = {}
            
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, Decimal('0'))
                weight_diff = target_weight - current_weight
                if abs(weight_diff) > request.tolerance:
                    weight_diffs[symbol] = weight_diff

            # Создаем ордера для ребалансировки
            orders = []
            for symbol, weight_diff in weight_diffs.items():
                order = await self._create_rebalance_order(symbol, weight_diff, str(request.portfolio_id))
                if order:
                    orders.append(order)

            rebalance_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Исправляем типы для PortfolioRebalanceResponse
            current_weights_symbol = {Symbol(k): v for k, v in current_weights.items()}
            target_weights_symbol = {Symbol(k): v for k, v in target_weights.items()}
            
            return PortfolioRebalanceResponse(
                success=True,
                orders_created=orders,
                current_weights=current_weights_symbol,
                target_weights=target_weights_symbol,
                rebalance_cost=Money(Decimal("0"), Currency.USDT),
                rebalanced=True,
            )

        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return PortfolioRebalanceResponse(
                success=False,
                errors=[str(e)],
                rebalance_cost=Money(Decimal("0"), Currency.USDT),
            )

    async def start_trading_session(
        self, portfolio_id: str, strategy_id: str
    ) -> TradingSession:
        """Запуск торговой сессии."""
        try:
            if not self.session_service:
                raise TradingOrchestrationError("Session service not available")

            # Создаем сессию - исправляем метод
            from domain.type_definitions import SessionType  # type: ignore[attr-defined]
            from domain.value_objects.timestamp import Timestamp
            session = await self.session_service.get_session_phase(  # type: ignore[misc]
                SessionType.TRADING, 
                Timestamp(datetime.now())
            )
            return session

        except Exception as e:
            logger.error(f"Error starting trading session: {e}")
            raise TradingOrchestrationError(f"Failed to start session: {e}")

    async def stop_trading_session(self, session_id: str) -> bool:
        """Остановка торговой сессии."""
        try:
            if not self.session_service:
                return False

            # Останавливаем сессию - исправляем метод
            from domain.type_definitions import SessionType  # type: ignore[attr-defined]
            from domain.value_objects.timestamp import Timestamp
            return await self.session_service.get_session_phase(  # type: ignore[misc]
                SessionType.TRADING, 
                Timestamp(datetime.now())
            )

        except Exception as e:
            logger.error(f"Error stopping trading session: {e}")
            return False

    async def get_trading_session(self, session_id: str) -> Optional[TradingSession]:
        """Получение торговой сессии."""
        try:
            if not self.session_service:
                return None

            # Получаем сессию - исправляем метод
            return await self.session_service.get_session_phase(session_id)  # type: ignore[misc, arg-type]

        except Exception as e:
            logger.error(f"Error getting trading session: {e}")
            return None

    async def calculate_portfolio_weights(
        self, portfolio_id: str
    ) -> Dict[str, Decimal]:
        """Расчет текущих весов портфеля."""
        try:
            # Получаем позиции портфеля - исправляем метод
            positions = await self.position_repository.get_open_positions()
            
            # Фильтруем позиции по портфелю (временное решение)
            portfolio_positions = [pos for pos in positions if hasattr(pos, 'portfolio_id') and str(pos.portfolio_id) == portfolio_id]
            
            # Рассчитываем общую стоимость портфеля
            total_value = Decimal('0')
            for position in portfolio_positions:
                if hasattr(position, 'unrealized_pnl'):
                    total_value += position.unrealized_pnl.amount

            # Рассчитываем веса
            weights = {}
            for position in portfolio_positions:
                if total_value > 0 and hasattr(position, 'unrealized_pnl') and hasattr(position, 'symbol'):
                    weight = position.unrealized_pnl.amount / total_value
                    weights[str(position.symbol)] = weight
                else:
                    weights[str(position.symbol)] = Decimal('0')

            return weights

        except Exception as e:
            logger.error(f"Error calculating portfolio weights: {e}")
            return {}

    async def validate_trading_conditions(
        self, portfolio_id: str, symbol: str
    ) -> Tuple[bool, List[str]]:
        """Проверка торговых условий."""
        try:
            errors = []
            
            # Проверяем существование портфеля
            portfolio = await self.portfolio_repository.get_by_id(EntityId(portfolio_id))
            if not portfolio:
                errors.append("Portfolio not found")

            # Проверяем валидность символа
            try:
                Symbol(symbol)
            except ValueError:
                errors.append("Invalid symbol")

            # Проверяем торговые условия
            # Здесь можно добавить дополнительные проверки

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating trading conditions: {e}")
            return False, [str(e)]

    async def _create_order_from_signal(
        self, signal: Signal, portfolio_id: str
    ) -> Optional[Order]:
        """Создание ордера из сигнала."""
        try:
            # Исправляем атрибуты Signal и типы Order
            from domain.entities.order import Order, OrderType, OrderStatus, OrderSide
            from domain.value_objects.signal_type import SignalType
            from domain.type_definitions import VolumeValue
            
            # Преобразуем SignalType в OrderSide
            side = OrderSide.BUY if signal.signal_type.value == "buy" else OrderSide.SELL
            
            # Создаем ордер на основе сигнала
            order = Order(
                id=EntityId.generate(),
                portfolio_id=EntityId(portfolio_id),
                symbol=Symbol("BTC/USD"),  # Используем дефолтный символ, так как Signal не имеет symbol
                side=side,
                order_type=OrderType.MARKET,
                quantity=VolumeValue(Decimal(str(signal.confidence))),  # Используем confidence как количество
                price=Price(Decimal(str(signal.price.value)) if signal.price else Decimal("0"), Currency.USD),
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),
            )
            
            # Сохраняем ордер
            await self.order_repository.save(order)
            return order

        except Exception as e:
            logger.error(f"Error creating order from signal: {e}")
            return None

    async def _create_rebalance_order(
        self, symbol: str, weight_diff: Decimal, portfolio_id: str
    ) -> Optional[Order]:
        """Создание ордера для ребалансировки."""
        try:
            from domain.entities.order import Order, OrderType, OrderStatus, OrderSide
            from domain.type_definitions import VolumeValue
            
            # Определяем сторону ордера
            side = OrderSide.BUY if weight_diff > 0 else OrderSide.SELL
            
            # Создаем ордер
            order = Order(
                id=EntityId.generate(),
                portfolio_id=EntityId(portfolio_id),
                symbol=Symbol(symbol),
                side=side,
                order_type=OrderType.MARKET,
                quantity=VolumeValue(abs(weight_diff)),
                price=Price(Decimal('0'), Currency.USD),
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),
            )
            
            # Сохраняем ордер
            await self.order_repository.save(order)
            return order

        except Exception as e:
            logger.error(f"Error creating rebalance order: {e}")
            return None

    async def _calculate_portfolio_risk(self, portfolio_id: str) -> Dict[str, float]:
        """Расчет риска портфеля."""
        try:
            # Получаем позиции портфеля - исправляем метод
            positions = await self.position_repository.get_open_positions()
            
            # Фильтруем позиции по портфелю (временное решение)
            portfolio_positions = [pos for pos in positions if hasattr(pos, 'portfolio_id') and str(pos.portfolio_id) == portfolio_id]
            
            # Рассчитываем базовые метрики риска
            total_value = sum(pos.unrealized_pnl.amount for pos in portfolio_positions if hasattr(pos, 'unrealized_pnl'))
            total_pnl = sum(pos.unrealized_pnl.amount for pos in portfolio_positions if hasattr(pos, 'unrealized_pnl'))
            
            return {
                "total_value": float(total_value),
                "total_pnl": float(total_pnl),
                "pnl_percent": float(total_pnl / total_value * 100) if total_value > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {}
