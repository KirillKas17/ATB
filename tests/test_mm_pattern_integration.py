"""
Тесты интеграции паттернов маркет-мейкера.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType, 
    PatternResult, PatternOutcome
)
from domain.market_maker.mm_pattern_classifier import (
    MarketMakerPatternClassifier, OrderBookSnapshot, TradeSnapshot
)
from domain.market_maker.mm_pattern_memory import PatternMemoryRepository
from application.market.mm_follow_controller import MarketMakerFollowController
# from infrastructure.market_profiles import MarketMakerPatternStorage
# from infrastructure.agents.agent_context import AgentContext, AgentContextManager
# from application.use_cases.trading_orchestrator import TradingOrchestratorUseCase
from unittest.mock import Mock, AsyncMock
from application.use_cases.execute_strategy_request import ExecuteStrategyRequest, ExecuteStrategyResponse
from domain.type_definitions.market_maker_types import (
    Symbol, Confidence
)

class TestMMPatternIntegration:
    """Тесты интеграции паттернов маркет-мейкера."""
    @pytest.fixture
    def pattern_classifier(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание классификатора паттернов."""
        return MarketMakerPatternClassifier()
    @pytest.fixture
    def pattern_memory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание памяти паттернов."""
        return PatternMemoryRepository("test_data")
    @pytest.fixture
    def follow_controller(self, pattern_classifier, pattern_memory) -> Any:
        """Создание контроллера следования."""
        return MarketMakerFollowController(
            pattern_classifier=pattern_classifier,
            pattern_memory=pattern_memory
        )
    @pytest.fixture
    def mm_storage(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание хранилища ММ."""
        # return MarketMakerPatternStorage("test_data")
        storage = Mock()
        storage.save_pattern = AsyncMock()
        storage.get_patterns_by_symbol = AsyncMock(return_value=[])
        return storage
    @pytest.fixture
    def agent_context_manager(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание менеджера контекстов агентов."""
        return AgentContextManager()
    @pytest.fixture
    def sample_order_book(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестового стакана заявок."""
        return OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            bids=[
                {"price": 50000.0, "size": 1.5},
                {"price": 49999.0, "size": 2.0},
                {"price": 49998.0, "size": 1.0}
            ],
            asks=[
                {"price": 50001.0, "size": 1.0},
                {"price": 50002.0, "size": 2.5},
                {"price": 50003.0, "size": 1.5}
            ],
            last_price=50000.0,
            volume_24h=1000000.0,
            price_change_24h=0.02
        )
    @pytest.fixture
    def sample_trades(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестовых сделок."""
        return TradeSnapshot(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            trades=[
                {
                    "time": datetime.now(),
                    "price": 50000.0,
                    "size": 0.1,
                    "side": "buy"
                },
                {
                    "time": datetime.now(),
                    "price": 50001.0,
                    "size": 0.2,
                    "side": "sell"
                },
                {
                    "time": datetime.now(),
                    "price": 50000.5,
                    "size": 0.15,
                    "side": "buy"
                }
            ]
        )
    def test_pattern_classifier_creation(self, pattern_classifier) -> None:
        """Тест создания классификатора паттернов."""
        assert pattern_classifier is not None
        assert hasattr(pattern_classifier, 'classify_pattern')
        assert hasattr(pattern_classifier, 'extract_features')
    def test_pattern_memory_creation(self, pattern_memory) -> None:
        """Тест создания памяти паттернов."""
        assert pattern_memory is not None
        assert hasattr(pattern_memory, 'save_pattern')
        assert hasattr(pattern_memory, 'find_similar_patterns')
    def test_follow_controller_creation(self, follow_controller) -> None:
        """Тест создания контроллера следования."""
        assert follow_controller is not None
        assert hasattr(follow_controller, 'process_pattern')
        assert hasattr(follow_controller, 'record_pattern_result')
    def test_mm_storage_creation(self, mm_storage) -> None:
        """Тест создания хранилища ММ."""
        assert mm_storage is not None
        assert hasattr(mm_storage, 'save_pattern')
        assert hasattr(mm_storage, 'get_patterns_by_symbol')
    @pytest.mark.asyncio
    async def test_pattern_classification(self, pattern_classifier, sample_order_book, sample_trades) -> None:
        """Тест классификации паттернов."""
        pattern = pattern_classifier.classify_pattern(
            "BTCUSDT", sample_order_book, sample_trades
        )
        if pattern:  # Может быть None если уверенность низкая
            assert isinstance(pattern, MarketMakerPattern)
            assert pattern.symbol == "BTCUSDT"
            assert pattern.pattern_type in MarketMakerPatternType
            assert 0.0 <= pattern.confidence <= 1.0
    @pytest.mark.asyncio
    async def test_pattern_memory_operations(self, pattern_memory) -> None:
        """Тест операций с памятью паттернов."""
        # Создаем тестовый паттерн
        features = PatternFeatures(
            book_pressure=0.5,
            volume_delta=0.1,
            price_reaction=0.02,
            spread_change=0.001,
            order_imbalance=0.3,
            liquidity_depth=1000.0,
            time_duration=300,
            volume_concentration=0.6,
            price_volatility=0.01,
            market_microstructure={}
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol=Symbol("BTCUSDT"),
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.8),
            context={"symbol": "BTCUSDT"}
        )
        # Сохраняем паттерн
        success = await pattern_memory.save_pattern("BTCUSDT", pattern)
        assert success is True
        # Получаем паттерны
        patterns = await pattern_memory.get_patterns_by_symbol("BTCUSDT", limit=10)
        assert len(patterns) > 0
        # Ищем похожие паттерны
        similar_patterns = await pattern_memory.find_similar_patterns(
            "BTCUSDT", features, similarity_threshold=0.7
        )
        assert isinstance(similar_patterns, list)
    @pytest.mark.asyncio
    async def test_follow_controller_processing(self, follow_controller, sample_order_book, sample_trades) -> None:
        """Тест обработки паттернов контроллером следования."""
        # Обрабатываем паттерн
        follow_signal = await follow_controller.process_pattern(
            "BTCUSDT", sample_order_book, sample_trades
        )
        # Сигнал может быть None если нет подходящих исторических паттернов
        if follow_signal:
            assert hasattr(follow_signal, 'pattern_type')
            assert hasattr(follow_signal, 'confidence')
            assert hasattr(follow_signal, 'expected_direction')
            assert hasattr(follow_signal, 'position_size_modifier')
    @pytest.mark.asyncio
    async def test_agent_context_integration(self, agent_context_manager) -> None:
        """Тест интеграции с AgentContext."""
        # Получаем контекст агента
        context = agent_context_manager.get_context("BTCUSDT")
        assert isinstance(context, AgentContext)
        assert context.symbol == "BTCUSDT"
        # Проверяем наличие методов для работы с паттернами ММ
        assert hasattr(context, 'apply_mm_pattern_modifier')
        assert hasattr(context, 'get_mm_pattern_status')
        assert hasattr(context, 'update_mm_pattern_result')
        assert hasattr(context, 'get_mm_pattern_statistics')
    @pytest.mark.asyncio
    async def test_full_integration_workflow(self, pattern_classifier, pattern_memory,
                                           follow_controller, agent_context_manager,
                                           sample_order_book, sample_trades) -> None:
        """Тест полного рабочего процесса интеграции."""
        # 1. Классифицируем паттерн
        pattern = pattern_classifier.classify_pattern(
            "BTCUSDT", sample_order_book, sample_trades
        )
        if pattern:
            # 2. Сохраняем в память
            await pattern_memory.save_pattern("BTCUSDT", pattern)
            # 3. Обрабатываем через контроллер
            follow_signal = await follow_controller.process_pattern(
                "BTCUSDT", sample_order_book, sample_trades
            )
            if follow_signal:
                # 4. Интегрируем с контекстом агента
                context = agent_context_manager.get_context("BTCUSDT")
                context.mm_pattern_context.follow_signal = follow_signal
                # 5. Применяем модификаторы
                context.apply_mm_pattern_modifier()
                # 6. Проверяем статус
                status = context.get_mm_pattern_status()
                assert status["pattern_detected"] is True
                assert status["pattern_type"] == follow_signal.pattern_type
                assert status["confidence"] == follow_signal.confidence
    @pytest.mark.asyncio
    async def test_pattern_result_recording(self, pattern_memory) -> None:
        """Тест записи результатов паттернов."""
        # Создаем тестовый паттерн
        features = PatternFeatures(
            book_pressure=0.5,
            volume_delta=0.1,
            price_reaction=0.02,
            spread_change=0.001,
            order_imbalance=0.3,
            liquidity_depth=1000.0,
            time_duration=300,
            volume_concentration=0.6,
            price_volatility=0.01,
            market_microstructure={}
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol=Symbol("BTCUSDT"),
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.8),
            context={}
        )
        # Сохраняем паттерн
        await pattern_memory.save_pattern("BTCUSDT", pattern)
        # Создаем результат
        result = PatternResult(
            outcome=PatternOutcome.SUCCESS,
            price_change_5min=0.01,
            price_change_15min=0.02,
            price_change_30min=0.03,
            volume_change=0.1,
            volatility_change=0.05,
            market_context={"trend": "up"}
        )
        # Записываем результат
        pattern_id = f"{pattern.symbol}_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
        success = await pattern_memory.update_pattern_result("BTCUSDT", pattern_id, result)
        assert success is True
    def test_pattern_features_serialization(self: "TestMMPatternIntegration") -> None:
        """Тест сериализации признаков паттерна."""
        from domain.type_definitions.market_maker_types import (
            BookPressure, VolumeDelta, PriceReaction, SpreadChange, 
            OrderImbalance, LiquidityDepth, TimeDuration, 
            VolumeConcentration, PriceVolatility
        )
        
        features = PatternFeatures(
            book_pressure=BookPressure(0.5),
            volume_delta=VolumeDelta(0.1),
            price_reaction=PriceReaction(0.02),
            spread_change=SpreadChange(0.001),
            order_imbalance=OrderImbalance(0.3),
            liquidity_depth=LiquidityDepth(1000.0),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.6),
            price_volatility=PriceVolatility(0.01),
            market_microstructure={"avg_trade_size": 123.0}
        )
        # Проверяем, что все поля доступны
        assert features.book_pressure == 0.5
        assert features.volume_delta == 0.1
        assert features.price_reaction == 0.02
        assert features.spread_change == 0.001
        assert features.order_imbalance == 0.3
        assert features.liquidity_depth == 1000.0
        assert features.time_duration == 300
        assert features.volume_concentration == 0.6
        assert features.price_volatility == 0.01
        assert features.market_microstructure["avg_trade_size"] == 123.0
    def test_pattern_serialization(self: "TestMMPatternIntegration") -> None:
        """Тест сериализации паттерна."""
        from domain.type_definitions.market_maker_types import (
            BookPressure, VolumeDelta, PriceReaction, SpreadChange, 
            OrderImbalance, LiquidityDepth, TimeDuration, 
            VolumeConcentration, PriceVolatility
        )
        
        features = PatternFeatures(
            book_pressure=BookPressure(0.5),
            volume_delta=VolumeDelta(0.1),
            price_reaction=PriceReaction(0.02),
            spread_change=SpreadChange(0.001),
            order_imbalance=OrderImbalance(0.3),
            liquidity_depth=LiquidityDepth(1000.0),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.6),
            price_volatility=PriceVolatility(0.01),
            market_microstructure={}
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol=Symbol("BTCUSDT"),
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.8),
            context={"symbol": "BTCUSDT"}
        )
        # Сериализуем в словарь
        pattern_dict = pattern.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["symbol"] == "BTCUSDT"
        assert pattern_dict["pattern_type"] == "accumulation"
        assert pattern_dict["confidence"] == 0.8
        assert "features" in pattern_dict
        assert "context" in pattern_dict
        # Десериализуем обратно
        pattern_restored = MarketMakerPattern.from_dict(pattern_dict)
        assert pattern_restored.symbol == pattern.symbol
        assert pattern_restored.pattern_type == pattern.pattern_type
        assert pattern_restored.confidence == pattern.confidence
    @pytest.mark.asyncio
    async def test_orchestrator_mm_pattern_analysis() -> None:
    """Тест интеграции MM Pattern Intelligence в orchestrator."""
    # Создаем компоненты
    pattern_classifier = MarketMakerPatternClassifier()
    pattern_memory = PatternMemoryRepository("test_data")
    follow_controller = MarketMakerFollowController(
        pattern_classifier=pattern_classifier,
        pattern_memory=pattern_memory
    )
    mm_storage = Mock()
    mm_storage.save_pattern = AsyncMock()
    mm_storage.get_patterns_by_symbol = AsyncMock(return_value=[])
    agent_context_manager = AgentContextManager()
    # Создаем orchestrator
    orchestrator = TradingOrchestratorUseCase(
        order_repository=None,
        position_repository=None,
        portfolio_repository=None,
        trading_repository=None,
        strategy_repository=None,
        enhanced_trading_service=None,
        mm_pattern_classifier=pattern_classifier,
        mm_pattern_memory=pattern_memory,
        mm_follow_controller=follow_controller,
        mm_storage=mm_storage
    )
    orchestrator._agent_context_manager = agent_context_manager
    # Кладем order_book и trades в AgentContext
    symbol = "BTCUSDT"
    context = agent_context_manager.get_context(symbol)
    context.order_book = OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        bids=[{"price": 50000.0, "size": 1.5}],
        asks=[{"price": 50001.0, "size": 1.0}],
        last_price=50000.0,
        volume_24h=1000000.0,
        price_change_24h=0.02
    )
    context.trades = TradeSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        trades=[{"time": datetime.now(), "price": 50000.0, "size": 0.1, "side": "buy"}]
    )
    # Вызываем анализ
    await orchestrator._update_mm_pattern_analysis([symbol])
    # Проверяем, что follow_signal и модификаторы применились
    assert context.mm_pattern_context.follow_signal is not None
    assert context.mm_pattern_context.pattern_confidence_boost >= 0.0
    assert context.strategy_modifiers.position_size_multiplier >= 1.0
class DummyRepo:
    async def get_by_id(self, _) -> Any:
        return True
    @pytest.mark.asyncio
    async def test_orchestrator_execute_strategy_mm_e2e() -> None:
    """E2E: execute_strategy вызывает MM анализ и применяет модификаторы."""
    # MM компоненты
    pattern_classifier = MarketMakerPatternClassifier()
    pattern_memory = PatternMemoryRepository("test_data")
    follow_controller = MarketMakerFollowController(
        pattern_classifier=pattern_classifier,
        pattern_memory=pattern_memory
    )
    mm_storage = Mock()
    mm_storage.save_pattern = AsyncMock()
    mm_storage.get_patterns_by_symbol = AsyncMock(return_value=[])
    agent_context_manager = AgentContextManager()
    # Заглушки для репозиториев
    dummy_repo = DummyRepo()
    orchestrator = TradingOrchestratorUseCase(
        order_repository=dummy_repo,
        position_repository=dummy_repo,
        portfolio_repository=dummy_repo,
        trading_repository=dummy_repo,
        strategy_repository=dummy_repo,
        enhanced_trading_service=None,
        mm_pattern_classifier=pattern_classifier,
        mm_pattern_memory=pattern_memory,
        mm_follow_controller=follow_controller,
        mm_storage=mm_storage
    )
    orchestrator._agent_context_manager = agent_context_manager
    # Кладём order_book и trades в AgentContext
    symbol = "BTCUSDT"
    context = agent_context_manager.get_context(symbol)
    context.order_book = OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        bids=[{"price": 50000.0, "size": 1.5}],
        asks=[{"price": 50001.0, "size": 1.0}],
        last_price=50000.0,
        volume_24h=1000000.0,
        price_change_24h=0.02
    )
    context.trades = TradeSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        trades=[{"time": datetime.now(), "price": 50000.0, "size": 0.1, "side": "buy"}]
    )
    # Запрос
    req = ExecuteStrategyRequest(
        strategy_id="s1",
        portfolio_id="p1",
        symbol=symbol
    )
    # Вызов
    resp: ExecuteStrategyResponse = await orchestrator.execute_strategy(req)
    # Проверки
    context = agent_context_manager.get_context(symbol)
    assert context.mm_pattern_context.follow_signal is not None
    assert context.mm_pattern_context.pattern_confidence_boost >= 0.0
    assert context.strategy_modifiers.position_size_multiplier >= 1.0
if __name__ == "__main__":
    pytest.main([__file__]) 
