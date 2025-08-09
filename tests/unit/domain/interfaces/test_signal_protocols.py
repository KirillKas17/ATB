"""
Unit тесты для signal_protocols.

Покрывает:
- Валидацию сигналов
- Тестирование протоколов
- Базовые классы
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from domain.interfaces.signal_protocols import (
    SessionInfluenceSignal,
    MarketMakerSignal,
    SignalEngineProtocol,
    MarketMakerSignalProtocol,
    BaseSignalEngine,
    BaseMarketMakerSignalEngine,
)
from domain.type_definitions.session_types import MarketConditions, MarketRegime, SessionIntensity, SessionType
from domain.value_objects.timestamp import Timestamp


class TestSessionInfluenceSignal:
    """Тесты для SessionInfluenceSignal."""

    @pytest.fixture
    def valid_signal_data(self) -> Dict[str, Any]:
        """Валидные данные для сигнала."""
        return {
            "session_type": SessionType.ASIAN,
            "influence_strength": 0.75,
            "market_conditions": MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            "confidence": 0.85,
            "timestamp": Timestamp.now(),
            "metadata": {"volatility": 0.02, "volume": 1000000.0},
            "predicted_impact": {"price_change": 0.015, "volume_change": 0.2},
        }

    def test_valid_signal_creation(self, valid_signal_data):
        """Тест создания валидного сигнала."""
        signal = SessionInfluenceSignal(**valid_signal_data)

        assert signal.session_type == valid_signal_data["session_type"]
        assert signal.influence_strength == 0.75
        assert signal.confidence == 0.85
        assert signal.metadata == valid_signal_data["metadata"]

    def test_invalid_influence_strength_too_high(self, valid_signal_data):
        """Тест валидации слишком высокой силы влияния."""
        valid_signal_data["influence_strength"] = 1.5

        with pytest.raises(ValueError, match="Influence strength must be between 0.0 and 1.0"):
            SessionInfluenceSignal(**valid_signal_data)

    def test_invalid_influence_strength_too_low(self, valid_signal_data):
        """Тест валидации слишком низкой силы влияния."""
        valid_signal_data["influence_strength"] = -0.1

        with pytest.raises(ValueError, match="Influence strength must be between 0.0 and 1.0"):
            SessionInfluenceSignal(**valid_signal_data)

    def test_invalid_confidence_too_high(self, valid_signal_data):
        """Тест валидации слишком высокой уверенности."""
        valid_signal_data["confidence"] = 1.2

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SessionInfluenceSignal(**valid_signal_data)

    def test_invalid_confidence_too_low(self, valid_signal_data):
        """Тест валидации слишком низкой уверенности."""
        valid_signal_data["confidence"] = -0.1

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SessionInfluenceSignal(**valid_signal_data)

    def test_boundary_values(self, valid_signal_data):
        """Тест граничных значений."""
        # Минимальные значения
        valid_signal_data["influence_strength"] = 0.0
        valid_signal_data["confidence"] = 0.0
        signal = SessionInfluenceSignal(**valid_signal_data)
        assert signal.influence_strength == 0.0
        assert signal.confidence == 0.0

        # Максимальные значения
        valid_signal_data["influence_strength"] = 1.0
        valid_signal_data["confidence"] = 1.0
        signal = SessionInfluenceSignal(**valid_signal_data)
        assert signal.influence_strength == 1.0
        assert signal.confidence == 1.0


class TestMarketMakerSignal:
    """Тесты для MarketMakerSignal."""

    @pytest.fixture
    def valid_mm_signal_data(self) -> Dict[str, Any]:
        """Валидные данные для сигнала маркет-мейкера."""
        return {
            "signal_type": "liquidity_grab",
            "strength": 0.8,
            "direction": "bullish",
            "confidence": 0.9,
            "timestamp": Timestamp.now(),
            "pattern_data": {"volume": 1000.0, "price_change": 0.02},
            "market_context": {"volatility": 0.025, "liquidity": "high"},
        }

    def test_valid_mm_signal_creation(self, valid_mm_signal_data):
        """Тест создания валидного сигнала маркет-мейкера."""
        signal = MarketMakerSignal(**valid_mm_signal_data)

        assert signal.signal_type == "liquidity_grab"
        assert signal.strength == 0.8
        assert signal.direction == "bullish"
        assert signal.confidence == 0.9

    def test_invalid_strength_too_high(self, valid_mm_signal_data):
        """Тест валидации слишком высокой силы сигнала."""
        valid_mm_signal_data["strength"] = 1.5

        with pytest.raises(ValueError, match="Signal strength must be between 0.0 and 1.0"):
            MarketMakerSignal(**valid_mm_signal_data)

    def test_invalid_strength_too_low(self, valid_mm_signal_data):
        """Тест валидации слишком низкой силы сигнала."""
        valid_mm_signal_data["strength"] = -0.1

        with pytest.raises(ValueError, match="Signal strength must be between 0.0 and 1.0"):
            MarketMakerSignal(**valid_mm_signal_data)

    def test_invalid_confidence_too_high(self, valid_mm_signal_data):
        """Тест валидации слишком высокой уверенности."""
        valid_mm_signal_data["confidence"] = 1.2

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            MarketMakerSignal(**valid_mm_signal_data)

    def test_invalid_confidence_too_low(self, valid_mm_signal_data):
        """Тест валидации слишком низкой уверенности."""
        valid_mm_signal_data["confidence"] = -0.1

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            MarketMakerSignal(**valid_mm_signal_data)

    def test_mm_signal_boundary_values(self, valid_mm_signal_data):
        """Тест граничных значений для сигнала маркет-мейкера."""
        # Минимальные значения
        valid_mm_signal_data["strength"] = 0.0
        valid_mm_signal_data["confidence"] = 0.0
        signal = MarketMakerSignal(**valid_mm_signal_data)
        assert signal.strength == 0.0
        assert signal.confidence == 0.0

        # Максимальные значения
        valid_mm_signal_data["strength"] = 1.0
        valid_mm_signal_data["confidence"] = 1.0
        signal = MarketMakerSignal(**valid_mm_signal_data)
        assert signal.strength == 1.0
        assert signal.confidence == 1.0


class TestSignalEngineProtocol:
    """Тесты для SignalEngineProtocol."""

    @pytest.fixture
    def mock_signal_engine(self) -> SignalEngineProtocol:
        """Мок реализации SignalEngineProtocol."""
        engine = Mock(spec=SignalEngineProtocol)
        engine.generate_session_signals = AsyncMock()
        engine.analyze_market_conditions = Mock()
        engine.validate_signal = Mock()
        return engine

    @pytest.fixture
    def sample_session_data(self) -> Dict[str, Any]:
        """Тестовые данные сессии."""
        return {
            "session_type": "asian",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T08:00:00Z",
            "volume": 1000000.0,
        }

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые рыночные данные."""
        return {"price": 50000.0, "volume": 1000000.0, "volatility": 0.02, "spread": 0.001}

    def test_protocol_definition(self):
        """Тест определения протокола."""
        assert hasattr(SignalEngineProtocol, "generate_session_signals")
        assert hasattr(SignalEngineProtocol, "analyze_market_conditions")
        assert hasattr(SignalEngineProtocol, "validate_signal")

    @pytest.mark.asyncio
    async def test_generate_session_signals(self, mock_signal_engine, sample_session_data, sample_market_data):
        """Тест генерации сигналов сессий."""
        expected_signals = [
            SessionInfluenceSignal(
                session_type=SessionType.ASIAN,
                influence_strength=0.75,
                market_conditions=MarketConditions(
                    volatility=0.02,
                    volume=1000000.0,
                    spread=0.001,
                    liquidity=0.8,
                    momentum=0.1,
                    trend_strength=0.6,
                    market_regime=MarketRegime.TRENDING,
                    session_intensity=SessionIntensity.HIGH,
                ),
                confidence=0.85,
                timestamp=Timestamp.now(),
                metadata={"volatility": 0.02},
                predicted_impact={"price_change": 0.015},
            )
        ]
        mock_signal_engine.generate_session_signals.return_value = expected_signals

        result = await mock_signal_engine.generate_session_signals(sample_session_data, sample_market_data)

        assert result == expected_signals
        mock_signal_engine.generate_session_signals.assert_called_once_with(sample_session_data, sample_market_data)

    def test_analyze_market_conditions(self, mock_signal_engine, sample_market_data):
        """Тест анализа рыночных условий."""
        expected_conditions = MarketConditions(
            volatility=0.02,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.1,
            trend_strength=0.6,
            market_regime=MarketRegime.TRENDING,
            session_intensity=SessionIntensity.HIGH,
        )
        mock_signal_engine.analyze_market_conditions.return_value = expected_conditions

        result = mock_signal_engine.analyze_market_conditions(sample_market_data)

        assert result == expected_conditions
        mock_signal_engine.analyze_market_conditions.assert_called_once_with(sample_market_data)

    def test_validate_signal(self, mock_signal_engine):
        """Тест валидации сигнала."""
        signal = SessionInfluenceSignal(
            session_type=SessionType.ASIAN,
            influence_strength=0.75,
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            confidence=0.85,
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )
        mock_signal_engine.validate_signal.return_value = True

        result = mock_signal_engine.validate_signal(signal)

        assert result is True
        mock_signal_engine.validate_signal.assert_called_once_with(signal)


class TestMarketMakerSignalProtocol:
    """Тесты для MarketMakerSignalProtocol."""

    @pytest.fixture
    def mock_mm_signal_engine(self) -> MarketMakerSignalProtocol:
        """Мок реализации MarketMakerSignalProtocol."""
        engine = Mock(spec=MarketMakerSignalProtocol)
        engine.detect_market_maker_patterns = AsyncMock()
        engine.analyze_order_flow = Mock()
        engine.predict_market_movement = Mock()
        return engine

    @pytest.fixture
    def sample_orderbook_data(self) -> Dict[str, Any]:
        """Тестовые данные ордербука."""
        return {
            "bids": [[50000.0, 1.0], [49999.0, 2.0]],
            "asks": [[50001.0, 1.5], [50002.0, 2.5]],
            "timestamp": "2024-01-01T00:00:00Z",
        }

    @pytest.fixture
    def sample_trade_data(self) -> Dict[str, Any]:
        """Тестовые данные сделок."""
        return {
            "trades": [
                {"price": 50000.0, "amount": 0.1, "side": "buy"},
                {"price": 50001.0, "amount": 0.2, "side": "sell"},
            ],
            "timestamp": "2024-01-01T00:00:00Z",
        }

    def test_protocol_definition(self):
        """Тест определения протокола."""
        assert hasattr(MarketMakerSignalProtocol, "detect_market_maker_patterns")
        assert hasattr(MarketMakerSignalProtocol, "analyze_order_flow")
        assert hasattr(MarketMakerSignalProtocol, "predict_market_movement")

    @pytest.mark.asyncio
    async def test_detect_market_maker_patterns(self, mock_mm_signal_engine, sample_orderbook_data, sample_trade_data):
        """Тест обнаружения паттернов маркет-мейкера."""
        expected_signals = [
            MarketMakerSignal(
                signal_type="liquidity_grab",
                strength=0.8,
                direction="bullish",
                confidence=0.9,
                timestamp=Timestamp.now(),
                pattern_data={"volume": 1000.0},
                market_context={"volatility": 0.025},
            )
        ]
        mock_mm_signal_engine.detect_market_maker_patterns.return_value = expected_signals

        result = await mock_mm_signal_engine.detect_market_maker_patterns(sample_orderbook_data, sample_trade_data)

        assert result == expected_signals
        mock_mm_signal_engine.detect_market_maker_patterns.assert_called_once_with(
            sample_orderbook_data, sample_trade_data
        )

    def test_analyze_order_flow(self, mock_mm_signal_engine, sample_orderbook_data):
        """Тест анализа потока ордеров."""
        expected_flow = {"buy_pressure": 0.6, "sell_pressure": 0.4, "imbalance": 0.2, "flow_direction": 0.1}
        mock_mm_signal_engine.analyze_order_flow.return_value = expected_flow

        result = mock_mm_signal_engine.analyze_order_flow(sample_orderbook_data)

        assert result == expected_flow
        mock_mm_signal_engine.analyze_order_flow.assert_called_once_with(sample_orderbook_data)

    def test_predict_market_movement(self, mock_mm_signal_engine):
        """Тест предсказания движения рынка."""
        signals = [
            MarketMakerSignal(
                signal_type="liquidity_grab",
                strength=0.8,
                direction="bullish",
                confidence=0.9,
                timestamp=Timestamp.now(),
                pattern_data={},
                market_context={},
            )
        ]
        expected_prediction = {"direction": "bullish", "strength": 0.8, "confidence": 0.9, "timeframe": "short"}
        mock_mm_signal_engine.predict_market_movement.return_value = expected_prediction

        result = mock_mm_signal_engine.predict_market_movement(signals)

        assert result == expected_prediction
        mock_mm_signal_engine.predict_market_movement.assert_called_once_with(signals)


class TestBaseSignalEngine:
    """Тесты для BaseSignalEngine."""

    @pytest.fixture
    def base_signal_engine(self) -> BaseSignalEngine:
        """Экземпляр базового движка сигналов."""
        return BaseSignalEngine({"test": "config"})

    @pytest.fixture
    def sample_session_data(self) -> Dict[str, Any]:
        """Тестовые данные сессии."""
        return {"session_type": "asian", "volume": 1000000.0}

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые рыночные данные."""
        return {"price": 50000.0, "volatility": 0.02}

    def test_initialization(self):
        """Тест инициализации."""
        engine = BaseSignalEngine()
        assert engine.config == {}
        assert engine._signal_history == []
        assert engine._market_conditions_history == []

        engine_with_config = BaseSignalEngine({"test": "value"})
        assert engine_with_config.config == {"test": "value"}

    def test_analyze_market_conditions(self, base_signal_engine, sample_market_data):
        """Тест анализа рыночных условий."""
        result = base_signal_engine.analyze_market_conditions(sample_market_data)

        assert isinstance(result, MarketConditions)
        assert result.volatility == 0.0
        assert result.volume == 0.0
        assert result.market_regime == MarketRegime.RANGING

    def test_validate_signal_valid(self, base_signal_engine):
        """Тест валидации валидного сигнала."""
        signal = SessionInfluenceSignal(
            session_type=SessionType.ASIAN,
            influence_strength=0.75,
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            confidence=0.85,
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )

        result = base_signal_engine.validate_signal(signal)
        assert result is True

    def test_validate_signal_invalid_strength(self, base_signal_engine):
        """Тест валидации сигнала с невалидной силой."""
        signal = SessionInfluenceSignal(
            session_type=SessionType.ASIAN,
            influence_strength=1.5,  # Невалидное значение
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            confidence=0.85,
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )

        result = base_signal_engine.validate_signal(signal)
        assert result is False

    def test_validate_signal_invalid_confidence(self, base_signal_engine):
        """Тест валидации сигнала с невалидной уверенностью."""
        signal = SessionInfluenceSignal(
            session_type=SessionType.ASIAN,
            influence_strength=0.75,
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            confidence=1.5,  # Невалидное значение
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )

        result = base_signal_engine.validate_signal(signal)
        assert result is False

    def test_get_signal_history(self, base_signal_engine):
        """Тест получения истории сигналов."""
        # Добавляем тестовые сигналы
        signal1 = SessionInfluenceSignal(
            session_type=SessionType.ASIAN,
            influence_strength=0.75,
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=1000000.0,
                spread=0.001,
                liquidity=0.8,
                momentum=0.1,
                trend_strength=0.6,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,
            ),
            confidence=0.85,
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )
        signal2 = SessionInfluenceSignal(
            session_type=SessionType.EUROPEAN,
            influence_strength=0.6,
            market_conditions=MarketConditions(
                volatility=0.015,
                volume=800000.0,
                spread=0.002,
                liquidity=0.7,
                momentum=0.05,
                trend_strength=0.4,
                market_regime=MarketRegime.RANGING,
                session_intensity=SessionIntensity.NORMAL,
            ),
            confidence=0.7,
            timestamp=Timestamp.now(),
            metadata={},
            predicted_impact={},
        )

        base_signal_engine._signal_history = [signal1, signal2]

        # Тест получения всей истории
        history = base_signal_engine.get_signal_history()
        assert len(history) == 2
        assert history[0] == signal1
        assert history[1] == signal2

        # Тест ограничения истории
        limited_history = base_signal_engine.get_signal_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == signal2  # Последний сигнал

    def test_get_market_conditions_history(self, base_signal_engine):
        """Тест получения истории рыночных условий."""
        conditions1 = MarketConditions(
            volatility=0.02,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.1,
            trend_strength=0.6,
            market_regime=MarketRegime.TRENDING,
            session_intensity=SessionIntensity.HIGH,
        )
        conditions2 = MarketConditions(
            volatility=0.015,
            volume=800000.0,
            spread=0.002,
            liquidity=0.7,
            momentum=0.05,
            trend_strength=0.4,
            market_regime=MarketRegime.RANGING,
            session_intensity=SessionIntensity.NORMAL,
        )

        base_signal_engine._market_conditions_history = [conditions1, conditions2]

        # Тест получения всей истории
        history = base_signal_engine.get_market_conditions_history()
        assert len(history) == 2
        assert history[0] == conditions1
        assert history[1] == conditions2

        # Тест ограничения истории
        limited_history = base_signal_engine.get_market_conditions_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == conditions2  # Последние условия

    @pytest.mark.asyncio
    async def test_generate_session_signals_abstract(self, base_signal_engine, sample_session_data, sample_market_data):
        """Тест что generate_session_signals является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_signal_engine.generate_session_signals(sample_session_data, sample_market_data)


class TestBaseMarketMakerSignalEngine:
    """Тесты для BaseMarketMakerSignalEngine."""

    @pytest.fixture
    def base_mm_signal_engine(self) -> BaseMarketMakerSignalEngine:
        """Экземпляр базового движка сигналов маркет-мейкера."""
        return BaseMarketMakerSignalEngine({"test": "config"})

    @pytest.fixture
    def sample_orderbook_data(self) -> Dict[str, Any]:
        """Тестовые данные ордербука."""
        return {"bids": [[50000.0, 1.0], [49999.0, 2.0]], "asks": [[50001.0, 1.5], [50002.0, 2.5]]}

    @pytest.fixture
    def sample_trade_data(self) -> Dict[str, Any]:
        """Тестовые данные сделок."""
        return {
            "trades": [
                {"price": 50000.0, "amount": 0.1, "side": "buy"},
                {"price": 50001.0, "amount": 0.2, "side": "sell"},
            ]
        }

    def test_initialization(self):
        """Тест инициализации."""
        engine = BaseMarketMakerSignalEngine()
        assert engine.config == {}
        assert engine._pattern_history == []
        assert engine._flow_history == []

        engine_with_config = BaseMarketMakerSignalEngine({"test": "value"})
        assert engine_with_config.config == {"test": "value"}

    def test_analyze_order_flow(self, base_mm_signal_engine, sample_orderbook_data):
        """Тест анализа потока ордеров."""
        result = base_mm_signal_engine.analyze_order_flow(sample_orderbook_data)

        assert isinstance(result, dict)
        assert "buy_pressure" in result
        assert "sell_pressure" in result
        assert "imbalance" in result
        assert "flow_direction" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_predict_market_movement(self, base_mm_signal_engine):
        """Тест предсказания движения рынка."""
        signals = [
            MarketMakerSignal(
                signal_type="liquidity_grab",
                strength=0.8,
                direction="bullish",
                confidence=0.9,
                timestamp=Timestamp.now(),
                pattern_data={},
                market_context={},
            )
        ]

        result = base_mm_signal_engine.predict_market_movement(signals)

        assert isinstance(result, dict)
        assert "direction" in result
        assert "strength" in result
        assert "confidence" in result
        assert "timeframe" in result
        assert result["direction"] == "neutral"
        assert result["strength"] == 0.0
        assert result["confidence"] == 0.0

    def test_get_pattern_history(self, base_mm_signal_engine):
        """Тест получения истории паттернов."""
        pattern1 = MarketMakerSignal(
            signal_type="liquidity_grab",
            strength=0.8,
            direction="bullish",
            confidence=0.9,
            timestamp=Timestamp.now(),
            pattern_data={"volume": 1000.0},
            market_context={"volatility": 0.025},
        )
        pattern2 = MarketMakerSignal(
            signal_type="stop_hunt",
            strength=0.6,
            direction="bearish",
            confidence=0.7,
            timestamp=Timestamp.now(),
            pattern_data={"volume": 800.0},
            market_context={"volatility": 0.02},
        )

        base_mm_signal_engine._pattern_history = [pattern1, pattern2]

        # Тест получения всей истории
        history = base_mm_signal_engine.get_pattern_history()
        assert len(history) == 2
        assert history[0] == pattern1
        assert history[1] == pattern2

        # Тест ограничения истории
        limited_history = base_mm_signal_engine.get_pattern_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == pattern2  # Последний паттерн

    def test_get_flow_history(self, base_mm_signal_engine):
        """Тест получения истории потока."""
        flow1 = {"buy_pressure": 0.6, "sell_pressure": 0.4, "imbalance": 0.2, "flow_direction": 0.1}
        flow2 = {"buy_pressure": 0.4, "sell_pressure": 0.6, "imbalance": -0.2, "flow_direction": -0.1}

        base_mm_signal_engine._flow_history = [flow1, flow2]

        # Тест получения всей истории
        history = base_mm_signal_engine.get_flow_history()
        assert len(history) == 2
        assert history[0] == flow1
        assert history[1] == flow2

        # Тест ограничения истории
        limited_history = base_mm_signal_engine.get_flow_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == flow2  # Последний поток

    @pytest.mark.asyncio
    async def test_detect_market_maker_patterns_abstract(
        self, base_mm_signal_engine, sample_orderbook_data, sample_trade_data
    ):
        """Тест что detect_market_maker_patterns является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_mm_signal_engine.detect_market_maker_patterns(sample_orderbook_data, sample_trade_data)
