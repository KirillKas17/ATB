"""
Тесты для сигналов application слоя.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from typing import Any, Dict, List
import pandas as pd

from application.signal.session_signal_engine import SessionSignalEngine, SessionInfluenceSignal
from domain.sessions.session_influence_analyzer import SessionInfluenceResult
from domain.sessions.session_marker import MarketSessionContext
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.session_type import SessionType
from domain.value_objects.session_phase import SessionPhase


class TestSessionSignalEngine:
    """Тесты для SessionSignalEngine."""

    @pytest.fixture
    def mock_session_analyzer(self) -> Mock:
        """Создает mock анализатора сессий."""
        analyzer = Mock()
        analyzer.analyze_session_influence = Mock()
        return analyzer

    @pytest.fixture
    def mock_session_marker(self) -> Mock:
        """Создает mock маркера сессий."""
        marker = Mock()
        context = MarketSessionContext(
            primary_session=None,
        )
        marker.get_session_context = Mock(return_value=context)
        return marker

    @pytest.fixture
    def engine(self, mock_session_analyzer: Mock, mock_session_marker: Mock) -> SessionSignalEngine:
        """Создает экземпляр движка сигналов."""
        with patch('application.signal.session_signal_engine.SessionInfluenceAnalyzer', return_value=mock_session_analyzer):
            with patch('application.signal.session_signal_engine.SessionMarker', return_value=mock_session_marker):
                return SessionSignalEngine(
                    session_analyzer=mock_session_analyzer,
                    session_marker=mock_session_marker
                )

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Создает образец рыночных данных."""
        return {
            "timestamp": "2024-01-01T00:00:00",
            "close": 50000.0,
            "volume": 1000.0,
            "high": 51000.0,
            "low": 49000.0,
            "open": 49500.0
        }

    @pytest.fixture
    def sample_influence_result(self) -> SessionInfluenceResult:
        """Создает образец результата анализа влияния сессии."""
        return SessionInfluenceResult(
            symbol="BTC/USD",
            session_type=SessionType.LONDON,
            session_phase=SessionPhase.ACTIVE,
            timestamp=Timestamp.now(),
            influence_metrics={},
            predicted_volatility=0.2,
            predicted_volume=0.3,
            predicted_direction="bullish",
            confidence=0.8,
            market_context={},
            historical_patterns={}
        )

    @pytest.mark.asyncio
    async def test_generate_signal(self, engine: SessionSignalEngine, sample_market_data: Dict[str, Any], sample_influence_result: SessionInfluenceResult) -> None:
        """Тест генерации сигнала."""
        symbol = "BTC/USD"
        timestamp = Timestamp.now()
        
        # Настраиваем mock
        engine.session_analyzer.analyze_session_influence.return_value = sample_influence_result
        
        result = await engine.generate_signal(symbol, sample_market_data, timestamp)
        
        assert result is not None
        assert isinstance(result, SessionInfluenceSignal)
        assert result.symbol == symbol
        assert result.session_type == "LONDON"
        assert result.session_phase == "ACTIVE"
        assert result.confidence == 0.8
        assert result.tendency in ["bullish", "bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_generate_signal_no_market_data(self, engine: SessionSignalEngine) -> None:
        """Тест генерации сигнала без рыночных данных."""
        symbol = "BTC/USD"
        
        result = await engine.generate_signal(symbol)
        
        # Должен вернуть None или базовый сигнал
        assert result is None or isinstance(result, SessionInfluenceSignal)

    @pytest.mark.asyncio
    async def test_get_current_signals(self, engine: SessionSignalEngine) -> None:
        """Тест получения текущих сигналов."""
        symbol = "BTC/USD"
        
        # Создаем тестовый сигнал
        test_signal = SessionInfluenceSignal(
            symbol=symbol,
            score=0.7,
            tendency="bullish",
            confidence=0.8,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now()
        )
        
        # Добавляем сигнал в хранилище
        engine.signals[symbol] = [test_signal]
        
        signals = await engine.get_current_signals(symbol)
        
        assert isinstance(signals, list)
        assert len(signals) == 1
        assert signals[0] == test_signal

    @pytest.mark.asyncio
    async def test_get_aggregated_signal(self, engine: SessionSignalEngine) -> None:
        """Тест получения агрегированного сигнала."""
        symbol = "BTC/USD"
        
        # Создаем тестовые сигналы
        signal1 = SessionInfluenceSignal(
            symbol=symbol,
            score=0.7,
            tendency="bullish",
            confidence=0.8,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now()
        )
        signal2 = SessionInfluenceSignal(
            symbol=symbol,
            score=0.5,
            tendency="bullish",
            confidence=0.6,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now()
        )
        
        # Добавляем сигналы в хранилище
        engine.signals[symbol] = [signal1, signal2]
        
        aggregated = await engine.get_aggregated_signal(symbol)
        
        assert aggregated is not None
        assert isinstance(aggregated, SessionInfluenceSignal)
        assert aggregated.symbol == symbol

    @pytest.mark.asyncio
    async def test_get_session_analysis(self, engine: SessionSignalEngine) -> None:
        """Тест получения анализа сессии."""
        symbol = "BTC/USD"
        
        analysis = engine.get_session_analysis(symbol)
        
        assert isinstance(analysis, dict)
        assert "current_session" in analysis
        assert "session_signals" in analysis
        assert "aggregated_signal" in analysis
        assert "statistics" in analysis

    def test_create_signal_from_influence_result(self, engine: SessionSignalEngine, sample_influence_result: SessionInfluenceResult) -> None:
        """Тест создания сигнала из результата анализа."""
        signal = engine._create_signal_from_influence_result(sample_influence_result)
        
        assert isinstance(signal, SessionInfluenceSignal)
        assert signal.symbol == sample_influence_result.symbol
        assert signal.session_type == sample_influence_result.session_type
        assert signal.session_phase == sample_influence_result.session_phase
        assert signal.confidence == sample_influence_result.confidence

    def test_generate_basic_influence_result(self, engine: SessionSignalEngine) -> None:
        """Тест генерации базового результата влияния."""
        symbol = "BTC/USD"
        context = MarketSessionContext(
            primary_session=None,
        )
        timestamp = Timestamp.now()
        
        result = engine._generate_basic_influence_result(symbol, context, timestamp)
        
        assert isinstance(result, SessionInfluenceResult)
        assert result.symbol == symbol
        assert result.session_type == "LONDON"
        assert result.session_phase == "ACTIVE"

    def test_store_signal(self, engine: SessionSignalEngine) -> None:
        """Тест сохранения сигнала."""
        symbol = "BTC/USD"
        signal = SessionInfluenceSignal(
            symbol=symbol,
            score=0.7,
            tendency="bullish",
            confidence=0.8,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now()
        )
        
        engine._store_signal(symbol, signal)
        
        assert symbol in engine.signals
        assert signal in engine.signals[symbol]
        assert symbol in engine.signal_history

    def test_update_stats(self, engine: SessionSignalEngine) -> None:
        """Тест обновления статистики."""
        signal = SessionInfluenceSignal(
            symbol="BTC/USD",
            score=0.7,
            tendency="bullish",
            confidence=0.8,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now()
        )
        
        initial_total = engine.stats["total_signals_generated"]
        initial_bullish = engine.stats["bullish_signals"]
        
        engine._update_stats(signal)
        
        assert engine.stats["total_signals_generated"] == initial_total + 1
        assert engine.stats["bullish_signals"] == initial_bullish + 1

    def test_get_signal_statistics(self, engine: SessionSignalEngine) -> None:
        """Тест получения статистики сигналов."""
        symbol = "BTC/USD"
        
        stats = engine._get_signal_statistics(symbol)
        
        assert isinstance(stats, dict)
        assert "total_signals" in stats
        assert "active_signals" in stats
        assert "high_confidence_signals" in stats
        assert "session_distribution" in stats

    def test_get_statistics(self, engine: SessionSignalEngine) -> None:
        """Тест получения общей статистики."""
        stats = engine.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_signals_generated" in stats
        assert "high_confidence_signals" in stats
        assert "bullish_signals" in stats
        assert "bearish_signals" in stats
        assert "neutral_signals" in stats

    @pytest.mark.asyncio
    async def test_start_stop(self, engine: SessionSignalEngine) -> None:
        """Тест запуска и остановки движка."""
        # Тест запуска
        await engine.start()
        assert engine._running is True
        
        # Тест остановки
        await engine.stop()
        assert engine._running is False

    def test_signal_to_dict(self) -> None:
        """Тест преобразования сигнала в словарь."""
        signal = SessionInfluenceSignal(
            symbol="BTC/USD",
            score=0.7,
            tendency="bullish",
            confidence=0.8,
            session_type="LONDON",
            session_phase="ACTIVE",
            timestamp=Timestamp.now(),
            volatility_impact=0.2,
            volume_impact=0.3,
            momentum_impact=0.4,
            reversal_probability=0.1,
            false_breakout_probability=0.05,
            metadata={"trend": "bullish"}
        )
        
        signal_dict = signal.to_dict()
        
        assert isinstance(signal_dict, dict)
        assert signal_dict["symbol"] == "BTC/USD"
        assert signal_dict["tendency"] == "bullish"
        assert signal_dict["confidence"] == 0.8
        assert signal_dict["session_type"] == "LONDON"
        assert signal_dict["session_phase"] == "ACTIVE"
        assert "timestamp" in signal_dict
        assert signal_dict["volatility_impact"] == 0.2
        assert signal_dict["volume_impact"] == 0.3
        assert signal_dict["momentum_impact"] == 0.4
        assert signal_dict["reversal_probability"] == 0.1
        assert signal_dict["false_breakout_probability"] == 0.05
        assert signal_dict["metadata"]["trend"] == "bullish" 
