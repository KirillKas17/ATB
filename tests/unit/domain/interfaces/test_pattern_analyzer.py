"""
Unit тесты для IPatternAnalyzer.

Покрывает:
- Валидацию интерфейса
- Тестирование всех методов
- Проверку типов
- Обработку ошибок
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from domain.interfaces.pattern_analyzer import IPatternAnalyzer
from domain.market_maker.mm_pattern import MarketMakerPattern, PatternMemory, PatternResult
from domain.type_definitions.market_maker_types import Confidence, SimilarityScore


class TestIPatternAnalyzer:
    """Тесты для IPatternAnalyzer."""

    @pytest.fixture
    def mock_pattern_analyzer(self) -> IPatternAnalyzer:
        """Мок реализации IPatternAnalyzer."""
        analyzer = Mock(spec=IPatternAnalyzer)
        analyzer.analyze_pattern_similarity = AsyncMock()
        analyzer.calculate_pattern_confidence = AsyncMock()
        analyzer.predict_pattern_outcome = AsyncMock()
        analyzer.analyze_market_context = AsyncMock()
        analyzer.calculate_pattern_effectiveness = AsyncMock()
        analyzer.get_pattern_recommendations = AsyncMock()
        return analyzer

    @pytest.fixture
    def sample_pattern(self) -> MarketMakerPattern:
        """Тестовый паттерн."""
        return MarketMakerPattern(
            pattern_id="test_pattern_1",
            symbol="BTC/USD",
            pattern_type="liquidity_grab",
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"volume": 1000.0, "price_change": 0.02},
        )

    @pytest.fixture
    def sample_pattern2(self) -> MarketMakerPattern:
        """Второй тестовый паттерн."""
        return MarketMakerPattern(
            pattern_id="test_pattern_2",
            symbol="BTC/USD",
            pattern_type="liquidity_grab",
            confidence=0.75,
            timestamp=datetime.now(),
            metadata={"volume": 1200.0, "price_change": 0.03},
        )

    @pytest.fixture
    def sample_historical_patterns(self) -> List[PatternMemory]:
        """Тестовые исторические паттерны."""
        return [
            PatternMemory(
                pattern_id="hist_1",
                symbol="BTC/USD",
                pattern_type="liquidity_grab",
                success_rate=0.8,
                avg_profit=0.015,
                occurrences=10,
                last_seen=datetime.now(),
            ),
            PatternMemory(
                pattern_id="hist_2",
                symbol="BTC/USD",
                pattern_type="liquidity_grab",
                success_rate=0.7,
                avg_profit=0.012,
                occurrences=15,
                last_seen=datetime.now(),
            ),
        ]

    def test_interface_definition(self):
        """Тест определения интерфейса."""
        assert hasattr(IPatternAnalyzer, "analyze_pattern_similarity")
        assert hasattr(IPatternAnalyzer, "calculate_pattern_confidence")
        assert hasattr(IPatternAnalyzer, "predict_pattern_outcome")
        assert hasattr(IPatternAnalyzer, "analyze_market_context")
        assert hasattr(IPatternAnalyzer, "calculate_pattern_effectiveness")
        assert hasattr(IPatternAnalyzer, "get_pattern_recommendations")

    @pytest.mark.asyncio
    async def test_analyze_pattern_similarity(self, mock_pattern_analyzer, sample_pattern, sample_pattern2):
        """Тест анализа схожести паттернов."""
        expected_similarity = SimilarityScore(value=0.85, confidence=0.9)
        mock_pattern_analyzer.analyze_pattern_similarity.return_value = expected_similarity

        result = await mock_pattern_analyzer.analyze_pattern_similarity(sample_pattern, sample_pattern2)

        assert result == expected_similarity
        mock_pattern_analyzer.analyze_pattern_similarity.assert_called_once_with(sample_pattern, sample_pattern2)

    @pytest.mark.asyncio
    async def test_calculate_pattern_confidence(
        self, mock_pattern_analyzer, sample_pattern, sample_historical_patterns
    ):
        """Тест расчета уверенности в паттерне."""
        expected_confidence = Confidence(value=0.82, factors={"historical_success": 0.8, "market_conditions": 0.85})
        mock_pattern_analyzer.calculate_pattern_confidence.return_value = expected_confidence

        result = await mock_pattern_analyzer.calculate_pattern_confidence(sample_pattern, sample_historical_patterns)

        assert result == expected_confidence
        mock_pattern_analyzer.calculate_pattern_confidence.assert_called_once_with(
            sample_pattern, sample_historical_patterns
        )

    @pytest.mark.asyncio
    async def test_predict_pattern_outcome(self, mock_pattern_analyzer, sample_pattern, sample_historical_patterns):
        """Тест предсказания исхода паттерна."""
        expected_result = PatternResult(
            pattern_id=sample_pattern.pattern_id,
            predicted_outcome="bullish",
            confidence=0.85,
            expected_profit=0.02,
            risk_level="medium",
        )
        mock_pattern_analyzer.predict_pattern_outcome.return_value = expected_result

        result = await mock_pattern_analyzer.predict_pattern_outcome(sample_pattern, sample_historical_patterns)

        assert result == expected_result
        mock_pattern_analyzer.predict_pattern_outcome.assert_called_once_with(
            sample_pattern, sample_historical_patterns
        )

    @pytest.mark.asyncio
    async def test_analyze_market_context(self, mock_pattern_analyzer):
        """Тест анализа рыночного контекста."""
        symbol = "BTC/USD"
        timestamp = datetime.now()
        expected_context = {"volatility": 0.025, "liquidity": "high", "market_regime": "trending", "correlation": 0.75}
        mock_pattern_analyzer.analyze_market_context.return_value = expected_context

        result = await mock_pattern_analyzer.analyze_market_context(symbol, timestamp)

        assert result == expected_context
        mock_pattern_analyzer.analyze_market_context.assert_called_once_with(symbol, timestamp)

    @pytest.mark.asyncio
    async def test_calculate_pattern_effectiveness(
        self, mock_pattern_analyzer, sample_pattern, sample_historical_patterns
    ):
        """Тест расчета эффективности паттерна."""
        expected_effectiveness = 0.78
        mock_pattern_analyzer.calculate_pattern_effectiveness.return_value = expected_effectiveness

        result = await mock_pattern_analyzer.calculate_pattern_effectiveness(sample_pattern, sample_historical_patterns)

        assert result == expected_effectiveness
        mock_pattern_analyzer.calculate_pattern_effectiveness.assert_called_once_with(
            sample_pattern, sample_historical_patterns
        )

    @pytest.mark.asyncio
    async def test_get_pattern_recommendations(self, mock_pattern_analyzer, sample_pattern):
        """Тест получения рекомендаций по паттернам."""
        current_patterns = [sample_pattern]
        expected_recommendations = [
            {
                "action": "buy",
                "confidence": 0.85,
                "reason": "Strong liquidity grab pattern detected",
                "risk_level": "medium",
            },
            {"action": "hold", "confidence": 0.6, "reason": "Wait for confirmation", "risk_level": "low"},
        ]
        mock_pattern_analyzer.get_pattern_recommendations.return_value = expected_recommendations

        result = await mock_pattern_analyzer.get_pattern_recommendations("BTC/USD", current_patterns)

        assert result == expected_recommendations
        mock_pattern_analyzer.get_pattern_recommendations.assert_called_once_with("BTC/USD", current_patterns)

    def test_runtime_checkable_protocol(self):
        """Тест что интерфейс является runtime_checkable."""
        assert hasattr(IPatternAnalyzer, "__runtime_checkable__")
        assert IPatternAnalyzer.__runtime_checkable__ is True

    @pytest.mark.asyncio
    async def test_interface_method_signatures(self: "TestIPatternAnalyzer") -> None:
        """Тест сигнатур методов интерфейса."""
        # Проверяем что методы существуют и являются async
        assert callable(IPatternAnalyzer.analyze_pattern_similarity)
        assert callable(IPatternAnalyzer.calculate_pattern_confidence)
        assert callable(IPatternAnalyzer.predict_pattern_outcome)
        assert callable(IPatternAnalyzer.analyze_market_context)
        assert callable(IPatternAnalyzer.calculate_pattern_effectiveness)
        assert callable(IPatternAnalyzer.get_pattern_recommendations)

    @pytest.mark.asyncio
    async def test_error_handling_in_analyze_pattern_similarity(self, mock_pattern_analyzer):
        """Тест обработки ошибок в analyze_pattern_similarity."""
        mock_pattern_analyzer.analyze_pattern_similarity.side_effect = ValueError("Invalid pattern data")

        with pytest.raises(ValueError, match="Invalid pattern data"):
            await mock_pattern_analyzer.analyze_pattern_similarity(Mock(), Mock())

    @pytest.mark.asyncio
    async def test_error_handling_in_calculate_pattern_confidence(self, mock_pattern_analyzer):
        """Тест обработки ошибок в calculate_pattern_confidence."""
        mock_pattern_analyzer.calculate_pattern_confidence.side_effect = RuntimeError("Historical data unavailable")

        with pytest.raises(RuntimeError, match="Historical data unavailable"):
            await mock_pattern_analyzer.calculate_pattern_confidence(Mock(), [])

    @pytest.mark.asyncio
    async def test_error_handling_in_predict_pattern_outcome(self, mock_pattern_analyzer):
        """Тест обработки ошибок в predict_pattern_outcome."""
        mock_pattern_analyzer.predict_pattern_outcome.side_effect = Exception("Prediction failed")

        with pytest.raises(Exception, match="Prediction failed"):
            await mock_pattern_analyzer.predict_pattern_outcome(Mock(), [])

    @pytest.mark.asyncio
    async def test_error_handling_in_analyze_market_context(self, mock_pattern_analyzer):
        """Тест обработки ошибок в analyze_market_context."""
        mock_pattern_analyzer.analyze_market_context.side_effect = ConnectionError("Market data unavailable")

        with pytest.raises(ConnectionError, match="Market data unavailable"):
            await mock_pattern_analyzer.analyze_market_context("BTC/USD", datetime.now())

    @pytest.mark.asyncio
    async def test_error_handling_in_calculate_pattern_effectiveness(self, mock_pattern_analyzer):
        """Тест обработки ошибок в calculate_pattern_effectiveness."""
        mock_pattern_analyzer.calculate_pattern_effectiveness.side_effect = ValueError(
            "Invalid effectiveness calculation"
        )

        with pytest.raises(ValueError, match="Invalid effectiveness calculation"):
            await mock_pattern_analyzer.calculate_pattern_effectiveness(Mock(), [])

    @pytest.mark.asyncio
    async def test_error_handling_in_get_pattern_recommendations(self, mock_pattern_analyzer):
        """Тест обработки ошибок в get_pattern_recommendations."""
        mock_pattern_analyzer.get_pattern_recommendations.side_effect = Exception("Recommendations unavailable")

        with pytest.raises(Exception, match="Recommendations unavailable"):
            await mock_pattern_analyzer.get_pattern_recommendations("BTC/USD", [])

    def test_interface_documentation(self):
        """Тест наличия документации в интерфейсе."""
        assert IPatternAnalyzer.__doc__ is not None
        assert "Интерфейс для анализа паттернов маркет-мейкера" in IPatternAnalyzer.__doc__

    def test_method_documentation(self):
        """Тест наличия документации методов."""
        assert IPatternAnalyzer.analyze_pattern_similarity.__doc__ is not None
        assert IPatternAnalyzer.calculate_pattern_confidence.__doc__ is not None
        assert IPatternAnalyzer.predict_pattern_outcome.__doc__ is not None
        assert IPatternAnalyzer.analyze_market_context.__doc__ is not None
        assert IPatternAnalyzer.calculate_pattern_effectiveness.__doc__ is not None
        assert IPatternAnalyzer.get_pattern_recommendations.__doc__ is not None
