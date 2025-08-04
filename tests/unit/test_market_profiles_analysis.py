"""
Юнит-тесты для модулей анализа market_profiles.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from infrastructure.market_profiles.analysis.pattern_analyzer import PatternAnalyzer
from infrastructure.market_profiles.analysis.similarity_calculator import SimilarityCalculator
from infrastructure.market_profiles.analysis.success_rate_analyzer import SuccessRateAnalyzer
from infrastructure.market_profiles.models.analysis_config import AnalysisConfig
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType,
    PatternResult, PatternOutcome, PatternMemory
)
from domain.type_definitions.market_maker_types import (
    BookPressure, VolumeDelta, PriceReaction, SpreadChange,
    OrderImbalance, LiquidityDepth, TimeDuration, VolumeConcentration,
    PriceVolatility, MarketMicrostructure, Confidence, Accuracy,
    AverageReturn, SuccessCount, TotalCount
)
class TestPatternAnalyzer:
    """Тесты для PatternAnalyzer."""
    @pytest.fixture
    def config(self) -> Any:
        """Конфигурация анализатора."""
        return AnalysisConfig(
            min_confidence=Confidence(0.6),
            similarity_threshold=0.8,
            accuracy_threshold=0.7
        )
    @pytest.fixture
    def analyzer(self, config) -> Any:
        """Экземпляр анализатора."""
        return PatternAnalyzer(config)
    @pytest.fixture
    def sample_pattern(self) -> Any:
        """Образец паттерна для тестов."""
        features = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.15),
            price_reaction=PriceReaction(0.02),
            spread_change=SpreadChange(0.05),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.8),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.75),
            price_volatility=PriceVolatility(0.03),
            market_microstructure=MarketMicrostructure({
                "depth_imbalance": 0.4,
                "flow_imbalance": 0.6
            })
        )
        return MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.85),
            context={"market_regime": "trending", "session": "asian"}
        )
    def test_analyzer_initialization(self, config) -> None:
        """Тест инициализации анализатора."""
        analyzer = PatternAnalyzer(config)
        assert analyzer.config == config
        assert analyzer.config.min_confidence == Confidence(0.6)
        assert analyzer.config.similarity_threshold == 0.8
        assert analyzer.config.accuracy_threshold == 0.7
    @pytest.mark.asyncio
    async def test_analyze_pattern_success(self, analyzer, sample_pattern) -> None:
        """Тест успешного анализа паттерна."""
        analysis = await analyzer.analyze_pattern("BTCUSDT", sample_pattern)
        assert analysis is not None
        assert "confidence" in analysis
        assert "similarity_score" in analysis
        assert "success_probability" in analysis
        assert "market_context" in analysis
        assert "risk_assessment" in analysis
        assert "recommendations" in analysis
    @pytest.mark.asyncio
    async def test_analyze_pattern_low_confidence(self, analyzer) -> None:
        """Тест анализа паттерна с низкой уверенностью."""
        features = PatternFeatures(
            book_pressure=BookPressure(0.1),
            volume_delta=VolumeDelta(0.05),
            price_reaction=PriceReaction(0.01),
            spread_change=SpreadChange(0.01),
            order_imbalance=OrderImbalance(0.1),
            liquidity_depth=LiquidityDepth(0.2),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.2),
            price_volatility=PriceVolatility(0.01),
            market_microstructure=MarketMicrostructure({})
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ABSORPTION,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.3),  # Низкая уверенность
            context={"market_regime": "sideways", "session": "asian"}
        )
        analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
        assert analysis is not None
        assert analysis["confidence"] < 0.6  # Должно быть ниже порога
    @pytest.mark.asyncio
    async def test_analyze_market_context(self, analyzer) -> None:
        """Тест анализа рыночного контекста."""
        context = await analyzer.analyze_market_context("BTCUSDT", datetime.now())
        assert context is not None
        assert "market_phase" in context
        assert "volatility_regime" in context
        assert "liquidity_regime" in context
        assert "volume_profile" in context
        assert "price_action" in context
        assert "order_flow" in context
    @pytest.mark.asyncio
    async def test_analyze_pattern_with_storage(self, analyzer, sample_pattern) -> None:
        """Тест анализа паттерна с хранилищем."""
        # Мокаем хранилище
        mock_storage = AsyncMock()
        mock_storage.get_patterns_by_symbol.return_value = []
        analyzer.storage = mock_storage
        analysis = await analyzer.analyze_pattern("BTCUSDT", sample_pattern)
        assert analysis is not None
        mock_storage.get_patterns_by_symbol.assert_called_once_with("BTCUSDT")
    @pytest.mark.asyncio
    async def test_analyze_pattern_with_similarity_calculator(self, analyzer, sample_pattern) -> None:
        """Тест анализа паттерна с калькулятором схожести."""
        # Мокаем калькулятор схожести
        mock_calculator = AsyncMock()
        mock_calculator.calculate_similarity.return_value = 0.85
        analyzer.similarity_calculator = mock_calculator
        analysis = await analyzer.analyze_pattern("BTCUSDT", sample_pattern)
        assert analysis is not None
        assert analysis["similarity_score"] == 0.85
    @pytest.mark.asyncio
    async def test_analyze_pattern_with_success_analyzer(self, analyzer, sample_pattern) -> None:
        """Тест анализа паттерна с анализатором успешности."""
        # Мокаем анализатор успешности
        mock_success_analyzer = AsyncMock()
        mock_success_analyzer.calculate_success_rate.return_value = 0.75
        analyzer.success_rate_analyzer = mock_success_analyzer
        analysis = await analyzer.analyze_pattern("BTCUSDT", sample_pattern)
        assert analysis is not None
        assert analysis["success_probability"] == 0.75
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await analyzer.analyze_pattern("", None)
    def test_config_validation(self) -> None:
        """Тест валидации конфигурации."""
        config = AnalysisConfig(
            min_confidence=Confidence(0.5),
            similarity_threshold=0.7,
            accuracy_threshold=0.6
        )
        assert config.min_confidence == Confidence(0.5)
        assert config.similarity_threshold == 0.7
        assert config.accuracy_threshold == 0.6
        assert len(config.feature_weights) > 0
        assert len(config.market_phase_weights) > 0
class TestSimilarityCalculator:
    """Тесты для SimilarityCalculator."""
    @pytest.fixture
    def calculator(self) -> Any:
        """Экземпляр калькулятора."""
        return SimilarityCalculator()
    @pytest.fixture
    def sample_features(self) -> Any:
        """Образец признаков для тестов."""
        return {
            "book_pressure": 0.7,
            "volume_delta": 0.15,
            "price_reaction": 0.02,
            "spread_change": 0.05,
            "order_imbalance": 0.6,
            "liquidity_depth": 0.8,
            "volume_concentration": 0.75
        }
    def test_calculator_initialization(self, calculator) -> None:
        """Тест инициализации калькулятора."""
        assert calculator is not None
        assert hasattr(calculator, 'calculate_similarity')
    @pytest.mark.asyncio
    async def test_calculate_similarity_identical(self, calculator, sample_features) -> None:
        """Тест расчета схожести для идентичных признаков."""
        similarity = await calculator.calculate_similarity(sample_features, sample_features)
        assert similarity == 1.0
    @pytest.mark.asyncio
    async def test_calculate_similarity_different(self, calculator, sample_features) -> None:
        """Тест расчета схожести для разных признаков."""
        different_features = {
            "book_pressure": 0.1,
            "volume_delta": 0.05,
            "price_reaction": 0.01,
            "spread_change": 0.01,
            "order_imbalance": 0.1,
            "liquidity_depth": 0.2,
            "volume_concentration": 0.2
        }
        similarity = await calculator.calculate_similarity(sample_features, different_features)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0
    @pytest.mark.asyncio
    async def test_calculate_similarity_partial(self, calculator, sample_features) -> None:
        """Тест расчета схожести для частично совпадающих признаков."""
        partial_features = {
            "book_pressure": 0.7,  # Идентично
            "volume_delta": 0.15,  # Идентично
            "price_reaction": 0.01,  # Различается
            "spread_change": 0.05,  # Идентично
            "order_imbalance": 0.6,  # Идентично
            "liquidity_depth": 0.8,  # Идентично
            "volume_concentration": 0.75  # Идентично
        }
        similarity = await calculator.calculate_similarity(sample_features, partial_features)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Должно быть достаточно высоким
    @pytest.mark.asyncio
    async def test_calculate_similarity_with_weights(self, calculator, sample_features) -> None:
        """Тест расчета схожести с весами."""
        different_features = {
            "book_pressure": 0.1,
            "volume_delta": 0.15,  # Идентично
            "price_reaction": 0.02,  # Идентично
            "spread_change": 0.01,
            "order_imbalance": 0.6,  # Идентично
            "liquidity_depth": 0.8,  # Идентично
            "volume_concentration": 0.75  # Идентично
        }
        # Используем кастомные веса
        weights = {
            "book_pressure": 0.5,  # Высокий вес
            "volume_delta": 0.1,
            "price_reaction": 0.1,
            "spread_change": 0.1,
            "order_imbalance": 0.1,
            "liquidity_depth": 0.05,
            "volume_concentration": 0.05
        }
        similarity = await calculator.calculate_similarity(
            sample_features, different_features, weights
        )
        assert 0.0 <= similarity <= 1.0
    @pytest.mark.asyncio
    async def test_calculate_similarity_empty_features(self, calculator) -> None:
        """Тест расчета схожести для пустых признаков."""
        similarity = await calculator.calculate_similarity({}, {})
        assert similarity == 0.0
    @pytest.mark.asyncio
    async def test_calculate_similarity_missing_features(self, calculator, sample_features) -> None:
        """Тест расчета схожести для отсутствующих признаков."""
        partial_features = {
            "book_pressure": 0.7,
            "volume_delta": 0.15
            # Остальные признаки отсутствуют
        }
        similarity = await calculator.calculate_similarity(sample_features, partial_features)
        assert 0.0 <= similarity <= 1.0
    @pytest.mark.asyncio
    async def test_calculate_similarity_edge_cases(self, calculator) -> None:
        """Тест граничных случаев расчета схожести."""
        # Тест с None значениями
        features1 = {"book_pressure": 0.7, "volume_delta": None}
        features2 = {"book_pressure": 0.7, "volume_delta": 0.15}
        similarity = await calculator.calculate_similarity(features1, features2)
        assert 0.0 <= similarity <= 1.0
        # Тест с очень большими значениями
        features1 = {"book_pressure": 1000.0, "volume_delta": 0.15}
        features2 = {"book_pressure": 0.7, "volume_delta": 0.15}
        similarity = await calculator.calculate_similarity(features1, features2)
        assert 0.0 <= similarity <= 1.0
class TestSuccessRateAnalyzer:
    """Тесты для SuccessRateAnalyzer."""
    @pytest.fixture
    def analyzer(self) -> Any:
        """Экземпляр анализатора."""
        return SuccessRateAnalyzer()
    @pytest.fixture
    def sample_pattern_memories(self) -> Any:
        """Образцы памяти паттернов для тестов."""
        features = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.15),
            price_reaction=PriceReaction(0.02),
            spread_change=SpreadChange(0.05),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.8),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.75),
            price_volatility=PriceVolatility(0.03),
            market_microstructure=MarketMicrostructure({
                "depth_imbalance": 0.4,
                "flow_imbalance": 0.6
            })
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.85),
            context={"market_regime": "trending", "session": "asian"}
        )
        # Создаем успешные результаты
        success_result = PatternResult(
            outcome=PatternOutcome.SUCCESS,
            price_change_15min=0.02,
            price_change_1h=0.05,
            volume_change=0.1,
            execution_time=300,
            confidence=Confidence(0.8)
        )
        # Создаем неуспешные результаты
        failure_result = PatternResult(
            outcome=PatternOutcome.FAILURE,
            price_change_15min=-0.01,
            price_change_1h=-0.02,
            volume_change=-0.05,
            execution_time=300,
            confidence=Confidence(0.6)
        )
        memories = []
        # Добавляем успешные паттерны
        for i in range(8):
            memory = PatternMemory(
                pattern=pattern,
                result=success_result,
                accuracy=Accuracy(0.8 + i * 0.02),
                avg_return=AverageReturn(0.02 + i * 0.001),
                success_count=SuccessCount(8),
                total_count=TotalCount(10),
                last_seen=datetime.now()
            )
            memories.append(memory)
        # Добавляем неуспешные паттерны
        for i in range(2):
            memory = PatternMemory(
                pattern=pattern,
                result=failure_result,
                accuracy=Accuracy(0.3 + i * 0.1),
                avg_return=AverageReturn(-0.01 + i * 0.001),
                success_count=SuccessCount(2),
                total_count=TotalCount(10),
                last_seen=datetime.now()
            )
            memories.append(memory)
        return memories
    def test_analyzer_initialization(self, analyzer) -> None:
        """Тест инициализации анализатора."""
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_success_rate')
        assert hasattr(analyzer, 'analyze_success_trends')
    @pytest.mark.asyncio
    async def test_calculate_success_rate(self, analyzer, sample_pattern_memories) -> None:
        """Тест расчета успешности."""
        success_rate = await analyzer.calculate_success_rate(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, sample_pattern_memories
        )
        assert 0.0 <= success_rate <= 1.0
        assert success_rate > 0.5  # Должно быть больше 50% для наших данных
    @pytest.mark.asyncio
    async def test_calculate_success_rate_empty_data(self, analyzer) -> None:
        """Тест расчета успешности для пустых данных."""
        success_rate = await analyzer.calculate_success_rate(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, []
        )
        assert success_rate == 0.0
    @pytest.mark.asyncio
    async def test_calculate_success_rate_all_successful(self, analyzer, sample_pattern_memories) -> None:
        """Тест расчета успешности для всех успешных паттернов."""
        # Берем только успешные паттерны
        successful_memories = sample_pattern_memories[:8]
        success_rate = await analyzer.calculate_success_rate(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, successful_memories
        )
        assert success_rate == 1.0
    @pytest.mark.asyncio
    async def test_calculate_success_rate_all_failed(self, analyzer, sample_pattern_memories) -> None:
        """Тест расчета успешности для всех неуспешных паттернов."""
        # Берем только неуспешные паттерны
        failed_memories = sample_pattern_memories[8:]
        success_rate = await analyzer.calculate_success_rate(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, failed_memories
        )
        assert success_rate == 0.0
    @pytest.mark.asyncio
    async def test_analyze_success_trends(self, analyzer, sample_pattern_memories) -> None:
        """Тест анализа трендов успешности."""
        trends = await analyzer.analyze_success_trends(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, sample_pattern_memories
        )
        assert trends is not None
        assert "trend_direction" in trends
        assert "trend_strength" in trends
        assert "confidence" in trends
        assert "periods" in trends
    @pytest.mark.asyncio
    async def test_analyze_success_trends_empty_data(self, analyzer) -> None:
        """Тест анализа трендов для пустых данных."""
        trends = await analyzer.analyze_success_trends(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, []
        )
        assert trends is not None
        assert trends["trend_direction"] == "neutral"
        assert trends["trend_strength"] == 0.0
    @pytest.mark.asyncio
    async def test_calculate_accuracy_metrics(self, analyzer, sample_pattern_memories) -> None:
        """Тест расчета метрик точности."""
        metrics = await analyzer.calculate_accuracy_metrics(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, sample_pattern_memories
        )
        assert metrics is not None
        assert "avg_accuracy" in metrics
        assert "accuracy_std" in metrics
        assert "min_accuracy" in metrics
        assert "max_accuracy" in metrics
        assert "accuracy_trend" in metrics
    @pytest.mark.asyncio
    async def test_calculate_return_metrics(self, analyzer, sample_pattern_memories) -> None:
        """Тест расчета метрик доходности."""
        metrics = await analyzer.calculate_return_metrics(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, sample_pattern_memories
        )
        assert metrics is not None
        assert "avg_return" in metrics
        assert "return_std" in metrics
        assert "min_return" in metrics
        assert "max_return" in metrics
        assert "return_trend" in metrics
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, analyzer, sample_pattern_memories) -> None:
        """Тест генерации рекомендаций."""
        recommendations = await analyzer.generate_recommendations(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, sample_pattern_memories
        )
        assert recommendations is not None
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await analyzer.calculate_success_rate("", None, None)
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
