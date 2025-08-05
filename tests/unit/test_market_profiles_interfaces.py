"""
Юнит-тесты для интерфейсов market_profiles.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from infrastructure.market_profiles.interfaces.storage_interfaces import (
    IPatternStorage, IBehaviorHistoryStorage, IPatternAnalyzer
)
from infrastructure.market_profiles.interfaces.analysis_interfaces import (
    ISimilarityCalculator, ISuccessRateAnalyzer
)
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType,
    PatternResult, PatternOutcome
)
from domain.type_definitions.market_maker_types import (
    BookPressure, VolumeDelta, PriceReaction, SpreadChange,
    OrderImbalance, LiquidityDepth, TimeDuration, VolumeConcentration,
    PriceVolatility, MarketMicrostructure, Confidence
)


class TestIPatternStorage:
    """Тесты для интерфейса IPatternStorage."""

    def test_interface_definition(self: "TestIPatternStorage") -> None:
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(IPatternStorage, '__protocol_attrs__')
        
        # Проверяем наличие обязательных методов
        required_methods = [
            'save_pattern',
            'get_patterns_by_symbol',
            'update_pattern_result',
            'get_successful_patterns',
            'find_similar_patterns',
            'get_storage_statistics',
            'cleanup_old_data',
            'backup_data',
            'validate_data_integrity',
            'get_pattern_metadata',
            'close'
        ]
        
        for method in required_methods:
            assert hasattr(IPatternStorage, method), f"Метод {method} отсутствует в интерфейсе"

    def test_interface_method_signatures(self: "TestIPatternStorage") -> None:
        """Тест сигнатур методов интерфейса."""
        # Создаем мок, который реализует интерфейс
        mock_storage = Mock(spec=IPatternStorage)
        
        # Проверяем, что методы могут быть вызваны
        mock_storage.save_pattern.return_value = True
        mock_storage.get_patterns_by_symbol.return_value = []
        mock_storage.update_pattern_result.return_value = True
        mock_storage.get_successful_patterns.return_value = []
        mock_storage.find_similar_patterns.return_value = []
        mock_storage.get_storage_statistics.return_value = Mock()
        mock_storage.cleanup_old_data.return_value = 0
        mock_storage.backup_data.return_value = True
        mock_storage.validate_data_integrity.return_value = True
        mock_storage.get_pattern_metadata.return_value = []
        mock_storage.close.return_value = None
        
        # Проверяем вызовы
        assert mock_storage.save_pattern("BTCUSDT", Mock()) is True
        assert mock_storage.get_patterns_by_symbol("BTCUSDT") == []
        assert mock_storage.update_pattern_result("BTCUSDT", "id", Mock()) is True
        assert mock_storage.get_successful_patterns("BTCUSDT", 0.7) == []
        assert mock_storage.find_similar_patterns("BTCUSDT", {}, 0.8) == []
        assert mock_storage.get_storage_statistics() is not None
        assert mock_storage.cleanup_old_data("BTCUSDT", 1) == 0
        assert mock_storage.backup_data("BTCUSDT") is True
        assert mock_storage.validate_data_integrity("BTCUSDT") is True
        assert mock_storage.get_pattern_metadata("BTCUSDT") == []
        assert mock_storage.close() is None

    @pytest.mark.asyncio
    def test_async_interface_implementation(self: "TestIPatternStorage") -> None:
        """Тест асинхронной реализации интерфейса."""
        # Создаем асинхронный мок
        async_mock_storage = AsyncMock(spec=IPatternStorage)
        
        # Настраиваем возвращаемые значения
        async_mock_storage.save_pattern.return_value = True
        async_mock_storage.get_patterns_by_symbol.return_value = []
        async_mock_storage.update_pattern_result.return_value = True
        async_mock_storage.get_successful_patterns.return_value = []
        async_mock_storage.find_similar_patterns.return_value = []
        async_mock_storage.get_storage_statistics.return_value = Mock()
        async_mock_storage.cleanup_old_data.return_value = 0
        async_mock_storage.backup_data.return_value = True
        async_mock_storage.validate_data_integrity.return_value = True
        async_mock_storage.get_pattern_metadata.return_value = []
        async_mock_storage.close.return_value = None
        
        # Проверяем асинхронные вызовы
        result = await async_mock_storage.save_pattern("BTCUSDT", Mock())
        assert result is True
        
        patterns = await async_mock_storage.get_patterns_by_symbol("BTCUSDT")
        assert patterns == []
        
        success = await async_mock_storage.update_pattern_result("BTCUSDT", "id", Mock())
        assert success is True
        
        successful_patterns = await async_mock_storage.get_successful_patterns("BTCUSDT", 0.7)
        assert successful_patterns == []
        
        similar_patterns = await async_mock_storage.find_similar_patterns("BTCUSDT", {}, 0.8)
        assert similar_patterns == []
        
        stats = await async_mock_storage.get_storage_statistics()
        assert stats is not None
        
        cleaned = await async_mock_storage.cleanup_old_data("BTCUSDT", 1)
        assert cleaned == 0
        
        backup_success = await async_mock_storage.backup_data("BTCUSDT")
        assert backup_success is True
        
        integrity = await async_mock_storage.validate_data_integrity("BTCUSDT")
        assert integrity is True
        
        metadata = await async_mock_storage.get_pattern_metadata("BTCUSDT")
        assert metadata == []
        
        await async_mock_storage.close()


class TestIBehaviorHistoryStorage:
    """Тесты для интерфейса IBehaviorHistoryStorage."""

    def test_interface_definition(self: "TestIBehaviorHistoryStorage") -> None:
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(IBehaviorHistoryStorage, '__protocol_attrs__')
        
        # Проверяем наличие обязательных методов
        required_methods = [
            'save_behavior_record',
            'get_behavior_history',
            'get_statistics'
        ]
        
        for method in required_methods:
            assert hasattr(IBehaviorHistoryStorage, method), f"Метод {method} отсутствует в интерфейсе"

    def test_interface_method_signatures(self: "TestIBehaviorHistoryStorage") -> None:
        """Тест сигнатур методов интерфейса."""
        # Создаем мок, который реализует интерфейс
        mock_storage = Mock(spec=IBehaviorHistoryStorage)
        
        # Настраиваем возвращаемые значения
        mock_storage.save_behavior_record.return_value = True
        mock_storage.get_behavior_history.return_value = []
        mock_storage.get_statistics.return_value = {}
        
        # Проверяем вызовы
        assert mock_storage.save_behavior_record("BTCUSDT", {}) is True
        assert mock_storage.get_behavior_history("BTCUSDT", 1) == []
        assert mock_storage.get_statistics("BTCUSDT") == {}

    @pytest.mark.asyncio
    def test_async_interface_implementation(self: "TestIBehaviorHistoryStorage") -> None:
        """Тест асинхронной реализации интерфейса."""
        # Создаем асинхронный мок
        async_mock_storage = AsyncMock(spec=IBehaviorHistoryStorage)
        
        # Настраиваем возвращаемые значения
        async_mock_storage.save_behavior_record.return_value = True
        async_mock_storage.get_behavior_history.return_value = []
        async_mock_storage.get_statistics.return_value = {
            "total_records": 0,
            "avg_volume": 0.0,
            "avg_spread": 0.0,
            "avg_imbalance": 0.0,
            "avg_pressure": 0.0
        }
        
        # Проверяем асинхронные вызовы
        success = await async_mock_storage.save_behavior_record("BTCUSDT", {})
        assert success is True
        
        history = await async_mock_storage.get_behavior_history("BTCUSDT", 1)
        assert history == []
        
        stats = await async_mock_storage.get_statistics("BTCUSDT")
        assert isinstance(stats, dict)
        assert "total_records" in stats


class TestIPatternAnalyzer:
    """Тесты для интерфейса IPatternAnalyzer."""

    def test_interface_definition(self: "TestIPatternAnalyzer") -> None:
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(IPatternAnalyzer, '__protocol_attrs__')
        
        # Проверяем наличие обязательных методов
        required_methods = [
            'analyze_pattern',
            'analyze_market_context'
        ]
        
        for method in required_methods:
            assert hasattr(IPatternAnalyzer, method), f"Метод {method} отсутствует в интерфейсе"

    def test_interface_method_signatures(self: "TestIPatternAnalyzer") -> None:
        """Тест сигнатур методов интерфейса."""
        # Создаем мок, который реализует интерфейс
        mock_analyzer = Mock(spec=IPatternAnalyzer)
        
        # Настраиваем возвращаемые значения
        mock_analyzer.analyze_pattern.return_value = {
            "confidence": 0.8,
            "similarity_score": 0.85,
            "success_probability": 0.75,
            "market_context": {},
            "risk_assessment": {},
            "recommendations": []
        }
        mock_analyzer.analyze_market_context.return_value = {
            "market_phase": "trending",
            "volatility_regime": "medium",
            "liquidity_regime": "high"
        }
        
        # Проверяем вызовы
        analysis = mock_analyzer.analyze_pattern("BTCUSDT", Mock())
        assert isinstance(analysis, dict)
        assert "confidence" in analysis
        
        context = mock_analyzer.analyze_market_context("BTCUSDT", datetime.now())
        assert isinstance(context, dict)
        assert "market_phase" in context

    @pytest.mark.asyncio
    def test_async_interface_implementation(self: "TestIPatternAnalyzer") -> None:
        """Тест асинхронной реализации интерфейса."""
        # Создаем асинхронный мок
        async_mock_analyzer = AsyncMock(spec=IPatternAnalyzer)
        
        # Настраиваем возвращаемые значения
        async_mock_analyzer.analyze_pattern.return_value = {
            "confidence": 0.8,
            "similarity_score": 0.85,
            "success_probability": 0.75,
            "market_context": {
                "market_phase": "trending",
                "volatility_regime": "medium",
                "liquidity_regime": "high",
                "volume_profile": "normal",
                "price_action": "trending",
                "order_flow": "positive"
            },
            "risk_assessment": {
                "risk_level": "medium",
                "risk_factors": ["high_volatility", "low_liquidity"],
                "risk_score": 0.6
            },
            "recommendations": [
                "Увеличить уверенность",
                "Снизить риск"
            ]
        }
        async_mock_analyzer.analyze_market_context.return_value = {
            "market_phase": "trending",
            "volatility_regime": "medium",
            "liquidity_regime": "high",
            "volume_profile": "normal",
            "price_action": "trending",
            "order_flow": "positive"
        }
        
        # Проверяем асинхронные вызовы
        analysis = await async_mock_analyzer.analyze_pattern("BTCUSDT", Mock())
        assert isinstance(analysis, dict)
        assert "confidence" in analysis
        assert "similarity_score" in analysis
        assert "success_probability" in analysis
        assert "market_context" in analysis
        assert "risk_assessment" in analysis
        assert "recommendations" in analysis
        
        context = await async_mock_analyzer.analyze_market_context("BTCUSDT", datetime.now())
        assert isinstance(context, dict)
        assert "market_phase" in context
        assert "volatility_regime" in context
        assert "liquidity_regime" in context


class TestISimilarityCalculator:
    """Тесты для интерфейса ISimilarityCalculator."""

    def test_interface_definition(self: "TestISimilarityCalculator") -> None:
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(ISimilarityCalculator, '__protocol_attrs__')
        
        # Проверяем наличие обязательных методов
        required_methods = [
            'calculate_similarity'
        ]
        
        for method in required_methods:
            assert hasattr(ISimilarityCalculator, method), f"Метод {method} отсутствует в интерфейсе"

    def test_interface_method_signatures(self: "TestISimilarityCalculator") -> None:
        """Тест сигнатур методов интерфейса."""
        # Создаем мок, который реализует интерфейс
        mock_calculator = Mock(spec=ISimilarityCalculator)
        
        # Настраиваем возвращаемые значения
        mock_calculator.calculate_similarity.return_value = 0.85
        
        # Проверяем вызовы
        similarity = mock_calculator.calculate_similarity({}, {})
        assert similarity == 0.85

    @pytest.mark.asyncio
    def test_async_interface_implementation(self: "TestISimilarityCalculator") -> None:
        """Тест асинхронной реализации интерфейса."""
        # Создаем асинхронный мок
        async_mock_calculator = AsyncMock(spec=ISimilarityCalculator)
        
        # Настраиваем возвращаемые значения
        async_mock_calculator.calculate_similarity.return_value = 0.85
        
        # Проверяем асинхронные вызовы
        similarity = await async_mock_calculator.calculate_similarity({}, {})
        assert similarity == 0.85
        
        # Тест с весами
        similarity_with_weights = await async_mock_calculator.calculate_similarity({}, {}, {})
        assert similarity_with_weights == 0.85


class TestISuccessRateAnalyzer:
    """Тесты для интерфейса ISuccessRateAnalyzer."""

    def test_interface_definition(self: "TestISuccessRateAnalyzer") -> None:
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(ISuccessRateAnalyzer, '__protocol_attrs__')
        
        # Проверяем наличие обязательных методов
        required_methods = [
            'calculate_success_rate',
            'analyze_success_trends',
            'calculate_accuracy_metrics',
            'calculate_return_metrics',
            'generate_recommendations'
        ]
        
        for method in required_methods:
            assert hasattr(ISuccessRateAnalyzer, method), f"Метод {method} отсутствует в интерфейсе"

    def test_interface_method_signatures(self: "TestISuccessRateAnalyzer") -> None:
        """Тест сигнатур методов интерфейса."""
        # Создаем мок, который реализует интерфейс
        mock_analyzer = Mock(spec=ISuccessRateAnalyzer)
        
        # Настраиваем возвращаемые значения
        mock_analyzer.calculate_success_rate.return_value = 0.75
        mock_analyzer.analyze_success_trends.return_value = {
            "trend_direction": "up",
            "trend_strength": 0.7,
            "confidence": 0.8,
            "periods": 5
        }
        mock_analyzer.calculate_accuracy_metrics.return_value = {
            "avg_accuracy": 0.8,
            "accuracy_std": 0.1,
            "min_accuracy": 0.6,
            "max_accuracy": 0.9,
            "accuracy_trend": "stable"
        }
        mock_analyzer.calculate_return_metrics.return_value = {
            "avg_return": 0.02,
            "return_std": 0.01,
            "min_return": 0.01,
            "max_return": 0.03,
            "return_trend": "positive"
        }
        mock_analyzer.generate_recommendations.return_value = [
            "Увеличить уверенность",
            "Снизить риск"
        ]
        
        # Проверяем вызовы
        success_rate = mock_analyzer.calculate_success_rate("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert success_rate == 0.75
        
        trends = mock_analyzer.analyze_success_trends("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(trends, dict)
        assert "trend_direction" in trends
        
        accuracy_metrics = mock_analyzer.calculate_accuracy_metrics("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(accuracy_metrics, dict)
        assert "avg_accuracy" in accuracy_metrics
        
        return_metrics = mock_analyzer.calculate_return_metrics("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(return_metrics, dict)
        assert "avg_return" in return_metrics
        
        recommendations = mock_analyzer.generate_recommendations("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    def test_async_interface_implementation(self: "TestISuccessRateAnalyzer") -> None:
        """Тест асинхронной реализации интерфейса."""
        # Создаем асинхронный мок
        async_mock_analyzer = AsyncMock(spec=ISuccessRateAnalyzer)
        
        # Настраиваем возвращаемые значения
        async_mock_analyzer.calculate_success_rate.return_value = 0.75
        async_mock_analyzer.analyze_success_trends.return_value = {
            "trend_direction": "up",
            "trend_strength": 0.7,
            "confidence": 0.8,
            "periods": 5
        }
        async_mock_analyzer.calculate_accuracy_metrics.return_value = {
            "avg_accuracy": 0.8,
            "accuracy_std": 0.1,
            "min_accuracy": 0.6,
            "max_accuracy": 0.9,
            "accuracy_trend": "stable"
        }
        async_mock_analyzer.calculate_return_metrics.return_value = {
            "avg_return": 0.02,
            "return_std": 0.01,
            "min_return": 0.01,
            "max_return": 0.03,
            "return_trend": "positive"
        }
        async_mock_analyzer.generate_recommendations.return_value = [
            "Увеличить уверенность для паттернов накопления",
            "Снизить риск для паттернов выхода",
            "Оптимизировать параметры для паттернов поглощения"
        ]
        
        # Проверяем асинхронные вызовы
        success_rate = await async_mock_analyzer.calculate_success_rate("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert success_rate == 0.75
        
        trends = await async_mock_analyzer.analyze_success_trends("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(trends, dict)
        assert "trend_direction" in trends
        assert "trend_strength" in trends
        assert "confidence" in trends
        assert "periods" in trends
        
        accuracy_metrics = await async_mock_analyzer.calculate_accuracy_metrics("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(accuracy_metrics, dict)
        assert "avg_accuracy" in accuracy_metrics
        assert "accuracy_std" in accuracy_metrics
        assert "min_accuracy" in accuracy_metrics
        assert "max_accuracy" in accuracy_metrics
        assert "accuracy_trend" in accuracy_metrics
        
        return_metrics = await async_mock_analyzer.calculate_return_metrics("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(return_metrics, dict)
        assert "avg_return" in return_metrics
        assert "return_std" in return_metrics
        assert "min_return" in return_metrics
        assert "max_return" in return_metrics
        assert "return_trend" in return_metrics
        
        recommendations = await async_mock_analyzer.generate_recommendations("BTCUSDT", MarketMakerPatternType.ACCUMULATION, [])
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestInterfaceCompatibility:
    """Тесты совместимости интерфейсов."""

    def test_storage_interface_compatibility(self: "TestInterfaceCompatibility") -> None:
        """Тест совместимости интерфейса хранилища."""
        # Создаем мок, который должен быть совместим с интерфейсом
        mock_storage = Mock()
        
        # Добавляем методы интерфейса
        mock_storage.save_pattern = AsyncMock(return_value=True)
        mock_storage.get_patterns_by_symbol = AsyncMock(return_value=[])
        mock_storage.update_pattern_result = AsyncMock(return_value=True)
        mock_storage.get_successful_patterns = AsyncMock(return_value=[])
        mock_storage.find_similar_patterns = AsyncMock(return_value=[])
        mock_storage.get_storage_statistics = AsyncMock(return_value=Mock())
        mock_storage.cleanup_old_data = AsyncMock(return_value=0)
        mock_storage.backup_data = AsyncMock(return_value=True)
        mock_storage.validate_data_integrity = AsyncMock(return_value=True)
        mock_storage.get_pattern_metadata = AsyncMock(return_value=[])
        mock_storage.close = AsyncMock(return_value=None)
        
        # Проверяем, что мок совместим с интерфейсом
        assert isinstance(mock_storage, IPatternStorage)

    def test_analyzer_interface_compatibility(self: "TestInterfaceCompatibility") -> None:
        """Тест совместимости интерфейса анализатора."""
        # Создаем мок, который должен быть совместим с интерфейсом
        mock_analyzer = Mock()
        
        # Добавляем методы интерфейса
        mock_analyzer.analyze_pattern = AsyncMock(return_value={})
        mock_analyzer.analyze_market_context = AsyncMock(return_value={})
        
        # Проверяем, что мок совместим с интерфейсом
        assert isinstance(mock_analyzer, IPatternAnalyzer)

    def test_calculator_interface_compatibility(self: "TestInterfaceCompatibility") -> None:
        """Тест совместимости интерфейса калькулятора."""
        # Создаем мок, который должен быть совместим с интерфейсом
        mock_calculator = Mock()
        
        # Добавляем методы интерфейса
        mock_calculator.calculate_similarity = AsyncMock(return_value=0.85)
        
        # Проверяем, что мок совместим с интерфейсом
        assert isinstance(mock_calculator, ISimilarityCalculator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
