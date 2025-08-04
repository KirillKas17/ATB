"""
Unit тесты для SignalService.

Покрывает:
- Основной функционал работы с сигналами
- Генерацию сигналов
- Валидацию сигналов
- Агрегацию сигналов
- Анализ сигналов
- Обработку ошибок
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from domain.services.signal_service import (
    SignalService,
    DefaultSignalService,
    SignalGenerationContext,
    AggregationWeights,
    AsyncSignalGenerator
)
from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import Strategy
from domain.type_definitions.signal_types import (
    SignalValidationResult,
    SignalAggregationResult,
    SignalAnalysisResult
)
from domain.type_definitions import MarketDataFrame


class TestSignalGenerationContext:
    """Тесты для SignalGenerationContext."""
    
    def test_creation(self):
        """Тест создания контекста генерации сигналов."""
        strategy = Mock(spec=Strategy)
        market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        context = SignalGenerationContext(
            strategy=strategy,
            market_data=market_data,
            current_price=100.0,
            volume=1000.0,
            volatility=0.02,
            trend_direction="up",
            support_level=95.0,
            resistance_level=105.0
        )
        
        assert context.strategy == strategy
        assert context.market_data.equals(market_data)
        assert context.current_price == 100.0
        assert context.volume == 1000.0
        assert context.volatility == 0.02
        assert context.trend_direction == "up"
        assert context.support_level == 95.0
        assert context.resistance_level == 105.0


class TestAggregationWeights:
    """Тесты для AggregationWeights."""
    
    def test_creation_default(self):
        """Тест создания с значениями по умолчанию."""
        weights = AggregationWeights()
        
        assert weights.confidence_weight == 0.4
        assert weights.strength_weight == 0.3
        assert weights.recency_weight == 0.2
        assert weights.volume_weight == 0.1
        assert weights.min_confidence == 0.3
        assert weights.max_signals == 10
    
    def test_creation_custom(self):
        """Тест создания с пользовательскими значениями."""
        weights = AggregationWeights(
            confidence_weight=0.5,
            strength_weight=0.4,
            recency_weight=0.1,
            volume_weight=0.0,
            min_confidence=0.5,
            max_signals=5
        )
        
        assert weights.confidence_weight == 0.5
        assert weights.strength_weight == 0.4
        assert weights.recency_weight == 0.1
        assert weights.volume_weight == 0.0
        assert weights.min_confidence == 0.5
        assert weights.max_signals == 5


class TestDefaultSignalService:
    """Тесты для DefaultSignalService."""
    
    @pytest.fixture
    def strategy(self) -> Strategy:
        """Тестовая стратегия."""
        strategy = Mock(spec=Strategy)
        strategy.name = "test_strategy"
        strategy.type = "trend_following"
        strategy.parameters = {"window": 20, "threshold": 0.02}
        return strategy
    
    @pytest.fixture
    def market_data(self) -> MarketDataFrame:
        """Тестовые рыночные данные."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 300, 100),
            'low': np.random.uniform(50, 100, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Создаем тренд
        data['close'] = data['close'] + np.arange(100) * 0.1
        return data
    
    @pytest.fixture
    def service(self) -> DefaultSignalService:
        """Экземпляр DefaultSignalService."""
        return DefaultSignalService()
    
    def test_creation(self):
        """Тест создания сервиса."""
        service = DefaultSignalService()
        
        assert service._signal_generators is not None
        assert len(service._signal_generators) > 0
        assert isinstance(service._aggregation_weights, AggregationWeights)
    
    def test_setup_signal_generators(self, service):
        """Тест настройки генераторов сигналов."""
        generators = service._setup_signal_generators()
        
        expected_generators = [
            "trend_following", "mean_reversion", "breakout",
            "scalping", "arbitrage", "grid", "momentum", "volatility"
        ]
        
        for generator_name in expected_generators:
            assert generator_name in generators
            assert callable(generators[generator_name])
    
    @pytest.mark.asyncio
    async def test_generate_signals_trend_following(self, service, strategy, market_data):
        """Тест генерации трендовых сигналов."""
        strategy.type = "trend_following"
        
        signals = await service.generate_signals(strategy, market_data)
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.strategy_name == strategy.name
    
    @pytest.mark.asyncio
    async def test_generate_signals_mean_reversion(self, service, strategy, market_data):
        """Тест генерации сигналов возврата к среднему."""
        strategy.type = "mean_reversion"
        
        signals = await service.generate_signals(strategy, market_data)
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.strategy_name == strategy.name
    
    @pytest.mark.asyncio
    async def test_generate_signals_breakout(self, service, strategy, market_data):
        """Тест генерации сигналов пробоя."""
        strategy.type = "breakout"
        
        signals = await service.generate_signals(strategy, market_data)
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.strategy_name == strategy.name
    
    @pytest.mark.asyncio
    async def test_generate_signals_unknown_type(self, service, strategy, market_data):
        """Тест генерации сигналов с неизвестным типом стратегии."""
        strategy.type = "unknown_type"
        
        signals = await service.generate_signals(strategy, market_data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    def test_calculate_trend_direction(self, service):
        """Тест расчета направления тренда."""
        # Восходящий тренд
        up_trend = [100, 101, 102, 103, 104]
        direction = service._calculate_trend_direction(up_trend)
        assert direction == "up"
        
        # Нисходящий тренд
        down_trend = [104, 103, 102, 101, 100]
        direction = service._calculate_trend_direction(down_trend)
        assert direction == "down"
        
        # Боковой тренд
        sideways_trend = [100, 101, 100, 101, 100]
        direction = service._calculate_trend_direction(sideways_trend)
        assert direction == "sideways"
    
    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, service):
        """Тест валидации валидного сигнала."""
        signal = Signal(
            id="test_signal",
            strategy_name="test_strategy",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            price=100.0,
            timestamp=datetime.now(),
            metadata={"reason": "trend_following"}
        )
        
        result = await service.validate_signal(signal)
        
        assert isinstance(result, SignalValidationResult)
        assert result.is_valid is True
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid(self, service):
        """Тест валидации невалидного сигнала."""
        signal = Signal(
            id="test_signal",
            strategy_name="test_strategy",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.1,  # Низкая уверенность
            price=100.0,
            timestamp=datetime.now(),
            metadata={"reason": "trend_following"}
        )
        
        result = await service.validate_signal(signal)
        
        assert isinstance(result, SignalValidationResult)
        assert result.is_valid is False
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_empty(self, service):
        """Тест агрегации пустого списка сигналов."""
        result = await service.aggregate_signals([])
        
        assert isinstance(result, SignalAggregationResult)
        assert result.aggregated_signal is None
        assert result.confidence_score == 0.0
        assert result.signal_count == 0
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_single(self, service):
        """Тест агрегации одного сигнала."""
        signal = Signal(
            id="test_signal",
            strategy_name="test_strategy",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            price=100.0,
            timestamp=datetime.now(),
            metadata={"reason": "trend_following"}
        )
        
        result = await service.aggregate_signals([signal])
        
        assert isinstance(result, SignalAggregationResult)
        assert result.aggregated_signal is not None
        assert result.confidence_score == 0.8
        assert result.signal_count == 1
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_multiple(self, service):
        """Тест агрегации нескольких сигналов."""
        signals = [
            Signal(
                id=f"signal_{i}",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                price=100.0 + i,
                timestamp=datetime.now(),
                metadata={"reason": "trend_following"}
            )
            for i in range(3)
        ]
        
        result = await service.aggregate_signals(signals)
        
        assert isinstance(result, SignalAggregationResult)
        assert result.aggregated_signal is not None
        assert result.confidence_score > 0.0
        assert result.signal_count == 3
    
    @pytest.mark.asyncio
    async def test_analyze_signals_empty(self, service):
        """Тест анализа пустого списка сигналов."""
        period = timedelta(days=1)
        result = await service.analyze_signals([], period)
        
        assert isinstance(result, SignalAnalysisResult)
        assert result.total_signals == 0
        assert result.success_rate == 0.0
        assert result.average_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_signals_success(self, service):
        """Тест успешного анализа сигналов."""
        signals = [
            Signal(
                id=f"signal_{i}",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                price=100.0 + i,
                timestamp=datetime.now(),
                metadata={"reason": "trend_following"}
            )
            for i in range(5)
        ]
        
        period = timedelta(days=1)
        result = await service.analyze_signals(signals, period)
        
        assert isinstance(result, SignalAnalysisResult)
        assert result.total_signals == 5
        assert result.average_confidence == 0.8
        assert isinstance(result.signal_distribution, dict)
        assert isinstance(result.recommendations, list)
    
    def test_get_strength_value(self, service):
        """Тест получения числового значения силы сигнала."""
        assert service._get_strength_value(SignalStrength.WEAK) == 0.3
        assert service._get_strength_value(SignalStrength.MEDIUM) == 0.6
        assert service._get_strength_value(SignalStrength.STRONG) == 1.0
    
    def test_determine_aggregated_strength(self, service):
        """Тест определения агрегированной силы сигнала."""
        signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.WEAK,
                confidence=0.8,
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_2",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                price=101.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        strength = service._determine_aggregated_strength(signals)
        
        assert isinstance(strength, SignalStrength)
        assert strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]
    
    def test_determine_aggregated_type(self, service):
        """Тест определения агрегированного типа сигнала."""
        buy_signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_2",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                price=101.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        signal_type = service._determine_aggregated_type(buy_signals)
        
        assert signal_type == SignalType.BUY
        
        # Смешанные сигналы
        mixed_signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_2",
                strategy_name="test_strategy",
                signal_type=SignalType.SELL,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                price=101.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        signal_type = service._determine_aggregated_type(mixed_signals)
        
        assert signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    
    def test_get_strength_distribution(self, service):
        """Тест получения распределения силы сигналов."""
        signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.WEAK,
                confidence=0.8,
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_2",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                price=101.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_3",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.7,
                price=102.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        distribution = service._get_strength_distribution(signals)
        
        assert isinstance(distribution, dict)
        assert "WEAK" in distribution
        assert "STRONG" in distribution
        assert distribution["WEAK"] == 1
        assert distribution["STRONG"] == 2
    
    def test_validate_signal_data(self, service):
        """Тест валидации данных сигнала."""
        valid_data = {
            "signal_type": "BUY",
            "strength": "STRONG",
            "confidence": 0.8,
            "price": 100.0
        }
        
        assert service._validate_signal_data(valid_data) is True
        
        invalid_data = {
            "signal_type": "INVALID",
            "strength": "INVALID",
            "confidence": 1.5,  # > 1.0
            "price": -100.0  # < 0
        }
        
        assert service._validate_signal_data(invalid_data) is False
    
    def test_process_signal(self, service):
        """Тест обработки сигнала."""
        signal = Signal(
            id="test_signal",
            strategy_name="test_strategy",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            price=100.0,
            timestamp=datetime.now(),
            metadata={"reason": "trend_following"}
        )
        
        processed = service._process_signal(signal)
        
        assert isinstance(processed, dict)
        assert "signal_type" in processed
        assert "strength" in processed
        assert "confidence" in processed
        assert "price" in processed
    
    @pytest.mark.asyncio
    async def test_filter_signals(self, service, strategy):
        """Тест фильтрации сигналов."""
        signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.WEAK,
                confidence=0.2,  # Низкая уверенность
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                id="signal_2",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.9,  # Высокая уверенность
                price=101.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        filtered = await service._filter_signals(signals, strategy)
        
        assert isinstance(filtered, list)
        assert len(filtered) <= len(signals)
        # Сигнал с низкой уверенностью должен быть отфильтрован
        assert all(signal.confidence >= 0.3 for signal in filtered)
    
    def test_generate_signal_recommendations(self, service):
        """Тест генерации рекомендаций по сигналам."""
        signals = [
            Signal(
                id="signal_1",
                strategy_name="test_strategy",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                price=100.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        distribution = {"STRONG": 1, "WEAK": 0}
        avg_confidence = 0.8
        
        recommendations = service._generate_signal_recommendations(
            signals, distribution, avg_confidence
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_strategy(self, service, market_data):
        """Тест обработки ошибок с невалидной стратегией."""
        invalid_strategy = None
        
        signals = await service.generate_signals(invalid_strategy, market_data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_market_data(self, service, strategy):
        """Тест обработки ошибок с невалидными рыночными данными."""
        invalid_market_data = None
        
        signals = await service.generate_signals(strategy, invalid_market_data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_empty_market_data(self, service, strategy):
        """Тест обработки ошибок с пустыми рыночными данными."""
        empty_market_data = pd.DataFrame()
        
        signals = await service.generate_signals(strategy, empty_market_data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, service, strategy, market_data):
        """Тест конкурентной генерации сигналов."""
        import asyncio
        
        async def generate_signals_batch():
            return await service.generate_signals(strategy, market_data)
        
        # Запускаем несколько задач одновременно
        tasks = [generate_signals_batch() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            assert isinstance(result, list)
            for signal in result:
                assert isinstance(signal, Signal)
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_different_strategies(self, service, market_data):
        """Тест генерации сигналов с разными типами стратегий."""
        strategy_types = [
            "trend_following", "mean_reversion", "breakout",
            "scalping", "arbitrage", "grid", "momentum", "volatility"
        ]
        
        for strategy_type in strategy_types:
            strategy = Mock(spec=Strategy)
            strategy.name = f"test_{strategy_type}"
            strategy.type = strategy_type
            strategy.parameters = {"window": 20, "threshold": 0.02}
            
            signals = await service.generate_signals(strategy, market_data)
            
            assert isinstance(signals, list)
            for signal in signals:
                assert isinstance(signal, Signal)
                assert signal.strategy_name == strategy.name 