"""
Unit тесты для mm_pattern.py.

Покрывает:
- PatternFeatures - признаки паттернов
- PatternResult - результаты паттернов  
- MarketMakerPattern - основные паттерны
- PatternMemory - память паттернов
- MatchedPattern - совпадения паттернов
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from domain.market_maker.mm_pattern import (
    PatternFeatures,
    PatternResult,
    MarketMakerPattern,
    PatternMemory,
    MatchedPattern,
    _empty_market_microstructure,
    _empty_pattern_context
)
from domain.types.market_maker_types import (
    MarketMakerPatternType,
    PatternOutcome,
    PatternConfidence,
    MarketPhase,
    BookPressure,
    VolumeDelta,
    PriceReaction,
    SpreadChange,
    OrderImbalance,
    LiquidityDepth,
    TimeDuration,
    VolumeConcentration,
    PriceVolatility,
    Confidence,
    Accuracy,
    AverageReturn,
    SuccessCount,
    TotalCount,
    SimilarityScore,
    SignalStrength
)
from domain.exceptions.base_exceptions import ValidationError


class TestPatternFeatures:
    """Тесты для PatternFeatures."""
    
    @pytest.fixture
    def sample_features_data(self) -> Dict[str, Any]:
        """Тестовые данные для PatternFeatures."""
        return {
            "book_pressure": BookPressure(0.5),
            "volume_delta": VolumeDelta(0.3),
            "price_reaction": PriceReaction(0.2),
            "spread_change": SpreadChange(-0.1),
            "order_imbalance": OrderImbalance(0.4),
            "liquidity_depth": LiquidityDepth(1000.0),
            "time_duration": TimeDuration(300),
            "volume_concentration": VolumeConcentration(0.6),
            "price_volatility": PriceVolatility(0.15),
            "market_microstructure": {
                "avg_trade_size": 100.0,
                "buy_sell_ratio": 1.2
            }
        }
    
    @pytest.fixture
    def pattern_features(self, sample_features_data) -> PatternFeatures:
        """Создает экземпляр PatternFeatures."""
        return PatternFeatures(**sample_features_data)
    
    def test_creation_with_valid_data(self, sample_features_data):
        """Тест создания с валидными данными."""
        features = PatternFeatures(**sample_features_data)
        
        assert features.book_pressure == sample_features_data["book_pressure"]
        assert features.volume_delta == sample_features_data["volume_delta"]
        assert features.price_reaction == sample_features_data["price_reaction"]
        assert features.spread_change == sample_features_data["spread_change"]
        assert features.order_imbalance == sample_features_data["order_imbalance"]
        assert features.liquidity_depth == sample_features_data["liquidity_depth"]
        assert features.time_duration == sample_features_data["time_duration"]
        assert features.volume_concentration == sample_features_data["volume_concentration"]
        assert features.price_volatility == sample_features_data["price_volatility"]
        assert features.market_microstructure == sample_features_data["market_microstructure"]
    
    def test_creation_with_defaults(self):
        """Тест создания с значениями по умолчанию."""
        features = PatternFeatures(
            book_pressure=BookPressure(0.1),
            volume_delta=VolumeDelta(0.1),
            price_reaction=PriceReaction(0.1),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.1),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        assert features.market_microstructure == _empty_market_microstructure()
    
    def test_validation_negative_time_duration(self):
        """Тест валидации отрицательного времени."""
        with pytest.raises(ValueError):
            PatternFeatures(
                book_pressure=BookPressure(0.1),
                volume_delta=VolumeDelta(0.1),
                price_reaction=PriceReaction(0.1),
                spread_change=SpreadChange(0.1),
                order_imbalance=OrderImbalance(0.1),
                liquidity_depth=LiquidityDepth(100.0),
                time_duration=TimeDuration(-1),  # Отрицательное время
                volume_concentration=VolumeConcentration(0.1),
                price_volatility=PriceVolatility(0.1)
            )
    
    def test_to_dict(self, pattern_features):
        """Тест сериализации в словарь."""
        result = pattern_features.to_dict()
        
        assert result["book_pressure"] == float(pattern_features.book_pressure)
        assert result["volume_delta"] == float(pattern_features.volume_delta)
        assert result["price_reaction"] == float(pattern_features.price_reaction)
        assert result["spread_change"] == float(pattern_features.spread_change)
        assert result["order_imbalance"] == float(pattern_features.order_imbalance)
        assert result["liquidity_depth"] == float(pattern_features.liquidity_depth)
        assert result["time_duration"] == int(pattern_features.time_duration)
        assert result["volume_concentration"] == float(pattern_features.volume_concentration)
        assert result["price_volatility"] == float(pattern_features.price_volatility)
        assert result["market_microstructure"] == pattern_features.market_microstructure
    
    def test_from_dict(self, sample_features_data):
        """Тест десериализации из словаря."""
        data_dict = {
            "book_pressure": float(sample_features_data["book_pressure"]),
            "volume_delta": float(sample_features_data["volume_delta"]),
            "price_reaction": float(sample_features_data["price_reaction"]),
            "spread_change": float(sample_features_data["spread_change"]),
            "order_imbalance": float(sample_features_data["order_imbalance"]),
            "liquidity_depth": float(sample_features_data["liquidity_depth"]),
            "time_duration": int(sample_features_data["time_duration"]),
            "volume_concentration": float(sample_features_data["volume_concentration"]),
            "price_volatility": float(sample_features_data["price_volatility"]),
            "market_microstructure": sample_features_data["market_microstructure"]
        }
        
        features = PatternFeatures.from_dict(data_dict)
        
        assert features.book_pressure == sample_features_data["book_pressure"]
        assert features.volume_delta == sample_features_data["volume_delta"]
        assert features.price_reaction == sample_features_data["price_reaction"]
        assert features.spread_change == sample_features_data["spread_change"]
        assert features.order_imbalance == sample_features_data["order_imbalance"]
        assert features.liquidity_depth == sample_features_data["liquidity_depth"]
        assert features.time_duration == sample_features_data["time_duration"]
        assert features.volume_concentration == sample_features_data["volume_concentration"]
        assert features.price_volatility == sample_features_data["price_volatility"]
        assert features.market_microstructure == sample_features_data["market_microstructure"]
    
    def test_get_overall_strength(self, pattern_features):
        """Тест расчета общей силы паттерна."""
        strength = pattern_features.get_overall_strength()
        
        assert 0.0 <= strength <= 1.0
        assert isinstance(strength, float)
    
    def test_get_direction_bias_bullish(self):
        """Тест расчета направленного смещения (бычий)."""
        features = PatternFeatures(
            book_pressure=BookPressure(0.5),
            volume_delta=VolumeDelta(0.3),
            price_reaction=PriceReaction(0.4),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        bias = features.get_direction_bias()
        assert bias > 0.1  # Бычий паттерн
        assert -1.0 <= bias <= 1.0
    
    def test_get_direction_bias_bearish(self):
        """Тест расчета направленного смещения (медвежий)."""
        features = PatternFeatures(
            book_pressure=BookPressure(-0.5),
            volume_delta=VolumeDelta(-0.3),
            price_reaction=PriceReaction(-0.4),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(-0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        bias = features.get_direction_bias()
        assert bias < -0.1  # Медвежий паттерн
        assert -1.0 <= bias <= 1.0


class TestPatternResult:
    """Тесты для PatternResult."""
    
    @pytest.fixture
    def sample_result_data(self) -> Dict[str, Any]:
        """Тестовые данные для PatternResult."""
        return {
            "outcome": PatternOutcome.SUCCESS,
            "price_change_5min": 0.02,
            "price_change_15min": 0.05,
            "price_change_30min": 0.08,
            "volume_change": 0.15,
            "volatility_change": 0.1,
            "market_context": {
                "symbol": "BTC/USDT",
                "last_price": 50000.0
            }
        }
    
    @pytest.fixture
    def pattern_result(self, sample_result_data) -> PatternResult:
        """Создает экземпляр PatternResult."""
        return PatternResult(**sample_result_data)
    
    def test_creation_with_valid_data(self, sample_result_data):
        """Тест создания с валидными данными."""
        result = PatternResult(**sample_result_data)
        
        assert result.outcome == sample_result_data["outcome"]
        assert result.price_change_5min == sample_result_data["price_change_5min"]
        assert result.price_change_15min == sample_result_data["price_change_15min"]
        assert result.price_change_30min == sample_result_data["price_change_30min"]
        assert result.volume_change == sample_result_data["volume_change"]
        assert result.volatility_change == sample_result_data["volatility_change"]
        assert result.market_context == sample_result_data["market_context"]
    
    def test_creation_with_defaults(self):
        """Тест создания с значениями по умолчанию."""
        result = PatternResult(
            outcome=PatternOutcome.NEUTRAL,
            price_change_5min=0.0,
            price_change_15min=0.0,
            price_change_30min=0.0,
            volume_change=0.0,
            volatility_change=0.0
        )
        
        assert result.market_context == _empty_pattern_context()
    
    def test_to_dict(self, pattern_result):
        """Тест сериализации в словарь."""
        result = pattern_result.to_dict()
        
        assert result["outcome"] == pattern_result.outcome.value
        assert result["price_change_5min"] == pattern_result.price_change_5min
        assert result["price_change_15min"] == pattern_result.price_change_15min
        assert result["price_change_30min"] == pattern_result.price_change_30min
        assert result["volume_change"] == pattern_result.volume_change
        assert result["volatility_change"] == pattern_result.volatility_change
        assert result["market_context"] == pattern_result.market_context
    
    def test_from_dict(self, sample_result_data):
        """Тест десериализации из словаря."""
        data_dict = {
            "outcome": sample_result_data["outcome"].value,
            "price_change_5min": sample_result_data["price_change_5min"],
            "price_change_15min": sample_result_data["price_change_15min"],
            "price_change_30min": sample_result_data["price_change_30min"],
            "volume_change": sample_result_data["volume_change"],
            "volatility_change": sample_result_data["volatility_change"],
            "market_context": sample_result_data["market_context"]
        }
        
        result = PatternResult.from_dict(data_dict)
        
        assert result.outcome == sample_result_data["outcome"]
        assert result.price_change_5min == sample_result_data["price_change_5min"]
        assert result.price_change_15min == sample_result_data["price_change_15min"]
        assert result.price_change_30min == sample_result_data["price_change_30min"]
        assert result.volume_change == sample_result_data["volume_change"]
        assert result.volatility_change == sample_result_data["volatility_change"]
        assert result.market_context == sample_result_data["market_context"]
    
    def test_get_expected_return_5min(self, pattern_result):
        """Тест получения ожидаемой доходности для 5 минут."""
        expected = pattern_result.get_expected_return("5min")
        assert expected == pattern_result.price_change_5min
    
    def test_get_expected_return_15min(self, pattern_result):
        """Тест получения ожидаемой доходности для 15 минут."""
        expected = pattern_result.get_expected_return("15min")
        assert expected == pattern_result.price_change_15min
    
    def test_get_expected_return_30min(self, pattern_result):
        """Тест получения ожидаемой доходности для 30 минут."""
        expected = pattern_result.get_expected_return("30min")
        assert expected == pattern_result.price_change_30min
    
    def test_get_expected_return_default(self, pattern_result):
        """Тест получения ожидаемой доходности по умолчанию."""
        expected = pattern_result.get_expected_return("unknown")
        assert expected == pattern_result.price_change_15min
    
    def test_is_successful_true(self, pattern_result):
        """Тест проверки успешности (успешный)."""
        assert pattern_result.is_successful(0.01) is True
    
    def test_is_successful_false(self):
        """Тест проверки успешности (неуспешный)."""
        result = PatternResult(
            outcome=PatternOutcome.FAILURE,
            price_change_5min=0.001,
            price_change_15min=0.001,
            price_change_30min=0.001,
            volume_change=0.0,
            volatility_change=0.0
        )
        assert result.is_successful(0.01) is False
    
    def test_get_risk_reward_ratio(self, pattern_result):
        """Тест расчета соотношения риск/доходность."""
        ratio = pattern_result.get_risk_reward_ratio()
        assert ratio > 0
        assert ratio == abs(pattern_result.price_change_15min) / abs(pattern_result.volatility_change)
    
    def test_get_risk_reward_ratio_zero_volatility(self):
        """Тест расчета соотношения риск/доходность при нулевой волатильности."""
        result = PatternResult(
            outcome=PatternOutcome.NEUTRAL,
            price_change_5min=0.0,
            price_change_15min=0.0,
            price_change_30min=0.0,
            volume_change=0.0,
            volatility_change=0.0
        )
        ratio = result.get_risk_reward_ratio()
        assert ratio == 0.0


class TestMarketMakerPattern:
    """Тесты для MarketMakerPattern."""
    
    @pytest.fixture
    def sample_pattern_data(self, sample_features_data) -> Dict[str, Any]:
        """Тестовые данные для MarketMakerPattern."""
        return {
            "pattern_type": MarketMakerPatternType.ACCUMULATION,
            "symbol": "BTC/USDT",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "features": PatternFeatures(**sample_features_data),
            "confidence": Confidence(0.8),
            "context": {
                "market_phase": "accumulation",
                "volatility_regime": "low"
            }
        }
    
    @pytest.fixture
    def market_maker_pattern(self, sample_pattern_data) -> MarketMakerPattern:
        """Создает экземпляр MarketMakerPattern."""
        return MarketMakerPattern(**sample_pattern_data)
    
    def test_creation_with_valid_data(self, sample_pattern_data):
        """Тест создания с валидными данными."""
        pattern = MarketMakerPattern(**sample_pattern_data)
        
        assert pattern.pattern_type == sample_pattern_data["pattern_type"]
        assert pattern.symbol == sample_pattern_data["symbol"]
        assert pattern.timestamp == sample_pattern_data["timestamp"]
        assert pattern.features == sample_pattern_data["features"]
        assert pattern.confidence == sample_pattern_data["confidence"]
        assert pattern.context == sample_pattern_data["context"]
    
    def test_creation_with_empty_symbol(self, sample_pattern_data):
        """Тест создания с пустым символом."""
        sample_pattern_data["symbol"] = ""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            MarketMakerPattern(**sample_pattern_data)
    
    def test_to_dict(self, market_maker_pattern):
        """Тест сериализации в словарь."""
        result = market_maker_pattern.to_dict()
        
        assert result["pattern_type"] == market_maker_pattern.pattern_type.value
        assert result["symbol"] == str(market_maker_pattern.symbol)
        assert result["timestamp"] == market_maker_pattern.timestamp.isoformat()
        assert result["features"] == market_maker_pattern.features.to_dict()
        assert result["confidence"] == float(market_maker_pattern.confidence)
        assert result["context"] == dict(market_maker_pattern.context)
    
    def test_from_dict(self, sample_pattern_data):
        """Тест десериализации из словаря."""
        data_dict = {
            "pattern_type": sample_pattern_data["pattern_type"].value,
            "symbol": str(sample_pattern_data["symbol"]),
            "timestamp": sample_pattern_data["timestamp"].isoformat(),
            "features": sample_pattern_data["features"].to_dict(),
            "confidence": float(sample_pattern_data["confidence"]),
            "context": dict(sample_pattern_data["context"])
        }
        
        pattern = MarketMakerPattern.from_dict(data_dict)
        
        assert pattern.pattern_type == sample_pattern_data["pattern_type"]
        assert pattern.symbol == sample_pattern_data["symbol"]
        assert pattern.timestamp == sample_pattern_data["timestamp"]
        assert pattern.features.to_dict() == sample_pattern_data["features"].to_dict()
        assert pattern.confidence == sample_pattern_data["confidence"]
        assert pattern.context == sample_pattern_data["context"]
    
    def test_get_pattern_strength(self, market_maker_pattern):
        """Тест расчета силы паттерна."""
        strength = market_maker_pattern.get_pattern_strength()
        
        assert 0.0 <= strength <= 1.0
        expected = market_maker_pattern.features.get_overall_strength() * float(market_maker_pattern.confidence)
        assert abs(strength - expected) < 0.001
    
    def test_get_direction_bullish(self, sample_pattern_data):
        """Тест определения направления (бычий)."""
        # Настраиваем признаки для бычьего паттерна
        sample_pattern_data["features"] = PatternFeatures(
            book_pressure=BookPressure(0.5),
            volume_delta=VolumeDelta(0.3),
            price_reaction=PriceReaction(0.4),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern = MarketMakerPattern(**sample_pattern_data)
        direction = pattern.get_direction()
        assert direction == "bullish"
    
    def test_get_direction_bearish(self, sample_pattern_data):
        """Тест определения направления (медвежий)."""
        # Настраиваем признаки для медвежьего паттерна
        sample_pattern_data["features"] = PatternFeatures(
            book_pressure=BookPressure(-0.5),
            volume_delta=VolumeDelta(-0.3),
            price_reaction=PriceReaction(-0.4),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(-0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern = MarketMakerPattern(**sample_pattern_data)
        direction = pattern.get_direction()
        assert direction == "bearish"
    
    def test_get_direction_neutral(self, sample_pattern_data):
        """Тест определения направления (нейтральный)."""
        # Настраиваем признаки для нейтрального паттерна
        sample_pattern_data["features"] = PatternFeatures(
            book_pressure=BookPressure(0.05),
            volume_delta=VolumeDelta(0.05),
            price_reaction=PriceReaction(0.05),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.05),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern = MarketMakerPattern(**sample_pattern_data)
        direction = pattern.get_direction()
        assert direction == "neutral"
    
    def test_get_confidence_level_very_high(self, sample_pattern_data):
        """Тест получения уровня уверенности (очень высокий)."""
        sample_pattern_data["confidence"] = Confidence(0.9)
        pattern = MarketMakerPattern(**sample_pattern_data)
        confidence_level = pattern.get_confidence_level()
        assert confidence_level == PatternConfidence.VERY_HIGH
    
    def test_get_confidence_level_high(self, sample_pattern_data):
        """Тест получения уровня уверенности (высокий)."""
        sample_pattern_data["confidence"] = Confidence(0.7)
        pattern = MarketMakerPattern(**sample_pattern_data)
        confidence_level = pattern.get_confidence_level()
        assert confidence_level == PatternConfidence.HIGH
    
    def test_get_confidence_level_medium(self, sample_pattern_data):
        """Тест получения уровня уверенности (средний)."""
        sample_pattern_data["confidence"] = Confidence(0.5)
        pattern = MarketMakerPattern(**sample_pattern_data)
        confidence_level = pattern.get_confidence_level()
        assert confidence_level == PatternConfidence.MEDIUM
    
    def test_get_confidence_level_low(self, sample_pattern_data):
        """Тест получения уровня уверенности (низкий)."""
        sample_pattern_data["confidence"] = Confidence(0.3)
        pattern = MarketMakerPattern(**sample_pattern_data)
        confidence_level = pattern.get_confidence_level()
        assert confidence_level == PatternConfidence.LOW
    
    def test_is_high_confidence_true(self, sample_pattern_data):
        """Тест проверки высокой уверенности (истина)."""
        sample_pattern_data["confidence"] = Confidence(0.8)
        pattern = MarketMakerPattern(**sample_pattern_data)
        assert pattern.is_high_confidence() is True
    
    def test_is_high_confidence_false(self, sample_pattern_data):
        """Тест проверки высокой уверенности (ложь)."""
        sample_pattern_data["confidence"] = Confidence(0.6)
        pattern = MarketMakerPattern(**sample_pattern_data)
        assert pattern.is_high_confidence() is False
    
    def test_get_market_phase_accumulation(self, sample_pattern_data):
        """Тест определения рыночной фазы (накопление)."""
        sample_pattern_data["pattern_type"] = MarketMakerPatternType.ACCUMULATION
        pattern = MarketMakerPattern(**sample_pattern_data)
        phase = pattern.get_market_phase()
        assert phase == MarketPhase.ACCUMULATION
    
    def test_get_market_phase_distribution(self, sample_pattern_data):
        """Тест определения рыночной фазы (распределение)."""
        sample_pattern_data["pattern_type"] = MarketMakerPatternType.EXIT
        pattern = MarketMakerPattern(**sample_pattern_data)
        phase = pattern.get_market_phase()
        assert phase == MarketPhase.DISTRIBUTION
    
    def test_get_market_phase_transition(self, sample_pattern_data):
        """Тест определения рыночной фазы (переход)."""
        sample_pattern_data["pattern_type"] = MarketMakerPatternType.SPOOFING
        pattern = MarketMakerPattern(**sample_pattern_data)
        phase = pattern.get_market_phase()
        assert phase == MarketPhase.TRANSITION


class TestPatternMemory:
    """Тесты для PatternMemory."""
    
    @pytest.fixture
    def sample_memory_data(self, sample_pattern_data) -> Dict[str, Any]:
        """Тестовые данные для PatternMemory."""
        return {
            "pattern": MarketMakerPattern(**sample_pattern_data),
            "result": None,
            "accuracy": Accuracy(0.0),
            "avg_return": AverageReturn(0.0),
            "success_count": SuccessCount(0),
            "total_count": TotalCount(0),
            "last_seen": None
        }
    
    @pytest.fixture
    def pattern_memory(self, sample_memory_data) -> PatternMemory:
        """Создает экземпляр PatternMemory."""
        return PatternMemory(**sample_memory_data)
    
    def test_creation_with_valid_data(self, sample_memory_data):
        """Тест создания с валидными данными."""
        memory = PatternMemory(**sample_memory_data)
        
        assert memory.pattern == sample_memory_data["pattern"]
        assert memory.result == sample_memory_data["result"]
        assert memory.accuracy == sample_memory_data["accuracy"]
        assert memory.avg_return == sample_memory_data["avg_return"]
        assert memory.success_count == sample_memory_data["success_count"]
        assert memory.total_count == sample_memory_data["total_count"]
        assert memory.last_seen == sample_memory_data["last_seen"]
    
    def test_validation_success_count_exceeds_total(self, sample_memory_data):
        """Тест валидации превышения успешных над общим количеством."""
        sample_memory_data["success_count"] = SuccessCount(5)
        sample_memory_data["total_count"] = TotalCount(3)
        
        with pytest.raises(ValueError, match="Success count cannot exceed total count"):
            PatternMemory(**sample_memory_data)
    
    def test_update_result_success(self, pattern_memory, sample_result_data):
        """Тест обновления результата (успех)."""
        result = PatternResult(**sample_result_data)
        pattern_memory.update_result(result)
        
        assert pattern_memory.result == result
        assert pattern_memory.total_count == TotalCount(1)
        assert pattern_memory.success_count == SuccessCount(1)
        assert pattern_memory.accuracy == Accuracy(1.0)
        assert pattern_memory.avg_return == AverageReturn(result.price_change_15min)
        assert pattern_memory.last_seen is not None
    
    def test_update_result_failure(self, pattern_memory):
        """Тест обновления результата (неудача)."""
        result_data = {
            "outcome": PatternOutcome.FAILURE,
            "price_change_5min": -0.02,
            "price_change_15min": -0.05,
            "price_change_30min": -0.08,
            "volume_change": 0.1,
            "volatility_change": 0.05
        }
        result = PatternResult(**result_data)
        pattern_memory.update_result(result)
        
        assert pattern_memory.result == result
        assert pattern_memory.total_count == TotalCount(1)
        assert pattern_memory.success_count == SuccessCount(0)
        assert pattern_memory.accuracy == Accuracy(0.0)
        assert pattern_memory.avg_return == AverageReturn(result.price_change_15min)
    
    def test_to_dict(self, pattern_memory):
        """Тест сериализации в словарь."""
        result = pattern_memory.to_dict()
        
        assert result["pattern"] == pattern_memory.pattern.to_dict()
        assert result["result"] is None  # result is None
        assert result["accuracy"] == float(pattern_memory.accuracy)
        assert result["avg_return"] == float(pattern_memory.avg_return)
        assert result["success_count"] == int(pattern_memory.success_count)
        assert result["total_count"] == int(pattern_memory.total_count)
        assert result["last_seen"] is None  # last_seen is None
    
    def test_from_dict(self, sample_memory_data):
        """Тест десериализации из словаря."""
        data_dict = {
            "pattern": sample_memory_data["pattern"].to_dict(),
            "result": None,
            "accuracy": float(sample_memory_data["accuracy"]),
            "avg_return": float(sample_memory_data["avg_return"]),
            "success_count": int(sample_memory_data["success_count"]),
            "total_count": int(sample_memory_data["total_count"]),
            "last_seen": None
        }
        
        memory = PatternMemory.from_dict(data_dict)
        
        assert memory.pattern.to_dict() == sample_memory_data["pattern"].to_dict()
        assert memory.result is None
        assert memory.accuracy == sample_memory_data["accuracy"]
        assert memory.avg_return == sample_memory_data["avg_return"]
        assert memory.success_count == sample_memory_data["success_count"]
        assert memory.total_count == sample_memory_data["total_count"]
        assert memory.last_seen is None
    
    def test_is_reliable_true(self, sample_memory_data):
        """Тест проверки надежности (истина)."""
        sample_memory_data["accuracy"] = Accuracy(0.7)
        sample_memory_data["total_count"] = TotalCount(10)
        memory = PatternMemory(**sample_memory_data)
        assert memory.is_reliable(0.6, 5) is True
    
    def test_is_reliable_false_low_accuracy(self, sample_memory_data):
        """Тест проверки надежности (низкая точность)."""
        sample_memory_data["accuracy"] = Accuracy(0.5)
        sample_memory_data["total_count"] = TotalCount(10)
        memory = PatternMemory(**sample_memory_data)
        assert memory.is_reliable(0.6, 5) is False
    
    def test_is_reliable_false_low_count(self, sample_memory_data):
        """Тест проверки надежности (мало наблюдений)."""
        sample_memory_data["accuracy"] = Accuracy(0.7)
        sample_memory_data["total_count"] = TotalCount(3)
        memory = PatternMemory(**sample_memory_data)
        assert memory.is_reliable(0.6, 5) is False
    
    def test_get_expected_outcome_reliable(self, sample_memory_data, sample_result_data):
        """Тест получения ожидаемого результата (надежный паттерн)."""
        sample_memory_data["accuracy"] = Accuracy(0.8)
        sample_memory_data["total_count"] = TotalCount(10)
        sample_memory_data["avg_return"] = AverageReturn(0.02)
        memory = PatternMemory(**sample_memory_data)
        
        expected = memory.get_expected_outcome()
        assert expected.outcome == PatternOutcome.SUCCESS
        assert expected.price_change_15min == 0.02
    
    def test_get_expected_outcome_unreliable(self, sample_memory_data):
        """Тест получения ожидаемого результата (ненадежный паттерн)."""
        sample_memory_data["accuracy"] = Accuracy(0.3)
        sample_memory_data["total_count"] = TotalCount(2)
        memory = PatternMemory(**sample_memory_data)
        
        expected = memory.get_expected_outcome()
        assert expected.outcome == PatternOutcome.NEUTRAL
        assert expected.price_change_15min == 0.0
    
    def test_get_confidence_boost(self, sample_memory_data):
        """Тест расчета увеличения уверенности."""
        sample_memory_data["accuracy"] = Accuracy(0.8)
        sample_memory_data["total_count"] = TotalCount(10)
        memory = PatternMemory(**sample_memory_data)
        
        boost = memory.get_confidence_boost(0.9)
        assert boost > 0
        assert boost <= 1.0
    
    def test_get_confidence_boost_unreliable(self, sample_memory_data):
        """Тест расчета увеличения уверенности (ненадежный паттерн)."""
        sample_memory_data["accuracy"] = Accuracy(0.3)
        sample_memory_data["total_count"] = TotalCount(2)
        memory = PatternMemory(**sample_memory_data)
        
        boost = memory.get_confidence_boost(0.9)
        assert boost == 0.0


class TestMatchedPattern:
    """Тесты для MatchedPattern."""
    
    @pytest.fixture
    def sample_matched_data(self, sample_memory_data, sample_result_data) -> Dict[str, Any]:
        """Тестовые данные для MatchedPattern."""
        memory = PatternMemory(**sample_memory_data)
        memory.update_result(PatternResult(**sample_result_data))
        
        return {
            "pattern_memory": memory,
            "similarity_score": SimilarityScore(0.85),
            "confidence_boost": Confidence(0.2),
            "expected_outcome": PatternResult(**sample_result_data),
            "signal_strength": SignalStrength(0.7)
        }
    
    @pytest.fixture
    def matched_pattern(self, sample_matched_data) -> MatchedPattern:
        """Создает экземпляр MatchedPattern."""
        return MatchedPattern(**sample_matched_data)
    
    def test_creation_with_valid_data(self, sample_matched_data):
        """Тест создания с валидными данными."""
        matched = MatchedPattern(**sample_matched_data)
        
        assert matched.pattern_memory == sample_matched_data["pattern_memory"]
        assert matched.similarity_score == sample_matched_data["similarity_score"]
        assert matched.confidence_boost == sample_matched_data["confidence_boost"]
        assert matched.expected_outcome == sample_matched_data["expected_outcome"]
        assert matched.signal_strength == sample_matched_data["signal_strength"]
    
    def test_to_dict(self, matched_pattern):
        """Тест сериализации в словарь."""
        result = matched_pattern.to_dict()
        
        assert result["pattern_memory"] == matched_pattern.pattern_memory.to_dict()
        assert result["similarity_score"] == float(matched_pattern.similarity_score)
        assert result["confidence_boost"] == float(matched_pattern.confidence_boost)
        assert result["expected_outcome"] == matched_pattern.expected_outcome.to_dict()
        assert result["signal_strength"] == float(matched_pattern.signal_strength)
    
    def test_from_dict(self, sample_matched_data):
        """Тест десериализации из словаря."""
        data_dict = {
            "pattern_memory": sample_matched_data["pattern_memory"].to_dict(),
            "similarity_score": float(sample_matched_data["similarity_score"]),
            "confidence_boost": float(sample_matched_data["confidence_boost"]),
            "expected_outcome": sample_matched_data["expected_outcome"].to_dict(),
            "signal_strength": float(sample_matched_data["signal_strength"])
        }
        
        matched = MatchedPattern.from_dict(data_dict)
        
        assert matched.pattern_memory.to_dict() == sample_matched_data["pattern_memory"].to_dict()
        assert matched.similarity_score == sample_matched_data["similarity_score"]
        assert matched.confidence_boost == sample_matched_data["confidence_boost"]
        assert matched.expected_outcome.to_dict() == sample_matched_data["expected_outcome"].to_dict()
        assert matched.signal_strength == sample_matched_data["signal_strength"]
    
    def test_get_combined_confidence(self, matched_pattern):
        """Тест получения комбинированной уверенности."""
        combined = matched_pattern.get_combined_confidence()
        
        base_confidence = float(matched_pattern.pattern_memory.pattern.confidence)
        boost = float(matched_pattern.confidence_boost)
        expected = min(1.0, base_confidence + boost)
        
        assert combined == expected
        assert 0.0 <= combined <= 1.0
    
    def test_is_high_quality_match_true(self, sample_matched_data):
        """Тест проверки высокого качества совпадения (истина)."""
        # Настраиваем для высокого качества
        sample_matched_data["pattern_memory"].accuracy = Accuracy(0.8)
        sample_matched_data["pattern_memory"].total_count = TotalCount(10)
        sample_matched_data["similarity_score"] = SimilarityScore(0.85)
        sample_matched_data["confidence_boost"] = Confidence(0.3)
        
        matched = MatchedPattern(**sample_matched_data)
        assert matched.is_high_quality_match() is True
    
    def test_is_high_quality_match_false_low_similarity(self, sample_matched_data):
        """Тест проверки высокого качества совпадения (низкая схожесть)."""
        sample_matched_data["similarity_score"] = SimilarityScore(0.7)
        matched = MatchedPattern(**sample_matched_data)
        assert matched.is_high_quality_match() is False
    
    def test_is_high_quality_match_false_low_boost(self, sample_matched_data):
        """Тест проверки высокого качества совпадения (низкий буст)."""
        sample_matched_data["confidence_boost"] = Confidence(0.1)
        matched = MatchedPattern(**sample_matched_data)
        assert matched.is_high_quality_match() is False
    
    def test_is_high_quality_match_false_unreliable(self, sample_matched_data):
        """Тест проверки высокого качества совпадения (ненадежный паттерн)."""
        sample_matched_data["pattern_memory"].accuracy = Accuracy(0.3)
        sample_matched_data["pattern_memory"].total_count = TotalCount(2)
        matched = MatchedPattern(**sample_matched_data)
        assert matched.is_high_quality_match() is False
    
    def test_get_trading_signal(self, matched_pattern):
        """Тест получения торгового сигнала."""
        signal = matched_pattern.get_trading_signal()
        
        assert signal["pattern_type"] == matched_pattern.pattern_memory.pattern.pattern_type.value
        assert signal["symbol"] == matched_pattern.pattern_memory.pattern.symbol
        assert signal["direction"] == matched_pattern.pattern_memory.pattern.get_direction()
        assert signal["confidence"] == matched_pattern.get_combined_confidence()
        assert signal["signal_strength"] == float(matched_pattern.signal_strength)
        assert signal["expected_return"] == matched_pattern.expected_outcome.get_expected_return()
        assert signal["time_horizon"] == "15min"
        assert "risk_level" in signal
    
    def test_get_trading_signal_high_strength(self, sample_matched_data):
        """Тест получения торгового сигнала (высокая сила)."""
        sample_matched_data["signal_strength"] = SignalStrength(0.8)
        matched = MatchedPattern(**sample_matched_data)
        signal = matched.get_trading_signal()
        assert signal["risk_level"] == "high"
    
    def test_get_trading_signal_medium_strength(self, sample_matched_data):
        """Тест получения торгового сигнала (средняя сила)."""
        sample_matched_data["signal_strength"] = SignalStrength(0.3)
        matched = MatchedPattern(**sample_matched_data)
        signal = matched.get_trading_signal()
        assert signal["risk_level"] == "medium"


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""
    
    def test_empty_market_microstructure(self):
        """Тест создания пустой микроструктуры рынка."""
        microstructure = _empty_market_microstructure()
        assert isinstance(microstructure, dict)
        assert len(microstructure) == 0
    
    def test_empty_pattern_context(self):
        """Тест создания пустого контекста паттерна."""
        context = _empty_pattern_context()
        assert isinstance(context, dict)
        assert len(context) == 0 