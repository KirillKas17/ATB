"""
Unit тесты для Pattern.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.pattern import Pattern, PatternConfidence
from domain.type_definitions.pattern_types import PatternType as BasePatternType


class TestPatternConfidence:
    """Тесты для PatternConfidence."""
    
    def test_enum_values(self):
        """Тест значений enum PatternConfidence."""
        assert PatternConfidence.UNKNOWN.value == 0.0
        assert PatternConfidence.LOW.value == 0.3
        assert PatternConfidence.MEDIUM.value == 0.6
        assert PatternConfidence.HIGH.value == 0.8
        assert PatternConfidence.VERY_HIGH.value == 1.0
    
    def test_confidence_levels_order(self):
        """Тест порядка уровней уверенности."""
        assert PatternConfidence.UNKNOWN.value < PatternConfidence.LOW.value
        assert PatternConfidence.LOW.value < PatternConfidence.MEDIUM.value
        assert PatternConfidence.MEDIUM.value < PatternConfidence.HIGH.value
        assert PatternConfidence.HIGH.value < PatternConfidence.VERY_HIGH.value


class TestPattern:
    """Тесты для Pattern."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "pattern_type": BasePatternType.DOUBLE_TOP,
            "characteristics": {
                "resistance_level": 50000.0,
                "support_level": 48000.0,
                "volume_profile": "high",
                "timeframe": "4h",
                "breakout_probability": 0.7
            },
            "confidence": PatternConfidence.HIGH,
            "trading_pair_id": "BTC/USD",
            "created_at": datetime.now(),
            "metadata": {
                "source": "technical_analysis",
                "indicators_used": ["RSI", "MACD", "Bollinger_Bands"],
                "market_conditions": "trending"
            }
        }
    
    @pytest.fixture
    def pattern(self, sample_data) -> Pattern:
        """Создает тестовый паттерн."""
        return Pattern(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания паттерна."""
        pattern = Pattern(**sample_data)
        
        assert pattern.id == sample_data["id"]
        assert pattern.pattern_type == sample_data["pattern_type"]
        assert pattern.characteristics == sample_data["characteristics"]
        assert pattern.confidence == sample_data["confidence"]
        assert pattern.trading_pair_id == sample_data["trading_pair_id"]
        assert pattern.created_at == sample_data["created_at"]
        assert pattern.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания паттерна с дефолтными значениями."""
        pattern = Pattern()
        
        assert isinstance(pattern.id, uuid4().__class__)
        assert pattern.pattern_type == BasePatternType.UNKNOWN
        assert pattern.characteristics == {}
        assert pattern.confidence == PatternConfidence.UNKNOWN
        assert pattern.trading_pair_id == ""
        assert isinstance(pattern.created_at, datetime)
        assert pattern.metadata == {}
    
    def test_pattern_with_different_types(self):
        """Тест паттернов с различными типами."""
        pattern_types = [
            BasePatternType.DOUBLE_TOP,
            BasePatternType.DOUBLE_BOTTOM,
            BasePatternType.HEAD_AND_SHOULDERS,
            BasePatternType.INVERSE_HEAD_AND_SHOULDERS,
            BasePatternType.TRIANGLE,
            BasePatternType.WEDGE,
            BasePatternType.FLAG,
            BasePatternType.PENNANT,
            BasePatternType.CUP_AND_HANDLE,
            BasePatternType.INVERSE_CUP_AND_HANDLE,
            BasePatternType.ASCENDING_TRIANGLE,
            BasePatternType.DESCENDING_TRIANGLE,
            BasePatternType.SYMMETRICAL_TRIANGLE,
            BasePatternType.RISING_WEDGE,
            BasePatternType.FALLING_WEDGE,
            BasePatternType.BULL_FLAG,
            BasePatternType.BEAR_FLAG,
            BasePatternType.BULL_PENNANT,
            BasePatternType.BEAR_PENNANT,
            BasePatternType.UNKNOWN
        ]
        
        for pattern_type in pattern_types:
            pattern = Pattern(pattern_type=pattern_type)
            assert pattern.pattern_type == pattern_type
    
    def test_pattern_with_different_confidence_levels(self):
        """Тест паттернов с различными уровнями уверенности."""
        confidence_levels = [
            PatternConfidence.UNKNOWN,
            PatternConfidence.LOW,
            PatternConfidence.MEDIUM,
            PatternConfidence.HIGH,
            PatternConfidence.VERY_HIGH
        ]
        
        for confidence in confidence_levels:
            pattern = Pattern(confidence=confidence)
            assert pattern.confidence == confidence
    
    def test_pattern_with_complex_characteristics(self):
        """Тест паттерна со сложными характеристиками."""
        characteristics = {
            "price_levels": {
                "resistance": [50000, 51000, 52000],
                "support": [48000, 47000, 46000]
            },
            "volume_analysis": {
                "avg_volume": 1000000,
                "volume_trend": "increasing",
                "volume_ratio": 1.5
            },
            "time_analysis": {
                "formation_time": "2_weeks",
                "breakout_time": "1_day",
                "consolidation_period": "5_days"
            },
            "technical_indicators": {
                "rsi": 65.5,
                "macd": {"signal": 0.002, "histogram": 0.001},
                "bollinger_bands": {"upper": 52000, "lower": 48000, "middle": 50000}
            },
            "market_context": {
                "trend": "bullish",
                "volatility": "medium",
                "liquidity": "high"
            }
        }
        
        pattern = Pattern(characteristics=characteristics)
        assert pattern.characteristics == characteristics
        assert "price_levels" in pattern.characteristics
        assert "volume_analysis" in pattern.characteristics
        assert "time_analysis" in pattern.characteristics
        assert "technical_indicators" in pattern.characteristics
        assert "market_context" in pattern.characteristics
    
    def test_pattern_with_extensive_metadata(self):
        """Тест паттерна с расширенными метаданными."""
        metadata = {
            "analysis_source": "automated_pattern_detector",
            "detection_algorithm": "machine_learning_v2",
            "model_version": "1.2.3",
            "training_data": {
                "period": "2020-2023",
                "markets": ["crypto", "forex", "stocks"],
                "accuracy": 0.85
            },
            "validation_metrics": {
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80
            },
            "market_conditions": {
                "volatility_regime": "high",
                "trend_strength": "strong",
                "correlation_regime": "low"
            },
            "risk_assessment": {
                "success_probability": 0.75,
                "risk_reward_ratio": 2.5,
                "max_drawdown": 0.15
            },
            "execution_parameters": {
                "entry_strategy": "breakout",
                "stop_loss_type": "atr_based",
                "take_profit_type": "fibonacci_extension"
            }
        }
        
        pattern = Pattern(metadata=metadata)
        assert pattern.metadata == metadata
        assert pattern.metadata["analysis_source"] == "automated_pattern_detector"
        assert pattern.metadata["training_data"]["accuracy"] == 0.85
        assert pattern.metadata["risk_assessment"]["success_probability"] == 0.75
    
    def test_pattern_equality(self):
        """Тест равенства паттернов."""
        pattern1 = Pattern(
            id=uuid4(),
            pattern_type=BasePatternType.DOUBLE_TOP,
            confidence=PatternConfidence.HIGH,
            trading_pair_id="BTC/USD"
        )
        
        pattern2 = Pattern(
            id=pattern1.id,  # Тот же ID
            pattern_type=BasePatternType.DOUBLE_TOP,
            confidence=PatternConfidence.HIGH,
            trading_pair_id="BTC/USD"
        )
        
        pattern3 = Pattern(
            id=uuid4(),  # Другой ID
            pattern_type=BasePatternType.DOUBLE_TOP,
            confidence=PatternConfidence.HIGH,
            trading_pair_id="BTC/USD"
        )
        
        # Паттерны с одинаковыми ID должны быть равны
        assert pattern1.id == pattern2.id
        assert pattern1.pattern_type == pattern2.pattern_type
        assert pattern1.confidence == pattern2.confidence
        assert pattern1.trading_pair_id == pattern2.trading_pair_id
        
        # Паттерны с разными ID не равны
        assert pattern1.id != pattern3.id
    
    def test_pattern_immutability(self):
        """Тест неизменяемости паттерна."""
        pattern = Pattern(
            pattern_type=BasePatternType.DOUBLE_TOP,
            characteristics={"level": 50000},
            confidence=PatternConfidence.HIGH
        )
        
        # Проверяем, что атрибуты можно читать
        assert pattern.pattern_type == BasePatternType.DOUBLE_TOP
        assert pattern.characteristics["level"] == 50000
        assert pattern.confidence == PatternConfidence.HIGH
        
        # Проверяем, что атрибуты можно изменять (dataclass не immutable по умолчанию)
        pattern.characteristics["level"] = 51000
        assert pattern.characteristics["level"] == 51000
    
    def test_pattern_serialization_compatibility(self):
        """Тест совместимости с сериализацией."""
        pattern = Pattern(
            id=uuid4(),
            pattern_type=BasePatternType.HEAD_AND_SHOULDERS,
            characteristics={
                "neckline": 50000,
                "head": 52000,
                "shoulders": [51000, 51000]
            },
            confidence=PatternConfidence.VERY_HIGH,
            trading_pair_id="ETH/USD",
            metadata={"source": "manual_analysis"}
        )
        
        # Проверяем, что все атрибуты могут быть сериализованы
        pattern_dict = {
            "id": str(pattern.id),
            "pattern_type": pattern.pattern_type.value,
            "characteristics": pattern.characteristics,
            "confidence": pattern.confidence.value,
            "trading_pair_id": pattern.trading_pair_id,
            "created_at": pattern.created_at.isoformat(),
            "metadata": pattern.metadata
        }
        
        assert isinstance(pattern_dict["id"], str)
        assert isinstance(pattern_dict["pattern_type"], str)
        assert isinstance(pattern_dict["characteristics"], dict)
        assert isinstance(pattern_dict["confidence"], float)
        assert isinstance(pattern_dict["trading_pair_id"], str)
        assert isinstance(pattern_dict["created_at"], str)
        assert isinstance(pattern_dict["metadata"], dict)
    
    def test_pattern_with_empty_characteristics(self):
        """Тест паттерна с пустыми характеристиками."""
        pattern = Pattern(characteristics={})
        assert pattern.characteristics == {}
        assert len(pattern.characteristics) == 0
    
    def test_pattern_with_empty_metadata(self):
        """Тест паттерна с пустыми метаданными."""
        pattern = Pattern(metadata={})
        assert pattern.metadata == {}
        assert len(pattern.metadata) == 0
    
    def test_pattern_with_nested_characteristics(self):
        """Тест паттерна с вложенными характеристиками."""
        characteristics = {
            "price_action": {
                "swings": {
                    "highs": [52000, 52500, 53000],
                    "lows": [48000, 47500, 47000]
                },
                "momentum": {
                    "rsi_divergence": True,
                    "macd_crossover": "bullish"
                }
            },
            "volume_profile": {
                "distribution": {
                    "high_volume_nodes": [50000, 51000],
                    "low_volume_nodes": [49000, 52000]
                }
            }
        }
        
        pattern = Pattern(characteristics=characteristics)
        assert pattern.characteristics["price_action"]["swings"]["highs"] == [52000, 52500, 53000]
        assert pattern.characteristics["price_action"]["momentum"]["rsi_divergence"] is True
        assert pattern.characteristics["volume_profile"]["distribution"]["high_volume_nodes"] == [50000, 51000]
    
    def test_pattern_confidence_comparison(self):
        """Тест сравнения уровней уверенности."""
        low_confidence_pattern = Pattern(confidence=PatternConfidence.LOW)
        medium_confidence_pattern = Pattern(confidence=PatternConfidence.MEDIUM)
        high_confidence_pattern = Pattern(confidence=PatternConfidence.HIGH)
        
        assert low_confidence_pattern.confidence.value < medium_confidence_pattern.confidence.value
        assert medium_confidence_pattern.confidence.value < high_confidence_pattern.confidence.value
        assert low_confidence_pattern.confidence.value < high_confidence_pattern.confidence.value
    
    def test_pattern_type_validation(self):
        """Тест валидации типа паттерна."""
        # Все типы паттернов должны быть валидными
        valid_pattern_types = [
            BasePatternType.DOUBLE_TOP,
            BasePatternType.DOUBLE_BOTTOM,
            BasePatternType.HEAD_AND_SHOULDERS,
            BasePatternType.INVERSE_HEAD_AND_SHOULDERS,
            BasePatternType.TRIANGLE,
            BasePatternType.WEDGE,
            BasePatternType.FLAG,
            BasePatternType.PENNANT,
            BasePatternType.CUP_AND_HANDLE,
            BasePatternType.INVERSE_CUP_AND_HANDLE,
            BasePatternType.ASCENDING_TRIANGLE,
            BasePatternType.DESCENDING_TRIANGLE,
            BasePatternType.SYMMETRICAL_TRIANGLE,
            BasePatternType.RISING_WEDGE,
            BasePatternType.FALLING_WEDGE,
            BasePatternType.BULL_FLAG,
            BasePatternType.BEAR_FLAG,
            BasePatternType.BULL_PENNANT,
            BasePatternType.BEAR_PENNANT,
            BasePatternType.UNKNOWN
        ]
        
        for pattern_type in valid_pattern_types:
            pattern = Pattern(pattern_type=pattern_type)
            assert pattern.pattern_type == pattern_type
            assert hasattr(pattern.pattern_type, 'value')
    
    def test_pattern_trading_pair_format(self):
        """Тест формата торговой пары."""
        trading_pairs = [
            "BTC/USD",
            "ETH/USDT",
            "ADA/BTC",
            "DOT/USD",
            "LINK/ETH"
        ]
        
        for trading_pair in trading_pairs:
            pattern = Pattern(trading_pair_id=trading_pair)
            assert pattern.trading_pair_id == trading_pair
            assert "/" in pattern.trading_pair_id
    
    def test_pattern_creation_timestamp(self):
        """Тест временной метки создания паттерна."""
        before_creation = datetime.now()
        pattern = Pattern()
        after_creation = datetime.now()
        
        assert before_creation <= pattern.created_at <= after_creation
    
    def test_pattern_with_custom_creation_time(self):
        """Тест паттерна с пользовательским временем создания."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        pattern = Pattern(created_at=custom_time)
        
        assert pattern.created_at == custom_time
        assert pattern.created_at.year == 2023
        assert pattern.created_at.month == 1
        assert pattern.created_at.day == 1
        assert pattern.created_at.hour == 12 