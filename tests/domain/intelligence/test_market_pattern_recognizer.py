"""
Тесты для MarketPatternRecognizer
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from domain.intelligence.market_pattern_recognizer import (
    MarketPatternRecognizer,
    PatternDetection,
    PatternType
)
from domain.value_objects.timestamp import Timestamp
class TestMarketPatternRecognizer:
    """Тесты для MarketPatternRecognizer"""
    @pytest.fixture
    def recognizer(self) -> Any:
        return MarketPatternRecognizer()
    @pytest.fixture
    def sample_market_data(self) -> Any:
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100),
        }
        return pd.DataFrame(data)
    @pytest.fixture
    def sample_order_book(self) -> Any:
        return {
            'bids': [
                {'price': 50000, 'size': 1.5},
                {'price': 49999, 'size': 2.0},
                {'price': 49998, 'size': 1.0},
            ],
            'asks': [
                {'price': 50001, 'size': 1.2},
                {'price': 50002, 'size': 2.5},
                {'price': 50003, 'size': 1.8},
            ],
            'timestamp': datetime.now().isoformat()
        }
    def test_recognizer_initialization(self, recognizer) -> None:
        """Тест инициализации распознавателя"""
        assert recognizer.config is not None
        assert 'volume_threshold' in recognizer.config
        assert 'price_impact_threshold' in recognizer.config
        assert 'confidence_threshold' in recognizer.config
    def test_recognizer_with_custom_config(self) -> None:
        """Тест инициализации с пользовательской конфигурацией"""
        custom_config = {
            'volume_threshold': 2000000,
            'price_impact_threshold': 0.03,
            'confidence_threshold': 0.8
        }
        recognizer = MarketPatternRecognizer(custom_config)
        assert recognizer.config['volume_threshold'] == 2000000
        assert recognizer.config['price_impact_threshold'] == 0.03
        assert recognizer.config['confidence_threshold'] == 0.8
    def test_analyze_volume_anomaly(self, recognizer, sample_market_data) -> None:
        """Тест анализа аномалий объема"""
        sample_market_data.loc[50, 'volume'] = 5000
        result = recognizer._analyze_volume_anomaly(sample_market_data)
        assert 'is_anomaly' in result
        assert 'anomaly_ratio' in result
        assert 'volume_sma' in result
        assert isinstance(result['is_anomaly'], bool)
        assert isinstance(result['anomaly_ratio'], float)
    def test_analyze_price_movement(self, recognizer, sample_market_data) -> None:
        """Тест анализа движения цены"""
        result = recognizer._analyze_price_movement(sample_market_data)
        assert 'price_change' in result
        assert 'price_volatility' in result
        assert 'trend_direction' in result
        assert isinstance(result['price_change'], float)
        assert result['trend_direction'] in ['up', 'down', 'neutral']
    def test_analyze_order_book_imbalance(self, recognizer, sample_order_book) -> None:
        """Тест анализа дисбаланса стакана"""
        result = recognizer._analyze_order_book_imbalance(sample_order_book)
        assert 'imbalance_ratio' in result
        assert 'absorption_ratio' in result
        assert 'bid_strength' in result
        assert 'ask_strength' in result
        assert isinstance(result['imbalance_ratio'], float)
        assert isinstance(result['absorption_ratio'], float)
    def test_detect_whale_absorption_insufficient_data(self, recognizer, sample_order_book) -> None:
        """Тест обнаружения поглощения китами с недостаточными данными"""
        insufficient_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1min'),
            'volume': np.random.uniform(100, 1000, 10),
            'close': np.random.uniform(50000, 51000, 10)
        })
        result = recognizer.detect_whale_absorption("BTC/USDT", insufficient_data, sample_order_book)
        assert result is None
    def test_detect_mm_spoofing_insufficient_data(self, recognizer, sample_order_book) -> None:
        """Тест обнаружения спуфинга с недостаточными данными"""
        insufficient_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1min'),
            'volume': np.random.uniform(100, 1000, 10),
            'close': np.random.uniform(50000, 51000, 10)
        })
        result = recognizer.detect_mm_spoofing("BTC/USDT", insufficient_data, sample_order_book)
        assert result is None
class TestPatternDetection:
    """Тесты для PatternDetection"""
    def test_pattern_detection_creation(self) -> None:
        """Тест создания PatternDetection"""
        detection = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTC/USDT",
            timestamp=Timestamp.now(),
            confidence=0.85,
            strength=2.5,
            direction="up",
            metadata={"test": "data"},
            volume_anomaly=2.5,
            price_impact=0.02,
            order_book_imbalance=0.4,
            spread_widening=0.001,
            depth_absorption=0.6
        )
        assert detection.pattern_type == PatternType.WHALE_ABSORPTION
        assert detection.symbol == "BTC/USDT"
        assert detection.confidence == 0.85
        assert detection.strength == 2.5
        assert detection.direction == "up"
        assert detection.metadata["test"] == "data"
    def test_pattern_detection_to_dict(self) -> None:
        """Тест преобразования PatternDetection в словарь"""
        detection = PatternDetection(
            pattern_type=PatternType.MM_SPOOFING,
            symbol="ETH/USDT",
            timestamp=Timestamp.now(),
            confidence=0.75,
            strength=1.8,
            direction="down",
            metadata={"strategy": "momentum"},
            volume_anomaly=1.8,
            price_impact=0.03,
            order_book_imbalance=0.5,
            spread_widening=0.002,
            depth_absorption=0.4
        )
        result = detection.to_dict()
        assert result["pattern_type"] == "mm_spoofing"
        assert result["symbol"] == "ETH/USDT"
        assert result["confidence"] == 0.75
        assert result["strength"] == 1.8
        assert result["direction"] == "down"
class TestPatternType:
    """Тесты для PatternType"""
    def test_pattern_type_values(self) -> None:
        """Тест значений типов паттернов"""
        assert PatternType.WHALE_ABSORPTION.value == "whale_absorption"
        assert PatternType.MM_SPOOFING.value == "mm_spoofing"
        assert PatternType.ICEBERG_DETECTION.value == "iceberg_detection"
        assert PatternType.LIQUIDITY_GRAB.value == "liquidity_grab"
        assert PatternType.PUMP_AND_DUMP.value == "pump_and_dump"
        assert PatternType.STOP_HUNTING.value == "stop_hunting"
        assert PatternType.ACCUMULATION.value == "accumulation"
        assert PatternType.DISTRIBUTION.value == "distribution" 
