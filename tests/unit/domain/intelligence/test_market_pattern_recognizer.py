"""
Unit тесты для market_pattern_recognizer.py.

Покрывает:
- MarketPatternRecognizer - распознаватель рыночных паттернов
- PatternDetection - результат обнаружения паттерна
- Методы детекции поглощения ликвидности
- Методы детекции спуфинга маркет-мейкеров
- Методы детекции айсберг-ордеров
- Анализ рыночных данных и ордербука
"""

import pytest
from shared.numpy_utils import np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

from domain.intelligence.market_pattern_recognizer import (
    MarketPatternRecognizer,
    PatternDetection
)
from domain.types.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.memory.pattern_memory import MarketFeatures
from domain.exceptions.base_exceptions import ValidationError


class TestPatternDetection:
    """Тесты для PatternDetection."""

    @pytest.fixture
    def sample_pattern_detection(self) -> PatternDetection:
        """Тестовое обнаружение паттерна."""
        return PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTC/USDT",
            timestamp=Timestamp.now(),
            confidence=0.8,
            strength=0.7,
            direction="up",
            metadata={"volume": 1000000, "price_change": 0.02},
            volume_anomaly=2.5,
            price_impact=0.015,
            order_book_imbalance=0.4,
            spread_widening=0.002,
            depth_absorption=0.6
        )

    def test_creation_valid(self, sample_pattern_detection: PatternDetection) -> None:
        """Тест создания с валидными данными."""
        assert sample_pattern_detection.pattern_type == PatternType.WHALE_ABSORPTION
        assert sample_pattern_detection.symbol == "BTC/USDT"
        assert isinstance(sample_pattern_detection.timestamp, Timestamp)
        assert sample_pattern_detection.confidence == 0.8
        assert sample_pattern_detection.strength == 0.7
        assert sample_pattern_detection.direction == "up"
        assert sample_pattern_detection.volume_anomaly == 2.5
        assert sample_pattern_detection.price_impact == 0.015
        assert sample_pattern_detection.order_book_imbalance == 0.4
        assert sample_pattern_detection.spread_widening == 0.002
        assert sample_pattern_detection.depth_absorption == 0.6

    def test_to_dict(self, sample_pattern_detection: PatternDetection) -> None:
        """Тест преобразования в словарь."""
        result = sample_pattern_detection.to_dict()
        
        assert isinstance(result, dict)
        assert result["pattern_type"] == PatternType.WHALE_ABSORPTION.value
        assert result["symbol"] == "BTC/USDT"
        assert result["confidence"] == 0.8
        assert result["strength"] == 0.7
        assert result["direction"] == "up"
        assert result["volume_anomaly"] == 2.5
        assert result["price_impact"] == 0.015
        assert result["order_book_imbalance"] == 0.4
        assert result["spread_widening"] == 0.002
        assert result["depth_absorption"] == 0.6


class TestMarketPatternRecognizer:
    """Тесты для MarketPatternRecognizer."""

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Тестовая конфигурация."""
        return {
            "volume_threshold": 500000,
            "price_impact_threshold": 0.01,
            "spread_threshold": 0.0005,
            "depth_imbalance_threshold": 0.2,
            "confidence_threshold": 0.6,
            "lookback_periods": 30,
            "volume_sma_periods": 15
        }

    @pytest.fixture
    def recognizer(self, sample_config: Dict[str, Any]) -> MarketPatternRecognizer:
        """Тестовый распознаватель."""
        return MarketPatternRecognizer(config=sample_config)

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Тестовые рыночные данные."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = {
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50100,
            'low': np.random.randn(100).cumsum() + 49900,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 100)
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_order_book(self) -> Dict[str, Any]:
        """Тестовый ордербук."""
        return {
            'bids': [
                {'price': 50000, 'size': 1.5},
                {'price': 49999, 'size': 2.0},
                {'price': 49998, 'size': 1.8}
            ],
            'asks': [
                {'price': 50001, 'size': 1.2},
                {'price': 50002, 'size': 2.1},
                {'price': 50003, 'size': 1.6}
            ],
            'spread': 1.0,
            'timestamp': Timestamp.now()
        }

    def test_initialization_default_config(self) -> None:
        """Тест инициализации с дефолтной конфигурацией."""
        recognizer = MarketPatternRecognizer()
        
        assert recognizer.config is not None
        assert "volume_threshold" in recognizer.config
        assert "price_impact_threshold" in recognizer.config
        assert "spread_threshold" in recognizer.config
        assert "depth_imbalance_threshold" in recognizer.config
        assert "confidence_threshold" in recognizer.config
        assert "lookback_periods" in recognizer.config
        assert "volume_sma_periods" in recognizer.config
        assert recognizer._volume_cache == {}
        assert recognizer._price_cache == {}
        assert recognizer._last_analysis == {}

    def test_initialization_custom_config(self, sample_config: Dict[str, Any]) -> None:
        """Тест инициализации с кастомной конфигурацией."""
        recognizer = MarketPatternRecognizer(config=sample_config)
        
        assert recognizer.config == sample_config

    def test_detect_whale_absorption_valid_data(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame, sample_order_book: Dict[str, Any]) -> None:
        """Тест детекции поглощения ликвидности с валидными данными."""
        # Создаем данные с признаками поглощения
        whale_data = sample_market_data.copy()
        whale_data['volume'].iloc[-10:] = 50000  # Большой объем
        whale_data['close'].iloc[-5:] = whale_data['close'].iloc[-5:] * 1.02  # Рост цены
        
        result = recognizer.detect_whale_absorption("BTC/USDT", whale_data, sample_order_book)
        
        if result is not None:
            assert isinstance(result, PatternDetection)
            assert result.pattern_type == PatternType.WHALE_ABSORPTION
            assert result.symbol == "BTC/USDT"
            assert isinstance(result.timestamp, Timestamp)
            assert isinstance(result.confidence, float)
            assert isinstance(result.strength, float)
            assert isinstance(result.direction, str)
            assert isinstance(result.metadata, dict)
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.strength <= 1.0

    def test_detect_whale_absorption_insufficient_data(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест детекции поглощения с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'open': [50000] * 10,
            'high': [50100] * 10,
            'low': [49900] * 10,
            'close': [50000] * 10,
            'volume': [1000] * 10
        })
        
        result = recognizer.detect_whale_absorption("BTC/USDT", insufficient_data, sample_order_book)
        
        assert result is None

    def test_detect_mm_spoofing_valid_data(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame, sample_order_book: Dict[str, Any]) -> None:
        """Тест детекции спуфинга маркет-мейкеров с валидными данными."""
        # Создаем данные с признаками спуфинга
        spoofing_data = sample_market_data.copy()
        spoofing_data['volume'].iloc[-5:] = 2000  # Низкий объем
        spoofing_data['close'].iloc[-3:] = spoofing_data['close'].iloc[-3:] * 0.99  # Падение цены
        
        result = recognizer.detect_mm_spoofing("BTC/USDT", spoofing_data, sample_order_book)
        
        if result is not None:
            assert isinstance(result, PatternDetection)
            assert result.pattern_type == PatternType.MM_SPOOFING
            assert result.symbol == "BTC/USDT"
            assert isinstance(result.confidence, float)
            assert isinstance(result.strength, float)
            assert isinstance(result.direction, str)

    def test_detect_iceberg_detection_valid_data(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame, sample_order_book: Dict[str, Any]) -> None:
        """Тест детекции айсберг-ордеров с валидными данными."""
        # Создаем данные с признаками айсберг-ордеров
        iceberg_data = sample_market_data.copy()
        iceberg_data['volume'].iloc[-20:] = 500  # Постоянный низкий объем
        iceberg_data['close'].iloc[-10:] = iceberg_data['close'].iloc[-10:] * 1.005  # Медленный рост
        
        result = recognizer.detect_iceberg_detection("BTC/USDT", iceberg_data, sample_order_book)
        
        if result is not None:
            assert isinstance(result, PatternDetection)
            assert result.pattern_type == PatternType.ICEBERG_DETECTION
            assert result.symbol == "BTC/USDT"
            assert isinstance(result.confidence, float)
            assert isinstance(result.strength, float)
            assert isinstance(result.direction, str)

    def test_analyze_volume_anomaly(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа аномалий объема."""
        result = recognizer._analyze_volume_anomaly(sample_market_data)
        
        assert isinstance(result, dict)
        assert "current_volume" in result
        assert "volume_sma" in result
        assert "volume_ratio" in result
        assert "volume_anomaly" in result
        assert isinstance(result["current_volume"], (int, float))
        assert isinstance(result["volume_sma"], (int, float))
        assert isinstance(result["volume_ratio"], float)
        assert isinstance(result["volume_anomaly"], float)

    def test_analyze_price_movement(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа движения цены."""
        result = recognizer._analyze_price_movement(sample_market_data)
        
        assert isinstance(result, dict)
        assert "price_change" in result
        assert "price_volatility" in result
        assert "price_momentum" in result
        assert "price_impact" in result
        assert isinstance(result["price_change"], float)
        assert isinstance(result["price_volatility"], float)
        assert isinstance(result["price_momentum"], float)
        assert isinstance(result["price_impact"], float)

    def test_analyze_order_book_imbalance(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест анализа дисбаланса ордербука."""
        result = recognizer._analyze_order_book_imbalance(sample_order_book)
        
        assert isinstance(result, dict)
        assert "bid_volume" in result
        assert "ask_volume" in result
        assert "imbalance_ratio" in result
        assert "imbalance_direction" in result
        assert isinstance(result["bid_volume"], float)
        assert isinstance(result["ask_volume"], float)
        assert isinstance(result["imbalance_ratio"], float)
        assert isinstance(result["imbalance_direction"], str)

    def test_analyze_spread_widening(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест анализа расширения спреда."""
        result = recognizer._analyze_spread_widening(sample_order_book)
        
        assert isinstance(result, dict)
        assert "current_spread" in result
        assert "spread_threshold" in result
        assert "spread_widening" in result
        assert "spread_anomaly" in result
        assert isinstance(result["current_spread"], float)
        assert isinstance(result["spread_threshold"], float)
        assert isinstance(result["spread_widening"], float)
        assert isinstance(result["spread_anomaly"], bool)

    def test_analyze_liquidity_dynamics(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест анализа динамики ликвидности."""
        result = recognizer._analyze_liquidity_dynamics(sample_order_book)
        
        assert isinstance(result, dict)
        assert "total_depth" in result
        assert "depth_absorption" in result
        assert "liquidity_ratio" in result
        assert "absorption_ratio" in result
        assert isinstance(result["total_depth"], float)
        assert isinstance(result["depth_absorption"], float)
        assert isinstance(result["liquidity_ratio"], float)
        assert isinstance(result["absorption_ratio"], float)

    def test_analyze_iceberg_patterns(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа паттернов айсберг-ордеров."""
        result = recognizer._analyze_iceberg_patterns(sample_market_data)
        
        assert isinstance(result, dict)
        assert "volume_consistency" in result
        assert "volume_pattern" in result
        assert "iceberg_probability" in result
        assert "consistency_score" in result
        assert isinstance(result["volume_consistency"], float)
        assert isinstance(result["volume_pattern"], str)
        assert isinstance(result["iceberg_probability"], float)
        assert isinstance(result["consistency_score"], float)

    def test_analyze_volume_consistency(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа консистентности объема."""
        result = recognizer._analyze_volume_consistency(sample_market_data)
        
        assert isinstance(result, dict)
        assert "volume_std" in result
        assert "volume_cv" in result
        assert "consistency_score" in result
        assert "is_consistent" in result
        assert isinstance(result["volume_std"], float)
        assert isinstance(result["volume_cv"], float)
        assert isinstance(result["consistency_score"], float)
        assert isinstance(result["is_consistent"], bool)

    def test_calculate_absorption_ratio(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест вычисления коэффициента поглощения."""
        ratio = recognizer._calculate_absorption_ratio(sample_order_book)
        
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_calculate_absorption_confidence(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест вычисления уверенности в поглощении."""
        volume_analysis = {"volume_anomaly": 2.5, "volume_ratio": 3.0}
        price_analysis = {"price_impact": 0.015, "price_change": 0.02}
        orderbook_analysis = {"imbalance_ratio": 0.4, "imbalance_direction": "bid"}
        spread_analysis = {"spread_widening": 0.002, "spread_anomaly": True}
        
        confidence = recognizer._calculate_absorption_confidence(
            volume_analysis, price_analysis, orderbook_analysis, spread_analysis
        )
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_spoofing_confidence(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест вычисления уверенности в спуфинге."""
        imbalance_analysis = {"imbalance_ratio": 0.6, "imbalance_direction": "ask"}
        price_analysis = {"price_impact": -0.01, "price_change": -0.015}
        liquidity_analysis = {"absorption_ratio": 0.3, "depth_absorption": 0.2}
        volume_analysis = {"volume_anomaly": 0.5, "volume_ratio": 0.8}
        
        confidence = recognizer._calculate_spoofing_confidence(
            imbalance_analysis, price_analysis, liquidity_analysis, volume_analysis
        )
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_iceberg_confidence(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест вычисления уверенности в айсберг-ордерах."""
        iceberg_analysis = {"iceberg_probability": 0.8, "consistency_score": 0.9}
        price_analysis = {"price_impact": 0.005, "price_change": 0.008}
        volume_analysis = {"volume_consistency": 0.85, "volume_pattern": "consistent"}
        
        confidence = recognizer._calculate_iceberg_confidence(
            iceberg_analysis, price_analysis, volume_analysis
        )
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_determine_absorption_direction(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест определения направления поглощения."""
        price_analysis = {"price_change": 0.02, "price_momentum": 0.015}
        orderbook_analysis = {"imbalance_direction": "bid", "imbalance_ratio": 0.4}
        
        direction = recognizer._determine_absorption_direction(price_analysis, orderbook_analysis)
        
        assert isinstance(direction, str)
        assert direction in ["up", "down", "neutral"]

    def test_determine_spoofing_direction(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест определения направления спуфинга."""
        imbalance_analysis = {"imbalance_direction": "ask", "imbalance_ratio": 0.6}
        price_analysis = {"price_change": -0.015, "price_momentum": -0.01}
        
        direction = recognizer._determine_spoofing_direction(imbalance_analysis, price_analysis)
        
        assert isinstance(direction, str)
        assert direction in ["up", "down", "neutral"]

    def test_create_default_pattern(self, recognizer: MarketPatternRecognizer) -> None:
        """Тест создания дефолтного паттерна."""
        result = recognizer._create_default_pattern("BTC/USDT", "insufficient_data")
        
        assert result is None

    def test_error_handling_invalid_market_data(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест обработки ошибок с невалидными рыночными данными."""
        invalid_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'close': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
        
        result = recognizer.detect_whale_absorption("BTC/USDT", invalid_data, sample_order_book)
        
        assert result is None

    def test_error_handling_invalid_order_book(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест обработки ошибок с невалидным ордербуком."""
        invalid_order_book = {
            'bids': [],
            'asks': [],
            'spread': 0.0
        }
        
        result = recognizer.detect_whale_absorption("BTC/USDT", sample_market_data, invalid_order_book)
        
        if result is not None:
            assert isinstance(result, PatternDetection)

    def test_performance_with_large_data(self, recognizer: MarketPatternRecognizer, sample_order_book: Dict[str, Any]) -> None:
        """Тест производительности с большими данными."""
        # Создаем большие рыночные данные
        large_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 50000,
            'high': np.random.randn(1000).cumsum() + 50100,
            'low': np.random.randn(1000).cumsum() + 49900,
            'close': np.random.randn(1000).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        start_time = datetime.now()
        result = recognizer.detect_whale_absorption("BTC/USDT", large_data, sample_order_book)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Обработка должна быть быстрой (менее 1 секунды)
        assert processing_time < 1.0
        assert result is None or isinstance(result, PatternDetection)

    def test_cache_functionality(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame) -> None:
        """Тест функциональности кэша."""
        # Первый вызов - кэш пустой
        result1 = recognizer._analyze_volume_anomaly(sample_market_data)
        
        # Второй вызов - данные должны быть в кэше
        result2 = recognizer._analyze_volume_anomaly(sample_market_data)
        
        assert result1 == result2
        assert "BTC/USDT" in recognizer._volume_cache

    def test_different_pattern_types(self, recognizer: MarketPatternRecognizer, sample_market_data: pd.DataFrame, sample_order_book: Dict[str, Any]) -> None:
        """Тест различных типов паттернов."""
        # Тестируем все типы паттернов
        patterns = [
            recognizer.detect_whale_absorption("BTC/USDT", sample_market_data, sample_order_book),
            recognizer.detect_mm_spoofing("BTC/USDT", sample_market_data, sample_order_book),
            recognizer.detect_iceberg_detection("BTC/USDT", sample_market_data, sample_order_book)
        ]
        
        for pattern in patterns:
            if pattern is not None:
                assert isinstance(pattern, PatternDetection)
                assert pattern.pattern_type in [
                    PatternType.WHALE_ABSORPTION,
                    PatternType.MM_SPOOFING,
                    PatternType.ICEBERG_DETECTION
                ]
                assert pattern.symbol == "BTC/USDT"
                assert isinstance(pattern.confidence, float)
                assert isinstance(pattern.strength, float)
                assert isinstance(pattern.direction, str) 