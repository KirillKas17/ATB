"""
Unit тесты для entanglement_detector.py.

Покрывает:
- EntanglementDetector - детектор квантовой запутанности
- Методы детекции запутанности между биржами
- Вычисление корреляционных матриц
- Анализ лагов между биржами
- Обработка данных ордербуков
- Статистика и метаданные
"""

import pytest
from shared.numpy_utils import np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
from collections import deque

from domain.intelligence.entanglement_detector import EntanglementDetector
from domain.types.intelligence_types import (
    EntanglementConfig,
    EntanglementResult,
    EntanglementStrength,
    EntanglementType,
    OrderBookSnapshot,
    OrderBookUpdate,
    CorrelationMethod
)
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.exceptions.base_exceptions import ValidationError


class TestEntanglementDetector:
    """Тесты для EntanglementDetector."""

    @pytest.fixture
    def sample_config(self) -> EntanglementConfig:
        """Тестовая конфигурация."""
        return EntanglementConfig(
            correlation_threshold=0.7,
            max_lag_ms=10.0,
            min_data_points=50,
            confidence_threshold=0.8,
            correlation_method=CorrelationMethod.PEARSON
        )

    @pytest.fixture
    def detector(self, sample_config: EntanglementConfig) -> EntanglementDetector:
        """Тестовый детектор."""
        return EntanglementDetector(
            config=sample_config,
            enable_advanced_metrics=True,
            enable_cross_correlation=True
        )

    @pytest.fixture
    def sample_order_book(self) -> OrderBookSnapshot:
        """Тестовый ордербук."""
        bids = [
            (Price(50000.0, Currency("USD")), Volume(1.0, Currency("BTC"))),
            (Price(49999.0, Currency("USD")), Volume(2.0, Currency("BTC"))),
            (Price(49998.0, Currency("USD")), Volume(1.5, Currency("BTC"))),
        ]
        asks = [
            (Price(50001.0, Currency("USD")), Volume(1.2, Currency("BTC"))),
            (Price(50002.0, Currency("USD")), Volume(2.1, Currency("BTC"))),
            (Price(50003.0, Currency("USD")), Volume(1.8, Currency("BTC"))),
        ]
        
        order_book = Mock(spec=OrderBookSnapshot)
        order_book.bids = bids
        order_book.asks = asks
        order_book.timestamp = Timestamp.now()
        
        return order_book

    @pytest.fixture
    def sample_exchange_data(self, sample_order_book: OrderBookSnapshot) -> Dict[str, Any]:
        """Тестовые данные бирж."""
        return {
            "binance": {
                "order_book": sample_order_book,
                "timestamp": Timestamp.now(),
                "symbol": "BTC/USDT"
            },
            "coinbase": {
                "order_book": sample_order_book,
                "timestamp": Timestamp.now(),
                "symbol": "BTC/USD"
            },
            "kraken": {
                "order_book": sample_order_book,
                "timestamp": Timestamp.now(),
                "symbol": "BTC/USD"
            }
        }

    def test_initialization_default_config(self) -> None:
        """Тест инициализации с дефолтной конфигурацией."""
        detector = EntanglementDetector()
        
        assert detector.config is not None
        assert detector.enable_advanced_metrics is True
        assert detector.enable_cross_correlation is True
        assert detector.price_buffers == {}
        assert detector.volume_buffers == {}
        assert detector.spread_buffers == {}
        assert detector.timestamp_buffers == {}
        assert detector.statistics["total_analyses"] == 0
        assert detector.statistics["entanglement_detections"] == 0

    def test_initialization_custom_config(self, sample_config: EntanglementConfig) -> None:
        """Тест инициализации с кастомной конфигурацией."""
        detector = EntanglementDetector(
            config=sample_config,
            enable_advanced_metrics=False,
            enable_cross_correlation=False
        )
        
        assert detector.config == sample_config
        assert detector.enable_advanced_metrics is False
        assert detector.enable_cross_correlation is False

    def test_detect_entanglement_valid_data(self, detector: EntanglementDetector, sample_exchange_data: Dict[str, Any]) -> None:
        """Тест детекции запутанности с валидными данными."""
        result = detector.detect_entanglement("BTC/USDT", sample_exchange_data)
        
        assert isinstance(result, EntanglementResult)
        assert result.symbol == "BTC/USDT"
        assert isinstance(result.entanglement_type, EntanglementType)
        assert isinstance(result.strength, EntanglementStrength)
        assert isinstance(result.confidence, float)
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.metadata, dict)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_entanglement_insufficient_exchanges(self, detector: EntanglementDetector) -> None:
        """Тест детекции запутанности с недостаточным количеством бирж."""
        exchange_data = {
            "binance": {
                "order_book": Mock(spec=OrderBookSnapshot),
                "timestamp": Timestamp.now(),
                "symbol": "BTC/USDT"
            }
        }
        
        result = detector.detect_entanglement("BTC/USDT", exchange_data)
        
        assert isinstance(result, EntanglementResult)
        assert result.entanglement_type == EntanglementType.NONE
        assert result.strength == EntanglementStrength.NONE
        assert result.confidence == 0.0

    def test_detect_entanglement_empty_data(self, detector: EntanglementDetector) -> None:
        """Тест детекции запутанности с пустыми данными."""
        result = detector.detect_entanglement("BTC/USDT", {})
        
        assert isinstance(result, EntanglementResult)
        assert result.entanglement_type == EntanglementType.NONE
        assert result.strength == EntanglementStrength.NONE
        assert result.confidence == 0.0

    def test_detect_entanglement_custom_parameters(self, detector: EntanglementDetector, sample_exchange_data: Dict[str, Any]) -> None:
        """Тест детекции запутанности с кастомными параметрами."""
        result = detector.detect_entanglement(
            "BTC/USDT", 
            sample_exchange_data,
            max_lag_ms=5.0,
            correlation_threshold=0.8
        )
        
        assert isinstance(result, EntanglementResult)
        assert result.symbol == "BTC/USDT"

    def test_extract_order_books_valid_data(self, detector: EntanglementDetector, sample_exchange_data: Dict[str, Any]) -> None:
        """Тест извлечения ордербуков с валидными данными."""
        order_books = detector._extract_order_books(sample_exchange_data)
        
        assert isinstance(order_books, dict)
        assert len(order_books) == 3
        assert "binance" in order_books
        assert "coinbase" in order_books
        assert "kraken" in order_books
        
        for order_book in order_books.values():
            assert isinstance(order_book, OrderBookSnapshot)

    def test_extract_order_books_invalid_data(self, detector: EntanglementDetector) -> None:
        """Тест извлечения ордербуков с невалидными данными."""
        invalid_data = {
            "binance": {"invalid": "data"},
            "coinbase": {"order_book": "not_an_order_book"}
        }
        
        order_books = detector._extract_order_books(invalid_data)
        
        assert isinstance(order_books, dict)
        assert len(order_books) == 0

    def test_compute_correlation_matrix(self, detector: EntanglementDetector, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления корреляционной матрицы."""
        order_books = {
            "binance": sample_order_book,
            "coinbase": sample_order_book,
            "kraken": sample_order_book
        }
        
        correlation_matrix = detector._compute_correlation_matrix(order_books)
        
        assert isinstance(correlation_matrix, dict)
        assert len(correlation_matrix) == 3  # 3 пары бирж
        
        for (exchange1, exchange2), correlation in correlation_matrix.items():
            assert isinstance(exchange1, str)
            assert isinstance(exchange2, str)
            assert isinstance(correlation, float)
            assert -1.0 <= correlation <= 1.0

    def test_extract_prices(self, detector: EntanglementDetector, sample_order_book: OrderBookSnapshot) -> None:
        """Тест извлечения цен из ордербука."""
        prices = detector._extract_prices(sample_order_book)
        
        assert isinstance(prices, list)
        assert len(prices) == 6  # 3 bid + 3 ask
        assert all(isinstance(price, float) for price in prices)

    def test_calculate_correlation_identical_prices(self, detector: EntanglementDetector) -> None:
        """Тест вычисления корреляции с идентичными ценами."""
        prices1 = [100.0, 101.0, 102.0, 103.0, 104.0]
        prices2 = [100.0, 101.0, 102.0, 103.0, 104.0]
        
        correlation = detector._calculate_correlation(prices1, prices2)
        
        assert correlation == 1.0  # Полная корреляция

    def test_calculate_correlation_opposite_prices(self, detector: EntanglementDetector) -> None:
        """Тест вычисления корреляции с противоположными ценами."""
        prices1 = [100.0, 101.0, 102.0, 103.0, 104.0]
        prices2 = [104.0, 103.0, 102.0, 101.0, 100.0]
        
        correlation = detector._calculate_correlation(prices1, prices2)
        
        assert correlation == -1.0  # Полная отрицательная корреляция

    def test_calculate_correlation_different_lengths(self, detector: EntanglementDetector) -> None:
        """Тест вычисления корреляции с разными длинами."""
        prices1 = [100.0, 101.0, 102.0]
        prices2 = [100.0, 101.0, 102.0, 103.0, 104.0]
        
        correlation = detector._calculate_correlation(prices1, prices2)
        
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0

    def test_compute_lag_matrix(self, detector: EntanglementDetector, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления матрицы лагов."""
        order_books = {
            "binance": sample_order_book,
            "coinbase": sample_order_book,
            "kraken": sample_order_book
        }
        
        lag_matrix = detector._compute_lag_matrix(order_books)
        
        assert isinstance(lag_matrix, dict)
        assert len(lag_matrix) == 3  # 3 пары бирж
        
        for (exchange1, exchange2), lag in lag_matrix.items():
            assert isinstance(exchange1, str)
            assert isinstance(exchange2, str)
            assert isinstance(lag, float)
            assert lag >= 0.0

    def test_find_best_entanglement_pair(self, detector: EntanglementDetector) -> None:
        """Тест поиска лучшей пары запутанности."""
        correlation_matrix = {
            ("binance", "coinbase"): 0.8,
            ("binance", "kraken"): 0.6,
            ("coinbase", "kraken"): 0.9
        }
        lag_matrix = {
            ("binance", "coinbase"): 5.0,
            ("binance", "kraken"): 15.0,
            ("coinbase", "kraken"): 3.0
        }
        
        best_pair, correlation, lag = detector._find_best_entanglement_pair(
            correlation_matrix, lag_matrix, 10.0
        )
        
        assert isinstance(best_pair, tuple)
        assert len(best_pair) == 2
        assert isinstance(correlation, float)
        assert isinstance(lag, float)
        assert correlation > 0.0
        assert lag >= 0.0

    def test_calculate_confidence(self, detector: EntanglementDetector) -> None:
        """Тест вычисления уверенности."""
        correlation = 0.8
        lag_ms = 5.0
        num_exchanges = 3
        
        confidence = detector._calculate_confidence(correlation, lag_ms, num_exchanges)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_high_correlation(self, detector: EntanglementDetector) -> None:
        """Тест вычисления уверенности с высокой корреляцией."""
        confidence = detector._calculate_confidence(0.95, 2.0, 3)
        
        assert confidence > 0.8  # Высокая уверенность

    def test_calculate_confidence_low_correlation(self, detector: EntanglementDetector) -> None:
        """Тест вычисления уверенности с низкой корреляцией."""
        confidence = detector._calculate_confidence(0.3, 15.0, 2)
        
        assert confidence < 0.5  # Низкая уверенность

    def test_create_analysis_metadata(self, detector: EntanglementDetector, sample_order_book: OrderBookSnapshot) -> None:
        """Тест создания метаданных анализа."""
        order_books = {
            "binance": sample_order_book,
            "coinbase": sample_order_book
        }
        correlation_matrix = {("binance", "coinbase"): 0.8}
        lag_matrix = {("binance", "coinbase"): 5.0}
        start_time = 1234567890.0
        
        metadata = detector._create_analysis_metadata(
            order_books, correlation_matrix, lag_matrix, start_time
        )
        
        assert isinstance(metadata, dict)
        assert "algorithm_version" in metadata
        assert "processing_time_ms" in metadata
        assert "num_exchanges" in metadata
        assert "correlation_matrix" in metadata
        assert "lag_matrix" in metadata

    def test_create_default_result(self, detector: EntanglementDetector) -> None:
        """Тест создания дефолтного результата."""
        symbol = "BTC/USDT"
        reason = "insufficient_data"
        start_time = 1234567890.0
        
        result = detector._create_default_result(symbol, reason, start_time)
        
        assert isinstance(result, EntanglementResult)
        assert result.symbol == symbol
        assert result.entanglement_type == EntanglementType.NONE
        assert result.strength == EntanglementStrength.NONE
        assert result.confidence == 0.0
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.metadata, dict)

    def test_update_statistics(self, detector: EntanglementDetector) -> None:
        """Тест обновления статистики."""
        result = Mock(spec=EntanglementResult)
        result.entanglement_type = EntanglementType.STRONG
        start_time = 1234567890.0
        
        initial_analyses = detector.statistics["total_analyses"]
        initial_detections = detector.statistics["entanglement_detections"]
        
        detector._update_statistics(result, start_time)
        
        assert detector.statistics["total_analyses"] == initial_analyses + 1
        assert detector.statistics["entanglement_detections"] == initial_detections + 1

    def test_get_detector_statistics(self, detector: EntanglementDetector) -> None:
        """Тест получения статистики детектора."""
        stats = detector.get_detector_statistics()
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "entanglement_detections" in stats
        assert "average_processing_time_ms" in stats
        assert "last_analysis_timestamp" in stats
        assert "config" in stats

    def test_reset_statistics(self, detector: EntanglementDetector, sample_exchange_data: Dict[str, Any]) -> None:
        """Тест сброса статистики."""
        # Выполняем анализ для накопления статистики
        detector.detect_entanglement("BTC/USDT", sample_exchange_data)
        assert detector.statistics["total_analyses"] > 0
        
        detector.reset_statistics()
        
        assert detector.statistics["total_analyses"] == 0
        assert detector.statistics["entanglement_detections"] == 0
        assert detector.statistics["average_processing_time_ms"] == 0.0
        assert detector.statistics["last_analysis_timestamp"] is None

    def test_advanced_metrics_disabled(self) -> None:
        """Тест работы с отключенными расширенными метриками."""
        detector = EntanglementDetector(enable_advanced_metrics=False)
        
        result = detector.detect_entanglement("BTC/USDT", {})
        
        assert isinstance(result, EntanglementResult)
        # Расширенные метрики не должны быть в metadata
        assert "advanced_metrics" not in result.metadata

    def test_cross_correlation_disabled(self) -> None:
        """Тест работы с отключенной кросс-корреляцией."""
        detector = EntanglementDetector(enable_cross_correlation=False)
        
        result = detector.detect_entanglement("BTC/USDT", {})
        
        assert isinstance(result, EntanglementResult)

    def test_entanglement_result_structure(self, detector: EntanglementDetector, sample_exchange_data: Dict[str, Any]) -> None:
        """Тест структуры результата запутанности."""
        result = detector.detect_entanglement("BTC/USDT", sample_exchange_data)
        
        assert result.symbol == "BTC/USDT"
        assert isinstance(result.entanglement_type, EntanglementType)
        assert isinstance(result.strength, EntanglementStrength)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.metadata, dict)

    def test_error_handling_invalid_order_book(self, detector: EntanglementDetector) -> None:
        """Тест обработки ошибок с невалидным ордербуком."""
        invalid_order_book = Mock(spec=OrderBookSnapshot)
        invalid_order_book.bids = [("invalid", "data")]
        invalid_order_book.asks = []
        
        exchange_data = {
            "binance": {"order_book": invalid_order_book, "timestamp": Timestamp.now()},
            "coinbase": {"order_book": invalid_order_book, "timestamp": Timestamp.now()}
        }
        
        result = detector.detect_entanglement("BTC/USDT", exchange_data)
        
        assert isinstance(result, EntanglementResult)
        assert result.entanglement_type == EntanglementType.NONE

    def test_performance_with_large_data(self, detector: EntanglementDetector) -> None:
        """Тест производительности с большими данными."""
        # Создаем большие данные
        large_order_book = Mock(spec=OrderBookSnapshot)
        large_order_book.bids = [(Price(50000 + i, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(100)]
        large_order_book.asks = [(Price(50100 + i, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(100)]
        large_order_book.timestamp = Timestamp.now()
        
        exchange_data = {
            f"exchange_{i}": {
                "order_book": large_order_book,
                "timestamp": Timestamp.now(),
                "symbol": "BTC/USDT"
            } for i in range(5)
        }
        
        start_time = datetime.now()
        result = detector.detect_entanglement("BTC/USDT", exchange_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Обработка должна быть быстрой (менее 2 секунд)
        assert processing_time < 2.0
        assert isinstance(result, EntanglementResult)

    def test_different_entanglement_types(self, detector: EntanglementDetector) -> None:
        """Тест различных типов запутанности."""
        # Создаем данные с разными уровнями корреляции
        high_corr_order_book = Mock(spec=OrderBookSnapshot)
        high_corr_order_book.bids = [(Price(50000 + i, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(10)]
        high_corr_order_book.asks = [(Price(50100 + i, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(10)]
        high_corr_order_book.timestamp = Timestamp.now()
        
        low_corr_order_book = Mock(spec=OrderBookSnapshot)
        low_corr_order_book.bids = [(Price(50000 + i * 10, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(10)]
        low_corr_order_book.asks = [(Price(50100 + i * 10, Currency("USD")), Volume(1.0, Currency("BTC"))) for i in range(10)]
        low_corr_order_book.timestamp = Timestamp.now()
        
        exchange_data = {
            "binance": {"order_book": high_corr_order_book, "timestamp": Timestamp.now()},
            "coinbase": {"order_book": low_corr_order_book, "timestamp": Timestamp.now()}
        }
        
        result = detector.detect_entanglement("BTC/USDT", exchange_data)
        
        assert isinstance(result, EntanglementResult)
        assert isinstance(result.entanglement_type, EntanglementType)
        assert isinstance(result.strength, EntanglementStrength) 