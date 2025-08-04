"""
Unit тесты для noise_analyzer.py.

Покрывает:
- NoiseAnalyzer - анализатор нейронного шума
- Методы анализа фрактальной размерности
- Методы анализа энтропии
- Классификация типов шума
- Обработка временных рядов
- Статистика и метаданные
"""

import pytest
from shared.numpy_utils import np
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from domain.intelligence.noise_analyzer import NoiseAnalyzer
from domain.type_definitions.intelligence_types import (
    NoiseAnalysisConfig,
    NoiseAnalysisResult,
    NoiseType,
    OrderBookSnapshot,
    AnalysisMetadata,
    NoiseMetrics
)
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.exceptions.base_exceptions import ValidationError


class TestNoiseAnalyzer:
    """Тесты для NoiseAnalyzer."""

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
        
        # Мокаем методы
        order_book.get_mid_price.return_value = Price(50000.5, Currency("USD"))
        order_book.get_total_volume.return_value = Volume(9.6, Currency("BTC"))
        order_book.get_spread.return_value = Price(1.0, Currency("USD"))
        order_book.get_volume_imbalance.return_value = 0.1
        
        return order_book

    @pytest.fixture
    def sample_config(self) -> NoiseAnalysisConfig:
        """Тестовая конфигурация."""
        return NoiseAnalysisConfig(
            fractal_dimension_lower=1.2,
            fractal_dimension_upper=1.8,
            entropy_threshold=1.5,
            min_data_points=20,
            window_size=100,
            confidence_threshold=0.7
        )

    @pytest.fixture
    def analyzer(self, sample_config: NoiseAnalysisConfig) -> NoiseAnalyzer:
        """Тестовый анализатор."""
        return NoiseAnalyzer(
            config=sample_config,
            enable_advanced_metrics=True,
            enable_frequency_analysis=True
        )

    def test_initialization_default_config(self) -> None:
        """Тест инициализации с дефолтной конфигурацией."""
        analyzer = NoiseAnalyzer()
        
        assert analyzer.config is not None
        assert analyzer.enable_advanced_metrics is True
        assert analyzer.enable_frequency_analysis is True
        assert analyzer.price_history == []
        assert analyzer.volume_history == []
        assert analyzer.spread_history == []
        assert analyzer.imbalance_history == []
        assert analyzer.statistics["total_analyses"] == 0
        assert analyzer.statistics["synthetic_noise_detections"] == 0

    def test_initialization_custom_config(self, sample_config: NoiseAnalysisConfig) -> None:
        """Тест инициализации с кастомной конфигурацией."""
        analyzer = NoiseAnalyzer(
            config=sample_config,
            enable_advanced_metrics=False,
            enable_frequency_analysis=False
        )
        
        assert analyzer.config == sample_config
        assert analyzer.enable_advanced_metrics is False
        assert analyzer.enable_frequency_analysis is False

    def test_extract_time_series_valid_data(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест извлечения временных рядов с валидными данными."""
        time_series = analyzer._extract_time_series(sample_order_book)
        
        assert "prices" in time_series
        assert "volumes" in time_series
        assert "bid_prices" in time_series
        assert "ask_prices" in time_series
        assert "bid_volumes" in time_series
        assert "ask_volumes" in time_series
        assert "spreads" in time_series
        assert "imbalances" in time_series
        
        assert len(time_series["prices"]) == 6  # 3 bid + 3 ask
        assert len(time_series["volumes"]) == 6
        assert len(time_series["bid_prices"]) == 3
        assert len(time_series["ask_prices"]) == 3
        assert len(time_series["spreads"]) == 3
        assert len(time_series["imbalances"]) == 3

    def test_extract_time_series_empty_order_book(self, analyzer: NoiseAnalyzer) -> None:
        """Тест извлечения временных рядов с пустым ордербуком."""
        empty_order_book = Mock(spec=OrderBookSnapshot)
        empty_order_book.bids = []
        empty_order_book.asks = []
        
        time_series = analyzer._extract_time_series(empty_order_book)
        
        assert all(len(series) == 0 for series in time_series.values())

    def test_extract_time_series_error_handling(self, analyzer: NoiseAnalyzer) -> None:
        """Тест обработки ошибок при извлечении временных рядов."""
        invalid_order_book = Mock(spec=OrderBookSnapshot)
        invalid_order_book.bids = [("invalid", "data")]
        invalid_order_book.asks = []
        
        time_series = analyzer._extract_time_series(invalid_order_book)
        
        assert all(len(series) == 0 for series in time_series.values())

    def test_update_history_valid_data(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест обновления истории с валидными данными."""
        analyzer._update_history(sample_order_book)
        
        assert len(analyzer.price_history) == 1
        assert len(analyzer.volume_history) == 1
        assert len(analyzer.spread_history) == 1
        assert len(analyzer.imbalance_history) == 1
        
        assert analyzer.price_history[0] == 50000.5
        assert analyzer.volume_history[0] == 9.6
        assert analyzer.spread_history[0] == 1.0
        assert analyzer.imbalance_history[0] == 0.1

    def test_update_history_window_size_limit(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест ограничения размера окна истории."""
        # Заполняем историю больше размера окна
        for _ in range(analyzer.config.window_size + 10):
            analyzer._update_history(sample_order_book)
        
        assert len(analyzer.price_history) == analyzer.config.window_size
        assert len(analyzer.volume_history) == analyzer.config.window_size
        assert len(analyzer.spread_history) == analyzer.config.window_size
        assert len(analyzer.imbalance_history) == analyzer.config.window_size

    def test_update_history_error_handling(self, analyzer: NoiseAnalyzer) -> None:
        """Тест обработки ошибок при обновлении истории."""
        invalid_order_book = Mock(spec=OrderBookSnapshot)
        invalid_order_book.get_mid_price.side_effect = Exception("Test error")
        
        analyzer._update_history(invalid_order_book)
        
        # История не должна измениться при ошибке
        assert len(analyzer.price_history) == 0

    def test_compute_higuchi_fractal_dimension_valid_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления фрактальной размерности Хигучи с валидными данными."""
        # Создаем тестовые данные с известной фрактальной размерностью
        data = np.random.randn(100)
        
        fd = analyzer._compute_higuchi_fractal_dimension(data)
        
        assert isinstance(fd, float)
        assert 1.0 <= fd <= 2.0

    def test_compute_higuchi_fractal_dimension_insufficient_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления фрактальной размерности с недостаточными данными."""
        data = np.array([1.0, 2.0, 3.0])  # Меньше MIN_DATA_POINTS
        
        fd = analyzer._compute_higuchi_fractal_dimension(data)
        
        assert fd == 1.0

    def test_compute_higuchi_fractal_dimension_constant_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления фрактальной размерности с постоянными данными."""
        data = np.ones(50)  # Постоянные данные
        
        fd = analyzer._compute_higuchi_fractal_dimension(data)
        
        assert fd == 1.0

    def test_compute_sample_entropy_valid_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления sample entropy с валидными данными."""
        data = np.random.randn(50)
        
        entropy = analyzer._compute_sample_entropy(data)
        
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_compute_sample_entropy_insufficient_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления sample entropy с недостаточными данными."""
        data = np.array([1.0, 2.0])  # Меньше m + 2
        
        entropy = analyzer._compute_sample_entropy(data)
        
        assert entropy == 0.0

    def test_compute_spectral_entropy_valid_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления спектральной энтропии с валидными данными."""
        data = np.random.randn(100)
        
        entropy = analyzer._compute_spectral_entropy(data)
        
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_compute_spectral_entropy_insufficient_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления спектральной энтропии с недостаточными данными."""
        data = np.array([1.0, 2.0, 3.0])  # Меньше 10
        
        entropy = analyzer._compute_spectral_entropy(data)
        
        assert entropy == 0.0

    def test_compute_fractal_dimension(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления фрактальной размерности."""
        # Заполняем историю
        for _ in range(25):  # Больше MIN_DATA_POINTS
            analyzer._update_history(sample_order_book)
        
        fd = analyzer.compute_fractal_dimension(sample_order_book)
        
        assert isinstance(fd, float)
        assert 1.0 <= fd <= 2.0

    def test_compute_fractal_dimension_insufficient_history(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления фрактальной размерности с недостаточной историей."""
        fd = analyzer.compute_fractal_dimension(sample_order_book)
        
        assert fd == 1.0

    def test_compute_entropy(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления энтропии."""
        # Заполняем историю
        for _ in range(25):  # Больше MIN_DATA_POINTS
            analyzer._update_history(sample_order_book)
        
        entropy = analyzer.compute_entropy(sample_order_book)
        
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_compute_entropy_insufficient_history(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест вычисления энтропии с недостаточной историей."""
        entropy = analyzer.compute_entropy(sample_order_book)
        
        assert entropy == 0.0

    def test_classify_noise_type_synthetic(self, analyzer: NoiseAnalyzer) -> None:
        """Тест классификации синтетического шума."""
        # Параметры, указывающие на синтетический шум
        fd = 1.1  # Низкая фрактальная размерность
        entropy = 0.5  # Низкая энтропия
        
        noise_type = analyzer._classify_noise_type(fd, entropy)
        
        assert noise_type in [NoiseType.SYNTHETIC, NoiseType.MIXED, NoiseType.NATURAL]

    def test_classify_noise_type_natural(self, analyzer: NoiseAnalyzer) -> None:
        """Тест классификации естественного шума."""
        # Параметры, указывающие на естественный шум
        fd = 1.6  # Средняя фрактальная размерность
        entropy = 2.0  # Высокая энтропия
        
        noise_type = analyzer._classify_noise_type(fd, entropy)
        
        assert noise_type in [NoiseType.SYNTHETIC, NoiseType.MIXED, NoiseType.NATURAL]

    def test_is_synthetic_noise(self, analyzer: NoiseAnalyzer) -> None:
        """Тест определения синтетического шума."""
        fd = 1.1
        entropy = 0.5
        
        is_synthetic = analyzer.is_synthetic_noise(fd, entropy)
        
        assert isinstance(is_synthetic, bool)

    def test_compute_confidence(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления уверенности."""
        # Заполняем историю
        for _ in range(25):
            analyzer._update_history(Mock(spec=OrderBookSnapshot))
        
        fd = 1.5
        entropy = 1.0
        
        confidence = analyzer.compute_confidence(fd, entropy)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_compute_confidence_insufficient_data(self, analyzer: NoiseAnalyzer) -> None:
        """Тест вычисления уверенности с недостаточными данными."""
        fd = 1.5
        entropy = 1.0
        
        confidence = analyzer.compute_confidence(fd, entropy)
        
        assert confidence < 1.0

    def test_analyze_noise_comprehensive(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест комплексного анализа шума."""
        # Заполняем историю
        for _ in range(25):
            analyzer._update_history(sample_order_book)
        
        result = analyzer.analyze_noise(sample_order_book)
        
        assert isinstance(result, NoiseAnalysisResult)
        assert isinstance(result.fractal_dimension, float)
        assert isinstance(result.entropy, float)
        assert isinstance(result.is_synthetic_noise, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.noise_type, NoiseType)
        assert isinstance(result.metrics, dict)
        
        assert 1.0 <= result.fractal_dimension <= 2.0
        assert result.entropy >= 0.0
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_noise_insufficient_data(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест анализа шума с недостаточными данными."""
        result = analyzer.analyze_noise(sample_order_book)
        
        assert isinstance(result, NoiseAnalysisResult)
        assert result.fractal_dimension == 1.0
        assert result.entropy == 0.0
        assert result.confidence == 0.0

    def test_analyze_noise_error_handling(self, analyzer: NoiseAnalyzer) -> None:
        """Тест обработки ошибок при анализе шума."""
        invalid_order_book = Mock(spec=OrderBookSnapshot)
        invalid_order_book.bids = [("invalid", "data")]
        invalid_order_book.asks = []
        
        result = analyzer.analyze_noise(invalid_order_book)
        
        assert isinstance(result, NoiseAnalysisResult)
        assert result.fractal_dimension == 1.0
        assert result.entropy == 0.0
        assert result.is_synthetic_noise is False
        assert result.confidence == 0.0
        assert result.noise_type == NoiseType.UNKNOWN

    def test_get_analysis_statistics(self, analyzer: NoiseAnalyzer) -> None:
        """Тест получения статистики анализа."""
        stats = analyzer.get_analysis_statistics()
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "synthetic_noise_detections" in stats
        assert "average_processing_time_ms" in stats
        assert "config" in stats
        assert "history_status" in stats
        assert "advanced_metrics_enabled" in stats
        assert "frequency_analysis_enabled" in stats

    def test_reset_history(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест сброса истории."""
        # Заполняем историю
        analyzer._update_history(sample_order_book)
        assert len(analyzer.price_history) > 0
        
        analyzer.reset_history()
        
        assert len(analyzer.price_history) == 0
        assert len(analyzer.volume_history) == 0
        assert len(analyzer.spread_history) == 0
        assert len(analyzer.imbalance_history) == 0

    def test_reset_statistics(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест сброса статистики."""
        # Выполняем анализ
        analyzer.analyze_noise(sample_order_book)
        assert analyzer.statistics["total_analyses"] > 0
        
        analyzer.reset_statistics()
        
        assert analyzer.statistics["total_analyses"] == 0
        assert analyzer.statistics["synthetic_noise_detections"] == 0
        assert analyzer.statistics["average_processing_time_ms"] == 0.0
        assert analyzer.statistics["last_analysis_timestamp"] is None

    def test_advanced_metrics_disabled(self) -> None:
        """Тест работы с отключенными расширенными метриками."""
        analyzer = NoiseAnalyzer(enable_advanced_metrics=False)
        
        result = analyzer.analyze_noise(Mock(spec=OrderBookSnapshot))
        
        assert isinstance(result, NoiseAnalysisResult)
        # Расширенные метрики не должны быть в metadata
        assert "price_volatility" not in result.metadata.get("quality_metrics", {})

    def test_frequency_analysis_disabled(self) -> None:
        """Тест работы с отключенным частотным анализом."""
        analyzer = NoiseAnalyzer(enable_frequency_analysis=False)
        
        result = analyzer.analyze_noise(Mock(spec=OrderBookSnapshot))
        
        assert isinstance(result, NoiseAnalysisResult)
        # Частотные метрики не должны быть в metadata
        assert "dominant_frequency" not in result.metadata.get("quality_metrics", {})

    def test_metadata_structure(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест структуры метаданных."""
        # Заполняем историю
        for _ in range(25):
            analyzer._update_history(sample_order_book)
        
        result = analyzer.analyze_noise(sample_order_book)
        metadata = result.metadata
        
        assert "data_points" in metadata
        assert "confidence" in metadata
        assert "processing_time_ms" in metadata
        assert "algorithm_version" in metadata
        assert "parameters" in metadata
        assert "quality_metrics" in metadata
        
        assert metadata["data_points"] == 25
        assert isinstance(metadata["processing_time_ms"], float)
        assert metadata["processing_time_ms"] > 0.0

    def test_metrics_structure(self, analyzer: NoiseAnalyzer, sample_order_book: OrderBookSnapshot) -> None:
        """Тест структуры метрик шума."""
        # Заполняем историю
        for _ in range(25):
            analyzer._update_history(sample_order_book)
        
        result = analyzer.analyze_noise(sample_order_book)
        metrics = result.metrics
        
        assert "fractal_dimension" in metrics
        assert "entropy" in metrics
        assert "noise_type" in metrics
        assert "synthetic_probability" in metrics
        assert "natural_probability" in metrics
        
        assert isinstance(metrics["fractal_dimension"], float)
        assert isinstance(metrics["entropy"], float)
        assert isinstance(metrics["noise_type"], NoiseType)
        assert isinstance(metrics["synthetic_probability"], float)
        assert isinstance(metrics["natural_probability"], float)
        
        assert 0.0 <= metrics["synthetic_probability"] <= 1.0
        assert 0.0 <= metrics["natural_probability"] <= 1.0 