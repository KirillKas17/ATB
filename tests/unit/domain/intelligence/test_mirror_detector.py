"""
Unit тесты для mirror_detector.py.

Покрывает:
- MirrorDetector - детектор зеркальных паттернов
- Методы предобработки данных
- Вычисление корреляций с лагом
- Детекция зеркальных сигналов
- Построение корреляционных матриц
- Кластеризация зеркальных паттернов
"""

import pytest
from shared.numpy_utils import np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

from domain.intelligence.mirror_detector import MirrorDetector
from domain.type_definitions.intelligence_types import (
    MirrorDetectionConfig,
    MirrorSignal,
    CorrelationMatrix,
    CorrelationMetrics,
    AnalysisMetadata,
)
from domain.value_objects.timestamp import Timestamp
from domain.exceptions.base_exceptions import ValidationError


class TestMirrorDetector:
    """Тесты для MirrorDetector."""

    @pytest.fixture
    def sample_config(self) -> MirrorDetectionConfig:
        """Тестовая конфигурация."""
        return MirrorDetectionConfig(
            min_correlation=0.7,
            max_lag=5,
            min_sample_size=30,
            normalize_data=True,
            remove_trend=True,
            confidence_threshold=0.8,
        )

    @pytest.fixture
    def detector(self, sample_config: MirrorDetectionConfig) -> MirrorDetector:
        """Тестовый детектор."""
        return MirrorDetector(config=sample_config, enable_advanced_metrics=True, enable_clustering=True)

    @pytest.fixture
    def sample_series1(self) -> pd.Series:
        """Тестовый временной ряд 1."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        values = np.random.randn(100).cumsum() + 100
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_series2(self) -> pd.Series:
        """Тестовый временной ряд 2 (зеркальный к первому)."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        values = -np.random.randn(100).cumsum() + 100  # Зеркальный паттерн
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_price_data(self) -> Dict[str, pd.Series]:
        """Тестовые данные цен."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="H")
        return {
            "BTC/USD": pd.Series(np.random.randn(50).cumsum() + 50000, index=dates),
            "ETH/USD": pd.Series(np.random.randn(50).cumsum() + 3000, index=dates),
            "ADA/USD": pd.Series(np.random.randn(50).cumsum() + 0.5, index=dates),
            "DOT/USD": pd.Series(np.random.randn(50).cumsum() + 7, index=dates),
        }

    def test_initialization_default_config(self: "TestMirrorDetector") -> None:
        """Тест инициализации с дефолтной конфигурацией."""
        detector = MirrorDetector()

        assert detector.config is not None
        assert detector.enable_advanced_metrics is True
        assert detector.enable_clustering is True
        assert detector.statistics["total_analyses"] == 0
        assert detector.statistics["mirror_signals_detected"] == 0

    def test_initialization_custom_config(self, sample_config: MirrorDetectionConfig) -> None:
        """Тест инициализации с кастомной конфигурацией."""
        detector = MirrorDetector(config=sample_config, enable_advanced_metrics=False, enable_clustering=False)

        assert detector.config == sample_config
        assert detector.enable_advanced_metrics is False
        assert detector.enable_clustering is False

    def test_preprocess_series_normalize_data(self, detector: MirrorDetector) -> None:
        """Тест предобработки с нормализацией данных."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)  # 50 элементов

        processed = detector._preprocess_series(series)

        assert len(processed) == 50
        # После нормализации среднее должно быть близко к 0
        assert abs(processed.mean()) < 0.1
        # Стандартное отклонение должно быть близко к 1
        assert abs(processed.std() - 1.0) < 0.1

    def test_preprocess_series_remove_trend(self, detector: MirrorDetector) -> None:
        """Тест предобработки с удалением тренда."""
        # Создаем ряд с линейным трендом
        x = np.arange(50)
        series = pd.Series(x + np.random.randn(50) * 0.1)  # Линейный тренд + шум

        processed = detector._preprocess_series(series)

        assert len(processed) == 50
        # После удаления тренда среднее должно быть близко к 0
        assert abs(processed.mean()) < 0.5

    def test_preprocess_series_insufficient_data(self, detector: MirrorDetector) -> None:
        """Тест предобработки с недостаточными данными."""
        series = pd.Series([1, 2, 3])  # Меньше MIN_SAMPLE_SIZE

        processed = detector._preprocess_series(series)

        assert len(processed) == 3
        # Данные не должны измениться при недостаточном количестве

    def test_preprocess_series_with_nan(self, detector: MirrorDetector) -> None:
        """Тест предобработки с NaN значениями."""
        series = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 5)

        processed = detector._preprocess_series(series)

        assert len(processed) == 49  # Один NaN удален
        assert not processed.isna().any()

    def test_compute_correlation_with_lag_valid_data(self, detector: MirrorDetector) -> None:
        """Тест вычисления корреляции с лагом с валидными данными."""
        series1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        series2 = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Лаг 1

        correlation, p_value = detector._compute_correlation_with_lag(series1, series2, 1)

        assert isinstance(correlation, float)
        assert isinstance(p_value, float)
        assert -1.0 <= correlation <= 1.0
        assert 0.0 <= p_value <= 1.0

    def test_compute_correlation_with_lag_zero_lag(self, detector: MirrorDetector) -> None:
        """Тест вычисления корреляции с нулевым лагом."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([1, 2, 3, 4, 5])

        correlation, p_value = detector._compute_correlation_with_lag(series1, series2, 0)

        assert correlation == 1.0  # Полная корреляция
        assert p_value < 0.05  # Статистически значимо

    def test_compute_correlation_with_lag_negative_correlation(self, detector: MirrorDetector) -> None:
        """Тест вычисления корреляции с отрицательной корреляцией."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([5, 4, 3, 2, 1])

        correlation, p_value = detector._compute_correlation_with_lag(series1, series2, 0)

        assert correlation == -1.0  # Полная отрицательная корреляция

    def test_compute_correlation_with_lag_invalid_lag(self, detector: MirrorDetector) -> None:
        """Тест вычисления корреляции с недопустимым лагом."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([1, 2, 3, 4, 5])

        correlation, p_value = detector._compute_correlation_with_lag(series1, series2, 10)

        assert correlation == 0.0
        assert p_value == 1.0

    def test_compute_correlation_with_lag_different_lengths(self, detector: MirrorDetector) -> None:
        """Тест вычисления корреляции с разными длинами рядов."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([1, 2, 3, 4])

        with pytest.raises(ValueError, match="Series must have the same length"):
            detector._compute_correlation_with_lag(series1, series2, 0)

    def test_compute_confidence(self, detector: MirrorDetector) -> None:
        """Тест вычисления уверенности."""
        correlation = 0.8
        p_value = 0.01
        sample_size = 100
        lag = 2

        confidence = detector._compute_confidence(correlation, p_value, sample_size, lag)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_compute_confidence_high_correlation(self, detector: MirrorDetector) -> None:
        """Тест вычисления уверенности с высокой корреляцией."""
        confidence = detector._compute_confidence(0.95, 0.001, 100, 0)

        assert confidence > 0.8  # Высокая уверенность

    def test_compute_confidence_low_correlation(self, detector: MirrorDetector) -> None:
        """Тест вычисления уверенности с низкой корреляцией."""
        confidence = detector._compute_confidence(0.3, 0.1, 100, 0)

        assert confidence < 0.5  # Низкая уверенность

    def test_detect_lagged_correlation(
        self, detector: MirrorDetector, sample_series1: pd.Series, sample_series2: pd.Series
    ) -> None:
        """Тест детекции корреляции с лагом."""
        optimal_lag, max_correlation = detector.detect_lagged_correlation(sample_series1, sample_series2)

        assert isinstance(optimal_lag, int)
        assert isinstance(max_correlation, float)
        assert -5 <= optimal_lag <= 5  # В пределах max_lag
        assert -1.0 <= max_correlation <= 1.0

    def test_detect_lagged_correlation_identical_series(self, detector: MirrorDetector) -> None:
        """Тест детекции корреляции с идентичными рядами."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)

        optimal_lag, max_correlation = detector.detect_lagged_correlation(series, series)

        assert optimal_lag == 0  # Нет лага для идентичных рядов
        assert max_correlation == 1.0  # Полная корреляция

    def test_detect_mirror_signal_valid_data(
        self, detector: MirrorDetector, sample_series1: pd.Series, sample_series2: pd.Series
    ) -> None:
        """Тест детекции зеркального сигнала с валидными данными."""
        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", sample_series1, sample_series2)

        if signal is not None:
            assert isinstance(signal, MirrorSignal)
            assert signal.asset1 == "BTC/USD"
            assert signal.asset2 == "ETH/USD"
            assert isinstance(signal.correlation, float)
            assert isinstance(signal.lag, int)
            assert isinstance(signal.confidence, float)
            assert isinstance(signal.timestamp, Timestamp)
            assert isinstance(signal.metadata, dict)

    def test_detect_mirror_signal_no_correlation(self, detector: MirrorDetector) -> None:
        """Тест детекции зеркального сигнала без корреляции."""
        # Создаем независимые ряды
        series1 = pd.Series(np.random.randn(50))
        series2 = pd.Series(np.random.randn(50))

        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", series1, series2)

        # Сигнал может быть None если корреляция ниже порога
        assert signal is None or isinstance(signal, MirrorSignal)

    def test_detect_mirror_signal_insufficient_data(self, detector: MirrorDetector) -> None:
        """Тест детекции зеркального сигнала с недостаточными данными."""
        series1 = pd.Series([1, 2, 3, 4, 5])  # Меньше min_sample_size
        series2 = pd.Series([1, 2, 3, 4, 5])

        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", series1, series2)

        assert signal is None

    def test_build_correlation_matrix(self, detector: MirrorDetector, sample_price_data: Dict[str, pd.Series]) -> None:
        """Тест построения корреляционной матрицы."""
        assets = list(sample_price_data.keys())

        matrix = detector.build_correlation_matrix(assets, sample_price_data)

        assert isinstance(matrix, CorrelationMatrix)
        assert matrix.assets == assets
        assert isinstance(matrix.correlations, dict)
        assert isinstance(matrix.lags, dict)
        assert isinstance(matrix.confidences, dict)
        assert isinstance(matrix.metadata, dict)

        # Проверяем структуру матрицы
        for asset1 in assets:
            for asset2 in assets:
                key = f"{asset1}_{asset2}"
                if key in matrix.correlations:
                    assert isinstance(matrix.correlations[key], float)
                    assert isinstance(matrix.lags[key], int)
                    assert isinstance(matrix.confidences[key], float)

    def test_build_correlation_matrix_empty_data(self, detector: MirrorDetector) -> None:
        """Тест построения корреляционной матрицы с пустыми данными."""
        matrix = detector.build_correlation_matrix([], {})

        assert isinstance(matrix, CorrelationMatrix)
        assert matrix.assets == []
        assert matrix.correlations == {}
        assert matrix.lags == {}
        assert matrix.confidences == {}

    def test_find_mirror_clusters(self, detector: MirrorDetector, sample_price_data: Dict[str, pd.Series]) -> None:
        """Тест поиска зеркальных кластеров."""
        assets = list(sample_price_data.keys())
        matrix = detector.build_correlation_matrix(assets, sample_price_data)

        clusters = detector.find_mirror_clusters(matrix)

        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, list)
            assert all(isinstance(asset, str) for asset in cluster)
            assert len(cluster) >= 1

    def test_find_mirror_clusters_high_threshold(
        self, detector: MirrorDetector, sample_price_data: Dict[str, pd.Series]
    ) -> None:
        """Тест поиска зеркальных кластеров с высоким порогом."""
        assets = list(sample_price_data.keys())
        matrix = detector.build_correlation_matrix(assets, sample_price_data)

        clusters = detector.find_mirror_clusters(matrix, min_correlation=0.95)

        assert isinstance(clusters, list)
        # При высоком пороге кластеров может быть меньше

    def test_get_detector_statistics(self, detector: MirrorDetector) -> None:
        """Тест получения статистики детектора."""
        stats = detector.get_detector_statistics()

        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "mirror_signals_detected" in stats
        assert "average_processing_time_ms" in stats
        assert "last_analysis_timestamp" in stats

    def test_reset_statistics(
        self, detector: MirrorDetector, sample_series1: pd.Series, sample_series2: pd.Series
    ) -> None:
        """Тест сброса статистики."""
        # Выполняем анализ для накопления статистики
        detector.detect_mirror_signal("BTC/USD", "ETH/USD", sample_series1, sample_series2)
        assert detector.statistics["total_analyses"] > 0

        detector.reset_statistics()

        assert detector.statistics["total_analyses"] == 0
        assert detector.statistics["mirror_signals_detected"] == 0
        assert detector.statistics["average_processing_time_ms"] == 0.0
        assert detector.statistics["last_analysis_timestamp"] is None

    def test_advanced_metrics_disabled(self: "TestMirrorDetector") -> None:
        """Тест работы с отключенными расширенными метриками."""
        detector = MirrorDetector(enable_advanced_metrics=False)

        signal = detector.detect_mirror_signal(
            "BTC/USD", "ETH/USD", pd.Series([1, 2, 3, 4, 5] * 10), pd.Series([1, 2, 3, 4, 5] * 10)
        )

        if signal is not None:
            assert isinstance(signal, MirrorSignal)
            # Расширенные метрики не должны быть в metadata
            assert "advanced_metrics" not in signal.metadata

    def test_clustering_disabled(self: "TestMirrorDetector") -> None:
        """Тест работы с отключенной кластеризацией."""
        detector = MirrorDetector(enable_clustering=False)

        # Создаем простую корреляционную матрицу
        matrix = Mock(spec=CorrelationMatrix)
        matrix.assets = ["BTC/USD", "ETH/USD"]
        matrix.correlations = {"BTC/USD_ETH/USD": 0.8}
        matrix.lags = {"BTC/USD_ETH/USD": 0}
        matrix.confidences = {"BTC/USD_ETH/USD": 0.9}

        clusters = detector.find_mirror_clusters(matrix)

        assert isinstance(clusters, list)

    def test_mirror_signal_structure(
        self, detector: MirrorDetector, sample_series1: pd.Series, sample_series2: pd.Series
    ) -> None:
        """Тест структуры зеркального сигнала."""
        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", sample_series1, sample_series2)

        if signal is not None:
            assert signal.asset1 == "BTC/USD"
            assert signal.asset2 == "ETH/USD"
            assert isinstance(signal.correlation, float)
            assert -1.0 <= signal.correlation <= 1.0
            assert isinstance(signal.lag, int)
            assert isinstance(signal.confidence, float)
            assert 0.0 <= signal.confidence <= 1.0
            assert isinstance(signal.timestamp, Timestamp)
            assert isinstance(signal.metadata, dict)

    def test_correlation_matrix_structure(
        self, detector: MirrorDetector, sample_price_data: Dict[str, pd.Series]
    ) -> None:
        """Тест структуры корреляционной матрицы."""
        assets = list(sample_price_data.keys())
        matrix = detector.build_correlation_matrix(assets, sample_price_data)

        assert matrix.assets == assets
        assert isinstance(matrix.correlations, dict)
        assert isinstance(matrix.lags, dict)
        assert isinstance(matrix.confidences, dict)
        assert isinstance(matrix.metadata, dict)

        # Проверяем метаданные
        assert "algorithm_version" in matrix.metadata
        assert "processing_time_ms" in matrix.metadata
        assert "total_pairs" in matrix.metadata

    def test_error_handling_invalid_data(self, detector: MirrorDetector) -> None:
        """Тест обработки ошибок с невалидными данными."""
        # Тест с пустыми рядами
        empty_series = pd.Series([])

        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", empty_series, empty_series)

        assert signal is None

    def test_performance_with_large_data(self, detector: MirrorDetector) -> None:
        """Тест производительности с большими данными."""
        # Создаем большие ряды
        large_series1 = pd.Series(np.random.randn(1000))
        large_series2 = pd.Series(np.random.randn(1000))

        start_time = datetime.now()
        signal = detector.detect_mirror_signal("BTC/USD", "ETH/USD", large_series1, large_series2)
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # Обработка должна быть быстрой (менее 1 секунды)
        assert processing_time < 1.0
        assert signal is None or isinstance(signal, MirrorSignal)
