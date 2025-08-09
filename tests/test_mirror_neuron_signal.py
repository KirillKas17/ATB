# -*- coding: utf-8 -*-
"""Тесты для системы Mirror Neuron Signal Detection."""
import time
from unittest.mock import Mock
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.strategy_advisor.mirror_map_builder import MirrorMap, MirrorMapBuilder, MirrorMapConfig
from domain.intelligence.mirror_detector import CorrelationMatrix, MirrorDetector, MirrorSignal
from domain.value_objects.timestamp import Timestamp


class TestMirrorDetector:
    """Тесты для MirrorDetector."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.detector = MirrorDetector(
            min_correlation=0.3,
            max_p_value=0.05,
            min_confidence=0.7,
            max_lag=5,
            correlation_method="pearson",
            normalize_data=True,
            remove_trend=True,
        )

    def test_initialization(self: "TestMirrorDetector") -> None:
        """Тест инициализации детектора."""
        assert self.detector.min_correlation == 0.3
        assert self.detector.max_p_value == 0.05
        assert self.detector.min_confidence == 0.7
        assert self.detector.correlation_method == "pearson"
        assert self.detector.normalize_data == True
        assert self.detector.remove_trend == True

    def test_preprocess_series(self: "TestMirrorDetector") -> None:
        """Тест предобработки временного ряда."""
        # Создаем тестовый ряд с трендом
        series = pd.Series([100 + i + np.random.normal(0, 1) for i in range(100)])
        processed = self.detector._preprocess_series(series)
        assert len(processed) > 0
        assert not processed.isna().any()
        # Проверяем, что тренд удален (если включено)
        if self.detector.remove_trend:
            # После удаления тренда среднее должно быть близко к нулю
            assert abs(processed.mean()) < 1.0

    def test_preprocess_series_empty(self: "TestMirrorDetector") -> None:
        """Тест предобработки пустого ряда."""
        empty_series = pd.Series(dtype=float)
        processed = self.detector._preprocess_series(empty_series)
        assert len(processed) == 0

    def test_preprocess_series_with_nan(self: "TestMirrorDetector") -> None:
        """Тест предобработки ряда с NaN значениями."""
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        processed = self.detector._preprocess_series(series_with_nan)
        assert not processed.isna().any()
        assert len(processed) < len(series_with_nan)

    def test_compute_correlation_with_lag(self: "TestMirrorDetector") -> None:
        """Тест вычисления корреляции с лагом."""
        # Создаем коррелированные ряды
        np.random.seed(42)
        series1 = pd.Series(np.random.randn(100))
        series2 = pd.Series(series1.shift(2) + np.random.randn(100) * 0.1)  # Лаг 2
        correlation, p_value = self.detector._compute_correlation_with_lag(series1, series2, 2)
        assert isinstance(correlation, float)
        assert isinstance(p_value, float)
        assert -1 <= correlation <= 1
        assert 0 <= p_value <= 1

    def test_compute_correlation_with_lag_negative(self: "TestMirrorDetector") -> None:
        """Тест вычисления корреляции с отрицательным лагом."""
        np.random.seed(42)
        series1 = pd.Series(np.random.randn(100))
        series2 = pd.Series(series1.shift(-2) + np.random.randn(100) * 0.1)  # Отрицательный лаг
        correlation, p_value = self.detector._compute_correlation_with_lag(series1, series2, -2)
        assert isinstance(correlation, float)
        assert isinstance(p_value, float)

    def test_compute_correlation_with_lag_insufficient_data(self: "TestMirrorDetector") -> None:
        """Тест корреляции с недостаточными данными."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([2, 3, 4, 5, 6])
        correlation, p_value = self.detector._compute_correlation_with_lag(series1, series2, 10)
        assert correlation == 0.0
        assert p_value == 1.0

    def test_detect_lagged_correlation(self: "TestMirrorDetector") -> None:
        """Тест обнаружения корреляции с лагом."""
        # Создаем коррелированные ряды
        np.random.seed(42)
        series1 = pd.Series(np.random.randn(100))
        series2 = pd.Series(series1.shift(2) + np.random.randn(100) * 0.1)
        best_lag, correlation = self.detector.detect_lagged_correlation(series1, series2, max_lag=5)
        assert isinstance(best_lag, int)
        assert isinstance(correlation, float)
        assert -5 <= best_lag <= 5
        assert -1 <= correlation <= 1

    def test_detect_lagged_correlation_identical(self: "TestMirrorDetector") -> None:
        """Тест корреляции идентичных рядов."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)  # Повторяем для достаточной длины
        best_lag, correlation = self.detector.detect_lagged_correlation(series, series, max_lag=5)
        assert best_lag == 0  # Нет лага для идентичных рядов
        assert abs(correlation - 1.0) < 0.01  # Корреляция близка к 1

    def test_detect_lagged_correlation_insufficient_data(self: "TestMirrorDetector") -> None:
        """Тест корреляции с недостаточными данными."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([2, 3, 4, 5, 6])
        best_lag, correlation = self.detector.detect_lagged_correlation(series1, series2, max_lag=5)
        assert best_lag == 0
        assert correlation == 0.0

    def test_detect_mirror_signal(self: "TestMirrorDetector") -> None:
        """Тест обнаружения зеркального сигнала."""
        # Создаем коррелированные ряды
        np.random.seed(42)
        series1 = pd.Series(np.random.randn(100))
        series2 = pd.Series(series1.shift(2) + np.random.randn(100) * 0.1)
        signal = self.detector.detect_mirror_signal("BTC", "ETH", series1, series2, max_lag=5)
        if signal:  # Сигнал может быть обнаружен или нет в зависимости от данных
            assert isinstance(signal, MirrorSignal)
            assert signal.asset1 == "BTC"
            assert signal.asset2 == "ETH"
            assert isinstance(signal.best_lag, int)
            assert isinstance(signal.correlation, float)
            assert isinstance(signal.p_value, float)
            assert isinstance(signal.confidence, float)
            assert isinstance(signal.signal_strength, float)
            assert isinstance(signal.timestamp, Timestamp)
            assert isinstance(signal.metadata, dict)

    def test_detect_mirror_signal_no_correlation(self: "TestMirrorDetector") -> None:
        """Тест обнаружения сигнала без корреляции."""
        # Создаем некоррелированные ряды
        np.random.seed(42)
        series1 = pd.Series(np.random.randn(100))
        series2 = pd.Series(np.random.randn(100))
        signal = self.detector.detect_mirror_signal("BTC", "ETH", series1, series2, max_lag=5)
        # Сигнал не должен быть обнаружен
        assert signal is None

    def test_compute_confidence(self: "TestMirrorDetector") -> None:
        """Тест вычисления уверенности."""
        confidence = self.detector._compute_confidence(0.8, 0.01, 1000)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_build_correlation_matrix(self: "TestMirrorDetector") -> None:
        """Тест построения матрицы корреляций."""
        # Создаем тестовые данные
        assets = ["BTC", "ETH", "ADA"]
        price_data = {
            "BTC": pd.Series(np.random.randn(100)),
            "ETH": pd.Series(np.random.randn(100)),
            "ADA": pd.Series(np.random.randn(100)),
        }
        matrix = self.detector.build_correlation_matrix(assets, price_data, max_lag=5)
        assert isinstance(matrix, CorrelationMatrix)
        assert matrix.assets == assets
        assert matrix.correlation_matrix.shape == (3, 3)
        assert matrix.lag_matrix.shape == (3, 3)
        assert matrix.p_value_matrix.shape == (3, 3)
        assert matrix.confidence_matrix.shape == (3, 3)
        # Проверяем диагональ
        for i in range(3):
            assert matrix.correlation_matrix[i, i] == 1.0
            assert matrix.lag_matrix[i, i] == 0
            assert matrix.p_value_matrix[i, i] == 0.0
            assert matrix.confidence_matrix[i, i] == 1.0

    def test_correlation_matrix_methods(self: "TestMirrorDetector") -> None:
        """Тест методов CorrelationMatrix."""
        # Создаем тестовую матрицу
        assets = ["BTC", "ETH"]
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        lag_matrix = np.array([[0, 2], [-2, 0]])
        p_value_matrix = np.array([[0.0, 0.01], [0.01, 0.0]])
        confidence_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=correlation_matrix,
            lag_matrix=lag_matrix,
            p_value_matrix=p_value_matrix,
            confidence_matrix=confidence_matrix,
            timestamp=Timestamp(time.time()),
        )
        # Тестируем методы
        assert matrix.get_correlation("BTC", "ETH") == 0.5
        assert matrix.get_lag("BTC", "ETH") == 2
        assert matrix.get_p_value("BTC", "ETH") == 0.01
        assert matrix.get_confidence("BTC", "ETH") == 0.8
        # Тестируем обратную корреляцию
        assert matrix.get_correlation("ETH", "BTC") == 0.5
        assert matrix.get_lag("ETH", "BTC") == -2

    def test_find_mirror_clusters(self: "TestMirrorDetector") -> None:
        """Тест поиска кластеров зеркальных активов."""
        # Создаем тестовую матрицу с кластерами
        assets = ["BTC", "ETH", "ADA", "DOT"]
        correlation_matrix = np.array(
            [
                [1.0, 0.8, 0.1, 0.1],
                [0.8, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.7],
                [0.1, 0.1, 0.7, 1.0],
            ]
        )
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=correlation_matrix,
            lag_matrix=np.zeros((4, 4), dtype=int),
            p_value_matrix=np.ones((4, 4)) * 0.01,
            confidence_matrix=np.ones((4, 4)),
            timestamp=Timestamp(time.time()),
        )
        clusters = self.detector.find_mirror_clusters(matrix, min_correlation=0.5)
        assert len(clusters) == 2  # Два кластера: [BTC, ETH] и [ADA, DOT]
        assert ["BTC", "ETH"] in clusters or ["ETH", "BTC"] in clusters
        assert ["ADA", "DOT"] in clusters or ["DOT", "ADA"] in clusters

    def test_get_detector_statistics(self: "TestMirrorDetector") -> None:
        """Тест получения статистики детектора."""
        stats = self.detector.get_detector_statistics()
        assert "min_correlation" in stats
        assert "max_p_value" in stats
        assert "min_confidence" in stats
        assert "correlation_method" in stats
        assert "normalize_data" in stats
        assert "remove_trend" in stats


class TestMirrorMapBuilder:
    """Тесты для MirrorMapBuilder."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = MirrorMapConfig(
            min_correlation=0.3,
            max_p_value=0.05,
            min_confidence=0.7,
            max_lag=5,
            correlation_method="pearson",
            normalize_data=True,
            remove_trend=True,
            min_cluster_size=2,
            max_cluster_size=10,
            update_interval=3600,
            parallel_processing=True,
            max_workers=4,
        )
        self.builder = MirrorMapBuilder(self.config)

    def test_initialization(self: "TestMirrorMapBuilder") -> None:
        """Тест инициализации построителя."""
        assert self.builder.config.min_correlation == 0.3
        assert self.builder.config.max_p_value == 0.05
        assert self.builder.config.parallel_processing == True
        assert self.builder.config.max_workers == 4
        assert self.builder._mirror_map_cache is None
        assert self.builder._last_update is None

    def test_validate_price_data(self: "TestMirrorMapBuilder") -> None:
        """Тест валидации данных о ценах."""
        assets = ["BTC", "ETH", "ADA"]
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2001, 2002, 2003, 2004]),
            "ADA": pd.Series([0.5, 0.51, 0.52, 0.53, 0.54]),
        }
        valid_data = self.builder._validate_price_data(assets, price_data)
        assert len(valid_data) == 3
        assert "BTC" in valid_data
        assert "ETH" in valid_data
        assert "ADA" in valid_data

    def test_validate_price_data_missing_assets(self: "TestMirrorMapBuilder") -> None:
        """Тест валидации с отсутствующими активами."""
        assets = ["BTC", "ETH", "ADA", "DOT"]
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2001, 2002, 2003, 2004]),
        }
        valid_data = self.builder._validate_price_data(assets, price_data)
        assert len(valid_data) == 2
        assert "BTC" in valid_data
        assert "ETH" in valid_data
        assert "ADA" not in valid_data
        assert "DOT" not in valid_data

    def test_validate_price_data_insufficient_length(self: "TestMirrorMapBuilder") -> None:
        """Тест валидации с недостаточной длиной данных."""
        assets = ["BTC", "ETH"]
        price_data = {
            "BTC": pd.Series([100, 101, 102]),  # Только 3 точки
            "ETH": pd.Series([2000, 2001, 2002]),
        }
        valid_data = self.builder._validate_price_data(assets, price_data)
        assert len(valid_data) == 0  # Недостаточно данных

    def test_validate_price_data_all_nan(self: "TestMirrorMapBuilder") -> None:
        """Тест валидации с полностью NaN данными."""
        assets = ["BTC"]
        price_data = {"BTC": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])}
        valid_data = self.builder._validate_price_data(assets, price_data)
        assert len(valid_data) == 0  # Все данные NaN

    def test_build_mirror_map(self: "TestMirrorMapBuilder") -> None:
        """Тест построения карты зеркальных зависимостей."""
        # Создаем тестовые данные
        assets = ["BTC", "ETH", "ADA"]
        price_data = {
            "BTC": pd.Series(np.random.randn(100)),
            "ETH": pd.Series(np.random.randn(100)),
            "ADA": pd.Series(np.random.randn(100)),
        }
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        assert isinstance(mirror_map, MirrorMap)
        assert mirror_map.assets == assets
        assert isinstance(mirror_map.mirror_map, dict)
        assert isinstance(mirror_map.clusters, list)
        assert isinstance(mirror_map.metadata, dict)
        assert isinstance(mirror_map.timestamp, Timestamp)
        assert mirror_map.correlation_matrix is not None

    def test_build_mirror_map_cache(self: "TestMirrorMapBuilder") -> None:
        """Тест кэширования карты."""
        assets = ["BTC", "ETH"]
        price_data = {
            "BTC": pd.Series(np.random.randn(100)),
            "ETH": pd.Series(np.random.randn(100)),
        }
        # Первое построение
        mirror_map1 = self.builder.build_mirror_map(assets, price_data, force_rebuild=False)
        # Второе построение (должно использовать кэш)
        mirror_map2 = self.builder.build_mirror_map(assets, price_data, force_rebuild=False)
        assert mirror_map1 is mirror_map2  # Тот же объект из кэша

    def test_build_mirror_map_force_rebuild(self: "TestMirrorMapBuilder") -> None:
        """Тест принудительного перестроения карты."""
        assets = ["BTC", "ETH"]
        price_data = {
            "BTC": pd.Series(np.random.randn(100)),
            "ETH": pd.Series(np.random.randn(100)),
        }
        # Первое построение
        mirror_map1 = self.builder.build_mirror_map(assets, price_data, force_rebuild=False)
        # Принудительное перестроение
        mirror_map2 = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        assert mirror_map1 is not mirror_map2  # Разные объекты

    @pytest.mark.asyncio
    async def test_build_mirror_map_async(self: "TestMirrorMapBuilder") -> None:
        """Тест асинхронного построения карты."""
        assets = ["BTC", "ETH"]
        price_data = {
            "BTC": pd.Series(np.random.randn(100)),
            "ETH": pd.Series(np.random.randn(100)),
        }
        mirror_map = await self.builder.build_mirror_map_async(assets, price_data, force_rebuild=True)
        assert isinstance(mirror_map, MirrorMap)
        assert mirror_map.assets == assets

    def test_get_mirror_assets_for_strategy(self: "TestMirrorMapBuilder") -> None:
        """Тест получения зеркальных активов для стратегии."""
        # Создаем тестовую карту
        assets = ["BTC", "ETH", "ADA"]
        correlation_matrix = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.1], [0.1, 0.1, 1.0]])
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=correlation_matrix,
            lag_matrix=np.zeros((3, 3), dtype=int),
            p_value_matrix=np.ones((3, 3)) * 0.01,
            confidence_matrix=np.ones((3, 3)),
            timestamp=Timestamp(time.time()),
        )
        mirror_map = MirrorMap(
            assets=assets,
            mirror_map={"BTC": ["ETH"], "ETH": ["BTC"]},
            correlation_matrix=matrix,
            clusters=[["BTC", "ETH"]],
        )
        mirror_assets = self.builder.get_mirror_assets_for_strategy(mirror_map, "BTC", min_correlation=0.5)
        assert len(mirror_assets) == 1
        assert mirror_assets[0][0] == "ETH"
        assert mirror_assets[0][1] == 0.8  # Корреляция

    def test_analyze_mirror_clusters(self: "TestMirrorMapBuilder") -> None:
        """Тест анализа кластеров зеркальных активов."""
        # Создаем тестовую карту с кластерами
        assets = ["BTC", "ETH", "ADA", "DOT"]
        correlation_matrix = np.array(
            [
                [1.0, 0.8, 0.1, 0.1],
                [0.8, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.7],
                [0.1, 0.1, 0.7, 1.0],
            ]
        )
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=correlation_matrix,
            lag_matrix=np.zeros((4, 4), dtype=int),
            p_value_matrix=np.ones((4, 4)) * 0.01,
            confidence_matrix=np.ones((4, 4)),
            timestamp=Timestamp(time.time()),
        )
        mirror_map = MirrorMap(
            assets=assets,
            mirror_map={"BTC": ["ETH"], "ETH": ["BTC"], "ADA": ["DOT"], "DOT": ["ADA"]},
            correlation_matrix=matrix,
            clusters=[["BTC", "ETH"], ["ADA", "DOT"]],
        )
        analysis = self.builder.analyze_mirror_clusters(mirror_map)
        assert analysis["total_clusters"] == 2
        assert analysis["largest_cluster"] == 2
        assert analysis["average_cluster_size"] == 2.0
        assert len(analysis["cluster_details"]) == 2

    def test_get_mirror_map_statistics(self: "TestMirrorMapBuilder") -> None:
        """Тест получения статистики построителя."""
        stats = self.builder.get_mirror_map_statistics()
        assert "config" in stats
        assert "detector_stats" in stats
        assert "cache_info" in stats
        assert stats["cache_info"]["has_cache"] == False

    def test_clear_cache(self: "TestMirrorMapBuilder") -> None:
        """Тест очистки кэша."""
        # Заполняем кэш
        self.builder._mirror_map_cache = Mock()
        self.builder._last_update = time.time()
        self.builder.clear_cache()
        assert self.builder._mirror_map_cache is None
        assert self.builder._last_update is None

    def test_update_config(self: "TestMirrorMapBuilder") -> None:
        """Тест обновления конфигурации."""
        new_config = MirrorMapConfig(min_correlation=0.5, max_p_value=0.01, max_lag=10)
        self.builder.update_config(new_config)
        assert self.builder.config.min_correlation == 0.5
        assert self.builder.config.max_p_value == 0.01
        assert self.builder.config.max_lag == 10
        assert self.builder._mirror_map_cache is None  # Кэш очищен


class TestMirrorMap:
    """Тесты для MirrorMap."""

    def test_creation(self: "TestMirrorMap") -> None:
        """Тест создания карты."""
        assets = ["BTC", "ETH"]
        mirror_map = MirrorMap(
            assets=assets,
            mirror_map={"BTC": ["ETH"], "ETH": ["BTC"]},
            clusters=[["BTC", "ETH"]],
        )
        assert mirror_map.assets == assets
        assert mirror_map.mirror_map == {"BTC": ["ETH"], "ETH": ["BTC"]}
        assert mirror_map.clusters == [["BTC", "ETH"]]
        assert isinstance(mirror_map.timestamp, Timestamp)

    def test_get_mirror_assets(self: "TestMirrorMap") -> None:
        """Тест получения зеркальных активов."""
        mirror_map = MirrorMap(assets=["BTC", "ETH"], mirror_map={"BTC": ["ETH"], "ETH": ["BTC"]})
        assert mirror_map.get_mirror_assets("BTC") == ["ETH"]
        assert mirror_map.get_mirror_assets("ETH") == ["BTC"]
        assert mirror_map.get_mirror_assets("ADA") == []  # Несуществующий актив

    def test_is_mirror_pair(self: "TestMirrorMap") -> None:
        """Тест проверки зеркальной пары."""
        mirror_map = MirrorMap(assets=["BTC", "ETH"], mirror_map={"BTC": ["ETH"], "ETH": ["BTC"]})
        assert mirror_map.is_mirror_pair("BTC", "ETH") == True
        assert mirror_map.is_mirror_pair("ETH", "BTC") == True
        assert mirror_map.is_mirror_pair("BTC", "ADA") == False

    def test_get_correlation(self: "TestMirrorMap") -> None:
        """Тест получения корреляции."""
        assets = ["BTC", "ETH"]
        correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=correlation_matrix,
            lag_matrix=np.zeros((2, 2), dtype=int),
            p_value_matrix=np.ones((2, 2)),
            confidence_matrix=np.ones((2, 2)),
            timestamp=Timestamp(time.time()),
        )
        mirror_map = MirrorMap(assets=assets, mirror_map={"BTC": ["ETH"]}, correlation_matrix=matrix)
        assert mirror_map.get_correlation("BTC", "ETH") == 0.8
        assert mirror_map.get_correlation("ETH", "BTC") == 0.8
        assert mirror_map.get_correlation("BTC", "ADA") == 0.0  # Несуществующий актив

    def test_get_lag(self: "TestMirrorMap") -> None:
        """Тест получения лага."""
        assets = ["BTC", "ETH"]
        lag_matrix = np.array([[0, 2], [-2, 0]])
        matrix = CorrelationMatrix(
            assets=assets,
            correlation_matrix=np.ones((2, 2)),
            lag_matrix=lag_matrix,
            p_value_matrix=np.ones((2, 2)),
            confidence_matrix=np.ones((2, 2)),
            timestamp=Timestamp(time.time()),
        )
        mirror_map = MirrorMap(assets=assets, mirror_map={"BTC": ["ETH"]}, correlation_matrix=matrix)
        assert mirror_map.get_lag("BTC", "ETH") == 2
        assert mirror_map.get_lag("ETH", "BTC") == -2
        assert mirror_map.get_lag("BTC", "ADA") == 0  # Несуществующий актив

    def test_to_dict(self: "TestMirrorMap") -> None:
        """Тест преобразования в словарь."""
        mirror_map = MirrorMap(
            assets=["BTC", "ETH"],
            mirror_map={"BTC": ["ETH"]},
            clusters=[["BTC", "ETH"]],
            metadata={"test": "data"},
        )
        data = mirror_map.to_dict()
        assert data["assets"] == ["BTC", "ETH"]
        assert data["mirror_map"] == {"BTC": ["ETH"]}
        assert data["clusters"] == [["BTC", "ETH"]]
        assert data["metadata"] == {"test": "data"}
        assert "timestamp" in data


class TestMirrorMapConfig:
    """Тесты для MirrorMapConfig."""

    def test_default_values(self: "TestMirrorMapConfig") -> None:
        """Тест значений по умолчанию."""
        config = MirrorMapConfig()
        assert config.min_correlation == 0.3
        assert config.max_p_value == 0.05
        assert config.min_confidence == 0.7
        assert config.max_lag == 5
        assert config.correlation_method == "pearson"
        assert config.normalize_data == True
        assert config.remove_trend == True
        assert config.min_cluster_size == 2
        assert config.max_cluster_size == 10
        assert config.update_interval == 3600
        assert config.parallel_processing == True
        assert config.max_workers == 4

    def test_custom_values(self: "TestMirrorMapConfig") -> None:
        """Тест пользовательских значений."""
        config = MirrorMapConfig(
            min_correlation=0.5,
            max_p_value=0.01,
            min_confidence=0.8,
            max_lag=10,
            correlation_method="spearman",
            normalize_data=False,
            remove_trend=False,
            min_cluster_size=3,
            max_cluster_size=20,
            update_interval=7200,
            parallel_processing=False,
            max_workers=8,
        )
        assert config.min_correlation == 0.5
        assert config.max_p_value == 0.01
        assert config.min_confidence == 0.8
        assert config.max_lag == 10
        assert config.correlation_method == "spearman"
        assert config.normalize_data == False
        assert config.remove_trend == False
        assert config.min_cluster_size == 3
        assert config.max_cluster_size == 20
        assert config.update_interval == 7200
        assert config.parallel_processing == False
        assert config.max_workers == 8


if __name__ == "__main__":
    pytest.main([__file__])
