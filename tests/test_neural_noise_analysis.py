# -*- coding: utf-8 -*-
"""Тесты для системы анализа нейронного шума ордербука."""

import time

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from application.filters.orderbook_filter import (FilterConfig,
                                                  OrderBookPreFilter)
from domain.intelligence.noise_analyzer import (NoiseAnalysisResult,
                                                NoiseAnalyzer,
                                                OrderBookSnapshot)
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


class TestNoiseAnalyzer:
    """Тесты для NoiseAnalyzer."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.analyzer = NoiseAnalyzer(
            fractal_dimension_lower=1.2,
            fractal_dimension_upper=1.4,
            entropy_threshold=0.7,
            min_data_points=20,
            window_size=50,
        )

    def test_initialization(self: "TestNoiseAnalyzer") -> None:
        """Тест инициализации анализатора."""
        assert self.analyzer.fractal_dimension_lower == 1.2
        assert self.analyzer.fractal_dimension_upper == 1.4
        assert self.analyzer.entropy_threshold == 0.7
        assert self.analyzer.min_data_points == 20
        assert self.analyzer.window_size == 50
        assert len(self.analyzer.price_history) == 0

    def test_create_order_book_snapshot(self: "TestNoiseAnalyzer") -> None:
        """Тест создания снимка ордербука."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        assert order_book.exchange == "test"
        assert order_book.symbol == "BTCUSDT"
        assert len(order_book.bids) == 1
        assert len(order_book.asks) == 1
        assert order_book.meta is not None

    def test_get_mid_price(self: "TestNoiseAnalyzer") -> None:
        """Тест получения средней цены."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        mid_price = order_book.get_mid_price()
        assert mid_price.value == 50005.0

    def test_get_spread(self: "TestNoiseAnalyzer") -> None:
        """Тест получения спреда."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        spread = order_book.get_spread()
        assert spread.value == 10.0

    def test_get_total_volume(self: "TestNoiseAnalyzer") -> None:
        """Тест получения общего объема."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0)), (Price(49990.0), Volume(2.0))],
            asks=[(Price(50010.0), Volume(1.5)), (Price(50020.0), Volume(2.5))],
            timestamp=Timestamp(time.time()),
        )

        total_volume = order_book.get_total_volume()
        assert total_volume.value == 7.0  # 1.0 + 2.0 + 1.5 + 2.5

    def test_get_volume_imbalance(self: "TestNoiseAnalyzer") -> None:
        """Тест получения дисбаланса объемов."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(3.0))],
            timestamp=Timestamp(time.time()),
        )

        imbalance = order_book.get_volume_imbalance()
        assert imbalance == -0.5  # (1.0 - 3.0) / (1.0 + 3.0) = -0.5

    def test_compute_entropy_uniform(self: "TestNoiseAnalyzer") -> None:
        """Тест вычисления энтропии для равномерного распределения."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0)), (Price(49990.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0)), (Price(50020.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        entropy = self.analyzer.compute_entropy(order_book)
        assert entropy > 0.8  # Высокая энтропия для равномерного распределения

    def test_compute_entropy_skewed(self: "TestNoiseAnalyzer") -> None:
        """Тест вычисления энтропии для неравномерного распределения."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(10.0)), (Price(49990.0), Volume(0.1))],
            asks=[(Price(50010.0), Volume(10.0)), (Price(50020.0), Volume(0.1))],
            timestamp=Timestamp(time.time()),
        )

        entropy = self.analyzer.compute_entropy(order_book)
        assert entropy < 0.5  # Низкая энтропия для неравномерного распределения

    def test_is_synthetic_noise(self: "TestNoiseAnalyzer") -> None:
        """Тест определения синтетического шума."""
        # Тест с синтетическими параметрами
        result = self.analyzer.is_synthetic_noise(fd=1.3, entropy=0.5)
        assert result == True

        # Тест с естественными параметрами
        result = self.analyzer.is_synthetic_noise(fd=1.8, entropy=0.8)
        assert result == False

        # Тест с граничными значениями
        result = self.analyzer.is_synthetic_noise(fd=1.2, entropy=0.7)
        assert result == True  # entropy < 0.7

    def test_compute_confidence(self: "TestNoiseAnalyzer") -> None:
        """Тест вычисления уверенности."""
        # Высокая уверенность для идеальных параметров
        confidence = self.analyzer.compute_confidence(fd=1.3, entropy=0.3)
        assert confidence > 0.8

        # Низкая уверенность для плохих параметров
        confidence = self.analyzer.compute_confidence(fd=2.0, entropy=0.9)
        assert confidence < 0.5

    def test_analyze_noise_insufficient_data(self: "TestNoiseAnalyzer") -> None:
        """Тест анализа с недостаточными данными."""
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        result = self.analyzer.analyze_noise(order_book)
        assert result.fractal_dimension == 1.0  # Значение по умолчанию
        assert result.is_synthetic_noise == False

    def test_analyze_noise_with_history(self: "TestNoiseAnalyzer") -> None:
        """Тест анализа с накопленной историей."""
        # Создаем ордербук с фрактальными свойствами
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )

        # Анализируем несколько раз для накопления истории
        for i in range(30):
            self.analyzer.analyze_noise(order_book)

        # Проверяем, что история накопилась
        assert len(self.analyzer.price_history) > 0
        assert len(self.analyzer.volume_history) > 0
        assert len(self.analyzer.spread_history) > 0

    def test_get_analysis_statistics(self: "TestNoiseAnalyzer") -> None:
        """Тест получения статистики анализа."""
        stats = self.analyzer.get_analysis_statistics()

        assert "price_history_length" in stats
        assert "fractal_dimension_range" in stats
        assert "entropy_threshold" in stats
        assert stats["fractal_dimension_range"] == [1.2, 1.4]
        assert stats["entropy_threshold"] == 0.7

    def test_reset_history(self: "TestNoiseAnalyzer") -> None:
        """Тест сброса истории."""
        # Добавляем данные в историю
        self.analyzer.price_history.append(50000.0)
        self.analyzer.volume_history.append(1.0)
        self.analyzer.spread_history.append(10.0)

        assert len(self.analyzer.price_history) > 0

        # Сбрасываем историю
        self.analyzer.reset_history()

        assert len(self.analyzer.price_history) == 0
        assert len(self.analyzer.volume_history) == 0
        assert len(self.analyzer.spread_history) == 0


class TestOrderBookPreFilter:
    """Тесты для OrderBookPreFilter."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = FilterConfig(
            enabled=True,
            fractal_dimension_lower=1.2,
            fractal_dimension_upper=1.4,
            entropy_threshold=0.7,
            min_data_points=20,
            window_size=50,
            log_filtered=True,
            log_analysis=False,
        )
        self.filter_obj = OrderBookPreFilter(self.config)

    def test_initialization(self: "TestOrderBookPreFilter") -> None:
        """Тест инициализации фильтра."""
        assert self.filter_obj.config.enabled == True
        assert self.filter_obj.config.fractal_dimension_lower == 1.2
        assert self.filter_obj.config.entropy_threshold == 0.7
        assert self.filter_obj.stats["total_processed"] == 0

    def test_create_order_book_snapshot(self: "TestOrderBookPreFilter") -> None:
        """Тест создания снимка ордербука."""
        bids = [(50000.0, 1.0), (49990.0, 2.0)]
        asks = [(50010.0, 1.5), (50020.0, 2.5)]

        snapshot = self.filter_obj._create_order_book_snapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

        assert snapshot.exchange == "test"
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert isinstance(snapshot.bids[0][0], Price)
        assert isinstance(snapshot.bids[0][1], Volume)

    def test_filter_order_book_disabled(self: "TestOrderBookPreFilter") -> None:
        """Тест фильтрации при отключенном фильтре."""
        # Отключаем фильтр
        self.filter_obj.config.enabled = False

        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]

        result = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

        assert result.meta.get("synthetic_noise") == False
        assert result.meta.get("filtered") == False

    def test_filter_order_book_enabled(self: "TestOrderBookPreFilter") -> None:
        """Тест фильтрации при включенном фильтре."""
        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]

        result = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

        assert "synthetic_noise" in result.meta
        assert "noise_analysis" in result.meta
        assert "filtered" in result.meta
        assert "filter_confidence" in result.meta

    def test_is_order_book_filtered(self: "TestOrderBookPreFilter") -> None:
        """Тест проверки фильтрации ордербука."""
        # Создаем ордербук с флагом фильтрации
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        order_book.meta["filtered"] = True

        assert self.filter_obj.is_order_book_filtered(order_book) == True

        # Тест с неотфильтрованным ордербуком
        order_book.meta["filtered"] = False
        assert self.filter_obj.is_order_book_filtered(order_book) == False

    def test_get_noise_analysis_result(self: "TestOrderBookPreFilter") -> None:
        """Тест получения результатов анализа."""
        # Создаем ордербук с результатами анализа
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        order_book.meta["noise_analysis"] = {
            "fractal_dimension": 1.3,
            "entropy": 0.5,
            "is_synthetic_noise": True,
        }

        result = self.filter_obj.get_noise_analysis_result(order_book)
        assert result["fractal_dimension"] == 1.3
        assert result["entropy"] == 0.5
        assert result["is_synthetic_noise"] == True

    def test_get_filter_statistics(self: "TestOrderBookPreFilter") -> None:
        """Тест получения статистики фильтра."""
        # Обрабатываем несколько ордербуков
        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]

        for i in range(5):
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )

        stats = self.filter_obj.get_filter_statistics()

        assert stats["total_processed"] == 5
        assert "filter_rate" in stats
        assert "synthetic_noise_rate" in stats
        assert "config" in stats
        assert "analyzer_stats" in stats

    def test_reset_statistics(self: "TestOrderBookPreFilter") -> None:
        """Тест сброса статистики."""
        # Обрабатываем ордербук
        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]

        self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

        assert self.filter_obj.stats["total_processed"] > 0

        # Сбрасываем статистику
        self.filter_obj.reset_statistics()

        assert self.filter_obj.stats["total_processed"] == 0
        assert self.filter_obj.stats["filtered_out"] == 0
        assert self.filter_obj.stats["synthetic_noise_detected"] == 0

    def test_update_config(self: "TestOrderBookPreFilter") -> None:
        """Тест обновления конфигурации."""
        new_config = FilterConfig(
            enabled=False,
            fractal_dimension_lower=1.0,
            fractal_dimension_upper=1.5,
            entropy_threshold=0.8,
        )

        self.filter_obj.update_config(new_config)

        assert self.filter_obj.config.enabled == False
        assert self.filter_obj.config.fractal_dimension_lower == 1.0
        assert self.filter_obj.config.entropy_threshold == 0.8


class TestNoiseAnalysisResult:
    """Тесты для NoiseAnalysisResult."""

    def test_creation(self: "TestNoiseAnalysisResult") -> None:
        """Тест создания результата анализа."""
        result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.5,
            is_synthetic_noise=True,
            confidence=0.8,
            metadata={"test": "data"},
            timestamp=Timestamp(time.time()),
        )

        assert result.fractal_dimension == 1.3
        assert result.entropy == 0.5
        assert result.is_synthetic_noise == True
        assert result.confidence == 0.8
        assert result.metadata == {"test": "data"}

    def test_to_dict(self: "TestNoiseAnalysisResult") -> None:
        """Тест преобразования в словарь."""
        result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.5,
            is_synthetic_noise=True,
            confidence=0.8,
            metadata={"test": "data"},
            timestamp=Timestamp(1640995200.0),
        )

        data = result.to_dict()

        assert data["fractal_dimension"] == 1.3
        assert data["entropy"] == 0.5
        assert data["is_synthetic_noise"] == True
        assert data["confidence"] == 0.8
        assert data["metadata"] == {"test": "data"}
        assert data["timestamp"] == 1640995200.0


class TestFilterConfig:
    """Тесты для FilterConfig."""

    def test_default_values(self: "TestFilterConfig") -> None:
        """Тест значений по умолчанию."""
        config = FilterConfig()

        assert config.enabled == True
        assert config.fractal_dimension_lower == 1.2
        assert config.fractal_dimension_upper == 1.4
        assert config.entropy_threshold == 0.7
        assert config.min_data_points == 50
        assert config.window_size == 100
        assert config.confidence_threshold == 0.8
        assert config.log_filtered == True
        assert config.log_analysis == False

    def test_custom_values(self: "TestFilterConfig") -> None:
        """Тест пользовательских значений."""
        config = FilterConfig(
            enabled=False,
            fractal_dimension_lower=1.0,
            fractal_dimension_upper=1.5,
            entropy_threshold=0.8,
            min_data_points=30,
            window_size=80,
            confidence_threshold=0.9,
            log_filtered=False,
            log_analysis=True,
        )

        assert config.enabled == False
        assert config.fractal_dimension_lower == 1.0
        assert config.fractal_dimension_upper == 1.5
        assert config.entropy_threshold == 0.8
        assert config.min_data_points == 30
        assert config.window_size == 80
        assert config.confidence_threshold == 0.9
        assert config.log_filtered == False
        assert config.log_analysis == True


if __name__ == "__main__":
    pytest.main([__file__])
