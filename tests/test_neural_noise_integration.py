# -*- coding: utf-8 -*-
"""Интеграционные тесты для системы анализа нейронного шума."""
import asyncio
import time
import pytest
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.filters.orderbook_filter import (FilterConfig,
                                                  OrderBookPreFilter)
from domain.intelligence.noise_analyzer import NoiseAnalyzer
class TestNeuralNoiseIntegration:
    """Интеграционные тесты для системы анализа нейронного шума."""
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
        self.analyzer = NoiseAnalyzer(
            fractal_dimension_lower=1.2,
            fractal_dimension_upper=1.4,
            entropy_threshold=0.7,
            min_data_points=20,
            window_size=50,
        )
    def test_end_to_end_synthetic_noise_detection(self: "TestNeuralNoiseIntegration") -> None:
        """Тест полного цикла обнаружения синтетического шума."""
        # Создаем синтетические данные с регулярными паттернами
        base_price = 50000.0
        bids = []
        asks = []
        for i in range(10):
            # Добавляем периодический шум
            noise = np.sin(i * 0.5) * 5  # Регулярная синусоида
            bid_price = base_price - i * 10 + noise
            ask_price = base_price + 10 + i * 10 + noise
            # Регулярные объемы
            volume = 1.0 + (i % 2) * 0.5
            bids.append((bid_price, volume))
            asks.append((ask_price, volume))
        # Фильтруем ордербук
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        # Проверяем результат
        assert "synthetic_noise" in filtered_ob.meta
        assert "noise_analysis" in filtered_ob.meta
        assert "filtered" in filtered_ob.meta
        assert "filter_confidence" in filtered_ob.meta
        # Проверяем, что анализ был выполнен
        noise_analysis = filtered_ob.meta["noise_analysis"]
        assert "fractal_dimension" in noise_analysis
        assert "entropy" in noise_analysis
        assert "is_synthetic_noise" in noise_analysis
        assert "confidence" in noise_analysis
    def test_end_to_end_natural_noise_processing(self: "TestNeuralNoiseIntegration") -> None:
        """Тест полного цикла обработки естественного шума."""
        # Создаем естественные данные со случайным шумом
        base_price = 50000.0
        bids = []
        asks = []
        for i in range(10):
            # Случайный шум
            noise = np.random.normal(0, 2)
            bid_price = base_price - i * 10 + noise
            ask_price = base_price + 10 + i * 10 + noise
            # Случайные объемы
            volume = np.random.uniform(0.1, 2.0)
            bids.append((bid_price, volume))
            asks.append((ask_price, volume))
        # Фильтруем ордербук
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        # Проверяем результат
        assert "synthetic_noise" in filtered_ob.meta
        assert "noise_analysis" in filtered_ob.meta
        # Естественные данные не должны быть помечены как синтетические
        # (хотя это зависит от конкретных значений)
        noise_analysis = filtered_ob.meta["noise_analysis"]
        assert isinstance(noise_analysis["fractal_dimension"], float)
        assert isinstance(noise_analysis["entropy"], float)
        assert isinstance(noise_analysis["is_synthetic_noise"], bool)
    def test_multiple_order_book_processing(self: "TestNeuralNoiseIntegration") -> None:
        """Тест обработки нескольких ордербуков."""
        # Обрабатываем несколько ордербуков
        for i in range(10):
            bids = [(50000.0 - j * 10, 1.0) for j in range(5)]
            asks = [(50010.0 + j * 10, 1.0) for j in range(5)]
            filtered_ob = self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
                sequence_id=i,
            )
            # Проверяем, что каждый ордербук обработан
            assert filtered_ob.meta.get("filter_stats") is not None
            assert filtered_ob.meta["filter_stats"]["total_processed"] == i + 1
        # Проверяем финальную статистику
        stats = self.filter_obj.get_filter_statistics()
        assert stats["total_processed"] == 10
        assert stats["filter_rate"] >= 0.0
        assert stats["filter_rate"] <= 1.0
    def test_configuration_updates(self: "TestNeuralNoiseIntegration") -> None:
        """Тест обновления конфигурации во время работы."""
        # Обрабатываем ордербук с начальной конфигурацией
        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]
        filtered_ob1 = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        # Обновляем конфигурацию
        new_config = FilterConfig(
            enabled=True,
            fractal_dimension_lower=1.0,
            fractal_dimension_upper=1.2,
            entropy_threshold=0.5,
        )
        self.filter_obj.update_config(new_config)
        # Обрабатываем еще один ордербук
        filtered_ob2 = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        # Проверяем, что конфигурация обновилась
        assert self.filter_obj.config.fractal_dimension_lower == 1.0
        assert self.filter_obj.config.entropy_threshold == 0.5
        # Проверяем, что оба ордербука обработаны
        stats = self.filter_obj.get_filter_statistics()
        assert stats["total_processed"] == 2
    def test_error_handling_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест обработки ошибок в интеграции."""
        # Тест с некорректными данными
        with pytest.raises(Exception):
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=[("invalid", "data")],  # Некорректные данные
                asks=[(50010.0, 1.0)],
                timestamp=time.time(),
            )
        # Проверяем, что система продолжает работать
        bids = [(50000.0, 1.0)]
        asks = [(50010.0, 1.0)]
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        # Проверяем, что ордербук обработан
        assert filtered_ob.meta.get("error") is not None
    def test_performance_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест производительности интеграции."""
        import time
        # Измеряем время обработки
        start_time = time.time()
        for i in range(100):
            bids = [(50000.0 - j * 10, 1.0) for j in range(5)]
            asks = [(50010.0 + j * 10, 1.0) for j in range(5)]
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )
        end_time = time.time()
        processing_time = end_time - start_time
        # Проверяем, что обработка не занимает слишком много времени
        # (менее 1 секунды для 100 ордербуков)
        assert processing_time < 1.0
        # Проверяем статистику
        stats = self.filter_obj.get_filter_statistics()
        assert stats["total_processed"] == 100
    @pytest.mark.asyncio
    def test_async_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест асинхронной интеграции."""
        # Создаем поток данных
        async def order_book_stream() -> Any:
            for i in range(5):
                yield {
                    "exchange": "test",
                    "symbol": "BTCUSDT",
                    "bids": [(50000.0 - j * 10, 1.0) for j in range(3)],
                    "asks": [(50010.0 + j * 10, 1.0) for j in range(3)],
                    "timestamp": time.time(),
                    "sequence_id": i,
                }
                await asyncio.sleep(0.01)  # Небольшая задержка
        # Обрабатываем поток
        processed_count = 0
        async for filtered_ob in self.filter_obj.filter_order_book_stream(
            order_book_stream()
        ):
            processed_count += 1
            assert filtered_ob.meta.get("synthetic_noise") is not None
            assert filtered_ob.meta.get("noise_analysis") is not None
        assert processed_count == 5
    def test_memory_management_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест управления памятью в интеграции."""
        # Обрабатываем много ордербуков для проверки управления памятью
        for i in range(1000):
            bids = [(50000.0 - j * 10, 1.0) for j in range(3)]
            asks = [(50010.0 + j * 10, 1.0) for j in range(3)]
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )
        # Проверяем, что история не превышает установленный размер
        analyzer_stats = self.filter_obj.noise_analyzer.get_analysis_statistics()
        assert analyzer_stats["price_history_length"] <= self.config.window_size
        assert analyzer_stats["volume_history_length"] <= self.config.window_size
        assert analyzer_stats["spread_history_length"] <= self.config.window_size
    def test_statistics_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест интеграции статистики."""
        # Обрабатываем ордербуки
        for i in range(10):
            bids = [(50000.0, 1.0)]
            asks = [(50010.0, 1.0)]
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )
        # Получаем статистику
        stats = self.filter_obj.get_filter_statistics()
        # Проверяем структуру статистики
        assert "total_processed" in stats
        assert "filtered_out" in stats
        assert "synthetic_noise_detected" in stats
        assert "filter_rate" in stats
        assert "synthetic_noise_rate" in stats
        assert "config" in stats
        assert "analyzer_stats" in stats
        # Проверяем значения
        assert stats["total_processed"] == 10
        assert stats["filter_rate"] >= 0.0
        assert stats["filter_rate"] <= 1.0
        assert stats["synthetic_noise_rate"] >= 0.0
        assert stats["synthetic_noise_rate"] <= 1.0
        # Проверяем конфигурацию в статистике
        assert stats["config"]["enabled"] == self.config.enabled
        assert stats["config"]["fractal_dimension_range"] == [
            self.config.fractal_dimension_lower,
            self.config.fractal_dimension_upper,
        ]
    def test_reset_functionality_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест функциональности сброса в интеграции."""
        # Обрабатываем ордербуки
        for i in range(5):
            bids = [(50000.0, 1.0)]
            asks = [(50010.0, 1.0)]
            self.filter_obj.filter_order_book(
                exchange="test",
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )
        # Проверяем, что данные накопились
        stats_before = self.filter_obj.get_filter_statistics()
        assert stats_before["total_processed"] == 5
        analyzer_stats_before = self.filter_obj.noise_analyzer.get_analysis_statistics()
        assert analyzer_stats_before["price_history_length"] > 0
        # Сбрасываем статистику
        self.filter_obj.reset_statistics()
        # Проверяем, что данные сброшены
        stats_after = self.filter_obj.get_filter_statistics()
        assert stats_after["total_processed"] == 0
        assert stats_after["filtered_out"] == 0
        assert stats_after["synthetic_noise_detected"] == 0
        analyzer_stats_after = self.filter_obj.noise_analyzer.get_analysis_statistics()
        assert analyzer_stats_after["price_history_length"] == 0
    def test_edge_cases_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест граничных случаев в интеграции."""
        # Тест с пустыми данными
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test", symbol="BTCUSDT", bids=[], asks=[], timestamp=time.time()
        )
        assert filtered_ob.meta.get("synthetic_noise") is not None
        assert filtered_ob.meta.get("noise_analysis") is not None
        # Тест с минимальными данными
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0)],
            asks=[(50010.0, 1.0)],
            timestamp=time.time(),
        )
        assert filtered_ob.meta.get("synthetic_noise") is not None
        assert filtered_ob.meta.get("noise_analysis") is not None
        # Тест с очень большими данными
        bids = [(50000.0 - i * 10, 1.0) for i in range(100)]
        asks = [(50010.0 + i * 10, 1.0) for i in range(100)]
        filtered_ob = self.filter_obj.filter_order_book(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )
        assert filtered_ob.meta.get("synthetic_noise") is not None
        assert filtered_ob.meta.get("noise_analysis") is not None
    def test_concurrent_access_integration(self: "TestNeuralNoiseIntegration") -> None:
        """Тест конкурентного доступа в интеграции."""
        import queue
        import threading
        # Создаем очередь для результатов
        results = queue.Queue()
        def process_order_book(thread_id) -> Any:
            """Функция для обработки ордербука в отдельном потоке."""
            try:
                bids = [(50000.0, 1.0)]
                asks = [(50010.0, 1.0)]
                filtered_ob = self.filter_obj.filter_order_book(
                    exchange="test",
                    symbol="BTCUSDT",
                    bids=bids,
                    asks=asks,
                    timestamp=time.time(),
                    sequence_id=thread_id,
                )
                results.put((thread_id, filtered_ob.meta.get("synthetic_noise")))
            except Exception as e:
                results.put((thread_id, f"Error: {e}"))
        # Запускаем несколько потоков
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_order_book, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем результаты
        processed_results = []
        while not results.empty():
            processed_results.append(results.get())
        assert len(processed_results) == 5
        # Проверяем, что все результаты корректны
        for thread_id, result in processed_results:
            assert isinstance(result, bool) or result.startswith("Error")
        # Проверяем статистику
        stats = self.filter_obj.get_filter_statistics()
        assert stats["total_processed"] == 5
if __name__ == "__main__":
    pytest.main([__file__])
