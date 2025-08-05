# -*- coding: utf-8 -*-
"""Интеграционные тесты для системы Quantum Order Entanglement Detection."""
import asyncio
import time
import pytest
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.analysis.entanglement_monitor import EntanglementMonitor
from domain.intelligence.entanglement_detector import (EntanglementDetector,
                                                       OrderBookUpdate)
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
class TestEntanglementIntegration:
    """Интеграционные тесты для системы обнаружения запутанности."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.detector = EntanglementDetector(
            max_lag_ms=3.0,
            correlation_threshold=0.95,
            window_size=50,
            min_data_points=20,
        )
        self.monitor = EntanglementMonitor(
            log_file_path="test_integration_events.json",
            detection_interval=0.1,
            max_lag_ms=3.0,
            correlation_threshold=0.95,
        )
    def test_full_pipeline_simulation(self: "TestEntanglementIntegration") -> None:
        """Тест полного пайплайна с симуляцией данных."""
        # Создаем симулированные данные с высокой корреляцией
        base_prices = np.linspace(50000, 51000, 100)
        noise1 = np.random.normal(0, 5, 100)
        noise2 = np.random.normal(0, 5, 100)
        # Данные с задержкой и корреляцией
        prices1 = base_prices + noise1
        prices2 = np.roll(base_prices + noise2, 2)  # Задержка на 2 тика
        updates = []
        # Создаем обновления для двух бирж
        for i in range(100):
            # Обновление для первой биржи
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(prices1[i] - 10), Volume(1.0))],
                asks=[(Price(prices1[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update1)
            # Обновление для второй биржи (с задержкой)
            if i >= 2:
                update2 = OrderBookUpdate(
                    exchange="coinbase",
                    symbol="BTCUSDT",
                    bids=[(Price(prices2[i] - 10), Volume(1.0))],
                    asks=[(Price(prices2[i] + 10), Volume(1.0))],
                    timestamp=Timestamp(time.time() + i * 0.001),
                )
                updates.append(update2)
        # Обрабатываем обновления через детектор
        results = self.detector.process_order_book_updates(updates)
        # Проверяем результаты
        assert len(results) > 0
        # Находим результат для пары binance-coinbase
        binance_coinbase_result = None
        for result in results:
            if set(result.exchange_pair) == {"binance", "coinbase"}:
                binance_coinbase_result = result
                break
        assert binance_coinbase_result is not None
        assert binance_coinbase_result.symbol == "BTCUSDT"
        assert binance_coinbase_result.correlation_score > 0.8  # Высокая корреляция
        assert abs(binance_coinbase_result.lag_ms) <= 5.0  # Разумный lag
    @pytest.mark.asyncio
    def test_monitor_with_simulated_data(self: "TestEntanglementIntegration") -> None:
        """Тест монитора с симулированными данными."""
        # Запускаем мониторинг в фоновом режиме
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        try:
            # Ждем немного для инициализации
            await asyncio.sleep(0.2)
            # Проверяем, что мониторинг запущен
            assert self.monitor.is_running == True
            # Проверяем статус
            status = self.monitor.get_status()
            assert status["is_running"] == True
            assert status["active_pairs"] > 0
            # Ждем еще немного для сбора данных
            await asyncio.sleep(0.3)
            # Проверяем, что данные собираются
            final_status = self.monitor.get_status()
            assert final_status["stats"]["total_detections"] >= 0
        finally:
            # Останавливаем мониторинг
            self.monitor.stop_monitoring()
            await monitor_task
    def test_multiple_exchange_pairs(self: "TestEntanglementIntegration") -> None:
        """Тест работы с несколькими парами бирж."""
        # Добавляем несколько пар
        self.monitor.add_exchange_pair("binance", "kraken", "ETHUSDT")
        self.monitor.add_exchange_pair("coinbase", "kraken", "ADAUSDT")
        # Проверяем количество пар
        assert len(self.monitor.exchange_pairs) >= 3  # Изначальные + добавленные
        # Проверяем, что новые пары активны
        active_pairs = [p for p in self.monitor.exchange_pairs if p.is_active]
        assert len(active_pairs) >= 3
    def test_buffer_management(self: "TestEntanglementIntegration") -> None:
        """Тест управления буферами данных."""
        # Добавляем много обновлений
        for i in range(200):
            update = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(50000.0 + i), Volume(1.0))],
                asks=[(Price(50010.0 + i), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            self.monitor._add_order_book_update(update)
        # Проверяем, что буфер не превышает максимальный размер
        buffer_size = len(self.monitor.order_book_buffers.get("binance", []))
        assert buffer_size <= self.monitor.buffer_max_size
    def test_correlation_calculation_accuracy(self: "TestEntanglementIntegration") -> None:
        """Тест точности расчета корреляции."""
        # Создаем идеально коррелированные данные
        n_points = 50
        x = np.linspace(0, 10, n_points)
        y1 = 2 * x + 1  # Линейная зависимость
        y2 = 2 * x + 1 + np.random.normal(0, 0.1, n_points)  # С небольшим шумом
        # Создаем обновления
        updates = []
        for i in range(n_points):
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(y1[i]), Volume(1.0))],
                asks=[(Price(y1[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update1)
            update2 = OrderBookUpdate(
                exchange="coinbase",
                symbol="BTCUSDT",
                bids=[(Price(y2[i]), Volume(1.0))],
                asks=[(Price(y2[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update2)
        # Обрабатываем через детектор
        results = self.detector.process_order_book_updates(updates)
        # Проверяем высокую корреляцию
        if results:
            result = results[0]
            assert result.correlation_score > 0.9  # Очень высокая корреляция
    def test_lag_detection_accuracy(self: "TestEntanglementIntegration") -> None:
        """Тест точности обнаружения lag."""
        # Создаем данные с известной задержкой
        n_points = 50
        base_prices = np.linspace(50000, 51000, n_points)
        known_lag = 3  # Известная задержка в тиках
        updates = []
        for i in range(n_points):
            # Первая биржа
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i]), Volume(1.0))],
                asks=[(Price(base_prices[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update1)
            # Вторая биржа с задержкой
            if i >= known_lag:
                update2 = OrderBookUpdate(
                    exchange="coinbase",
                    symbol="BTCUSDT",
                    bids=[(Price(base_prices[i - known_lag]), Volume(1.0))],
                    asks=[(Price(base_prices[i - known_lag] + 10), Volume(1.0))],
                    timestamp=Timestamp(time.time() + i * 0.001),
                )
                updates.append(update2)
        # Обрабатываем через детектор
        results = self.detector.process_order_book_updates(updates)
        # Проверяем обнаружение lag
        if results:
            result = results[0]
            detected_lag = abs(result.lag_ms)
            # Lag должен быть близок к известному значению (с учетом погрешности)
            assert abs(detected_lag - known_lag) <= 2
    def test_confidence_calculation(self: "TestEntanglementIntegration") -> None:
        """Тест расчета уверенности в результатах."""
        # Создаем данные с разной качественностью
        n_points = 50
        base_prices = np.linspace(50000, 51000, n_points)
        # Высококачественные данные
        updates_high_quality = []
        for i in range(n_points):
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i]), Volume(1.0))],
                asks=[(Price(base_prices[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates_high_quality.append(update1)
            update2 = OrderBookUpdate(
                exchange="coinbase",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i]), Volume(1.0))],
                asks=[(Price(base_prices[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates_high_quality.append(update2)
        # Обрабатываем высококачественные данные
        results_high = self.detector.process_order_book_updates(updates_high_quality)
        # Создаем низкокачественные данные (с большим шумом)
        updates_low_quality = []
        for i in range(n_points):
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i] + np.random.normal(0, 100)), Volume(1.0))],
                asks=[
                    (Price(base_prices[i] + np.random.normal(0, 100) + 10), Volume(1.0))
                ],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates_low_quality.append(update1)
            update2 = OrderBookUpdate(
                exchange="coinbase",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i] + np.random.normal(0, 100)), Volume(1.0))],
                asks=[
                    (Price(base_prices[i] + np.random.normal(0, 100) + 10), Volume(1.0))
                ],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates_low_quality.append(update2)
        # Обрабатываем низкокачественные данные
        results_low = self.detector.process_order_book_updates(updates_low_quality)
        # Проверяем, что уверенность выше для высококачественных данных
        if results_high and results_low:
            confidence_high = results_high[0].confidence
            confidence_low = results_low[0].confidence
            assert confidence_high > confidence_low
    def test_error_handling(self: "TestEntanglementIntegration") -> None:
        """Тест обработки ошибок."""
        # Тестируем обработку некорректных данных
        invalid_updates = [
            OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[],  # Пустые данные
                asks=[],
                timestamp=Timestamp(time.time()),
            ),
            OrderBookUpdate(
                exchange="coinbase",
                symbol="BTCUSDT",
                bids=[(Price(50000.0), Volume(1.0))],
                asks=[(Price(50010.0), Volume(1.0))],
                timestamp=Timestamp(time.time()),
            ),
        ]
        # Обработка не должна вызывать исключений
        try:
            results = self.detector.process_order_book_updates(invalid_updates)
            # Результат может быть пустым, но не должен вызывать ошибку
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Error handling failed: {e}")
    def test_performance_under_load(self: "TestEntanglementIntegration") -> None:
        """Тест производительности под нагрузкой."""
        # Создаем большое количество обновлений
        n_updates = 1000
        updates = []
        start_time = time.time()
        for i in range(n_updates):
            update = OrderBookUpdate(
                exchange="binance" if i % 2 == 0 else "coinbase",
                symbol="BTCUSDT",
                bids=[(Price(50000.0 + i), Volume(1.0))],
                asks=[(Price(50010.0 + i), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update)
        # Обрабатываем все обновления
        results = self.detector.process_order_book_updates(updates)
        processing_time = time.time() - start_time
        # Проверяем, что обработка не занимает слишком много времени
        assert processing_time < 1.0  # Менее 1 секунды для 1000 обновлений
        # Проверяем, что результаты получены
        assert isinstance(results, list)
if __name__ == "__main__":
    pytest.main([__file__])
