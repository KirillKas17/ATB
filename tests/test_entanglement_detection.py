# -*- coding: utf-8 -*-
"""Тесты для системы Quantum Order Entanglement Detection."""
import asyncio
import time
import pytest
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.analysis.entanglement_monitor import EntanglementMonitor
from domain.intelligence.entanglement_detector import (EntanglementDetector,
                                                       EntanglementResult,
                                                       OrderBookUpdate)
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
class TestEntanglementDetector:
    """Тесты для EntanglementDetector."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.detector = EntanglementDetector(
            max_lag_ms=3.0,
            correlation_threshold=0.95,
            window_size=50,
            min_data_points=20,
        )
    def test_initialization(self) -> None:
        """Тест инициализации детектора."""
        assert self.detector.max_lag_ms == 3.0
        assert self.detector.correlation_threshold == 0.95
        assert self.detector.window_size == 50
        assert self.detector.min_data_points == 20
        assert len(self.detector.exchange_buffers) == 0
    def test_add_order_book_update(self) -> None:
        """Тест добавления обновления ордербука."""
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        self.detector._add_order_book_update(update)
        assert "binance" in self.detector.exchange_buffers
        assert len(self.detector.exchange_buffers["binance"]) == 1
        buffer_item = self.detector.exchange_buffers["binance"][0]
        assert buffer_item["price"].value == 50005.0  # Средняя цена
        assert buffer_item["spread"].value == 10.0
    def test_normalize_price_changes(self) -> None:
        """Тест нормализации изменений цен."""
        # Создаем тестовые данные
        test_data = [
            {"price": Price(100.0)},
            {"price": Price(101.0)},
            {"price": Price(102.0)},
            {"price": Price(103.0)},
            {"price": Price(104.0)},
        ]
        normalized = self.detector._normalize_price_changes(test_data)
        assert len(normalized) == 4  # n-1 изменений
        assert isinstance(normalized, np.ndarray)
        assert np.all(np.isfinite(normalized))  # Все значения конечные
    def test_calculate_cross_correlation(self) -> None:
        """Тест расчета кросс-корреляции."""
        # Создаем коррелированные данные
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Сдвинутые данные
        correlation, lag = self.detector._calculate_cross_correlation(data1, data2)
        assert isinstance(correlation, float)
        assert isinstance(lag, int)
        assert correlation > 0.9  # Высокая корреляция
        assert abs(lag) <= len(data1)  # Lag в разумных пределах
    def test_detect_entanglement_no_data(self) -> None:
        """Тест обнаружения запутанности без данных."""
        result = self.detector.detect_entanglement("binance", "coinbase", "BTCUSDT")
        assert result is None
    def test_detect_entanglement_insufficient_data(self) -> None:
        """Тест обнаружения запутанности с недостаточными данными."""
        # Добавляем мало данных
        for i in range(10):
            update = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(50000.0 + i), Volume(1.0))],
                asks=[(Price(50010.0 + i), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            self.detector._add_order_book_update(update)
        result = self.detector.detect_entanglement("binance", "coinbase", "BTCUSDT")
        assert result is None
    def test_detect_entanglement_with_data(self) -> None:
        """Тест обнаружения запутанности с достаточными данными."""
        # Создаем коррелированные данные для двух бирж
        base_prices = np.linspace(50000, 51000, 50)
        for i in range(50):
            # Данные для первой биржи
            update1 = OrderBookUpdate(
                exchange="binance",
                symbol="BTCUSDT",
                bids=[(Price(base_prices[i] - 10), Volume(1.0))],
                asks=[(Price(base_prices[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            self.detector._add_order_book_update(update1)
            # Данные для второй биржи (с небольшой задержкой)
            if i >= 2:
                update2 = OrderBookUpdate(
                    exchange="coinbase",
                    symbol="BTCUSDT",
                    bids=[(Price(base_prices[i - 2] - 10), Volume(1.0))],
                    asks=[(Price(base_prices[i - 2] + 10), Volume(1.0))],
                    timestamp=Timestamp(time.time() + i * 0.001),
                )
                self.detector._add_order_book_update(update2)
        result = self.detector.detect_entanglement("binance", "coinbase", "BTCUSDT")
        assert result is not None
        assert isinstance(result, EntanglementResult)
        assert result.exchange_pair == ("binance", "coinbase")
        assert result.symbol == "BTCUSDT"
        assert result.confidence > 0
    def test_get_buffer_status(self) -> None:
        """Тест получения статуса буферов."""
        # Добавляем данные
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        self.detector._add_order_book_update(update)
        status = self.detector.get_buffer_status()
        assert "binance" in status
        assert status["binance"]["buffer_size"] == 1
        assert status["binance"]["price_range"]["min"] == 50005.0
        assert status["binance"]["price_range"]["max"] == 50005.0
    def test_clear_buffers(self) -> None:
        """Тест очистки буферов."""
        # Добавляем данные
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        self.detector._add_order_book_update(update)
        assert len(self.detector.exchange_buffers["binance"]) == 1
        self.detector.clear_buffers()
        assert len(self.detector.exchange_buffers) == 0
        assert len(self.detector.last_processed) == 0
class TestEntanglementMonitor:
    """Тесты для EntanglementMonitor."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.monitor = EntanglementMonitor(
            log_file_path="test_entanglement_events.json",
            detection_interval=0.1,
            max_lag_ms=3.0,
            correlation_threshold=0.95,
        )
    def test_initialization(self) -> None:
        """Тест инициализации монитора."""
        assert self.monitor.detection_interval == 0.1
        assert self.monitor.is_running == False
        assert len(self.monitor.connectors) == 3  # binance, coinbase, kraken
        assert len(self.monitor.exchange_pairs) > 0
    def test_add_exchange_pair(self) -> None:
        """Тест добавления пары бирж."""
        initial_count = len(self.monitor.exchange_pairs)
        self.monitor.add_exchange_pair("test1", "test2", "TESTUSDT")
        assert len(self.monitor.exchange_pairs) == initial_count + 1
        # Проверяем, что пара добавлена
        pair = self.monitor.exchange_pairs[-1]
        assert pair.exchange1 == "test1"
        assert pair.exchange2 == "test2"
        assert pair.symbol == "TESTUSDT"
        assert pair.is_active == True
    def test_remove_exchange_pair(self) -> None:
        """Тест удаления пары бирж."""
        # Добавляем пару
        self.monitor.add_exchange_pair("test1", "test2", "TESTUSDT")
        initial_count = len(self.monitor.exchange_pairs)
        # Удаляем пару
        self.monitor.remove_exchange_pair("test1", "test2", "TESTUSDT")
        assert len(self.monitor.exchange_pairs) == initial_count - 1
    def test_get_status(self) -> None:
        """Тест получения статуса."""
        status = self.monitor.get_status()
        assert "is_running" in status
        assert "stats" in status
        assert "active_pairs" in status
        assert "total_pairs" in status
        assert "buffer_sizes" in status
        assert "detector_status" in status
        assert status["is_running"] == False
        assert status["stats"]["total_detections"] == 0
        assert status["stats"]["entangled_detections"] == 0
    def test_stop_monitoring(self) -> None:
        """Тест остановки мониторинга."""
        self.monitor.is_running = True
        self.monitor.stop_monitoring()
        assert self.monitor.is_running == False
    @pytest.mark.asyncio
    async def test_start_monitoring_short(self) -> None:
        """Тест кратковременного запуска мониторинга."""
        # Запускаем мониторинг на короткое время
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        # Ждем немного
        await asyncio.sleep(0.1)
        # Останавливаем
        self.monitor.stop_monitoring()
        await monitor_task
        # Проверяем, что мониторинг остановлен
        assert self.monitor.is_running == False
    def test_get_entanglement_history_empty(self) -> None:
        """Тест получения истории запутанности (пустая)."""
        history = self.monitor.get_entanglement_history()
        assert isinstance(history, list)
        assert len(history) == 0
    def test_log_entanglement_event(self) -> None:
        """Тест логирования события запутанности."""
        result = EntanglementResult(
            is_entangled=True,
            lag_ms=2.5,
            correlation_score=0.97,
            exchange_pair=("binance", "coinbase"),
            symbol="BTCUSDT",
            timestamp=Timestamp(time.time()),
            confidence=0.95,
            metadata={"test": "data"},
        )
        self.monitor._log_entanglement_event(result)
        # Проверяем, что статистика обновилась
        assert self.monitor.stats["total_detections"] == 1
        assert self.monitor.stats["entangled_detections"] == 1
        assert self.monitor.stats["last_detection_time"] is not None
class TestOrderBookUpdate:
    """Тесты для OrderBookUpdate."""
    def test_creation(self) -> None:
        """Тест создания OrderBookUpdate."""
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        assert update.exchange == "binance"
        assert update.symbol == "BTCUSDT"
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        assert update.sequence_id is None
    def test_get_mid_price(self) -> None:
        """Тест получения средней цены."""
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        mid_price = update.get_mid_price()
        assert mid_price.value == 50005.0
    def test_get_spread(self) -> None:
        """Тест получения спреда."""
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(time.time()),
        )
        spread = update.get_spread()
        assert spread.value == 10.0
    def test_get_mid_price_empty(self) -> None:
        """Тест получения средней цены при пустых данных."""
        update = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            timestamp=Timestamp(time.time()),
        )
        mid_price = update.get_mid_price()
        assert mid_price.value == 0.0
class TestEntanglementResult:
    """Тесты для EntanglementResult."""
    def test_creation(self) -> None:
        """Тест создания EntanglementResult."""
        result = EntanglementResult(
            is_entangled=True,
            lag_ms=2.5,
            correlation_score=0.97,
            exchange_pair=("binance", "coinbase"),
            symbol="BTCUSDT",
            timestamp=Timestamp(time.time()),
            confidence=0.95,
            metadata={"test": "data"},
        )
        assert result.is_entangled == True
        assert result.lag_ms == 2.5
        assert result.correlation_score == 0.97
        assert result.exchange_pair == ("binance", "coinbase")
        assert result.symbol == "BTCUSDT"
        assert result.confidence == 0.95
        assert result.metadata == {"test": "data"}
    def test_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        result = EntanglementResult(
            is_entangled=True,
            lag_ms=2.5,
            correlation_score=0.97,
            exchange_pair=("binance", "coinbase"),
            symbol="BTCUSDT",
            timestamp=Timestamp(1640995200.0),
            confidence=0.95,
            metadata={"test": "data"},
        )
        data = result.to_dict()
        assert data["is_entangled"] == True
        assert data["lag_ms"] == 2.5
        assert data["correlation_score"] == 0.97
        assert data["exchange_pair"] == ("binance", "coinbase")
        assert data["symbol"] == "BTCUSDT"
        assert data["timestamp"] == 1640995200.0
        assert data["confidence"] == 0.95
        assert data["metadata"] == {"test": "data"}
if __name__ == "__main__":
    pytest.main([__file__])
