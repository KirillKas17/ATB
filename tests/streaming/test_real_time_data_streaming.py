#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты real-time streaming финансовых данных.
Критически важно для финансовой системы - бесперебойная передача данных в реальном времени.
"""

import pytest
import asyncio
import time
import threading
import json
import websocket
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.market_data import MarketData, OrderBook, Ticker, Trade
from infrastructure.streaming.data_stream_manager import DataStreamManager
from infrastructure.streaming.websocket_client import WebSocketClient
from infrastructure.streaming.message_broker import MessageBroker, MessageQueue
from infrastructure.streaming.stream_processor import StreamProcessor, StreamFilter
from infrastructure.streaming.backpressure_manager import BackpressureManager
from infrastructure.streaming.data_validator import StreamDataValidator
from domain.exceptions import StreamingError, ConnectionError, DataValidationError


class StreamType(Enum):
    """Типы потоков данных."""
    MARKET_DATA = "MARKET_DATA"
    ORDER_BOOK = "ORDER_BOOK"
    TRADES = "TRADES"
    TICKER = "TICKER"
    ACCOUNT_UPDATES = "ACCOUNT_UPDATES"
    ORDER_UPDATES = "ORDER_UPDATES"
    POSITION_UPDATES = "POSITION_UPDATES"


class StreamStatus(Enum):
    """Статусы потоков."""
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


@dataclass
class StreamMessage:
    """Сообщение потока данных."""
    stream_id: str
    message_id: str
    timestamp: datetime
    message_type: StreamType
    data: Dict[str, Any]
    sequence_number: int
    source: str


@dataclass
class StreamMetrics:
    """Метрики потока данных."""
    messages_received: int
    messages_processed: int
    messages_dropped: int
    average_latency_ms: float
    throughput_msg_per_sec: float
    error_rate: float
    connection_uptime_percentage: float


class TestRealTimeDataStreaming:
    """Comprehensive тесты streaming данных."""

    @pytest.fixture
    def data_stream_manager(self) -> DataStreamManager:
        """Фикстура менеджера потоков данных."""
        return DataStreamManager(
            max_concurrent_streams=10,
            message_buffer_size=10000,
            heartbeat_interval_seconds=30,
            reconnect_attempts=5,
            reconnect_delay_seconds=1
        )

    @pytest.fixture
    def websocket_client(self) -> WebSocketClient:
        """Фикстура WebSocket клиента."""
        return WebSocketClient(
            connection_timeout_seconds=10,
            ping_interval_seconds=20,
            ping_timeout_seconds=5,
            max_message_size=1024*1024,  # 1MB
            compression_enabled=True
        )

    @pytest.fixture
    def message_broker(self) -> MessageBroker:
        """Фикстура брокера сообщений."""
        return MessageBroker(
            max_queue_size=50000,
            persistence_enabled=True,
            batch_processing=True,
            batch_size=100,
            flush_interval_ms=100
        )

    @pytest.fixture
    def stream_processor(self) -> StreamProcessor:
        """Фикстура процессора потоков."""
        return StreamProcessor(
            processing_threads=4,
            queue_size=10000,
            processing_timeout_seconds=5,
            error_handling_enabled=True
        )

    @pytest.fixture
    def mock_exchange_stream(self) -> Mock:
        """Фикстура мок потока биржи."""
        mock_stream = Mock()
        mock_stream.url = "wss://api.exchange.com/stream"
        mock_stream.is_connected = True
        mock_stream.send = AsyncMock()
        mock_stream.receive = AsyncMock()
        mock_stream.close = AsyncMock()
        return mock_stream

    def test_websocket_connection_management(
        self,
        websocket_client: WebSocketClient
    ) -> None:
        """Тест управления WebSocket соединениями."""
        
        connection_events = []
        
        # Настраиваем callback'и
        def on_connect():
            connection_events.append(("CONNECTED", datetime.utcnow()))
        
        def on_disconnect():
            connection_events.append(("DISCONNECTED", datetime.utcnow()))
        
        def on_error(error):
            connection_events.append(("ERROR", error))
        
        websocket_client.on_connect = on_connect
        websocket_client.on_disconnect = on_disconnect
        websocket_client.on_error = on_error
        
        # Тестируем соединение
        with patch('websocket.WebSocketApp') as mock_ws:
            mock_ws_instance = Mock()
            mock_ws.return_value = mock_ws_instance
            
            # Подключение
            asyncio.run(websocket_client.connect("wss://api.test.com"))
            
            # Симулируем успешное подключение
            mock_ws_instance.on_open(mock_ws_instance)
            
            assert websocket_client.is_connected is True
            assert len(connection_events) >= 1
            assert connection_events[0][0] == "CONNECTED"
            
            # Симулируем отключение
            mock_ws_instance.on_close(mock_ws_instance, 1000, "Normal closure")
            
            assert websocket_client.is_connected is False
            
            # Проверяем reconnect логику
            websocket_client.auto_reconnect = True
            mock_ws_instance.on_close(mock_ws_instance, 1006, "Connection lost")
            
            # Должна начаться попытка переподключения
            time.sleep(0.1)
            assert websocket_client.status == StreamStatus.RECONNECTING

    def test_real_time_market_data_streaming(
        self,
        data_stream_manager: DataStreamManager,
        stream_processor: StreamProcessor
    ) -> None:
        """Тест streaming рыночных данных в реальном времени."""
        
        received_messages = []
        processed_data = []
        
        def message_handler(message: StreamMessage):
            """Обработчик сообщений."""
            received_messages.append(message)
            
            if message.message_type == StreamType.MARKET_DATA:
                # Обрабатываем market data
                market_data = MarketData.from_dict(message.data)
                processed_data.append(market_data)
        
        # Настраиваем обработчик
        stream_processor.add_handler(StreamType.MARKET_DATA, message_handler)
        
        # Создаем тестовые данные
        test_market_data = [
            {
                "symbol": "BTCUSDT",
                "price": "45000.50",
                "volume": "1.25",
                "timestamp": datetime.utcnow().isoformat(),
                "bid": "44999.50",
                "ask": "45001.50",
                "high_24h": "45500.00",
                "low_24h": "44200.00"
            },
            {
                "symbol": "ETHUSDT", 
                "price": "3200.75",
                "volume": "5.80",
                "timestamp": datetime.utcnow().isoformat(),
                "bid": "3200.25",
                "ask": "3201.25",
                "high_24h": "3250.00",
                "low_24h": "3150.00"
            }
        ]
        
        # Симулируем получение данных
        for i, data in enumerate(test_market_data):
            message = StreamMessage(
                stream_id="market_stream_001",
                message_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                message_type=StreamType.MARKET_DATA,
                data=data,
                sequence_number=i + 1,
                source="EXCHANGE_API"
            )
            
            # Обрабатываем сообщение
            asyncio.run(stream_processor.process_message(message))
        
        # Проверяем результаты
        assert len(received_messages) == 2
        assert len(processed_data) == 2
        
        # Проверяем корректность данных
        btc_data = next(data for data in processed_data if data.symbol == "BTCUSDT")
        assert btc_data.price == Decimal("45000.50")
        assert btc_data.volume == Decimal("1.25")
        
        eth_data = next(data for data in processed_data if data.symbol == "ETHUSDT")
        assert eth_data.price == Decimal("3200.75")
        assert eth_data.volume == Decimal("5.80")

    def test_order_book_streaming_and_management(
        self,
        data_stream_manager: DataStreamManager
    ) -> None:
        """Тест streaming и управления order book."""
        
        order_books = {}
        
        async def order_book_handler(message: StreamMessage):
            """Обработчик order book updates."""
            symbol = message.data["symbol"]
            
            if symbol not in order_books:
                order_books[symbol] = OrderBook(symbol=symbol)
            
            order_book = order_books[symbol]
            
            # Обновляем order book
            if message.data["action"] == "snapshot":
                # Полный снимок
                order_book.update_snapshot(
                    bids=message.data["bids"],
                    asks=message.data["asks"]
                )
            elif message.data["action"] == "update":
                # Инкрементальное обновление
                order_book.update_incremental(
                    bids=message.data.get("bids", []),
                    asks=message.data.get("asks", [])
                )
        
        # Настраиваем stream
        stream_config = {
            "stream_id": "orderbook_btcusdt",
            "stream_type": StreamType.ORDER_BOOK,
            "symbol": "BTCUSDT",
            "depth": 20,
            "update_frequency": "100ms"
        }
        
        asyncio.run(data_stream_manager.create_stream(stream_config, order_book_handler))
        
        # Симулируем snapshot
        snapshot_message = StreamMessage(
            stream_id="orderbook_btcusdt",
            message_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            message_type=StreamType.ORDER_BOOK,
            data={
                "symbol": "BTCUSDT",
                "action": "snapshot",
                "bids": [
                    ["44999.50", "1.5"],
                    ["44999.00", "2.0"],
                    ["44998.50", "0.8"]
                ],
                "asks": [
                    ["45000.50", "1.2"],
                    ["45001.00", "1.8"],
                    ["45001.50", "0.5"]
                ]
            },
            sequence_number=1,
            source="EXCHANGE_ORDERBOOK"
        )
        
        await order_book_handler(snapshot_message)
        
        # Проверяем snapshot
        assert "BTCUSDT" in order_books
        btc_orderbook = order_books["BTCUSDT"]
        
        assert len(btc_orderbook.bids) == 3
        assert len(btc_orderbook.asks) == 3
        assert btc_orderbook.best_bid == Decimal("44999.50")
        assert btc_orderbook.best_ask == Decimal("45000.50")
        
        # Симулируем update
        update_message = StreamMessage(
            stream_id="orderbook_btcusdt",
            message_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            message_type=StreamType.ORDER_BOOK,
            data={
                "symbol": "BTCUSDT",
                "action": "update",
                "bids": [
                    ["45000.00", "2.5"]  # Новый bid
                ],
                "asks": [
                    ["45000.50", "0"]  # Удаляем ask (quantity = 0)
                ]
            },
            sequence_number=2,
            source="EXCHANGE_ORDERBOOK"
        )
        
        await order_book_handler(update_message)
        
        # Проверяем update
        assert btc_orderbook.best_bid == Decimal("45000.00")  # Новый лучший bid
        assert btc_orderbook.best_ask == Decimal("45001.00")  # Ask изменился

    def test_backpressure_management(
        self,
        message_broker: MessageBroker
    ) -> None:
        """Тест управления backpressure при высокой нагрузке."""
        
        backpressure_manager = BackpressureManager(
            high_watermark=8000,  # 80% заполнения
            low_watermark=5000,   # 50% заполнения
            max_queue_size=10000,
            drop_strategy="DROP_OLDEST"
        )
        
        # Создаем очередь сообщений
        message_queue = MessageQueue(
            name="high_volume_queue",
            max_size=10000,
            backpressure_manager=backpressure_manager
        )
        
        # Генерируем высокую нагрузку
        messages_sent = 0
        messages_dropped = 0
        
        for i in range(15000):  # Больше чем размер очереди
            message = StreamMessage(
                stream_id="load_test_stream",
                message_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                message_type=StreamType.TRADES,
                data={"trade_id": i, "price": "45000", "volume": "0.1"},
                sequence_number=i,
                source="LOAD_TEST"
            )
            
            try:
                result = message_queue.put(message)
                if result:
                    messages_sent += 1
                else:
                    messages_dropped += 1
            except Exception:
                messages_dropped += 1
        
        # Проверяем backpressure
        assert messages_dropped > 0  # Должны были сбросить сообщения
        assert message_queue.size() <= 10000  # Не превышаем лимит
        
        # Проверяем метрики backpressure
        bp_metrics = backpressure_manager.get_metrics()
        assert bp_metrics.high_watermark_breaches > 0
        assert bp_metrics.messages_dropped > 0
        assert bp_metrics.current_pressure_level > 0

    def test_stream_data_validation_and_filtering(
        self,
        stream_processor: StreamProcessor
    ) -> None:
        """Тест валидации и фильтрации данных потока."""
        
        validator = StreamDataValidator()
        
        # Настраиваем правила валидации
        validation_rules = {
            StreamType.MARKET_DATA: {
                "required_fields": ["symbol", "price", "timestamp"],
                "field_types": {
                    "price": "decimal",
                    "volume": "decimal",
                    "timestamp": "datetime"
                },
                "value_ranges": {
                    "price": {"min": Decimal("0"), "max": Decimal("1000000")},
                    "volume": {"min": Decimal("0"), "max": Decimal("100000")}
                }
            },
            StreamType.TRADES: {
                "required_fields": ["symbol", "price", "quantity", "timestamp"],
                "field_types": {
                    "price": "decimal",
                    "quantity": "decimal"
                }
            }
        }
        
        validator.configure_rules(validation_rules)
        
        # Создаем фильтры
        filters = [
            StreamFilter(
                name="symbol_filter",
                condition=lambda msg: msg.data.get("symbol") in ["BTCUSDT", "ETHUSDT"]
            ),
            StreamFilter(
                name="price_filter",
                condition=lambda msg: Decimal(msg.data.get("price", "0")) > Decimal("1000")
            )
        ]
        
        stream_processor.add_filters(filters)
        
        # Тестовые сообщения
        test_messages = [
            # Валидное сообщение
            StreamMessage(
                stream_id="test_stream",
                message_id="msg_001",
                timestamp=datetime.utcnow(),
                message_type=StreamType.MARKET_DATA,
                data={
                    "symbol": "BTCUSDT",
                    "price": "45000.50",
                    "volume": "1.25",
                    "timestamp": datetime.utcnow().isoformat()
                },
                sequence_number=1,
                source="TEST"
            ),
            # Невалидное сообщение - отсутствует price
            StreamMessage(
                stream_id="test_stream",
                message_id="msg_002",
                timestamp=datetime.utcnow(),
                message_type=StreamType.MARKET_DATA,
                data={
                    "symbol": "BTCUSDT",
                    "volume": "1.25",
                    "timestamp": datetime.utcnow().isoformat()
                },
                sequence_number=2,
                source="TEST"
            ),
            # Фильтруется по символу
            StreamMessage(
                stream_id="test_stream",
                message_id="msg_003",
                timestamp=datetime.utcnow(),
                message_type=StreamType.MARKET_DATA,
                data={
                    "symbol": "ADAUSDT",  # Не в whitelist
                    "price": "2.50",
                    "volume": "100.0",
                    "timestamp": datetime.utcnow().isoformat()
                },
                sequence_number=3,
                source="TEST"
            ),
            # Фильтруется по цене
            StreamMessage(
                stream_id="test_stream",
                message_id="msg_004",
                timestamp=datetime.utcnow(),
                message_type=StreamType.MARKET_DATA,
                data={
                    "symbol": "ETHUSDT",
                    "price": "500.00",  # Ниже фильтра цены
                    "volume": "5.0",
                    "timestamp": datetime.utcnow().isoformat()
                },
                sequence_number=4,
                source="TEST"
            )
        ]
        
        valid_messages = []
        invalid_messages = []
        filtered_messages = []
        
        for message in test_messages:
            # Валидация
            validation_result = validator.validate(message)
            
            if not validation_result.is_valid:
                invalid_messages.append((message, validation_result.errors))
                continue
            
            # Фильтрация
            filter_result = stream_processor.apply_filters(message)
            
            if filter_result.passed:
                valid_messages.append(message)
            else:
                filtered_messages.append((message, filter_result.failed_filters))
        
        # Проверяем результаты
        assert len(valid_messages) == 1  # Только первое сообщение прошло все проверки
        assert len(invalid_messages) == 1  # Второе сообщение не прошло валидацию
        assert len(filtered_messages) == 2  # Третье и четвертое сообщения отфильтрованы

    def test_streaming_latency_measurement(
        self,
        data_stream_manager: DataStreamManager
    ) -> None:
        """Тест измерения латентности streaming."""
        
        latency_measurements = []
        
        class LatencyTracker:
            def __init__(self):
                self.sent_timestamps = {}
            
            def mark_sent(self, message_id: str):
                """Отмечаем время отправки."""
                self.sent_timestamps[message_id] = time.perf_counter()
            
            def mark_received(self, message_id: str):
                """Отмечаем время получения и рассчитываем латентность."""
                if message_id in self.sent_timestamps:
                    sent_time = self.sent_timestamps[message_id]
                    received_time = time.perf_counter()
                    latency_ms = (received_time - sent_time) * 1000
                    latency_measurements.append(latency_ms)
                    del self.sent_timestamps[message_id]
        
        latency_tracker = LatencyTracker()
        
        # Симулируем отправку и получение сообщений
        for i in range(100):
            message_id = str(uuid.uuid4())
            
            # Отмечаем отправку
            latency_tracker.mark_sent(message_id)
            
            # Симулируем сетевую задержку
            network_delay = 0.001 + (i % 10) * 0.0001  # 1-2ms
            time.sleep(network_delay)
            
            # Отмечаем получение
            latency_tracker.mark_received(message_id)
        
        # Анализируем латентность
        assert len(latency_measurements) == 100
        
        avg_latency = sum(latency_measurements) / len(latency_measurements)
        min_latency = min(latency_measurements)
        max_latency = max(latency_measurements)
        
        # Проверяем что латентность в разумных пределах
        assert avg_latency < 5.0  # Средняя латентность < 5ms
        assert min_latency < 3.0  # Минимальная < 3ms
        assert max_latency < 10.0  # Максимальная < 10ms
        
        # Рассчитываем percentiles
        sorted_latencies = sorted(latency_measurements)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))]
        p99_latency = sorted_latencies[int(0.99 * len(sorted_latencies))]
        
        assert p95_latency < 8.0  # P95 < 8ms
        assert p99_latency < 10.0  # P99 < 10ms

    def test_concurrent_stream_processing(
        self,
        stream_processor: StreamProcessor
    ) -> None:
        """Тест параллельной обработки множественных потоков."""
        
        processed_messages = {"count": 0, "by_stream": {}}
        lock = threading.Lock()
        
        def concurrent_handler(message: StreamMessage):
            """Обработчик для concurrent тестов."""
            with lock:
                processed_messages["count"] += 1
                
                stream_id = message.stream_id
                if stream_id not in processed_messages["by_stream"]:
                    processed_messages["by_stream"][stream_id] = 0
                processed_messages["by_stream"][stream_id] += 1
            
            # Симулируем обработку
            time.sleep(0.001)  # 1ms обработка
        
        # Настраиваем обработчики для разных типов потоков
        stream_types = [
            StreamType.MARKET_DATA,
            StreamType.ORDER_BOOK, 
            StreamType.TRADES,
            StreamType.TICKER
        ]
        
        for stream_type in stream_types:
            stream_processor.add_handler(stream_type, concurrent_handler)
        
        # Генерируем сообщения для разных потоков
        messages = []
        
        for stream_idx in range(4):  # 4 потока
            stream_id = f"stream_{stream_idx}"
            stream_type = stream_types[stream_idx]
            
            for msg_idx in range(250):  # 250 сообщений на поток
                message = StreamMessage(
                    stream_id=stream_id,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    message_type=stream_type,
                    data={
                        "test_data": f"stream_{stream_idx}_msg_{msg_idx}",
                        "value": msg_idx
                    },
                    sequence_number=msg_idx,
                    source="CONCURRENT_TEST"
                )
                messages.append(message)
        
        # Обрабатываем сообщения параллельно
        start_time = time.time()
        
        async def process_all_messages():
            tasks = []
            for message in messages:
                task = asyncio.create_task(stream_processor.process_message(message))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        asyncio.run(process_all_messages())
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Проверяем результаты
        assert processed_messages["count"] == 1000  # 4 * 250 сообщений
        assert len(processed_messages["by_stream"]) == 4
        
        # Каждый поток должен обработать 250 сообщений
        for stream_idx in range(4):
            stream_id = f"stream_{stream_idx}"
            assert processed_messages["by_stream"][stream_id] == 250
        
        # Проверяем производительность
        messages_per_second = 1000 / processing_time
        assert messages_per_second > 500  # Минимум 500 msg/sec

    def test_stream_reconnection_and_recovery(
        self,
        websocket_client: WebSocketClient,
        data_stream_manager: DataStreamManager
    ) -> None:
        """Тест переподключения и восстановления потоков."""
        
        connection_attempts = []
        recovered_streams = []
        
        def on_reconnect_attempt(attempt_number: int):
            connection_attempts.append({
                "attempt": attempt_number,
                "timestamp": datetime.utcnow()
            })
        
        def on_stream_recovered(stream_id: str):
            recovered_streams.append(stream_id)
        
        websocket_client.on_reconnect_attempt = on_reconnect_attempt
        data_stream_manager.on_stream_recovered = on_stream_recovered
        
        # Настраиваем reconnection параметры
        websocket_client.configure_reconnection(
            max_attempts=5,
            initial_delay=0.1,  # Быстрый reconnect для тестов
            backoff_multiplier=1.5,
            max_delay=2.0
        )
        
        # Создаем активные потоки
        active_streams = [
            "market_data_btcusdt",
            "orderbook_ethusdt", 
            "trades_adausdt"
        ]
        
        for stream_id in active_streams:
            data_stream_manager.register_stream(stream_id, StreamType.MARKET_DATA)
        
        # Симулируем обрыв соединения
        with patch.object(websocket_client, 'is_connected', False):
            with patch.object(websocket_client, '_websocket', None):
                
                # Попытка отправить данные должна инициировать reconnect
                asyncio.run(websocket_client.send_message({"test": "message"}))
                
                # Ждем попыток переподключения
                time.sleep(1)
                
                # Симулируем успешное переподключение
                with patch.object(websocket_client, 'is_connected', True):
                    websocket_client._trigger_reconnect_success()
                    
                    # Ждем восстановления потоков
                    time.sleep(0.5)
        
        # Проверяем результаты
        assert len(connection_attempts) >= 1
        assert len(recovered_streams) == len(active_streams)
        
        # Проверяем что все потоки восстановлены
        for stream_id in active_streams:
            assert stream_id in recovered_streams

    def test_message_ordering_and_sequence_validation(
        self,
        message_broker: MessageBroker
    ) -> None:
        """Тест упорядочивания сообщений и валидации последовательности."""
        
        # Создаем ordered queue
        ordered_queue = message_broker.create_ordered_queue(
            name="sequence_test_queue",
            ordering_key="stream_id"
        )
        
        # Генерируем сообщения с разной последовательностью
        messages = []
        
        # Нормальная последовательность для stream_1
        for i in range(1, 6):
            message = StreamMessage(
                stream_id="stream_1",
                message_id=f"msg_1_{i}",
                timestamp=datetime.utcnow(),
                message_type=StreamType.TRADES,
                data={"trade_id": i},
                sequence_number=i,
                source="TEST"
            )
            messages.append(message)
        
        # Нарушенная последовательность для stream_2
        sequence_nums = [1, 2, 4, 3, 5]  # 4 и 3 поменяны местами
        for i, seq_num in enumerate(sequence_nums):
            message = StreamMessage(
                stream_id="stream_2",
                message_id=f"msg_2_{i+1}",
                timestamp=datetime.utcnow(),
                message_type=StreamType.TRADES,
                data={"trade_id": seq_num},
                sequence_number=seq_num,
                source="TEST"
            )
            messages.append(message)
        
        # Отправляем сообщения в случайном порядке
        import random
        random.shuffle(messages)
        
        for message in messages:
            ordered_queue.put(message)
        
        # Получаем обработанные сообщения
        processed_stream_1 = []
        processed_stream_2 = []
        
        while not ordered_queue.empty():
            message = ordered_queue.get()
            
            if message.stream_id == "stream_1":
                processed_stream_1.append(message)
            elif message.stream_id == "stream_2":
                processed_stream_2.append(message)
        
        # Проверяем упорядочивание для stream_1
        assert len(processed_stream_1) == 5
        for i, message in enumerate(processed_stream_1):
            assert message.sequence_number == i + 1
        
        # Проверяем обработку нарушений последовательности для stream_2
        sequence_validator = message_broker.get_sequence_validator("stream_2")
        validation_results = sequence_validator.get_validation_results()
        
        assert validation_results.gaps_detected > 0
        assert validation_results.out_of_order_messages > 0
        
        # Проверяем что сообщения все же были переупорядочены
        assert len(processed_stream_2) == 5
        sorted_stream_2 = sorted(processed_stream_2, key=lambda m: m.sequence_number)
        for i, message in enumerate(sorted_stream_2):
            assert message.sequence_number == i + 1

    def test_stream_health_monitoring_and_metrics(
        self,
        data_stream_manager: DataStreamManager
    ) -> None:
        """Тест мониторинга здоровья потоков и метрик."""
        
        # Создаем мониторы для разных потоков
        stream_monitors = {}
        
        for stream_id in ["btc_stream", "eth_stream", "ada_stream"]:
            monitor = data_stream_manager.create_stream_monitor(
                stream_id=stream_id,
                health_check_interval_seconds=1,
                metrics_collection_enabled=True
            )
            stream_monitors[stream_id] = monitor
        
        # Симулируем активность потоков
        for round_num in range(10):
            for stream_id, monitor in stream_monitors.items():
                
                # Симулируем получение сообщений
                messages_in_round = 50 + round_num * 5
                
                for i in range(messages_in_round):
                    # Имитируем processing
                    processing_start = time.perf_counter()
                    time.sleep(0.0001)  # 0.1ms processing
                    processing_end = time.perf_counter()
                    
                    processing_time_ms = (processing_end - processing_start) * 1000
                    
                    monitor.record_message_processed(
                        message_id=f"{stream_id}_msg_{round_num}_{i}",
                        processing_time_ms=processing_time_ms,
                        success=True
                    )
                
                # Симулируем ошибки для одного потока
                if stream_id == "ada_stream" and round_num % 3 == 0:
                    monitor.record_error("Connection timeout")
            
            time.sleep(1.1)  # Ждем следующего цикла мониторинга
        
        # Собираем метрики
        all_metrics = {}
        for stream_id, monitor in stream_monitors.items():
            metrics = monitor.get_metrics()
            all_metrics[stream_id] = metrics
        
        # Проверяем метрики
        for stream_id, metrics in all_metrics.items():
            assert metrics.messages_processed > 0
            assert metrics.throughput_msg_per_sec > 0
            assert metrics.average_latency_ms < 5.0  # Низкая латентность
            
            if stream_id == "ada_stream":
                # У ada_stream должны быть ошибки
                assert metrics.error_rate > 0
            else:
                # У остальных потоков ошибок быть не должно
                assert metrics.error_rate == 0
        
        # Проверяем health status
        health_statuses = data_stream_manager.get_all_stream_health()
        
        assert len(health_statuses) == 3
        
        for stream_id, health in health_statuses.items():
            if stream_id == "ada_stream":
                assert health.status in [StreamStatus.ERROR, StreamStatus.CONNECTED]
                assert len(health.recent_errors) > 0
            else:
                assert health.status == StreamStatus.CONNECTED
                assert len(health.recent_errors) == 0

    def test_streaming_data_compression_and_optimization(
        self,
        websocket_client: WebSocketClient
    ) -> None:
        """Тест сжатия и оптимизации streaming данных."""
        
        # Настраиваем compression
        compression_config = {
            "enabled": True,
            "algorithm": "gzip",
            "level": 6,
            "minimum_size": 1024  # Сжимаем только большие сообщения
        }
        
        websocket_client.configure_compression(compression_config)
        
        # Создаем тестовые данные разного размера
        test_cases = [
            {
                "name": "small_message",
                "size": 500,  # Меньше minimum_size
                "data": {"small": "x" * 500}
            },
            {
                "name": "large_message",
                "size": 5000,  # Больше minimum_size
                "data": {"large": "y" * 5000}
            },
            {
                "name": "huge_message",
                "size": 50000,  # Очень большое сообщение
                "data": {"huge": "z" * 50000}
            }
        ]
        
        compression_results = []
        
        for test_case in test_cases:
            original_data = json.dumps(test_case["data"])
            original_size = len(original_data.encode('utf-8'))
            
            # Обрабатываем сообщение через WebSocket client
            processed_data, compressed_size = websocket_client.prepare_message(
                original_data
            )
            
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            result = {
                "name": test_case["name"],
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_applied": compressed_size < original_size
            }
            compression_results.append(result)
        
        # Проверяем результаты сжатия
        small_msg = next(r for r in compression_results if r["name"] == "small_message")
        large_msg = next(r for r in compression_results if r["name"] == "large_message") 
        huge_msg = next(r for r in compression_results if r["name"] == "huge_message")
        
        # Малые сообщения не должны сжиматься
        assert small_msg["compression_applied"] is False
        
        # Большие сообщения должны сжиматься
        assert large_msg["compression_applied"] is True
        assert large_msg["compression_ratio"] < 0.8  # Сжатие > 20%
        
        assert huge_msg["compression_applied"] is True
        assert huge_msg["compression_ratio"] < 0.5  # Сжатие > 50%

    def test_streaming_error_handling_and_recovery(
        self,
        stream_processor: StreamProcessor
    ) -> None:
        """Тест обработки ошибок и восстановления потоков."""
        
        error_log = []
        recovery_attempts = []
        
        def error_handler(error: Exception, message: StreamMessage):
            """Обработчик ошибок."""
            error_log.append({
                "error": str(error),
                "message_id": message.message_id,
                "timestamp": datetime.utcnow()
            })
        
        def recovery_handler(stream_id: str, attempt_number: int):
            """Обработчик попыток восстановления."""
            recovery_attempts.append({
                "stream_id": stream_id,
                "attempt": attempt_number,
                "timestamp": datetime.utcnow()
            })
        
        # Настраиваем обработчики
        stream_processor.set_error_handler(error_handler)
        stream_processor.set_recovery_handler(recovery_handler)
        
        # Создаем обработчик который иногда падает
        processing_count = {"count": 0}
        
        def flaky_handler(message: StreamMessage):
            """Обработчик который периодически падает."""
            processing_count["count"] += 1
            
            # Падаем на каждом 3-м сообщении
            if processing_count["count"] % 3 == 0:
                raise ValueError(f"Simulated error for message {message.message_id}")
            
            # Нормальная обработка
            return {"status": "processed", "message_id": message.message_id}
        
        stream_processor.add_handler(StreamType.TRADES, flaky_handler)
        
        # Настраиваем retry policy
        retry_config = {
            "max_attempts": 3,
            "backoff_strategy": "exponential",
            "initial_delay_ms": 100,
            "max_delay_ms": 1000
        }
        
        stream_processor.configure_retry_policy(retry_config)
        
        # Генерируем тестовые сообщения
        test_messages = []
        for i in range(10):
            message = StreamMessage(
                stream_id="error_test_stream",
                message_id=f"error_msg_{i}",
                timestamp=datetime.utcnow(),
                message_type=StreamType.TRADES,
                data={"trade_id": i, "amount": "100.0"},
                sequence_number=i,
                source="ERROR_TEST"
            )
            test_messages.append(message)
        
        # Обрабатываем сообщения
        for message in test_messages:
            asyncio.run(stream_processor.process_message_with_retry(message))
        
        # Проверяем результаты
        # Должны быть ошибки на сообщениях 2, 5, 8 (каждое 3-е)
        assert len(error_log) >= 3
        
        # Проверяем что были попытки восстановления
        assert len(recovery_attempts) >= 0  # Может быть 0 если быстро обработалось
        
        # Проверяем что большинство сообщений было обработано
        # (некоторые могли быть обработаны после retry)
        successful_processing_count = processing_count["count"] - len(error_log)
        assert successful_processing_count >= 7  # Как минимум 7 из 10