# -*- coding: utf-8 -*-
"""Тесты для расширенной системы обнаружения запутанности."""
import time
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from typing import cast
import pytest
from unittest.mock import Mock, patch
from application.entanglement.stream_manager import StreamManager
from domain.intelligence.entanglement_detector import EntanglementResult, AnalysisMetadata
from shared.models.orderbook import OrderBookUpdate
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from infrastructure.exchange_streams.bingx_ws_client import BingXWebSocketClient
from infrastructure.exchange_streams.bitget_ws_client import BitgetWebSocketClient
from infrastructure.exchange_streams.bybit_ws_client import BybitWebSocketClient
from infrastructure.exchange_streams.market_stream_aggregator import MarketStreamAggregator
class TestStreamManager:
    """Тесты для StreamManager."""
    @pytest.fixture
    def stream_manager(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра StreamManager для тестов."""
        return StreamManager(
            max_lag_ms=3.0,
            correlation_threshold=0.95,
            detection_interval=0.1
        )
    @pytest.fixture
    def mock_order_book_update(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок-обновления ордербука."""
        base_currency = Currency.from_string("BTC") or Currency.USD
        quote_currency = Currency.from_string("USD") or Currency.USD
        return OrderBookUpdate(
            exchange="test_exchange",
            symbol="BTCUSDT",
            bids=[(Price(Decimal("50000.0"), base_currency, quote_currency), Volume(Decimal("1.0"), quote_currency))],
            asks=[(Price(Decimal("50010.0"), base_currency, quote_currency), Volume(Decimal("1.0"), quote_currency))],
            timestamp=Timestamp.now(),
            sequence_id=12345
        )
    def test_initialization(self, stream_manager) -> None:
        """Тест инициализации StreamManager."""
        assert stream_manager.aggregator is not None
        assert stream_manager.detector is not None
        assert stream_manager.detection_interval == 0.1
        assert not stream_manager.is_running
        assert len(stream_manager.monitored_symbols) == 0
        assert len(stream_manager.entanglement_callbacks) == 0
    @pytest.mark.asyncio
    async def test_initialize_exchanges(self, stream_manager) -> None:
        """Тест инициализации бирж."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        api_keys = {
            "bingx": {"api_key": "test", "api_secret": "test"},
            "bitget": {"api_key": "test", "api_secret": "test"},
            "bybit": {"api_key": "test", "api_secret": "test"}
        }
        # Мокаем клиенты
        with patch('application.entanglement.stream_manager.BingXWebSocketClient') as mock_bingx, \
             patch('application.entanglement.stream_manager.BitgetWebSocketClient') as mock_bitget, \
             patch('application.entanglement.stream_manager.BybitWebSocketClient') as mock_bybit:
            mock_bingx.return_value = MagicMock()
            mock_bitget.return_value = MagicMock()
            mock_bybit.return_value = MagicMock()
            await stream_manager.initialize_exchanges(symbols, api_keys)
            # Проверяем, что источники добавлены
            assert len(stream_manager.aggregator.sources) == 3
            assert "bingx" in stream_manager.aggregator.sources
            assert "bitget" in stream_manager.aggregator.sources
            assert "bybit" in stream_manager.aggregator.sources
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_symbol(self, stream_manager) -> None:
        """Тест подписки и отписки от символов."""
        # Мокаем агрегатор
        stream_manager.aggregator.subscribe_symbol = AsyncMock(return_value=True)
        stream_manager.aggregator.unsubscribe_symbol = AsyncMock(return_value=True)
        # Тестируем подписку
        result = await stream_manager.subscribe_symbol("BTCUSDT")
        assert result is True
        assert "BTCUSDT" in stream_manager.monitored_symbols
        stream_manager.aggregator.subscribe_symbol.assert_called_once_with("BTCUSDT")
        # Тестируем отписку
        result = await stream_manager.unsubscribe_symbol("BTCUSDT")
        assert result is True
        assert "BTCUSDT" not in stream_manager.monitored_symbols
        stream_manager.aggregator.unsubscribe_symbol.assert_called_once_with("BTCUSDT")
    @pytest.mark.asyncio
    async def test_entanglement_callback(self, stream_manager, mock_order_book_update) -> None:
        """Тест callback для обработки запутанности."""
        callback_called = False
        callback_result = None
    def test_callback(result) -> None:
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result
        # Добавляем callback
        stream_manager.add_entanglement_callback(test_callback)
        assert len(stream_manager.entanglement_callbacks) == 1
        # Создаем мок-результат запутанности
        mock_result = EntanglementResult(
            is_entangled=True,
            lag_ms=2.5,
            correlation_score=0.98,
            exchange_pair=("bingx", "bitget"),
            symbol="BTCUSDT",
            timestamp=Timestamp.now(),
            confidence=0.95,
            metadata=cast(AnalysisMetadata, {"test": "data"})
        )
        # Вызываем обработчик
        await stream_manager._handle_entanglement_result(mock_result)
        # Проверяем, что callback был вызван
        assert callback_called
        assert callback_result == mock_result
    def test_get_status(self, stream_manager) -> None:
        """Тест получения статуса."""
        status = stream_manager.get_status()
        assert "is_running" in status
        assert "monitored_symbols" in status
        assert "aggregator_stats" in status
        assert "source_status" in status
        assert "detector_stats" in status
        assert "entanglement_stats" in status
        assert "uptime" in status
    def test_get_entanglement_stats(self, stream_manager) -> None:
        """Тест получения статистики запутанности."""
        # Устанавливаем тестовые данные
        stream_manager.stats["total_detections"] = 100
        stream_manager.stats["entangled_detections"] = 25
        stream_manager.stats["start_time"] = time.time() - 60  # 1 минута назад
        stats = stream_manager.get_entanglement_stats()
        assert stats["total_detections"] == 100
        assert stats["entangled_detections"] == 25
        assert "detection_rate" in stats
        assert "entanglement_rate" in stats
        assert stats["entanglement_rate"] == 0.25
class TestMarketStreamAggregator:
    """Тесты для MarketStreamAggregator."""
    @pytest.fixture
    def aggregator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра MarketStreamAggregator для тестов."""
        return MarketStreamAggregator()
    @pytest.fixture
    def mock_client(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок-клиента."""
        client = MagicMock()
        client.get_status.return_value = {"is_connected": True}
        return client
    def test_initialization(self, aggregator) -> None:
        """Тест инициализации агрегатора."""
        assert len(aggregator.sources) == 0
        assert len(aggregator.callbacks) == 0
        assert not aggregator.is_running
        assert len(aggregator.symbols) == 0
        assert aggregator.sync_tolerance_ms == 100.0
        assert aggregator.buffer_size == 1000
    def test_add_remove_source(self, aggregator, mock_client) -> None:
        """Тест добавления и удаления источников."""
        # Добавляем источник
        result = aggregator.add_source("test_exchange", mock_client)
        assert result is True
        assert "test_exchange" in aggregator.sources
        assert aggregator.sources["test_exchange"].name == "test_exchange"
        assert aggregator.sources["test_exchange"].client == mock_client
        # Удаляем источник
        result = aggregator.remove_source("test_exchange")
        assert result is True
        assert "test_exchange" not in aggregator.sources
    def test_add_remove_callback(self, aggregator) -> None:
        """Тест добавления и удаления callbacks."""
    def test_callback(update) -> None:
            pass
        # Добавляем callback
        aggregator.add_callback(test_callback)
        assert len(aggregator.callbacks) == 1
        assert test_callback in aggregator.callbacks
        # Удаляем callback
        aggregator.remove_callback(test_callback)
        assert len(aggregator.callbacks) == 0
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_symbol(self, aggregator, mock_client) -> None:
        """Тест подписки и отписки от символов."""
        # Добавляем источник
        aggregator.add_source("test_exchange", mock_client)
        # Мокаем методы клиента
        mock_client.subscribe = AsyncMock(return_value=True)
        mock_client.unsubscribe = AsyncMock(return_value=True)
        # Тестируем подписку
        result = await aggregator.subscribe_symbol("BTCUSDT")
        assert result is True
        assert "BTCUSDT" in aggregator.symbols
        mock_client.subscribe.assert_called_once_with("BTCUSDT")
        # Тестируем отписку
        result = await aggregator.unsubscribe_symbol("BTCUSDT")
        assert result is True
        assert "BTCUSDT" not in aggregator.symbols
        mock_client.unsubscribe.assert_called_once_with("BTCUSDT")
    def test_get_synchronized_updates(self, aggregator) -> None:
        """Тест получения синхронизированных обновлений."""
        # Добавляем тестовые обновления в буфер
        current_time = time.time()
        update1 = OrderBookUpdate(
            exchange="test1",
            symbol="BTCUSDT",
            bids=[(Price(50000.0), Volume(1.0))],
            asks=[(Price(50010.0), Volume(1.0))],
            timestamp=Timestamp(current_time)
        )
        update2 = OrderBookUpdate(
            exchange="test2",
            symbol="BTCUSDT",
            bids=[(Price(50001.0), Volume(1.0))],
            asks=[(Price(50011.0), Volume(1.0))],
            timestamp=Timestamp(current_time + 0.05)  # 50ms разница
        )
        aggregator.update_buffer = [update1, update2]
        # Получаем синхронизированные обновления
        synchronized = aggregator.get_synchronized_updates(tolerance_ms=100)
        assert len(synchronized) == 2
        # Тестируем с меньшей толерантностью
        synchronized = aggregator.get_synchronized_updates(tolerance_ms=10)
        assert len(synchronized) == 1  # Только update1
    def test_get_source_status(self, aggregator, mock_client) -> None:
        """Тест получения статуса источников."""
        # Добавляем источник
        aggregator.add_source("test_exchange", mock_client)
        status = aggregator.get_source_status()
        assert "test_exchange" in status
        assert status["test_exchange"]["is_active"] is True
        assert "last_update" in status["test_exchange"]
        assert "update_count" in status["test_exchange"]
        assert "error_count" in status["test_exchange"]
    def test_get_aggregator_stats(self, aggregator) -> None:
        """Тест получения статистики агрегатора."""
        stats = aggregator.get_aggregator_stats()
        assert "total_updates" in stats
        assert "uptime" in stats
        assert "active_sources" in stats
        assert "total_sources" in stats
        assert "buffer_size" in stats
        assert "subscribed_symbols" in stats
class TestWebSocketClients:
    """Тесты для WebSocket клиентов."""
    @pytest.fixture
    def bingx_client(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание BingX клиента."""
        return BingXWebSocketClient()
    @pytest.fixture
    def bitget_client(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание Bitget клиента."""
        return BitgetWebSocketClient()
    @pytest.fixture
    def bybit_client(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание Bybit клиента."""
        return BybitWebSocketClient()
    def test_bingx_client_initialization(self, bingx_client) -> None:
        """Тест инициализации BingX клиента."""
        assert bingx_client.base_url == "wss://open-api-swap.bingx.com/swap-market"
        assert not bingx_client.is_connected
        assert len(bingx_client.subscribed_symbols) == 0
    def test_bitget_client_initialization(self, bitget_client) -> None:
        """Тест инициализации Bitget клиента."""
        assert bitget_client.base_url == "wss://ws.bitget.com/spot/v1/stream"
        assert not bitget_client.is_connected
        assert len(bitget_client.subscribed_symbols) == 0
    def test_bybit_client_initialization(self, bybit_client) -> None:
        """Тест инициализации Bybit клиента."""
        assert bybit_client.base_url == "wss://stream.bybit.com/v5/public/spot"
        assert not bybit_client.is_connected
        assert len(bybit_client.subscribed_symbols) == 0
    def test_symbol_normalization(self, bingx_client, bitget_client, bybit_client) -> None:
        """Тест нормализации символов."""
        # BingX
        assert bingx_client._normalize_symbol("BTCUSDT") == "BTC-USDT"
        assert bingx_client._denormalize_symbol("BTC-USDT") == "BTCUSDT"
        # Bitget
        assert bitget_client._normalize_symbol("BTCUSDT") == "BTCUSDT_SPBL"
        assert bitget_client._denormalize_symbol("BTCUSDT_SPBL") == "BTCUSDT"
        # Bybit
        assert bybit_client._normalize_symbol("BTCUSDT") == "BTCUSDT"
        assert bybit_client._denormalize_symbol("BTCUSDT") == "BTCUSDT"
    def test_subscription_messages(self, bingx_client, bitget_client, bybit_client) -> None:
        """Тест сообщений подписки."""
        # BingX
        bingx_msg = bingx_client.get_subscription_message("BTCUSDT")
        assert bingx_msg["event"] == "subscribe"
        assert bingx_msg["topic"] == "market.depth"
        assert "BTC-USDT" in str(bingx_msg)
        # Bitget
        bitget_msg = bitget_client.get_subscription_message("BTCUSDT")
        assert bitget_msg["op"] == "subscribe"
        assert "BTCUSDT_SPBL" in str(bitget_msg)
        # Bybit
        bybit_msg = bybit_client.get_subscription_message("BTCUSDT")
        assert bybit_msg["op"] == "subscribe"
        assert "BTCUSDT" in str(bybit_msg)
    def test_get_status(self, bingx_client, bitget_client, bybit_client) -> None:
        """Тест получения статуса клиентов."""
        for client, name in [(bingx_client, "bingx"), (bitget_client, "bitget"), (bybit_client, "bybit")]:
            status = client.get_status()
            assert status["exchange"] == name
            assert "is_connected" in status
            assert "subscribed_symbols" in status
            assert "reconnect_attempts" in status
            assert "last_ping" in status
    @pytest.mark.asyncio
    async def test_integration_workflow() -> None:
    """Интеграционный тест рабочего процесса."""
    # Создаем StreamManager
    stream_manager = StreamManager(
        max_lag_ms=3.0,
        correlation_threshold=0.95,
        detection_interval=0.1
    )
    # Мокаем клиенты
    with patch('application.entanglement.stream_manager.BingXWebSocketClient') as mock_bingx, \
         patch('application.entanglement.stream_manager.BitgetWebSocketClient') as mock_bitget, \
         patch('application.entanglement.stream_manager.BybitWebSocketClient') as mock_bybit:
        mock_bingx.return_value = MagicMock()
        mock_bitget.return_value = MagicMock()
        mock_bybit.return_value = MagicMock()
        # Инициализируем биржи
        symbols = ["BTCUSDT"]
        api_keys = {
            "bingx": {"api_key": "test", "api_secret": "test"},
            "bitget": {"api_key": "test", "api_secret": "test"},
            "bybit": {"api_key": "test", "api_secret": "test"}
        }
        await stream_manager.initialize_exchanges(symbols, api_keys)
        # Проверяем, что источники добавлены
        assert len(stream_manager.aggregator.sources) == 3
        # Проверяем статус
        status = stream_manager.get_status()
        assert "is_running" in status
        assert "monitored_symbols" in status
        assert "aggregator_stats" in status
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
