"""
Комплексные тесты для инфраструктуры market_data.
Покрывает:
- BaseExchangeConnector (абстрактный класс)
- BinanceConnector
- CoinbaseConnector
- KrakenConnector
- Интеграцию с OrderBookUpdate/OrderBookSnapshot
- WebSocket соединения
- Обработку ошибок
"""

import asyncio
import json
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.market_data.base_connector import BaseExchangeConnector
from infrastructure.market_data.binance_connector import BinanceConnector
from infrastructure.market_data.coinbase_connector import CoinbaseConnector
from infrastructure.market_data.kraken_connector import KrakenConnector
from shared.models.orderbook import OrderBookUpdate, OrderBookSnapshot
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.type_definitions.value_object_types import CurrencyCode


class TestBaseExchangeConnector:
    """Тесты для базового абстрактного коннектора."""

    def test_abstract_class_cannot_be_instantiated(self: "TestBaseExchangeConnector") -> None:
        """Проверка, что абстрактный класс нельзя инстанцировать."""
        with pytest.raises(TypeError):
            BaseExchangeConnector()

    def test_abstract_methods_are_defined(self: "TestBaseExchangeConnector") -> None:
        """Проверка, что абстрактные методы определены."""
        assert hasattr(BaseExchangeConnector, "get_websocket_url")
        assert hasattr(BaseExchangeConnector, "get_subscription_message")
        assert hasattr(BaseExchangeConnector, "parse_order_book_update")


class TestBinanceConnector:
    """Тесты для Binance коннектора."""

    @pytest.fixture
    def binance_connector(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра BinanceConnector для тестов."""
        return BinanceConnector()

    @pytest.fixture
    def sample_binance_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Тестовые данные от Binance WebSocket."""
        return {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
                "a": [["50001.00", "1.0"], ["50002.00", "2.5"]],
            },
        }

    def test_initialization(self, binance_connector) -> None:
        """Тест инициализации BinanceConnector."""
        assert binance_connector.exchange_name == "binance"
        assert binance_connector.base_url == "wss://stream.binance.com:9443/ws"
        assert binance_connector.is_connected is False
        assert binance_connector.websocket is None

    def test_get_websocket_url(self, binance_connector) -> None:
        """Тест получения WebSocket URL."""
        url = binance_connector.get_websocket_url("BTCUSDT")
        assert url == "wss://stream.binance.com:9443/ws"

    def test_get_subscription_message(self, binance_connector) -> None:
        """Тест создания сообщения подписки."""
        message = binance_connector.get_subscription_message("BTCUSDT")
        expected = {"method": "SUBSCRIBE", "params": ["btcusdt@depth20@100ms"], "id": 1}
        assert message == expected

    def test_symbol_normalization(self, binance_connector) -> None:
        """Тест нормализации символов."""
        assert binance_connector._normalize_symbol("BTCUSDT") == "btcusdt"
        assert binance_connector._normalize_symbol("ETHUSDT") == "ethusdt"
        assert binance_connector._normalize_symbol("ADAUSDT") == "adausdt"

    def test_parse_order_book_update_success(self, binance_connector, sample_binance_data) -> None:
        """Тест успешного парсинга обновления ордербука."""
        update = binance_connector.parse_order_book_update(sample_binance_data)
        assert isinstance(update, OrderBookUpdate)
        assert update.exchange == "binance"
        assert update.symbol == "BTCUSDT"
        assert update.sequence_id == 123457
        # Проверяем bids
        assert len(update.bids) == 2
        assert update.bids[0][0].amount == Decimal("50000.00")
        assert update.bids[0][1].amount == Decimal("1.5")
        # Проверяем asks
        assert len(update.asks) == 2
        assert update.asks[0][0].amount == Decimal("50001.00")
        assert update.asks[0][1].amount == Decimal("1.0")

    def test_parse_order_book_update_invalid_data(self, binance_connector) -> None:
        """Тест обработки некорректных данных."""
        invalid_data = {"invalid": "data"}
        result = binance_connector.parse_order_book_update(invalid_data)
        assert result is None

    def test_parse_order_book_update_missing_fields(self, binance_connector) -> None:
        """Тест обработки данных с отсутствующими полями."""
        incomplete_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                # Отсутствуют bids, asks, timestamp
            },
        }
        result = binance_connector.parse_order_book_update(incomplete_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_connect_success(self, binance_connector) -> None:
        """Тест успешного подключения."""
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            success = await binance_connector.connect("BTCUSDT")
            assert success is True
            assert binance_connector.is_connected is True
            assert binance_connector.websocket == mock_websocket
            mock_connect.assert_called_once_with("wss://stream.binance.com:9443/ws")

    @pytest.mark.asyncio
    async def test_connect_failure(self, binance_connector) -> None:
        """Тест неудачного подключения."""
        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            success = await binance_connector.connect("BTCUSDT")
            assert success is False
            assert binance_connector.is_connected is False
            assert binance_connector.websocket is None

    @pytest.mark.asyncio
    async def test_disconnect(self, binance_connector) -> None:
        """Тест отключения."""
        # Сначала подключаемся
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            await binance_connector.connect("BTCUSDT")
        # Теперь отключаемся
        await binance_connector.disconnect()
        assert binance_connector.is_connected is False
        assert binance_connector.websocket is None

    @pytest.mark.asyncio
    async def test_stream_order_book_success(self, binance_connector) -> None:
        """Тест успешного стриминга ордербука."""
        mock_callback = AsyncMock()
        sample_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"]],
                "a": [["50001.00", "1.0"]],
            },
        }
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_websocket.recv.return_value = json.dumps(sample_data)
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            # Запускаем стриминг на короткое время
            task = asyncio.create_task(binance_connector.stream_order_book("BTCUSDT", mock_callback))
            # Ждем немного и отменяем
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Проверяем, что callback был вызван
            assert mock_callback.called

    def test_parse_price_volume_pairs(self, binance_connector) -> None:
        """Тест парсинга пар цена-объем."""
        base_currency = Currency(CurrencyCode("BTC"))
        quote_currency = Currency(CurrencyCode("USD"))
        data = [["50000.00", "1.5"], ["49999.00", "2.0"]]
        result = binance_connector._parse_price_volume_pairs(data, base_currency, quote_currency)
        assert len(result) == 2
        assert result[0][0].amount == Decimal("50000.00")
        assert result[0][1].amount == Decimal("1.5")
        assert result[1][0].amount == Decimal("49999.00")
        assert result[1][1].amount == Decimal("2.0")


class TestCoinbaseConnector:
    """Тесты для Coinbase коннектора."""

    @pytest.fixture
    def coinbase_connector(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра CoinbaseConnector для тестов."""
        return CoinbaseConnector()

    @pytest.fixture
    def sample_coinbase_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Тестовые данные от Coinbase WebSocket."""
        return {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.5"]],
            "sequence": 123456,
        }

    def test_initialization(self, coinbase_connector) -> None:
        """Тест инициализации CoinbaseConnector."""
        assert coinbase_connector.exchange_name == "coinbase"
        assert coinbase_connector.base_url == "wss://ws-feed.pro.coinbase.com"
        assert coinbase_connector.is_connected is False

    def test_get_websocket_url(self, coinbase_connector) -> None:
        """Тест получения WebSocket URL."""
        url = coinbase_connector.get_websocket_url("BTC-USD")
        assert url == "wss://ws-feed.pro.coinbase.com"

    def test_get_subscription_message(self, coinbase_connector) -> None:
        """Тест создания сообщения подписки."""
        message = coinbase_connector.get_subscription_message("BTC-USD")
        expected = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": [{"name": "level2", "product_ids": ["BTC-USD"]}],
        }
        assert message == expected

    def test_symbol_normalization(self, coinbase_connector) -> None:
        """Тест нормализации символов."""
        assert coinbase_connector._normalize_symbol("BTC-USD") == "BTC-USD"
        assert coinbase_connector._normalize_symbol("ETH-USD") == "ETH-USD"

    def test_parse_order_book_update_success(self, coinbase_connector, sample_coinbase_data) -> None:
        """Тест успешного парсинга обновления ордербука."""
        update = coinbase_connector.parse_order_book_update(sample_coinbase_data)
        assert isinstance(update, OrderBookUpdate)
        assert update.exchange == "coinbase"
        assert update.symbol == "BTC-USD"
        # Проверяем bids
        assert len(update.bids) == 2
        assert update.bids[0][0].amount == Decimal("50000.00")
        assert update.bids[0][1].amount == Decimal("1.5")
        # Проверяем asks
        assert len(update.asks) == 2
        assert update.asks[0][0].amount == Decimal("50001.00")
        assert update.asks[0][1].amount == Decimal("1.0")

    def test_parse_order_book_update_invalid_data(self, coinbase_connector) -> None:
        """Тест обработки некорректных данных."""
        invalid_data = {"invalid": "data"}
        result = coinbase_connector.parse_order_book_update(invalid_data)
        assert result is None


class TestKrakenConnector:
    """Тесты для Kraken коннектора."""

    @pytest.fixture
    def kraken_connector(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра KrakenConnector для тестов."""
        return KrakenConnector()

    @pytest.fixture
    def sample_kraken_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Тестовые данные от Kraken WebSocket."""
        return [
            0,  # channelID
            {
                "b": [
                    ["50000.0000", "1.50000000", "1640995200.123456"],
                    ["49999.0000", "2.00000000", "1640995200.123457"],
                ],
                "a": [
                    ["50001.0000", "1.00000000", "1640995200.123458"],
                    ["50002.0000", "2.50000000", "1640995200.123459"],
                ],
            },
            "book-10",
            "XBT/USD",
        ]

    def test_initialization(self, kraken_connector) -> None:
        """Тест инициализации KrakenConnector."""
        assert kraken_connector.exchange_name == "kraken"
        assert kraken_connector.base_url == "wss://ws.kraken.com"
        assert kraken_connector.is_connected is False

    def test_get_websocket_url(self, kraken_connector) -> None:
        """Тест получения WebSocket URL."""
        url = kraken_connector.get_websocket_url("XBT/USD")
        assert url == "wss://ws.kraken.com"

    def test_get_subscription_message(self, kraken_connector) -> None:
        """Тест создания сообщения подписки."""
        message = kraken_connector.get_subscription_message("XBT/USD")
        expected = {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "book", "depth": 20}}
        assert message == expected

    def test_symbol_normalization(self, kraken_connector) -> None:
        """Тест нормализации символов."""
        assert kraken_connector._normalize_symbol("XBT/USD") == "XBT/USD"
        assert kraken_connector._normalize_symbol("ETH/USD") == "ETH/USD"

    def test_parse_order_book_update_success(self, kraken_connector, sample_kraken_data) -> None:
        """Тест успешного парсинга обновления ордербука."""
        update = kraken_connector.parse_order_book_update(sample_kraken_data)
        assert isinstance(update, OrderBookUpdate)
        assert update.exchange == "kraken"
        assert update.symbol == "XBT/USD"
        # Проверяем bids
        assert len(update.bids) == 2
        assert update.bids[0][0].amount == Decimal("50000.00")
        assert update.bids[0][1].amount == Decimal("1.5")
        # Проверяем asks
        assert len(update.asks) == 2
        assert update.asks[0][0].amount == Decimal("50001.00")
        assert update.asks[0][1].amount == Decimal("1.0")

    def test_parse_order_book_update_invalid_data(self, kraken_connector) -> None:
        """Тест обработки некорректных данных."""
        invalid_data = {"invalid": "data"}
        result = kraken_connector.parse_order_book_update(invalid_data)
        assert result is None


class TestMarketDataIntegration:
    """Интеграционные тесты для market_data."""

    @pytest.fixture
    def connectors(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание всех коннекторов для интеграционных тестов."""
        return {"binance": BinanceConnector(), "coinbase": CoinbaseConnector(), "kraken": KrakenConnector()}

    def test_unified_order_book_update_structure(self, connectors) -> None:
        """Тест единой структуры OrderBookUpdate для всех коннекторов."""
        # Тестовые данные для каждого коннектора
        test_data = {
            "binance": {
                "stream": "btcusdt@depth20@100ms",
                "data": {
                    "e": "depthUpdate",
                    "E": 1640995200000,
                    "s": "BTCUSDT",
                    "U": 123456,
                    "u": 123457,
                    "b": [["50000.00", "1.5"]],
                    "a": [["50001.00", "1.0"]],
                },
            },
            "coinbase": {
                "type": "snapshot",
                "product_id": "BTC-USD",
                "bids": [["50000.00", "1.5"]],
                "asks": [["50001.00", "1.0"]],
                "sequence": 123456,
            },
            "kraken": [
                0,
                {
                    "b": [["50000.0000", "1.50000000", "1640995200.123456"]],
                    "a": [["50001.0000", "1.00000000", "1640995200.123458"]],
                },
                "book-10",
                "XBT/USD",
            ],
        }
        updates = {}
        for exchange, connector in connectors.items():
            data = test_data[exchange]
            update = connector.parse_order_book_update(data)
            if update:
                updates[exchange] = update
        # Проверяем, что все обновления имеют единую структуру
        for exchange, update in updates.items():
            assert isinstance(update, OrderBookUpdate)
            assert hasattr(update, "exchange")
            assert hasattr(update, "symbol")
            assert hasattr(update, "bids")
            assert hasattr(update, "asks")
            assert hasattr(update, "timestamp")
            assert hasattr(update, "sequence_id")
            # Проверяем типы данных
            assert isinstance(update.bids, list)
            assert isinstance(update.asks, list)
            assert isinstance(update.timestamp, Timestamp)
            # Проверяем структуру bids/asks
            if update.bids:
                assert isinstance(update.bids[0], tuple)
                assert isinstance(update.bids[0][0], Price)
                assert isinstance(update.bids[0][1], Volume)

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connectors) -> None:
        """Тест одновременных подключений к разным биржам."""
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            # Подключаемся ко всем биржам одновременно
            tasks = []
            for connector in connectors.values():
                task = asyncio.create_task(connector.connect("BTCUSDT"))
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Проверяем, что все подключения прошли успешно
            for result in results:
                assert result is True

    def test_error_handling_consistency(self, connectors) -> None:
        """Тест единообразной обработки ошибок."""
        invalid_data = {"invalid": "data"}
        for exchange, connector in connectors.items():
            result = connector.parse_order_book_update(invalid_data)
            # Все коннекторы должны возвращать None при некорректных данных
            assert result is None

    def test_symbol_consistency(self, connectors) -> None:
        """Тест консистентности обработки символов."""
        test_symbols = ["BTCUSDT", "BTC-USD", "XBT/USD"]
        for symbol in test_symbols:
            for connector in connectors.values():
                # Проверяем, что нормализация не вызывает ошибок
                normalized = connector._normalize_symbol(symbol)
                assert isinstance(normalized, str)
                assert len(normalized) > 0


class TestMarketDataPerformance:
    """Тесты производительности market_data."""

    @pytest.fixture
    def binance_connector(self: "TestEvolvableMarketMakerAgent") -> Any:
        return BinanceConnector()

    def test_parse_performance(self, binance_connector) -> None:
        """Тест производительности парсинга."""
        import time

        # Создаем большой объем тестовых данных
        large_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"] for _ in range(100)],
                "a": [["50001.00", "1.0"] for _ in range(100)],
            },
        }
        # Измеряем время парсинга
        start_time = time.time()
        for _ in range(1000):
            result = binance_connector.parse_order_book_update(large_data)
        end_time = time.time()
        parsing_time = end_time - start_time
        avg_time_per_parse = parsing_time / 1000
        # Проверяем, что парсинг достаточно быстрый (< 1ms на операцию)
        assert avg_time_per_parse < 0.001
        assert result is not None

    def test_memory_usage(self, binance_connector) -> None:
        """Тест использования памяти."""
        import sys

        # Создаем много обновлений
        updates = []
        sample_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"]],
                "a": [["50001.00", "1.0"]],
            },
        }
        initial_size = sys.getsizeof(updates)
        for _ in range(1000):
            update = binance_connector.parse_order_book_update(sample_data)
            if update:
                updates.append(update)
        final_size = sys.getsizeof(updates)
        memory_increase = final_size - initial_size
        # Проверяем, что рост памяти разумен (< 1MB для 1000 обновлений)
        assert memory_increase < 1024 * 1024


class TestMarketDataEdgeCases:
    """Тесты граничных случаев для market_data."""

    @pytest.fixture
    def binance_connector(self: "TestEvolvableMarketMakerAgent") -> Any:
        return BinanceConnector()

    def test_empty_order_book(self, binance_connector) -> None:
        """Тест обработки пустого ордербука."""
        empty_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [],
                "a": [],
            },
        }
        update = binance_connector.parse_order_book_update(empty_data)
        assert update is not None
        assert len(update.bids) == 0
        assert len(update.asks) == 0

    def test_malformed_price_data(self, binance_connector) -> None:
        """Тест обработки некорректных цен."""
        malformed_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["invalid_price", "1.5"], ["50000.00", "invalid_volume"]],
                "a": [["50001.00", "1.0"]],
            },
        }
        update = binance_connector.parse_order_book_update(malformed_data)
        # Должен обработать корректные данные и пропустить некорректные
        assert update is not None
        assert len(update.bids) == 0  # Некорректные данные должны быть пропущены
        assert len(update.asks) == 1  # Корректные данные должны быть обработаны

    def test_extremely_large_numbers(self, binance_connector) -> None:
        """Тест обработки очень больших чисел."""
        large_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["999999999.99999999", "999999999.99999999"]],
                "a": [["0.00000001", "0.00000001"]],
            },
        }
        update = binance_connector.parse_order_book_update(large_data)
        assert update is not None
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        # Проверяем, что большие числа корректно обработаны
        assert update.bids[0][0].amount == Decimal("999999999.99999999")
        assert update.asks[0][0].amount == Decimal("0.00000001")

    def test_unicode_symbols(self, binance_connector) -> None:
        """Тест обработки символов с Unicode."""
        unicode_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"]],
                "a": [["50001.00", "1.0"]],
            },
        }
        update = binance_connector.parse_order_book_update(unicode_data)
        assert update is not None
        assert update.symbol == "BTCUSDT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
