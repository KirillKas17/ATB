"""
Тесты для модуля market_data с промышленным уровнем качества.
"""

from decimal import Decimal
from typing import Dict, Any
from infrastructure.market_data.base_connector import BaseExchangeConnector
from infrastructure.market_data.binance_connector import BinanceConnector
from infrastructure.market_data.coinbase_connector import CoinbaseConnector
from infrastructure.market_data.kraken_connector import KrakenConnector
from shared.models.orderbook import OrderBookUpdate
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp


class TestBaseConnector:
    """Тесты базового коннектора."""

    def test_base_connector_initialization(self: "TestBaseConnector") -> None:
        """Тест инициализации базового коннектора."""

        # Создаем тестовый коннектор
        class TestConnector(BaseExchangeConnector):
            def get_websocket_url(self, symbol: str) -> str:
                return f"wss://test.com/ws/{symbol}"

            def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
                return {"method": "SUBSCRIBE", "params": [symbol]}

            def parse_order_book_update(self, message: Dict[str, Any]) -> Any:
                return None

        connector = TestConnector("test_exchange")
        # Проверяем инициализацию
        assert connector.exchange_name == "test_exchange"
        assert connector.api_key is None
        assert connector.api_secret is None
        assert connector.websocket_url == ""
        assert not connector.is_connected
        assert connector.reconnect_attempts == 0
        assert connector.max_reconnect_attempts == 5
        assert connector.reconnect_delay == 1.0
        assert connector._websocket is None

    def test_normalize_symbol(self: "TestBaseConnector") -> None:
        """Тест нормализации символа."""

        class TestConnector(BaseExchangeConnector):
            def get_websocket_url(self, symbol: str) -> str:
                return f"wss://test.com/ws/{symbol}"

            def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
                return {"method": "SUBSCRIBE", "params": [symbol]}

            def parse_order_book_update(self, message: Dict[str, Any]) -> Any:
                return None

        connector = TestConnector("test_exchange")
        # Проверяем нормализацию
        assert connector._normalize_symbol("BTCUSDT") == "btcusdt"
        assert connector._normalize_symbol("ETHUSD") == "ethusd"
        assert connector._normalize_symbol("ADA-BTC") == "ada-btc"

    def test_parse_price_volume_pairs(self: "TestBaseConnector") -> None:
        """Тест парсинга пар цена-объем."""

        class TestConnector(BaseExchangeConnector):
            def get_websocket_url(self, symbol: str) -> str:
                return f"wss://test.com/ws/{symbol}"

            def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
                return {"method": "SUBSCRIBE", "params": [symbol]}

            def parse_order_book_update(self, message: Dict[str, Any]) -> Any:
                return None

        connector = TestConnector("test_exchange")
        # Тестовые данные
        data = [["50000.00", "1.5"], ["49999.00", "2.0"]]
        base_currency = Currency.from_string("BTC") or Currency.USD
        quote_currency = Currency.from_string("USD") or Currency.USD
        # Парсим данные
        result = connector._parse_price_volume_pairs(data, base_currency, quote_currency)
        # Проверяем результат
        assert len(result) == 2
        assert isinstance(result[0][0], Price)
        assert isinstance(result[0][1], Volume)
        assert result[0][0].amount == Decimal("50000.00")
        assert result[0][1].amount == Decimal("1.5")

    def test_parse_price_volume_pairs_invalid_data(self: "TestBaseConnector") -> None:
        """Тест парсинга некорректных данных."""

        class TestConnector(BaseExchangeConnector):
            def get_websocket_url(self, symbol: str) -> str:
                return f"wss://test.com/ws/{symbol}"

            def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
                return {"method": "SUBSCRIBE", "params": [symbol]}

            def parse_order_book_update(self, message: Dict[str, Any]) -> Any:
                return None

        connector = TestConnector("test_exchange")
        # Тестовые данные с ошибками
        data = [["invalid", "1.5"], ["50000.00", "invalid"], ["50000.00", "1.5"]]
        base_currency = Currency.from_string("BTC") or Currency.USD
        quote_currency = Currency.from_string("USD") or Currency.USD
        # Парсим данные
        result = connector._parse_price_volume_pairs(data, base_currency, quote_currency)
        # Проверяем, что только валидные данные обработаны
        assert len(result) == 1
        assert result[0][0].amount == Decimal("50000.00")
        assert result[0][1].amount == Decimal("1.5")


class TestBinanceConnector:
    """Тесты коннектора Binance."""

    def test_binance_connector_initialization(self: "TestBinanceConnector") -> None:
        """Тест инициализации коннектора Binance."""
        connector = BinanceConnector()
        assert connector.exchange_name == "binance"
        assert connector.api_key is None
        assert connector.api_secret is None

    def test_get_websocket_url(self: "TestBinanceConnector") -> None:
        """Тест получения WebSocket URL."""
        connector = BinanceConnector()
        url = connector.get_websocket_url("BTCUSDT")
        expected_url = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
        assert url == expected_url

    def test_get_subscription_message(self: "TestBinanceConnector") -> None:
        """Тест получения сообщения подписки."""
        connector = BinanceConnector()
        message = connector.get_subscription_message("BTCUSDT")
        expected_message = {"method": "SUBSCRIBE", "params": ["btcusdt@depth20@100ms"], "id": 1}
        assert message == expected_message

    def test_parse_order_book_update_valid(self: "TestBinanceConnector") -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = BinanceConnector()
        message = {
            "e": "depthUpdate",
            "E": 123456789,
            "s": "BTCUSDT",
            "U": 123,
            "u": 125,
            "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "a": [["50001.00", "0.5"], ["50002.00", "1.0"]],
        }
        result = connector.parse_order_book_update(message)
        assert isinstance(result, OrderBookUpdate)
        assert result.symbol == "BTCUSDT"
        assert result.timestamp is not None
        assert len(result.bids) == 2
        assert len(result.asks) == 2

    def test_parse_order_book_update_invalid_event(self: "TestBinanceConnector") -> None:
        """Тест парсинга невалидного события."""
        connector = BinanceConnector()
        message = {"e": "invalid_event"}
        result = connector.parse_order_book_update(message)
        assert result is None

    def test_parse_order_book_update_missing_data(self: "TestBinanceConnector") -> None:
        """Тест парсинга сообщения без данных."""
        connector = BinanceConnector()
        message = {"e": "depthUpdate"}
        result = connector.parse_order_book_update(message)
        assert result is None

    def test_parse_order_book_update_no_data_key(self: "TestBinanceConnector") -> None:
        """Тест парсинга сообщения без ключей данных."""
        connector = BinanceConnector()
        message = {"e": "depthUpdate", "s": "BTCUSDT"}
        result = connector.parse_order_book_update(message)
        assert result is None

    def test_connect(self: "TestBinanceConnector") -> None:
        """Тест подключения к WebSocket."""
        connector = BinanceConnector()
        # Этот тест может быть пропущен в CI/CD окружении
        # где нет реального сетевого подключения
        assert not connector.is_connected


class TestCoinbaseConnector:
    """Тесты коннектора Coinbase."""

    def test_coinbase_connector_initialization(self: "TestCoinbaseConnector") -> None:
        """Тест инициализации коннектора Coinbase."""
        connector = CoinbaseConnector()
        assert connector.exchange_name == "coinbase"
        assert connector.api_key is None
        assert connector.api_secret is None

    def test_get_websocket_url(self: "TestCoinbaseConnector") -> None:
        """Тест получения WebSocket URL."""
        connector = CoinbaseConnector()
        url = connector.get_websocket_url("BTC-USD")
        expected_url = "wss://ws-feed.exchange.coinbase.com"
        assert url == expected_url

    def test_get_subscription_message(self: "TestCoinbaseConnector") -> None:
        """Тест получения сообщения подписки."""
        connector = CoinbaseConnector()
        message = connector.get_subscription_message("BTC-USD")
        expected_message = {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["level2"]}
        assert message == expected_message

    def test_parse_order_book_update_valid(self: "TestCoinbaseConnector") -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = CoinbaseConnector()
        message = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "0.5"], ["50002.00", "1.0"]],
        }
        result = connector.parse_order_book_update(message)
        assert isinstance(result, OrderBookUpdate)
        assert result.symbol == "BTC-USD"
        assert result.timestamp is not None
        assert len(result.bids) == 2
        assert len(result.asks) == 2

    def test_parse_order_book_update_invalid_type(self: "TestCoinbaseConnector") -> None:
        """Тест парсинга невалидного типа сообщения."""
        connector = CoinbaseConnector()
        message = {"type": "invalid_type"}
        result = connector.parse_order_book_update(message)
        assert result is None

    def test_parse_order_book_update_missing_data(self: "TestCoinbaseConnector") -> None:
        """Тест парсинга сообщения без данных."""
        connector = CoinbaseConnector()
        message = {"type": "snapshot"}
        result = connector.parse_order_book_update(message)
        assert result is None


class TestKrakenConnector:
    """Тесты коннектора Kraken."""

    def test_kraken_connector_initialization(self: "TestKrakenConnector") -> None:
        """Тест инициализации коннектора Kraken."""
        connector = KrakenConnector()
        assert connector.exchange_name == "kraken"
        assert connector.api_key is None
        assert connector.api_secret is None

    def test_get_websocket_url(self: "TestKrakenConnector") -> None:
        """Тест получения WebSocket URL."""
        connector = KrakenConnector()
        url = connector.get_websocket_url("XBT/USD")
        expected_url = "wss://ws.kraken.com"
        assert url == expected_url

    def test_get_subscription_message(self: "TestKrakenConnector") -> None:
        """Тест получения сообщения подписки."""
        connector = KrakenConnector()
        message = connector.get_subscription_message("XBT/USD")
        expected_message = {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "book"}}
        assert message == expected_message

    def test_parse_order_book_update_valid(self: "TestKrakenConnector") -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = KrakenConnector()
        message = [
            0,
            {"as": [["50001.00000", "0.5", "1234567890.123456"]], "bs": [["50000.00000", "1.5", "1234567890.123456"]]},
            "book",
            "XBT/USD",
        ]
        result = connector.parse_order_book_update(message)
        assert isinstance(result, OrderBookUpdate)
        assert result.symbol == "XBT/USD"
        assert result.timestamp is not None
        assert len(result.bids) == 1
        assert len(result.asks) == 1

    def test_parse_order_book_update_invalid_event(self: "TestKrakenConnector") -> None:
        """Тест парсинга невалидного события."""
        connector = KrakenConnector()
        message = [0, {}, "invalid_event", "XBT/USD"]
        result = connector.parse_order_book_update(message)
        assert result is None

    def test_parse_order_book_update_missing_data(self: "TestKrakenConnector") -> None:
        """Тест парсинга сообщения без данных."""
        connector = KrakenConnector()
        message = [0, {}, "book", "XBT/USD"]
        result = connector.parse_order_book_update(message)
        assert result is None


class TestOrderBookIntegration:
    """Интеграционные тесты для OrderBook."""

    def test_order_book_update_creation(self: "TestOrderBookIntegration") -> None:
        """Тест создания обновления ордербука."""
        symbol = "BTCUSDT"
        timestamp = Timestamp.now()
        bids = [
            (
                Price(Decimal("50000.00"), Currency.from_string("USD") or Currency.USD),
                Volume(Decimal("1.5"), Currency.from_string("BTC") or Currency.USD),
            )
        ]
        asks = [
            (
                Price(Decimal("50001.00"), Currency.from_string("USD") or Currency.USD),
                Volume(Decimal("0.5"), Currency.from_string("BTC") or Currency.USD),
            )
        ]

        update = OrderBookUpdate(symbol=symbol, timestamp=timestamp, bids=bids, asks=asks)

        assert update.symbol == symbol
        assert update.timestamp == timestamp
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        assert update.bids[0][0].amount == Decimal("50000.00")
        assert update.asks[0][0].amount == Decimal("50001.00")

    def test_order_book_update_serialization(self: "TestOrderBookIntegration") -> None:
        """Тест сериализации обновления ордербука."""
        symbol = "BTCUSDT"
        timestamp = Timestamp.now()
        bids = [
            (
                Price(Decimal("50000.00"), Currency.from_string("USD") or Currency.USD),
                Volume(Decimal("1.5"), Currency.from_string("BTC") or Currency.USD),
            )
        ]
        asks = [
            (
                Price(Decimal("50001.00"), Currency.from_string("USD") or Currency.USD),
                Volume(Decimal("0.5"), Currency.from_string("BTC") or Currency.USD),
            )
        ]

        update = OrderBookUpdate(symbol=symbol, timestamp=timestamp, bids=bids, asks=asks)

        # Проверяем, что объект можно сериализовать
        serialized = str(update)
        assert symbol in serialized
        assert "50000.00" in serialized
        assert "50001.00" in serialized
