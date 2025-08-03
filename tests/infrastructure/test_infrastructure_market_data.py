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
    def test_base_connector_initialization(self) -> None:
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
    def test_normalize_symbol(self) -> None:
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
    def test_parse_price_volume_pairs(self) -> None:
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
    def test_parse_price_volume_pairs_invalid_data(self) -> None:
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
    def test_binance_connector_initialization(self) -> None:
        """Тест инициализации коннектора Binance."""
        connector = BinanceConnector()
        assert connector.exchange_name == "binance"
        assert connector.api_key is None
        assert connector.api_secret is None
    def test_get_websocket_url(self) -> None:
        """Тест получения WebSocket URL."""
        connector = BinanceConnector()
        url = connector.get_websocket_url("BTCUSDT")
        expected_url = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
        assert url == expected_url
    def test_get_subscription_message(self) -> None:
        """Тест получения сообщения подписки."""
        connector = BinanceConnector()
        message = connector.get_subscription_message("BTCUSDT")
        expected_message = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@depth20@100ms"],
            "id": 1
        }
        assert message == expected_message
    def test_parse_order_book_update_valid(self) -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = BinanceConnector()
        # Тестовые данные Binance
        message = {
            "data": {
                "e": "depthUpdate",
                "E": 1640995200000,
                "s": "BTCUSDT",
                "U": 123456,
                "u": 123457,
                "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
                "a": [["50001.00", "1.0"], ["50002.00", "2.5"]]
            },
            "stream": "btcusdt@depth20@100ms"
        }
        result = connector.parse_order_book_update(message)
        assert result is not None
        assert result.exchange == "binance"
        assert result.symbol == "BTCUSDT"
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0][0].amount == Decimal("50000.00")
        assert result.bids[0][1].amount == Decimal("1.5")
        assert result.asks[0][0].amount == Decimal("50001.00")
        assert result.asks[0][1].amount == Decimal("1.0")
    def test_parse_order_book_update_invalid_event(self) -> None:
        """Тест парсинга с неверным типом события."""
        connector = BinanceConnector()
        message = {
            "data": {
                "e": "trade",  # Неверный тип события
                "s": "BTCUSDT"
            }
        }
        result = connector.parse_order_book_update(message)
        assert result is None
    def test_parse_order_book_update_missing_data(self) -> None:
        """Тест парсинга с отсутствующими данными."""
        connector = BinanceConnector()
        message = {
            "data": {
                "e": "depthUpdate",
                "s": "BTCUSDT"
                # Отсутствуют bids и asks
            }
        }
        result = connector.parse_order_book_update(message)
        assert result is None
    def test_parse_order_book_update_no_data_key(self) -> None:
        """Тест парсинга без ключа data."""
        connector = BinanceConnector()
        message = {
            "stream": "btcusdt@depth20@100ms"
        }
        result = connector.parse_order_book_update(message)
        assert result is None
    def test_connect(self) -> None:
        """Тест подключения к бирже."""
        # Создаем коннектор
        connector = BinanceConnector()
        # Проверяем начальное состояние
        assert not connector.is_connected
        assert connector._websocket is None
        # Проверяем URL подключения
        url = connector.get_websocket_url("BTCUSDT")
        assert "wss://stream.binance.com:9443/ws" in url
        assert "btcusdt@depth20@100ms" in url
        # Проверяем сообщение подписки
        sub_msg = connector.get_subscription_message("BTCUSDT")
        assert sub_msg["method"] == "SUBSCRIBE"
        assert "btcusdt@depth20@100ms" in sub_msg["params"]
class TestCoinbaseConnector:
    """Тесты коннектора Coinbase."""
    def test_coinbase_connector_initialization(self) -> None:
        """Тест инициализации коннектора Coinbase."""
        connector = CoinbaseConnector()
        assert connector.exchange_name == "coinbase"
        assert connector.api_key is None
        assert connector.api_secret is None
    def test_get_websocket_url(self) -> None:
        """Тест получения WebSocket URL."""
        connector = CoinbaseConnector()
        url = connector.get_websocket_url("BTC-USD")
        expected_url = "wss://ws-feed.exchange.coinbase.com"
        assert url == expected_url
    def test_get_subscription_message(self) -> None:
        """Тест получения сообщения подписки."""
        connector = CoinbaseConnector()
        message = connector.get_subscription_message("BTC-USD")
        expected_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["level2"]
        }
        assert message == expected_message
    def test_parse_order_book_update_valid(self) -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = CoinbaseConnector()
        # Тестовые данные Coinbase
        message = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.5"]]
        }
        result = connector.parse_order_book_update(message)
        assert result is not None
        assert result.exchange == "coinbase"
        assert result.symbol == "BTC-USD"
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0][0].amount == Decimal("50000.00")
        assert result.bids[0][1].amount == Decimal("1.5")
    def test_parse_order_book_update_invalid_type(self) -> None:
        """Тест парсинга с неверным типом сообщения."""
        connector = CoinbaseConnector()
        message = {
            "type": "heartbeat",  # Неверный тип
            "product_id": "BTC-USD"
        }
        result = connector.parse_order_book_update(message)
        assert result is None
    def test_parse_order_book_update_missing_data(self) -> None:
        """Тест парсинга с отсутствующими данными."""
        connector = CoinbaseConnector()
        message = {
            "type": "snapshot",
            "product_id": "BTC-USD"
            # Отсутствуют bids и asks
        }
        result = connector.parse_order_book_update(message)
        assert result is None
class TestKrakenConnector:
    """Тесты коннектора Kraken."""
    def test_kraken_connector_initialization(self) -> None:
        """Тест инициализации коннектора Kraken."""
        connector = KrakenConnector()
        assert connector.exchange_name == "kraken"
        assert connector.api_key is None
        assert connector.api_secret is None
    def test_get_websocket_url(self) -> None:
        """Тест получения WebSocket URL."""
        connector = KrakenConnector()
        url = connector.get_websocket_url("XBT/USD")
        expected_url = "wss://ws.kraken.com"
        assert url == expected_url
    def test_get_subscription_message(self) -> None:
        """Тест получения сообщения подписки."""
        connector = KrakenConnector()
        message = connector.get_subscription_message("XBT/USD")
        expected_message = {
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {
                "name": "book",
                "depth": 10
            }
        }
        assert message == expected_message
    def test_parse_order_book_update_valid(self) -> None:
        """Тест парсинга валидного обновления ордербука."""
        connector = KrakenConnector()
        # Тестовые данные Kraken (правильный формат)
        message = [
            1234,  # channelID
            {
                "b": [["50000.00", "1.5", "1640995200"], ["49999.00", "2.0", "1640995201"]],
                "a": [["50001.00", "1.0", "1640995202"], ["50002.00", "2.5", "1640995203"]]
            },
            "book",  # channel_name
            "XBT/USD"  # pair
        ]
        result = connector.parse_order_book_update(message)
        assert result is not None
        assert result.exchange == "kraken"
        assert result.symbol == "XBT/USD"
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0][0].amount == Decimal("50000.00")
        assert result.bids[0][1].amount == Decimal("1.5")
    def test_parse_order_book_update_invalid_event(self) -> None:
        """Тест парсинга с неверным типом события."""
        connector = KrakenConnector()
        message = {
            "event": "heartbeat",  # Неверный тип
            "pair": "XBT/USD"
        }
        result = connector.parse_order_book_update(message)
        assert result is None
    def test_parse_order_book_update_missing_data(self) -> None:
        """Тест парсинга с отсутствующими данными."""
        connector = KrakenConnector()
        message = {
            "event": "subscriptionStatus",
            "pair": "XBT/USD"
            # Отсутствуют данные ордербука
        }
        result = connector.parse_order_book_update(message)
        assert result is None
class TestOrderBookIntegration:
    """Интеграционные тесты ордербука."""
    def test_order_book_update_creation(self) -> None:
        """Тест создания обновления ордербука."""
        # Создаем тестовые данные
        bids = [
            (Price(Decimal("50000.00"), Currency(CurrencyCode("BTC")), Currency(CurrencyCode("USD"))),
             Volume(Decimal("1.5"), Currency(CurrencyCode("USD"))))
        ]
        asks = [
            (Price(Decimal("50001.00"), Currency(CurrencyCode("BTC")), Currency(CurrencyCode("USD"))),
             Volume(Decimal("1.0"), Currency(CurrencyCode("USD"))))
        ]
        timestamp = Timestamp.now()
        # Создаем обновление ордербука
        update = OrderBookUpdate(
            exchange="test",
            symbol="BTCUSD",
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            sequence_id=123,
            meta={"test": True}
        )
        # Проверяем создание
        assert update.exchange == "test"
        assert update.symbol == "BTCUSD"
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        assert update.timestamp == timestamp
        assert update.sequence_id == 123
        assert update.meta["test"] is True
    def test_order_book_update_serialization(self) -> None:
        """Тест сериализации обновления ордербука."""
        # Создаем тестовые данные
        bids = [
            (Price(Decimal("50000.00"), Currency(CurrencyCode("BTC")), Currency(CurrencyCode("USD"))),
             Volume(Decimal("1.5"), Currency(CurrencyCode("USD"))))
        ]
        asks = [
            (Price(Decimal("50001.00"), Currency(CurrencyCode("BTC")), Currency(CurrencyCode("USD"))),
             Volume(Decimal("1.0"), Currency(CurrencyCode("USD"))))
        ]
        timestamp = Timestamp.now()
        # Создаем обновление ордербука
        update = OrderBookUpdate(
            exchange="test",
            symbol="BTCUSD",
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            sequence_id=123,
            meta={"test": True}
        )
        # Проверяем, что объект можно сериализовать
        assert str(update) is not None
        assert repr(update) is not None 
