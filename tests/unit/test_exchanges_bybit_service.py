"""
Unit тесты для BybitExchangeService
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.type_definitions.external_service_types import ExchangeName, ExchangeCredentials, ConnectionConfig
from infrastructure.external_services.exchanges.config import ExchangeServiceConfig
from infrastructure.external_services.exchanges.bybit_exchange_service import BybitExchangeService
class TestBybitExchangeService:
    """Тесты для BybitExchangeService."""
    @pytest.fixture
    def config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Конфигурация для тестов."""
        return ExchangeServiceConfig(
            exchange_name=ExchangeName.BYBIT,
            credentials=ExchangeCredentials(
                api_key="test_key",
                api_secret="test_secret",
                api_passphrase="test_passphrase",
                testnet=True,
                sandbox=True
            ),
            connection_config=ConnectionConfig(
                rate_limit=100,
                rate_limit_window=60,
                timeout=30.0,
                retry_attempts=3
            )
        )
    @pytest.fixture
    def service(self, config) -> Any:
        """Экземпляр BybitExchangeService."""
        with patch('infrastructure.external_services.exchanges.base_exchange_service.BaseExchangeService.__init__'):
            return BybitExchangeService(config)
    def test_init(self, config) -> None:
        """Тест инициализации."""
        with patch('infrastructure.external_services.exchanges.base_exchange_service.BaseExchangeService.__init__'):
            service = BybitExchangeService(config)
            assert service._exchange_name == ExchangeName("bybit")
    def test_get_websocket_url_testnet(self, service) -> None:
        """Тест получения WebSocket URL для testnet."""
        service.config.credentials.testnet = True
        url = service._get_websocket_url()
        assert url == "wss://testnet.stream.bybit.com/v5/public/spot"
    def test_get_websocket_url_production(self, service) -> None:
        """Тест получения WebSocket URL для production."""
        service.config.credentials.testnet = False
        url = service._get_websocket_url()
        assert url == "wss://stream.bybit.com/v5/public/spot"
    @pytest.mark.asyncio
    async def test_subscribe_websocket_channels_with_websocket(self, service) -> None:
        """Тест подписки на WebSocket каналы с активным WebSocket."""
        # Мокаем WebSocket
        service.websocket = AsyncMock()
        await service._subscribe_websocket_channels()
        # Проверяем, что сообщение было отправлено
        service.websocket.send.assert_called_once()
        # Проверяем содержимое сообщения
        call_args = service.websocket.send.call_args[0][0]
        message = json.loads(call_args)
        assert message["op"] == "subscribe"
        assert "tickers.BTCUSDT" in message["args"]
        assert "tickers.ETHUSDT" in message["args"]
    @pytest.mark.asyncio
    async def test_subscribe_websocket_channels_without_websocket(self, service) -> None:
        """Тест подписки на WebSocket каналы без WebSocket."""
        service.websocket = None
        # Не должно вызывать ошибок
        await service._subscribe_websocket_channels()
    @pytest.mark.asyncio
    async def test_process_websocket_message_valid_ticker(self, service) -> None:
        """Тест обработки валидного WebSocket сообщения с тикером."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Валидные данные тикера
        ticker_message = {
            "topic": "tickers.BTCUSDT",
            "data": {
                "symbol": "BTCUSDT",
                "lastPrice": "50000.00",
                "volume24h": "1000.50",
                "price24hPcnt": "2.5",
                "highPrice24h": "51000.00",
                "lowPrice24h": "49000.00"
            }
        }
        await service._process_websocket_message(ticker_message)
        # Проверяем, что данные были сохранены в кэш
        service.cache.set.assert_called_once()
        call_args = service.cache.set.call_args
        cache_key = call_args[0][0]
        ticker_info = call_args[0][1]
        assert cache_key == "ticker_BTCUSDT"
        assert ticker_info["symbol"] == "BTCUSDT"
        assert ticker_info["price"] == 50000.00
        assert ticker_info["volume"] == 1000.50
        assert ticker_info["change"] == 2.5
        assert ticker_info["high"] == 51000.00
        assert ticker_info["low"] == 49000.00
        assert "timestamp" in ticker_info
    @pytest.mark.asyncio
    async def test_process_websocket_message_invalid_ticker(self, service) -> None:
        """Тест обработки невалидного WebSocket сообщения."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Невалидные данные (отсутствуют обязательные поля)
        invalid_data = {
            "some_field": "some_value"
        }
        await service._process_websocket_message(invalid_data)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()
    @pytest.mark.asyncio
    async def test_process_websocket_message_non_ticker_topic(self, service) -> None:
        """Тест обработки WebSocket сообщения с не-тикер топиком."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Сообщение с другим топиком
        non_ticker_message = {
            "topic": "orderbook.BTCUSDT",
            "data": {"some": "data"}
        }
        await service._process_websocket_message(non_ticker_message)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()
    @pytest.mark.asyncio
    async def test_process_websocket_message_missing_data(self, service) -> None:
        """Тест обработки WebSocket сообщения без данных."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Сообщение без данных
        message_without_data = {
            "topic": "tickers.BTCUSDT"
            # Отсутствует поле "data"
        }
        await service._process_websocket_message(message_without_data)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()
    @pytest.mark.asyncio
    async def test_process_ticker_data_valid(self, service) -> None:
        """Тест обработки валидных данных тикера."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Валидные данные тикера
        ticker_data = {
            "symbol": "ETHUSDT",
            "lastPrice": "3000.00",
            "volume24h": "500.75",
            "price24hPcnt": "-1.25",
            "highPrice24h": "3100.00",
            "lowPrice24h": "2900.00"
        }
        await service._process_ticker_data(ticker_data)
        # Проверяем, что данные были сохранены в кэш
        service.cache.set.assert_called_once()
        call_args = service.cache.set.call_args
        cache_key = call_args[0][0]
        ticker_info = call_args[0][1]
        assert cache_key == "ticker_ETHUSDT"
        assert ticker_info["symbol"] == "ETHUSDT"
        assert ticker_info["price"] == 3000.00
        assert ticker_info["volume"] == 500.75
        assert ticker_info["change"] == -1.25
        assert ticker_info["high"] == 3100.00
        assert ticker_info["low"] == 2900.00
        assert "timestamp" in ticker_info
    @pytest.mark.asyncio
    async def test_process_ticker_data_partial(self, service) -> None:
        """Тест обработки частичных данных тикера."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Частичные данные тикера
        partial_data = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.00"
            # Отсутствуют другие поля
        }
        await service._process_ticker_data(partial_data)
        # Проверяем, что данные были сохранены с значениями по умолчанию
        service.cache.set.assert_called_once()
        call_args = service.cache.set.call_args
        ticker_info = call_args[0][1]
        assert ticker_info["symbol"] == "BTCUSDT"
        assert ticker_info["price"] == 50000.00
        assert ticker_info["volume"] == 0.0  # Значение по умолчанию
        assert ticker_info["change"] == 0.0  # Значение по умолчанию
        assert ticker_info["high"] == 0.0    # Значение по умолчанию
        assert ticker_info["low"] == 0.0     # Значение по умолчанию
    @pytest.mark.asyncio
    async def test_process_ticker_data_numeric_strings(self, service) -> None:
        """Тест обработки числовых строк в данных тикера."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Данные с числовыми строками
        ticker_data = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.50",
            "volume24h": "1000.75",
            "price24hPcnt": "-1.25",
            "highPrice24h": "51000.00",
            "lowPrice24h": "49000.00"
        }
        await service._process_ticker_data(ticker_data)
        # Проверяем, что строки корректно преобразованы в числа
        call_args = service.cache.set.call_args
        ticker_info = call_args[0][1]
        assert isinstance(ticker_info["price"], float)
        assert isinstance(ticker_info["volume"], float)
        assert isinstance(ticker_info["change"], float)
        assert isinstance(ticker_info["high"], float)
        assert isinstance(ticker_info["low"], float)
        assert ticker_info["price"] == 50000.50
        assert ticker_info["volume"] == 1000.75
        assert ticker_info["change"] == -1.25
        assert ticker_info["high"] == 51000.00
        assert ticker_info["low"] == 49000.00
    @pytest.mark.asyncio
    async def test_process_ticker_data_empty_symbol(self, service) -> None:
        """Тест обработки данных тикера с пустым символом."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Данные с пустым символом
        ticker_data = {
            "symbol": "",
            "lastPrice": "50000.00",
            "volume24h": "1000.50",
            "price24hPcnt": "2.5",
            "highPrice24h": "51000.00",
            "lowPrice24h": "49000.00"
        }
        await service._process_ticker_data(ticker_data)
        # Проверяем, что данные были сохранены
        service.cache.set.assert_called_once()
        call_args = service.cache.set.call_args
        cache_key = call_args[0][0]
        ticker_info = call_args[0][1]
        assert cache_key == "ticker_"
        assert ticker_info["symbol"] == ""
    @pytest.mark.asyncio
    async def test_process_ticker_data_cache_error(self, service) -> None:
        """Тест обработки ошибки кэша в данных тикера."""
        # Мокаем кэш с ошибкой
        service.cache = AsyncMock()
        service.cache.set.side_effect = Exception("Cache error")
        ticker_data = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.00",
            "volume24h": "1000.50",
            "price24hPcnt": "2.5",
            "highPrice24h": "51000.00",
            "lowPrice24h": "49000.00"
        }
        # Не должно вызывать исключение
        await service._process_ticker_data(ticker_data)
        # Кэш должен был вызваться
        service.cache.set.assert_called_once()
    @pytest.mark.asyncio
    async def test_process_websocket_message_empty_data(self, service) -> None:
        """Тест обработки пустого WebSocket сообщения."""
        # Мокаем кэш
        service.cache = AsyncMock()
        empty_data = {}
        await service._process_websocket_message(empty_data)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()
    @pytest.mark.asyncio
    async def test_process_websocket_message_none_data(self, service) -> None:
        """Тест обработки None данных WebSocket."""
        # Мокаем кэш
        service.cache = AsyncMock()
        await service._process_websocket_message(None)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()
    def test_inheritance_from_base_service(self, service) -> None:
        """Тест наследования от BaseExchangeService."""
        assert isinstance(service, BybitExchangeService)
        # Проверяем, что методы базового класса доступны
        assert hasattr(service, 'connect')
        assert hasattr(service, 'disconnect')
        assert hasattr(service, 'get_market_data')
        assert hasattr(service, 'place_order')
        assert hasattr(service, 'cancel_order')
        assert hasattr(service, 'get_order_status')
        assert hasattr(service, 'get_balance')
        assert hasattr(service, 'get_positions')
    def test_bybit_specific_methods(self, service) -> None:
        """Тест специфичных для Bybit методов."""
        assert hasattr(service, '_get_websocket_url')
        assert hasattr(service, '_subscribe_websocket_channels')
        assert hasattr(service, '_process_websocket_message')
        assert hasattr(service, '_process_ticker_data')
        # Проверяем, что методы переопределены
        assert service._get_websocket_url.__qualname__ == 'BybitExchangeService._get_websocket_url'
        assert service._subscribe_websocket_channels.__qualname__ == 'BybitExchangeService._subscribe_websocket_channels'
        assert service._process_websocket_message.__qualname__ == 'BybitExchangeService._process_websocket_message'
        assert service._process_ticker_data.__qualname__ == 'BybitExchangeService._process_ticker_data'
    @pytest.mark.asyncio
    async def test_websocket_integration(self, service) -> None:
        """Тест интеграции WebSocket компонентов."""
        # Мокаем WebSocket
        service.websocket = AsyncMock()
        service.cache = AsyncMock()
        # Получаем URL
        url = service._get_websocket_url()
        assert "bybit" in url
        # Подписываемся на каналы
        await service._subscribe_websocket_channels()
        service.websocket.send.assert_called_once()
        # Обрабатываем сообщение
        ticker_message = {
            "topic": "tickers.BTCUSDT",
            "data": {
                "symbol": "BTCUSDT",
                "lastPrice": "50000.00",
                "volume24h": "1000.50",
                "price24hPcnt": "2.5",
                "highPrice24h": "51000.00",
                "lowPrice24h": "49000.00"
            }
        }
        await service._process_websocket_message(ticker_message)
        service.cache.set.assert_called_once()
    def test_exchange_name_consistency(self, service) -> None:
        """Тест консистентности имени биржи."""
        assert service._exchange_name == ExchangeName("bybit")
        assert str(service._exchange_name) == "bybit"
        assert service._exchange_name.value == "bybit"
    @pytest.mark.asyncio
    async def test_multiple_ticker_processing(self, service) -> None:
        """Тест обработки нескольких тикеров."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Обрабатываем несколько тикеров
        tickers = [
            {
                "topic": "tickers.BTCUSDT",
                "data": {
                    "symbol": "BTCUSDT",
                    "lastPrice": "50000.00",
                    "volume24h": "1000.50",
                    "price24hPcnt": "2.5",
                    "highPrice24h": "51000.00",
                    "lowPrice24h": "49000.00"
                }
            },
            {
                "topic": "tickers.ETHUSDT",
                "data": {
                    "symbol": "ETHUSDT",
                    "lastPrice": "3000.00",
                    "volume24h": "500.75",
                    "price24hPcnt": "-1.25",
                    "highPrice24h": "3100.00",
                    "lowPrice24h": "2900.00"
                }
            }
        ]
        for ticker in tickers:
            await service._process_websocket_message(ticker)
        # Проверяем, что каждый тикер был обработан
        assert service.cache.set.call_count == 2
        # Проверяем, что ключи кэша разные
        call_args_list = service.cache.set.call_args_list
        cache_keys = [call[0][0] for call in call_args_list]
        assert "ticker_BTCUSDT" in cache_keys
        assert "ticker_ETHUSDT" in cache_keys 
