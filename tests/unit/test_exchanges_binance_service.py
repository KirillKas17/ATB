"""
Unit тесты для BinanceExchangeService
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.type_definitions.external_service_types import ExchangeName, ExchangeCredentials, ConnectionConfig
from infrastructure.external_services.exchanges.config import ExchangeServiceConfig
from infrastructure.external_services.exchanges.binance_exchange_service import BinanceExchangeService


class TestBinanceExchangeService:
    """Тесты для BinanceExchangeService."""

    @pytest.fixture
    def config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Конфигурация для тестов."""
        return ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=ExchangeCredentials(
                api_key="test_key",
                api_secret="test_secret",
                api_passphrase="test_passphrase",
                testnet=True,
                sandbox=True,
            ),
            connection_config=ConnectionConfig(rate_limit=100, rate_limit_window=60, timeout=30.0, retry_attempts=3),
        )

    @pytest.fixture
    def service(self, config) -> Any:
        """Экземпляр BinanceExchangeService."""
        with patch("infrastructure.external_services.exchanges.base_exchange_service.BaseExchangeService.__init__"):
            return BinanceExchangeService(config)

    def test_init(self, config) -> None:
        """Тест инициализации."""
        with patch("infrastructure.external_services.exchanges.base_exchange_service.BaseExchangeService.__init__"):
            service = BinanceExchangeService(config)
            assert service._exchange_name == ExchangeName("binance")

    def test_get_websocket_url_testnet(self, service) -> None:
        """Тест получения WebSocket URL для testnet."""
        service.config.credentials.testnet = True
        url = service._get_websocket_url()
        assert url == "wss://testnet.binance.vision/ws"

    def test_get_websocket_url_production(self, service) -> None:
        """Тест получения WebSocket URL для production."""
        service.config.credentials.testnet = False
        url = service._get_websocket_url()
        assert url == "wss://stream.binance.com:9443/ws"

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
        assert message["method"] == "SUBSCRIBE"
        assert "btcusdt@ticker" in message["params"]
        assert "ethusdt@ticker" in message["params"]
        assert message["id"] == 1

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
        ticker_data = {"s": "BTCUSDT", "c": "50000.00", "v": "1000.50", "P": "2.5", "h": "51000.00", "l": "49000.00"}
        await service._process_websocket_message(ticker_data)
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
        invalid_data = {"some_field": "some_value"}
        await service._process_websocket_message(invalid_data)
        # Кэш не должен вызываться
        service.cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_websocket_message_partial_ticker(self, service) -> None:
        """Тест обработки частичного WebSocket сообщения."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Частичные данные тикера
        partial_data = {
            "s": "ETHUSDT",
            "c": "3000.00",
            # Отсутствуют другие поля
        }
        await service._process_websocket_message(partial_data)
        # Проверяем, что данные были сохранены с значениями по умолчанию
        service.cache.set.assert_called_once()
        call_args = service.cache.set.call_args
        ticker_info = call_args[0][1]
        assert ticker_info["symbol"] == "ETHUSDT"
        assert ticker_info["price"] == 3000.00
        assert ticker_info["volume"] == 0.0  # Значение по умолчанию
        assert ticker_info["change"] == 0.0  # Значение по умолчанию
        assert ticker_info["high"] == 0.0  # Значение по умолчанию
        assert ticker_info["low"] == 0.0  # Значение по умолчанию

    @pytest.mark.asyncio
    async def test_process_websocket_message_numeric_strings(self, service) -> None:
        """Тест обработки числовых строк в WebSocket сообщении."""
        # Мокаем кэш
        service.cache = AsyncMock()
        # Данные с числовыми строками
        ticker_data = {"s": "BTCUSDT", "c": "50000.50", "v": "1000.75", "P": "-1.25", "h": "51000.00", "l": "49000.00"}
        await service._process_websocket_message(ticker_data)
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

    @pytest.mark.asyncio
    async def test_process_websocket_message_cache_error(self, service) -> None:
        """Тест обработки ошибки кэша в WebSocket сообщении."""
        # Мокаем кэш с ошибкой
        service.cache = AsyncMock()
        service.cache.set.side_effect = Exception("Cache error")
        ticker_data = {"s": "BTCUSDT", "c": "50000.00", "v": "1000.50", "P": "2.5", "h": "51000.00", "l": "49000.00"}
        # Не должно вызывать исключение
        await service._process_websocket_message(ticker_data)
        # Кэш должен был вызваться
        service.cache.set.assert_called_once()

    def test_inheritance_from_base_service(self, service) -> None:
        """Тест наследования от BaseExchangeService."""
        assert isinstance(service, BinanceExchangeService)
        # Проверяем, что методы базового класса доступны
        assert hasattr(service, "connect")
        assert hasattr(service, "disconnect")
        assert hasattr(service, "get_market_data")
        assert hasattr(service, "place_order")
        assert hasattr(service, "cancel_order")
        assert hasattr(service, "get_order_status")
        assert hasattr(service, "get_balance")
        assert hasattr(service, "get_positions")

    def test_binance_specific_methods(self, service) -> None:
        """Тест специфичных для Binance методов."""
        assert hasattr(service, "_get_websocket_url")
        assert hasattr(service, "_subscribe_websocket_channels")
        assert hasattr(service, "_process_websocket_message")
        # Проверяем, что методы переопределены
        assert service._get_websocket_url.__qualname__ == "BinanceExchangeService._get_websocket_url"
        assert (
            service._subscribe_websocket_channels.__qualname__ == "BinanceExchangeService._subscribe_websocket_channels"
        )
        assert service._process_websocket_message.__qualname__ == "BinanceExchangeService._process_websocket_message"

    @pytest.mark.asyncio
    async def test_websocket_integration(self, service) -> None:
        """Тест интеграции WebSocket компонентов."""
        # Мокаем WebSocket
        service.websocket = AsyncMock()
        service.cache = AsyncMock()
        # Получаем URL
        url = service._get_websocket_url()
        assert "binance" in url
        # Подписываемся на каналы
        await service._subscribe_websocket_channels()
        service.websocket.send.assert_called_once()
        # Обрабатываем сообщение
        ticker_data = {"s": "BTCUSDT", "c": "50000.00", "v": "1000.50", "P": "2.5", "h": "51000.00", "l": "49000.00"}
        await service._process_websocket_message(ticker_data)
        service.cache.set.assert_called_once()

    def test_exchange_name_consistency(self, service) -> None:
        """Тест консистентности имени биржи."""
        assert service._exchange_name == ExchangeName("binance")
        assert str(service._exchange_name) == "binance"
        assert service._exchange_name.value == "binance"
