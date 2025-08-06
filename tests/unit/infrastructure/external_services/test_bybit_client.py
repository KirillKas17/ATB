#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Bybit Client Infrastructure.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from infrastructure.external_services.bybit_client import BybitClient
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.order import Order, OrderSide, OrderType
from domain.exceptions import ValidationError, ExchangeError


class TestBybitClient:
    """Тесты для Bybit Client."""

    @pytest.fixture
    def mock_api_credentials(self) -> Dict[str, str]:
        """Фикстура API credentials."""
        return {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "testnet": True
        }

    @pytest.fixture
    def bybit_client(self, mock_api_credentials: Dict[str, str]) -> BybitClient:
        """Фикстура Bybit клиента."""
        return BybitClient(**mock_api_credentials)

    @pytest.fixture
    def mock_market_data(self) -> Dict[str, Any]:
        """Фикстура рыночных данных."""
        return {
            "symbol": "BTCUSDT",
            "price": "45000.50",
            "volume": "1000.25",
            "timestamp": 1640995200000,
            "bid": "44999.75",
            "ask": "45001.25",
            "high24h": "46000.00",
            "low24h": "44000.00"
        }

    @pytest.fixture
    def mock_order_data(self) -> Dict[str, Any]:
        """Фикстура данных ордера."""
        return {
            "orderId": "test_order_123",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "qty": "0.001",
            "price": "45000.00",
            "orderStatus": "New",
            "createdTime": "1640995200000",
            "updatedTime": "1640995200000"
        }

    def test_client_initialization(self, mock_api_credentials: Dict[str, str]) -> None:
        """Тест инициализации клиента."""
        client = BybitClient(**mock_api_credentials)
        
        assert client.api_key == "test_api_key"
        assert client.api_secret == "test_api_secret"
        assert client.testnet is True
        assert client.base_url is not None
        assert hasattr(client, 'session')

    def test_client_initialization_invalid_credentials(self) -> None:
        """Тест инициализации с невалидными credentials."""
        with pytest.raises(ValidationError, match="API key is required"):
            BybitClient(api_key="", api_secret="test")
        
        with pytest.raises(ValidationError, match="API secret is required"):
            BybitClient(api_key="test", api_secret="")

    @pytest.mark.asyncio
    async def test_get_server_time(self, bybit_client: BybitClient) -> None:
        """Тест получения времени сервера."""
        mock_response = {"result": {"timeSecond": "1640995200"}}
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            server_time = await bybit_client.get_server_time()
            
            assert isinstance(server_time, int)
            assert server_time == 1640995200

    @pytest.mark.asyncio
    async def test_get_account_balance(self, bybit_client: BybitClient) -> None:
        """Тест получения баланса аккаунта."""
        mock_response = {
            "result": {
                "list": [{
                    "accountType": "UNIFIED",
                    "coin": [{
                        "coin": "USDT",
                        "walletBalance": "10000.50",
                        "availableBalance": "9500.25"
                    }]
                }]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            balance = await bybit_client.get_account_balance()
            
            assert isinstance(balance, dict)
            assert "USDT" in balance
            assert balance["USDT"]["total"] == Decimal("10000.50")
            assert balance["USDT"]["available"] == Decimal("9500.25")

    @pytest.mark.asyncio
    async def test_get_ticker(self, bybit_client: BybitClient, mock_market_data: Dict[str, Any]) -> None:
        """Тест получения тикера."""
        mock_response = {
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": mock_market_data["price"],
                    "volume24h": mock_market_data["volume"],
                    "bid1Price": mock_market_data["bid"],
                    "ask1Price": mock_market_data["ask"],
                    "highPrice24h": mock_market_data["high24h"],
                    "lowPrice24h": mock_market_data["low24h"]
                }]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            ticker = await bybit_client.get_ticker("BTCUSDT")
            
            assert ticker["symbol"] == "BTCUSDT"
            assert ticker["price"] == Decimal(mock_market_data["price"])
            assert ticker["volume"] == Decimal(mock_market_data["volume"])
            assert ticker["bid"] == Decimal(mock_market_data["bid"])
            assert ticker["ask"] == Decimal(mock_market_data["ask"])

    @pytest.mark.asyncio
    async def test_get_orderbook(self, bybit_client: BybitClient) -> None:
        """Тест получения стакана заявок."""
        mock_response = {
            "result": {
                "s": "BTCUSDT",
                "b": [["44999.50", "1.5"], ["44999.00", "2.0"]],
                "a": [["45000.50", "1.2"], ["45001.00", "1.8"]],
                "ts": 1640995200000
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            orderbook = await bybit_client.get_orderbook("BTCUSDT", limit=50)
            
            assert orderbook["symbol"] == "BTCUSDT"
            assert len(orderbook["bids"]) == 2
            assert len(orderbook["asks"]) == 2
            assert orderbook["bids"][0] == [Decimal("44999.50"), Decimal("1.5")]
            assert orderbook["asks"][0] == [Decimal("45000.50"), Decimal("1.2")]
            assert orderbook["timestamp"] == 1640995200000

    @pytest.mark.asyncio
    async def test_place_limit_order(self, bybit_client: BybitClient, mock_order_data: Dict[str, Any]) -> None:
        """Тест размещения лимитного ордера."""
        mock_response = {
            "result": {
                "orderId": mock_order_data["orderId"],
                "orderLinkId": "custom_order_id"
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            order_result = await bybit_client.place_limit_order(
                symbol="BTCUSDT",
                side="Buy",
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )
            
            assert order_result["order_id"] == mock_order_data["orderId"]
            assert order_result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_place_market_order(self, bybit_client: BybitClient, mock_order_data: Dict[str, Any]) -> None:
        """Тест размещения рыночного ордера."""
        mock_response = {
            "result": {
                "orderId": mock_order_data["orderId"],
                "orderLinkId": "market_order_id"
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            order_result = await bybit_client.place_market_order(
                symbol="BTCUSDT",
                side="Sell",
                quantity=Decimal("0.001")
            )
            
            assert order_result["order_id"] == mock_order_data["orderId"]
            assert order_result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_cancel_order(self, bybit_client: BybitClient) -> None:
        """Тест отмены ордера."""
        mock_response = {
            "result": {
                "orderId": "test_order_123",
                "orderLinkId": "custom_order_id"
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            result = await bybit_client.cancel_order("BTCUSDT", "test_order_123")
            
            assert result["order_id"] == "test_order_123"
            assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_get_order_status(self, bybit_client: BybitClient, mock_order_data: Dict[str, Any]) -> None:
        """Тест получения статуса ордера."""
        mock_response = {
            "result": {
                "list": [{
                    "orderId": mock_order_data["orderId"],
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": "0.001",
                    "price": "45000.00",
                    "orderStatus": "Filled",
                    "avgPrice": "45000.25",
                    "cumExecQty": "0.001",
                    "createdTime": "1640995200000"
                }]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            order = await bybit_client.get_order_status("BTCUSDT", "test_order_123")
            
            assert order["order_id"] == mock_order_data["orderId"]
            assert order["symbol"] == "BTCUSDT"
            assert order["side"] == "Buy"
            assert order["status"] == "Filled"
            assert order["filled_quantity"] == Decimal("0.001")
            assert order["average_price"] == Decimal("45000.25")

    @pytest.mark.asyncio
    async def test_get_open_orders(self, bybit_client: BybitClient) -> None:
        """Тест получения открытых ордеров."""
        mock_response = {
            "result": {
                "list": [
                    {
                        "orderId": "order_1",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderType": "Limit",
                        "qty": "0.001",
                        "price": "45000.00",
                        "orderStatus": "New"
                    },
                    {
                        "orderId": "order_2",
                        "symbol": "ETHUSDT",
                        "side": "Sell",
                        "orderType": "Limit",
                        "qty": "0.1",
                        "price": "3000.00",
                        "orderStatus": "PartiallyFilled"
                    }
                ]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            orders = await bybit_client.get_open_orders()
            
            assert len(orders) == 2
            assert orders[0]["order_id"] == "order_1"
            assert orders[1]["order_id"] == "order_2"
            assert orders[0]["status"] == "New"
            assert orders[1]["status"] == "PartiallyFilled"

    @pytest.mark.asyncio
    async def test_get_trade_history(self, bybit_client: BybitClient) -> None:
        """Тест получения истории сделок."""
        mock_response = {
            "result": {
                "list": [{
                    "execId": "trade_1",
                    "orderId": "order_1",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "execQty": "0.001",
                    "execPrice": "45000.50",
                    "execFee": "0.045",
                    "execTime": "1640995200000"
                }]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            trades = await bybit_client.get_trade_history("BTCUSDT")
            
            assert len(trades) == 1
            assert trades[0]["trade_id"] == "trade_1"
            assert trades[0]["order_id"] == "order_1"
            assert trades[0]["symbol"] == "BTCUSDT"
            assert trades[0]["quantity"] == Decimal("0.001")
            assert trades[0]["price"] == Decimal("45000.50")
            assert trades[0]["fee"] == Decimal("0.045")

    @pytest.mark.asyncio
    async def test_get_positions(self, bybit_client: BybitClient) -> None:
        """Тест получения позиций."""
        mock_response = {
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.001",
                    "avgPrice": "45000.00",
                    "markPrice": "45100.00",
                    "unrealisedPnl": "0.1",
                    "leverage": "10"
                }]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            positions = await bybit_client.get_positions()
            
            assert len(positions) == 1
            assert positions[0]["symbol"] == "BTCUSDT"
            assert positions[0]["side"] == "Buy"
            assert positions[0]["size"] == Decimal("0.001")
            assert positions[0]["entry_price"] == Decimal("45000.00")
            assert positions[0]["mark_price"] == Decimal("45100.00")
            assert positions[0]["unrealized_pnl"] == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_set_leverage(self, bybit_client: BybitClient) -> None:
        """Тест установки кредитного плеча."""
        mock_response = {"result": {"symbol": "BTCUSDT", "leverage": "20"}}
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            result = await bybit_client.set_leverage("BTCUSDT", 20)
            
            assert result["symbol"] == "BTCUSDT"
            assert result["leverage"] == 20

    @pytest.mark.asyncio
    async def test_api_error_handling(self, bybit_client: BybitClient) -> None:
        """Тест обработки ошибок API."""
        mock_error_response = {
            "retCode": 10001,
            "retMsg": "Invalid API key",
            "result": {}
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_error_response):
            with pytest.raises(ExchangeError, match="Invalid API key"):
                await bybit_client.get_account_balance()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, bybit_client: BybitClient) -> None:
        """Тест ограничения скорости запросов."""
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                return {"retCode": 10006, "retMsg": "Rate limit exceeded"}
            return {"result": {"timeSecond": "1640995200"}}
        
        with patch.object(bybit_client, '_make_request', side_effect=mock_request):
            # Первые 3 запроса должны пройти
            for _ in range(3):
                await bybit_client.get_server_time()
            
            # 4-й запрос должен вызвать ошибку
            with pytest.raises(ExchangeError, match="Rate limit exceeded"):
                await bybit_client.get_server_time()

    @pytest.mark.asyncio
    async def test_connection_timeout(self, bybit_client: BybitClient) -> None:
        """Тест тайм-аута соединения."""
        with patch.object(bybit_client, '_make_request', side_effect=asyncio.TimeoutError):
            with pytest.raises(ExchangeError, match="Request timeout"):
                await bybit_client.get_server_time()

    @pytest.mark.asyncio
    async def test_network_error(self, bybit_client: BybitClient) -> None:
        """Тест сетевой ошибки."""
        import aiohttp
        
        with patch.object(bybit_client, '_make_request', side_effect=aiohttp.ClientError):
            with pytest.raises(ExchangeError, match="Network error"):
                await bybit_client.get_server_time()

    def test_signature_generation(self, bybit_client: BybitClient) -> None:
        """Тест генерации подписи."""
        timestamp = "1640995200000"
        params = {"symbol": "BTCUSDT", "side": "Buy"}
        
        signature = bybit_client._generate_signature(timestamp, params)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # HMAC SHA256 hex digest length

    def test_prepare_request_params(self, bybit_client: BybitClient) -> None:
        """Тест подготовки параметров запроса."""
        params = {"symbol": "BTCUSDT", "side": "Buy", "qty": Decimal("0.001")}
        
        prepared_params = bybit_client._prepare_params(params)
        
        assert prepared_params["symbol"] == "BTCUSDT"
        assert prepared_params["side"] == "Buy"
        assert prepared_params["qty"] == "0.001"  # Decimal преобразован в строку
        assert "api_key" in prepared_params
        assert "timestamp" in prepared_params
        assert "sign" in prepared_params

    def test_validate_symbol(self, bybit_client: BybitClient) -> None:
        """Тест валидации символа."""
        # Валидные символы
        assert bybit_client._validate_symbol("BTCUSDT") is True
        assert bybit_client._validate_symbol("ETHUSDT") is True
        
        # Невалидные символы
        with pytest.raises(ValidationError):
            bybit_client._validate_symbol("")
        
        with pytest.raises(ValidationError):
            bybit_client._validate_symbol("INVALID")

    def test_validate_order_params(self, bybit_client: BybitClient) -> None:
        """Тест валидации параметров ордера."""
        # Валидные параметры
        params = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "qty": Decimal("0.001"),
            "price": Decimal("45000.00")
        }
        
        assert bybit_client._validate_order_params(params) is True
        
        # Невалидное количество
        invalid_params = params.copy()
        invalid_params["qty"] = Decimal("0")
        
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            bybit_client._validate_order_params(invalid_params)
        
        # Невалидная цена
        invalid_params = params.copy()
        invalid_params["price"] = Decimal("-100")
        
        with pytest.raises(ValidationError, match="Price must be positive"):
            bybit_client._validate_order_params(invalid_params)

    @pytest.mark.asyncio
    async def test_websocket_connection(self, bybit_client: BybitClient) -> None:
        """Тест WebSocket соединения."""
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_websocket):
            await bybit_client.connect_websocket()
            
            assert bybit_client.websocket is not None
            assert bybit_client.is_websocket_connected() is True

    @pytest.mark.asyncio
    async def test_websocket_subscription(self, bybit_client: BybitClient) -> None:
        """Тест подписки на WebSocket."""
        mock_websocket = AsyncMock()
        bybit_client.websocket = mock_websocket
        
        await bybit_client.subscribe_ticker("BTCUSDT")
        
        # Проверяем, что было отправлено сообщение подписки
        mock_websocket.send.assert_called_once()
        sent_message = mock_websocket.send.call_args[0][0]
        assert "BTCUSDT" in sent_message
        assert "subscribe" in sent_message

    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, bybit_client: BybitClient) -> None:
        """Тест обработки WebSocket сообщений."""
        mock_message = {
            "topic": "tickers.BTCUSDT",
            "data": {
                "symbol": "BTCUSDT",
                "lastPrice": "45000.50",
                "volume24h": "1000.25"
            }
        }
        
        processed_message = bybit_client._process_websocket_message(mock_message)
        
        assert processed_message["topic"] == "tickers.BTCUSDT"
        assert processed_message["symbol"] == "BTCUSDT"
        assert processed_message["price"] == Decimal("45000.50")

    def test_client_configuration(self, bybit_client: BybitClient) -> None:
        """Тест конфигурации клиента."""
        config = bybit_client.get_configuration()
        
        assert "api_key" in config
        assert "testnet" in config
        assert config["testnet"] is True
        assert "rate_limit" in config
        assert "timeout" in config

    def test_client_health_check(self, bybit_client: BybitClient) -> None:
        """Тест проверки здоровья клиента."""
        # Мокаем успешный ответ
        with patch.object(bybit_client, 'get_server_time', return_value=1640995200):
            health = bybit_client.health_check()
            
            assert health["status"] == "healthy"
            assert health["api_connection"] is True

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, bybit_client: BybitClient) -> None:
        """Тест механизма повторных попыток."""
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ExchangeError("Temporary error")
            return {"result": {"timeSecond": "1640995200"}}
        
        with patch.object(bybit_client, '_make_request', side_effect=mock_request):
            result = await bybit_client.get_server_time()
            
            assert call_count == 3  # 2 повтора + 1 успешный вызов
            assert result == 1640995200

    def test_data_transformation(self, bybit_client: BybitClient) -> None:
        """Тест преобразования данных."""
        raw_order_data = {
            "orderId": "123",
            "orderType": "Limit",
            "orderStatus": "Filled",
            "qty": "0.001",
            "price": "45000.00"
        }
        
        transformed = bybit_client._transform_order_data(raw_order_data)
        
        assert transformed["order_id"] == "123"
        assert transformed["type"] == "limit"
        assert transformed["status"] == "filled"
        assert transformed["quantity"] == Decimal("0.001")
        assert transformed["price"] == Decimal("45000.00")

    @pytest.mark.asyncio
    async def test_batch_operations(self, bybit_client: BybitClient) -> None:
        """Тест пакетных операций."""
        orders = [
            {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.001", "price": "45000"},
            {"symbol": "ETHUSDT", "side": "Sell", "qty": "0.1", "price": "3000"}
        ]
        
        mock_response = {
            "result": {
                "list": [
                    {"orderId": "order_1", "orderLinkId": "link_1"},
                    {"orderId": "order_2", "orderLinkId": "link_2"}
                ]
            }
        }
        
        with patch.object(bybit_client, '_make_request', return_value=mock_response):
            results = await bybit_client.place_batch_orders(orders)
            
            assert len(results) == 2
            assert results[0]["order_id"] == "order_1"
            assert results[1]["order_id"] == "order_2"

    @pytest.mark.asyncio
    async def test_performance_metrics(self, bybit_client: BybitClient) -> None:
        """Тест метрик производительности."""
        with patch.object(bybit_client, '_make_request', return_value={"result": {"timeSecond": "1640995200"}}):
            import time
            start_time = time.time()
            
            # Выполняем несколько запросов
            for _ in range(10):
                await bybit_client.get_server_time()
            
            end_time = time.time()
            
            # Проверяем метрики
            metrics = bybit_client.get_performance_metrics()
            assert metrics["total_requests"] == 10
            assert metrics["average_response_time"] > 0
            assert end_time - start_time < 1.0  # Должно быть быстро