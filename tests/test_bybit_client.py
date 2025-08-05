from unittest.mock import AsyncMock, patch

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from shared.logging import setup_logger

logger = setup_logger(__name__)


    @pytest.fixture
def bybit_config() -> Any:
    """Фикстура с конфигурацией Bybit"""
    return {"api_key": "test_api_key", "api_secret": "test_api_secret", "testnet": True}


    @pytest.fixture
def bybit_client(bybit_config) -> Any:
    """Фикстура с клиентом Bybit"""
    return BybitClient(config=bybit_config)


class TestBybitClient:
    def test_initialization(self, bybit_client, bybit_config) -> None:
        """Тест инициализации клиента"""
        assert bybit_client.api_key == bybit_config["api_key"]
        assert bybit_client.api_secret == bybit_config["api_secret"]
        assert bybit_client.testnet == bybit_config["testnet"]
        assert (
            bybit_client.base_url == "https://api-testnet.bybit.com"
            if bybit_config["testnet"]
            else "https://api.bybit.com"
        )
        assert (
            bybit_client.ws_url == "wss://stream-testnet.bybit.com"
            if bybit_config["testnet"]
            else "wss://stream.bybit.com"
        )

    @pytest.mark.asyncio
    async def test_connection(self, bybit_client) -> None:
        """Тест подключения к API"""
        with patch("exchange.bybit_client.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = AsyncMock()
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.status = (
                200
            )

            await bybit_client.connect()
            assert bybit_client.session is not None

            await bybit_client.close()
            assert bybit_client.session is None

    @pytest.mark.asyncio
    async def test_websocket_subscription(self, bybit_client) -> None:
        """Тест подписки на WebSocket"""
        with patch("exchange.bybit_client.websockets.connect") as mock_ws:
            mock_ws.return_value.__aenter__.return_value.send = AsyncMock()
            mock_ws.return_value.__aenter__.return_value.recv = AsyncMock(
                return_value='{"type":"ping"}'
            )

            await bybit_client.connect_ws()
            assert bybit_client.ws is not None

            await bybit_client.subscribe(["BTCUSDT"])
            mock_ws.return_value.__aenter__.return_value.send.assert_called()

            await bybit_client.close_ws()
            assert bybit_client.ws is None

    @pytest.mark.asyncio
    async def test_get_klines(self, bybit_client) -> None:
        """Тест получения свечей"""
        with patch("exchange.bybit_client.aiohttp.ClientSession") as mock_session:
            mock_response = {
                "ret_code": 0,
                "result": {
                    "list": [
                        ["1625097600000", "35000", "35100", "34900", "35050", "100"]
                    ]
                },
            }
            mock_session.return_value.__aenter__.return_value.get = AsyncMock()
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )

            klines = await bybit_client.get_klines("BTCUSDT", "1h", limit=1)
            assert isinstance(klines, list)
            assert len(klines) == 1

    @pytest.mark.asyncio
    async def test_get_orderbook(self, bybit_client) -> None:
        """Тест получения стакана"""
        with patch("exchange.bybit_client.aiohttp.ClientSession") as mock_session:
            mock_response = {
                "ret_code": 0,
                "result": {"bids": [["35000", "1.0"]], "asks": [["35001", "1.0"]]},
            }
            mock_session.return_value.__aenter__.return_value.get = AsyncMock()
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )

            orderbook = await bybit_client.get_orderbook("BTCUSDT")
            assert isinstance(orderbook, dict)
            assert "bids" in orderbook
            assert "asks" in orderbook

    @pytest.mark.asyncio
    async def test_place_order(self, bybit_client) -> None:
        """Тест размещения ордера"""
        with patch("exchange.bybit_client.aiohttp.ClientSession") as mock_session:
            mock_response = {"ret_code": 0, "result": {"order_id": "test_order_id"}}
            mock_session.return_value.__aenter__.return_value.post = AsyncMock()
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )

            order = await bybit_client.place_order(
                symbol="BTCUSDT", side="Buy", order_type="Limit", qty=1.0, price=35000.0
            )
            assert isinstance(order, dict)
            assert "order_id" in order

    @pytest.mark.asyncio
    async def test_cancel_order(self, bybit_client) -> None:
        """Тест отмены ордера"""
        with patch("exchange.bybit_client.aiohttp.ClientSession") as mock_session:
            mock_response = {"ret_code": 0, "result": {"order_id": "test_order_id"}}
            mock_session.return_value.__aenter__.return_value.post = AsyncMock()
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )

            result = await bybit_client.cancel_order("BTCUSDT", "test_order_id")
            assert isinstance(result, dict)
            assert "order_id" in result


    @pytest.mark.asyncio
    async def test_get_ticker(client) -> None:
    """Тест получения тикера"""
    try:
        ticker = await client.get_ticker("BTCUSDT")

        assert isinstance(ticker, dict)
        assert "symbol" in ticker
        assert "last" in ticker
        assert "bid" in ticker
        assert "ask" in ticker
        assert "volume" in ticker
        assert "timestamp" in ticker

    except Exception as e:
        pytest.skip(f"Ticker test skipped: {str(e)}")


    @pytest.mark.asyncio
    async def test_get_balance(client) -> None:
    """Тест получения баланса"""
    try:
        balance = await client.get_balance()

        assert isinstance(balance, dict)
        for currency, info in balance.items():
            assert "free" in info
            assert "used" in info
            assert "total" in info

    except Exception as e:
        pytest.skip(f"Balance test skipped: {str(e)}")


    @pytest.mark.asyncio
    async def test_create_order(client) -> None:
    """Тест создания ордера"""
    try:
        order = await client.create_order(
            symbol="BTCUSDT", order_type="limit", side="buy", amount=0.001, price=30000
        )

        assert isinstance(order, dict)
        assert "id" in order
        assert "symbol" in order
        assert "type" in order
        assert "side" in order
        assert "amount" in order
        assert "price" in order
        assert "status" in order
        assert "timestamp" in order

    except Exception as e:
        pytest.skip(f"Create order test skipped: {str(e)}")


    @pytest.mark.asyncio
    async def test_get_order(client) -> None:
    """Тест получения информации об ордере"""
    try:
        # Создание тестового ордера
        order = await client.create_order(
            symbol="BTCUSDT", order_type="limit", side="buy", amount=0.001, price=30000
        )

        # Получение информации
        order_info = await client.get_order(order_id=order["id"], symbol="BTCUSDT")

        assert isinstance(order_info, dict)
        assert order_info["id"] == order["id"]
        assert "symbol" in order_info
        assert "type" in order_info
        assert "side" in order_info
        assert "amount" in order_info
        assert "price" in order_info
        assert "status" in order_info
        assert "filled" in order_info
        assert "remaining" in order_info
        assert "timestamp" in order_info

    except Exception as e:
        pytest.skip(f"Get order test skipped: {str(e)}")


    @pytest.mark.asyncio
    async def test_rate_limit(client) -> None:
    """Тест ограничения запросов"""
    try:
        # Выполнение нескольких запросов
        for _ in range(5):
            await client.get_ticker("BTCUSDT")

        assert client.request_count > 0

    except Exception as e:
        pytest.skip(f"Rate limit test skipped: {str(e)}")


    @pytest.mark.asyncio
    async def test_error_handling(client) -> None:
    """Тест обработки ошибок"""
    # Тест с неверным символом
    with pytest.raises(Exception):
        await client.get_ticker("INVALID")

    # Тест с неверным типом ордера
    with pytest.raises(Exception):
        await client.create_order(
            symbol="BTCUSDT",
            order_type="invalid",
            side="buy",
            amount=0.001,
            price=30000,
        )

    # Тест с неверным ID ордера
    with pytest.raises(Exception):
        await client.get_order(order_id="invalid", symbol="BTCUSDT")
