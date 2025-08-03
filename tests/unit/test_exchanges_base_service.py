"""
Unit тесты для BaseExchangeService
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.exceptions import (
    ExchangeError, ConnectionError, AuthenticationError, 
    InvalidOrderError, NetworkError
)
from domain.types.external_service_types import (
    ConnectionStatus, TimeFrame, ExchangeName, ExchangeCredentials, 
    MarketDataRequest, OrderRequest, OrderType, OrderSide
)
from domain.types import OrderId, Symbol
from infrastructure.external_services.exchanges.config import ExchangeServiceConfig
from infrastructure.external_services.exchanges.base_exchange_service import BaseExchangeService
class TestBaseExchangeService:
    """Тесты для BaseExchangeService."""
    @pytest.fixture
    def config(self) -> Any:
        """Конфигурация для тестов."""
        return ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=ExchangeCredentials(
                api_key="test_key",
                api_secret="test_secret",
                api_passphrase="test_passphrase",
                testnet=True,
                sandbox=True
            ),
            connection_config=MagicMock(
                rate_limit=100,
                rate_limit_window=60
            ),
            max_cache_size=1000,
            cache_ttl=300,
            timeout=30.0,
            enable_rate_limiting=True
        )
    @pytest.fixture
    def service(self, config) -> Any:
        """Экземпляр BaseExchangeService."""
        with patch('ccxt.binance'):
            return BaseExchangeService(config)
    @pytest.fixture
    def mock_ccxt_client(self) -> Any:
        """Мок CCXT клиента."""
        client = AsyncMock()
        client.load_markets = AsyncMock()
        client.fetch_ohlcv = AsyncMock()
        client.create_order = AsyncMock()
        client.cancel_order = AsyncMock()
        client.fetch_order = AsyncMock()
        client.fetch_balance = AsyncMock()
        client.fetch_positions = AsyncMock()
        client.close = AsyncMock()
        return client
    @pytest.fixture
    def sample_market_data_request(self) -> Any:
        """Пример запроса рыночных данных."""
        return MarketDataRequest(
            symbol=Symbol("BTC/USDT"),
            timeframe=TimeFrame.HOUR_1,
            limit=100,
            since=datetime.now() - timedelta(days=1)
        )
    @pytest.fixture
    def sample_order_request(self) -> Any:
        """Пример запроса на размещение ордера."""
        return OrderRequest(
            symbol=Symbol("BTC/USDT"),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            time_in_force="GTC",
            post_only=False,
            reduce_only=False
        )
    def test_init(self, config) -> None:
        """Тест инициализации."""
        with patch('ccxt.binance'):
            service = BaseExchangeService(config)
            assert service.exchange_name == ExchangeName.BINANCE
            assert service.connection_status == ConnectionStatus.DISCONNECTED
            assert service.is_running is False
            assert service.is_websocket_connected is False
            assert service.ccxt_client is not None
            assert service.cache is not None
            assert service.rate_limiter is not None
    def test_init_ccxt_client(self, config) -> None:
        """Тест инициализации CCXT клиента."""
        with patch('ccxt.binance') as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange
            service = BaseExchangeService(config)
            mock_binance.assert_called_once()
            assert service.ccxt_client == mock_exchange
    @pytest.mark.asyncio
    async def test_connect_success(self, service, mock_ccxt_client) -> None:
        """Тест успешного подключения."""
        service.ccxt_client = mock_ccxt_client
        credentials = ExchangeCredentials(
            api_key="new_key",
            api_secret="new_secret",
            api_passphrase="new_passphrase",
            testnet=True,
            sandbox=True
        )
        result = await service.connect(credentials)
        assert result is True
        assert service.connection_status == ConnectionStatus.CONNECTED
        assert service.is_running is True
        assert service.metrics["uptime"] > 0
        mock_ccxt_client.load_markets.assert_called_once()
    @pytest.mark.asyncio
    async def test_connect_failure(self, service, mock_ccxt_client) -> None:
        """Тест неудачного подключения."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.load_markets.side_effect = Exception("Connection failed")
        credentials = ExchangeCredentials(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_passphrase",
            testnet=True,
            sandbox=True
        )
        with pytest.raises(ConnectionError):
            await service.connect(credentials)
        assert service.connection_status == ConnectionStatus.ERROR
        assert service.metrics["total_errors"] == 1
        assert service.metrics["last_error"] == "Connection failed"
    @pytest.mark.asyncio
    async def test_disconnect_success(self, service, mock_ccxt_client) -> None:
        """Тест успешного отключения."""
        service.ccxt_client = mock_ccxt_client
        service.is_running = True
        service.connection_status = ConnectionStatus.CONNECTED
        await service.disconnect()
        assert service.connection_status == ConnectionStatus.DISCONNECTED
        assert service.is_running is False
        assert service.is_websocket_connected is False
        mock_ccxt_client.close.assert_called_once()
    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, service, mock_ccxt_client) -> None:
        """Тест отключения с ошибкой."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.close.side_effect = Exception("Close failed")
        with pytest.raises(Exception):
            await service.disconnect()
        assert service.metrics["total_errors"] == 1
        assert service.metrics["last_error"] == "Close failed"
    def test_get_websocket_url_not_implemented(self, service) -> None:
        """Тест нереализованного метода _get_websocket_url."""
        with pytest.raises(NotImplementedError):
            service._get_websocket_url()
    @pytest.mark.asyncio
    async def test_subscribe_websocket_channels_not_implemented(self, service) -> None:
        """Тест нереализованного метода _subscribe_websocket_channels."""
        with pytest.raises(NotImplementedError):
            await service._subscribe_websocket_channels()
    @pytest.mark.asyncio
    async def test_process_websocket_message(self, service) -> None:
        """Тест обработки WebSocket сообщения."""
        data = {"test": "data"}
        # Метод должен выполняться без ошибок
        await service._process_websocket_message(data)
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест успешного получения рыночных данных."""
        service.ccxt_client = mock_ccxt_client
        # Мокаем данные OHLCV
        mock_ohlcv = [
            [1640995200000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0],
            [1640998800000, 50050.0, 50200.0, 50000.0, 50150.0, 1200.0],
        ]
        mock_ccxt_client.fetch_ohlcv.return_value = mock_ohlcv
        result = await service.get_market_data(sample_market_data_request)
        assert len(result) == 2
        assert result[0]["timestamp"] == 1640995200000
        assert result[0]["open"] == 50000.0
        assert result[0]["high"] == 50100.0
        assert result[0]["low"] == 49900.0
        assert result[0]["close"] == 50050.0
        assert result[0]["volume"] == 1000.0
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.fetch_ohlcv.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_market_data_from_cache(self, service, sample_market_data_request) -> None:
        """Тест получения рыночных данных из кэша."""
        cached_data = [
            {"timestamp": 1640995200000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1000.0}
        ]
        # Мокаем кэш
        service.cache.get = AsyncMock(return_value=cached_data)
        result = await service.get_market_data(sample_market_data_request)
        assert result == cached_data
        service.cache.get.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_market_data_error(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест ошибки получения рыночных данных."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.fetch_ohlcv.side_effect = Exception("API error")
        with pytest.raises(ExchangeError):
            await service.get_market_data(sample_market_data_request)
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "API error"
    def test_convert_timeframe(self, service) -> None:
        """Тест конвертации временных интервалов."""
        assert service._convert_timeframe(TimeFrame.MINUTE_1) == "1m"
        assert service._convert_timeframe(TimeFrame.MINUTE_5) == "5m"
        assert service._convert_timeframe(TimeFrame.MINUTE_15) == "15m"
        assert service._convert_timeframe(TimeFrame.MINUTE_30) == "30m"
        assert service._convert_timeframe(TimeFrame.HOUR_1) == "1h"
        assert service._convert_timeframe(TimeFrame.HOUR_4) == "4h"
        assert service._convert_timeframe(TimeFrame.DAY_1) == "1d"
        assert service._convert_timeframe(TimeFrame.MINUTE_1) == "1m"  # default
    @pytest.mark.asyncio
    async def test_place_order_success(self, service, mock_ccxt_client, sample_order_request) -> None:
        """Тест успешного размещения ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_order = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "limit",
            "side": "buy",
            "amount": 0.1,
            "price": 50000.0,
            "status": "open",
            "timestamp": 1640995200000,
            "filled": 0.0,
            "remaining": 0.1,
            "cost": 0.0,
        }
        mock_ccxt_client.create_order.return_value = mock_order
        result = await service.place_order(sample_order_request)
        assert result["id"] == "12345"
        assert result["symbol"] == "BTC/USDT"
        assert result["type"] == "limit"
        assert result["side"] == "buy"
        assert result["amount"] == 0.1
        assert result["price"] == 50000.0
        assert result["status"] == "open"
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.create_order.assert_called_once()
    @pytest.mark.asyncio
    async def test_place_order_validation_error(self, service, sample_order_request) -> None:
        """Тест ошибки валидации ордера."""
        # Устанавливаем невалидные данные
        sample_order_request.quantity = 0
        with pytest.raises(InvalidOrderError):
            await service.place_order(sample_order_request)
        assert service.metrics["failed_requests"] == 1
    @pytest.mark.asyncio
    async def test_place_order_missing_symbol(self, service, sample_order_request) -> None:
        """Тест размещения ордера без символа."""
        sample_order_request.symbol = None
        with pytest.raises(InvalidOrderError):
            await service.place_order(sample_order_request)
    @pytest.mark.asyncio
    async def test_place_order_limit_without_price(self, service, sample_order_request) -> None:
        """Тест лимитного ордера без цены."""
        sample_order_request.price = None
        with pytest.raises(InvalidOrderError):
            await service.place_order(sample_order_request)
    @pytest.mark.asyncio
    async def test_place_order_error(self, service, mock_ccxt_client, sample_order_request) -> None:
        """Тест ошибки размещения ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.create_order.side_effect = Exception("Order failed")
        with pytest.raises(ExchangeError):
            await service.place_order(sample_order_request)
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "Order failed"
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, service, mock_ccxt_client) -> None:
        """Тест успешной отмены ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_result = {"status": "canceled"}
        mock_ccxt_client.cancel_order.return_value = mock_result
        result = await service.cancel_order(OrderId("12345"))
        assert result is True
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.cancel_order.assert_called_once_with("12345")
    @pytest.mark.asyncio
    async def test_cancel_order_not_canceled(self, service, mock_ccxt_client) -> None:
        """Тест отмены ордера, который не был отменен."""
        service.ccxt_client = mock_ccxt_client
        mock_result = {"status": "open"}
        mock_ccxt_client.cancel_order.return_value = mock_result
        result = await service.cancel_order(OrderId("12345"))
        assert result is False
    @pytest.mark.asyncio
    async def test_cancel_order_error(self, service, mock_ccxt_client) -> None:
        """Тест ошибки отмены ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.cancel_order.side_effect = Exception("Cancel failed")
        with pytest.raises(ExchangeError):
            await service.cancel_order(OrderId("12345"))
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "Cancel failed"
    @pytest.mark.asyncio
    async def test_get_order_status_success(self, service, mock_ccxt_client) -> None:
        """Тест успешного получения статуса ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_order = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "limit",
            "side": "buy",
            "amount": 0.1,
            "price": 50000.0,
            "status": "filled",
            "timestamp": 1640995200000,
            "filled": 0.1,
            "remaining": 0.0,
            "cost": 5000.0,
        }
        mock_ccxt_client.fetch_order.return_value = mock_order
        result = await service.get_order_status(OrderId("12345"))
        assert result["id"] == "12345"
        assert result["status"] == "filled"
        assert result["filled"] == 0.1
        assert result["remaining"] == 0.0
        assert result["cost"] == 5000.0
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.fetch_order.assert_called_once_with("12345")
    @pytest.mark.asyncio
    async def test_get_order_status_error(self, service, mock_ccxt_client) -> None:
        """Тест ошибки получения статуса ордера."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.fetch_order.side_effect = Exception("Fetch failed")
        with pytest.raises(ExchangeError):
            await service.get_order_status(OrderId("12345"))
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "Fetch failed"
    @pytest.mark.asyncio
    async def test_get_balance_success(self, service, mock_ccxt_client) -> None:
        """Тест успешного получения баланса."""
        service.ccxt_client = mock_ccxt_client
        mock_balance = {
            "total": {"BTC": 1.0, "USDT": 50000.0, "ETH": 0.0},
            "free": {"BTC": 0.5, "USDT": 25000.0, "ETH": 0.0},
            "used": {"BTC": 0.5, "USDT": 25000.0, "ETH": 0.0},
        }
        mock_ccxt_client.fetch_balance.return_value = mock_balance
        result = await service.get_balance()
        assert "BTC" in result
        assert "USDT" in result
        assert "ETH" not in result  # Нулевой баланс не включается
        assert result["BTC"]["total"] == 1.0
        assert result["BTC"]["free"] == 0.5
        assert result["BTC"]["used"] == 0.5
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.fetch_balance.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_balance_from_cache(self, service) -> None:
        """Тест получения баланса из кэша."""
        cached_balance = {"BTC": {"total": 1.0, "free": 0.5, "used": 0.5}}
        # Мокаем кэш
        service.cache.get = AsyncMock(return_value=cached_balance)
        result = await service.get_balance()
        assert result == cached_balance
        service.cache.get.assert_called_once_with("balance")
    @pytest.mark.asyncio
    async def test_get_balance_error(self, service, mock_ccxt_client) -> None:
        """Тест ошибки получения баланса."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.fetch_balance.side_effect = Exception("Balance failed")
        with pytest.raises(ExchangeError):
            await service.get_balance()
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "Balance failed"
    @pytest.mark.asyncio
    async def test_get_positions_success(self, service, mock_ccxt_client) -> None:
        """Тест успешного получения позиций."""
        service.ccxt_client = mock_ccxt_client
        mock_positions = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": 0.1,
                "notional": 5000.0,
                "leverage": 10.0,
                "unrealizedPnl": 100.0,
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "liquidationPrice": 45000.0,
            },
            {
                "symbol": "ETH/USDT",
                "side": "short",
                "contracts": 0.0,  # Нулевая позиция
                "notional": 0.0,
                "leverage": 5.0,
                "unrealizedPnl": 0.0,
                "entryPrice": 3000.0,
                "markPrice": 3000.0,
                "liquidationPrice": 3500.0,
            }
        ]
        mock_ccxt_client.fetch_positions.return_value = mock_positions
        result = await service.get_positions()
        assert len(result) == 1  # Только ненулевые позиции
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["side"] == "long"
        assert result[0]["contracts"] == 0.1
        assert result[0]["notional"] == 5000.0
        assert result[0]["leverage"] == 10.0
        assert result[0]["unrealized_pnl"] == 100.0
        assert result[0]["entry_price"] == 50000.0
        assert result[0]["mark_price"] == 51000.0
        assert result[0]["liquidation_price"] == 45000.0
        assert service.metrics["successful_requests"] == 1
        mock_ccxt_client.fetch_positions.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_positions_from_cache(self, service) -> None:
        """Тест получения позиций из кэша."""
        cached_positions = [
            {"symbol": "BTC/USDT", "side": "long", "contracts": 0.1}
        ]
        # Мокаем кэш
        service.cache.get = AsyncMock(return_value=cached_positions)
        result = await service.get_positions()
        assert result == cached_positions
        service.cache.get.assert_called_once_with("positions")
    @pytest.mark.asyncio
    async def test_get_positions_error(self, service, mock_ccxt_client) -> None:
        """Тест ошибки получения позиций."""
        service.ccxt_client = mock_ccxt_client
        mock_ccxt_client.fetch_positions.side_effect = Exception("Positions failed")
        with pytest.raises(ExchangeError):
            await service.get_positions()
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["last_error"] == "Positions failed"
    @pytest.mark.asyncio
    async def test_rate_limiting_enabled(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест работы rate limiting."""
        service.ccxt_client = mock_ccxt_client
        service.config.enable_rate_limiting = True
        mock_ohlcv = [[1640995200000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0]]
        mock_ccxt_client.fetch_ohlcv.return_value = mock_ohlcv
        # Мокаем rate limiter
        service.rate_limiter.acquire = AsyncMock()
        await service.get_market_data(sample_market_data_request)
        service.rate_limiter.acquire.assert_called_once()
    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест отключенного rate limiting."""
        service.ccxt_client = mock_ccxt_client
        service.config.enable_rate_limiting = False
        mock_ohlcv = [[1640995200000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0]]
        mock_ccxt_client.fetch_ohlcv.return_value = mock_ohlcv
        # Мокаем rate limiter
        service.rate_limiter.acquire = AsyncMock()
        await service.get_market_data(sample_market_data_request)
        service.rate_limiter.acquire.assert_not_called()
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест отслеживания метрик."""
        service.ccxt_client = mock_ccxt_client
        # Успешный запрос
        mock_ohlcv = [[1640995200000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0]]
        mock_ccxt_client.fetch_ohlcv.return_value = mock_ohlcv
        await service.get_market_data(sample_market_data_request)
        assert service.metrics["total_requests"] == 1
        assert service.metrics["successful_requests"] == 1
        assert service.metrics["failed_requests"] == 0
        # Неудачный запрос
        mock_ccxt_client.fetch_ohlcv.side_effect = Exception("Error")
        with pytest.raises(ExchangeError):
            await service.get_market_data(sample_market_data_request)
        assert service.metrics["total_requests"] == 2
        assert service.metrics["successful_requests"] == 1
        assert service.metrics["failed_requests"] == 1
        assert service.metrics["total_errors"] == 1
        assert service.metrics["last_error"] == "Error"
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, service, mock_ccxt_client, sample_market_data_request) -> None:
        """Тест конкурентных запросов."""
        service.ccxt_client = mock_ccxt_client
        mock_ohlcv = [[1640995200000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0]]
        mock_ccxt_client.fetch_ohlcv.return_value = mock_ohlcv
        # Выполняем несколько запросов одновременно
        tasks = [
            service.get_market_data(sample_market_data_request)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(len(result) == 1 for result in results)
        assert service.metrics["successful_requests"] == 5 
