"""
Интеграционные тесты для exchanges
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.type_definitions.external_service_types import (
    ExchangeType, ExchangeCredentials, ConnectionConfig, 
    MarketDataRequest, OrderRequest, OrderType, OrderSide, TimeFrame
)
from domain.type_definitions import Symbol
from infrastructure.external_services.exchanges.factory import ExchangeServiceFactory
from infrastructure.external_services.exchanges.bybit_exchange_service import BybitExchangeService
from infrastructure.external_services.exchanges.binance_exchange_service import BinanceExchangeService
class TestExchangesIntegration:
    """Интеграционные тесты для exchanges."""
    @pytest.fixture
    def sample_credentials(self) -> Any:
        """Пример учетных данных."""
        return ExchangeCredentials(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_passphrase",
            testnet=True,
            sandbox=True
        )
    @pytest.fixture
    def sample_connection_config(self) -> Any:
        """Пример конфигурации соединения."""
        return ConnectionConfig(
            rate_limit=100,
            rate_limit_window=60,
            timeout=30.0,
            retry_attempts=3
        )
    @pytest.fixture
    def sample_market_data_request(self) -> Any:
        """Пример запроса рыночных данных."""
        return MarketDataRequest(
            symbol=Symbol("BTC/USDT"),
            timeframe=TimeFrame.HOUR_1,
            limit=100
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
    @pytest.mark.asyncio
    async def test_factory_creates_bybit_service(self, sample_credentials, sample_connection_config) -> None:
        """Тест создания сервиса Bybit через фабрику."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            assert isinstance(service, BybitExchangeService)
            assert service.exchange_name == ExchangeType.BYBIT.value
    @pytest.mark.asyncio
    async def test_factory_creates_binance_service(self, sample_credentials, sample_connection_config) -> None:
        """Тест создания сервиса Binance через фабрику."""
        with patch('infrastructure.external_services.exchanges.binance_exchange_service.BinanceExchangeService'):
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BINANCE,
                sample_credentials,
                sample_connection_config
            )
            assert isinstance(service, BinanceExchangeService)
            assert service.exchange_name == ExchangeType.BINANCE.value
    @pytest.mark.asyncio
    async def test_factory_specialized_methods(self, sample_connection_config) -> None:
        """Тест специализированных методов фабрики."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            bybit_service = ExchangeServiceFactory.create_bybit_service(
                api_key="bybit_key",
                api_secret="bybit_secret",
                testnet=True,
                connection_config=sample_connection_config
            )
            assert isinstance(bybit_service, BybitExchangeService)
            assert bybit_service.exchange_name == "bybit"
        with patch('infrastructure.external_services.exchanges.binance_exchange_service.BinanceExchangeService'):
            binance_service = ExchangeServiceFactory.create_binance_service(
                api_key="binance_key",
                api_secret="binance_secret",
                testnet=True,
                connection_config=sample_connection_config
            )
            assert isinstance(binance_service, BinanceExchangeService)
            assert binance_service.exchange_name == "binance"
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, sample_credentials, sample_connection_config) -> None:
        """Тест жизненного цикла сервиса."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Подключение
            await service.connect(sample_credentials)
            mock_service.connect.assert_called_once_with(sample_credentials)
            # Отключение
            await service.disconnect()
            mock_service.disconnect.assert_called_once()
    @pytest.mark.asyncio
    async def test_market_data_flow(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест потока рыночных данных."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем рыночные данные
            mock_market_data = [
                {
                    "timestamp": 1640995200000,
                    "open": 50000.0,
                    "high": 50100.0,
                    "low": 49900.0,
                    "close": 50050.0,
                    "volume": 1000.0
                }
            ]
            mock_service.get_market_data.return_value = mock_market_data
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Получаем рыночные данные
            result = await service.get_market_data(sample_market_data_request)
            assert result == mock_market_data
            mock_service.get_market_data.assert_called_once_with(sample_market_data_request)
    @pytest.mark.asyncio
    async def test_order_flow(self, sample_credentials, sample_connection_config, sample_order_request) -> None:
        """Тест потока ордеров."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем размещение ордера
            mock_order = {
                "id": "12345",
                "symbol": "BTC/USDT",
                "type": "limit",
                "side": "buy",
                "amount": 0.1,
                "price": 50000.0,
                "status": "open"
            }
            mock_service.place_order.return_value = mock_order
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Размещаем ордер
            result = await service.place_order(sample_order_request)
            assert result == mock_order
            mock_service.place_order.assert_called_once_with(sample_order_request)
    @pytest.mark.asyncio
    async def test_balance_and_positions_flow(self, sample_credentials, sample_connection_config) -> None:
        """Тест потока баланса и позиций."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем баланс
            mock_balance = {
                "BTC": {"total": 1.0, "free": 0.5, "used": 0.5},
                "USDT": {"total": 50000.0, "free": 25000.0, "used": 25000.0}
            }
            mock_service.get_balance.return_value = mock_balance
            # Мокаем позиции
            mock_positions = [
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.1,
                    "notional": 5000.0,
                    "unrealized_pnl": 100.0
                }
            ]
            mock_service.get_positions.return_value = mock_positions
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Получаем баланс
            balance = await service.get_balance()
            assert balance == mock_balance
            # Получаем позиции
            positions = await service.get_positions()
            assert positions == mock_positions
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест обработки ошибок."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем ошибку
            mock_service.get_market_data.side_effect = Exception("API Error")
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Проверяем, что ошибка пробрасывается
            with pytest.raises(Exception, match="API Error"):
                await service.get_market_data(sample_market_data_request)
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест интеграции rate limiting."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем рыночные данные
            mock_market_data = [{"timestamp": 1640995200000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1000.0}]
            mock_service.get_market_data.return_value = mock_market_data
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Выполняем несколько запросов для проверки rate limiting
            tasks = [
                service.get_market_data(sample_market_data_request)
                for _ in range(3)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            assert all(result == mock_market_data for result in results)
            assert mock_service.get_market_data.call_count == 3
    @pytest.mark.asyncio
    async def test_caching_integration(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест интеграции кэширования."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем рыночные данные
            mock_market_data = [{"timestamp": 1640995200000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1000.0}]
            mock_service.get_market_data.return_value = mock_market_data
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Первый запрос - должен вызвать API
            result1 = await service.get_market_data(sample_market_data_request)
            assert result1 == mock_market_data
            # Второй запрос - должен использовать кэш
            result2 = await service.get_market_data(sample_market_data_request)
            assert result2 == mock_market_data
            # Проверяем, что API был вызван только один раз
            assert mock_service.get_market_data.call_count == 1
    @pytest.mark.asyncio
    async def test_websocket_integration(self, sample_credentials, sample_connection_config) -> None:
        """Тест интеграции WebSocket."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Проверяем, что WebSocket методы доступны
            assert hasattr(service, '_get_websocket_url')
            assert hasattr(service, '_subscribe_websocket_channels')
            assert hasattr(service, '_process_websocket_message')
            # Проверяем URL для testnet
            url = service._get_websocket_url()
            assert "testnet" in url
            assert "bybit" in url
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест конкурентных операций."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем различные операции
            mock_service.get_market_data.return_value = [{"data": "market"}]
            mock_service.get_balance.return_value = {"BTC": {"total": 1.0}}
            mock_service.get_positions.return_value = [{"symbol": "BTC/USDT"}]
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Выполняем конкурентные операции
            tasks = [
                service.get_market_data(sample_market_data_request),
                service.get_balance(),
                service.get_positions()
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            assert results[0] == [{"data": "market"}]
            assert results[1] == {"BTC": {"total": 1.0}}
            assert results[2] == [{"symbol": "BTC/USDT"}]
    @pytest.mark.asyncio
    async def test_service_metrics_integration(self, sample_credentials, sample_connection_config, sample_market_data_request) -> None:
        """Тест интеграции метрик сервиса."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            # Мокаем рыночные данные
            mock_market_data = [{"timestamp": 1640995200000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1000.0}]
            mock_service.get_market_data.return_value = mock_market_data
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Выполняем операции
            await service.get_market_data(sample_market_data_request)
            await service.get_market_data(sample_market_data_request)
            # Проверяем, что метрики обновляются
            assert hasattr(service, 'metrics')
            assert 'total_requests' in service.metrics
            assert 'successful_requests' in service.metrics
            assert 'failed_requests' in service.metrics
    @pytest.mark.asyncio
    async def test_config_integration(self, sample_credentials, sample_connection_config) -> None:
        """Тест интеграции конфигурации."""
        # Создаем сервис с кастомной конфигурацией
        custom_config = ExchangeServiceConfig(
            exchange_name=ExchangeType.BYBIT.value,
            credentials=sample_credentials,
            connection_config=sample_connection_config,
            enable_rate_limiting=False,
            enable_caching=False,
            timeout=60.0,
            max_retries=5
        )
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService') as mock_bybit_class:
            mock_service = AsyncMock()
            mock_bybit_class.return_value = mock_service
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            # Проверяем, что конфигурация применяется
            assert service.config.exchange_name == ExchangeType.BYBIT.value
            assert service.config.credentials == sample_credentials
            assert service.config.connection_config == sample_connection_config 
