"""
Unit тесты для ExchangeServiceFactory
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import patch
from domain.types.external_service_types import (
    ExchangeType, ExchangeCredentials, ConnectionConfig
)
from infrastructure.external_services.exchanges.factory import ExchangeServiceFactory
from infrastructure.external_services.exchanges.bybit_exchange_service import BybitExchangeService
from infrastructure.external_services.exchanges.binance_exchange_service import BinanceExchangeService
class TestExchangeServiceFactory:
    """Тесты для ExchangeServiceFactory."""
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
    def test_create_exchange_service_bybit(self, sample_credentials, sample_connection_config) -> None:
        """Тест создания сервиса Bybit."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials,
                sample_connection_config
            )
            assert isinstance(service, BybitExchangeService)
    def test_create_exchange_service_binance(self, sample_credentials, sample_connection_config) -> None:
        """Тест создания сервиса Binance."""
        with patch('infrastructure.external_services.exchanges.binance_exchange_service.BinanceExchangeService'):
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BINANCE,
                sample_credentials,
                sample_connection_config
            )
            assert isinstance(service, BinanceExchangeService)
    def test_create_exchange_service_unsupported_type(self, sample_credentials) -> None:
        """Тест создания сервиса с неподдерживаемым типом."""
        with pytest.raises(ValueError, match="Unsupported exchange type"):
            ExchangeServiceFactory.create_exchange_service(
                ExchangeType.UNKNOWN,  # Предполагаем, что такой тип существует
                sample_credentials
            )
    def test_create_exchange_service_default_connection_config(self, sample_credentials) -> None:
        """Тест создания сервиса с конфигурацией по умолчанию."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                sample_credentials
            )
            assert isinstance(service, BybitExchangeService)
    def test_create_bybit_service(self) -> None:
        """Тест создания сервиса Bybit через специализированный метод."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service = ExchangeServiceFactory.create_bybit_service(
                api_key="bybit_key",
                api_secret="bybit_secret",
                testnet=True
            )
            assert isinstance(service, BybitExchangeService)
    def test_create_bybit_service_with_connection_config(self, sample_connection_config) -> None:
        """Тест создания сервиса Bybit с конфигурацией соединения."""
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service = ExchangeServiceFactory.create_bybit_service(
                api_key="bybit_key",
                api_secret="bybit_secret",
                testnet=False,
                connection_config=sample_connection_config
            )
            assert isinstance(service, BybitExchangeService)
    def test_create_binance_service(self) -> None:
        """Тест создания сервиса Binance через специализированный метод."""
        with patch('infrastructure.external_services.exchanges.binance_exchange_service.BinanceExchangeService'):
            service = ExchangeServiceFactory.create_binance_service(
                api_key="binance_key",
                api_secret="binance_secret",
                testnet=True
            )
            assert isinstance(service, BinanceExchangeService)
    def test_create_binance_service_with_connection_config(self, sample_connection_config) -> None:
        """Тест создания сервиса Binance с конфигурацией соединения."""
        with patch('infrastructure.external_services.exchanges.binance_exchange_service.BinanceExchangeService'):
            service = ExchangeServiceFactory.create_binance_service(
                api_key="binance_key",
                api_secret="binance_secret",
                testnet=False,
                connection_config=sample_connection_config
            )
            assert isinstance(service, BinanceExchangeService)
    def test_factory_methods_consistency(self) -> None:
        """Тест консистентности методов фабрики."""
        # Проверяем, что общий метод и специализированные методы создают одинаковые сервисы
        credentials = ExchangeCredentials(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        with patch('infrastructure.external_services.exchanges.bybit_exchange_service.BybitExchangeService'):
            service1 = ExchangeServiceFactory.create_exchange_service(
                ExchangeType.BYBIT,
                credentials
            )
            service2 = ExchangeServiceFactory.create_bybit_service(
                api_key="test_key",
                api_secret="test_secret",
                testnet=True
            )
            assert type(service1) == type(service2)
    def test_factory_static_methods(self) -> None:
        """Тест, что все методы фабрики являются статическими."""
        # Проверяем, что можно вызывать методы без создания экземпляра
        assert hasattr(ExchangeServiceFactory, 'create_exchange_service')
        assert hasattr(ExchangeServiceFactory, 'create_bybit_service')
        assert hasattr(ExchangeServiceFactory, 'create_binance_service')
        # Проверяем, что методы статические
        import inspect
        assert inspect.isfunction(ExchangeServiceFactory.create_exchange_service)
        assert inspect.isfunction(ExchangeServiceFactory.create_bybit_service)
        assert inspect.isfunction(ExchangeServiceFactory.create_binance_service) 
