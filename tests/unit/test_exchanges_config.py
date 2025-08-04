"""
Unit тесты для ExchangeServiceConfig
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from domain.type_definitions.external_service_types import (
    ExchangeName, ExchangeCredentials, ConnectionConfig
)
from infrastructure.external_services.exchanges.config import ExchangeServiceConfig


class TestExchangeServiceConfig:
    """Тесты для ExchangeServiceConfig."""
    
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
    
    def test_init_with_required_fields(self, sample_credentials, sample_connection_config) -> None:
        """Тест инициализации с обязательными полями."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert config.exchange_name == ExchangeName.BINANCE
        assert config.credentials == sample_credentials
        assert config.connection_config == sample_connection_config
    
    def test_init_with_all_fields(self, sample_credentials, sample_connection_config) -> None:
        """Тест инициализации со всеми полями."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BYBIT,
            credentials=sample_credentials,
            connection_config=sample_connection_config,
            enable_websocket=False,
            enable_rest=False,
            enable_rate_limiting=False,
            enable_caching=False,
            cache_ttl=120,
            max_cache_size=500,
            retry_on_failure=False,
            max_retries=5,
            retry_delay=2.0,
            timeout=60.0,
            ping_interval=60,
            reconnect_interval=10,
            max_reconnects=10
        )
        
        assert config.exchange_name == ExchangeName.BYBIT
        assert config.credentials == sample_credentials
        assert config.connection_config == sample_connection_config
        assert config.enable_websocket is False
        assert config.enable_rest is False
        assert config.enable_rate_limiting is False
        assert config.enable_caching is False
        assert config.cache_ttl == 120
        assert config.max_cache_size == 500
        assert config.retry_on_failure is False
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.timeout == 60.0
        assert config.ping_interval == 60
        assert config.reconnect_interval == 10
        assert config.max_reconnects == 10
    
    def test_default_values(self, sample_credentials, sample_connection_config) -> None:
        """Тест значений по умолчанию."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert config.enable_websocket is True
        assert config.enable_rest is True
        assert config.enable_rate_limiting is True
        assert config.enable_caching is True
        assert config.cache_ttl == 60
        assert config.max_cache_size == 1000
        assert config.retry_on_failure is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 30.0
        assert config.ping_interval == 30
        assert config.reconnect_interval == 5
        assert config.max_reconnects == 5
    
    def test_different_exchange_names(self, sample_credentials, sample_connection_config) -> None:
        """Тест разных имен бирж."""
        binance_config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        bybit_config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BYBIT,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert binance_config.exchange_name == ExchangeName.BINANCE
        assert bybit_config.exchange_name == ExchangeName.BYBIT
        assert binance_config.exchange_name != bybit_config.exchange_name
    
    def test_config_immutability(self, sample_credentials, sample_connection_config) -> None:
        """Тест неизменяемости конфигурации."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        # Проверяем, что поля можно изменять (dataclass по умолчанию изменяемый)
        config.enable_websocket = False
        config.cache_ttl = 120
        
        assert config.enable_websocket is False
        assert config.cache_ttl == 120
    
    def test_config_equality(self, sample_credentials, sample_connection_config) -> None:
        """Тест равенства конфигураций."""
        config1 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config2 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert config1 == config2
    
    def test_config_inequality(self, sample_credentials, sample_connection_config) -> None:
        """Тест неравенства конфигураций."""
        config1 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config2 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BYBIT,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert config1 != config2
    
    def test_config_repr(self, sample_credentials, sample_connection_config) -> None:
        """Тест строкового представления конфигурации."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        repr_str = repr(config)
        
        assert "ExchangeServiceConfig" in repr_str
        assert "exchange_name=" in repr_str
        assert "credentials=" in repr_str
        assert "connection_config=" in repr_str
    
    def test_config_hash(self, sample_credentials, sample_connection_config) -> None:
        """Тест хеширования конфигурации."""
        config1 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config2 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        # Проверяем, что одинаковые конфигурации имеют одинаковый хеш
        assert hash(config1) == hash(config2)
    
    def test_config_in_dict(self, sample_credentials, sample_connection_config) -> None:
        """Тест использования конфигурации в словаре."""
        config1 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config2 = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config_dict = {config1: "value1", config2: "value2"}
        
        # Должен быть только один ключ, так как конфигурации одинаковые
        assert len(config_dict) == 1
        assert config_dict[config1] == "value2"  # Последнее значение
    
    def test_config_copy(self, sample_credentials, sample_connection_config) -> None:
        """Тест копирования конфигурации."""
        from copy import copy
        
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config,
            enable_websocket=False,
            cache_ttl=120
        )
        
        config_copy = copy(config)
        
        assert config == config_copy
        assert config is not config_copy
    
    def test_config_deep_copy(self, sample_credentials, sample_connection_config) -> None:
        """Тест глубокого копирования конфигурации."""
        from copy import deepcopy
        
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        config_deep_copy = deepcopy(config)
        
        assert config == config_deep_copy
        assert config is not config_deep_copy
    
    def test_config_field_types(self, sample_credentials, sample_connection_config) -> None:
        """Тест типов полей конфигурации."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config
        )
        
        assert type(config.exchange_name) == ExchangeName
        assert type(config.credentials) == ExchangeCredentials
        assert type(config.connection_config) == ConnectionConfig
        assert isinstance(config.enable_websocket, bool)
        assert isinstance(config.enable_rest, bool)
        assert isinstance(config.enable_rate_limiting, bool)
        assert isinstance(config.enable_caching, bool)
        assert isinstance(config.cache_ttl, int)
        assert isinstance(config.max_cache_size, int)
        assert isinstance(config.retry_on_failure, bool)
        assert isinstance(config.max_retries, int)
        assert isinstance(config.retry_delay, float)
        assert isinstance(config.timeout, float)
        assert isinstance(config.ping_interval, int)
        assert isinstance(config.reconnect_interval, int)
        assert isinstance(config.max_reconnects, int)
    
    def test_config_validation_positive_values(self, sample_credentials, sample_connection_config) -> None:
        """Тест валидации положительных значений."""
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config,
            cache_ttl=1,
            max_cache_size=1,
            max_retries=1,
            retry_delay=0.1,
            timeout=0.1,
            ping_interval=1,
            reconnect_interval=1,
            max_reconnects=1
        )
        
        # Проверяем, что конфигурация создается без ошибок
        assert config.cache_ttl == 1
        assert config.max_cache_size == 1
        assert config.max_retries == 1
        assert config.retry_delay == 0.1
        assert config.timeout == 0.1
        assert config.ping_interval == 1
        assert config.reconnect_interval == 1
        assert config.max_reconnects == 1
    
    def test_config_edge_cases(self, sample_credentials, sample_connection_config) -> None:
        """Тест граничных случаев конфигурации."""
        # Очень большие значения
        config = ExchangeServiceConfig(
            exchange_name=ExchangeName.BINANCE,
            credentials=sample_credentials,
            connection_config=sample_connection_config,
            cache_ttl=999999,
            max_cache_size=999999,
            max_retries=999999,
            retry_delay=999999.0,
            timeout=999999.0,
            ping_interval=999999,
            reconnect_interval=999999,
            max_reconnects=999999
        )
        
        # Проверяем, что конфигурация создается без ошибок
        assert config.cache_ttl == 999999
        assert config.max_cache_size == 999999
        assert config.max_retries == 999999
        assert config.retry_delay == 999999.0
        assert config.timeout == 999999.0
        assert config.ping_interval == 999999
        assert config.reconnect_interval == 999999
        assert config.max_reconnects == 999999 
