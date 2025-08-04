"""
Фабрика для создания сервисов бирж - Production Ready
"""

from abc import ABC, abstractmethod
from typing import Optional

from domain.type_definitions.external_service_types import (
    ConnectionConfig,
    ExchangeCredentials,
    ExchangeType,
    APIKey,
    APISecret,
    ExchangeName,
)

from .base_exchange_service import BaseExchangeService
from .binance_exchange_service import BinanceExchangeService
from .bybit_exchange_service import BybitExchangeService
from .config import ExchangeServiceConfig


class ExchangeServiceFactory:
    """Фабрика для создания сервисов бирж."""

    @staticmethod
    def create_exchange_service(
        exchange_type: ExchangeType,
        credentials: ExchangeCredentials,
        connection_config: Optional[ConnectionConfig] = None,
    ) -> BaseExchangeService:
        """Создать сервис биржи."""
        if connection_config is None:
            connection_config = ConnectionConfig()

        config = ExchangeServiceConfig(
            exchange_name=ExchangeName(exchange_type.value),
            credentials=credentials,
            connection_config=connection_config,
        )

        if exchange_type == ExchangeType.BYBIT:
            return BybitExchangeService(config)
        elif exchange_type == ExchangeType.BINANCE:
            return BinanceExchangeService(config)
        else:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")

    @staticmethod
    def create_bybit_service(
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        connection_config: Optional[ConnectionConfig] = None,
    ) -> BybitExchangeService:
        """Создать сервис Bybit."""
        credentials = ExchangeCredentials(
            api_key=APIKey(api_key), api_secret=APISecret(api_secret), testnet=testnet
        )

        if connection_config is None:
            connection_config = ConnectionConfig()

        config = ExchangeServiceConfig(
            exchange_name=ExchangeName("bybit"),
            credentials=credentials,
            connection_config=connection_config,
        )

        return BybitExchangeService(config)

    @staticmethod
    def create_binance_service(
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        connection_config: Optional[ConnectionConfig] = None,
    ) -> BinanceExchangeService:
        """Создать сервис Binance."""
        credentials = ExchangeCredentials(
            api_key=APIKey(api_key), api_secret=APISecret(api_secret), testnet=testnet
        )

        if connection_config is None:
            connection_config = ConnectionConfig()

        config = ExchangeServiceConfig(
            exchange_name=ExchangeName("binance"),
            credentials=credentials,
            connection_config=connection_config,
        )

        return BinanceExchangeService(config)
