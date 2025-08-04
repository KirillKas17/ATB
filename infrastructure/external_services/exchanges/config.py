"""
Конфигурация для сервисов бирж - Production Ready
"""

from dataclasses import dataclass

from domain.type_definitions.external_service_types import (
    ConnectionConfig,
    ExchangeCredentials,
    ExchangeName,
)


@dataclass
class ExchangeServiceConfig:
    """Конфигурация сервиса биржи."""

    exchange_name: ExchangeName
    credentials: ExchangeCredentials
    connection_config: ConnectionConfig
    enable_websocket: bool = True
    enable_rest: bool = True
    enable_rate_limiting: bool = True
    enable_caching: bool = True
    cache_ttl: int = 60
    max_cache_size: int = 1000
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    ping_interval: int = 30
    reconnect_interval: int = 5
    max_reconnects: int = 5
