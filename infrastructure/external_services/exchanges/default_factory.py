"""
Фабрика с дефолтными конфигурациями для Exchange Services.
"""

from typing import Optional, Dict, Any
from loguru import logger

from domain.type_definitions.external_service_types import (
    ConnectionConfig,
    ExchangeCredentials,
    ExchangeType,
    APIKey,
    APISecret,
    ExchangeName,
)


from .config import ExchangeServiceConfig
from .binance_exchange_service import BinanceExchangeService
from .bybit_exchange_service import BybitExchangeService


class DefaultExchangeFactory:
    """Фабрика с дефолтными конфигурациями для быстрого создания exchange services."""
    
    @staticmethod
    def create_default_config() -> ExchangeServiceConfig:
        """Создаёт дефолтную конфигурацию для exchange service."""
        # Дефолтные credentials (для тестирования/демо режима)
        default_credentials = ExchangeCredentials(
            api_key=APIKey("demo_api_key"),
            api_secret=APISecret("demo_api_secret"),
            api_passphrase=None,
            sandbox=True  # Используем sandbox по умолчанию
        )
        
        # Дефолтный connection config
        default_connection = ConnectionConfig(
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0
        )
        
        return ExchangeServiceConfig(
            exchange_name=ExchangeName("default"),
            credentials=default_credentials,
            connection_config=default_connection
        )
    
    @staticmethod 
    def create_binance_service(config: Optional[ExchangeServiceConfig] = None) -> BinanceExchangeService:
        """Создаёт Binance service с дефолтной конфигурацией."""
        try:
            if config is None:
                config = DefaultExchangeFactory.create_default_config()
                config.exchange_name = ExchangeName("binance")
            
            service = BinanceExchangeService(config)
            logger.info("BinanceExchangeService создан с дефолтной конфигурацией")
            return service
            
        except Exception as e:
            logger.error(f"Ошибка создания BinanceExchangeService: {e}")
            raise
    
    @staticmethod
    def create_bybit_service(config: Optional[ExchangeServiceConfig] = None) -> BybitExchangeService:
        """Создаёт Bybit service с дефолтной конфигурацией."""
        try:
            if config is None:
                config = DefaultExchangeFactory.create_default_config()
                config.exchange_name = ExchangeName("bybit")
            
            service = BybitExchangeService(config)
            logger.info("BybitExchangeService создан с дефолтной конфигурацией")
            return service
            
        except Exception as e:
            logger.error(f"Ошибка создания BybitExchangeService: {e}")
            raise
    
    @staticmethod
    def create_exchange_from_dict(exchange_type: str, config_dict: Optional[Dict[str, Any]] = None) -> object:
        """Создаёт exchange service из словаря конфигурации."""
        try:
            if config_dict is None:
                config_dict = {}
            
            # Создаём credentials из словаря
            credentials = ExchangeCredentials(
                api_key=APIKey(config_dict.get("api_key", "demo_api_key")),
                api_secret=APISecret(config_dict.get("api_secret", "demo_api_secret")),
                api_passphrase=config_dict.get("api_passphrase"),
                sandbox=config_dict.get("sandbox", True)
            )
            
            # Создаём connection config из словаря
            connection_config = ConnectionConfig(
                timeout=config_dict.get("timeout", 30.0),
                max_retries=config_dict.get("max_retries", 3),
                retry_delay=config_dict.get("retry_delay", 1.0)
            )
            
            # Создаём полную конфигурацию
            config = ExchangeServiceConfig(
                exchange_name=ExchangeName(exchange_type.lower()),
                credentials=credentials,
                connection_config=connection_config
            )
            
            # Создаём соответствующий сервис
            if exchange_type.lower() == "binance":
                return BinanceExchangeService(config)
            elif exchange_type.lower() == "bybit":
                return BybitExchangeService(config)
            else:
                raise ValueError(f"Неподдерживаемый тип биржи: {exchange_type}")
                
        except Exception as e:
            logger.error(f"Ошибка создания exchange service из словаря: {e}")
            raise
    
    @staticmethod
    def get_supported_exchanges() -> list:
        """Возвращает список поддерживаемых бирж."""
        return ["binance", "bybit"]
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any]) -> bool:
        """Валидирует конфигурацию exchange."""
        required_fields = ["api_key", "api_secret"]
        
        for field in required_fields:
            if field not in config_dict:
                logger.warning(f"Отсутствует обязательное поле: {field}")
                return False
        
        return True


def create_default_binance() -> BinanceExchangeService:
    """Быстрое создание Binance service с дефолтными настройками."""
    return DefaultExchangeFactory.create_binance_service()


def create_default_bybit() -> BybitExchangeService:
    """Быстрое создание Bybit service с дефолтными настройками."""
    return DefaultExchangeFactory.create_bybit_service()


def create_demo_exchange(exchange_type: str = "binance") -> object:
    """Создаёт demo exchange service для тестирования."""
    demo_config = {
        "api_key": "demo_api_key_12345",
        "api_secret": "demo_secret_67890", 
        "sandbox": True,
        "timeout": 30.0,
        "max_retries": 3
    }
    
    return DefaultExchangeFactory.create_exchange_from_dict(exchange_type, demo_config)