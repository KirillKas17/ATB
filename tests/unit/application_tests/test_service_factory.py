"""
Тесты для фабрики сервисов.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, patch
from application.services.service_factory import (
    ServiceFactory,
    DefaultServiceFactory,
    ServiceFactoryRegistry,
    get_service_factory,
)
from application.protocols.service_protocols import (
    MarketService,
    MLService,
    TradingService,
    StrategyService,
    PortfolioService,
    RiskService,
    CacheService,
    NotificationService,
)


class TestServiceFactory:
    """Тесты для абстрактной фабрики сервисов."""

    def test_service_factory_is_abstract(self: "TestServiceFactory") -> None:
        """Тест что ServiceFactory является абстрактным классом."""
        with pytest.raises(TypeError):
            ServiceFactory()


class TestDefaultServiceFactory:
    """Тесты для реализации фабрики сервисов по умолчанию."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            "market_service": {"cache_enabled": True},
            "ml_service": {"model_path": "/path/to/model"},
            "trading_service": {"max_orders": 100},
            "strategy_service": {"max_strategies": 50},
            "portfolio_service": {"max_portfolios": 10},
            "risk_service": {"max_risk": 0.1},
            "cache_service": {"max_size": 1000},
            "notification_service": {"email_enabled": True},
        }
        self.factory = DefaultServiceFactory(self.config)

    def test_create_market_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса рынка."""
        service = self.factory.create_market_service()
        assert isinstance(service, MarketService)
        assert service is self.factory.create_market_service()  # Синглтон

    def test_create_ml_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания ML сервиса."""
        service = self.factory.create_ml_service()
        assert isinstance(service, MLService)
        assert service is self.factory.create_ml_service()  # Синглтон

    def test_create_trading_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания торгового сервиса."""
        service = self.factory.create_trading_service()
        assert isinstance(service, TradingService)
        assert service is self.factory.create_trading_service()  # Синглтон

    def test_create_strategy_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса стратегий."""
        service = self.factory.create_strategy_service()
        assert isinstance(service, StrategyService)
        assert service is self.factory.create_strategy_service()  # Синглтон

    def test_create_portfolio_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса портфелей."""
        service = self.factory.create_portfolio_service()
        assert isinstance(service, PortfolioService)
        assert service is self.factory.create_portfolio_service()  # Синглтон

    def test_create_risk_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса рисков."""
        service = self.factory.create_risk_service()
        assert isinstance(service, RiskService)
        assert service is self.factory.create_risk_service()  # Синглтон

    def test_create_cache_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса кэширования."""
        service = self.factory.create_cache_service()
        assert isinstance(service, CacheService)
        assert service is self.factory.create_cache_service()  # Синглтон

    def test_create_notification_service(self: "TestDefaultServiceFactory") -> None:
        """Тест создания сервиса уведомлений."""
        service = self.factory.create_notification_service()
        assert isinstance(service, NotificationService)
        assert service is self.factory.create_notification_service()  # Синглтон

    def test_merge_configs(self: "TestDefaultServiceFactory") -> None:
        """Тест объединения конфигураций."""
        base_config = {"key1": "value1", "key2": "value2"}
        override_config = {"key2": "new_value", "key3": "value3"}
        merged = self.factory._merge_configs(base_config, override_config)
        assert merged["key1"] == "value1"
        assert merged["key2"] == "new_value"
        assert merged["key3"] == "value3"

    def test_get_service_instance(self: "TestDefaultServiceFactory") -> None:
        """Тест получения экземпляра сервиса."""
        service = self.factory.create_market_service()
        retrieved = self.factory.get_service_instance("market_service")
        assert retrieved is service

    def test_get_all_services(self: "TestDefaultServiceFactory") -> None:
        """Тест получения всех сервисов."""
        # Создаем несколько сервисов
        self.factory.create_market_service()
        self.factory.create_ml_service()
        all_services = self.factory.get_all_services()
        assert "market_service" in all_services
        assert "ml_service" in all_services
        assert len(all_services) == 2

    @pytest.mark.asyncio
    async def test_initialize_all_services(self: "TestDefaultServiceFactory") -> None:
        """Тест инициализации всех сервисов."""
        # Создаем сервисы
        self.factory.create_market_service()
        self.factory.create_ml_service()
        # Инициализируем все сервисы
        await self.factory.initialize_all_services()
        # Проверяем что все сервисы инициализированы
        all_services = self.factory.get_all_services()
        for service in all_services.values():
            assert hasattr(service, "is_running")

    @pytest.mark.asyncio
    async def test_stop_all_services(self: "TestDefaultServiceFactory") -> None:
        """Тест остановки всех сервисов."""
        # Создаем и инициализируем сервисы
        self.factory.create_market_service()
        self.factory.create_ml_service()
        await self.factory.initialize_all_services()
        # Останавливаем все сервисы
        await self.factory.stop_all_services()
        # Проверяем что сервисы очищены
        assert len(self.factory.get_all_services()) == 0


class TestServiceFactoryRegistry:
    """Тесты для реестра фабрик сервисов."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = ServiceFactoryRegistry()
        self.mock_factory_class = Mock()

    def test_register_factory(self: "TestServiceFactoryRegistry") -> None:
        """Тест регистрации фабрики."""
        self.registry.register_factory("test_factory", self.mock_factory_class)
        retrieved = self.registry.get_factory("test_factory")
        assert retrieved is self.mock_factory_class

    def test_set_default_factory(self: "TestServiceFactoryRegistry") -> None:
        """Тест установки фабрики по умолчанию."""
        self.registry.set_default_factory(self.mock_factory_class)
        default = self.registry.get_default_factory()
        assert default is self.mock_factory_class

    def test_create_factory(self: "TestServiceFactoryRegistry") -> None:
        """Тест создания экземпляра фабрики."""
        config = {"test": "config"}
        mock_factory_instance = Mock()
        self.mock_factory_class.return_value = mock_factory_instance
        self.registry.register_factory("test_factory", self.mock_factory_class)
        created = self.registry.create_factory("test_factory", config)
        assert created is mock_factory_instance
        self.mock_factory_class.assert_called_once_with(config)

    def test_create_default_factory(self: "TestServiceFactoryRegistry") -> None:
        """Тест создания экземпляра фабрики по умолчанию."""
        config = {"test": "config"}
        mock_factory_instance = Mock()
        self.mock_factory_class.return_value = mock_factory_instance
        self.registry.set_default_factory(self.mock_factory_class)
        created = self.registry.create_default_factory(config)
        assert created is mock_factory_instance
        self.mock_factory_class.assert_called_once_with(config)

    def test_create_factory_not_found(self: "TestServiceFactoryRegistry") -> None:
        """Тест создания несуществующей фабрики."""
        created = self.registry.create_factory("non_existent")
        assert created is None

    def test_create_default_factory_not_set(self: "TestServiceFactoryRegistry") -> None:
        """Тест создания фабрики по умолчанию когда она не установлена."""
        created = self.registry.create_default_factory()
        assert created is None


class TestGetServiceFactory:
    """Тесты для функции получения фабрики сервисов."""

    def test_get_service_factory_default(self: "TestGetServiceFactory") -> None:
        """Тест получения фабрики по умолчанию."""
        config = {"test": "config"}
        factory = get_service_factory(config=config)
        assert isinstance(factory, DefaultServiceFactory)

    def test_get_service_factory_named(self: "TestGetServiceFactory") -> None:
        """Тест получения именованной фабрики."""
        config = {"test": "config"}
        factory = get_service_factory("default", config)
        assert isinstance(factory, DefaultServiceFactory)

    def test_get_service_factory_fallback(self: "TestGetServiceFactory") -> None:
        """Тест fallback к фабрике по умолчанию."""
        config = {"test": "config"}
        factory = get_service_factory("non_existent", config)
        assert isinstance(factory, DefaultServiceFactory)


class TestServiceFactoryIntegration:
    """Интеграционные тесты фабрики сервисов."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            "market_service": {"cache_enabled": True, "update_interval": 60},
            "ml_service": {"model_path": "/path/to/model", "prediction_threshold": 0.7},
            "trading_service": {"max_orders": 100, "order_timeout": 30},
            "risk_service": {"max_risk": 0.1, "var_confidence_level": 0.95},
            "cache_service": {"max_size": 1000, "default_ttl": 300},
            "notification_service": {"email_enabled": True, "webhook_enabled": False},
        }
        self.factory = DefaultServiceFactory(self.config)

    @pytest.mark.asyncio
    async def test_full_service_lifecycle(self: "TestServiceFactoryIntegration") -> None:
        """Тест полного жизненного цикла сервисов."""
        # Создаем все сервисы
        market_service = self.factory.create_market_service()
        ml_service = self.factory.create_ml_service()
        trading_service = self.factory.create_trading_service()
        strategy_service = self.factory.create_strategy_service()
        portfolio_service = self.factory.create_portfolio_service()
        risk_service = self.factory.create_risk_service()
        cache_service = self.factory.create_cache_service()
        notification_service = self.factory.create_notification_service()
        # Проверяем что все сервисы созданы
        assert all(
            [
                market_service,
                ml_service,
                trading_service,
                strategy_service,
                portfolio_service,
                risk_service,
                cache_service,
                notification_service,
            ]
        )
        # Инициализируем все сервисы
        await self.factory.initialize_all_services()
        # Проверяем что все сервисы инициализированы
        all_services = self.factory.get_all_services()
        assert len(all_services) == 8
        for service in all_services.values():
            assert hasattr(service, "is_running")
        # Останавливаем все сервисы
        await self.factory.stop_all_services()
        # Проверяем что сервисы очищены
        assert len(self.factory.get_all_services()) == 0

    def test_service_configuration_inheritance(self: "TestServiceFactoryIntegration") -> None:
        """Тест наследования конфигурации сервисами."""
        # Создаем сервис с переопределенной конфигурацией
        override_config = {"cache_enabled": False}
        market_service = self.factory.create_market_service(override_config)
        # Проверяем что конфигурация применена
        assert hasattr(market_service, "config")

    def test_service_singleton_behavior(self: "TestServiceFactoryIntegration") -> None:
        """Тест поведения синглтона сервисов."""
        # Создаем сервис дважды
        service1 = self.factory.create_market_service()
        service2 = self.factory.create_market_service()
        # Проверяем что это один и тот же экземпляр
        assert service1 is service2
        # Проверяем что в реестре только один экземпляр
        all_services = self.factory.get_all_services()
        assert len(all_services) == 1
        assert "market_service" in all_services
