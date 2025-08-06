"""
Фабрика сервисов для централизованного создания и управления сервисами.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from application.protocols.service_protocols import (
    CacheService,
    MarketService,
    MLService,
    NotificationService,
    PortfolioService,
    RiskService,
    StrategyService,
    TradingService,
)
from application.services.implementations.cache_service_impl import CacheServiceImpl
from application.services.implementations.market_service_impl import MarketServiceImpl
from application.services.implementations.ml_service_impl import MLServiceImpl
from application.services.implementations.notification_service_impl import (
    NotificationServiceImpl,
)
from application.services.implementations.portfolio_service_impl import (
    PortfolioServiceImpl,
)
from application.services.implementations.risk_service_impl import RiskServiceImpl
# from application.services.implementations.strategy_service_impl import (
#     StrategyServiceImpl,
# )
from application.services.implementations.trading_service_impl import TradingServiceImpl


class ServiceFactory(ABC):
    """Абстрактная фабрика сервисов."""

    @abstractmethod
    def create_market_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> MarketService:
        """Создание сервиса рынка."""
        pass

    @abstractmethod
    def create_ml_service(self, config: Optional[Dict[str, Any]] = None) -> MLService:
        """Создание ML сервиса."""
        pass

    @abstractmethod
    def create_trading_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> TradingService:
        """Создание торгового сервиса."""
        pass

    @abstractmethod
    def create_strategy_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> StrategyService:
        """Создание сервиса стратегий."""
        pass

    @abstractmethod
    def create_portfolio_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> PortfolioService:
        """Создание сервиса портфелей."""
        pass

    @abstractmethod
    def create_risk_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> RiskService:
        """Создание сервиса рисков."""
        pass

    @abstractmethod
    def create_cache_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> CacheService:
        """Создание сервиса кэширования."""
        pass

    @abstractmethod
    def create_notification_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> NotificationService:
        """Создание сервиса уведомлений."""
        pass


class DefaultServiceFactory(ServiceFactory):
    """Реализация фабрики сервисов по умолчанию."""

    def __init__(self) -> None:
        """Инициализация фабрики сервисов."""
        self._services: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        # Исправление: определяем global_config как пустой словарь по умолчанию
        self.global_config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._service_instances: Dict[str, Any] = {}

    def create_service(self, service_type: str, **kwargs: Any) -> Any:
        """Создание сервиса."""
        if service_type == "trading":
            return self.create_trading_service(**kwargs)
        elif service_type == "risk":
            return self.create_risk_service(**kwargs)
        elif service_type == "portfolio":
            return self.create_portfolio_service(**kwargs)
        elif service_type == "ml":
            return self.create_ml_service(**kwargs)
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    def create_market_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> MarketService:
        """Создание сервиса рынка."""
        service_key = "market_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("market_service", {}), config or {}
            )

            # Получаем зависимости
            technical_analysis_service = self._get_technical_analysis_service()
            market_metrics_service = self._get_market_metrics_service()
            market_repository = self._get_market_repository()

            self._service_instances[service_key] = MarketServiceImpl(
                market_repository=market_repository,
                technical_analysis_service=technical_analysis_service,
                market_metrics_service=market_metrics_service,
                config=service_config,
            )
            self.logger.info("Created MarketService instance")

        return self._service_instances[service_key]

    def create_ml_service(self, config: Optional[Dict[str, Any]] = None) -> MLService:
        """Создание ML сервиса."""
        service_key = "ml_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("ml_service", {}), config or {}
            )

            # Получаем зависимости
            ml_predictor = self._get_ml_predictor()
            ml_repository = self._get_ml_repository()

            self._service_instances[service_key] = MLServiceImpl(
                ml_repository=ml_repository,
                ml_predictor=ml_predictor,
                config=service_config,
            )
            self.logger.info("Created MLService instance")

        return self._service_instances[service_key]

    def create_trading_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> TradingService:
        """Создание торгового сервиса."""
        service_key = "trading_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("trading_service", {}), config or {}
            )

            # Получаем зависимости
            signal_service = self._get_signal_service()
            trading_repository = self._get_trading_repository()

            self._service_instances[service_key] = TradingServiceImpl(
                trading_repository=trading_repository,
                signal_service=signal_service,
                config=service_config,
            )
            self.logger.info("Created TradingService instance")

        return self._service_instances[service_key]

    def create_strategy_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> StrategyService:
        """Создание сервиса стратегий."""
        service_key = "strategy_service"

        if service_key not in self._service_instances:
            service_config = config or {}
            service_config.update({"name": "strategy_service"})

            # Используем реализованный StrategyService из domain.services.strategy_service
            from domain.services.strategy_service import strategy_service
            self._service_instances[service_key] = strategy_service
            self.logger.info("Created StrategyService instance using domain service")

        return self._service_instances[service_key]

    def create_portfolio_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> PortfolioService:
        """Создание сервиса портфелей."""
        service_key = "portfolio_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("portfolio_service", {}), config or {}
            )

            # Получаем зависимости
            portfolio_optimizer = self._get_portfolio_optimizer()
            portfolio_repository = self._get_portfolio_repository()

            self._service_instances[service_key] = PortfolioServiceImpl(
                portfolio_repository=portfolio_repository,
                portfolio_optimizer=portfolio_optimizer,
                config=service_config,
            )
            self.logger.info("Created PortfolioService instance")

        return self._service_instances[service_key]

    def create_risk_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> RiskService:
        """Создание сервиса рисков."""
        service_key = "risk_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("risk_service", {}), config or {}
            )

            # Получаем зависимости
            risk_repository = self._get_risk_repository()

            self._service_instances[service_key] = RiskServiceImpl(
                risk_repository=risk_repository,
                config=service_config,
            )
            self.logger.info("Created RiskService instance")

        return self._service_instances[service_key]

    def create_cache_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> CacheService:
        """Создание сервиса кэширования."""
        service_key = "cache_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("cache_service", {}), config or {}
            )

            self._service_instances[service_key] = CacheServiceImpl(service_config)
            self.logger.info("Created CacheService instance")

        return self._service_instances[service_key]

    def create_notification_service(
        self, config: Optional[Dict[str, Any]] = None
    ) -> NotificationService:
        """Создание сервиса уведомлений."""
        service_key = "notification_service"

        if service_key not in self._service_instances:
            service_config = self._merge_configs(
                self.global_config.get("notification_service", {}), config or {}
            )

            self._service_instances[service_key] = NotificationServiceImpl(service_config)
            self.logger.info("Created NotificationService instance")

        return self._service_instances[service_key]

    def _merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Объединение конфигураций."""
        merged = base_config.copy()
        merged.update(override_config)
        return merged

    def _get_risk_repository(self) -> Any:
        """Получить репозиторий рисков."""
        try:
            from infrastructure.repositories.risk_repository import RiskRepositoryImpl
            return RiskRepositoryImpl()
        except ImportError:
            self.logger.warning("RiskRepository not available, using mock")
            from application.safe_services import SafeRiskService
            return SafeRiskService()

    def _get_technical_analysis_service(self) -> Any:
        """Получить сервис технического анализа."""
        try:
            from infrastructure.services.technical_analysis_service import TechnicalAnalysisService
            return TechnicalAnalysisService()
        except ImportError:
            self.logger.warning("TechnicalAnalysisService not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("TechnicalAnalysisService")

    def _get_market_metrics_service(self) -> Any:
        """Получить сервис рыночных метрик."""
        try:
            from infrastructure.services.market_metrics_service import MarketMetricsService
            return MarketMetricsService()
        except ImportError:
            self.logger.warning("MarketMetricsService not available, using market service")
            try:
                from application.services.implementations.market_service_impl import MarketServiceImpl
                return MarketServiceImpl()
            except ImportError:
                from application.safe_services import SafeMarketService
                return SafeMarketService()

    def _get_market_repository(self) -> Any:
        """Получить репозиторий рынка."""
        try:
            from infrastructure.repositories.market_repository import MarketRepositoryImpl
            return MarketRepositoryImpl()
        except ImportError:
            self.logger.warning("MarketRepository not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("MarketRepository")

    def _get_ml_predictor(self) -> Any:
        """Получить ML предиктор."""
        try:
            from infrastructure.ml_services.predictor import MLPredictor
            return MLPredictor()
        except ImportError:
            self.logger.warning("MLPredictor not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("MLPredictor")

    def _get_ml_repository(self) -> Any:
        """Получить ML репозиторий."""
        try:
            from infrastructure.repositories.ml_repository import MLRepositoryImpl
            return MLRepositoryImpl()
        except ImportError:
            self.logger.warning("MLRepository not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("MLRepository")

    def _get_signal_service(self) -> Any:
        """Получить сервис сигналов."""
        try:
            from infrastructure.services.signal_service import SignalService
            return SignalService()
        except ImportError:
            self.logger.warning("SignalService not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("SignalService")

    def _get_trading_repository(self) -> Any:
        """Получить торговый репозиторий."""
        try:
            from infrastructure.repositories.trading_repository import TradingRepositoryImpl
            return TradingRepositoryImpl()
        except ImportError:
            self.logger.warning("TradingRepository not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("TradingRepository")

    def _get_portfolio_optimizer(self) -> Any:
        """Получить оптимизатор портфеля."""
        try:
            from infrastructure.agents.portfolio.optimizers import PortfolioOptimizer
            return PortfolioOptimizer()
        except ImportError:
            self.logger.warning("PortfolioOptimizer not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("PortfolioOptimizer")

    def _get_portfolio_repository(self) -> Any:
        """Получить репозиторий портфеля."""
        try:
            from infrastructure.repositories.portfolio_repository import PortfolioRepositoryImpl
            return PortfolioRepositoryImpl()
        except ImportError:
            self.logger.warning("PortfolioRepository not available, using mock")
            from safe_import_wrapper import SafeImportMock
            return SafeImportMock("PortfolioRepository")

    def get_service_instance(self, service_type: str) -> Optional[Any]:
        """Получить экземпляр сервиса по типу."""
        return self._service_instances.get(service_type)

    def get_all_services(self) -> Dict[str, Any]:
        """Получить все сервисы."""
        return self._service_instances.copy()

    async def initialize_all_services(self) -> None:
        """Инициализация всех сервисов."""
        try:
            # Создаем все сервисы
            self.create_market_service()
            self.create_ml_service()
            self.create_trading_service()
            self.create_strategy_service()
            self.create_portfolio_service()
            self.create_risk_service()
            self.create_cache_service()
            self.create_notification_service()
            self.logger.info("All services initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise

    async def stop_all_services(self) -> None:
        """Остановка всех сервисов."""
        try:
            for service_name, service in self._service_instances.items():
                if hasattr(service, 'stop'):
                    await service.stop()
                self.logger.info(f"Stopped service: {service_name}")
            self._service_instances.clear()
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")


class ServiceFactoryRegistry:
    """Реестр фабрик сервисов."""

    def __init__(self) -> None:
        self._factories: Dict[str, Type[ServiceFactory]] = {}
        self._default_factory: Optional[Type[ServiceFactory]] = None

    def register_factory(self, name: str, factory_class: Type[ServiceFactory]) -> None:
        """Регистрация фабрики."""
        self._factories[name] = factory_class

    def set_default_factory(self, factory_class: Type[ServiceFactory]) -> None:
        """Установка фабрики по умолчанию."""
        self._default_factory = factory_class

    def get_factory(self, name: str) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по имени."""
        return self._factories.get(name)

    def get_default_factory(self) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по умолчанию."""
        return self._default_factory

    def create_factory(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по имени."""
        factory_class = self.get_factory(name)
        if factory_class:
            return factory_class()
        return None

    def create_default_factory(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по умолчанию."""
        if self._default_factory:
            return self._default_factory()
        return None


# Глобальный реестр фабрик
_factory_registry = ServiceFactoryRegistry()
_factory_registry.register_factory("default", DefaultServiceFactory)
_factory_registry.set_default_factory(DefaultServiceFactory)


def get_service_factory(
    name: str = "default", config: Optional[Dict[str, Any]] = None
) -> ServiceFactory:
    """Получение фабрики сервисов."""
    factory = _factory_registry.create_factory(name, config)
    if factory is None:
        factory = _factory_registry.create_default_factory(config)
    if factory is None:
        raise ValueError(f"Could not create service factory: {name}")
    return factory


def register_service_factory(name: str, factory_class: Type[ServiceFactory]) -> None:
    """Регистрация фабрики сервисов."""
    _factory_registry.register_factory(name, factory_class)


def set_default_service_factory(factory_class: Type[ServiceFactory]) -> None:
    """Установка фабрики сервисов по умолчанию."""
    _factory_registry.set_default_factory(factory_class)
