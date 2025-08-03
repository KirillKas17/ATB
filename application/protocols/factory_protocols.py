"""
Протоколы фабрик для application слоя.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from .service_protocols import (
    AnalyticsService,
    CacheService,
    EvolutionService,
    MarketService,
    MetricsService,
    MLService,
    NotificationService,
    PortfolioService,
    RiskService,
    StrategyService,
    SymbolSelectionService,
    TradingService,
)

# ============================================================================
# FACTORY PROTOCOLS
# ============================================================================


@runtime_checkable
class ServiceFactory(Protocol):
    """Протокол для фабрики сервисов."""

    @abstractmethod
    def create_market_service(self) -> MarketService:
        """Создание сервиса рыночных данных."""
        ...

    @abstractmethod
    def create_ml_service(self) -> MLService:
        """Создание ML сервиса."""
        ...

    @abstractmethod
    def create_trading_service(self) -> TradingService:
        """Создание торгового сервиса."""
        ...

    @abstractmethod
    def create_strategy_service(self) -> StrategyService:
        """Создание сервиса стратегий."""
        ...

    @abstractmethod
    def create_portfolio_service(self) -> PortfolioService:
        """Создание сервиса портфеля."""
        ...

    @abstractmethod
    def create_risk_service(self) -> RiskService:
        """Создание сервиса рисков."""
        ...

    @abstractmethod
    def create_cache_service(self) -> CacheService:
        """Создание сервиса кэширования."""
        ...

    @abstractmethod
    def create_notification_service(self) -> NotificationService:
        """Создание сервиса уведомлений."""
        ...

    @abstractmethod
    def create_analytics_service(self) -> AnalyticsService:
        """Создание аналитического сервиса."""
        ...

    @abstractmethod
    def create_metrics_service(self) -> MetricsService:
        """Создание сервиса метрик."""
        ...

    @abstractmethod
    def create_evolution_service(self) -> EvolutionService:
        """Создание сервиса эволюции."""
        ...

    @abstractmethod
    def create_symbol_selection_service(self) -> SymbolSelectionService:
        """Создание сервиса выбора символов."""
        ...
