"""
Репозитории инфраструктуры - сверхпродвинутые промышленные реализации доменных репозиториев.

Этот модуль предоставляет реализации репозиториев для всех доменных сущностей
с поддержкой кэширования, валидации, метрик и событийной архитектуры.
"""

from typing import Any, Dict, Type

from .market_repository import (
    InMemoryMarketRepository,
    MarketRepositoryProtocol,
    PostgresMarketRepository,
)
from .ml_repository import (
    InMemoryMLRepository,
    MLRepositoryProtocol,
    PostgresMLRepository,
)
from .order_repository import (
    InMemoryOrderRepository,
    OrderRepositoryProtocol,
    PostgresOrderRepository,
)
from .portfolio_repository import (
    InMemoryPortfolioRepository,
    PortfolioRepositoryProtocol,
    PostgresPortfolioRepository,
)
from .position_repository import (
    InMemoryPositionRepository,
    PositionRepositoryProtocol,
    PostgresPositionRepository,
)
from .risk_repository import (
    InMemoryRiskRepository,
    PostgresRiskRepository,
    RiskRepositoryProtocol,
)
from .strategy_repository import (
    InMemoryStrategyRepository,
    PostgresStrategyRepository,
    StrategyRepositoryProtocol,
)
from .trading import (
    InMemoryTradingRepository,
    PostgresTradingRepository,
    TradingBusinessRuleValidator,
    TradingDataValidator,
    TradingEventBus,
    TradingRepositoryCache,
    TradingRepositoryProtocol,
    TradingRepositoryServices,
)

# Реестр репозиториев для DI
REPOSITORY_REGISTRY: Dict[str, Type[Any]] = {
    "market": PostgresMarketRepository,
    "ml": PostgresMLRepository,
    "portfolio": PostgresPortfolioRepository,
    "risk": PostgresRiskRepository,
    "strategy": PostgresStrategyRepository,
    "trading": PostgresTradingRepository,
    "position": PostgresPositionRepository,
    "order": PostgresOrderRepository,
}

# Реестр in-memory репозиториев для тестирования
IN_MEMORY_REGISTRY: Dict[str, Type[Any]] = {
    "market": InMemoryMarketRepository,
    "ml": InMemoryMLRepository,
    "portfolio": InMemoryPortfolioRepository,
    "risk": InMemoryRiskRepository,
    "strategy": InMemoryStrategyRepository,
    "trading": InMemoryTradingRepository,
    "position": InMemoryPositionRepository,
    "order": InMemoryOrderRepository,
}

__all__ = [
    # Протоколы репозиториев
    "MarketRepositoryProtocol",
    "MLRepositoryProtocol",
    "PortfolioRepositoryProtocol",
    "RiskRepositoryProtocol",
    "StrategyRepositoryProtocol",
    "TradingRepositoryProtocol",
    "PositionRepositoryProtocol",
    "OrderRepositoryProtocol",
    # Trading репозитории и сервисы
    "PostgresTradingRepository",
    "InMemoryTradingRepository",
    "TradingRepositoryServices",
    "TradingEventBus",
    "TradingRepositoryCache",
    "TradingDataValidator",
    "TradingBusinessRuleValidator",
    # Portfolio репозитории
    "PostgresPortfolioRepository",
    "InMemoryPortfolioRepository",
    # Strategy репозитории
    "PostgresStrategyRepository",
    "InMemoryStrategyRepository",
    # Market репозитории
    "PostgresMarketRepository",
    "InMemoryMarketRepository",
    # Risk репозитории
    "PostgresRiskRepository",
    "InMemoryRiskRepository",
    # ML репозитории
    "PostgresMLRepository",
    "InMemoryMLRepository",
    # Position репозитории
    "PostgresPositionRepository",
    "InMemoryPositionRepository",
    # Order репозитории
    "PostgresOrderRepository",
    "InMemoryOrderRepository",
    # Реестры
    "REPOSITORY_REGISTRY",
    "IN_MEMORY_REGISTRY",
]
