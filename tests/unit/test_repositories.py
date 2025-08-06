"""
Unit тесты для репозиториев.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from uuid import uuid4
from decimal import Decimal
from datetime import datetime
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.trading import Trade, Position
from domain.entities.strategy import Strategy, StrategyStatus, StrategyType, StrategyParameters
from domain.entities.ml import Model, ModelStatus, ModelType
from domain.entities.portfolio import Portfolio, PortfolioStatus
from domain.entities.risk import RiskManager, RiskLevel, RiskProfile
from domain.entities.market import MarketData, MarketRegime, Timeframe
from domain.entities.trading_pair import TradingPair
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.type_definitions import OrderId, TradeId, Symbol, TimestampValue
from infrastructure.repositories.trading_repository import InMemoryTradingRepository, PostgresTradingRepository
from infrastructure.repositories.strategy_repository import InMemoryStrategyRepository, PostgresStrategyRepository
from infrastructure.repositories.ml_repository import InMemoryMLRepository, PostgresMLRepository
from infrastructure.repositories.portfolio_repository import InMemoryPortfolioRepository, PostgresPortfolioRepository
from infrastructure.repositories.risk_repository import InMemoryRiskRepository, PostgresRiskRepository
from infrastructure.repositories.market_repository import InMemoryMarketRepository, PostgresMarketRepository
class TestTradingRepository:
    """Тесты для торгового репозитория."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryTradingRepository()
class TestStrategyRepository:
    """Тесты для репозитория стратегий."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryStrategyRepository()
class TestMLRepository:
    """Тесты для репозитория ML моделей."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryMLRepository()
class TestPortfolioRepository:
    """Тесты для репозитория портфеля."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryPortfolioRepository()
        # Обновляем цену
class TestRiskRepository:
    """Тесты для репозитория рисков."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryRiskRepository()
class TestMarketRepository:
    """Тесты для репозитория рыночных данных."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryMarketRepository()
class TestPostgresRepositories:
    """Тесты для PostgreSQL репозиториев."""
    @pytest.mark.asyncio
    def test_postgres_trading_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория торговли."""
        from infrastructure.repositories.trading_repository import PostgresTradingRepository
        repository = PostgresTradingRepository("postgresql://test")
        assert repository is not None
    @pytest.mark.asyncio
    def test_postgres_strategy_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория стратегий."""
        from infrastructure.repositories.strategy_repository import PostgresStrategyRepository
        repository = PostgresStrategyRepository("postgresql://test")
        assert repository is not None
    @pytest.mark.asyncio
    def test_postgres_ml_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория ML."""
        from infrastructure.repositories.ml_repository import PostgresMLRepository
        repository = PostgresMLRepository("postgresql://test")
        assert repository is not None
    @pytest.mark.asyncio
    def test_postgres_portfolio_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория портфеля."""
        from infrastructure.repositories.portfolio_repository import PostgresPortfolioRepository
        repository = PostgresPortfolioRepository("postgresql://test")
        assert repository is not None
    @pytest.mark.asyncio
    def test_postgres_risk_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория рисков."""
        from infrastructure.repositories.risk_repository import PostgresRiskRepository
        repository = PostgresRiskRepository("postgresql://test")
        assert repository is not None
    @pytest.mark.asyncio
    def test_postgres_market_repository_initialization(self: "TestPostgresRepositories") -> None:
        """Тест инициализации PostgreSQL репозитория рынка."""
        from infrastructure.repositories.market_repository import PostgresMarketRepository
        repository = PostgresMarketRepository("postgresql://test")
        assert repository is not None 
