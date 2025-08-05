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
    @pytest.fixture
    def sample_order(self: "TestEvolvableMarketMakerAgent") -> Any:
        return Order(
            id=OrderId(uuid4()),
            trading_pair=TradingPair(
                Symbol("BTC/USDT"),
                Currency.BTC,
                Currency.USDT
            ),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.1"), Currency.BTC),
            price=Price(Decimal("50000"), Currency.USDT),
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
    @pytest.fixture
    def sample_trade(self: "TestEvolvableMarketMakerAgent") -> Any:
        return Trade(
            id=TradeId(uuid4()),
            order_id=OrderId(uuid4()),
            trading_pair=TradingPair(
                Symbol("BTC/USDT"),
                Currency.BTC,
                Currency.USDT
            ),
            side=OrderSide.BUY,
            quantity=Volume(Decimal("0.1"), Currency.BTC),
            price=Price(Decimal("50000"), Currency.USDT),
            timestamp=TimestampValue(datetime.now())
        )
    @pytest.mark.asyncio
    async def test_save_order(self, repository, sample_order) -> None:
        """Тест сохранения ордера."""
        result = await repository.save_order(sample_order)
        assert result is not None
        assert result.id == sample_order.id
        saved_order = await repository.get_order(sample_order.id)
        assert saved_order is not None
        assert saved_order.id == sample_order.id
    @pytest.mark.asyncio
    async def test_get_order_not_found(self, repository) -> None:
        """Тест получения несуществующего ордера."""
        order = await repository.get_order("non-existent-id")
        assert order is None
    @pytest.mark.asyncio
    async def test_save_trade(self, repository, sample_trade) -> None:
        """Тест сохранения сделки."""
        result = await repository.save_trade(sample_trade)
        assert result is not None
        assert result.id == sample_trade.id
        saved_trade = await repository.get_trade(sample_trade.id)
        assert saved_trade is not None
        assert saved_trade.id == sample_trade.id
    @pytest.mark.asyncio
    async def test_get_trades_by_order(self, repository, sample_trade) -> None:
        """Тест получения сделок по ордеру."""
        await repository.save_trade(sample_trade)
        trades = await repository.get_trades_by_order(sample_trade.order_id)
        assert len(trades) == 1
        assert trades[0].id == sample_trade.id
    @pytest.mark.asyncio
    async def test_delete_order(self, repository, sample_order) -> None:
        """Тест удаления ордера."""
        await repository.save_order(sample_order)
        result = await repository.delete(sample_order.id)
        assert result is True
        order = await repository.get_order(sample_order.id)
        assert order is None
class TestStrategyRepository:
    """Тесты для репозитория стратегий."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryStrategyRepository()
    @pytest.fixture
    def sample_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        return Strategy(
            id=uuid4(),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            status=StrategyStatus.ACTIVE,
            parameters=StrategyParameters(parameters={"param1": "value1"}),
            created_at=datetime.now()
        )
    @pytest.mark.asyncio
    async def test_save_strategy(self, repository, sample_strategy) -> None:
        """Тест сохранения стратегии."""
        result = await repository.save_strategy(sample_strategy)
        assert result is True
        saved_strategy = await repository.get_strategy(sample_strategy.id)
        assert saved_strategy is not None
        assert saved_strategy.id == sample_strategy.id
    @pytest.mark.asyncio
    async def test_get_strategies_by_type(self, repository, sample_strategy) -> None:
        """Тест получения стратегий по типу."""
        await repository.save_strategy(sample_strategy)
        strategies = await repository.get_strategies_by_type("trend_following")
        assert len(strategies) == 1
        assert strategies[0].id == sample_strategy.id
    @pytest.mark.asyncio
    async def test_get_active_strategies(self, repository, sample_strategy) -> None:
        """Тест получения активных стратегий."""
        await repository.save_strategy(sample_strategy)
        strategies = await repository.get_active_strategies()
        assert len(strategies) == 1
        assert strategies[0].id == sample_strategy.id
class TestMLRepository:
    """Тесты для репозитория ML моделей."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryMLRepository()
    @pytest.fixture
    def sample_model(self: "TestEvolvableMarketMakerAgent") -> Any:
        return Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.LINEAR_REGRESSION,
            status=ModelStatus.ACTIVE,
            accuracy=Decimal("0.85"),
            created_at=datetime.now()
        )
    @pytest.mark.asyncio
    async def test_save_model(self, repository, sample_model) -> None:
        """Тест сохранения модели."""
        result = await repository.save_model(sample_model)
        assert result is True
        saved_model = await repository.get_model(sample_model.id)
        assert saved_model is not None
        assert saved_model.id == sample_model.id
    @pytest.mark.asyncio
    async def test_get_models_by_type(self, repository, sample_model) -> None:
        """Тест получения моделей по типу."""
        await repository.save_model(sample_model)
        models = await repository.get_models_by_type("linear_regression")
        assert len(models) == 1
        assert models[0].id == sample_model.id
    @pytest.mark.asyncio
    async def test_get_best_model(self, repository, sample_model) -> None:
        """Тест получения лучшей модели."""
        await repository.save_model(sample_model)
        best_model = await repository.get_best_model("linear_regression")
        assert best_model is not None
        assert best_model.id == sample_model.id
class TestPortfolioRepository:
    """Тесты для репозитория портфеля."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryPortfolioRepository()
    @pytest.fixture
    def sample_portfolio(self: "TestEvolvableMarketMakerAgent") -> Any:
        return Portfolio(
            id="test-portfolio",
            name="Test Portfolio",
            status=PortfolioStatus.ACTIVE,
            risk_profile=RiskLevel.MEDIUM,
            max_leverage=Decimal("1"),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
    @pytest.fixture
    def sample_position(self: "TestEvolvableMarketMakerAgent") -> Any:
        from domain.type_definitions import Symbol
        return PortfolioPosition(
            id="test-position",
            portfolio_id="test-portfolio",
            trading_pair=TradingPair(
                Symbol("BTC/USDT"),
                Currency.BTC,
                Currency.USDT
            ),
            side="long",
            volume=Volume(Decimal("0.1"), Currency.BTC),
            entry_price=Price(Decimal("50000"), Currency.USDT),
            current_price=Price(Decimal("51000"), Currency.USDT),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
    @pytest.mark.asyncio
    async def test_save_portfolio(self, repository, sample_portfolio) -> None:
        """Тест сохранения портфеля."""
        result = await repository.save_portfolio(sample_portfolio)
        assert result is True
        saved_portfolio = await repository.get_portfolio(sample_portfolio.id)
        assert saved_portfolio is not None
        assert saved_portfolio.id == sample_portfolio.id
    @pytest.mark.asyncio
    async def test_save_position(self, repository, sample_position) -> None:
        """Тест сохранения позиции."""
        result = await repository.save_position(sample_position)
        assert result is True
        saved_position = await repository.get_position(sample_position.id)
        assert saved_position is not None
        assert saved_position.id == sample_position.id
    @pytest.mark.asyncio
    async def test_get_positions_by_symbol(self, repository, sample_position) -> None:
        """Тест получения позиций по символу."""
        await repository.save_position(sample_position)
        positions = await repository.get_positions_by_symbol("BTCUSDT")
        assert len(positions) == 1
        assert positions[0].id == sample_position.id
    @pytest.mark.asyncio
    async def test_update_position(self, repository, sample_position) -> None:
        """Тест обновления позиции."""
        await repository.save_position(sample_position)
        # Обновляем цену
        sample_position.current_price = Price(Decimal("52000"), Currency.USDT)
        result = await repository.update_position(sample_position)
        assert result is True
        updated_position = await repository.get_position(sample_position.id)
        assert updated_position.current_price.to_decimal() == Decimal("52000")
class TestRiskRepository:
    """Тесты для репозитория рисков."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryRiskRepository()
    @pytest.fixture
    def sample_risk_profile(self: "TestEvolvableMarketMakerAgent") -> Any:
        return RiskProfile(
            name="Test Profile",
            risk_level=RiskLevel.MEDIUM,
            max_risk_per_trade=Decimal("0.02"),
            max_daily_loss=Decimal("0.05"),
            max_leverage=Decimal("3")
        )
    @pytest.fixture
    def sample_risk_manager(self: "TestEvolvableMarketMakerAgent") -> Any:
        risk_profile = RiskProfile(
            name="Test Profile",
            risk_level=RiskLevel.MEDIUM,
            max_drawdown=Decimal("0.20")
        )
        return RiskManager(risk_profile)
    @pytest.mark.asyncio
    async def test_save_risk_profile(self, repository, sample_risk_profile) -> None:
        """Тест сохранения профиля риска."""
        result = await repository.save_risk_profile(sample_risk_profile)
        assert result is True
        saved_profile = await repository.get_risk_profile(sample_risk_profile.id)
        assert saved_profile is not None
        assert saved_profile.id == sample_risk_profile.id
    @pytest.mark.asyncio
    async def test_save_risk_manager(self, repository, sample_risk_manager) -> None:
        """Тест сохранения менеджера рисков."""
        result = await repository.save_risk_manager(sample_risk_manager)
        assert result is True
        saved_manager = await repository.get_risk_manager(sample_risk_manager.risk_profile.name)
        assert saved_manager is not None
        assert saved_manager.risk_profile.name == sample_risk_manager.risk_profile.name
class TestMarketRepository:
    """Тесты для репозитория рыночных данных."""
    @pytest.fixture
    def repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        return InMemoryMarketRepository()
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        return MarketData(
            id=uuid4(),
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open=Price(Decimal("50000"), Currency.USDT),
            high=Price(Decimal("51000"), Currency.USDT),
            low=Price(Decimal("49000"), Currency.USDT),
            close=Price(Decimal("50500"), Currency.USDT),
            volume=Volume(Decimal("100"), Currency.BTC)
        )
    @pytest.fixture
    def sample_market_regime(self: "TestEvolvableMarketMakerAgent") -> Any:
        return MarketRegime.TRENDING_UP
    @pytest.mark.asyncio
    async def test_save_market_data(self, repository, sample_market_data) -> None:
        """Тест сохранения рыночных данных."""
        result = await repository.save_market_data(sample_market_data)
        assert result is not None
        assert result.id == sample_market_data.id
    @pytest.mark.asyncio
    async def test_save_market_regime(self, repository, sample_market_regime) -> None:
        """Тест сохранения рыночного режима."""
        result = await repository.save_market_regime("BTCUSDT", sample_market_regime)
        assert result is True
        saved_regime = await repository.get_market_regime("BTCUSDT")
        assert saved_regime is not None
        assert saved_regime == sample_market_regime
    @pytest.mark.asyncio
    async def test_get_market_data_by_time_range(self, repository, sample_market_data) -> None:
        """Тест получения рыночных данных по временному диапазону."""
        await repository.save_market_data(sample_market_data)
        start_time = sample_market_data.timestamp
        end_time = datetime.now()
        data = await repository.get_market_data_by_time_range(
            sample_market_data.symbol, start_time, end_time
        )
        assert len(data) == 1
        assert data[0].id == sample_market_data.id
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
