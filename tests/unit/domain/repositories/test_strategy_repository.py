"""
Unit тесты для StrategyRepository.

Покрывает:
- Основной функционал репозитория
- CRUD операции
- Фильтрацию и поиск
- Обработку метрик и сигналов
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.repositories.strategy_repository import StrategyRepository, InMemoryStrategyRepository
from domain.types.repository_types import EntityId, QueryOptions, QueryFilter
from domain.exceptions.base_exceptions import ValidationError


class TestStrategyRepository:
    """Тесты для абстрактного StrategyRepository."""

    @pytest.fixture
    def mock_strategy_repository(self) -> Mock:
        """Мок репозитория стратегий."""
        return Mock(spec=StrategyRepository)

    @pytest.fixture
    def sample_strategy(self) -> Strategy:
        """Тестовая стратегия."""
        return Strategy(
            id=uuid4(),
            name="Test Strategy",
            strategy_type=StrategyType.ARBITRAGE,
            status=StrategyStatus.ACTIVE,
            trading_pair="BTC/USDT",
            parameters={"param1": "value1"},
            risk_limits={"max_loss": 0.1}
        )

    @pytest.fixture
    def sample_strategies(self) -> List[Strategy]:
        """Список тестовых стратегий."""
        return [
            Strategy(
                id=uuid4(),
                name="Strategy 1",
                strategy_type=StrategyType.ARBITRAGE,
                status=StrategyStatus.ACTIVE,
                trading_pair="BTC/USDT"
            ),
            Strategy(
                id=uuid4(),
                name="Strategy 2",
                strategy_type=StrategyType.MARKET_MAKING,
                status=StrategyStatus.PAUSED,
                trading_pair="ETH/USDT"
            ),
            Strategy(
                id=uuid4(),
                name="Strategy 3",
                strategy_type=StrategyType.ARBITRAGE,
                status=StrategyStatus.ACTIVE,
                trading_pair="BTC/USDT"
            )
        ]

    def test_save_method_exists(self, mock_strategy_repository, sample_strategy):
        """Тест наличия метода save."""
        mock_strategy_repository.save = AsyncMock(return_value=sample_strategy)
        assert hasattr(mock_strategy_repository, 'save')
        assert callable(mock_strategy_repository.save)

    def test_get_by_id_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_by_id."""
        mock_strategy_repository.get_by_id = AsyncMock(return_value=None)
        assert hasattr(mock_strategy_repository, 'get_by_id')
        assert callable(mock_strategy_repository.get_by_id)

    def test_get_all_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_all."""
        mock_strategy_repository.get_all = AsyncMock(return_value=[])
        assert hasattr(mock_strategy_repository, 'get_all')
        assert callable(mock_strategy_repository.get_all)

    def test_delete_method_exists(self, mock_strategy_repository):
        """Тест наличия метода delete."""
        mock_strategy_repository.delete = AsyncMock(return_value=True)
        assert hasattr(mock_strategy_repository, 'delete')
        assert callable(mock_strategy_repository.delete)

    def test_get_by_type_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_by_type."""
        mock_strategy_repository.get_by_type = AsyncMock(return_value=[])
        assert hasattr(mock_strategy_repository, 'get_by_type')
        assert callable(mock_strategy_repository.get_by_type)

    def test_get_by_status_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_by_status."""
        mock_strategy_repository.get_by_status = AsyncMock(return_value=[])
        assert hasattr(mock_strategy_repository, 'get_by_status')
        assert callable(mock_strategy_repository.get_by_status)

    def test_get_by_trading_pair_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_by_trading_pair."""
        mock_strategy_repository.get_by_trading_pair = AsyncMock(return_value=[])
        assert hasattr(mock_strategy_repository, 'get_by_trading_pair')
        assert callable(mock_strategy_repository.get_by_trading_pair)

    def test_update_metrics_method_exists(self, mock_strategy_repository):
        """Тест наличия метода update_metrics."""
        mock_strategy_repository.update_metrics = AsyncMock()
        assert hasattr(mock_strategy_repository, 'update_metrics')
        assert callable(mock_strategy_repository.update_metrics)

    def test_add_signal_method_exists(self, mock_strategy_repository):
        """Тест наличия метода add_signal."""
        mock_strategy_repository.add_signal = AsyncMock()
        assert hasattr(mock_strategy_repository, 'add_signal')
        assert callable(mock_strategy_repository.add_signal)

    def test_get_latest_signal_method_exists(self, mock_strategy_repository):
        """Тест наличия метода get_latest_signal."""
        mock_strategy_repository.get_latest_signal = AsyncMock(return_value=None)
        assert hasattr(mock_strategy_repository, 'get_latest_signal')
        assert callable(mock_strategy_repository.get_latest_signal)


class TestInMemoryStrategyRepository:
    """Тесты для InMemoryStrategyRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryStrategyRepository:
        """Экземпляр репозитория."""
        return InMemoryStrategyRepository()

    @pytest.fixture
    def sample_strategy(self) -> Strategy:
        """Тестовая стратегия."""
        return Strategy(
            id=uuid4(),
            name="Test Strategy",
            strategy_type=StrategyType.ARBITRAGE,
            status=StrategyStatus.ACTIVE,
            trading_pair="BTC/USDT",
            parameters={"param1": "value1"},
            risk_limits={"max_loss": 0.1}
        )

    @pytest.fixture
    def sample_strategies(self) -> List[Strategy]:
        """Список тестовых стратегий."""
        return [
            Strategy(
                id=uuid4(),
                name="Strategy 1",
                strategy_type=StrategyType.ARBITRAGE,
                status=StrategyStatus.ACTIVE,
                trading_pair="BTC/USDT"
            ),
            Strategy(
                id=uuid4(),
                name="Strategy 2",
                strategy_type=StrategyType.MARKET_MAKING,
                status=StrategyStatus.PAUSED,
                trading_pair="ETH/USDT"
            ),
            Strategy(
                id=uuid4(),
                name="Strategy 3",
                strategy_type=StrategyType.ARBITRAGE,
                status=StrategyStatus.ACTIVE,
                trading_pair="BTC/USDT"
            )
        ]

    @pytest.mark.asyncio
    async def test_save_strategy(self, repository, sample_strategy):
        """Тест сохранения стратегии."""
        saved_strategy = await repository.save(sample_strategy)
        
        assert saved_strategy == sample_strategy
        assert EntityId(sample_strategy.id) in repository._strategies
        assert repository._strategies[EntityId(sample_strategy.id)] == sample_strategy

    @pytest.mark.asyncio
    async def test_get_by_id_existing(self, repository, sample_strategy):
        """Тест получения существующей стратегии по ID."""
        await repository.save(sample_strategy)
        
        retrieved_strategy = await repository.get_by_id(EntityId(sample_strategy.id))
        
        assert retrieved_strategy == sample_strategy

    @pytest.mark.asyncio
    async def test_get_by_id_not_existing(self, repository):
        """Тест получения несуществующей стратегии по ID."""
        strategy_id = EntityId(uuid4())
        retrieved_strategy = await repository.get_by_id(strategy_id)
        
        assert retrieved_strategy is None

    @pytest.mark.asyncio
    async def test_get_all_empty(self, repository):
        """Тест получения всех стратегий из пустого репозитория."""
        strategies = await repository.get_all()
        
        assert strategies == []

    @pytest.mark.asyncio
    async def test_get_all_with_strategies(self, repository, sample_strategies):
        """Тест получения всех стратегий."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.get_all()
        
        assert len(strategies) == 3
        assert all(strategy in strategies for strategy in sample_strategies)

    @pytest.mark.asyncio
    async def test_delete_existing_strategy(self, repository, sample_strategy):
        """Тест удаления существующей стратегии."""
        await repository.save(sample_strategy)
        
        result = await repository.delete(sample_strategy.id)
        
        assert result is True
        assert EntityId(sample_strategy.id) not in repository._strategies

    @pytest.mark.asyncio
    async def test_delete_not_existing_strategy(self, repository):
        """Тест удаления несуществующей стратегии."""
        strategy_id = uuid4()
        result = await repository.delete(strategy_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_string_id(self, repository, sample_strategy):
        """Тест удаления стратегии по строковому ID."""
        await repository.save(sample_strategy)
        
        result = await repository.delete(str(sample_strategy.id))
        
        assert result is True
        assert EntityId(sample_strategy.id) not in repository._strategies

    @pytest.mark.asyncio
    async def test_get_by_type(self, repository, sample_strategies):
        """Тест получения стратегий по типу."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        arbitrage_strategies = await repository.get_by_type(StrategyType.ARBITRAGE)
        market_making_strategies = await repository.get_by_type(StrategyType.MARKET_MAKING)
        
        assert len(arbitrage_strategies) == 2
        assert len(market_making_strategies) == 1
        assert all(s.strategy_type == StrategyType.ARBITRAGE for s in arbitrage_strategies)
        assert all(s.strategy_type == StrategyType.MARKET_MAKING for s in market_making_strategies)

    @pytest.mark.asyncio
    async def test_get_by_status(self, repository, sample_strategies):
        """Тест получения стратегий по статусу."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        active_strategies = await repository.get_by_status(StrategyStatus.ACTIVE)
        paused_strategies = await repository.get_by_status(StrategyStatus.PAUSED)
        
        assert len(active_strategies) == 2
        assert len(paused_strategies) == 1
        assert all(s.status == StrategyStatus.ACTIVE for s in active_strategies)
        assert all(s.status == StrategyStatus.PAUSED for s in paused_strategies)

    @pytest.mark.asyncio
    async def test_get_by_trading_pair(self, repository, sample_strategies):
        """Тест получения стратегий по торговой паре."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        btc_strategies = await repository.get_by_trading_pair("BTC/USDT")
        eth_strategies = await repository.get_by_trading_pair("ETH/USDT")
        
        assert len(btc_strategies) == 2
        assert len(eth_strategies) == 1
        assert all(getattr(s, "trading_pair", None) == "BTC/USDT" for s in btc_strategies)
        assert all(getattr(s, "trading_pair", None) == "ETH/USDT" for s in eth_strategies)

    @pytest.mark.asyncio
    async def test_update_metrics_existing_strategy(self, repository, sample_strategy):
        """Тест обновления метрик существующей стратегии."""
        await repository.save(sample_strategy)
        metrics = {"profit": 100.0, "trades_count": 50}
        
        updated_strategy = await repository.update_metrics(EntityId(sample_strategy.id), metrics)
        
        assert updated_strategy == sample_strategy
        assert repository._strategy_metrics[EntityId(sample_strategy.id)] == metrics

    @pytest.mark.asyncio
    async def test_update_metrics_not_existing_strategy(self, repository):
        """Тест обновления метрик несуществующей стратегии."""
        strategy_id = EntityId(uuid4())
        metrics = {"profit": 100.0}
        
        with pytest.raises(ValueError, match="Strategy not found"):
            await repository.update_metrics(strategy_id, metrics)

    @pytest.mark.asyncio
    async def test_add_signal_existing_strategy(self, repository, sample_strategy):
        """Тест добавления сигнала к существующей стратегии."""
        await repository.save(sample_strategy)
        signal = {"type": "buy", "price": 50000.0, "timestamp": "2023-01-01T00:00:00Z"}
        
        updated_strategy = await repository.add_signal(EntityId(sample_strategy.id), signal)
        
        assert updated_strategy == sample_strategy
        assert EntityId(sample_strategy.id) in repository._signals
        assert signal in repository._signals[EntityId(sample_strategy.id)]

    @pytest.mark.asyncio
    async def test_add_signal_not_existing_strategy(self, repository):
        """Тест добавления сигнала к несуществующей стратегии."""
        strategy_id = EntityId(uuid4())
        signal = {"type": "buy", "price": 50000.0}
        
        with pytest.raises(ValueError, match="Strategy not found"):
            await repository.add_signal(strategy_id, signal)

    @pytest.mark.asyncio
    async def test_get_latest_signal_with_signals(self, repository, sample_strategy):
        """Тест получения последнего сигнала."""
        await repository.save(sample_strategy)
        signals = [
            {"type": "buy", "price": 50000.0},
            {"type": "sell", "price": 51000.0},
            {"type": "buy", "price": 52000.0}
        ]
        
        for signal in signals:
            await repository.add_signal(EntityId(sample_strategy.id), signal)
        
        latest_signal = await repository.get_latest_signal(EntityId(sample_strategy.id))
        
        assert latest_signal == signals[-1]

    @pytest.mark.asyncio
    async def test_get_latest_signal_no_signals(self, repository, sample_strategy):
        """Тест получения последнего сигнала при отсутствии сигналов."""
        await repository.save(sample_strategy)
        
        latest_signal = await repository.get_latest_signal(EntityId(sample_strategy.id))
        
        assert latest_signal is None

    @pytest.mark.asyncio
    async def test_get_latest_signal_not_existing_strategy(self, repository):
        """Тест получения последнего сигнала несуществующей стратегии."""
        strategy_id = EntityId(uuid4())
        
        latest_signal = await repository.get_latest_signal(strategy_id)
        
        assert latest_signal is None

    @pytest.mark.asyncio
    async def test_find_by_id_existing(self, repository, sample_strategy):
        """Тест find_by_id для существующей стратегии."""
        await repository.save(sample_strategy)
        
        found_strategy = await repository.find_by_id(EntityId(sample_strategy.id))
        
        assert found_strategy == sample_strategy

    @pytest.mark.asyncio
    async def test_find_by_id_not_existing(self, repository):
        """Тест find_by_id для несуществующей стратегии."""
        strategy_id = EntityId(uuid4())
        
        found_strategy = await repository.find_by_id(strategy_id)
        
        assert found_strategy is None

    @pytest.mark.asyncio
    async def test_find_all_with_limit(self, repository, sample_strategies):
        """Тест find_all с лимитом."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_all(limit=2)
        
        assert len(strategies) == 2

    @pytest.mark.asyncio
    async def test_find_all_with_offset(self, repository, sample_strategies):
        """Тест find_all со смещением."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_all(offset=1)
        
        assert len(strategies) == 2

    @pytest.mark.asyncio
    async def test_find_all_with_limit_and_offset(self, repository, sample_strategies):
        """Тест find_all с лимитом и смещением."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_all(limit=1, offset=1)
        
        assert len(strategies) == 1

    @pytest.mark.asyncio
    async def test_find_by_criteria_by_type(self, repository, sample_strategies):
        """Тест find_by_criteria по типу стратегии."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_by_criteria({"strategy_type": StrategyType.ARBITRAGE})
        
        assert len(strategies) == 2
        assert all(s.strategy_type == StrategyType.ARBITRAGE for s in strategies)

    @pytest.mark.asyncio
    async def test_find_by_criteria_by_status(self, repository, sample_strategies):
        """Тест find_by_criteria по статусу стратегии."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_by_criteria({"status": StrategyStatus.ACTIVE})
        
        assert len(strategies) == 2
        assert all(s.status == StrategyStatus.ACTIVE for s in strategies)

    @pytest.mark.asyncio
    async def test_find_by_criteria_by_trading_pair(self, repository, sample_strategies):
        """Тест find_by_criteria по торговой паре."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_by_criteria({"trading_pair": "BTC/USDT"})
        
        assert len(strategies) == 2
        assert all(getattr(s, "trading_pair", None) == "BTC/USDT" for s in strategies)

    @pytest.mark.asyncio
    async def test_find_by_criteria_with_limit_and_offset(self, repository, sample_strategies):
        """Тест find_by_criteria с лимитом и смещением."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_by_criteria(
            {"strategy_type": StrategyType.ARBITRAGE}, 
            limit=1, 
            offset=1
        )
        
        assert len(strategies) == 1

    @pytest.mark.asyncio
    async def test_find_by_criteria_no_matches(self, repository, sample_strategies):
        """Тест find_by_criteria без совпадений."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        strategies = await repository.find_by_criteria({"name": "NonExistent"})
        
        assert len(strategies) == 0

    @pytest.mark.asyncio
    async def test_update_strategy(self, repository, sample_strategy):
        """Тест обновления стратегии."""
        await repository.save(sample_strategy)
        
        # Изменяем стратегию
        sample_strategy.name = "Updated Strategy"
        updated_strategy = await repository.update(sample_strategy)
        
        assert updated_strategy == sample_strategy
        assert repository._strategies[EntityId(sample_strategy.id)].name == "Updated Strategy"

    @pytest.mark.asyncio
    async def test_exists_true(self, repository, sample_strategy):
        """Тест exists для существующей стратегии."""
        await repository.save(sample_strategy)
        
        exists = await repository.exists(sample_strategy.id)
        
        assert exists is True

    @pytest.mark.asyncio
    async def test_exists_false(self, repository):
        """Тест exists для несуществующей стратегии."""
        strategy_id = uuid4()
        
        exists = await repository.exists(strategy_id)
        
        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_with_string_id(self, repository, sample_strategy):
        """Тест exists со строковым ID."""
        await repository.save(sample_strategy)
        
        exists = await repository.exists(str(sample_strategy.id))
        
        assert exists is True

    @pytest.mark.asyncio
    async def test_count_no_filters(self, repository, sample_strategies):
        """Тест count без фильтров."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        count = await repository.count()
        
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_with_filters(self, repository, sample_strategies):
        """Тест count с фильтрами."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        filters = [
            QueryFilter(field="strategy_type", value=StrategyType.ARBITRAGE)
        ]
        
        count = await repository.count(filters)
        
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_with_multiple_filters(self, repository, sample_strategies):
        """Тест count с несколькими фильтрами."""
        for strategy in sample_strategies:
            await repository.save(strategy)
        
        filters = [
            QueryFilter(field="strategy_type", value=StrategyType.ARBITRAGE),
            QueryFilter(field="status", value=StrategyStatus.ACTIVE)
        ]
        
        count = await repository.count(filters)
        
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_empty_repository(self, repository):
        """Тест count для пустого репозитория."""
        count = await repository.count()
        
        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_signals_per_strategy(self, repository, sample_strategy):
        """Тест добавления нескольких сигналов к стратегии."""
        await repository.save(sample_strategy)
        signals = [
            {"type": "buy", "price": 50000.0, "timestamp": "2023-01-01T00:00:00Z"},
            {"type": "sell", "price": 51000.0, "timestamp": "2023-01-01T01:00:00Z"},
            {"type": "buy", "price": 52000.0, "timestamp": "2023-01-01T02:00:00Z"}
        ]
        
        for signal in signals:
            await repository.add_signal(EntityId(sample_strategy.id), signal)
        
        strategy_signals = repository._signals[EntityId(sample_strategy.id)]
        assert len(strategy_signals) == 3
        assert strategy_signals == signals

    @pytest.mark.asyncio
    async def test_metrics_persistence(self, repository, sample_strategy):
        """Тест сохранения метрик стратегии."""
        await repository.save(sample_strategy)
        initial_metrics = {"profit": 100.0, "trades_count": 50}
        updated_metrics = {"profit": 150.0, "trades_count": 75, "win_rate": 0.8}
        
        await repository.update_metrics(EntityId(sample_strategy.id), initial_metrics)
        await repository.update_metrics(EntityId(sample_strategy.id), updated_metrics)
        
        stored_metrics = repository._strategy_metrics[EntityId(sample_strategy.id)]
        assert stored_metrics == updated_metrics
        assert "profit" in stored_metrics
        assert "trades_count" in stored_metrics
        assert "win_rate" in stored_metrics

    @pytest.mark.asyncio
    async def test_repository_isolation(self):
        """Тест изоляции между экземплярами репозитория."""
        repo1 = InMemoryStrategyRepository()
        repo2 = InMemoryStrategyRepository()
        
        strategy1 = Strategy(
            id=uuid4(),
            name="Strategy 1",
            strategy_type=StrategyType.ARBITRAGE,
            status=StrategyStatus.ACTIVE,
            trading_pair="BTC/USDT"
        )
        
        strategy2 = Strategy(
            id=uuid4(),
            name="Strategy 2",
            strategy_type=StrategyType.MARKET_MAKING,
            status=StrategyStatus.PAUSED,
            trading_pair="ETH/USDT"
        )
        
        await repo1.save(strategy1)
        await repo2.save(strategy2)
        
        assert len(await repo1.get_all()) == 1
        assert len(await repo2.get_all()) == 1
        assert await repo1.get_by_id(EntityId(strategy1.id)) == strategy1
        assert await repo2.get_by_id(EntityId(strategy2.id)) == strategy2
        assert await repo1.get_by_id(EntityId(strategy2.id)) is None
        assert await repo2.get_by_id(EntityId(strategy1.id)) is None 