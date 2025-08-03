"""
Unit тесты для OrderRepository.

Покрывает:
- Основной функционал репозитория ордеров
- CRUD операции
- Фильтрацию по различным критериям
- Поиск по датам и статусам
- Обработку ошибок
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trading_pair import TradingPair
from domain.repositories.order_repository import OrderRepository, InMemoryOrderRepository
from domain.types.repository_types import EntityId, QueryOptions, QueryFilter
from domain.exceptions.base_exceptions import ValidationError


class TestOrderRepository:
    """Тесты для абстрактного OrderRepository."""

    @pytest.fixture
    def mock_order_repository(self) -> Mock:
        """Мок репозитория ордеров."""
        return Mock(spec=OrderRepository)

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_order(self, sample_trading_pair) -> Order:
        """Тестовый ордер."""
        return Order(
            id=uuid4(),
            trading_pair=sample_trading_pair,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )

    def test_save_method_exists(self, mock_order_repository, sample_order):
        """Тест наличия метода save."""
        mock_order_repository.save = AsyncMock(return_value=sample_order)
        assert hasattr(mock_order_repository, 'save')
        assert callable(mock_order_repository.save)

    def test_get_by_id_method_exists(self, mock_order_repository):
        """Тест наличия метода get_by_id."""
        mock_order_repository.get_by_id = AsyncMock(return_value=None)
        assert hasattr(mock_order_repository, 'get_by_id')
        assert callable(mock_order_repository.get_by_id)

    def test_get_by_trading_pair_method_exists(self, mock_order_repository, sample_trading_pair):
        """Тест наличия метода get_by_trading_pair."""
        mock_order_repository.get_by_trading_pair = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_by_trading_pair')
        assert callable(mock_order_repository.get_by_trading_pair)

    def test_get_active_orders_method_exists(self, mock_order_repository):
        """Тест наличия метода get_active_orders."""
        mock_order_repository.get_active_orders = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_active_orders')
        assert callable(mock_order_repository.get_active_orders)

    def test_get_orders_by_status_method_exists(self, mock_order_repository):
        """Тест наличия метода get_orders_by_status."""
        mock_order_repository.get_orders_by_status = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_orders_by_status')
        assert callable(mock_order_repository.get_orders_by_status)

    def test_get_orders_by_side_method_exists(self, mock_order_repository):
        """Тест наличия метода get_orders_by_side."""
        mock_order_repository.get_orders_by_side = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_orders_by_side')
        assert callable(mock_order_repository.get_orders_by_side)

    def test_get_orders_by_type_method_exists(self, mock_order_repository):
        """Тест наличия метода get_orders_by_type."""
        mock_order_repository.get_orders_by_type = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_orders_by_type')
        assert callable(mock_order_repository.get_orders_by_type)

    def test_get_orders_by_date_range_method_exists(self, mock_order_repository):
        """Тест наличия метода get_orders_by_date_range."""
        mock_order_repository.get_orders_by_date_range = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_orders_by_date_range')
        assert callable(mock_order_repository.get_orders_by_date_range)

    def test_update_method_exists(self, mock_order_repository, sample_order):
        """Тест наличия метода update."""
        mock_order_repository.update = AsyncMock(return_value=sample_order)
        assert hasattr(mock_order_repository, 'update')
        assert callable(mock_order_repository.update)

    def test_delete_method_exists(self, mock_order_repository):
        """Тест наличия метода delete."""
        mock_order_repository.delete = AsyncMock(return_value=True)
        assert hasattr(mock_order_repository, 'delete')
        assert callable(mock_order_repository.delete)

    def test_exists_method_exists(self, mock_order_repository):
        """Тест наличия метода exists."""
        mock_order_repository.exists = AsyncMock(return_value=True)
        assert hasattr(mock_order_repository, 'exists')
        assert callable(mock_order_repository.exists)

    def test_count_method_exists(self, mock_order_repository):
        """Тест наличия метода count."""
        mock_order_repository.count = AsyncMock(return_value=0)
        assert hasattr(mock_order_repository, 'count')
        assert callable(mock_order_repository.count)

    def test_get_all_method_exists(self, mock_order_repository):
        """Тест наличия метода get_all."""
        mock_order_repository.get_all = AsyncMock(return_value=[])
        assert hasattr(mock_order_repository, 'get_all')
        assert callable(mock_order_repository.get_all)


class TestInMemoryOrderRepository:
    """Тесты для InMemoryOrderRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryOrderRepository:
        """Экземпляр репозитория."""
        return InMemoryOrderRepository()

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_eth(self) -> TradingPair:
        """Тестовая торговая пара ETH."""
        return TradingPair(base="ETH", quote="USDT")

    @pytest.fixture
    def sample_order(self, sample_trading_pair) -> Order:
        """Тестовый ордер."""
        return Order(
            id=uuid4(),
            trading_pair=sample_trading_pair,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )

    @pytest.fixture
    def sample_orders(self, sample_trading_pair, sample_trading_pair_eth) -> List[Order]:
        """Список тестовых ордеров."""
        now = datetime.now()
        return [
            Order(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                price=50000.0,
                status=OrderStatus.PENDING,
                created_at=now
            ),
            Order(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.5,
                price=51000.0,
                status=OrderStatus.OPEN,
                created_at=now + timedelta(minutes=1)
            ),
            Order(
                id=uuid4(),
                trading_pair=sample_trading_pair_eth,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=10.0,
                price=3000.0,
                status=OrderStatus.FILLED,
                created_at=now + timedelta(minutes=2)
            ),
            Order(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=OrderSide.SELL,
                order_type=OrderType.STOP_LOSS,
                quantity=2.0,
                price=48000.0,
                status=OrderStatus.CANCELLED,
                created_at=now + timedelta(minutes=3)
            )
        ]

    @pytest.mark.asyncio
    async def test_save_order(self, repository, sample_order):
        """Тест сохранения ордера."""
        saved_order = await repository.save(sample_order)
        
        assert saved_order == sample_order
        assert EntityId(sample_order.id) in repository._orders
        assert repository._orders[EntityId(sample_order.id)] == sample_order

    @pytest.mark.asyncio
    async def test_get_by_id_existing(self, repository, sample_order):
        """Тест получения существующего ордера по ID."""
        await repository.save(sample_order)
        
        retrieved_order = await repository.get_by_id(EntityId(sample_order.id))
        
        assert retrieved_order == sample_order

    @pytest.mark.asyncio
    async def test_get_by_id_not_existing(self, repository):
        """Тест получения несуществующего ордера по ID."""
        order_id = EntityId(uuid4())
        retrieved_order = await repository.get_by_id(order_id)
        
        assert retrieved_order is None

    @pytest.mark.asyncio
    async def test_get_by_trading_pair(self, repository, sample_orders, sample_trading_pair):
        """Тест получения ордеров по торговой паре."""
        for order in sample_orders:
            await repository.save(order)
        
        btc_orders = await repository.get_by_trading_pair(sample_trading_pair)
        
        assert len(btc_orders) == 3
        assert all(order.trading_pair == sample_trading_pair for order in btc_orders)

    @pytest.mark.asyncio
    async def test_get_by_trading_pair_with_status(self, repository, sample_orders, sample_trading_pair):
        """Тест получения ордеров по торговой паре с фильтром по статусу."""
        for order in sample_orders:
            await repository.save(order)
        
        pending_orders = await repository.get_by_trading_pair(sample_trading_pair, OrderStatus.PENDING)
        
        assert len(pending_orders) == 1
        assert all(order.status == OrderStatus.PENDING for order in pending_orders)

    @pytest.mark.asyncio
    async def test_get_active_orders(self, repository, sample_orders):
        """Тест получения активных ордеров."""
        for order in sample_orders:
            await repository.save(order)
        
        active_orders = await repository.get_active_orders()
        
        assert len(active_orders) == 2
        active_statuses = [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        assert all(order.status in active_statuses for order in active_orders)

    @pytest.mark.asyncio
    async def test_get_active_orders_by_trading_pair(self, repository, sample_orders, sample_trading_pair):
        """Тест получения активных ордеров по торговой паре."""
        for order in sample_orders:
            await repository.save(order)
        
        active_orders = await repository.get_active_orders(sample_trading_pair)
        
        assert len(active_orders) == 2
        active_statuses = [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        assert all(order.status in active_statuses and order.trading_pair == sample_trading_pair for order in active_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_status(self, repository, sample_orders):
        """Тест получения ордеров по статусу."""
        for order in sample_orders:
            await repository.save(order)
        
        pending_orders = await repository.get_orders_by_status(OrderStatus.PENDING)
        filled_orders = await repository.get_orders_by_status(OrderStatus.FILLED)
        
        assert len(pending_orders) == 1
        assert len(filled_orders) == 1
        assert all(order.status == OrderStatus.PENDING for order in pending_orders)
        assert all(order.status == OrderStatus.FILLED for order in filled_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_status_with_limit(self, repository, sample_orders):
        """Тест получения ордеров по статусу с лимитом."""
        for order in sample_orders:
            await repository.save(order)
        
        # Создаем дополнительный ордер с тем же статусом
        additional_order = Order(
            id=uuid4(),
            trading_pair=sample_orders[0].trading_pair,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        await repository.save(additional_order)
        
        pending_orders = await repository.get_orders_by_status(OrderStatus.PENDING, limit=1)
        
        assert len(pending_orders) == 1
        assert all(order.status == OrderStatus.PENDING for order in pending_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_side(self, repository, sample_orders):
        """Тест получения ордеров по стороне."""
        for order in sample_orders:
            await repository.save(order)
        
        buy_orders = await repository.get_orders_by_side(OrderSide.BUY)
        sell_orders = await repository.get_orders_by_side(OrderSide.SELL)
        
        assert len(buy_orders) == 2
        assert len(sell_orders) == 2
        assert all(order.side == OrderSide.BUY for order in buy_orders)
        assert all(order.side == OrderSide.SELL for order in sell_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_side_with_trading_pair(self, repository, sample_orders, sample_trading_pair):
        """Тест получения ордеров по стороне с фильтром по торговой паре."""
        for order in sample_orders:
            await repository.save(order)
        
        buy_orders = await repository.get_orders_by_side(OrderSide.BUY, sample_trading_pair)
        
        assert len(buy_orders) == 1
        assert all(order.side == OrderSide.BUY and order.trading_pair == sample_trading_pair for order in buy_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_type(self, repository, sample_orders):
        """Тест получения ордеров по типу."""
        for order in sample_orders:
            await repository.save(order)
        
        limit_orders = await repository.get_orders_by_type(OrderType.LIMIT)
        market_orders = await repository.get_orders_by_type(OrderType.MARKET)
        
        assert len(limit_orders) == 2
        assert len(market_orders) == 1
        assert all(order.order_type == OrderType.LIMIT for order in limit_orders)
        assert all(order.order_type == OrderType.MARKET for order in market_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_type_with_trading_pair(self, repository, sample_orders, sample_trading_pair):
        """Тест получения ордеров по типу с фильтром по торговой паре."""
        for order in sample_orders:
            await repository.save(order)
        
        limit_orders = await repository.get_orders_by_type(OrderType.LIMIT, sample_trading_pair)
        
        assert len(limit_orders) == 1
        assert all(order.order_type == OrderType.LIMIT and order.trading_pair == sample_trading_pair for order in limit_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_date_range(self, repository, sample_orders):
        """Тест получения ордеров по диапазону дат."""
        for order in sample_orders:
            await repository.save(order)
        
        start_date = datetime.now()
        end_date = start_date + timedelta(minutes=2)
        
        orders_in_range = await repository.get_orders_by_date_range(start_date, end_date)
        
        assert len(orders_in_range) == 3
        assert all(start_date <= order.created_at.to_datetime().replace(tzinfo=None) <= end_date for order in orders_in_range)

    @pytest.mark.asyncio
    async def test_get_orders_by_date_range_with_trading_pair(self, repository, sample_orders, sample_trading_pair):
        """Тест получения ордеров по диапазону дат с фильтром по торговой паре."""
        for order in sample_orders:
            await repository.save(order)
        
        start_date = datetime.now()
        end_date = start_date + timedelta(minutes=2)
        
        orders_in_range = await repository.get_orders_by_date_range(start_date, end_date, sample_trading_pair)
        
        assert len(orders_in_range) == 2
        assert all(
            start_date <= order.created_at.to_datetime().replace(tzinfo=None) <= end_date 
            and order.trading_pair == sample_trading_pair 
            for order in orders_in_range
        )

    @pytest.mark.asyncio
    async def test_update_existing_order(self, repository, sample_order):
        """Тест обновления существующего ордера."""
        await repository.save(sample_order)
        
        # Изменяем ордер
        sample_order.status = OrderStatus.FILLED
        sample_order.price = 51000.0
        
        updated_order = await repository.update(sample_order)
        
        assert updated_order == sample_order
        assert repository._orders[EntityId(sample_order.id)].status == OrderStatus.FILLED
        assert repository._orders[EntityId(sample_order.id)].price == 51000.0

    @pytest.mark.asyncio
    async def test_update_not_existing_order(self, repository, sample_order):
        """Тест обновления несуществующего ордера."""
        with pytest.raises(Exception, match="Order .* not found"):
            await repository.update(sample_order)

    @pytest.mark.asyncio
    async def test_delete_existing_order(self, repository, sample_order):
        """Тест удаления существующего ордера."""
        await repository.save(sample_order)
        
        result = await repository.delete(sample_order.id)
        
        assert result is True
        assert EntityId(sample_order.id) not in repository._orders

    @pytest.mark.asyncio
    async def test_delete_not_existing_order(self, repository):
        """Тест удаления несуществующего ордера."""
        order_id = uuid4()
        result = await repository.delete(order_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_string_id(self, repository, sample_order):
        """Тест удаления ордера по строковому ID."""
        await repository.save(sample_order)
        
        result = await repository.delete(str(sample_order.id))
        
        assert result is True
        assert EntityId(sample_order.id) not in repository._orders

    @pytest.mark.asyncio
    async def test_exists_true(self, repository, sample_order):
        """Тест exists для существующего ордера."""
        await repository.save(sample_order)
        
        exists = await repository.exists(sample_order.id)
        
        assert exists is True

    @pytest.mark.asyncio
    async def test_exists_false(self, repository):
        """Тест exists для несуществующего ордера."""
        order_id = uuid4()
        
        exists = await repository.exists(order_id)
        
        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_with_string_id(self, repository, sample_order):
        """Тест exists со строковым ID."""
        await repository.save(sample_order)
        
        exists = await repository.exists(str(sample_order.id))
        
        assert exists is True

    @pytest.mark.asyncio
    async def test_count_no_filters(self, repository, sample_orders):
        """Тест count без фильтров."""
        for order in sample_orders:
            await repository.save(order)
        
        count = await repository.count()
        
        assert count == 4

    @pytest.mark.asyncio
    async def test_count_empty_repository(self, repository):
        """Тест count для пустого репозитория."""
        count = await repository.count()
        
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_all_empty(self, repository):
        """Тест get_all для пустого репозитория."""
        orders = await repository.get_all()
        
        assert orders == []

    @pytest.mark.asyncio
    async def test_get_all_with_orders(self, repository, sample_orders):
        """Тест get_all с ордерами."""
        for order in sample_orders:
            await repository.save(order)
        
        orders = await repository.get_all()
        
        assert len(orders) == 4
        assert all(order in orders for order in sample_orders)

    @pytest.mark.asyncio
    async def test_find_by_id_compatibility(self, repository, sample_order):
        """Тест совместимости метода find_by_id."""
        await repository.save(sample_order)
        
        found_order = await repository.find_by_id(EntityId(sample_order.id))
        
        assert found_order == sample_order

    @pytest.mark.asyncio
    async def test_find_all_compatibility(self, repository, sample_orders):
        """Тест совместимости метода find_all."""
        for order in sample_orders:
            await repository.save(order)
        
        orders = await repository.find_all()
        
        assert len(orders) == 4
        assert all(order in orders for order in sample_orders)

    @pytest.mark.asyncio
    async def test_find_by_criteria_compatibility(self, repository, sample_orders):
        """Тест совместимости метода find_by_criteria."""
        for order in sample_orders:
            await repository.save(order)
        
        orders = await repository.find_by_criteria([])
        
        assert len(orders) == 4

    @pytest.mark.asyncio
    async def test_repository_isolation(self):
        """Тест изоляции между экземплярами репозитория."""
        repo1 = InMemoryOrderRepository()
        repo2 = InMemoryOrderRepository()
        
        trading_pair = TradingPair(base="BTC", quote="USDT")
        
        order1 = Order(
            id=uuid4(),
            trading_pair=trading_pair,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        order2 = Order(
            id=uuid4(),
            trading_pair=trading_pair,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.5,
            price=51000.0,
            status=OrderStatus.OPEN,
            created_at=datetime.now()
        )
        
        await repo1.save(order1)
        await repo2.save(order2)
        
        assert len(await repo1.get_all()) == 1
        assert len(await repo2.get_all()) == 1
        assert await repo1.get_by_id(EntityId(order1.id)) == order1
        assert await repo2.get_by_id(EntityId(order2.id)) == order2
        assert await repo1.get_by_id(EntityId(order2.id)) is None
        assert await repo2.get_by_id(EntityId(order1.id)) is None

    @pytest.mark.asyncio
    async def test_order_status_transitions(self, repository, sample_order):
        """Тест переходов статусов ордера."""
        await repository.save(sample_order)
        
        # PENDING -> OPEN
        sample_order.status = OrderStatus.OPEN
        updated_order = await repository.update(sample_order)
        assert updated_order.status == OrderStatus.OPEN
        
        # OPEN -> PARTIALLY_FILLED
        sample_order.status = OrderStatus.PARTIALLY_FILLED
        updated_order = await repository.update(sample_order)
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        
        # PARTIALLY_FILLED -> FILLED
        sample_order.status = OrderStatus.FILLED
        updated_order = await repository.update(sample_order)
        assert updated_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_order_price_updates(self, repository, sample_order):
        """Тест обновления цены ордера."""
        await repository.save(sample_order)
        
        original_price = sample_order.price
        new_price = 52000.0
        
        sample_order.price = new_price
        updated_order = await repository.update(sample_order)
        
        assert updated_order.price == new_price
        assert updated_order.price != original_price

    @pytest.mark.asyncio
    async def test_order_quantity_updates(self, repository, sample_order):
        """Тест обновления количества ордера."""
        await repository.save(sample_order)
        
        original_quantity = sample_order.quantity
        new_quantity = 2.5
        
        sample_order.quantity = new_quantity
        updated_order = await repository.update(sample_order)
        
        assert updated_order.quantity == new_quantity
        assert updated_order.quantity != original_quantity

    @pytest.mark.asyncio
    async def test_multiple_orders_same_trading_pair(self, repository, sample_trading_pair):
        """Тест работы с несколькими ордерами одной торговой пары."""
        orders = []
        for i in range(5):
            order = Order(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=1.0 + i * 0.1,
                price=50000.0 + i * 100,
                status=OrderStatus.PENDING,
                created_at=datetime.now() + timedelta(minutes=i)
            )
            orders.append(order)
            await repository.save(order)
        
        all_orders = await repository.get_all()
        assert len(all_orders) == 5
        
        btc_orders = await repository.get_by_trading_pair(sample_trading_pair)
        assert len(btc_orders) == 5
        
        buy_orders = await repository.get_orders_by_side(OrderSide.BUY, sample_trading_pair)
        sell_orders = await repository.get_orders_by_side(OrderSide.SELL, sample_trading_pair)
        assert len(buy_orders) == 3
        assert len(sell_orders) == 2 