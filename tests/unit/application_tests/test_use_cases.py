"""
Тесты для use cases в application слое.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from decimal import Decimal
from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.use_cases.manage_positions import (
    PositionManagementUseCase, DefaultPositionManagementUseCase
)
from application.types import (
    CreateOrderRequest, CreateOrderResponse, CancelOrderRequest, CancelOrderResponse,
    GetOrdersRequest, GetOrdersResponse
)
from domain.value_objects import Price, Volume, Money
class TestOrderManagementUseCase:
    """Тесты для OrderManagementUseCase."""
    @pytest.fixture
    def mock_repositories(self) -> Any:
        """Создает mock репозитории."""
        order_repo = Mock()
        portfolio_repo = Mock()
        position_repo = Mock()
        # Настройка mock методов
        order_repo.create = AsyncMock()
        order_repo.get_by_id = AsyncMock()
        order_repo.update = AsyncMock()
        order_repo.get_by_portfolio_id = AsyncMock()
        portfolio_repo.get_by_id = AsyncMock()
        position_repo.get_by_symbol = AsyncMock()
        return order_repo, portfolio_repo, position_repo
    @pytest.fixture
    def use_case(self, mock_repositories) -> Any:
        """Создает экземпляр use case."""
        order_repo, portfolio_repo, position_repo = mock_repositories
        return DefaultOrderManagementUseCase(order_repo, portfolio_repo, position_repo)
    @pytest.fixture
    def sample_portfolio(self) -> Any:
        """Создает образец портфеля."""
        portfolio = Mock()
        portfolio.available_balance = Money(Decimal("10000"), "USD")
        return portfolio
    @pytest.fixture
    def sample_order_request(self) -> Any:
        """Создает образец запроса на создание ордера."""
        return CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=Price(Decimal("50000"), "USD", "BTC")
        )
    @pytest.mark.asyncio
    async def test_create_order_success(self, use_case, mock_repositories, sample_portfolio, sample_order_request) -> None:
        """Тест успешного создания ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        # Настройка mock'ов
        portfolio_repo.get_by_id.return_value = sample_portfolio
        order_repo.create.return_value = None  # create не возвращает значение
        # Act
        result = await use_case.create_order(sample_order_request)
        # Assert
        assert result.success is True
        assert "Order created successfully" in result.message
        assert result.order is not None
        assert result.estimated_cost == Decimal("5000")  # 0.1 * 50000
        # Проверяем вызовы
        portfolio_repo.get_by_id.assert_called_once_with(sample_order_request.portfolio_id)
        order_repo.create.assert_called_once()
    @pytest.mark.asyncio
    async def test_create_order_insufficient_funds(self, use_case, mock_repositories, sample_order_request) -> None:
        """Тест создания ордера с недостаточными средствами."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        # Портфель с недостаточными средствами
        poor_portfolio = Mock()
        poor_portfolio.available_balance = Money(Decimal("100"), "USD")
        portfolio_repo.get_by_id.return_value = poor_portfolio
        # Act
        result = await use_case.create_order(sample_order_request)
        # Assert
        assert result.success is False
        assert "Insufficient funds" in result.message
        assert "Insufficient funds" in result.errors
        # Проверяем, что ордер не был создан
        order_repo.create.assert_not_called()
    @pytest.mark.asyncio
    async def test_create_order_portfolio_not_found(self, use_case, mock_repositories, sample_order_request) -> None:
        """Тест создания ордера с несуществующим портфелем."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        portfolio_repo.get_by_id.return_value = None
        # Act
        result = await use_case.create_order(sample_order_request)
        # Assert
        assert result.success is False
        assert "Portfolio not found" in result.message
        assert "Portfolio not found" in result.errors
        # Проверяем, что ордер не был создан
        order_repo.create.assert_not_called()
    @pytest.mark.asyncio
    async def test_create_order_validation_failure(self, use_case, mock_repositories, sample_order_request) -> None:
        """Тест создания ордера с ошибкой валидации."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        # Создаем невалидный запрос (без цены для лимитного ордера)
        invalid_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=None  # Отсутствует цена для лимитного ордера
        )
        # Act
        result = await use_case.create_order(invalid_request)
        # Assert
        assert result.success is False
        assert "validation failed" in result.message.lower()
        # Проверяем, что ордер не был создан
        order_repo.create.assert_not_called()
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, use_case, mock_repositories) -> None:
        """Тест успешной отмены ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_id = uuid4()
        portfolio_id = uuid4()
        # Создаем mock ордер
        mock_order = Mock()
        mock_order.portfolio_id = portfolio_id
        mock_order.status = OrderStatus.PENDING
        mock_order.update_status = Mock()
        order_repo.get_by_id.return_value = mock_order
        cancel_request = CancelOrderRequest(
            order_id=order_id,
            portfolio_id=portfolio_id
        )
        # Act
        result = await use_case.cancel_order(cancel_request)
        # Assert
        assert result.cancelled is True
        assert "Order cancelled successfully" in result.message
        assert result.order == mock_order
        # Проверяем вызовы
        order_repo.get_by_id.assert_called_once_with(order_id)
        mock_order.update_status.assert_called_once_with(OrderStatus.CANCELLED)
        order_repo.update.assert_called_once_with(mock_order)
    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, use_case, mock_repositories) -> None:
        """Тест отмены несуществующего ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_repo.get_by_id.return_value = None
        cancel_request = CancelOrderRequest(
            order_id=uuid4(),
            portfolio_id=uuid4()
        )
        # Act
        result = await use_case.cancel_order(cancel_request)
        # Assert
        assert result.cancelled is False
        assert "Order not found" in result.message
    @pytest.mark.asyncio
    async def test_cancel_order_wrong_portfolio(self, use_case, mock_repositories) -> None:
        """Тест отмены ордера из другого портфеля."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        # Создаем mock ордер с другим portfolio_id
        mock_order = Mock()
        mock_order.portfolio_id = uuid4()
        order_repo.get_by_id.return_value = mock_order
        cancel_request = CancelOrderRequest(
            order_id=uuid4(),
            portfolio_id=uuid4()  # Другой portfolio_id
        )
        # Act
        result = await use_case.cancel_order(cancel_request)
        # Assert
        assert result.cancelled is False
        assert "does not belong to portfolio" in result.message
    @pytest.mark.asyncio
    async def test_cancel_order_already_filled(self, use_case, mock_repositories) -> None:
        """Тест отмены уже исполненного ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        # Создаем mock ордер со статусом FILLED
        mock_order = Mock()
        mock_order.portfolio_id = uuid4()
        mock_order.status = OrderStatus.FILLED
        order_repo.get_by_id.return_value = mock_order
        cancel_request = CancelOrderRequest(
            order_id=uuid4(),
            portfolio_id=mock_order.portfolio_id
        )
        # Act
        result = await use_case.cancel_order(cancel_request)
        # Assert
        assert result.cancelled is False
        assert "Cannot cancel order with status" in result.message
    @pytest.mark.asyncio
    async def test_get_orders_success(self, use_case, mock_repositories) -> None:
        """Тест успешного получения списка ордеров."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        portfolio_id = uuid4()
        mock_orders = [
            Mock(symbol="BTC/USD", status=OrderStatus.PENDING),
            Mock(symbol="ETH/USD", status=OrderStatus.FILLED)
        ]
        order_repo.get_by_portfolio_id.return_value = mock_orders
        get_request = GetOrdersRequest(
            portfolio_id=portfolio_id,
            symbol="BTC/USD",
            status=OrderStatus.PENDING,
            limit=10,
            offset=0
        )
        # Act
        result = await use_case.get_orders(get_request)
        # Assert
        assert result.success is True
        assert len(result.orders) == 1  # Только BTC/USD с PENDING статусом
        assert result.total_count == 1
        assert result.has_more is False
        # Проверяем вызовы
        order_repo.get_by_portfolio_id.assert_called_once_with(portfolio_id)
    @pytest.mark.asyncio
    async def test_get_orders_empty(self, use_case, mock_repositories) -> None:
        """Тест получения пустого списка ордеров."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_repo.get_by_portfolio_id.return_value = []
        get_request = GetOrdersRequest(
            portfolio_id=uuid4(),
            limit=10,
            offset=0
        )
        # Act
        result = await use_case.get_orders(get_request)
        # Assert
        assert result.success is True
        assert len(result.orders) == 0
        assert result.total_count == 0
        assert result.has_more is False
    @pytest.mark.asyncio
    async def test_get_order_by_id_success(self, use_case, mock_repositories) -> None:
        """Тест успешного получения ордера по ID."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_id = OrderId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        mock_order = Mock()
        mock_order.portfolio_id = portfolio_id
        order_repo.get_by_id.return_value = mock_order
        # Act
        result = await use_case.get_order_by_id(order_id, portfolio_id)
        # Assert
        assert result == mock_order
        order_repo.get_by_id.assert_called_once_with(order_id)
    @pytest.mark.asyncio
    async def test_get_order_by_id_not_found(self, use_case, mock_repositories) -> None:
        """Тест получения несуществующего ордера по ID."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_repo.get_by_id.return_value = None
        order_id = OrderId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        # Act
        result = await use_case.get_order_by_id(order_id, portfolio_id)
        # Assert
        assert result is None
    @pytest.mark.asyncio
    async def test_get_order_by_id_wrong_portfolio(self, use_case, mock_repositories) -> None:
        """Тест получения ордера по ID из другого портфеля."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_id = OrderId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        other_portfolio_id = PortfolioId(uuid4())
        mock_order = Mock()
        mock_order.portfolio_id = other_portfolio_id
        order_repo.get_by_id.return_value = mock_order
        # Act
        result = await use_case.get_order_by_id(order_id, portfolio_id)
        # Assert
        assert result is None
    @pytest.mark.asyncio
    async def test_update_order_status_success(self, use_case, mock_repositories) -> None:
        """Тест успешного обновления статуса ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_id = OrderId(uuid4())
        mock_order = Mock()
        mock_order.update_status = Mock()
        order_repo.get_by_id.return_value = mock_order
        # Act
        result = await use_case.update_order_status(
            order_id, OrderStatus.FILLED, Decimal("0.1")
        )
        # Assert
        assert result is True
        mock_order.update_status.assert_called_once_with(OrderStatus.FILLED)
        order_repo.update.assert_called_once_with(mock_order)
    @pytest.mark.asyncio
    async def test_update_order_status_not_found(self, use_case, mock_repositories) -> None:
        """Тест обновления статуса несуществующего ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        order_repo.get_by_id.return_value = None
        order_id = OrderId(uuid4())
        # Act
        result = await use_case.update_order_status(order_id, OrderStatus.FILLED)
        # Assert
        assert result is False
    @pytest.mark.asyncio
    async def test_validate_order_success(self, use_case, mock_repositories, sample_portfolio) -> None:
        """Тест успешной валидации ордера."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        portfolio_repo.get_by_id.return_value = sample_portfolio
        position_repo.get_by_symbol.return_value = None
        sample_order_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=Price(Decimal("50000"), "USD", "BTC")
        )
        # Act
        is_valid, errors = await use_case.validate_order(sample_order_request)
        # Assert
        assert is_valid is True
        assert len(errors) == 0
    @pytest.mark.asyncio
    async def test_validate_order_missing_symbol(self, use_case, mock_repositories) -> None:
        """Тест валидации ордера без символа."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        invalid_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="",  # Пустой символ
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=Price(Decimal("50000"), "USD", "BTC")
        )
        # Act
        is_valid, errors = await use_case.validate_order(invalid_request)
        # Assert
        assert is_valid is False
        assert "Symbol is required" in errors
    @pytest.mark.asyncio
    async def test_validate_order_missing_price_for_limit(self, use_case, mock_repositories, sample_portfolio) -> None:
        """Тест валидации лимитного ордера без цены."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        portfolio_repo.get_by_id.return_value = sample_portfolio
        invalid_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=None  # Отсутствует цена для лимитного ордера
        )
        # Act
        is_valid, errors = await use_case.validate_order(invalid_request)
        # Assert
        assert is_valid is False
        assert "Price is required for limit orders" in errors
    @pytest.mark.asyncio
    async def test_validate_order_insufficient_position_for_sell(self, use_case, mock_repositories, sample_portfolio) -> None:
        """Тест валидации продажи с недостаточной позицией."""
        # Arrange
        order_repo, portfolio_repo, position_repo = mock_repositories
        portfolio_repo.get_by_id.return_value = sample_portfolio
        # Создаем позицию с недостаточным объемом
        mock_position = Mock()
        mock_position.volume = Volume(Decimal("0.05"), "BTC")  # Меньше чем 0.1
        position_repo.get_by_symbol.return_value = mock_position
        sell_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            amount=Volume(Decimal("0.1"), "BTC"),  # Больше чем есть в позиции
            price=None
        )
        # Act
        is_valid, errors = await use_case.validate_order(sell_request)
        # Assert
        assert is_valid is False
        assert "Insufficient position size" in errors[0]
    @pytest.mark.asyncio
    async def test_calculate_order_cost_market_order(self, use_case, sample_order_request) -> None:
        """Тест расчета стоимости рыночного ордера."""
        # Arrange
        market_request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=None
        )
        # Act
        cost = await use_case._calculate_order_cost(market_request)
        # Assert
        assert cost == Decimal("0.1")  # Упрощенная логика для рыночных ордеров
    @pytest.mark.asyncio
    async def test_calculate_order_cost_limit_order(self, use_case, sample_order_request) -> None:
        """Тест расчета стоимости лимитного ордера."""
        # Act
        cost = await use_case._calculate_order_cost(sample_order_request)
        # Assert
        assert cost == Decimal("5000")  # 0.1 * 50000
    @pytest.mark.asyncio
    async def test_calculate_order_cost_no_price(self, use_case) -> None:
        """Тест расчета стоимости ордера без цены."""
        # Arrange
        request_without_price = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Volume(Decimal("0.1"), "BTC"),
            price=None
        )
        # Act
        cost = await use_case._calculate_order_cost(request_without_price)
        # Assert
        assert cost == Decimal("0.1")  # Упрощенная логика
class TestPositionManagementUseCase:
    """Тесты для PositionManagementUseCase."""
    @pytest.fixture
    def mock_repositories(self) -> Any:
        """Создает mock репозитории для позиций."""
        position_repo = Mock()
        portfolio_repo = Mock()
        position_repo.create = AsyncMock()
        position_repo.get_by_id = AsyncMock()
        position_repo.update = AsyncMock()
        position_repo.get_by_symbol = AsyncMock()
        position_repo.get_active_positions = AsyncMock()
        position_repo.get_position_history = AsyncMock()
        return position_repo, portfolio_repo
    @pytest.fixture
    def use_case(self, mock_repositories) -> Any:
        """Создает экземпляр use case для позиций."""
        position_repo, portfolio_repo = mock_repositories
        return DefaultPositionManagementUseCase(position_repo, portfolio_repo)
    @pytest.mark.asyncio
    async def test_create_position_success(self, use_case, mock_repositories) -> None:
        """Тест успешного создания позиции."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        position_id = uuid4()
        mock_position = Mock()
        mock_position.id = position_id
        position_repo.create.return_value = mock_position
        # Act
        result = await use_case.create_position(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            side="LONG",
            quantity=Volume(Decimal("0.5"), "BTC"),
            entry_price=Price(Decimal("50000"), "USD", "BTC")
        )
        # Assert
        assert result == mock_position
        position_repo.create.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_position_by_id_success(self, use_case, mock_repositories) -> None:
        """Тест успешного получения позиции по ID."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        position_id = uuid4()
        portfolio_id = uuid4()
        mock_position = Mock()
        mock_position.portfolio_id = portfolio_id
        position_repo.get_by_id.return_value = mock_position
        # Act
        result = await use_case.get_position_by_id(position_id, portfolio_id)
        # Assert
        assert result == mock_position
        position_repo.get_by_id.assert_called_once_with(position_id)
    @pytest.mark.asyncio
    async def test_get_position_by_symbol_success(self, use_case, mock_repositories) -> None:
        """Тест успешного получения позиции по символу."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        portfolio_id = uuid4()
        symbol = "BTC/USD"
        mock_position = Mock()
        mock_position.symbol = symbol
        position_repo.get_by_symbol.return_value = mock_position
        # Act
        result = await use_case.get_position_by_symbol(portfolio_id, symbol)
        # Assert
        assert result == mock_position
        position_repo.get_by_symbol.assert_called_once_with(portfolio_id, symbol)
    @pytest.mark.asyncio
    async def test_update_position_success(self, use_case, mock_repositories) -> None:
        """Тест успешного обновления позиции."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        position_id = uuid4()
        current_price = Price(Decimal("51000"), "USD", "BTC")
        unrealized_pnl = Money(Decimal("500"), "USD")
        mock_position = Mock()
        position_repo.get_by_id.return_value = mock_position
        # Act
        result = await use_case.update_position(position_id, current_price, unrealized_pnl)
        # Assert
        assert result == mock_position
        position_repo.update.assert_called_once_with(mock_position)
    @pytest.mark.asyncio
    async def test_close_position_success(self, use_case, mock_repositories) -> None:
        """Тест успешного закрытия позиции."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        position_id = uuid4()
        close_price = Price(Decimal("51000"), "USD", "BTC")
        close_quantity = Volume(Decimal("0.5"), "BTC")
        realized_pnl = Money(Decimal("500"), "USD")
        mock_position = Mock()
        position_repo.get_by_id.return_value = mock_position
        # Act
        result = await use_case.close_position(
            position_id, close_price, close_quantity, realized_pnl
        )
        # Assert
        assert result is True
        position_repo.update.assert_called_once_with(mock_position)
    @pytest.mark.asyncio
    async def test_get_active_positions(self, use_case, mock_repositories) -> None:
        """Тест получения активных позиций."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        mock_positions = [
            Mock(symbol="BTC/USD", status="OPEN"),
            Mock(symbol="ETH/USD", status="OPEN")
        ]
        position_repo.get_active_positions.return_value = mock_positions
        # Act
        result = await use_case.get_active_positions()
        # Assert
        assert result == mock_positions
        assert len(result) == 2
        position_repo.get_active_positions.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_position_history(self, use_case, mock_repositories) -> None:
        """Тест получения истории позиций."""
        # Arrange
        position_repo, portfolio_repo = mock_repositories
        mock_history = [
            Mock(symbol="BTC/USD", status="CLOSED"),
            Mock(symbol="ETH/USD", status="CLOSED")
        ]
        position_repo.get_position_history.return_value = mock_history
        # Act
        result = await use_case.get_position_history(
            symbol="BTC/USD",
            start_date=None,
            end_date=None,
            limit=10
        )
        # Assert
        assert result == mock_history
        assert len(result) == 2
        position_repo.get_position_history.assert_called_once() 
