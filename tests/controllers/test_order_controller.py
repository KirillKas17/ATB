from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from unittest.mock import Mock, patch
from infrastructure.core.controllers.order_controller import OrderController
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.entities import Order, Position
else:
    try:
        from domain.entities import Order, Position
    except ImportError:
        # Создаем заглушки для тестирования
        class Order:
            def __init__(
                self,
                id: str,
                pair: str,
                type: str,
                side: str,
                price: float,
                size: float,
                status: str,
                timestamp: datetime,
            ) -> None:
                self.id = id
                self.pair = pair
                self.type = type
                self.side = side
                self.price = price
                self.size = size
                self.status = status
                self.timestamp = timestamp

        class Position:
            def __init__(
                self,
                pair: str,
                side: str,
                size: float,
                entry_price: float,
                current_price: float,
                pnl: float,
                leverage: float,
                entry_time: datetime,
            ) -> None:
                self.pair = pair
                self.side = side
                self.size = size
                self.entry_price = entry_price
                self.current_price = current_price
                self.pnl = pnl
                self.leverage = leverage
                self.entry_time = entry_time


@pytest.fixture
def mock_exchange() -> Any:
    exchange = AsyncMock()
    exchange.create_order = Mock()
    exchange.cancel_order = Mock()
    exchange.get_order = Mock()
    exchange.get_open_orders = Mock()
    return exchange


@pytest.fixture
def order_controller(mock_exchange) -> Any:
    return OrderController(mock_exchange, {})


@pytest.fixture
def sample_order() -> Any:
    return Order(
        id="",
        pair="BTC/USDT",
        type="market",
        side="buy",
        price=50000.0,
        size=0.1,
        status="new",
        timestamp=datetime.now(),
    )


@pytest.mark.asyncio
async def test_create_order(order_controller, mock_exchange) -> None:
    """Тест создания ордера"""
    mock_exchange.create_order.return_value = {"id": "123", "status": "open"}

    result = await order_controller.create_order("BTC/USDT", "buy", "market", 0.1)

    assert result["id"] == "123"
    assert result["status"] == "open"
    mock_exchange.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 0.1, None)


@pytest.mark.asyncio
async def test_cancel_order(order_controller, mock_exchange) -> None:
    """Тест отмены ордера"""
    order_id = "123"
    symbol = "BTC/USDT"
    mock_exchange.cancel_order.return_value = True
    
    result = await order_controller.cancel_order(order_id, symbol)

    assert result is True
    mock_exchange.cancel_order.assert_called_once_with(order_id)


@pytest.mark.asyncio
async def test_get_order(order_controller, mock_exchange) -> None:
    """Тест получения ордера"""
    order_data = {
        "id": "123",
        "symbol": "BTC/USDT",
        "type": "market",
        "side": "buy",
        "price": 50000.0,
        "amount": 0.1,
        "status": "open",
        "timestamp": datetime.now().timestamp() * 1000,
    }
    mock_exchange.get_order.return_value = order_data

    result = await order_controller.get_order("123", "BTC/USDT")

    # Проверяем основные поля
    assert result["id"] == "123"
    assert result["symbol"] == "BTC/USDT"
    assert result["type"] == "market"
    assert result["side"] == "buy"
    assert result["price"] == 50000.0
    assert result["amount"] == 0.1
    assert result["status"] == "open"
    # Проверяем дополнительные поля, которые добавляет контроллер
    assert "filled" in result


@pytest.mark.asyncio
async def test_get_open_orders(order_controller, mock_exchange) -> None:
    """Тест получения открытых ордеров"""
    orders_data = [
        {
            "id": "123",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "price": 50000.0,
            "amount": 0.1,
            "status": "open",
            "timestamp": datetime.now().timestamp() * 1000,
        }
    ]
    mock_exchange.get_open_orders.return_value = orders_data

    result = await order_controller.get_open_orders()

    assert len(result) == 1
    assert result[0]["id"] == "123"
    assert result[0]["symbol"] == "BTC/USDT"
    mock_exchange.get_open_orders.assert_called_once_with(None)
