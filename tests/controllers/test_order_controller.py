from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.controllers.order_controller import OrderController
from core.models import Order


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.create_order = AsyncMock()
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock()
    exchange.fetch_open_orders = AsyncMock()
    return exchange


@pytest.fixture
def order_controller(mock_exchange):
    return OrderController(mock_exchange, {})


@pytest.fixture
def sample_order():
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
async def test_place_order(order_controller, sample_order, mock_exchange):
    """Тест размещения ордера"""
    mock_exchange.create_order.return_value = {"id": "123", "status": "open"}

    result = await order_controller.place_order(sample_order)

    assert result.id == "123"
    assert result.status == "open"
    assert result.id in order_controller.active_orders


@pytest.mark.asyncio
async def test_cancel_order(order_controller, mock_exchange):
    """Тест отмены ордера"""
    order_id = "123"
    await order_controller.cancel_order(order_id)
    mock_exchange.cancel_order.assert_called_once_with(order_id)


@pytest.mark.asyncio
async def test_get_order(order_controller, mock_exchange):
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
    mock_exchange.fetch_order.return_value = order_data

    result = await order_controller.get_order("123")

    assert result is not None
    assert result.id == "123"
    assert result.pair == "BTC/USDT"


@pytest.mark.asyncio
async def test_get_open_orders(order_controller, mock_exchange):
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
    mock_exchange.fetch_open_orders.return_value = orders_data

    result = await order_controller.get_open_orders()

    assert len(result) == 1
    assert result[0].id == "123"
    assert result[0].pair == "BTC/USDT"
