from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.controllers.position_controller import PositionController
from core.models import Order, Position


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.fetch_positions = AsyncMock()
    return exchange


@pytest.fixture
def mock_order_controller():
    controller = AsyncMock()
    controller.place_order = AsyncMock()
    return controller


@pytest.fixture
def position_controller(mock_exchange, mock_order_controller):
    return PositionController(mock_exchange, mock_order_controller)


@pytest.fixture
def sample_position():
    return Position(
        pair="BTC/USDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        entry_time=datetime.now(),
    )


@pytest.mark.asyncio
async def test_open_position(
    position_controller, sample_position, mock_order_controller
):
    """Тест открытия позиции"""
    mock_order_controller.place_order.return_value = Order(
        id="123",
        pair="BTC/USDT",
        type="market",
        side="buy",
        price=50000.0,
        size=0.1,
        status="open",
        timestamp=datetime.now(),
    )
    result = await position_controller.open_position(sample_position)
    assert result is not None
    assert result.pair == "BTC/USDT"


@pytest.mark.asyncio
async def test_close_position(
    position_controller, sample_position, mock_order_controller
):
    """Тест закрытия позиции"""
    position_controller.positions["BTC/USDT"] = sample_position
    mock_order_controller.place_order.return_value = Order(
        id="123",
        pair="BTC/USDT",
        type="market",
        side="sell",
        price=50000.0,
        size=0.1,
        status="closed",
        timestamp=datetime.now(),
    )
    await position_controller.close_position("BTC/USDT")
    assert "BTC/USDT" not in position_controller.positions


@pytest.mark.asyncio
async def test_update_positions(position_controller, mock_exchange):
    """Тест обновления позиций"""
    positions_data = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "contracts": 0.1,
            "entryPrice": 50000.0,
            "markPrice": 50000.0,
            "unrealizedPnl": 0.0,
            "timestamp": datetime.now().timestamp() * 1000,
        }
    ]
    mock_exchange.fetch_positions.return_value = positions_data

    await position_controller.update_positions()

    assert "BTC/USDT" in position_controller.positions
    assert position_controller.positions["BTC/USDT"].pair == "BTC/USDT"


def test_get_position(position_controller, sample_position):
    """Тест получения позиции"""
    position_controller.positions["BTC/USDT"] = sample_position

    result = position_controller.get_position("BTC/USDT")

    assert result is not None
    assert result.pair == "BTC/USDT"
    assert result.side == "long"


def test_get_all_positions(position_controller, sample_position):
    """Тест получения всех позиций"""
    position_controller.positions["BTC/USDT"] = sample_position

    result = position_controller.get_all_positions()

    assert len(result) == 1
    assert result[0].pair == "BTC/USDT"
    assert result[0].side == "long"
