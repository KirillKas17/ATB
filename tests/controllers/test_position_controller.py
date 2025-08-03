from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from unittest.mock import Mock, patch
from core.controllers.position_controller import PositionController
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.entities import Order, Position
else:
    try:
        from domain.entities import Order, Position
    except ImportError:
        # Создаем заглушки для тестирования
        class Order:
            def __init__(self, id: str, pair: str, type: str, side: str, price: float, size: float, status: str, timestamp: datetime) -> Any:
                self.id = id
                self.pair = pair
                self.type = type
                self.side = side
                self.price = price
                self.size = size
                self.status = status
                self.timestamp = timestamp
        class Position:
            def __init__(self, pair: str, side: str, size: float, entry_price: float, current_price: float, pnl: float, leverage: float, entry_time: datetime) -> Any:
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
    exchange.fetch_positions = AsyncMock()
    return exchange


@pytest.fixture
def mock_order_controller() -> Any:
    controller = AsyncMock()
    controller.place_order = AsyncMock()
    return controller


@pytest.fixture
def position_controller(mock_exchange, mock_order_controller) -> Any:
    return PositionController(mock_exchange, mock_order_controller)


@pytest.fixture
def sample_position() -> Any:
    return Position(
        pair="BTC/USDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        leverage=1,
        entry_time=datetime.now(),
    )


@pytest.mark.asyncio
async def test_open_position(
    position_controller, sample_position, mock_order_controller
) -> None:
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
) -> None:
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
async def test_update_positions(position_controller, mock_exchange) -> None:
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


def test_get_position(position_controller, sample_position) -> None:
    """Тест получения позиции"""
    position_controller.positions["BTC/USDT"] = sample_position

    result = position_controller.get_position("BTC/USDT")

    assert result is not None
    assert result.pair == "BTC/USDT"
    assert result.side == "long"


def test_get_all_positions(position_controller, sample_position) -> None:
    """Тест получения всех позиций"""
    position_controller.positions["BTC/USDT"] = sample_position

    result = position_controller.get_all_positions()

    assert len(result) == 1
    assert result[0].pair == "BTC/USDT"
    assert result[0].side == "long"
