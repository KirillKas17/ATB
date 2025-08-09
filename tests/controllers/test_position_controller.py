from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from unittest.mock import Mock, patch
from infrastructure.core.controllers.position_controller import PositionController
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
            ) -> Any:
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
            ) -> Any:
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
    exchange.get_positions = Mock()
    return exchange


@pytest.fixture
def mock_order_controller() -> Any:
    controller = AsyncMock()
    controller.create_order = AsyncMock()
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
async def test_open_position(position_controller, sample_position, mock_order_controller) -> None:
    """Тест открытия позиции"""
    mock_order_controller.create_order.return_value = {
        "id": "123",
        "symbol": "BTC/USDT",
        "type": "market",
        "side": "buy",
        "status": "open"
    }
    result = await position_controller.open_position("BTC/USDT", "buy", 0.1)

    assert result["id"] == "123"
    assert result["status"] == "open"
    mock_order_controller.create_order.assert_called_once_with(
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        amount=0.1
    )


@pytest.mark.asyncio
async def test_close_position(position_controller, mock_order_controller) -> None:
    """Тест закрытия позиции"""
    # Мокаем get_position чтобы вернуть позицию
    position_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "size": 0.1,
        "entry_price": 50000.0,
        "mark_price": 50000.0,
        "unrealized_pnl": 0.0
    }
    
    with patch.object(position_controller, 'get_position', return_value=position_data):
        mock_order_controller.create_order.return_value = {
            "id": "123",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "sell",
            "status": "closed"
        }

        result = await position_controller.close_position("BTC/USDT")

        assert result["id"] == "123"
        assert result["status"] == "closed"
        mock_order_controller.create_order.assert_called_once_with(
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            amount=0.1
        )


@pytest.mark.asyncio
async def test_get_positions(position_controller, mock_exchange) -> None:
    """Тест получения позиций"""
    positions_data = [
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "size": 0.1,
            "entry_price": 50000.0,
            "mark_price": 50000.0,
            "unrealized_pnl": 0.0,
            "timestamp": datetime.now().timestamp() * 1000,
        }
    ]
    mock_exchange.get_positions.return_value = positions_data

    result = await position_controller.get_positions()

    assert len(result) == 1
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["side"] == "buy"
    assert result[0]["size"] == 0.1


@pytest.mark.asyncio
async def test_get_position(position_controller, mock_exchange) -> None:
    """Тест получения позиции"""
    position_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "size": 0.1,
        "entry_price": 50000.0,
        "mark_price": 50000.0,
        "unrealized_pnl": 0.0
    }
    
    with patch.object(position_controller, 'get_positions', return_value=[position_data]):
        result = await position_controller.get_position("BTC/USDT")

        assert result is not None
        assert result["symbol"] == "BTC/USDT"
        assert result["side"] == "buy"


@pytest.mark.asyncio
async def test_get_all_positions(position_controller, mock_exchange) -> None:
    """Тест получения всех позиций"""
    positions_data = [
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "size": 0.1,
            "entry_price": 50000.0,
            "mark_price": 50000.0,
            "unrealized_pnl": 0.0
        },
        {
            "symbol": "ETH/USDT",
            "side": "sell",
            "size": 1.0,
            "entry_price": 3000.0,
            "mark_price": 3000.0,
            "unrealized_pnl": 0.0
        }
    ]
    mock_exchange.get_positions.return_value = positions_data

    result = await position_controller.get_positions()

    assert len(result) == 2
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["side"] == "buy"
    assert result[1]["symbol"] == "ETH/USDT"
    assert result[1]["side"] == "sell"
