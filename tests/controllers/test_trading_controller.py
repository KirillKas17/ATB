from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.controllers.trading_controller import TradingController
from core.models import Order, Position


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.fetch_balance = AsyncMock(
        return_value={
            "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
            "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
        }
    )
    exchange.fetch_ticker = AsyncMock(
        return_value={"last": 50000.0, "bid": 49900.0, "ask": 50100.0, "volume": 100.0}
    )
    exchange.create_order = AsyncMock(
        return_value={
            "id": "test_order",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "sell",
            "amount": 0.1,
            "price": 50000.0,
            "status": "closed",
        }
    )
    exchange.cancel_order = AsyncMock(
        return_value={"id": "test_order", "status": "canceled"}
    )
    exchange.fetch_markets = AsyncMock(
        return_value=[{"symbol": "BTC/USDT", "active": True}]
    )
    return exchange


@pytest.fixture
def config():
    return {
        "trading_pairs": ["BTC/USDT"],
        "risk_limits": {
            "max_position_size": 1.0,
            "max_daily_loss": 1000.0,
            "max_leverage": 3.0,
        },
        "order_settings": {"default_amount": 0.1, "default_leverage": 1.0},
        "market_update_interval": 60,
        "position_update_interval": 60,
        "order_update_interval": 60,
    }


@pytest.fixture
def controller(mock_exchange, config):
    return TradingController(mock_exchange, config)


@pytest.mark.asyncio
async def test_start(controller):
    await controller.start()
    assert controller.state.is_running
    assert controller.monitoring_tasks


@pytest.mark.asyncio
async def test_stop(controller):
    await controller.start()
    await controller.stop()
    assert not controller.state.is_running
    assert not controller.monitoring_tasks


@pytest.mark.asyncio
async def test_load_config(controller):
    await controller._load_config()
    assert controller._validate_config() is True


@pytest.mark.asyncio
async def test_init_trading_pairs(controller):
    await controller._init_trading_pairs()
    assert "BTC/USDT" in controller.trading_pairs


@pytest.mark.asyncio
async def test_start_monitoring(controller):
    await controller._start_monitoring()
    assert controller.monitoring_tasks
    assert len(controller.monitoring_tasks) == 3


@pytest.mark.asyncio
async def test_stop_monitoring(controller):
    await controller._start_monitoring()
    await controller._stop_monitoring()
    assert not controller.monitoring_tasks


@pytest.mark.asyncio
async def test_monitor_market_data(controller):
    await controller._monitor_market()
    assert hasattr(controller.market_controller, "current_state")


@pytest.mark.asyncio
async def test_monitor_positions(controller):
    await controller._monitor_positions()
    assert controller.position_controller.positions is not None


@pytest.mark.asyncio
async def test_monitor_orders(controller):
    await controller._monitor_orders()
    assert controller.order_controller.active_orders is not None


@pytest.mark.asyncio
async def test_close_all_positions(controller):
    position = Position(
        pair="BTC/USDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        leverage=1.0,
        entry_time=datetime.now(),
    )
    controller.position_controller.positions = {"BTC/USDT": position}

    mock_order = Order(
        id="test_order",
        pair="BTC/USDT",
        type="market",
        side="sell",
        price=50000.0,
        size=0.1,
        status="closed",
        timestamp=datetime.now(),
    )

    async def close_position_and_remove(pair):
        controller.position_controller.positions.pop(pair, None)
        return mock_order

    controller.position_controller.close_position = AsyncMock(
        side_effect=close_position_and_remove
    )

    await controller.close_all_positions()
    assert len(controller.position_controller.positions) == 0


@pytest.mark.asyncio
async def test_cancel_all_orders(controller):
    order = Order(
        id="test_order",
        pair="BTC/USDT",
        type="limit",
        side="buy",
        size=0.1,
        price=50000.0,
        status="open",
        timestamp=datetime.now(),
    )
    controller.order_controller.active_orders = {"test_order": order}
    controller.order_controller.cancel_order = AsyncMock(return_value=True)
    controller.exchange.cancel_order = AsyncMock(
        return_value={
            "id": "test_order",
            "status": "canceled",
            "timestamp": datetime.now().timestamp() * 1000,
        }
    )
    await controller.cancel_all_orders()
    assert len(controller.order_controller.active_orders) == 0


def test_validate_config(controller):
    """Тест валидации конфигурации"""
    assert controller._validate_config() is True

    # Тест с невалидной конфигурацией
    controller.config = {}
    assert controller._validate_config() is False
