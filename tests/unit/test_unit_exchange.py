"""
Unit тесты для Exchange.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from infrastructure.external_services.exchange import Exchange, ExchangeConfig, Position
@pytest.fixture
def event_loop() -> Any:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
@pytest.fixture
def exchange() -> Any:
    config = ExchangeConfig(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        symbols=["BTC/USDT"],
        intervals=["1m"]
    )
    return Exchange(config)
@pytest.mark.asyncio
async def test_init_and_basic_properties(exchange) -> None:
    assert exchange.config.api_key == "test_key"
    assert exchange.config.testnet is True
    assert isinstance(exchange.orders, dict)
    assert isinstance(exchange.positions, dict)
    assert isinstance(exchange.balance, dict)
    assert exchange.is_running is False
@pytest.mark.asyncio
async def test_start_and_stop(exchange) -> None:
    exchange.account_manager.start = AsyncMock()
    exchange.account_manager.stop = AsyncMock()
    exchange._update_market_data = AsyncMock()
    exchange._monitor_trades = AsyncMock()
    exchange._health_check = AsyncMock()
    exchange._save_metrics = AsyncMock()
    with patch("asyncio.create_task", new=lambda coro: AsyncMock()):
        await exchange.start()
        assert exchange.is_running is True
        await exchange.stop()
        assert exchange.is_running is False
@pytest.mark.asyncio
async def test_get_market_data(exchange) -> None:
    exchange.client.fetch_ohlcv = AsyncMock(return_value=[[1,2,3,4,5,6]]*5)
    df = await exchange.get_market_data("BTC/USDT", "1m", limit=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
@pytest.mark.asyncio
async def test_get_market_data_no_data(exchange) -> None:
    exchange.client.fetch_ohlcv = AsyncMock(return_value=None)
    df = await exchange.get_market_data("BTC/USDT", "1m", limit=5)
    assert df is None
@pytest.mark.asyncio
async def test_create_and_cancel_order(exchange) -> None:
    exchange.order_manager.create_order = AsyncMock(return_value=MagicMock(id="order1"))
    order = await exchange.create_order(
        symbol="BTC/USDT", side="buy", amount=1, price=10000, stop_loss=9500, take_profit=11000, confidence=0.9
    )
    assert order.id == "order1"
    exchange.order_manager.cancel_order = AsyncMock()
    await exchange.cancel_order("order1")
    exchange.order_manager.cancel_order.assert_called_with("order1")
@pytest.mark.asyncio
async def test_place_order(exchange) -> None:
    exchange.client.place_order = AsyncMock(return_value={"id": "order2"})
    result = await exchange.place_order("BTC/USDT", "buy", "limit", 1, price=10000)
    assert isinstance(result, dict)
    assert result["id"] == "order2"
@pytest.mark.asyncio
async def test_get_order_status_and_position(exchange) -> None:
    order = MagicMock(id="order3")
    exchange.orders["order3"] = order
    assert exchange.get_order_status("order3") == order
    pos = Position(symbol="BTC/USDT", quantity=1, entry_price=10000, current_price=10500, unrealized_pnl=500, realized_pnl=0)
    exchange.positions["BTC/USDT"] = pos
    assert exchange.get_position("BTC/USDT") == pos
@pytest.mark.asyncio
async def test_get_balance_and_update_market_data(exchange) -> None:
    exchange.balance["USDT"] = 1000.0
    assert exchange.get_balance("USDT") == 1000.0
    df = pd.DataFrame({"close": [1,2,3]})
    exchange.update_market_data("BTC/USDT", df)
    assert "BTC/USDT" in exchange.data_cache._cache
@pytest.mark.asyncio
async def test_error_handling(exchange) -> None:
    # Ошибка при старте
    exchange.account_manager.start = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await exchange.start()
    # Ошибка при остановке
    exchange.account_manager.stop = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await exchange.stop()
@pytest.mark.asyncio
async def test_edge_cases(exchange) -> None:
    # Некорректный order_id
    assert exchange.get_order_status("bad_id") is None
    # Некорректный symbol
    assert exchange.get_position("BAD/PAIR") is None
    # Некорректный asset
    assert exchange.get_balance("BAD") == 0.0 
