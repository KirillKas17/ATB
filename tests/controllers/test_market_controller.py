from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.controllers.market_controller import MarketController
from core.models import MarketData


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock()
    exchange.fetch_order_book = AsyncMock()
    exchange.fetch_trades = AsyncMock()
    return exchange


@pytest.fixture
def market_controller(mock_exchange):
    return MarketController(mock_exchange)


@pytest.mark.asyncio
async def test_get_ticker(market_controller, mock_exchange):
    """Тест получения тикера"""
    ticker_data = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49900.0,
        "ask": 50100.0,
        "volume": 100.0,
    }
    mock_exchange.fetch_ticker.return_value = ticker_data

    result = await market_controller.get_ticker("BTC/USDT")

    assert result == ticker_data
    mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_get_ohlcv(market_controller, mock_exchange):
    """Тест получения OHLCV данных"""
    ohlcv_data = [[datetime.now().timestamp() * 1000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]]
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data

    result = await market_controller.get_ohlcv("BTC/USDT")

    assert len(result) == 1
    assert isinstance(result[0], MarketData)
    assert result[0].pair == "BTC/USDT"
    assert result[0].open == 50000.0
    assert result[0].high == 50100.0
    assert result[0].low == 49900.0
    assert result[0].close == 50050.0
    assert result[0].volume == 100.0


@pytest.mark.asyncio
async def test_get_order_book(market_controller, mock_exchange):
    """Тест получения книги ордеров"""
    order_book = {"bids": [[49900.0, 1.0]], "asks": [[50100.0, 1.0]]}
    mock_exchange.fetch_order_book.return_value = order_book

    result = await market_controller.get_order_book("BTC/USDT")

    assert result == order_book
    mock_exchange.fetch_order_book.assert_called_once_with("BTC/USDT", 20)


@pytest.mark.asyncio
async def test_get_trades(market_controller, mock_exchange):
    """Тест получения сделок"""
    trades = [
        {
            "id": "123",
            "symbol": "BTC/USDT",
            "side": "buy",
            "price": 50000.0,
            "amount": 0.1,
            "timestamp": datetime.now().timestamp() * 1000,
        }
    ]
    mock_exchange.fetch_trades.return_value = trades

    result = await market_controller.get_trades("BTC/USDT")

    assert result == trades
    mock_exchange.fetch_trades.assert_called_once_with("BTC/USDT", limit=50)


def test_get_market_data(market_controller):
    """Тест получения рыночных данных"""
    market_data = [
        MarketData(
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            pair="BTC/USDT",
        )
    ]
    market_controller.market_data["BTC/USDT"] = market_data

    result = market_controller.get_market_data("BTC/USDT")

    assert result == market_data
