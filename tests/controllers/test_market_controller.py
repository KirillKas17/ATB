from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from unittest.mock import Mock, patch
from infrastructure.core.controllers.market_controller import MarketController
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.entities import MarketData
else:
    try:
        from domain.entities import MarketData
    except ImportError:
        # Fallback для тестов
        class MarketData:
            def __init__(
                self, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float, symbol: str
            ) -> None:
                self.timestamp = timestamp
                self.open = open
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.symbol = symbol


@pytest.fixture
def mock_exchange() -> Any:
    exchange = AsyncMock()
    exchange.get_ticker = Mock()
    exchange.get_ohlcv = Mock()
    exchange.get_orderbook = Mock()
    exchange.get_trades = Mock()
    return exchange


@pytest.fixture
def market_controller(mock_exchange) -> Any:
    return MarketController(mock_exchange)


@pytest.mark.asyncio
async def test_get_ticker(market_controller, mock_exchange) -> None:
    """Тест получения тикера"""
    ticker_data = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49900.0,
        "ask": 50100.0,
        "volume": 100.0,
    }
    mock_exchange.get_ticker.return_value = ticker_data

    result = await market_controller.get_ticker("BTC/USDT")

    # Проверяем основные поля
    assert result["symbol"] == "BTC/USDT"
    assert result["last"] == 50000.0
    assert result["bid"] == 49900.0
    assert result["ask"] == 50100.0
    assert result["volume"] == 100.0
    # Проверяем дополнительные поля, которые добавляет контроллер
    assert "high" in result
    assert "low" in result
    assert "timestamp" in result
    
    mock_exchange.get_ticker.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_get_ohlcv(market_controller, mock_exchange) -> None:
    """Тест получения OHLCV данных"""
    # Создаем данные в формате, который ожидает контроллер
    timestamp = datetime.now().timestamp() * 1000
    ohlcv_data = [
        {
            "timestamp": timestamp,
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.0
        }
    ]
    mock_exchange.get_ohlcv.return_value = ohlcv_data

    result = await market_controller.get_ohlcv("BTC/USDT")

    assert len(result) == 1
    # Проверяем, что результат содержит MarketData объект
    market_data = result[0]
    assert hasattr(market_data, 'symbol')
    assert hasattr(market_data, 'open')
    assert hasattr(market_data, 'high')
    assert hasattr(market_data, 'low')
    assert hasattr(market_data, 'close')
    assert hasattr(market_data, 'volume')
    assert hasattr(market_data, 'timestamp')
    
    mock_exchange.get_ohlcv.assert_called_once_with("BTC/USDT", "1h", limit=100)


@pytest.mark.asyncio
async def test_get_order_book(market_controller, mock_exchange) -> None:
    """Тест получения книги ордеров"""
    order_book = {"bids": [[49900.0, 1.0]], "asks": [[50100.0, 1.0]]}
    mock_exchange.get_orderbook.return_value = order_book

    result = await market_controller.get_order_book("BTC/USDT")

    # Проверяем основные поля
    assert result["symbol"] == "BTC/USDT"
    assert result["bids"] == [[49900.0, 1.0]]
    assert result["asks"] == [[50100.0, 1.0]]
    assert "timestamp" in result
    
    mock_exchange.get_orderbook.assert_called_once_with("BTC/USDT", 20)


@pytest.mark.asyncio
async def test_get_trades(market_controller, mock_exchange) -> None:
    """Тест получения сделок"""
    # Создаем мок объекты сделок
    from unittest.mock import Mock
    trade1 = Mock()
    trade1.id = "123"
    trade1.side = "buy"
    trade1.price = Mock()
    trade1.price.amount = 50000.0
    trade1.quantity = 0.1
    trade1.timestamp = datetime.now()
    
    trades = [trade1]
    mock_exchange.get_trades.return_value = trades

    result = await market_controller.get_trades("BTC/USDT")

    assert len(result) == 1
    assert result[0]["id"] == "123"
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["side"] == "buy"
    assert result[0]["price"] == 50000.0
    assert result[0]["amount"] == 0.1
    assert "timestamp" in result[0]
    
    mock_exchange.get_trades.assert_called_once_with("BTC/USDT", 100)


@pytest.mark.asyncio
async def test_get_market_data(market_controller, mock_exchange) -> None:
    """Тест получения рыночных данных"""
    # Настраиваем моки для всех методов
    mock_exchange.get_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49900.0,
        "ask": 50100.0,
        "volume": 100.0,
    }
    
    mock_exchange.get_ohlcv.return_value = []
    mock_exchange.get_orderbook.return_value = {"bids": [], "asks": []}
    mock_exchange.get_trades.return_value = []

    result = await market_controller.get_market_data("BTC/USDT")

    # Проверяем структуру ответа
    assert result["symbol"] == "BTC/USDT"
    assert "ticker" in result
    assert "ohlcv" in result
    assert "orderbook" in result
    assert "trades" in result
    assert "timestamp" in result
