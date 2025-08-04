"""
Unit тесты для MarketData.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from infrastructure.external_services.market_data import MarketData, MarketDataConfig, DataCache
@pytest.fixture
def event_loop() -> Any:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
@pytest.fixture
def market_data() -> Any:
    return MarketData(symbol="BTCUSDT", interval="1m", max_candles=10)
@pytest.mark.asyncio
async def test_init_and_basic_properties(market_data) -> None:
    assert market_data.symbol == "BTCUSDT"
    assert market_data.interval == "1m"
    assert market_data.max_candles == 10
    assert isinstance(market_data.df, pd.DataFrame)
    assert market_data.is_connected is False
    assert market_data.websocket is None
@pytest.mark.asyncio
async def test_start_and_stop(market_data) -> None:
    with patch("websockets.connect", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = AsyncMock()
        await market_data.start()
        assert market_data.is_connected is True
        await market_data.stop()
        assert market_data.is_connected is False
@pytest.mark.asyncio
async def test_subscribe(market_data) -> None:
    market_data.websocket = AsyncMock()
    await market_data._subscribe()
    market_data.websocket.send.assert_called()
@pytest.mark.asyncio
async def test_process_candle_and_update(market_data) -> None:
    candle = {
        "timestamp": int(datetime.now().timestamp() * 1000),
        "open": "100",
        "high": "110",
        "low": "90",
        "close": "105",
        "volume": "1000"
    }
    market_data._update_indicators = AsyncMock()
    market_data._update_metrics = AsyncMock()
    market_data._check_signals = AsyncMock()
    await market_data._process_candle(candle)
    assert not market_data.df.empty
    assert market_data.df.iloc[-1]["close"] == 105.0
@pytest.mark.asyncio
async def test_handle_messages_reconnect(market_data) -> None:
    market_data.is_connected = True
    market_data.websocket = AsyncMock()
    market_data.websocket.recv = AsyncMock(side_effect=[
        json.dumps({"data": {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "1000"
        }}),
        Exception("Test error")
    ])
    market_data._process_candle = AsyncMock()
    with patch.object(market_data, "_reconnect", new=AsyncMock()):
        with patch("websockets.exceptions.ConnectionClosed", Exception):
            await market_data._handle_messages()
    market_data._process_candle.assert_called()
@pytest.mark.asyncio
async def test_update_indicators_and_metrics(market_data) -> None:
    # Заполняем df тестовыми данными
    now = datetime.now()
    market_data.df = pd.DataFrame({
        "timestamp": [now for _ in range(10)],
        "open": np.linspace(100, 110, 10),
        "high": np.linspace(110, 120, 10),
        "low": np.linspace(90, 100, 10),
        "close": np.linspace(100, 110, 10),
        "volume": np.linspace(1000, 2000, 10)
    })
    await market_data._update_indicators()
    await market_data._update_metrics()
    assert isinstance(market_data.indicators, dict)
    assert isinstance(market_data.volatility, float)
@pytest.mark.asyncio
async def test_get_indicators_and_metrics(market_data) -> None:
    market_data.indicators = {"rsi": 50, "macd": 0.1}
    result = await market_data.get_indicators()
    assert "rsi" in result
    result = await market_data.get_metrics()
    assert isinstance(result, dict)
@pytest.mark.asyncio
async def test_get_support_resistance(market_data) -> None:
    now = datetime.now()
    market_data.df = pd.DataFrame({
        "timestamp": [now for _ in range(10)],
        "open": np.linspace(100, 110, 10),
        "high": np.linspace(110, 120, 10),
        "low": np.linspace(90, 100, 10),
        "close": np.linspace(100, 110, 10),
        "volume": np.linspace(1000, 2000, 10)
    })
    result = await market_data.get_support_resistance()
    assert "support" in result
    assert "resistance" in result
@pytest.mark.asyncio
async def test_data_cache_add_get_clear() -> None:
    cache = DataCache(max_size=2, ttl=1)
    df = pd.DataFrame({"close": [1, 2, 3]})
    await cache.add("BTCUSDT", df)
    result = await cache.get("BTCUSDT")
    assert isinstance(result, pd.DataFrame)
    await cache.clear()
    result = await cache.get("BTCUSDT")
    assert result is None
@pytest.mark.asyncio
async def test_market_data_edge_cases(market_data) -> None:
    # Пустая свеча
    with pytest.raises(Exception):
        await market_data._process_candle({})
    # Некорректные данные
    with pytest.raises(Exception):
        await market_data._process_candle({"timestamp": "bad", "open": "bad"})
@pytest.mark.asyncio
async def test_market_data_error_handling(market_data) -> None:
    # Ошибка при подписке
    market_data.websocket = MagicMock()
    market_data.websocket.send = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await market_data._subscribe()
    # Ошибка при старте
    with patch("websockets.connect", new_callable=AsyncMock) as mock_ws:
        mock_ws.side_effect = Exception("fail")
        with pytest.raises(Exception):
            await market_data.start()
    # Ошибка при остановке
    market_data.websocket = AsyncMock()
    market_data.websocket.close = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await market_data.stop() 
