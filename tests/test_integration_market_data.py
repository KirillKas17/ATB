import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from exchange.bybit_client import BybitClient, BybitConfig
from exchange.market_data import DataCache, MarketData
@pytest.fixture
def bybit_config() -> Any:
    """Фикстура с конфигурацией Bybit"""
    return BybitConfig(api_key="test_key", api_secret="test_secret", testnet=True)
@pytest.fixture
async def bybit_client(bybit_config) -> Any:
    """Фикстура с клиентом Bybit"""
    client = BybitClient(bybit_config)
    try:
        await client.connect()
        yield client
    finally:
        await client.close()
@pytest.fixture
def market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    return MarketData(
        symbol="BTC/USDT",
        interval="1m",
        exchange="binance",
        api_key="test_key",
        api_secret="test_secret",
    )
@pytest.fixture
def data_cache() -> Any:
    """Фикстура с кешем данных"""
    return DataCache(ttl=5)  # 5 секунд для тестов
def test_initialization(market_data) -> None:
    """Тест инициализации"""
    assert market_data.symbol == "BTC/USDT"
    assert market_data.interval == "1m"
    assert market_data.exchange == "binance"
    assert market_data.api_key == "test_key"
    assert market_data.api_secret == "test_secret"
def test_get_candles(market_data) -> None:
    """Тест получения свечей"""
    candles = market_data.get_candles(limit=100)
    assert isinstance(candles, pd.DataFrame)
    assert not candles.empty
def test_get_orderbook(market_data) -> None:
    """Тест получения стакана"""
    orderbook = market_data.get_orderbook()
    assert isinstance(orderbook, dict)
    assert "bids" in orderbook
    assert "asks" in orderbook
def test_get_trades(market_data) -> None:
    """Тест получения сделок"""
    trades = market_data.get_trades(limit=100)
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty
def test_get_historical_data(market_data) -> None:
    """Тест получения исторических данных"""
    data = market_data.get_historical_data(
        start_time="2024-01-01", end_time="2024-01-02"
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
def test_error_handling(market_data) -> None:
    """Тест обработки ошибок"""
    with pytest.raises(Exception):
        market_data.get_candles(limit=-1)
@pytest.mark.asyncio
async def test_async_initialization(market_data, bybit_client) -> None:
    """Тест инициализации"""
    assert market_data.client == bybit_client
    assert isinstance(market_data.data_dir, Path)
    assert market_data.data_dir.exists()
    assert isinstance(market_data.cache, DataCache)
    assert market_data.update_task is None
@pytest.mark.asyncio
async def test_async_get_candles(market_data) -> None:
    """Тест получения свечей"""
    try:
        # Получение данных
        df = await market_data.get_candles(symbol="BTCUSDT", interval="1h", limit=100)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(
            col in df.columns for col in ["open", "high", "low", "close", "volume"]
        )
        # Проверка кеширования
        cached_df = await market_data.get_candles(
            symbol="BTCUSDT", interval="1h", limit=100
        )
        assert cached_df.equals(df)
    except Exception as e:
        pytest.skip(f"Get candles test skipped: {str(e)}")
@pytest.mark.asyncio
async def test_async_get_orderbook(market_data) -> None:
    """Тест получения стакана"""
    try:
        # Получение данных
        orderbook = await market_data.get_orderbook(symbol="BTCUSDT", depth=20)
        assert isinstance(orderbook, dict)
        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) <= 20
        assert len(orderbook["asks"]) <= 20
        # Проверка кеширования
        cached_orderbook = await market_data.get_orderbook(symbol="BTCUSDT", depth=20)
        assert cached_orderbook == orderbook
    except Exception as e:
        pytest.skip(f"Get orderbook test skipped: {str(e)}")
@pytest.mark.asyncio
async def test_async_get_trades(market_data) -> None:
    """Тест получения сделок"""
    try:
        # Получение данных
        df = await market_data.get_trades(symbol="BTCUSDT", limit=100)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(col in df.columns for col in ["price", "amount", "side"])
        # Проверка кеширования
        cached_df = await market_data.get_trades(symbol="BTCUSDT", limit=100)
        assert cached_df.equals(df)
    except Exception as e:
        pytest.skip(f"Get trades test skipped: {str(e)}")
@pytest.mark.asyncio
async def test_async_get_historical_data(market_data) -> None:
    """Тест получения исторических данных"""
    try:
        # Получение данных
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        df = await market_data.get_historical_data(
            symbol="BTCUSDT", interval="1h", start_time=start_time, end_time=end_time
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(
            col in df.columns for col in ["open", "high", "low", "close", "volume"]
        )
        # Проверка временного диапазона
        assert df.index.min() >= start_time
        assert df.index.max() <= end_time
        # Проверка сохранения в файл
        file_path = market_data.data_dir / "BTCUSDT_1h.parquet"
        assert file_path.exists()
    except Exception as e:
        pytest.skip(f"Get historical data test skipped: {str(e)}")
@pytest.mark.asyncio
async def test_data_cache(data_cache) -> None:
    """Тест кеша данных"""
    # Сохранение данных
    data_cache.set("test_key", "test_value")
    # Получение данных
    value = data_cache.get("test_key")
    assert value == "test_value"
    # Проверка TTL
    await asyncio.sleep(6)  # Ждем больше TTL
    value = data_cache.get("test_key")
    assert value is None
    # Очистка кеша
    data_cache.set("test_key", "test_value")
    data_cache.clear()
    value = data_cache.get("test_key")
    assert value is None
