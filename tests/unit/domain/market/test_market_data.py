"""
Unit тесты для market_data.

Покрывает:
- Класс OHLCV
- Класс MarketData
- Все методы и свойства
- Сериализацию и десериализацию
- Работу с pandas DataFrame
"""

import pytest
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock

from domain.market.market_data import (
    OHLCV,
    MarketData,
    _default_metadata
)
from domain.market.market_types import Timeframe


class TestOHLCV:
    """Тесты для класса OHLCV."""

    @pytest.fixture
    def sample_ohlcv(self) -> OHLCV:
        """Тестовый OHLCV."""
        return OHLCV(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )

    def test_ohlcv_creation(self, sample_ohlcv):
        """Тест создания OHLCV."""
        assert sample_ohlcv.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert sample_ohlcv.open == 50000.0
        assert sample_ohlcv.high == 51000.0
        assert sample_ohlcv.low == 49000.0
        assert sample_ohlcv.close == 50500.0
        assert sample_ohlcv.volume == 1000.0

    def test_ohlcv_immutable(self, sample_ohlcv):
        """Тест что OHLCV является неизменяемым."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample_ohlcv.open = 51000.0

    def test_to_dict(self, sample_ohlcv):
        """Тест преобразования в словарь."""
        result = sample_ohlcv.to_dict()
        
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert result["open"] == 50000.0
        assert result["high"] == 51000.0
        assert result["low"] == 49000.0
        assert result["close"] == 50500.0
        assert result["volume"] == 1000.0

    def test_from_dict(self):
        """Тест создания из словаря."""
        data = {
            "timestamp": "2024-01-01T12:00:00",
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 1000.0
        }
        
        ohlcv = OHLCV.from_dict(data)
        
        assert ohlcv.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert ohlcv.open == 50000.0
        assert ohlcv.high == 51000.0
        assert ohlcv.low == 49000.0
        assert ohlcv.close == 50500.0
        assert ohlcv.volume == 1000.0

    def test_from_dict_with_string_numbers(self):
        """Тест создания из словаря со строковыми числами."""
        data = {
            "timestamp": "2024-01-01T12:00:00",
            "open": "50000.0",
            "high": "51000.0",
            "low": "49000.0",
            "close": "50500.0",
            "volume": "1000.0"
        }
        
        ohlcv = OHLCV.from_dict(data)
        
        assert ohlcv.open == 50000.0
        assert ohlcv.high == 51000.0
        assert ohlcv.low == 49000.0
        assert ohlcv.close == 50500.0
        assert ohlcv.volume == 1000.0


class TestDefaultMetadata:
    """Тесты для функции _default_metadata."""

    def test_default_metadata(self):
        """Тест функции _default_metadata."""
        metadata = _default_metadata()
        
        assert metadata["source"] == ""
        assert metadata["exchange"] == ""
        assert metadata["extra"] == {}


class TestMarketData:
    """Тесты для класса MarketData."""

    @pytest.fixture
    def sample_market_data(self) -> MarketData:
        """Тестовые рыночные данные."""
        return MarketData(
            symbol="BTC/USD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            quote_volume=50000000.0,
            trades_count=150,
            taker_buy_volume=600.0,
            taker_buy_quote_volume=30000000.0,
            metadata={"source": "binance", "exchange": "binance", "extra": {"test": "value"}}
        )

    @pytest.fixture
    def bullish_market_data(self) -> MarketData:
        """Бычьи рыночные данные."""
        return MarketData(
            symbol="BTC/USD",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )

    @pytest.fixture
    def bearish_market_data(self) -> MarketData:
        """Медвежьи рыночные данные."""
        return MarketData(
            symbol="BTC/USD",
            open=50500.0,
            high=51000.0,
            low=49000.0,
            close=50000.0,
            volume=1000.0
        )

    @pytest.fixture
    def doji_market_data(self) -> MarketData:
        """Доджи рыночные данные."""
        return MarketData(
            symbol="BTC/USD",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=1000.0
        )

    def test_market_data_creation(self, sample_market_data):
        """Тест создания рыночных данных."""
        assert sample_market_data.symbol == "BTC/USD"
        assert sample_market_data.timeframe == Timeframe.MINUTE_1
        assert sample_market_data.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert sample_market_data.open == 50000.0
        assert sample_market_data.high == 51000.0
        assert sample_market_data.low == 49000.0
        assert sample_market_data.close == 50500.0
        assert sample_market_data.volume == 1000.0
        assert sample_market_data.quote_volume == 50000000.0
        assert sample_market_data.trades_count == 150
        assert sample_market_data.taker_buy_volume == 600.0
        assert sample_market_data.taker_buy_quote_volume == 30000000.0
        assert sample_market_data.metadata["source"] == "binance"

    def test_market_data_defaults(self):
        """Тест значений по умолчанию."""
        market_data = MarketData()
        
        assert market_data.symbol == ""
        assert market_data.timeframe == Timeframe.MINUTE_1
        assert isinstance(market_data.timestamp, datetime)
        assert market_data.open == 0.0
        assert market_data.high == 0.0
        assert market_data.low == 0.0
        assert market_data.close == 0.0
        assert market_data.volume == 0.0
        assert market_data.quote_volume is None
        assert market_data.trades_count is None
        assert market_data.taker_buy_volume is None
        assert market_data.taker_buy_quote_volume is None
        assert market_data.metadata == {"source": "", "exchange": "", "extra": {}}

    def test_price_properties(self, sample_market_data):
        """Тест свойств цен."""
        assert sample_market_data.open_price == 50000.0
        assert sample_market_data.high_price == 51000.0
        assert sample_market_data.low_price == 49000.0
        assert sample_market_data.close_price == 50500.0

    def test_get_price_range(self, sample_market_data):
        """Тест получения ценового диапазона."""
        price_range = sample_market_data.get_price_range()
        assert price_range == 2000.0  # 51000.0 - 49000.0

    def test_get_body_size(self, sample_market_data):
        """Тест получения размера тела свечи."""
        body_size = sample_market_data.get_body_size()
        assert body_size == 500.0  # abs(50500.0 - 50000.0)

    def test_get_upper_shadow(self, sample_market_data):
        """Тест получения верхней тени."""
        upper_shadow = sample_market_data.get_upper_shadow()
        assert upper_shadow == 500.0  # 51000.0 - max(50000.0, 50500.0)

    def test_get_lower_shadow(self, sample_market_data):
        """Тест получения нижней тени."""
        lower_shadow = sample_market_data.get_lower_shadow()
        assert lower_shadow == 1000.0  # min(50000.0, 50500.0) - 49000.0

    def test_is_bullish(self, bullish_market_data, bearish_market_data):
        """Тест определения бычьей свечи."""
        assert bullish_market_data.is_bullish() is True
        assert bearish_market_data.is_bullish() is False

    def test_is_bearish(self, bullish_market_data, bearish_market_data):
        """Тест определения медвежьей свечи."""
        assert bullish_market_data.is_bearish() is False
        assert bearish_market_data.is_bearish() is True

    def test_is_doji(self, doji_market_data, sample_market_data):
        """Тест определения доджи."""
        assert doji_market_data.is_doji() is True
        assert sample_market_data.is_doji() is False

    def test_get_volume_price_trend(self, sample_market_data):
        """Тест получения тренда объема-цены."""
        trend = sample_market_data.get_volume_price_trend()
        expected_trend = (50500.0 - 50000.0) / 1000.0
        assert trend == expected_trend

    def test_get_volume_price_trend_zero_volume(self):
        """Тест получения тренда объема-цены при нулевом объеме."""
        market_data = MarketData(volume=0.0)
        trend = market_data.get_volume_price_trend()
        assert trend is None

    def test_to_dict(self, sample_market_data):
        """Тест преобразования в словарь."""
        result = sample_market_data.to_dict()
        
        assert result["symbol"] == "BTC/USD"
        assert result["timeframe"] == "1m"
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert result["open"] == 50000.0
        assert result["high"] == 51000.0
        assert result["low"] == 49000.0
        assert result["close"] == 50500.0
        assert result["volume"] == 1000.0
        assert result["quote_volume"] == 50000000.0
        assert result["trades_count"] == 150
        assert result["taker_buy_volume"] == 600.0
        assert result["taker_buy_quote_volume"] == 30000000.0
        assert result["metadata"]["source"] == "binance"

    def test_from_dict(self):
        """Тест создания из словаря."""
        data = {
            "id": "test-id",
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "timestamp": "2024-01-01T12:00:00",
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 1000.0,
            "quote_volume": 50000000.0,
            "trades_count": 150,
            "taker_buy_volume": 600.0,
            "taker_buy_quote_volume": 30000000.0,
            "metadata": {"source": "binance", "exchange": "binance", "extra": {"test": "value"}}
        }
        
        market_data = MarketData.from_dict(data)
        
        assert market_data.id == "test-id"
        assert market_data.symbol == "BTC/USD"
        assert market_data.timeframe == Timeframe.MINUTE_1
        assert market_data.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert market_data.open == 50000.0
        assert market_data.high == 51000.0
        assert market_data.low == 49000.0
        assert market_data.close == 50500.0
        assert market_data.volume == 1000.0
        assert market_data.quote_volume == 50000000.0
        assert market_data.trades_count == 150
        assert market_data.taker_buy_volume == 600.0
        assert market_data.taker_buy_quote_volume == 30000000.0
        assert market_data.metadata["source"] == "binance"

    def test_from_dict_with_optional_fields(self):
        """Тест создания из словаря с опциональными полями."""
        data = {
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "timestamp": "2024-01-01T12:00:00",
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 1000.0
        }
        
        market_data = MarketData.from_dict(data)
        
        assert market_data.symbol == "BTC/USD"
        assert market_data.quote_volume is None
        assert market_data.trades_count is None
        assert market_data.taker_buy_volume is None
        assert market_data.taker_buy_quote_volume is None
        assert market_data.metadata == {"source": "", "exchange": "", "extra": {}}

    def test_from_dict_with_string_numbers(self):
        """Тест создания из словаря со строковыми числами."""
        data = {
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "timestamp": "2024-01-01T12:00:00",
            "open": "50000.0",
            "high": "51000.0",
            "low": "49000.0",
            "close": "50500.0",
            "volume": "1000.0",
            "quote_volume": "50000000.0",
            "trades_count": "150",
            "taker_buy_volume": "600.0",
            "taker_buy_quote_volume": "30000000.0"
        }
        
        market_data = MarketData.from_dict(data)
        
        assert market_data.open == 50000.0
        assert market_data.high == 51000.0
        assert market_data.low == 49000.0
        assert market_data.close == 50500.0
        assert market_data.volume == 1000.0
        assert market_data.quote_volume == 50000000.0
        assert market_data.trades_count == 150
        assert market_data.taker_buy_volume == 600.0
        assert market_data.taker_buy_quote_volume == 30000000.0

    def test_from_dataframe(self):
        """Тест создания из pandas DataFrame."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01T12:00:00", "2024-01-01T12:01:00"],
            "open": [50000.0, 50500.0],
            "high": [51000.0, 51500.0],
            "low": [49000.0, 49500.0],
            "close": [50500.0, 51000.0],
            "volume": [1000.0, 1200.0],
            "quote_volume": [50000000.0, 60000000.0],
            "trades_count": [150, 180],
            "taker_buy_volume": [600.0, 720.0],
            "taker_buy_quote_volume": [30000000.0, 36000000.0]
        })
        
        result = MarketData.from_dataframe(df, "BTC/USD", Timeframe.MINUTE_1)
        
        assert len(result) == 2
        assert all(isinstance(item, MarketData) for item in result)
        assert result[0].symbol == "BTC/USD"
        assert result[0].timeframe == Timeframe.MINUTE_1
        assert result[0].open == 50000.0
        assert result[1].open == 50500.0

    def test_from_dataframe_with_datetime_timestamp(self):
        """Тест создания из DataFrame с datetime timestamp."""
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 1, 0)],
            "open": [50000.0, 50500.0],
            "high": [51000.0, 51500.0],
            "low": [49000.0, 49500.0],
            "close": [50500.0, 51000.0],
            "volume": [1000.0, 1200.0]
        })
        
        result = MarketData.from_dataframe(df, "BTC/USD", Timeframe.MINUTE_1)
        
        assert len(result) == 2
        assert result[0].timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert result[1].timestamp == datetime(2024, 1, 1, 12, 1, 0)

    def test_from_dataframe_with_missing_optional_fields(self):
        """Тест создания из DataFrame с отсутствующими опциональными полями."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01T12:00:00"],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        result = MarketData.from_dataframe(df, "BTC/USD", Timeframe.MINUTE_1)
        
        assert len(result) == 1
        assert result[0].quote_volume is None
        assert result[0].trades_count is None
        assert result[0].taker_buy_volume is None
        assert result[0].taker_buy_quote_volume is None

    def test_unique_id_generation(self):
        """Тест генерации уникальных ID."""
        market_data1 = MarketData()
        market_data2 = MarketData()
        
        assert market_data1.id != market_data2.id
        assert isinstance(market_data1.id, str)
        assert isinstance(market_data2.id, str)

    def test_candle_analysis_methods(self, bullish_market_data, bearish_market_data, doji_market_data):
        """Тест методов анализа свечей."""
        # Тест бычьей свечи
        assert bullish_market_data.is_bullish() is True
        assert bullish_market_data.is_bearish() is False
        assert bullish_market_data.get_body_size() == 500.0
        assert bullish_market_data.get_upper_shadow() == 500.0
        assert bullish_market_data.get_lower_shadow() == 1000.0
        
        # Тест медвежьей свечи
        assert bearish_market_data.is_bullish() is False
        assert bearish_market_data.is_bearish() is True
        assert bearish_market_data.get_body_size() == 500.0
        assert bearish_market_data.get_upper_shadow() == 500.0
        assert bearish_market_data.get_lower_shadow() == 1000.0
        
        # Тест доджи
        assert doji_market_data.is_doji() is True
        assert doji_market_data.get_body_size() == 0.0
        assert doji_market_data.get_upper_shadow() == 100.0
        assert doji_market_data.get_lower_shadow() == 100.0

    def test_protocol_compliance(self, sample_market_data):
        """Тест соответствия протоколу MarketDataProtocol."""
        from domain.market.market_protocols import MarketDataProtocol
        assert isinstance(sample_market_data, MarketDataProtocol)


class TestMarketDataIntegration:
    """Интеграционные тесты для MarketData."""

    def test_full_workflow(self):
        """Тест полного рабочего процесса."""
        # Создаем рыночные данные
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        # Проверяем свойства
        assert market_data.open_price == 50000.0
        assert market_data.close_price == 50500.0
        
        # Проверяем анализ свечи
        assert market_data.is_bullish() is True
        assert market_data.get_price_range() == 2000.0
        assert market_data.get_body_size() == 500.0
        
        # Преобразуем в словарь и обратно
        data_dict = market_data.to_dict()
        restored_market_data = MarketData.from_dict(data_dict)
        
        assert restored_market_data.symbol == market_data.symbol
        assert restored_market_data.open == market_data.open
        assert restored_market_data.close == market_data.close

    def test_dataframe_workflow(self):
        """Тест рабочего процесса с DataFrame."""
        # Создаем DataFrame
        df = pd.DataFrame({
            "timestamp": ["2024-01-01T12:00:00", "2024-01-01T12:01:00"],
            "open": [50000.0, 50500.0],
            "high": [51000.0, 51500.0],
            "low": [49000.0, 49500.0],
            "close": [50500.0, 51000.0],
            "volume": [1000.0, 1200.0]
        })
        
        # Создаем MarketData из DataFrame
        market_data_list = MarketData.from_dataframe(df, "BTC/USD", Timeframe.MINUTE_1)
        
        # Проверяем результаты
        assert len(market_data_list) == 2
        assert all(isinstance(item, MarketData) for item in market_data_list)
        assert market_data_list[0].symbol == "BTC/USD"
        assert market_data_list[1].symbol == "BTC/USD"
        
        # Проверяем анализ
        assert market_data_list[0].is_bullish() is True
        assert market_data_list[1].is_bullish() is True 