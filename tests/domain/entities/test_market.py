"""Тесты для Market entities."""
import pytest
import pandas as pd
import dataclasses
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal
from uuid import uuid4, UUID
from domain.entities.market import (
    Market,
    OHLCV,
    MarketData,
    MarketState,
    MarketRegime,
    Timeframe,
)
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.volume import Volume
class TestMarket:
    """Тесты для класса Market."""
    def test_market_creation(self) -> None:
        """Тест создания рынка."""
        market = Market(
            symbol="BTCUSDT",
            name="Bitcoin/USDT",
            is_active=True
        )
        assert market.symbol == "BTCUSDT"
        assert market.name == "Bitcoin/USDT"
        assert market.is_active is True
        assert isinstance(market.id, UUID)
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
    def test_market_default_values(self) -> None:
        """Тест значений по умолчанию."""
        market = Market()
        assert market.symbol == ""
        assert market.name == ""
        assert market.is_active is True
        assert market.metadata == {}
    def test_market_serialization(self) -> None:
        """Тест сериализации."""
        market = Market(
            symbol="BTCUSDT",
            name="Bitcoin/USDT",
            is_active=True,
            metadata={"exchange": "binance"}
        )
        data = market.to_dict()
        assert data["symbol"] == "BTCUSDT"
        assert data["name"] == "Bitcoin/USDT"
        assert data["is_active"] is True
        assert data["metadata"]["exchange"] == "binance"
        restored = Market.from_dict(data)
        assert restored.symbol == market.symbol
        assert restored.name == market.name
        assert restored.is_active == market.is_active
        assert restored.metadata == market.metadata
class TestOHLCV:
    """Тесты для класса OHLCV."""
    def test_ohlcv_creation(self) -> None:
        """Тест создания OHLCV."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("95.0"),
            close=Decimal("105.0"),
            volume=Decimal("1000.0")
        )
        assert ohlcv.timestamp == timestamp
        assert ohlcv.open == Decimal("100.0")
        assert ohlcv.high == Decimal("110.0")
        assert ohlcv.low == Decimal("95.0")
        assert ohlcv.close == Decimal("105.0")
        assert ohlcv.volume == Decimal("1000.0")
    def test_ohlcv_immutability(self) -> None:
        """Тест неизменяемости."""
        ohlcv = OHLCV(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("95.0"),
            close=Decimal("105.0"),
            volume=Decimal("1000.0")
        )
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(dataclasses.FrozenInstanceError):
            ohlcv.open = Decimal("200.0")
    def test_ohlcv_serialization(self) -> None:
        """Тест сериализации."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("95.0"),
            close=Decimal("105.0"),
            volume=Decimal("1000.0")
        )
        data = ohlcv.to_dict()
        assert data["timestamp"] == timestamp.isoformat()
        assert data["open"] == "100.0"
        assert data["high"] == "110.0"
        assert data["low"] == "95.0"
        assert data["close"] == "105.0"
        assert data["volume"] == "1000.0"
        restored = OHLCV.from_dict(data)
        assert restored == ohlcv
class TestMarketData:
    """Тесты для класса MarketData."""
    def test_market_data_creation(self) -> None:
        """Тест создания рыночных данных."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        market_data = MarketData(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=timestamp,
            open=Price(Decimal("100.0"), Currency.USD),
            high=Price(Decimal("110.0"), Currency.USD),
            low=Price(Decimal("95.0"), Currency.USD),
            close=Price(Decimal("105.0"), Currency.USD),
            volume=Volume(Decimal("1000.0"))
        )
        assert market_data.symbol == "BTCUSDT"
        assert market_data.timeframe == Timeframe.MINUTE_1
        assert market_data.timestamp == timestamp
        assert market_data.open.value == Decimal("100.0")
        assert market_data.high.value == Decimal("110.0")
        assert market_data.low.value == Decimal("95.0")
        assert market_data.close.value == Decimal("105.0")
        assert market_data.volume.value == Decimal("1000.0")
    def test_market_data_properties(self) -> None:
        """Тест свойств рыночных данных."""
        market_data = MarketData(
            open=Price(Decimal("100.0"), Currency.USD),
            high=Price(Decimal("110.0"), Currency.USD),
            low=Price(Decimal("95.0"), Currency.USD),
            close=Price(Decimal("105.0"), Currency.USD),
            volume=Volume(Decimal("1000.0"))
        )
        assert market_data.open_price == Price(Decimal("100.0"), Currency.USD)
        assert market_data.high_price == Price(Decimal("110.0"), Currency.USD)
        assert market_data.low_price == Price(Decimal("95.0"), Currency.USD)
        assert market_data.close_price == Price(Decimal("105.0"), Currency.USD)
    def test_market_data_price_analysis(self) -> None:
        """Тест анализа цен."""
        market_data = MarketData(
            open=Price(Decimal("100.0"), Currency.USD),
            high=Price(Decimal("110.0"), Currency.USD),
            low=Price(Decimal("95.0"), Currency.USD),
            close=Price(Decimal("105.0"), Currency.USD),
            volume=Volume(Decimal("1000.0"))
        )
        # Диапазон цен
        price_range = market_data.get_price_range()
        assert price_range.value == Decimal("15.0")  # 110 - 95
        # Размер тела свечи
        body_size = market_data.get_body_size()
        assert body_size.value == Decimal("5.0")  # |105 - 100|
        # Верхняя тень
        upper_shadow = market_data.get_upper_shadow()
        assert upper_shadow.value == Decimal("5.0")  # 110 - 105
        # Нижняя тень
        lower_shadow = market_data.get_lower_shadow()
        assert lower_shadow.value == Decimal("5.0")  # 100 - 95
    def test_market_data_candle_patterns(self) -> None:
        """Тест паттернов свечей."""
        # Бычья свеча
        bullish_data = MarketData(
            open=Price(Decimal("100.0"), Currency.USD),
            close=Price(Decimal("105.0"), Currency.USD)
        )
        assert bullish_data.is_bullish() is True
        assert bullish_data.is_bearish() is False
        # Медвежья свеча
        bearish_data = MarketData(
            open=Price(Decimal("105.0"), Currency.USD),
            close=Price(Decimal("100.0"), Currency.USD)
        )
        assert bearish_data.is_bullish() is False
        assert bearish_data.is_bearish() is True
        # Доджи
        doji_data = MarketData(
            open=Price(Decimal("100.0"), Currency.USD),
            high=Price(Decimal("101.0"), Currency.USD),
            low=Price(Decimal("99.0"), Currency.USD),
            close=Price(Decimal("100.1"), Currency.USD)
        )
        assert doji_data.is_doji() is True
    def test_market_data_volume_trend(self) -> None:
        """Тест тренда объема."""
        market_data = MarketData(
            volume=Volume(Decimal("1000.0")),
            quote_volume=Volume(Decimal("100000.0"))
        )
        trend = market_data.get_volume_price_trend()
        assert trend == Decimal("100.0")  # 100000 / 1000
        # Без quote_volume
        market_data_no_quote = MarketData(volume=Volume(Decimal("1000.0")))
        trend = market_data_no_quote.get_volume_price_trend()
        assert trend is None
    def test_market_data_serialization(self) -> None:
        """Тест сериализации."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        market_data = MarketData(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=timestamp,
            open=Price(Decimal("100.0"), Currency.USD),
            high=Price(Decimal("110.0"), Currency.USD),
            low=Price(Decimal("95.0"), Currency.USD),
            close=Price(Decimal("105.0"), Currency.USD),
            volume=Volume(Decimal("1000.0")),
            metadata={"source": "binance"}
        )
        data = market_data.to_dict()
        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1m"
        assert data["metadata"]["source"] == "binance"
        restored = MarketData.from_dict(data)
        assert restored.symbol == market_data.symbol
        assert restored.timeframe == market_data.timeframe
    def test_market_data_from_dataframe(self) -> None:
        """Тест создания из DataFrame."""
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 12, 0, 0)],
            'open': [100.0],
            'high': [110.0],
            'low': [95.0],
            'close': [105.0],
            'volume': [1000.0]
        })
        market_data_list = MarketData.from_dataframe(
            df, "BTCUSDT", Timeframe.MINUTE_1
        )
        assert len(market_data_list) == 1
        assert market_data_list[0].symbol == "BTCUSDT"
        assert market_data_list[0].timeframe == Timeframe.MINUTE_1
class TestMarketState:
    """Тесты для класса MarketState."""
    def test_market_state_creation(self) -> None:
        """Тест создания состояния рынка."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        market_state = MarketState(
            symbol="BTCUSDT",
            timestamp=timestamp,
            regime=MarketRegime.TRENDING_UP,
            volatility=Decimal("0.05"),
            trend_strength=Decimal("0.8"),
            support_level=Price(Decimal("100.0"), Currency.USD),
            resistance_level=Price(Decimal("110.0"), Currency.USD)
        )
        assert market_state.symbol == "BTCUSDT"
        assert market_state.timestamp == timestamp
        assert market_state.regime == MarketRegime.TRENDING_UP
        assert market_state.volatility == Decimal("0.05")
        assert market_state.trend_strength == Decimal("0.8")
        assert market_state.support_level.value == Decimal("100.0")
        assert market_state.resistance_level.value == Decimal("110.0")
    def test_market_state_regime_checks(self) -> None:
        """Тест проверок режима рынка."""
        # Трендовый режим
        trending_state = MarketState(regime=MarketRegime.TRENDING_UP)
        assert trending_state.is_trending() is True
        assert trending_state.is_sideways() is False
        assert trending_state.is_volatile() is False
        assert trending_state.is_breakout() is False
        # Боковой режим
        sideways_state = MarketState(regime=MarketRegime.SIDEWAYS)
        assert sideways_state.is_trending() is False
        assert sideways_state.is_sideways() is True
        # Волатильный режим
        volatile_state = MarketState(regime=MarketRegime.VOLATILE)
        assert volatile_state.is_volatile() is True
        # Режим пробоя
        breakout_state = MarketState(regime=MarketRegime.BREAKOUT)
        assert breakout_state.is_breakout() is True
    def test_market_state_trend_direction(self) -> None:
        """Тест направления тренда."""
        # Восходящий тренд
        up_trend = MarketState(regime=MarketRegime.TRENDING_UP)
        assert up_trend.get_trend_direction() == "up"
        # Нисходящий тренд
        down_trend = MarketState(regime=MarketRegime.TRENDING_DOWN)
        assert down_trend.get_trend_direction() == "down"
        # Боковой тренд
        sideways = MarketState(regime=MarketRegime.SIDEWAYS)
        assert sideways.get_trend_direction() is None
    def test_market_state_price_position(self) -> None:
        """Тест позиции цены."""
        market_state = MarketState(
            support_level=Price(Decimal("100.0"), Currency.USD),
            resistance_level=Price(Decimal("110.0"), Currency.USD),
            pivot_point=Price(Decimal("105.0"), Currency.USD)
        )
        # Цена выше сопротивления
        position = market_state.get_price_position(Price(Decimal("115.0"), Currency.USD))
        assert position == "above_resistance"
        # Цена ниже поддержки
        position = market_state.get_price_position(Price(Decimal("95.0"), Currency.USD))
        assert position == "below_support"
        # Цена в диапазоне
        position = market_state.get_price_position(Price(Decimal("105.0"), Currency.USD))
        assert position == "in_range"
    def test_market_state_overbought_oversold(self) -> None:
        """Тест перекупленности/перепроданности."""
        # Перекупленность
        overbought_state = MarketState(rsi=Decimal("75.0"))
        assert overbought_state.is_overbought() is True
        assert overbought_state.is_oversold() is False
        # Перепроданность
        oversold_state = MarketState(rsi=Decimal("25.0"))
        assert oversold_state.is_overbought() is False
        assert oversold_state.is_oversold() is True
        # Нейтральное состояние
        neutral_state = MarketState(rsi=Decimal("50.0"))
        assert neutral_state.is_overbought() is False
        assert neutral_state.is_oversold() is False
    def test_market_state_serialization(self) -> None:
        """Тест сериализации."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        market_state = MarketState(
            symbol="BTCUSDT",
            timestamp=timestamp,
            regime=MarketRegime.TRENDING_UP,
            volatility=Decimal("0.05"),
            support_level=Price(Decimal("100.0"), Currency.USD),
            resistance_level=Price(Decimal("110.0"), Currency.USD),
            metadata={"source": "analysis"}
        )
        data = market_state.to_dict()
        assert data["symbol"] == "BTCUSDT"
        assert data["regime"] == "trending_up"
        assert data["metadata"]["source"] == "analysis"
        restored = MarketState.from_dict(data)
        assert restored.symbol == market_state.symbol
        assert restored.regime == market_state.regime
class TestMarketRegime:
    """Тесты для перечисления MarketRegime."""
    def test_market_regime_values(self) -> None:
        """Тест значений рыночных режимов."""
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.TRENDING_DOWN.value == "trending_down"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.BREAKOUT.value == "breakout"
        assert MarketRegime.UNKNOWN.value == "unknown"
class TestTimeframe:
    """Тесты для перечисления Timeframe."""
    def test_timeframe_values(self) -> None:
        """Тест значений временных интервалов."""
        assert Timeframe.TICK.value == "tick"
        assert Timeframe.SECOND_1.value == "1s"
        assert Timeframe.MINUTE_1.value == "1m"
        assert Timeframe.HOUR_1.value == "1h"
        assert Timeframe.DAY_1.value == "1d"
        assert Timeframe.WEEK_1.value == "1w"
        assert Timeframe.MONTH_1.value == "1M" 
