"""
Unit тесты для Market.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.market import (
    Market, MarketData, MarketState, OHLCV, TechnicalIndicator,
    OrderBook, OrderBookEntry, Trade, MarketSnapshot,
    MarketRegime, Timeframe, MarketProtocol, MarketDataProtocol, MarketStateProtocol
)
from domain.type_definitions import (
    MarketId, MarketName, Symbol, MetadataDict, TimestampValue,
    VolatilityValue, TrendStrengthValue, VolumeTrendValue, PriceMomentumValue,
    RSIMetric, MACDMetric, ATRMetric
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.type_definitions.common_types import Timestamp


class TestMarket:
    """Тесты для Market."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": MarketId(uuid4()),
            "symbol": Symbol("BTC/USDT"),
            "name": MarketName("Bitcoin Market"),
            "is_active": True,
            "metadata": {"exchange": "binance", "type": "spot"}
        }
    
    @pytest.fixture
    def market(self, sample_data) -> Market:
        """Создает тестовый рынок."""
        return Market(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания рынка."""
        market = Market(**sample_data)
        
        assert market.id == sample_data["id"]
        assert market.symbol == sample_data["symbol"]
        assert market.name == sample_data["name"]
        assert market.is_active == sample_data["is_active"]
        assert market.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания рынка с дефолтными значениями."""
        market = Market()
        
        assert isinstance(market.id, MarketId)
        assert market.symbol == Symbol("")
        assert market.name == MarketName("")
        assert market.is_active is True
        assert market.metadata == MetadataDict({})
    
    def test_to_dict(self, market):
        """Тест сериализации в словарь."""
        data = market.to_dict()
        
        assert data["id"] == str(market.id)
        assert data["symbol"] == str(market.symbol)
        assert data["name"] == str(market.name)
        assert data["is_active"] == market.is_active
        assert "created_at" in data
        assert "updated_at" in data
        assert "metadata" in data
    
    def test_from_dict(self, market):
        """Тест десериализации из словаря."""
        data = market.to_dict()
        new_market = Market.from_dict(data)
        
        assert new_market.id == market.id
        assert new_market.symbol == market.symbol
        assert new_market.name == market.name
        assert new_market.is_active == market.is_active
        assert new_market.metadata == market.metadata
    
    def test_market_protocol_compliance(self, market):
        """Тест соответствия протоколу MarketProtocol."""
        assert isinstance(market, MarketProtocol)
    
    def test_market_regime_enum(self):
        """Тест enum MarketRegime."""
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.TRENDING_DOWN.value == "trending_down"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.BREAKOUT.value == "breakout"
        assert MarketRegime.UNKNOWN.value == "unknown"
    
    def test_timeframe_enum(self):
        """Тест enum Timeframe."""
        assert Timeframe.TICK.value == "tick"
        assert Timeframe.MINUTE_1.value == "1m"
        assert Timeframe.HOUR_1.value == "1h"
        assert Timeframe.DAY_1.value == "1d"


class TestMarketData:
    """Тесты для MarketData."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": MarketId(uuid4()),
            "symbol": Symbol("BTC/USDT"),
            "timeframe": Timeframe.MINUTE_1,
            "timestamp": TimestampValue(datetime.now()),
            "open": Price(Decimal("50000"), Currency.USDT, Currency.USDT),
            "high": Price(Decimal("51000"), Currency.USDT, Currency.USDT),
            "low": Price(Decimal("49000"), Currency.USDT, Currency.USDT),
            "close": Price(Decimal("50500"), Currency.USDT, Currency.USDT),
            "volume": Volume(Decimal("1000"), Currency.USDT),
            "metadata": {"source": "binance"}
        }
    
    @pytest.fixture
    def market_data(self, sample_data) -> MarketData:
        """Создает тестовые рыночные данные."""
        return MarketData(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания рыночных данных."""
        data = MarketData(**sample_data)
        
        assert data.id == sample_data["id"]
        assert data.symbol == sample_data["symbol"]
        assert data.timeframe == sample_data["timeframe"]
        assert data.open == sample_data["open"]
        assert data.high == sample_data["high"]
        assert data.low == sample_data["low"]
        assert data.close == sample_data["close"]
        assert data.volume == sample_data["volume"]
    
    def test_default_creation(self):
        """Тест создания с дефолтными значениями."""
        data = MarketData()
        
        assert isinstance(data.id, MarketId)
        assert data.symbol == Symbol("")
        assert data.timeframe == Timeframe.MINUTE_1
        assert data.open.value == Decimal("0")
        assert data.high.value == Decimal("0")
        assert data.low.value == Decimal("0")
        assert data.close.value == Decimal("0")
        assert data.volume.value == Decimal("0")
    
    def test_price_properties(self, market_data):
        """Тест свойств цен."""
        assert market_data.open_price == market_data.open
        assert market_data.high_price == market_data.high
        assert market_data.low_price == market_data.low
        assert market_data.close_price == market_data.close
    
    def test_get_price_range(self, market_data):
        """Тест получения диапазона цен."""
        price_range = market_data.get_price_range()
        expected_range = market_data.high.value - market_data.low.value
        assert price_range.value == expected_range
    
    def test_get_body_size(self, market_data):
        """Тест получения размера тела свечи."""
        body_size = market_data.get_body_size()
        expected_size = abs(market_data.close.value - market_data.open.value)
        assert body_size.value == expected_size
    
    def test_get_upper_shadow(self, market_data):
        """Тест получения верхней тени."""
        upper_shadow = market_data.get_upper_shadow()
        body_high = max(market_data.open.value, market_data.close.value)
        expected_shadow = market_data.high.value - body_high
        assert upper_shadow.value == expected_shadow
    
    def test_get_lower_shadow(self, market_data):
        """Тест получения нижней тени."""
        lower_shadow = market_data.get_lower_shadow()
        body_low = min(market_data.open.value, market_data.close.value)
        expected_shadow = body_low - market_data.low.value
        assert lower_shadow.value == expected_shadow
    
    def test_is_bullish(self, market_data):
        """Тест проверки бычьей свечи."""
        # Устанавливаем close > open для бычьей свечи
        market_data.close = Price(Decimal("51000"), Currency.USDT, Currency.USDT)
        market_data.open = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_bullish() is True
        
        # Устанавливаем close < open для медвежьей свечи
        market_data.close = Price(Decimal("49000"), Currency.USDT, Currency.USDT)
        market_data.open = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_bullish() is False
    
    def test_is_bearish(self, market_data):
        """Тест проверки медвежьей свечи."""
        # Устанавливаем close < open для медвежьей свечи
        market_data.close = Price(Decimal("49000"), Currency.USDT, Currency.USDT)
        market_data.open = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_bearish() is True
        
        # Устанавливаем close > open для бычьей свечи
        market_data.close = Price(Decimal("51000"), Currency.USDT, Currency.USDT)
        market_data.open = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_bearish() is False
    
    def test_is_doji(self, market_data):
        """Тест проверки доджи."""
        # Устанавливаем близкие значения open и close
        market_data.open = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        market_data.close = Price(Decimal("50001"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_doji() is True
        
        # Устанавливаем разные значения open и close
        market_data.close = Price(Decimal("51000"), Currency.USDT, Currency.USDT)
        
        assert market_data.is_doji() is False
    
    def test_get_volume_price_trend(self, market_data):
        """Тест получения тренда объема и цены."""
        # Устанавливаем quote_volume
        market_data.quote_volume = Volume(Decimal("50000000"), Currency.USDT)
        
        vpt = market_data.get_volume_price_trend()
        expected_vpt = market_data.volume.value * (
            (market_data.close.value - market_data.open.value) / market_data.open.value
        )
        assert vpt == expected_vpt
    
    def test_get_volume_price_trend_no_quote_volume(self, market_data):
        """Тест получения VPT без quote_volume."""
        market_data.quote_volume = None
        
        vpt = market_data.get_volume_price_trend()
        assert vpt is None
    
    def test_to_dict(self, market_data):
        """Тест сериализации в словарь."""
        data = market_data.to_dict()
        
        assert data["id"] == str(market_data.id)
        assert data["symbol"] == str(market_data.symbol)
        assert data["timeframe"] == market_data.timeframe.value
        assert "timestamp" in data
        assert "open" in data
        assert "high" in data
        assert "low" in data
        assert "close" in data
        assert "volume" in data
        assert "metadata" in data
    
    def test_from_dict(self, market_data):
        """Тест десериализации из словаря."""
        data = market_data.to_dict()
        new_data = MarketData.from_dict(data)
        
        assert new_data.id == market_data.id
        assert new_data.symbol == market_data.symbol
        assert new_data.timeframe == market_data.timeframe
        assert new_data.open.value == market_data.open.value
        assert new_data.high.value == market_data.high.value
        assert new_data.low.value == market_data.low.value
        assert new_data.close.value == market_data.close.value
        assert new_data.volume.value == market_data.volume.value
    
    def test_from_dataframe(self):
        """Тест создания из DataFrame."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='1min'),
            'open': [50000, 50100, 50200],
            'high': [51000, 51100, 51200],
            'low': [49000, 49100, 49200],
            'close': [50500, 50600, 50700],
            'volume': [1000, 1100, 1200]
        })
        
        market_data_list = MarketData.from_dataframe(df, "BTC/USDT", Timeframe.MINUTE_1)
        
        assert len(market_data_list) == 3
        assert all(isinstance(data, MarketData) for data in market_data_list)
        assert all(data.symbol == Symbol("BTC/USDT") for data in market_data_list)
        assert all(data.timeframe == Timeframe.MINUTE_1 for data in market_data_list)
    
    def test_market_data_protocol_compliance(self, market_data):
        """Тест соответствия протоколу MarketDataProtocol."""
        assert isinstance(market_data, MarketDataProtocol)


class TestMarketState:
    """Тесты для MarketState."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": MarketId(uuid4()),
            "symbol": Symbol("BTC/USDT"),
            "timestamp": TimestampValue(datetime.now()),
            "regime": MarketRegime.TRENDING_UP,
            "volatility": VolatilityValue(Decimal("0.15")),
            "trend_strength": TrendStrengthValue(Decimal("0.8")),
            "volume_trend": VolumeTrendValue(Decimal("0.6")),
            "price_momentum": PriceMomentumValue(Decimal("0.7")),
            "support_level": Price(Decimal("49000"), Currency.USDT, Currency.USDT),
            "resistance_level": Price(Decimal("51000"), Currency.USDT, Currency.USDT),
            "pivot_point": Price(Decimal("50000"), Currency.USDT, Currency.USDT),
            "metadata": {"analysis": "technical"}
        }
    
    @pytest.fixture
    def market_state(self, sample_data) -> MarketState:
        """Создает тестовое состояние рынка."""
        return MarketState(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания состояния рынка."""
        state = MarketState(**sample_data)
        
        assert state.id == sample_data["id"]
        assert state.symbol == sample_data["symbol"]
        assert state.regime == sample_data["regime"]
        assert state.volatility == sample_data["volatility"]
        assert state.trend_strength == sample_data["trend_strength"]
        assert state.volume_trend == sample_data["volume_trend"]
        assert state.price_momentum == sample_data["price_momentum"]
        assert state.support_level == sample_data["support_level"]
        assert state.resistance_level == sample_data["resistance_level"]
        assert state.pivot_point == sample_data["pivot_point"]
    
    def test_default_creation(self):
        """Тест создания с дефолтными значениями."""
        state = MarketState()
        
        assert isinstance(state.id, MarketId)
        assert state.symbol == Symbol("")
        assert state.regime == MarketRegime.UNKNOWN
        assert state.volatility.value == Decimal("0")
        assert state.trend_strength.value == Decimal("0")
        assert state.volume_trend.value == Decimal("0")
        assert state.price_momentum.value == Decimal("0")
    
    def test_is_trending(self, market_state):
        """Тест проверки трендового режима."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.is_trending() is True
        
        market_state.regime = MarketRegime.TRENDING_DOWN
        assert market_state.is_trending() is True
        
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_trending() is False
    
    def test_is_sideways(self, market_state):
        """Тест проверки бокового режима."""
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_sideways() is True
        
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.is_sideways() is False
    
    def test_is_volatile(self, market_state):
        """Тест проверки волатильного режима."""
        market_state.regime = MarketRegime.VOLATILE
        assert market_state.is_volatile() is True
        
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_volatile() is False
    
    def test_is_breakout(self, market_state):
        """Тест проверки режима пробоя."""
        market_state.regime = MarketRegime.BREAKOUT
        assert market_state.is_breakout() is True
        
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_breakout() is False
    
    def test_get_trend_direction(self, market_state):
        """Тест получения направления тренда."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.get_trend_direction() == "up"
        
        market_state.regime = MarketRegime.TRENDING_DOWN
        assert market_state.get_trend_direction() == "down"
        
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.get_trend_direction() is None
    
    def test_get_price_position(self, market_state):
        """Тест получения позиции цены."""
        current_price = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        
        # Цена на уровне поддержки
        market_state.support_level = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        assert market_state.get_price_position(current_price) == "support"
        
        # Цена на уровне сопротивления
        market_state.resistance_level = Price(Decimal("50000"), Currency.USDT, Currency.USDT)
        assert market_state.get_price_position(current_price) == "resistance"
        
        # Цена между уровнями
        market_state.support_level = Price(Decimal("49000"), Currency.USDT, Currency.USDT)
        market_state.resistance_level = Price(Decimal("51000"), Currency.USDT, Currency.USDT)
        assert market_state.get_price_position(current_price) == "between"
    
    def test_is_overbought(self, market_state):
        """Тест проверки перекупленности."""
        market_state.rsi = RSIMetric(Decimal("75"))
        assert market_state.is_overbought() is True
        
        market_state.rsi = RSIMetric(Decimal("65"))
        assert market_state.is_overbought() is False
    
    def test_is_oversold(self, market_state):
        """Тест проверки перепроданности."""
        market_state.rsi = RSIMetric(Decimal("25"))
        assert market_state.is_oversold() is True
        
        market_state.rsi = RSIMetric(Decimal("35"))
        assert market_state.is_oversold() is False
    
    def test_to_dict(self, market_state):
        """Тест сериализации в словарь."""
        data = market_state.to_dict()
        
        assert data["id"] == str(market_state.id)
        assert data["symbol"] == str(market_state.symbol)
        assert data["regime"] == market_state.regime.value
        assert "timestamp" in data
        assert "volatility" in data
        assert "trend_strength" in data
        assert "volume_trend" in data
        assert "price_momentum" in data
        assert "metadata" in data
    
    def test_from_dict(self, market_state):
        """Тест десериализации из словаря."""
        data = market_state.to_dict()
        new_state = MarketState.from_dict(data)
        
        assert new_state.id == market_state.id
        assert new_state.symbol == market_state.symbol
        assert new_state.regime == market_state.regime
        assert new_state.volatility.value == market_state.volatility.value
        assert new_state.trend_strength.value == market_state.trend_strength.value
        assert new_state.volume_trend.value == market_state.volume_trend.value
        assert new_state.price_momentum.value == market_state.price_momentum.value
    
    def test_market_state_protocol_compliance(self, market_state):
        """Тест соответствия протоколу MarketStateProtocol."""
        assert isinstance(market_state, MarketStateProtocol)


class TestTechnicalIndicator:
    """Тесты для TechnicalIndicator."""
    
    def test_creation(self):
        """Тест создания технического индикатора."""
        indicator = TechnicalIndicator(
            name="RSI",
            value=65.5,
            signal="BUY",
            strength=0.8,
            metadata={"period": 14}
        )
        
        assert indicator.name == "RSI"
        assert indicator.value == 65.5
        assert indicator.signal == "BUY"
        assert indicator.strength == 0.8
        assert indicator.metadata == {"period": 14}
    
    def test_validation_strength_out_of_range(self):
        """Тест валидации силы вне диапазона."""
        with pytest.raises(ValueError, match="Strength must be between 0.0 and 1.0"):
            TechnicalIndicator(
                name="RSI",
                value=65.5,
                signal="BUY",
                strength=1.5
            )
    
    def test_validation_invalid_signal(self):
        """Тест валидации некорректного сигнала."""
        with pytest.raises(ValueError, match="Signal must be BUY, SELL, or HOLD"):
            TechnicalIndicator(
                name="RSI",
                value=65.5,
                signal="INVALID",
                strength=0.8
            )
    
    def test_to_dict(self):
        """Тест сериализации в словарь."""
        indicator = TechnicalIndicator(
            name="RSI",
            value=65.5,
            signal="BUY",
            strength=0.8,
            metadata={"period": 14}
        )
        
        data = indicator.to_dict()
        
        assert data["name"] == "RSI"
        assert data["value"] == 65.5
        assert data["signal"] == "BUY"
        assert data["strength"] == 0.8
        assert data["metadata"] == {"period": 14}
    
    def test_from_dict(self):
        """Тест десериализации из словаря."""
        data = {
            "name": "RSI",
            "value": 65.5,
            "signal": "BUY",
            "strength": 0.8,
            "metadata": {"period": 14}
        }
        
        indicator = TechnicalIndicator.from_dict(data)
        
        assert indicator.name == "RSI"
        assert indicator.value == 65.5
        assert indicator.signal == "BUY"
        assert indicator.strength == 0.8
        assert indicator.metadata == {"period": 14}


class TestOrderBook:
    """Тесты для OrderBook."""
    
    @pytest.fixture
    def order_book(self) -> OrderBook:
        """Создает тестовый order book."""
        bids = [
            OrderBookEntry(
                price=Price(Decimal("50000"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("1.0"), Currency.BTC),
                timestamp=Timestamp(datetime.now())
            ),
            OrderBookEntry(
                price=Price(Decimal("49900"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("2.0"), Currency.BTC),
                timestamp=Timestamp(datetime.now())
            )
        ]
        asks = [
            OrderBookEntry(
                price=Price(Decimal("50100"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("1.5"), Currency.BTC),
                timestamp=Timestamp(datetime.now())
            ),
            OrderBookEntry(
                price=Price(Decimal("50200"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("2.5"), Currency.BTC),
                timestamp=Timestamp(datetime.now())
            )
        ]
        
        return OrderBook(
            symbol=Currency.BTC,
            bids=bids,
            asks=asks,
            timestamp=Timestamp(datetime.now())
        )
    
    def test_get_best_bid(self, order_book):
        """Тест получения лучшего bid."""
        best_bid = order_book.get_best_bid()
        assert best_bid.price.value == Decimal("50000")
    
    def test_get_best_bid_empty(self):
        """Тест получения лучшего bid при пустом списке."""
        order_book = OrderBook(symbol=Currency.BTC)
        best_bid = order_book.get_best_bid()
        assert best_bid is None
    
    def test_get_best_ask(self, order_book):
        """Тест получения лучшего ask."""
        best_ask = order_book.get_best_ask()
        assert best_ask.price.value == Decimal("50100")
    
    def test_get_best_ask_empty(self):
        """Тест получения лучшего ask при пустом списке."""
        order_book = OrderBook(symbol=Currency.BTC)
        best_ask = order_book.get_best_ask()
        assert best_ask is None
    
    def test_get_spread(self, order_book):
        """Тест получения спреда."""
        spread = order_book.get_spread()
        expected_spread = Decimal("50100") - Decimal("50000")
        assert spread == expected_spread
    
    def test_get_spread_no_orders(self):
        """Тест получения спреда без ордеров."""
        order_book = OrderBook(symbol=Currency.BTC)
        spread = order_book.get_spread()
        assert spread is None
    
    def test_get_mid_price(self, order_book):
        """Тест получения средней цены."""
        mid_price = order_book.get_mid_price()
        expected_mid_price = (Decimal("50100") + Decimal("50000")) / 2
        assert mid_price == expected_mid_price
    
    def test_get_mid_price_no_orders(self):
        """Тест получения средней цены без ордеров."""
        order_book = OrderBook(symbol=Currency.BTC)
        mid_price = order_book.get_mid_price()
        assert mid_price is None


class TestTrade:
    """Тесты для Trade."""
    
    def test_creation(self):
        """Тест создания сделки."""
        trade = Trade(
            trade_id="12345",
            symbol=Currency.BTC,
            price=Price(Decimal("50000"), Currency.USDT, Currency.USDT),
            volume=Volume(Decimal("1.0"), Currency.BTC),
            side="buy",
            timestamp=Timestamp(datetime.now()),
            trade_type="market"
        )
        
        assert trade.trade_id == "12345"
        assert trade.symbol == Currency.BTC
        assert trade.price.value == Decimal("50000")
        assert trade.volume.value == Decimal("1.0")
        assert trade.side == "buy"
        assert trade.trade_type == "market"


class TestMarketSnapshot:
    """Тесты для MarketSnapshot."""
    
    def test_creation(self):
        """Тест создания снимка рынка."""
        market_data = MarketData(
            symbol=Symbol("BTC/USDT"),
            open=Price(Decimal("50000"), Currency.USDT, Currency.USDT),
            high=Price(Decimal("51000"), Currency.USDT, Currency.USDT),
            low=Price(Decimal("49000"), Currency.USDT, Currency.USDT),
            close=Price(Decimal("50500"), Currency.USDT, Currency.USDT),
            volume=Volume(Decimal("1000"), Currency.USDT)
        )
        
        order_book = OrderBook(symbol=Currency.BTC)
        
        recent_trades = [
            Trade(
                trade_id="12345",
                symbol=Currency.BTC,
                price=Price(Decimal("50500"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("1.0"), Currency.BTC),
                side="buy",
                timestamp=Timestamp(datetime.now())
            )
        ]
        
        snapshot = MarketSnapshot(
            symbol=Currency.BTC,
            timestamp=Timestamp(datetime.now()),
            market_data=market_data,
            order_book=order_book,
            recent_trades=recent_trades,
            metadata={"source": "binance"}
        )
        
        assert snapshot.symbol == Currency.BTC
        assert snapshot.market_data == market_data
        assert snapshot.order_book == order_book
        assert len(snapshot.recent_trades) == 1
        assert snapshot.metadata == {"source": "binance"}


class TestOHLCV:
    """Тесты для OHLCV."""
    
    def test_creation(self):
        """Тест создания OHLCV."""
        ohlcv = OHLCV(
            timestamp=TimestampValue(datetime.now()),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000")
        )
        
        assert ohlcv.timestamp.value == ohlcv.timestamp.value
        assert ohlcv.open == Decimal("50000")
        assert ohlcv.high == Decimal("51000")
        assert ohlcv.low == Decimal("49000")
        assert ohlcv.close == Decimal("50500")
        assert ohlcv.volume == Decimal("1000")
    
    def test_to_dict(self):
        """Тест сериализации в словарь."""
        ohlcv = OHLCV(
            timestamp=TimestampValue(datetime.now()),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000")
        )
        
        data = ohlcv.to_dict()
        
        assert "timestamp" in data
        assert data["open"] == "50000"
        assert data["high"] == "51000"
        assert data["low"] == "49000"
        assert data["close"] == "50500"
        assert data["volume"] == "1000"
    
    def test_from_dict(self):
        """Тест десериализации из словаря."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "open": "50000",
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "1000"
        }
        
        ohlcv = OHLCV.from_dict(data)
        
        assert ohlcv.open == Decimal("50000")
        assert ohlcv.high == Decimal("51000")
        assert ohlcv.low == Decimal("49000")
        assert ohlcv.close == Decimal("50500")
        assert ohlcv.volume == Decimal("1000") 