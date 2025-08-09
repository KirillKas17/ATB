"""
Unit тесты для MarketState.

Покрывает:
- Основной функционал создания и валидации
- Бизнес-логику определения состояния рынка
- Обработку ошибок и граничные случаи
- Сериализацию и десериализацию
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from domain.market.market_state import MarketState
from domain.market.market_types import MarketRegime, MarketMetadataDict
from domain.exceptions.base_exceptions import ValidationError


class TestMarketState:
    """Тесты для MarketState."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные для создания MarketState."""
        return {
            "id": "test_state_001",
            "symbol": "BTC/USDT",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "regime": MarketRegime.TRENDING_UP,
            "volatility": 0.15,
            "trend_strength": 0.8,
            "volume_trend": 0.6,
            "price_momentum": 0.7,
            "support_level": 45000.0,
            "resistance_level": 48000.0,
            "pivot_point": 46500.0,
            "rsi": 65.0,
            "macd": 0.002,
            "bollinger_upper": 47000.0,
            "bollinger_lower": 46000.0,
            "bollinger_middle": 46500.0,
            "atr": 500.0,
            "metadata": {"source": "binance", "exchange": "binance", "extra": {"confidence": 0.9}},
        }

    @pytest.fixture
    def market_state(self, sample_data) -> MarketState:
        """Создает экземпляр MarketState с тестовыми данными."""
        return MarketState(**sample_data)

    def test_creation_with_default_values(self):
        """Тест создания MarketState с значениями по умолчанию."""
        state = MarketState()

        assert state.id == ""
        assert state.symbol == ""
        assert isinstance(state.timestamp, datetime)
        assert state.regime == MarketRegime.UNKNOWN
        assert state.volatility == 0.0
        assert state.trend_strength == 0.0
        assert state.volume_trend == 0.0
        assert state.price_momentum == 0.0
        assert state.support_level is None
        assert state.resistance_level is None
        assert state.pivot_point is None
        assert state.rsi is None
        assert state.macd is None
        assert state.bollinger_upper is None
        assert state.bollinger_lower is None
        assert state.bollinger_middle is None
        assert state.atr is None
        assert state.metadata == {"source": "", "exchange": "", "extra": {}}

    def test_creation_with_custom_values(self, sample_data):
        """Тест создания MarketState с пользовательскими значениями."""
        state = MarketState(**sample_data)

        assert state.id == sample_data["id"]
        assert state.symbol == sample_data["symbol"]
        assert state.timestamp == sample_data["timestamp"]
        assert state.regime == sample_data["regime"]
        assert state.volatility == sample_data["volatility"]
        assert state.trend_strength == sample_data["trend_strength"]
        assert state.volume_trend == sample_data["volume_trend"]
        assert state.price_momentum == sample_data["price_momentum"]
        assert state.support_level == sample_data["support_level"]
        assert state.resistance_level == sample_data["resistance_level"]
        assert state.pivot_point == sample_data["pivot_point"]
        assert state.rsi == sample_data["rsi"]
        assert state.macd == sample_data["macd"]
        assert state.bollinger_upper == sample_data["bollinger_upper"]
        assert state.bollinger_lower == sample_data["bollinger_lower"]
        assert state.bollinger_middle == sample_data["bollinger_middle"]
        assert state.atr == sample_data["atr"]
        assert state.metadata == sample_data["metadata"]

    def test_is_trending_trending_up(self, market_state):
        """Тест определения трендового состояния (восходящий тренд)."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.is_trending() is True

    def test_is_trending_trending_down(self, market_state):
        """Тест определения трендового состояния (нисходящий тренд)."""
        market_state.regime = MarketRegime.TRENDING_DOWN
        assert market_state.is_trending() is True

    def test_is_trending_sideways(self, market_state):
        """Тест определения нетрендового состояния (боковик)."""
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_trending() is False

    def test_is_trending_volatile(self, market_state):
        """Тест определения нетрендового состояния (волатильность)."""
        market_state.regime = MarketRegime.VOLATILE
        assert market_state.is_trending() is False

    def test_is_sideways_true(self, market_state):
        """Тест определения бокового движения."""
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_sideways() is True

    def test_is_sideways_false(self, market_state):
        """Тест определения не бокового движения."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.is_sideways() is False

    def test_is_volatile_true(self, market_state):
        """Тест определения волатильного состояния."""
        market_state.regime = MarketRegime.VOLATILE
        assert market_state.is_volatile() is True

    def test_is_volatile_false(self, market_state):
        """Тест определения не волатильного состояния."""
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.is_volatile() is False

    def test_is_breakout_true(self, market_state):
        """Тест определения состояния пробоя."""
        market_state.regime = MarketRegime.BREAKOUT
        assert market_state.is_breakout() is True

    def test_is_breakout_false(self, market_state):
        """Тест определения состояния не пробоя."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.is_breakout() is False

    def test_get_trend_direction_up(self, market_state):
        """Тест получения направления тренда (вверх)."""
        market_state.regime = MarketRegime.TRENDING_UP
        assert market_state.get_trend_direction() == "up"

    def test_get_trend_direction_down(self, market_state):
        """Тест получения направления тренда (вниз)."""
        market_state.regime = MarketRegime.TRENDING_DOWN
        assert market_state.get_trend_direction() == "down"

    def test_get_trend_direction_none(self, market_state):
        """Тест получения направления тренда (нет тренда)."""
        market_state.regime = MarketRegime.SIDEWAYS
        assert market_state.get_trend_direction() is None

    def test_get_price_position_below_support(self, market_state):
        """Тест определения позиции цены ниже поддержки."""
        market_state.support_level = 45000.0
        current_price = 44000.0
        assert market_state.get_price_position(current_price) == "below_support"

    def test_get_price_position_above_resistance(self, market_state):
        """Тест определения позиции цены выше сопротивления."""
        market_state.resistance_level = 48000.0
        current_price = 49000.0
        assert market_state.get_price_position(current_price) == "above_resistance"

    def test_get_price_position_inside_range(self, market_state):
        """Тест определения позиции цены внутри диапазона."""
        market_state.support_level = 45000.0
        market_state.resistance_level = 48000.0
        current_price = 46500.0
        assert market_state.get_price_position(current_price) == "inside_range"

    def test_get_price_position_no_levels(self, market_state):
        """Тест определения позиции цены без уровней."""
        market_state.support_level = None
        market_state.resistance_level = None
        current_price = 46500.0
        assert market_state.get_price_position(current_price) == "inside_range"

    def test_is_overbought_true(self, market_state):
        """Тест определения перекупленности."""
        market_state.rsi = 75.0
        assert market_state.is_overbought() is True

    def test_is_overbought_false(self, market_state):
        """Тест определения не перекупленности."""
        market_state.rsi = 65.0
        assert market_state.is_overbought() is False

    def test_is_overbought_none(self, market_state):
        """Тест определения перекупленности при отсутствии RSI."""
        market_state.rsi = None
        assert market_state.is_overbought() is False

    def test_is_oversold_true(self, market_state):
        """Тест определения перепроданности."""
        market_state.rsi = 25.0
        assert market_state.is_oversold() is True

    def test_is_oversold_false(self, market_state):
        """Тест определения не перепроданности."""
        market_state.rsi = 35.0
        assert market_state.is_oversold() is False

    def test_is_oversold_none(self, market_state):
        """Тест определения перепроданности при отсутствии RSI."""
        market_state.rsi = None
        assert market_state.is_oversold() is False

    def test_to_dict_complete_data(self, market_state):
        """Тест сериализации в словарь с полными данными."""
        result = market_state.to_dict()

        assert result["id"] == market_state.id
        assert result["symbol"] == market_state.symbol
        assert result["timestamp"] == market_state.timestamp.isoformat()
        assert result["regime"] == market_state.regime.value
        assert result["volatility"] == market_state.volatility
        assert result["trend_strength"] == market_state.trend_strength
        assert result["volume_trend"] == market_state.volume_trend
        assert result["price_momentum"] == market_state.price_momentum
        assert result["support_level"] == market_state.support_level
        assert result["resistance_level"] == market_state.resistance_level
        assert result["pivot_point"] == market_state.pivot_point
        assert result["rsi"] == market_state.rsi
        assert result["macd"] == market_state.macd
        assert result["bollinger_upper"] == market_state.bollinger_upper
        assert result["bollinger_lower"] == market_state.bollinger_lower
        assert result["bollinger_middle"] == market_state.bollinger_middle
        assert result["atr"] == market_state.atr
        assert result["metadata"] == market_state.metadata

    def test_to_dict_with_none_values(self):
        """Тест сериализации в словарь с None значениями."""
        state = MarketState()
        result = state.to_dict()

        assert result["support_level"] is None
        assert result["resistance_level"] is None
        assert result["pivot_point"] is None
        assert result["rsi"] is None
        assert result["macd"] is None
        assert result["bollinger_upper"] is None
        assert result["bollinger_lower"] is None
        assert result["bollinger_middle"] is None
        assert result["atr"] is None

    def test_from_dict_complete_data(self, sample_data):
        """Тест десериализации из словаря с полными данными."""
        data_dict = {
            "id": sample_data["id"],
            "symbol": sample_data["symbol"],
            "timestamp": sample_data["timestamp"].isoformat(),
            "regime": sample_data["regime"].value,
            "volatility": sample_data["volatility"],
            "trend_strength": sample_data["trend_strength"],
            "volume_trend": sample_data["volume_trend"],
            "price_momentum": sample_data["price_momentum"],
            "support_level": sample_data["support_level"],
            "resistance_level": sample_data["resistance_level"],
            "pivot_point": sample_data["pivot_point"],
            "rsi": sample_data["rsi"],
            "macd": sample_data["macd"],
            "bollinger_upper": sample_data["bollinger_upper"],
            "bollinger_lower": sample_data["bollinger_lower"],
            "bollinger_middle": sample_data["bollinger_middle"],
            "atr": sample_data["atr"],
            "metadata": sample_data["metadata"],
        }

        state = MarketState.from_dict(data_dict)

        assert state.id == sample_data["id"]
        assert state.symbol == sample_data["symbol"]
        assert state.timestamp == sample_data["timestamp"]
        assert state.regime == sample_data["regime"]
        assert state.volatility == sample_data["volatility"]
        assert state.trend_strength == sample_data["trend_strength"]
        assert state.volume_trend == sample_data["volume_trend"]
        assert state.price_momentum == sample_data["price_momentum"]
        assert state.support_level == sample_data["support_level"]
        assert state.resistance_level == sample_data["resistance_level"]
        assert state.pivot_point == sample_data["pivot_point"]
        assert state.rsi == sample_data["rsi"]
        assert state.macd == sample_data["macd"]
        assert state.bollinger_upper == sample_data["bollinger_upper"]
        assert state.bollinger_lower == sample_data["bollinger_lower"]
        assert state.bollinger_middle == sample_data["bollinger_middle"]
        assert state.atr == sample_data["atr"]
        assert state.metadata == sample_data["metadata"]

    def test_from_dict_with_none_values(self):
        """Тест десериализации из словаря с None значениями."""
        data_dict = {
            "id": "test",
            "symbol": "BTC/USDT",
            "timestamp": datetime.now().isoformat(),
            "regime": MarketRegime.UNKNOWN.value,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "volume_trend": 0.0,
            "price_momentum": 0.0,
            "support_level": None,
            "resistance_level": None,
            "pivot_point": None,
            "rsi": None,
            "macd": None,
            "bollinger_upper": None,
            "bollinger_lower": None,
            "bollinger_middle": None,
            "atr": None,
            "metadata": {"source": "", "exchange": "", "extra": {}},
        }

        state = MarketState.from_dict(data_dict)

        assert state.support_level is None
        assert state.resistance_level is None
        assert state.pivot_point is None
        assert state.rsi is None
        assert state.macd is None
        assert state.bollinger_upper is None
        assert state.bollinger_lower is None
        assert state.bollinger_middle is None
        assert state.atr is None

    def test_from_dict_with_missing_optional_fields(self):
        """Тест десериализации из словаря с отсутствующими опциональными полями."""
        data_dict = {
            "id": "test",
            "symbol": "BTC/USDT",
            "timestamp": datetime.now().isoformat(),
            "regime": MarketRegime.UNKNOWN.value,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "volume_trend": 0.0,
            "price_momentum": 0.0,
            "metadata": {"source": "", "exchange": "", "extra": {}},
        }

        state = MarketState.from_dict(data_dict)

        assert state.support_level is None
        assert state.resistance_level is None
        assert state.pivot_point is None
        assert state.rsi is None
        assert state.macd is None
        assert state.bollinger_upper is None
        assert state.bollinger_lower is None
        assert state.bollinger_middle is None
        assert state.atr is None

    def test_from_dict_with_default_metadata(self):
        """Тест десериализации из словаря с метаданными по умолчанию."""
        data_dict = {
            "id": "test",
            "symbol": "BTC/USDT",
            "timestamp": datetime.now().isoformat(),
            "regime": MarketRegime.UNKNOWN.value,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "volume_trend": 0.0,
            "price_momentum": 0.0,
        }

        state = MarketState.from_dict(data_dict)

        expected_metadata = {"source": "", "exchange": "", "extra": {}}
        assert state.metadata == expected_metadata

    def test_protocol_compliance(self, market_state):
        """Тест соответствия протоколу MarketStateProtocol."""
        from domain.market.market_protocols import MarketStateProtocol

        assert isinstance(market_state, MarketStateProtocol)

        # Проверяем наличие всех обязательных атрибутов
        assert hasattr(market_state, "id")
        assert hasattr(market_state, "symbol")
        assert hasattr(market_state, "timestamp")
        assert hasattr(market_state, "regime")
        assert hasattr(market_state, "volatility")
        assert hasattr(market_state, "trend_strength")
        assert hasattr(market_state, "volume_trend")
        assert hasattr(market_state, "price_momentum")
        assert hasattr(market_state, "support_level")
        assert hasattr(market_state, "resistance_level")
        assert hasattr(market_state, "pivot_point")
        assert hasattr(market_state, "rsi")
        assert hasattr(market_state, "macd")
        assert hasattr(market_state, "bollinger_upper")
        assert hasattr(market_state, "bollinger_lower")
        assert hasattr(market_state, "bollinger_middle")
        assert hasattr(market_state, "atr")
        assert hasattr(market_state, "metadata")

        # Проверяем наличие всех методов
        assert hasattr(market_state, "is_trending")
        assert hasattr(market_state, "is_sideways")
        assert hasattr(market_state, "is_volatile")
        assert hasattr(market_state, "is_breakout")
        assert hasattr(market_state, "get_trend_direction")
        assert hasattr(market_state, "get_price_position")
        assert hasattr(market_state, "is_overbought")
        assert hasattr(market_state, "is_oversold")
        assert hasattr(market_state, "to_dict")
        assert hasattr(market_state, "from_dict")

    def test_edge_case_rsi_boundary_values(self, market_state):
        """Тест граничных значений RSI."""
        # Граничные значения для перекупленности
        market_state.rsi = 70.0
        assert market_state.is_overbought() is True

        market_state.rsi = 69.9
        assert market_state.is_overbought() is False

        # Граничные значения для перепроданности
        market_state.rsi = 30.0
        assert market_state.is_oversold() is True

        market_state.rsi = 30.1
        assert market_state.is_oversold() is False

    def test_edge_case_price_position_exact_levels(self, market_state):
        """Тест граничных случаев позиции цены на точных уровнях."""
        market_state.support_level = 45000.0
        market_state.resistance_level = 48000.0

        # Цена точно на уровне поддержки
        assert market_state.get_price_position(45000.0) == "inside_range"

        # Цена точно на уровне сопротивления
        assert market_state.get_price_position(48000.0) == "inside_range"

    def test_comprehensive_market_analysis(self, market_state):
        """Комплексный тест анализа состояния рынка."""
        # Настройка для восходящего тренда
        market_state.regime = MarketRegime.TRENDING_UP
        market_state.trend_strength = 0.8
        market_state.rsi = 65.0
        market_state.support_level = 45000.0
        market_state.resistance_level = 48000.0

        # Проверка состояния
        assert market_state.is_trending() is True
        assert market_state.get_trend_direction() == "up"
        assert market_state.is_overbought() is False
        assert market_state.is_oversold() is False
        assert market_state.get_price_position(46500.0) == "inside_range"

        # Настройка для перекупленного состояния
        market_state.rsi = 75.0
        assert market_state.is_overbought() is True

        # Настройка для пробоя сопротивления
        assert market_state.get_price_position(49000.0) == "above_resistance"

    def test_serialization_round_trip(self, market_state):
        """Тест полного цикла сериализации и десериализации."""
        # Сериализация
        data_dict = market_state.to_dict()

        # Десериализация
        restored_state = MarketState.from_dict(data_dict)

        # Проверка идентичности
        assert restored_state.id == market_state.id
        assert restored_state.symbol == market_state.symbol
        assert restored_state.timestamp == market_state.timestamp
        assert restored_state.regime == market_state.regime
        assert restored_state.volatility == market_state.volatility
        assert restored_state.trend_strength == market_state.trend_strength
        assert restored_state.volume_trend == market_state.volume_trend
        assert restored_state.price_momentum == market_state.price_momentum
        assert restored_state.support_level == market_state.support_level
        assert restored_state.resistance_level == market_state.resistance_level
        assert restored_state.pivot_point == market_state.pivot_point
        assert restored_state.rsi == market_state.rsi
        assert restored_state.macd == market_state.macd
        assert restored_state.bollinger_upper == market_state.bollinger_upper
        assert restored_state.bollinger_lower == market_state.bollinger_lower
        assert restored_state.bollinger_middle == market_state.bollinger_middle
        assert restored_state.atr == market_state.atr
        assert restored_state.metadata == market_state.metadata
