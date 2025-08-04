"""
Unit тесты для Trading.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.trading import (
    Signal, Trade, Position, TradingSession,
    OrderType, OrderSide, OrderStatus, PositionSide, SignalType,
    OrderProtocol, TradeProtocol, PositionProtocol
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import TimestampValue
from domain.type_definitions import MetadataDict


class TestSignal:
    """Тесты для Signal."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": str(uuid4()),
            "symbol": "BTC/USD",
            "signal_type": SignalType.BUY,
            "strength": 0.8,
            "confidence": 0.9,
            "price": Price(Decimal("50000"), Currency.USD),
            "timestamp": datetime.now(),
            "metadata": MetadataDict({"source": "strategy", "confidence": "high"})
        }
    
    @pytest.fixture
    def signal(self, sample_data) -> Signal:
        """Создает тестовый сигнал."""
        return Signal(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания сигнала."""
        signal = Signal(**sample_data)
        
        assert signal.id == sample_data["id"]
        assert signal.symbol == sample_data["symbol"]
        assert signal.signal_type == sample_data["signal_type"]
        assert signal.strength == sample_data["strength"]
        assert signal.confidence == sample_data["confidence"]
        assert signal.price == sample_data["price"]
        assert signal.timestamp == sample_data["timestamp"]
        assert signal.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания сигнала с дефолтными значениями."""
        signal = Signal()
        
        assert isinstance(signal.id, str)
        assert signal.symbol == ""
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.price is None
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == MetadataDict({})
    
    def test_post_init_price_conversion(self):
        """Тест конвертации цены в __post_init__."""
        # Тест с Price объектом
        price = Price(Decimal("50000"), Currency.USD)
        signal = Signal(price=price)
        assert signal.price == price
        
        # Тест с Decimal
        signal = Signal(price=Decimal("50000"))
        assert signal.price.value == Decimal("50000")
        assert signal.price.currency == Currency.USD
        
        # Тест с float
        signal = Signal(price=50000.0)
        assert signal.price.value == Decimal("50000")
        assert signal.price.currency == Currency.USD
    
    def test_to_dict(self, signal):
        """Тест сериализации в словарь."""
        data = signal.to_dict()
        
        assert data["id"] == signal.id
        assert data["symbol"] == signal.symbol
        assert data["signal_type"] == signal.signal_type.value
        assert data["strength"] == signal.strength
        assert data["confidence"] == signal.confidence
        assert data["price"] == str(signal.price.value)
        assert data["timestamp"] == signal.timestamp.isoformat()
        assert data["metadata"] == signal.metadata
    
    def test_to_dict_no_price(self):
        """Тест сериализации в словарь без цены."""
        signal = Signal(price=None)
        data = signal.to_dict()
        
        assert data["price"] is None
    
    def test_from_dict(self, signal):
        """Тест десериализации из словаря."""
        data = signal.to_dict()
        new_signal = Signal.from_dict(data)
        
        assert new_signal.id == signal.id
        assert new_signal.symbol == signal.symbol
        assert new_signal.signal_type == signal.signal_type
        assert new_signal.strength == signal.strength
        assert new_signal.confidence == signal.confidence
        assert new_signal.price.value == signal.price.value
        assert new_signal.timestamp == signal.timestamp
        assert new_signal.metadata == signal.metadata
    
    def test_from_dict_no_price(self):
        """Тест десериализации из словаря без цены."""
        data = {
            "id": str(uuid4()),
            "symbol": "BTC/USD",
            "signal_type": SignalType.SELL.value,
            "strength": 0.7,
            "confidence": 0.8,
            "price": None,
            "timestamp": datetime.now().isoformat(),
            "metadata": {}
        }
        signal = Signal.from_dict(data)
        
        assert signal.price is None
    
    def test_signal_type_enum(self):
        """Тест enum SignalType."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.CLOSE.value == "close"


class TestTrade:
    """Тесты для Trade."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": str(uuid4()),
            "order_id": str(uuid4()),
            "trading_pair": "BTC/USD",
            "side": OrderSide.BUY,
            "quantity": Volume(Decimal("1.5")),
            "price": Price(Decimal("50000"), Currency.USD),
            "commission": Money(Decimal("25"), Currency.USD),
            "timestamp": TimestampValue(datetime.now())
        }
    
    @pytest.fixture
    def trade(self, sample_data) -> Trade:
        """Создает тестовую сделку."""
        return Trade(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания сделки."""
        trade = Trade(**sample_data)
        
        assert trade.id == sample_data["id"]
        assert trade.order_id == sample_data["order_id"]
        assert trade.trading_pair == sample_data["trading_pair"]
        assert trade.side == sample_data["side"]
        assert trade.quantity == sample_data["quantity"]
        assert trade.price == sample_data["price"]
        assert trade.commission == sample_data["commission"]
        assert trade.timestamp == sample_data["timestamp"]
    
    def test_default_creation(self):
        """Тест создания сделки с дефолтными значениями."""
        trade = Trade()
        
        assert isinstance(trade.id, str)
        assert isinstance(trade.order_id, str)
        assert trade.trading_pair == ""
        assert trade.side == OrderSide.BUY
        assert trade.quantity.value == Decimal("0")
        assert trade.price.value == Decimal("0")
        assert trade.commission.value == Decimal("0")
        assert isinstance(trade.timestamp.value, datetime)
    
    def test_post_init_conversion(self):
        """Тест конвертации в __post_init__."""
        # Тест с различными типами данных
        trade = Trade(
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
            commission=Decimal("25")
        )
        
        assert isinstance(trade.quantity, Volume)
        assert isinstance(trade.price, Price)
        assert isinstance(trade.commission, Money)
        assert trade.quantity.value == Decimal("1.5")
        assert trade.price.value == Decimal("50000")
        assert trade.commission.value == Decimal("25")
    
    def test_total_value(self, trade):
        """Тест расчета общей стоимости сделки."""
        total_value = trade.total_value
        
        expected_value = trade.price.value * trade.quantity.value
        assert total_value.value == expected_value
        assert total_value.currency == Currency.USD
    
    def test_to_dict(self, trade):
        """Тест сериализации в словарь."""
        data = trade.to_dict()
        
        assert data["id"] == trade.id
        assert data["order_id"] == trade.order_id
        assert data["trading_pair"] == trade.trading_pair
        assert data["side"] == trade.side.value
        assert data["quantity"] == str(trade.quantity.value)
        assert data["price"] == str(trade.price.value)
        assert data["commission"] == str(trade.commission.value)
        assert data["timestamp"] == trade.timestamp.isoformat()
    
    def test_from_dict(self, trade):
        """Тест десериализации из словаря."""
        data = trade.to_dict()
        new_trade = Trade.from_dict(data)
        
        assert new_trade.id == trade.id
        assert new_trade.order_id == trade.order_id
        assert new_trade.trading_pair == trade.trading_pair
        assert new_trade.side == trade.side
        assert new_trade.quantity.value == trade.quantity.value
        assert new_trade.price.value == trade.price.value
        assert new_trade.commission.value == trade.commission.value
        assert new_trade.timestamp.value == trade.timestamp.value
    
    def test_trade_protocol_compliance(self, trade):
        """Тест соответствия протоколу TradeProtocol."""
        assert isinstance(trade, TradeProtocol)
        
        total_value = trade.total_value
        assert isinstance(total_value, Money)
        
        data = trade.to_dict()
        assert isinstance(data, dict)
        
        new_trade = Trade.from_dict(data)
        assert isinstance(new_trade, TradeProtocol)


class TestPosition:
    """Тесты для Position."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": str(uuid4()),
            "symbol": "BTC/USD",
            "side": PositionSide.LONG,
            "quantity": Volume(Decimal("2.0")),
            "entry_price": Price(Decimal("50000"), Currency.USD),
            "current_price": Price(Decimal("52000"), Currency.USD),
            "entry_time": datetime.now(),
            "updated_at": datetime.now(),
            "stop_loss": Price(Decimal("48000"), Currency.USD),
            "take_profit": Price(Decimal("55000"), Currency.USD),
            "unrealized_pnl": Money(Decimal("4000"), Currency.USD),
            "realized_pnl": Money(Decimal("1000"), Currency.USD),
            "metadata": MetadataDict({"strategy": "trend_following"})
        }
    
    @pytest.fixture
    def position(self, sample_data) -> Position:
        """Создает тестовую позицию."""
        return Position(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания позиции."""
        position = Position(**sample_data)
        
        assert position.id == sample_data["id"]
        assert position.symbol == sample_data["symbol"]
        assert position.side == sample_data["side"]
        assert position.quantity == sample_data["quantity"]
        assert position.entry_price == sample_data["entry_price"]
        assert position.current_price == sample_data["current_price"]
        assert position.entry_time == sample_data["entry_time"]
        assert position.updated_at == sample_data["updated_at"]
        assert position.stop_loss == sample_data["stop_loss"]
        assert position.take_profit == sample_data["take_profit"]
        assert position.unrealized_pnl == sample_data["unrealized_pnl"]
        assert position.realized_pnl == sample_data["realized_pnl"]
        assert position.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания позиции с дефолтными значениями."""
        position = Position()
        
        assert isinstance(position.id, str)
        assert position.symbol == ""
        assert position.side == PositionSide.LONG
        assert position.quantity.value == Decimal("0")
        assert position.entry_price.value == Decimal("0")
        assert position.current_price.value == Decimal("0")
        assert isinstance(position.entry_time, datetime)
        assert isinstance(position.updated_at, datetime)
        assert position.stop_loss is None
        assert position.take_profit is None
        assert position.unrealized_pnl.value == Decimal("0")
        assert position.realized_pnl.value == Decimal("0")
        assert position.metadata == MetadataDict({})
    
    def test_post_init_conversion(self):
        """Тест конвертации в __post_init__."""
        position = Position(
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000"),
            unrealized_pnl=Decimal("4000"),
            realized_pnl=Decimal("1000")
        )
        
        assert isinstance(position.quantity, Volume)
        assert isinstance(position.entry_price, Price)
        assert isinstance(position.current_price, Price)
        assert isinstance(position.unrealized_pnl, Money)
        assert isinstance(position.realized_pnl, Money)
    
    def test_get_market_value(self, position):
        """Тест получения рыночной стоимости позиции."""
        market_value = position.get_market_value()
        
        expected_value = position.current_price.value * position.quantity.value
        assert market_value.value == expected_value
        assert market_value.currency == Currency.USD
    
    def test_get_entry_value(self, position):
        """Тест получения стоимости входа в позицию."""
        entry_value = position.get_entry_value()
        
        expected_value = position.entry_price.value * position.quantity.value
        assert entry_value.value == expected_value
        assert entry_value.currency == Currency.USD
    
    def test_calculate_unrealized_pnl_long_profitable(self, position):
        """Тест расчета нереализованного P&L для длинной прибыльной позиции."""
        position.side = PositionSide.LONG
        position.entry_price = Price(Decimal("50000"), Currency.USD)
        position.current_price = Price(Decimal("52000"), Currency.USD)
        position.quantity = Volume(Decimal("2.0"))
        
        pnl = position.calculate_unrealized_pnl()
        expected_pnl = (Decimal("52000") - Decimal("50000")) * Decimal("2.0")
        
        assert pnl.value == expected_pnl
        assert pnl.currency == Currency.USD
    
    def test_calculate_unrealized_pnl_long_loss(self, position):
        """Тест расчета нереализованного P&L для длинной убыточной позиции."""
        position.side = PositionSide.LONG
        position.entry_price = Price(Decimal("52000"), Currency.USD)
        position.current_price = Price(Decimal("50000"), Currency.USD)
        position.quantity = Volume(Decimal("2.0"))
        
        pnl = position.calculate_unrealized_pnl()
        expected_pnl = (Decimal("50000") - Decimal("52000")) * Decimal("2.0")
        
        assert pnl.value == expected_pnl
        assert pnl.currency == Currency.USD
    
    def test_calculate_unrealized_pnl_short_profitable(self, position):
        """Тест расчета нереализованного P&L для короткой прибыльной позиции."""
        position.side = PositionSide.SHORT
        position.entry_price = Price(Decimal("52000"), Currency.USD)
        position.current_price = Price(Decimal("50000"), Currency.USD)
        position.quantity = Volume(Decimal("2.0"))
        
        pnl = position.calculate_unrealized_pnl()
        expected_pnl = (Decimal("52000") - Decimal("50000")) * Decimal("2.0")
        
        assert pnl.value == expected_pnl
        assert pnl.currency == Currency.USD
    
    def test_calculate_unrealized_pnl_short_loss(self, position):
        """Тест расчета нереализованного P&L для короткой убыточной позиции."""
        position.side = PositionSide.SHORT
        position.entry_price = Price(Decimal("50000"), Currency.USD)
        position.current_price = Price(Decimal("52000"), Currency.USD)
        position.quantity = Volume(Decimal("2.0"))
        
        pnl = position.calculate_unrealized_pnl()
        expected_pnl = (Decimal("50000") - Decimal("52000")) * Decimal("2.0")
        
        assert pnl.value == expected_pnl
        assert pnl.currency == Currency.USD
    
    def test_is_profitable(self, position):
        """Тест проверки прибыльности позиции."""
        # Прибыльная позиция
        position.side = PositionSide.LONG
        position.entry_price = Price(Decimal("50000"), Currency.USD)
        position.current_price = Price(Decimal("52000"), Currency.USD)
        position.quantity = Volume(Decimal("2.0"))
        
        assert position.is_profitable() is True
        
        # Убыточная позиция
        position.current_price = Price(Decimal("48000"), Currency.USD)
        assert position.is_profitable() is False
    
    def test_is_open(self, position):
        """Тест проверки открытости позиции."""
        # Открытая позиция
        position.quantity = Volume(Decimal("2.0"))
        assert position.is_open is True
        
        # Закрытая позиция
        position.quantity = Volume(Decimal("0"))
        assert position.is_open is False
        
        # Отрицательная позиция (не должно быть)
        position.quantity = Volume(Decimal("-1"))
        assert position.is_open is False
    
    def test_to_dict(self, position):
        """Тест сериализации в словарь."""
        data = position.to_dict()
        
        assert data["id"] == position.id
        assert data["symbol"] == position.symbol
        assert data["side"] == position.side.value
        assert data["quantity"] == str(position.quantity.value)
        assert data["entry_price"] == str(position.entry_price.value)
        assert data["current_price"] == str(position.current_price.value)
        assert data["entry_time"] == position.entry_time.isoformat()
        assert data["updated_at"] == position.updated_at.isoformat()
        assert data["stop_loss"] == str(position.stop_loss.value)
        assert data["take_profit"] == str(position.take_profit.value)
        assert data["unrealized_pnl"] == str(position.unrealized_pnl.value)
        assert data["realized_pnl"] == str(position.realized_pnl.value)
        assert data["metadata"] == position.metadata
    
    def test_to_dict_no_stop_loss_take_profit(self):
        """Тест сериализации в словарь без stop_loss и take_profit."""
        position = Position(stop_loss=None, take_profit=None)
        data = position.to_dict()
        
        assert data["stop_loss"] is None
        assert data["take_profit"] is None
    
    def test_from_dict(self, position):
        """Тест десериализации из словаря."""
        data = position.to_dict()
        new_position = Position.from_dict(data)
        
        assert new_position.id == position.id
        assert new_position.symbol == position.symbol
        assert new_position.side == position.side
        assert new_position.quantity.value == position.quantity.value
        assert new_position.entry_price.value == position.entry_price.value
        assert new_position.current_price.value == position.current_price.value
        assert new_position.entry_time == position.entry_time
        assert new_position.updated_at == position.updated_at
        assert new_position.stop_loss.value == position.stop_loss.value
        assert new_position.take_profit.value == position.take_profit.value
        assert new_position.unrealized_pnl.value == position.unrealized_pnl.value
        assert new_position.realized_pnl.value == position.realized_pnl.value
        assert new_position.metadata == position.metadata
    
    def test_from_dict_no_stop_loss_take_profit(self):
        """Тест десериализации из словаря без stop_loss и take_profit."""
        data = {
            "id": str(uuid4()),
            "symbol": "BTC/USD",
            "side": PositionSide.LONG.value,
            "quantity": "2.0",
            "entry_price": "50000",
            "current_price": "52000",
            "entry_time": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stop_loss": None,
            "take_profit": None,
            "unrealized_pnl": "4000",
            "realized_pnl": "1000",
            "metadata": {}
        }
        position = Position.from_dict(data)
        
        assert position.stop_loss is None
        assert position.take_profit is None
    
    def test_position_protocol_compliance(self, position):
        """Тест соответствия протоколу PositionProtocol."""
        assert isinstance(position, PositionProtocol)
        
        market_value = position.get_market_value()
        assert isinstance(market_value, Money)
        
        entry_value = position.get_entry_value()
        assert isinstance(entry_value, Money)
        
        unrealized_pnl = position.calculate_unrealized_pnl()
        assert isinstance(unrealized_pnl, Money)
        
        is_profitable = position.is_profitable()
        assert isinstance(is_profitable, bool)
        
        data = position.to_dict()
        assert isinstance(data, dict)
        
        new_position = Position.from_dict(data)
        assert isinstance(new_position, PositionProtocol)
    
    def test_position_side_enum(self):
        """Тест enum PositionSide."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"


class TestTradingSession:
    """Тесты для TradingSession."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": str(uuid4()),
            "start_time": datetime.now(),
            "end_time": None,
            "trading_pairs": ["BTC/USD", "ETH/USD"],
            "total_trades": 10,
            "total_volume": Volume(Decimal("15.5")),
            "total_commission": Money(Decimal("250"), Currency.USD),
            "pnl": Money(Decimal("5000"), Currency.USD)
        }
    
    @pytest.fixture
    def trading_session(self, sample_data) -> TradingSession:
        """Создает тестовую торговую сессию."""
        return TradingSession(**sample_data)
    
    @pytest.fixture
    def trade(self) -> Trade:
        """Создает тестовую сделку."""
        return Trade(
            trading_pair="BTC/USD",
            side=OrderSide.BUY,
            quantity=Volume(Decimal("1.0")),
            price=Price(Decimal("50000"), Currency.USD),
            commission=Money(Decimal("25"), Currency.USD)
        )
    
    def test_creation(self, sample_data):
        """Тест создания торговой сессии."""
        session = TradingSession(**sample_data)
        
        assert session.id == sample_data["id"]
        assert session.start_time == sample_data["start_time"]
        assert session.end_time == sample_data["end_time"]
        assert session.trading_pairs == sample_data["trading_pairs"]
        assert session.total_trades == sample_data["total_trades"]
        assert session.total_volume == sample_data["total_volume"]
        assert session.total_commission == sample_data["total_commission"]
        assert session.pnl == sample_data["pnl"]
    
    def test_default_creation(self):
        """Тест создания торговой сессии с дефолтными значениями."""
        session = TradingSession()
        
        assert isinstance(session.id, str)
        assert isinstance(session.start_time, datetime)
        assert session.end_time is None
        assert session.trading_pairs == []
        assert session.total_trades == 0
        assert session.total_volume.value == Decimal("0")
        assert session.total_commission.value == Decimal("0")
        assert session.pnl.value == Decimal("0")
    
    def test_post_init_conversion(self):
        """Тест конвертации в __post_init__."""
        session = TradingSession(
            total_volume=Decimal("15.5"),
            total_commission=Decimal("250"),
            pnl=Decimal("5000")
        )
        
        assert isinstance(session.total_volume, Volume)
        assert isinstance(session.total_commission, Money)
        assert isinstance(session.pnl, Money)
    
    def test_add_trade(self, trading_session, trade):
        """Тест добавления сделки в сессию."""
        initial_trades = trading_session.total_trades
        initial_volume = trading_session.total_volume.value
        initial_commission = trading_session.total_commission.value
        
        trading_session.add_trade(trade)
        
        assert trading_session.total_trades == initial_trades + 1
        assert trading_session.total_volume.value == initial_volume + trade.quantity.value
        assert trading_session.total_commission.value == initial_commission + trade.commission.value
    
    def test_close(self, trading_session):
        """Тест закрытия сессии."""
        assert trading_session.end_time is None
        
        trading_session.close()
        
        assert trading_session.end_time is not None
        assert isinstance(trading_session.end_time, datetime)
    
    def test_duration_open_session(self, trading_session):
        """Тест длительности открытой сессии."""
        duration = trading_session.duration
        assert duration is None
    
    def test_duration_closed_session(self, trading_session):
        """Тест длительности закрытой сессии."""
        import time
        time.sleep(0.1)  # Небольшая задержка для тестирования
        
        trading_session.close()
        duration = trading_session.duration
        
        assert duration is not None
        assert duration > 0
    
    def test_to_dict(self, trading_session):
        """Тест сериализации в словарь."""
        data = trading_session.to_dict()
        
        assert data["id"] == trading_session.id
        assert data["start_time"] == trading_session.start_time.isoformat()
        assert data["end_time"] is None
        assert data["trading_pairs"] == [str(pair) for pair in trading_session.trading_pairs]
        assert data["total_trades"] == trading_session.total_trades
        assert data["total_volume"] == str(trading_session.total_volume.value)
        assert data["total_commission"] == str(trading_session.total_commission.value)
        assert data["pnl"] == str(trading_session.pnl.value)
    
    def test_to_dict_closed_session(self, trading_session):
        """Тест сериализации в словарь закрытой сессии."""
        trading_session.close()
        data = trading_session.to_dict()
        
        assert data["end_time"] == trading_session.end_time.isoformat()
    
    def test_from_dict(self, trading_session):
        """Тест десериализации из словаря."""
        data = trading_session.to_dict()
        new_session = TradingSession.from_dict(data)
        
        assert new_session.id == trading_session.id
        assert new_session.start_time == trading_session.start_time
        assert new_session.end_time == trading_session.end_time
        assert new_session.trading_pairs == trading_session.trading_pairs
        assert new_session.total_trades == trading_session.total_trades
        assert new_session.total_volume.value == trading_session.total_volume.value
        assert new_session.total_commission.value == trading_session.total_commission.value
        assert new_session.pnl.value == trading_session.pnl.value
    
    def test_from_dict_closed_session(self, trading_session):
        """Тест десериализации из словаря закрытой сессии."""
        trading_session.close()
        data = trading_session.to_dict()
        new_session = TradingSession.from_dict(data)
        
        assert new_session.end_time == trading_session.end_time


class TestEnums:
    """Тесты для enum'ов."""
    
    def test_order_type_enum(self):
        """Тест enum OrderType."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TAKE_PROFIT.value == "take_profit"
    
    def test_order_side_enum(self):
        """Тест enum OrderSide."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_status_enum(self):
        """Тест enum OrderStatus."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired" 