"""Тесты для Trading entities."""
from datetime import datetime
from decimal import Decimal
from uuid import uuid4
from domain.entities.order import Order, OrderType, OrderSide, OrderStatus
from domain.entities.trade import Trade
from domain.entities.position import Position, PositionSide
from domain.entities.trading_pair import TradingPair
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
def test_order_creation() -> None:
    order = Order(
        id=uuid4(),
        strategy_id=uuid4(),
        signal_id=None,
        exchange_order_id="ex123",
        trading_pair="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=Money(Decimal("50000"), Currency.USDT),
        stop_price=None,
        status=OrderStatus.OPEN,
        filled_quantity=Decimal("0.0"),
        average_price=None,
        commission=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        filled_at=None,
        metadata={}
    )
    assert order.trading_pair == "BTC/USDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.status == OrderStatus.OPEN
    assert order.quantity == Decimal("1.0")
def test_trade_creation() -> None:
    trade = Trade(
        id="trade_001",
        symbol="BTC/USDT",
        side="buy",
        price=Price(Decimal("50000"), Currency.USDT),
        volume=Volume(Decimal("1.0"), Currency.BTC),
        executed_at=Timestamp(datetime.now()),
        fee=Money(Decimal("25"), Currency.USDT),
        realized_pnl=None,
        metadata={}
    )
    assert trade.id == "trade_001"
    assert trade.symbol == "BTC/USDT"
    assert trade.side == "buy"
    assert trade.price.value == Decimal("50000")
    assert trade.volume.value == Decimal("1.0")
    assert trade.fee.value == Decimal("25")
def test_position_creation() -> None:
    pair = TradingPair(symbol="BTC/USDT", base_currency=Currency.BTC, quote_currency=Currency.USDT)
    position = Position(
        id="pos_001",
        portfolio_id="test_portfolio",
        trading_pair=pair,
        side=PositionSide.LONG,
        volume=Volume(Decimal("0.5"), Currency.BTC),
        entry_price=Price(Decimal("50000"), Currency.USDT),
        current_price=Price(Decimal("51000"), Currency.USDT),
        unrealized_pnl=None,
        realized_pnl=None,
        margin_used=None,
        leverage=Decimal("1"),
        created_at=Timestamp(datetime.now()),
        updated_at=Timestamp(datetime.now()),
        closed_at=None,
        stop_loss=None,
        take_profit=None,
        metadata={}
    )
    assert position.id == "pos_001"
    assert position.trading_pair.symbol == "BTC/USDT"
    assert position.side == PositionSide.LONG
    assert position.volume.value == Decimal("0.5")
    assert position.entry_price.value == Decimal("50000")
    assert position.current_price.value == Decimal("51000")
def test_order_status_transitions() -> None:
    order = Order(
        id=uuid4(),
        strategy_id=uuid4(),
        signal_id=None,
        exchange_order_id="ex124",
        trading_pair="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=Decimal("0.2"),
        price=Money(Decimal("50500"), Currency.USDT),
        stop_price=None,
        status=OrderStatus.OPEN,
        filled_quantity=Decimal("0.0"),
        average_price=None,
        commission=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        filled_at=None,
        metadata={}
    )
    order.status = OrderStatus.FILLED
    assert order.status == OrderStatus.FILLED
    order.status = OrderStatus.CANCELLED
    assert order.status == OrderStatus.CANCELLED
def test_trade_metadata() -> None:
    trade = Trade(
        id="trade_002",
        symbol="BTC/USDT",
        side="sell",
        price=Price(Decimal("51000"), Currency.USDT),
        volume=Volume(Decimal("0.3"), Currency.BTC),
        executed_at=Timestamp(datetime.now()),
        fee=Money(Decimal("15"), Currency.USDT),
        realized_pnl=None,
        metadata={"note": "test"}
    )
    assert trade.metadata["note"] == "test"
def test_position_pnl() -> None:
    pair = TradingPair(symbol="BTC/USDT", base_currency=Currency.BTC, quote_currency=Currency.USDT)
    position = Position(
        id="pos_002",
        portfolio_id="test_portfolio",
        trading_pair=pair,
        side=PositionSide.LONG,
        volume=Volume(Decimal("1.0"), Currency.BTC),
        entry_price=Price(Decimal("50000"), Currency.USDT),
        current_price=Price(Decimal("52000"), Currency.USDT),
        unrealized_pnl=None,
        realized_pnl=None,
        margin_used=None,
        leverage=Decimal("1"),
        created_at=Timestamp(datetime.now()),
        updated_at=Timestamp(datetime.now()),
        closed_at=None,
        stop_loss=None,
        take_profit=None,
        metadata={}
    )
    pnl = (position.current_price.value - position.entry_price.value) * position.volume.value
    assert pnl == Decimal("2000") 
