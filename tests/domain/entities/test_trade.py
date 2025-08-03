"""
Тесты для доменной сущности Trade
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal

from domain.entities.trade import Trade
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


class TestTrade:
    """Тесты для Trade"""

    @pytest.fixture
    def btc_currency(self) -> Any:
        return Currency.BTC

    @pytest.fixture
    def usdt_currency(self) -> Any:
        return Currency.USDT

    def test_trade_creation(self, btc_currency, usdt_currency) -> None:
        """Тест создания сделки"""
        trade = Trade(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            price=Price(value=Decimal("50000"), currency=usdt_currency),
            volume=Volume(value=Decimal("1.0"), currency=btc_currency),
            executed_at=Timestamp(value=datetime.now()),
            fee=Money(Decimal("25"), usdt_currency)
        )

        assert trade.id == "trade_001"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.price.value == Decimal("50000")
        assert trade.volume.value == Decimal("1.0")
        assert trade.fee.value == Decimal("25")
        assert trade.realized_pnl is None
        assert trade.metadata == {}

    def test_empty_id_validation(self, btc_currency, usdt_currency) -> None:
        """Тест валидации пустого ID"""
        with pytest.raises(ValueError, match="Trade ID cannot be empty"):
            Trade(
                id="",
                symbol="BTC/USDT",
                side="buy",
                price=Price(value=Decimal("50000"), currency=usdt_currency),
                volume=Volume(value=Decimal("1.0"), currency=btc_currency),
                executed_at=Timestamp(value=datetime.now()),
                fee=Money(Decimal("25"), usdt_currency)
            )

    def test_invalid_side_validation(self, btc_currency, usdt_currency) -> None:
        """Тест валидации неверной стороны сделки"""
        with pytest.raises(ValueError, match="Trade side must be 'buy' or 'sell'"):
            Trade(
                id="trade_001",
                symbol="BTC/USDT",
                side="invalid",
                price=Price(value=Decimal("50000"), currency=usdt_currency),
                volume=Volume(value=Decimal("1.0"), currency=btc_currency),
                executed_at=Timestamp(value=datetime.now()),
                fee=Money(Decimal("25"), usdt_currency)
            )

    def test_notional_value_calculation(self, btc_currency, usdt_currency) -> None:
        """Тест расчета номинальной стоимости"""
        trade = Trade(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            price=Price(value=Decimal("50000"), currency=usdt_currency),
            volume=Volume(value=Decimal("1.0"), currency=btc_currency),
            executed_at=Timestamp(value=datetime.now()),
            fee=Money(Decimal("25"), usdt_currency)
        )
        
        notional = trade.notional_value
        
        expected_value = Decimal("50000") * Decimal("1.0")
        assert notional.value == expected_value
        assert notional.currency == "USDT"

    def test_is_buy_property(self, btc_currency, usdt_currency) -> None:
        """Тест свойства is_buy"""
        trade = Trade(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            price=Price(value=Decimal("50000"), currency=usdt_currency),
            volume=Volume(value=Decimal("1.0"), currency=btc_currency),
            executed_at=Timestamp(value=datetime.now()),
            fee=Money(Decimal("25"), usdt_currency)
        )
        
        assert trade.is_buy is True
        assert trade.is_sell is False

    def test_is_sell_property(self, btc_currency, usdt_currency) -> None:
        """Тест свойства is_sell"""
        trade = Trade(
            id="trade_001",
            symbol="BTC/USDT",
            side="sell",
            price=Price(value=Decimal("51000"), currency=usdt_currency),
            volume=Volume(value=Decimal("1.0"), currency=btc_currency),
            executed_at=Timestamp(value=datetime.now()),
            fee=Money(Decimal("25.5"), usdt_currency)
        )
        
        assert trade.is_buy is False
        assert trade.is_sell is True
