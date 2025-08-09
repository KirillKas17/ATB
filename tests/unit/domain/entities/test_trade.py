"""
Unit тесты для Trade entity.

Покрывает:
- Создание и инициализацию сделки
- Расчеты P&L
- Валидацию данных
- Бизнес-логику торговли
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime

from domain.entities.trade import Trade
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions import ValidationError


class TestTrade:
    """Тесты для Trade entity."""

    @pytest.fixture
    def sample_trade_data(self) -> Dict[str, Any]:
        """Тестовые данные для сделки."""
        return {
            "trade_id": "trade_001",
            "order_id": "order_001",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": "1.5",
            "price": "50000.00",
            "cost": "75000.00",
            "fee": "75.00",
            "fee_currency": "USDT",
            "timestamp": "2024-01-01T12:00:00Z",
            "exchange_id": "binance",
        }

    def test_trade_creation(self, sample_trade_data: Dict[str, Any]):
        """Тест создания сделки."""
        trade = Trade(**sample_trade_data)

        assert trade.trade_id == "trade_001"
        assert trade.order_id == "order_001"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.amount == Decimal("1.5")
        assert trade.price == Decimal("50000.00")
        assert trade.cost == Decimal("75000.00")
        assert trade.fee == Decimal("75.00")
        assert trade.fee_currency == "USDT"
        assert trade.exchange_id == "binance"

    def test_trade_creation_with_optional_fields(self, sample_trade_data: Dict[str, Any]):
        """Тест создания сделки с опциональными полями."""
        trade_data = sample_trade_data.copy()
        trade_data.update({"taker": True, "info": {"additional": "data"}})

        trade = Trade(**trade_data)

        assert trade.taker is True
        assert trade.info == {"additional": "data"}

    def test_trade_validation_empty_trade_id(self):
        """Тест валидации пустого trade_id."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="",
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                amount="1.5",
                price="50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_empty_order_id(self):
        """Тест валидации пустого order_id."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="",
                symbol="BTC/USDT",
                side="buy",
                amount="1.5",
                price="50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_empty_symbol(self):
        """Тест валидации пустого symbol."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="",
                side="buy",
                amount="1.5",
                price="50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_invalid_side(self):
        """Тест валидации неверной стороны сделки."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="BTC/USDT",
                side="invalid_side",
                amount="1.5",
                price="50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_negative_amount(self):
        """Тест валидации отрицательного количества."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                amount="-1.5",
                price="50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_negative_price(self):
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                amount="1.5",
                price="-50000.00",
                cost="75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_negative_cost(self):
        """Тест валидации отрицательной стоимости."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                amount="1.5",
                price="50000.00",
                cost="-75000.00",
                fee="75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_validation_negative_fee(self):
        """Тест валидации отрицательной комиссии."""
        with pytest.raises(ValidationError):
            Trade(
                trade_id="trade_001",
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                amount="1.5",
                price="50000.00",
                cost="75000.00",
                fee="-75.00",
                fee_currency="USDT",
                timestamp="2024-01-01T12:00:00Z",
                exchange_id="binance",
            )

    def test_trade_equality(self, sample_trade_data: Dict[str, Any]):
        """Тест равенства сделок."""
        trade1 = Trade(**sample_trade_data)
        trade2 = Trade(**sample_trade_data)

        assert trade1 == trade2

    def test_trade_inequality(self, sample_trade_data: Dict[str, Any]):
        """Тест неравенства сделок."""
        trade1 = Trade(**sample_trade_data)

        different_data = sample_trade_data.copy()
        different_data["trade_id"] = "different_trade_id"
        trade2 = Trade(**different_data)

        assert trade1 != trade2

    def test_trade_hash(self, sample_trade_data: Dict[str, Any]):
        """Тест хеширования сделки."""
        trade1 = Trade(**sample_trade_data)
        trade2 = Trade(**sample_trade_data)

        assert hash(trade1) == hash(trade2)

    def test_trade_str_representation(self, sample_trade_data: Dict[str, Any]):
        """Тест строкового представления сделки."""
        trade = Trade(**sample_trade_data)
        str_repr = str(trade)

        assert "trade_001" in str_repr
        assert "BTC/USDT" in str_repr
        assert "buy" in str_repr

    def test_trade_repr_representation(self, sample_trade_data: Dict[str, Any]):
        """Тест repr представления сделки."""
        trade = Trade(**sample_trade_data)
        repr_str = repr(trade)

        assert "Trade" in repr_str
        assert "trade_001" in repr_str

    def test_trade_to_dict(self, sample_trade_data: Dict[str, Any]):
        """Тест преобразования в словарь."""
        trade = Trade(**sample_trade_data)
        trade_dict = trade.to_dict()

        assert trade_dict["trade_id"] == "trade_001"
        assert trade_dict["order_id"] == "order_001"
        assert trade_dict["symbol"] == "BTC/USDT"
        assert trade_dict["side"] == "buy"
        assert trade_dict["amount"] == "1.5"
        assert trade_dict["price"] == "50000.00"
        assert trade_dict["cost"] == "75000.00"
        assert trade_dict["fee"] == "75.00"
        assert trade_dict["fee_currency"] == "USDT"
        assert trade_dict["exchange_id"] == "binance"

    def test_trade_from_dict(self, sample_trade_data: Dict[str, Any]):
        """Тест создания из словаря."""
        trade = Trade.from_dict(sample_trade_data)

        assert trade.trade_id == "trade_001"
        assert trade.order_id == "order_001"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"

    def test_trade_is_buy(self, sample_trade_data: Dict[str, Any]):
        """Тест проверки покупки."""
        trade = Trade(**sample_trade_data)
        assert trade.is_buy() is True

        sell_trade_data = sample_trade_data.copy()
        sell_trade_data["side"] = "sell"
        sell_trade = Trade(**sell_trade_data)
        assert sell_trade.is_buy() is False

    def test_trade_is_sell(self, sample_trade_data: Dict[str, Any]):
        """Тест проверки продажи."""
        trade = Trade(**sample_trade_data)
        assert trade.is_sell() is False

        sell_trade_data = sample_trade_data.copy()
        sell_trade_data["side"] = "sell"
        sell_trade = Trade(**sell_trade_data)
        assert sell_trade.is_sell() is True

    def test_trade_get_base_currency(self, sample_trade_data: Dict[str, Any]):
        """Тест получения базовой валюты."""
        trade = Trade(**sample_trade_data)
        assert trade.get_base_currency() == "BTC"

    def test_trade_get_quote_currency(self, sample_trade_data: Dict[str, Any]):
        """Тест получения котируемой валюты."""
        trade = Trade(**sample_trade_data)
        assert trade.get_quote_currency() == "USDT"

    def test_trade_calculate_net_amount(self, sample_trade_data: Dict[str, Any]):
        """Тест расчета чистого количества."""
        trade = Trade(**sample_trade_data)
        net_amount = trade.calculate_net_amount()

        # Чистое количество = количество - комиссия в базовой валюте
        expected_net = Decimal("1.5") - (Decimal("75.00") / Decimal("50000.00"))
        assert net_amount == expected_net

    def test_trade_calculate_net_cost(self, sample_trade_data: Dict[str, Any]):
        """Тест расчета чистой стоимости."""
        trade = Trade(**sample_trade_data)
        net_cost = trade.calculate_net_cost()

        # Чистая стоимость = стоимость + комиссия
        expected_net_cost = Decimal("75000.00") + Decimal("75.00")
        assert net_cost == expected_net_cost

    def test_trade_calculate_average_price(self, sample_trade_data: Dict[str, Any]):
        """Тест расчета средней цены."""
        trade = Trade(**sample_trade_data)
        avg_price = trade.calculate_average_price()

        # Средняя цена = чистая стоимость / чистое количество
        net_cost = trade.calculate_net_cost()
        net_amount = trade.calculate_net_amount()
        expected_avg_price = net_cost / net_amount

        assert avg_price == expected_avg_price

    def test_trade_get_timestamp_datetime(self, sample_trade_data: Dict[str, Any]):
        """Тест получения временной метки как datetime."""
        trade = Trade(**sample_trade_data)
        timestamp_dt = trade.get_timestamp_datetime()

        assert isinstance(timestamp_dt, datetime)
        assert timestamp_dt.year == 2024
        assert timestamp_dt.month == 1
        assert timestamp_dt.day == 1
        assert timestamp_dt.hour == 12
        assert timestamp_dt.minute == 0
        assert timestamp_dt.second == 0
