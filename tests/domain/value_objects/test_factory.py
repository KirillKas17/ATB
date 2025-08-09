"""
Unit тесты для domain/value_objects/factory.py.
"""

import pytest
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timezone

from domain.value_objects.factory import ValueObjectFactory
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.percentage import Percentage
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.value_objects.balance import Balance
from domain.value_objects.signal import Signal, SignalType, SignalDirection, SignalStrength
from domain.value_objects.trading_pair import TradingPair
from domain.exceptions.base_exceptions import ValidationError


class TestValueObjectFactory:
    """Тесты для ValueObjectFactory."""

    @pytest.fixture
    def factory(self):
        """Создание фабрики."""
        return ValueObjectFactory()

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "currency_code": "USD",
            "amount": Decimal("1000.50"),
            "price_value": Decimal("50000.00"),
            "volume_value": Decimal("100.0"),
            "percentage_value": Decimal("25.5"),
            "timestamp_value": datetime.now(timezone.utc),
            "base_currency": "BTC",
            "quote_currency": "USD",
            "signal_id": "signal_001",
        }

    def test_create_money(self, factory, sample_data):
        """Тест создания Money."""
        money = factory.create_money(amount=sample_data["amount"], currency_code=sample_data["currency_code"])

        assert isinstance(money, Money)
        assert money.amount == sample_data["amount"]
        assert money.currency.code == sample_data["currency_code"]

    def test_create_money_with_currency_object(self, factory, sample_data):
        """Тест создания Money с объектом Currency."""
        currency = Currency(code=sample_data["currency_code"])
        money = factory.create_money(amount=sample_data["amount"], currency=currency)

        assert isinstance(money, Money)
        assert money.amount == sample_data["amount"]
        assert money.currency == currency

    def test_create_price(self, factory, sample_data):
        """Тест создания Price."""
        price = factory.create_price(value=sample_data["price_value"], currency_code=sample_data["currency_code"])

        assert isinstance(price, Price)
        assert price.value == sample_data["price_value"]
        assert price.currency == sample_data["currency_code"]

    def test_create_price_with_precision(self, factory, sample_data):
        """Тест создания Price с точностью."""
        price = factory.create_price(
            value=sample_data["price_value"], currency_code=sample_data["currency_code"], precision=4
        )

        assert isinstance(price, Price)
        assert price.value == sample_data["price_value"]
        assert price.currency == sample_data["currency_code"]
        assert price.precision == 4

    def test_create_volume(self, factory, sample_data):
        """Тест создания Volume."""
        volume = factory.create_volume(value=sample_data["volume_value"], unit="BTC")

        assert isinstance(volume, Volume)
        assert volume.value == sample_data["volume_value"]
        assert volume.unit == "BTC"

    def test_create_volume_with_precision(self, factory, sample_data):
        """Тест создания Volume с точностью."""
        volume = factory.create_volume(value=sample_data["volume_value"], unit="BTC", precision=6)

        assert isinstance(volume, Volume)
        assert volume.value == sample_data["volume_value"]
        assert volume.unit == "BTC"
        assert volume.precision == 6

    def test_create_percentage(self, factory, sample_data):
        """Тест создания Percentage."""
        percentage = factory.create_percentage(value=sample_data["percentage_value"])

        assert isinstance(percentage, Percentage)
        assert percentage.value == sample_data["percentage_value"]

    def test_create_percentage_with_precision(self, factory, sample_data):
        """Тест создания Percentage с точностью."""
        percentage = factory.create_percentage(value=sample_data["percentage_value"], precision=2)

        assert isinstance(percentage, Percentage)
        assert percentage.value == sample_data["percentage_value"]
        assert percentage.precision == 2

    def test_create_timestamp(self, factory, sample_data):
        """Тест создания Timestamp."""
        timestamp = factory.create_timestamp(value=sample_data["timestamp_value"])

        assert isinstance(timestamp, Timestamp)
        assert timestamp.value == sample_data["timestamp_value"]

    def test_create_timestamp_from_string(self, factory):
        """Тест создания Timestamp из строки."""
        timestamp_str = "2023-12-01T10:30:00Z"
        timestamp = factory.create_timestamp_from_string(timestamp_str)

        assert isinstance(timestamp, Timestamp)
        assert timestamp.value.year == 2023
        assert timestamp.value.month == 12
        assert timestamp.value.day == 1

    def test_create_currency(self, factory, sample_data):
        """Тест создания Currency."""
        currency = factory.create_currency(code=sample_data["currency_code"])

        assert isinstance(currency, Currency)
        assert currency.code == sample_data["currency_code"]

    def test_create_balance(self, factory, sample_data):
        """Тест создания Balance."""
        balance = factory.create_balance(
            currency_code=sample_data["currency_code"], available=sample_data["amount"], reserved=Decimal("100.00")
        )

        assert isinstance(balance, Balance)
        assert balance.currency == sample_data["currency_code"]
        assert balance.available == sample_data["amount"]
        assert balance.reserved == Decimal("100.00")

    def test_create_signal(self, factory, sample_data):
        """Тест создания Signal."""
        signal = factory.create_signal(
            id=sample_data["signal_id"],
            type=SignalType.TECHNICAL,
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            price=sample_data["price_value"],
            volume=sample_data["volume_value"],
            timestamp=sample_data["timestamp_value"],
        )

        assert isinstance(signal, Signal)
        assert signal.id == sample_data["signal_id"]
        assert signal.type == SignalType.TECHNICAL
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.price == sample_data["price_value"]
        assert signal.volume == sample_data["volume_value"]
        assert signal.timestamp == sample_data["timestamp_value"]

    def test_create_trading_pair(self, factory, sample_data):
        """Тест создания TradingPair."""
        trading_pair = factory.create_trading_pair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100.0"),
            price_precision=2,
            quantity_precision=6,
            min_notional=Decimal("10.00"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.000001"),
        )

        assert isinstance(trading_pair, TradingPair)
        assert trading_pair.base_currency == sample_data["base_currency"]
        assert trading_pair.quote_currency == sample_data["quote_currency"]
        assert trading_pair.min_order_size == Decimal("0.001")
        assert trading_pair.max_order_size == Decimal("100.0")
        assert trading_pair.price_precision == 2
        assert trading_pair.quantity_precision == 6
        assert trading_pair.min_notional == Decimal("10.00")
        assert trading_pair.tick_size == Decimal("0.01")
        assert trading_pair.step_size == Decimal("0.000001")

    def test_create_from_dict_money(self, factory, sample_data):
        """Тест создания Money из словаря."""
        data = {"type": "money", "amount": str(sample_data["amount"]), "currency_code": sample_data["currency_code"]}

        money = factory.create_from_dict(data)

        assert isinstance(money, Money)
        assert money.amount == sample_data["amount"]
        assert money.currency.code == sample_data["currency_code"]

    def test_create_from_dict_price(self, factory, sample_data):
        """Тест создания Price из словаря."""
        data = {
            "type": "price",
            "value": str(sample_data["price_value"]),
            "currency": sample_data["currency_code"],
            "precision": 2,
        }

        price = factory.create_from_dict(data)

        assert isinstance(price, Price)
        assert price.value == sample_data["price_value"]
        assert price.currency == sample_data["currency_code"]
        assert price.precision == 2

    def test_create_from_dict_volume(self, factory, sample_data):
        """Тест создания Volume из словаря."""
        data = {"type": "volume", "value": str(sample_data["volume_value"]), "unit": "BTC", "precision": 6}

        volume = factory.create_from_dict(data)

        assert isinstance(volume, Volume)
        assert volume.value == sample_data["volume_value"]
        assert volume.unit == "BTC"
        assert volume.precision == 6

    def test_create_from_dict_percentage(self, factory, sample_data):
        """Тест создания Percentage из словаря."""
        data = {"type": "percentage", "value": str(sample_data["percentage_value"]), "precision": 2}

        percentage = factory.create_from_dict(data)

        assert isinstance(percentage, Percentage)
        assert percentage.value == sample_data["percentage_value"]
        assert percentage.precision == 2

    def test_create_from_dict_timestamp(self, factory, sample_data):
        """Тест создания Timestamp из словаря."""
        data = {"type": "timestamp", "value": sample_data["timestamp_value"].isoformat()}

        timestamp = factory.create_from_dict(data)

        assert isinstance(timestamp, Timestamp)
        assert timestamp.value == sample_data["timestamp_value"]

    def test_create_from_dict_currency(self, factory, sample_data):
        """Тест создания Currency из словаря."""
        data = {"type": "currency", "code": sample_data["currency_code"]}

        currency = factory.create_from_dict(data)

        assert isinstance(currency, Currency)
        assert currency.code == sample_data["currency_code"]

    def test_create_from_dict_balance(self, factory, sample_data):
        """Тест создания Balance из словаря."""
        data = {
            "type": "balance",
            "currency": sample_data["currency_code"],
            "available": str(sample_data["amount"]),
            "reserved": "100.00",
        }

        balance = factory.create_from_dict(data)

        assert isinstance(balance, Balance)
        assert balance.currency == sample_data["currency_code"]
        assert balance.available == sample_data["amount"]
        assert balance.reserved == Decimal("100.00")

    def test_create_from_dict_signal(self, factory, sample_data):
        """Тест создания Signal из словаря."""
        data = {
            "type": "signal",
            "id": sample_data["signal_id"],
            "type": "technical",
            "direction": "buy",
            "strength": "strong",
            "price": str(sample_data["price_value"]),
            "volume": str(sample_data["volume_value"]),
            "timestamp": sample_data["timestamp_value"].isoformat(),
        }

        signal = factory.create_from_dict(data)

        assert isinstance(signal, Signal)
        assert signal.id == sample_data["signal_id"]
        assert signal.type == SignalType.TECHNICAL
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.price == sample_data["price_value"]
        assert signal.volume == sample_data["volume_value"]

    def test_create_from_dict_trading_pair(self, factory, sample_data):
        """Тест создания TradingPair из словаря."""
        data = {
            "type": "trading_pair",
            "base_currency": sample_data["base_currency"],
            "quote_currency": sample_data["quote_currency"],
            "min_order_size": "0.001",
            "max_order_size": "100.0",
            "price_precision": 2,
            "quantity_precision": 6,
            "min_notional": "10.00",
            "tick_size": "0.01",
            "step_size": "0.000001",
        }

        trading_pair = factory.create_from_dict(data)

        assert isinstance(trading_pair, TradingPair)
        assert trading_pair.base_currency == sample_data["base_currency"]
        assert trading_pair.quote_currency == sample_data["quote_currency"]
        assert trading_pair.min_order_size == Decimal("0.001")
        assert trading_pair.max_order_size == Decimal("100.0")

    def test_create_from_dict_unknown_type(self, factory):
        """Тест создания неизвестного типа."""
        data = {"type": "unknown_type", "value": "test"}

        with pytest.raises(ValueError, match="Unknown value object type: unknown_type"):
            factory.create_from_dict(data)

    def test_create_from_dict_missing_type(self, factory):
        """Тест создания без указания типа."""
        data = {"value": "test"}

        with pytest.raises(ValueError, match="Type is required in data dictionary"):
            factory.create_from_dict(data)

    def test_create_money_with_invalid_currency(self, factory, sample_data):
        """Тест создания Money с невалидной валютой."""
        with pytest.raises(ValidationError):
            factory.create_money(amount=sample_data["amount"], currency_code="INVALID")

    def test_create_price_with_invalid_currency(self, factory, sample_data):
        """Тест создания Price с невалидной валютой."""
        with pytest.raises(ValidationError):
            factory.create_price(value=sample_data["price_value"], currency_code="INVALID")

    def test_create_balance_with_invalid_currency(self, factory, sample_data):
        """Тест создания Balance с невалидной валютой."""
        with pytest.raises(ValidationError):
            factory.create_balance(currency_code="INVALID", available=sample_data["amount"], reserved=Decimal("100.00"))

    def test_create_trading_pair_with_same_currencies(self, factory, sample_data):
        """Тест создания TradingPair с одинаковыми валютами."""
        with pytest.raises(ValidationError, match="Base and quote currencies cannot be the same"):
            factory.create_trading_pair(
                base_currency="BTC",
                quote_currency="BTC",
                min_order_size=Decimal("0.001"),
                max_order_size=Decimal("100.0"),
                price_precision=2,
                quantity_precision=6,
                min_notional=Decimal("10.00"),
                tick_size=Decimal("0.01"),
                step_size=Decimal("0.000001"),
            )

    def test_create_timestamp_from_invalid_string(self, factory):
        """Тест создания Timestamp из невалидной строки."""
        with pytest.raises(ValueError):
            factory.create_timestamp_from_string("invalid_timestamp")

    def test_create_percentage_with_invalid_value(self, factory):
        """Тест создания Percentage с невалидным значением."""
        with pytest.raises(ValidationError):
            factory.create_percentage(value=Decimal("150.0"))  # > 100%

    def test_create_volume_with_invalid_value(self, factory):
        """Тест создания Volume с невалидным значением."""
        with pytest.raises(ValidationError):
            factory.create_volume(value=Decimal("-100.0"), unit="BTC")  # Отрицательное значение

    def test_create_price_with_invalid_value(self, factory, sample_data):
        """Тест создания Price с невалидным значением."""
        with pytest.raises(ValidationError):
            factory.create_price(
                value=Decimal("-100.00"), currency_code=sample_data["currency_code"]  # Отрицательное значение
            )

    def test_create_money_with_invalid_amount(self, factory, sample_data):
        """Тест создания Money с невалидной суммой."""
        with pytest.raises(ValidationError):
            factory.create_money(
                amount=Decimal("-100.00"), currency_code=sample_data["currency_code"]  # Отрицательное значение
            )

    def test_create_balance_with_invalid_amounts(self, factory, sample_data):
        """Тест создания Balance с невалидными суммами."""
        with pytest.raises(ValidationError):
            factory.create_balance(
                currency_code=sample_data["currency_code"],
                available=Decimal("-100.00"),  # Отрицательное значение
                reserved=Decimal("100.00"),
            )

    def test_create_signal_with_invalid_data(self, factory, sample_data):
        """Тест создания Signal с невалидными данными."""
        with pytest.raises(ValidationError):
            factory.create_signal(
                id="",  # Пустой ID
                type=SignalType.TECHNICAL,
                direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG,
                price=sample_data["price_value"],
                volume=sample_data["volume_value"],
                timestamp=sample_data["timestamp_value"],
            )

    def test_get_supported_types(self, factory):
        """Тест получения поддерживаемых типов."""
        supported_types = factory.get_supported_types()

        expected_types = [
            "money",
            "price",
            "volume",
            "percentage",
            "timestamp",
            "currency",
            "balance",
            "signal",
            "trading_pair",
        ]

        for expected_type in expected_types:
            assert expected_type in supported_types

    def test_is_supported_type(self, factory):
        """Тест проверки поддерживаемого типа."""
        assert factory.is_supported_type("money") is True
        assert factory.is_supported_type("price") is True
        assert factory.is_supported_type("unknown_type") is False

    def test_create_batch(self, factory, sample_data):
        """Тест создания нескольких объектов."""
        data_list = [
            {"type": "money", "amount": str(sample_data["amount"]), "currency_code": sample_data["currency_code"]},
            {
                "type": "price",
                "value": str(sample_data["price_value"]),
                "currency": sample_data["currency_code"],
                "precision": 2,
            },
            {"type": "volume", "value": str(sample_data["volume_value"]), "unit": "BTC", "precision": 6},
        ]

        objects = factory.create_batch(data_list)

        assert len(objects) == 3
        assert isinstance(objects[0], Money)
        assert isinstance(objects[1], Price)
        assert isinstance(objects[2], Volume)

    def test_create_batch_with_invalid_data(self, factory):
        """Тест создания batch с невалидными данными."""
        data_list = [
            {"type": "money", "amount": "100.00", "currency_code": "USD"},
            {"type": "unknown_type", "value": "test"},
        ]

        with pytest.raises(ValueError, match="Unknown value object type: unknown_type"):
            factory.create_batch(data_list)
