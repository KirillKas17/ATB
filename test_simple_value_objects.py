#!/usr/bin/env python3
"""
Простые тесты Value Objects без pytest.
"""

from decimal import Decimal
from datetime import datetime
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage


def test_price_creation():
    """Тест создания Price."""
    price = Price(amount=Decimal("45000.0"), currency="USD")
    assert price.amount == Decimal("45000.0")
    assert price.currency == "USD"
    assert str(price) == "45000.0 USD"


def test_price_validation():
    """Тест валидации Price."""
    # Позитивный тест
    price = Price(amount=Decimal("100.0"), currency="BTC")
    assert price.amount == Decimal("100.0")
    
    # Негативный тест
    try:
        Price(amount=Decimal("-100.0"), currency="USD")
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_volume_creation():
    """Тест создания Volume."""
    volume = Volume(amount=Decimal("1.5"))
    assert volume.amount == Decimal("1.5")
    assert str(volume) == "1.5"


def test_volume_validation():
    """Тест валидации Volume."""
    # Позитивный тест
    volume = Volume(amount=Decimal("10.0"))
    assert volume.amount == Decimal("10.0")
    
    # Негативный тест
    try:
        Volume(amount=Decimal("-5.0"))
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_currency_enum():
    """Тест Currency enum."""
    btc = Currency.BTC
    assert btc.code == "BTC"
    assert btc.value == "BTC"
    
    usd = Currency.USD
    assert usd.code == "USD"
    assert usd.value == "USD"
    
    eth = Currency.ETH
    assert eth.code == "ETH"
    
    eur = Currency.EUR
    assert eur.code == "EUR"


def test_timestamp_creation():
    """Тест создания Timestamp."""
    ts = Timestamp.now()
    assert ts.value is not None
    assert isinstance(ts.value, datetime)
    
    # Проверяем что timestamp можно конвертировать в строку
    ts_str = str(ts)
    assert isinstance(ts_str, str)
    assert len(ts_str) > 10  # Должен быть в ISO формате


def test_timestamp_from_iso():
    """Тест создания Timestamp из ISO строки."""
    iso_str = "2024-01-15T10:30:00+00:00"
    ts = Timestamp.from_iso(iso_str)
    assert ts.value is not None
    assert isinstance(ts.value, datetime)


def test_money_creation():
    """Тест создания Money."""
    money = Money(amount=Decimal("1000.0"), currency=Currency.USD)
    assert money.amount == Decimal("1000.0")
    assert money.currency == Currency.USD
    assert str(money) == "1000.0 USD"


def test_money_validation():
    """Тест валидации Money."""
    # Позитивный тест
    money = Money(amount=Decimal("500.0"), currency=Currency.BTC)
    assert money.amount == Decimal("500.0")
    
    # Негативный тест
    try:
        Money(amount=Decimal("-100.0"), currency=Currency.USD)
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_percentage_creation():
    """Тест создания Percentage."""
    percent = Percentage(value=Decimal("25.5"))
    assert percent.value == Decimal("25.5")
    assert str(percent) == "25.5%"


def test_percentage_validation():
    """Тест валидации Percentage."""
    # Позитивные тесты
    percent1 = Percentage(value=Decimal("0"))
    assert percent1.value == Decimal("0")
    
    percent2 = Percentage(value=Decimal("100"))
    assert percent2.value == Decimal("100")
    
    percent3 = Percentage(value=Decimal("50.75"))
    assert percent3.value == Decimal("50.75")
    
    # Негативные тесты
    try:
        Percentage(value=Decimal("-5"))
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение
    
    try:
        Percentage(value=Decimal("105"))
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_value_object_immutability():
    """Тест неизменяемости value objects."""
    price = Price(amount=Decimal("100.0"), currency="USD")
    volume = Volume(amount=Decimal("5.0"))
    
    # Попытка изменить значения должна вызвать ошибку
    try:
        price.amount = Decimal("200.0")
        raise AssertionError("Should have raised error - Price should be immutable")
    except (AttributeError, TypeError):
        pass  # Ожидаемое исключение
    
    try:
        volume.amount = Decimal("10.0")
        raise AssertionError("Should have raised error - Volume should be immutable")
    except (AttributeError, TypeError):
        pass  # Ожидаемое исключение


def test_decimal_precision():
    """Тест точности Decimal в value objects."""
    # Проверяем что Decimal сохраняет точность
    price1 = Price(amount=Decimal("0.00000001"), currency="BTC")
    assert price1.amount == Decimal("0.00000001")
    
    price2 = Price(amount=Decimal("99999999.99999999"), currency="USD")
    assert price2.amount == Decimal("99999999.99999999")
    
    volume = Volume(amount=Decimal("0.12345678"))
    assert volume.amount == Decimal("0.12345678")