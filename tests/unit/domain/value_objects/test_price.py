"""
Unit тесты для domain.value_objects.price

Покрывает:
- Создание и валидацию Price объектов
- Арифметические операции
- Сравнения
- Сериализация/десериализация
- Округление и форматирование
"""

import pytest
from decimal import Decimal
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency


class TestPrice:
    """Тесты для Price value object"""

    def test_price_creation_valid(self):
        """Тест создания Price с валидными данными"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        assert price.amount == Decimal("50000.00")
        assert price.currency == Currency.USD

    def test_price_creation_from_string(self):
        """Тест создания Price из строки"""
        price = Price.from_string("50000.00", "USD")
        assert price.amount == Decimal("50000.00")
        assert price.currency == Currency.USD

    def test_price_creation_from_float(self):
        """Тест создания Price из float"""
        price = Price.from_float(50000.00, Currency.USD)
        assert price.amount == Decimal("50000.00")
        assert price.currency == Currency.USD

    def test_price_creation_invalid_amount(self):
        """Тест создания Price с невалидной суммой"""
        with pytest.raises(ValueError):
            Price(amount=Decimal("-100"), currency=Currency.USD)

    def test_price_creation_zero_amount(self):
        """Тест создания Price с нулевой суммой"""
        with pytest.raises(ValueError):
            Price(amount=Decimal("0"), currency=Currency.USD)

    def test_price_addition(self):
        """Тест сложения Price объектов"""
        price1 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price2 = Price(amount=Decimal("1000.00"), currency=Currency.USD)
        result = price1 + price2
        assert result.amount == Decimal("51000.00")
        assert result.currency == Currency.USD

    def test_price_subtraction(self):
        """Тест вычитания Price объектов"""
        price1 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price2 = Price(amount=Decimal("1000.00"), currency=Currency.USD)
        result = price1 - price2
        assert result.amount == Decimal("49000.00")
        assert result.currency == Currency.USD

    def test_price_subtraction_negative_result(self):
        """Тест вычитания с отрицательным результатом"""
        price1 = Price(amount=Decimal("1000.00"), currency=Currency.USD)
        price2 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        with pytest.raises(ValueError):
            price1 - price2

    def test_price_multiplication(self):
        """Тест умножения Price на число"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        result = price * Decimal("1.1")
        assert result.amount == Decimal("55000.00")
        assert result.currency == Currency.USD

    def test_price_division(self):
        """Тест деления Price на число"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        result = price / Decimal("2")
        assert result.amount == Decimal("25000.00")
        assert result.currency == Currency.USD

    def test_price_division_by_zero(self):
        """Тест деления Price на ноль"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        with pytest.raises(ValueError):
            price / Decimal("0")

    def test_price_comparison(self):
        """Тест сравнения Price объектов"""
        price1 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price2 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price3 = Price(amount=Decimal("60000.00"), currency=Currency.USD)

        assert price1 == price2
        assert price1 != price3
        assert price1 < price3
        assert price3 > price1
        assert price1 <= price2
        assert price1 >= price2

    def test_price_percentage_change(self):
        """Тест расчета процентного изменения"""
        old_price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        new_price = Price(amount=Decimal("55000.00"), currency=Currency.USD)
        change = new_price.percentage_change(old_price)
        assert change == Decimal("10.00")

    def test_price_percentage_change_decrease(self):
        """Тест расчета процентного изменения (уменьшение)"""
        old_price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        new_price = Price(amount=Decimal("45000.00"), currency=Currency.USD)
        change = new_price.percentage_change(old_price)
        assert change == Decimal("-10.00")

    def test_price_percentage_change_same_currency(self):
        """Тест расчета процентного изменения с разными валютами"""
        old_price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        new_price = Price(amount=Decimal("55000.00"), currency=Currency.EUR)
        with pytest.raises(ValueError):
            new_price.percentage_change(old_price)

    def test_price_rounding(self):
        """Тест округления Price"""
        price = Price(amount=Decimal("50000.567"), currency=Currency.USD)
        rounded = price.round_to_currency_precision()
        assert rounded.amount == Decimal("50000.57")
        assert rounded.currency == Currency.USD

    def test_price_formatting(self):
        """Тест форматирования Price"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        formatted = price.format()
        assert "50000.00" in formatted
        assert "USD" in formatted

    def test_price_to_dict(self):
        """Тест сериализации Price в словарь"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        data = price.to_dict()
        assert data["amount"] == "50000.00"
        assert data["currency"] == "USD"

    def test_price_from_dict(self):
        """Тест десериализации Price из словаря"""
        data = {"amount": "50000.00", "currency": "USD"}
        price = Price.from_dict(data)
        assert price.amount == Decimal("50000.00")
        assert price.currency == Currency.USD

    def test_price_hash(self):
        """Тест хеширования Price"""
        price1 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price2 = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        price3 = Price(amount=Decimal("60000.00"), currency=Currency.USD)

        assert hash(price1) == hash(price2)
        assert hash(price1) != hash(price3)

    def test_price_repr(self):
        """Тест строкового представления"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        repr_str = repr(price)
        assert "Price" in repr_str
        assert "50000.00" in repr_str
        assert "USD" in repr_str

    def test_price_str(self):
        """Тест строкового представления для пользователя"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        str_repr = str(price)
        assert "50000.00" in str_repr
        assert "USD" in str_repr

    def test_price_validation_precision(self):
        """Тест валидации точности"""
        with pytest.raises(ValueError):
            Price(amount=Decimal("50000.123456"), currency=Currency.USD)

    def test_price_validation_max_amount(self):
        """Тест валидации максимальной суммы"""
        with pytest.raises(ValueError):
            Price(amount=Decimal("999999999.99"), currency=Currency.USD)

    def test_price_with_amount(self):
        """Тест создания нового Price с измененной суммой"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        new_price = price.with_amount(Decimal("60000.00"))
        assert new_price.amount == Decimal("60000.00")
        assert new_price.currency == Currency.USD
        assert price.amount == Decimal("50000.00")  # оригинал не изменился

    def test_price_with_currency(self):
        """Тест создания нового Price с измененной валютой"""
        price = Price(amount=Decimal("50000.00"), currency=Currency.USD)
        new_price = price.with_currency(Currency.EUR)
        assert new_price.amount == Decimal("50000.00")
        assert new_price.currency == Currency.EUR
        assert price.currency == Currency.USD  # оригинал не изменился
