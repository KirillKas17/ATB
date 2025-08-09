#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit тесты для value object Money.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal, InvalidOperation
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


class TestMoney:
    """Тесты для класса Money."""

    def test_money_creation(self: "TestMoney") -> None:
        """Тест создания денежной суммы."""
        money = Money(100.50, Currency.USD)
        assert money.amount == Decimal("100.50000000")
        assert money.currency == Currency.USD

    def test_money_creation_with_different_types(self: "TestMoney") -> None:
        """Тест создания с разными типами данных."""
        # С int
        money1 = Money(100, Currency.BTC)
        assert money1.amount == Decimal("100.00000000")
        # С float
        money2 = Money(100.5, Currency.ETH)
        assert money2.amount == Decimal("100.50000000")
        # С Decimal
        money3 = Money(Decimal("100.75"), Currency.USDT)
        assert money3.amount == Decimal("100.75000000")

    def test_money_invalid_currency(self: "TestMoney") -> None:
        """Тест создания с невалидной валютой."""
        with pytest.raises(ValueError, match="currency must be a Currency instance"):
            Money(100, "USD")

    def test_money_immutability(self: "TestMoney") -> None:
        """Тест неизменяемости."""
        money = Money(100, Currency.USD)
        original_amount = money.amount
        original_currency = money.currency
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(AttributeError):
            money._amount = Decimal("200")
        with pytest.raises(AttributeError):
            money._currency = Currency.EUR
        assert money.amount == original_amount
        assert money.currency == original_currency

    def test_money_addition(self: "TestMoney") -> None:
        """Тест сложения денежных сумм."""
        money1 = Money(100, Currency.USD)
        money2 = Money(50, Currency.USD)
        result = money1 + money2
        assert result.amount == Decimal("150.00000000")
        assert result.currency == Currency.USD

    def test_money_addition_different_currencies(self: "TestMoney") -> None:
        """Тест сложения с разными валютами."""
        money1 = Money(100, Currency.USD)
        money2 = Money(50, Currency.EUR)
        with pytest.raises(ValueError, match="Cannot add money with different currencies"):
            money1 + money2

    def test_money_addition_with_non_money(self: "TestMoney") -> None:
        """Тест сложения с не-денежным типом."""
        money = Money(100, Currency.USD)
        with pytest.raises(TypeError, match="Can only add Money to Money"):
            money + 50

    def test_money_subtraction(self: "TestMoney") -> None:
        """Тест вычитания денежных сумм."""
        money1 = Money(100, Currency.USD)
        money2 = Money(30, Currency.USD)
        result = money1 - money2
        assert result.amount == Decimal("70.00000000")
        assert result.currency == Currency.USD

    def test_money_subtraction_different_currencies(self: "TestMoney") -> None:
        """Тест вычитания с разными валютами."""
        money1 = Money(100, Currency.USD)
        money2 = Money(30, Currency.EUR)
        with pytest.raises(ValueError, match="Cannot subtract money with different currencies"):
            money1 - money2

    def test_money_subtraction_with_non_money(self: "TestMoney") -> None:
        """Тест вычитания с не-денежным типом."""
        money = Money(100, Currency.USD)
        with pytest.raises(TypeError, match="Can only subtract Money from Money"):
            money - 30

    def test_money_multiplication(self: "TestMoney") -> None:
        """Тест умножения на коэффициент."""
        money = Money(100, Currency.USD)
        result = money * 2
        assert result.amount == Decimal("200.00000000")
        assert result.currency == Currency.USD
        result = money * 1.5
        assert result.amount == Decimal("150.00000000")
        result = money * Decimal("0.5")
        assert result.amount == Decimal("50.00000000")

    def test_money_multiplication_with_non_number(self: "TestMoney") -> None:
        """Тест умножения с не-числовым типом."""
        money = Money(100, Currency.USD)
        with pytest.raises(TypeError, match="Factor must be a number"):
            money * "2"

    def test_money_division(self: "TestMoney") -> None:
        """Тест деления на коэффициент."""
        money = Money(100, Currency.USD)
        result = money / 2
        assert result.amount == Decimal("50.00000000")
        assert result.currency == Currency.USD
        result = money / 1.5
        assert result.amount == Decimal("66.66666667")
        result = money / Decimal("4")
        assert result.amount == Decimal("25.00000000")

    def test_money_division_by_zero(self: "TestMoney") -> None:
        """Тест деления на ноль."""
        money = Money(100, Currency.USD)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money / 0

    def test_money_division_with_non_number(self: "TestMoney") -> None:
        """Тест деления с не-числовым типом."""
        money = Money(100, Currency.USD)
        with pytest.raises(TypeError, match="Divisor must be a number"):
            money / "2"

    def test_money_comparison(self: "TestMoney") -> None:
        """Тест сравнения денежных сумм."""
        money1 = Money(100, Currency.USD)
        money2 = Money(200, Currency.USD)
        money3 = Money(100, Currency.USD)
        assert money1 < money2
        assert money2 > money1
        assert money1 == money3
        assert money1 <= money3
        assert money1 >= money3

    def test_money_comparison_different_currencies(self: "TestMoney") -> None:
        """Тест сравнения с разными валютами."""
        money1 = Money(100, Currency.USD)
        money2 = Money(200, Currency.EUR)
        with pytest.raises(ValueError, match="Cannot compare money with different currencies"):
            money1 < money2

    def test_money_comparison_with_non_money(self: "TestMoney") -> None:
        """Тест сравнения с не-денежным типом."""
        money = Money(100, Currency.USD)
        with pytest.raises(TypeError, match="Can only compare Money with Money"):
            _ = money < 100

    def test_money_string_representation(self: "TestMoney") -> None:
        """Тест строкового представления."""
        money = Money(100.50, Currency.USD)
        assert str(money) == "100.50000000 USD"

    def test_money_repr_representation(self: "TestMoney") -> None:
        """Тест представления для отладки."""
        money = Money(100.50, Currency.USD)
        assert repr(money) == "Money(100.50000000, Currency.USD)"

    def test_money_equality(self: "TestMoney") -> None:
        """Тест равенства."""
        money1 = Money(100.50, Currency.USD)
        money2 = Money(100.50, Currency.USD)
        money3 = Money(200.00, Currency.USD)
        money4 = Money(100.50, Currency.EUR)
        assert money1 == money2
        assert money1 != money3
        assert money1 != money4
        assert money1 != "100.50 USD"

    def test_money_conversion(self: "TestMoney") -> None:
        """Тест преобразований."""
        money = Money(100.50, Currency.USD)
        assert money.to_float() == 100.5
        assert money.to_decimal() == Decimal("100.50000000")

    def test_money_utility_methods(self: "TestMoney") -> None:
        """Тест утилитарных методов."""
        money_positive = Money(100.50, Currency.USD)
        money_negative = Money(-50.25, Currency.EUR)
        money_zero = Money(0, Currency.BTC)
        # Проверка на положительность
        assert money_positive.is_positive()
        assert not money_negative.is_positive()
        assert not money_zero.is_positive()
        # Проверка на отрицательность
        assert not money_positive.is_negative()
        assert money_negative.is_negative()
        assert not money_zero.is_negative()
        # Проверка на ноль
        assert not money_positive.is_zero()
        assert not money_negative.is_zero()
        assert money_zero.is_zero()
        # Абсолютное значение
        abs_money = money_negative.abs()
        assert abs_money.amount == Decimal("50.25000000")
        assert abs_money.currency == Currency.EUR

    def test_money_class_methods(self: "TestMoney") -> None:
        """Тест классовых методов."""
        # Создание нулевой суммы
        zero_money = Money.zero(Currency.USD)
        assert zero_money.amount == Decimal("0.00000000")
        assert zero_money.currency == Currency.USD
        # Создание из строки
        money = Money.from_string("100.50", Currency.USD)
        assert money.amount == Decimal("100.50000000")
        assert money.currency == Currency.USD

    def test_money_from_string_invalid(self: "TestMoney") -> None:
        """Тест создания из невалидной строки."""
        with pytest.raises(ValueError, match="Invalid amount string"):
            Money.from_string("invalid", Currency.USD)
        with pytest.raises(ValueError, match="Invalid amount string"):
            Money.from_string("", Currency.USD)

    def test_money_serialization(self: "TestMoney") -> None:
        """Тест сериализации."""
        money = Money(100.50, Currency.USD)
        data = money.to_dict()
        assert data == {"amount": "100.50000000", "currency": "USD"}
        restored = Money.from_dict(data)
        assert restored == money

    def test_money_hash(self: "TestMoney") -> None:
        """Тест хеширования."""
        money1 = Money(100.50, Currency.USD)
        money2 = Money(100.50, Currency.USD)
        money3 = Money(200.00, Currency.USD)
        money_set = {money1, money2, money3}
        assert len(money_set) == 2  # money1 и money2 одинаковые
        assert money1 in money_set

    def test_money_precision(self: "TestMoney") -> None:
        """Тест точности вычислений."""
        money1 = Money(0.1, Currency.BTC)
        money2 = Money(0.2, Currency.BTC)
        result = money1 + money2
        assert result.amount == Decimal("0.30000000")
        # Проверка округления
        money3 = Money(0.123456789, Currency.ETH)
        assert money3.amount == Decimal("0.12345679")  # Округление до 8 знаков
