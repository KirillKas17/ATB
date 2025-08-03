"""
Unit тесты для Money value object.

Покрывает:
- Создание и инициализацию денежных сумм
- Математические операции
- Валидацию данных
- Форматирование и конвертацию
"""

import pytest
from decimal import Decimal, InvalidOperation
from typing import Dict, Any
from unittest.mock import Mock, patch

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions import ValidationError


class TestMoney:
    """Тесты для Money value object."""
    
    @pytest.fixture
    def sample_money_data(self) -> Dict[str, Any]:
        """Тестовые данные для денежной суммы."""
        return {
            "amount": "1000.50",
            "currency": "USDT"
        }
    
    def test_money_creation(self, sample_money_data: Dict[str, Any]):
        """Тест создания денежной суммы."""
        money = Money(**sample_money_data)
        
        assert money.amount == Decimal("1000.50")
        assert money.currency == "USDT"
    
    def test_money_creation_with_decimal(self):
        """Тест создания с Decimal."""
        money = Money(amount=Decimal("1000.50"), currency="USDT")
        
        assert money.amount == Decimal("1000.50")
        assert money.currency == "USDT"
    
    def test_money_creation_with_float(self):
        """Тест создания с float."""
        money = Money(amount=1000.50, currency="USDT")
        
        assert money.amount == Decimal("1000.50")
        assert money.currency == "USDT"
    
    def test_money_creation_with_int(self):
        """Тест создания с int."""
        money = Money(amount=1000, currency="USDT")
        
        assert money.amount == Decimal("1000")
        assert money.currency == "USDT"
    
    def test_money_validation_empty_currency(self):
        """Тест валидации пустой валюты."""
        with pytest.raises(ValidationError):
            Money(amount="1000.50", currency="")
    
    def test_money_validation_invalid_amount_string(self):
        """Тест валидации неверной строки суммы."""
        with pytest.raises(ValidationError):
            Money(amount="invalid_amount", currency="USDT")
    
    def test_money_validation_none_amount(self):
        """Тест валидации None суммы."""
        with pytest.raises(ValidationError):
            Money(amount=None, currency="USDT")
    
    def test_money_validation_none_currency(self):
        """Тест валидации None валюты."""
        with pytest.raises(ValidationError):
            Money(amount="1000.50", currency=None)
    
    def test_money_equality(self, sample_money_data: Dict[str, Any]):
        """Тест равенства денежных сумм."""
        money1 = Money(**sample_money_data)
        money2 = Money(**sample_money_data)
        
        assert money1 == money2
    
    def test_money_inequality(self, sample_money_data: Dict[str, Any]):
        """Тест неравенства денежных сумм."""
        money1 = Money(**sample_money_data)
        
        different_data = sample_money_data.copy()
        different_data["amount"] = "2000.00"
        money2 = Money(**different_data)
        
        assert money1 != money2
    
    def test_money_hash(self, sample_money_data: Dict[str, Any]):
        """Тест хеширования денежной суммы."""
        money1 = Money(**sample_money_data)
        money2 = Money(**sample_money_data)
        
        assert hash(money1) == hash(money2)
    
    def test_money_str_representation(self, sample_money_data: Dict[str, Any]):
        """Тест строкового представления."""
        money = Money(**sample_money_data)
        str_repr = str(money)
        
        assert "1000.50" in str_repr
        assert "USDT" in str_repr
    
    def test_money_repr_representation(self, sample_money_data: Dict[str, Any]):
        """Тест repr представления."""
        money = Money(**sample_money_data)
        repr_str = repr(money)
        
        assert "Money" in repr_str
        assert "1000.50" in str_repr
        assert "USDT" in str_repr
    
    def test_money_to_dict(self, sample_money_data: Dict[str, Any]):
        """Тест преобразования в словарь."""
        money = Money(**sample_money_data)
        money_dict = money.to_dict()
        
        assert money_dict["amount"] == "1000.50"
        assert money_dict["currency"] == "USDT"
    
    def test_money_from_dict(self, sample_money_data: Dict[str, Any]):
        """Тест создания из словаря."""
        money = Money.from_dict(sample_money_data)
        
        assert money.amount == Decimal("1000.50")
        assert money.currency == "USDT"
    
    def test_money_addition_same_currency(self):
        """Тест сложения в одной валюте."""
        money1 = Money(amount="1000.00", currency="USDT")
        money2 = Money(amount="500.00", currency="USDT")
        
        result = money1 + money2
        
        assert result.amount == Decimal("1500.00")
        assert result.currency == "USDT"
    
    def test_money_addition_different_currencies(self):
        """Тест сложения в разных валютах."""
        money1 = Money(amount="1000.00", currency="USDT")
        money2 = Money(amount="500.00", currency="BTC")
        
        with pytest.raises(ValueError):
            money1 + money2
    
    def test_money_subtraction_same_currency(self):
        """Тест вычитания в одной валюте."""
        money1 = Money(amount="1000.00", currency="USDT")
        money2 = Money(amount="300.00", currency="USDT")
        
        result = money1 - money2
        
        assert result.amount == Decimal("700.00")
        assert result.currency == "USDT"
    
    def test_money_subtraction_different_currencies(self):
        """Тест вычитания в разных валютах."""
        money1 = Money(amount="1000.00", currency="USDT")
        money2 = Money(amount="300.00", currency="BTC")
        
        with pytest.raises(ValueError):
            money1 - money2
    
    def test_money_multiplication(self):
        """Тест умножения."""
        money = Money(amount="100.00", currency="USDT")
        multiplier = Decimal("2.5")
        
        result = money * multiplier
        
        assert result.amount == Decimal("250.00")
        assert result.currency == "USDT"
    
    def test_money_division(self):
        """Тест деления."""
        money = Money(amount="1000.00", currency="USDT")
        divisor = Decimal("4")
        
        result = money / divisor
        
        assert result.amount == Decimal("250.00")
        assert result.currency == "USDT"
    
    def test_money_division_by_zero(self):
        """Тест деления на ноль."""
        money = Money(amount="1000.00", currency="USDT")
        
        with pytest.raises(ZeroDivisionError):
            money / Decimal("0")
    
    def test_money_comparison_less_than(self):
        """Тест сравнения меньше."""
        money1 = Money(amount="100.00", currency="USDT")
        money2 = Money(amount="200.00", currency="USDT")
        
        assert money1 < money2
        assert not money2 < money1
    
    def test_money_comparison_less_equal(self):
        """Тест сравнения меньше или равно."""
        money1 = Money(amount="100.00", currency="USDT")
        money2 = Money(amount="200.00", currency="USDT")
        money3 = Money(amount="100.00", currency="USDT")
        
        assert money1 <= money2
        assert money1 <= money3
        assert not money2 <= money1
    
    def test_money_comparison_greater_than(self):
        """Тест сравнения больше."""
        money1 = Money(amount="200.00", currency="USDT")
        money2 = Money(amount="100.00", currency="USDT")
        
        assert money1 > money2
        assert not money2 > money1
    
    def test_money_comparison_greater_equal(self):
        """Тест сравнения больше или равно."""
        money1 = Money(amount="200.00", currency="USDT")
        money2 = Money(amount="100.00", currency="USDT")
        money3 = Money(amount="200.00", currency="USDT")
        
        assert money1 >= money2
        assert money1 >= money3
        assert not money2 >= money1
    
    def test_money_comparison_different_currencies(self):
        """Тест сравнения разных валют."""
        money1 = Money(amount="100.00", currency="USDT")
        money2 = Money(amount="200.00", currency="BTC")
        
        with pytest.raises(ValueError):
            money1 < money2
    
    def test_money_abs(self):
        """Тест абсолютного значения."""
        money = Money(amount="-100.00", currency="USDT")
        
        result = abs(money)
        
        assert result.amount == Decimal("100.00")
        assert result.currency == "USDT"
    
    def test_money_round(self):
        """Тест округления."""
        money = Money(amount="100.567", currency="USDT")
        
        result = round(money, 2)
        
        assert result.amount == Decimal("100.57")
        assert result.currency == "USDT"
    
    def test_money_is_zero(self):
        """Тест проверки нулевой суммы."""
        zero_money = Money(amount="0.00", currency="USDT")
        non_zero_money = Money(amount="100.00", currency="USDT")
        
        assert zero_money.is_zero() is True
        assert non_zero_money.is_zero() is False
    
    def test_money_is_positive(self):
        """Тест проверки положительной суммы."""
        positive_money = Money(amount="100.00", currency="USDT")
        negative_money = Money(amount="-100.00", currency="USDT")
        zero_money = Money(amount="0.00", currency="USDT")
        
        assert positive_money.is_positive() is True
        assert negative_money.is_positive() is False
        assert zero_money.is_positive() is False
    
    def test_money_is_negative(self):
        """Тест проверки отрицательной суммы."""
        positive_money = Money(amount="100.00", currency="USDT")
        negative_money = Money(amount="-100.00", currency="USDT")
        zero_money = Money(amount="0.00", currency="USDT")
        
        assert positive_money.is_negative() is False
        assert negative_money.is_negative() is True
        assert zero_money.is_negative() is False
    
    def test_money_format(self):
        """Тест форматирования."""
        money = Money(amount="1234.56", currency="USDT")
        
        formatted = money.format()
        
        assert "1234.56" in formatted
        assert "USDT" in formatted
    
    def test_money_format_with_precision(self):
        """Тест форматирования с точностью."""
        money = Money(amount="1234.567", currency="USDT")
        
        formatted = money.format(precision=2)
        
        assert "1234.57" in formatted
        assert "USDT" in formatted
    
    def test_money_zero(self):
        """Тест создания нулевой суммы."""
        zero_money = Money.zero("USDT")
        
        assert zero_money.amount == Decimal("0")
        assert zero_money.currency == "USDT"
    
    def test_money_from_cents(self):
        """Тест создания из центов."""
        money = Money.from_cents(12345, "USDT")
        
        assert money.amount == Decimal("123.45")
        assert money.currency == "USDT"
    
    def test_money_to_cents(self):
        """Тест преобразования в центы."""
        money = Money(amount="123.45", currency="USDT")
        
        cents = money.to_cents()
        
        assert cents == 12345 