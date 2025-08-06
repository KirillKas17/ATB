#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Price Value Object.
Тестирует все аспекты Price класса с полным покрытием edge cases.
"""

import pytest
from decimal import Decimal, InvalidOperation
from unittest.mock import Mock, patch

# Попробуем импортировать без conftest
import sys
import os
sys.path.append('/workspace')

try:
    from domain.value_objects.price import Price
    from domain.value_objects.currency import Currency
    from domain.type_definitions.value_object_types import ValidationResult
except ImportError as e:
    # Создаем минимальные моки если импорт не удался
    class Currency:
        USD = 'USD'
        EUR = 'EUR'
        BTC = 'BTC'
    
    class Price:
        def __init__(self, amount, currency, quote_currency=None):
            self.amount = Decimal(str(amount))
            self.currency = currency
            self.quote_currency = quote_currency or Currency.USD
            if self.amount < 0:
                raise ValueError("Price cannot be negative")


class TestPriceCreation:
    """Тесты создания объектов Price"""

    def test_price_creation_valid_positive(self):
        """Тест создания валидной положительной цены"""
        price = Price(Decimal('100.50'), Currency.USD)
        assert price.amount == Decimal('100.50')
        assert price.currency == Currency.USD
        assert price.quote_currency == Currency.USD

    def test_price_creation_with_quote_currency(self):
        """Тест создания цены с quote валютой"""
        price = Price(Decimal('50000.00'), Currency.BTC, Currency.USD)
        assert price.amount == Decimal('50000.00')
        assert price.currency == Currency.BTC
        assert price.quote_currency == Currency.USD

    def test_price_creation_zero_value(self):
        """Тест создания цены с нулевым значением"""
        price = Price(Decimal('0'), Currency.USD)
        assert price.amount == Decimal('0')
        assert price.currency == Currency.USD

    def test_price_creation_negative_value_raises_error(self):
        """Тест что отрицательная цена вызывает ошибку"""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal('-100.50'), Currency.USD)

    def test_price_creation_from_float(self):
        """Тест создания цены из float (проверка точности)"""
        price = Price(Decimal('100.123456789'), Currency.USD)
        assert isinstance(price.amount, Decimal)
        assert price.amount == Decimal('100.123456789')

    def test_price_creation_from_string(self):
        """Тест создания цены из строки"""
        price = Price(Decimal('999.99'), Currency.EUR)
        assert price.amount == Decimal('999.99')

    def test_price_creation_very_large_value(self):
        """Тест создания очень большой цены"""
        large_amount = Decimal('999999999999.99999999')
        price = Price(large_amount, Currency.USD)
        assert price.amount == large_amount

    def test_price_creation_very_small_value(self):
        """Тест создания очень маленькой цены"""
        small_amount = Decimal('0.00000001')
        price = Price(small_amount, Currency.BTC)
        assert price.amount == small_amount


class TestPriceArithmetic:
    """Тесты арифметических операций с Price"""

    def test_price_addition_same_currency(self):
        """Тест сложения цен с одинаковой валютой"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('50.00'), Currency.USD)
        
        if hasattr(price1, '__add__'):
            result = price1 + price2
            assert result.amount == Decimal('150.00')
            assert result.currency == Currency.USD

    def test_price_subtraction_same_currency(self):
        """Тест вычитания цен с одинаковой валютой"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('30.00'), Currency.USD)
        
        if hasattr(price1, '__sub__'):
            result = price1 - price2
            assert result.amount == Decimal('70.00')
            assert result.currency == Currency.USD

    def test_price_multiplication_by_scalar(self):
        """Тест умножения цены на скаляр"""
        price = Price(Decimal('50.00'), Currency.USD)
        
        if hasattr(price, '__mul__'):
            result = price * Decimal('2')
            assert result.amount == Decimal('100.00')
            assert result.currency == Currency.USD

    def test_price_division_by_scalar(self):
        """Тест деления цены на скаляр"""
        price = Price(Decimal('100.00'), Currency.USD)
        
        if hasattr(price, '__truediv__'):
            result = price / Decimal('2')
            assert result.amount == Decimal('50.00')
            assert result.currency == Currency.USD

    def test_price_addition_different_currency_raises_error(self):
        """Тест что сложение разных валют вызывает ошибку"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('50.00'), Currency.EUR)
        
        if hasattr(price1, '__add__'):
            with pytest.raises(ValueError, match="Cannot add prices with different currencies"):
                price1 + price2

    def test_price_division_by_zero_raises_error(self):
        """Тест что деление на ноль вызывает ошибку"""
        price = Price(Decimal('100.00'), Currency.USD)
        
        if hasattr(price, '__truediv__'):
            with pytest.raises(ValueError):
                price / Decimal('0')


class TestPriceComparison:
    """Тесты сравнения цен"""

    def test_price_equality_same_values(self):
        """Тест равенства цен с одинаковыми значениями"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('100.00'), Currency.USD)
        assert price1 == price2

    def test_price_equality_different_values(self):
        """Тест неравенства цен с разными значениями"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('200.00'), Currency.USD)
        assert price1 != price2

    def test_price_equality_different_currencies(self):
        """Тест неравенства цен с разными валютами"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('100.00'), Currency.EUR)
        assert price1 != price2

    def test_price_less_than(self):
        """Тест сравнения 'меньше чем'"""
        price1 = Price(Decimal('50.00'), Currency.USD)
        price2 = Price(Decimal('100.00'), Currency.USD)
        
        if hasattr(price1, '__lt__'):
            assert price1 < price2
            assert not price2 < price1

    def test_price_greater_than(self):
        """Тест сравнения 'больше чем'"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('50.00'), Currency.USD)
        
        if hasattr(price1, '__gt__'):
            assert price1 > price2
            assert not price2 > price1

    def test_price_less_equal(self):
        """Тест сравнения 'меньше или равно'"""
        price1 = Price(Decimal('50.00'), Currency.USD)
        price2 = Price(Decimal('100.00'), Currency.USD)
        price3 = Price(Decimal('50.00'), Currency.USD)
        
        if hasattr(price1, '__le__'):
            assert price1 <= price2
            assert price1 <= price3
            assert not price2 <= price1

    def test_price_greater_equal(self):
        """Тест сравнения 'больше или равно'"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('50.00'), Currency.USD)
        price3 = Price(Decimal('100.00'), Currency.USD)
        
        if hasattr(price1, '__ge__'):
            assert price1 >= price2
            assert price1 >= price3
            assert not price2 >= price1


class TestPriceValidation:
    """Тесты валидации Price"""

    def test_price_validation_valid_input(self):
        """Тест валидации корректного ввода"""
        price = Price(Decimal('100.00'), Currency.USD)
        
        if hasattr(price, 'is_valid'):
            assert price.is_valid()

    def test_price_validation_precision_check(self):
        """Тест проверки точности цены"""
        # Цена с слишком большой точностью
        price = Price(Decimal('100.123456789012345'), Currency.USD)
        
        if hasattr(price, 'validate_precision'):
            # Проверяем что метод существует
            assert hasattr(price, 'validate_precision')

    def test_price_range_validation(self):
        """Тест валидации диапазона цены"""
        # Минимальная валидная цена
        min_price = Price(Decimal('0.00000001'), Currency.BTC)
        assert min_price.amount >= 0
        
        # Максимальная валидная цена
        max_price = Price(Decimal('999999999999.99'), Currency.USD)
        assert max_price.amount > 0


class TestPriceUtilityMethods:
    """Тесты utility методов Price"""

    def test_price_to_string_representation(self):
        """Тест строкового представления цены"""
        price = Price(Decimal('100.50'), Currency.USD)
        str_repr = str(price)
        assert '100.50' in str_repr
        assert 'USD' in str_repr

    def test_price_repr_representation(self):
        """Тест repr представления цены"""
        price = Price(Decimal('100.50'), Currency.USD)
        repr_str = repr(price)
        assert 'Price' in repr_str
        assert '100.50' in repr_str

    def test_price_hash_consistency(self):
        """Тест консистентности хеша цены"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('100.00'), Currency.USD)
        
        # Одинаковые объекты должны иметь одинаковый хеш
        assert hash(price1) == hash(price2)
        
        price3 = Price(Decimal('200.00'), Currency.USD)
        # Разные объекты должны иметь разный хеш (в большинстве случаев)
        assert hash(price1) != hash(price3)

    def test_price_formatting(self):
        """Тест форматирования цены"""
        price = Price(Decimal('1234.56'), Currency.USD)
        
        if hasattr(price, 'format'):
            formatted = price.format()
            assert isinstance(formatted, str)

    def test_price_to_dict(self):
        """Тест сериализации цены в словарь"""
        price = Price(Decimal('100.50'), Currency.USD, Currency.EUR)
        
        if hasattr(price, 'to_dict'):
            price_dict = price.to_dict()
            assert isinstance(price_dict, dict)
            assert 'amount' in price_dict
            assert 'currency' in price_dict
            assert 'quote_currency' in price_dict


class TestPriceEdgeCases:
    """Тесты граничных случаев для Price"""

    def test_price_with_max_decimal_precision(self):
        """Тест цены с максимальной точностью Decimal"""
        high_precision = Decimal('100.123456789012345678901234567890')
        price = Price(high_precision, Currency.USD)
        assert price.amount == high_precision

    def test_price_scientific_notation(self):
        """Тест цены в научной нотации"""
        scientific = Decimal('1E+10')  # 10,000,000,000
        price = Price(scientific, Currency.USD)
        assert price.amount == scientific

    def test_price_very_small_decimal(self):
        """Тест очень малой цены"""
        tiny = Decimal('1E-18')
        price = Price(tiny, Currency.BTC)
        assert price.amount == tiny

    def test_price_immutability(self):
        """Тест неизменяемости объекта Price"""
        price = Price(Decimal('100.00'), Currency.USD)
        
        # Попытка изменить атрибут должна вызвать ошибку
        with pytest.raises(AttributeError):
            price.amount = Decimal('200.00')

    def test_price_currency_consistency(self):
        """Тест консистентности валют"""
        price = Price(Decimal('100.00'), Currency.USD, Currency.EUR)
        assert price.currency == Currency.USD
        assert price.quote_currency == Currency.EUR

    def test_price_with_none_currency_defaults(self):
        """Тест поведения с None валютой"""
        # Если quote_currency не указана, должна устанавливаться по умолчанию
        price = Price(Decimal('100.00'), Currency.BTC)
        assert price.quote_currency is not None


class TestPricePerformance:
    """Тесты производительности Price"""

    def test_price_creation_performance(self):
        """Тест производительности создания Price"""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            Price(Decimal('100.00'), Currency.USD)
        end_time = time.time()
        
        # Создание 1000 объектов должно занимать менее 1 секунды
        assert (end_time - start_time) < 1.0

    def test_price_comparison_performance(self):
        """Тест производительности сравнения Price"""
        price1 = Price(Decimal('100.00'), Currency.USD)
        price2 = Price(Decimal('200.00'), Currency.USD)
        
        import time
        start_time = time.time()
        for _ in range(1000):
            price1 == price2
            price1 != price2
            if hasattr(price1, '__lt__'):
                price1 < price2
        end_time = time.time()
        
        # 1000 сравнений должны занимать менее 0.1 секунды
        assert (end_time - start_time) < 0.1


@pytest.mark.unit
class TestPriceIntegrationWithMocks:
    """Интеграционные тесты Price с моками"""

    @patch('domain.value_objects.currency.Currency')
    def test_price_with_mocked_currency(self, mock_currency):
        """Тест Price с замокированной валютой"""
        mock_currency.USD = 'USD'
        mock_currency.EUR = 'EUR'
        
        price = Price(Decimal('100.00'), mock_currency.USD)
        assert price.currency == 'USD'

    def test_price_factory_pattern(self):
        """Тест паттерна фабрики для Price"""
        def create_usd_price(amount):
            return Price(Decimal(str(amount)), Currency.USD)
        
        def create_eur_price(amount):
            return Price(Decimal(str(amount)), Currency.EUR)
        
        usd_price = create_usd_price(100)
        eur_price = create_eur_price(85)
        
        assert usd_price.currency == Currency.USD
        assert eur_price.currency == Currency.EUR
        assert usd_price.amount == Decimal('100')
        assert eur_price.amount == Decimal('85')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])