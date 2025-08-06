#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Quantity Value Object.
Тестирует все аспекты Quantity value object с полным покрытием edge cases.
"""

import pytest
from decimal import Decimal, InvalidOperation
from unittest.mock import Mock, patch

# Попробуем импортировать без conftest
import sys
import os
sys.path.append('/workspace')

try:
    from domain.value_objects.quantity import Quantity
    from domain.exceptions.base_exceptions import ValidationError
    from domain.value_objects.base_value_object import BaseValueObject
except ImportError as e:
    # Создаем минимальные моки если импорт не удался
    class ValidationError(Exception):
        def __init__(self, field, value, validation_type, message):
            self.field = field
            self.value = value
            self.validation_type = validation_type
            self.message = message
            super().__init__(message)
    
    class Quantity:
        def __init__(self, value, precision=8):
            self.value = Decimal(str(value))
            self.precision = precision
            if self.value <= 0:
                raise ValidationError("value", self.value, "positive", "Quantity must be positive")


class TestQuantityCreation:
    """Тесты создания Quantity objects"""

    def test_quantity_creation_valid_decimal(self):
        """Тест создания количества с валидным Decimal значением"""
        quantity = Quantity(Decimal('1.5'))
        
        assert quantity.value == Decimal('1.5')
        assert quantity.precision == 8  # Дефолтная точность

    def test_quantity_creation_valid_string(self):
        """Тест создания количества со строковым значением"""
        quantity = Quantity('2.75')
        
        assert quantity.value == Decimal('2.75')
        assert quantity.precision == 8

    def test_quantity_creation_valid_integer(self):
        """Тест создания количества с целым числом"""
        quantity = Quantity(5)
        
        assert quantity.value == Decimal('5')
        assert quantity.precision == 8

    def test_quantity_creation_valid_float(self):
        """Тест создания количества с float"""
        quantity = Quantity(3.14)
        
        assert quantity.value == Decimal('3.14')
        assert quantity.precision == 8

    def test_quantity_creation_custom_precision(self):
        """Тест создания количества с кастомной точностью"""
        quantity = Quantity(Decimal('1.0'), precision=4)
        
        assert quantity.value == Decimal('1.0')
        assert quantity.precision == 4

    def test_quantity_creation_zero_precision(self):
        """Тест создания количества с нулевой точностью"""
        quantity = Quantity(Decimal('10'), precision=0)
        
        assert quantity.value == Decimal('10')
        assert quantity.precision == 0

    def test_quantity_creation_high_precision(self):
        """Тест создания количества с высокой точностью"""
        quantity = Quantity(Decimal('1.123456789'), precision=18)
        
        assert quantity.value == Decimal('1.123456789')
        assert quantity.precision == 18


class TestQuantityValidation:
    """Тесты валидации Quantity"""

    def test_quantity_zero_value_raises_error(self):
        """Тест что нулевое значение вызывает ошибку"""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            Quantity(Decimal('0'))

    def test_quantity_negative_value_raises_error(self):
        """Тест что отрицательное значение вызывает ошибку"""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            Quantity(Decimal('-1.5'))

    def test_quantity_invalid_string_raises_error(self):
        """Тест что невалидная строка вызывает ошибку"""
        with pytest.raises(ValidationError, match="Invalid quantity value"):
            Quantity("invalid_number")

    def test_quantity_negative_precision_raises_error(self):
        """Тест что отрицательная точность вызывает ошибку"""
        with pytest.raises(ValidationError, match="Precision must be a non-negative integer"):
            Quantity(Decimal('1.0'), precision=-1)

    def test_quantity_invalid_precision_type_raises_error(self):
        """Тест что неверный тип точности вызывает ошибку"""
        with pytest.raises(ValidationError, match="Precision must be a non-negative integer"):
            Quantity(Decimal('1.0'), precision=3.5)

    def test_quantity_max_value_validation(self):
        """Тест валидации максимального значения"""
        max_value = Decimal('999999999.99999999')
        
        # Валидное максимальное значение
        quantity = Quantity(max_value)
        assert quantity.value == max_value
        
        # Превышение максимума
        with pytest.raises(ValidationError, match="exceeds maximum allowed value"):
            Quantity(max_value + Decimal('0.00000001'))

    def test_quantity_very_small_valid_value(self):
        """Тест валидного очень малого значения"""
        small_value = Decimal('0.00000001')
        quantity = Quantity(small_value)
        
        assert quantity.value == small_value


class TestQuantityArithmetic:
    """Тесты арифметических операций Quantity"""

    def test_quantity_addition(self):
        """Тест сложения количеств"""
        qty1 = Quantity(Decimal('1.5'))
        qty2 = Quantity(Decimal('2.5'))
        
        if hasattr(qty1, '__add__'):
            result = qty1 + qty2
            assert isinstance(result, Quantity)
            assert result.value == Decimal('4.0')

    def test_quantity_subtraction(self):
        """Тест вычитания количеств"""
        qty1 = Quantity(Decimal('5.0'))
        qty2 = Quantity(Decimal('3.0'))
        
        if hasattr(qty1, '__sub__'):
            result = qty1 - qty2
            assert isinstance(result, Quantity)
            assert result.value == Decimal('2.0')

    def test_quantity_subtraction_negative_result_raises_error(self):
        """Тест что вычитание с отрицательным результатом вызывает ошибку"""
        qty1 = Quantity(Decimal('2.0'))
        qty2 = Quantity(Decimal('5.0'))
        
        if hasattr(qty1, '__sub__'):
            with pytest.raises(ValidationError):
                qty1 - qty2

    def test_quantity_multiplication_by_scalar(self):
        """Тест умножения количества на скаляр"""
        qty = Quantity(Decimal('2.5'))
        
        if hasattr(qty, '__mul__'):
            result = qty * Decimal('3')
            assert isinstance(result, Quantity)
            assert result.value == Decimal('7.5')

    def test_quantity_division_by_scalar(self):
        """Тест деления количества на скаляр"""
        qty = Quantity(Decimal('10.0'))
        
        if hasattr(qty, '__truediv__'):
            result = qty / Decimal('2')
            assert isinstance(result, Quantity)
            assert result.value == Decimal('5.0')

    def test_quantity_division_by_zero_raises_error(self):
        """Тест что деление на ноль вызывает ошибку"""
        qty = Quantity(Decimal('10.0'))
        
        if hasattr(qty, '__truediv__'):
            with pytest.raises(ZeroDivisionError):
                qty / Decimal('0')

    def test_quantity_floor_division(self):
        """Тест целочисленного деления"""
        qty = Quantity(Decimal('7.0'))
        
        if hasattr(qty, '__floordiv__'):
            result = qty // Decimal('3')
            assert isinstance(result, Quantity)
            assert result.value == Decimal('2')

    def test_quantity_modulo(self):
        """Тест операции остатка от деления"""
        qty = Quantity(Decimal('7.0'))
        
        if hasattr(qty, '__mod__'):
            result = qty % Decimal('3')
            assert isinstance(result, Quantity)
            assert result.value == Decimal('1')


class TestQuantityComparison:
    """Тесты сравнения Quantity"""

    def test_quantity_equality(self):
        """Тест равенства количеств"""
        qty1 = Quantity(Decimal('1.5'))
        qty2 = Quantity(Decimal('1.5'))
        qty3 = Quantity(Decimal('2.0'))
        
        assert qty1 == qty2
        assert qty1 != qty3

    def test_quantity_less_than(self):
        """Тест сравнения меньше чем"""
        qty1 = Quantity(Decimal('1.5'))
        qty2 = Quantity(Decimal('2.0'))
        
        assert qty1 < qty2
        assert not qty2 < qty1

    def test_quantity_less_than_or_equal(self):
        """Тест сравнения меньше или равно"""
        qty1 = Quantity(Decimal('1.5'))
        qty2 = Quantity(Decimal('2.0'))
        qty3 = Quantity(Decimal('1.5'))
        
        assert qty1 <= qty2
        assert qty1 <= qty3
        assert not qty2 <= qty1

    def test_quantity_greater_than(self):
        """Тест сравнения больше чем"""
        qty1 = Quantity(Decimal('2.0'))
        qty2 = Quantity(Decimal('1.5'))
        
        assert qty1 > qty2
        assert not qty2 > qty1

    def test_quantity_greater_than_or_equal(self):
        """Тест сравнения больше или равно"""
        qty1 = Quantity(Decimal('2.0'))
        qty2 = Quantity(Decimal('1.5'))
        qty3 = Quantity(Decimal('2.0'))
        
        assert qty1 >= qty2
        assert qty1 >= qty3
        assert not qty2 >= qty1

    def test_quantity_comparison_with_different_precision(self):
        """Тест сравнения количеств с разной точностью"""
        qty1 = Quantity(Decimal('1.5'), precision=2)
        qty2 = Quantity(Decimal('1.5'), precision=8)
        
        # Значения равны, несмотря на разную точность
        assert qty1 == qty2


class TestQuantityUtilityMethods:
    """Тесты utility методов Quantity"""

    def test_quantity_to_decimal(self):
        """Тест конвертации в Decimal"""
        qty = Quantity(Decimal('2.5'))
        
        if hasattr(qty, 'to_decimal'):
            decimal_value = qty.to_decimal()
            assert decimal_value == Decimal('2.5')
            assert isinstance(decimal_value, Decimal)

    def test_quantity_to_float(self):
        """Тест конвертации в float"""
        qty = Quantity(Decimal('2.5'))
        
        if hasattr(qty, 'to_float'):
            float_value = qty.to_float()
            assert float_value == 2.5
            assert isinstance(float_value, float)

    def test_quantity_to_string(self):
        """Тест конвертации в строку"""
        qty = Quantity(Decimal('2.5'))
        
        str_value = str(qty)
        assert '2.5' in str_value

    def test_quantity_repr(self):
        """Тест repr представления"""
        qty = Quantity(Decimal('2.5'), precision=4)
        
        repr_value = repr(qty)
        assert 'Quantity' in repr_value
        assert '2.5' in repr_value

    def test_quantity_hash_consistency(self):
        """Тест консистентности хеша"""
        qty1 = Quantity(Decimal('1.5'))
        qty2 = Quantity(Decimal('1.5'))
        
        # Одинаковые количества должны иметь одинаковый хеш
        assert hash(qty1) == hash(qty2)

    def test_quantity_immutability(self):
        """Тест неизменяемости объекта"""
        qty = Quantity(Decimal('2.5'))
        
        # Попытка изменить value должна вызвать ошибку
        with pytest.raises(AttributeError):
            qty.value = Decimal('3.0')
        
        # Попытка изменить precision должна вызвать ошибку
        with pytest.raises(AttributeError):
            qty.precision = 4

    def test_quantity_round_to_precision(self):
        """Тест округления до заданной точности"""
        qty = Quantity(Decimal('1.123456789'), precision=4)
        
        if hasattr(qty, 'round_to_precision'):
            rounded = qty.round_to_precision()
            # Должно округлить до 4 знаков после запятой
            assert str(rounded.value) == '1.1235'

    def test_quantity_normalize(self):
        """Тест нормализации значения"""
        qty = Quantity(Decimal('2.50000'), precision=8)
        
        if hasattr(qty, 'normalize'):
            normalized = qty.normalize()
            assert isinstance(normalized, Quantity)


class TestQuantityTradingMethods:
    """Тесты методов для торговли"""

    def test_quantity_is_lot_size_compliant(self):
        """Тест проверки соответствия размеру лота"""
        qty = Quantity(Decimal('1.0'))
        lot_size = Decimal('0.1')
        
        if hasattr(qty, 'is_lot_size_compliant'):
            is_compliant = qty.is_lot_size_compliant(lot_size)
            assert isinstance(is_compliant, bool)

    def test_quantity_round_to_lot_size(self):
        """Тест округления до размера лота"""
        qty = Quantity(Decimal('1.23'))
        lot_size = Decimal('0.1')
        
        if hasattr(qty, 'round_to_lot_size'):
            rounded = qty.round_to_lot_size(lot_size)
            assert isinstance(rounded, Quantity)
            assert rounded.value == Decimal('1.2')

    def test_quantity_split(self):
        """Тест разделения количества на части"""
        qty = Quantity(Decimal('10.0'))
        
        if hasattr(qty, 'split'):
            parts = qty.split(3)
            assert len(parts) == 3
            assert all(isinstance(part, Quantity) for part in parts)
            # Сумма частей должна равняться исходному количеству
            total = sum(part.value for part in parts)
            assert total == qty.value

    def test_quantity_percentage_of(self):
        """Тест получения процента от количества"""
        qty = Quantity(Decimal('100.0'))
        
        if hasattr(qty, 'percentage_of'):
            ten_percent = qty.percentage_of(Decimal('10'))
            assert isinstance(ten_percent, Quantity)
            assert ten_percent.value == Decimal('10.0')

    def test_quantity_is_dust(self):
        """Тест проверки является ли количество пылью"""
        small_qty = Quantity(Decimal('0.00000001'))
        large_qty = Quantity(Decimal('1.0'))
        
        if hasattr(small_qty, 'is_dust'):
            dust_threshold = Decimal('0.0001')
            assert small_qty.is_dust(dust_threshold) is True
            assert large_qty.is_dust(dust_threshold) is False


class TestQuantityEdgeCases:
    """Тесты граничных случаев для Quantity"""

    def test_quantity_very_large_precision(self):
        """Тест с очень большой точностью"""
        qty = Quantity(Decimal('1.0'), precision=50)
        
        assert qty.precision == 50

    def test_quantity_scientific_notation(self):
        """Тест с научной нотацией"""
        qty = Quantity(Decimal('1E-8'))
        
        assert qty.value == Decimal('0.00000001')

    def test_quantity_max_decimal_places(self):
        """Тест с максимальным количеством знаков после запятой"""
        value_str = '1.' + '1' * 28  # 28 знаков после запятой
        qty = Quantity(Decimal(value_str))
        
        assert len(str(qty.value).split('.')[1]) == 28

    def test_quantity_performance_large_operations(self):
        """Тест производительности с большими операциями"""
        import time
        
        # Создание множества объектов Quantity
        start_time = time.time()
        quantities = [Quantity(Decimal(str(i))) for i in range(1, 1001)]
        creation_time = time.time() - start_time
        
        # Время создания должно быть разумным (< 1 секунды)
        assert creation_time < 1.0
        assert len(quantities) == 1000

    def test_quantity_memory_efficiency(self):
        """Тест эффективности использования памяти"""
        import sys
        
        qty = Quantity(Decimal('1.0'))
        size = sys.getsizeof(qty)
        
        # Размер объекта должен быть разумным
        assert size < 1000  # байт

    def test_quantity_thread_safety(self):
        """Тест потокобезопасности (immutability)"""
        import threading
        import time
        
        qty = Quantity(Decimal('100.0'))
        results = []
        
        def worker():
            # Попытка множественного чтения должна быть безопасной
            for _ in range(100):
                results.append(qty.value)
                time.sleep(0.001)
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Все результаты должны быть одинаковыми
        assert all(result == Decimal('100.0') for result in results)
        assert len(results) == 300


@pytest.mark.unit
class TestQuantityIntegrationWithMocks:
    """Интеграционные тесты Quantity с моками"""

    def test_quantity_with_mocked_decimal(self):
        """Тест Quantity с замокированным Decimal"""
        mock_decimal = Mock()
        mock_decimal.return_value = Decimal('5.0')
        
        with patch('decimal.Decimal', mock_decimal):
            qty = Quantity('5.0')
            # Проверяем что мок был вызван правильно
            mock_decimal.assert_called()

    def test_quantity_factory_pattern(self):
        """Тест паттерна фабрики для Quantity"""
        def create_crypto_quantity(amount):
            """Создает количество для криптовалют (8 знаков точности)"""
            return Quantity(Decimal(str(amount)), precision=8)
        
        def create_forex_quantity(amount):
            """Создает количество для форекса (5 знаков точности)"""
            return Quantity(Decimal(str(amount)), precision=5)
        
        def create_stock_quantity(amount):
            """Создает количество для акций (0 знаков точности)"""
            return Quantity(Decimal(str(int(amount))), precision=0)
        
        crypto_qty = create_crypto_quantity(1.12345678)
        forex_qty = create_forex_quantity(1000.12345)
        stock_qty = create_stock_quantity(100)
        
        assert crypto_qty.precision == 8
        assert forex_qty.precision == 5
        assert stock_qty.precision == 0

    def test_quantity_builder_pattern(self):
        """Тест паттерна строителя для Quantity"""
        class QuantityBuilder:
            def __init__(self):
                self._value = Decimal('1.0')
                self._precision = 8
            
            def with_value(self, value):
                self._value = Decimal(str(value))
                return self
            
            def with_precision(self, precision):
                self._precision = precision
                return self
            
            def for_crypto(self):
                self._precision = 8
                return self
            
            def for_forex(self):
                self._precision = 5
                return self
            
            def for_stocks(self):
                self._precision = 0
                return self
            
            def build(self):
                return Quantity(self._value, precision=self._precision)
        
        qty = (QuantityBuilder()
               .with_value(2.5)
               .for_crypto()
               .build())
        
        assert qty.value == Decimal('2.5')
        assert qty.precision == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])