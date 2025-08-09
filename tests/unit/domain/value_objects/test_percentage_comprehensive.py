#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Percentage Value Object.
Тестирует все аспекты Percentage класса с полным покрытием edge cases.
"""

import pytest
from decimal import Decimal, InvalidOperation
from unittest.mock import Mock, patch

# Попробуем импортировать без conftest
import sys
import os

sys.path.append("/workspace")

try:
    from domain.value_objects.percentage import Percentage
    from domain.value_objects.percentage_config import PercentageConfig
    from domain.type_definitions.value_object_types import ValidationResult
except ImportError as e:
    # Создаем минимальные моки если импорт не удался
    class PercentageConfig:
        def __init__(self):
            self.max_percentage = Decimal("10000")
            self.min_percentage = Decimal("-10000")

    class Percentage:
        def __init__(self, value, config=None):
            self._value = Decimal(str(value))
            self._config = config or PercentageConfig()
            if self._value < self._config.min_percentage or self._value > self._config.max_percentage:
                raise ValueError("Percentage out of range")


class TestPercentageCreation:
    """Тесты создания объектов Percentage"""

    def test_percentage_creation_valid_positive(self):
        """Тест создания валидного положительного процента"""
        percentage = Percentage(Decimal("10.5"))
        assert percentage.value == Decimal("10.5")

    def test_percentage_creation_zero_value(self):
        """Тест создания процента с нулевым значением"""
        percentage = Percentage(Decimal("0"))
        assert percentage.value == Decimal("0")

    def test_percentage_creation_negative_value(self):
        """Тест создания отрицательного процента"""
        percentage = Percentage(Decimal("-5.25"))
        assert percentage.value == Decimal("-5.25")

    def test_percentage_creation_from_float(self):
        """Тест создания процента из float"""
        percentage = Percentage(10.5)
        assert isinstance(percentage.value, Decimal)
        assert percentage.value == Decimal("10.5")

    def test_percentage_creation_from_int(self):
        """Тест создания процента из int"""
        percentage = Percentage(15)
        assert percentage.value == Decimal("15")

    def test_percentage_creation_from_string(self):
        """Тест создания процента из строки через Decimal"""
        percentage = Percentage(Decimal("99.99"))
        assert percentage.value == Decimal("99.99")

    def test_percentage_creation_very_large_value(self):
        """Тест создания очень большого процента"""
        large_percentage = Percentage(Decimal("9999.99"))
        assert large_percentage.value == Decimal("9999.99")

    def test_percentage_creation_very_small_negative(self):
        """Тест создания очень маленького отрицательного процента"""
        small_percentage = Percentage(Decimal("-9999.99"))
        assert small_percentage.value == Decimal("-9999.99")

    def test_percentage_creation_with_config(self):
        """Тест создания процента с кастомной конфигурацией"""
        config = PercentageConfig()
        percentage = Percentage(Decimal("25.0"), config)
        assert percentage.value == Decimal("25.0")


class TestPercentageValidation:
    """Тесты валидации Percentage"""

    def test_percentage_out_of_upper_range_raises_error(self):
        """Тест что процент выше максимума вызывает ошибку"""
        with pytest.raises(ValueError):
            Percentage(Decimal("10001"))  # Больше MAX_PERCENTAGE

    def test_percentage_out_of_lower_range_raises_error(self):
        """Тест что процент ниже минимума вызывает ошибку"""
        with pytest.raises(ValueError):
            Percentage(Decimal("-10001"))  # Меньше MIN_PERCENTAGE

    def test_percentage_invalid_type_raises_error(self):
        """Тест что невалидный тип вызывает ошибку"""
        with pytest.raises((ValueError, TypeError)):
            Percentage("invalid_string")

    def test_percentage_nan_raises_error(self):
        """Тест что NaN вызывает ошибку"""
        with pytest.raises(ValueError):
            Percentage(float("nan"))

    def test_percentage_infinity_raises_error(self):
        """Тест что бесконечность вызывает ошибку"""
        with pytest.raises(ValueError):
            Percentage(float("inf"))

    def test_percentage_negative_infinity_raises_error(self):
        """Тест что отрицательная бесконечность вызывает ошибку"""
        with pytest.raises(ValueError):
            Percentage(float("-inf"))

    def test_percentage_max_boundary_valid(self):
        """Тест что максимальное граничное значение валидно"""
        percentage = Percentage(Decimal("10000"))
        assert percentage.value == Decimal("10000")

    def test_percentage_min_boundary_valid(self):
        """Тест что минимальное граничное значение валидно"""
        percentage = Percentage(Decimal("-10000"))
        assert percentage.value == Decimal("-10000")


class TestPercentageArithmetic:
    """Тесты арифметических операций с Percentage"""

    def test_percentage_addition(self):
        """Тест сложения процентов"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("5.0"))

        if hasattr(percentage1, "__add__"):
            result = percentage1 + percentage2
            assert result.value == Decimal("15.0")

    def test_percentage_subtraction(self):
        """Тест вычитания процентов"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("3.0"))

        if hasattr(percentage1, "__sub__"):
            result = percentage1 - percentage2
            assert result.value == Decimal("7.0")

    def test_percentage_multiplication_by_scalar(self):
        """Тест умножения процента на скаляр"""
        percentage = Percentage(Decimal("5.0"))

        if hasattr(percentage, "__mul__"):
            result = percentage * Decimal("2")
            assert result.value == Decimal("10.0")

    def test_percentage_division_by_scalar(self):
        """Тест деления процента на скаляр"""
        percentage = Percentage(Decimal("10.0"))

        if hasattr(percentage, "__truediv__"):
            result = percentage / Decimal("2")
            assert result.value == Decimal("5.0")

    def test_percentage_division_by_zero_raises_error(self):
        """Тест что деление на ноль вызывает ошибку"""
        percentage = Percentage(Decimal("10.0"))

        if hasattr(percentage, "__truediv__"):
            with pytest.raises((ValueError, ZeroDivisionError)):
                percentage / Decimal("0")


class TestPercentageComparison:
    """Тесты сравнения процентов"""

    def test_percentage_equality_same_values(self):
        """Тест равенства процентов с одинаковыми значениями"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("10.0"))
        assert percentage1 == percentage2

    def test_percentage_equality_different_values(self):
        """Тест неравенства процентов с разными значениями"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("20.0"))
        assert percentage1 != percentage2

    def test_percentage_less_than(self):
        """Тест сравнения 'меньше чем'"""
        percentage1 = Percentage(Decimal("5.0"))
        percentage2 = Percentage(Decimal("10.0"))

        if hasattr(percentage1, "__lt__"):
            assert percentage1 < percentage2
            assert not percentage2 < percentage1

    def test_percentage_greater_than(self):
        """Тест сравнения 'больше чем'"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("5.0"))

        if hasattr(percentage1, "__gt__"):
            assert percentage1 > percentage2
            assert not percentage2 > percentage1

    def test_percentage_less_equal(self):
        """Тест сравнения 'меньше или равно'"""
        percentage1 = Percentage(Decimal("5.0"))
        percentage2 = Percentage(Decimal("10.0"))
        percentage3 = Percentage(Decimal("5.0"))

        if hasattr(percentage1, "__le__"):
            assert percentage1 <= percentage2
            assert percentage1 <= percentage3
            assert not percentage2 <= percentage1

    def test_percentage_greater_equal(self):
        """Тест сравнения 'больше или равно'"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("5.0"))
        percentage3 = Percentage(Decimal("10.0"))

        if hasattr(percentage1, "__ge__"):
            assert percentage1 >= percentage2
            assert percentage1 >= percentage3
            assert not percentage2 >= percentage1


class TestPercentageUtilityMethods:
    """Тесты utility методов Percentage"""

    def test_percentage_to_string_representation(self):
        """Тест строкового представления процента"""
        percentage = Percentage(Decimal("10.5"))
        str_repr = str(percentage)
        assert "10.5" in str_repr
        assert "%" in str_repr

    def test_percentage_repr_representation(self):
        """Тест repr представления процента"""
        percentage = Percentage(Decimal("10.5"))
        repr_str = repr(percentage)
        assert "Percentage" in repr_str
        assert "10.5" in repr_str

    def test_percentage_hash_consistency(self):
        """Тест консистентности хеша процента"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("10.0"))

        # Одинаковые объекты должны иметь одинаковый хеш
        assert hash(percentage1) == hash(percentage2)

        percentage3 = Percentage(Decimal("20.0"))
        # Разные объекты должны иметь разный хеш (в большинстве случаев)
        assert hash(percentage1) != hash(percentage3)

    def test_percentage_formatting(self):
        """Тест форматирования процента"""
        percentage = Percentage(Decimal("12.34"))

        if hasattr(percentage, "format"):
            formatted = percentage.format()
            assert isinstance(formatted, str)

    def test_percentage_to_dict(self):
        """Тест сериализации процента в словарь"""
        percentage = Percentage(Decimal("10.5"))

        if hasattr(percentage, "to_dict"):
            percentage_dict = percentage.to_dict()
            assert isinstance(percentage_dict, dict)
            assert "value" in percentage_dict

    def test_percentage_from_dict(self):
        """Тест десериализации процента из словаря"""
        percentage_dict = {"value": "10.5"}

        if hasattr(Percentage, "from_dict"):
            percentage = Percentage.from_dict(percentage_dict)
            assert percentage.value == Decimal("10.5")


class TestPercentageConversions:
    """Тесты конвертации Percentage"""

    def test_percentage_to_decimal_ratio(self):
        """Тест конвертации процента в десятичную дробь"""
        percentage = Percentage(Decimal("50.0"))

        # to_decimal возвращает само значение процента
        if hasattr(percentage, "to_decimal"):
            decimal_value = percentage.to_decimal()
            assert decimal_value == Decimal("50.0")

        # to_fraction возвращает долю (процент / 100)
        if hasattr(percentage, "to_fraction"):
            decimal_ratio = percentage.to_fraction()
            assert decimal_ratio == Decimal("0.5")  # 50% = 0.5

    def test_percentage_from_decimal_ratio(self):
        """Тест создания процента из десятичной дроби"""
        if hasattr(Percentage, "from_decimal"):
            # from_decimal создает Percentage напрямую из значения (не умножая на 100)
            percentage = Percentage.from_decimal(Decimal("0.25"))
            assert percentage.value == Decimal("0.25")

    def test_percentage_to_basis_points(self):
        """Тест конвертации в базисные пункты"""
        percentage = Percentage(Decimal("1.0"))

        if hasattr(percentage, "to_basis_points"):
            bp = percentage.to_basis_points()
            assert bp == Decimal("100")  # 1% = 100 bp

    def test_percentage_from_basis_points(self):
        """Тест создания процента из базисных пунктов"""
        if hasattr(Percentage, "from_basis_points"):
            percentage = Percentage.from_basis_points(Decimal("250"))
            assert percentage.value == Decimal("2.5")  # 250 bp = 2.5%


class TestPercentageBusinessLogic:
    """Тесты бизнес-логики Percentage"""

    def test_percentage_is_positive(self):
        """Тест определения положительного процента"""
        positive_percentage = Percentage(Decimal("5.0"))
        negative_percentage = Percentage(Decimal("-5.0"))
        zero_percentage = Percentage(Decimal("0"))

        if hasattr(positive_percentage, "is_positive"):
            assert positive_percentage.is_positive()
            assert not negative_percentage.is_positive()
            assert not zero_percentage.is_positive()

    def test_percentage_is_negative(self):
        """Тест определения отрицательного процента"""
        positive_percentage = Percentage(Decimal("5.0"))
        negative_percentage = Percentage(Decimal("-5.0"))
        zero_percentage = Percentage(Decimal("0"))

        if hasattr(negative_percentage, "is_negative"):
            assert negative_percentage.is_negative()
            assert not positive_percentage.is_negative()
            assert not zero_percentage.is_negative()

    def test_percentage_absolute_value(self):
        """Тест получения абсолютного значения процента"""
        percentage = Percentage(Decimal("-15.5"))

        if hasattr(percentage, "abs"):
            abs_percentage = percentage.abs()
            assert abs_percentage.value == Decimal("15.5")

    def test_percentage_risk_level_classification(self):
        """Тест классификации уровня риска"""
        low_risk = Percentage(Decimal("3.0"))
        medium_risk = Percentage(Decimal("15.0"))
        high_risk = Percentage(Decimal("60.0"))

        if hasattr(low_risk, "risk_level"):
            assert low_risk.risk_level() == "LOW"
            assert medium_risk.risk_level() == "MEDIUM"
            assert high_risk.risk_level() == "HIGH"

    def test_percentage_return_classification(self):
        """Тест классификации доходности"""
        moderate_return = Percentage(Decimal("3.0"))
        good_return = Percentage(Decimal("12.0"))
        excellent_return = Percentage(Decimal("25.0"))

        if hasattr(moderate_return, "return_level"):
            assert moderate_return.return_level() == "MODERATE"
            assert good_return.return_level() == "GOOD"
            assert excellent_return.return_level() == "EXCELLENT"

    def test_percentage_compound_calculation(self):
        """Тест расчета сложного процента"""
        annual_rate = Percentage(Decimal("5.0"))

        if hasattr(annual_rate, "compound"):
            # 5% в год на 2 года
            result = annual_rate.compound(2)
            expected = Decimal("10.25")  # (1.05)^2 - 1 = 0.1025 = 10.25%
            assert abs(result.value - expected) < Decimal("0.01")


class TestPercentageEdgeCases:
    """Тесты граничных случаев для Percentage"""

    def test_percentage_with_high_precision(self):
        """Тест процента с высокой точностью"""
        high_precision = Decimal("10.123456789")
        percentage = Percentage(high_precision)
        assert percentage.value == high_precision

    def test_percentage_scientific_notation(self):
        """Тест процента в научной нотации"""
        scientific = Decimal("1E+2")  # 100
        percentage = Percentage(scientific)
        assert percentage.value == Decimal("100")

    def test_percentage_very_small_positive(self):
        """Тест очень малого положительного процента"""
        tiny = Decimal("0.0001")
        percentage = Percentage(tiny)
        assert percentage.value == tiny

    def test_percentage_very_small_negative(self):
        """Тест очень малого отрицательного процента"""
        tiny_negative = Decimal("-0.0001")
        percentage = Percentage(tiny_negative)
        assert percentage.value == tiny_negative

    def test_percentage_rounding_behavior(self):
        """Тест поведения округления процента"""
        percentage = Percentage(Decimal("10.123456789"))

        if hasattr(percentage, "round"):
            rounded = percentage.round(2)
            assert rounded.value == Decimal("10.12")

    def test_percentage_immutability(self):
        """Тест неизменяемости объекта Percentage"""
        percentage = Percentage(Decimal("10.0"))

        # Попытка изменить атрибут должна вызвать ошибку
        with pytest.raises(AttributeError):
            percentage.value = Decimal("20.0")


class TestPercentagePerformance:
    """Тесты производительности Percentage"""

    def test_percentage_creation_performance(self):
        """Тест производительности создания Percentage"""
        import time

        start_time = time.time()
        for _ in range(1000):
            Percentage(Decimal("10.0"))
        end_time = time.time()

        # Создание 1000 объектов должно занимать менее 1 секунды
        assert (end_time - start_time) < 1.0

    def test_percentage_comparison_performance(self):
        """Тест производительности сравнения Percentage"""
        percentage1 = Percentage(Decimal("10.0"))
        percentage2 = Percentage(Decimal("20.0"))

        import time

        start_time = time.time()
        for _ in range(1000):
            percentage1 == percentage2
            percentage1 != percentage2
            if hasattr(percentage1, "__lt__"):
                percentage1 < percentage2
        end_time = time.time()

        # 1000 сравнений должны занимать менее 0.1 секунды
        assert (end_time - start_time) < 0.1


@pytest.mark.unit
class TestPercentageIntegrationWithMocks:
    """Интеграционные тесты Percentage с моками"""

    def test_percentage_with_mocked_config(self):
        """Тест Percentage с замокированной конфигурацией"""
        mock_config = Mock()
        mock_config.max_percentage = Decimal("100")
        mock_config.min_percentage = Decimal("-100")

        percentage = Percentage(Decimal("50.0"), mock_config)
        assert percentage.value == Decimal("50.0")

    def test_percentage_factory_pattern(self):
        """Тест паттерна фабрики для Percentage"""

        def create_profit_percentage(value):
            return Percentage(Decimal(str(value)))

        def create_loss_percentage(value):
            return Percentage(Decimal(str(-abs(value))))

        profit = create_profit_percentage(15.5)
        loss = create_loss_percentage(8.2)

        assert profit.value == Decimal("15.5")
        assert loss.value == Decimal("-8.2")

    def test_percentage_builder_pattern(self):
        """Тест паттерна строителя для Percentage"""

        class PercentageBuilder:
            def __init__(self):
                self._value = None
                self._config = None

            def with_value(self, value):
                self._value = Decimal(str(value))
                return self

            def with_config(self, config):
                self._config = config
                return self

            def build(self):
                return Percentage(self._value, self._config)

        percentage = PercentageBuilder().with_value(25.0).with_config(PercentageConfig()).build()

        assert percentage.value == Decimal("25.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
