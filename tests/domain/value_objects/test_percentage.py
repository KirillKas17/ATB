"""Тесты для Percentage value object."""

import dataclasses
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from domain.value_objects.percentage import Percentage


class TestPercentage:
    """Тесты для класса Percentage."""

    def test_percentage_creation(self) -> None:
        """Тест создания процента."""
        p1 = Percentage(Decimal("10.5"))
        assert p1.value == Decimal("10.5")
        
        p2 = Percentage(10.5)
        assert p2.value == Decimal("10.5")
        
        p3 = Percentage(10)
        assert p3.value == Decimal("10")

    def test_percentage_immutability(self) -> None:
        """Тест неизменяемости."""
        p = Percentage(Decimal("10.5"))
        original_value = p.value
        
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.value = Decimal("20.0")
            
        assert p.value == original_value

    def test_percentage_addition(self) -> None:
        """Тест сложения процентов."""
        p1 = Percentage(Decimal("10.5"))
        p2 = Percentage(Decimal("5.5"))
        
        result = p1 + p2
        assert result.value == Decimal("16.0")
        
        # Сложение с числом
        result = p1 + 5.5
        assert result.value == Decimal("16.0")

    def test_percentage_subtraction(self) -> None:
        """Тест вычитания процентов."""
        p1 = Percentage(Decimal("10.5"))
        p2 = Percentage(Decimal("3.5"))
        
        result = p1 - p2
        assert result.value == Decimal("7.0")
        
        # Вычитание числа
        result = p1 - 3.5
        assert result.value == Decimal("7.0")

    def test_percentage_multiplication(self) -> None:
        """Тест умножения процента."""
        p = Percentage(Decimal("10.0"))
        
        result = p * 2
        assert result.value == Decimal("20.0")
        
        result = p * Decimal("1.5")
        assert result.value == Decimal("15.0")

    def test_percentage_division(self) -> None:
        """Тест деления процента."""
        p1 = Percentage(Decimal("10.0"))
        p2 = Percentage(Decimal("2.0"))
        
        result = p1 / p2
        assert result == Decimal("5.0")
        
        result = p1 / 2
        assert result.value == Decimal("5.0")

    def test_percentage_division_by_zero(self) -> None:
        """Тест деления на ноль."""
        p = Percentage(Decimal("10.0"))
        
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            p / 0
            
        with pytest.raises(ValueError, match="Cannot divide by zero percentage"):
            p / Percentage(Decimal("0"))

    def test_percentage_comparison(self) -> None:
        """Тест сравнения процентов."""
        p1 = Percentage(Decimal("10.0"))
        p2 = Percentage(Decimal("20.0"))
        p3 = Percentage(Decimal("10.0"))
        
        assert p1 < p2
        assert p2 > p1
        assert p1 == p3
        assert p1 <= p3
        assert p1 >= p3

    def test_percentage_comparison_with_non_percentage(self) -> None:
        """Тест сравнения с не-процентами."""
        p = Percentage(Decimal("10.0"))
        
        with pytest.raises(TypeError, match="Can only compare Percentage with Percentage"):
            _ = p < 10

    def test_percentage_string_representation(self) -> None:
        """Тест строкового представления."""
        p = Percentage(Decimal("10.5"))
        assert str(p) == "10.50%"
        
        p2 = Percentage(Decimal("0"))
        assert str(p2) == "0.00%"

    def test_percentage_repr_representation(self) -> None:
        """Тест представления для отладки."""
        p = Percentage(Decimal("10.5"))
        assert repr(p) == "Percentage(10.5)"

    def test_percentage_equality(self) -> None:
        """Тест равенства."""
        p1 = Percentage(Decimal("10.5"))
        p2 = Percentage(Decimal("10.5"))
        p3 = Percentage(Decimal("20.0"))
        
        assert p1 == p2
        assert p1 != p3
        assert p1 != "10.5"  # Сравнение с не-процентом

    def test_percentage_utility_methods(self) -> None:
        """Тест утилитарных методов."""
        p_positive = Percentage(Decimal("10.5"))
        p_negative = Percentage(Decimal("-5.5"))
        p_zero = Percentage(Decimal("0"))
        
        # Проверка на ноль
        assert not p_positive.is_zero()
        assert not p_negative.is_zero()
        assert p_zero.is_zero()
        
        # Проверка на положительность
        assert p_positive.is_positive()
        assert not p_negative.is_positive()
        assert not p_zero.is_positive()
        
        # Проверка на отрицательность
        assert not p_positive.is_negative()
        assert p_negative.is_negative()
        assert not p_zero.is_negative()
        
        # Абсолютное значение
        assert p_negative.abs().value == Decimal("5.5")

    def test_percentage_rounding(self) -> None:
        """Тест округления."""
        p = Percentage(Decimal("10.567"))
        
        rounded = p.round(2)
        assert rounded.value == Decimal("10.57")
        
        rounded = p.round(1)
        assert rounded.value == Decimal("10.6")

    def test_percentage_conversion(self) -> None:
        """Тест преобразований."""
        p = Percentage(Decimal("10.5"))
        
        assert p.to_float() == 10.5
        assert p.to_decimal() == Decimal("10.5")

    def test_percentage_application(self) -> None:
        """Тест применения процента."""
        p = Percentage(Decimal("10.0"))  # 10%
        
        # Применение процента
        result = p.apply_to(100)
        assert result == Decimal("10.0")
        
        # Увеличение на процент
        result = p.increase_by(100)
        assert result == Decimal("110.0")
        
        # Уменьшение на процент
        result = p.decrease_by(100)
        assert result == Decimal("90.0")

    def test_percentage_range_check(self) -> None:
        """Тест проверки диапазона."""
        p = Percentage(Decimal("10.0"))
        
        assert p.is_within_range(5, 15)
        assert not p.is_within_range(15, 20)
        assert not p.is_within_range(0, 5)

    def test_percentage_serialization(self) -> None:
        """Тест сериализации."""
        p = Percentage(Decimal("10.5"))
        
        data = p.to_dict()
        assert data == {"value": "10.5"}
        
        restored = Percentage.from_dict(data)
        assert restored == p

    def test_percentage_class_methods(self) -> None:
        """Тест классовых методов."""
        # Создание нулевого процента
        zero = Percentage.zero()
        assert zero.value == Decimal("0")
        
        # Создание из Decimal (в долях)
        p1 = Percentage.from_decimal(Decimal("0.1"))  # 10%
        assert p1.value == Decimal("10.0")
        
        # Создание из float (в долях)
        p2 = Percentage.from_float(0.05)  # 5%
        assert p2.value == Decimal("5.0")
        
        # Создание из строки
        p3 = Percentage.from_string("15.5%")
        assert p3.value == Decimal("15.5")
        
        p4 = Percentage.from_string("20")
        assert p4.value == Decimal("20")

    def test_percentage_from_string_invalid(self) -> None:
        """Тест создания из невалидной строки."""
        with pytest.raises(ValueError, match="Invalid percentage string"):
            Percentage.from_string("invalid")
            
        with pytest.raises(ValueError, match="Invalid percentage string"):
            Percentage.from_string("")

    def test_percentage_hash(self) -> None:
        """Тест хеширования."""
        p1 = Percentage(Decimal("10.5"))
        p2 = Percentage(Decimal("10.5"))
        p3 = Percentage(Decimal("20.0"))
        
        percentage_set = {p1, p2, p3}
        assert len(percentage_set) == 2  # p1 и p2 одинаковые
        assert p1 in percentage_set 
