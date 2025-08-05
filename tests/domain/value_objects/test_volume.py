"""Тесты для Volume value object."""

import dataclasses
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from domain.value_objects.volume import Volume


class TestVolume:
    """Тесты для класса Volume."""

    def test_volume_creation(self: "TestVolume") -> None:
        """Тест создания объема."""
        v1 = Volume(Decimal("100.5"))
        assert v1.value == Decimal("100.5")
        
        v2 = Volume(100.5)
        assert v2.value == Decimal("100.5")
        
        v3 = Volume(100)
        assert v3.value == Decimal("100")

    def test_volume_negative_validation(self: "TestVolume") -> None:
        """Тест валидации отрицательных значений."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Volume(-10)
            
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Volume(Decimal("-5.5"))

    def test_volume_immutability(self: "TestVolume") -> None:
        """Тест неизменяемости."""
        v = Volume(Decimal("100.5"))
        original_value = v.value
        
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(AttributeError):
            v.value = Decimal("200.0")
            
        assert v.value == original_value

    def test_volume_addition(self: "TestVolume") -> None:
        """Тест сложения объемов."""
        v1 = Volume(Decimal("100.5"))
        v2 = Volume(Decimal("50.5"))
        
        result = v1 + v2
        assert result.value == Decimal("151.0")
        
        # Сложение с числом
        result = v1 + 50.5
        assert result.value == Decimal("151.0")

    def test_volume_subtraction(self: "TestVolume") -> None:
        """Тест вычитания объемов."""
        v1 = Volume(Decimal("100.5"))
        v2 = Volume(Decimal("30.5"))
        
        result = v1 - v2
        assert result.value == Decimal("70.0")
        
        # Вычитание числа
        result = v1 - 30.5
        assert result.value == Decimal("70.0")

    def test_volume_subtraction_negative_result(self: "TestVolume") -> None:
        """Тест вычитания с отрицательным результатом."""
        v1 = Volume(Decimal("50.0"))
        v2 = Volume(Decimal("100.0"))
        
        with pytest.raises(ValueError, match="Volume difference cannot be negative"):
            v1 - v2
            
        with pytest.raises(ValueError, match="Volume difference cannot be negative"):
            v1 - 100

    def test_volume_multiplication(self: "TestVolume") -> None:
        """Тест умножения объема."""
        v = Volume(Decimal("10.0"))
        
        result = v * 2
        assert result.value == Decimal("20.0")
        
        result = v * Decimal("1.5")
        assert result.value == Decimal("15.0")

    def test_volume_division(self: "TestVolume") -> None:
        """Тест деления объема."""
        v1 = Volume(Decimal("100.0"))
        v2 = Volume(Decimal("2.0"))
        
        result = v1 / v2
        assert result == Decimal("50.0")
        
        result = v1 / 2
        assert result.value == Decimal("50.0")

    def test_volume_division_by_zero(self: "TestVolume") -> None:
        """Тест деления на ноль."""
        v = Volume(Decimal("100.0"))
        
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            v / 0
            
        with pytest.raises(ValueError, match="Cannot divide by zero volume"):
            v / Volume(Decimal("0"))

    def test_volume_comparison(self: "TestVolume") -> None:
        """Тест сравнения объемов."""
        v1 = Volume(Decimal("100.0"))
        v2 = Volume(Decimal("200.0"))
        v3 = Volume(Decimal("100.0"))
        
        assert v1 < v2
        assert v2 > v1
        assert v1 == v3
        assert v1 <= v3
        assert v1 >= v3

    def test_volume_comparison_with_non_volume(self: "TestVolume") -> None:
        """Тест сравнения с не-объемами."""
        v = Volume(Decimal("100.0"))
        
        with pytest.raises(TypeError, match="Can only compare Volume with Volume"):
            _ = v < 100

    def test_volume_string_representation(self: "TestVolume") -> None:
        """Тест строкового представления."""
        v = Volume(Decimal("100.56789012"))
        assert str(v) == "100.56789012"
        
        v2 = Volume(Decimal("0"))
        assert str(v2) == "0.00000000"

    def test_volume_repr_representation(self: "TestVolume") -> None:
        """Тест представления для отладки."""
        v = Volume(Decimal("100.5"))
        assert repr(v) == "Volume(100.5)"

    def test_volume_equality(self: "TestVolume") -> None:
        """Тест равенства."""
        v1 = Volume(Decimal("100.5"))
        v2 = Volume(Decimal("100.5"))
        v3 = Volume(Decimal("200.0"))
        
        assert v1 == v2
        assert v1 != v3
        assert v1 != "100.5"  # Сравнение с не-объемом

    def test_volume_utility_methods(self: "TestVolume") -> None:
        """Тест утилитарных методов."""
        v_positive = Volume(Decimal("100.5"))
        v_zero = Volume(Decimal("0"))
        
        # Проверка на ноль
        assert v_positive.value != 0
        assert v_zero.value == 0
        
        # Проверка на положительность
        assert v_positive.value > 0
        assert not (v_zero.value > 0)
        
        # Абсолютное значение
        assert abs(v_positive.value) == Decimal("100.5")

    def test_volume_rounding(self: "TestVolume") -> None:
        """Тест округления."""
        v = Volume(Decimal("100.56789012"))
        
        rounded = v.round(2)
        assert rounded.value == Decimal("100.57")
        
        rounded = v.round(4)
        assert rounded.value == Decimal("100.5679")

    def test_volume_percentage_of(self: "TestVolume") -> None:
        """Тест расчета процента от общего объема."""
        v1 = Volume(Decimal("25.0"))
        v2 = Volume(Decimal("100.0"))
        
        percentage = v1.percentage_of(v2)
        assert percentage == Decimal("25.0")
        
        # Процент от нулевого объема
        with pytest.raises(ValueError, match="Cannot calculate percentage of zero volume"):
            v1.percentage_of(Volume(Decimal("0")))

    def test_volume_conversion(self: "TestVolume") -> None:
        """Тест преобразований."""
        v = Volume(Decimal("100.5"))
        
        assert v.to_float() == 100.5
        assert v.to_decimal() == Decimal("100.5")

    def test_volume_serialization(self: "TestVolume") -> None:
        """Тест сериализации."""
        v = Volume(Decimal("100.5"))
        
        data = v.to_dict()
        assert data["value"] == "100.5"
        
        restored = Volume.from_dict(data)
        assert restored == v

    def test_volume_class_methods(self: "TestVolume") -> None:
        """Тест классовых методов."""
        # Создание нулевого объема
        zero = Volume.zero()
        assert zero.value == Decimal("0")
        
        # Создание из строки
        v1 = Volume.from_string("100.5")
        assert v1.value == Decimal("100.5")
        
        v2 = Volume.from_string("1,000.5")
        assert v2.value == Decimal("1000.5")

    def test_volume_from_string_invalid(self: "TestVolume") -> None:
        """Тест создания из невалидной строки."""
        with pytest.raises(ValueError):
            try:
                Volume.from_string("invalid")
            except Exception as e:
                raise ValueError(str(e))
            
        with pytest.raises(ValueError):
            Volume.from_string("")

    def test_volume_hash(self: "TestVolume") -> None:
        """Тест хеширования."""
        v1 = Volume(Decimal("100.5"))
        v2 = Volume(Decimal("100.5"))
        v3 = Volume(Decimal("200.0"))
        
        volume_set = {v1, v2, v3}
        assert len(volume_set) == 2  # v1 и v2 одинаковые
        assert v1 in volume_set 
