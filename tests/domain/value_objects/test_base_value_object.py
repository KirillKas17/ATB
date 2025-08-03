"""
Unit тесты для domain/value_objects/base_value_object.py.
"""

import pytest
from typing import Dict, Any
from decimal import Decimal

from domain.value_objects.base_value_object import BaseValueObject
from domain.exceptions.base_exceptions import ValidationError


class TestValueObject(BaseValueObject):
    """Тестовый Value Object для тестирования базового класса."""
    
    def __init__(self, value: Decimal, name: str):
        self._value = value
        self._name = name
        self._validate()
    
    @property
    def value(self) -> Decimal:
        return self._value
    
    @property
    def name(self) -> str:
        return self._name
    
    def _validate(self) -> None:
        if self._value < 0:
            raise ValidationError("Value cannot be negative")
        if not self._name or not self._name.strip():
            raise ValidationError("Name cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": str(self._value),
            "name": self._name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestValueObject":
        return cls(
            value=Decimal(data["value"]),
            name=data["name"]
        )
    
    def __str__(self) -> str:
        return f"TestValueObject({self._name}: {self._value})"
    
    def __repr__(self) -> str:
        return f"TestValueObject(value={self._value}, name='{self._name}')"


class TestBaseValueObject:
    """Тесты для BaseValueObject."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "value": Decimal("100.50"),
            "name": "test_object"
        }
    
    def test_creation(self, sample_data):
        """Тест создания Value Object."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        assert obj.value == sample_data["value"]
        assert obj.name == sample_data["name"]
    
    def test_validation_negative_value(self, sample_data):
        """Тест валидации отрицательного значения."""
        with pytest.raises(ValidationError, match="Value cannot be negative"):
            TestValueObject(
                value=Decimal("-100.00"),
                name=sample_data["name"]
            )
    
    def test_validation_empty_name(self, sample_data):
        """Тест валидации пустого имени."""
        with pytest.raises(ValidationError, match="Name cannot be empty"):
            TestValueObject(
                value=sample_data["value"],
                name=""
            )
    
    def test_validation_whitespace_name(self, sample_data):
        """Тест валидации имени из пробелов."""
        with pytest.raises(ValidationError, match="Name cannot be empty"):
            TestValueObject(
                value=sample_data["value"],
                name="   "
            )
    
    def test_to_dict(self, sample_data):
        """Тест сериализации в словарь."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        result = obj.to_dict()
        
        assert result["value"] == str(sample_data["value"])
        assert result["name"] == sample_data["name"]
    
    def test_from_dict(self, sample_data):
        """Тест десериализации из словаря."""
        data = {
            "value": str(sample_data["value"]),
            "name": sample_data["name"]
        }
        
        obj = TestValueObject.from_dict(data)
        
        assert obj.value == sample_data["value"]
        assert obj.name == sample_data["name"]
    
    def test_equality(self, sample_data):
        """Тест равенства объектов."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        assert obj1 == obj2
        assert hash(obj1) == hash(obj2)
    
    def test_inequality(self, sample_data):
        """Тест неравенства объектов."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=Decimal("200.00"),
            name=sample_data["name"]
        )
        
        assert obj1 != obj2
        assert hash(obj1) != hash(obj2)
    
    def test_inequality_different_name(self, sample_data):
        """Тест неравенства объектов с разными именами."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=sample_data["value"],
            name="different_name"
        )
        
        assert obj1 != obj2
        assert hash(obj1) != hash(obj2)
    
    def test_inequality_different_type(self, sample_data):
        """Тест неравенства объектов разных типов."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        # Создаем объект другого типа с теми же значениями
        class DifferentValueObject(BaseValueObject):
            def __init__(self, value: Decimal, name: str):
                self._value = value
                self._name = name
            
            @property
            def value(self) -> Decimal:
                return self._value
            
            @property
            def name(self) -> str:
                return self._name
            
            def to_dict(self) -> Dict[str, Any]:
                return {"value": str(self._value), "name": self._name}
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "DifferentValueObject":
                return cls(value=Decimal(data["value"]), name=data["name"])
        
        obj2 = DifferentValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        assert obj1 != obj2
        assert hash(obj1) != hash(obj2)
    
    def test_str_representation(self, sample_data):
        """Тест строкового представления."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        expected = f"TestValueObject({sample_data['name']}: {sample_data['value']})"
        assert str(obj) == expected
    
    def test_repr_representation(self, sample_data):
        """Тест repr представления."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        expected = f"TestValueObject(value={sample_data['value']}, name='{sample_data['name']}')"
        assert repr(obj) == expected
    
    def test_immutability(self, sample_data):
        """Тест неизменяемости объекта."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        # Попытка изменить атрибуты должна вызвать ошибку
        with pytest.raises(AttributeError):
            obj._value = Decimal("200.00")
        
        with pytest.raises(AttributeError):
            obj._name = "new_name"
        
        # Значения должны остаться неизменными
        assert obj.value == sample_data["value"]
        assert obj.name == sample_data["name"]
    
    def test_hash_consistency(self, sample_data):
        """Тест консистентности хеша."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        # Хеши должны быть одинаковыми для одинаковых объектов
        assert hash(obj1) == hash(obj2)
        
        # Хеш должен быть консистентным при повторных вызовах
        assert hash(obj1) == hash(obj1)
        assert hash(obj2) == hash(obj2)
    
    def test_hash_different_objects(self, sample_data):
        """Тест хешей для разных объектов."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=Decimal("200.00"),
            name=sample_data["name"]
        )
        
        obj3 = TestValueObject(
            value=sample_data["value"],
            name="different_name"
        )
        
        # Хеши должны быть разными для разных объектов
        assert hash(obj1) != hash(obj2)
        assert hash(obj1) != hash(obj3)
        assert hash(obj2) != hash(obj3)
    
    def test_comparison_with_non_value_object(self, sample_data):
        """Тест сравнения с не-Value Object."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        # Сравнение с обычным объектом должно возвращать False
        assert obj != "string"
        assert obj != 123
        assert obj != {"key": "value"}
        assert obj != [1, 2, 3]
    
    def test_comparison_with_none(self, sample_data):
        """Тест сравнения с None."""
        obj = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        assert obj != None
    
    def test_from_dict_invalid_data(self, sample_data):
        """Тест десериализации с невалидными данными."""
        # Отсутствует обязательное поле
        data = {"value": str(sample_data["value"])}
        
        with pytest.raises(KeyError):
            TestValueObject.from_dict(data)
        
        # Неверный тип данных
        data = {
            "value": "invalid_decimal",
            "name": sample_data["name"]
        }
        
        with pytest.raises(ValueError):
            TestValueObject.from_dict(data)
    
    def test_abstract_methods_implementation(self):
        """Тест реализации абстрактных методов."""
        # Проверяем, что TestValueObject реализует все необходимые методы
        obj = TestValueObject(value=Decimal("100.00"), name="test")
        
        # Методы должны быть доступны
        assert hasattr(obj, 'to_dict')
        assert hasattr(obj, 'from_dict')
        assert callable(obj.to_dict)
        assert callable(obj.from_dict)
        
        # Методы должны работать
        data = obj.to_dict()
        assert isinstance(data, dict)
        
        new_obj = TestValueObject.from_dict(data)
        assert isinstance(new_obj, TestValueObject)
    
    def test_value_object_in_collections(self, sample_data):
        """Тест использования Value Object в коллекциях."""
        obj1 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        obj2 = TestValueObject(
            value=Decimal("200.00"),
            name="obj2"
        )
        
        obj3 = TestValueObject(
            value=sample_data["value"],
            name=sample_data["name"]
        )
        
        # Тест в set
        obj_set = {obj1, obj2, obj3}
        assert len(obj_set) == 2  # obj1 и obj3 одинаковые
        
        # Тест в dict
        obj_dict = {obj1: "value1", obj2: "value2"}
        assert len(obj_dict) == 2
        assert obj_dict[obj1] == "value1"
        assert obj_dict[obj3] == "value1"  # obj1 и obj3 одинаковые
        
        # Тест в list
        obj_list = [obj1, obj2, obj3]
        assert len(obj_list) == 3
        assert obj_list.count(obj1) == 2  # obj1 и obj3 одинаковые 