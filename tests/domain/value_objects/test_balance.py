"""
Unit тесты для domain/value_objects/balance.py.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
from decimal import Decimal

from domain.value_objects.balance import Balance
from domain.exceptions.base_exceptions import ValidationError


class TestBalance:
    """Тесты для Balance."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "currency": "USD",
            "available": Decimal("1000.50"),
            "reserved": Decimal("100.00"),
            "total": Decimal("1100.50")
        }
    
    def test_creation(self, sample_data):
        """Тест создания Balance."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        assert balance.currency == sample_data["currency"]
        assert balance.available == sample_data["available"]
        assert balance.reserved == sample_data["reserved"]
        assert balance.total == sample_data["total"]
    
    def test_creation_with_total(self, sample_data):
        """Тест создания Balance с явным указанием total."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"],
            total=sample_data["total"]
        )
        
        assert balance.total == sample_data["total"]
    
    def test_validation_negative_available(self):
        """Тест валидации отрицательного available."""
        with pytest.raises(ValidationError, match="Available balance cannot be negative"):
            Balance(
                currency="USD",
                available=Decimal("-100.00"),
                reserved=Decimal("0.00")
            )
    
    def test_validation_negative_reserved(self):
        """Тест валидации отрицательного reserved."""
        with pytest.raises(ValidationError, match="Reserved balance cannot be negative"):
            Balance(
                currency="USD",
                available=Decimal("100.00"),
                reserved=Decimal("-50.00")
            )
    
    def test_validation_invalid_total(self):
        """Тест валидации некорректного total."""
        with pytest.raises(ValidationError, match="Total must equal available plus reserved"):
            Balance(
                currency="USD",
                available=Decimal("100.00"),
                reserved=Decimal("50.00"),
                total=Decimal("200.00")
            )
    
    def test_validation_empty_currency(self):
        """Тест валидации пустой валюты."""
        with pytest.raises(ValidationError, match="Currency cannot be empty"):
            Balance(
                currency="",
                available=Decimal("100.00"),
                reserved=Decimal("0.00")
            )
    
    def test_add_available(self, sample_data):
        """Тест добавления к available."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        new_balance = balance.add_available(Decimal("100.00"))
        
        assert new_balance.available == Decimal("1100.50")
        assert new_balance.reserved == sample_data["reserved"]
        assert new_balance.total == Decimal("1200.50")
    
    def test_subtract_available(self, sample_data):
        """Тест вычитания из available."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        new_balance = balance.subtract_available(Decimal("100.00"))
        
        assert new_balance.available == Decimal("900.50")
        assert new_balance.reserved == sample_data["reserved"]
        assert new_balance.total == Decimal("1000.50")
    
    def test_subtract_available_insufficient_funds(self, sample_data):
        """Тест вычитания при недостаточных средствах."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        with pytest.raises(ValidationError, match="Insufficient available balance"):
            balance.subtract_available(Decimal("2000.00"))
    
    def test_reserve_funds(self, sample_data):
        """Тест резервирования средств."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        new_balance = balance.reserve_funds(Decimal("50.00"))
        
        assert new_balance.available == Decimal("950.50")
        assert new_balance.reserved == Decimal("150.00")
        assert new_balance.total == sample_data["total"]
    
    def test_reserve_funds_insufficient_available(self, sample_data):
        """Тест резервирования при недостаточных available."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        with pytest.raises(ValidationError, match="Insufficient available balance"):
            balance.reserve_funds(Decimal("2000.00"))
    
    def test_release_reserved(self, sample_data):
        """Тест освобождения зарезервированных средств."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        new_balance = balance.release_reserved(Decimal("50.00"))
        
        assert new_balance.available == Decimal("1050.50")
        assert new_balance.reserved == Decimal("50.00")
        assert new_balance.total == sample_data["total"]
    
    def test_release_reserved_insufficient_reserved(self, sample_data):
        """Тест освобождения при недостаточных reserved."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        with pytest.raises(ValidationError, match="Insufficient reserved balance"):
            balance.release_reserved(Decimal("200.00"))
    
    def test_consume_reserved(self, sample_data):
        """Тест потребления зарезервированных средств."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        new_balance = balance.consume_reserved(Decimal("50.00"))
        
        assert new_balance.available == sample_data["available"]
        assert new_balance.reserved == Decimal("50.00")
        assert new_balance.total == Decimal("1050.50")
    
    def test_consume_reserved_insufficient_reserved(self, sample_data):
        """Тест потребления при недостаточных reserved."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        with pytest.raises(ValidationError, match="Insufficient reserved balance"):
            balance.consume_reserved(Decimal("200.00"))
    
    def test_is_sufficient_available(self, sample_data):
        """Тест проверки достаточности available."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        assert balance.is_sufficient_available(Decimal("500.00")) is True
        assert balance.is_sufficient_available(Decimal("2000.00")) is False
    
    def test_is_sufficient_reserved(self, sample_data):
        """Тест проверки достаточности reserved."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        assert balance.is_sufficient_reserved(Decimal("50.00")) is True
        assert balance.is_sufficient_reserved(Decimal("200.00")) is False
    
    def test_to_dict(self, sample_data):
        """Тест сериализации в словарь."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        result = balance.to_dict()
        
        assert result["currency"] == sample_data["currency"]
        assert result["available"] == str(sample_data["available"])
        assert result["reserved"] == str(sample_data["reserved"])
        assert result["total"] == str(sample_data["total"])
    
    def test_from_dict(self, sample_data):
        """Тест десериализации из словаря."""
        data = {
            "currency": sample_data["currency"],
            "available": str(sample_data["available"]),
            "reserved": str(sample_data["reserved"]),
            "total": str(sample_data["total"])
        }
        
        balance = Balance.from_dict(data)
        
        assert balance.currency == sample_data["currency"]
        assert balance.available == sample_data["available"]
        assert balance.reserved == sample_data["reserved"]
        assert balance.total == sample_data["total"]
    
    def test_equality(self, sample_data):
        """Тест равенства объектов."""
        balance1 = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        balance2 = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        assert balance1 == balance2
        assert hash(balance1) == hash(balance2)
    
    def test_inequality(self, sample_data):
        """Тест неравенства объектов."""
        balance1 = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        balance2 = Balance(
            currency=sample_data["currency"],
            available=Decimal("2000.00"),
            reserved=sample_data["reserved"]
        )
        
        assert balance1 != balance2
        assert hash(balance1) != hash(balance2)
    
    def test_str_representation(self, sample_data):
        """Тест строкового представления."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        expected = f"Balance(USD: available=1000.50, reserved=100.00, total=1100.50)"
        assert str(balance) == expected
    
    def test_repr_representation(self, sample_data):
        """Тест repr представления."""
        balance = Balance(
            currency=sample_data["currency"],
            available=sample_data["available"],
            reserved=sample_data["reserved"]
        )
        
        expected = f"Balance(currency='USD', available=Decimal('1000.50'), reserved=Decimal('100.00'), total=Decimal('1100.50'))"
        assert repr(balance) == expected 