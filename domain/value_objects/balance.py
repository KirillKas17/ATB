"""
Value Object для баланса торгового счета.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


@dataclass(frozen=True)
class Balance:
    """
    Баланс торгового счета.
    
    Attributes:
        currency: Валюта баланса
        free: Доступные средства
        used: Используемые средства (в ордерах)
        total: Общий баланс (free + used)
    """
    
    currency: Currency
    free: Money
    used: Money
    
    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.free.currency != self.currency:
            raise ValueError("Валюта free должна совпадать с currency")
        if self.used.currency != self.currency:
            raise ValueError("Валюта used должна совпадать с currency")
        if self.free.amount < 0:
            raise ValueError("Free баланс не может быть отрицательным")
        if self.used.amount < 0:
            raise ValueError("Used баланс не может быть отрицательным")
    
    @property
    def total(self) -> Money:
        """Общий баланс."""
        return Money(
            amount=self.free.amount + self.used.amount,
            currency=self.currency
        )
    
    @property
    def available(self) -> Money:
        """Доступные средства (alias для free)."""
        return self.free
    
    def can_afford(self, amount: Money) -> bool:
        """
        Проверка, достаточно ли средств для операции.
        
        Args:
            amount: Требуемая сумма
            
        Returns:
            True если средств достаточно
        """
        if amount.currency != self.currency:
            return False
        return self.free.amount >= amount.amount
    
    def reserve(self, amount: Money) -> 'Balance':
        """
        Резервирование средств для ордера.
        
        Args:
            amount: Сумма для резервирования
            
        Returns:
            Новый баланс с зарезервированными средствами
        """
        if not self.can_afford(amount):
            raise ValueError(f"Недостаточно средств: {self.free} < {amount}")
        
        return Balance(
            currency=self.currency,
            free=self.free - amount,
            used=self.used + amount
        )
    
    def release(self, amount: Money) -> 'Balance':
        """
        Освобождение зарезервированных средств.
        
        Args:
            amount: Сумма для освобождения
            
        Returns:
            Новый баланс с освобожденными средствами
        """
        if amount.currency != self.currency:
            raise ValueError("Валюта должна совпадать")
        if amount.amount > self.used.amount:
            raise ValueError(f"Нельзя освободить больше чем зарезервировано: {self.used} < {amount}")
        
        return Balance(
            currency=self.currency,
            free=self.free + amount,
            used=self.used - amount
        )
    
    def add(self, amount: Money) -> 'Balance':
        """
        Добавление средств к балансу.
        
        Args:
            amount: Сумма для добавления
            
        Returns:
            Новый баланс с добавленными средствами
        """
        if amount.currency != self.currency:
            raise ValueError("Валюта должна совпадать")
        
        return Balance(
            currency=self.currency,
            free=self.free + amount,
            used=self.used
        )
    
    def subtract(self, amount: Money) -> 'Balance':
        """
        Вычитание средств из баланса.
        
        Args:
            amount: Сумма для вычитания
            
        Returns:
            Новый баланс с вычтенными средствами
        """
        if amount.currency != self.currency:
            raise ValueError("Валюта должна совпадать")
        if amount.amount > self.free.amount:
            raise ValueError(f"Недостаточно средств: {self.free} < {amount}")
        
        return Balance(
            currency=self.currency,
            free=self.free - amount,
            used=self.used
        )
    
    def to_dict(self) -> Dict[str, any]:
        """Конвертация в словарь."""
        return {
            'currency': self.currency.code,
            'free': self.free.to_dict(),
            'used': self.used.to_dict(),
            'total': self.total.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'Balance':
        """Создание из словаря."""
        return cls(
            currency=Currency.from_code(data['currency']),
            free=Money.from_dict(data['free']),
            used=Money.from_dict(data['used'])
        )
    
    def __str__(self) -> str:
        return f"Balance({self.currency.code}: free={self.free}, used={self.used}, total={self.total})"
    
    def __repr__(self) -> str:
        return self.__str__() 