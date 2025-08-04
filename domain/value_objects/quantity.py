"""
Quantity Value Object для торговой системы ATB.
Обеспечивает безопасную работу с торговыми объемами и количествами.
"""

from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Union, Optional
from dataclasses import dataclass

from domain.value_objects.base_value_object import BaseValueObject
from domain.exceptions.base_exceptions import DomainValidationError


@dataclass(frozen=True)
class Quantity(BaseValueObject[Decimal]):
    """
    Value Object для представления количества/объема в торговле.
    
    Обеспечивает:
    - Точные вычисления с использованием Decimal
    - Валидацию положительности
    - Соблюдение точности символа
    - Иммутабельность
    """
    
    value: Decimal
    precision: int = 8  # Точность для большинства криптовалют
    
    def __post_init__(self):
        """Валидация quantity после создания."""
        if not isinstance(self.value, Decimal):
            try:
                # Конвертируем в Decimal если возможно
                object.__setattr__(self, 'value', Decimal(str(self.value)))
            except (InvalidOperation, ValueError) as e:
                raise DomainValidationError(f"Invalid quantity value: {e}")
        
        if self.value <= 0:
            raise DomainValidationError("Quantity must be positive")
        
        if not isinstance(self.precision, int) or self.precision < 0:
            raise DomainValidationError("Precision must be a non-negative integer")
        
        # Проверяем, что значение не превышает максимально возможное
        max_value = Decimal('999999999.99999999')  # Разумный максимум
        if self.value > max_value:
            raise DomainValidationError(f"Quantity {self.value} exceeds maximum allowed value")
        
        # Округляем до нужной точности
        rounded_value = self.value.quantize(Decimal('0.' + '0' * self.precision), rounding=ROUND_DOWN)
        object.__setattr__(self, 'value', rounded_value)
    
    @classmethod
    def create(cls, value: Union[str, int, float, Decimal], precision: int = 8) -> 'Quantity':
        """
        Фабричный метод для создания Quantity с валидацией.
        
        Args:
            value: Значение quantity
            precision: Точность (количество знаков после запятой)
            
        Returns:
            Quantity: Валидный объект количества
            
        Raises:
            DomainValidationError: При невалидных данных
        """
        try:
            decimal_value = Decimal(str(value))
            return cls(value=decimal_value, precision=precision)
        except (InvalidOperation, ValueError) as e:
            raise DomainValidationError(f"Cannot create quantity from value '{value}': {e}")
    
    @classmethod
    def zero(cls, precision: int = 8) -> 'Quantity':
        """Создание нулевого quantity (технически невалидно для торговли)."""
        # Создаем минимально возможное значение вместо нуля
        min_value = Decimal('0.' + '0' * (precision - 1) + '1')
        return cls(value=min_value, precision=precision)
    
    @classmethod
    def from_string(cls, value_str: str, precision: int = 8) -> 'Quantity':
        """Создание из строки с дополнительной валидацией."""
        if not isinstance(value_str, str):
            raise DomainValidationError("Value must be a string")
        
        # Удаляем лишние пробелы
        value_str = value_str.strip()
        
        if not value_str:
            raise DomainValidationError("Quantity string cannot be empty")
        
        return cls.create(value_str, precision)
    
    def add(self, other: 'Quantity') -> 'Quantity':
        """Сложение двух quantity с сохранением большей точности."""
        if not isinstance(other, Quantity):
            raise DomainValidationError("Can only add Quantity to Quantity")
        
        max_precision = max(self.precision, other.precision)
        result_value = self.value + other.value
        
        return Quantity(value=result_value, precision=max_precision)
    
    def subtract(self, other: 'Quantity') -> 'Quantity':
        """Вычитание quantity с проверкой на отрицательность."""
        if not isinstance(other, Quantity):
            raise DomainValidationError("Can only subtract Quantity from Quantity")
        
        if other.value > self.value:
            raise DomainValidationError("Subtraction would result in negative quantity")
        
        max_precision = max(self.precision, other.precision)
        result_value = self.value - other.value
        
        return Quantity(value=result_value, precision=max_precision)
    
    def multiply(self, multiplier: Union[int, float, Decimal]) -> 'Quantity':
        """Умножение на скаляр."""
        try:
            multiplier_decimal = Decimal(str(multiplier))
        except (InvalidOperation, ValueError) as e:
            raise DomainValidationError(f"Invalid multiplier: {e}")
        
        if multiplier_decimal <= 0:
            raise DomainValidationError("Multiplier must be positive")
        
        result_value = self.value * multiplier_decimal
        return Quantity(value=result_value, precision=self.precision)
    
    def divide(self, divisor: Union[int, float, Decimal]) -> 'Quantity':
        """Деление на скаляр."""
        try:
            divisor_decimal = Decimal(str(divisor))
        except (InvalidOperation, ValueError) as e:
            raise DomainValidationError(f"Invalid divisor: {e}")
        
        if divisor_decimal <= 0:
            raise DomainValidationError("Divisor must be positive")
        
        result_value = self.value / divisor_decimal
        return Quantity(value=result_value, precision=self.precision)
    
    def percentage_of(self, total: 'Quantity') -> Decimal:
        """Вычисление процента от общего количества."""
        if not isinstance(total, Quantity):
            raise DomainValidationError("Total must be a Quantity")
        
        if total.value == 0:
            raise DomainValidationError("Cannot calculate percentage of zero")
        
        return (self.value / total.value) * 100
    
    def is_greater_than(self, other: 'Quantity') -> bool:
        """Сравнение больше."""
        if not isinstance(other, Quantity):
            raise DomainValidationError("Can only compare with another Quantity")
        return self.value > other.value
    
    def is_less_than(self, other: 'Quantity') -> bool:
        """Сравнение меньше."""
        if not isinstance(other, Quantity):
            raise DomainValidationError("Can only compare with another Quantity")
        return self.value < other.value
    
    def is_equal_to(self, other: 'Quantity') -> bool:
        """Сравнение равенства с учетом точности."""
        if not isinstance(other, Quantity):
            return False
        
        # Сравниваем с учетом минимальной точности
        min_precision = min(self.precision, other.precision)
        
        self_rounded = self.value.quantize(Decimal('0.' + '0' * min_precision))
        other_rounded = other.value.quantize(Decimal('0.' + '0' * min_precision))
        
        return self_rounded == other_rounded
    
    def to_float(self) -> float:
        """Преобразование в float (с потенциальной потерей точности)."""
        return float(self.value)
    
    def to_string(self, format_precision: Optional[int] = None) -> str:
        """
        Форматирование в строку.
        
        Args:
            format_precision: Точность для отображения (если отличается от внутренней)
        """
        precision = format_precision if format_precision is not None else self.precision
        
        if precision == 0:
            return str(int(self.value))
        
        format_str = f"{{:.{precision}f}}"
        return format_str.format(self.value)
    
    def meets_minimum(self, minimum: 'Quantity') -> bool:
        """Проверка соответствия минимальному количеству."""
        if not isinstance(minimum, Quantity):
            raise DomainValidationError("Minimum must be a Quantity")
        
        return self.value >= minimum.value
    
    def apply_lot_size(self, lot_size: 'Quantity') -> 'Quantity':
        """Применение размера лота (округление вниз до кратного лоту)."""
        if not isinstance(lot_size, Quantity):
            raise DomainValidationError("Lot size must be a Quantity")
        
        if lot_size.value <= 0:
            raise DomainValidationError("Lot size must be positive")
        
        # Вычисляем количество полных лотов
        lots = int(self.value / lot_size.value)
        
        if lots == 0:
            raise DomainValidationError("Quantity is less than minimum lot size")
        
        # Возвращаем количество, кратное лоту
        result_value = lot_size.value * lots
        return Quantity(value=result_value, precision=self.precision)
    
    def __str__(self) -> str:
        """Строковое представление."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"Quantity(value={self.value}, precision={self.precision})"
    
    def __hash__(self) -> int:
        """Хэш для использования в множествах и словарях."""
        return hash((self.value, self.precision))
    
    def __eq__(self, other: Any) -> bool:
        """Операция равенства."""
        if not isinstance(other, Quantity):
            return False
        return self.is_equal_to(other)
    
    def __lt__(self, other: 'Quantity') -> bool:
        """Операция меньше."""
        return self.is_less_than(other)
    
    def __le__(self, other: 'Quantity') -> bool:
        """Операция меньше или равно."""
        return self.is_less_than(other) or self.is_equal_to(other)
    
    def __gt__(self, other: 'Quantity') -> bool:
        """Операция больше."""
        return self.is_greater_than(other)
    
    def __ge__(self, other: 'Quantity') -> bool:
        """Операция больше или равно."""
        return self.is_greater_than(other) or self.is_equal_to(other)
    
    def __add__(self, other: 'Quantity') -> 'Quantity':
        """Операция сложения."""
        return self.add(other)
    
    def __sub__(self, other: 'Quantity') -> 'Quantity':
        """Операция вычитания."""
        return self.subtract(other)
    
    def __mul__(self, multiplier: Union[int, float, Decimal]) -> 'Quantity':
        """Операция умножения."""
        return self.multiply(multiplier)
    
    def __truediv__(self, divisor: Union[int, float, Decimal]) -> 'Quantity':
        """Операция деления."""
        return self.divide(divisor)