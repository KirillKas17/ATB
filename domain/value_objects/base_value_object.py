"""
Базовый класс для Value Objects.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseValueObject(ABC):
    """
    Базовый абстрактный класс для всех Value Objects.
    
    Определяет общий интерфейс и поведение для всех value objects
    в доменной модели.
    """
    
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Сравнение на равенство."""
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        """Хеш для использования в словарях и множествах."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Строковое представление."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Представление для отладки."""
        pass 