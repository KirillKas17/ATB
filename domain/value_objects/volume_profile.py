# -*- coding: utf-8 -*-
"""Volume Profile Value Object for Market Analysis."""
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from shared.numpy_utils import np
import pandas as pd

from domain.value_objects.base_value_object import BaseValueObject
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp


class VolumeLevel(BaseValueObject):
    """Уровень объема с ценой."""
    
    def __init__(self, price: Price, volume: float, timestamp: Timestamp) -> None:
        """
        Инициализация уровня объема.
        
        Args:
            price: Цена уровня
            volume: Объем на уровне
            timestamp: Временная метка
        """
        super().__init__()
        self._price = price
        self._volume = volume
        self._timestamp = timestamp
    
    @property
    def price(self) -> Price:
        """Цена уровня."""
        return self._price
    
    @property
    def volume(self) -> float:
        """Объем на уровне."""
        return self._volume
    
    @property
    def timestamp(self) -> Timestamp:
        """Временная метка."""
        return self._timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "price": float(self._price.amount),
            "volume": self._volume,
            "timestamp": self._timestamp.value.isoformat()
        }


class VolumeProfile(BaseValueObject):
    """Профиль объема для анализа рыночной структуры."""

    def __init__(
        self,
        price_levels: List[Price],
        volume_at_levels: List[float],
        poc_price: Price,
        value_area_high: Price,
        value_area_low: Price,
        timestamp: Timestamp,
        symbol: str = "",
    ) -> None:
        """
        Инициализация профиля объема.
        
        Args:
            price_levels: Уровни цен
            volume_at_levels: Объемы на уровнях
            poc_price: Цена точки контроля (Point of Control)
            value_area_high: Верхняя граница области значений
            value_area_low: Нижняя граница области значений
            timestamp: Временная метка
            symbol: Символ торговой пары
        """
        super().__init__()
        self._price_levels = price_levels
        self._volume_at_levels = volume_at_levels
        self._poc_price = poc_price
        self._value_area_high = value_area_high
        self._value_area_low = value_area_low
        self._timestamp = timestamp
        self._symbol = symbol

    @property
    def price_levels(self) -> List[Price]:
        """Получить уровни цен."""
        return self._price_levels

    @property
    def volume_at_levels(self) -> List[float]:
        """Получить объемы на уровнях."""
        return self._volume_at_levels

    @property
    def poc_price(self) -> Price:
        """Получить цену точки контроля."""
        return self._poc_price

    @property
    def value_area_high(self) -> Price:
        """Получить верхнюю границу области значений."""
        return self._value_area_high

    @property
    def value_area_low(self) -> Price:
        """Получить нижнюю границу области значений."""
        return self._value_area_low

    @property
    def timestamp(self) -> Timestamp:
        """Получить временную метку."""
        return self._timestamp

    @property
    def symbol(self) -> str:
        """Получить символ торговой пары."""
        return self._symbol

    def get_volume_at_price(self, price: Price) -> float:
        """
        Получить объем на конкретной цене.
        
        Args:
            price: Цена для поиска
            
        Returns:
            float: Объем на указанной цене
        """
        for i, level in enumerate(self._price_levels):
            if abs(level.value - price.value) < 0.0001:
                return self._volume_at_levels[i]
        return 0.0

    def get_support_resistance_levels(self, threshold: float = 0.1) -> Dict[str, List[Price]]:
        """
        Получить уровни поддержки и сопротивления.
        
        Args:
            threshold: Порог для определения значимых уровней
            
        Returns:
            Dict[str, List[Price]]: Словарь с уровнями поддержки и сопротивления
        """
        max_volume = max(self._volume_at_levels) if self._volume_at_levels else 0.0
        support_levels: List[Price] = []
        resistance_levels: List[Price] = []
        
        for i, volume in enumerate(self._volume_at_levels):
            if volume >= max_volume * threshold:
                price = self._price_levels[i]
                if price.value < self._poc_price.value:
                    support_levels.append(price)
                else:
                    resistance_levels.append(price)
        
        return {
            "support": sorted(support_levels, key=lambda x: x.value),
            "resistance": sorted(resistance_levels, key=lambda x: x.value)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "price_levels": [price.to_dict() for price in self._price_levels],
            "volume_at_levels": self._volume_at_levels,
            "poc_price": self._poc_price.to_dict(),
            "value_area_high": self._value_area_high.to_dict(),
            "value_area_low": self._value_area_low.to_dict(),
            "timestamp": self._timestamp.to_dict(),
            "symbol": self._symbol
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolumeProfile":
        """Создать из словаря."""
        return cls(
            price_levels=[Price.from_dict(p) for p in data["price_levels"]],
            volume_at_levels=data["volume_at_levels"],
            poc_price=Price.from_dict(data["poc_price"]),
            value_area_high=Price.from_dict(data["value_area_high"]),
            value_area_low=Price.from_dict(data["value_area_low"]),
            timestamp=Timestamp.from_dict(data["timestamp"]),
            symbol=data.get("symbol", "")
        )

    def __eq__(self, other: Any) -> bool:
        """Проверка равенства."""
        if not isinstance(other, VolumeProfile):
            return False
        return (
            self._price_levels == other._price_levels
            and self._volume_at_levels == other._volume_at_levels
            and self._poc_price == other._poc_price
            and self._value_area_high == other._value_area_high
            and self._value_area_low == other._value_area_low
            and self._timestamp == other._timestamp
            and self._symbol == other._symbol
        )

    def __hash__(self) -> int:
        """Хеш объекта."""
        return hash((
            tuple(self._price_levels),
            tuple(self._volume_at_levels),
            self._poc_price,
            self._value_area_high,
            self._value_area_low,
            self._timestamp,
            self._symbol
        ))

    def __repr__(self) -> str:
        """Строковое представление."""
        return (
            f"VolumeProfile(symbol={self._symbol}, "
            f"poc_price={self._poc_price}, "
            f"levels_count={len(self._price_levels)})"
        ) 