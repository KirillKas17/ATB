"""
Доменная сущность ордера с промышленной типизацией и бизнес-валидацией.
"""

import ast
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4
from datetime import datetime


class OrderType(Enum):
    """Типы ордеров."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderSide(Enum):
    """Сторона ордера."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Статус ордера."""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderTimeInForce(Enum):
    """Время действия ордера."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day order


@dataclass
class Order:
    """Доменная сущность ордера."""
    
    id: str
    user_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    timestamp: Optional[datetime] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    
    def __post_init__(self):
        """Пост-инициализация для валидации."""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        # Валидация
        self._validate()
    
    def _validate(self) -> None:
        """Валидация ордера."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("Stop price must be positive")
    
    def fill(self, quantity: Decimal, price: Decimal) -> None:
        """Частичное или полное исполнение ордера."""
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        
        if quantity > self.remaining_quantity:
            raise ValueError("Cannot fill more than remaining quantity")
        
        # Обновляем заполненное количество
        self.filled_quantity += quantity
        self.remaining_quantity -= quantity
        
        # Обновляем среднюю цену
        if self.average_price is None:
            self.average_price = price
        else:
            total_value = (self.filled_quantity - quantity) * self.average_price + quantity * price
            self.average_price = total_value / self.filled_quantity
        
        # Обновляем статус
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> None:
        """Отмена ордера."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order with status {self.status}")
        
        self.status = OrderStatus.CANCELLED


def create_order(
    user_id: str,
    symbol: str,
    side: OrderSide,
    order_type: OrderType,
    quantity: Decimal,
    price: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None,
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
) -> Order:
    """Фабричная функция для создания ордера."""
    return Order(
        id=str(uuid4()),
        user_id=user_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price,
        time_in_force=time_in_force
    )


def validate_order_params(
    quantity: Decimal,
    price: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None
) -> bool:
    """Валидация параметров ордера."""
    try:
        if quantity <= 0:
            return False
        
        if price is not None and price <= 0:
            return False
        
        if stop_price is not None and stop_price <= 0:
            return False
        
        return True
    except Exception:
        return False
