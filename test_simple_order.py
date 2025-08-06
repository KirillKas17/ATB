#!/usr/bin/env python3
"""
Простые тесты Order entity без pytest.
"""

from decimal import Decimal
from domain.entities.order import (
    Order, OrderSide, OrderType, OrderStatus, OrderTimeInForce,
    create_order, validate_order_params
)


def test_order_creation():
    """Тест создания ордера."""
    order = create_order(
        user_id="test_user",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("45000.0")
    )
    
    assert order.user_id == "test_user"
    assert order.symbol == "BTCUSDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.quantity == Decimal("1.0")
    assert order.price == Decimal("45000.0")
    assert order.status == OrderStatus.PENDING


def test_order_fill():
    """Тест заполнения ордера."""
    order = create_order(
        user_id="test_user",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("45000.0")
    )
    
    # Частичное заполнение
    order.fill(Decimal("0.5"), Decimal("45100.0"))
    assert order.filled_quantity == Decimal("0.5")
    assert order.remaining_quantity == Decimal("0.5")
    assert order.status == OrderStatus.PARTIAL
    assert order.average_price == Decimal("45100.0")
    
    # Полное заполнение
    order.fill(Decimal("0.5"), Decimal("45200.0"))
    assert order.filled_quantity == Decimal("1.0")
    assert order.remaining_quantity == Decimal("0.0")
    assert order.status == OrderStatus.FILLED


def test_order_cancel():
    """Тест отмены ордера."""
    order = create_order(
        user_id="test_user",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.5"),
        price=Decimal("46000.0")
    )
    
    order.cancel()
    assert order.status == OrderStatus.CANCELLED


def test_order_validation():
    """Тест валидации ордеров."""
    # Позитивные тесты
    assert validate_order_params(Decimal("1.0"), Decimal("45000.0")) == True
    assert validate_order_params(Decimal("0.1")) == True
    
    # Негативные тесты
    assert validate_order_params(Decimal("-1.0")) == False
    assert validate_order_params(Decimal("1.0"), Decimal("-45000.0")) == False
    assert validate_order_params(Decimal("0")) == False


def test_order_validation_exceptions():
    """Тест исключений при валидации."""
    try:
        order = Order(
            id="test",
            user_id="test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("-1.0"),  # Некорректное количество
            price=Decimal("45000.0")
        )
        raise AssertionError("Should have raised validation error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_order_fill_exceptions():
    """Тест исключений при заполнении."""
    order = create_order(
        user_id="test_user",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("45000.0")
    )
    
    # Попытка заполнить больше чем доступно
    try:
        order.fill(Decimal("2.0"), Decimal("45000.0"))
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение
    
    # Попытка заполнить отрицательным количеством
    try:
        order.fill(Decimal("-0.5"), Decimal("45000.0"))
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_order_cancel_exceptions():
    """Тест исключений при отмене."""
    order = create_order(
        user_id="test_user",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("45000.0")
    )
    
    # Заполняем ордер полностью
    order.fill(Decimal("1.0"), Decimal("45000.0"))
    
    # Попытка отменить заполненный ордер
    try:
        order.cancel()
        raise AssertionError("Should have raised error")
    except ValueError:
        pass  # Ожидаемое исключение


def test_order_enums():
    """Тест значений енумов."""
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"
    
    assert OrderType.MARKET.value == "MARKET"
    assert OrderType.LIMIT.value == "LIMIT"
    
    assert OrderStatus.PENDING.value == "PENDING"
    assert OrderStatus.PARTIAL.value == "PARTIAL"
    assert OrderStatus.FILLED.value == "FILLED"
    assert OrderStatus.CANCELLED.value == "CANCELLED"
    
    assert OrderTimeInForce.GTC.value == "GTC"
    assert OrderTimeInForce.IOC.value == "IOC"