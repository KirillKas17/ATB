#!/usr/bin/env python3
"""
Архитектурная проверка и тестирование основных компонентов.
"""

import sys
import traceback
from decimal import Decimal
from datetime import datetime


def test_order_entity():
    """Тест Order entity."""
    print("Testing Order entity...")
    
    try:
        # Импорт
        from domain.entities.order import Order, OrderSide, OrderType, OrderStatus, create_order
        
        # Создание ордера
        order = create_order(
            user_id="test_user",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000.0")
        )
        
        # Проверки
        assert order.user_id == "test_user"
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("1.0")
        assert order.price == Decimal("45000.0")
        assert order.status == OrderStatus.PENDING
        assert order.remaining_quantity == Decimal("1.0")
        
        # Тест fill
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
        
        print("✅ Order entity tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Order entity test failed: {e}")
        traceback.print_exc()
        return False


def test_value_objects():
    """Тест Value Objects."""
    print("Testing Value Objects...")
    
    try:
        # Price
        from domain.value_objects.price import Price
        price = Price(amount=Decimal("45000.0"), currency="USD")
        assert price.amount == Decimal("45000.0")
        assert price.currency == "USD"
        assert str(price) == "45000.0 USD"
        
        # Volume
        from domain.value_objects.volume import Volume
        volume = Volume(amount=Decimal("1.5"))
        assert volume.amount == Decimal("1.5")
        assert str(volume) == "1.5"
        
        # Currency
        from domain.value_objects.currency import Currency
        currency = Currency.BTC
        assert currency.code == "BTC"
        
        # Timestamp
        from domain.value_objects.timestamp import Timestamp
        ts = Timestamp.now()
        assert ts.value is not None
        assert isinstance(ts.value, datetime)
        
        print("✅ Value Objects tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Value Objects test failed: {e}")
        traceback.print_exc()
        return False


def test_exceptions():
    """Тест исключений."""
    print("Testing Exceptions...")
    
    try:
        from domain.exceptions import (
            DomainError, ValidationError, OrderError, TradingError,
            MLModelError, CryptographyError, SecurityError
        )
        
        # Проверяем что исключения создаются
        assert issubclass(ValidationError, DomainError)
        assert issubclass(OrderError, DomainError)
        assert issubclass(TradingError, DomainError)
        assert issubclass(MLModelError, DomainError)
        assert issubclass(CryptographyError, DomainError)
        assert issubclass(SecurityError, DomainError)
        
        # Проверяем что исключения можно вызывать
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            assert str(e) == "Test validation error"
        
        print("✅ Exceptions tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Exceptions test failed: {e}")
        traceback.print_exc()
        return False


def test_order_validation():
    """Тест валидации ордеров."""
    print("Testing Order validation...")
    
    try:
        from domain.entities.order import Order, OrderSide, OrderType, validate_order_params
        from decimal import Decimal
        
        # Позитивные тесты
        assert validate_order_params(Decimal("1.0"), Decimal("45000.0")) == True
        assert validate_order_params(Decimal("0.1")) == True
        
        # Негативные тесты
        assert validate_order_params(Decimal("-1.0")) == False
        assert validate_order_params(Decimal("1.0"), Decimal("-45000.0")) == False
        assert validate_order_params(Decimal("0")) == False
        
        # Тест создания некорректного ордера
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
            print("❌ Should have raised validation error")
            return False
        except ValueError:
            pass  # Ожидаемое исключение
        
        print("✅ Order validation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Order validation test failed: {e}")
        traceback.print_exc()
        return False


def test_imports():
    """Тест основных импортов."""
    print("Testing basic imports...")
    
    try:
        # Проверяем что все основные модули импортируются
        import domain
        import domain.entities
        import domain.value_objects
        import domain.exceptions
        import application
        import infrastructure
        
        print("✅ Basic imports tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic imports test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Главная функция тестирования."""
    print("🚀 Starting Architecture Testing...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_exceptions,
        test_value_objects,
        test_order_entity,
        test_order_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("🎉 All architectural tests passed!")
        return 0
    else:
        print(f"💥 {failed} tests failed - architecture needs fixes!")
        return 1


if __name__ == "__main__":
    sys.exit(main())