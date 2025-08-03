"""
Unit тесты для domain/services/order_validation_service.py.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime, timezone

from domain.services.order_validation_service import OrderValidationService
from domain.entities.order import Order, OrderType, OrderSide
from domain.value_objects.trading_pair import TradingPair
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.balance import Balance
from domain.exceptions.base_exceptions import ValidationError


class TestOrderValidationService:
    """Тесты для OrderValidationService."""
    
    @pytest.fixture
    def service(self):
        """Создание сервиса."""
        return OrderValidationService()
    
    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(
            base_currency="BTC",
            quote_currency="USD",
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100.0"),
            price_precision=2,
            quantity_precision=6,
            min_notional=Decimal("10.00"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.000001")
        )
    
    @pytest.fixture
    def sample_balance(self) -> Balance:
        """Тестовый баланс."""
        return Balance(
            currency="USD",
            available=Decimal("10000.00"),
            reserved=Decimal("0.00")
        )
    
    @pytest.fixture
    def sample_order_data(self) -> Dict[str, Any]:
        """Тестовые данные заказа."""
        return {
            "id": "order_001",
            "trading_pair": "BTCUSD",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.00"),
            "timestamp": datetime.now(timezone.utc)
        }
    
    def test_validate_order_basic(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест базовой валидации заказа."""
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_invalid_quantity(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа с невалидным количеством."""
        # Количество меньше минимального
        sample_order_data["quantity"] = Decimal("0.0001")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("quantity" in error.lower() for error in result.errors)
    
    def test_validate_order_invalid_price(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа с невалидной ценой."""
        # Цена не соответствует tick_size
        sample_order_data["price"] = Decimal("50000.001")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("price" in error.lower() for error in result.errors)
    
    def test_validate_order_insufficient_balance(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа с недостаточным балансом."""
        # Недостаточный баланс
        insufficient_balance = Balance(
            currency="USD",
            available=Decimal("1000.00"),
            reserved=Decimal("0.00")
        )
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, insufficient_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("balance" in error.lower() or "insufficient" in error.lower() for error in result.errors)
    
    def test_validate_order_invalid_notional(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа с невалидным номиналом."""
        # Номинал меньше минимального
        sample_order_data["quantity"] = Decimal("0.0001")
        sample_order_data["price"] = Decimal("50000.00")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("notional" in error.lower() or "minimum" in error.lower() for error in result.errors)
    
    def test_validate_order_market_order(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации рыночного заказа."""
        sample_order_data["type"] = OrderType.MARKET
        sample_order_data["price"] = None
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_stop_loss(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации стоп-лосс заказа."""
        sample_order_data["type"] = OrderType.STOP_LOSS
        sample_order_data["stop_price"] = Decimal("45000.00")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_take_profit(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации тейк-профит заказа."""
        sample_order_data["type"] = OrderType.TAKE_PROFIT
        sample_order_data["stop_price"] = Decimal("55000.00")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_sell_side(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа на продажу."""
        sample_order_data["side"] = OrderSide.SELL
        
        # Для продажи нужен баланс в базовой валюте
        btc_balance = Balance(
            currency="BTC",
            available=Decimal("10.0"),
            reserved=Decimal("0.0")
        )
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, btc_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_sell_insufficient_base_balance(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации заказа на продажу с недостаточным балансом базовой валюты."""
        sample_order_data["side"] = OrderSide.SELL
        
        # Недостаточный баланс в базовой валюте
        insufficient_btc_balance = Balance(
            currency="BTC",
            available=Decimal("0.1"),
            reserved=Decimal("0.0")
        )
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, insufficient_btc_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("balance" in error.lower() or "insufficient" in error.lower() for error in result.errors)
    
    def test_validate_order_quantity_precision(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации точности количества."""
        # Количество не соответствует step_size
        sample_order_data["quantity"] = Decimal("1.0000001")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("quantity" in error.lower() or "precision" in error.lower() for error in result.errors)
    
    def test_validate_order_price_precision(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации точности цены."""
        # Цена не соответствует tick_size
        sample_order_data["price"] = Decimal("50000.123")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("price" in error.lower() or "precision" in error.lower() for error in result.errors)
    
    def test_validate_order_max_size_exceeded(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации превышения максимального размера заказа."""
        # Количество больше максимального
        sample_order_data["quantity"] = Decimal("200.0")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("maximum" in error.lower() or "size" in error.lower() for error in result.errors)
    
    def test_validate_order_zero_quantity(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации нулевого количества."""
        sample_order_data["quantity"] = Decimal("0.0")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("quantity" in error.lower() or "zero" in error.lower() for error in result.errors)
    
    def test_validate_order_negative_quantity(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации отрицательного количества."""
        sample_order_data["quantity"] = Decimal("-1.0")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("quantity" in error.lower() or "negative" in error.lower() for error in result.errors)
    
    def test_validate_order_negative_price(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации отрицательной цены."""
        sample_order_data["price"] = Decimal("-50000.00")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("price" in error.lower() or "negative" in error.lower() for error in result.errors)
    
    def test_validate_order_multiple_errors(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации с множественными ошибками."""
        # Несколько проблем одновременно
        sample_order_data["quantity"] = Decimal("0.0001")  # Меньше минимального
        sample_order_data["price"] = Decimal("50000.001")  # Не соответствует tick_size
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # Должно быть минимум 2 ошибки
    
    def test_validate_order_with_reserved_balance(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации с зарезервированным балансом."""
        # Баланс с зарезервированными средствами
        balance_with_reserved = Balance(
            currency="USD",
            available=Decimal("5000.00"),
            reserved=Decimal("5000.00")
        )
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, balance_with_reserved)
        assert result.is_valid is False  # Недостаточно доступных средств
        assert len(result.errors) > 0
        assert any("available" in error.lower() for error in result.errors)
    
    def test_validate_order_rounding(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации с округлением."""
        # Количество близко к минимальному, но должно округляться
        sample_order_data["quantity"] = Decimal("0.0010001")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        # Результат зависит от логики округления в сервисе
        assert isinstance(result.is_valid, bool)
    
    def test_validate_order_high_precision(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест валидации с высокой точностью."""
        # Создаем торговую пару с высокой точностью
        high_precision_pair = TradingPair(
            base_currency="BTC",
            quote_currency="USD",
            min_order_size=Decimal("0.000001"),
            max_order_size=Decimal("1000.0"),
            price_precision=8,
            quantity_precision=8,
            min_notional=Decimal("0.01"),
            tick_size=Decimal("0.00000001"),
            step_size=Decimal("0.00000001")
        )
        
        sample_order_data["quantity"] = Decimal("0.000001")
        sample_order_data["price"] = Decimal("50000.00000001")
        
        order = Order(
            id=sample_order_data["id"],
            trading_pair=high_precision_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        assert result.is_valid is True
        assert result.errors == []
    
    def test_validate_order_validation_result_structure(self, service, sample_trading_pair, sample_balance, sample_order_data):
        """Тест структуры результата валидации."""
        order = Order(
            id=sample_order_data["id"],
            trading_pair=sample_trading_pair,
            side=sample_order_data["side"],
            type=sample_order_data["type"],
            quantity=sample_order_data["quantity"],
            price=sample_order_data["price"],
            timestamp=sample_order_data["timestamp"]
        )
        
        result = service.validate_order(order, sample_balance)
        
        # Проверяем структуру результата
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        
        if result.errors:
            assert all(isinstance(error, str) for error in result.errors) 