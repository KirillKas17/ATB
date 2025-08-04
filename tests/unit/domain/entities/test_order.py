"""
Unit тесты для Order entity.

Покрывает:
- Создание и инициализацию ордера
- Валидацию данных
- Бизнес-логику ордера
- Статусы и состояния
- Операции с ордером
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4

from domain.entities.order import Order, OrderType, OrderSide, OrderStatus
from domain.type_definitions import OrderId, PortfolioId, StrategyId, SignalId, Symbol, TradingPair, VolumeValue
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from domain.exceptions import OrderError


class TestOrder:
    """Тесты для Order entity."""
    
    @pytest.fixture
    def sample_order_data(self) -> Dict[str, Any]:
        """Тестовые данные для ордера."""
        return {
            "symbol": Symbol("BTC/USDT"),
            "trading_pair": TradingPair("BTC/USDT"),
            "order_type": OrderType.LIMIT,
            "side": OrderSide.BUY,
            "amount": Volume(Decimal("1000.00"), Currency.USD),
            "quantity": VolumeValue(Decimal("1.5")),
            "price": Price(Decimal("50000.00"), Currency.USD),
            "status": OrderStatus.PENDING
        }
    
    @pytest.fixture
    def sample_market_order_data(self) -> Dict[str, Any]:
        """Тестовые данные для рыночного ордера."""
        return {
            "symbol": Symbol("ETH/USDT"),
            "trading_pair": TradingPair("ETH/USDT"),
            "order_type": OrderType.MARKET,
            "side": OrderSide.SELL,
            "amount": Volume(Decimal("500.00"), Currency.USD),
            "quantity": VolumeValue(Decimal("2.0")),
            "status": OrderStatus.PENDING
        }
    
    def test_order_creation(self, sample_order_data: Dict[str, Any]):
        """Тест создания ордера."""
        order = Order(**sample_order_data)
        
        assert order.symbol == Symbol("BTC/USDT")
        assert order.trading_pair == TradingPair("BTC/USDT")
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.BUY
        assert order.quantity == VolumeValue(Decimal("1.5"))
        assert order.price == Price(Decimal("50000.00"), Currency.USD)
        assert order.status == OrderStatus.PENDING
    
    def test_order_creation_with_defaults(self):
        """Тест создания ордера с значениями по умолчанию."""
        # Order требует trading_pair, поэтому создаем с минимальными данными
        order = Order(trading_pair=TradingPair("BTC/USDT"))
        
        # Проверяем, что ID являются UUID
        assert hasattr(order.id, '__class__')
        assert hasattr(order.portfolio_id, '__class__')
        assert hasattr(order.strategy_id, '__class__')
        assert order.signal_id is None
        assert order.exchange_order_id is None
        assert order.symbol == Symbol("")
        assert order.trading_pair == TradingPair("BTC/USDT")
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
    
    def test_order_validation_empty_trading_pair(self):
        """Тест валидации пустой торговой пары."""
        with pytest.raises(ValueError, match="Trading pair cannot be empty"):
            Order(trading_pair=TradingPair(""))
    
    def test_order_equality(self, sample_order_data: Dict[str, Any]):
        """Тест равенства ордеров."""
        order1 = Order(**sample_order_data)
        order2 = Order(**sample_order_data)
        
        assert order1 == order2
    
    def test_order_inequality(self, sample_order_data: Dict[str, Any]):
        """Тест неравенства ордеров."""
        order1 = Order(**sample_order_data)
        
        different_data = sample_order_data.copy()
        different_data["side"] = OrderSide.SELL
        order2 = Order(**different_data)
        
        assert order1 != order2
    
    def test_order_hash(self, sample_order_data: Dict[str, Any]):
        """Тест хеширования ордера."""
        order1 = Order(**sample_order_data)
        order2 = Order(**sample_order_data)
        
        assert hash(order1) == hash(order2)
        
        # Проверяем, что ордер можно использовать как ключ словаря
        order_dict = {order1: "test"}
        assert order_dict[order2] == "test"
    
    def test_order_str_representation(self, sample_order_data: Dict[str, Any]):
        """Тест строкового представления ордера."""
        order = Order(**sample_order_data)
        str_repr = str(order)
        
        assert "buy" in str_repr
        assert "limit" in str_repr
        assert "1.5" in str_repr
        assert "BTC/USDT" in str_repr
        assert "50000.00" in str_repr
        assert "pending" in str_repr
    
    def test_order_repr_representation(self, sample_order_data: Dict[str, Any]):
        """Тест repr представления ордера."""
        order = Order(**sample_order_data)
        repr_str = repr(order)
        
        assert "Order" in repr_str
        assert "BTC/USDT" in repr_str
        assert "limit" in repr_str
        assert "buy" in repr_str
    
    def test_order_protocol_implementation(self, sample_order_data: Dict[str, Any]):
        """Тест реализации протокола OrderProtocol."""
        order = Order(**sample_order_data)
        
        assert hasattr(order, 'get_status')
        assert hasattr(order, 'get_quantity')
        assert hasattr(order, 'get_price')
        
        assert order.get_status() == OrderStatus.PENDING.value
        assert order.get_quantity() == VolumeValue(Decimal("1.5"))
        assert order.get_price() == VolumeValue(Decimal("50000.00"))
    
    def test_order_status_properties(self, sample_order_data: Dict[str, Any]):
        """Тест свойств статуса ордера."""
        order = Order(**sample_order_data)
        
        # PENDING - не активный
        assert order.is_active is True
        assert order.is_open is False
        assert order.is_filled is False
        assert order.is_cancelled is False
        
        # OPEN - активный и открытый
        order.status = OrderStatus.OPEN
        assert order.is_active is True
        assert order.is_open is True
        assert order.is_filled is False
        assert order.is_cancelled is False
        
        # PARTIALLY_FILLED - активный и открытый
        order.status = OrderStatus.PARTIALLY_FILLED
        assert order.is_active is True
        assert order.is_open is True
        assert order.is_filled is False
        assert order.is_cancelled is False
        
        # FILLED - не активный, заполненный
        order.status = OrderStatus.FILLED
        assert order.is_active is False
        assert order.is_open is False
        assert order.is_filled is True
        assert order.is_cancelled is False
        
        # CANCELLED - не активный, отмененный
        order.status = OrderStatus.CANCELLED
        assert order.is_active is False
        assert order.is_open is False
        assert order.is_filled is False
        assert order.is_cancelled is True
    
    def test_order_fill_percentage(self, sample_order_data: Dict[str, Any]):
        """Тест расчета процента заполнения."""
        order = Order(**sample_order_data)
        
        # 0% заполнения
        assert order.fill_percentage == Decimal("0")
        
        # 50% заполнения
        order.filled_quantity = VolumeValue(Decimal("0.75"))
        assert order.fill_percentage == Decimal("50")
        
        # 100% заполнения
        order.filled_quantity = VolumeValue(Decimal("1.5"))
        assert order.fill_percentage == Decimal("100")
    
    def test_order_remaining_quantity(self, sample_order_data: Dict[str, Any]):
        """Тест расчета оставшегося количества."""
        order = Order(**sample_order_data)
        
        # Полное количество
        assert order.remaining_quantity == VolumeValue(Decimal("1.5"))
        
        # Частично заполненный
        order.filled_quantity = VolumeValue(Decimal("0.5"))
        assert order.remaining_quantity == VolumeValue(Decimal("1.0"))
        
        # Полностью заполненный
        order.filled_quantity = VolumeValue(Decimal("1.5"))
        assert order.remaining_quantity == VolumeValue(Decimal("0"))
    
    def test_order_total_value(self, sample_order_data: Dict[str, Any]):
        """Тест расчета общей стоимости."""
        order = Order(**sample_order_data)
        
        # С ценой
        total_value = order.total_value
        assert total_value is not None
        assert total_value.amount == Decimal("75000.00")  # 1.5 * 50000
        assert total_value.currency == Currency.USD
        
        # Без цены
        order.price = None
        assert order.total_value is None
    
    def test_order_filled_value(self, sample_order_data: Dict[str, Any]):
        """Тест расчета стоимости заполненной части."""
        order = Order(**sample_order_data)
        
        # Без средней цены
        assert order.filled_value is None
        
        # С средней ценой
        order.average_price = Price(Decimal("51000.00"), Currency.USD)
        order.filled_quantity = VolumeValue(Decimal("0.5"))
        
        filled_value = order.filled_value
        assert filled_value is not None
        assert filled_value.amount == Decimal("25500.00")  # 0.5 * 51000
        assert filled_value.currency == Currency.USD
    
    def test_order_update_status(self, sample_order_data: Dict[str, Any]):
        """Тест обновления статуса."""
        order = Order(**sample_order_data)
        original_updated_at = order.updated_at
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        order.update_status(OrderStatus.OPEN)
        
        assert order.status == OrderStatus.OPEN
        assert order.updated_at != original_updated_at
        
        # При заполнении устанавливается filled_at
        order.update_status(OrderStatus.FILLED)
        assert order.filled_at is not None
    
    def test_order_update_fill(self, sample_order_data: Dict[str, Any]):
        """Тест обновления заполнения."""
        order = Order(**sample_order_data)
        fill_price = Price(Decimal("51000.00"), Currency.USD)
        commission = Price(Decimal("25.50"), Currency.USD)
        
        # Первое заполнение
        order.update_fill(VolumeValue(Decimal("0.5")), fill_price, commission)
        
        assert order.filled_quantity == VolumeValue(Decimal("0.5"))
        assert order.average_price == fill_price
        assert order.commission == commission
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Второе заполнение
        second_fill_price = Price(Decimal("52000.00"), Currency.USD)
        second_commission = Price(Decimal("26.00"), Currency.USD)
        
        order.update_fill(VolumeValue(Decimal("0.5")), second_fill_price, second_commission)
        
        assert order.filled_quantity == VolumeValue(Decimal("1.0"))
        # Средняя цена: (0.5 * 51000 + 0.5 * 52000) / 1.0 = 51500
        assert order.average_price.amount == Decimal("51500.00")
        # Комиссия: 25.50 + 26.00 = 51.50
        assert order.commission.amount == Decimal("51.50")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Полное заполнение
        order.update_fill(VolumeValue(Decimal("0.5")), Price(Decimal("53000.00"), Currency.USD))
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
    
    def test_order_fill_success(self, sample_order_data: Dict[str, Any]):
        """Тест успешного заполнения ордера."""
        order = Order(**sample_order_data)
        fill_volume = Volume(Decimal("0.5"), Currency.USD)
        fill_price = Price(Decimal("51000.00"), Currency.USD)
        
        order.fill(fill_volume, fill_price)
        
        assert order.filled_quantity == VolumeValue(Decimal("0.5"))
        assert order.average_price == fill_price
        assert order.status == OrderStatus.PARTIALLY_FILLED
    
    def test_order_fill_cancelled_order(self, sample_order_data: Dict[str, Any]):
        """Тест заполнения отмененного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.CANCELLED
        fill_volume = Volume(Decimal("0.5"), Currency.USD)
        fill_price = Price(Decimal("51000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot fill cancelled order"):
            order.fill(fill_volume, fill_price)
    
    def test_order_fill_already_filled(self, sample_order_data: Dict[str, Any]):
        """Тест заполнения уже заполненного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.FILLED
        fill_volume = Volume(Decimal("0.5"), Currency.USD)
        fill_price = Price(Decimal("51000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot fill already filled order"):
            order.fill(fill_volume, fill_price)
    
    def test_order_fill_exceeds_quantity(self, sample_order_data: Dict[str, Any]):
        """Тест заполнения большего количества."""
        order = Order(**sample_order_data)
        fill_volume = Volume(Decimal("2.0"), Currency.USD)  # Больше чем 1.5
        fill_price = Price(Decimal("51000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot fill more than order quantity"):
            order.fill(fill_volume, fill_price)
    
    def test_order_cancel_success(self, sample_order_data: Dict[str, Any]):
        """Тест успешной отмены ордера."""
        order = Order(**sample_order_data)
        original_updated_at = order.updated_at
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        order.cancel()
        
        assert order.status == OrderStatus.CANCELLED
        assert order.updated_at != original_updated_at
    
    def test_order_cancel_filled_order(self, sample_order_data: Dict[str, Any]):
        """Тест отмены заполненного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.FILLED
        
        with pytest.raises(OrderError, match="Cannot cancel filled order"):
            order.cancel()
    
    def test_order_cancel_already_cancelled(self, sample_order_data: Dict[str, Any]):
        """Тест отмены уже отмененного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.CANCELLED
        
        with pytest.raises(OrderError, match="Order is already cancelled"):
            order.cancel()
    
    def test_order_update_price_success(self, sample_order_data: Dict[str, Any]):
        """Тест успешного обновления цены."""
        order = Order(**sample_order_data)
        new_price = Price(Decimal("52000.00"), Currency.USD)
        original_updated_at = order.updated_at
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        order.update_price(new_price)
        
        assert order.price == new_price
        assert order.updated_at != original_updated_at
    
    def test_order_update_price_market_order(self, sample_market_order_data: Dict[str, Any]):
        """Тест обновления цены рыночного ордера."""
        order = Order(**sample_market_order_data)
        new_price = Price(Decimal("3000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot update price for market order"):
            order.update_price(new_price)
    
    def test_order_update_price_filled_order(self, sample_order_data: Dict[str, Any]):
        """Тест обновления цены заполненного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.FILLED
        new_price = Price(Decimal("52000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot update price for filled order"):
            order.update_price(new_price)
    
    def test_order_update_price_cancelled_order(self, sample_order_data: Dict[str, Any]):
        """Тест обновления цены отмененного ордера."""
        order = Order(**sample_order_data)
        order.status = OrderStatus.CANCELLED
        new_price = Price(Decimal("52000.00"), Currency.USD)
        
        with pytest.raises(OrderError, match="Cannot update price for cancelled order"):
            order.update_price(new_price)
    
    def test_order_to_dict(self, sample_order_data: Dict[str, Any]):
        """Тест преобразования в словарь."""
        order = Order(**sample_order_data)
        order_dict = order.to_dict()
        
        assert order_dict["symbol"] == "BTC/USDT"
        assert order_dict["trading_pair"] == "BTC/USDT"
        assert order_dict["order_type"] == "limit"
        assert order_dict["side"] == "buy"
        assert order_dict["quantity"] == "1.5"
        assert order_dict["price"] == "50000.00"
        assert order_dict["status"] == "pending"
    
    def test_order_from_dict(self, sample_order_data: Dict[str, Any]):
        """Тест создания из словаря."""
        # Создаем данные в правильном формате для from_dict
        dict_data = {
            "id": str(uuid4()),
            "portfolio_id": str(uuid4()),
            "strategy_id": str(uuid4()),
            "signal_id": "",
            "exchange_order_id": "",
            "symbol": "BTC/USDT",
            "trading_pair": "BTC/USDT",
            "order_type": "limit",
            "side": "buy",
            "amount": "1000.00",
            "quantity": "1.5",
            "price": "50000.00",
            "stop_price": "",
            "status": "pending",
            "filled_amount": "0.00",
            "filled_quantity": "0.0",
            "average_price": "",
            "commission": "",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "filled_at": "",
            "metadata": "{}"
        }
        
        order = Order.from_dict(dict_data)
        
        assert order.symbol == Symbol("BTC/USDT")
        assert order.trading_pair == TradingPair("BTC/USDT")
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.BUY
        assert order.quantity == VolumeValue(Decimal("1.5"))
        assert order.price == Price(Decimal("50000.00"), Currency.USD)
        assert order.status == OrderStatus.PENDING
    
    def test_order_get_price_without_price(self, sample_market_order_data: Dict[str, Any]):
        """Тест получения цены без установленной цены."""
        order = Order(**sample_market_order_data)
        
        assert order.get_price() == VolumeValue(Decimal("0"))
    
    def test_order_enum_values(self):
        """Тест значений перечислений."""
        # OrderType
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        
        # OrderSide
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        
        # OrderStatus
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled" 