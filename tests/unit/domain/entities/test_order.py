"""
Unit тесты для domain.entities.order

Покрывает:
- Создание и валидацию Order объектов
- Изменение статуса
- Расчеты и валидации
- Сериализация/десериализация
"""

import pytest
from datetime import datetime
from decimal import Decimal
from domain.entities.order import Order, OrderStatus, OrderType, OrderSide
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp


class TestOrder:
    """Тесты для Order entity"""

    def test_order_creation_valid(self):
        """Тест создания Order с валидными данными"""
        order = Order(
            order_id="test_order_1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )
        assert order.order_id == "test_order_1"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.PENDING

    def test_order_creation_market_order(self):
        """Тест создания рыночного ордера"""
        order = Order(
            order_id="test_order_2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Volume(amount=Decimal("0.5"), currency=Currency.BTC),
            price=None,
            timestamp=Timestamp(datetime.now()),
        )
        assert order.order_type == OrderType.MARKET
        assert order.price is None

    def test_order_creation_invalid_quantity(self):
        """Тест создания Order с невалидным количеством"""
        with pytest.raises(ValueError):
            Order(
                order_id="test_order_3",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Volume(amount=Decimal("0"), currency=Currency.BTC),
                price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
                timestamp=Timestamp(datetime.now()),
            )

    def test_order_creation_invalid_price_for_limit(self):
        """Тест создания лимитного ордера без цены"""
        with pytest.raises(ValueError):
            Order(
                order_id="test_order_4",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
                price=None,
                timestamp=Timestamp(datetime.now()),
            )

    def test_order_status_transitions(self):
        """Тест переходов статуса ордера"""
        order = Order(
            order_id="test_order_5",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        # PENDING -> OPEN
        order.open()
        assert order.status == OrderStatus.OPEN
        assert order.opened_at is not None

        # OPEN -> PARTIALLY_FILLED
        order.partially_fill(Volume(amount=Decimal("0.5"), currency=Currency.BTC))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity.amount == Decimal("0.5")

        # PARTIALLY_FILLED -> FILLED
        order.fill(Volume(amount=Decimal("0.5"), currency=Currency.BTC))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity.amount == Decimal("1.0")
        assert order.filled_at is not None

    def test_order_cancellation(self):
        """Тест отмены ордера"""
        order = Order(
            order_id="test_order_6",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        order.open()
        order.cancel("User cancelled")
        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert order.cancel_reason == "User cancelled"

    def test_order_rejection(self):
        """Тест отклонения ордера"""
        order = Order(
            order_id="test_order_7",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        order.reject("Insufficient funds")
        assert order.status == OrderStatus.REJECTED
        assert order.rejected_at is not None
        assert order.reject_reason == "Insufficient funds"

    def test_order_total_value(self):
        """Тест расчета общей стоимости ордера"""
        order = Order(
            order_id="test_order_8",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("2.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        total_value = order.total_value
        assert total_value.amount == Decimal("100000.00")
        assert total_value.currency == Currency.USD

    def test_order_remaining_quantity(self):
        """Тест расчета оставшегося количества"""
        order = Order(
            order_id="test_order_9",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        # До заполнения
        assert order.remaining_quantity.amount == Decimal("1.0")

        # После частичного заполнения
        order.partially_fill(Volume(amount=Decimal("0.3"), currency=Currency.BTC))
        assert order.remaining_quantity.amount == Decimal("0.7")

        # После полного заполнения
        order.fill(Volume(amount=Decimal("0.7"), currency=Currency.BTC))
        assert order.remaining_quantity.amount == Decimal("0")

    def test_order_fill_validation(self):
        """Тест валидации заполнения ордера"""
        order = Order(
            order_id="test_order_10",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        # Попытка заполнить больше, чем заказано
        with pytest.raises(ValueError):
            order.fill(Volume(amount=Decimal("1.5"), currency=Currency.BTC))

    def test_order_to_dict(self):
        """Тест сериализации Order в словарь"""
        order = Order(
            order_id="test_order_11",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        data = order.to_dict()
        assert data["order_id"] == "test_order_11"
        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == OrderSide.BUY.value
        assert data["order_type"] == OrderType.LIMIT.value
        assert data["status"] == OrderStatus.PENDING.value

    def test_order_from_dict(self):
        """Тест десериализации Order из словаря"""
        data = {
            "order_id": "test_order_12",
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY.value,
            "order_type": OrderType.LIMIT.value,
            "quantity": {"amount": "1.0", "currency": "BTC"},
            "price": {"amount": "50000.00", "currency": "USD"},
            "timestamp": datetime.now().isoformat(),
            "status": OrderStatus.PENDING.value,
        }

        order = Order.from_dict(data)
        assert order.order_id == "test_order_12"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT

    def test_order_equality(self):
        """Тест равенства Order объектов"""
        order1 = Order(
            order_id="test_order_13",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        order2 = Order(
            order_id="test_order_13",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        assert order1 == order2
        assert hash(order1) == hash(order2)

    def test_order_repr(self):
        """Тест строкового представления"""
        order = Order(
            order_id="test_order_14",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        repr_str = repr(order)
        assert "Order" in repr_str
        assert "test_order_14" in repr_str
        assert "BTC/USDT" in repr_str

    def test_order_str(self):
        """Тест строкового представления для пользователя"""
        order = Order(
            order_id="test_order_15",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(amount=Decimal("1.0"), currency=Currency.BTC),
            price=Price(amount=Decimal("50000.00"), currency=Currency.USD),
            timestamp=Timestamp(datetime.now()),
        )

        str_repr = str(order)
        assert "test_order_15" in str_repr
        assert "BTC/USDT" in str_repr
        assert "BUY" in str_repr
