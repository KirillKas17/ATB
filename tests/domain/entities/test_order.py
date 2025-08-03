#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit тесты для entity Order.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.exceptions import OrderError

class TestOrder:
    """Тесты для entity Order."""
    def test_creation_with_valid_data(self) -> None:
        """Тест создания ордера с валидными данными."""
        # Arrange & Act
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Assert
        assert order.trading_pair == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity.value == Decimal("0.001")
        assert order.price.value == Decimal("50000")
        assert order.status == OrderStatus.PENDING
        assert order.is_active is True
        assert order.filled_quantity == Decimal("0")
        assert order.remaining_quantity == Decimal("0.001")
    def test_creation_market_order(self) -> None:
        """Тест создания рыночного ордера."""
        # Arrange & Act
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )
        # Assert
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        assert order.status == OrderStatus.PENDING
    def test_creation_with_invalid_data(self) -> None:
        """Тест создания ордера с невалидными данными."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            Order(
            portfolio_id="test_portfolio",
            trading_pair="",  # Пустая торговая пара
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Volume(Decimal("0.001")),
                price=Price(Decimal("50000"), Currency.USD),
            )
    def test_fill_partial(self) -> None:
        """Тест частичного заполнения ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000"), Currency.USD))
        # Assert
        assert order.filled_quantity == Decimal("0.0005")
        assert order.average_price.value == Decimal("50000")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == Decimal("0.0005")
    def test_fill_complete(self) -> None:
        """Тест полного заполнения ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000"), Currency.USD))
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50100"), Currency.USD))
        # Assert
        assert order.filled_quantity == Decimal("0.001")
        assert order.average_price.value == Decimal("50050")  # Средняя цена
        assert order.status == OrderStatus.FILLED
        assert order.remaining_quantity == Decimal("0")
        assert order.is_active is False
    def test_fill_more_than_quantity(self) -> None:
        """Тест заполнения больше чем количество."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act & Assert
        with pytest.raises(OrderError, match="Cannot fill more than order quantity"):
            order.fill(Volume(Decimal("0.002")), Price(Decimal("50000"), Currency.USD))
    def test_cancel_order(self) -> None:
        """Тест отмены ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        order.cancel()
        # Assert
        assert order.status == OrderStatus.CANCELLED
        assert order.is_active is False
    def test_cancel_filled_order(self) -> None:
        """Тест отмены уже заполненного ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.fill(Volume(Decimal("0.001")), Price(Decimal("50000"), Currency.USD))
        # Act & Assert
        with pytest.raises(OrderError, match="Cannot cancel filled order"):
            order.cancel()
    def test_cancel_cancelled_order(self) -> None:
        """Тест отмены уже отмененного ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.cancel()
        # Act & Assert
        with pytest.raises(OrderError, match="Order is already cancelled"):
            order.cancel()
    def test_update_price(self) -> None:
        """Тест обновления цены ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        order.update_price(Price(Decimal("51000"), Currency.USD))
        # Assert
        assert order.price.value == Decimal("51000")
    def test_update_price_market_order(self) -> None:
        """Тест обновления цены рыночного ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )
        # Act & Assert
        with pytest.raises(OrderError, match="Cannot update price for market order"):
            order.update_price(Price(Decimal("51000"), Currency.USD))
    def test_update_price_filled_order(self) -> None:
        """Тест обновления цены заполненного ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.fill(Volume(Decimal("0.001")), Price(Decimal("50000"), Currency.USD))
        # Act & Assert
        with pytest.raises(OrderError, match="Cannot update price for filled order"):
            order.update_price(Price(Decimal("51000"), Currency.USD))
    def test_calculate_total_value(self) -> None:
        """Тест расчета общей стоимости ордера."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        total_value = order.total_value
        # Assert
        assert total_value.value == Decimal("50.00")  # 0.001 * 50000
    def test_calculate_filled_value(self) -> None:
        """Тест расчета стоимости заполненной части."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000"), Currency.USD))
        # Act
        filled_value = order.filled_value
        # Assert
        assert filled_value.value == Decimal("25.00")  # 0.0005 * 50000
    def test_calculate_remaining_value(self) -> None:
        """Тест расчета стоимости оставшейся части."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000"), Currency.USD))
        # Act
        remaining_value = order.remaining_quantity * order.price.amount
        # Assert
        assert remaining_value == Decimal("25.00")  # 0.0005 * 50000
    def test_get_fill_percentage(self) -> None:
        """Тест получения процента заполнения."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000"), Currency.USD))
        # Act
        fill_percentage = order.fill_percentage
        # Assert
        assert fill_percentage == 50.0
    def test_to_dict(self) -> None:
        """Тест сериализации в словарь."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        result = order.to_dict()
        # Assert
        assert result["trading_pair"] == "BTCUSDT"
        assert result["side"] == "buy"
        assert result["order_type"] == "limit"
        assert result["quantity"] == "0.00100000"
        assert result["price"] == "50000"
        assert result["status"] == "pending"
    def test_from_dict(self) -> None:
        """Тест десериализации из словаря."""
        # Arrange
        data = {
            "id": "12345678-1234-1234-1234-123456789012",
            "portfolio_id": "12345678-1234-1234-1234-123456789013",
            "strategy_id": "12345678-1234-1234-1234-123456789014",
            "signal_id": "",
            "exchange_order_id": "",
            "symbol": "BTCUSDT",
            "trading_pair": "BTCUSDT",
            "side": "buy",
            "order_type": "limit",
            "amount": "0.00000000",
            "quantity": "0.001",
            "price": "50000",
            "stop_price": "",
            "status": "pending",
            "filled_amount": "0.00000000",
            "filled_quantity": "0.00000000",
            "average_price": "",
            "commission": "",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "filled_at": "",
            "metadata": "{}",
        }
        # Act
        order = Order.from_dict(data)
        # Assert
        assert order.trading_pair == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("0.001")
        assert order.price.value == Decimal("50000")
        assert order.status == OrderStatus.PENDING
    def test_equality(self) -> None:
        """Тест равенства ордеров."""
        # Arrange
        order1 = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order2 = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Assert
        assert order1 == order2
        assert hash(order1) == hash(order2)
    def test_inequality(self) -> None:
        """Тест неравенства ордеров."""
        # Arrange
        order1 = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        order2 = Order(
            portfolio_id="test_portfolio",
            trading_pair="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Assert
        assert order1 != order2
        assert hash(order1) != hash(order2)
    def test_str_representation(self) -> None:
        """Тест строкового представления."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        result = str(order)
        # Assert
        assert "BTCUSDT" in result
        assert "buy" in result
        assert "limit" in result
        assert "0.001" in result
        assert "50000" in result
    def test_repr_representation(self) -> None:
        """Тест представления для отладки."""
        # Arrange
        order = Order(
            portfolio_id="test_portfolio",
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000"), Currency.USD),
        )
        # Act
        result = repr(order)
        # Assert
        assert "Order" in result
        assert "BTCUSDT" in result
        assert "buy" in result
        assert "limit" in result 
