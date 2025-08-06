#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Order Domain Entity.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime

from domain.entities.order import (
    Order, OrderType, OrderSide, OrderStatus, OrderTimeInForce,
    create_order, validate_order_params
)
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from domain.exceptions import OrderError, ValidationError


class TestOrder:
    """Тесты для Order Entity."""

    @pytest.fixture
    def usd_currency(self) -> Currency:
        """Фикстура USD валюты."""
        return Currency("USD")

    @pytest.fixture
    def btc_currency(self) -> Currency:
        """Фикстура BTC валюты."""
        return Currency("BTC")

    @pytest.fixture
    def sample_price(self, usd_currency: Currency) -> Price:
        """Фикстура цены."""
        return Price(value=Decimal("45000.00"), currency=usd_currency)

    @pytest.fixture
    def sample_volume(self, btc_currency: Currency) -> Volume:
        """Фикстура объема."""
        return Volume(value=Decimal("0.001"), currency=btc_currency)

    @pytest.fixture
    def sample_order_data(self) -> Dict[str, Any]:
        """Фикстура данных ордера."""
        return {
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "strategy_id": "test_strategy_001",
            "portfolio_id": "portfolio_001"
        }

    def test_order_creation_valid(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест создания валидного ордера."""
        order = Order(**sample_order_data)
        
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("0.001")
        assert order.price == Decimal("45000.00")
        assert order.status == OrderStatus.PENDING
        assert order.strategy_id == "test_strategy_001"
        assert order.portfolio_id == "portfolio_001"
        assert order.order_id is not None
        assert isinstance(order.order_id, str)

    def test_order_creation_market_order(self) -> None:
        """Тест создания рыночного ордера."""
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            strategy_id="market_strategy"
        )
        
        assert order.order_type == OrderType.MARKET
        assert order.price is None  # Рыночный ордер без цены
        assert order.status == OrderStatus.PENDING

    def test_order_creation_stop_order(self) -> None:
        """Тест создания стоп-ордера."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.001"),
            price=Decimal("44000.00"),
            stop_price=Decimal("44500.00"),
            strategy_id="stop_strategy"
        )
        
        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price == Decimal("44500.00")
        assert order.price == Decimal("44000.00")

    def test_order_validation_invalid_quantity(self) -> None:
        """Тест валидации невалидного количества."""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0"),  # Нулевое количество
                price=Decimal("45000.00")
            )

    def test_order_validation_negative_price(self) -> None:
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValidationError, match="Price must be positive"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("-100.00")  # Отрицательная цена
            )

    def test_order_validation_limit_order_without_price(self) -> None:
        """Тест валидации лимитного ордера без цены."""
        with pytest.raises(ValidationError, match="Limit order requires price"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001")
                # Отсутствует price для LIMIT ордера
            )

    def test_order_validation_stop_order_without_stop_price(self) -> None:
        """Тест валидации стоп-ордера без стоп-цены."""
        with pytest.raises(ValidationError, match="Stop order requires stop price"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
                # Отсутствует stop_price для STOP ордера
            )

    def test_order_status_transitions(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест переходов статусов ордера."""
        order = Order(**sample_order_data)
        
        # PENDING -> OPEN
        order.set_status(OrderStatus.OPEN)
        assert order.status == OrderStatus.OPEN
        
        # OPEN -> PARTIALLY_FILLED
        order.set_status(OrderStatus.PARTIALLY_FILLED)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # PARTIALLY_FILLED -> FILLED
        order.set_status(OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED

    def test_order_invalid_status_transition(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест невалидного перехода статуса."""
        order = Order(**sample_order_data)
        
        # Нельзя перейти из PENDING сразу в FILLED
        with pytest.raises(OrderError, match="Invalid status transition"):
            order.set_status(OrderStatus.FILLED)

    def test_order_fill_execution(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест исполнения ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Частичное исполнение
        fill_quantity = Decimal("0.0005")
        fill_price = Decimal("45050.00")
        
        order.add_fill(fill_quantity, fill_price, "execution_001")
        
        assert order.filled_quantity == fill_quantity
        assert order.remaining_quantity == order.quantity - fill_quantity
        assert order.average_fill_price == fill_price
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert len(order.fills) == 1

    def test_order_multiple_fills(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест множественных исполнений ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Первое исполнение
        order.add_fill(Decimal("0.0003"), Decimal("45000.00"), "exec_1")
        
        # Второе исполнение
        order.add_fill(Decimal("0.0004"), Decimal("45100.00"), "exec_2")
        
        # Третье исполнение (завершающее)
        order.add_fill(Decimal("0.0003"), Decimal("45200.00"), "exec_3")
        
        assert order.filled_quantity == Decimal("0.001")
        assert order.status == OrderStatus.FILLED
        assert len(order.fills) == 3
        
        # Проверяем средневзвешенную цену
        expected_avg = (
            Decimal("0.0003") * Decimal("45000.00") +
            Decimal("0.0004") * Decimal("45100.00") +
            Decimal("0.0003") * Decimal("45200.00")
        ) / Decimal("0.001")
        assert order.average_fill_price == expected_avg

    def test_order_overfill_protection(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест защиты от переисполнения ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Попытка исполнить больше, чем заявлено
        with pytest.raises(OrderError, match="Fill quantity exceeds remaining"):
            order.add_fill(Decimal("0.002"), Decimal("45000.00"), "overfill")

    def test_order_cancellation(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест отмены ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Отмена ордера
        order.cancel("Manual cancellation")
        
        assert order.status == OrderStatus.CANCELLED
        assert order.cancel_reason == "Manual cancellation"
        assert order.cancelled_at is not None

    def test_order_cancellation_after_partial_fill(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест отмены частично исполненного ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Частичное исполнение
        order.add_fill(Decimal("0.0005"), Decimal("45000.00"), "partial_exec")
        
        # Отмена оставшейся части
        order.cancel("Partial cancellation")
        
        assert order.status == OrderStatus.PARTIALLY_CANCELLED
        assert order.filled_quantity == Decimal("0.0005")
        assert order.remaining_quantity == Decimal("0.0005")

    def test_order_time_in_force_ioc(self) -> None:
        """Тест ордера с IOC (Immediate or Cancel)."""
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("3000.00"),
            time_in_force=OrderTimeInForce.IOC
        )
        
        assert order.time_in_force == OrderTimeInForce.IOC
        assert order.is_immediate_or_cancel() is True

    def test_order_time_in_force_fok(self) -> None:
        """Тест ордера с FOK (Fill or Kill)."""
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("3000.00"),
            time_in_force=OrderTimeInForce.FOK
        )
        
        assert order.time_in_force == OrderTimeInForce.FOK
        assert order.is_fill_or_kill() is True

    def test_order_expiry_handling(self) -> None:
        """Тест обработки истечения ордера."""
        expiry_time = datetime.now().timestamp() + 3600  # +1 час
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000.00"),
            expires_at=expiry_time
        )
        
        assert order.expires_at == expiry_time
        assert order.is_expired() is False
        
        # Симулируем истечение
        order.expires_at = datetime.now().timestamp() - 1
        assert order.is_expired() is True

    def test_order_commission_calculation(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест расчета комиссии ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Исполнение с комиссией
        fill_quantity = Decimal("0.001")
        fill_price = Decimal("45000.00")
        commission = Decimal("0.000001")  # 0.1% комиссия
        
        order.add_fill(fill_quantity, fill_price, "exec_with_commission", commission)
        
        assert order.total_commission == commission
        assert order.fills[0]["commission"] == commission

    def test_order_modification(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест модификации ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Изменение цены
        new_price = Decimal("44500.00")
        order.modify_price(new_price)
        
        assert order.price == new_price
        assert order.modified_at is not None

    def test_order_modification_invalid_status(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест невалидной модификации ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.FILLED)
        
        # Нельзя изменить исполненный ордер
        with pytest.raises(OrderError, match="Cannot modify order in status"):
            order.modify_price(Decimal("44000.00"))

    def test_order_risk_validation(self) -> None:
        """Тест валидации риска ордера."""
        # Слишком большой ордер
        with pytest.raises(ValidationError, match="Order size exceeds risk limits"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1000"),  # Очень большое количество
                max_position_size=Decimal("10")
            )

    def test_order_price_deviation_check(self) -> None:
        """Тест проверки отклонения цены."""
        # Цена слишком далека от рыночной
        with pytest.raises(ValidationError, match="Price deviates too much from market"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("10000.00"),  # Слишком низкая цена
                market_price=Decimal("45000.00"),
                max_price_deviation=Decimal("0.05")  # 5% максимальное отклонение
            )

    def test_order_serialization(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест сериализации ордера."""
        order = Order(**sample_order_data)
        
        # Сериализация в словарь
        order_dict = order.to_dict()
        
        assert order_dict["order_id"] == order.order_id
        assert order_dict["symbol"] == order.symbol
        assert order_dict["side"] == order.side.value
        assert order_dict["order_type"] == order.order_type.value
        assert order_dict["quantity"] == str(order.quantity)
        assert order_dict["price"] == str(order.price)

    def test_order_deserialization(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест десериализации ордера."""
        original_order = Order(**sample_order_data)
        order_dict = original_order.to_dict()
        
        # Десериализация из словаря
        restored_order = Order.from_dict(order_dict)
        
        assert restored_order.order_id == original_order.order_id
        assert restored_order.symbol == original_order.symbol
        assert restored_order.side == original_order.side
        assert restored_order.quantity == original_order.quantity
        assert restored_order.price == original_order.price

    def test_order_factory_method(self) -> None:
        """Тест фабричного метода создания ордера."""
        order = create_order(
            symbol="ETHUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity="0.1",
            price="3000.00",
            strategy_id="factory_strategy"
        )
        
        assert isinstance(order, Order)
        assert order.symbol == "ETHUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("0.1")
        assert order.price == Decimal("3000.00")

    def test_order_comparison(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест сравнения ордеров."""
        order1 = Order(**sample_order_data)
        order2 = Order(**sample_order_data)
        order3 = Order(**{**sample_order_data, "quantity": Decimal("0.002")})
        
        # Ордеры с одинаковыми параметрами равны
        assert order1 == order2
        assert hash(order1) == hash(order2)
        
        # Ордеры с разными параметрами не равны
        assert order1 != order3
        assert hash(order1) != hash(order3)

    def test_order_priority_calculation(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест расчета приоритета ордера."""
        order = Order(**sample_order_data)
        
        # Приоритет зависит от цены и времени создания
        priority = order.calculate_priority()
        
        assert isinstance(priority, int)
        assert priority > 0

    def test_order_matching_compatibility(self) -> None:
        """Тест совместимости ордеров для сопоставления."""
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000.00")
        )
        
        sell_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("44999.00")
        )
        
        # Ордера совместимы для исполнения
        assert buy_order.can_match_with(sell_order) is True
        
        # Проверяем детали сопоставления
        match_details = buy_order.get_match_details(sell_order)
        assert match_details["executable_quantity"] == Decimal("0.001")
        assert match_details["execution_price"] == Decimal("44999.00")  # Цена продавца

    def test_order_performance_metrics(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест метрик производительности ордера."""
        order = Order(**sample_order_data)
        order.set_status(OrderStatus.OPEN)
        
        # Симулируем задержку исполнения
        import time
        time.sleep(0.01)
        
        order.add_fill(Decimal("0.001"), Decimal("45000.00"), "perf_exec")
        
        metrics = order.get_performance_metrics()
        
        assert "execution_time" in metrics
        assert "slippage" in metrics
        assert "fill_rate" in metrics
        assert metrics["execution_time"] > 0

    def test_order_edge_cases(self) -> None:
        """Тест граничных случаев."""
        # Минимальное количество
        min_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.00000001"),  # Минимум
            price=Decimal("45000.00")
        )
        
        assert min_order.quantity == Decimal("0.00000001")
        
        # Максимальная цена
        max_price_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("999999999.99")  # Очень высокая цена
        )
        
        assert max_price_order.price == Decimal("999999999.99")

    def test_order_memory_efficiency(self, sample_order_data: Dict[str, Any]) -> None:
        """Тест эффективности использования памяти."""
        import sys
        
        # Создаем множество ордеров
        orders = [Order(**sample_order_data) for _ in range(1000)]
        
        # Проверяем размер одного ордера
        order_size = sys.getsizeof(orders[0])
        
        # Размер должен быть разумным
        assert order_size < 2048  # Менее 2KB на ордер
        
        # Все ордера должны быть валидными
        assert all(order.status == OrderStatus.PENDING for order in orders) 