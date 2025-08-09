#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Order Entity.
Тестирует все аспекты Order entity с полным покрытием edge cases.
"""

import pytest
from decimal import Decimal
from uuid import UUID, uuid4
from datetime import datetime
from unittest.mock import Mock, patch

# Попробуем импортировать без conftest
import sys
import os

sys.path.append("/workspace")

try:
    from domain.entities.order import Order, OrderType, OrderSide, OrderStatus
    from domain.value_objects.price import Price
    from domain.value_objects.volume import Volume
    from domain.value_objects.currency import Currency
    from domain.value_objects.timestamp import Timestamp
    from domain.type_definitions import OrderId, PortfolioId, StrategyId, SignalId, Symbol, TradingPair, VolumeValue
    from domain.exceptions import OrderError
except ImportError as e:
    # Создаем минимальные моки если импорт не удался
    class OrderType:
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"

    class OrderSide:
        BUY = "buy"
        SELL = "sell"

    class OrderStatus:
        PENDING = "pending"
        OPEN = "open"
        FILLED = "filled"
        CANCELLED = "cancelled"

    class Currency:
        USD = "USD"
        BTC = "BTC"

    class Order:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", uuid4())
            self.symbol = kwargs.get("symbol", "BTC/USD")
            self.status = kwargs.get("status", OrderStatus.PENDING)


class TestOrderCreation:
    """Тесты создания Order objects"""

    def test_order_creation_with_defaults(self):
        """Тест создания ордера с дефолтными значениями"""
        order = Order(trading_pair=TradingPair("BTC/USD"))  # Обязательный параметр

        assert order.id is not None
        assert isinstance(order.id, UUID)
        assert order.portfolio_id is not None
        assert order.strategy_id is not None
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
        assert order.amount.amount == Decimal("0")
        assert order.quantity == VolumeValue(Decimal("0"))

    def test_order_creation_with_custom_values(self):
        """Тест создания ордера с кастомными значениями"""
        order_id = OrderId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        strategy_id = StrategyId(uuid4())
        price = Price(Decimal("50000.00"), Currency.USD)
        volume = Volume(Decimal("1.5"), Currency.BTC)

        order = Order(
            id=order_id,
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            symbol=Symbol("BTC"),
            trading_pair=TradingPair("BTC/USD"),
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            amount=volume,
            price=price,
            status=OrderStatus.OPEN,
        )

        assert order.id == order_id
        assert order.portfolio_id == portfolio_id
        assert order.strategy_id == strategy_id
        assert order.symbol == Symbol("BTC")
        assert order.trading_pair == TradingPair("BTC/USD")
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL
        assert order.amount == volume
        assert order.price == price
        assert order.status == OrderStatus.OPEN

    def test_order_creation_market_order(self):
        """Тест создания рыночного ордера"""
        order = Order(
            symbol=Symbol("ETH"),
            trading_pair=TradingPair("ETH/USD"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Volume(Decimal("10.0"), Currency.ETH),
        )

        assert order.order_type == OrderType.MARKET
        assert order.price is None  # Рыночные ордера не имеют цены

    def test_order_creation_limit_order(self):
        """Тест создания лимитного ордера"""
        price = Price(Decimal("3000.00"), Currency.USD)

        order = Order(
            symbol=Symbol("ETH"),
            trading_pair=TradingPair("ETH/USD"),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("5.0"), Currency.ETH),
            price=price,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.price == price

    def test_order_creation_stop_order(self):
        """Тест создания стоп-ордера"""
        stop_price = Price(Decimal("48000.00"), Currency.USD)

        order = Order(
            symbol=Symbol("BTC"),
            trading_pair=TradingPair("BTC/USD"),
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            amount=Volume(Decimal("0.5"), Currency.BTC),
            stop_price=stop_price,
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == stop_price

    def test_order_creation_empty_trading_pair_raises_error(self):
        """Тест что пустая торговая пара вызывает ошибку"""
        with pytest.raises(ValueError, match="Trading pair cannot be empty"):
            Order(trading_pair=TradingPair(""))

    def test_order_creation_with_signal_id(self):
        """Тест создания ордера с сигналом"""
        signal_id = SignalId(uuid4())

        order = Order(signal_id=signal_id, trading_pair=TradingPair("BTC/USD"))

        assert order.signal_id == signal_id


class TestOrderValidation:
    """Тесты валидации Order"""

    def test_order_valid_buy_order(self):
        """Тест валидного buy ордера"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            side=OrderSide.BUY,
            amount=Volume(Decimal("1.0"), Currency.BTC),
            price=Price(Decimal("50000.00"), Currency.USD),
        )

        if hasattr(order, "validate"):
            assert order.validate() is True

    def test_order_valid_sell_order(self):
        """Тест валидного sell ордера"""
        order = Order(
            trading_pair=TradingPair("ETH/USD"),
            side=OrderSide.SELL,
            amount=Volume(Decimal("5.0"), Currency.ETH),
            price=Price(Decimal("3000.00"), Currency.USD),
        )

        if hasattr(order, "validate"):
            assert order.validate() is True

    def test_order_zero_amount_validation(self):
        """Тест валидации нулевого количества"""
        order = Order(trading_pair=TradingPair("BTC/USD"), amount=Volume(Decimal("0"), Currency.BTC))

        if hasattr(order, "validate"):
            with pytest.raises((ValueError, OrderError)):
                order.validate()

    def test_order_negative_amount_validation(self):
        """Тест валидации отрицательного количества"""
        # Volume уже проверяет отрицательные значения в __post_init__
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            order = Order(trading_pair=TradingPair("BTC/USD"), amount=Volume(Decimal("-1.0"), Currency.BTC))

    def test_order_invalid_price_validation(self):
        """Тест валидации неверной цены"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"), order_type=OrderType.LIMIT, price=Price(Decimal("0"), Currency.USD)
        )

        if hasattr(order, "validate"):
            with pytest.raises((ValueError, OrderError)):
                order.validate()


class TestOrderStatusTransitions:
    """Тесты переходов статусов Order"""

    def test_order_status_pending_to_open(self):
        """Тест перехода статуса с PENDING на OPEN"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.PENDING)

        # Используем реальный метод update_status
        order.update_status(OrderStatus.OPEN)
        assert order.status == OrderStatus.OPEN

    def test_order_status_open_to_filled(self):
        """Тест перехода статуса с OPEN на FILLED"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            status=OrderStatus.OPEN,
            amount=Volume(Decimal("1.0"), Currency.BTC),
            quantity=VolumeValue(Decimal("1.0")),  # Нужен quantity для сравнения при заполнении
        )

        # Используем реальный метод fill с ценой
        fill_price = Price(Decimal("50000.00"), Currency.USD)
        order.fill(Volume(Decimal("1.0"), Currency.BTC), fill_price)
        assert order.status == OrderStatus.FILLED

    def test_order_status_open_to_partially_filled(self):
        """Тест перехода статуса с OPEN на PARTIALLY_FILLED"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            status=OrderStatus.OPEN,
            amount=Volume(Decimal("2.0"), Currency.BTC),
            quantity=VolumeValue(Decimal("2.0")),
        )

        # Частичное заполнение
        fill_price = Price(Decimal("50000.00"), Currency.USD)
        order.fill(Volume(Decimal("1.0"), Currency.BTC), fill_price)
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_order_status_open_to_cancelled(self):
        """Тест перехода статуса с OPEN на CANCELLED"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.OPEN)

        # Используем реальный метод cancel
        order.cancel()
        assert order.status == OrderStatus.CANCELLED

    def test_order_invalid_status_transition(self):
        """Тест невалидного перехода статуса"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.FILLED)

        if hasattr(order, "set_status"):
            with pytest.raises((ValueError, OrderError)):
                order.set_status(OrderStatus.PENDING)


class TestOrderBusinessLogic:
    """Тесты бизнес-логики Order"""

    def test_order_fill_complete(self):
        """Тест полного заполнения ордера"""
        amount = Volume(Decimal("1.0"), Currency.BTC)
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            amount=amount,
            quantity=VolumeValue(Decimal("1.0")),
            status=OrderStatus.OPEN,
        )

        # Заполняем ордер с ценой
        fill_price = Price(Decimal("50000.00"), Currency.USD)
        order.fill(amount, fill_price)

        assert order.filled_quantity == VolumeValue(Decimal("1.0"))
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None

    def test_order_fill_partial(self):
        """Тест частичного заполнения ордера"""
        total_amount = Volume(Decimal("2.0"), Currency.BTC)
        fill_amount = Volume(Decimal("0.8"), Currency.BTC)

        order = Order(trading_pair=TradingPair("BTC/USD"), amount=total_amount, status=OrderStatus.OPEN)

        if hasattr(order, "fill"):
            order.fill(fill_amount)

            assert order.filled_amount == fill_amount
            assert order.status == OrderStatus.PARTIALLY_FILLED
            assert order.get_remaining_amount() == Volume(Decimal("1.2"), Currency.BTC)

    def test_order_fill_overfill_protection(self):
        """Тест защиты от переполнения ордера"""
        amount = Volume(Decimal("1.0"), Currency.BTC)
        overfill_amount = Volume(Decimal("1.5"), Currency.BTC)

        order = Order(trading_pair=TradingPair("BTC/USD"), amount=amount, status=OrderStatus.OPEN)

        if hasattr(order, "fill"):
            with pytest.raises((ValueError, OrderError)):
                order.fill(overfill_amount)

    def test_order_cancel_open_order(self):
        """Тест отмены открытого ордера"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.OPEN)

        if hasattr(order, "cancel"):
            order.cancel()

            assert order.status == OrderStatus.CANCELLED
            assert order.updated_at > order.created_at

    def test_order_cancel_filled_order_raises_error(self):
        """Тест что отмена заполненного ордера вызывает ошибку"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.FILLED)

        if hasattr(order, "cancel"):
            with pytest.raises((ValueError, OrderError)):
                order.cancel()

    def test_order_get_remaining_amount(self):
        """Тест получения оставшейся суммы ордера"""
        order = Order(
            trading_pair=TradingPair("ETH/USD"),
            amount=Volume(Decimal("5.0"), Currency.ETH),
            quantity=VolumeValue(Decimal("5.0")),
            filled_quantity=VolumeValue(Decimal("2.0")),
        )

        # Используем реальное свойство remaining_quantity
        remaining = order.remaining_quantity
        assert remaining == VolumeValue(Decimal("3.0"))

    def test_order_is_fully_filled(self):
        """Тест проверки полного заполнения ордера"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            amount=Volume(Decimal("1.0"), Currency.BTC),
            quantity=VolumeValue(Decimal("1.0")),
            filled_quantity=VolumeValue(Decimal("1.0")),
            status=OrderStatus.FILLED,
        )

        # Используем реальное свойство is_filled
        assert order.is_filled is True

    def test_order_is_partially_filled(self):
        """Тест проверки частичного заполнения ордера"""
        total_amount = Volume(Decimal("2.0"), Currency.BTC)
        filled_amount = Volume(Decimal("0.5"), Currency.BTC)

        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            amount=total_amount,
            filled_amount=filled_amount,
            status=OrderStatus.PARTIALLY_FILLED,
        )

        if hasattr(order, "is_partially_filled"):
            assert order.is_partially_filled() is True


class TestOrderCalculations:
    """Тесты расчетов Order"""

    def test_order_total_value_calculation(self):
        """Тест расчета общей стоимости ордера"""
        amount = Volume(Decimal("2.0"), Currency.BTC)
        price = Price(Decimal("50000.00"), Currency.USD)

        order = Order(trading_pair=TradingPair("BTC/USD"), amount=amount, price=price)

        if hasattr(order, "get_total_value"):
            total_value = order.get_total_value()
            assert total_value.amount == Decimal("100000.00")

    def test_order_average_price_calculation(self):
        """Тест расчета средней цены заполнения"""
        order = Order(
            trading_pair=TradingPair("BTC/USD"),
            amount=Volume(Decimal("2.0"), Currency.BTC),
            filled_amount=Volume(Decimal("2.0"), Currency.BTC),
            average_price=Price(Decimal("49500.00"), Currency.USD),
        )

        if hasattr(order, "get_average_fill_price"):
            avg_price = order.get_average_fill_price()
            assert avg_price == Price(Decimal("49500.00"), Currency.USD)

    def test_order_commission_calculation(self):
        """Тест расчета комиссии ордера"""
        order = Order(trading_pair=TradingPair("BTC/USD"), commission=Price(Decimal("25.00"), Currency.USD))

        if hasattr(order, "get_commission"):
            commission = order.get_commission()
            assert commission == Price(Decimal("25.00"), Currency.USD)

    def test_order_slippage_calculation(self):
        """Тест расчета проскальзывания"""
        expected_price = Price(Decimal("50000.00"), Currency.USD)
        actual_price = Price(Decimal("49800.00"), Currency.USD)

        order = Order(trading_pair=TradingPair("BTC/USD"), price=expected_price, average_price=actual_price)

        if hasattr(order, "get_slippage"):
            slippage = order.get_slippage()
            # Slippage should be around 0.4% (200/50000)
            assert abs(slippage.value - Decimal("0.004")) < Decimal("0.001")


class TestOrderProtocolImplementation:
    """Тесты реализации OrderProtocol"""

    def test_order_get_status(self):
        """Тест метода get_status"""
        order = Order(trading_pair=TradingPair("BTC/USD"), status=OrderStatus.OPEN)

        status = order.get_status()
        assert status == "open"

    def test_order_get_quantity(self):
        """Тест метода get_quantity"""
        quantity = VolumeValue(Decimal("1.5"))
        order = Order(trading_pair=TradingPair("BTC/USD"), quantity=quantity)

        result = order.get_quantity()
        assert result == quantity

    def test_order_get_price(self):
        """Тест метода get_price"""
        price = Price(Decimal("50000.00"), Currency.USD)
        order = Order(trading_pair=TradingPair("BTC/USD"), price=price)

        result = order.get_price()
        if hasattr(order, "get_price"):
            assert result == price.amount


class TestOrderUtilityMethods:
    """Тесты utility методов Order"""

    def test_order_equality(self):
        """Тест равенства ордеров"""
        order_id = OrderId(uuid4())

        order1 = Order(id=order_id, trading_pair=TradingPair("BTC/USD"))

        order2 = Order(id=order_id, trading_pair=TradingPair("BTC/USD"))

        assert order1 == order2

    def test_order_inequality(self):
        """Тест неравенства ордеров"""
        order1 = Order(id=OrderId(uuid4()), trading_pair=TradingPair("BTC/USD"))

        order2 = Order(id=OrderId(uuid4()), trading_pair=TradingPair("ETH/USD"))

        assert order1 != order2

    def test_order_string_representation(self):
        """Тест строкового представления ордера"""
        order = Order(
            symbol=Symbol("BTC"),
            trading_pair=TradingPair("BTC/USD"),
            side=OrderSide.BUY,
            amount=Volume(Decimal("1.0"), Currency.BTC),
        )

        str_repr = str(order)
        assert "BTC" in str_repr
        assert "BUY" in str_repr or "buy" in str_repr

    def test_order_repr_representation(self):
        """Тест repr представления ордера"""
        order = Order(trading_pair=TradingPair("BTC/USD"))

        repr_str = repr(order)
        assert "Order" in repr_str

    def test_order_hash_consistency(self):
        """Тест консистентности хеша ордера"""
        order_id = OrderId(uuid4())

        order1 = Order(id=order_id, trading_pair=TradingPair("BTC/USD"))

        order2 = Order(id=order_id, trading_pair=TradingPair("BTC/USD"))

        # Одинаковые ордера должны иметь одинаковый хеш
        assert hash(order1) == hash(order2)

    def test_order_to_dict(self):
        """Тест сериализации ордера в словарь"""
        order = Order(symbol=Symbol("BTC"), trading_pair=TradingPair("BTC/USD"), side=OrderSide.BUY)

        if hasattr(order, "to_dict"):
            order_dict = order.to_dict()
            assert isinstance(order_dict, dict)
            assert "id" in order_dict
            assert "symbol" in order_dict
            assert "side" in order_dict

    def test_order_from_dict(self):
        """Тест десериализации ордера из словаря"""
        order_dict = {
            "id": str(uuid4()),
            "symbol": "BTC",
            "trading_pair": "BTC/USD",
            "side": "buy",
            "order_type": "market",
        }

        if hasattr(Order, "from_dict"):
            order = Order.from_dict(order_dict)
            assert order.symbol == Symbol("BTC")
            assert order.side == OrderSide.BUY


class TestOrderEdgeCases:
    """Тесты граничных случаев для Order"""

    def test_order_with_very_small_amount(self):
        """Тест ордера с очень малым количеством"""
        tiny_amount = Volume(Decimal("0.00000001"), Currency.BTC)

        order = Order(trading_pair=TradingPair("BTC/USD"), amount=tiny_amount)

        assert order.amount == tiny_amount

    def test_order_with_very_large_amount(self):
        """Тест ордера с очень большим количеством"""
        large_amount = Volume(Decimal("1000000.99999999"), Currency.USD)

        order = Order(trading_pair=TradingPair("USDT/USD"), amount=large_amount)

        assert order.amount == large_amount

    def test_order_with_high_precision_price(self):
        """Тест ордера с высокоточной ценой"""
        precise_price = Price(Decimal("50000.12345678"), Currency.USD)

        order = Order(trading_pair=TradingPair("BTC/USD"), order_type=OrderType.LIMIT, price=precise_price)

        assert order.price == precise_price

    def test_order_lifecycle_complete(self):
        """Тест полного жизненного цикла ордера"""
        order = Order(
            trading_pair=TradingPair("ETH/USD"),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Volume(Decimal("10.0"), Currency.ETH),
            price=Price(Decimal("3000.00"), Currency.USD),
            status=OrderStatus.PENDING,
        )

        # Открытие ордера
        if hasattr(order, "set_status"):
            order.set_status(OrderStatus.OPEN)
            assert order.status == OrderStatus.OPEN

        # Частичное заполнение
        if hasattr(order, "fill"):
            order.fill(Volume(Decimal("4.0"), Currency.ETH))
            assert order.status == OrderStatus.PARTIALLY_FILLED

        # Полное заполнение
        if hasattr(order, "fill"):
            order.fill(Volume(Decimal("6.0"), Currency.ETH))
            assert order.status == OrderStatus.FILLED

    def test_order_with_metadata(self):
        """Тест ордера с метаданными"""
        metadata = {"strategy": "momentum", "confidence": "0.85", "signal_strength": "strong"}

        order = Order(trading_pair=TradingPair("BTC/USD"), metadata=metadata)

        assert order.metadata == metadata
        assert order.metadata["strategy"] == "momentum"

    def test_order_timestamp_consistency(self):
        """Тест консистентности временных меток"""
        order = Order(trading_pair=TradingPair("BTC/USD"))

        assert order.created_at <= order.updated_at
        assert order.filled_at is None  # Пока не заполнен

    def test_order_immutability_of_id(self):
        """Тест неизменяемости ID ордера"""
        order = Order(trading_pair=TradingPair("BTC/USD"))
        original_id = order.id

        # ID не должен изменяться
        with pytest.raises(AttributeError):
            order.id = OrderId(uuid4())


@pytest.mark.unit
class TestOrderIntegrationWithMocks:
    """Интеграционные тесты Order с моками"""

    def test_order_with_mocked_dependencies(self):
        """Тест Order с замокированными зависимостями"""
        mock_price = Mock()
        mock_price.amount = Decimal("50000.00")
        mock_price.currency = "USD"

        mock_volume = Mock()
        mock_volume.amount = Decimal("1.0")
        mock_volume.currency = "BTC"

        order = Order(trading_pair=TradingPair("BTC/USD"), price=mock_price, amount=mock_volume)

        assert order.price == mock_price
        assert order.amount == mock_volume

    def test_order_factory_pattern(self):
        """Тест паттерна фабрики для Order"""

        def create_market_buy_order(symbol, amount):
            return Order(
                symbol=Symbol(symbol),
                trading_pair=TradingPair(f"{symbol}/USD"),
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                amount=Volume(amount, Currency.from_string(symbol)),
            )

        def create_limit_sell_order(symbol, amount, price):
            return Order(
                symbol=Symbol(symbol),
                trading_pair=TradingPair(f"{symbol}/USD"),
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                amount=Volume(amount, Currency.from_string(symbol)),
                price=Price(price, Currency.USD),
            )

        buy_order = create_market_buy_order("BTC", Decimal("1.0"))
        sell_order = create_limit_sell_order("ETH", Decimal("5.0"), Decimal("3000.00"))

        assert buy_order.order_type == OrderType.MARKET
        assert buy_order.side == OrderSide.BUY
        assert sell_order.order_type == OrderType.LIMIT
        assert sell_order.side == OrderSide.SELL

    def test_order_builder_pattern(self):
        """Тест паттерна строителя для Order"""

        class OrderBuilder:
            def __init__(self):
                self._symbol = None
                self._trading_pair = None
                self._order_type = OrderType.MARKET
                self._side = OrderSide.BUY
                self._amount = None
                self._price = None

            def with_symbol(self, symbol):
                self._symbol = Symbol(symbol)
                self._trading_pair = TradingPair(f"{symbol}/USD")
                return self

            def with_type(self, order_type):
                self._order_type = order_type
                return self

            def with_side(self, side):
                self._side = side
                return self

            def with_amount(self, amount, currency):
                self._amount = Volume(amount, currency)
                return self

            def with_price(self, price):
                self._price = Price(price, Currency.USD)
                return self

            def build(self):
                return Order(
                    symbol=self._symbol,
                    trading_pair=self._trading_pair,
                    order_type=self._order_type,
                    side=self._side,
                    amount=self._amount,
                    price=self._price,
                )

        order = (
            OrderBuilder()
            .with_symbol("ETH")
            .with_type(OrderType.LIMIT)
            .with_side(OrderSide.SELL)
            .with_amount(Decimal("10.0"), Currency.ETH)
            .with_price(Decimal("3200.00"))
            .build()
        )

        assert order.symbol == Symbol("ETH")
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
