"""
Сервис для валидации ордеров.
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from uuid import UUID

from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.portfolio import Portfolio
from domain.types import PortfolioId
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume


class OrderValidator:
    """Сервис для валидации ордеров."""

    def __init__(self):
        self.config = {
            "min_order_size": Decimal("10"),
            "max_order_size": Decimal("1000000"),
            "min_price": Decimal("0.000001"),
            "max_price": Decimal("1000000"),
        }

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
        min_order_size: Optional[Decimal] = None,
        max_order_size: Optional[Decimal] = None,
    ) -> Tuple[bool, List[str]]:
        """Валидация ордера."""
        errors = []
        # Базовая валидация
        basic_errors = self._validate_basic_order(order)
        errors.extend(basic_errors)
        # Валидация размера
        min_size = (
            min_order_size
            if min_order_size is not None
            else self.config["min_order_size"]
        )
        max_size = (
            max_order_size
            if max_order_size is not None
            else self.config["max_order_size"]
        )
        size_errors = self._validate_order_size(order, min_size, max_size)
        errors.extend(size_errors)
        # Валидация цены
        price_errors = self._validate_order_price(order, current_price)
        errors.extend(price_errors)
        # Валидация средств
        funds_errors = self._validate_sufficient_funds(order, portfolio)
        errors.extend(funds_errors)
        # Валидация лимитов
        limit_errors = self._validate_order_limits(order, portfolio)
        errors.extend(limit_errors)
        return len(errors) == 0, errors

    def _validate_basic_order(self, order: Order) -> List[str]:
        """Валидация базовых параметров ордера."""
        errors = []
        # Проверка количества
        if hasattr(order.quantity, 'amount'):
            quantity_value = order.quantity.amount
        else:
            quantity_value = order.quantity
        if quantity_value <= 0:
            errors.append("Order quantity must be positive")
        # Проверка цены
        if order.price is None:
            errors.append("Order price must be positive")
        else:
            # Безопасное преобразование Price в Decimal для сравнения
            if hasattr(order.price, 'amount'):
                price_value = order.price.amount
            else:
                price_value = order.price  # type: ignore
            if price_value <= 0:
                errors.append("Order price must be positive")
        return errors

    def _validate_order_size(
        self, order: Order, min_size: Decimal, max_size: Decimal
    ) -> List[str]:
        """Валидация размера ордера."""
        errors = []
        # Используем цену из ордера или текущую рыночную цену
        if hasattr(order.price, 'amount'):
            order_price = order.price.amount if order.price else Decimal("0")
        else:
            order_price = order.price if order.price else Decimal("0")  # type: ignore
        
        if hasattr(order.quantity, 'amount'):
            quantity_value = order.quantity.amount
        else:
            quantity_value = order.quantity  # type: ignore
            
        order_value = quantity_value * order_price
        if order_value < min_size:
            errors.append(f"Order value {order_value} is below minimum {min_size}")
        if order_value > max_size:
            errors.append(f"Order value {order_value} exceeds maximum {max_size}")
        return errors

    def _validate_order_price(self, order: Order, current_price: Decimal) -> List[str]:
        """Валидация цены ордера."""
        errors = []
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            # Получаем цену ордера
            if hasattr(order.price, 'amount'):
                order_price_value = order.price.amount
            else:
                order_price_value = order.price  # type: ignore
                
            # Проверка разумности цены относительно текущей
            price_diff_ratio = abs(order_price_value - current_price) / current_price
            if price_diff_ratio > Decimal("0.5"):
                errors.append("Order price deviates more than 50% from current price")
            # Проверка минимального спреда для лимитных ордеров
            min_spread = current_price * Decimal("0.001")  # 0.1%
            if abs(order_price_value - current_price) < min_spread:
                errors.append("Order price too close to current price")
        return errors

    def _validate_sufficient_funds(
        self, order: Order, portfolio: Portfolio
    ) -> List[str]:
        """Валидация достаточности средств."""
        errors = []
        if order.price:
            # Получаем значения цены и количества
            if hasattr(order.price, 'amount'):
                price_value = order.price.amount
            else:
                price_value = order.price  # type: ignore
                
            if hasattr(order.quantity, 'amount'):
                quantity_value = order.quantity.amount
            else:
                quantity_value = order.quantity  # type: ignore
                
            # Проверка достаточности средств
            required_funds = quantity_value * price_value
            if hasattr(portfolio.free_margin, 'amount'):
                free_margin_value = portfolio.free_margin.amount
            else:
                free_margin_value = portfolio.free_margin  # type: ignore
                
            if free_margin_value < required_funds:
                errors.append(
                    f"Insufficient funds. Required: {required_funds}, Available: {free_margin_value}"
                )
        return errors

    def _validate_order_limits(self, order: Order, portfolio: Portfolio) -> List[str]:
        """Валидация лимитов портфеля."""
        errors = []
        # Проверка размера позиции
        if hasattr(portfolio.total_balance, 'amount'):
            total_balance_value = portfolio.total_balance.amount
        else:
            total_balance_value = portfolio.total_balance  # type: ignore
            
        max_position_size = total_balance_value * Decimal(
            self.config.get("max_position_size_ratio", "0.2")
        )
        
        if hasattr(order.price, 'amount'):
            price_value = order.price.amount if order.price else Decimal("0")
        else:
            price_value = order.price if order.price else Decimal("0")  # type: ignore
            
        if hasattr(order.quantity, 'amount'):
            quantity_value = order.quantity.amount
        else:
            quantity_value = order.quantity  # type: ignore
            
        order_value = quantity_value * price_value
        if order_value > max_position_size:
            errors.append(
                f"Order size {order_value} exceeds maximum position size {max_position_size}"
            )
        # Проверка дневного лимита ордеров
        # Здесь можно добавить логику проверки количества ордеров за день
        return errors

    def validate_stop_loss(
        self,
        order: Order,
        entry_price: Decimal,
        max_stop_distance: Decimal = Decimal("0.1"),
    ) -> Tuple[bool, List[str]]:
        """Валидация стоп-лосса."""
        errors = []
        if order.order_type == OrderType.STOP_LIMIT and order.price:
            # Проверка стоп-лосса
            if hasattr(order, 'stop_loss') and order.stop_loss and entry_price > 0:
                if hasattr(order.price, 'amount'):
                    price_value = order.price.amount
                else:
                    price_value = order.price  # type: ignore
                stop_distance = abs(price_value - entry_price) / entry_price
                if stop_distance > max_stop_distance:
                    errors.append(f"Stop loss distance {stop_distance:.2%} exceeds maximum {max_stop_distance:.2%}")
                if stop_distance < Decimal("0.001"):  # 0.1%
                    errors.append("Stop loss too close to entry price")
        return len(errors) == 0, errors

    def validate_take_profit(
        self,
        order: Order,
        entry_price: Decimal,
        min_tp_distance: Decimal = Decimal("0.005"),
    ) -> Tuple[bool, List[str]]:
        """Валидация тейк-профита."""
        errors = []
        if order.order_type == OrderType.LIMIT and order.price:
            if hasattr(order.price, 'amount'):
                price_value = order.price.amount
            else:
                price_value = order.price  # type: ignore
            tp_distance = abs(price_value - entry_price) / entry_price
            if tp_distance < min_tp_distance:
                errors.append(
                    f"Take profit distance {tp_distance:.2%} below minimum {min_tp_distance:.2%}"
                )
        return len(errors) == 0, errors

    def validate_batch_orders(
        self, orders: List[Order], portfolio: Portfolio
    ) -> Tuple[bool, List[str]]:
        """Валидация пакета ордеров."""
        errors = []
        # Проверка пакетных ордеров
        total_value = Decimal("0")
        for order in orders:
            if hasattr(order.price, 'amount'):
                price_value = order.price.amount if order.price else Decimal("0")
            else:
                price_value = order.price if order.price else Decimal("0")  # type: ignore
                
            if hasattr(order.quantity, 'amount'):
                quantity_value = order.quantity.amount
            else:
                quantity_value = order.quantity  # type: ignore
                
            total_value += quantity_value * price_value
            
        if hasattr(portfolio.free_margin, 'amount'):
            free_margin_value = portfolio.free_margin.amount
        else:
            free_margin_value = portfolio.free_margin  # type: ignore
            
        if total_value > free_margin_value:
            errors.append(
                f"Total batch value {total_value} exceeds available funds {free_margin_value}"
            )
        # Проверка каждого ордера
        for i, order in enumerate(orders):
            if hasattr(order.price, 'amount'):
                current_price = order.price.amount if order.price else Decimal("0")
            else:
                current_price = order.price if order.price else Decimal("0")  # type: ignore
            is_valid, order_errors = self.validate_order(
                order, portfolio, current_price
            )
            if not is_valid:
                errors.extend([f"Order {i+1}: {error}" for error in order_errors])
        return len(errors) == 0, errors

    def validate_order_modification(
        self, original_order: Order, modified_order: Order
    ) -> Tuple[bool, List[str]]:
        """Валидация модификации ордера."""
        errors = []
        # Проверка, что ордер еще не исполнен
        if original_order.status in ["FILLED", "CANCELLED"]:
            errors.append("Cannot modify filled or cancelled order")
        # Проверка, что основные параметры не изменились
        if original_order.trading_pair != modified_order.trading_pair:
            errors.append("Cannot change trading pair")
        if original_order.side != modified_order.side:
            errors.append("Cannot change order side")
        if original_order.order_type != modified_order.order_type:
            errors.append("Cannot change order type")
        return len(errors) == 0, errors
