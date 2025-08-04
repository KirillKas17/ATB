"""
Сервис для создания ордеров.
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from uuid import uuid4

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.portfolio import Portfolio
from domain.exceptions import (
    InsufficientFundsError,
    InvalidOrderError,
    OrderManagementError,
)
from domain.type_definitions import OrderId, TradingPair, VolumeValue, PortfolioId
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp


class OrderCreator:
    """Сервис для создания ордеров."""

    def create_order(
        self,
        portfolio: Portfolio,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> Tuple[Order, Decimal, List[str]]:
        """Создание нового ордера."""
        try:
            # Валидация ордера
            is_valid, validation_errors = self._validate_order(
                portfolio, symbol, order_type, side, amount, price, stop_price
            )
            if not is_valid:
                raise InvalidOrderError(
                    f"Order validation failed: {', '.join(validation_errors)}"
                )
            # Проверка достаточности средств
            estimated_cost = self._calculate_order_cost(side, amount, price)
            if side == OrderSide.BUY:
                if portfolio.free_margin.amount < estimated_cost:
                    raise InsufficientFundsError(
                        f"Insufficient funds. Required: {estimated_cost}, Available: {portfolio.free_margin.amount}"
                    )
            # Создаем ордер
            order = Order(
                id=OrderId(uuid4()),
                portfolio_id=portfolio.id,
                trading_pair=TradingPair(symbol),
                order_type=order_type,
                side=side,
                quantity=VolumeValue(amount),
                price=Price(price, Currency.USD) if price else None,
                stop_price=Price(stop_price, Currency.USD) if stop_price else None,
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now(),
            )
            # Генерируем предупреждения
            warnings: List[str] = self._generate_warnings(order, portfolio)
            return order, estimated_cost, warnings
        except Exception as e:
            raise OrderManagementError(f"Error creating order: {str(e)}")

    def _validate_order(
        self,
        portfolio: Portfolio,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
    ) -> Tuple[bool, List[str]]:
        """Валидация ордера."""
        errors = []
        # Проверка символа
        if not symbol or len(symbol.strip()) == 0:
            errors.append("Symbol is required")
        # Проверка количества
        if amount <= 0:
            errors.append("Amount must be positive")
        # Проверка цены для лимитных ордеров
        if order_type == OrderType.LIMIT and (price is None or price <= 0):
            errors.append("Price is required for limit orders")
        # Проверка стоп-цены для стоп-ордеров
        if order_type == OrderType.STOP and (stop_price is None or stop_price <= 0):
            errors.append("Stop price is required for stop orders")
        # Проверка портфеля
        if not portfolio:
            errors.append("Portfolio is required")
        return len(errors) == 0, errors

    def _calculate_order_cost(
        self, side: OrderSide, amount: Decimal, price: Optional[Decimal]
    ) -> Decimal:
        """Расчет стоимости ордера."""
        if side == OrderSide.BUY and price:
            return amount * price
        return Decimal("0")

    def _generate_warnings(self, order: Order, portfolio: Portfolio) -> List[str]:
        """Генерация предупреждений для ордера."""
        warnings: List[str] = []
        # Проверка для рыночных ордеров на продажу
        if order.order_type == OrderType.MARKET and order.side == OrderSide.SELL:
            # Проверка достаточности позиции для продажи
            try:
                # В реальной системе здесь был бы запрос к репозиторию позиций
                # position = await self.position_repository.get_position_by_symbol(
                #     portfolio.id, order.trading_pair
                # )
                # if not position or position.volume.amount < order.quantity:
                #     warnings.append("Insufficient position for sell order")
                # Упрощенная проверка - предполагаем, что позиция есть
                warnings.append(
                    "Position check required - implement position repository integration"
                )
            except Exception as e:
                warnings.append(f"Error checking position: {str(e)}")
        return warnings
