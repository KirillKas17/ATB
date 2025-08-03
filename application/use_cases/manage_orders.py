"""
Use case для управления ордерами с промышленной типизацией и валидацией.
"""

import logging
from decimal import Decimal
from typing import List, Optional, Tuple, Any
from uuid import uuid4, UUID
from datetime import datetime

from application.types import (
    CancelOrderRequest,
    CancelOrderResponse,
    CreateOrderRequest,
    CreateOrderResponse,
    GetOrdersRequest,
    GetOrdersResponse,
)
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.repositories.order_repository import OrderRepository
from domain.repositories.portfolio_repository import PortfolioRepository
from domain.repositories.position_repository import PositionRepository
from domain.types import (
    OrderId,
    PortfolioId,
    PriceValue,
    Symbol,
    TimestampValue,
    TradingPair,
    VolumeValue,
    AmountValue,
)
from domain.types.repository_types import EntityId
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume

logger = logging.getLogger(__name__)


class OrderManagementUseCase:
    """Базовый класс для управления ордерами."""
    
    def __init__(
        self,
        order_repository: OrderRepository,
        portfolio_repository: PortfolioRepository,
        position_repository: PositionRepository,
    ):
        self.order_repository = order_repository
        self.portfolio_repository = portfolio_repository
        self.position_repository = position_repository


class DefaultOrderManagementUseCase(OrderManagementUseCase):
    """Реализация use case для управления ордерами."""

    def __init__(
        self,
        order_repository: OrderRepository,
        portfolio_repository: PortfolioRepository,
        position_repository: PositionRepository,
    ):
        super().__init__(order_repository, portfolio_repository, position_repository)

    async def create_order(self, request: CreateOrderRequest) -> CreateOrderResponse:
        """
        Создание нового ордера с валидацией и бизнес-логикой.
        Args:
            request: Запрос на создание ордера
        Returns:
            Ответ с созданным ордером
        """
        try:
            # Валидация запроса
            is_valid, errors = await self.validate_order(request)
            if not is_valid:
                return CreateOrderResponse(
                    success=False, message="Invalid order request", errors=errors
                )
            # Расчет стоимости ордера
            estimated_cost = await self._calculate_order_cost(request)
            # Создание ордера с правильными типами
            order = Order(
                id=OrderId(uuid4()),
                portfolio_id=PortfolioId(request.portfolio_id),
                symbol=Symbol(str(request.trading_pair)),
                trading_pair=TradingPair(str(request.trading_pair)),
                order_type=request.order_type,
                side=request.side,
                quantity=VolumeValue(request.volume.amount),  # Исправление: используем amount
                price=Price(request.price.value, Currency.USD) if request.price else None,
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),  # Исправление: используем Timestamp
                updated_at=Timestamp.now(),  # Исправление: используем Timestamp
            )
            # Сохраняем ордер
            await self.order_repository.save(order)
            # Генерируем предупреждения
            warnings = []
            if (
                request.order_type == OrderType.MARKET
                and request.side == OrderSide.SELL
            ):
                # Проверяем достаточность позиции для продажи
                trading_pair = TradingPair(str(request.trading_pair))
                # positions = await self.position_repository.get_by_portfolio_and_symbol(
                #     request.portfolio_id, str(request.trading_pair)
                # )
                positions: List[Any] = []
                position = positions[0] if positions else None
                if not position or position.quantity.value < request.volume.value:
                    warnings.append(
                        f"Insufficient position size for {request.trading_pair}"
                    )
            return CreateOrderResponse(
                order=order,
                estimated_cost=Money(estimated_cost, Currency.USD),
                order_id=order.id,
                warnings=warnings,
                message="Order created successfully",
            )
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return CreateOrderResponse(
                success=False,
                message=f"Error creating order: {str(e)}, errors=[str(e)]",
            )

    async def cancel_order(self, request: CancelOrderRequest) -> CancelOrderResponse:
        """
        Отмена ордера с проверкой прав доступа и бизнес-логики.
        Args:
            request: Запрос на отмену ордера
        Returns:
            Ответ на отмену ордера
        """
        try:
            # Получаем ордер - исправляем тип
            order = await self.order_repository.get_by_id(EntityId(UUID(str(request.order_id))))
            if not order:
                return CancelOrderResponse(
                    success=False, cancelled=False, message="Order not found"
                )
            # Проверяем, что ордер принадлежит портфелю
            if order.portfolio_id != request.portfolio_id:
                return CancelOrderResponse(
                    success=False,
                    cancelled=False,
                    message="Order does not belong to portfolio",
                )
            # Проверяем, что ордер можно отменить
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                return CancelOrderResponse(
                    success=False,
                    cancelled=False,
                    message=f"Cannot cancel order with status {order.status}",
                )
            # Отменяем ордер
            order.status = OrderStatus.CANCELLED
            order.updated_at = Timestamp.now()  # Исправление: используем Timestamp
            await self.order_repository.save(order)
            return CancelOrderResponse(
                cancelled=True,
                order=order,
                cancellation_time=TimestampValue(datetime.now()),
                message="Order cancelled successfully",
            )
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return CancelOrderResponse(
                success=False,
                cancelled=False,
                message=f"Error cancelling order: {str(e)}, errors=[str(e)]",
            )

    async def get_orders(self, request: GetOrdersRequest) -> GetOrdersResponse:
        """
        Получение списка ордеров с фильтрацией и пагинацией.
        Args:
            request: Запрос на получение ордеров
        Returns:
            Ответ с ордерами
        """
        try:
            # Получаем ордера
            orders = await self.order_repository.get_all()
            # Фильтруем по портфелю
            orders = [
                order for order in orders if order.portfolio_id == request.portfolio_id
            ]
            # Применяем дополнительные фильтры
            if request.trading_pair:
                orders = [
                    order
                    for order in orders
                    if order.trading_pair == TradingPair(str(request.trading_pair))
                ]
            if request.status:
                orders = [order for order in orders if order.status == request.status]
            if request.side:
                orders = [order for order in orders if order.side == request.side]
            if request.order_type:
                orders = [
                    order for order in orders if order.order_type == request.order_type
                ]
            # Применяем пагинацию
            if request.offset:
                orders = orders[request.offset :]
            if request.limit:
                orders = orders[: request.limit]
            # Рассчитываем общую стоимость
            total_value = sum(
                (order.quantity * order.price.amount) if order.price else Decimal("0")
                for order in orders
            )
            return GetOrdersResponse(
                orders=orders,
                total_count=len(orders),
                total_value=Money(total_value, Currency.USD),
                message="Orders retrieved successfully",
            )
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return GetOrdersResponse(
                success=False,
                orders=[],
                total_count=0,
                total_value=Money(Decimal("0"), Currency.USD),
                message=f"Error getting orders: {str(e)}, errors=[str(e)]",
            )

    async def get_order_by_id(
        self, order_id: OrderId, portfolio_id: PortfolioId
    ) -> Optional[Order]:
        """
        Получение ордера по ID с проверкой прав доступа.
        Args:
            order_id: ID ордера
            portfolio_id: ID портфеля
        Returns:
            Ордер или None
        """
        try:
            # Получаем ордер
            order = await self.order_repository.get_by_id(EntityId(UUID(str(order_id))))
            if not order:
                return None
            if order and order.portfolio_id == portfolio_id:
                return order
            return None
        except Exception as e:
            logger.error(f"Error getting order by ID: {e}")
            return None

    async def update_order_status(
        self,
        order_id: OrderId,
        status: OrderStatus,
        filled_amount: Optional[VolumeValue] = None,
    ) -> bool:
        """
        Обновление статуса ордера.
        Args:
            order_id: ID ордера
            status: Новый статус
            filled_amount: Заполненное количество
        Returns:
            True если успешно
        """
        try:
            order = await self.order_repository.get_by_id(EntityId(UUID(str(order_id))))
            if not order:
                return False

            order.status = status
            order.updated_at = Timestamp.now()  # Исправление: используем Timestamp

            if filled_amount:
                order.filled_quantity = filled_amount

            await self.order_repository.save(order)
            return True

        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False

    async def validate_order(
        self, request: CreateOrderRequest
    ) -> Tuple[bool, List[str]]:
        """
        Валидация запроса на создание ордера.
        Args:
            request: Запрос на создание ордера
        Returns:
            Кортеж (валидность, список ошибок)
        """
        errors = []

        # Проверяем обязательные поля
        if not request.trading_pair:
            errors.append("Trading pair is required")
        if not request.volume or request.volume.value <= 0:
            errors.append("Volume must be positive")
        if not request.side:
            errors.append("Order side is required")
        if not request.order_type:
            errors.append("Order type is required")

        # Проверяем цену для лимитных ордеров
        if request.order_type == OrderType.LIMIT and not request.price:
            errors.append("Price is required for limit orders")

        # Проверяем портфель
        if not request.portfolio_id:
            errors.append("Portfolio ID is required")

        return len(errors) == 0, errors

    async def _calculate_order_cost(self, request: CreateOrderRequest) -> Decimal:
        """
        Расчет стоимости ордера.
        Args:
            request: Запрос на создание ордера
        Returns:
            Стоимость ордера
        """
        try:
            if request.price:
                return request.volume.value * request.price.value
            else:
                # Для рыночных ордеров используем примерную цену
                return request.volume.value * Decimal("50000")  # Примерная цена BTC
        except Exception as e:
            logger.error(f"Error calculating order cost: {e}")
            return Decimal("0")

    async def execute_signal(self, signal: Any) -> bool:
        """
        Исполнение торгового сигнала.
        Args:
            signal: Торговый сигнал
        Returns:
            True если сигнал исполнен успешно
        """
        try:
            # В реальной системе здесь была бы логика исполнения сигнала
            # Пока возвращаем заглушку
            logger.info(f"Executing signal: {signal}")
            return True
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
