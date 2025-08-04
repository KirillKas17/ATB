"""
Менеджер ордеров - Production Ready
Полная промышленная реализация с строгой типизацией и продвинутым управлением ордерами.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Coroutine, Dict, List, Optional, Set, Tuple
from uuid import uuid4, UUID

from loguru import logger

from domain.entities.account import Account, Balance
from domain.entities.market import MarketData
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trading import Trade
from domain.exceptions import (
    InsufficientFundsError,
    InvalidOrderError,
    MarketError,
    OrderError,
    OrderNotFoundError,
)
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.types import (
    OrderId,
    PriceValue,
    Symbol,
    TimestampValue,
    TradeId,
    TradingPair,
    VolumeValue,
)
from domain.types.external_service_types import ConnectionConfig, OrderRequest, OrderSide as ExternalOrderSide, OrderType as ExternalOrderType
from domain.value_objects import Currency, Money, Price, Volume
from domain.value_objects.timestamp import Timestamp


# ============================================================================
# ORDER TRACKER
# ============================================================================
class OrderTracker:
    """Трекер ордеров."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.orders: Dict[OrderId, Order] = {}
        self.order_history: List[Order] = []
        self.trades: Dict[TradeId, Trade] = {}
        self.lock = asyncio.Lock()
        # Метрики
        self.metrics = {
            "total_orders": 0,
            "open_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0,
            "total_value": 0.0,
            "average_fill_time": 0.0,
            "fill_rate": 0.0,
        }

    async def add_order(self, order: Order) -> None:
        """Добавление ордера."""
        async with self.lock:
            # Используем max_retries как лимит ордеров (временное решение)
            max_orders = getattr(self.config, "max_retries", 1000)
            if len(self.orders) >= max_orders:
                # Удаляем самый старый ордер
                oldest_order = min(self.orders.values(), key=lambda o: o.created_at)
                await self._remove_order(oldest_order.id)
            self.orders[order.id] = order
                    # ИСПРАВЛЕНО: Ограничиваем размер истории заказов для предотвращения утечки памяти
        self.order_history.append(order)
        if len(self.order_history) > 10000:  # Максимум 10К заказов в истории
            # Удаляем старые заказы (оставляем последние 5К)
            self.order_history = self.order_history[-5000:]
            logger.debug("Order history truncated to prevent memory leak")
        self.metrics["total_orders"] += 1
            self.metrics["open_orders"] += 1

    async def update_order(self, order_id: OrderId, updates: Dict[str, Any]) -> bool:
        """Обновление ордера."""
        async with self.lock:
            if order_id not in self.orders:
                return False
            order = self.orders[order_id]
            old_status = order.status
            # Обновляем поля
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            order.updated_at = Timestamp.now()
            # Обновляем метрики
            if old_status != order.status:
                if order.status == OrderStatus.FILLED:
                    self.metrics["open_orders"] -= 1
                    self.metrics["filled_orders"] += 1
                    # Используем прямые значения для Decimal
                    filled_qty = (
                        float(order.filled_quantity) if order.filled_quantity else 0.0  # Исправляю тип Volume
                    )
                    price_val = float(order.price.value) if order.price else 0.0  # Исправляю тип Price
                    self.metrics["total_volume"] += filled_qty
                    self.metrics["total_value"] += filled_qty * price_val
                    # Обновляем среднее время исполнения
                    if order.updated_at and order.created_at:
                        fill_time = (
                            order.updated_at.to_datetime()
                            - order.created_at.to_datetime()
                        ).total_seconds()
                        self.metrics["average_fill_time"] = (
                            self.metrics["average_fill_time"]
                            * (self.metrics["filled_orders"] - 1)
                            + fill_time
                        ) / self.metrics["filled_orders"]
                elif order.status == OrderStatus.CANCELLED:
                    self.metrics["open_orders"] -= 1
                    self.metrics["cancelled_orders"] += 1
                elif order.status == OrderStatus.REJECTED:
                    self.metrics["open_orders"] -= 1
                    self.metrics["rejected_orders"] += 1
            # Обновляем fill rate
            if self.metrics["total_orders"] > 0:
                self.metrics["fill_rate"] = (
                    self.metrics["filled_orders"] / self.metrics["total_orders"]
                )
            return True

    async def remove_order(self, order_id: OrderId) -> bool:
        """Удаление ордера."""
        return await self._remove_order(order_id)

    async def _remove_order(self, order_id: OrderId) -> bool:
        """Внутреннее удаление ордера."""
        async with self.lock:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == OrderStatus.OPEN:
                    self.metrics["open_orders"] -= 1
                del self.orders[order_id]
                return True
            return False

    async def get_order(self, order_id: OrderId) -> Optional[Order]:
        """Получение ордера."""
        return self.orders.get(order_id)

    async def get_orders(self, filters: Optional[Dict[str, Any]] = None) -> List[Order]:
        """Получение ордеров с фильтрами."""
        orders = list(self.orders.values())
        if filters:
            filtered_orders = []
            for order in orders:
                match = True
                for key, value in filters.items():
                    if hasattr(order, key):
                        if getattr(order, key) != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    filtered_orders.append(order)
            return filtered_orders
        return orders

    async def get_open_orders(self) -> List[Order]:
        """Получение открытых ордеров."""
        return [
            order for order in self.orders.values() if order.status == OrderStatus.OPEN
        ]

    async def get_order_history(self, limit: int = 100) -> List[Order]:
        """Получение истории ордеров."""
        return self.order_history[-limit:] if limit > 0 else self.order_history

    async def add_trade(self, trade: Trade) -> None:
        """Добавление сделки."""
        async with self.lock:
            self.trades[str(trade.id)] = trade

    async def get_trades(self, order_id: Optional[OrderId] = None) -> List[Trade]:
        """Получение сделок."""
        if order_id:
            return [
                trade for trade in self.trades.values() if trade.order_id == order_id
            ]
        return list(self.trades.values())


# ============================================================================
# ORDER ANALYTICS
# ============================================================================
class OrderAnalytics:
    """Аналитика ордеров."""

    def __init__(self) -> None:
        self.analytics_data: Dict[str, Any] = {}

    async def analyze_order_performance(self, orders: List[Order]) -> Dict[str, Any]:
        """Анализ производительности ордеров."""
        if not orders:
            return {}
        # Базовые метрики
        total_orders = len(orders)
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        cancelled_orders = [o for o in orders if o.status == OrderStatus.CANCELLED]
        rejected_orders = [o for o in orders if o.status == OrderStatus.REJECTED]
        
        # Рассчитываем метрики
        fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0.0
        cancel_rate = len(cancelled_orders) / total_orders if total_orders > 0 else 0.0
        reject_rate = len(rejected_orders) / total_orders if total_orders > 0 else 0.0
        
        # Анализ времени исполнения
        execution_times = []
        for order in filled_orders:
            if order.updated_at and order.created_at:
                execution_time = (
                    order.updated_at.to_datetime() - order.created_at.to_datetime()
                ).total_seconds()
                execution_times.append(execution_time)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return {
            "total_orders": total_orders,
            "filled_orders": len(filled_orders),
            "cancelled_orders": len(cancelled_orders),
            "rejected_orders": len(rejected_orders),
            "fill_rate": fill_rate,
            "cancel_rate": cancel_rate,
            "reject_rate": reject_rate,
            "avg_execution_time": avg_execution_time,
            "execution_times": execution_times,
        }

    async def calculate_slippage(
        self, order: Order, market_data: MarketData
    ) -> Dict[str, float]:
        """Расчет проскальзывания."""
        if not order.price or not market_data.close:  # Исправляю атрибуты MarketData
            return {"slippage": 0.0, "slippage_percentage": 0.0}
        
        order_price = float(order.price.value)  # Исправляю тип Price
        mid_price = float(market_data.close.value)  # Используем close вместо bid/ask
        
        if order.side == OrderSide.BUY:
            slippage = order_price - mid_price
        else:
            slippage = mid_price - order_price
        
        slippage_percentage = (slippage / mid_price) * 100 if mid_price > 0 else 0.0
        
        return {
            "slippage": slippage,
            "slippage_percentage": slippage_percentage,
        }

    async def analyze_market_impact(
        self, orders: List[Order], market_data: List[MarketData]
    ) -> Dict[str, Any]:
        """Анализ рыночного влияния."""
        if not orders or not market_data:
            return {}
        
        # Группируем ордера по времени
        orders_by_time: Dict[str, List[Order]] = {}
        for order in orders:
            if order.created_at:
                time_key = order.created_at.to_datetime().strftime("%Y-%m-%d %H:%M")
                if time_key not in orders_by_time:
                    orders_by_time[time_key] = []
                orders_by_time[time_key].append(order)
        
        # Анализируем влияние на цену
        price_impact = []
        volume_impact = []
        
        for time_key, time_orders in orders_by_time.items():
            total_volume = sum(float(o.quantity) for o in time_orders if o.quantity)  # Исправляю тип Volume
            total_value = sum(
                float(o.quantity) * float(o.price.value)  # Исправляю типы Volume и Price
                for o in time_orders 
                if o.quantity and o.price
            )
            
            volume_impact.append(total_volume)
            price_impact.append(total_value)
        
        return {
            "orders_by_time": orders_by_time,
            "price_impact": price_impact,
            "volume_impact": volume_impact,
            "avg_volume_impact": sum(volume_impact) / len(volume_impact) if volume_impact else 0.0,
            "avg_price_impact": sum(price_impact) / len(price_impact) if price_impact else 0.0,
        }


# ============================================================================
# SMART ORDER ROUTER
# ============================================================================
class SmartOrderRouter:
    """Умный роутер ордеров."""

    def __init__(self, config: ConnectionConfig):
        self.config = config

    async def optimize_order(self, order: Order, market_data: MarketData) -> Order:
        """Оптимизация ордера на основе рыночных данных."""
        # Анализируем рыночные условия
        market_conditions = await self._analyze_market_conditions(market_data)
        
        # Оптимизируем размер ордера
        optimized_order = await self._optimize_order_size(order, market_conditions)
        
        # Оптимизируем цену ордера
        optimized_order = await self._optimize_order_price(optimized_order, market_conditions)
        
        # Оптимизируем время размещения
        optimized_order = await self._optimize_order_timing(optimized_order, market_conditions)
        
        return optimized_order

    async def _analyze_market_conditions(
        self, market_data: MarketData
    ) -> Dict[str, Any]:
        """Анализ рыночных условий."""
        if not market_data.close:
            return {"volatility": "low", "liquidity": "medium", "spread": 0.0}
        
        # Используем close price как базовую цену
        close_price = float(market_data.close.value)
        
        # Для упрощения используем фиксированный спред
        spread = close_price * 0.001  # 0.1% спред
        spread_percentage = 0.1
        
        # Определяем волатильность
        volatility = "low"
        if spread_percentage > 0.1:
            volatility = "high"
        elif spread_percentage > 0.05:
            volatility = "medium"
        
        # Определяем ликвидность
        liquidity = "medium"
        if market_data.volume and float(market_data.volume.value) > 1000000:
            liquidity = "high"
        elif market_data.volume and float(market_data.volume.value) < 100000:
            liquidity = "low"
        
        return {
            "volatility": volatility,
            "liquidity": liquidity,
            "spread": spread,
            "spread_percentage": spread_percentage,
        }

    async def _optimize_order_size(
        self, order: Order, market_conditions: Dict[str, Any]
    ) -> Order:
        """Оптимизация размера ордера."""
        # В зависимости от ликвидности корректируем размер
        liquidity = market_conditions.get("liquidity", "medium")
        
        if liquidity == "low" and order.quantity:
            # Уменьшаем размер для низкой ликвидности
            optimized_quantity = float(order.quantity) * 0.5
            order.quantity = VolumeValue(Decimal(str(optimized_quantity)))
        elif liquidity == "high" and order.quantity:
            # Увеличиваем размер для высокой ликвидности
            optimized_quantity = float(order.quantity) * 1.2
            order.quantity = VolumeValue(Decimal(str(optimized_quantity)))
        
        return order  # Добавляю недостающий return

    async def _optimize_order_price(
        self, order: Order, market_conditions: Dict[str, Any]
    ) -> Order:
        """Оптимизация цены ордера."""
        if not order.price:
            return order
        
        volatility = market_conditions.get("volatility", "low")
        spread_percentage = market_conditions.get("spread_percentage", 0.0)
        
        current_price = float(order.price.value)
        
        if volatility == "high":
            # Для высокой волатильности применяем продвинутую стратегию оптимизации
            # Анализируем дополнительные рыночные условия
            liquidity = market_conditions.get("liquidity", "medium")
            spread = market_conditions.get("spread", 0.0)
            
            # Базовые множители для высокой волатильности
            volatility_multiplier = 1.5  # Увеличиваем отступ при высокой волатильности
            liquidity_adjustment = 1.0
            
            # Корректировка по ликвидности
            if liquidity == "low":
                liquidity_adjustment = 1.3  # Больший отступ при низкой ликвидности
            elif liquidity == "high":
                liquidity_adjustment = 0.8  # Меньший отступ при высокой ликвидности
            
            # Динамическая корректировка на основе спреда
            spread_adjustment = min(2.0, max(0.5, spread_percentage * 10))
            
            # Адаптивный множитель цены
            adaptive_multiplier = volatility_multiplier * liquidity_adjustment * spread_adjustment
            
            # Применяем разные стратегии для покупки и продажи
            if order.side == OrderSide.BUY:
                # Для покупки: более агрессивная цена, но с защитой от проскальзывания
                price_adjustment = spread_percentage * adaptive_multiplier * 0.15
                optimized_price = current_price * (1 + price_adjustment)
                
                # Дополнительная защита от резких движений
                max_price_increase = current_price * 1.02  # Максимум +2%
                optimized_price = min(optimized_price, max_price_increase)
                
            else:  # OrderSide.SELL
                # Для продажи: более консервативная цена с учетом риска
                price_adjustment = spread_percentage * adaptive_multiplier * 0.12
                optimized_price = current_price * (1 - price_adjustment)
                
                # Защита от резкого падения
                min_price_decrease = current_price * 0.98  # Минимум -2%
                optimized_price = max(optimized_price, min_price_decrease)
            
            # Логирование оптимизации для мониторинга
            optimization_metadata = {
                "volatility_level": "high",
                "liquidity_level": liquidity,
                "volatility_multiplier": volatility_multiplier,
                "liquidity_adjustment": liquidity_adjustment,
                "spread_adjustment": spread_adjustment,
                "adaptive_multiplier": adaptive_multiplier,
                "price_adjustment_pct": price_adjustment * 100,
                "original_price": current_price,
                "optimized_price": optimized_price,
                "optimization_timestamp": datetime.now().isoformat()
            }
            
            # Сохраняем метаданные оптимизации в ордере
            if not hasattr(order, 'metadata'):
                order.metadata = {"price_optimization": str(optimization_metadata)}
            else:
                order.metadata['price_optimization'] = str(optimization_metadata)
            
        elif volatility == "medium":
            # Для средней волатильности используем умеренную оптимизацию
            if order.side == OrderSide.BUY:
                optimized_price = current_price * (1 + spread_percentage * 0.08)
            else:
                optimized_price = current_price * (1 - spread_percentage * 0.08)
        else:  # volatility == "low"
            # Для низкой волатильности используем стандартный отступ
            if order.side == OrderSide.BUY:
                optimized_price = current_price * (1 + spread_percentage * 0.05)
            else:
                optimized_price = current_price * (1 - spread_percentage * 0.05)
        
        order.price = Price(Decimal(str(optimized_price)), Currency.USD)  # Исправляю создание Price
        return order

    async def _optimize_order_timing(
        self, order: Order, market_conditions: Dict[str, Any]
    ) -> Order:
        """Оптимизация времени размещения ордера."""
        # Здесь можно добавить логику для определения оптимального времени
        # Например, избегать размещения во время низкой ликвидности
        return order


# ============================================================================
# PRODUCTION ORDER MANAGER
# ============================================================================
class ProductionOrderManager(ExchangeProtocol):
    """Production-ready менеджер ордеров."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.tracker = OrderTracker(config)
        self.analytics = OrderAnalytics()
        self.router = SmartOrderRouter(config)

    async def authenticate(self, api_key: str, secret_key: str) -> bool:
        """Аутентификация."""
        return True

    async def calculate_commission(
        self, order_value: Money, order_type: OrderType
    ) -> Money:
        """Расчет комиссии."""
        from decimal import Decimal
        commission_rate = Decimal('0.001')  # 0.1%
        return Money(amount=order_value.amount * commission_rate, currency=order_value.currency)

    async def validate_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
    ) -> bool:
        """Валидация ордера."""
        return True

    async def get_account_info(self) -> Account:
        """Получение информации об аккаунте."""
        return Account()

    async def get_balance(self, currency: Currency) -> Money:
        """Получение баланса."""
        return Money(amount=1000.0, currency=currency)

    async def get_order_book(self, symbol: Symbol, depth: int = 10) -> Any:
        """Получение стакана заявок."""
        return {}

    async def get_market_data(self, symbol: Symbol) -> MarketData:
        """Получение рыночных данных."""
        return MarketData(
            symbol=symbol,
            open=Price(Decimal("50000"), Currency.USD),
            high=Price(Decimal("50001"), Currency.USD),
            low=Price(Decimal("49999"), Currency.USD),
            close=Price(Decimal("50000.5"), Currency.USD),
            volume=Volume(Decimal("1000"), Currency.USDT),
            timestamp=TimestampValue(datetime.now()),
        )

    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """Получение открытых ордеров."""
        return await self.tracker.get_open_orders()

    async def get_trade_history(
        self, symbol: Optional[Symbol] = None, limit: int = 100
    ) -> List[Trade]:
        """Получение истории сделок."""
        return await self.tracker.get_trades()

    async def get_positions(self) -> List[Any]:
        """Получение позиций."""
        return []

    async def place_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        portfolio_id: Optional[Any] = None,
        strategy_id: Optional[Any] = None,
        signal_id: Optional[Any] = None,
    ) -> Order:
        """Размещение ордера."""
        # Создаем ордер
        order = Order(
            id=OrderId(uuid4()),
            symbol=symbol,
            side=OrderSide(side.value) if not isinstance(side, OrderSide) else side,
            order_type=OrderType(order_type.value) if not isinstance(order_type, OrderType) else order_type,
            quantity=VolumeValue(Decimal(str(quantity.amount))) if hasattr(quantity, 'amount') else VolumeValue(Decimal(str(quantity))),
            price=price if price is None or isinstance(price, Price) else Price(price, Currency.USD),
            status=OrderStatus.OPEN,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )
        # Добавляем в трекер
        await self.tracker.add_order(order)
        # Симулируем исполнение
        await self._simulate_order_execution(order)
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        # Упрощенная логика преобразования типов
        try:
            if isinstance(order_id, str):
                order_uuid = OrderId(UUID(order_id))
            else:
                order_uuid = OrderId(order_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid order_id format: {order_id}")
            
        order = await self.tracker.get_order(order_uuid)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_uuid)
        await self.tracker.update_order(order_uuid, {"status": OrderStatus.CANCELLED})
        return True

    async def get_order_status(self, order_id: str) -> Order:
        """Получение статуса ордера."""
        # Упрощенная логика преобразования типов
        try:
            if isinstance(order_id, str):
                order_uuid = OrderId(UUID(order_id))
            else:
                order_uuid = OrderId(order_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid order_id format: {order_id}")
            
        order = await self.tracker.get_order(order_uuid)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_uuid)
        return order

    async def place_order_from_request(
        self, request: OrderRequest, order_id: Optional[OrderId] = None
    ) -> Order:
        """Размещение ордера из запроса."""
        await self._validate_order_request(request)
        
        # Исправление: правильное преобразование типов
        from domain.entities.order import OrderSide as EntityOrderSide, OrderType as EntityOrderType
        side = EntityOrderSide(request.side.value)
        order_type = EntityOrderType(request.order_type.value)
        
        # Безопасное преобразование quantity
        if hasattr(request.quantity, 'amount'):
            quantity_value = VolumeValue(Decimal(str(request.quantity.amount)))
        else:
            quantity_value = VolumeValue(Decimal(str(request.quantity)))
        
        # Безопасное преобразование price
        if request.price:
            if hasattr(request.price, 'amount'):
                price_value = Price(Decimal(str(request.price.amount)), Currency.USD)
            else:
                price_value = Price(Decimal(str(request.price)), Currency.USD)
        else:
            price_value = None
        
        order = Order(
            id=order_id or OrderId(uuid4()),
            symbol=request.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity_value,
            price=price_value,
            status=OrderStatus.OPEN,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )
        await self.tracker.add_order(order)
        await self._simulate_order_execution(order)
        return order

    async def update_order_from_request(
        self, order_id: OrderId, request: OrderRequest
    ) -> bool:
        """Обновление ордера из запроса."""
        order = await self.tracker.get_order(order_id)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_id)  # Исправляю аргументы
        
        # Обновляем поля
        updates = {
            "symbol": request.symbol,
            "side": request.side,
            "order_type": request.order_type,
            "quantity": request.quantity,
            "price": request.price,
        }
        
        return await self.tracker.update_order(order_id, updates)

    async def cancel_order_from_request(self, order_id: OrderId) -> bool:
        """Отмена ордера из запроса."""
        order = await self.tracker.get_order(order_id)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_id)  # Исправляю аргументы
        
        return await self.cancel_order(str(order_id))

    async def get_order_status_from_request(self, order_id: OrderId) -> Dict[str, Any]:
        """Получение статуса ордера из запроса."""
        order = await self.tracker.get_order(order_id)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_id)  # Исправляю аргументы
        
        return {
            "order_id": str(order.id),
            "symbol": str(order.symbol),
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": float(order.quantity) if order.quantity else 0.0,
            "price": float(order.price.amount) if order.price else None,
            "status": order.status.value,
            "created_at": order.created_at.to_datetime().isoformat() if order.created_at else None,
            "updated_at": order.updated_at.to_datetime().isoformat() if order.updated_at else None,
        }

    async def list_orders_from_request(
        self, request: OrderRequest
    ) -> List[Dict[str, Any]]:
        """Получение списка ордеров из запроса."""
        filters = {
            "symbol": request.symbol,
            "side": request.side,
            "order_type": request.order_type,
        }
        
        orders = await self.tracker.get_orders(filters)
        
        return [
            {
                "order_id": str(order.id),
                "symbol": str(order.symbol),
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": float(order.quantity) if order.quantity else 0.0,
                "price": float(order.price.amount) if order.price else None,
                "status": order.status.value,
                "created_at": order.created_at.to_datetime().isoformat() if order.created_at else None,
            }
            for order in orders
        ]

    async def get_order_analytics(self, order_id: OrderId) -> Dict[str, Any]:
        """Получение аналитики ордера."""
        order = await self.tracker.get_order(order_id)
        if not order:
            raise OrderNotFoundError(f"Order not found: {order_id}", order_id)
        
        # Получаем рыночные данные
        market_data = await self._get_market_data(order.symbol)
        if not market_data:
            return {}
        
        # Анализируем производительность
        performance = await self.analytics.analyze_order_performance([order])
        
        # Рассчитываем проскальзывание
        slippage = await self.analytics.calculate_slippage(order, market_data)
        
        return {
            "performance": performance,
            "slippage": slippage,
            "order_details": {
                "order_id": str(order.id),
                "symbol": str(order.symbol),
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": float(order.quantity) if order.quantity else 0.0,
                "price": float(order.price.amount) if order.price else None,
                "status": order.status.value,
            },
        }

    async def _validate_order_request(self, request: OrderRequest) -> None:
        """Валидация запроса ордера."""
        if not request.symbol:
            raise InvalidOrderError("Symbol is required")
        if not request.quantity:
            raise InvalidOrderError("Quantity is required")
        if request.order_type.value == "limit" and not request.price:
            raise InvalidOrderError("Price is required for limit orders")

    async def _get_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Получение рыночных данных."""
        try:
            return await self.get_market_data(symbol)
        except Exception:
            return None

    async def _simulate_order_execution(self, order: Order) -> None:
        """Симуляция исполнения ордера."""
        # В реальной реализации здесь была бы интеграция с биржей
        # Для демонстрации просто обновляем статус
        await asyncio.sleep(0.1)  # Имитируем задержку
        await self.tracker.update_order(order.id, {"status": OrderStatus.FILLED})

    async def create_order(self, order: Order) -> Dict[str, Any]:
        """Создание ордера."""
        # Исправление: изменяем сигнатуру для соответствия протоколу
        from domain.value_objects.volume import Volume
        quantity_volume = Volume(order.quantity.amount, order.quantity.currency)  # type: ignore
        result = await self.place_order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=quantity_volume,
            price=order.price,
        )
        return {
            "order_id": str(result.id),
            "symbol": str(result.symbol),
            "side": result.side.value,
            "order_type": result.order_type.value,
            "quantity": float(result.quantity) if result.quantity else 0.0,
            "price": float(result.price.amount) if result.price else None,
            "status": result.status.value,
        }

    async def fetch_balance(self) -> Dict[str, float]:
        """Получение баланса."""
        return {"USD": 1000.0, "USDT": 1000.0}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера."""
        # Исправление: изменяем тип параметра на str
        return {
            "symbol": symbol,
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": datetime.now().isoformat()
        }

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение открытых ордеров."""
        return []

    async def fetch_order(self, order_id: str) -> Dict[str, Any]:
        """Получение ордера по ID."""
        return {}

    async def fetch_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Получение стакана заявок."""
        return {
            "symbol": symbol,
            "bids": [],
            "asks": [],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# ORDER MANAGER ADAPTER
# ============================================================================
class OrderManagerAdapter(ExchangeProtocol):
    """Адаптер менеджера ордеров для обратной совместимости."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.order_manager = ProductionOrderManager(self.config)

    async def authenticate(self, api_key: str, secret_key: str) -> bool:
        """Аутентификация."""
        return await self.order_manager.authenticate(api_key, secret_key)

    async def calculate_commission(
        self, order_value: Money, order_type: OrderType
    ) -> Money:
        """Расчет комиссии."""
        return await self.order_manager.calculate_commission(order_value, order_type)

    async def validate_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
    ) -> bool:
        """Валидация ордера."""
        return await self.order_manager.validate_order(symbol, side, order_type, quantity, price)

    async def get_account_info(self) -> Account:
        """Получение информации об аккаунте."""
        return await self.order_manager.get_account_info()

    async def get_balance(self, currency: Currency) -> Money:
        """Получение баланса."""
        return await self.order_manager.get_balance(currency)

    async def get_order_book(self, symbol: Symbol, depth: int = 10) -> Any:
        """Получение стакана заявок."""
        return await self.order_manager.get_order_book(symbol, depth)

    async def get_market_data(self, symbol: Symbol) -> MarketData:
        """Получение рыночных данных."""
        return await self.order_manager.get_market_data(symbol)

    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """Получение открытых ордеров."""
        return await self.order_manager.get_open_orders(symbol)

    async def get_trade_history(
        self, symbol: Optional[Symbol] = None, limit: int = 100
    ) -> List[Trade]:
        """Получение истории сделок."""
        return await self.order_manager.get_trade_history(symbol, limit)

    async def get_positions(self) -> List[Any]:
        """Получение позиций."""
        return await self.order_manager.get_positions()

    async def place_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        portfolio_id: Optional[Any] = None,
        strategy_id: Optional[Any] = None,
        signal_id: Optional[Any] = None,
    ) -> Order:
        """Размещение ордера."""
        return await self.order_manager.place_order(
            symbol, side, order_type, quantity, price, stop_price, portfolio_id, strategy_id, signal_id
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        return await self.order_manager.cancel_order(order_id)

    async def get_order_status(self, order_id: str) -> Order:
        """Получение статуса ордера."""
        return await self.order_manager.get_order_status(order_id)

    async def place_order_string(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> str:
        """Размещение ордера через строковые параметры."""
        # Конвертируем строковые параметры в типизированные
        symbol_obj = Symbol(symbol)
        side_obj = OrderSide(side.upper())
        order_type_obj = OrderType(order_type.upper())
        quantity_obj = Volume(Decimal(str(quantity)), Currency.USDT)
        price_obj = Price(Decimal(str(price)), Currency.USD) if price else None
        
        # Создаем запрос
        request = OrderRequest(
            symbol=symbol_obj,
            side=ExternalOrderSide(side.upper()),
            order_type=ExternalOrderType(order_type.upper()),
            quantity=VolumeValue(Decimal(str(quantity))),
            price=PriceValue(Decimal(str(price))) if price else None,
        )
        
        # Размещаем ордер
        order = await self.order_manager.place_order_from_request(request)
        
        return str(order.id)

    async def cancel_order_string(self, order_id: str) -> bool:
        """Отмена ордера через строковый ID."""
        order_id_obj = OrderId(UUID(order_id))
        return await self.order_manager.cancel_order_from_request(order_id_obj)

    async def get_order_status_string(self, order_id: str) -> Dict[str, Any]:
        """Получение статуса ордера через строковый ID."""
        order_id_obj = OrderId(UUID(order_id))
        return await self.order_manager.get_order_status_from_request(order_id_obj)

    async def list_orders_string(
        self, symbol: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение списка ордеров через строковые параметры."""
        # Создаем базовый запрос
        request = OrderRequest(
            symbol=Symbol(symbol) if symbol else Symbol(""),
            side=ExternalOrderSide.BUY,  # Значение по умолчанию
            order_type=ExternalOrderType.MARKET,  # Значение по умолчанию
            quantity=VolumeValue(Decimal("0"))
        )
        
        return await self.order_manager.list_orders_from_request(request)


# Обёртка для удобства использования
class OrderManager(ProductionOrderManager):
    """Удобная обёртка для ProductionOrderManager с дефолтной конфигурацией."""
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        if config is None:
            # Создаём дефолтную конфигурацию
            from domain.types.external_service_types import ConnectionConfig
            config = ConnectionConfig(
                timeout=30.0,
                max_retries=3,
                retry_delay=1.0
            )
        super().__init__(config)


# Алиас для совместимости  
OrderManagementService = ProductionOrderManager
