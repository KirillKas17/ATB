import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .bybit_client import BybitClient


@dataclass
class OrderConfig:
    """Конфигурация ордеров"""

    max_leverage: float = 10.0  # максимальное плечо
    min_leverage: float = 1.0  # минимальное плечо
    confidence_threshold: float = 0.7  # порог уверенности для плеча
    trailing_stop: bool = False  # использование трейлинг-стопа
    trailing_distance: float = 0.02  # дистанция трейлинг-стопа (2%)
    break_even_threshold: float = 0.01  # порог для брейк-ивен (1%)
    take_profit_levels: List[float] = field(default_factory=lambda: [0.02, 0.03, 0.05])
    take_profit_quantities: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    def __post_init__(self):
        if self.take_profit_levels is None:
            self.take_profit_levels = [0.02, 0.03, 0.05]  # 2%, 3%, 5%
        if self.take_profit_quantities is None:
            self.take_profit_quantities = [0.4, 0.3, 0.3]  # 40%, 30%, 30%


@dataclass
class Order:
    """Информация об ордере"""

    id: str
    symbol: str
    type: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: datetime
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    break_even_price: Optional[float] = None
    take_profit_levels: Optional[List[float]] = None
    take_profit_quantities: Optional[List[float]] = None


class OrderManager:
    def __init__(self, client: BybitClient, config: Optional[OrderConfig] = None):
        """
        Инициализация менеджера ордеров.

        Args:
            client (BybitClient): Клиент биржи
            config (OrderConfig): Конфигурация
        """
        self.client = client
        self.config = config or OrderConfig()

        # Инициализация списков ордеров
        self.active_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}

        # Запуск мониторинга
        self.monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Запуск менеджера"""
        try:
            # Запуск мониторинга
            self.monitor_task = asyncio.create_task(self._monitor_orders())

            logger.info("Order manager started")

        except Exception as e:
            logger.error(f"Error starting order manager: {str(e)}")
            raise

    async def stop(self):
        """Остановка менеджера"""
        try:
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            logger.info("Order manager stopped")

        except Exception as e:
            logger.error(f"Error stopping order manager: {str(e)}")
            raise

    async def create_entry_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        leverage: Optional[float] = None,
    ) -> Order:
        """
        Создание входного ордера.

        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            amount: Количество
            price: Цена
            stop_loss: Стоп-лосс
            take_profit: Тейк-профит
            confidence: Уверенность
            leverage: Плечо (опционально)

        Returns:
            Order: Созданный ордер
        """
        try:
            # Расчет плеча
            if leverage is None:
                leverage = self._calculate_leverage(confidence)

            # Установка плеча
            await self._set_leverage(symbol, leverage)

            # Создание ордера
            order = await self.client.create_order(
                symbol=symbol, order_type="limit", side=side, amount=amount, price=price
            )

            # Создание стоп-лосса
            stop_order = await self.client.create_order(
                symbol=symbol,
                order_type="stop",
                side="sell" if side == "buy" else "buy",
                amount=amount,
                price=stop_loss,
                params={"stopPrice": stop_loss},
            )

            # Создание тейк-профита
            take_profit_order = await self.client.create_order(
                symbol=symbol,
                order_type="limit",
                side="sell" if side == "buy" else "buy",
                amount=amount,
                price=take_profit,
            )

            # Создание объекта ордера
            order_obj = Order(
                id=order["id"],
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price,
                status=order["status"],
                timestamp=datetime.fromtimestamp(order["timestamp"] / 1000),
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=self.config.trailing_stop,
                break_even_price=self._calculate_break_even(price, stop_loss),
                take_profit_levels=self.config.take_profit_levels,
                take_profit_quantities=self.config.take_profit_quantities,
            )

            # Сохранение ордера
            self.active_orders[order["id"]] = order_obj

            logger.info(f"Created entry order: {order_obj}")

            return order_obj

        except Exception as e:
            logger.error(f"Error creating entry order: {str(e)}")
            raise

    async def create_take_profit_ladder(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        base_price: float,
        levels: List[float],
        quantities: List[float],
    ) -> List[Order]:
        """
        Создание лестницы тейк-профитов.

        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            total_amount: Общее количество
            base_price: Базовая цена
            levels: Уровни цен
            quantities: Количества для каждого уровня

        Returns:
            List[Order]: Список созданных ордеров
        """
        try:
            if len(levels) != len(quantities):
                raise ValueError("Number of levels must match number of quantities")

            if not all(0 < q <= 1 for q in quantities):
                raise ValueError("All quantities must be between 0 and 1")

            if sum(quantities) != 1:
                raise ValueError("Quantities must sum to 1")

            orders: List[Order] = []
            for level, quantity in zip(levels, quantities):
                amount = total_amount * quantity
                price = base_price * (1 + level if side == "buy" else 1 - level)

                order = await self.client.create_order(
                    symbol=symbol,
                    order_type="limit",
                    side="sell" if side == "buy" else "buy",
                    amount=amount,
                    price=price,
                )

                order_obj = Order(
                    id=order["id"],
                    symbol=symbol,
                    type="limit",
                    side="sell" if side == "buy" else "buy",
                    amount=amount,
                    price=price,
                    status=order["status"],
                    timestamp=datetime.fromtimestamp(order["timestamp"] / 1000),
                    leverage=1.0,  # для тейк-профитов плечо не используется
                )

                self.active_orders[order["id"]] = order_obj
                orders.append(order_obj)

            logger.info(f"Created take profit ladder: {orders}")

            return orders

        except Exception as e:
            logger.error(f"Error creating take profit ladder: {str(e)}")
            raise

    async def update_trailing_stop(self, order_id: str, current_price: float):
        """
        Обновление трейлинг-стопа.

        Args:
            order_id: ID ордера
            current_price: Текущая цена
        """
        try:
            order = self.active_orders.get(order_id)
            if not order or not order.trailing_stop or order.stop_loss is None:
                return

            # Расчет нового стоп-лосса
            if order.side == "buy":
                new_stop = current_price * (1 - self.config.trailing_distance)
                if new_stop > order.stop_loss:
                    await self._update_stop_loss(order, new_stop)
            else:
                new_stop = current_price * (1 + self.config.trailing_distance)
                if new_stop < order.stop_loss:
                    await self._update_stop_loss(order, new_stop)

        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
            raise

    async def check_break_even(self, order_id: str, current_price: float) -> None:
        """Проверка достижения брейк-ивен"""
        try:
            order = self.active_orders.get(order_id)
            if not order or not order.break_even_price:
                return

            break_even_price = float(order.break_even_price)
            if order.side == "buy" and current_price > break_even_price:
                await self._move_stop_to_break_even(order)
            elif order.side == "sell" and current_price < break_even_price:
                await self._move_stop_to_break_even(order)

        except Exception as e:
            logger.error(f"Error checking break even: {str(e)}")
            raise

    async def cancel_order(self, order_id: str):
        """
        Отмена ордера.

        Args:
            order_id: ID ордера
        """
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return

            # Отмена ордера
            await self.client.cancel_order(order_id, order.symbol)

            # Обновление статуса
            order.status = "cancelled"
            self.closed_orders[order_id] = order
            del self.active_orders[order_id]

            logger.info(f"Cancelled order: {order}")

        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise

    async def _monitor_orders(self):
        """Мониторинг ордеров"""
        try:
            while True:
                for order_id, order in list(self.active_orders.items()):
                    try:
                        # Получение текущего состояния
                        current_order = await self.client.get_order(
                            order_id, order.symbol
                        )

                        # Обновление статуса
                        if current_order["status"] != order.status:
                            order.status = current_order["status"]

                            if order.status in ["closed", "cancelled"]:
                                self.closed_orders[order_id] = order
                                del self.active_orders[order_id]

                        # Проверка трейлинг-стопа
                        if order.trailing_stop:
                            current_price = float(current_order["price"])
                            await self.update_trailing_stop(order_id, current_price)

                        # Проверка брейк-ивен
                        if order.break_even_price:
                            current_price = float(current_order["price"])
                            await self.check_break_even(order_id, current_price)

                    except Exception as e:
                        logger.error(f"Error monitoring order {order_id}: {str(e)}")

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in order monitor: {str(e)}")
            raise

    def _calculate_leverage(self, confidence: float) -> float:
        """Расчет плеча на основе уверенности"""
        try:
            if confidence < self.config.confidence_threshold:
                return self.config.min_leverage

            # Линейное масштабирование
            leverage = self.config.min_leverage + (
                self.config.max_leverage - self.config.min_leverage
            ) * (confidence - self.config.confidence_threshold) / (
                1 - self.config.confidence_threshold
            )

            return min(leverage, self.config.max_leverage)

        except Exception as e:
            logger.error(f"Error calculating leverage: {str(e)}")
            raise

    async def _set_leverage(self, symbol: str, leverage: float):
        """Установка плеча"""
        try:
            await self.client.set_leverage(symbol=symbol, leverage=int(leverage))
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            raise

    def _calculate_break_even(self, entry_price: float, stop_loss: float) -> float:
        """Расчет цены брейк-ивен"""
        try:
            return (
                entry_price
                + (entry_price - stop_loss) * self.config.break_even_threshold
            )

        except Exception as e:
            logger.error(f"Error calculating break even: {str(e)}")
            raise

    async def _update_stop_loss(self, order: Order, new_stop: Optional[float]) -> None:
        """Обновление стоп-лосса"""
        try:
            if new_stop is None:
                return

            await self.client.cancel_order(symbol=order.symbol, order_id=order.id)

            stop_order = await self.client.create_order(
                symbol=order.symbol,
                order_type="stop",
                side="sell" if order.side == "buy" else "buy",
                amount=order.amount,
                price=new_stop,
                params={"stopPrice": new_stop},
            )

            order.stop_loss = new_stop
            logger.info(f"Updated stop loss for order {order.id} to {new_stop}")

        except Exception as e:
            logger.error(f"Error updating stop loss: {str(e)}")
            raise

    async def _move_stop_to_break_even(self, order: Order):
        """Перемещение стопа в брейк-ивен"""
        try:
            await self._update_stop_loss(order, order.break_even_price)

            logger.info(f"Moved stop to break even for order {order.id}")

        except Exception as e:
            logger.error(f"Error moving stop to break even: {str(e)}")
            raise
