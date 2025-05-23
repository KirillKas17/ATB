from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from ..models import Order
from ..order_utils import is_valid_order


class OrderController:
    """Контроллер для управления ордерами"""

    def __init__(self, exchange, config: Dict):
        self.exchange = exchange
        self.config = config
        self.active_orders: Dict[str, Order] = {}

    async def place_order(self, order: Order) -> Order:
        """
        Размещение ордера.

        Args:
            order: Ордер для размещения

        Returns:
            Order: Размещенный ордер
        """
        try:
            # Безопасное получение цены
            price = 0.0
            if order.price is not None:
                try:
                    price = float(order.price)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid price value for order {order.id}")

            # Безопасное получение размера
            size = 0.0
            if order.size is not None:
                try:
                    size = float(order.size)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid size value for order {order.id}")

            result = await self.exchange.create_order(
                symbol=order.pair,
                type=order.type,
                side=order.side,
                amount=size,
                price=price,
            )

            if result is None:
                raise ValueError("Failed to place order")

            # Безопасное получение filled_price и filled_size
            filled_price = None
            filled_size = None
            if result.get("average") is not None:
                try:
                    filled_price = float(result["average"])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid average value for order {result.get('id')}"
                    )
            if result.get("filled") is not None:
                try:
                    filled_size = float(result["filled"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid filled value for order {result.get('id')}")

            placed_order = Order(
                id=str(result.get("id", "")),
                pair=str(result.get("symbol", "")),
                type=str(result.get("type", "market")),
                side=str(result.get("side", "buy")),
                price=price,
                size=size,
                status=str(result.get("status", "new")),
                timestamp=(
                    datetime.fromtimestamp(result.get("timestamp", 0) / 1000)
                    if result.get("timestamp")
                    else datetime.now()
                ),
                filled_price=filled_price,
                filled_size=filled_size,
                filled_timestamp=(
                    datetime.fromtimestamp(result.get("lastTradeTimestamp", 0) / 1000)
                    if result.get("lastTradeTimestamp")
                    else None
                ),
            )

            self.active_orders[placed_order.id] = placed_order
            logger.info(f"Order placed: {placed_order.id}")

            return placed_order

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    async def cancel_order(self, order_id: str) -> None:
        """
        Отмена ордера.

        Args:
            order_id: ID ордера
        """
        try:
            await self.exchange.cancel_order(order_id)
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            logger.info(f"Order canceled: {order_id}")

        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            raise

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Получение ордера.

        Args:
            order_id: ID ордера

        Returns:
            Optional[Order]: Ордер или None
        """
        try:
            result = await self.exchange.fetch_order(order_id)
            if result is None:
                return None

            # Безопасное получение цены
            price = 0.0
            if result.get("price") is not None:
                try:
                    price = float(result["price"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid price value for order {order_id}")

            # Безопасное получение filled_price и filled_size
            filled_price = None
            filled_size = None
            if result.get("average") is not None:
                try:
                    filled_price = float(result["average"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid average value for order {order_id}")
            if result.get("filled") is not None:
                try:
                    filled_size = float(result["filled"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid filled value for order {order_id}")

            # Безопасное получение размера
            size = 0.0
            if result.get("amount") is not None:
                try:
                    size = float(result["amount"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid amount value for order {order_id}")

            order = Order(
                id=str(result.get("id", "")),
                pair=str(result.get("symbol", "")),
                type=str(result.get("type", "market")),
                side=str(result.get("side", "buy")),
                price=price,
                size=size,
                status=str(result.get("status", "new")),
                timestamp=(
                    datetime.fromtimestamp(result.get("timestamp", 0) / 1000)
                    if result.get("timestamp")
                    else datetime.now()
                ),
                filled_price=filled_price,
                filled_size=filled_size,
                filled_timestamp=(
                    datetime.fromtimestamp(result.get("lastTradeTimestamp", 0) / 1000)
                    if result.get("lastTradeTimestamp")
                    else None
                ),
            )

            return order

        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            return None

    async def get_open_orders(self) -> List[Order]:
        """
        Получение открытых ордеров.

        Returns:
            List[Order]: Список открытых ордеров
        """
        try:
            orders = await self.exchange.fetch_open_orders()
            result = []

            for order_data in orders:
                # Безопасное получение цены
                price = 0.0
                if order_data.get("price") is not None:
                    try:
                        price = float(order_data["price"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid price value for order {order_data.get('id')}"
                        )

                # Безопасное получение размера
                size = 0.0
                if order_data.get("amount") is not None:
                    try:
                        size = float(order_data["amount"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid amount value for order {order_data.get('id')}"
                        )

                # Безопасное получение filled_price и filled_size
                filled_price = None
                filled_size = None
                if order_data.get("average") is not None:
                    try:
                        filled_price = float(order_data["average"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid average value for order {order_data.get('id')}"
                        )
                if order_data.get("filled") is not None:
                    try:
                        filled_size = float(order_data["filled"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid filled value for order {order_data.get('id')}"
                        )

                order = Order(
                    id=str(order_data.get("id", "")),
                    pair=str(order_data.get("symbol", "")),
                    type=str(order_data.get("type", "market")),
                    side=str(order_data.get("side", "buy")),
                    price=price,
                    size=size,
                    status=str(order_data.get("status", "new")),
                    timestamp=(
                        datetime.fromtimestamp(order_data.get("timestamp", 0) / 1000)
                        if order_data.get("timestamp")
                        else datetime.now()
                    ),
                    filled_price=filled_price,
                    filled_size=filled_size,
                    filled_timestamp=(
                        datetime.fromtimestamp(
                            order_data.get("lastTradeTimestamp", 0) / 1000
                        )
                        if order_data.get("lastTradeTimestamp")
                        else None
                    ),
                )

                result.append(order)
                self.active_orders[order.id] = order

            return result

        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []

    async def clear_invalid_orders(self) -> None:
        """Очистка невалидных ордеров"""
        try:
            for order_id, order in list(self.active_orders.items()):
                if not is_valid_order(order.to_dict()):
                    await self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error clearing invalid orders: {str(e)}")
            raise
