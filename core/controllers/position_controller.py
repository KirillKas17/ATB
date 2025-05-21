import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logger import Logger

from ..models import Order, Position
from .base import BaseController

logger = Logger()


class PositionController:
    """Контроллер для управления позициями"""

    def __init__(self, exchange, order_controller):
        self.exchange = exchange
        self.order_controller = order_controller
        self.positions: Dict[str, Position] = {}

    async def open_position(self, position: Position) -> Position:
        """
        Открытие позиции.

        Args:
            position: Позиция для открытия

        Returns:
            Position: Открытая позиция
        """
        try:
            # Создание ордера
            order = Order(
                id=str(uuid.uuid4()),  # Генерируем уникальный ID
                pair=position.pair,
                type="market",
                side="buy" if position.side == "long" else "sell",
                price=position.entry_price,
                size=position.size,
                status="new",
                timestamp=datetime.now(),
            )

            # Размещение ордера
            placed_order = await self.order_controller.place_order(order)

            # Обновление позиции
            position.entry_time = datetime.now()
            position.timestamp = datetime.now()
            self.positions[position.pair] = position

            logger.info(f"Position opened: {position.pair}")
            return position

        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            raise

    async def close_position(self, pair: str) -> None:
        """
        Закрытие позиции.

        Args:
            pair: Торговая пара
        """
        try:
            if pair not in self.positions:
                return

            position = self.positions[pair]

            # Создание ордера
            order = Order(
                id=str(uuid.uuid4()),  # Генерируем уникальный ID
                pair=position.pair,
                type="market",
                side="sell" if position.side == "long" else "buy",
                price=position.current_price,
                size=position.size,
                status="new",
                timestamp=datetime.now(),
            )

            # Размещение ордера
            result = await self.order_controller.place_order(order)

            # Удаление позиции только если ордер успешно размещен
            if result and result.status == "closed":
                del self.positions[pair]
                logger.info(f"Position closed: {pair}")

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise

    async def update_positions(self, leverage: float = 1.0) -> None:
        """Обновление позиций"""
        try:
            positions = await self.exchange.fetch_positions()

            for pos in positions:
                position = Position(
                    pair=pos["symbol"],
                    side=pos["side"],
                    size=float(pos["contracts"]),
                    entry_price=float(pos["entryPrice"]),
                    current_price=float(pos["markPrice"]),
                    pnl=float(pos["unrealizedPnl"]),
                    entry_time=datetime.fromtimestamp(pos["timestamp"] / 1000),
                )
                self.positions[position.pair] = position

            logger.info("Positions updated")

        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            raise

    def get_position(self, pair: str) -> Optional[Position]:
        """
        Получение позиции.

        Args:
            pair: Торговая пара

        Returns:
            Optional[Position]: Позиция или None
        """
        return self.positions.get(pair)

    def get_all_positions(self) -> List[Position]:
        """
        Получение всех позиций.

        Returns:
            List[Position]: Список позиций
        """
        return list(self.positions.values())

    async def get_positions(self, symbol: str) -> List[Position]:
        """
        Получение позиций по символу.

        Args:
            symbol: Торговая пара

        Returns:
            List[Position]: Список позиций
        """
        return [pos for pos in self.positions.values() if pos.pair == symbol]
