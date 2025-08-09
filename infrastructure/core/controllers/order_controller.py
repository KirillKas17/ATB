"""
Контроллер для управления ордерами.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from infrastructure.core.exchange import Exchange
from domain.entities.order import Order

logger = logging.getLogger(__name__)


class OrderController:
    """Контроллер для управления ордерами."""

    def __init__(self, exchange: Exchange, config: Dict[str, Any]) -> None:
        """Инициализация контроллера.
        
        Args:
            exchange: Экземпляр биржи
            config: Конфигурация контроллера
        """
        self.exchange = exchange
        self.config = config

    async def create_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        amount: float, 
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Создание ордера.
        
        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            order_type: Тип ордера (market/limit)
            amount: Количество
            price: Цена (для лимитных ордеров)
            
        Returns:
            Информация о созданном ордере
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            if order:
                if isinstance(order, Order):
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "amount": order.amount,
                        "filled": order.filled_amount,
                        "price": order.price,
                        "status": order.status,
                        "timestamp": order.created_at.isoformat() if hasattr(order, 'created_at') and hasattr(order.created_at, 'isoformat') else None
                    }
                elif isinstance(order, dict):
                    return {
                        "id": order.get('id', None),  # type: ignore[attr-defined]
                        "symbol": order.get('symbol', None),
                        "side": order.get('side', None),
                        "type": order.get('type', None),
                        "amount": order.get('amount', None),
                        "filled": order.get('filled', None),
                        "price": order.get('price', None),
                        "status": order.get('status', None),
                        "timestamp": order.get('timestamp', None)
                    }
            return None
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Отмена ордера.
        
        Args:
            order_id: ID ордера
            symbol: Торговая пара
            
        Returns:
            True если ордер отменен успешно
        """
        try:
            return self.exchange.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение информации об ордере.
        
        Args:
            order_id: ID ордера
            symbol: Торговая пара
            
        Returns:
            Информация об ордере
        """
        try:
            order = self.exchange.get_order(order_id)
            if order:
                if isinstance(order, Order):
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "amount": order.amount,
                        "filled": order.filled_amount,
                        "price": order.price,
                        "status": order.status,
                        "timestamp": order.created_at.isoformat() if hasattr(order, 'created_at') and hasattr(order.created_at, 'isoformat') else None
                    }
                elif isinstance(order, dict):
                    return {
                        "id": order.get('id', None),
                        "symbol": order.get('symbol', None),
                        "side": order.get('side', None),
                        "type": order.get('type', None),
                        "amount": order.get('amount', None),
                        "filled": order.get('filled', None),
                        "price": order.get('price', None),
                        "status": order.get('status', None),
                        "timestamp": order.get('timestamp', None)
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение открытых ордеров.
        
        Args:
            symbol: Торговая пара (опционально)
            
        Returns:
            Список открытых ордеров
        """
        try:
            orders = self.exchange.get_open_orders(symbol)
            result = []
            for order in orders:
                if isinstance(order, Order):
                    result.append({
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "amount": order.amount,
                        "filled": order.filled_amount,
                        "price": order.price,
                        "status": order.status,
                        "timestamp": order.created_at.isoformat() if hasattr(order, 'created_at') and hasattr(order.created_at, 'isoformat') else None
                    })
                elif isinstance(order, dict):
                    result.append({
                        "id": order.get('id', None),
                        "symbol": order.get('symbol', None),
                        "side": order.get('side', None),
                        "type": order.get('type', None),
                        "amount": order.get('amount', None),
                        "filled": order.get('filled', None),
                        "price": order.get('price', None),
                        "status": order.get('status', None),
                        "timestamp": order.get('timestamp', None)
                    })
            return result
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def get_closed_orders(
        self, 
        symbol: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение закрытых ордеров.
        
        Args:
            symbol: Торговая пара (опционально)
            limit: Количество ордеров
            
        Returns:
            Список закрытых ордеров
        """
        try:
            orders = self.exchange.get_closed_orders(symbol, limit=limit)
            result = []
            for order in orders:
                if isinstance(order, Order):
                    result.append({
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "amount": order.amount,
                        "filled": order.filled_amount,
                        "price": order.price,
                        "status": order.status,
                        "timestamp": order.created_at.isoformat() if hasattr(order, 'created_at') and hasattr(order.created_at, 'isoformat') else None
                    })
                elif isinstance(order, dict):
                    result.append({
                        "id": order.get('id', None),
                        "symbol": order.get('symbol', None),
                        "side": order.get('side', None),
                        "type": order.get('type', None),
                        "amount": order.get('amount', None),
                        "filled": order.get('filled', None),
                        "price": order.get('price', None),
                        "status": order.get('status', None),
                        "timestamp": order.get('timestamp', None)
                    })
            return result
        except Exception as e:
            logger.error(f"Error getting closed orders: {e}")
            return []

    async def update_order(
        self, 
        order_id: str, 
        symbol: str, 
        new_price: Optional[float] = None,
        new_amount: Optional[float] = None
    ) -> bool:
        """Обновление ордера.
        
        Args:
            order_id: ID ордера
            symbol: Торговая пара
            new_price: Новая цена
            new_amount: Новое количество
            
        Returns:
            True если ордер обновлен успешно
        """
        try:
            # Сначала отменяем старый ордер
            if not await self.cancel_order(order_id, symbol):
                return False
                
            # Получаем информацию о старом ордере
            old_order = await self.get_order(order_id, symbol)
            if not old_order:
                return False
                
            # Создаем новый ордер с обновленными параметрами
            price = new_price if new_price is not None else old_order["price"]
            amount = new_amount if new_amount is not None else old_order["amount"]
            
            new_order = await self.create_order(
                symbol, 
                old_order["side"], 
                old_order["type"], 
                amount, 
                price
            )
            
            return new_order is not None
        except Exception as e:
            logger.error(f"Error updating order {order_id}: {e}")
            return False

    async def get_order_history(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение истории ордеров.
        
        Args:
            symbol: Торговая пара
            start_time: Время начала
            end_time: Время окончания
            limit: Количество ордеров
            
        Returns:
            История ордеров
        """
        try:
            # Получаем закрытые ордера
            orders = await self.get_closed_orders(symbol, limit)
            
            # Фильтруем по времени если указано
            if start_time or end_time:
                filtered_orders = []
                for order in orders:
                    order_time = datetime.fromisoformat(order["timestamp"])
                    if start_time and order_time < start_time:
                        continue
                    if end_time and order_time > end_time:
                        continue
                    filtered_orders.append(order)
                return filtered_orders
            
            return orders
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return [] 