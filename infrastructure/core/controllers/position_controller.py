"""
Контроллер для управления позициями.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from infrastructure.core.exchange import Exchange
from infrastructure.core.controllers.order_controller import OrderController

logger = logging.getLogger(__name__)


class PositionController:
    """Контроллер для управления позициями."""

    def __init__(self, exchange: Exchange, order_controller: OrderController):
        """Инициализация контроллера.
        
        Args:
            exchange: Экземпляр биржи
            order_controller: Контроллер ордеров
        """
        self.exchange = exchange
        self.order_controller = order_controller

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение позиций.
        
        Args:
            symbol: Торговая пара (опционально)
            
        Returns:
            Список позиций
        """
        try:
            positions = self.exchange.get_positions()
            result = []
            
            for position in positions:
                if symbol and getattr(position, 'symbol', position.get('symbol', '')) != symbol:  # type: ignore[attr-defined]
                    continue
                    
                result.append({
                    "symbol": getattr(position, 'symbol', position.get('symbol', '')),  # type: ignore[attr-defined]
                    "side": getattr(position, 'side', position.get('side', '')),  # type: ignore[attr-defined]
                    "size": getattr(position, 'size', position.get('size', 0.0)),  # type: ignore[attr-defined]
                    "entry_price": getattr(position, 'entry_price', position.get('entry_price', 0.0)),  # type: ignore[attr-defined]
                    "mark_price": getattr(position, 'mark_price', position.get('mark_price', 0.0)),  # type: ignore[attr-defined]
                    "unrealized_pnl": getattr(position, 'unrealized_pnl', position.get('unrealized_pnl', 0.0)),  # type: ignore[attr-defined]
                    "realized_pnl": getattr(position, 'realized_pnl', position.get('realized_pnl', 0.0)),  # type: ignore[attr-defined]
                    "leverage": getattr(position, 'leverage', position.get('leverage', 1)),  # type: ignore[attr-defined]
                    "margin_type": getattr(position, 'margin_type', position.get('margin_type', '')),  # type: ignore[attr-defined]
                    "timestamp": getattr(position, 'timestamp', position.get('timestamp', datetime.now())).isoformat() if hasattr(getattr(position, 'timestamp', position.get('timestamp', datetime.now())), 'isoformat') else str(getattr(position, 'timestamp', position.get('timestamp', datetime.now())))  # type: ignore[attr-defined]
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение конкретной позиции.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Информация о позиции
        """
        try:
            positions = await self.get_positions(symbol)
            return positions[0] if positions else None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None

    async def open_position(
        self, 
        symbol: str, 
        side: str, 
        size: float, 
        leverage: int = 1,
        order_type: str = "market"
    ) -> Optional[Dict[str, Any]]:
        """Открытие позиции.
        
        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            size: Размер позиции
            leverage: Плечо
            order_type: Тип ордера
            
        Returns:
            Информация о созданном ордере
        """
        try:
            # Устанавливаем плечо
            self.exchange.set_leverage(symbol, leverage)
            
            # Создаем ордер
            order = await self.order_controller.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=size
            )
            
            return order
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None

    async def close_position(
        self, 
        symbol: str, 
        order_type: str = "market"
    ) -> Optional[Dict[str, Any]]:
        """Закрытие позиции.
        
        Args:
            symbol: Торговая пара
            order_type: Тип ордера
            
        Returns:
            Информация о созданном ордере
        """
        try:
            # Получаем текущую позицию
            position = await self.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return None
                
            # Определяем сторону для закрытия
            close_side = "sell" if position["side"] == "buy" else "buy"
            
            # Создаем ордер на закрытие
            order = await self.order_controller.create_order(
                symbol=symbol,
                side=close_side,
                order_type=order_type,
                amount=position["size"]
            )
            
            return order
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    async def modify_position(
        self, 
        symbol: str, 
        new_size: Optional[float] = None,
        new_leverage: Optional[int] = None
    ) -> bool:
        """Модификация позиции.
        
        Args:
            symbol: Торговая пара
            new_size: Новый размер позиции
            new_leverage: Новое плечо
            
        Returns:
            True если позиция модифицирована успешно
        """
        try:
            # Получаем текущую позицию
            position = await self.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
                
            # Изменяем плечо если указано
            if new_leverage is not None:
                self.exchange.set_leverage(symbol, new_leverage)
                
            # Изменяем размер позиции если указано
            if new_size is not None and new_size != position["size"]:
                if new_size > position["size"]:
                    # Увеличиваем позицию
                    additional_size = new_size - position["size"]
                    side = position["side"]
                    await self.order_controller.create_order(
                        symbol=symbol,
                        side=side,
                        order_type="market",
                        amount=additional_size
                    )
                else:
                    # Уменьшаем позицию
                    reduce_size = position["size"] - new_size
                    close_side = "sell" if position["side"] == "buy" else "buy"
                    await self.order_controller.create_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="market",
                        amount=reduce_size
                    )
                    
            return True
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False

    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Получение рисков позиции.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Информация о рисках
        """
        try:
            position = await self.get_position(symbol)
            if not position:
                return {
                    "symbol": symbol,
                    "has_position": False,
                    "risk_level": "none"
                }
                
            # Рассчитываем риски
            unrealized_pnl = position["unrealized_pnl"]
            entry_price = position["entry_price"]
            mark_price = position["mark_price"]
            size = position["size"]
            leverage = position["leverage"]
            
            # Процент изменения цены
            price_change_pct = ((mark_price - entry_price) / entry_price) * 100
            
            # Определяем уровень риска
            if abs(price_change_pct) > 10:
                risk_level = "high"
            elif abs(price_change_pct) > 5:
                risk_level = "medium"
            else:
                risk_level = "low"
                
            return {
                "symbol": symbol,
                "has_position": True,
                "risk_level": risk_level,
                "unrealized_pnl": unrealized_pnl,
                "price_change_pct": price_change_pct,
                "leverage": leverage,
                "position_size": size,
                "entry_price": entry_price,
                "mark_price": mark_price
            }
        except Exception as e:
            logger.error(f"Error getting position risk: {e}")
            return {
                "symbol": symbol,
                "has_position": False,
                "risk_level": "unknown"
            }

    async def get_position_history(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Получение истории позиций.
        
        Args:
            symbol: Торговая пара
            start_time: Время начала
            end_time: Время окончания
            
        Returns:
            История позиций
        """
        try:
            # Получаем историю ордеров
            orders = await self.order_controller.get_order_history(
                symbol, start_time, end_time
            )
            
            # Группируем по позициям
            positions = []
            current_position = None
            
            for order in orders:
                if order["status"] == "closed":
                    if current_position is None:
                        current_position = {
                            "symbol": symbol,
                            "side": order["side"],
                            "entry_order": order,
                            "exit_order": None,
                            "entry_time": order["timestamp"],
                            "exit_time": None,
                            "pnl": 0.0
                        }
                    else:
                        # Закрываем позицию
                        current_position["exit_order"] = order
                        current_position["exit_time"] = order["timestamp"]
                        
                        # Рассчитываем PnL
                        entry_price = current_position["entry_order"]["price"]
                        exit_price = order["price"]
                        size = order["filled"]
                        
                        if current_position["side"] == "buy":
                            pnl = (exit_price - entry_price) * size
                        else:
                            pnl = (entry_price - exit_price) * size
                            
                        current_position["pnl"] = pnl
                        positions.append(current_position)
                        current_position = None
                        
            return positions
        except Exception as e:
            logger.error(f"Error getting position history: {e}")
            return [] 