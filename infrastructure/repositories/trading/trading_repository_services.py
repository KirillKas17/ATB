"""
Сервисы торгового репозитория.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal

from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trading import Trade
from domain.value_objects.money import Money


class TradingRepositoryServices:
    """Сервисы для торгового репозитория."""
    
    def __init__(self) -> None:
        pass
    
    def calculate_order_metrics(self, orders: List[Order]) -> Dict[str, Any]:
        """Расчет метрик ордеров."""
        if not orders:
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "cancelled_orders": 0,
                "total_volume": Decimal("0"),
                "total_value": Decimal("0"),
            }
        
        total_orders = len(orders)
        filled_orders = len([o for o in orders if o.is_filled])
        cancelled_orders = len([o for o in orders if o.is_cancelled])
        total_volume = sum(o.quantity for o in orders)
        total_value = sum(o.total_value.amount if o.total_value is not None else Decimal("0") for o in orders)
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "total_volume": total_volume,
            "total_value": total_value,
        }
    
    def calculate_position_metrics(self, positions: List[Position]) -> Dict[str, Any]:
        """Расчет метрик позиций."""
        if not positions:
            return {
                "total_positions": 0,
                "open_positions": 0,
                "total_pnl": Decimal("0"),
                "profitable_positions": 0,
            }
        
        total_positions = len(positions)
        open_positions = len([p for p in positions if p.is_open])
        total_pnl = sum(p.unrealized_pnl.amount if p.unrealized_pnl is not None else Decimal("0") for p in positions)
        profitable_positions = len([p for p in positions if p.unrealized_pnl and p.unrealized_pnl.amount > 0])
        
        return {
            "total_positions": total_positions,
            "open_positions": open_positions,
            "total_pnl": total_pnl,
            "profitable_positions": profitable_positions,
        }
    
    def calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Расчет метрик сделок."""
        if not trades:
            return {
                "total_trades": 0,
                "total_volume": Decimal("0"),
                "total_commission": Decimal("0"),
                "total_value": Decimal("0"),
            }
        
        total_trades = len(trades)
        total_volume = sum(t.quantity.value for t in trades)
        total_commission = sum(t.commission.value for t in trades)
        total_value = sum(t.total_value.value for t in trades)
        
        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "total_commission": total_commission,
            "total_value": total_value,
        }