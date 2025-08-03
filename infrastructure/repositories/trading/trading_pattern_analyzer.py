"""
Анализатор торговых паттернов.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal

from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trading import Trade


class TradingPatternAnalyzer:
    """Анализатор торговых паттернов."""
    
    def __init__(self) -> None:
        pass
    
    def analyze_order_patterns(self, orders: List[Order]) -> Dict[str, Any]:
        """Анализ паттернов ордеров."""
        if not orders:
            return {"patterns": [], "insights": []}
        
        patterns = []
        insights = []
        
        # Анализ частоты ордеров
        buy_orders = [o for o in orders if o.side.value == "buy"]
        sell_orders = [o for o in orders if o.side.value == "sell"]
        
        if len(buy_orders) > len(sell_orders) * 1.5:
            patterns.append("buy_dominance")
            insights.append("Преобладают покупки")
        elif len(sell_orders) > len(buy_orders) * 1.5:
            patterns.append("sell_dominance")
            insights.append("Преобладают продажи")
        
        # Анализ размеров ордеров
        volumes = [o.quantity for o in orders if o.quantity > 0]
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            large_orders = [o for o in orders if o.quantity > avg_volume * 2]
            if large_orders:
                patterns.append("large_orders")
                insights.append(f"Обнаружено {len(large_orders)} крупных ордеров")
        
        return {
            "patterns": patterns,
            "insights": insights,
            "buy_count": len(buy_orders),
            "sell_count": len(sell_orders),
        }
    
    def analyze_position_patterns(self, positions: List[Position]) -> Dict[str, Any]:
        """Анализ паттернов позиций."""
        if not positions:
            return {"patterns": [], "insights": []}
        
        patterns = []
        insights = []
        
        # Анализ распределения позиций
        long_positions = [p for p in positions if p.side.value == "long"]
        short_positions = [p for p in positions if p.side.value == "short"]
        
        if len(long_positions) > len(short_positions) * 2:
            patterns.append("long_bias")
            insights.append("Преобладают длинные позиции")
        elif len(short_positions) > len(long_positions) * 2:
            patterns.append("short_bias")
            insights.append("Преобладают короткие позиции")
        
        # Анализ прибыльности
        profitable_positions = [p for p in positions if p.unrealized_pnl and p.unrealized_pnl.value > 0]
        if profitable_positions:
            profit_ratio = len(profitable_positions) / len(positions)
            if profit_ratio > 0.7:
                patterns.append("high_profitability")
                insights.append("Высокая прибыльность позиций")
        
        return {
            "patterns": patterns,
            "insights": insights,
            "long_count": len(long_positions),
            "short_count": len(short_positions),
            "profitable_ratio": len(profitable_positions) / len(positions) if positions else 0,
        }
    
    def analyze_trade_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Анализ паттернов сделок."""
        if not trades:
            return {"patterns": [], "insights": []}
        
        patterns = []
        insights = []
        
        # Анализ объема сделок
        volumes = [float(t.quantity.value) for t in trades]
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            large_trades = [t for t in trades if float(t.quantity.value) > avg_volume * 2]
            if large_trades:
                patterns.append("large_trades")
                insights.append(f"Обнаружено {len(large_trades)} крупных сделок")
        
        # Анализ комиссий
        commissions = [float(t.commission.value) for t in trades]
        if commissions:
            avg_commission = sum(commissions) / len(commissions)
            high_commission_trades = [t for t in trades if float(t.commission.value) > avg_commission * 1.5]
            if high_commission_trades:
                patterns.append("high_commissions")
                insights.append("Высокие комиссии в некоторых сделках")
        
        return {
            "patterns": patterns,
            "insights": insights,
            "total_trades": len(trades),
            "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
            "avg_commission": sum(commissions) / len(commissions) if commissions else 0,
        }