"""
Анализатор ликвидности.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal

from domain.entities.order import Order
from domain.entities.trading import Trade
from domain.value_objects.money import Money


class ConcreteLiquidityAnalyzer:
    """Конкретный анализатор ликвидности."""
    
    def __init__(self) -> None:
        pass
    
    def analyze_order_liquidity(self, orders: List[Order]) -> Dict[str, Any]:
        """Анализ ликвидности ордеров."""
        if not orders:
            return {
                "liquidity_score": 0.0,
                "depth_analysis": {},
                "spread_analysis": {},
            }
        
        # Анализ глубины рынка
        buy_orders = [o for o in orders if o.side.value == "buy" and o.status.value in ["open", "pending"]]
        sell_orders = [o for o in orders if o.side.value == "sell" and o.status.value in ["open", "pending"]]
        
        buy_depth = sum(o.quantity for o in buy_orders)
        sell_depth = sum(o.quantity for o in sell_orders)
        
        # Анализ спреда
        if buy_orders and sell_orders:
            best_bid = max(o.price.amount if o.price else Decimal("0") for o in buy_orders)
            best_ask = min(o.price.amount if o.price else Decimal("0") for o in sell_orders)
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid * 100) if best_bid > 0 else 0
        else:
            spread = Decimal("0")
            spread_percentage = 0
        
        # Расчет ликвидности
        total_depth = buy_depth + sell_depth
        liquidity_score = min(1.0, float(total_depth / 1000)) if total_depth > 0 else 0.0
        
        return {
            "liquidity_score": liquidity_score,
            "depth_analysis": {
                "buy_depth": float(buy_depth),
                "sell_depth": float(sell_depth),
                "total_depth": float(total_depth),
            },
            "spread_analysis": {
                "spread": float(spread),
                "spread_percentage": float(spread_percentage),
            },
        }
    
    def analyze_trade_liquidity(self, trades: List[Trade]) -> Dict[str, Any]:
        """Анализ ликвидности сделок."""
        if not trades:
            return {
                "volume_profile": {},
                "execution_quality": {},
            }
        
        # Профиль объема
        volumes = [t.quantity.value for t in trades]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        # Качество исполнения
        execution_times = []
        for i in range(1, len(trades)):
            time_diff = (trades[i].timestamp - trades[i-1].timestamp).total_seconds()
            execution_times.append(time_diff)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "volume_profile": {
                "total_volume": float(sum(volumes)),
                "avg_volume": float(avg_volume),
                "max_volume": float(max(volumes)) if volumes else 0,
                "min_volume": float(min(volumes)) if volumes else 0,
            },
            "execution_quality": {
                "avg_execution_time": float(avg_execution_time),
                "total_trades": len(trades),
            },
        }
    
    def calculate_liquidity_metrics(self, orders: List[Order], trades: List[Trade]) -> Dict[str, Any]:
        """Расчет метрик ликвидности."""
        order_liquidity = self.analyze_order_liquidity(orders)
        trade_liquidity = self.analyze_trade_liquidity(trades)
        
        # Комбинированная оценка ликвидности
        order_score = order_liquidity["liquidity_score"]
        trade_score = min(1.0, trade_liquidity["execution_quality"]["total_trades"] / 100)
        
        combined_score = (order_score + trade_score) / 2
        
        return {
            "combined_liquidity_score": combined_score,
            "order_liquidity": order_liquidity,
            "trade_liquidity": trade_liquidity,
            "recommendations": self._generate_recommendations(combined_score),
        }
    
    def _generate_recommendations(self, liquidity_score: float) -> List[str]:
        """Генерация рекомендаций на основе ликвидности."""
        recommendations = []
        
        if liquidity_score < 0.3:
            recommendations.append("Низкая ликвидность - рекомендуется увеличить размеры ордеров")
        elif liquidity_score < 0.6:
            recommendations.append("Средняя ликвидность - мониторинг необходим")
        else:
            recommendations.append("Высокая ликвидность - оптимальные условия для торговли")
        
        return recommendations