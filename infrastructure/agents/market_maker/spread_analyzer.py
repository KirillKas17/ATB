"""
Сервис для анализа спредов в маркет-мейкинге.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SpreadAnalyzer:
    """Анализатор спредов для маркет-мейкинга."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "spread_threshold": 0.001,
            "confidence_threshold": 0.7,
        }

    def analyze_spread(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ спреда в ордербуке."""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return {"spread": 0.0, "imbalance": 0.0, "confidence": 0.0}

            bids = order_book["bids"]
            asks = order_book["asks"]

            if not bids or not asks:
                return {"spread": 0.0, "imbalance": 0.0, "confidence": 0.0}

            # Получаем лучшие цены
            best_bid = bids[0]["price"] if isinstance(bids[0], dict) else bids[0]
            best_ask = asks[0]["price"] if isinstance(asks[0], dict) else asks[0]

            # Рассчитываем спред
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0

            # Рассчитываем дисбаланс
            bid_volume = (
                sum(order["size"] for order in bids if isinstance(order, dict))
                if isinstance(bids[0], dict)
                else sum(bids)
            )
            ask_volume = (
                sum(order["size"] for order in asks if isinstance(order, dict))
                if isinstance(asks[0], dict)
                else sum(asks)
            )

            total_volume = bid_volume + ask_volume
            imbalance = (
                (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
            )

            # Рассчитываем уверенность
            confidence = min(abs(imbalance), 1.0)

            return {
                "spread": spread,
                "imbalance": imbalance,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {"spread": 0.0, "imbalance": 0.0, "confidence": 0.0}

    def is_spread_acceptable(self, spread: float) -> bool:
        """Проверка приемлемости спреда."""
        spread_threshold = self.config.get("spread_threshold", 0.001)
        if isinstance(spread_threshold, (int, float)):
            return spread <= float(spread_threshold)
        return False

    def get_spread_recommendation(
        self, spread_analysis: Dict[str, float]
    ) -> Dict[str, Any]:
        """Получение рекомендаций на основе анализа спреда."""
        spread = spread_analysis.get("spread", 0.0)
        imbalance = spread_analysis.get("imbalance", 0.0)
        confidence = spread_analysis.get("confidence", 0.0)

        recommendation = {
            "action": "hold",
            "reason": "normal_spread",
            "confidence": confidence,
        }

        if spread > self.config["spread_threshold"]:
            recommendation["action"] = "avoid"
            recommendation["reason"] = "high_spread"
        elif abs(imbalance) > 0.5:
            if imbalance > 0:
                recommendation["action"] = "buy"
                recommendation["reason"] = "bid_imbalance"
            else:
                recommendation["action"] = "sell"
                recommendation["reason"] = "ask_imbalance"

        return recommendation
