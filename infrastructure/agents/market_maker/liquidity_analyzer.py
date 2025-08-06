"""
Сервис для анализа ликвидности в маркет-мейкинге.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """Анализатор ликвидности для маркет-мейкинга."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {
            "liquidity_zone_size": 0.005,
            "volume_threshold": 100000,
            "confidence_threshold": 0.7,
        }

    def analyze_liquidity(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Dict[str, float]:
        """Анализ ликвидности на основе рыночных данных и ордербука."""
        try:
            if market_data.empty:
                return {"support": 0.0, "resistance": 0.0, "neutral": 0.0}
            # Анализ объемного профиля
            volume_profile = self._calculate_volume_profile(market_data)
            # Анализ уровней поддержки и сопротивления
            support_resistance = self._find_support_resistance_levels(market_data)
            # Анализ ликвидности в ордербуке
            order_book_liquidity = self._analyze_order_book_liquidity(order_book)
            # Комбинируем результаты
            support_score = self._calculate_support_score(
                volume_profile, support_resistance, order_book_liquidity
            )
            resistance_score = self._calculate_resistance_score(
                volume_profile, support_resistance, order_book_liquidity
            )
            neutral_score = 1.0 - max(support_score, resistance_score)
            return {
                "support": support_score,
                "resistance": resistance_score,
                "neutral": neutral_score,
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"support": 0.0, "resistance": 0.0, "neutral": 0.0}

    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет объемного профиля."""
        try:
            # Группируем данные по ценовым уровням
            price_bins = pd.cut(market_data["close"], bins=20)
            volume_profile = market_data.groupby(price_bins)["volume"].sum()
            # Находим уровни с высоким объемом
            high_volume_levels = volume_profile[
                volume_profile > volume_profile.quantile(0.8)
            ]
            return {
                "high_volume_levels": high_volume_levels.index.tolist(),
                "volume_distribution": volume_profile.to_dict(),
            }
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {"high_volume_levels": [], "volume_distribution": {}}

    def _find_support_resistance_levels(
        self, market_data: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Поиск уровней поддержки и сопротивления."""
        try:
            levels: Dict[str, List[float]] = {"support": [], "resistance": []}
            # Простой алгоритм поиска локальных экстремумов
            for i in range(2, len(market_data) - 2):
                # Поддержка (локальный минимум)
                if (
                    market_data["low"].iloc[i] < market_data["low"].iloc[i - 1]
                    and market_data["low"].iloc[i] < market_data["low"].iloc[i - 2]
                    and market_data["low"].iloc[i] < market_data["low"].iloc[i + 1]
                    and market_data["low"].iloc[i] < market_data["low"].iloc[i + 2]
                ):
                    levels["support"].append(market_data["low"].iloc[i])
                # Сопротивление (локальный максимум)
                if (
                    market_data["high"].iloc[i] > market_data["high"].iloc[i - 1]
                    and market_data["high"].iloc[i] > market_data["high"].iloc[i - 2]
                    and market_data["high"].iloc[i] > market_data["high"].iloc[i + 1]
                    and market_data["high"].iloc[i] > market_data["high"].iloc[i + 2]
                ):
                    levels["resistance"].append(market_data["high"].iloc[i])
            return levels
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {str(e)}")
            return {"support": [], "resistance": []}

    def _analyze_order_book_liquidity(
        self, order_book: Dict[str, Any]
    ) -> Dict[str, float]:
        """Анализ ликвидности в ордербуке."""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return {"bid_liquidity": 0.0, "ask_liquidity": 0.0}
            bids = order_book["bids"]
            asks = order_book["asks"]
            # Рассчитываем ликвидность на стороне покупок
            bid_liquidity = (
                sum(order["size"] for order in bids[:5])
                if isinstance(bids[0], dict)
                else sum(bids[:5])
            )
            # Рассчитываем ликвидность на стороне продаж
            ask_liquidity = (
                sum(order["size"] for order in asks[:5])
                if isinstance(asks[0], dict)
                else sum(asks[:5])
            )
            return {
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
            }
        except Exception as e:
            logger.error(f"Error analyzing order book liquidity: {str(e)}")
            return {"bid_liquidity": 0.0, "ask_liquidity": 0.0}

    def _calculate_support_score(
        self,
        volume_profile: Dict[str, Any],
        support_resistance: Dict[str, List[float]],
        order_book_liquidity: Dict[str, float],
    ) -> float:
        """Расчет скора поддержки."""
        try:
            score = 0.0
            # Учитываем объемный профиль
            if volume_profile["high_volume_levels"]:
                score += 0.3
            # Учитываем уровни поддержки
            if support_resistance["support"]:
                score += 0.4
            # Учитываем ликвидность в ордербуке
            if order_book_liquidity["bid_liquidity"] > self.config["volume_threshold"]:
                score += 0.3
            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating support score: {str(e)}")
            return 0.0

    def _calculate_resistance_score(
        self,
        volume_profile: Dict[str, Any],
        support_resistance: Dict[str, List[float]],
        order_book_liquidity: Dict[str, float],
    ) -> float:
        """Расчет скора сопротивления."""
        try:
            score = 0.0
            # Учитываем объемный профиль
            if volume_profile["high_volume_levels"]:
                score += 0.3
            # Учитываем уровни сопротивления
            if support_resistance["resistance"]:
                score += 0.4
            # Учитываем ликвидность в ордербуке
            if order_book_liquidity["ask_liquidity"] > self.config["volume_threshold"]:
                score += 0.3
            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating resistance score: {str(e)}")
            return 0.0

    def get_liquidity_recommendation(
        self, liquidity_analysis: Dict[str, float]
    ) -> Dict[str, Any]:
        """Получение рекомендаций на основе анализа ликвидности."""
        support = liquidity_analysis.get("support", 0.0)
        resistance = liquidity_analysis.get("resistance", 0.0)
        neutral = liquidity_analysis.get("neutral", 0.0)
        recommendation = {
            "action": "hold",
            "reason": "neutral_liquidity",
            "confidence": max(support, resistance, neutral),
        }
        if support > resistance and support > neutral:
            recommendation["action"] = "buy"
            recommendation["reason"] = "strong_support"
        elif resistance > support and resistance > neutral:
            recommendation["action"] = "sell"
            recommendation["reason"] = "strong_resistance"
        return recommendation
