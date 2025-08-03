from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from domain.types.ml_types import SpreadAnalysisResult, SpreadMovementPrediction


class AnalysisError(Exception):
    """Ошибка анализа."""

    pass


class ISpreadAnalyzer(ABC):
    @abstractmethod
    async def analyze_spread(self, order_book: Dict[str, Any]) -> SpreadAnalysisResult:
        """Анализирует спред в ордербуке."""
        pass

    @abstractmethod
    async def calculate_imbalance(self, order_book: Dict[str, Any]) -> float:
        """Вычисляет дисбаланс в ордербуке."""
        pass

    @abstractmethod
    async def predict_spread_movement(
        self, historical_data: pd.DataFrame
    ) -> SpreadMovementPrediction:
        """Предсказывает движение спреда."""
        pass


class SpreadAnalyzer(ISpreadAnalyzer):
    """Сервис для анализа спредов и дисбалансов в ордербуке"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "spread_threshold": 0.001,
            "imbalance_threshold": 0.1,
            "confidence_threshold": 0.7,
        }
        self.logger = None  # Будет установлен через dependency injection

    async def analyze_spread(self, order_book: Dict[str, Any]) -> SpreadAnalysisResult:
        """Анализирует спред и дисбаланс в ордербуке"""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                raise AnalysisError("Invalid order book structure")
            best_bid = order_book["bids"][0]["price"]
            best_ask = order_book["asks"][0]["price"]
            spread = (best_ask - best_bid) / best_bid
            imbalance = await self.calculate_imbalance(order_book)
            confidence = min(abs(imbalance), 1.0)
            return {
                "spread": spread,
                "imbalance": imbalance,
                "confidence": confidence,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }
        except Exception as e:
            if self.logger:
                try:
                    self.logger.error(f"Error analyzing spread: {str(e)}")
                except Exception:
                    pass  # Игнорируем ошибки логирования
            return {
                "spread": 0.0, 
                "imbalance": 0.0, 
                "confidence": 0.0, 
                "best_bid": 0.0, 
                "best_ask": 0.0,
            }

    async def calculate_imbalance(self, order_book: Dict[str, Any]) -> float:
        """Вычисляет дисбаланс между bid и ask объемами"""
        try:
            bid_volume = sum(order["size"] for order in order_book["bids"])
            ask_volume = sum(order["size"] for order in order_book["asks"])
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            else:
                imbalance = (bid_volume - ask_volume) / total_volume
                return float(imbalance)
        except Exception as e:
            if self.logger:
                try:
                    self.logger.error(f"Error calculating imbalance: {str(e)}")
                except Exception:
                    pass  # Игнорируем ошибки логирования
            return 0.0

    async def predict_spread_movement(
        self, historical_data: pd.DataFrame
    ) -> SpreadMovementPrediction:
        """Предсказывает движение спреда на основе исторических данных"""
        try:
            if historical_data.empty:
                return {
                    "prediction": 0.0, 
                    "confidence": 0.0, 
                    "ma_short": 0.0, 
                    "ma_long": 0.0,
                }
            else:
                # Простая модель на основе скользящих средних
                spread_series = historical_data.get(
                    "spread", pd.Series([0.0] * len(historical_data))
                )
                if len(spread_series) < 10:
                    return {
                        "prediction": 0.0, 
                        "confidence": 0.0, 
                        "ma_short": 0.0, 
                        "ma_long": 0.0,
                    }
                else:
                    ma_short = spread_series.rolling(window=5).mean().iloc[-1]
                    ma_long = spread_series.rolling(window=20).mean().iloc[-1]
                    prediction = ma_short - ma_long
                    confidence = min(abs(prediction), 1.0)
                    return {
                        "prediction": prediction,
                        "confidence": confidence,
                        "ma_short": ma_short,
                        "ma_long": ma_long,
                    }
        except Exception as e:
            if self.logger:
                try:
                    self.logger.error(f"Error predicting spread movement: {str(e)}")
                except Exception:
                    pass  # Игнорируем ошибки логирования
            return {
                "prediction": 0.0, 
                "confidence": 0.0, 
                "ma_short": 0.0, 
                "ma_long": 0.0,
            }

    async def get_spread_statistics(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Получает статистику по спреду"""
        spread_analysis = await self.analyze_spread(order_book)
        return {
            "current_spread": spread_analysis["spread"],
            "spread_percentage": spread_analysis["spread"] * 100,
            "imbalance": spread_analysis["imbalance"],
            "confidence": spread_analysis["confidence"],
            "is_wide_spread": spread_analysis["spread"] > self.config["spread_threshold"],
            "is_imbalanced": abs(spread_analysis["imbalance"]) > self.config["imbalance_threshold"],
        }
