# Миграция из utils/market_regime.py
# Содержимое файла переносится без изменений для сохранения бизнес-логики.
import pandas as pd
from enum import Enum
from typing import Any, Dict

from .technical import calculate_momentum, calculate_volatility


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    def __init__(
        self,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.01,
        window_size: int = 20,
    ):
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.window_size = window_size

    def detect_regime(self, data: pd.Series) -> MarketRegime:
        """Определение текущего рыночного режима"""
        if len(data) < self.window_size:
            return MarketRegime.UNKNOWN
        # Расчет волатильности
        volatility = calculate_volatility(data, self.window_size).iloc[-1]
        # Расчет тренда
        momentum = calculate_momentum(data, self.window_size).iloc[-1]
        # Определение режима
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        elif momentum > self.trend_threshold:
            return MarketRegime.TRENDING_UP
        elif momentum < -self.trend_threshold:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def get_regime_metrics(self, data: pd.Series) -> Dict[str, Any]:
        """Получение метрик рыночного режима"""
        volatility = calculate_volatility(data, self.window_size)
        momentum = calculate_momentum(data, self.window_size)
        return {
            "volatility": volatility.iloc[-1],
            "momentum": momentum.iloc[-1],
            "regime": self.detect_regime(data).value,
        }


def detect_market_regime(prices: list, window: int = 20) -> str:
    """Определение рыночного режима: тренд, флэт, высокая волатильность"""
    if len(prices) < window:
        return "unknown"
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    avg_diff = sum(diffs[-window:]) / window
    if avg_diff > 2:
        return "volatile"
    elif prices[-1] > prices[-window]:
        return "uptrend"
    elif prices[-1] < prices[-window]:
        return "downtrend"
    else:
        return "sideways"
