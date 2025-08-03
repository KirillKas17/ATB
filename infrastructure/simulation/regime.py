import random
from typing import List

from .types import (
    MarketRegimeType,
    MarketSimulationConfig,
    SimulationConfig,
    SimulationConstants,
    SimulationUtils,
)


class MarketRegimeAnalyzer:
    """Анализатор рыночных режимов и фаз рынка."""

    @staticmethod
    def detect_regime(
        prices: List[float], volatility: float, trend_strength: float
    ) -> MarketRegimeType:
        """Определение рыночного режима на основе цен, волатильности и силы тренда."""
        if not prices:
            return MarketRegimeType.UNKNOWN

        # Используем SimulationUtils для определения режима
        return SimulationUtils.detect_market_regime(prices, volatility, trend_strength)

    @staticmethod
    def regime_switching(
        current_regime: MarketRegimeType, probability: float
    ) -> MarketRegimeType:
        """Переключение рыночного режима с заданной вероятностью."""
        if random.random() < probability:
            # Случайный выбор нового режима
            regimes = list(MarketRegimeType)
            regimes.remove(current_regime)
            return random.choice(regimes)
        return current_regime
