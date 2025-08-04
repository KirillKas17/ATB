"""Модуль для управления состоянием рынка и анализа рыночных метрик."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MarketState:
    """Состояние рынка с расширенными метриками."""

    timestamp: datetime
    price: float
    volume: float
    volatility: float
    trend: str
    indicators: Dict[str, float]
    market_regime: str  # bull, bear, sideways, volatile
    liquidity: float
    momentum: float
    sentiment: float  # -1 to 1
    support_levels: List[float]
    resistance_levels: List[float]
    market_depth: Dict[str, float]  # bid/ask volumes
    correlation_matrix: Dict[str, Dict[str, float]]
    market_impact: float
    volume_profile: Dict[float, float]  # price levels and their volumes

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.
        Returns:
            Dict[str, Any]: Словарь с данными состояния рынка
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "volatility": self.volatility,
            "trend": self.trend,
            "indicators": self.indicators,
            "market_regime": self.market_regime,
            "liquidity": self.liquidity,
            "momentum": self.momentum,
            "sentiment": self.sentiment,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "market_depth": self.market_depth,
            "correlation_matrix": self.correlation_matrix,
            "market_impact": self.market_impact,
            "volume_profile": self.volume_profile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        """
        Создание из словаря.
        Args:
            data: Словарь с данными состояния рынка
        Returns:
            MarketState: Объект состояния рынка
        """
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class MarketStateManager:
    """Расширенный менеджер состояния рынка."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация менеджера состояния рынка.
        Args:
            config: Конфигурация менеджера
        """
        self.config = config or {}
        self.states: List[MarketState] = []
        self.lookback_period = self.config.get("lookback_period", 100)
        self.regime_threshold = self.config.get("regime_threshold", 0.7)
        self.volatility_window = self.config.get("volatility_window", 20)
        self.momentum_window = self.config.get("momentum_window", 14)

    def add_state(self, state: MarketState) -> None:
        """
        Добавление нового состояния с анализом.
        Args:
            state: Новое состояние рынка
        """
        self.states.append(state)
        if len(self.states) > self.lookback_period:
            self.states.pop(0)
        # Обновление метрик
        self._update_market_regime()
        self._update_support_resistance()
        self._update_correlation_matrix()

    def get_latest_state(self) -> Optional[MarketState]:
        """
        Получение последнего состояния.
        Returns:
            Optional[MarketState]: Последнее состояние или None
        """
        return self.states[-1] if self.states else None

    def get_state_at(self, timestamp: datetime) -> Optional[MarketState]:
        """
        Получение состояния на определенный момент времени.
        Args:
            timestamp: Временная метка
        Returns:
            Optional[MarketState]: Состояние на указанное время или None
        """
        for state in reversed(self.states):
            if state.timestamp <= timestamp:
                return state
        return None

    def get_market_regime(self) -> str:
        """
        Определение текущего режима рынка.
        Returns:
            str: Режим рынка (bull, bear, sideways, volatile, unknown)
        """
        if not self.states:
            return "unknown"
        self.states[-1]
        prices = [s.price for s in self.states[-self.volatility_window :]]
        # Расчет тренда
        trend = np.polyfit(np.arange(len(prices)), np.array(prices), 1)[0]
        # Расчет волатильности
        volatility = np.std(prices) / np.mean(prices)
        # Определение режима
        if volatility > self.regime_threshold:
            return "volatile"
        elif trend > 0.001:
            return "bull"
        elif trend < -0.001:
            return "bear"
        else:
            return "sideways"

    def _update_market_regime(self) -> None:
        """Обновление режима рынка."""
        if not self.states:
            return
        regime = self.get_market_regime()
        self.states[-1].market_regime = regime

    def _update_support_resistance(self) -> None:
        """Обновление уровней поддержки и сопротивления."""
        if len(self.states) < 20:
            return
        prices = [s.price for s in self.states]
        # Поиск локальных минимумов и максимумов
        from scipy.signal import argrelextrema

        local_min = argrelextrema(np.array(prices), np.less, order=5)[0]
        local_max = argrelextrema(np.array(prices), np.greater, order=5)[0]
        # Кластеризация уровней
        support_levels = self._cluster_levels([prices[i] for i in local_min])
        resistance_levels = self._cluster_levels([prices[i] for i in local_max])
        self.states[-1].support_levels = support_levels
        self.states[-1].resistance_levels = resistance_levels

    def _cluster_levels(
        self, levels: List[float], threshold: float = 0.01
    ) -> List[float]:
        """
        Кластеризация ценовых уровней.
        Args:
            levels: Список ценовых уровней
            threshold: Порог для кластеризации
        Returns:
            List[float]: Кластеризованные уровни
        """
        if not levels:
            return []
        levels = sorted(levels)
        clusters: List[float] = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                clusters.append(float(np.mean(current_cluster)))
                current_cluster = [level]
        clusters.append(float(np.mean(current_cluster)))
        return clusters

    def _update_correlation_matrix(self) -> None:
        """Обновление матрицы корреляций."""
        if not self.states:
            return
        # Получение всех доступных пар
        pairs: set[str] = set()
        for state in self.states:
            pairs.update(state.correlation_matrix.keys())
        # Расчет корреляций
        correlations: Dict[str, Dict[str, float]] = {}
        for pair1 in pairs:
            correlations[pair1] = {}
            for pair2 in pairs:
                if pair1 != pair2:
                    corr = self._calculate_correlation(pair1, pair2)
                    correlations[pair1][pair2] = corr
        self.states[-1].correlation_matrix = correlations

    def _calculate_correlation(self, pair1: str, pair2: str) -> float:
        """
        Расчет корреляции между парами.
        Args:
            pair1: Первая торговая пара
            pair2: Вторая торговая пара
        Returns:
            float: Коэффициент корреляции
        """
        prices1 = []
        prices2 = []
        for state in self.states:
            if (
                pair1 in state.correlation_matrix
                and pair2 in state.correlation_matrix[pair1]
            ):
                prices1.append(state.price)
                prices2.append(state.correlation_matrix[pair1][pair2])
        if len(prices1) < 2:
            return 0.0
        return float(np.corrcoef(prices1, prices2)[0, 1])

    def get_market_metrics(self) -> Dict[str, float]:
        """
        Получение расширенных метрик рынка.
        Returns:
            Dict[str, float]: Словарь с метриками рынка
        """
        if not self.states:
            return {}
        latest = self.states[-1]
        prices = [s.price for s in self.states[-self.volatility_window :]]
        volatility = float(np.std(prices) / np.mean(prices))
        momentum = float((latest.price - prices[0]) / prices[0])
        trend_strength = float(abs(np.polyfit(np.arange(len(prices)), np.array(prices), 1)[0]))
        volume_trend = float(
            np.mean([s.volume for s in self.states[-5:]])
            / np.mean([s.volume for s in self.states[-10:-5]])
        )
        regime_score = float(self._calculate_regime_score())
        liquidity_score = float(latest.liquidity)
        sentiment_score = float(latest.sentiment)
        return {
            "volatility": volatility,
            "momentum": momentum,
            "trend_strength": trend_strength,
            "volume_trend": volume_trend,
            "market_regime_score": regime_score,
            "liquidity_score": liquidity_score,
            "sentiment_score": sentiment_score,
        }

    def _calculate_regime_score(self) -> float:
        """
        Расчет оценки режима рынка.
        Returns:
            float: Оценка режима рынка
        """
        if not self.states:
            return 0.0
        regime = self.states[-1].market_regime
        if regime == "bull":
            return 1.0
        elif regime == "bear":
            return -1.0
        elif regime == "volatile":
            return 0.5
        else:
            return 0.0
