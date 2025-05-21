import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.technical import (
    calculate_adx,
    calculate_rsi,
    calculate_atr,
    calculate_obv,
    calculate_fractals,
    calculate_wave_clusters
)

from core.strategy import Signal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketRegime(Enum):
    TREND = 1
    SIDEWAYS = 2
    REVERSAL = 3
    MANIPULATION = 4
    VOLATILITY = 5
    ANOMALY = 6


class IIndicatorCalculator(ABC):
    @abstractmethod
    def calculate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        pass


class DefaultIndicatorCalculator(IIndicatorCalculator):
    def calculate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        # ... расчёт всех индикаторов, unit-тестируемый ...
        return {}


class MarketRegimeAgent:
    """
    Агент для определения рыночного режима: ансамбль индикаторов, ML/кластеризация, unit-тестируемость.
    """

    config: Dict[str, Any]
    current_regime: Optional[MarketRegime]
    regime_confidence: float
    regime_history: List[Tuple[MarketRegime, float]]
    indicators: Dict[str, Dict[str, Any]]
    calculator: IIndicatorCalculator

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента с параметрами конфигурации.
        :param config: словарь с параметрами для детекции режима
        """
        self.config = config or {
            "adx_period": 14,
            "adx_threshold": 25,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "volatility_threshold": 2.0,
            "obv_threshold": 0.7,
            "confidence_threshold": 0.8,
        }
        self.current_regime: Optional[MarketRegime] = None
        self.regime_confidence: float = 0.0
        self.regime_history: List[Tuple[MarketRegime, float]] = []
        self.calculator = DefaultIndicatorCalculator()
        self._initialize_indicators()

    def _initialize_indicators(self) -> None:
        """Инициализация структуры для хранения индикаторов."""
        self.indicators = {
            "adx": {"value": None, "trend": None},
            "rsi": {"value": None, "signal": None},
            "atr": {"value": None, "volatility": None},
            "obv": {"value": None, "imbalance": None},
            "fractals": {"value": None, "pattern": None},
            "wave_clusters": {"value": None, "pattern": None},
        }

    def detect_regime(self, dataframe: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Определить текущий рыночный режим по OHLCV-данным.
        :param dataframe: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
        :return: (MarketRegime, confidence)
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(dataframe.columns):
            logger.error(f"DataFrame должен содержать колонки: {required_columns}")
            return MarketRegime.SIDEWAYS, 0.0
        try:
            self._calculate_indicators(dataframe)
            regime_scores = self._calculate_regime_scores()
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
            self.current_regime = dominant_regime[0]
            self.regime_confidence = dominant_regime[1]
            self.regime_history.append((self.current_regime, self.regime_confidence))
            logger.info(
                f"Detected regime: {self.current_regime.name} with confidence: {self.regime_confidence:.2f}"
            )
            return self.current_regime, self.regime_confidence
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return MarketRegime.SIDEWAYS, 0.0

    def _calculate_indicators(self, dataframe: pd.DataFrame) -> None:
        """Вычислить все технические индикаторы для режима."""
        try:
            self.indicators["adx"]["value"] = calculate_adx(
                pd.Series(dataframe["high"]),
                pd.Series(dataframe["low"]),
                pd.Series(dataframe["close"]),
                period=self.config["adx_period"]
            )
            self.indicators["rsi"]["value"] = calculate_rsi(
                pd.Series(dataframe["close"]),
                period=self.config["rsi_period"]
            )
            self.indicators["atr"]["value"] = calculate_atr(
                pd.Series(dataframe["high"]),
                pd.Series(dataframe["low"]),
                pd.Series(dataframe["close"]),
                period=self.config["atr_period"]
            )
            self.indicators["obv"]["value"] = calculate_obv(
                pd.Series(dataframe["close"]),
                pd.Series(dataframe["volume"])
            )
            self.indicators["fractals"]["value"] = calculate_fractals(
                pd.Series(dataframe["high"]),
                pd.Series(dataframe["low"])
            )
            self.indicators["wave_clusters"]["value"] = calculate_wave_clusters(
                np.asarray(dataframe["close"].values)
            )
        except Exception as e:
            logger.error(f"Ошибка при расчёте индикаторов: {str(e)}")
            raise

    def _calculate_regime_scores(self) -> Dict[MarketRegime, float]:
        """Вычислить score для каждого режима."""
        scores = {
            MarketRegime.TREND: self._calculate_trend_score(),
            MarketRegime.SIDEWAYS: self._calculate_sideways_score(),
            MarketRegime.REVERSAL: self._calculate_reversal_score(),
            MarketRegime.MANIPULATION: self._calculate_manipulation_score(),
            MarketRegime.VOLATILITY: self._calculate_volatility_score(),
            MarketRegime.ANOMALY: self._calculate_anomaly_score(),
        }
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    def _get_last(self, arr: Any, n: int = 1) -> Optional[Any]:
        """Безопасно получить последние n элементов массива/Series."""
        if arr is None or len(arr) < n:
            return None
        return arr[-n]

    def _calculate_trend_score(self) -> float:
        adx = self._get_last(self.indicators["adx"]["value"])
        rsi = self._get_last(self.indicators["rsi"]["value"])
        if adx is None or rsi is None:
            return 0.0
        trend_strength = min(1.0, adx / 100.0)
        momentum_alignment = 1.0 if (rsi > 50 and adx > self.config["adx_threshold"]) else 0.5
        return trend_strength * momentum_alignment

    def _calculate_sideways_score(self) -> float:
        adx = self._get_last(self.indicators["adx"]["value"])
        rsi = self._get_last(self.indicators["rsi"]["value"])
        if adx is None or rsi is None:
            return 0.0
        low_trend = 1.0 - min(1.0, adx / self.config["adx_threshold"])
        neutral_momentum = 1.0 - abs(rsi - 50) / 50.0
        return low_trend * neutral_momentum

    def _calculate_reversal_score(self) -> float:
        rsi = self._get_last(self.indicators["rsi"]["value"])
        fractals = self.indicators["fractals"]["value"]
        if rsi is None or fractals is None:
            return 0.0
        extreme_rsi = (
            1.0
            if (rsi > self.config["rsi_overbought"] or rsi < self.config["rsi_oversold"])
            else 0.0
        )
        fractal_pattern = 1.0 if fractals.get("reversal_pattern") else 0.0
        return extreme_rsi * fractal_pattern

    def _calculate_manipulation_score(self) -> float:
        obv = self.indicators["obv"]["value"]
        if obv is None or len(obv) < 2 or obv[-2] == 0:
            return 0.0
        volume_imbalance = abs(obv[-1] - obv[-2]) / abs(obv[-2])
        return min(1.0, volume_imbalance / self.config["obv_threshold"])

    def _calculate_volatility_score(self) -> float:
        atr = self.indicators["atr"]["value"]
        if atr is None or len(atr) < 20:
            return 0.0
        last_atr = atr[-1]
        avg_atr = np.mean(atr[-20:])
        if avg_atr == 0:
            return 0.0
        volatility_ratio = last_atr / avg_atr
        return min(1.0, volatility_ratio / self.config["volatility_threshold"])

    def _calculate_anomaly_score(self) -> float:
        atr = self.indicators["atr"]["value"]
        obv = self.indicators["obv"]["value"]
        if (
            atr is None
            or obv is None
            or len(atr) < 2
            or len(obv) < 2
            or atr[-2] == 0
            or obv[-2] == 0
        ):
            return 0.0
        price_change = abs(atr[-1] / atr[-2] - 1)
        volume_spike = abs(obv[-1] / obv[-2] - 1)
        return min(1.0, (price_change + volume_spike) / 2)

    def get_current_regime_label(self) -> str:
        """Вернуть строковую метку текущего режима."""
        return self.current_regime.name if self.current_regime else "UNKNOWN"

    def regime_confidence_score(self) -> float:
        """Вернуть confidence текущего режима."""
        return self.regime_confidence

    def get_regime_history(self) -> List[Tuple[MarketRegime, float]]:
        """Вернуть историю режимов."""
        return self.regime_history

    async def get_signals(self) -> List[Signal]:
        """Получение сигналов от агента."""
        try:
            if not self.current_regime:
                return []

            # Создание сигнала на основе текущего режима
            signal = Signal(
                pair="BTC/USDT",  # TODO: сделать динамическим
                action="buy" if self.current_regime == MarketRegime.TREND else "sell",
                price=0.0,  # TODO: добавить текущую цену
                size=0.0,  # TODO: добавить размер позиции
                metadata={
                    "regime": self.current_regime.name,
                    "indicators": self.indicators,
                    "strength": min(1.0, self.regime_confidence * 1.5),
                    "confidence": self.regime_confidence,
                    "timestamp": datetime.now().isoformat(),
                    "source": "market_regime"
                }
            )
            return [signal]
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return []
