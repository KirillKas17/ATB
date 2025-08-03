import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


class MarketRegime(Enum):
    """Перечисление рыночных режимов"""

    TREND = 1
    SIDEWAYS = 2
    REVERSAL = 3
    MANIPULATION = 4
    VOLATILITY = 5
    ANOMALY = 6


class IIndicatorCalculator(ABC):
    """Абстрактный интерфейс для калькулятора индикаторов"""

    @abstractmethod
    def calculate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет индикаторов для определения рыночного режима.
        Args:
            dataframe: DataFrame с рыночными данными
        Returns:
            Словарь с рассчитанными индикаторами
        """


from dataclasses import dataclass

# ============================================================================
# Реализация калькулятора индикаторов
# ============================================================================
from typing import Any, Dict, Optional


@dataclass
class RegimeIndicators:
    """Индикаторы для определения рыночного режима."""

    trend_strength: float
    volatility: float
    momentum: float
    mean_reversion: float
    volume_trend: float
    price_range: float
    regime_confidence: float
    regime_type: MarketRegime


class DefaultIndicatorCalculator(IIndicatorCalculator):
    """Промышленная реализация калькулятора индикаторов."""

    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period

    def calculate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет индикаторов для определения рыночного режима.
        Args:
            dataframe: DataFrame с рыночными данными
        Returns:
            Словарь с рассчитанными индикаторами
        """
        try:
            if len(dataframe) < self.lookback_period:
                return self._get_default_indicators()
            # Базовые расчеты
            close_prices = dataframe["close"].values
            high_prices = dataframe["high"].values
            low_prices = dataframe["low"].values
            volumes = dataframe["volume"].values if "volume" in dataframe.columns else np.ones(len(dataframe))
            # Расчет индикаторов
            trend_strength = self._calculate_trend_strength(np.asarray(close_prices))
            volatility = self._calculate_volatility(np.asarray(close_prices))
            momentum = self._calculate_momentum(np.asarray(close_prices))
            mean_reversion = self._calculate_mean_reversion(np.asarray(close_prices))
            volume_trend = self._calculate_volume_trend(np.asarray(volumes))
            price_range = self._calculate_price_range(np.asarray(high_prices), np.asarray(low_prices))
            # Определение режима
            regime_type = self._determine_regime(
                trend_strength, volatility, momentum, mean_reversion, volume_trend
            )
            # Уверенность в режиме
            regime_confidence = self._calculate_regime_confidence(
                trend_strength, volatility, momentum, mean_reversion, volume_trend
            )
            indicators = RegimeIndicators(
                trend_strength=trend_strength,
                volatility=volatility,
                momentum=momentum,
                mean_reversion=mean_reversion,
                volume_trend=volume_trend,
                price_range=price_range,
                regime_confidence=regime_confidence,
                regime_type=regime_type,
            )
            return {
                "trend_strength": trend_strength,
                "volatility": volatility,
                "momentum": momentum,
                "mean_reversion": mean_reversion,
                "volume_trend": volume_trend,
                "price_range": price_range,
                "regime_confidence": regime_confidence,
                "regime_type": regime_type.value,
                "regime_name": regime_type.name,
                "indicators": indicators,
            }
        except Exception as e:
            return self._get_default_indicators(error=str(e))

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Расчет силы тренда."""
        try:
            if len(prices) < self.lookback_period:
                return 0.0
            # Линейная регрессия для определения тренда
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            # Нормализуем наклон
            price_range = np.max(prices) - np.min(prices)
            if price_range > 0:
                normalized_slope = abs(slope) / price_range
                trend_strength = min(1.0, normalized_slope * 100)
            else:
                trend_strength = 0.0
            return float(trend_strength)
        except Exception:
            return 0.0

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Расчет волатильности."""
        try:
            if len(prices) < 2:
                return 0.0
            # Логарифмические доходности
            log_returns = np.diff(np.log(prices))
            # Годовая волатильность
            volatility = np.std(log_returns) * np.sqrt(252)
            # Нормализуем волатильность
            normalized_volatility = min(
                1.0, volatility / 0.5
            )  # 50% годовая волатильность как максимум
            return float(normalized_volatility)
        except Exception:
            return 0.0

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Расчет моментума."""
        try:
            if len(prices) < self.lookback_period:
                return 0.0
            # Моментум как отношение текущей цены к цене n периодов назад
            current_price = prices[-1]
            past_price = prices[-self.lookback_period]
            momentum = (
                (current_price - past_price) / past_price if past_price > 0 else 0.0
            )
            # Нормализуем моментум
            normalized_momentum = np.tanh(
                momentum * 10
            )  # Используем tanh для нормализации
            return float(normalized_momentum)
        except Exception:
            return 0.0

    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Расчет силы возврата к среднему."""
        try:
            if len(prices) < self.lookback_period:
                return 0.0
            # Скользящее среднее
            ma = np.mean(prices[-self.lookback_period :])
            current_price = prices[-1]
            # Отклонение от среднего
            deviation = (current_price - ma) / ma if ma > 0 else 0.0
            # Сила возврата к среднему (обратная величина отклонения)
            mean_reversion = 1.0 - min(1.0, abs(deviation))
            return float(mean_reversion)
        except Exception:
            return 0.0

    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Расчет тренда объема."""
        try:
            if len(volumes) < self.lookback_period:
                return 0.0
            # Тренд объема через линейную регрессию
            x = np.arange(len(volumes))
            slope, _ = np.polyfit(x, volumes, 1)
            # Нормализуем тренд объема
            volume_trend = (
                np.tanh(slope / np.mean(volumes) * 100) if np.mean(volumes) > 0 else 0.0
            )
            return float(volume_trend)
        except Exception:
            return 0.0

    def _calculate_price_range(
        self, high_prices: np.ndarray, low_prices: np.ndarray
    ) -> float:
        """Расчет диапазона цен."""
        try:
            if len(high_prices) < self.lookback_period:
                return 0.0
            # Средний диапазон за период
            ranges = (high_prices - low_prices) / low_prices
            avg_range = np.mean(ranges[-self.lookback_period :])
            # Нормализуем диапазон
            normalized_range = min(1.0, avg_range * 10)
            return float(normalized_range)
        except Exception:
            return 0.0

    def _determine_regime(
        self,
        trend_strength: float,
        volatility: float,
        momentum: float,
        mean_reversion: float,
        volume_trend: float,
    ) -> MarketRegime:
        """Определение рыночного режима."""
        try:
            # Высокая волатильность
            if volatility > 0.7:
                return MarketRegime.VOLATILITY
            # Сильный тренд
            if trend_strength > 0.6 and abs(momentum) > 0.5:
                return MarketRegime.TREND
            # Возврат к среднему
            if mean_reversion > 0.7 and volatility < 0.4:
                return MarketRegime.REVERSAL
            # Боковое движение
            if trend_strength < 0.3 and volatility < 0.5:
                return MarketRegime.SIDEWAYS
            # Аномалия (высокий объем при низкой волатильности)
            if abs(volume_trend) > 0.8 and volatility < 0.3:
                return MarketRegime.ANOMALY
            # Манипуляция (противоречивые сигналы)
            if (trend_strength > 0.5 and mean_reversion > 0.5) or (
                momentum > 0.5 and volume_trend < -0.5
            ):
                return MarketRegime.MANIPULATION
            # По умолчанию
            return MarketRegime.SIDEWAYS
        except Exception:
            return MarketRegime.SIDEWAYS

    def _calculate_regime_confidence(
        self,
        trend_strength: float,
        volatility: float,
        momentum: float,
        mean_reversion: float,
        volume_trend: float,
    ) -> float:
        """Расчет уверенности в определенном режиме."""
        try:
            # Берем максимальное значение среди индикаторов
            max_indicator = max(
                trend_strength,
                volatility,
                abs(momentum),
                mean_reversion,
                abs(volume_trend),
            )
            # Добавляем вес на основе согласованности индикаторов
            indicators = [
                trend_strength,
                volatility,
                abs(momentum),
                mean_reversion,
                abs(volume_trend),
            ]
            consistency = 1.0 - np.std(indicators)
            confidence = (max_indicator + consistency) / 2
            return float(min(1.0, max(0.0, float(confidence))))
        except Exception:
            return 0.5

    def _get_default_indicators(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Получение индикаторов по умолчанию."""
        default_indicators = {
            "trend_strength": 0.0,
            "volatility": 0.0,
            "momentum": 0.0,
            "mean_reversion": 0.0,
            "volume_trend": 0.0,
            "price_range": 0.0,
            "regime_confidence": 0.0,
            "regime_type": MarketRegime.SIDEWAYS.value,
            "regime_name": MarketRegime.SIDEWAYS.name,
            "indicators": None,
        }
        if error:
            default_indicators["error"] = error
        return default_indicators


class MarketRegimeClassifier:
    """Классификатор рыночных режимов."""

    def __init__(self, calculator: Optional[IIndicatorCalculator] = None):
        self.calculator = calculator or DefaultIndicatorCalculator()

    def classify_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Классификация рыночного режима."""
        try:
            indicators = self.calculator.calculate(market_data)
            return MarketRegime(indicators["regime_type"])
        except Exception:
            return MarketRegime.SIDEWAYS

    def get_regime_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Получение полного анализа режима."""
        try:
            return self.calculator.calculate(market_data)
        except Exception as e:
            return {
                "regime_type": MarketRegime.SIDEWAYS.value,
                "regime_name": MarketRegime.SIDEWAYS.name,
                "regime_confidence": 0.0,
                "error": str(e),
            }
