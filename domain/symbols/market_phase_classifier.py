"""
Классификатор фаз рынка - Production Ready
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd
from scipy import stats

from domain.types import MarketDataFrame
from domain.types import OrderBookData
from domain.exceptions import ConfigurationError
from domain.types import MarketPhaseClassifierProtocol
from domain.types import MarketPhase, MarketPhaseResult, ConfidenceValue

logger = logging.getLogger(__name__)


@dataclass
class PhaseDetectionConfig:
    """Конфигурация для определения фаз рынка."""

    # Периоды для анализа
    atr_period: int = 14
    vwap_period: int = 20
    volume_period: int = 20
    entropy_period: int = 50
    # Пороги для классификации
    atr_threshold: float = 0.02  # 2% от цены
    vwap_deviation_threshold: float = 0.01  # 1% от VWAP
    volume_slope_threshold: float = 0.1  # 10% изменение объема
    entropy_threshold: float = 0.7  # Порог энтропии для структуры
    # Пороги для фаз
    accumulation_volume_threshold: float = 0.5  # Низкий объем
    breakout_volume_threshold: float = 1.5  # Высокий объем
    exhaustion_volume_threshold: float = 2.0  # Очень высокий объем
    reversal_momentum_threshold: float = 0.3  # Сила разворота

    def __post_init__(self) -> None:
        """Валидация конфигурации."""
        if self.atr_period <= 0:
            raise ConfigurationError("ATR period must be positive", "atr_period", self.atr_period)
        if self.vwap_period <= 0:
            raise ConfigurationError("VWAP period must be positive", "vwap_period", self.vwap_period)
        if self.volume_period <= 0:
            raise ConfigurationError("Volume period must be positive", "volume_period", self.volume_period)
        if self.entropy_period <= 0:
            raise ConfigurationError("Entropy period must be positive", "entropy_period", self.entropy_period)
        if self.atr_threshold <= 0:
            raise ConfigurationError("ATR threshold must be positive", "atr_threshold", self.atr_threshold)
        if self.vwap_deviation_threshold <= 0:
            raise ConfigurationError("VWAP deviation threshold must be positive", "vwap_deviation_threshold", self.vwap_deviation_threshold)
        if self.volume_slope_threshold <= 0:
            raise ConfigurationError("Volume slope threshold must be positive", "volume_slope_threshold", self.volume_slope_threshold)
        if not 0.0 <= self.entropy_threshold <= 1.0:
            raise ConfigurationError("Entropy threshold must be between 0.0 and 1.0", "entropy_threshold", self.entropy_threshold)


class MarketPhaseClassifier(MarketPhaseClassifierProtocol):
    """Классификатор фаз рынка."""

    def __init__(self, config: Optional[PhaseDetectionConfig] = None):
        self.config = config or PhaseDetectionConfig()
        self.logger = logger

    def classify_market_phase(
        self, market_data: MarketDataFrame, order_book: Optional[OrderBookData] = None
    ) -> MarketPhaseResult:
        """
        Классифицировать текущую фазу рынка.
        
        Args:
            market_data: Рыночные данные
            order_book: Данные стакана заявок (опционально)
            
        Returns:
            MarketPhaseResult: Результат классификации
        """
        try:
            # Валидация входных данных
            self._validate_market_data(market_data)
            
            # Расчет индикаторов
            indicators = self._calculate_indicators(market_data)
            
            # Анализ структуры цены
            price_structure = self._analyze_price_structure(market_data, indicators)
            
            # Анализ паттернов объема
            volume_analysis = self._analyze_volume_pattern(market_data)
            
            # Анализ волатильности
            volatility_analysis = self._analyze_volatility(market_data, indicators)
            
            # Определение фазы
            phase, confidence = self._determine_phase(
                price_structure, volume_analysis, volatility_analysis, order_book
            )
            
            # Формирование результата
            result = MarketPhaseResult(
                phase=phase,
                confidence=ConfidenceValue(confidence),
                indicators=indicators,
                metadata={
                    "price_structure": price_structure,
                    "volume_analysis": volume_analysis,
                    "volatility_analysis": volatility_analysis,
                    "timestamp": market_data.timestamp.iloc[-1] if len(market_data.timestamp) > 0 else None,
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying market phase: {e}")
            # Возвращаем нейтральную фазу при ошибке
            return MarketPhaseResult(
                phase=MarketPhase.NO_STRUCTURE,
                confidence=ConfidenceValue(0.0),
                indicators={},
                metadata={},
            )

    def get_phase_description(self, phase: MarketPhase) -> str:
        """Получить описание фазы рынка."""
        descriptions = {
            MarketPhase.ACCUMULATION: "Накопление - крупные игроки накапливают позиции",
            MarketPhase.BREAKOUT_SETUP: "Подготовка к пробою - формирование паттерна",
            MarketPhase.BREAKOUT_ACTIVE: "Активный пробой - сильное движение цены",
            MarketPhase.EXHAUSTION: "Истощение - завершение тренда",
            MarketPhase.REVERSION_POTENTIAL: "Потенциал разворота - признаки смены тренда",
            MarketPhase.NO_STRUCTURE: "Отсутствие структуры - боковое движение",
        }
        return descriptions.get(phase, "Неизвестная фаза")

    def _validate_market_data(self, market_data: MarketDataFrame) -> None:
        """Валидация рыночных данных."""
        if market_data is None or market_data.empty:
            raise ValueError("Market data is empty or None")
        
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(market_data) < self.config.entropy_period:
            raise ValueError(f"Insufficient data points. Need at least {self.config.entropy_period}")

    def _calculate_indicators(self, market_data: MarketDataFrame) -> Dict[str, Any]:
        """Расчет технических индикаторов."""
        indicators = {}
        
        # ATR (Average True Range)
        high_low = pd.Series(market_data["high"] - market_data["low"])
        high_close = pd.Series(np.abs(market_data["high"] - market_data["close"].shift()))
        low_close = pd.Series(np.abs(market_data["low"] - market_data["close"].shift()))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=self.config.atr_period).mean()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (market_data["high"] + market_data["low"] + market_data["close"]) / 3
        volume_price = typical_price * market_data["volume"]
        cumulative_vp = volume_price.rolling(window=self.config.vwap_period).sum()
        cumulative_volume = market_data["volume"].rolling(window=self.config.vwap_period).sum()
        indicators["vwap"] = cumulative_vp / cumulative_volume
        
        # Volume indicators
        indicators["volume_sma"] = market_data["volume"].rolling(window=self.config.volume_period).mean()
        indicators["volume_ratio"] = market_data["volume"] / indicators["volume_sma"]
        
        # Price entropy
        indicators["price_entropy"] = self._calculate_price_entropy(market_data["close"])
        
        return indicators

    def _analyze_price_structure(
        self, market_data: MarketDataFrame, indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Анализ структуры цены."""
        analysis = {}
        
        # Тренд
        analysis["trend_strength"] = self._calculate_trend_strength(market_data["close"])
        
        # Отклонение от VWAP
        current_price = market_data["close"].iloc[-1]
        current_vwap = indicators["vwap"].iloc[-1]
        if current_vwap > 0:
            analysis["vwap_deviation"] = (current_price - current_vwap) / current_vwap
        else:
            analysis["vwap_deviation"] = 0.0
        
        # Уровни поддержки и сопротивления
        support_level = self._find_support_level(market_data["low"])
        resistance_level = self._find_resistance_level(market_data["high"])
        analysis["support_level"] = support_level if support_level is not None else 0.0
        analysis["resistance_level"] = resistance_level if resistance_level is not None else 0.0
        
        # Энтропия цены
        analysis["price_entropy"] = indicators["price_entropy"].iloc[-1]
        
        return analysis

    def _analyze_volume_pattern(self, market_data: MarketDataFrame) -> Dict[str, Any]:
        """Анализ паттернов объема."""
        analysis = {}
        
        # Тренд объема
        analysis["volume_trend"] = self._calculate_volume_trend(market_data["volume"])
        
        # Стабильность объема
        analysis["volume_stability"] = self._calculate_volume_stability(market_data["volume"])
        
        # Аномалии объема
        analysis["volume_anomalies"] = self._detect_volume_anomalies(market_data["volume"])
        
        # Отношение текущего объема к среднему
        volume_sma = market_data["volume"].rolling(window=self.config.volume_period).mean()
        analysis["volume_ratio"] = market_data["volume"].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        return analysis

    def _analyze_volatility(
        self, market_data: MarketDataFrame, indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Анализ волатильности."""
        analysis = {}
        
        # Тренд ATR
        analysis["atr_trend"] = self._calculate_atr_trend(indicators["atr"])
        
        # Сжатие волатильности
        analysis["volatility_compression"] = self._detect_volatility_compression(indicators["atr"])
        
        # Текущая волатильность относительно цены
        current_atr = indicators["atr"].iloc[-1]
        current_price = market_data["close"].iloc[-1]
        if current_price > 0:
            analysis["volatility_ratio"] = current_atr / current_price
        else:
            analysis["volatility_ratio"] = 0.0
        
        return analysis

    def _determine_phase(
        self,
        price_structure: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        volatility_analysis: Dict[str, Any],
        order_book: Optional[OrderBookData],
    ) -> Tuple[MarketPhase, float]:
        """Определение фазы рынка."""
        # Извлекаем ключевые метрики
        volume_ratio = volume_analysis.get("volume_ratio", 1.0)
        volume_trend = volume_analysis.get("volume_trend", 0.0)
        price_entropy = price_structure.get("price_entropy", 0.5)
        vwap_deviation = price_structure.get("vwap_deviation", 0.0)
        volatility_compression = volatility_analysis.get("volatility_compression", 0.0)
        
        # Оценка каждой фазы
        scores = {}
        scores[MarketPhase.ACCUMULATION] = self._score_accumulation(
            volume_ratio, volume_trend, price_entropy, vwap_deviation, volatility_compression
        )
        scores[MarketPhase.BREAKOUT_SETUP] = self._score_breakout_setup(
            volume_ratio, volume_trend, price_entropy, vwap_deviation, volatility_compression
        )
        scores[MarketPhase.BREAKOUT_ACTIVE] = self._score_breakout_active(
            volume_ratio, volume_trend, price_entropy, vwap_deviation
        )
        scores[MarketPhase.EXHAUSTION] = self._score_exhaustion(
            volume_ratio, volume_trend, price_entropy
        )
        scores[MarketPhase.REVERSION_POTENTIAL] = self._score_reversion_potential(
            volume_ratio, volume_trend, price_entropy, vwap_deviation
        )
        scores[MarketPhase.NO_STRUCTURE] = self._score_no_structure(
            volume_ratio, price_entropy, vwap_deviation
        )
        
        # Выбор фазы с максимальным скором
        best_phase = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[best_phase]
        
        return best_phase, confidence

    def _score_accumulation(
        self,
        volume_ratio: float,
        volume_trend: float,
        price_entropy: float,
        vwap_deviation: float,
        volatility_compression: float,
    ) -> float:
        """Оценка фазы накопления."""
        score = 0.0
        
        # Низкий объем
        if volume_ratio < self.config.accumulation_volume_threshold:
            score += 0.3
        
        # Низкая энтропия (структурированное движение)
        if price_entropy < self.config.entropy_threshold:
            score += 0.2
        
        # Сжатие волатильности
        if volatility_compression > 0.5:
            score += 0.2
        
        # Небольшое отклонение от VWAP
        if abs(vwap_deviation) < self.config.vwap_deviation_threshold:
            score += 0.2
        
        # Слабый тренд объема
        if abs(volume_trend) < self.config.volume_slope_threshold:
            score += 0.1
        
        return min(score, 1.0)

    def _score_breakout_setup(
        self,
        volume_ratio: float,
        volume_trend: float,
        price_entropy: float,
        vwap_deviation: float,
        volatility_compression: float,
    ) -> float:
        """Оценка фазы подготовки к пробою."""
        score = 0.0
        
        # Умеренный рост объема
        if 0.8 < volume_ratio < 1.5:
            score += 0.3
        
        # Сжатие волатильности
        if volatility_compression > 0.6:
            score += 0.3
        
        # Низкая энтропия
        if price_entropy < self.config.entropy_threshold:
            score += 0.2
        
        # Небольшое отклонение от VWAP
        if abs(vwap_deviation) < self.config.vwap_deviation_threshold * 2:
            score += 0.2
        
        return min(score, 1.0)

    def _score_breakout_active(
        self,
        volume_ratio: float,
        volume_trend: float,
        price_entropy: float,
        vwap_deviation: float,
    ) -> float:
        """Оценка активной фазы пробоя."""
        score = 0.0
        
        # Высокий объем
        if volume_ratio > self.config.breakout_volume_threshold:
            score += 0.4
        
        # Сильный тренд объема
        if abs(volume_trend) > self.config.volume_slope_threshold * 2:
            score += 0.3
        
        # Высокая энтропия (хаотичное движение)
        if price_entropy > self.config.entropy_threshold:
            score += 0.2
        
        # Значительное отклонение от VWAP
        if abs(vwap_deviation) > self.config.vwap_deviation_threshold * 3:
            score += 0.1
        
        return min(score, 1.0)

    def _score_exhaustion(
        self, volume_ratio: float, volume_trend: float, price_entropy: float
    ) -> float:
        """Оценка фазы истощения."""
        score = 0.0
        
        # Очень высокий объем
        if volume_ratio > self.config.exhaustion_volume_threshold:
            score += 0.4
        
        # Высокая энтропия
        if price_entropy > self.config.entropy_threshold:
            score += 0.3
        
        # Снижение тренда объема
        if volume_trend < -self.config.volume_slope_threshold:
            score += 0.3
        
        return min(score, 1.0)

    def _score_reversion_potential(
        self,
        volume_ratio: float,
        volume_trend: float,
        price_entropy: float,
        vwap_deviation: float,
    ) -> float:
        """Оценка потенциала разворота."""
        score = 0.0
        
        # Умеренный объем
        if 0.8 < volume_ratio < 1.8:
            score += 0.3
        
        # Высокая энтропия
        if price_entropy > self.config.entropy_threshold:
            score += 0.3
        
        # Значительное отклонение от VWAP
        if abs(vwap_deviation) > self.config.vwap_deviation_threshold * 2:
            score += 0.2
        
        # Изменение тренда объема
        if abs(volume_trend) > self.config.volume_slope_threshold:
            score += 0.2
        
        return min(score, 1.0)

    def _score_no_structure(
        self, volume_ratio: float, price_entropy: float, vwap_deviation: float
    ) -> float:
        """Оценка отсутствия структуры."""
        score = 0.0
        
        # Нормальный объем
        if 0.5 < volume_ratio < 1.5:
            score += 0.3
        
        # Высокая энтропия
        if price_entropy > self.config.entropy_threshold:
            score += 0.3
        
        # Небольшое отклонение от VWAP
        if abs(vwap_deviation) < self.config.vwap_deviation_threshold:
            score += 0.4
        
        return min(score, 1.0)

    def _find_support_level(self, low: pd.Series) -> Optional[float]:
        """Поиск уровня поддержки."""
        try:
            if len(low) < 20:
                return None
            
            # Ищем локальные минимумы
            window = 5
            local_mins = []
            for i in range(window, len(low) - window):
                if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, window+1)) and \
                   all(low.iloc[i] <= low.iloc[i+j] for j in range(1, window+1)):
                    local_mins.append(low.iloc[i])
            
            if local_mins:
                # Возвращаем наиболее частый уровень
                return float(np.median(local_mins))
            return None
        except Exception as e:
            self.logger.warning(f"Error finding support level: {e}")
            return None

    def _find_resistance_level(self, high: pd.Series) -> Optional[float]:
        """Поиск уровня сопротивления."""
        try:
            if len(high) < 20:
                return None
            
            # Ищем локальные максимумы
            window = 5
            local_maxs = []
            for i in range(window, len(high) - window):
                if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, window+1)) and \
                   all(high.iloc[i] >= high.iloc[i+j] for j in range(1, window+1)):
                    local_maxs.append(high.iloc[i])
            
            if local_maxs:
                # Возвращаем наиболее частый уровень
                return float(np.median(local_maxs))
            return None
        except Exception as e:
            self.logger.warning(f"Error finding resistance level: {e}")
            return None

    def _calculate_trend_strength(self, close: pd.Series) -> float:
        """Расчет силы тренда."""
        try:
            if len(close) < 10:
                return 0.0
            
            # Линейная регрессия
            x = np.arange(len(close))
            slope, _, r_value, _, _ = stats.linregress(x, close)
            
            # Нормализация к [-1, 1]
            trend = np.sign(slope) * abs(r_value) if not np.isnan(r_value) else 0.0
            return float(np.clip(trend, -1.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating trend strength: {e}")
            return 0.0

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Расчет тренда объема."""
        try:
            if len(volume) < 10:
                return 0.0
            x = np.arange(len(volume))
            volume_values = volume.to_numpy() if hasattr(volume, 'to_numpy') else np.array(volume)
            # Убеждаемся, что volume_values является numpy массивом
            volume_values = np.asarray(volume_values)
            slope, _, r_value, _, _ = stats.linregress(x, volume_values)
            trend = np.sign(slope) * abs(r_value) if not np.isnan(r_value) else 0.0
            return float(np.clip(trend, -1.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating volume trend: {e}")
            return 0.0

    def _calculate_volume_stability(self, volume: pd.Series) -> float:
        """Расчет стабильности объема."""
        try:
            if callable(volume):
                volume = volume()
            if len(volume) < 10:
                return 0.0
            volume_values = volume.to_numpy() if hasattr(volume, 'to_numpy') else np.array(volume)

            # Убеждаемся, что volume_values является numpy массивом
            volume_values = np.asarray(volume_values)
            # Коэффициент вариации
            mean_volume = np.mean(volume_values)
            if mean_volume == 0:
                return 0.0
            cv = np.std(volume_values) / mean_volume
            # Стабильность = 1 / (1 + CV)
            stability = 1.0 / (1.0 + cv)
            return float(np.clip(stability, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating volume stability: {e}")
            return 0.0

    def _detect_volume_anomalies(self, volume: pd.Series) -> float:
        """Обнаружение аномалий объема."""
        try:
            if len(volume) < 20:
                return 1.0
            # Z-score для текущего объема
            mean_volume = volume.rolling(window=20).mean().iloc[-1]
            std_volume = volume.rolling(window=20).std().iloc[-1]
            if std_volume == 0:
                return 1.0
            current_volume = volume.iloc[-1]
            z_score = abs(current_volume - mean_volume) / std_volume
            # Нормализация к [0, 1]
            anomaly_ratio = min(z_score / 3.0, 1.0)
            return float(anomaly_ratio)
        except Exception as e:
            self.logger.warning(f"Error detecting volume anomalies: {e}")
            return 1.0

    def _calculate_atr_trend(self, atr: pd.Series) -> float:
        """Расчет тренда ATR."""
        try:
            if len(atr) < 10:
                return 0.0
            # Линейная регрессия для ATR
            x = np.arange(len(atr))
            atr_clean = atr.dropna()
            if len(atr_clean) < 10:
                return 0.0
            atr_values = atr_clean.to_numpy() if hasattr(atr_clean, 'to_numpy') else np.array(atr_clean)

            # Убеждаемся, что atr_values является numpy массивом
            atr_values = np.asarray(atr_values)
            slope, _, r_value, _, _ = stats.linregress(x[:len(atr_clean)], atr_values)
            # Нормализация к [-1, 1]
            trend = np.sign(slope) * abs(r_value) if not np.isnan(r_value) else 0.0
            return float(np.clip(trend, -1.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating ATR trend: {e}")
            return 0.0

    def _detect_volatility_compression(self, atr: pd.Series) -> float:
        """Обнаружение сжатия волатильности."""
        try:
            if len(atr) < 20:
                return 0.0
            # Сравнение текущего ATR с историческим
            current_atr = atr.iloc[-1]
            historical_atr = atr.rolling(window=20).mean().iloc[-1]
            if historical_atr == 0:
                return 0.0
            # Нормализация к [0, 1]
            compression = current_atr / historical_atr
            compression = 1.0 - min(compression, 1.0)
            return float(compression)
        except Exception as e:
            self.logger.warning(f"Error detecting volatility compression: {e}")
            return 0.0

    def _calculate_price_entropy(self, close: pd.Series) -> pd.Series:
        """Расчет энтропии цены."""
        try:
            # Используем изменения цены
            price_changes = close.pct_change().dropna()
            
            def entropy(x: np.ndarray) -> float:
                if len(x) < 2:
                    return 0.0
                # Создаем гистограмму
                hist, _ = np.histogram(x, bins=min(10, len(x)))
                hist = hist[hist > 0]
                if len(hist) < 2:
                    return 0.0
                # Вычисляем энтропию
                p = hist / hist.sum()
                return float(-np.sum(p * np.log2(p)))
            
            # Применяем скользящее окно
            return price_changes.rolling(window=self.config.entropy_period).apply(entropy)
        except Exception as e:
            self.logger.warning(f"Error calculating price entropy: {e}")
            return pd.Series([0.5] * len(close)) 