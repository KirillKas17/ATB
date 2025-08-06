# -*- coding: utf-8 -*-
"""Neural Noise Divergence Analysis for Order Book Intelligence."""
import time
from typing import Any, Dict, Final, List, Optional, Tuple

from shared.numpy_utils import np
from loguru import logger
from scipy import stats
from scipy.signal import welch

from domain.type_definitions.intelligence_types import (
    AnalysisMetadata,
    NoiseAnalysisConfig,
    NoiseAnalysisResult,
    NoiseMetrics,
    NoiseType,
    OrderBookSnapshot,
)
from domain.value_objects.timestamp import Timestamp

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_CONFIG: Final[NoiseAnalysisConfig] = NoiseAnalysisConfig()
ALGORITHM_VERSION: Final[str] = "2.0.0"
MIN_DATA_POINTS: Final[int] = 20
MAX_FRACTAL_DIMENSION: Final[float] = 2.0
MIN_FRACTAL_DIMENSION: Final[float] = 1.0


# =============================================================================
# ENHANCED NOISE ANALYZER
# =============================================================================
class NoiseAnalyzer:
    """Продвинутый анализатор нейронного шума ордербука."""

    def __init__(
        self,
        config: Optional[NoiseAnalysisConfig] = None,
        enable_advanced_metrics: bool = True,
        enable_frequency_analysis: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.enable_advanced_metrics = enable_advanced_metrics
        self.enable_frequency_analysis = enable_frequency_analysis
        # Буфер для исторических данных
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.spread_history: List[float] = []
        self.imbalance_history: List[float] = []
        # Статистика анализа
        self.statistics: Dict[str, Any] = {
            "total_analyses": 0,
            "synthetic_noise_detections": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info(
            f"NoiseAnalyzer initialized with config: {self.config}, "
            f"advanced_metrics: {enable_advanced_metrics}, "
            f"frequency_analysis: {enable_frequency_analysis}"
        )

    def _extract_time_series(
        self, order_book: OrderBookSnapshot
    ) -> Dict[str, np.ndarray]:
        """Извлечение временных рядов из ордербука с расширенной обработкой."""
        try:
            # Цены (bid и ask)
            bid_prices = [float(price.value) for price, _ in order_book.bids]
            ask_prices = [float(price.value) for price, _ in order_book.asks]
            # Объемы
            bid_volumes = [float(vol.value) for _, vol in order_book.bids]
            ask_volumes = [float(vol.value) for _, vol in order_book.asks]
            # Объединяем bid и ask данные
            all_prices = bid_prices + ask_prices
            all_volumes = bid_volumes + ask_volumes
            # Дополнительные метрики
            spreads = []
            imbalances = []
            if bid_prices and ask_prices:
                for i in range(min(len(bid_prices), len(ask_prices))):
                    spread = ask_prices[i] - bid_prices[i]
                    spreads.append(spread)
                    if i < len(bid_volumes) and i < len(ask_volumes):
                        total_vol = bid_volumes[i] + ask_volumes[i]
                        if total_vol > 0:
                            imbalance = (bid_volumes[i] - ask_volumes[i]) / total_vol
                            imbalances.append(imbalance)
            # Создаем временные ряды
            time_series = {
                "prices": np.array(all_prices) if all_prices else np.array([]),
                "volumes": np.array(all_volumes) if all_volumes else np.array([]),
                "bid_prices": np.array(bid_prices) if bid_prices else np.array([]),
                "ask_prices": np.array(ask_prices) if ask_prices else np.array([]),
                "bid_volumes": np.array(bid_volumes) if bid_volumes else np.array([]),
                "ask_volumes": np.array(ask_volumes) if ask_volumes else np.array([]),
                "spreads": np.array(spreads) if spreads else np.array([]),
                "imbalances": np.array(imbalances) if imbalances else np.array([]),
            }
            return time_series
        except Exception as e:
            logger.error(f"Error extracting time series: {e}")
            return {
                "prices": np.array([]),
                "volumes": np.array([]),
                "bid_prices": np.array([]),
                "ask_prices": np.array([]),
                "bid_volumes": np.array([]),
                "ask_volumes": np.array([]),
                "spreads": np.array([]),
                "imbalances": np.array([]),
            }

    def _update_history(self, order_book: OrderBookSnapshot) -> None:
        """Обновление исторических данных с расширенными метриками."""
        try:
            mid_price = float(order_book.get_mid_price().value)
            total_volume = float(order_book.get_total_volume().value)
            spread = float(order_book.get_spread().value)
            imbalance = order_book.get_volume_imbalance()
            self.price_history.append(mid_price)
            self.volume_history.append(total_volume)
            self.spread_history.append(spread)
            self.imbalance_history.append(imbalance)
            # Ограничиваем размер истории
            if len(self.price_history) > self.config.window_size:
                self.price_history.pop(0)
                self.volume_history.pop(0)
                self.spread_history.pop(0)
                self.imbalance_history.pop(0)
        except Exception as e:
            logger.error(f"Error updating history: {e}")

    def _compute_higuchi_fractal_dimension(self, data: np.ndarray) -> float:
        """
        Вычисление фрактальной размерности по методу Хигучи.
        Args:
            data: Временной ряд данных
        Returns:
            float: Фрактальная размерность
        """
        try:
            if len(data) < MIN_DATA_POINTS:
                return 1.0
            # Нормализация данных
            if np.std(data) > 0:
                data = (data - np.mean(data)) / np.std(data)
            # Параметры для вычисления
            k_max = min(20, len(data) // 4)
            k_values = np.arange(1, k_max + 1)
            # Вычисляем L(k) для каждого k
            L_values = []
            for k in k_values:
                L_k = 0.0
                for m in range(k):
                    # Вычисляем L(m,k)
                    L_mk = 0.0
                    N_mk = int((len(data) - m - 1) / k)
                    if N_mk > 0:
                        for i in range(N_mk):
                            idx1 = m + i * k
                            idx2 = m + (i + 1) * k
                            if idx2 < len(data):
                                L_mk += abs(data[idx2] - data[idx1])
                        L_mk = L_mk * (len(data) - 1) / (k * k * N_mk)
                        L_k += L_mk
                L_k /= k
                L_values.append(L_k)
            # Линейная регрессия log(L(k)) vs log(1/k)
            if len(L_values) > 1 and all(l > 0 for l in L_values):
                log_L = np.log(L_values)
                log_k = np.log(1.0 / k_values)
                # Вычисляем наклон (фрактальная размерность)
                slope, _, r_value, _, _ = stats.linregress(log_k, log_L)
                # Проверяем качество регрессии
                if r_value > 0.8:
                    fractal_dimension = float(-slope)
                    return max(
                        MIN_FRACTAL_DIMENSION,
                        min(MAX_FRACTAL_DIMENSION, fractal_dimension),
                    )
            return 1.0
        except Exception as e:
            logger.error(f"Error computing Higuchi fractal dimension: {e}")
            return 1.0

    def _compute_sample_entropy(
        self, data: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """
        Вычисление sample entropy.
        Args:
            data: Временной ряд
            m: Размер шаблона
            r: Порог толерантности
        Returns:
            float: Sample entropy
        """
        try:
            if len(data) < m + 2:
                return 0.0
            # Нормализация
            if np.std(data) > 0:
                data = (data - np.mean(data)) / np.std(data)
            r = r * np.std(data)
            # Подсчет совпадений для m и m+1
            A = 0  # совпадения для m+1
            B = 0  # совпадения для m
            for i in range(len(data) - m):
                for j in range(i + 1, len(data) - m):
                    # Проверяем совпадение для m
                    if all(abs(data[i + k] - data[j + k]) < r for k in range(m)):
                        B += 1
                        # Проверяем совпадение для m+1
                        if abs(data[i + m] - data[j + m]) < r:
                            A += 1
            if B == 0:
                return 0.0
            return float(-np.log(A / B)) if A > 0 else float('inf')
        except Exception as e:
            logger.error(f"Error computing sample entropy: {e}")
            return 0.0

    def _compute_spectral_entropy(self, data: np.ndarray) -> float:
        """
        Вычисление спектральной энтропии.
        Args:
            data: Временной ряд
        Returns:
            float: Спектральная энтропия
        """
        try:
            if len(data) < 10:
                return 0.0
            # Вычисляем спектр мощности
            freqs, psd = welch(data, nperseg=min(256, len(data) // 4))
            # Нормализуем спектр
            psd = psd / np.sum(psd)
            # Вычисляем энтропию
            entropy = -np.sum(psd * np.log2(psd + np.finfo(float).eps))
            return float(entropy)
        except Exception as e:
            logger.error(f"Error computing spectral entropy: {e}")
            return 0.0

    def compute_fractal_dimension(self, order_book: OrderBookSnapshot) -> float:
        """
        Вычисление фрактальной размерности ордербука.
        Args:
            order_book: Снимок ордербука
        Returns:
            float: Фрактальная размерность
        """
        try:
            # Обновляем историю
            self._update_history(order_book)
            if len(self.price_history) < MIN_DATA_POINTS:
                return 1.0
            # Вычисляем фрактальную размерность для цен
            price_data = np.array(self.price_history)
            fractal_dimension = self._compute_higuchi_fractal_dimension(price_data)
            return fractal_dimension
        except Exception as e:
            logger.error(f"Error computing fractal dimension: {e}")
            return 1.0

    def compute_entropy(self, order_book: OrderBookSnapshot) -> float:
        """
        Вычисление энтропии ордербука.
        Args:
            order_book: Снимок ордербука
        Returns:
            float: Энтропия
        """
        try:
            # Обновляем историю
            self._update_history(order_book)
            if len(self.price_history) < MIN_DATA_POINTS:
                return 0.0
            # Вычисляем различные типы энтропии
            price_data = np.array(self.price_history)
            # Sample entropy
            sample_entropy = self._compute_sample_entropy(price_data)
            # Спектральная энтропия
            spectral_entropy = self._compute_spectral_entropy(price_data)
            # Комбинированная энтропия
            combined_entropy = (sample_entropy + spectral_entropy) / 2.0
            return float(combined_entropy)
        except Exception as e:
            logger.error(f"Error computing entropy: {e}")
            return 0.0

    def _classify_noise_type(self, fd: float, entropy: float) -> NoiseType:
        """
        Классификация типа шума на основе фрактальной размерности и энтропии.
        Args:
            fd: Фрактальная размерность
            entropy: Энтропия
        Returns:
            NoiseType: Тип шума
        """
        try:
            # Определяем вероятность синтетического шума
            synthetic_prob = 0.0
            # Фактор фрактальной размерности
            if fd < self.config.fractal_dimension_lower:
                synthetic_prob += 0.4
            elif fd > self.config.fractal_dimension_upper:
                synthetic_prob += 0.3
            else:
                synthetic_prob += 0.1
            # Фактор энтропии
            if entropy < self.config.entropy_threshold:
                synthetic_prob += 0.4
            else:
                synthetic_prob += 0.1
            # Дополнительные факторы
            if self.enable_advanced_metrics and len(self.price_history) > 50:
                # Анализ периодичности
                price_data = np.array(self.price_history)
                autocorr = np.correlate(price_data, price_data, mode="full")
                autocorr = autocorr[len(autocorr) // 2 :]
                # Ищем пики автокорреляции
                peaks = []
                for i in range(1, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                        peaks.append(i)
                if len(peaks) > 3:  # Слишком много пиков - возможен синтетический шум
                    synthetic_prob += 0.2
            # Классификация
            if synthetic_prob > 0.7:
                return NoiseType.SYNTHETIC
            elif synthetic_prob > 0.3:
                return NoiseType.MIXED
            else:
                return NoiseType.NATURAL
        except Exception as e:
            logger.error(f"Error classifying noise type: {e}")
            return NoiseType.UNKNOWN

    def is_synthetic_noise(self, fd: float, entropy: float) -> bool:
        """
        Определение синтетического шума.
        Args:
            fd: Фрактальная размерность
            entropy: Энтропия
        Returns:
            bool: True если шум синтетический
        """
        try:
            noise_type = self._classify_noise_type(fd, entropy)
            return noise_type == NoiseType.SYNTHETIC
        except Exception as e:
            logger.error(f"Error determining synthetic noise: {e}")
            return False

    def compute_confidence(self, fd: float, entropy: float) -> float:
        """
        Вычисление уверенности в анализе.
        Args:
            fd: Фрактальная размерность
            entropy: Энтропия
        Returns:
            float: Уверенность (0.0 - 1.0)
        """
        try:
            # Базовая уверенность на основе количества данных
            data_confidence = min(
                1.0, len(self.price_history) / self.config.min_data_points
            )
            # Уверенность на основе качества фрактальной размерности
            fd_confidence = 1.0 - abs(fd - 1.5) / 0.5  # 1.5 - идеальная FD
            fd_confidence = max(0.0, min(1.0, fd_confidence))
            # Уверенность на основе энтропии
            entropy_confidence = min(1.0, entropy / 2.0)  # Нормализуем энтропию
            # Комбинированная уверенность
            confidence = (
                data_confidence * 0.4 + fd_confidence * 0.3 + entropy_confidence * 0.3
            )
            return max(0.0, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error computing confidence: {e}")
            return 0.0

    def analyze_noise(self, order_book: OrderBookSnapshot) -> NoiseAnalysisResult:
        """
        Комплексный анализ нейронного шума.
        Args:
            order_book: Снимок ордербука
        Returns:
            NoiseAnalysisResult: Результат анализа
        """
        start_time = time.time()
        try:
            # Вычисляем основные метрики
            fractal_dimension = self.compute_fractal_dimension(order_book)
            entropy = self.compute_entropy(order_book)
            # Определяем тип шума
            noise_type = self._classify_noise_type(fractal_dimension, entropy)
            is_synthetic = noise_type == NoiseType.SYNTHETIC
            # Вычисляем уверенность
            confidence = self.compute_confidence(fractal_dimension, entropy)
            # Дополнительные метрики
            additional_metrics = {}
            if self.enable_advanced_metrics:
                additional_metrics.update(
                    {
                        "price_volatility": (
                            float(np.std(self.price_history))
                            if self.price_history
                            else 0.0
                        ),
                        "volume_volatility": (
                            float(np.std(self.volume_history))
                            if self.volume_history
                            else 0.0
                        ),
                        "spread_volatility": (
                            float(np.std(self.spread_history))
                            if self.spread_history
                            else 0.0
                        ),
                        "imbalance_volatility": (
                            float(np.std(self.imbalance_history))
                            if self.imbalance_history
                            else 0.0
                        ),
                    }
                )
                if self.enable_frequency_analysis and len(self.price_history) > 50:
                    price_data = np.array(self.price_history)
                    freqs, psd = welch(
                        price_data, nperseg=min(256, len(price_data) // 4)
                    )
                    dominant_freq = freqs[np.argmax(psd)]
                    additional_metrics["dominant_frequency"] = float(dominant_freq)
                    additional_metrics["spectral_entropy"] = (
                        self._compute_spectral_entropy(price_data)
                    )
            # Метаданные для анализа
            metadata: AnalysisMetadata = {
                "data_points": len(self.price_history),
                "confidence": confidence,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "algorithm_version": ALGORITHM_VERSION,
                "parameters": {
                    "fractal_dimension_lower": self.config.fractal_dimension_lower,
                    "fractal_dimension_upper": self.config.fractal_dimension_upper,
                    "entropy_threshold": self.config.entropy_threshold,
                    "min_data_points": self.config.min_data_points,
                    "window_size": self.config.window_size,
                },
                "quality_metrics": additional_metrics,
            }
            # Метрики шума
            noise_metrics: NoiseMetrics = {
                "fractal_dimension": fractal_dimension,
                "entropy": entropy,
                "noise_type": noise_type,
                "synthetic_probability": 0.7 if is_synthetic else 0.3,
                "natural_probability": 0.3 if is_synthetic else 0.7,
            }
            result = NoiseAnalysisResult(
                fractal_dimension=fractal_dimension,
                entropy=entropy,
                is_synthetic_noise=is_synthetic,
                confidence=confidence,
                metadata=metadata,
                timestamp=Timestamp.now(),
                noise_type=noise_type,
                metrics=noise_metrics,
            )
            # Обновляем статистику
            self.statistics["total_analyses"] += 1
            if is_synthetic:
                self.statistics["synthetic_noise_detections"] += 1
            self.statistics["average_processing_time_ms"] = (
                self.statistics["average_processing_time_ms"]
                * (self.statistics["total_analyses"] - 1)
                + metadata["processing_time_ms"]
            ) / self.statistics["total_analyses"]
            self.statistics["last_analysis_timestamp"] = Timestamp.now()
            return result
        except Exception as e:
            logger.error(f"Error analyzing noise: {e}")
            # Возвращаем дефолтный результат в случае ошибки
            return NoiseAnalysisResult(
                fractal_dimension=1.0,
                entropy=0.0,
                is_synthetic_noise=False,
                confidence=0.0,
                metadata={
                    "data_points": 0,
                    "confidence": 0.0,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "algorithm_version": ALGORITHM_VERSION,
                    "parameters": {},
                    "quality_metrics": {"error": float(0.0)},
                },
                timestamp=Timestamp.now(),
                noise_type=NoiseType.UNKNOWN,
                metrics={
                    "fractal_dimension": 1.0,
                    "entropy": 0.0,
                    "noise_type": NoiseType.UNKNOWN,
                    "synthetic_probability": 0.0,
                    "natural_probability": 0.0,
                },
            )

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Получение расширенной статистики анализа."""
        return {
            **self.statistics,
            "config": {
                "fractal_dimension_lower": self.config.fractal_dimension_lower,
                "fractal_dimension_upper": self.config.fractal_dimension_upper,
                "entropy_threshold": self.config.entropy_threshold,
                "min_data_points": self.config.min_data_points,
                "window_size": self.config.window_size,
                "confidence_threshold": self.config.confidence_threshold,
            },
            "history_status": {
                "price_history_length": len(self.price_history),
                "volume_history_length": len(self.volume_history),
                "spread_history_length": len(self.spread_history),
                "imbalance_history_length": len(self.imbalance_history),
            },
            "advanced_metrics_enabled": self.enable_advanced_metrics,
            "frequency_analysis_enabled": self.enable_frequency_analysis,
        }

    def reset_history(self) -> None:
        """Сброс исторических данных."""
        self.price_history.clear()
        self.volume_history.clear()
        self.spread_history.clear()
        self.imbalance_history.clear()
        logger.info("Noise analyzer history reset")

    def reset_statistics(self) -> None:
        """Сброс статистики."""
        self.statistics = {
            "total_analyses": 0,
            "synthetic_noise_detections": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info("Noise analyzer statistics reset")
