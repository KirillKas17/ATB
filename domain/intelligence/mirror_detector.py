# -*- coding: utf-8 -*-
"""Advanced Mirror Pattern Detection for Market Intelligence."""
import time
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from domain.types.intelligence_types import (
    AnalysisMetadata,
    CorrelationMatrix,
    CorrelationMetrics,
    MirrorDetectionConfig,
    MirrorSignal,
)
from domain.value_objects.timestamp import Timestamp

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_CONFIG: Final[MirrorDetectionConfig] = MirrorDetectionConfig()
ALGORITHM_VERSION: Final[str] = "2.0.0"
MIN_SAMPLE_SIZE: Final[int] = 30
MAX_LAG_DEFAULT: Final[int] = 10
MIN_CORRELATION_DEFAULT: Final[float] = 0.7


# =============================================================================
# ENHANCED MIRROR DETECTOR
# =============================================================================
class MirrorDetector:
    """Продвинутый детектор зеркальных паттернов на рынке."""

    def __init__(
        self,
        config: Optional[MirrorDetectionConfig] = None,
        enable_advanced_metrics: bool = True,
        enable_clustering: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.enable_advanced_metrics = enable_advanced_metrics
        self.enable_clustering = enable_clustering
        # Статистика детектора
        self.statistics: Dict[str, Any] = {
            "total_analyses": 0,
            "mirror_signals_detected": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info(
            f"MirrorDetector initialized with config: {self.config}, "
            f"advanced_metrics: {enable_advanced_metrics}, "
            f"clustering: {enable_clustering}"
        )

    def _preprocess_series(self, series: pd.Series) -> pd.Series:
        """Предобработка временного ряда."""
        try:
            # Удаляем NaN значения
            series = series.dropna()
            if len(series) < MIN_SAMPLE_SIZE:
                return series
            # Нормализация данных
            if self.config.normalize_data:
                mean_val = series.mean()
                std_val = series.std()
                if std_val > 0:
                    series = (series - mean_val) / std_val
            # Удаление тренда
            if self.config.remove_trend:
                # Линейная детрендизация
                x = np.arange(len(series))
                slope, intercept = np.polyfit(x, np.array(series.values), 1)
                trend = slope * x + intercept
                series = series - trend
            return series
        except Exception as e:
            logger.error(f"Error preprocessing series: {e}")
            return series

    def _compute_correlation_with_lag(
        self, series1: pd.Series, series2: pd.Series, lag: int
    ) -> Tuple[float, float]:
        """
        Вычисление корреляции с учетом лага.
        Args:
            series1: Первый временной ряд
            series2: Второй временной ряд
            lag: Лаг (положительный = series2 отстает, отрицательный = series1 отстает)
        Returns:
            Tuple[float, float]: (корреляция, p-value)
        """
        try:
            if len(series1) != len(series2):
                raise ValueError("Series must have the same length")
            if abs(lag) >= len(series1):
                return 0.0, 1.0
            # Вычисляем корреляцию с учетом лага
            if lag == 0:
                # Преобразуем pandas Series в numpy массивы для совместимости с pearsonr
                series1_array = series1.to_numpy() if hasattr(series1, 'to_numpy') else np.array(series1)
                series2_array = series2.to_numpy() if hasattr(series2, 'to_numpy') else np.array(series2)
                pearson_result = stats.pearsonr(series1_array, series2_array)
                correlation = float(pearson_result.statistic)
                p_value = float(pearson_result.pvalue)
            elif lag > 0:
                series1_array = series1.to_numpy() if hasattr(series1, 'to_numpy') else np.array(series1)
                series2_array = series2.to_numpy() if hasattr(series2, 'to_numpy') else np.array(series2)
                pearson_result = stats.pearsonr(
                    series1_array[lag:], series2_array[:-lag]
                )
                correlation = float(pearson_result.statistic)
                p_value = float(pearson_result.pvalue)
            else:
                series1_array = series1.to_numpy() if hasattr(series1, 'to_numpy') else np.array(series1)
                series2_array = series2.to_numpy() if hasattr(series2, 'to_numpy') else np.array(series2)
                pearson_result = stats.pearsonr(
                    series1_array[:-lag], series2_array[-lag:]
                )
                correlation = float(pearson_result.statistic)
                p_value = float(pearson_result.pvalue)
            return float(correlation), float(p_value)
        except Exception as e:
            logger.error(f"Error computing correlation with lag: {e}")
            return 0.0, 1.0

    def _compute_confidence(
        self, correlation: float, p_value: float, sample_size: int, lag: int
    ) -> float:
        """
        Вычисление уверенности в корреляции.
        Args:
            correlation: Коэффициент корреляции
            p_value: P-value
            sample_size: Размер выборки
            lag: Лаг
        Returns:
            float: Уверенность (0.0 - 1.0)
        """
        try:
            # Базовая уверенность на основе размера выборки
            sample_confidence = min(1.0, sample_size / 100.0)
            # Уверенность на основе p-value
            p_confidence = 1.0 - p_value if p_value <= 1.0 else 0.0
            # Уверенность на основе силы корреляции
            correlation_confidence = abs(correlation)
            # Уверенность на основе лага (меньший лаг = большая уверенность)
            lag_confidence = max(0.0, 1.0 - abs(lag) / 10.0)
            # Комбинированная уверенность
            confidence = (
                sample_confidence * 0.2
                + p_confidence * 0.3
                + correlation_confidence * 0.3
                + lag_confidence * 0.2
            )
            return max(0.0, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error computing confidence: {e}")
            return 0.0

    def detect_lagged_correlation(
        self, asset1: pd.Series, asset2: pd.Series, max_lag: int = 5
    ) -> Tuple[int, float]:
        """
        Обнаружение корреляции с оптимальным лагом.
        Args:
            asset1: Временной ряд первого актива
            asset2: Временной ряд второго актива
            max_lag: Максимальный лаг для поиска
        Returns:
            Tuple[int, float]: (оптимальный лаг, максимальная корреляция)
        """
        try:
            # Предобработка данных
            series1 = self._preprocess_series(asset1)
            series2 = self._preprocess_series(asset2)
            if len(series1) < MIN_SAMPLE_SIZE or len(series2) < MIN_SAMPLE_SIZE:
                return 0, 0.0
            best_lag = 0
            best_correlation = 0.0
            # Перебираем все возможные лаги
            for lag in range(-max_lag, max_lag + 1):
                try:
                    correlation, _ = self._compute_correlation_with_lag(
                        series1, series2, lag
                    )
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_lag = lag
                except Exception:
                    continue
            return best_lag, float(best_correlation)
        except Exception as e:
            logger.error(f"Error detecting lagged correlation: {e}")
            return 0, 0.0

    def detect_mirror_signal(
        self,
        asset1: str,
        asset2: str,
        series1: pd.Series,
        series2: pd.Series,
        max_lag: int = 5,
    ) -> Optional[MirrorSignal]:
        """
        Обнаружение зеркального сигнала между двумя активами.
        Args:
            asset1: Название первого актива
            asset2: Название второго актива
            series1: Временной ряд первого актива
            series2: Временной ряд второго актива
            max_lag: Максимальный лаг для поиска
        Returns:
            Optional[MirrorSignal]: Зеркальный сигнал или None
        """
        start_time = time.time()
        try:
            # Обнаруживаем корреляцию с лагом
            best_lag, correlation = self.detect_lagged_correlation(
                series1, series2, max_lag
            )
            # Проверяем значимость корреляции
            if abs(correlation) < self.config.min_correlation:
                return None
            # Вычисляем p-value для статистической значимости
            _, p_value = self._compute_correlation_with_lag(series1, series2, best_lag)
            # Вычисляем уверенность
            sample_size = min(len(series1), len(series2))
            confidence = self._compute_confidence(
                correlation, p_value, sample_size, best_lag
            )
            # Создаем метаданные анализа
            metadata: AnalysisMetadata = {
                "data_points": sample_size,
                "confidence": confidence,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "algorithm_version": ALGORITHM_VERSION,
                "parameters": {
                    "max_lag": max_lag,
                    "min_correlation": self.config.min_correlation,
                    "normalize_data": self.config.normalize_data,
                    "remove_trend": self.config.remove_trend,
                },
                "quality_metrics": {
                    "p_value": p_value,
                    "lag_search_range": max_lag,
                    "preprocessing_applied": self.config.normalize_data
                    or self.config.remove_trend,
                },
            }
            # Создаем метрики корреляции
            correlation_metrics: CorrelationMetrics = {
                "correlation": correlation,
                "p_value": p_value,
                "confidence": confidence,
                "lag": best_lag,
                "sample_size": sample_size,
                "significance": p_value < 0.05,
            }
            # Создаем зеркальный сигнал
            mirror_signal = MirrorSignal(
                asset1=asset1,
                asset2=asset2,
                best_lag=best_lag,
                correlation=correlation,
                p_value=p_value,
                confidence=confidence,
                signal_strength=abs(correlation),
                timestamp=Timestamp.now(),
                metadata=metadata,
            )
            # Обновляем статистику
            self.statistics["total_analyses"] += 1
            self.statistics["mirror_signals_detected"] += 1
            self.statistics["last_analysis_timestamp"] = time.time()
            # Обновляем среднее время обработки
            processing_time = (time.time() - start_time) * 1000
            current_avg = self.statistics["average_processing_time_ms"]
            total_analyses = self.statistics["total_analyses"]
            self.statistics["average_processing_time_ms"] = (
                current_avg * (total_analyses - 1) + processing_time
            ) / total_analyses
            logger.info(
                f"Mirror signal detected: {asset1} <-> {asset2}, "
                f"correlation: {correlation:.4f}, lag: {best_lag}, "
                f"confidence: {confidence:.4f}"
            )
            return mirror_signal
        except Exception as e:
            logger.error(
                f"Error detecting mirror signal between {asset1} and {asset2}: {e}"
            )
            return None

    def build_correlation_matrix(
        self, assets: List[str], price_data: Dict[str, pd.Series], max_lag: int = 5
    ) -> CorrelationMatrix:
        """
        Построение матрицы корреляций между всеми активами.
        Args:
            assets: Список активов для анализа
            price_data: Словарь с временными рядами цен
            max_lag: Максимальный лаг для поиска
        Returns:
            CorrelationMatrix: Матрица корреляций
        """
        try:
            n_assets = len(assets)
            correlation_matrix = np.zeros((n_assets, n_assets))
            lag_matrix = np.zeros((n_assets, n_assets), dtype=int)
            confidence_matrix = np.zeros((n_assets, n_assets))
            p_value_matrix = np.zeros((n_assets, n_assets))
            # Анализируем все пары активов
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i == j:
                        # Диагональ - корреляция с самим собой
                        correlation_matrix[i, j] = 1.0
                        lag_matrix[i, j] = 0
                        confidence_matrix[i, j] = 1.0
                        p_value_matrix[i, j] = 0.0
                    elif i < j:
                        # Анализируем только верхнюю треугольную матрицу
                        series1 = price_data.get(asset1)
                        series2 = price_data.get(asset2)
                        if series1 is not None and series2 is not None:
                            # Обнаруживаем корреляцию с лагом
                            best_lag, correlation = self.detect_lagged_correlation(
                                series1, series2, max_lag
                            )
                            # Вычисляем уверенность и p-value
                            sample_size = min(len(series1), len(series2))
                            _, p_value = self._compute_correlation_with_lag(
                                series1, series2, best_lag
                            )
                            confidence = self._compute_confidence(
                                correlation, p_value, sample_size, best_lag
                            )
                            # Заполняем матрицы
                            correlation_matrix[i, j] = correlation
                            correlation_matrix[j, i] = (
                                correlation  # Симметричная матрица
                            )
                            lag_matrix[i, j] = best_lag
                            lag_matrix[j, i] = -best_lag  # Обратный лаг
                            confidence_matrix[i, j] = confidence
                            confidence_matrix[j, i] = confidence
                            p_value_matrix[i, j] = p_value
                            p_value_matrix[j, i] = p_value
            # Создаем метаданные
            metadata: AnalysisMetadata = {
                "data_points": n_assets,
                "confidence": float(np.mean(confidence_matrix)),
                "processing_time_ms": 0.0,  # Будет обновлено позже
                "algorithm_version": ALGORITHM_VERSION,
                "parameters": {
                    "max_lag": max_lag,
                    "min_correlation": self.config.min_correlation,
                    "matrix_size": n_assets,
                },
                "quality_metrics": {
                    "matrix_size": n_assets,
                    "max_lag": max_lag,
                    "total_pairs": n_assets * (n_assets - 1) // 2,
                },
            }
            return CorrelationMatrix(
                assets=assets,
                correlation_matrix=correlation_matrix,
                lag_matrix=lag_matrix,
                confidence_matrix=confidence_matrix,
                p_value_matrix=p_value_matrix,
                metadata=metadata,
                timestamp=Timestamp.now(),
            )
        except Exception as e:
            logger.error(f"Error building correlation matrix: {e}")
            # Возвращаем пустую матрицу в случае ошибки
            return CorrelationMatrix(
                assets=assets,
                correlation_matrix=np.zeros((len(assets), len(assets))),
                lag_matrix=np.zeros((len(assets), len(assets)), dtype=int),
                confidence_matrix=np.zeros((len(assets), len(assets))),
                p_value_matrix=np.zeros((len(assets), len(assets))),
                metadata={
                    "data_points": 0,
                    "confidence": 0.0,
                    "processing_time_ms": 0.0,
                    "algorithm_version": ALGORITHM_VERSION,
                    "parameters": {},
                    "quality_metrics": {"error_code": 1.0},
                },
                timestamp=Timestamp.now(),
            )

    def find_mirror_clusters(
        self,
        correlation_matrix: CorrelationMatrix,
        min_correlation: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Поиск кластеров зеркальных активов.
        Args:
            correlation_matrix: Матрица корреляций
            min_correlation: Минимальная корреляция для включения в кластер
        Returns:
            List[List[str]]: Список кластеров активов
        """
        try:
            if min_correlation is None:
                min_correlation = self.config.min_correlation
            n_assets = len(correlation_matrix.assets)
            visited = [False] * n_assets
            clusters = []

            # Поиск в глубину для нахождения связанных компонентов
            def dfs(asset_idx: int, cluster: List[str]) -> None:
                visited[asset_idx] = True
                cluster.append(correlation_matrix.assets[asset_idx])
                for j in range(n_assets):
                    if (
                        not visited[j]
                        and abs(correlation_matrix.correlation_matrix[asset_idx, j])
                        >= min_correlation
                    ):
                        dfs(j, cluster)

            # Находим все кластеры
            for i in range(n_assets):
                if not visited[i]:
                    cluster: List[str] = []
                    dfs(i, cluster)
                    if len(cluster) > 1:  # Кластер должен содержать минимум 2 актива
                        clusters.append(cluster)
            return clusters
        except Exception as e:
            logger.error(f"Error finding mirror clusters: {e}")
            return []

    def get_detector_statistics(self) -> Dict[str, Any]:
        """Получение статистики детектора."""
        return {
            **self.statistics,
            "config": {
                "min_correlation": self.config.min_correlation,
                "normalize_data": self.config.normalize_data,
                "remove_trend": self.config.remove_trend,
            },
            "advanced_metrics_enabled": self.enable_advanced_metrics,
            "clustering_enabled": self.enable_clustering,
        }

    def reset_statistics(self) -> None:
        """Сброс статистики."""
        self.statistics = {
            "total_analyses": 0,
            "mirror_signals_detected": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info("Mirror detector statistics reset")
