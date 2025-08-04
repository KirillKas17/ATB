from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.tsa.stattools import acf, coint, grangercausalitytests  # type: ignore


@dataclass
class CorrelationMetrics:
    """Расширенные метрики корреляции"""

    correlation: float
    p_value: float
    lag: int
    strength: float
    cointegration: Optional[float] = None
    granger_causality: Optional[float] = None
    lead_lag_relationship: Optional[str] = None
    regime_dependency: Optional[Dict[str, float]] = None
    stability_score: Optional[float] = None
    breakpoints: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "correlation": self.correlation,
            "p_value": self.p_value,
            "lag": self.lag,
            "strength": self.strength,
            "cointegration": self.cointegration,
            "granger_causality": self.granger_causality,
            "lead_lag_relationship": self.lead_lag_relationship,
            "regime_dependency": self.regime_dependency,
            "stability_score": self.stability_score,
            "breakpoints": self.breakpoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationMetrics":
        """Создание из словаря"""
        return cls(**data)


class CorrelationChain:
    """Расширенный анализ корреляций"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация цепочки корреляций."""
        self.config = config or {}
        self.metrics: Dict[str, Dict[str, CorrelationMetrics]] = {}
        self.min_correlation = self.config.get("min_correlation", 0.3)
        self.max_lag = self.config.get("max_lag", 20)
        self.regime_window = self.config.get("regime_window", 50)
        self.stability_window = self.config.get("stability_window", 30)
        self.breakpoint_window = self.config.get("breakpoint_window", 20)
        self.breakpoint_threshold = self.config.get("breakpoint_threshold", 0.2)

    def calculate_correlation(
        self, array1: np.ndarray, array2: np.ndarray
    ) -> CorrelationMetrics:
        """
        Расчет корреляции между двумя массивами.
        Args:
            array1: Первый массив
            array2: Второй массив
        Returns:
            CorrelationMetrics: Метрики корреляции
        """
        try:
            if len(array1) != len(array2) or len(array1) < 10:
                return CorrelationMetrics(
                    correlation=0.0, p_value=1.0, lag=0, strength=0.0
                )
            # Преобразование в pandas Series
            series1 = pd.Series(array1)
            series2 = pd.Series(array2)
            # Удаление NaN значений
            series1 = series1.dropna()
            series2 = series2.dropna()
            if len(series1) < 10 or len(series2) < 10:
                return CorrelationMetrics(
                    correlation=0.0, p_value=1.0, lag=0, strength=0.0
                )
            # Рассчитываем корреляцию
            # Исправляем использование corr для Series
            if hasattr(series1, 'corr') and hasattr(series2, 'corr'):
                corr = float(series1.corr(series2))
            else:
                # Преобразуем в numpy массивы для совместимости
                series1_array = series1.to_numpy() if hasattr(series1, 'to_numpy') else np.array(series1)
                series2_array = series2.to_numpy() if hasattr(series2, 'to_numpy') else np.array(series2)
                corr = float(np.corrcoef(series1_array, series2_array)[0, 1])
            # Рассчитываем p-value
            # Исправляем использование values для Series
            series1_array = series1.to_numpy() if hasattr(series1, 'to_numpy') else np.array(series1)
            series2_array = series2.to_numpy() if hasattr(series2, 'to_numpy') else np.array(series2)
            _, p_value = stats.pearsonr(series1_array, series2_array)
            # Рассчитываем лаг
            lag = self._calculate_lag(series1, series2)
            # Рассчитываем силу корреляции
            strength = abs(corr)
            # Рассчитываем дополнительные метрики
            cointegration = self._calculate_cointegration(series1, series2)
            granger_causality = self._calculate_granger_causality(series1, series2)
            lead_lag = self._calculate_lead_lag_relationship(series1, series2)
            regime_dep = self._analyze_regime_dependency(series1, series2)
            stability = self._calculate_stability_score(series1, series2)
            breakpoints = self._find_breakpoints(series1, series2)
            return CorrelationMetrics(
                correlation=corr,
                p_value=float(p_value),
                lag=lag,
                strength=strength,
                cointegration=cointegration,
                granger_causality=granger_causality,
                lead_lag_relationship=lead_lag,
                regime_dependency=regime_dep,
                stability_score=stability,
                breakpoints=breakpoints,
            )
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return CorrelationMetrics(correlation=0.0, p_value=1.0, lag=0, strength=0.0)

    def add_metrics(self, pair1: str, pair2: str, metrics: CorrelationMetrics) -> None:
        """
        Добавление метрик для пары инструментов.
        Args:
            pair1: Первый инструмент
            pair2: Второй инструмент
            metrics: Метрики корреляции
        """
        if pair1 not in self.metrics:
            self.metrics[pair1] = {}
        self.metrics[pair1][pair2] = metrics

    def get_metrics(self, pair1: str, pair2: str) -> Optional[CorrelationMetrics]:
        """
        Получение метрик для пары инструментов.
        Args:
            pair1: Первый инструмент
            pair2: Второй инструмент
        Returns:
            Optional[CorrelationMetrics]: Метрики корреляции
        """
        return self.metrics.get(pair1, {}).get(pair2)

    def get_correlation_matrix(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Получение матрицы корреляций."""
        try:
            pairs = list(data.keys())
            matrix = pd.DataFrame(
                index=pd.Index(pairs), columns=pd.Index(pairs), dtype=float
            )
            # Заполнение матрицы корреляций
            for pair1 in data.keys():
                for pair2 in data.keys():
                    if pair1 != pair2:
                        # Получаем метрики корреляции
                        metrics = self.get_metrics(pair1, pair2)
                        if metrics:
                            matrix.loc[pair1, pair2] = metrics.correlation
                        else:
                            # Если метрики не найдены, рассчитываем их
                            if len(data[pair1]) == len(data[pair2]) and len(data[pair1]) > 10:
                                metrics = self.calculate_correlation(data[pair1], data[pair2])
                                self.add_metrics(pair1, pair2, metrics)
                                matrix.loc[pair1, pair2] = metrics.correlation
                            else:
                                # Проверяем, что matrix не является функцией
                                if callable(matrix):
                                    matrix_result = matrix()
                                    if hasattr(matrix_result, 'loc'):
                                        matrix_result.loc[pair1, pair2] = 1.0
                                else:
                                    if hasattr(matrix, 'loc'):
                                        matrix.loc[pair1, pair2] = 1.0
            return matrix
        except Exception as e:
            logger.error(f"Error getting correlation matrix: {e}")
            return pd.DataFrame()

    def _calculate_cointegration(self, series1: pd.Series, series2: pd.Series) -> float:
        """Расчет коинтеграции между рядами"""
        try:
            _, p_value, _ = coint(series1, series2)
            return float(1.0 - p_value)
        except Exception as e:
            logger.warning(f"Error calculating cointegration: {str(e)}")
            return 0.0

    def _calculate_granger_causality(
        self, series1: pd.Series, series2: pd.Series
    ) -> float:
        """Расчет причинности по Грейнджеру"""
        try:
            data = pd.concat([series1, series2], axis=1, ignore_index=True)
            result = grangercausalitytests(data, maxlag=5, verbose=False)
            # Агрегация результатов по всем лагам
            p_values = []
            for lag in range(1, 6):
                if lag in result:
                    p_values.append(result[lag][0]["ssr_chi2test"][1])
            return float(1.0 - min(p_values)) if p_values else 0.0
        except Exception as e:
            logger.warning(f"Error calculating Granger causality: {str(e)}")
            return 0.0

    def _calculate_lead_lag_relationship(
        self, series1: pd.Series, series2: pd.Series
    ) -> str:
        """Расчет лидирующего ряда"""
        try:
            # Расчет лага
            lag = self._calculate_lag(series1, series2)
            # Определение лидирующего ряда
            if lag > 0:
                name1 = getattr(series1, 'name', 'Series1')
                name2 = getattr(series2, 'name', 'Series2')
                return f"{name1} leads {name2} by {lag}"
            elif lag < 0:
                name1 = getattr(series1, 'name', 'Series1')
                name2 = getattr(series2, 'name', 'Series2')
                return f"{name2} leads {name1} by {abs(lag)}"
            else:
                return "No clear lead-lag relationship"
        except Exception as e:
            logger.warning(f"Error calculating lead-lag relationship: {str(e)}")
            return "No clear lead-lag relationship"

    def _calculate_lag(self, series1: pd.Series, series2: pd.Series) -> int:
        """Расчет лага между рядами"""
        try:
            # Расчет автокорреляции
            acf1_result = acf(series1, nlags=self.max_lag)
            acf2_result = acf(series2, nlags=self.max_lag)
            # Преобразуем в numpy массивы если это функции
            if callable(acf1_result):
                acf1 = acf1_result()
            else:
                acf1 = acf1_result
            if callable(acf2_result):
                acf2 = acf2_result()
            else:
                acf2 = acf2_result
            # Поиск максимального значения автокорреляции
            max_acf1 = max(acf1)
            max_acf2 = max(acf2)
            # Определение лага
            lag = int(np.argmax(acf1) - np.argmax(acf2))
            return lag
        except Exception as e:
            logger.warning(f"Error calculating lag: {str(e)}")
            return 0

    def _analyze_regime_dependency(
        self, series1: pd.Series, series2: pd.Series
    ) -> Dict[str, float]:
        """Анализ зависимости от режима рынка"""
        try:
            # Проверяем, что series1 и series2 не являются функциями
            if callable(series1):
                series1_data = series1()
            else:
                series1_data = series1
            if callable(series2):
                series2_data = series2()
            else:
                series2_data = series2
                
            # Разделение на окна
            window_size = self.regime_window
            correlations = []
            volatilities = []
            for i in range(0, len(series1_data) - window_size, window_size // 2):
                window1 = series1_data.iloc[i : i + window_size]
                window2 = series2_data.iloc[i : i + window_size]
                if len(window1) >= 10 and len(window2) >= 10:
                    corr = window1.corr(window2)
                    vol = window1.std() + window2.std()
                    correlations.append(corr)
                    volatilities.append(vol)
            # Анализ зависимости
            if len(correlations) > 1:
                vol_corr = np.corrcoef(volatilities, correlations)[0, 1]
                return {
                    "volatility_dependency": float(vol_corr),
                    "correlation_stability": float(np.std(correlations)),
                    "regime_count": len(correlations),
                }
            else:
                return {
                    "volatility_dependency": 0.0,
                    "correlation_stability": 0.0,
                    "regime_count": 0,
                }
        except Exception as e:
            logger.warning(f"Error analyzing regime dependency: {str(e)}")
            return {
                "volatility_dependency": 0.0,
                "correlation_stability": 0.0,
                "regime_count": 0,
            }

    def _calculate_stability_score(
        self, series1: pd.Series, series2: pd.Series
    ) -> float:
        """Расчет стабильности корреляции"""
        try:
            # Проверяем, что series1 и series2 не являются функциями
            if callable(series1):
                series1_data = series1()
            else:
                series1_data = series1
            if callable(series2):
                series2_data = series2()
            else:
                series2_data = series2
                
            # Расчет корреляции в скользящем окне
            window_size = self.stability_window
            correlations = []
            for i in range(0, len(series1_data) - window_size, window_size // 4):
                window1 = series1_data.iloc[i : i + window_size]
                window2 = series2_data.iloc[i : i + window_size]
                if len(window1) >= 10 and len(window2) >= 10:
                    corr = window1.corr(window2)
                    if not np.isnan(corr):
                        correlations.append(corr)
            # Расчет стабильности
            if len(correlations) > 1:
                stability = 1.0 - float(np.std(correlations))
                return float(max(0.0, min(1.0, stability)))
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating stability score: {str(e)}")
            return 0.0

    def _find_breakpoints(self, series1: pd.Series, series2: pd.Series) -> List[int]:
        """Поиск точек разрыва корреляции"""
        try:
            # Проверяем, что series1 и series2 не являются функциями
            if callable(series1):
                series1_data = series1()
            else:
                series1_data = series1
            if callable(series2):
                series2_data = series2()
            else:
                series2_data = series2
                
            # Расчет корреляции в скользящем окне
            window_size = self.breakpoint_window
            breakpoints = []
            for i in range(window_size, len(series1_data) - window_size, window_size // 2):
                # Корреляция до точки
                before1 = series1_data.iloc[i - window_size : i]
                before2 = series2_data.iloc[i - window_size : i]
                corr_before = before1.corr(before2)
                # Корреляция после точки
                after1 = series1_data.iloc[i : i + window_size]
                after2 = series2_data.iloc[i : i + window_size]
                corr_after = after1.corr(after2)
                # Проверка разрыва
                if not np.isnan(corr_before) and not np.isnan(corr_after):
                    if abs(corr_before - corr_after) > self.breakpoint_threshold:
                        breakpoints.append(i)
            return breakpoints
        except Exception as e:
            logger.warning(f"Error finding breakpoints: {str(e)}")
            return []

    def get_strong_correlations(
        self, threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение сильных корреляций.
        Args:
            threshold: Порог корреляции (по умолчанию использует min_correlation)
        Returns:
            Список сильных корреляций
        """
        threshold = threshold or self.min_correlation
        strong_correlations = []
        for pair1, pairs in self.metrics.items():
            for pair2, metrics in pairs.items():
                if abs(metrics.correlation) >= threshold:
                    strong_correlations.append({
                        "pair1": pair1,
                        "pair2": pair2,
                        "correlation": metrics.correlation,
                        "strength": metrics.strength,
                        "lag": metrics.lag,
                        "p_value": metrics.p_value,
                    })
        return strong_correlations

    def get_correlation_summary(self) -> Dict[str, Any]:
        """
        Получение сводки по корреляциям.
        Returns:
            Словарь со сводкой
        """
        all_correlations = []
        for pair1, pairs in self.metrics.items():
            for pair2, metrics in pairs.items():
                all_correlations.append(metrics.correlation)
        if all_correlations:
            return {
                "total_pairs": len(all_correlations),
                "mean_correlation": float(np.mean(all_correlations)),
                "std_correlation": float(np.std(all_correlations)),
                "max_correlation": float(np.max(all_correlations)),
                "min_correlation": float(np.min(all_correlations)),
                "strong_correlations": len([c for c in all_correlations if abs(c) >= 0.7]),
            }
        else:
            return {
                "total_pairs": 0,
                "mean_correlation": 0.0,
                "std_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "strong_correlations": 0,
            }
