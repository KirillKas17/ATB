from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


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
        self.config = config or {}
        self.max_lag = self.config.get("max_lag", 20)
        self.min_correlation = self.config.get("min_correlation", 0.5)
        self.significance_level = self.config.get("significance_level", 0.05)
        self.regime_window = self.config.get("regime_window", 100)
        self.stability_window = self.config.get("stability_window", 50)
        self.metrics: Dict[str, Dict[str, CorrelationMetrics]] = {}

    def calculate_correlation(
        self, array1: np.ndarray, array2: np.ndarray
    ) -> CorrelationMetrics:
        """Расчет корреляции между массивами."""
        try:
            # Преобразуем массивы в pd.Series
            series1 = pd.Series(array1)
            series2 = pd.Series(array2)

            # Проверяем на достаточное количество данных
            if len(series1) < 2 or len(series2) < 2:
                logger.warning("Insufficient data for correlation calculation")
                return CorrelationMetrics(
                    correlation=0.0, p_value=1.0, lag=0, strength=0.0
                )

            # Рассчитываем корреляцию
            corr = float(series1.corr(series2))

            # Рассчитываем p-value
            _, p_value = stats.pearsonr(series1.dropna(), series2.dropna())

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

            for i, pair1 in enumerate(pairs):
                for j, pair2 in enumerate(pairs):
                    if i != j:
                        metrics = self.calculate_correlation(data[pair1], data[pair2])
                        matrix.loc[pair1, pair2] = float(metrics.correlation)
                    else:
                        matrix.loc[pair1, pair2] = 1.0

            return matrix

        except Exception as e:
            logger.error(f"Error getting correlation matrix: {e}")
            return pd.DataFrame()

    def _calculate_cointegration(self, series1: pd.Series, series2: pd.Series) -> float:
        """Расчет коинтеграции между рядами"""
        try:
            from statsmodels.tsa.stattools import coint

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
            from statsmodels.tsa.stattools import grangercausalitytests

            data = pd.concat([series1, series2], axis=1)
            result = grangercausalitytests(data, maxlag=5, verbose=False)

            # Агрегация результатов по всем лагам
            p_values = [result[lag][0]["ssr_chi2test"][1] for lag in range(1, 6)]
            return float(1.0 - min(p_values))
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
                return f"{series1.name} leads {series2.name} by {lag}"
            elif lag < 0:
                return f"{series2.name} leads {series1.name} by {abs(lag)}"
            else:
                return "No clear lead-lag relationship"
        except Exception as e:
            logger.warning(f"Error calculating lead-lag relationship: {str(e)}")
            return "No clear lead-lag relationship"

    def _calculate_lag(self, series1: pd.Series, series2: pd.Series) -> int:
        """Расчет лага между рядами"""
        try:
            # Расчет автокорреляции
            acf1 = stats.acf(series1, nlags=self.max_lag)
            acf2 = stats.acf(series2, nlags=self.max_lag)

            # Поиск максимального значения автокорреляции
            max(acf1, acf2)

            # Определение лага
            lag = np.argmax(acf1) - np.argmax(acf2)

            return lag
        except Exception as e:
            logger.warning(f"Error calculating lag: {str(e)}")
            return 0

    def _analyze_regime_dependency(
        self, series1: pd.Series, series2: pd.Series
    ) -> Dict[str, float]:
        """Анализ зависимости корреляции от режима рынка"""
        try:
            # Определение режимов
            returns1 = series1.pct_change().fillna(0)
            returns2 = series2.pct_change().fillna(0)

            # Проверка на достаточное количество данных
            if len(returns1) < 2 or len(returns2) < 2:
                logger.warning("Insufficient data for regime analysis")
                return {"bull": 0.0, "bear": 0.0, "volatile": 0.0}

            # Разделение на режимы
            bull_mask = (returns1 > 0) & (returns2 > 0)
            bear_mask = (returns1 < 0) & (returns2 < 0)
            volatile_mask = (returns1.abs() > returns1.std()) | (
                returns2.abs() > returns2.std()
            )

            # Проверка на наличие данных в каждом режиме
            if not any(bull_mask) or not any(bear_mask) or not any(volatile_mask):
                logger.warning("Insufficient data in one or more regimes")
                return {"bull": 0.0, "bear": 0.0, "volatile": 0.0}

            # Расчет корреляций для каждого режима
            bull_corr = float(returns1[bull_mask].corr(returns2[bull_mask]))
            bear_corr = float(returns1[bear_mask].corr(returns2[bear_mask]))
            volatile_corr = float(returns1[volatile_mask].corr(returns2[volatile_mask]))

            return {
                "bull": bull_corr,
                "bear": bear_corr,
                "volatile": volatile_corr,
            }
        except Exception as e:
            logger.warning(f"Error analyzing regime dependency: {str(e)}")
            return {"bull": 0.0, "bear": 0.0, "volatile": 0.0}

    def _calculate_stability_score(
        self, series1: pd.Series, series2: pd.Series
    ) -> float:
        """Расчет оценки стабильности корреляции"""
        try:
            window_size = min(self.stability_window, len(series1) // 2)
            if window_size < 2:
                logger.warning("Window size too small for stability calculation")
                return 0.0

            correlations = []

            for i in range(0, len(series1) - window_size, window_size):
                window1 = pd.Series(series1[i : i + window_size])
                window2 = pd.Series(series2[i : i + window_size])

                # Проверка на NaN в окне
                if window1.isna().any() or window2.isna().any():
                    continue

                # Проверка на достаточное количество данных
                if len(window1) < 2 or len(window2) < 2:
                    continue

                corr = window1.corr(window2)
                if not pd.isna(corr):
                    correlations.append(corr)

            if not correlations:
                logger.warning("No valid correlations found for stability calculation")
                return 0.0

            # Оценка стабильности через стандартное отклонение корреляций
            return float(1.0 - np.std(correlations))
        except Exception as e:
            logger.warning(f"Error calculating stability score: {str(e)}")
            return 0.0

    def _find_breakpoints(self, series1: pd.Series, series2: pd.Series) -> List[int]:
        """Поиск точек разрыва корреляции"""
        try:
            from statsmodels.tsa.stattools import adfuller

            # Расчет скользящей корреляции
            window_size = min(self.stability_window, len(series1) // 2)
            rolling_corr = pd.Series(index=series1.index)

            for i in range(window_size, len(series1)):
                window1 = pd.Series(series1[i - window_size : i])
                window2 = pd.Series(series2[i - window_size : i])
                rolling_corr.iloc[i] = window1.corr(window2)

            # Поиск точек разрыва через тест на стационарность
            breakpoints = []
            for i in range(window_size, len(rolling_corr) - window_size):
                window = pd.Series(rolling_corr[i - window_size : i + window_size])
                if len(window.dropna()) > 0:
                    adf_result = adfuller(window.dropna())
                    if adf_result[1] < self.significance_level:
                        breakpoints.append(i)

            return breakpoints
        except Exception as e:
            logger.warning(f"Error finding breakpoints: {str(e)}")
            return []
