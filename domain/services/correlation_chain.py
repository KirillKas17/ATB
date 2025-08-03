"""
Модуль для анализа корреляций между инструментами.
Промышленная реализация с строгой типизацией и продвинутыми
алгоритмами анализа корреляций.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr, kendalltau, norm


class CorrelationMethod(Enum):
    """Методы расчета корреляции."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationStrength(Enum):
    """Уровни силы корреляции."""

    VERY_WEAK = (0.0, 0.1)
    WEAK = (0.1, 0.3)
    MODERATE = (0.3, 0.5)
    STRONG = (0.5, 0.7)
    VERY_STRONG = (0.7, 0.9)
    PERFECT = (0.9, 1.0)


@dataclass
class CorrelationPair:
    """Класс для хранения информации о коррелирующей паре."""

    symbol1: str
    symbol2: str
    correlation: float
    lag: int
    strength: CorrelationStrength
    p_value: float
    confidence_interval: Tuple[float, float] = field(default=(0.0, 0.0))
    sample_size: int = field(default=0)


@dataclass
class CorrelationSummary:
    """Сводка по корреляциям."""

    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    total_pairs: int
    significant_pairs: int
    strong_correlations: int
    average_lag: float


@runtime_checkable
class CorrelationAnalyzer(Protocol):
    """Протокол для анализаторов корреляций."""

    def calculate_correlation(self, x: pd.Series, y: pd.Series) -> float: ...
    def calculate_lag(self, x: pd.Series, y: pd.Series, max_lag: int) -> int: ...
    def assess_significance(self, correlation: float, sample_size: int) -> float: ...


class CorrelationChain(ABC):
    """Абстрактный класс для анализа корреляций между активами."""

    @abstractmethod
    def build_correlation_matrix(self) -> pd.DataFrame:
        """Строит матрицу корреляций."""
        pass

    @abstractmethod
    def find_correlation_chain(self, threshold: float) -> List[CorrelationPair]:
        """Находит цепочки коррелированных активов."""
        pass

    @abstractmethod
    def get_correlation_summary(self) -> CorrelationSummary:
        """Возвращает сводку по корреляциям."""
        pass


class DefaultCorrelationChain(CorrelationChain):
    """Промышленная реализация анализа корреляций между активами."""

    def __init__(
        self, data: pd.DataFrame, method: CorrelationMethod = CorrelationMethod.PEARSON
    ) -> None:
        """
        Инициализация анализа корреляций.
        Args:
            data: DataFrame с ценами активов
            method: Метод расчета корреляции
        """
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        self.data: pd.DataFrame = data
        self.method: CorrelationMethod = method
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.lag_matrix: Optional[pd.DataFrame] = None
        self.correlation_pairs: List[CorrelationPair] = []
        self._validate_data()

    def _validate_data(self) -> None:
        """Валидация входных данных."""
        # Проверяем наличие атрибута to_numpy у DataFrame
        if hasattr(self.data.isnull(), 'values'):
            # Исправление: безопасное получение numpy array из DataFrame
            null_array = self.data.isnull().values
            if hasattr(null_array, 'any'):
                if null_array.any():
                    raise ValueError("Data contains null values")
        elif hasattr(self.data.isnull(), 'any'):
            if self.data.isnull().any().any():
                raise ValueError("Data contains null values")
        else:
            # Альтернативная проверка на null значения
            if self.data.isnull().any():
                raise ValueError("Data contains null values")
        if len(self.data.columns) < 2:
            raise ValueError("At least 2 assets required for correlation analysis")
        if len(self.data) < 30:
            raise ValueError(
                "Insufficient data points for reliable correlation analysis"
            )

    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """
        Вычисляет корреляцию между двумя Series.
        Returns:
            Tuple[float, float]: (correlation, p_value)
        """
        if len(x) != len(y):
            raise ValueError("Series must have the same length")
        # Удаляем NaN значения
        # Проверяем наличие атрибута to_numpy у Series
        if hasattr(x.isna(), 'values'):
            mask = ~(x.isna().values.astype(bool) | y.isna().values.astype(bool))
        else:
            # Альтернативный способ создания маски
            mask = ~(x.isna().astype(bool) | y.isna().astype(bool))
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 10:
            return 0.0, 1.0
        if self.method == CorrelationMethod.PEARSON:
            correlation, p_value = pearsonr(x_clean, y_clean)
        elif self.method == CorrelationMethod.SPEARMAN:
            correlation, p_value = spearmanr(x_clean, y_clean)
        else:
            # Kendall correlation
            correlation, p_value = kendalltau(x_clean, y_clean)
        return float(correlation), float(p_value)

    def _assess_correlation_strength(self, correlation: float) -> CorrelationStrength:
        """Оценка силы корреляции."""
        abs_corr = abs(correlation)
        for strength in CorrelationStrength:
            min_val, max_val = strength.value
            if min_val <= abs_corr <= max_val:
                return strength
        return CorrelationStrength.PERFECT

    def _calculate_confidence_interval(
        self, correlation: float, sample_size: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Вычисление доверительного интервала для корреляции."""
        if sample_size < 3:
            return (0.0, 0.0)
        # Преобразование Фишера
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        # Стандартная ошибка
        se = 1 / np.sqrt(sample_size - 3)
        # Z-критическое значение
        z_critical = norm.ppf((1 + confidence) / 2)
        # Доверительный интервал в Z-пространстве
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        # Обратное преобразование
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        return (float(r_lower), float(r_upper))

    def build_correlation_matrix(self) -> pd.DataFrame:
        """Строит матрицу корреляций."""
        if self.correlation_matrix is not None:
            return self.correlation_matrix
        assets = self.data.columns
        corr_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    if hasattr(corr_matrix, 'loc'):
                        corr_matrix.loc[asset1, asset2] = 1.0
                else:
                    x = self.data[asset1]
                    y = self.data[asset2]
                    if callable(x):
                        x = x()
                    if callable(y):
                        y = y()
                    correlation, _ = self._calculate_correlation(x, y)
                    # Проверяем, что corr_matrix не является callable перед индексированием
                    if callable(corr_matrix):
                        corr_matrix = corr_matrix()
                    if hasattr(corr_matrix, 'loc'):
                        corr_matrix.loc[asset1, asset2] = correlation
        self.correlation_matrix = corr_matrix
        return corr_matrix

    def calculate_lag(self, x: pd.Series, y: pd.Series, max_lag: int = 10) -> int:
        """
        Расчет оптимального лага между активами.
        Args:
            x: Первый актив
            y: Второй актив
            max_lag: Максимальный лаг
        Returns:
            int: Оптимальный лаг
        """
        if len(x) != len(y):
            raise ValueError("Series must have the same length")
        # Нормализация данных
        x_norm = (x - x.mean()) / x.std()
        y_norm = (y - y.mean()) / y.std()
        # Удаляем NaN значения
        mask = ~(x_norm.isna() | y_norm.isna())
        x_clean = x_norm[mask]
        y_clean = y_norm[mask]
        if len(x_clean) < max_lag * 2:
            return 0
        # Расчет кросс-корреляции
        correlation = correlate(x_clean, y_clean, mode="full")
        # Находим максимальную корреляцию в пределах max_lag
        mid_point = len(correlation) // 2
        start_idx = max(0, mid_point - max_lag)
        end_idx = min(len(correlation), mid_point + max_lag + 1)
        lag_range = correlation[start_idx:end_idx]
        max_idx = np.argmax(np.abs(lag_range))
        # Вычисляем фактический лаг
        actual_lag = max_idx - (mid_point - start_idx)
        return int(actual_lag)

    def build_lag_matrix(self, max_lag: int = 10) -> pd.DataFrame:
        """
        Построение матрицы лагов.
        Args:
            max_lag: Максимальный лаг
        Returns:
            pd.DataFrame: Матрица лагов
        """
        if self.lag_matrix is not None:
            return self.lag_matrix
        assets = self.data.columns
        lag_matrix = pd.DataFrame(index=assets, columns=assets, dtype=int)
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    if hasattr(lag_matrix, 'loc'):
                        lag_matrix.loc[asset1, asset2] = 0
                else:
                    lag = self.calculate_lag(
                        self.data[asset1], self.data[asset2], max_lag=max_lag
                    )
                    # Проверяем, что lag_matrix не является callable перед индексированием
                    if callable(lag_matrix):
                        lag_matrix = lag_matrix()
                    if hasattr(lag_matrix, 'loc'):
                        lag_matrix.loc[asset1, asset2] = lag
        self.lag_matrix = lag_matrix
        return lag_matrix

    def find_correlation_chain(self, threshold: float = 0.7) -> List[CorrelationPair]:
        """Находит цепочки коррелированных активов."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self.build_correlation_matrix()
        if self.correlation_matrix.empty:
            return []
        symbols = list(self.correlation_matrix.index)
        pairs: List[CorrelationPair] = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                corr_value = self.correlation_matrix.loc[symbol1, symbol2]
                if abs(corr_value) >= threshold:
                    # Вычисляем p-value и доверительный интервал
                    correlation, p_value = self._calculate_correlation(
                        self.data[symbol1], self.data[symbol2]
                    )
                    # Вычисляем лаг
                    lag = self.calculate_lag(self.data[symbol1], self.data[symbol2])
                    # Оцениваем силу корреляции
                    strength = self._assess_correlation_strength(correlation)
                    # Вычисляем доверительный интервал
                    sample_size = len(self.data)
                    confidence_interval = self._calculate_confidence_interval(
                        correlation, sample_size
                    )
                    pair = CorrelationPair(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        correlation=correlation,
                        lag=lag,
                        strength=strength,
                        p_value=p_value,
                        confidence_interval=confidence_interval,
                        sample_size=sample_size,
                    )
                    pairs.append(pair)
        # Сортируем по абсолютному значению корреляции
        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        self.correlation_pairs = pairs
        return pairs

    def get_leading_assets(self) -> List[str]:
        """Возвращает ведущие активы на основе анализа лагов."""
        if self.lag_matrix is None:
            self.lag_matrix = self.build_lag_matrix()
        if self.lag_matrix.empty:
            return []
        # Находим активы с минимальными средними лагами
        # Проверяем наличие атрибута to_numpy у DataFrame
        if hasattr(self.lag_matrix, 'to_numpy'):
            avg_lags = self.lag_matrix.to_numpy().mean(axis=1)
        else:
            # Альтернативная проверка на null значения
            avg_lags = np.array([self.lag_matrix.iloc[i].mean() for i in range(len(self.lag_matrix))])
        min_lag = avg_lags.min()
        leading_assets: List[str] = self.lag_matrix.index[avg_lags <= min_lag + 1].tolist()
        return leading_assets

    def get_correlation_groups(self, threshold: float = 0.7) -> List[List[str]]:
        """Возвращает группы коррелированных активов."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self.build_correlation_matrix()
        if self.correlation_matrix.empty:
            return []
        # Используем алгоритм кластеризации для группировки
        from sklearn.cluster import AgglomerativeClustering

        # Создаем матрицу расстояний на основе корреляций
        # Проверяем наличие атрибута to_numpy у DataFrame
        if hasattr(self.correlation_matrix, 'to_numpy'):
            distance_matrix = 1 - np.abs(self.correlation_matrix.to_numpy())
        else:
            # Альтернативный способ получения значений
            distance_matrix = 1 - np.abs(self.correlation_matrix.to_numpy())
        # Кластеризация
        n_clusters = max(1, len(self.correlation_matrix.columns) // 3)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="complete"
        )
        clusters = clustering.fit_predict(distance_matrix)
        # Группируем активы по кластерам
        groups: Dict[int, List[str]] = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append(self.correlation_matrix.columns[i])
        # Фильтруем группы по размеру и корреляции
        filtered_groups = []
        for group in groups.values():
            if len(group) >= 2:
                # Проверяем среднюю корреляцию в группе
                group_corr = self.correlation_matrix.loc[group, group]
                # Проверяем наличие атрибута to_numpy у DataFrame
                if hasattr(group_corr, 'to_numpy'):
                    values = group_corr.to_numpy()
                    mean_corr = values[np.triu_indices_from(values, k=1)].mean()
                else:
                    # Альтернативный способ вычисления средней корреляции
                    values = group_corr.to_numpy()
                    mean_corr = values[np.triu_indices_from(values, k=1)].mean()
                if mean_corr >= threshold:
                    filtered_groups.append(group)
        return filtered_groups

    def get_correlation_summary(self) -> CorrelationSummary:
        """Возвращает сводку по корреляциям."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self.build_correlation_matrix()
        if self.correlation_matrix.empty:
            return CorrelationSummary(
                mean_correlation=0.0,
                std_correlation=0.0,
                min_correlation=0.0,
                max_correlation=0.0,
                total_pairs=0,
                significant_pairs=0,
                strong_correlations=0,
                average_lag=0.0,
            )
        # Базовые статистики
        # Проверяем наличие атрибута to_numpy у DataFrame
        if hasattr(self.correlation_matrix, 'to_numpy'):
            values = self.correlation_matrix.to_numpy()
        else:
            values = self.correlation_matrix.to_numpy()
        upper_triangle = values[np.triu_indices_from(values, k=1)]
        mean_corr = float(np.mean(upper_triangle))
        std_corr = float(np.std(upper_triangle))
        min_corr = float(np.min(upper_triangle))
        max_corr = float(np.max(upper_triangle))
        # Количество пар
        total_pairs = len(upper_triangle)
        # Значимые корреляции (p < 0.05)
        significant_pairs = sum(1 for corr in upper_triangle if abs(corr) > 0.3)
        # Сильные корреляции
        strong_correlations = sum(1 for corr in upper_triangle if abs(corr) > 0.7)
        # Средний лаг
        if self.lag_matrix is not None:
            # Проверяем наличие атрибута to_numpy у DataFrame
            if hasattr(self.lag_matrix, 'to_numpy'):
                lag_values = self.lag_matrix.to_numpy()
            else:
                lag_values = self.lag_matrix.to_numpy()
            lag_upper_triangle = lag_values[np.triu_indices_from(lag_values, k=1)]
            average_lag = float(np.mean(np.abs(lag_upper_triangle)))
        else:
            average_lag = 0.0
        return CorrelationSummary(
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            total_pairs=total_pairs,
            significant_pairs=significant_pairs,
            strong_correlations=strong_correlations,
            average_lag=average_lag,
        )

    def get_rolling_correlation(
        self, symbol1: str, symbol2: str, window: int = 30
    ) -> pd.Series:
        """Вычисляет скользящую корреляцию между двумя активами."""
        if symbol1 not in self.data.columns or symbol2 not in self.data.columns:
            raise ValueError(f"Symbols {symbol1} or {symbol2} not found in data")
        series1 = self.data[symbol1]
        series2 = self.data[symbol2]
        rolling_corr = series1.rolling(window=window).corr(series2)
        return rolling_corr

    def detect_correlation_breakdown(
        self, threshold: float = 0.3
    ) -> List[Tuple[str, str, pd.Timestamp]]:
        """Обнаруживает моменты разрыва корреляций."""
        breakdowns = []
        for pair in self.correlation_pairs:
            if abs(pair.correlation) > 0.7:  # Изначально сильная корреляция
                rolling_corr = self.get_rolling_correlation(pair.symbol1, pair.symbol2)
                # Находим моменты, когда корреляция падает ниже порога
                # Проверяем, что rolling_corr поддерживает сравнение с float
                if hasattr(rolling_corr, '__lt__'):
                    rolling_corr_array = rolling_corr.to_numpy() if hasattr(rolling_corr, 'to_numpy') else np.array(rolling_corr)
                    breakdown_points = rolling_corr.index[rolling_corr_array < threshold]
                else:
                    # Альтернативный способ фильтрации
                    rolling_corr_array = rolling_corr.to_numpy() if hasattr(rolling_corr, 'to_numpy') else np.array(rolling_corr)
                    breakdown_points = rolling_corr.index[rolling_corr_array < threshold] if len(rolling_corr_array) > 0 else pd.Index([])
                for point in breakdown_points:
                    breakdowns.append((pair.symbol1, pair.symbol2, point))
        return breakdowns


# Экспорт интерфейса для обратной совместимости
ICorrelationChain = CorrelationChain
