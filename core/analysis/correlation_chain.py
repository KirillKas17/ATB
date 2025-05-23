"""
Модуль для анализа корреляций между инструментами
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


@dataclass
class CorrelationPair:
    """Класс для хранения информации о коррелирующей паре"""

    symbol1: str
    symbol2: str
    correlation: float
    lag: int
    strength: float


class CorrelationChain:
    """Класс для анализа цепочек корреляций"""

    def __init__(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data = data or {}
        self.config = config or {}
        self.min_correlation = float(self.config.get("min_correlation", 0.7))
        self.max_lag = int(self.config.get("max_lag", 10))
        self.window_size = int(self.config.get("window_size", 100))
        self.pairs: List[CorrelationPair] = []
        self.chain: List[Tuple[str, str, float]] = []

    def _calculate_correlation(
        self, x: Union[pd.Series, pd.DataFrame], y: Union[pd.Series, pd.DataFrame]
    ) -> float:
        """Расчет корреляции между двумя временными рядами."""
        if isinstance(x, pd.DataFrame):
            x = pd.Series(x.squeeze())
        if isinstance(y, pd.DataFrame):
            y = pd.Series(y.squeeze())
        return float(x.corr(y))

    def build_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Построение матрицы корреляций."""
        symbols = data.columns.tolist()
        corr_matrix = pd.DataFrame(index=pd.Index(symbols), columns=pd.Index(symbols))

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:
                    corr = self._calculate_correlation(data[symbol1], data[symbol2])
                    corr_matrix.loc[symbol1, symbol2] = corr
                    corr_matrix.loc[symbol2, symbol1] = corr

        return corr_matrix

    def find_correlation_chain(
        self, corr_matrix: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """Поиск цепочки корреляций."""
        chain = []
        symbols = corr_matrix.index.tolist()

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:
                    corr_value = float(corr_matrix.loc[symbol1, symbol2])
                    if abs(corr_value) >= self.min_correlation:
                        chain.append((symbol1, symbol2, corr_value))

        return chain

    def get_correlation_groups(
        self, data: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """Получение групп коррелирующих активов."""
        corr_matrix = self.build_correlation_matrix(data)
        self.chain = self.find_correlation_chain(corr_matrix)
        return self.chain

    def find_correlations(self) -> List[CorrelationPair]:
        """Поиск корреляций между всеми парами"""
        if not self.data:
            return []

        symbols = list(self.data.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr, lag = self._calculate_correlation_with_lag(
                    self.data[symbols[i]], self.data[symbols[j]]
                )
                if abs(corr) >= self.min_correlation:
                    self.pairs.append(
                        CorrelationPair(
                            symbol1=symbols[i],
                            symbol2=symbols[j],
                            correlation=float(corr),
                            lag=lag,
                            strength=abs(float(corr)),
                        )
                    )
        return sorted(self.pairs, key=lambda x: abs(x.correlation), reverse=True)

    def _calculate_correlation_with_lag(
        self, series1: pd.DataFrame, series2: pd.DataFrame
    ) -> Tuple[float, int]:
        """Расчет корреляции с учетом лага"""
        best_corr = 0.0
        best_lag = 0

        # Преобразуем DataFrame в Series
        s1 = pd.Series(series1["close"].values)
        s2 = pd.Series(series2["close"].values)

        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                corr = float(s1.corr(s2.shift(-lag)))
            else:
                corr = float(s1.shift(lag).corr(s2))

            if abs(corr) > abs(best_corr):
                best_corr = float(corr)
                best_lag = lag

        return best_corr, best_lag

    def get_strongest_pairs(self, n: int = 5) -> List[CorrelationPair]:
        """Получение n самых сильных корреляций"""
        return sorted(self.pairs, key=lambda x: abs(x.correlation), reverse=True)[:n]

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Получение матрицы корреляций"""
        if not self.data:
            return pd.DataFrame()

        symbols = list(self.data.keys())
        matrix = pd.DataFrame(index=pd.Index(symbols), columns=pd.Index(symbols))

        for pair in self.pairs:
            matrix.loc[pair.symbol1, pair.symbol2] = float(pair.correlation)
            matrix.loc[pair.symbol2, pair.symbol1] = float(pair.correlation)

        return matrix.fillna(0)

    def calculate_lag(self, series1: np.ndarray, series2: np.ndarray) -> int:
        """Расчет оптимального лага между рядами"""
        s1 = pd.Series(series1)
        s2 = pd.Series(series2)
        corr = s1.corr(s2)
        return int(np.argmax(acf(corr, nlags=self.max_lag)))


def calculate_acf(data: np.ndarray, nlags: int = 40) -> np.ndarray:
    """
    Расчет автокорреляционной функции.

    Args:
        data: Входные данные
        nlags: Количество лагов

    Returns:
        np.ndarray: Значения ACF
    """
    result = acf(data, nlags=nlags)
    if isinstance(result, tuple):
        return result[0]  # Возвращаем только массив значений
    return result


def find_correlation_chain(data: List[float], threshold: float = 0.7) -> List[int]:
    """
    Поиск цепочки корреляций.

    Args:
        data: Входные данные
        threshold: Порог корреляции

    Returns:
        List[int]: Индексы коррелирующих элементов
    """
    chain = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            corr = _calculate_correlation(data[i], data[j])
            if abs(corr) >= threshold:
                chain.append(int(i))
                chain.append(int(j))
    return list(set(chain))  # Убираем дубликаты


def find_max_correlation(data: np.ndarray) -> int:
    """
    Поиск максимальной корреляции.

    Args:
        data: Входные данные

    Returns:
        int: Индекс максимальной корреляции
    """
    values = calculate_acf(data)
    return int(np.argmax(values))  # Явное приведение к int


def _find_optimal_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> int:
    """Поиск оптимального лага между рядами."""
    # Преобразуем в Series для корректного расчета корреляции
    x_series = pd.Series(x)
    y_series = pd.Series(y)

    correlations = []
    for lag in range(max_lag):
        corr = x_series.corr(y_series.shift(lag))
        correlations.append(corr)

    # Явно приводим к int
    return int(np.argmax(np.abs(correlations)))


def _calculate_autocorrelation(series: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Расчет автокорреляции ряда."""
    # Используем acf из statsmodels и возвращаем только массив значений
    result = acf(series, nlags=nlags)
    if isinstance(result, tuple):
        return result[0]  # Возвращаем только массив значений
    return result


def _calculate_correlation(x: float, y: float) -> float:
    """
    Расчет корреляции между двумя значениями.

    Args:
        x: Первое значение
        y: Второе значение

    Returns:
        float: Значение корреляции
    """
    return float(np.corrcoef([x], [y])[0, 1])
