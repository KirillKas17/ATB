"""
Модуль для анализа корреляций между инструментами
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any, Optional, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from loguru import logger


@dataclass
class CorrelationPair:
    """Класс для хранения информации о коррелирующей паре"""

    symbol1: str
    symbol2: str
    correlation: float
    lag: int
    strength: float


class CorrelationChain:
    """Анализ корреляций между активами."""

    def __init__(self, data: pd.DataFrame):
        """
        Инициализация анализа корреляций.

        Args:
            data: DataFrame с ценами активов
        """
        self.data = data
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.lag_matrix: Optional[pd.DataFrame] = None
        self.chain: Optional[List[Tuple[str, str, float]]] = None

    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Вычисляет корреляцию между двумя Series."""
        if isinstance(x, pd.DataFrame):
            x = pd.Series(x.squeeze())
        if isinstance(y, pd.DataFrame):
            y = pd.Series(y.squeeze())
        return x.corr(y, method="pearson")

    def build_correlation_matrix(self) -> pd.DataFrame:
        """Строит матрицу корреляций."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        corr_matrix = self.data.corr(method="pearson")
        return corr_matrix

    def calculate_lag(
        self,
        x: Union[pd.Series, pd.DataFrame],
        y: Union[pd.Series, pd.DataFrame],
        max_lag: int = 10,
    ) -> int:
        """
        Расчет оптимального лага между активами.

        Args:
            x: Первый актив
            y: Второй актив
            max_lag: Максимальный лаг

        Returns:
            int: Оптимальный лаг
        """
        if isinstance(x, pd.DataFrame):
            x = pd.Series(x.squeeze())
        if isinstance(y, pd.DataFrame):
            y = pd.Series(y.squeeze())

        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise ValueError("Inputs must be pandas Series")

        # Расчет автокорреляционной функции
        acf_x = acf(x, nlags=max_lag)
        acf_y = acf(y, nlags=max_lag)

        # Поиск оптимального лага
        return int(np.argmax(np.correlate(acf_x, acf_y)))

    def build_lag_matrix(self, max_lag: int = 10) -> pd.DataFrame:
        """
        Построение матрицы лагов.

        Args:
            max_lag: Максимальный лаг

        Returns:
            pd.DataFrame: Матрица лагов
        """
        assets = self.data.columns
        lag_matrix = pd.DataFrame(index=pd.Index(assets), columns=pd.Index(assets))

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j:
                    lag = self.calculate_lag(self.data[asset1], self.data[asset2], max_lag=max_lag)
                    lag_matrix.loc[asset1, asset2] = lag
                else:
                    lag_matrix.loc[asset1, asset2] = 0

        self.lag_matrix = lag_matrix
        return lag_matrix

    def find_correlation_chain(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Находит цепочки коррелированных активов."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self.build_correlation_matrix()
        if self.correlation_matrix.empty:
            return []
        symbols = list(self.correlation_matrix.index)
        chains = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr_value = self.correlation_matrix.loc[symbols[i], symbols[j]]
                if isinstance(corr_value, tuple):
                    corr_value = corr_value[0]
                if abs(corr_value) >= threshold:
                    chains.append((symbols[i], symbols[j], float(corr_value)))
        return chains

    def get_leading_assets(self) -> List[str]:
        """Возвращает ведущие активы."""
        if self.chain is None:
            self.chain = self.find_correlation_chain()
        if not self.chain:
            return []
        leading = []
        for chain in self.chain:
            if len(chain) > 0:
                leading.append(chain[0])
        return leading

    def get_correlation_groups(self) -> List[Tuple[str, str, float]]:
        """Возвращает группы коррелированных активов."""
        if self.chain is None:
            self.chain = self.find_correlation_chain()
        return self.chain

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по корреляциям."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self.build_correlation_matrix()
        if self.correlation_matrix.empty:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": self.correlation_matrix.values.mean(),
            "std": self.correlation_matrix.values.std(),
            "min": self.correlation_matrix.values.min(),
            "max": self.correlation_matrix.values.max(),
        }
