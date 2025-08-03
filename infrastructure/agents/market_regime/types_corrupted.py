import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict


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
