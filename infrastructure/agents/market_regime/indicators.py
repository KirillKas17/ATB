import pandas as pd
import numpy as np
from typing import Any, Dict

from infrastructure.core.technical import (
    calculate_adx,
    calculate_atr,
    calculate_fractals,
    calculate_obv,
    calculate_wave_clusters,
    rsi as calculate_rsi,
)
from shared.logging import setup_logger

from .types import IIndicatorCalculator

logger = setup_logger(__name__)


class DefaultIndicatorCalculator(IIndicatorCalculator):
    """Реализация калькулятора индикаторов по умолчанию"""

    def calculate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет всех ключевых индикаторов: ADX, RSI, ATR, OBV, fractals, wave_clusters.
        Возвращает словарь значений для unit-тестов и дальнейшей обработки.
        """
        try:
            result = {}
            # ADX - Average Directional Index
            result["adx"] = calculate_adx(
                pd.Series(dataframe["high"]),
                pd.Series(dataframe["low"]),
                pd.Series(dataframe["close"]),
                period=14,
            )
            # RSI - Relative Strength Index
            result["rsi"] = calculate_rsi(pd.Series(dataframe["close"]), period=14)
            # ATR - Average True Range
            result["atr"] = calculate_atr(
                pd.Series(dataframe["high"]),
                pd.Series(dataframe["low"]),
                pd.Series(dataframe["close"]),
                period=14,
            )
            # OBV - On-Balance Volume
            result["obv"] = calculate_obv(
                pd.Series(dataframe["close"]), pd.Series(dataframe["volume"])
            )
            # Fractals - фрактальные паттерны
            fractals_result = calculate_fractals(
                pd.Series(dataframe["high"]), pd.Series(dataframe["low"])
            )
            if isinstance(fractals_result, tuple):
                result["fractals"] = fractals_result[0]
            else:
                result["fractals"] = fractals_result
            # Wave clusters - волновые кластеры
            wave_clusters_result = calculate_wave_clusters(
                np.asarray(dataframe["close"].values)
            )
            if isinstance(wave_clusters_result, list):
                result["wave_clusters"] = wave_clusters_result[0] if wave_clusters_result else []
            else:
                result["wave_clusters"] = wave_clusters_result
            return result
        except Exception as e:
            logger.error(
                f"Ошибка при расчёте индикаторов (DefaultIndicatorCalculator): {str(e)}"
            )
            return {}
