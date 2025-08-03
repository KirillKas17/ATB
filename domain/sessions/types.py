"""
Строгие типы для модуля торговых сессий.
"""

from typing import List, Optional, Union, TypeVar, Type
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# Определение типов pandas и numpy
PandasSeries = Series
PandasDataFrame = DataFrame
PandasIndex = pd.Index
PandasTimestamp = pd.Timestamp
NumpyArray = np.ndarray
NumpyFloat = np.float64
NumpyInt = np.int64

# Типы для рыночных данных
MarketDataFrame = PandasDataFrame
MarketDataSeries = PandasSeries

# Типы для числовых значений
NumericValue = Union[float, int, NumpyFloat, NumpyInt]
FloatValue = Union[float, NumpyFloat]
IntValue = Union[int, NumpyInt]

# Типы для временных меток
TimestampValue = Union[str, PandasTimestamp]

# Типы для индексов
IndexValue = PandasIndex

# Типы для массивов
ArrayValue = Union[NumpyArray, List[float]]
