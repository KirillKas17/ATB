"""
Строгие типы для модуля торговых сессий.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from typing_extensions import TypeAlias
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    import pandas as pd

# Type aliases for numpy and pandas types
NumpyFloat: TypeAlias = float
NumpyInt: TypeAlias = int
NumpyArray: TypeAlias = List[float]
PandasTimestamp: TypeAlias = datetime

# Типы для рыночных данных
MarketDataFrame = pd.DataFrame
MarketDataSeries = pd.Series

# Типы для числовых значений
NumericValue = Union[float, int, NumpyFloat, NumpyInt]
FloatValue = Union[float, NumpyFloat]
IntValue = Union[int, NumpyInt]

# Типы для временных меток
TimestampValue = Union[str, PandasTimestamp]

# Типы для индексов
IndexValue = pd.Index

# Типы для массивов
ArrayValue = Union[NumpyArray, List[float]]
