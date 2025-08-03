"""
Type stubs for talib library
"""

from typing import Any, Tuple, Union
import pandas as pd

def RSI(data: pd.Series, timeperiod: int = 14) -> pd.Series: ...

def MACD(
    data: pd.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]: ...

def BBANDS(
    data: pd.Series,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]: ...

def ATR(
    high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
) -> pd.Series: ...

def EMA(data: pd.Series, timeperiod: int = 14) -> pd.Series: ...

def SMA(data: pd.Series, timeperiod: int = 20) -> pd.Series: ...

def ADX(
    high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
) -> pd.Series: ...

def STOCHRSI(data: pd.Series) -> Tuple[pd.Series, pd.Series]: ...

def KELTNER(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]: ...

def VWAP(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series: ...

def FRACTAL(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]: ... 