"""
Базовые реализации стратегий.
"""

import logging
from typing import List, Type, Union

from domain.strategies.trend_following_strategy import TrendFollowingStrategy
from domain.strategies.mean_reversion_strategy import MeanReversionStrategy
from domain.strategies.breakout_strategy import BreakoutStrategy
from domain.strategies.scalping_strategy import ScalpingStrategy
from domain.strategies.arbitrage_strategy import ArbitrageStrategy
from domain.strategies.strategy_types import (
    ArbitrageParams,
    BreakoutParams,
    MeanReversionParams,
    ScalpingParams,
    StrategyParameters,
    TrendFollowingParams,
)
from domain.strategies.utils import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_momentum,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
    calculate_volume_sma,
    detect_support_resistance,
)

logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "ScalpingStrategy",
    "ArbitrageStrategy",
]
