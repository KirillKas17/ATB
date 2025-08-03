from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class TrendDirection(Enum):
    """Направление тренда."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class VolatilityTrend(Enum):
    """Тренд волатильности."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class VolumeTrend(Enum):
    """Тренд объема."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class VolatilityMetrics:
    """Метрики волатильности."""
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    volatility_trend: VolatilityTrend


@dataclass
class TrendMetrics:
    """Метрики тренда."""
    trend_direction: TrendDirection
    trend_strength: float
    trend_confidence: float = 0.0
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


@dataclass
class VolumeMetrics:
    """Метрики объема."""
    current_volume: float
    average_volume: float
    volume_trend: VolumeTrend
    volume_ratio: float = 0.0
    unusual_volume: bool = False


@dataclass
class MomentumMetrics:
    """Метрики моментума."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    momentum_score: float = 0.0


@dataclass
class LiquidityMetrics:
    """Метрики ликвидности."""
    bid_ask_spread: float
    market_depth: float
    order_book_imbalance: float
    liquidity_score: float = 0.0


@dataclass
class MarketStressMetrics:
    """Метрики рыночного стресса."""
    stress_index: float
    fear_greed_index: float
    market_regime: str = "normal"
    stress_level: str = "low"


@dataclass
class MarketMetricsResult:
    """Результат расчета рыночных метрик."""
    volatility: VolatilityMetrics
    trend: TrendMetrics
    volume: VolumeMetrics
    momentum: MomentumMetrics
    liquidity: LiquidityMetrics
    stress: MarketStressMetrics
    timestamp: Optional[str] = None
    symbol: Optional[str] = None 