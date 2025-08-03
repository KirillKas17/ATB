"""
Типы данных для портфельного агента.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class AssetMetrics:
    """Data class to store asset-specific metrics"""

    symbol: str
    expected_return: float
    risk_score: float
    liquidity_score: float
    trend_strength: float
    volume_score: float
    correlation_with_btc: float
    current_weight: float
    target_weight: float


@dataclass
class PortfolioState:
    """Состояние портфеля"""

    total_value: float
    total_pnl: float
    total_pnl_percent: float
    risk_metrics: Dict[str, float]
    allocation: Dict[str, float]
    last_updated: datetime
    performance_history: List[Dict[str, Any]]


@dataclass
class TradeSuggestion:
    """Предложение сделки"""

    symbol: str
    action: str  # "buy" или "sell"
    amount: float
    current_price: float
    target_price: float
    confidence: float
    reason: str
    timestamp: datetime


@dataclass
class PortfolioConstraints:
    """Ограничения портфеля"""

    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_aversion: float = 1.0
    max_correlation: float = 0.7
    rebalance_threshold: float = 0.1
    max_position_size: float = 0.3
    min_position_size: float = 0.05
