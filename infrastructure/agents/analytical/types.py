"""
Типы данных для аналитической интеграции.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


@dataclass
class AnalyticalIntegrationConfig:
    """Конфигурация интеграции аналитических модулей."""

    # Включение модулей
    entanglement_enabled: bool = True
    noise_enabled: bool = True
    mirror_enabled: bool = True
    gravity_enabled: bool = True
    # Пороги активации
    entanglement_threshold: float = 0.95
    noise_threshold: float = 0.7
    mirror_threshold: float = 0.8
    gravity_threshold: float = 1.0
    # Интервалы обновления (секунды)
    update_interval: float = 1.0
    context_cleanup_interval: float = 3600.0  # 1 час
    # Логирование
    enable_detailed_logging: bool = True
    log_analysis_results: bool = True


@dataclass
class AnalyticalData:
    """Аналитические данные."""

    volume_metrics: Dict[str, float] = field(default_factory=dict)
    volatility_metrics: Dict[str, float] = field(default_factory=dict)
    spread_metrics: Dict[str, float] = field(default_factory=dict)
    liquidity_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


@dataclass
class MarketMakerContext:
    """Контекст для маркет-мейкера."""

    symbol: str
    market_regime: str
    liquidity_score: float
    volatility_score: float
    spread_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationConfig:
    """Конфигурация интеграции."""

    cache_ttl: float = 300.0  # 5 минут
    max_cache_size: int = 1000
    enable_analytics: bool = True
    analytics_weight: float = 0.3


@dataclass
class AnalyticalResult:
    """Результат аналитического анализа."""

    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    entanglement_score: Optional[float] = None
    noise_level: Optional[float] = None
    mirror_signals: Optional[List[Dict[str, Any]]] = None
    gravity_risk: Optional[float] = None
    confidence: float = 0.0
    recommendations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingRecommendation:
    """Рекомендация для торговли."""

    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float
    price_offset: float
    position_size_adjustment: float
    aggressiveness_adjustment: float
    reasoning: Dict[str, Any] = field(default_factory=dict)


class IAnalyticalIntegrator(Protocol):
    """Протокол для интегратора аналитических данных."""

    async def integrate_data(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> AnalyticalData:
        """Интегрирует различные источники аналитических данных."""
        ...

    async def get_market_maker_context(self, symbol: str) -> MarketMakerContext:
        """Получает контекст для маркет-мейкера."""
        ...

    def get_trading_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Получает торговые рекомендации."""
        ...

    def should_proceed_with_trade(
        self, symbol: str, trade_aggression: float = 1.0
    ) -> bool:
        """Определяет, следует ли продолжать торговлю."""
        ...

    def get_adjusted_aggressiveness(
        self, symbol: str, base_aggressiveness: float
    ) -> float:
        """Получает скорректированную агрессивность."""
        ...

    def get_adjusted_position_size(self, symbol: str, base_size: float) -> float:
        """Получает скорректированный размер позиции."""
        ...

    def get_adjusted_confidence(self, symbol: str, base_confidence: float) -> float:
        """Получает скорректированную уверенность."""
        ...

    def get_price_offset(self, symbol: str, base_price: float, side: str) -> float:
        """Получает смещение цены."""
        ...
