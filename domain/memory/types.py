# -*- coding: utf-8 -*-
"""Типы для модуля памяти паттернов."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from shared.numpy_utils import np
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from domain.types.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp


# =============================================================================
# ENUMS
# =============================================================================
class OutcomeType(Enum):
    """Типы исходов паттернов."""

    PROFITABLE = "profitable"
    UNPROFITABLE = "unprofitable"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class VolumeProfile(Enum):
    """Профили объема."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class MarketRegime(Enum):
    """Рыночные режимы."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"


class PredictionDirection(Enum):
    """Направления прогноза."""

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


# =============================================================================
# TYPED DICTS
# =============================================================================
class MarketFeaturesDict(TypedDict):
    """Словарь рыночных характеристик."""

    price: float
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    volatility: float
    volume: float
    volume_change_1m: float
    volume_change_5m: float
    volume_sma_ratio: float
    spread: float
    spread_change: float
    bid_volume: float
    ask_volume: float
    order_book_imbalance: float
    depth_absorption: float
    entropy: float
    gravity: float
    latency: float
    correlation: float
    whale_signal: float
    mm_signal: float
    external_sync: bool


class PatternSnapshotDict(TypedDict):
    """Словарь снимка паттерна."""

    pattern_id: str
    timestamp: str
    symbol: str
    pattern_type: str
    confidence: float
    strength: float
    direction: str
    features: MarketFeaturesDict
    metadata: Dict[str, Any]


class PatternOutcomeDict(TypedDict):
    """Словарь исхода паттерна."""

    pattern_id: str
    symbol: str
    outcome_type: str
    timestamp: str
    price_change_percent: float
    volume_change_percent: float
    duration_minutes: int
    max_profit_percent: float
    max_loss_percent: float
    final_return_percent: float
    volatility_during: float
    volume_profile: str
    market_regime: str
    metadata: Dict[str, Any]


class PredictionResultDict(TypedDict):
    """Словарь результата прогнозирования."""

    pattern_id: str
    symbol: str
    confidence: float
    predicted_direction: str
    predicted_return_percent: float
    predicted_duration_minutes: int
    predicted_volatility: float
    similar_cases_count: int
    success_rate: float
    avg_return: float
    avg_duration: float
    metadata: Dict[str, Any]


class MemoryStatisticsDict(TypedDict):
    """Словарь статистики памяти."""

    total_snapshots: int
    total_outcomes: int
    pattern_type_stats: Dict[str, int]
    outcome_type_stats: Dict[str, int]
    symbol_stats: Dict[str, int]
    avg_confidence: float
    avg_success_rate: float
    last_cleanup: NotRequired[Optional[str]]


class SimilarityMetricsDict(TypedDict):
    """Словарь метрик сходства."""

    similarity_score: float
    confidence_boost: float
    signal_strength: float
    pattern_type: str
    timestamp: str
    accuracy: float
    avg_return: float


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass(frozen=True)
class PatternMemoryConfig:
    """Конфигурация системы памяти паттернов."""

    # Пути к данным
    db_path: str = "data/pattern_memory.db"
    # Пороги сходства
    similarity_threshold: float = 0.9
    max_similar_cases: int = 10
    min_similar_cases: int = 3
    # Параметры очистки
    days_to_keep: int = 30
    max_snapshots_per_symbol: int = 1000
    # Параметры прогнозирования
    confidence_threshold: float = 0.7
    min_success_rate: float = 0.6
    min_accuracy: float = 0.5
    # Параметры производительности
    cache_size: int = 1000
    batch_size: int = 100
    connection_timeout: float = 30.0


@dataclass(frozen=True)
class MemoryStatistics:
    """Статистика системы памяти."""

    total_snapshots: int
    total_outcomes: int
    pattern_type_stats: Dict[str, int]
    outcome_type_stats: Dict[str, int]
    symbol_stats: Dict[str, int]
    avg_confidence: float
    avg_success_rate: float
    last_cleanup: Optional[datetime] = None

    def to_dict(self) -> MemoryStatisticsDict:
        """Преобразование в словарь."""
        return {
            "total_snapshots": self.total_snapshots,
            "total_outcomes": self.total_outcomes,
            "pattern_type_stats": self.pattern_type_stats,
            "outcome_type_stats": self.outcome_type_stats,
            "symbol_stats": self.symbol_stats,
            "avg_confidence": self.avg_confidence,
            "avg_success_rate": self.avg_success_rate,
            "last_cleanup": (
                self.last_cleanup.isoformat() if self.last_cleanup else None
            ),
        }


@dataclass(frozen=True)
class SimilarityMetrics:
    """Метрики сходства паттернов."""

    similarity_score: float
    confidence_boost: float
    signal_strength: float
    pattern_type: PatternType
    timestamp: Timestamp
    accuracy: float
    avg_return: float

    def to_dict(self) -> SimilarityMetricsDict:
        """Преобразование в словарь."""
        return {
            "similarity_score": self.similarity_score,
            "confidence_boost": self.confidence_boost,
            "signal_strength": self.signal_strength,
            "pattern_type": self.pattern_type.value,
            "timestamp": self.timestamp.to_iso(),
            "accuracy": self.accuracy,
            "avg_return": self.avg_return,
        }


@dataclass(frozen=True)
class PredictionMetadata:
    """Метаданные прогнозирования."""

    algorithm_version: str
    processing_time_ms: float
    data_points_used: int
    confidence_interval: Tuple[float, float]
    model_parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "algorithm_version": self.algorithm_version,
            "processing_time_ms": self.processing_time_ms,
            "data_points_used": self.data_points_used,
            "confidence_interval": self.confidence_interval,
            "model_parameters": self.model_parameters,
            "quality_metrics": self.quality_metrics,
        }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class MarketFeatures(BaseModel):
    """Рыночные характеристики для snapshot."""

    # Ценовые характеристики
    price: float = Field(..., ge=0.0, description="Текущая цена")
    price_change_1m: float = Field(..., description="Изменение цены за 1 минуту")
    price_change_5m: float = Field(..., description="Изменение цены за 5 минут")
    price_change_15m: float = Field(..., description="Изменение цены за 15 минут")
    volatility: float = Field(..., ge=0.0, description="Волатильность")
    # Объемные характеристики
    volume: float = Field(..., ge=0.0, description="Объем торгов")
    volume_change_1m: float = Field(..., description="Изменение объема за 1 минуту")
    volume_change_5m: float = Field(..., description="Изменение объема за 5 минут")
    volume_sma_ratio: float = Field(..., description="Отношение объема к SMA")
    # Характеристики стакана
    spread: float = Field(..., ge=0.0, description="Спред")
    spread_change: float = Field(..., description="Изменение спреда")
    bid_volume: float = Field(..., ge=0.0, description="Объем на покупку")
    ask_volume: float = Field(..., ge=0.0, description="Объем на продажу")
    order_book_imbalance: float = Field(
        ..., ge=-1.0, le=1.0, description="Дисбаланс ордербука"
    )
    depth_absorption: float = Field(..., ge=0.0, description="Поглощение глубины")
    # Аналитические характеристики
    entropy: float = Field(..., ge=0.0, description="Энтропия")
    gravity: float = Field(..., description="Гравитация ликвидности")
    latency: float = Field(..., ge=0.0, description="Латентность")
    correlation: float = Field(..., ge=-1.0, le=1.0, description="Корреляция")
    # Сигналы крупного капитала
    whale_signal: float = Field(..., ge=-1.0, le=1.0, description="Сигнал китов")
    mm_signal: float = Field(..., ge=-1.0, le=1.0, description="Сигнал маркет-мейкера")
    external_sync: bool = Field(..., description="Внешняя синхронизация")

    def to_vector(self) -> np.ndarray:
        """Преобразование в числовой вектор для сравнения."""
        return np.array(
            [
                self.price_change_1m,
                self.price_change_5m,
                self.price_change_15m,
                self.volatility,
                self.volume_change_1m,
                self.volume_change_5m,
                self.volume_sma_ratio,
                self.spread,
                self.spread_change,
                self.order_book_imbalance,
                self.depth_absorption,
                self.entropy,
                self.gravity,
                self.correlation,
                self.whale_signal,
                self.mm_signal,
            ]
        )

    def to_dict(self) -> MarketFeaturesDict:
        """Преобразование в словарь."""
        return {
            "price": self.price,
            "price_change_1m": self.price_change_1m,
            "price_change_5m": self.price_change_5m,
            "price_change_15m": self.price_change_15m,
            "volatility": self.volatility,
            "volume": self.volume,
            "volume_change_1m": self.volume_change_1m,
            "volume_change_5m": self.volume_change_5m,
            "volume_sma_ratio": self.volume_sma_ratio,
            "spread": self.spread,
            "spread_change": self.spread_change,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "order_book_imbalance": self.order_book_imbalance,
            "depth_absorption": self.depth_absorption,
            "entropy": self.entropy,
            "gravity": self.gravity,
            "latency": self.latency,
            "correlation": self.correlation,
            "whale_signal": self.whale_signal,
            "mm_signal": self.mm_signal,
            "external_sync": self.external_sync,
        }

    @classmethod
    def from_dict(cls, data: MarketFeaturesDict) -> "MarketFeatures":
        """Создание из словаря."""
        return cls(**data)
