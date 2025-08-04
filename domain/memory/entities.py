# -*- coding: utf-8 -*-
"""Сущности модуля памяти паттернов."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from shared.numpy_utils import np

from domain.memory.types import (
    MarketFeatures,
    MarketRegime,
    OutcomeType,
    PredictionDirection,
    VolumeProfile,
)
from domain.types.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp


@dataclass
class PatternSnapshot:
    """Снимок рыночного состояния при обнаружении паттерна."""

    pattern_id: str
    timestamp: Timestamp
    symbol: str
    pattern_type: PatternType
    confidence: float
    strength: float
    direction: str
    features: MarketFeatures
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "timestamp": self.timestamp.to_iso(),
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "direction": self.direction,
            "features": self.features.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternSnapshot":
        """Создание из словаря."""
        return cls(
            pattern_id=data["pattern_id"],
            timestamp=Timestamp.from_iso(data["timestamp"]),
            symbol=data["symbol"],
            pattern_type=PatternType(data["pattern_type"]),
            confidence=data["confidence"],
            strength=data["strength"],
            direction=data["direction"],
            features=MarketFeatures.from_dict(data["features"]),
            metadata=data["metadata"],
        )


@dataclass
class PatternOutcome:
    """Результат развития паттерна."""

    pattern_id: str
    symbol: str
    outcome_type: OutcomeType
    timestamp: Timestamp
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

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "outcome_type": self.outcome_type.value,
            "timestamp": self.timestamp.to_iso(),
            "price_change_percent": self.price_change_percent,
            "volume_change_percent": self.volume_change_percent,
            "duration_minutes": self.duration_minutes,
            "max_profit_percent": self.max_profit_percent,
            "max_loss_percent": self.max_loss_percent,
            "final_return_percent": self.final_return_percent,
            "volatility_during": self.volatility_during,
            "volume_profile": self.volume_profile,
            "market_regime": self.market_regime,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternOutcome":
        """Создание из словаря."""
        return cls(
            pattern_id=data["pattern_id"],
            symbol=data["symbol"],
            outcome_type=OutcomeType(data["outcome_type"]),
            timestamp=Timestamp.from_iso(data["timestamp"]),
            price_change_percent=data["price_change_percent"],
            volume_change_percent=data["volume_change_percent"],
            duration_minutes=data["duration_minutes"],
            max_profit_percent=data["max_profit_percent"],
            max_loss_percent=data["max_loss_percent"],
            final_return_percent=data["final_return_percent"],
            volatility_during=data["volatility_during"],
            volume_profile=data["volume_profile"],
            market_regime=data["market_regime"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PredictionResult:
    """Результат прогнозирования на основе исторических паттернов."""

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

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "confidence": self.confidence,
            "predicted_direction": self.predicted_direction,
            "predicted_return_percent": self.predicted_return_percent,
            "predicted_duration_minutes": self.predicted_duration_minutes,
            "predicted_volatility": self.predicted_volatility,
            "similar_cases_count": self.similar_cases_count,
            "success_rate": self.success_rate,
            "avg_return": self.avg_return,
            "avg_duration": self.avg_duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        """Создание из словаря."""
        return cls(
            pattern_id=data["pattern_id"],
            symbol=data["symbol"],
            confidence=data["confidence"],
            predicted_direction=data["predicted_direction"],
            predicted_return_percent=data["predicted_return_percent"],
            predicted_duration_minutes=data["predicted_duration_minutes"],
            predicted_volatility=data["predicted_volatility"],
            similar_cases_count=data["similar_cases_count"],
            success_rate=data["success_rate"],
            avg_return=data["avg_return"],
            avg_duration=data["avg_duration"],
            metadata=data["metadata"],
        )


@dataclass
class PatternCluster:
    """Кластер похожих паттернов."""

    cluster_id: str
    center_features: MarketFeatures
    patterns: List[PatternSnapshot]
    outcomes: List[PatternOutcome]
    avg_similarity: float
    avg_confidence: float
    avg_return: float
    success_rate: float
    size: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "cluster_id": self.cluster_id,
            "center_features": self.center_features.to_dict(),
            "patterns": [p.to_dict() for p in self.patterns],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "avg_similarity": self.avg_similarity,
            "avg_confidence": self.avg_confidence,
            "avg_return": self.avg_return,
            "success_rate": self.success_rate,
            "size": self.size,
            "metadata": self.metadata,
        }


@dataclass
class PatternAnalysis:
    """Анализ паттерна."""

    pattern_id: str
    symbol: str
    pattern_type: PatternType
    timestamp: Timestamp
    # Метрики качества
    quality_score: float
    reliability_score: float
    uniqueness_score: float
    # Аналитические метрики
    market_impact: float
    volume_significance: float
    price_momentum: float
    volatility_impact: float
    # Кластерная информация
    cluster_id: Optional[str] = None
    cluster_similarity: Optional[float] = None
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "timestamp": self.timestamp.to_iso(),
            "quality_score": self.quality_score,
            "reliability_score": self.reliability_score,
            "uniqueness_score": self.uniqueness_score,
            "market_impact": self.market_impact,
            "volume_significance": self.volume_significance,
            "price_momentum": self.price_momentum,
            "volatility_impact": self.volatility_impact,
            "cluster_id": self.cluster_id,
            "cluster_similarity": self.cluster_similarity,
            "metadata": self.metadata,
        }


@dataclass
class MemoryOptimizationResult:
    """Результат оптимизации памяти."""

    # Оптимизированные параметры
    optimal_similarity_threshold: float
    optimal_feature_weights: np.ndarray
    optimal_prediction_params: Dict[str, Any]
    # Метрики качества
    accuracy_improvement: float
    precision_improvement: float
    recall_improvement: float
    f1_improvement: float
    # Статистика
    total_patterns_analyzed: int
    patterns_removed: int
    clusters_created: int
    # Метаданные
    optimization_method: str
    processing_time_seconds: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "optimal_similarity_threshold": self.optimal_similarity_threshold,
            "optimal_feature_weights": self.optimal_feature_weights.tolist(),
            "optimal_prediction_params": self.optimal_prediction_params,
            "accuracy_improvement": self.accuracy_improvement,
            "precision_improvement": self.precision_improvement,
            "recall_improvement": self.recall_improvement,
            "f1_improvement": self.f1_improvement,
            "total_patterns_analyzed": self.total_patterns_analyzed,
            "patterns_removed": self.patterns_removed,
            "clusters_created": self.clusters_created,
            "optimization_method": self.optimization_method,
            "processing_time_seconds": self.processing_time_seconds,
            "metadata": self.metadata,
        }
