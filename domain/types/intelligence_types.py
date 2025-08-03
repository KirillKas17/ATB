# -*- coding: utf-8 -*-
"""Типы и интерфейсы для модуля intelligence."""
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from typing_extensions import NotRequired

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


# =============================================================================
# ENUMS
# =============================================================================
class PatternType(Enum):
    """Типы паттернов крупного капитала."""

    WHALE_ABSORPTION = "whale_absorption"
    MM_SPOOFING = "mm_spoofing"
    ICEBERG_DETECTION = "iceberg_detection"
    LIQUIDITY_GRAB = "liquidity_grab"
    PUMP_AND_DUMP = "pump_and_dump"
    STOP_HUNTING = "stop_hunting"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    SYNTHETIC_NOISE = "synthetic_noise"
    MIRROR_SIGNAL = "mirror_signal"
    ENTANGLEMENT = "entanglement"


class SignalDirection(Enum):
    """Направления сигналов."""

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


class CorrelationMethod(Enum):
    """Методы корреляции."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    CROSS_CORRELATION = "cross_correlation"


class NoiseType(Enum):
    """Типы шума."""

    NATURAL = "natural"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EntanglementType(Enum):
    """Типы запутанности (entanglement) между активами/рынками."""

    NONE = "none"
    STRONG = "strong"
    WEAK = "weak"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    UNKNOWN = "unknown"


class EntanglementStrength(Enum):
    """Сила запутанности между активами/рынками."""

    NONE = "none"
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREME = "extreme"


# =============================================================================
# TYPED DICTS
# =============================================================================
class OrderBookData(TypedDict):
    """Данные ордербука."""

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: NotRequired[Optional[int]]
    meta: NotRequired[Dict[str, Any]]


class MarketData(TypedDict):
    """Рыночные данные."""

    symbol: str
    timestamp: Timestamp
    price: Price
    volume: Volume
    high: Price
    low: Price
    open: Price
    close: Price


class AnalysisMetadata(TypedDict):
    """Метаданные анализа."""

    data_points: int
    confidence: float
    processing_time_ms: float
    algorithm_version: str
    parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]


class PatternMetrics(TypedDict):
    """Метрики паттерна."""

    volume_anomaly: float
    price_impact: float
    order_book_imbalance: float
    spread_widening: float
    depth_absorption: float
    momentum: float
    volatility: float


class CorrelationMetrics(TypedDict):
    """Метрики корреляции."""

    correlation: float
    p_value: float
    lag: int
    confidence: float
    sample_size: int
    significance: bool


class NoiseMetrics(TypedDict):
    """Метрики шума."""

    fractal_dimension: float
    entropy: float
    noise_type: NoiseType
    synthetic_probability: float
    natural_probability: float


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass(frozen=True)
class OrderBookUpdate:
    """Обновление ордербука."""

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            object.__setattr__(self, "meta", {})


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Снимок ордербука для анализа."""

    exchange: str
    symbol: str
    bids: List[Tuple[Price, Volume]]
    asks: List[Tuple[Price, Volume]]
    timestamp: Timestamp
    sequence_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            object.__setattr__(self, "meta", {})

    def get_mid_price(self) -> Price:
        """Получение средней цены."""
        if not self.bids or not self.asks:
            return Price(Decimal("0"), Currency.USDT, Currency.USDT)
        best_bid = self.bids[0][0].value
        best_ask = self.asks[0][0].value
        mid_price = (best_bid + best_ask) / Decimal("2")
        return Price(mid_price, Currency.USDT, Currency.USDT)

    def get_spread(self) -> Price:
        """Получение спреда."""
        if not self.bids or not self.asks:
            return Price(Decimal("0"), Currency.USDT, Currency.USDT)
        best_bid = self.bids[0][0].value
        best_ask = self.asks[0][0].value
        return Price(best_ask - best_bid, Currency.USDT, Currency.USDT)

    def get_total_volume(self) -> Volume:
        """Получение общего объема."""
        bid_volume = sum(vol.value for _, vol in self.bids)
        ask_volume = sum(vol.value for _, vol in self.asks)
        total_volume = bid_volume + ask_volume
        return Volume(Decimal(str(total_volume)))

    def get_volume_imbalance(self) -> float:
        """Получение дисбаланса объемов."""
        if not self.bids or not self.asks:
            return 0.0
        bid_volume = sum(vol.value for _, vol in self.bids)
        ask_volume = sum(vol.value for _, vol in self.asks)
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        imbalance = float((bid_volume - ask_volume) / total_volume)
        return imbalance


@dataclass(frozen=True)
class PatternDetection:
    """Результат обнаружения паттерна."""

    pattern_type: PatternType
    symbol: str
    timestamp: Timestamp
    confidence: float  # 0.0 - 1.0
    strength: float  # Сила сигнала
    direction: SignalDirection
    metadata: AnalysisMetadata
    metrics: PatternMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_type": self.pattern_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.to_iso(),
            "confidence": self.confidence,
            "strength": self.strength,
            "direction": self.direction.value,
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class EntanglementResult:
    """Результат обнаружения запутанности."""

    is_entangled: bool
    lag_ms: float
    correlation_score: float
    exchange_pair: Tuple[str, str]
    symbol: str
    timestamp: Timestamp
    confidence: float
    metadata: AnalysisMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "is_entangled": self.is_entangled,
            "lag_ms": self.lag_ms,
            "correlation_score": self.correlation_score,
            "exchange_pair": self.exchange_pair,
            "symbol": self.symbol,
            "timestamp": self.timestamp.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MirrorSignal:
    """Результат обнаружения зеркального сигнала."""

    asset1: str
    asset2: str
    best_lag: int
    correlation: float
    p_value: float
    confidence: float
    signal_strength: float
    timestamp: Timestamp
    metadata: AnalysisMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "best_lag": self.best_lag,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "confidence": self.confidence,
            "signal_strength": self.signal_strength,
            "timestamp": self.timestamp.value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class NoiseAnalysisResult:
    """Результат анализа нейронного шума."""

    fractal_dimension: float
    entropy: float
    is_synthetic_noise: bool
    confidence: float
    metadata: AnalysisMetadata
    timestamp: Timestamp
    noise_type: NoiseType
    metrics: NoiseMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "fractal_dimension": self.fractal_dimension,
            "entropy": self.entropy,
            "is_synthetic_noise": self.is_synthetic_noise,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.value,
            "noise_type": self.noise_type.value,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class CorrelationMatrix:
    """Матрица корреляций с лагами."""

    assets: List[str]
    correlation_matrix: np.ndarray
    lag_matrix: np.ndarray
    p_value_matrix: np.ndarray
    confidence_matrix: np.ndarray
    timestamp: Timestamp
    metadata: AnalysisMetadata

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Получение корреляции между активами."""
        try:
            i = self.assets.index(asset1)
            j = self.assets.index(asset2)
            return float(self.correlation_matrix[i, j])
        except ValueError:
            return 0.0

    def get_lag(self, asset1: str, asset2: str) -> int:
        """Получение лага между активами."""
        try:
            i = self.assets.index(asset1)
            j = self.assets.index(asset2)
            return int(self.lag_matrix[i, j])
        except ValueError:
            return 0

    def get_p_value(self, asset1: str, asset2: str) -> float:
        """Получение p-value между активами."""
        try:
            i = self.assets.index(asset1)
            j = self.assets.index(asset2)
            return float(self.p_value_matrix[i, j])
        except ValueError:
            return 1.0

    def get_confidence(self, asset1: str, asset2: str) -> float:
        """Получение уверенности между активами."""
        try:
            i = self.assets.index(asset1)
            j = self.assets.index(asset2)
            return float(self.confidence_matrix[i, j])
        except ValueError:
            return 0.0


# =============================================================================
# PROTOCOLS (INTERFACES)
# =============================================================================
class PatternDetector(Protocol):
    """Протокол для детекторов паттернов."""

    def detect_pattern(
        self, symbol: str, market_data: pd.DataFrame, order_book: OrderBookSnapshot
    ) -> Optional[PatternDetection]:
        """Обнаружение паттерна."""
        ...


class EntanglementDetector(Protocol):
    """Протокол для детекторов запутанности."""

    def detect_entanglement(
        self, exchange1: str, exchange2: str, symbol: str
    ) -> Optional[EntanglementResult]:
        """Обнаружение запутанности."""
        ...

    def process_order_book_updates(
        self, updates: List[OrderBookSnapshot]
    ) -> List[EntanglementResult]:
        """Обработка обновлений ордербуков."""
        ...


class NoiseAnalyzer(Protocol):
    """Протокол для анализаторов шума."""

    def analyze_noise(self, order_book: OrderBookSnapshot) -> NoiseAnalysisResult:
        """Анализ шума."""
        ...

    def compute_fractal_dimension(self, order_book: OrderBookSnapshot) -> float:
        """Вычисление фрактальной размерности."""
        ...

    def compute_entropy(self, order_book: OrderBookSnapshot) -> float:
        """Вычисление энтропии."""
        ...


class MirrorDetector(Protocol):
    """Протокол для детекторов зеркальных сигналов."""

    def detect_mirror_signal(
        self,
        asset1: str,
        asset2: str,
        series1: pd.Series,
        series2: pd.Series,
        max_lag: int = 5,
    ) -> Optional[MirrorSignal]:
        """Обнаружение зеркального сигнала."""
        ...

    def build_correlation_matrix(
        self, assets: List[str], price_data: Dict[str, pd.Series], max_lag: int = 5
    ) -> CorrelationMatrix:
        """Построение матрицы корреляций."""
        ...


class IntelligenceAnalyzer(Protocol):
    """Протокол для интеллектуального анализатора."""

    def analyze_market_intelligence(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        order_book: OrderBookSnapshot,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Комплексный анализ рыночной разведки."""
        ...

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Получение статистики анализа."""
        ...


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================
@dataclass(frozen=True)
class PatternDetectionConfig:
    """Конфигурация детектора паттернов."""

    volume_threshold: float = 1_000_000.0
    price_impact_threshold: float = 0.02
    spread_threshold: float = 0.001
    depth_imbalance_threshold: float = 0.3
    confidence_threshold: float = 0.7
    lookback_periods: int = 50
    volume_sma_periods: int = 20
    min_data_points: int = 30


@dataclass(frozen=True)
class EntanglementConfig:
    """Конфигурация детектора запутанности."""

    max_lag_ms: float = 3.0
    correlation_threshold: float = 0.95
    window_size: int = 100
    min_data_points: int = 50
    price_change_threshold: float = 0.0001
    confidence_threshold: float = 0.8


@dataclass(frozen=True)
class NoiseAnalysisConfig:
    """Конфигурация анализатора шума."""

    fractal_dimension_lower: float = 1.2
    fractal_dimension_upper: float = 1.4
    entropy_threshold: float = 0.7
    min_data_points: int = 50
    window_size: int = 100
    confidence_threshold: float = 0.8


@dataclass(frozen=True)
class MirrorDetectionConfig:
    """Конфигурация детектора зеркальных сигналов."""

    min_correlation: float = 0.3
    max_p_value: float = 0.05
    min_confidence: float = 0.7
    correlation_method: CorrelationMethod = CorrelationMethod.PEARSON
    normalize_data: bool = True
    remove_trend: bool = True
    max_lag: int = 5


# =============================================================================
# TYPE ALIASES
# =============================================================================
OrderBookLevel = Tuple[Price, Volume]
OrderBookLevels = List[OrderBookLevel]
PriceSeries = pd.Series
VolumeSeries = pd.Series
TimeSeries = pd.Series
CorrelationResult = Tuple[float, float]  # (correlation, p_value)
LagResult = Tuple[int, float]  # (lag, correlation)
AnalysisResult = Union[
    PatternDetection, EntanglementResult, MirrorSignal, NoiseAnalysisResult
]
