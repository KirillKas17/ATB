# -*- coding: utf-8 -*-
"""Reversal Signal Domain Model for Price Reversal Prediction."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from domain.type_definitions.prediction_types import (
    ConfidenceScore,
    DivergenceType,
    ReversalDirection,
    RiskScore,
    SignalStrength,
    SignalStrengthScore,
)
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp


@dataclass
class PivotPoint:
    """Точка разворота (пивот)."""

    price: Price
    timestamp: Timestamp
    volume: float
    pivot_type: Literal["high", "low"]
    strength: float  # 0.0 - 1.0
    confirmation_levels: List[float] = field(default_factory=list)
    volume_cluster: Optional[float] = None
    fibonacci_levels: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Pivot strength must be between 0.0 and 1.0, got {self.strength}"
            )
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass
class FibonacciLevel:
    """Уровень Фибоначчи."""

    level: float  # 23.6, 38.2, 50.0, 61.8, 78.6
    price: Price
    strength: float  # 0.0 - 1.0
    volume_cluster: Optional[float] = None
    confluence_count: int = 0

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        valid_levels = [23.6, 38.2, 50.0, 61.8, 78.6]
        if self.level not in valid_levels:
            raise ValueError(
                f"Invalid Fibonacci level: {self.level}. Must be one of {valid_levels}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Fibonacci strength must be between 0.0 and 1.0, got {self.strength}"
            )
        if self.confluence_count < 0:
            raise ValueError(
                f"Confluence count cannot be negative: {self.confluence_count}"
            )


@dataclass
class VolumeProfile:
    """Профиль объема."""

    price_level: Price
    volume_density: float
    poc_price: Price  # Point of Control
    value_area_high: Price
    value_area_low: Price
    timestamp: Timestamp
    volume_nodes: List[Dict[str, float]] = field(default_factory=list)
    imbalance_ratio: float = 0.0

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.volume_density < 0:
            raise ValueError(
                f"Volume density cannot be negative: {self.volume_density}"
            )
        if not -1.0 <= self.imbalance_ratio <= 1.0:
            raise ValueError(
                f"Imbalance ratio must be between -1.0 and 1.0, got {self.imbalance_ratio}"
            )


@dataclass
class LiquidityCluster:
    """Кластер ликвидности."""

    price: Price
    volume: float
    side: Literal["bid", "ask"]
    cluster_size: int
    strength: float  # 0.0 - 1.0
    timestamp: Timestamp

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")
        if self.cluster_size < 1:
            raise ValueError(f"Cluster size must be positive: {self.cluster_size}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Strength must be between 0.0 and 1.0, got {self.strength}"
            )


@dataclass
class DivergenceSignal:
    """Сигнал дивергенции."""

    type: DivergenceType
    indicator: str  # RSI, MACD, Stochastic, etc.
    price_highs: List[float]
    price_lows: List[float]
    indicator_highs: List[float]
    indicator_lows: List[float]
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    timestamp: Timestamp

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Strength must be between 0.0 and 1.0, got {self.strength}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not self.indicator:
            raise ValueError("Indicator cannot be empty")


@dataclass
class CandlestickPattern:
    """Свечной паттерн."""

    name: str
    direction: ReversalDirection
    strength: float  # 0.0 - 1.0
    confirmation_level: float  # 0.0 - 1.0
    volume_confirmation: bool
    timestamp: Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.name:
            raise ValueError("Pattern name cannot be empty")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Strength must be between 0.0 and 1.0, got {self.strength}"
            )
        if not 0.0 <= self.confirmation_level <= 1.0:
            raise ValueError(
                f"Confirmation level must be between 0.0 and 1.0, got {self.confirmation_level}"
            )


@dataclass
class MomentumAnalysis:
    """Анализ импульса."""

    timestamp: Timestamp
    momentum_loss: float
    velocity_change: float
    acceleration: float
    volume_momentum: float
    price_momentum: float
    momentum_divergence: Optional[float] = None

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not isinstance(self.momentum_loss, (int, float)):
            raise ValueError(
                f"Momentum loss must be numeric, got {type(self.momentum_loss)}"
            )
        if not isinstance(self.velocity_change, (int, float)):
            raise ValueError(
                f"Velocity change must be numeric, got {type(self.velocity_change)}"
            )
        if not isinstance(self.acceleration, (int, float)):
            raise ValueError(
                f"Acceleration must be numeric, got {type(self.acceleration)}"
            )
        if not isinstance(self.volume_momentum, (int, float)):
            raise ValueError(
                f"Volume momentum must be numeric, got {type(self.volume_momentum)}"
            )
        if not isinstance(self.price_momentum, (int, float)):
            raise ValueError(
                f"Price momentum must be numeric, got {type(self.price_momentum)}"
            )


@dataclass
class MeanReversionBand:
    """Полоса возврата к среднему."""

    upper_band: Price
    lower_band: Price
    middle_line: Price
    deviation: float
    band_width: float
    current_position: float  # 0.0 - 1.0 (где находится цена в полосе)
    timestamp: Timestamp

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.deviation < 0:
            raise ValueError(f"Deviation cannot be negative: {self.deviation}")
        if self.band_width < 0:
            raise ValueError(f"Band width cannot be negative: {self.band_width}")
        if not 0.0 <= self.current_position <= 1.0:
            raise ValueError(
                f"Current position must be between 0.0 and 1.0, got {self.current_position}"
            )


@dataclass
class ReversalSignal:
    """Сигнал разворота цены."""

    symbol: str
    direction: ReversalDirection
    pivot_price: Price
    confidence: ConfidenceScore
    horizon: timedelta
    signal_strength: SignalStrengthScore
    timestamp: Timestamp
    is_controversial: bool = False
    agreement_score: float = 0.0
    pivot_points: List[PivotPoint] = field(default_factory=list)
    fibonacci_levels: List[FibonacciLevel] = field(default_factory=list)
    liquidity_clusters: List[LiquidityCluster] = field(default_factory=list)
    divergence_signals: List[DivergenceSignal] = field(default_factory=list)
    candlestick_patterns: List[CandlestickPattern] = field(default_factory=list)
    controversy_reasons: List[Dict[str, Any]] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: Timestamp = field(default_factory=lambda: Timestamp(datetime.now()))
    volume_profile: Optional[VolumeProfile] = None
    momentum_analysis: Optional[MomentumAnalysis] = None
    mean_reversion_band: Optional[MeanReversionBand] = None

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError(
                f"Signal strength must be between 0.0 and 1.0, got {self.signal_strength}"
            )
        if not 0.0 <= self.agreement_score <= 1.0:
            raise ValueError(
                f"Agreement score must be between 0.0 and 1.0, got {self.agreement_score}"
            )
        if self.horizon.total_seconds() <= 0:
            raise ValueError("Horizon must be positive")

    @property
    def strength_category(self) -> SignalStrength:
        """Категория силы сигнала."""
        if self.signal_strength < 0.3:
            return SignalStrength.WEAK
        elif self.signal_strength < 0.6:
            return SignalStrength.MODERATE
        elif self.signal_strength < 0.8:
            return SignalStrength.STRONG
        else:
            return SignalStrength.VERY_STRONG

    @property
    def is_expired(self) -> bool:
        """Проверка истечения срока действия сигнала."""
        current_time = datetime.now()
        signal_time = datetime.fromtimestamp(self.timestamp.value.timestamp())
        return current_time > signal_time + self.horizon

    @property
    def time_to_expiry(self) -> timedelta:
        """Время до истечения сигнала."""
        current_time = datetime.now()
        signal_time = datetime.fromtimestamp(self.timestamp.value.timestamp())
        expiry_time = signal_time + self.horizon
        return expiry_time - current_time

    @property
    def risk_level(self) -> str:
        """Уровень риска сигнала."""
        risk_score = self._calculate_risk_score()
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "extreme"

    def enhance_confidence(self, factor: float) -> None:
        """Усиление уверенности в сигнале."""
        if not 0.0 <= factor <= 1.0:
            raise ValueError(
                f"Enhancement factor must be between 0.0 and 1.0, got {factor}"
            )
        new_confidence = min(
            1.0, float(self.confidence) + factor * (1.0 - float(self.confidence))
        )
        self.confidence = ConfidenceScore(new_confidence)
        self.last_updated = Timestamp(datetime.now())
        logger.debug(f"Enhanced confidence for {self.symbol}: {self.confidence:.3f}")

    def reduce_confidence(self, factor: float) -> None:
        """Снижение уверенности в сигнале."""
        if not 0.0 <= factor <= 1.0:
            raise ValueError(
                f"Reduction factor must be between 0.0 and 1.0, got {factor}"
            )
        new_confidence = max(
            0.0, float(self.confidence) - factor * float(self.confidence)
        )
        self.confidence = ConfidenceScore(new_confidence)
        self.last_updated = Timestamp(datetime.now())
        logger.debug(f"Reduced confidence for {self.symbol}: {self.confidence:.3f}")

    def mark_controversial(
        self, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Пометка сигнала как спорного."""
        if not reason:
            raise ValueError("Controversy reason cannot be empty")
        self.is_controversial = True
        controversy_record = {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        self.controversy_reasons.append(controversy_record)
        logger.warning(
            f"Reversal signal marked as controversial for {self.symbol}: {reason}"
        )

    def update_agreement_score(self, score: float) -> None:
        """Обновление оценки согласованности."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Agreement score must be between 0.0 and 1.0, got {score}"
            )
        self.agreement_score = score
        self.last_updated = Timestamp(datetime.now())
        # Автоматическое усиление/ослабление на основе согласованности
        if score > 0.7:
            self.enhance_confidence(0.1)
        elif score < 0.3:
            self.reduce_confidence(0.1)

    def add_divergence_signal(self, divergence: DivergenceSignal) -> None:
        """Добавление сигнала дивергенции."""
        if not isinstance(divergence, DivergenceSignal):
            raise TypeError(f"Expected DivergenceSignal, got {type(divergence)}")
        self.divergence_signals.append(divergence)
        self._recalculate_signal_strength()

    def add_candlestick_pattern(self, pattern: CandlestickPattern) -> None:
        """Добавление свечного паттерна."""
        if not isinstance(pattern, CandlestickPattern):
            raise TypeError(f"Expected CandlestickPattern, got {type(pattern)}")
        self.candlestick_patterns.append(pattern)
        self._recalculate_signal_strength()

    def update_momentum_analysis(self, momentum: MomentumAnalysis) -> None:
        """Обновление анализа импульса."""
        if not isinstance(momentum, MomentumAnalysis):
            raise TypeError(f"Expected MomentumAnalysis, got {type(momentum)}")
        self.momentum_analysis = momentum
        self._recalculate_signal_strength()

    def update_mean_reversion_band(self, band: MeanReversionBand) -> None:
        """Обновление полосы возврата к среднему."""
        if not isinstance(band, MeanReversionBand):
            raise TypeError(f"Expected MeanReversionBand, got {type(band)}")
        self.mean_reversion_band = band
        self._recalculate_signal_strength()

    def _recalculate_signal_strength(self) -> None:
        """Пересчет силы сигнала на основе всех компонентов."""
        try:
            base_strength = float(self.confidence)
            # Усиление на основе дивергенций
            if self.divergence_signals:
                divergence_strength = sum(d.strength for d in self.divergence_signals)
                base_strength *= 1.0 + divergence_strength * 0.2
            # Усиление на основе свечных паттернов
            if self.candlestick_patterns:
                pattern_strength = sum(p.strength for p in self.candlestick_patterns)
                base_strength *= 1.0 + pattern_strength * 0.15
            # Усиление на основе импульса
            if self.momentum_analysis:
                momentum_factor = abs(self.momentum_analysis.momentum_loss)
                base_strength *= 1.0 + momentum_factor * 0.1
            # Усиление на основе полосы возврата к среднему
            if self.mean_reversion_band:
                if (
                    self.mean_reversion_band.current_position > 0.8
                    or self.mean_reversion_band.current_position < 0.2
                ):
                    base_strength *= 1.1
            # Усиление на основе согласованности
            base_strength *= 1.0 + self.agreement_score * 0.1
            self.signal_strength = SignalStrengthScore(min(1.0, base_strength))
        except Exception as e:
            logger.error(f"Error recalculating signal strength: {e}")

    def _calculate_risk_score(self) -> float:
        """Вычисление оценки риска."""
        try:
            risk_factors = []
            # Риск на основе уверенности
            risk_factors.append(1.0 - float(self.confidence))
            # Риск на основе спорности
            if self.is_controversial:
                risk_factors.append(0.3)
            # Риск на основе согласованности
            risk_factors.append(1.0 - self.agreement_score)
            # Риск на основе времени до истечения
            if self.time_to_expiry.total_seconds() < 300:  # 5 минут
                risk_factors.append(0.2)
            # Риск на основе силы сигнала
            risk_factors.append(1.0 - float(self.signal_strength))
            return sum(risk_factors) / len(risk_factors)
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "pivot_price": self.pivot_price.value,
            "confidence": self.confidence,
            "horizon_seconds": self.horizon.total_seconds(),
            "signal_strength": self.signal_strength,
            "timestamp": self.timestamp.value,
            "strength_category": self.strength_category.value,
            "risk_level": self.risk_level,
            "is_expired": self.is_expired,
            "time_to_expiry_seconds": self.time_to_expiry.total_seconds(),
            "is_controversial": self.is_controversial,
            "agreement_score": self.agreement_score,
            "last_updated": self.last_updated.value,
            "pivot_points_count": len(self.pivot_points),
            "fibonacci_levels_count": len(self.fibonacci_levels),
            "divergence_signals_count": len(self.divergence_signals),
            "candlestick_patterns_count": len(self.candlestick_patterns),
            "liquidity_clusters_count": len(self.liquidity_clusters),
            "controversy_reasons_count": len(self.controversy_reasons),
            "analysis_metadata": self.analysis_metadata,
            "risk_metrics": self.risk_metrics,
        }

    def __str__(self) -> str:
        """Строковое представление сигнала."""
        return (
            f"ReversalSignal({self.symbol}, {self.direction.value}, "
            f"price={self.pivot_price.value:.2f}, confidence={self.confidence:.2f}, "
            f"strength={self.signal_strength:.2f}, horizon={self.horizon}, "
            f"risk={self.risk_level})"
        )

    def __repr__(self) -> str:
        """Представление для отладки."""
        return self.__str__()
