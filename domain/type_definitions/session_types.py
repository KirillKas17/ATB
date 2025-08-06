"""
Централизованные типы для торговых сессий.
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from domain.value_objects.timestamp import Timestamp

# Новые типы для строгой типизации
SessionId = NewType("SessionId", str)
Symbol = NewType("Symbol", str)
VolumeMultiplier = NewType("VolumeMultiplier", float)
VolatilityMultiplier = NewType("VolatilityMultiplier", float)
ConfidenceScore = NewType("ConfidenceScore", float)
CorrelationScore = NewType("CorrelationScore", float)
# Константы
DEFAULT_LOOKBACK_DAYS: Final[int] = 30
DEFAULT_MIN_DATA_POINTS: Final[int] = 100
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.6
DEFAULT_VOLATILITY_WINDOW_MINUTES: Final[int] = 60
DEFAULT_VOLUME_WINDOW_MINUTES: Final[int] = 30
DEFAULT_PATTERN_MATCHING_THRESHOLD: Final[float] = 0.8


class SessionType(Enum):
    """Типы торговых сессий."""

    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    FRANKFURT = "frankfurt"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"
    OVERLAP_LONDON_NEWYORK = "overlap_london_newyork"
    OVERLAP_NEWYORK_ASIAN = "overlap_newyork_asian"
    GLOBAL = "global"
    CRYPTO_24H = "crypto_24h"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"


class SessionPhase(Enum):
    """Фазы торговой сессии."""

    PRE_OPENING = "pre_opening"
    OPENING = "opening"
    EARLY_SESSION = "early_session"
    MID_SESSION = "mid_session"
    LATE_SESSION = "late_session"
    CLOSING = "closing"
    POST_CLOSING = "post_closing"
    TRANSITION = "transition"
    BREAK = "break"
    OVERLAP_START = "overlap_start"
    OVERLAP_PEAK = "overlap_peak"
    OVERLAP_END = "overlap_end"


class MarketRegime(Enum):
    """Рыночные режимы."""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MANIPULATION = "manipulation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    FLASH_RALLY = "flash_rally"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"


class SessionIntensity(Enum):
    """Интенсивность сессии."""

    EXTREMELY_LOW = "extremely_low"
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREMELY_HIGH = "extremely_high"
    CRITICAL = "critical"


class LiquidityProfile(Enum):
    """Профили ликвидности."""

    EXCESSIVE = "excessive"
    ABUNDANT = "abundant"
    NORMAL = "normal"
    TIGHT = "tight"
    SCARCE = "scarce"
    CRITICAL = "critical"
    DRY = "dry"
    FROZEN = "frozen"


class InfluenceType(Enum):
    """Типы влияния сессии."""

    VOLATILITY = "volatility"
    VOLUME = "volume"
    DIRECTION = "direction"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"


class PriceDirection(Enum):
    """Направления движения цены."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# TypedDict для строгой типизации словарей
class SessionMetrics(TypedDict):
    """Метрики сессии."""

    volume_change_percent: float
    volatility_change_percent: float
    price_direction_bias: float
    momentum_strength: float
    false_breakout_probability: float
    reversal_probability: float
    trend_continuation_probability: float
    influence_duration_minutes: int
    peak_influence_time_minutes: int
    spread_impact: float
    liquidity_impact: float
    correlation_with_other_sessions: float


class MarketConditions(TypedDict):
    """Рыночные условия."""

    volatility: float
    volume: float
    spread: float
    liquidity: float
    momentum: float
    trend_strength: float
    market_regime: MarketRegime
    session_intensity: SessionIntensity


class SessionTransition(TypedDict):
    """Переход между сессиями."""

    from_session: SessionType
    to_session: SessionType
    transition_duration_minutes: int
    volume_decay_rate: float
    volatility_spike_probability: float
    gap_probability: float
    correlation_shift_probability: float
    liquidity_drain_rate: float
    manipulation_window_minutes: int


# Protocol интерфейсы
class SessionTimeProvider(Protocol):
    """Протокол для провайдера времени сессий."""

    def is_active(self, current_time: time) -> bool:
        """Проверка активности сессии."""
        ...

    def get_phase(self, current_time: time) -> SessionPhase:
        """Определение фазы сессии."""
        ...

    def get_duration(self) -> float:
        """Получение длительности сессии в секундах."""
        ...


class SessionMetricsCalculator(Protocol):
    """Протокол для калькулятора метрик сессии."""

    def calculate_volume_impact(self, base_volume: float, phase: SessionPhase) -> float:
        """Расчет влияния на объем."""
        ...

    def calculate_volatility_impact(
        self, base_volatility: float, phase: SessionPhase
    ) -> float:
        """Расчет влияния на волатильность."""
        ...

    def calculate_direction_bias(
        self, phase: SessionPhase, market_conditions: MarketConditions
    ) -> float:
        """Расчет смещения направления."""
        ...


# Pydantic модели для валидации
class SessionTimeWindow(BaseModel):
    """Временное окно сессии."""

    start_time: time
    end_time: time
    timezone: str = Field(default="UTC")
    overlap_start: Optional[time] = None
    overlap_end: Optional[time] = None

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v: time, info: Any) -> time:
        """Валидация времени окончания."""
        if "start_time" in info.data and v == info.data["start_time"]:
            raise ValueError("End time cannot be equal to start time")
        return v

    def is_active(self, current_time: time) -> bool:
        """Проверка активности сессии."""
        if self.overlap_start and self.overlap_end:
            return (
                self.start_time <= current_time <= self.end_time
                or self.overlap_start <= current_time <= self.overlap_end
            )
        return self.start_time <= current_time <= self.end_time

    def get_phase(self, current_time: time) -> SessionPhase:
        """Определение фазы сессии."""
        session_duration = self._get_duration()
        if current_time < self.start_time:
            return SessionPhase.PRE_OPENING
        if current_time == self.start_time:
            return SessionPhase.OPENING
        if self.overlap_start and self.overlap_end:
            if self.overlap_start <= current_time <= self.overlap_end:
                return SessionPhase.OVERLAP_PEAK
        # Разбиваем сессию на фазы
        elapsed = self._get_elapsed_time(current_time)
        progress = elapsed / session_duration
        if progress <= 0.1:
            return SessionPhase.EARLY_SESSION
        elif progress <= 0.4:
            return SessionPhase.MID_SESSION
        elif progress <= 0.8:
            return SessionPhase.LATE_SESSION
        elif progress <= 0.95:
            return SessionPhase.CLOSING
        else:
            return SessionPhase.POST_CLOSING

    def _get_duration(self) -> float:
        """Получение длительности сессии в секундах."""
        start_seconds = self.start_time.hour * 3600 + self.start_time.minute * 60
        end_seconds = self.end_time.hour * 3600 + self.end_time.minute * 60
        if end_seconds < start_seconds:
            end_seconds += 24 * 3600  # Переход через полночь
        return end_seconds - start_seconds

    def _get_elapsed_time(self, current_time: time) -> float:
        """Получение прошедшего времени в секундах."""
        start_seconds = self.start_time.hour * 3600 + self.start_time.minute * 60
        current_seconds = current_time.hour * 3600 + current_time.minute * 60
        if current_seconds < start_seconds:
            current_seconds += 24 * 3600  # Переход через полночь
        return current_seconds - start_seconds


class SessionBehavior(BaseModel):
    """Поведенческие характеристики сессии."""

    # Временные характеристики
    typical_volatility_spike_minutes: int = Field(default=30, ge=1, le=1440)
    volume_peak_hours: List[int] = Field(default_factory=lambda: [2, 4, 6])
    quiet_hours: List[int] = Field(default_factory=lambda: [1, 5])
    # Рыночные характеристики
    avg_volume_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    avg_volatility_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    typical_direction_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Паттерны поведения
    common_patterns: List[str] = Field(default_factory=list)
    false_breakout_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    reversal_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    # Влияние на другие сессии
    overlap_impact: Dict[str, float] = Field(default_factory=dict)

    @field_validator("volume_peak_hours", "quiet_hours")
    @classmethod
    def validate_hours(cls, v: List[int]) -> List[int]:
        """Валидация часов."""
        for hour in v:
            if not 0 <= hour <= 23:
                raise ValueError(f"Hour must be between 0 and 23, got {hour}")
        return v


class SessionProfile(BaseModel):
    """Профиль торговой сессии."""

    session_type: SessionType
    time_window: SessionTimeWindow
    behavior: SessionBehavior
    # Дополнительные характеристики
    description: str = Field(default="")
    is_active: bool = Field(default=True)
    # Расширенные метрики
    typical_volume_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    typical_volatility_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    typical_spread_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    typical_direction_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    liquidity_profile: LiquidityProfile = Field(default=LiquidityProfile.NORMAL)
    intensity_profile: SessionIntensity = Field(default=SessionIntensity.NORMAL)
    market_regime_tendency: MarketRegime = Field(default=MarketRegime.RANGING)
    # Вероятностные характеристики
    whale_activity_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    mm_activity_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    news_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    technical_signal_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    fundamental_impact_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    correlation_breakdown_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    gap_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    false_breakout_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    reversal_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    continuation_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    manipulation_susceptibility: float = Field(default=0.3, ge=0.0, le=1.0)
    # Временные метки
    last_updated: Optional[datetime] = Field(default=None)

    def calculate_session_impact(
        self,
        base_metrics: SessionMetrics,
        current_phase: SessionPhase,
        market_conditions: MarketConditions,
    ) -> SessionMetrics:
        """Расчет влияния сессии на рыночные метрики."""
        # Базовые множители
        volume_mult = self.typical_volume_multiplier
        volatility_mult = self.typical_volatility_multiplier
        spread_mult = self.typical_spread_multiplier
        # Корректировка по фазе
        phase_adjustments = self._get_phase_adjustments(current_phase)
        volume_mult *= phase_adjustments["volume"]
        volatility_mult *= phase_adjustments["volatility"]
        spread_mult *= phase_adjustments["spread"]
        # Корректировка по рыночным условиям
        market_adjustments = self._get_market_adjustments(market_conditions)
        volume_mult *= market_adjustments["volume"]
        volatility_mult *= market_adjustments["volatility"]
        spread_mult *= market_adjustments["spread"]
        # Расчет итоговых метрик
        return SessionMetrics(
            volume_change_percent=base_metrics["volume_change_percent"] * volume_mult,
            volatility_change_percent=base_metrics["volatility_change_percent"]
            * volatility_mult,
            price_direction_bias=base_metrics["price_direction_bias"]
            + self.typical_direction_bias,
            momentum_strength=base_metrics["momentum_strength"]
            * self.technical_signal_strength,
            false_breakout_probability=base_metrics["false_breakout_probability"]
            * (1 + self.manipulation_susceptibility),
            reversal_probability=self.reversal_probability,
            trend_continuation_probability=self.continuation_probability,
            influence_duration_minutes=base_metrics["influence_duration_minutes"],
            peak_influence_time_minutes=base_metrics["peak_influence_time_minutes"],
            spread_impact=base_metrics["spread_impact"] * spread_mult,
            liquidity_impact=base_metrics["liquidity_impact"]
            * (1 - self.manipulation_susceptibility),
            correlation_with_other_sessions=base_metrics[
                "correlation_with_other_sessions"
            ]
            * (1 - self.correlation_breakdown_probability),
        )

    def _get_phase_adjustments(self, phase: SessionPhase) -> Dict[str, float]:
        """Получение корректировок по фазе."""
        adjustments = {"volume": 1.0, "volatility": 1.0, "spread": 1.0}
        if phase == SessionPhase.OPENING:
            adjustments["volume"] = 1.5
            adjustments["volatility"] = 1.8
            adjustments["spread"] = 1.3
        elif phase == SessionPhase.CLOSING:
            adjustments["volume"] = 1.2
            adjustments["volatility"] = 1.4
            adjustments["spread"] = 1.1
        elif phase == SessionPhase.OVERLAP_PEAK:
            adjustments["volume"] = 2.0
            adjustments["volatility"] = 1.6
            adjustments["spread"] = 0.9
        return adjustments

    def _get_market_adjustments(self, conditions: MarketConditions) -> Dict[str, float]:
        """Получение корректировок по рыночным условиям."""
        adjustments = {"volume": 1.0, "volatility": 1.0, "spread": 1.0}
        # Корректировка по режиму рынка
        if conditions["market_regime"] == MarketRegime.VOLATILE:
            adjustments["volatility"] *= 1.5
            adjustments["spread"] *= 1.2
        elif conditions["market_regime"] == MarketRegime.MANIPULATION:
            adjustments["volume"] *= 0.8
            adjustments["spread"] *= 1.5
        # Корректировка по интенсивности
        if conditions["session_intensity"] == SessionIntensity.HIGH:
            adjustments["volume"] *= 1.3
            adjustments["volatility"] *= 1.2
        elif conditions["session_intensity"] == SessionIntensity.LOW:
            adjustments["volume"] *= 0.7
            adjustments["volatility"] *= 0.8
        return adjustments


# Dataclass для результатов анализа
@dataclass(frozen=True)
class SessionAnalysisResult:
    """Результат анализа сессии."""

    session_type: SessionType
    session_phase: SessionPhase
    timestamp: Timestamp
    confidence: ConfidenceScore
    metrics: SessionMetrics
    market_conditions: MarketConditions
    predictions: Dict[str, float]
    risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "session_type": self.session_type.value,
            "session_phase": self.session_phase.value,
            "timestamp": self.timestamp.to_iso(),
            "confidence": float(self.confidence),
            "metrics": dict(self.metrics),
            "market_conditions": {
                **dict(self.market_conditions),
                "market_regime": self.market_conditions["market_regime"].value,
                "session_intensity": self.market_conditions["session_intensity"].value,
            },
            "predictions": self.predictions,
            "risk_factors": self.risk_factors,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SessionAnalysisResult":
        """Создание SessionAnalysisResult из словаря."""
        # Формируем словарь с нужными типами для SessionMetrics
        metrics_dict = {
            "volume_change_percent": float(
                data.get("metrics", {}).get("volume_change_percent", 0.0)
            ),
            "volatility_change_percent": float(
                data.get("metrics", {}).get("volatility_change_percent", 0.0)
            ),
            "price_direction_bias": float(
                data.get("metrics", {}).get("price_direction_bias", 0.0)
            ),
            "momentum_strength": float(
                data.get("metrics", {}).get("momentum_strength", 0.0)
            ),
            "false_breakout_probability": float(
                data.get("metrics", {}).get("false_breakout_probability", 0.0)
            ),
            "reversal_probability": float(
                data.get("metrics", {}).get("reversal_probability", 0.0)
            ),
            "trend_continuation_probability": float(
                data.get("metrics", {}).get("trend_continuation_probability", 0.0)
            ),
            "influence_duration_minutes": int(
                data.get("metrics", {}).get("influence_duration_minutes", 0)
            ),
            "peak_influence_time_minutes": int(
                data.get("metrics", {}).get("peak_influence_time_minutes", 0)
            ),
            "spread_impact": float(data.get("metrics", {}).get("spread_impact", 0.0)),
            "liquidity_impact": float(
                data.get("metrics", {}).get("liquidity_impact", 0.0)
            ),
            "correlation_with_other_sessions": float(
                data.get("metrics", {}).get("correlation_with_other_sessions", 0.0)
            ),
        }
        metrics = SessionMetrics(
            volume_change_percent=float(metrics_dict["volume_change_percent"]),
            volatility_change_percent=float(metrics_dict["volatility_change_percent"]),
            price_direction_bias=float(metrics_dict["price_direction_bias"]),
            momentum_strength=float(metrics_dict["momentum_strength"]),
            false_breakout_probability=float(
                metrics_dict["false_breakout_probability"]
            ),
            reversal_probability=float(metrics_dict["reversal_probability"]),
            trend_continuation_probability=float(
                metrics_dict["trend_continuation_probability"]
            ),
            influence_duration_minutes=int(metrics_dict["influence_duration_minutes"]),
            peak_influence_time_minutes=int(
                metrics_dict["peak_influence_time_minutes"]
            ),
            spread_impact=float(metrics_dict["spread_impact"]),
            liquidity_impact=float(metrics_dict["liquidity_impact"]),
            correlation_with_other_sessions=float(
                metrics_dict["correlation_with_other_sessions"]
            ),
        )
        # Формируем словарь с нужными типами для MarketConditions
        mc = data.get("market_conditions", {})
        market_conditions_dict = {
            "volatility": float(mc.get("volatility", 0.0)),
            "volume": float(mc.get("volume", 0.0)),
            "spread": float(mc.get("spread", 0.0)),
            "liquidity": float(mc.get("liquidity", 0.0)),
            "momentum": float(mc.get("momentum", 0.0)),
            "trend_strength": float(mc.get("trend_strength", 0.0)),
            "market_regime": mc.get("market_regime", "RANGING"),
            "session_intensity": mc.get("session_intensity", "NORMAL"),
        }

        def safe_float(val: object) -> float:
            # Если Enum — берём value
            if hasattr(val, "value"):
                val = getattr(val, "value")
            try:
                return float(str(val))
            except (TypeError, ValueError):
                return 0.0

        def safe_enum(enum_cls: Any, val: Any, default: Any) -> Any:
            try:
                return enum_cls(val)
            except Exception:
                return default

        market_conditions = MarketConditions(
            volatility=safe_float(market_conditions_dict["volatility"]),
            volume=safe_float(market_conditions_dict["volume"]),
            spread=safe_float(market_conditions_dict["spread"]),
            liquidity=safe_float(market_conditions_dict["liquidity"]),
            momentum=safe_float(market_conditions_dict["momentum"]),
            trend_strength=safe_float(market_conditions_dict["trend_strength"]),
            market_regime=safe_enum(
                MarketRegime,
                market_conditions_dict["market_regime"],
                MarketRegime.RANGING,
            ),
            session_intensity=safe_enum(
                SessionIntensity,
                market_conditions_dict["session_intensity"],
                SessionIntensity.NORMAL,
            ),
        )
        return SessionAnalysisResult(
            session_type=SessionType(data["session_type"]),
            session_phase=SessionPhase(data["session_phase"]),
            timestamp=Timestamp.from_iso(data["timestamp"]),
            confidence=ConfidenceScore(float(data["confidence"])),
            metrics=metrics,
            market_conditions=market_conditions,
            predictions={k: float(v) for k, v in data["predictions"].items()},
            risk_factors=list(data["risk_factors"]),
        )
