"""
Модели паттернов маркет-мейкера.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.types.market_maker_types import (
    Accuracy,
    AverageReturn,
    BookPressure,
    Confidence,
    LiquidityDepth,
    MarketMakerPatternProtocol,
    MarketMakerPatternType,
    MarketMicrostructure,
    MarketPhase,
    MatchedPatternDict,
    MatchedPatternProtocol,
    OrderBookLevel,
    OrderImbalance,
    PatternConfidence,
    PatternContext,
    PatternFeaturesDict,
    PatternFeaturesProtocol,
    PatternMemoryDict,
    PatternMemoryProtocol,
    PatternOutcome,
    PatternResultDict,
    PatternResultProtocol,
    PriceReaction,
    PriceVolatility,
    SignalStrength,
    SimilarityScore,
    SpreadChange,
    SuccessCount,
    Symbol,
    TimeDuration,
    TotalCount,
    TradeData,
    VolumeConcentration,
    VolumeDelta,
    validate_accuracy,
    validate_book_pressure,
    validate_confidence,
    validate_non_negative_float,
    validate_order_imbalance,
    validate_positive_int,
    validate_signal_strength,
    validate_similarity_score,
)

logger = logging.getLogger(__name__)


def _empty_market_microstructure() -> MarketMicrostructure:
    """Создать пустую микроструктуру рынка."""
    return {}


def _empty_pattern_context() -> PatternContext:
    """Создать пустой контекст паттерна."""
    return {}


@dataclass(frozen=True)
class PatternFeatures(PatternFeaturesProtocol):
    """Признаки паттерна маркет-мейкера с валидацией."""

    book_pressure: BookPressure
    volume_delta: VolumeDelta
    price_reaction: PriceReaction
    spread_change: SpreadChange
    order_imbalance: OrderImbalance
    liquidity_depth: LiquidityDepth
    time_duration: TimeDuration
    volume_concentration: VolumeConcentration
    price_volatility: PriceVolatility
    market_microstructure: MarketMicrostructure = field(
        default_factory=_empty_market_microstructure
    )

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        # Валидация диапазонов
        validate_book_pressure(self.book_pressure)
        validate_order_imbalance(self.order_imbalance)
        validate_positive_int(self.time_duration, "time_duration")
        validate_non_negative_float(self.liquidity_depth, "liquidity_depth")
        validate_non_negative_float(self.volume_concentration, "volume_concentration")
        validate_non_negative_float(self.price_volatility, "price_volatility")

    def to_dict(self) -> PatternFeaturesDict:
        """Сериализация в словарь."""
        return {
            "book_pressure": float(self.book_pressure),
            "volume_delta": float(self.volume_delta),
            "price_reaction": float(self.price_reaction),
            "spread_change": float(self.spread_change),
            "order_imbalance": float(self.order_imbalance),
            "liquidity_depth": float(self.liquidity_depth),
            "time_duration": int(self.time_duration),
            "volume_concentration": float(self.volume_concentration),
            "price_volatility": float(self.price_volatility),
            "market_microstructure": self.market_microstructure,
        }

    @classmethod
    def from_dict(cls, data: PatternFeaturesDict) -> "PatternFeatures":
        """Десериализация из словаря."""
        return cls(
            book_pressure=BookPressure(data["book_pressure"]),
            volume_delta=VolumeDelta(data["volume_delta"]),
            price_reaction=PriceReaction(data["price_reaction"]),
            spread_change=SpreadChange(data["spread_change"]),
            order_imbalance=OrderImbalance(data["order_imbalance"]),
            liquidity_depth=LiquidityDepth(data["liquidity_depth"]),
            time_duration=TimeDuration(data["time_duration"]),
            volume_concentration=VolumeConcentration(data["volume_concentration"]),
            price_volatility=PriceVolatility(data["price_volatility"]),
            market_microstructure=data.get(
                "market_microstructure", _empty_market_microstructure()
            ),
        )

    def get_overall_strength(self) -> float:
        """Расчет общей силы паттерна."""
        weights = {
            "book_pressure": 0.15,
            "volume_delta": 0.12,
            "price_reaction": 0.15,
            "spread_change": 0.10,
            "order_imbalance": 0.15,
            "liquidity_depth": 0.08,
            "volume_concentration": 0.10,
            "price_volatility": 0.05,
        }
        strength = 0.0
        for attr, weight in weights.items():
            value = abs(getattr(self, attr))
            strength += value * weight
        return min(1.0, strength)

    def get_direction_bias(self) -> float:
        """Расчет направленного смещения."""
        # Положительные значения указывают на бычий паттерн
        direction_score = (
            self.book_pressure * 0.3
            + self.order_imbalance * 0.3
            + self.price_reaction * 0.4
        )
        return max(-1.0, min(1.0, direction_score))


@dataclass(frozen=True)
class PatternResult(PatternResultProtocol):
    """Результат паттерна через определенное время."""

    outcome: PatternOutcome
    price_change_5min: float
    price_change_15min: float
    price_change_30min: float
    volume_change: float
    volatility_change: float
    market_context: PatternContext = field(default_factory=_empty_pattern_context)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        validate_non_negative_float(abs(self.volume_change), "volume_change")
        validate_non_negative_float(abs(self.volatility_change), "volatility_change")

    def to_dict(self) -> PatternResultDict:
        """Сериализация в словарь."""
        return {
            "outcome": self.outcome.value,
            "price_change_5min": self.price_change_5min,
            "price_change_15min": self.price_change_15min,
            "price_change_30min": self.price_change_30min,
            "volume_change": self.volume_change,
            "volatility_change": self.volatility_change,
            "market_context": self.market_context,
        }

    @classmethod
    def from_dict(cls, data: PatternResultDict) -> "PatternResult":
        """Десериализация из словаря."""
        return cls(
            outcome=PatternOutcome(data["outcome"]),
            price_change_5min=data["price_change_5min"],
            price_change_15min=data["price_change_15min"],
            price_change_30min=data["price_change_30min"],
            volume_change=data["volume_change"],
            volatility_change=data["volatility_change"],
            market_context=data.get("market_context", _empty_pattern_context()),
        )

    def get_expected_return(self, time_horizon: str = "15min") -> float:
        """Получение ожидаемой доходности для временного горизонта."""
        horizon_map = {
            "5min": self.price_change_5min,
            "15min": self.price_change_15min,
            "30min": self.price_change_30min,
        }
        return horizon_map.get(time_horizon, self.price_change_15min)

    def is_successful(self, threshold: float = 0.01) -> bool:
        """Проверка успешности паттерна."""
        return abs(self.price_change_15min) >= threshold

    def get_risk_reward_ratio(self) -> float:
        """Расчет соотношения риск/доходность."""
        if self.volatility_change == 0:
            return 0.0
        return abs(self.price_change_15min) / abs(self.volatility_change)


@dataclass
class MarketMakerPattern(MarketMakerPatternProtocol):
    """Паттерн маркет-мейкера с улучшенной логикой."""

    pattern_type: MarketMakerPatternType
    symbol: Symbol
    timestamp: datetime
    features: PatternFeatures
    confidence: Confidence
    context: PatternContext = field(default_factory=_empty_pattern_context)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        validate_confidence(self.confidence)
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "pattern_type": self.pattern_type.value,
            "symbol": str(self.symbol),
            "timestamp": self.timestamp.isoformat(),
            "features": self.features.to_dict(),
            "confidence": float(self.confidence),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketMakerPattern":
        """Десериализация из словаря."""
        features = PatternFeatures.from_dict(data["features"])
        return cls(
            pattern_type=MarketMakerPatternType(data["pattern_type"]),
            symbol=Symbol(data["symbol"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            features=features,
            confidence=Confidence(data["confidence"]),
            context=data.get("context", _empty_pattern_context()),
        )

    def get_pattern_strength(self) -> float:
        """Расчет силы паттерна."""
        return self.features.get_overall_strength() * float(self.confidence)

    def get_direction(self) -> str:
        """Определение направления паттерна."""
        bias = self.features.get_direction_bias()
        if bias > 0.1:
            return "bullish"
        elif bias < -0.1:
            return "bearish"
        else:
            return "neutral"

    def get_confidence_level(self) -> PatternConfidence:
        """Получение уровня уверенности."""
        conf_value = float(self.confidence)
        if conf_value >= 0.8:
            return PatternConfidence.VERY_HIGH
        elif conf_value >= 0.6:
            return PatternConfidence.HIGH
        elif conf_value >= 0.4:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW

    def is_high_confidence(self) -> bool:
        """Проверка высокой уверенности."""
        return float(self.confidence) >= 0.7

    def get_market_phase(self) -> MarketPhase:
        """Определение рыночной фазы на основе паттерна."""
        pattern_phase_map = {
            MarketMakerPatternType.ACCUMULATION: MarketPhase.ACCUMULATION,
            MarketMakerPatternType.ABSORPTION: MarketPhase.ACCUMULATION,
            MarketMakerPatternType.SPOOFING: MarketPhase.TRANSITION,
            MarketMakerPatternType.EXIT: MarketPhase.DISTRIBUTION,
            MarketMakerPatternType.PRESSURE_ZONE: MarketPhase.TRANSITION,
            MarketMakerPatternType.LIQUIDITY_GRAB: MarketPhase.MARKUP,
            MarketMakerPatternType.STOP_HUNT: MarketPhase.MARKDOWN,
            MarketMakerPatternType.FAKE_BREAKOUT: MarketPhase.TRANSITION,
            MarketMakerPatternType.WASH_TRADING: MarketPhase.TRANSITION,
            MarketMakerPatternType.PUMP_AND_DUMP: MarketPhase.MARKUP,
        }
        return pattern_phase_map.get(self.pattern_type, MarketPhase.TRANSITION)


@dataclass
class PatternMemory(PatternMemoryProtocol):
    """Память паттерна с результатами и статистикой."""

    pattern: MarketMakerPattern
    result: Optional[PatternResultProtocol] = None
    accuracy: Accuracy = Accuracy(0.0)
    avg_return: AverageReturn = AverageReturn(0.0)
    success_count: SuccessCount = SuccessCount(0)
    total_count: TotalCount = TotalCount(0)
    last_seen: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        validate_accuracy(self.accuracy)
        validate_positive_int(self.total_count, "total_count")
        validate_positive_int(self.success_count, "success_count")
        if self.success_count > self.total_count:
            raise ValueError("Success count cannot exceed total count")

    def update_result(self, result: PatternResultProtocol) -> None:
        """Обновление результата паттерна с пересчетом статистики."""
        self.result = result
        self.total_count = TotalCount(self.total_count + 1)
        if result.outcome == PatternOutcome.SUCCESS:
            self.success_count = SuccessCount(self.success_count + 1)
        # Пересчет точности
        self.accuracy = Accuracy(self.success_count / self.total_count)
        # Пересчет средней доходности
        if self.total_count > 1:
            self.avg_return = AverageReturn(
                (
                    float(self.avg_return) * (self.total_count - 1)
                    + result.price_change_15min
                )
                / self.total_count
            )
        else:
            self.avg_return = AverageReturn(result.price_change_15min)
        self.last_seen = datetime.now()
        logger.info(
            f"Updated pattern memory: {self.pattern.symbol} {self.pattern.pattern_type.value} "
            f"accuracy={self.accuracy:.3f} avg_return={self.avg_return:.3f}"
        )

    def to_dict(self) -> PatternMemoryDict:
        """Сериализация в словарь."""
        return {
            "pattern": self.pattern.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "accuracy": float(self.accuracy),
            "avg_return": float(self.avg_return),
            "success_count": int(self.success_count),
            "total_count": int(self.total_count),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }

    @classmethod
    def from_dict(cls, data: PatternMemoryDict) -> "PatternMemory":
        """Десериализация из словаря."""
        pattern = MarketMakerPattern.from_dict(data["pattern"])
        result = None
        if data["result"]:
            result = PatternResult.from_dict(data["result"])
        return cls(
            pattern=pattern,
            result=result,
            accuracy=Accuracy(data["accuracy"]),
            avg_return=AverageReturn(data["avg_return"]),
            success_count=SuccessCount(data["success_count"]),
            total_count=TotalCount(data["total_count"]),
            last_seen=(
                datetime.fromisoformat(data["last_seen"]) if data["last_seen"] else None
            ),
        )

    def is_reliable(self, min_accuracy: float = 0.6, min_count: int = 5) -> bool:
        """Проверка надежности паттерна."""
        return float(self.accuracy) >= min_accuracy and self.total_count >= min_count

    def get_expected_outcome(self) -> PatternResult:
        """Получение ожидаемого результата на основе исторических данных."""
        if not self.is_reliable():
            return PatternResult(
                outcome=PatternOutcome.NEUTRAL,
                price_change_5min=0.0,
                price_change_15min=0.0,
                price_change_30min=0.0,
                volume_change=0.0,
                volatility_change=0.0,
            )
        # Прогнозируем результат на основе исторических данных
        expected_return = float(self.avg_return)
        outcome = (
            PatternOutcome.SUCCESS
            if expected_return > 0.005
            else (
                PatternOutcome.FAILURE
                if expected_return < -0.005
                else PatternOutcome.NEUTRAL
            )
        )
        return PatternResult(
            outcome=outcome,
            price_change_5min=expected_return * 0.3,
            price_change_15min=expected_return,
            price_change_30min=expected_return * 1.5,
            volume_change=0.0,  # Требует дополнительного анализа
            volatility_change=0.0,  # Требует дополнительного анализа
        )

    def get_confidence_boost(self, similarity_score: float) -> float:
        """Расчет увеличения уверенности на основе схожести."""
        if not self.is_reliable():
            return 0.0
        # Учитываем точность, количество наблюдений и схожесть
        reliability_factor = float(self.accuracy) * min(1.0, self.total_count / 10)
        return similarity_score * reliability_factor


@dataclass(frozen=True)
class MatchedPattern(MatchedPatternProtocol):
    """Совпадение текущего поведения с историческим паттерном."""

    pattern_memory: PatternMemory
    similarity_score: SimilarityScore
    confidence_boost: Confidence
    expected_outcome: PatternResult
    signal_strength: SignalStrength

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        validate_similarity_score(self.similarity_score)
        validate_confidence(self.confidence_boost)
        validate_signal_strength(self.signal_strength)

    def to_dict(self) -> MatchedPatternDict:
        """Сериализация в словарь."""
        return {
            "pattern_memory": self.pattern_memory.to_dict(),
            "similarity_score": float(self.similarity_score),
            "confidence_boost": float(self.confidence_boost),
            "expected_outcome": self.expected_outcome.to_dict(),
            "signal_strength": float(self.signal_strength),
        }

    @classmethod
    def from_dict(cls, data: MatchedPatternDict) -> "MatchedPattern":
        """Десериализация из словаря."""
        return cls(
            pattern_memory=PatternMemory.from_dict(data["pattern_memory"]),
            similarity_score=SimilarityScore(data["similarity_score"]),
            confidence_boost=Confidence(data["confidence_boost"]),
            expected_outcome=PatternResult.from_dict(data["expected_outcome"]),
            signal_strength=SignalStrength(data["signal_strength"]),
        )

    def get_combined_confidence(self) -> float:
        """Получение комбинированной уверенности."""
        base_confidence = float(self.pattern_memory.pattern.confidence)
        return min(1.0, base_confidence + float(self.confidence_boost))

    def is_high_quality_match(self) -> bool:
        """Проверка высокого качества совпадения."""
        return (
            float(self.similarity_score) >= 0.8
            and float(self.confidence_boost) >= 0.2
            and self.pattern_memory.is_reliable()
        )

    def get_trading_signal(self) -> Dict[str, Any]:
        """Получение торгового сигнала."""
        return {
            "pattern_type": self.pattern_memory.pattern.pattern_type.value,
            "symbol": self.pattern_memory.pattern.symbol,
            "direction": self.pattern_memory.pattern.get_direction(),
            "confidence": self.get_combined_confidence(),
            "signal_strength": float(self.signal_strength),
            "expected_return": self.expected_outcome.get_expected_return(),
            "time_horizon": "15min",
            "risk_level": "medium" if float(self.signal_strength) < 0.5 else "high",
        }
