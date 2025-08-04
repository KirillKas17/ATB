# -*- coding: utf-8 -*-
"""Профиль торгового символа с текущей структурой и фазой рынка."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from loguru import logger

from domain.types import (
    DataInsufficientError,
    MarketPhase,
    PriceStructureMetrics,
    SymbolProfileProtocol,
    TrendStrengthValue,
    ValidationError,
    VolumeProfileMetrics,
)


@dataclass
class VolumeProfile:
    """Профиль объема торгов с валидацией."""

    current_volume: float = field(default=0.0)
    avg_volume_1m: float = field(default=0.0)
    avg_volume_5m: float = field(default=0.0)
    avg_volume_15m: float = field(default=0.0)
    volume_trend: float = field(default=0.0)  # -1.0 to 1.0
    volume_stability: float = field(default=0.0)  # 0.0 to 1.0
    volume_anomaly_ratio: float = field(default=1.0)

    def __post_init__(self) -> None:
        """Валидация данных профиля объема."""
        if self.current_volume < 0:
            raise ValidationError("value", "", "validation", "Current volume cannot be negative")
        if not -1.0 <= self.volume_trend <= 1.0:
            raise ValidationError("value", "", "validation", "Volume trend must be between -1.0 and 1.0")
        if not 0.0 <= self.volume_stability <= 1.0:
            raise ValidationError("value", "", "validation", "Volume stability must be between 0.0 and 1.0")
        if self.volume_anomaly_ratio < 0:
            raise ValidationError("value", "", "validation", "Volume anomaly ratio cannot be negative")

    def to_metrics(self) -> Dict[str, float]:
        """Преобразование в метрики."""
        return {
            "current_volume": self.current_volume,
            "avg_volume_1m": self.avg_volume_1m,
            "avg_volume_5m": self.avg_volume_5m,
            "avg_volume_15m": self.avg_volume_15m,
            "volume_trend": self.volume_trend,
            "volume_stability": self.volume_stability,
            "volume_anomaly_ratio": self.volume_anomaly_ratio,
        }


@dataclass
class PriceStructure:
    """Структура цены и технические уровни с валидацией."""

    current_price: float = field(default=0.0)
    atr: float = field(default=0.0)
    atr_percent: float = field(default=0.0)  # ATR как процент от цены
    vwap: float = field(default=0.0)
    vwap_deviation: float = field(default=0.0)
    support_level: Optional[float] = field(default=None)
    resistance_level: Optional[float] = field(default=None)
    pivot_point: Optional[float] = field(default=None)
    price_entropy: float = field(default=0.0)
    volatility_compression: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Валидация данных структуры цены."""
        if self.current_price < 0:
            raise ValidationError("value", "", "validation", "Current price cannot be negative")
        if self.atr < 0:
            raise ValidationError("value", "", "validation", "ATR cannot be negative")
        if self.atr_percent < 0:
            raise ValidationError("value", "", "validation", "ATR percent cannot be negative")
        if self.vwap < 0:
            raise ValidationError("value", "", "validation", "VWAP cannot be negative")
        if not 0.0 <= self.price_entropy <= 1.0:
            raise ValidationError("value", "", "validation", "Price entropy must be between 0.0 and 1.0")
        if not 0.0 <= self.volatility_compression <= 1.0:
            raise ValidationError("value", "", "validation", "Volatility compression must be between 0.0 and 1.0")
        # Валидация уровней
        if self.support_level is not None and self.support_level < 0:
            raise ValidationError("value", "", "validation", "Support level cannot be negative")
        if self.resistance_level is not None and self.resistance_level < 0:
            raise ValidationError("value", "", "validation", "Resistance level cannot be negative")
        if self.pivot_point is not None and self.pivot_point < 0:
            raise ValidationError("value", "", "validation", "Pivot point cannot be negative")

    def to_metrics(self) -> Dict[str, Any]:
        """Преобразование в метрики."""
        return {
            "current_price": self.current_price,
            "atr": self.atr,
            "atr_percent": self.atr_percent,
            "vwap": self.vwap,
            "vwap_deviation": self.vwap_deviation,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "pivot_point": self.pivot_point,
            "price_entropy": self.price_entropy,
            "volatility_compression": self.volatility_compression,
        }


@dataclass
class OrderBookMetricsData:
    """Метрики стакана заявок с валидацией."""

    bid_ask_spread: float = field(default=0.0)
    spread_percent: float = field(default=0.0)
    bid_volume: float = field(default=0.0)
    ask_volume: float = field(default=0.0)
    volume_imbalance: float = field(default=0.0)  # -1.0 to 1.0
    order_book_symmetry: float = field(default=0.0)  # 0.0 to 1.0
    liquidity_depth: float = field(default=0.0)
    absorption_ratio: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Валидация метрик стакана заявок."""
        if self.bid_ask_spread < 0:
            raise ValidationError("value", "", "validation", "Bid-ask spread cannot be negative")
        if self.spread_percent < 0:
            raise ValidationError("value", "", "validation", "Spread percent cannot be negative")
        if self.bid_volume < 0:
            raise ValidationError("value", "", "validation", "Bid volume cannot be negative")
        if self.ask_volume < 0:
            raise ValidationError("value", "", "validation", "Ask volume cannot be negative")
        if not -1.0 <= self.volume_imbalance <= 1.0:
            raise ValidationError("value", "", "validation", "Volume imbalance must be between -1.0 and 1.0")
        if not 0.0 <= self.order_book_symmetry <= 1.0:
            raise ValidationError("value", "", "validation", "Order book symmetry must be between 0.0 and 1.0")
        if self.liquidity_depth < 0:
            raise ValidationError("value", "", "validation", "Liquidity depth cannot be negative")
        if not 0.0 <= self.absorption_ratio <= 1.0:
            raise ValidationError("value", "", "validation", "Absorption ratio must be between 0.0 and 1.0")

    def to_metrics(self) -> Dict[str, float]:
        """Преобразование в метрики."""
        return {
            "bid_ask_spread": self.bid_ask_spread,
            "spread_percent": self.spread_percent,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "volume_imbalance": self.volume_imbalance,
            "order_book_symmetry": self.order_book_symmetry,
            "liquidity_depth": self.liquidity_depth,
            "absorption_ratio": self.absorption_ratio,
        }


@dataclass
class PatternMetricsData:
    """Метрики паттернов и сигналов с валидацией."""

    mirror_neuron_score: float = field(default=0.0)
    gravity_anomaly_score: float = field(default=0.0)
    reversal_setup_score: float = field(default=0.0)
    pattern_confidence: float = field(default=0.0)
    historical_pattern_match: float = field(default=0.0)
    pattern_complexity: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Валидация метрик паттернов."""
        if not 0.0 <= self.mirror_neuron_score <= 1.0:
            raise ValidationError("value", "", "validation", "Mirror neuron score must be between 0.0 and 1.0")
        if not 0.0 <= self.gravity_anomaly_score <= 1.0:
            raise ValidationError("value", "", "validation", "Gravity anomaly score must be between 0.0 and 1.0")
        if not 0.0 <= self.reversal_setup_score <= 1.0:
            raise ValidationError("value", "", "validation", "Reversal setup score must be between 0.0 and 1.0")
        if not 0.0 <= self.pattern_confidence <= 1.0:
            raise ValidationError("value", "", "validation", "Pattern confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.historical_pattern_match <= 1.0:
            raise ValidationError(
                "Historical pattern match must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.pattern_complexity <= 1.0:
            raise ValidationError("value", "", "validation", "Pattern complexity must be between 0.0 and 1.0")

    def to_metrics(self) -> Dict[str, float]:
        """Преобразование в метрики."""
        return {
            "mirror_neuron_score": self.mirror_neuron_score,
            "gravity_anomaly_score": self.gravity_anomaly_score,
            "reversal_setup_score": self.reversal_setup_score,
            "pattern_confidence": self.pattern_confidence,
            "historical_pattern_match": self.historical_pattern_match,
            "pattern_complexity": self.pattern_complexity,
        }


@dataclass
class SessionMetricsData:
    """Метрики торговой сессии с валидацией."""

    session_alignment: float = field(default=0.0)  # 0.0 to 1.0
    session_activity: float = field(default=0.0)
    session_volatility: float = field(default=0.0)
    session_momentum: float = field(default=0.0)
    session_influence_score: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Валидация метрик сессии."""
        if not 0.0 <= self.session_alignment <= 1.0:
            raise ValidationError("value", "", "validation", "Session alignment must be between 0.0 and 1.0")
        if self.session_activity < 0:
            raise ValidationError("value", "", "validation", "Session activity cannot be negative")
        if self.session_volatility < 0:
            raise ValidationError("value", "", "validation", "Session volatility cannot be negative")
        if not 0.0 <= self.session_influence_score <= 1.0:
            raise ValidationError("value", "", "validation", "Session influence score must be between 0.0 and 1.0")

    def to_metrics(self) -> Dict[str, float]:
        """Преобразование в метрики."""
        return {
            "session_alignment": self.session_alignment,
            "session_activity": self.session_activity,
            "session_volatility": self.session_volatility,
            "session_momentum": self.session_momentum,
            "session_influence_score": self.session_influence_score,
        }


@dataclass
class SymbolProfile(SymbolProfileProtocol):
    """Профиль торгового символа с полной аналитической информацией."""

    id: UUID = field(default_factory=uuid4)
    symbol: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные характеристики
    market_phase: MarketPhase = field(default=MarketPhase.NO_STRUCTURE)
    opportunity_score: float = field(default=0.0)
    confidence: float = field(default=0.0)
    # Детальные метрики
    volume_profile: VolumeProfile = field(default_factory=VolumeProfile)
    price_structure: PriceStructure = field(default_factory=PriceStructure)
    order_book_metrics: OrderBookMetricsData = field(default_factory=OrderBookMetricsData)
    pattern_metrics: PatternMetricsData = field(default_factory=PatternMetricsData)
    session_metrics: SessionMetricsData = field(default_factory=SessionMetricsData)
    # Дополнительные данные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация профиля символа."""
        if not self.symbol:
            raise ValidationError("value", "", "validation", "Symbol cannot be empty")
        if not 0.0 <= self.opportunity_score <= 1.0:
            raise ValidationError("value", "", "validation", "Opportunity score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("value", "", "validation", "Confidence must be between 0.0 and 1.0")

    def is_opportunity(self, min_score: float = 0.78) -> bool:
        """Проверка, является ли символ торговой возможностью."""
        if not 0.0 <= min_score <= 1.0:
            raise ValidationError("value", "", "validation", "Minimum score must be between 0.0 and 1.0")
        return (
            self.opportunity_score >= min_score
            and self.market_phase != MarketPhase.NO_STRUCTURE
            and self.confidence >= 0.6
        )

    def get_phase_description(self) -> str:
        """Получение описания фазы рынка."""
        phase_descriptions = {
            MarketPhase.ACCUMULATION: "Накопление - низкая активность, подготовка к движению",
            MarketPhase.BREAKOUT_SETUP: "Подготовка к пробою - формирование структуры",
            MarketPhase.BREAKOUT_ACTIVE: "Активный пробой - сильное движение",
            MarketPhase.EXHAUSTION: "Истощение - завершение движения",
            MarketPhase.REVERSION_POTENTIAL: "Потенциал разворота - признаки смены тренда",
            MarketPhase.NO_STRUCTURE: "Отсутствие структуры - шумовой режим",
        }
        return phase_descriptions.get(self.market_phase, "Неизвестная фаза")

    def get_opportunity_summary(self) -> Dict[str, Any]:
        """Получение краткого описания торговой возможности."""
        return {
            "symbol": self.symbol,
            "opportunity_score": self.opportunity_score,
            "market_phase": self.market_phase.value,
            "confidence": self.confidence,
            "volume_trend": self.volume_profile.volume_trend,
            "price_entropy": self.price_structure.price_entropy,
            "spread_percent": self.order_book_metrics.spread_percent,
            "pattern_confidence": self.pattern_metrics.pattern_confidence,
            "session_alignment": self.session_metrics.session_alignment,
            "is_opportunity": self.is_opportunity(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации."""
        return {
            "id": str(self.id),
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "market_phase": self.market_phase.value,
            "opportunity_score": self.opportunity_score,
            "confidence": self.confidence,
            "volume_profile": {
                "current_volume": self.volume_profile.current_volume,
                "avg_volume_1m": self.volume_profile.avg_volume_1m,
                "avg_volume_5m": self.volume_profile.avg_volume_5m,
                "avg_volume_15m": self.volume_profile.avg_volume_15m,
                "volume_trend": self.volume_profile.volume_trend,
                "volume_stability": self.volume_profile.volume_stability,
                "volume_anomaly_ratio": self.volume_profile.volume_anomaly_ratio,
            },
            "price_structure": {
                "current_price": self.price_structure.current_price,
                "atr": self.price_structure.atr,
                "atr_percent": self.price_structure.atr_percent,
                "vwap": self.price_structure.vwap,
                "vwap_deviation": self.price_structure.vwap_deviation,
                "support_level": self.price_structure.support_level,
                "resistance_level": self.price_structure.resistance_level,
                "pivot_point": self.price_structure.pivot_point,
                "price_entropy": self.price_structure.price_entropy,
                "volatility_compression": self.price_structure.volatility_compression,
            },
            "order_book_metrics": {
                "bid_ask_spread": self.order_book_metrics.bid_ask_spread,
                "spread_percent": self.order_book_metrics.spread_percent,
                "bid_volume": self.order_book_metrics.bid_volume,
                "ask_volume": self.order_book_metrics.ask_volume,
                "volume_imbalance": self.order_book_metrics.volume_imbalance,
                "order_book_symmetry": self.order_book_metrics.order_book_symmetry,
                "liquidity_depth": self.order_book_metrics.liquidity_depth,
                "absorption_ratio": self.order_book_metrics.absorption_ratio,
            },
            "pattern_metrics": {
                "mirror_neuron_score": self.pattern_metrics.mirror_neuron_score,
                "gravity_anomaly_score": self.pattern_metrics.gravity_anomaly_score,
                "reversal_setup_score": self.pattern_metrics.reversal_setup_score,
                "pattern_confidence": self.pattern_metrics.pattern_confidence,
                "historical_pattern_match": self.pattern_metrics.historical_pattern_match,
                "pattern_complexity": self.pattern_metrics.pattern_complexity,
            },
            "session_metrics": {
                "session_alignment": self.session_metrics.session_alignment,
                "session_activity": self.session_metrics.session_activity,
                "session_volatility": self.session_metrics.session_volatility,
                "session_momentum": self.session_metrics.session_momentum,
                "session_influence_score": self.session_metrics.session_influence_score,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolProfile":
        """Создание из словаря с валидацией."""
        try:
            # Валидация обязательных полей
            if not isinstance(data, dict):
                raise ValidationError("value", "", "validation", "Data must be a dictionary")
            symbol = data.get("symbol", "")
            if not symbol:
                raise ValidationError("value", "", "validation", "Symbol is required")
            # Создание вложенных объектов
            volume_profile = VolumeProfile(**data.get("volume_profile", {}))
            price_structure = PriceStructure(**data.get("price_structure", {}))
            order_book_metrics = OrderBookMetricsData(**data.get("order_book_metrics", {}))
            pattern_metrics = PatternMetricsData(**data.get("pattern_metrics", {}))
            session_metrics = SessionMetricsData(**data.get("session_metrics", {}))
            return cls(
                id=UUID(data.get("id", str(uuid4()))),
                symbol=symbol,
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.now().isoformat())
                ),
                market_phase=MarketPhase(
                    data.get("market_phase", MarketPhase.NO_STRUCTURE.value)
                ),
                opportunity_score=data.get("opportunity_score", 0.0),
                confidence=data.get("confidence", 0.0),
                volume_profile=volume_profile,
                price_structure=price_structure,
                order_book_metrics=order_book_metrics,
                pattern_metrics=pattern_metrics,
                session_metrics=session_metrics,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error creating SymbolProfile from dict: {e}")
            raise ValidationError("value", "", "format", f"Invalid data format: {e}")

    def get_all_metrics(self) -> Dict[str, Any]:
        """Получение всех метрик в структурированном виде."""
        return {
            "volume_metrics": self.volume_profile.to_metrics(),
            "price_metrics": self.price_structure.to_metrics(),
            "order_book_metrics": self.order_book_metrics.to_metrics(),
            "pattern_metrics": self.pattern_metrics.to_metrics(),
            "session_metrics": self.session_metrics.to_metrics(),
        }

    def validate_data_quality(self) -> Dict[str, bool]:
        """Проверка качества данных."""
        return {
            "has_sufficient_volume": self.volume_profile.current_volume > 0,
            "has_price_data": self.price_structure.current_price > 0,
            "has_order_book_data": self.order_book_metrics.bid_ask_spread > 0,
            "has_pattern_data": self.pattern_metrics.pattern_confidence > 0,
            "has_session_data": self.session_metrics.session_alignment > 0,
        }

    def get_risk_assessment(self) -> Dict[str, float]:
        """Оценка рисков для символа."""
        return {
            "volatility_risk": self.price_structure.volatility_compression,
            "liquidity_risk": 1.0 - self.order_book_metrics.liquidity_depth,
            "pattern_risk": 1.0 - self.pattern_metrics.pattern_confidence,
            "session_risk": 1.0 - self.session_metrics.session_alignment,
            "overall_risk": (
                self.price_structure.volatility_compression * 0.3
                + (1.0 - self.order_book_metrics.liquidity_depth) * 0.25
                + (1.0 - self.pattern_metrics.pattern_confidence) * 0.25
                + (1.0 - self.session_metrics.session_alignment) * 0.2
            ),
        }
