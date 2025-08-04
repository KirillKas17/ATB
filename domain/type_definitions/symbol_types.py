# -*- coding: utf-8 -*-
"""Типизированные типы для модуля symbols."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)
from uuid import UUID
import pandas as pd

from typing_extensions import NewType, NotRequired

# Новые типы для строгой типизации
SymbolId = NewType("SymbolId", UUID)
OpportunityScoreValue = NewType("OpportunityScoreValue", float)
ConfidenceValue = NewType("ConfidenceValue", float)
VolumeValue = NewType("VolumeValue", float)
PriceValue = NewType("PriceValue", float)
ATRValue = NewType("ATRValue", float)
VWAPValue = NewType("VWAPValue", float)
SpreadValue = NewType("SpreadValue", float)
EntropyValue = NewType("EntropyValue", float)
MomentumValue = NewType("MomentumValue", float)
VolatilityValue = NewType("VolatilityValue", float)
TrendStrengthValue = NewType("TrendStrengthValue", float)
PatternConfidenceValue = NewType("PatternConfidenceValue", float)
SessionAlignmentValue = NewType("SessionAlignmentValue", float)
# Типы для рыночных данных
MarketDataFrame = NewType("MarketDataFrame", Dict[str, Any])  # Используем Dict вместо Any
OrderBookData = NewType("OrderBookData", Dict[str, Any])
PatternMemoryData = NewType("PatternMemoryData", Dict[str, Any])
SessionData = NewType("SessionData", Dict[str, Any])


# Перечисления
class MarketPhase(str, Enum):
    """Фазы рынка для торгового символа."""

    ACCUMULATION = "accumulation"
    BREAKOUT_SETUP = "breakout_setup"
    BREAKOUT_ACTIVE = "breakout_active"
    EXHAUSTION = "exhaustion"
    REVERSION_POTENTIAL = "reversion_potential"
    NO_STRUCTURE = "no_structure"
    UNKNOWN = "unknown"


class VolumeTrend(str, Enum):
    """Тренды объема."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class PriceStructure(str, Enum):
    """Структуры цены."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    REVERSION = "reversion"


class OrderBookSymmetry(str, Enum):
    """Симметрия стакана заявок."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    BALANCED = "balanced"
    ASYMMETRIC = "asymmetric"


# TypedDict для конфигураций
class PhaseDetectionConfig(TypedDict, total=False):
    """Конфигурация для определения фаз рынка."""

    # Периоды для анализа
    atr_period: int
    vwap_period: int
    volume_period: int
    entropy_period: int
    # Пороги для классификации
    atr_threshold: float
    vwap_deviation_threshold: float
    volume_slope_threshold: float
    entropy_threshold: float
    # Пороги для фаз
    accumulation_volume_threshold: float
    breakout_volume_threshold: float
    exhaustion_volume_threshold: float
    reversal_momentum_threshold: float


class OpportunityScoreConfig(TypedDict, total=False):
    """Конфигурация для расчета opportunity score."""

    # Веса компонентов
    alpha1_liquidity_score: float
    alpha2_volume_stability: float
    alpha3_structural_predictability: float
    alpha4_orderbook_symmetry: float
    alpha5_session_alignment: float
    alpha6_historical_pattern_match: float
    # Пороги для нормализации
    min_volume_threshold: float
    max_spread_threshold: float
    min_atr_threshold: float
    max_entropy_threshold: float
    # Пороги для паттернов
    min_pattern_confidence: float
    min_historical_match: float
    # Множители для фаз рынка
    phase_multipliers: Dict[MarketPhase, float]


# TypedDict для метрик
class VolumeProfileMetrics(TypedDict, total=False):
    """Метрики профиля объема."""

    current_volume: VolumeValue
    avg_volume_1m: VolumeValue
    avg_volume_5m: VolumeValue
    avg_volume_15m: VolumeValue
    volume_trend: VolumeValue
    volume_stability: ConfidenceValue
    volume_anomaly_ratio: float


class PriceStructureMetrics(TypedDict, total=False):
    """Метрики структуры цены."""

    current_price: PriceValue
    atr: ATRValue
    atr_percent: float
    vwap: VWAPValue
    vwap_deviation: float
    support_level: NotRequired[Optional[PriceValue]]
    resistance_level: NotRequired[Optional[PriceValue]]
    pivot_point: NotRequired[Optional[PriceValue]]
    price_entropy: EntropyValue
    volatility_compression: VolatilityValue


class OrderBookMetrics(TypedDict, total=False):
    """Метрики стакана заявок."""

    bid_ask_spread: SpreadValue
    spread_percent: float
    bid_volume: VolumeValue
    ask_volume: VolumeValue
    volume_imbalance: float
    order_book_symmetry: float
    liquidity_depth: float
    absorption_ratio: float


class PatternMetrics(TypedDict, total=False):
    """Метрики паттернов и сигналов."""

    mirror_neuron_score: float
    gravity_anomaly_score: float
    reversal_setup_score: float
    pattern_confidence: PatternConfidenceValue
    historical_pattern_match: float
    pattern_complexity: float


class SessionMetrics(TypedDict, total=False):
    """Метрики торговой сессии."""

    session_alignment: SessionAlignmentValue
    session_activity: float
    session_volatility: VolatilityValue
    session_momentum: MomentumValue
    session_influence_score: float


# TypedDict для результатов анализа
class MarketPhaseResult(TypedDict):
    """Результат классификации фазы рынка."""

    phase: MarketPhase
    confidence: ConfidenceValue
    indicators: Dict[str, float]
    metadata: Dict[str, Any]


class OpportunityScoreResult(TypedDict):
    """Результат расчета opportunity score."""

    symbol: str
    total_score: OpportunityScoreValue
    confidence: ConfidenceValue
    market_phase: MarketPhase
    phase_confidence: ConfidenceValue
    components: Dict[str, float]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


# Протоколы для интерфейсов
@runtime_checkable
class MarketPhaseClassifierProtocol(Protocol):
    """Протокол для классификатора фаз рынка."""

    def classify_market_phase(
        self, market_data: MarketDataFrame, order_book: Optional[OrderBookData] = None
    ) -> MarketPhaseResult:
        """Определение фазы рынка."""
        ...

    def get_phase_description(self, phase: MarketPhase) -> str:
        """Получение описания фазы."""
        ...


@runtime_checkable
class OpportunityScoreCalculatorProtocol(Protocol):
    """Протокол для калькулятора opportunity score."""

    def calculate_opportunity_score(
        self,
        symbol: str,
        market_data: MarketDataFrame,
        order_book: OrderBookData,
        pattern_memory: Optional[PatternMemoryData] = None,
        session_data: Optional[SessionData] = None,
    ) -> OpportunityScoreResult:
        """Расчет opportunity score."""
        ...

    def is_opportunity(
        self, score: OpportunityScoreValue, min_score: float = 0.78
    ) -> bool:
        """Проверка, является ли символ торговой возможностью."""
        ...


@runtime_checkable
class SymbolProfileProtocol(Protocol):
    """Протокол для профиля символа."""

    def is_opportunity(self, min_score: float = 0.78) -> bool:
        """Проверка торговой возможности."""
        ...

    def get_opportunity_summary(self) -> Dict[str, Any]:
        """Получение краткого описания."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolProfileProtocol":
        """Создание из словаря."""
        ...


# Dataclass для конфигураций
@dataclass(frozen=True)
class PhaseDetectionConfigData:
    """Неизменяемая конфигурация для определения фаз рынка."""

    # Периоды для анализа
    atr_period: int = 14
    vwap_period: int = 20
    volume_period: int = 20
    entropy_period: int = 50
    # Пороги для классификации
    atr_threshold: float = 0.02
    vwap_deviation_threshold: float = 0.01
    volume_slope_threshold: float = 0.1
    entropy_threshold: float = 0.7
    # Пороги для фаз
    accumulation_volume_threshold: float = 0.5
    breakout_volume_threshold: float = 1.5
    exhaustion_volume_threshold: float = 2.0
    reversal_momentum_threshold: float = 0.3


@dataclass(frozen=True)
class OpportunityScoreConfigData:
    """Неизменяемая конфигурация для расчета opportunity score."""

    # Веса компонентов
    alpha1_liquidity_score: float = 0.20
    alpha2_volume_stability: float = 0.15
    alpha3_structural_predictability: float = 0.20
    alpha4_orderbook_symmetry: float = 0.15
    alpha5_session_alignment: float = 0.15
    alpha6_historical_pattern_match: float = 0.15
    # Пороги для нормализации
    min_volume_threshold: float = 1000000.0
    max_spread_threshold: float = 0.005
    min_atr_threshold: float = 0.001
    max_entropy_threshold: float = 0.8
    # Пороги для паттернов
    min_pattern_confidence: float = 0.6
    min_historical_match: float = 0.7
    # Множители для фаз рынка
    phase_multipliers: Optional[Dict[MarketPhase, float]] = None

    def __post_init__(self) -> None:
        """Проверка и установка значений по умолчанию."""
        if self.phase_multipliers is None:
            default_multipliers = {
                MarketPhase.ACCUMULATION: 0.8,
                MarketPhase.BREAKOUT_SETUP: 1.2,
                MarketPhase.BREAKOUT_ACTIVE: 1.0,
                MarketPhase.EXHAUSTION: 0.6,
                MarketPhase.REVERSION_POTENTIAL: 1.1,
                MarketPhase.NO_STRUCTURE: 0.3,
            }
            object.__setattr__(self, "phase_multipliers", default_multipliers)


# Дополнительные типы для валидации и обработки ошибок
class ValidationError(Exception):
    """Ошибка валидации данных."""

    pass


class ConfigurationError(Exception):
    """Ошибка конфигурации."""

    pass


class DataInsufficientError(Exception):
    """Недостаточно данных для анализа."""

    pass


class MarketDataError(Exception):
    """Ошибка рыночных данных."""

    pass


class OrderBookError(Exception):
    """Ошибка стакана заявок."""

    pass


# Типы для валидации данных
class MarketDataValidator(Protocol):
    """Протокол для валидации рыночных данных."""

    def validate_ohlcv_data(self, data: MarketDataFrame) -> bool:
        """Валидация OHLCV данных."""
        return True

    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """Валидация стакана заявок."""
        return True

    def validate_pattern_memory(self, pattern_memory: PatternMemoryData) -> bool:
        """Валидация данных паттернов."""
        return True


# Типы для кэширования
class SymbolProfileCache(Protocol):
    """Протокол для кэширования профилей символов."""

    def get_profile(self, symbol: str) -> Optional[SymbolProfileProtocol]:
        """Получение профиля из кэша."""
        ...

    def set_profile(self, symbol: str, profile: SymbolProfileProtocol) -> None:
        """Сохранение профиля в кэш."""
        ...

    def invalidate_profile(self, symbol: str) -> None:
        """Инвалидация профиля."""
        ...

    def clear_cache(self) -> None:
        """Очистка кэша."""
        ...


# Типы для метрик производительности
class SymbolAnalysisMetrics(TypedDict, total=False):
    """Метрики производительности анализа символов."""

    analysis_time_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    error_rate: float
    throughput_symbols_per_second: float
    accuracy_score: float


# Типы для событий
class SymbolAnalysisEvent(TypedDict):
    """Событие анализа символа."""

    symbol: str
    timestamp: datetime
    event_type: str
    market_phase: MarketPhase
    opportunity_score: OpportunityScoreValue
    confidence: ConfidenceValue
    metadata: Dict[str, Any]
