"""
Конфигурация для анализа паттернов маркет-мейкера.
"""

from dataclasses import dataclass, field
from typing import Dict

from domain.types.market_maker_types import (
    Accuracy,
    BookPressure,
    Confidence,
    OrderImbalance,
    PriceVolatility,
    SimilarityScore,
    VolumeDelta,
)


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Конфигурация для анализа паттернов маркет-мейкера.
    Attributes:
        min_confidence: Минимальная уверенность для анализа
        similarity_threshold: Порог схожести паттернов
        accuracy_threshold: Порог точности для надежных паттернов
        volume_threshold: Порог объема для значимых паттернов
        spread_threshold: Порог спреда для анализа
        imbalance_threshold: Порог дисбаланса ордеров
        pressure_threshold: Порог давления на стакан
        time_window_seconds: Временное окно анализа в секундах
        min_trades_count: Минимальное количество сделок
        max_history_size: Максимальный размер истории
        feature_weights: Веса признаков для расчета схожести
        market_phase_weights: Веса для разных рыночных фаз
        volatility_regime_weights: Веса для разных режимов волатильности
        liquidity_regime_weights: Веса для разных режимов ликвидности
        pattern_type_weights: Веса для разных типов паттернов
        time_decay_factor: Фактор временного затухания
        confidence_boost_factors: Факторы повышения уверенности
        signal_strength_factors: Факторы силы сигнала
    """

    # Основные пороги
    min_confidence: Confidence = Confidence(0.6)
    similarity_threshold: SimilarityScore = SimilarityScore(0.8)
    accuracy_threshold: Accuracy = Accuracy(0.7)
    # Рыночные пороги
    volume_threshold: float = 1000.0
    spread_threshold: float = 0.001
    imbalance_threshold: OrderImbalance = OrderImbalance(0.3)
    pressure_threshold: BookPressure = BookPressure(0.2)
    # Временные параметры
    time_window_seconds: int = 300  # 5 минут
    min_trades_count: int = 10
    max_history_size: int = 1000
    # Веса признаков для расчета схожести
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "book_pressure": 0.25,
            "volume_delta": 0.20,
            "price_reaction": 0.15,
            "spread_change": 0.10,
            "order_imbalance": 0.15,
            "liquidity_depth": 0.10,
            "volume_concentration": 0.05,
        }
    )
    # Веса для разных рыночных фаз
    market_phase_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accumulation": 1.2,
            "markup": 1.0,
            "distribution": 0.8,
            "markdown": 0.9,
            "transition": 1.1,
        }
    )
    # Веса для разных режимов волатильности
    volatility_regime_weights: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.8, "medium": 1.0, "high": 1.3, "extreme": 1.5}
    )
    # Веса для разных режимов ликвидности
    liquidity_regime_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "high": 1.1,
            "medium": 1.0,
            "low": 0.9,
            "very_low": 0.7,
        }
    )
    # Веса для разных типов паттернов
    pattern_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accumulation": 1.0,
            "absorption": 0.9,
            "spoofing": 1.2,
            "exit": 1.1,
            "pressure_zone": 1.0,
            "liquidity_grab": 1.3,
            "stop_hunt": 1.2,
            "fake_breakout": 1.1,
            "wash_trading": 0.8,
            "pump_and_dump": 1.4,
        }
    )
    # Факторы временного затухания
    time_decay_factor: float = 0.95  # Затухание за день
    # Факторы повышения уверенности
    confidence_boost_factors: Dict[str, float] = field(
        default_factory=lambda: {
            "high_accuracy": 0.3,
            "high_volume": 0.2,
            "pattern_frequency": 0.15,
            "market_regime_match": 0.2,
            "time_recency": 0.15,
        }
    )
    # Факторы силы сигнала
    signal_strength_factors: Dict[str, float] = field(
        default_factory=lambda: {
            "pattern_accuracy": 0.4,
            "volume_significance": 0.2,
            "price_impact": 0.2,
            "market_context": 0.1,
            "historical_success": 0.1,
        }
    )

    def __post_init__(self) -> None:
        """Валидация конфигурации после инициализации."""
        if not (0.0 <= float(self.min_confidence) <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not (0.0 <= float(self.similarity_threshold) <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= float(self.accuracy_threshold) <= 1.0):
            raise ValueError("accuracy_threshold must be between 0.0 and 1.0")
        if self.volume_threshold <= 0:
            raise ValueError("volume_threshold must be positive")
        if self.spread_threshold <= 0:
            raise ValueError("spread_threshold must be positive")
        if self.time_window_seconds <= 0:
            raise ValueError("time_window_seconds must be positive")
        if self.min_trades_count <= 0:
            raise ValueError("min_trades_count must be positive")
        if self.max_history_size <= 0:
            raise ValueError("max_history_size must be positive")
        if not (0.0 <= self.time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be between 0.0 and 1.0")
        # Проверка весов
        total_feature_weight = sum(self.feature_weights.values())
        if abs(total_feature_weight - 1.0) > 0.01:
            raise ValueError(
                f"Feature weights must sum to 1.0, got {total_feature_weight}"
            )
        total_confidence_boost = sum(self.confidence_boost_factors.values())
        if total_confidence_boost > 1.0:
            raise ValueError(
                f"Confidence boost factors sum exceeds 1.0: {total_confidence_boost}"
            )
        total_signal_strength = sum(self.signal_strength_factors.values())
        if abs(total_signal_strength - 1.0) > 0.01:
            raise ValueError(
                f"Signal strength factors must sum to 1.0, got {total_signal_strength}"
            )

    def get_feature_weight(self, feature_name: str) -> float:
        """Получение веса признака."""
        return self.feature_weights.get(feature_name, 0.0)

    def get_market_phase_weight(self, phase: str) -> float:
        """Получение веса рыночной фазы."""
        return self.market_phase_weights.get(phase, 1.0)

    def get_volatility_regime_weight(self, regime: str) -> float:
        """Получение веса режима волатильности."""
        return self.volatility_regime_weights.get(regime, 1.0)

    def get_liquidity_regime_weight(self, regime: str) -> float:
        """Получение веса режима ликвидности."""
        return self.liquidity_regime_weights.get(regime, 1.0)

    def get_pattern_type_weight(self, pattern_type: str) -> float:
        """Получение веса типа паттерна."""
        return self.pattern_type_weights.get(pattern_type, 1.0)

    def get_confidence_boost_factor(self, factor_name: str) -> float:
        """Получение фактора повышения уверенности."""
        return self.confidence_boost_factors.get(factor_name, 0.0)

    def get_signal_strength_factor(self, factor_name: str) -> float:
        """Получение фактора силы сигнала."""
        return self.signal_strength_factors.get(factor_name, 0.0)

    def calculate_time_decay(self, days_old: int) -> float:
        """Расчет временного затухания."""
        return self.time_decay_factor**days_old
