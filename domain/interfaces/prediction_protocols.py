"""
Протоколы для предсказаний в доменном слое.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.types.prediction_types import (
    AnalysisMetadata,
    CandlestickMetadata,
    CandlestickPatternType,
    ControversyDetails,
    DivergenceAnalysis,
    DivergenceType,
    LiquidityClusterData,
    MarketPhase,
    MeanReversionData,
    MomentumMetrics,
    OrderBookSnapshot,
    PivotPointData,
    PredictionConfig,
    ReversalDirection,
    RiskLevel,
    RiskMetrics,
    SignalStrength,
    VolumeProfileData,
)


class PredictionResultProtocol(Protocol):
    """Протокол результата предсказания."""

    confidence: float
    direction: ReversalDirection
    strength: SignalStrength
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class EnhancedPredictionResult:
    """Улучшенный результат предсказания."""

    confidence: float
    direction: ReversalDirection
    strength: SignalStrength
    timestamp: datetime
    metadata: Dict[str, Any]
    risk_level: RiskLevel
    market_phase: MarketPhase
    pivot_points: List[PivotPointData]
    candlestick_patterns: List[CandlestickPatternType]
    divergences: List[DivergenceAnalysis]
    momentum_metrics: MomentumMetrics
    volume_profile: VolumeProfileData
    liquidity_clusters: List[LiquidityClusterData]
    mean_reversion_data: MeanReversionData
    controversy_details: Optional[ControversyDetails] = None

    def __post_init__(self) -> None:
        """Валидация результата предсказания."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.timestamp > datetime.now():
            raise ValueError("Timestamp cannot be in the future")


@runtime_checkable
class PatternPredictorProtocol(Protocol):
    """Протокол предсказателя паттернов."""

    async def predict_patterns(
        self, market_data: Dict[str, Any], config: Optional[PredictionConfig] = None
    ) -> EnhancedPredictionResult:
        """Предсказание паттернов."""
        ...

    def validate_input(self, market_data: Dict[str, Any]) -> bool:
        """Валидация входных данных."""
        ...

    def get_prediction_confidence(self, result: EnhancedPredictionResult) -> float:
        """Получение уверенности в предсказании."""
        ...


@runtime_checkable
class ReversalPredictorProtocol(Protocol):
    """Протокол предсказателя разворотов."""

    async def predict_reversal(
        self,
        market_data: Dict[str, Any],
        orderbook: OrderBookSnapshot,
        config: Optional[PredictionConfig] = None,
    ) -> EnhancedPredictionResult:
        """Предсказание разворота."""
        ...

    def analyze_momentum(self, metrics: MomentumMetrics) -> SignalStrength:
        """Анализ импульса."""
        ...

    def detect_divergences(self, data: Dict[str, Any]) -> List[DivergenceAnalysis]:
        """Обнаружение дивергенций."""
        ...


@runtime_checkable
class MarketPhasePredictorProtocol(Protocol):
    """Протокол предсказателя фазы рынка."""

    async def predict_market_phase(
        self, market_data: Dict[str, Any], config: Optional[PredictionConfig] = None
    ) -> MarketPhase:
        """Предсказание фазы рынка."""
        ...

    def analyze_volatility(self, data: Dict[str, Any]) -> float:
        """Анализ волатильности."""
        ...

    def detect_trend_strength(self, data: Dict[str, Any]) -> float:
        """Обнаружение силы тренда."""
        ...


class BasePatternPredictor(ABC):
    """Базовый класс предсказателя паттернов."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self._prediction_history: List[EnhancedPredictionResult] = []
        self._accuracy_metrics: Dict[str, float] = {}

    @abstractmethod
    async def predict_patterns(
        self, market_data: Dict[str, Any], config: Optional[PredictionConfig] = None
    ) -> EnhancedPredictionResult:
        """Предсказание паттернов."""
        pass

    def validate_input(self, market_data: Dict[str, Any]) -> bool:
        """Валидация входных данных."""
        required_keys = ["price", "volume", "timestamp"]
        return all(key in market_data for key in required_keys)

    def get_prediction_confidence(self, result: EnhancedPredictionResult) -> float:
        """Получение уверенности в предсказании."""
        return result.confidence

    def update_accuracy_metrics(
        self, prediction: EnhancedPredictionResult, actual: Any
    ) -> None:
        """Обновление метрик точности."""
        # Реализация обновления метрик точности
        pass

    def get_historical_predictions(
        self, limit: int = 100
    ) -> List[EnhancedPredictionResult]:
        """Получение исторических предсказаний."""
        return self._prediction_history[-limit:]

    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Получение метрик точности."""
        return self._accuracy_metrics.copy()


class BaseReversalPredictor(ABC):
    """Базовый класс предсказателя разворотов."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self._reversal_history: List[EnhancedPredictionResult] = []

    @abstractmethod
    async def predict_reversal(
        self,
        market_data: Dict[str, Any],
        orderbook: OrderBookSnapshot,
        config: Optional[PredictionConfig] = None,
    ) -> EnhancedPredictionResult:
        """Предсказание разворота."""
        pass

    def analyze_momentum(self, metrics: MomentumMetrics) -> SignalStrength:
        """Анализ импульса."""
        # Реализация анализа импульса
        return SignalStrength.MODERATE

    def detect_divergences(self, data: Dict[str, Any]) -> List[DivergenceAnalysis]:
        """Обнаружение дивергенций."""
        # Реализация обнаружения дивергенций
        return []

    def get_reversal_history(self, limit: int = 100) -> List[EnhancedPredictionResult]:
        """Получение истории разворотов."""
        return self._reversal_history[-limit:]


class BaseMarketPhasePredictor(ABC):
    """Базовый класс предсказателя фазы рынка."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self._phase_history: List[MarketPhase] = []

    @abstractmethod
    async def predict_market_phase(
        self, market_data: Dict[str, Any], config: Optional[PredictionConfig] = None
    ) -> MarketPhase:
        """Предсказание фазы рынка."""
        pass

    def analyze_volatility(self, data: Dict[str, Any]) -> float:
        """Анализ волатильности."""
        # Реализация анализа волатильности
        return 0.0

    def detect_trend_strength(self, data: Dict[str, Any]) -> float:
        """Обнаружение силы тренда."""
        # Реализация обнаружения силы тренда
        return 0.0

    def get_phase_history(self, limit: int = 100) -> List[MarketPhase]:
        """Получение истории фаз."""
        return self._phase_history[-limit:]
