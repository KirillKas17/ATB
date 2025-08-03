"""
Протоколы для сигналов в доменном слое.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.types.session_types import (
    MarketConditions,
    MarketRegime,
    SessionIntensity,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp


@dataclass
class SessionInfluenceSignal:
    """Сигнал влияния сессии."""

    session_type: SessionType
    influence_strength: float
    market_conditions: MarketConditions
    confidence: float
    timestamp: Timestamp
    metadata: Dict[str, Any]
    predicted_impact: Dict[str, float]

    def __post_init__(self) -> None:
        """Валидация сигнала влияния сессии."""
        if not 0.0 <= self.influence_strength <= 1.0:
            raise ValueError("Influence strength must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class MarketMakerSignal:
    """Сигнал маркет-мейкера."""

    signal_type: str
    strength: float
    direction: str
    confidence: float
    timestamp: Timestamp
    pattern_data: Dict[str, Any]
    market_context: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация сигнала маркет-мейкера."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Signal strength must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@runtime_checkable
class SignalEngineProtocol(Protocol):
    """Протокол движка сигналов."""

    async def generate_session_signals(
        self, session_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> List[SessionInfluenceSignal]:
        """Генерация сигналов сессий."""
        ...

    def analyze_market_conditions(self, data: Dict[str, Any]) -> MarketConditions:
        """Анализ рыночных условий."""
        ...

    def validate_signal(self, signal: SessionInfluenceSignal) -> bool:
        """Валидация сигнала."""
        ...


@runtime_checkable
class MarketMakerSignalProtocol(Protocol):
    """Протокол сигналов маркет-мейкера."""

    async def detect_market_maker_patterns(
        self, orderbook_data: Dict[str, Any], trade_data: Dict[str, Any]
    ) -> List[MarketMakerSignal]:
        """Обнаружение паттернов маркет-мейкера."""
        ...

    def analyze_order_flow(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Анализ потока ордеров."""
        ...

    def predict_market_movement(
        self, signals: List[MarketMakerSignal]
    ) -> Dict[str, Any]:
        """Предсказание движения рынка."""
        ...


class BaseSignalEngine(ABC):
    """Базовый класс движка сигналов."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._signal_history: List[SessionInfluenceSignal] = []
        self._market_conditions_history: List[MarketConditions] = []

    @abstractmethod
    async def generate_session_signals(
        self, session_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> List[SessionInfluenceSignal]:
        """Генерация сигналов сессий."""
        pass

    def analyze_market_conditions(self, data: Dict[str, Any]) -> MarketConditions:
        """Анализ рыночных условий."""
        # Реализация анализа рыночных условий
        return MarketConditions(
            volatility=0.0,
            volume=0.0,
            spread=0.0,
            liquidity=0.0,
            momentum=0.0,
            trend_strength=0.0,
            market_regime=MarketRegime.RANGING,
            session_intensity=SessionIntensity.NORMAL,
        )

    def validate_signal(self, signal: SessionInfluenceSignal) -> bool:
        """Валидация сигнала."""
        try:
            # Проверка обязательных полей
            # if not signal.session_type:
            #     return False
            if not 0.0 <= signal.influence_strength <= 1.0:
                return False
            if not 0.0 <= signal.confidence <= 1.0:
                return False
            return True
        except Exception:
            return False

    def get_signal_history(self, limit: int = 100) -> List[SessionInfluenceSignal]:
        """Получение истории сигналов."""
        return self._signal_history[-limit:]

    def get_market_conditions_history(self, limit: int = 100) -> List[MarketConditions]:
        """Получение истории рыночных условий."""
        return self._market_conditions_history[-limit:]


class BaseMarketMakerSignalEngine(ABC):
    """Базовый класс движка сигналов маркет-мейкера."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._pattern_history: List[MarketMakerSignal] = []
        self._flow_history: List[Dict[str, float]] = []

    @abstractmethod
    async def detect_market_maker_patterns(
        self, orderbook_data: Dict[str, Any], trade_data: Dict[str, Any]
    ) -> List[MarketMakerSignal]:
        """Обнаружение паттернов маркет-мейкера."""
        pass

    def analyze_order_flow(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Анализ потока ордеров."""
        # Реализация анализа потока ордеров
        return {
            "buy_pressure": 0.0,
            "sell_pressure": 0.0,
            "imbalance": 0.0,
            "flow_direction": 0.0,
        }

    def predict_market_movement(
        self, signals: List[MarketMakerSignal]
    ) -> Dict[str, Any]:
        """Предсказание движения рынка."""
        # Реализация предсказания движения
        return {
            "direction": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "timeframe": "short",
        }

    def get_pattern_history(self, limit: int = 100) -> List[MarketMakerSignal]:
        """Получение истории паттернов."""
        return self._pattern_history[-limit:]

    def get_flow_history(self, limit: int = 100) -> List[Dict[str, float]]:
        """Получение истории потока."""
        return self._flow_history[-limit:]
