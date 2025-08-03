"""
Протоколы для стратегий в доменном слое.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.types.strategy_types import (
    MarketRegime,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)


@dataclass
class MirrorMap:
    """Карта зеркальных отражений рынка."""

    market_regime: MarketRegime
    reflection_points: List[Dict[str, float]]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    symmetry_score: float

    def __post_init__(self) -> None:
        """Валидация карты зеркальных отражений."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.symmetry_score <= 1.0:
            raise ValueError("Symmetry score must be between 0.0 and 1.0")


@dataclass
class FollowSignal:
    """Сигнал следования за рынком."""

    direction: StrategyDirection
    strength: float
    confidence: float
    market_regime: MarketRegime
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация сигнала следования."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Signal strength must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class FollowResult:
    """Результат следования за рынком."""

    signal: FollowSignal
    execution_quality: float
    market_impact: float
    slippage: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация результата следования."""
        if not 0.0 <= self.execution_quality <= 1.0:
            raise ValueError("Execution quality must be between 0.0 and 1.0")
        if self.slippage < 0.0:
            raise ValueError("Slippage cannot be negative")


@dataclass
class SymbolSelectionResult:
    """Результат выбора символа."""

    symbol: str
    opportunity_score: float
    risk_score: float
    liquidity_score: float
    volatility_score: float
    market_regime: MarketRegime
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация результата выбора символа."""
        if not 0.0 <= self.opportunity_score <= 1.0:
            raise ValueError("Opportunity score must be between 0.0 and 1.0")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("Risk score must be between 0.0 and 1.0")
        if not 0.0 <= self.liquidity_score <= 1.0:
            raise ValueError("Liquidity score must be between 0.0 and 1.0")
        if not 0.0 <= self.volatility_score <= 1.0:
            raise ValueError("Volatility score must be between 0.0 and 1.0")


@runtime_checkable
class StrategyAdvisorProtocol(Protocol):
    """Протокол советника стратегий."""

    async def build_mirror_map(
        self, market_data: Dict[str, Any], historical_data: Dict[str, Any]
    ) -> MirrorMap:
        """Построение карты зеркальных отражений."""
        ...

    def analyze_market_symmetry(self, data: Dict[str, Any]) -> float:
        """Анализ симметрии рынка."""
        ...

    def validate_mirror_map(self, mirror_map: MirrorMap) -> bool:
        """Валидация карты зеркальных отражений."""
        ...


@runtime_checkable
class MarketFollowerProtocol(Protocol):
    """Протокол следования за рынком."""

    async def generate_follow_signals(
        self, market_data: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> List[FollowSignal]:
        """Генерация сигналов следования."""
        ...

    def analyze_market_momentum(self, data: Dict[str, Any]) -> float:
        """Анализ импульса рынка."""
        ...

    def execute_follow_strategy(self, signal: FollowSignal) -> FollowResult:
        """Исполнение стратегии следования."""
        ...


@runtime_checkable
class SymbolSelectorProtocol(Protocol):
    """Протокол выбора символов."""

    async def select_opportunities(
        self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]
    ) -> List[SymbolSelectionResult]:
        """Выбор торговых возможностей."""
        ...

    def calculate_opportunity_score(self, symbol_data: Dict[str, Any]) -> float:
        """Расчет оценки возможности."""
        ...

    def filter_by_risk(
        self, opportunities: List[SymbolSelectionResult], max_risk: float
    ) -> List[SymbolSelectionResult]:
        """Фильтрация по риску."""
        ...


class BaseStrategyAdvisor(ABC):
    """Базовый класс советника стратегий."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._mirror_map_history: List[MirrorMap] = []
        self._symmetry_history: List[float] = []

    @abstractmethod
    async def build_mirror_map(
        self, market_data: Dict[str, Any], historical_data: Dict[str, Any]
    ) -> MirrorMap:
        """Построение карты зеркальных отражений."""
        pass

    def analyze_market_symmetry(self, data: Dict[str, Any]) -> float:
        """Анализ симметрии рынка."""
        # Реализация анализа симметрии
        return 0.5

    def validate_mirror_map(self, mirror_map: MirrorMap) -> bool:
        """Валидация карты зеркальных отражений."""
        try:
            if not mirror_map.market_regime:
                return False
            if not 0.0 <= mirror_map.confidence <= 1.0:
                return False
            if not 0.0 <= mirror_map.symmetry_score <= 1.0:
                return False
            return True
        except Exception:
            return False

    def get_mirror_map_history(self, limit: int = 100) -> List[MirrorMap]:
        """Получение истории карт зеркальных отражений."""
        return self._mirror_map_history[-limit:]

    def get_symmetry_history(self, limit: int = 100) -> List[float]:
        """Получение истории симметрии."""
        return self._symmetry_history[-limit:]


class BaseMarketFollower(ABC):
    """Базовый класс следования за рынком."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._signal_history: List[FollowSignal] = []
        self._result_history: List[FollowResult] = []

    @abstractmethod
    async def generate_follow_signals(
        self, market_data: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> List[FollowSignal]:
        """Генерация сигналов следования."""
        pass

    def analyze_market_momentum(self, data: Dict[str, Any]) -> float:
        """Анализ импульса рынка."""
        # Реализация анализа импульса
        return 0.0

    def execute_follow_strategy(self, signal: FollowSignal) -> FollowResult:
        """Исполнение стратегии следования."""
        # Реализация исполнения стратегии
        return FollowResult(
            signal=signal,
            execution_quality=0.8,
            market_impact=0.01,
            slippage=0.005,
            timestamp=datetime.now(),
            metadata={},
        )

    def get_signal_history(self, limit: int = 100) -> List[FollowSignal]:
        """Получение истории сигналов."""
        return self._signal_history[-limit:]

    def get_result_history(self, limit: int = 100) -> List[FollowResult]:
        """Получение истории результатов."""
        return self._result_history[-limit:]


class BaseSymbolSelector(ABC):
    """Базовый класс выбора символов."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._selection_history: List[SymbolSelectionResult] = []
        self._opportunity_history: List[float] = []

    @abstractmethod
    async def select_opportunities(
        self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]
    ) -> List[SymbolSelectionResult]:
        """Выбор торговых возможностей."""
        pass

    def calculate_opportunity_score(self, symbol_data: Dict[str, Any]) -> float:
        """Расчет оценки возможности."""
        # Реализация расчета оценки
        return 0.5

    def filter_by_risk(
        self, opportunities: List[SymbolSelectionResult], max_risk: float
    ) -> List[SymbolSelectionResult]:
        """Фильтрация по риску."""
        return [opp for opp in opportunities if opp.risk_score <= max_risk]

    def get_selection_history(self, limit: int = 100) -> List[SymbolSelectionResult]:
        """Получение истории выбора."""
        return self._selection_history[-limit:]

    def get_opportunity_history(self, limit: int = 100) -> List[float]:
        """Получение истории возможностей."""
        return self._opportunity_history[-limit:]
