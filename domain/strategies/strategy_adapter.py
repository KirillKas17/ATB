"""
Адаптер для объединения domain и infrastructure стратегий.
Этот модуль создаёт мост между доменными интерфейсами и инфраструктурными
реализациями стратегий, обеспечивая единообразный доступ к ним.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Type, Union, Protocol, runtime_checkable, cast
from uuid import uuid4

from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import StrategyStatus
from domain.strategies.strategy_interface import (
    StrategyAnalysisResult,
    StrategyInterface,
)
from domain.type_definitions import ConfidenceLevel
from shared.exception_handler import (
    ExceptionCategory,
    ExceptionSeverity,
    handle_exceptions,
)


@runtime_checkable
class InfrastructureStrategyProtocol(Protocol):
    """Протокол для инфраструктурных стратегий."""
    
    def analyze(self, market_data: MarketData) -> Dict[str, Any]:
        """Анализировать рынок."""
        ...
    
    def generate_signal(self, market_data: MarketData) -> Any:
        """Генерировать сигнал."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики."""
        ...
    
    def update_metrics(self, signal: Signal, result: Dict[str, Any]) -> None:
        """Обновить метрики."""
        ...


class StrategyAdapter:
    """
    Адаптер для объединения стратегий из разных слоёв.
    Обеспечивает единообразный интерфейс для работы со стратегиями
    как из domain, так и из infrastructure слоёв.
    """

    def __init__(self) -> None:
        self._domain_strategies: Dict[str, StrategyInterface] = {}
        self._infrastructure_strategies: Dict[str, InfrastructureStrategyProtocol] = {}
        self._strategy_mapping: Dict[str, str] = {}  # strategy_id -> layer_type

    def register_domain_strategy(self, strategy: StrategyInterface) -> None:
        """Зарегистрировать доменную стратегию."""
        strategy_id = str(strategy.get_strategy_id())
        self._domain_strategies[strategy_id] = strategy
        self._strategy_mapping[strategy_id] = "domain"

    def register_infrastructure_strategy(self, strategy: InfrastructureStrategyProtocol) -> None:
        """Зарегистрировать инфраструктурную стратегию."""
        strategy_id = str(uuid4())
        self._infrastructure_strategies[strategy_id] = strategy
        self._strategy_mapping[strategy_id] = "infrastructure"

    def get_strategy(
        self, strategy_id: str
    ) -> Optional[Union[StrategyInterface, InfrastructureStrategyProtocol]]:
        layer_type = self._strategy_mapping.get(strategy_id)
        if layer_type == "domain":
            domain_strategy = self._domain_strategies.get(strategy_id)
            if domain_strategy is not None:
                return domain_strategy
        elif layer_type == "infrastructure":
            infra_strategy = self._infrastructure_strategies.get(strategy_id)
            if infra_strategy is not None:
                return infra_strategy
        return None

    @handle_exceptions(
        "StrategyAdapter",
        "analyze_market",
        severity=ExceptionSeverity.MEDIUM,
        category=ExceptionCategory.BUSINESS_LOGIC,
    )
    def analyze_market(
        self, strategy_id: str, market_data: MarketData
    ) -> Optional[StrategyAnalysisResult]:
        """Анализировать рынок через стратегию."""
        strategy = self.get_strategy(strategy_id)
        if isinstance(strategy, StrategyInterface):
            return strategy.analyze_market(market_data)
        elif isinstance(strategy, InfrastructureStrategyProtocol):
            # Адаптируем инфраструктурную стратегию к доменному интерфейсу
            analysis_data = strategy.analyze(market_data)
            return self._convert_infrastructure_analysis(analysis_data)
        return None

    @handle_exceptions(
        "StrategyAdapter",
        "generate_signal",
        severity=ExceptionSeverity.MEDIUM,
        category=ExceptionCategory.BUSINESS_LOGIC,
    )
    def generate_signal(
        self, strategy_id: str, market_data: MarketData
    ) -> Optional[Signal]:
        """Генерировать сигнал через стратегию."""
        strategy = self.get_strategy(strategy_id)
        if isinstance(strategy, StrategyInterface):
            return strategy.generate_signal(market_data)
        elif isinstance(strategy, InfrastructureStrategyProtocol):
            # Адаптируем инфраструктурную стратегию к доменному интерфейсу
            infrastructure_signal = strategy.generate_signal(market_data)
            if infrastructure_signal:
                return self._convert_infrastructure_signal(
                    infrastructure_signal, market_data
                )
        return None

    @handle_exceptions(
        "StrategyAdapter",
        "validate_signal",
        severity=ExceptionSeverity.LOW,
        category=ExceptionCategory.VALIDATION,
    )
    def validate_signal(self, strategy_id: str, signal: Signal) -> bool:
        """Валидировать сигнал через стратегию."""
        strategy = self.get_strategy(strategy_id)
        if isinstance(strategy, StrategyInterface):
            # StrategyInterface не имеет метода validate_signal, используем базовую валидацию
            return self._validate_domain_signal(signal)
        elif isinstance(strategy, InfrastructureStrategyProtocol):
            # Базовая валидация для инфраструктурных стратегий
            return self._validate_infrastructure_signal(signal)
        return False

    @handle_exceptions(
        "StrategyAdapter",
        "get_metrics",
        severity=ExceptionSeverity.LOW,
        category=ExceptionCategory.INFRASTRUCTURE,
    )
    def get_strategy_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Получить метрики стратегии."""
        strategy = self.get_strategy(strategy_id)
        if isinstance(strategy, StrategyInterface):
            return strategy.get_execution_stats()
        elif isinstance(strategy, InfrastructureStrategyProtocol):
            return strategy.get_metrics()
        return {}

    @handle_exceptions(
        "StrategyAdapter",
        "update_performance",
        severity=ExceptionSeverity.LOW,
        category=ExceptionCategory.INFRASTRUCTURE,
    )
    def update_strategy_performance(
        self, strategy_id: str, signal: Signal, result: Dict[str, Any]
    ) -> None:
        """Обновить производительность стратегии."""
        strategy = self.get_strategy(strategy_id)
        if isinstance(strategy, StrategyInterface):
            # StrategyInterface не имеет метода update_metrics
            pass
        elif isinstance(strategy, InfrastructureStrategyProtocol):
            strategy.update_metrics(signal, result)

    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Получить все зарегистрированные стратегии."""
        strategies: Dict[str, Dict[str, Any]] = {}
        # Доменные стратегии
        for strategy_id, strategy in self._domain_strategies.items():
            strategies[strategy_id] = {
                "type": "domain",
                "name": getattr(strategy, "_name", "Unknown"),
                "strategy_type": strategy.get_strategy_type().value,
                "status": getattr(strategy, "_status", StrategyStatus.INACTIVE).value,
                "trading_pairs": [str(pair) for pair in strategy.get_trading_pairs()],
                "metrics": strategy.get_execution_stats(),
            }
        # Инфраструктурные стратегии
        for strategy_id, strategy in self._infrastructure_strategies.items():  # type: ignore[assignment]
            # Получаем информацию о стратегии
            strategy_name = getattr(strategy, "__class__", type(strategy)).__name__
            strategy_symbols = getattr(strategy, "symbols", [])
            strategy_metrics = strategy.get_metrics() if hasattr(strategy, "get_metrics") else {}
            
            strategies[strategy_id] = {
                "type": "infrastructure",
                "name": strategy_name,
                "strategy_type": "infrastructure",
                "status": "active",  # Инфраструктурные стратегии всегда активны
                "trading_pairs": strategy_symbols,
                "metrics": strategy_metrics,
            }
        return strategies

    def _convert_infrastructure_analysis(
        self, analysis_data: Dict[str, Any]
    ) -> StrategyAnalysisResult:
        """Конвертировать анализ инфраструктурной стратегии в доменный формат."""
        return StrategyAnalysisResult(
            confidence_score=analysis_data.get("confidence", 0.5),
            trend_direction=analysis_data.get("trend", "neutral"),
            trend_strength=analysis_data.get("trend_strength", 0.0),
            volatility_level=analysis_data.get("volatility", 0.0),
            volume_analysis=analysis_data.get("volume_analysis", {}),
            technical_indicators=analysis_data.get("indicators", {}),
            market_regime=analysis_data.get("market_regime", "unknown"),
            risk_assessment=analysis_data.get("risk_assessment", {}),
            support_resistance=(
                analysis_data.get("support"),
                analysis_data.get("resistance"),
            ),
            momentum_indicators=analysis_data.get("momentum", {}),
            pattern_recognition=analysis_data.get("patterns", []),
            market_sentiment=analysis_data.get("sentiment", "neutral"),
        )

    def _convert_infrastructure_signal(
        self, infrastructure_signal: Any, market_data: MarketData
    ) -> Signal:
        """Конвертировать сигнал инфраструктурной стратегии в доменный формат."""
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency

        # Определяем тип сигнала
        direction = getattr(infrastructure_signal, "direction", "").lower()
        if direction in ["buy", "long"]:
            signal_type = SignalType.BUY
        elif direction in ["sell", "short"]:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Определяем силу сигнала
        confidence = getattr(infrastructure_signal, "confidence", 0.5)
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.STRONG
        elif confidence >= 0.4:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        # Создаем цену
        entry_price = getattr(infrastructure_signal, "entry_price", market_data.close.value)
        price = Money(Decimal(str(entry_price)), Currency.USD)
        
        return Signal(
            strategy_id=uuid4(),
            trading_pair=str(market_data.symbol),
            signal_type=signal_type,
            strength=strength,
            confidence=Decimal(str(confidence)),
            price=price,
            metadata=getattr(infrastructure_signal, "metadata", {}),
        )

    def _validate_domain_signal(self, signal: Signal) -> bool:
        """Валидировать доменный сигнал."""
        if not signal:
            return False
        if not signal.price or signal.price.value <= 0:
            return False
        if signal.confidence and float(signal.confidence) < 0.1:
            return False
        return True

    def _validate_infrastructure_signal(self, signal: Signal) -> bool:
        """Валидировать сигнал инфраструктурной стратегии."""
        return self._validate_domain_signal(signal)
