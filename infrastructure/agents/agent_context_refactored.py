"""
Рефакторенный Agent Context System для интеграции аналитических модулей.
Оптимизированная версия с устранением дублирования и улучшенной производительностью.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

from domain.types.monitoring_types import LogContext
from infrastructure.shared.logging import get_logger, AgentLogger
from infrastructure.agents.market_maker.types import MarketMakerPattern
from infrastructure.agents.market_maker.signals import FollowSignal, FollowResult
from infrastructure.agents.entanglement.types import EntanglementResult
from infrastructure.agents.analytical.types import NoiseAnalysisResult
from infrastructure.agents.analytical.types import MirrorSignal, MirrorMap
from infrastructure.agents.analytical.types import LiquidityGravityResult
from infrastructure.agents.risk.types import RiskAssessmentResult
from infrastructure.agents.analytical.types import PatternDetection
from infrastructure.agents.analytical.types import SessionInfluenceResult, SessionInfluenceSignal
from infrastructure.agents.analytical.types import MarketSessionContext
from infrastructure.agents.analytical.types import SymbolSelectionResult
from domain.types.strategy_types import EnhancedPredictionResult  # type: ignore


class ModifierType(Enum):
    """Типы модификаторов."""

    ORDER_AGGRESSIVENESS = "order_aggressiveness"
    POSITION_SIZE_MULTIPLIER = "position_size_multiplier"
    CONFIDENCE_MULTIPLIER = "confidence_multiplier"
    PRICE_OFFSET_PERCENT = "price_offset_percent"
    EXECUTION_DELAY_MS = "execution_delay_ms"
    RISK_MULTIPLIER = "risk_multiplier"


class PriorityLevel(Enum):
    """Уровни приоритета модификаторов."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    AUXILIARY = "auxiliary"
    ML = "ml"
    AGENTS = "agents"
    TRAINING = "training"


@dataclass
class MarketContext:
    """Контекст рыночных условий."""

    is_clean: bool = True
    external_sync: bool = False
    regime_shift: bool = False
    gravity_bias: float = 0.0
    unreliable_depth: bool = False
    synthetic_noise: bool = False
    leader_asset: Optional[str] = None
    mirror_correlation: float = 0.0
    price_influence_bias: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyModifiers:
    """Модификаторы стратегий."""

    # Базовые модификаторы
    order_aggressiveness: float = 1.0
    position_size_multiplier: float = 1.0
    confidence_multiplier: float = 1.0
    price_offset_percent: float = 0.0
    execution_delay_ms: int = 0
    risk_multiplier: float = 1.0

    # Флаги стратегий
    scalping_enabled: bool = True
    mean_reversion_enabled: bool = True
    momentum_enabled: bool = True

    # Временные параметры
    valid_until: Optional[datetime] = None
    priority: int = 0

    # Специализированные модификаторы
    market_pattern_confidence_multiplier: float = 1.1
    entanglement_confidence_multiplier: float = 0.9
    mirror_confidence_multiplier: float = 1.15
    noise_confidence_multiplier: float = 0.85
    session_influence_confidence_multiplier: float = 0.6
    session_marker_confidence_multiplier: float = 1.1
    live_adaptation_confidence_multiplier: float = 1.2
    evolutionary_transformer_confidence_multiplier: float = 1.25
    whale_analysis_confidence_multiplier: float = 0.8
    risk_analysis_confidence_multiplier: float = 0.9
    portfolio_analysis_confidence_multiplier: float = 1.05
    mm_pattern_confidence_multiplier: float = 1.2

    def apply_modifier(self, modifier_type: ModifierType, value: float) -> None:
        """Применение модификатора."""
        if modifier_type == ModifierType.ORDER_AGGRESSIVENESS:
            self.order_aggressiveness *= value
        elif modifier_type == ModifierType.POSITION_SIZE_MULTIPLIER:
            self.position_size_multiplier *= value
        elif modifier_type == ModifierType.CONFIDENCE_MULTIPLIER:
            self.confidence_multiplier *= value
        elif modifier_type == ModifierType.PRICE_OFFSET_PERCENT:
            self.price_offset_percent *= value
        elif modifier_type == ModifierType.EXECUTION_DELAY_MS:
            self.execution_delay_ms = max(0, int(self.execution_delay_ms * value))
        elif modifier_type == ModifierType.RISK_MULTIPLIER:
            self.risk_multiplier *= value

    def validate(self) -> Dict[str, Any]:
        """Валидация модификаторов."""
        validation_result = {"is_valid": True, "warnings": [], "errors": []}

        # Проверка граничных значений
        if self.order_aggressiveness < 0.1:
            validation_result["warnings"].append("Very low order aggressiveness")  # type: ignore[attr-defined]
        elif self.order_aggressiveness > 5.0:
            validation_result["warnings"].append("Very high order aggressiveness")  # type: ignore[attr-defined]

        if self.position_size_multiplier < 0.1:
            validation_result["warnings"].append("Very low position size multiplier")  # type: ignore[attr-defined]
        elif self.position_size_multiplier > 3.0:
            validation_result["warnings"].append("Very high position size multiplier")  # type: ignore[attr-defined]

        if self.confidence_multiplier < 0.1:
            validation_result["errors"].append("Invalid confidence multiplier")  # type: ignore[attr-defined]
            validation_result["is_valid"] = False
        elif self.confidence_multiplier > 2.0:
            validation_result["warnings"].append("Very high confidence multiplier")  # type: ignore[attr-defined]

        if self.risk_multiplier < 0.1:
            validation_result["errors"].append("Invalid risk multiplier")  # type: ignore[attr-defined]
            validation_result["is_valid"] = False
        elif self.risk_multiplier > 5.0:
            validation_result["warnings"].append("Very high risk multiplier")  # type: ignore[attr-defined]

        if self.execution_delay_ms < 0:
            validation_result["errors"].append("Invalid execution delay")  # type: ignore[attr-defined]
            validation_result["is_valid"] = False
        elif self.execution_delay_ms > 10000:
            validation_result["warnings"].append("Very high execution delay")  # type: ignore[attr-defined]

        # Проверка логической согласованности
        if self.order_aggressiveness > 1.5 and self.risk_multiplier > 1.5:
            validation_result["warnings"].append("High aggressiveness with high risk")  # type: ignore[attr-defined]

        if self.confidence_multiplier < 0.5 and self.position_size_multiplier > 1.2:
            warnings = validation_result.get("warnings")
            if isinstance(warnings, list):
                warnings.append("Low confidence with high position size")  # type: ignore[attr-defined]
            else:
                validation_result["warnings"] = ["Low confidence with high position size"]

        return validation_result


@dataclass
class PatternPredictionContext:
    """Контекст прогнозирования паттернов."""

    prediction_result: Optional[EnhancedPredictionResult] = None
    is_prediction_available: bool = False
    prediction_confidence: float = 0.0
    predicted_direction: Optional[str] = None
    predicted_return_percent: float = 0.0
    predicted_duration_minutes: int = 0
    pattern_confidence_boost: float = 1.0
    pattern_risk_adjustment: float = 1.0
    pattern_position_multiplier: float = 1.0
    pattern_type: Optional[str] = None
    similar_cases_count: int = 0
    success_rate: float = 0.0
    data_quality_score: float = 0.0


@dataclass
class SessionContext:
    """Контекст торговых сессий."""

    session_signals: Dict[str, SessionInfluenceSignal] = field(default_factory=dict)
    aggregated_signal: Optional[SessionInfluenceSignal] = None
    session_statistics: Dict[str, Any] = field(default_factory=dict)
    session_confidence_boost: float = 1.0
    session_aggressiveness_modifier: float = 1.0
    session_position_multiplier: float = 1.0
    primary_session: Optional[str] = None
    session_phase: Optional[str] = None
    session_overlap: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketMakerPatternContext:
    """Контекст паттернов маркет-мейкера."""

    current_pattern: Optional[MarketMakerPattern] = None
    follow_signal: Optional[FollowSignal] = None
    follow_result: Optional[FollowResult] = None
    pattern_statistics: Dict[str, Any] = field(default_factory=dict)
    pattern_confidence_boost: float = 1.0
    pattern_aggressiveness_modifier: float = 1.0
    pattern_position_multiplier: float = 1.0
    pattern_risk_modifier: float = 1.0
    pattern_price_offset_modifier: float = 1.0
    pattern_type: Optional[str] = None
    pattern_confidence: float = 0.0
    expected_direction: Optional[str] = None
    expected_return: float = 0.0
    entry_timing: Optional[str] = None
    historical_accuracy: float = 0.0
    similarity_score: float = 0.0


@runtime_checkable
class ModifierApplicator(Protocol):
    """Протокол для применения модификаторов."""

    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение модификатора к контексту."""
        ...


class BaseModifierApplicator:
    """Базовый класс для применения модификаторов."""

    def __init__(self, name: str, priority: PriorityLevel):
        self.name = name
        self.priority = priority
        self.logger = get_logger(f"modifier.{name}", context=LogContext.AGENT)  # type: ignore

    def apply_modifier(self, context: "AgentContext") -> None:
        """Базовое применение модификатора."""
        raise NotImplementedError

    def is_applicable(self, context: "AgentContext") -> bool:
        """Проверка применимости модификатора."""
        return True


class EntanglementModifierApplicator(BaseModifierApplicator):
    """Применение модификатора запутанности."""

    def __init__(self):
        super().__init__("entanglement", PriorityLevel.CRITICAL)

    def is_applicable(self, context: "AgentContext") -> bool:
        return context.entanglement_result is not None

    def apply_modifier(self, context: "AgentContext") -> None:
        if not self.is_applicable(context):
            return

        result = context.entanglement_result
        if result is None:
            return

        correlation_score = result.correlation_score
        lag_ms = result.lag_ms
        confidence = result.confidence

        # Модификаторы на основе корреляции
        if correlation_score > 0.95:
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 0.6
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.5
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 0.7
            )
            context.strategy_modifiers.apply_modifier(ModifierType.RISK_MULTIPLIER, 1.5)
        elif correlation_score > 0.8:
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 0.8
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.7
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 0.9
            )
            context.strategy_modifiers.apply_modifier(ModifierType.RISK_MULTIPLIER, 1.2)

        # Модификаторы на основе лага
        if lag_ms < 1.0:
            context.strategy_modifiers.apply_modifier(
                ModifierType.EXECUTION_DELAY_MS, 1.5
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.PRICE_OFFSET_PERCENT, 1.3
            )
        elif lag_ms > 5.0:
            context.strategy_modifiers.apply_modifier(
                ModifierType.EXECUTION_DELAY_MS, 0.8
            )

        # Модификаторы на основе уверенности
        if confidence < 0.7:
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 0.9
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.8
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 0.8
            )

        if self.logger:
            self.logger.debug(
                f"Applied entanglement modifiers: correlation={correlation_score:.3f}, "
                f"lag={lag_ms:.3f}, confidence={confidence:.3f}"
            )


class MarketMakerPatternModifierApplicator(BaseModifierApplicator):
    """Применение модификатора паттернов маркет-мейкера."""

    def __init__(self):
        super().__init__("mm_pattern", PriorityLevel.IMPORTANT)

    def is_applicable(self, context: "AgentContext") -> bool:
        return context.mm_pattern_context.current_pattern is not None

    def apply_modifier(self, context: "AgentContext") -> None:
        if not self.is_applicable(context):
            return

        pattern_context = context.mm_pattern_context
        pattern = pattern_context.current_pattern
        if pattern is None:
            return

        # Применяем модификаторы на основе типа паттерна
        pattern_type = pattern.pattern_type
        confidence = pattern_context.pattern_confidence

        if pattern_type == "accumulation":
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 1.2
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 1.1
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 1.15
            )
        elif pattern_type == "distribution":
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 0.8
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.9
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 0.85
            )
        elif pattern_type == "manipulation":
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 1.5
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.7
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 0.9
            )
            context.strategy_modifiers.apply_modifier(ModifierType.RISK_MULTIPLIER, 1.3)

        # Применяем модификаторы на основе уверенности
        if confidence > 0.8:
            context.strategy_modifiers.apply_modifier(
                ModifierType.CONFIDENCE_MULTIPLIER, 1.1
            )
        elif confidence < 0.5:
            context.strategy_modifiers.apply_modifier(
                ModifierType.ORDER_AGGRESSIVENESS, 0.9
            )
            context.strategy_modifiers.apply_modifier(
                ModifierType.POSITION_SIZE_MULTIPLIER, 0.8
            )

        # Обновляем контекст паттерна
        if pattern_context.follow_signal:
            self._update_pattern_context(context, pattern_context.follow_signal)

        if self.logger:
            self.logger.debug(
                f"Applied MM pattern modifiers: type={pattern_type}, "
                f"confidence={confidence:.3f}"
            )

    def _update_pattern_context(
        self, context: "AgentContext", signal: FollowSignal
    ) -> None:
        """Обновление контекста паттерна."""
        context.mm_pattern_context.follow_signal = signal
        context.mm_pattern_context.expected_direction = signal.direction
        context.mm_pattern_context.entry_timing = signal.timing


@dataclass
class AgentContext:
    """Оптимизированный контекст агента для интеграции аналитических модулей."""

    # Базовые параметры
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Контексты
    market_context: MarketContext = field(default_factory=MarketContext)
    strategy_modifiers: StrategyModifiers = field(default_factory=StrategyModifiers)
    pattern_prediction: PatternPredictionContext = field(
        default_factory=PatternPredictionContext
    )
    session_context: SessionContext = field(default_factory=SessionContext)
    mm_pattern_context: MarketMakerPatternContext = field(
        default_factory=MarketMakerPatternContext
    )

    # Результаты аналитических модулей
    entanglement_result: Optional[EntanglementResult] = None
    noise_result: Optional[NoiseAnalysisResult] = None
    mirror_signal: Optional[MirrorSignal] = None
    mirror_map: Optional[MirrorMap] = None
    gravity_result: Optional[LiquidityGravityResult] = None
    risk_assessment: Optional[RiskAssessmentResult] = None
    market_pattern_result: Optional[PatternDetection] = None
    session_influence_result: Optional[SessionInfluenceResult] = None
    session_marker_result: Optional[MarketSessionContext] = None
    live_adaptation_result: Optional[Dict[str, Any]] = None
    decision_reasoning_result: Optional[Dict[str, Any]] = None
    evolutionary_transformer_result: Optional[Dict[str, Any]] = None
    pattern_discovery_result: Optional[Dict[str, Any]] = None
    meta_learning_result: Optional[Dict[str, Any]] = None
    whale_analysis_result: Optional[Dict[str, Any]] = None
    risk_analysis_result: Optional[Dict[str, Any]] = None
    portfolio_analysis_result: Optional[Dict[str, Any]] = None
    meta_controller_result: Optional[Dict[str, Any]] = None
    genetic_optimization_result: Optional[Dict[str, Any]] = None
    doass_result: Optional[SymbolSelectionResult] = None

    # Внутренние данные
    _flags: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _logger: Optional[AgentLogger] = None

    def __post_init__(self):
        """Инициализация после создания объекта."""
        self._logger = get_logger(f"agent_context.{self.symbol}")
        self._modifier_applicators = self._create_modifier_applicators()

    def _create_modifier_applicators(self) -> List[ModifierApplicator]:
        """Создание списка применятелей модификаторов."""
        return [
            EntanglementModifierApplicator(),
            MarketMakerPatternModifierApplicator(),
            # Добавьте другие применятели здесь
        ]

    def set(self, key: str, value: Any) -> None:
        """Установка флага в контексте."""
        self._flags[key] = value
        if self._logger:
            self._logger.debug(f"Set {key} = {value}")

    def get(self, key: str, default: Any = None) -> Any:
        """Получение флага из контекста."""
        return self._flags.get(key, default)

    def has(self, key: str) -> bool:
        """Проверка наличия флага."""
        return key in self._flags

    def remove(self, key: str) -> None:
        """Удаление флага."""
        if key in self._flags:
            del self._flags[key]

    def clear(self) -> None:
        """Очистка всех флагов."""
        self._flags.clear()
        self._metadata.clear()

    def is_market_clean(self) -> bool:
        """Проверка чистоты рынка."""
        return (
            self.market_context.is_clean
            and not self.market_context.external_sync
            and not self.market_context.unreliable_depth
            and not self.market_context.synthetic_noise
        )

    def get_modifier(self, modifier_type: str) -> float:
        """Получение модификатора стратегии."""
        modifier_map = {
            "order_aggressiveness": self.strategy_modifiers.order_aggressiveness,
            "position_size_multiplier": self.strategy_modifiers.position_size_multiplier,
            "confidence_multiplier": self.strategy_modifiers.confidence_multiplier,
            "price_offset_percent": self.strategy_modifiers.price_offset_percent,
            "execution_delay_ms": self.strategy_modifiers.execution_delay_ms,
            "risk_multiplier": self.strategy_modifiers.risk_multiplier,
        }
        return modifier_map.get(modifier_type, 1.0)

    def apply_all_modifiers(self) -> Dict[str, Any]:
        """Применение всех модификаторов."""
        start_time = time.time()
        modifiers_applied = 0
        priority_levels_processed = 0

        # Проверяем кэш
        cache_key = self._generate_modifiers_cache_key()
        cached_result = self.get("modifiers_cache_key")
        if cached_result == cache_key:
            self.set("modifiers_cache_hits", self.get("modifiers_cache_hits", 0) + 1)
            return {
                "execution_time_ms": 0,
                "modifiers_applied": 0,
                "cache_hit": True,
                "priority_levels_processed": 0,
            }

        # Применяем модификаторы
        processed_priorities = set()

        for applicator in self._modifier_applicators:
            try:
                if hasattr(applicator, 'is_applicable') and applicator.is_applicable(self):
                    applicator.apply_modifier(self)
                    modifiers_applied += 1

                    if hasattr(applicator, 'priority') and applicator.priority not in processed_priorities:
                        processed_priorities.add(applicator.priority)
                        priority_levels_processed += 1

            except Exception as e:
                if self._logger:
                    self._logger.error(f"Error applying modifier {applicator.name}: {e}")
                continue

        # Сохраняем кэш
        self.set("modifiers_cache_key", cache_key)
        self.set("last_modifiers_application", time.time())

        # Вычисляем время выполнения
        execution_time_ms = (time.time() - start_time) * 1000

        if self._logger:
            self._logger.debug(
                f"Applied {modifiers_applied} modifiers in {execution_time_ms:.2f}ms, "
                f"priority levels: {priority_levels_processed}"
            )

        return {
            "execution_time_ms": execution_time_ms,
            "modifiers_applied": modifiers_applied,
            "cache_hit": False,
            "priority_levels_processed": priority_levels_processed,
        }

    def _generate_modifiers_cache_key(self) -> str:
        """Генерация ключа кэша для модификаторов."""
        state_string = ""

        # Добавляем хеши результатов аналитических модулей
        result_attributes = [
            "entanglement_result",
            "noise_result",
            "mirror_signal",
            "gravity_result",
            "risk_assessment",
            "market_pattern_result",
            "session_influence_result",
            "session_marker_result",
            "live_adaptation_result",
            "decision_reasoning_result",
            "evolutionary_transformer_result",
            "pattern_discovery_result",
            "meta_learning_result",
            "whale_analysis_result",
            "risk_analysis_result",
            "portfolio_analysis_result",
            "meta_controller_result",
            "genetic_optimization_result",
            "doass_result",
        ]

        for attr in result_attributes:
            result = getattr(self, attr, None)
            if result:
                state_string += f"{attr}:{hash(str(result))}"

        # Добавляем хеш модификаторов стратегий
        state_string += f"modifiers:{hash(str(self.strategy_modifiers))}"

        return hashlib.md5(state_string.encode()).hexdigest()

    def validate_modifiers(self) -> Dict[str, Any]:
        """Валидация модификаторов стратегий."""
        return self.strategy_modifiers.validate()

    def get_modifiers_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности модификаторов."""
        return {
            "last_application_time": self.get("last_modifiers_application", 0),
            "cache_hits": self.get("modifiers_cache_hits", 0),
            "total_applications": self.get("total_modifiers_applications", 0),
            "average_execution_time_ms": self.get(
                "average_modifiers_execution_time", 0.0
            ),
            "priority_levels_processed": len(
                self._metadata.get("processed_priorities", [])
            ),
            "active_modifiers_count": self._count_active_modifiers(),
        }

    def _count_active_modifiers(self) -> int:
        """Подсчет активных модификаторов."""
        active_count = 0

        for applicator in self._modifier_applicators:
            if applicator.is_applicable(self):
                active_count += 1

        return active_count

    def reset_modifiers_cache(self) -> None:
        """Сброс кэша модификаторов."""
        self.remove("modifiers_cache_key")
        self.remove("last_modifiers_application")
        self._metadata.pop("processed_priorities", None)
        if self._logger:
            self._logger.debug("Reset modifiers cache")


class AgentContextManager:
    """Менеджер контекстов агентов."""

    def __init__(self):
        self._contexts: Dict[str, AgentContext] = {}
        self._logger = get_logger("agent_context_manager", context=LogContext.AGENT)

    def get_context(self, symbol: str) -> AgentContext:
        """Получение контекста для символа."""
        if symbol not in self._contexts:
            self._contexts[symbol] = AgentContext(symbol=symbol)
        return self._contexts[symbol]

    def update_context(self, symbol: str, context: AgentContext) -> None:
        """Обновление контекста."""
        self._contexts[symbol] = context

    def clear_context(self, symbol: str) -> None:
        """Очистка контекста."""
        if symbol in self._contexts:
            del self._contexts[symbol]

    def get_context_statistics(self) -> Dict[str, Any]:
        """Получение статистики контекстов."""
        return {
            "total_contexts": len(self._contexts),
            "symbols": list(self._contexts.keys()),
        }
