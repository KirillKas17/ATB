# -*- coding: utf-8 -*-
"""Agent Context System для интеграции аналитических модулей."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Iterable

from loguru import logger

from application.market.mm_follow_controller import FollowResult as MMFollowResult, FollowSignal as MMFollowSignal
from domain.intelligence.entanglement_detector import EntanglementResult
from domain.intelligence.market_pattern_recognizer import PatternDetection
from domain.intelligence.mirror_detector import MirrorSignal
from domain.intelligence.noise_analyzer import NoiseAnalysisResult
from domain.interfaces.prediction_protocols import EnhancedPredictionResult
from domain.interfaces.risk_protocols import RiskAssessmentResult
from domain.interfaces.signal_protocols import SessionInfluenceSignal
from domain.interfaces.strategy_protocols import (
    MirrorMap,
    SymbolSelectionResult,
)
from domain.market.liquidity_gravity import LiquidityGravityResult
from domain.market_maker.mm_pattern import MarketMakerPattern
from domain.services.pattern_discovery import Pattern
from domain.sessions.session_influence_analyzer import SessionInfluenceResult
from domain.sessions.session_marker import MarketSessionContext
from domain.types.ml_types import AggregatedSignal as TradeDecision
from domain.value_objects.percentage import Percentage
from domain.value_objects.signal import Signal
from domain.types.session_types import SessionType
from domain.types.base_types import RiskLevel, SignalDirection, TradingPair
from domain.entities.signal import Signal


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

    # Модификаторы агрессивности
    order_aggressiveness: float = 1.0
    position_size_multiplier: float = 1.0
    confidence_multiplier: float = 1.0

    # Модификаторы исполнения
    price_offset_percent: float = 0.0
    execution_delay_ms: int = 0
    risk_multiplier: float = 1.0

    # Модификаторы стратегий
    scalping_enabled: bool = True
    mean_reversion_enabled: bool = True
    momentum_enabled: bool = True

    # Временные модификаторы
    valid_until: Optional[datetime] = None
    priority: int = 0

    # Модификаторы для анализа паттернов рынка
    market_pattern_confidence_multiplier: float = 1.1
    market_pattern_strength_multiplier: float = 1.05
    market_pattern_execution_delay_ms: int = 0

    # Модификаторы для детекции запутанности
    entanglement_confidence_multiplier: float = 0.9
    entanglement_strength_multiplier: float = 0.8
    entanglement_execution_delay_ms: int = 500

    # Модификаторы для детекции зеркальных сигналов
    mirror_confidence_multiplier: float = 1.15
    mirror_strength_multiplier: float = 1.2
    mirror_execution_delay_ms: int = 0

    # Модификаторы для анализа шума
    noise_confidence_multiplier: float = 0.85
    noise_strength_multiplier: float = 0.9
    noise_execution_delay_ms: int = 200

    # Модификаторы для анализа влияния сессий
    session_influence_confidence_multiplier: float = 0.6
    session_influence_strength_multiplier: float = 0.7
    session_influence_execution_delay_ms: int = 500

    # Модификаторы для маркера сессий
    session_marker_confidence_multiplier: float = 1.1
    session_marker_strength_multiplier: float = 1.15
    session_marker_execution_delay_ms: int = 0

    # Модификаторы для адаптации в реальном времени
    live_adaptation_confidence_multiplier: float = 1.2
    live_adaptation_strength_multiplier: float = 1.1
    live_adaptation_execution_delay_ms: int = 0

    # Модификаторы для эволюционного трансформера
    evolutionary_transformer_confidence_multiplier: float = 1.25
    evolutionary_transformer_strength_multiplier: float = 1.2
    evolutionary_transformer_execution_delay_ms: int = 0

    # Модификаторы для анализа активности китов
    whale_analysis_confidence_multiplier: float = 0.8
    whale_analysis_strength_multiplier: float = 0.7
    whale_analysis_execution_delay_ms: int = 200
    whale_analysis_risk_multiplier: float = 1.3

    # Модификаторы для анализа рисков
    risk_analysis_confidence_multiplier: float = 0.9
    risk_analysis_strength_multiplier: float = 0.8
    risk_analysis_execution_delay_ms: int = 100
    risk_analysis_position_size_multiplier: float = 0.7
    risk_analysis_leverage_multiplier: float = 0.8

    # Модификаторы для анализа портфеля
    portfolio_analysis_confidence_multiplier: float = 1.05
    portfolio_analysis_strength_multiplier: float = 1.1
    portfolio_analysis_execution_delay_ms: int = 50
    portfolio_analysis_position_size_multiplier: float = 1.1
    portfolio_analysis_weight_adjustment: float = 1.0

    meta_controller_boost: float = 1.0

    # StateManager
    state_manager_confidence_multiplier: float = 1.0
    state_manager_strength_multiplier: float = 1.0
    state_manager_execution_delay_ms: int = 0

    # DatasetManager
    dataset_manager_confidence_multiplier: float = 1.05
    dataset_manager_strength_multiplier: float = 1.1
    dataset_manager_execution_delay_ms: int = 0
    dataset_manager_data_quality_boost: float = 1.0

    # Модификаторы для паттернов маркет-мейкера
    mm_pattern_confidence_multiplier: float = 1.2
    mm_pattern_strength_multiplier: float = 1.15
    mm_pattern_execution_delay_ms: int = 0
    mm_pattern_position_size_multiplier: float = 1.1
    mm_pattern_risk_multiplier: float = 0.9
    mm_pattern_price_offset_multiplier: float = 1.05


@dataclass
class PatternPredictionContext:
    """Контекст прогнозирования паттернов."""

    # Прогноз паттерна
    prediction_result: Optional[EnhancedPredictionResult] = None

    # Статус прогнозирования
    is_prediction_available: bool = False
    prediction_confidence: float = 0.0
    predicted_direction: Optional[str] = None
    predicted_return_percent: float = 0.0
    predicted_duration_minutes: int = 0

    # Модификаторы на основе прогноза
    pattern_confidence_boost: float = 1.0
    pattern_risk_adjustment: float = 1.0
    pattern_position_multiplier: float = 1.0

    # Метаданные
    pattern_type: Optional[str] = None
    similar_cases_count: Union[int, float] = 0
    success_rate: float = 0.0
    data_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "is_prediction_available": self.is_prediction_available,
            "prediction_confidence": self.prediction_confidence,
            "predicted_direction": self.predicted_direction,
            "predicted_return_percent": self.predicted_return_percent,
            "predicted_duration_minutes": self.predicted_duration_minutes,
            "pattern_confidence_boost": self.pattern_confidence_boost,
            "pattern_risk_adjustment": self.pattern_risk_adjustment,
            "pattern_position_multiplier": self.pattern_position_multiplier,
            "pattern_type": self.pattern_type,
            "similar_cases_count": self.similar_cases_count,
            "success_rate": self.success_rate,
            "data_quality_score": self.data_quality_score,
        }


@dataclass
class SessionContext:
    """Контекст торговых сессий."""

    # Текущие сигналы сессий
    session_signals: Dict[str, SessionInfluenceSignal] = field(default_factory=dict)

    # Агрегированный сигнал
    aggregated_signal: Optional[SessionInfluenceSignal] = None

    # Статистика сессий
    session_statistics: Dict[str, Any] = field(default_factory=dict)

    session_confidence_boost: float = 1.0
    session_aggressiveness_modifier: float = 1.0
    session_position_multiplier: float = 1.0

    # Метаданные
    primary_session: Optional[str] = None
    session_phase: Optional[str] = None
    session_overlap: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "session_signals": {
                k: {
                    "session_type": v.session_type.value,
                    "influence_strength": v.influence_strength,
                    "confidence": v.confidence,
                    "predicted_impact": v.predicted_impact,
                    "metadata": v.metadata
                } for k, v in self.session_signals.items()
            },
            "aggregated_signal": (
                {
                    "session_type": self.aggregated_signal.session_type.value,
                    "influence_strength": self.aggregated_signal.influence_strength,
                    "confidence": self.aggregated_signal.confidence,
                    "predicted_impact": self.aggregated_signal.predicted_impact,
                    "metadata": self.aggregated_signal.metadata
                } if self.aggregated_signal else None
            ),
            "session_statistics": self.session_statistics,
            "session_confidence_boost": self.session_confidence_boost,
            "session_aggressiveness_modifier": self.session_aggressiveness_modifier,
            "session_position_multiplier": self.session_position_multiplier,
            "primary_session": self.primary_session,
            "session_phase": self.session_phase,
            "session_overlap": self.session_overlap,
        }


@dataclass
class MarketMakerPatternContext:
    """Контекст паттернов маркет-мейкера."""

    # Текущий паттерн
    current_pattern: Optional[MarketMakerPattern] = None

    # Сигнал следования
    follow_signal: Optional[MMFollowSignal] = None

    # Результат следования
    follow_result: Optional[MMFollowResult] = None

    # Статистика паттернов
    pattern_statistics: Dict[str, Any] = field(default_factory=dict)

    # Модификаторы на основе паттернов
    pattern_confidence_boost: float = 1.0
    pattern_aggressiveness_modifier: float = 1.0
    pattern_position_multiplier: float = 1.0
    pattern_risk_modifier: float = 1.0
    pattern_price_offset_modifier: float = 1.0

    # Метаданные
    pattern_type: Optional[str] = None
    pattern_confidence: float = 0.0
    expected_direction: Optional[str] = None
    expected_return: float = 0.0
    entry_timing: Optional[str] = None
    historical_accuracy: float = 0.0
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "current_pattern": (
                self.current_pattern.to_dict() if self.current_pattern else None
            ),
            "follow_signal": (
                {
                    "pattern_type": getattr(self.follow_signal, 'pattern_type', None),
                    "expected_direction": getattr(self.follow_signal, 'expected_direction', None),
                    "expected_return": getattr(self.follow_signal, 'expected_return', 0.0),
                    "entry_timing": getattr(self.follow_signal, 'entry_timing', None),
                    "position_size_modifier": getattr(self.follow_signal, 'position_size_modifier', 1.0),
                    "risk_modifier": getattr(self.follow_signal, 'risk_modifier', 1.0),
                    "stop_loss_modifier": getattr(self.follow_signal, 'stop_loss_modifier', 1.0),
                    "take_profit_modifier": getattr(self.follow_signal, 'take_profit_modifier', 1.0),
                    "confidence": getattr(self.follow_signal, 'confidence', 0.0),
                    "metadata": getattr(self.follow_signal, 'metadata', {})
                } if self.follow_signal else None
            ),
            "follow_result": (
                {
                    "success": getattr(self.follow_result, 'success', False),
                    "actual_return": getattr(self.follow_result, 'actual_return', 0.0),
                    "metadata": getattr(self.follow_result, 'metadata', {})
                } if self.follow_result else None
            ),
            "pattern_statistics": self.pattern_statistics,
            "pattern_confidence_boost": self.pattern_confidence_boost,
            "pattern_aggressiveness_modifier": self.pattern_aggressiveness_modifier,
            "pattern_position_multiplier": self.pattern_position_multiplier,
            "pattern_risk_modifier": self.pattern_risk_modifier,
            "pattern_price_offset_modifier": self.pattern_price_offset_modifier,
            "pattern_type": self.pattern_type,
            "pattern_confidence": self.pattern_confidence,
            "expected_direction": self.expected_direction,
            "expected_return": self.expected_return,
            "entry_timing": self.entry_timing,
            "historical_accuracy": self.historical_accuracy,
            "similarity_score": self.similarity_score,
        }


@dataclass
class AgentContext:
    """Контекст агента для интеграции аналитических модулей."""

    # Базовые параметры
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Рыночный контекст
    market_context: MarketContext = field(default_factory=MarketContext)

    # Модификаторы стратегий
    strategy_modifiers: StrategyModifiers = field(default_factory=StrategyModifiers)

    # Результаты аналитических модулей
    entanglement_result: Optional[EntanglementResult] = None
    noise_result: Optional[NoiseAnalysisResult] = None
    mirror_signal: Optional[MirrorSignal] = None
    mirror_map: Optional[MirrorMap] = None
    gravity_result: Optional[LiquidityGravityResult] = None
    risk_assessment: Optional[RiskAssessmentResult] = None
    market_pattern_result: Optional[PatternDetection] = None
    session_influence_result: Optional[SessionInfluenceResult] = None

    # Контекст прогнозирования паттернов
    pattern_prediction: PatternPredictionContext = field(
        default_factory=PatternPredictionContext
    )

    # Контекст торговых сессий
    session_context: SessionContext = field(default_factory=SessionContext)

    # Контекст паттернов маркет-мейкера
    mm_pattern_context: MarketMakerPatternContext = field(
        default_factory=MarketMakerPatternContext
    )

    # Внутренние флаги и метаданные
    _flags: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)

    session_marker_result: Optional[MarketSessionContext] = None

    # Результаты адаптации в реальном времени
    live_adaptation_result: Optional[Dict[str, Any]] = None

    decision_reasoning_result: Optional[TradeDecision] = None

    # Результаты эволюционного трансформера
    evolutionary_transformer_result: Optional[Dict[str, Any]] = None

    # Результаты паттерна открытия
    pattern_discovery_result: Optional[Pattern] = None

    # --- MetaLearning ---
    meta_learning_result: Optional[Dict[str, Any]] = None
    # --- AgentWhales ---
    whale_analysis_result: Optional[Dict[str, Any]] = None
    # --- AgentRisk ---
    risk_analysis_result: Optional[Dict[str, Any]] = None
    portfolio_analysis_result: Optional[Dict[str, Any]] = None

    meta_controller_result: Optional[Dict[str, Any]] = None

    genetic_optimization_result: Optional[Dict[str, Any]] = None

    # Эволюционные агенты
    evolvable_news_result: Optional[Dict[str, Any]] = None
    evolvable_market_regime_result: Optional[Dict[str, Any]] = None
    evolvable_strategy_result: Optional[Dict[str, Any]] = None
    evolvable_risk_result: Optional[Dict[str, Any]] = None
    evolvable_portfolio_result: Optional[Dict[str, Any]] = None
    evolvable_order_executor_result: Optional[Dict[str, Any]] = None
    evolvable_meta_controller_result: Optional[Dict[str, Any]] = None
    evolvable_market_maker_result: Optional[Dict[str, Any]] = None

    # ML сервисы
    model_selector_result: Optional[Dict[str, Any]] = None

    # AdvancedPricePredictor
    advanced_price_predictor_result: Optional[Dict[str, Any]] = None

    # WindowOptimizer
    window_optimizer_result: Optional[Dict[str, Any]] = None

    # StateManager
    state_manager_result: Optional[Dict[str, Any]] = None

    # DatasetManager
    dataset_manager_result: Optional[Dict[str, Any]] = None

    # EvolvableDecisionReasoner
    evolvable_decision_reasoner_result: Optional[Dict[str, Any]] = None

    # RegimeDiscovery
    regime_discovery_result: Optional[Dict[str, Any]] = None

    # AdvancedMarketMaker
    advanced_market_maker_result: Optional[Dict[str, Any]] = None

    # MarketMemoryIntegration
    market_memory_integration_result: Optional[Dict[str, Any]] = None

    # MarketMemoryWhaleIntegration
    market_memory_whale_integration_result: Optional[Dict[str, Any]] = None

    # LocalAIController
    local_ai_controller_result: Optional[Dict[str, Any]] = None

    # AnalyticalIntegration
    analytical_integration_result: Optional[Dict[str, Any]] = None

    # EntanglementIntegration
    entanglement_integration_result: Optional[Dict[str, Any]] = None

    # AgentOrderExecutor
    agent_order_executor_result: Optional[Dict[str, Any]] = None

    # AgentMarketRegime
    agent_market_regime_result: Optional[Dict[str, Any]] = None

    # AgentMarketMakerModel
    agent_market_maker_model_result: Optional[Dict[str, Any]] = None

    # SandboxTrainer
    sandbox_trainer_result: Optional[Dict[str, Any]] = None

    # ModelTrainer
    model_trainer_result: Optional[Dict[str, Any]] = None

    # WindowModelTrainer
    window_model_trainer_result: Optional[Dict[str, Any]] = None

    # DOASS интеграция
    doass_result: Optional[SymbolSelectionResult] = None

    # Базовый сигнал для модификации
    base_signal: Optional[Signal] = None

    def set(self, key: str, value: Any) -> None:
        """Установка флага в контексте."""
        self._flags[key] = value
        logger.debug(f"AgentContext[{self.symbol}] set {key} = {value}")

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
        if modifier_type == "order_aggressiveness":
            return self.strategy_modifiers.order_aggressiveness
        elif modifier_type == "position_size_multiplier":
            return self.strategy_modifiers.position_size_multiplier
        elif modifier_type == "confidence_multiplier":
            return self.strategy_modifiers.confidence_multiplier
        elif modifier_type == "price_offset_percent":
            return self.strategy_modifiers.price_offset_percent
        elif modifier_type == "execution_delay_ms":
            return self.strategy_modifiers.execution_delay_ms
        elif modifier_type == "risk_multiplier":
            return self.strategy_modifiers.risk_multiplier
        return 1.0

    # Новые методы для работы с паттернами ММ
    def apply_mm_pattern_modifier(self) -> None:
        """Применение модификатора паттернов маркет-мейкера."""
        try:
            if not self.mm_pattern_context.follow_signal:
                return

            signal = self.mm_pattern_context.follow_signal

            # Применяем модификаторы на основе сигнала следования
            if signal.confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= (
                    getattr(signal, 'position_size_modifier', 1.0)
                )
                self.strategy_modifiers.position_size_multiplier *= (
                    getattr(signal, 'position_size_modifier', 1.0)
                )
                self.strategy_modifiers.confidence_multiplier *= signal.confidence

                # Корректируем риск
                self.strategy_modifiers.risk_multiplier *= getattr(signal, 'risk_modifier', 1.0)

                # Корректируем смещение цены
                self.strategy_modifiers.price_offset_percent *= (
                    getattr(signal, 'stop_loss_modifier', 1.0)
                )

                # Корректируем задержку исполнения на основе времени входа
                entry_timing = getattr(signal, 'entry_timing', 'immediate')
                if entry_timing == "immediate":
                    self.strategy_modifiers.execution_delay_ms = 0
                elif entry_timing == "wait":
                    self.strategy_modifiers.execution_delay_ms = max(
                        0, self.strategy_modifiers.execution_delay_ms + 200
                    )
                elif entry_timing == "gradual":
                    self.strategy_modifiers.execution_delay_ms = max(
                        0, self.strategy_modifiers.execution_delay_ms + 100
                    )

            elif signal.confidence > 0.6:
                # Средняя уверенность - умеренная корректировка
                position_modifier = getattr(signal, 'position_size_modifier', 1.0)
                self.strategy_modifiers.order_aggressiveness *= (
                    1.0 + (position_modifier - 1.0) * 0.5
                )
                self.strategy_modifiers.position_size_multiplier *= (
                    1.0 + (position_modifier - 1.0) * 0.5
                )
                self.strategy_modifiers.confidence_multiplier *= (
                    1.0 + (signal.confidence - 0.6) * 0.5
                )

            else:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Обновляем контекст паттерна
            self.mm_pattern_context.pattern_confidence_boost = signal.confidence
            self.mm_pattern_context.pattern_aggressiveness_modifier = (
                getattr(signal, 'position_size_modifier', 1.0)
            )
            self.mm_pattern_context.pattern_position_multiplier = (
                getattr(signal, 'position_size_modifier', 1.0)
            )
            self.mm_pattern_context.pattern_risk_modifier = getattr(signal, 'risk_modifier', 1.0)
            self.mm_pattern_context.pattern_price_offset_modifier = (
                getattr(signal, 'stop_loss_modifier', 1.0)
            )
            self.mm_pattern_context.pattern_type = getattr(signal, 'pattern_type', None)
            self.mm_pattern_context.pattern_confidence = signal.confidence
            self.mm_pattern_context.expected_direction = getattr(signal, 'expected_direction', None)
            self.mm_pattern_context.expected_return = getattr(signal, 'expected_return', 0.0)
            self.mm_pattern_context.entry_timing = getattr(signal, 'entry_timing', None)

            # Извлекаем метаданные
            if signal.metadata:
                self.mm_pattern_context.historical_accuracy = signal.metadata.get(
                    "historical_accuracy", 0.0
                )
                self.mm_pattern_context.similarity_score = signal.metadata.get(
                    "similarity_score", 0.0
                )

            logger.debug(
                f"Applied MM pattern modifier for {self.symbol}: pattern={getattr(signal, 'pattern_type', 'unknown')}, confidence={signal.confidence:.3f}, direction={getattr(signal, 'expected_direction', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"Error applying MM pattern modifier: {e}")

    def get_mm_pattern_status(self) -> Dict[str, Any]:
        """Получение статуса паттернов маркет-мейкера."""
        try:
            if not self.mm_pattern_context.follow_signal:
                return {"pattern_detected": False, "status": "no_pattern"}

            signal = self.mm_pattern_context.follow_signal

            return {
                "pattern_detected": True,
                "pattern_type": signal.pattern_type,
                "confidence": signal.confidence,
                "expected_direction": signal.expected_direction,
                "expected_return": signal.expected_return,
                "entry_timing": signal.entry_timing,
                "historical_accuracy": self.mm_pattern_context.historical_accuracy,
                "similarity_score": self.mm_pattern_context.similarity_score,
                "position_size_modifier": signal.position_size_modifier,
                "risk_modifier": signal.risk_modifier,
                "stop_loss_modifier": signal.stop_loss_modifier,
                "take_profit_modifier": signal.take_profit_modifier,
                "status": (
                    "high_confidence"
                    if signal.confidence > 0.8
                    else (
                        "medium_confidence"
                        if signal.confidence > 0.6
                        else "low_confidence"
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Error getting MM pattern status: {e}")
            return {"pattern_detected": False, "status": "error"}

    def update_mm_pattern_result(self, result: MMFollowResult) -> None:
        """Обновление результата следования за паттерном ММ."""
        try:
            self.mm_pattern_context.follow_result = result

            # Обновляем статистику
            if "total_follows" not in self.mm_pattern_context.pattern_statistics:
                self.mm_pattern_context.pattern_statistics["total_follows"] = 0
                self.mm_pattern_context.pattern_statistics["successful_follows"] = 0
                self.mm_pattern_context.pattern_statistics["total_return"] = 0.0

            self.mm_pattern_context.pattern_statistics["total_follows"] += 1
            self.mm_pattern_context.pattern_statistics[
                "total_return"
            ] += float(result.actual_return) if hasattr(result, 'actual_return') else 0.0

            if hasattr(result, 'success') and result.success:
                self.mm_pattern_context.pattern_statistics["successful_follows"] += 1

            # Обновляем успешность
            total_follows = self.mm_pattern_context.pattern_statistics["total_follows"]
            successful_follows = self.mm_pattern_context.pattern_statistics[
                "successful_follows"
            ]
            self.mm_pattern_context.pattern_statistics["success_rate"] = (
                successful_follows / total_follows if total_follows > 0 else 0.0
            )

            # Обновляем среднюю доходность
            self.mm_pattern_context.pattern_statistics["avg_return"] = (
                self.mm_pattern_context.pattern_statistics["total_return"]
                / total_follows
            )

            logger.debug(
                f"Updated MM pattern result for {self.symbol}: success={result.success}, return={result.actual_return:.3f}"
            )

        except Exception as e:
            logger.error(f"Error updating MM pattern result: {e}")

    def get_mm_pattern_statistics(self) -> Dict[str, Any]:
        """Получение статистики паттернов ММ."""
        try:
            stats = self.mm_pattern_context.pattern_statistics.copy()

            # Добавляем текущий статус
            current_status = self.get_mm_pattern_status()
            stats.update(current_status)

            return stats

        except Exception as e:
            logger.error(f"Error getting MM pattern statistics: {e}")
            return {"error": str(e)}

    def apply_session_signal(self, signal: SessionInfluenceSignal) -> None:
        """Применение сигнала сессии к контексту."""
        try:
            # Сохраняем сигнал в контексте сессий
            self.session_context.session_signals[signal.session_type.value] = signal

            # Обновляем агрегированный сигнал
            self.session_context.aggregated_signal = signal

            # Применяем модификаторы на основе сигнала
            score = signal.influence_strength - 0.5  # Преобразуем в диапазон [-0.5, 0.5]
            if score > 0.5:
                # Позитивный сигнал - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.0 + score * 0.2
                self.strategy_modifiers.position_size_multiplier *= (
                    1.0 + score * 0.1
                )
            elif score < -0.5:
                # Негативный сигнал - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= (
                    1.0 - abs(score) * 0.2
                )
                self.strategy_modifiers.position_size_multiplier *= (
                    1.0 - abs(score) * 0.1
                )

            # Обновляем модификаторы уверенности
            self.strategy_modifiers.confidence_multiplier *= (
                1.0 + abs(score) * 0.1
            )

            # Обновляем метаданные сессии
            if hasattr(signal, 'influence_score'):
                self.session_context.session_confidence_boost = signal.influence_score
            if hasattr(signal, 'volume_impact'):
                self.session_context.session_aggressiveness_modifier = signal.volume_impact
            if hasattr(signal, 'volatility_impact'):
                self.session_context.session_position_multiplier = signal.volatility_impact
            if hasattr(signal, 'liquidity_impact'):
                self.session_context.session_position_multiplier *= signal.liquidity_impact
            if hasattr(signal, 'price_impact'):
                self.session_context.session_position_multiplier *= signal.price_impact

            logger.debug(
                f"Applied session signal for {self.symbol}: score={score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying session signal: {e}")

    def get_session_signal(self, session_type: str) -> Optional[SessionInfluenceSignal]:
        """Получение сигнала сессии."""
        return self.session_context.session_signals.get(session_type)

    def get_aggregated_session_signal(self) -> Optional[SessionInfluenceSignal]:
        """Получение агрегированного сигнала сессий."""
        return self.session_context.aggregated_signal

    def apply_aggregated_session_signal(self) -> None:
        """Применение агрегированного сигнала сессий."""
        try:
            if not self.session_context.aggregated_signal:
                return

            signal = self.session_context.aggregated_signal

            # Применяем модификаторы на основе агрегированного сигнала
            score = signal.influence_strength - 0.5  # Преобразуем в диапазон [-0.5, 0.5]
            if score > 0.3:
                # Позитивный сигнал
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif score < -0.3:
                # Негативный сигнал
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.95
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Применяем модификаторы волатильности
            volatility_impact = signal.predicted_impact.get("volatility", 0.0)
            if volatility_impact > 0.5:
                self.strategy_modifiers.price_offset_percent *= 1.2
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 1.5)

            logger.debug(f"Applied aggregated session signal: score={score:.3f}")

        except Exception as e:
            logger.error(f"Error applying aggregated session signal: {e}")

    def apply_entanglement_modifier(self) -> None:
        """Применение модификатора запутанности."""
        try:
            if not self.entanglement_result:
                return

            if self.entanglement_result.is_entangled:
                # Высокая запутанность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.9

                # Увеличиваем задержку исполнения
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms + 200
                )

                logger.debug(f"Applied entanglement modifier for {self.symbol}")

        except Exception as e:
            logger.error(f"Error applying entanglement modifier: {e}")

    def apply_entanglement_monitor_modifier(self) -> None:
        """Применение модификатора монитора запутанности."""
        try:
            if not self.entanglement_result:
                return

            # Применяем модификаторы на основе результатов запутанности
            correlation_score = self.entanglement_result.correlation_score
            lag_ms = self.entanglement_result.lag_ms
            confidence = self.entanglement_result.confidence

            # Модификаторы на основе корреляции
            if correlation_score > 0.95:
                # Очень высокая корреляция - резко снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.6
                self.strategy_modifiers.position_size_multiplier *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.7
                self.strategy_modifiers.risk_multiplier *= 1.5

            elif correlation_score > 0.8:
                # Высокая корреляция - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.risk_multiplier *= 1.2

            # Модификаторы на основе лага
            if lag_ms < 1.0:
                # Очень быстрая синхронизация - увеличиваем осторожность
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms + 300
                )
                self.strategy_modifiers.price_offset_percent *= 1.3

            elif lag_ms > 5.0:
                # Медленная синхронизация - снижаем задержки
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms - 100
                )

            # Модификаторы на основе уверенности
            if confidence < 0.7:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.8

            logger.debug(
                f"Applied entanglement monitor modifier: correlation={correlation_score:.3f}, lag={lag_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error applying entanglement monitor modifier: {e}")

    def get_entanglement_status(self) -> Dict[str, Any]:
        """Получение статуса запутанности."""
        try:
            if not self.entanglement_result:
                return {"is_entangled": False, "status": "unknown"}

            return {
                "is_entangled": self.entanglement_result.is_entangled,
                "correlation_score": self.entanglement_result.correlation_score,
                "lag_ms": self.entanglement_result.lag_ms,
                "confidence": self.entanglement_result.confidence,
                "exchange_pair": self.entanglement_result.exchange_pair,
                "status": (
                    "high_risk" if self.entanglement_result.is_entangled else "normal"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting entanglement status: {e}")
            return {"is_entangled": False, "status": "error"}

    def apply_noise_modifier(self) -> None:
        """Применение модификатора шума."""
        try:
            if not self.noise_result:
                return

            if self.noise_result.is_synthetic_noise:
                # Синтетический шум - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.confidence_multiplier *= 0.8

                # Увеличиваем задержку исполнения
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms + 300
                )

                logger.debug(f"Applied noise modifier for {self.symbol}")

        except Exception as e:
            logger.error(f"Error applying noise modifier: {e}")

    def apply_noise_analyzer_modifier(self) -> None:
        """Применение модификатора анализатора шума."""
        try:
            if not self.noise_result:
                return

            # Применяем модификаторы на основе результатов анализа шума
            fractal_dimension = self.noise_result.fractal_dimension
            entropy = self.noise_result.entropy
            confidence = self.noise_result.confidence

            # Модификаторы на основе фрактальной размерности
            if fractal_dimension < 1.2:
                # Низкая фрактальная размерность - возможен синтетический шум
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.risk_multiplier *= 1.3

            elif fractal_dimension > 1.6:
                # Высокая фрактальная размерность - хаотичное поведение
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.85

            # Модификаторы на основе энтропии
            if entropy > 0.8:
                # Высокая энтропия - неопределенность
                self.strategy_modifiers.order_aggressiveness *= 0.85
                self.strategy_modifiers.position_size_multiplier *= 0.75
                self.strategy_modifiers.confidence_multiplier *= 0.8

            elif entropy < 0.3:
                # Низкая энтропия - предсказуемость
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1

            # Модификаторы на основе уверенности
            if confidence < 0.6:
                # Низкая уверенность в анализе
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.85
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Специальные модификаторы для синтетического шума
            if self.noise_result.is_synthetic_noise:
                # Синтетический шум - резко снижаем активность
                self.strategy_modifiers.order_aggressiveness *= 0.5
                self.strategy_modifiers.position_size_multiplier *= 0.4
                self.strategy_modifiers.confidence_multiplier *= 0.6
                self.strategy_modifiers.risk_multiplier *= 2.0

                # Увеличиваем задержки и смещения цен
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms + 500
                )
                self.strategy_modifiers.price_offset_percent *= 1.5

            logger.debug(
                f"Applied noise analyzer modifier: FD={fractal_dimension:.3f}, entropy={entropy:.3f}, confidence={confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying noise analyzer modifier: {e}")

    def get_noise_analysis_status(self) -> Dict[str, Any]:
        """Получение статуса анализа шума."""
        try:
            if not self.noise_result:
                return {"is_synthetic_noise": False, "status": "unknown"}

            return {
                "is_synthetic_noise": self.noise_result.is_synthetic_noise,
                "fractal_dimension": self.noise_result.fractal_dimension,
                "entropy": self.noise_result.entropy,
                "confidence": self.noise_result.confidence,
                "status": (
                    "synthetic_noise"
                    if self.noise_result.is_synthetic_noise
                    else "natural_noise"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting noise analysis status: {e}")
            return {"is_synthetic_noise": False, "status": "error"}

    def apply_mirror_modifier(self) -> None:
        """Применение модификатора зеркальных сигналов."""
        try:
            if not self.mirror_signal:
                return

            # Применяем модификаторы на основе зеркального сигнала
            if self.mirror_signal.correlation > 0.8:
                # Высокая корреляция - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.15

            elif self.mirror_signal.correlation < 0.3:
                # Низкая корреляция - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.9
                self.strategy_modifiers.confidence_multiplier *= 0.85

            logger.debug(
                f"Applied mirror modifier: correlation={self.mirror_signal.correlation:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying mirror modifier: {e}")

    def apply_mirror_map_modifier(self) -> None:
        """Применение модификатора зеркальной карты."""
        try:
            if not self.mirror_map:
                return

            # Применяем модификаторы на основе симметрии
            symmetry_score = self.mirror_map.symmetry_score
            confidence = self.mirror_map.confidence

            if symmetry_score > 0.8:
                # Высокая симметрия - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif symmetry_score > 0.6:
                # Средняя симметрия - умеренная корректировка
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
            else:
                # Низкая симметрия - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8

            # Модификаторы на основе уверенности
            if confidence > 0.8:
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif confidence < 0.5:
                self.strategy_modifiers.confidence_multiplier *= 0.9

            logger.debug(
                f"Applied mirror map modifier for {self.symbol}: symmetry={symmetry_score:.3f}, confidence={confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying mirror map modifier: {e}")

    def get_mirror_map_status(self) -> Dict[str, Any]:
        """Получение статуса зеркальной карты."""
        try:
            if not self.mirror_map:
                return {"mirror_detected": False, "status": "no_mirror"}

            return {
                "mirror_detected": True,
                "symmetry_score": self.mirror_map.symmetry_score,
                "confidence": self.mirror_map.confidence,
                "market_regime": self.mirror_map.market_regime.value,
                "reflection_points_count": len(self.mirror_map.reflection_points),
                "status": (
                    "high_symmetry"
                    if self.mirror_map.symmetry_score > 0.8
                    else (
                        "medium_symmetry"
                        if self.mirror_map.symmetry_score > 0.6
                        else "low_symmetry"
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Error getting mirror map status: {e}")
            return {"mirror_detected": False, "status": "error"}

    def get_mirror_assets(self) -> list[str]:
        try:
            if not self.mirror_map:
                return []
            mirror_assets: list[str] = []
            for point in self.mirror_map.reflection_points:
                if "symbol" in point and point["symbol"] != self.symbol:
                    mirror_assets.append(str(point["symbol"]))
            return list(set(mirror_assets))
        except Exception as e:
            logger.error(f"Error getting mirror assets: {e}")
            return []

    def get_mirror_correlation(self, asset: str) -> float:
        """Получение корреляции с зеркальным активом."""
        try:
            if not self.mirror_map:
                return 0.0

            # Ищем корреляцию в reflection_points
            for point in self.mirror_map.reflection_points:
                if point.get("symbol") == asset:
                    return point.get("correlation", 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting mirror correlation: {e}")
            return 0.0

    def apply_gravity_modifier(self) -> None:
        """Применение модификатора гравитации ликвидности."""
        try:
            if not self.gravity_result:
                return

            # Применяем модификаторы на основе гравитации ликвидности
            gravity_strength = self.gravity_result.total_gravity

            if gravity_strength > 0.8:
                # Высокая гравитация - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.confidence_multiplier *= 0.8

                # Увеличиваем задержку исполнения
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 2.5)

            elif gravity_strength < 0.2:
                # Низкая гравитация - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1

            logger.debug(f"Applied gravity modifier: strength={gravity_strength:.3f}")

        except Exception as e:
            logger.error(f"Error applying gravity modifier: {e}")

    def apply_risk_modifier(self) -> None:
        try:
            if not self.risk_assessment:
                return
            risk_level = self.risk_assessment.risk_level
            risk_score = self.risk_assessment.risk_score
            if risk_level == RiskLevel.CRITICAL:
                self.strategy_modifiers.order_aggressiveness *= 0.3
                self.strategy_modifiers.position_size_multiplier *= 0.2
                self.strategy_modifiers.confidence_multiplier *= 0.5
                self.strategy_modifiers.risk_multiplier *= 2.0
            elif risk_level == RiskLevel.HIGH:
                self.strategy_modifiers.order_aggressiveness *= 0.6
                self.strategy_modifiers.position_size_multiplier *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.7
                self.strategy_modifiers.risk_multiplier *= 1.5
            elif risk_level == RiskLevel.MEDIUM:
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.risk_multiplier *= 1.2
            if hasattr(self.risk_assessment, 'gravity_score') and self.risk_assessment.gravity_score > 0.7:
                self.strategy_modifiers.price_offset_percent *= 1.3
            if hasattr(self.risk_assessment, 'liquidity_score') and self.risk_assessment.liquidity_score < 0.3:
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 2.0)
            if hasattr(self.risk_assessment, 'volatility_score') and self.risk_assessment.volatility_score > 0.7:
                self.strategy_modifiers.price_offset_percent *= 1.2
            logger.debug(
                f"Applied risk modifier: level={risk_level}, score={risk_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Error applying risk modifier: {e}")

    def apply_market_pattern_modifier(self) -> None:
        """Применение модификатора распознавания паттернов рынка."""
        try:
            if not self.market_pattern_result:
                return

            # Применяем модификаторы на основе типа паттерна
            pattern_type = self.market_pattern_result.pattern_type.value
            confidence = self.market_pattern_result.confidence
            strength = self.market_pattern_result.strength
            direction = self.market_pattern_result.direction

            # Базовые модификаторы на основе уверенности
            if confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.15
            elif confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.9
                self.strategy_modifiers.confidence_multiplier *= 0.85

            # Специфичные модификаторы для разных типов паттернов
            if pattern_type == "whale_absorption":
                # Поглощение китами - осторожность
                if direction == "up":
                    # Восходящее поглощение - умеренная агрессивность
                    self.strategy_modifiers.order_aggressiveness *= 1.1
                    self.strategy_modifiers.position_size_multiplier *= 1.05
                else:
                    # Нисходящее поглощение - снижаем активность
                    self.strategy_modifiers.order_aggressiveness *= 0.7
                    self.strategy_modifiers.position_size_multiplier *= 0.6
                    self.strategy_modifiers.risk_multiplier *= 1.3

            elif pattern_type == "mm_spoofing":
                # Спуфинг маркет-мейкеров - высокая осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.6
                self.strategy_modifiers.position_size_multiplier *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.7
                self.strategy_modifiers.risk_multiplier *= 1.5
                self.strategy_modifiers.execution_delay_ms = int(
                    self.strategy_modifiers.execution_delay_ms + 300
                )

            elif pattern_type == "iceberg_detection":
                # Обнаружение айсбергов - умеренная осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.price_offset_percent *= 1.2

            elif pattern_type == "liquidity_grab":
                # Захват ликвидности - высокая осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.5
                self.strategy_modifiers.position_size_multiplier *= 0.4
                self.strategy_modifiers.confidence_multiplier *= 0.6
                self.strategy_modifiers.risk_multiplier *= 1.8
                self.strategy_modifiers.execution_delay_ms = int(
                    self.strategy_modifiers.execution_delay_ms + 500
                )

            elif pattern_type == "pump_and_dump":
                # Накачка и сброс - избегаем
                self.strategy_modifiers.order_aggressiveness *= 0.3
                self.strategy_modifiers.position_size_multiplier *= 0.2
                self.strategy_modifiers.confidence_multiplier *= 0.4
                self.strategy_modifiers.risk_multiplier *= 2.0

            elif pattern_type == "stop_hunting":
                # Охота за стопами - высокая осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.4
                self.strategy_modifiers.position_size_multiplier *= 0.3
                self.strategy_modifiers.confidence_multiplier *= 0.5
                self.strategy_modifiers.risk_multiplier *= 1.7
                self.strategy_modifiers.price_offset_percent *= 1.5

            elif pattern_type == "accumulation":
                # Накопление - умеренная агрессивность
                if direction == "up":
                    self.strategy_modifiers.order_aggressiveness *= 1.1
                    self.strategy_modifiers.position_size_multiplier *= 1.05
                    self.strategy_modifiers.confidence_multiplier *= 1.1

            elif pattern_type == "distribution":
                # Распределение - снижаем активность
                if direction == "down":
                    self.strategy_modifiers.order_aggressiveness *= 0.7
                    self.strategy_modifiers.position_size_multiplier *= 0.6
                    self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе силы паттерна
            if strength > 0.8:
                # Сильный паттерн - усиливаем модификаторы
                self.strategy_modifiers.confidence_multiplier *= 1.2
            elif strength < 0.3:
                # Слабый паттерн - ослабляем модификаторы
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе метрик паттерна
            if self.market_pattern_result.volume_anomaly > 2.0:
                # Аномально высокий объем - осторожность
                self.strategy_modifiers.risk_multiplier *= 1.2

            if self.market_pattern_result.price_impact > 0.05:
                # Высокое влияние на цену - осторожность
                self.strategy_modifiers.price_offset_percent *= 1.3

            if self.market_pattern_result.order_book_imbalance > 0.7:
                # Сильный дисбаланс стакана - осторожность
                self.strategy_modifiers.execution_delay_ms = int(
                    self.strategy_modifiers.execution_delay_ms + 200
                )

            logger.debug(
                f"Applied market pattern modifier: type={pattern_type}, confidence={confidence:.3f}, strength={strength:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying market pattern modifier: {e}")

    def get_market_pattern_status(self) -> Dict[str, Any]:
        """Получение статуса распознавания паттернов рынка."""
        try:
            if not self.market_pattern_result:
                return {"pattern_detected": False, "status": "unknown"}

            return {
                "pattern_detected": True,
                "pattern_type": self.market_pattern_result.pattern_type.value,
                "confidence": self.market_pattern_result.confidence,
                "strength": self.market_pattern_result.strength,
                "direction": self.market_pattern_result.direction,
                "volume_anomaly": self.market_pattern_result.volume_anomaly,
                "price_impact": self.market_pattern_result.price_impact,
                "order_book_imbalance": self.market_pattern_result.order_book_imbalance,
                "spread_widening": self.market_pattern_result.spread_widening,
                "depth_absorption": self.market_pattern_result.depth_absorption,
                "status": (
                    "high_risk"
                    if self.market_pattern_result.confidence > 0.8
                    else "moderate_risk"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting market pattern status: {e}")
            return {"pattern_detected": False, "status": "error"}

    def apply_session_influence_modifier(self) -> None:
        """Применение модификатора влияния сессий."""
        try:
            if not self.session_influence_result:
                return

            # Применяем модификаторы на основе влияния сессий
            influence_score = getattr(self.session_influence_result, 'influence_score', 0.0)
            volume_impact = getattr(self.session_influence_result, 'volume_impact', 0.0)
            volatility_impact = getattr(self.session_influence_result, 'volatility_impact', 0.0)
            liquidity_impact = getattr(self.session_influence_result, 'liquidity_impact', 0.0)

            if influence_score > 0.7:
                # Высокое влияние - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif influence_score < 0.3:
                # Низкое влияние - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.9
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Модификаторы на основе объема
            if volume_impact > 0.5:
                self.strategy_modifiers.position_size_multiplier *= 1.1

            # Модификаторы на основе волатильности
            if volatility_impact > 0.5:
                self.strategy_modifiers.price_offset_percent *= 1.2
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 1.5)

            # Модификаторы на основе ликвидности
            if liquidity_impact < 0.3:
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 2.0)

            logger.debug(
                f"Applied session influence modifier: score={influence_score:.3f}, volume={volume_impact:.3f}, volatility={volatility_impact:.3f}, liquidity={liquidity_impact:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying session influence modifier: {e}")

    def get_session_influence_status(self) -> Dict[str, Any]:
        """Получение статуса анализа влияния сессий."""
        try:
            if not self.session_influence_result:
                return {"influence_detected": False, "status": "unknown"}

            return {
                "influence_detected": True,
                "session_type": self.session_influence_result.session_type,
                "influence_score": self.session_influence_result.confidence,
                "confidence": self.session_influence_result.confidence,
                "volume_impact": self.session_influence_result.influence_metrics.volume_change_percent,
                "volatility_impact": self.session_influence_result.influence_metrics.volatility_change_percent,
                "liquidity_impact": self.session_influence_result.influence_metrics.liquidity_impact,
                "price_impact": self.session_influence_result.influence_metrics.price_direction_bias,
                "timestamp": self.session_influence_result.timestamp.to_iso(),
                "status": (
                    "high_influence"
                    if self.session_influence_result.confidence > 0.7
                    else "moderate_influence"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting session influence status: {e}")
            return {"influence_detected": False, "status": "error"}

    def update_session_influence_result(self, result: SessionInfluenceResult) -> None:
        """Обновление результата анализа влияния сессий."""
        try:
            self.session_influence_result = result
            logger.debug(
                f"Updated session influence result for {self.symbol}: type={result.session_type}, score={result.confidence:.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating session influence result: {e}")

    def get_session_influence_result(self) -> Optional[SessionInfluenceResult]:
        """Получение результата анализа влияния сессий."""
        return self.session_influence_result

    def apply_pattern_prediction_modifier(self) -> None:
        """Применение модификатора прогнозирования паттернов."""
        try:
            if not self.pattern_prediction.prediction_result:
                return

            prediction_result = self.pattern_prediction.prediction_result
            prediction = getattr(prediction_result, 'prediction', None)
            confidence = getattr(prediction_result, 'confidence', 0.0)

            if prediction and confidence > 0.7:
                # Высокая уверенность в прогнозе - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.15
            elif prediction and confidence > 0.5:
                # Средняя уверенность - умеренная корректировка
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.05
            else:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.9
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Обновляем контекст прогнозирования
            self.pattern_prediction.is_prediction_available = True
            self.pattern_prediction.prediction_confidence = confidence
            self.pattern_prediction.predicted_direction = str(prediction) if prediction else None

            logger.debug(
                f"Applied pattern prediction modifier: prediction={prediction}, confidence={confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying pattern prediction modifier: {e}")

    def set_predicted_direction(self, direction: str) -> None:
        """Установка предсказанного направления."""
        self.pattern_prediction.predicted_direction = direction

    def should_proceed_with_trade(self) -> bool:
        """Проверка, следует ли продолжать торговлю."""
        try:
            # Проверяем чистоту рынка
            if not self.is_market_clean():
                logger.warning(f"Market not clean for {self.symbol}")
                return False

            # Проверяем уровень риска
            if self.risk_assessment and self.risk_assessment.risk_level == "critical":
                logger.warning(f"Critical risk level for {self.symbol}")
                return False

            # Проверяем запутанность
            if self.entanglement_result and self.entanglement_result.is_entangled:
                logger.warning(f"High entanglement for {self.symbol}")
                return False

            # Проверяем синтетический шум
            if self.noise_result and self.noise_result.is_synthetic_noise:
                logger.warning(f"Synthetic noise detected for {self.symbol}")
                return False

            # Проверяем гравитацию ликвидности
            if self.gravity_result and self.gravity_result.total_gravity > 0.9:
                logger.warning(f"High liquidity gravity for {self.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking trade conditions: {e}")
            return False

    def get_adjusted_aggressiveness(self) -> float:
        """Получение скорректированной агрессивности."""
        return self.strategy_modifiers.order_aggressiveness

    def get_adjusted_position_size(self) -> float:
        """Получение скорректированного размера позиции."""
        return self.strategy_modifiers.position_size_multiplier

    def get_adjusted_confidence(self) -> float:
        """Получение скорректированной уверенности."""
        return self.strategy_modifiers.confidence_multiplier

    def get_price_offset(self) -> float:
        """Получение смещения цены."""
        return self.strategy_modifiers.price_offset_percent

    def get_trading_recommendations(self) -> Dict[str, Any]:
        """Получение торговых рекомендаций."""
        recommendations = {
            "whale_confidence": 0.0,
            "whale_boost": False,
            "position_size_multiplier": 1.0,
            "confidence_multiplier": 1.0,
            "risk_level": "normal",
            "market_conditions": [],
            "analytical_signals": [],
            "execution_advice": [],
        }

        # Рекомендации на основе оценки риска
        if self.risk_assessment and hasattr(self.risk_assessment, 'recommendations'):
            if isinstance(self.risk_assessment.recommendations, list):
                if isinstance(recommendations["analytical_signals"], list):
                    recommendations["analytical_signals"].extend(
                        self.risk_assessment.recommendations
                    )

        # Рекомендации на основе рыночного контекста
        if not self.market_context.is_clean:
            if isinstance(recommendations["market_conditions"], list):
                recommendations["market_conditions"].append(
                    "Рынок нестабилен - соблюдать осторожность"
                )

        if self.market_context.regime_shift:
            if isinstance(recommendations["market_conditions"], list):
                recommendations["market_conditions"].append(
                    "Смена режима рынка - адаптировать стратегию"
                )

        if self.market_context.unreliable_depth:
            if isinstance(recommendations["market_conditions"], list):
                recommendations["market_conditions"].append(
                    "Ненадежная глубина рынка - использовать лимитные ордера"
                )

        # Рекомендации на основе аналитических модулей
        if self.entanglement_result and self.entanglement_result.is_entangled:
            if isinstance(recommendations["analytical_signals"], list):
                recommendations["analytical_signals"].append(
                    "Высокая запутанность - избегать крупных позиций"
                )

        if self.noise_result and self.noise_result.is_synthetic_noise:
            if isinstance(recommendations["analytical_signals"], list):
                recommendations["analytical_signals"].append(
                    "Синтетический шум - снизить активность"
                )

        if self.gravity_result and self.gravity_result.total_gravity > 0.8:
            if isinstance(recommendations["analytical_signals"], list):
                recommendations["analytical_signals"].append(
                    "Высокая гравитация ликвидности - возможны резкие движения"
                )

        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование контекста в словарь."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "market_context": {
                "is_clean": self.market_context.is_clean,
                "external_sync": self.market_context.external_sync,
                "regime_shift": self.market_context.regime_shift,
                "gravity_bias": self.market_context.gravity_bias,
                "unreliable_depth": self.market_context.unreliable_depth,
                "synthetic_noise": self.market_context.synthetic_noise,
                "leader_asset": self.market_context.leader_asset,
                "mirror_correlation": self.market_context.mirror_correlation,
                "price_influence_bias": self.market_context.price_influence_bias,
            },
            "strategy_modifiers": {
                "order_aggressiveness": self.strategy_modifiers.order_aggressiveness,
                "position_size_multiplier": self.strategy_modifiers.position_size_multiplier,
                "confidence_multiplier": self.strategy_modifiers.confidence_multiplier,
                "price_offset_percent": self.strategy_modifiers.price_offset_percent,
                "execution_delay_ms": self.strategy_modifiers.execution_delay_ms,
                "risk_multiplier": self.strategy_modifiers.risk_multiplier,
                "scalping_enabled": self.strategy_modifiers.scalping_enabled,
                "mean_reversion_enabled": self.strategy_modifiers.mean_reversion_enabled,
                "momentum_enabled": self.strategy_modifiers.momentum_enabled,
                "meta_controller_boost": self.strategy_modifiers.meta_controller_boost,
                "state_manager_confidence_multiplier": self.strategy_modifiers.state_manager_confidence_multiplier,
                "state_manager_strength_multiplier": self.strategy_modifiers.state_manager_strength_multiplier,
                "state_manager_execution_delay_ms": self.strategy_modifiers.state_manager_execution_delay_ms,
                "dataset_manager_confidence_multiplier": self.strategy_modifiers.dataset_manager_confidence_multiplier,
                "dataset_manager_strength_multiplier": self.strategy_modifiers.dataset_manager_strength_multiplier,
                "dataset_manager_execution_delay_ms": self.strategy_modifiers.dataset_manager_execution_delay_ms,
                "dataset_manager_data_quality_boost": self.strategy_modifiers.dataset_manager_data_quality_boost,
            },
            "pattern_prediction": self.pattern_prediction.to_dict(),
            "session_context": self.session_context.to_dict(),
            "flags": self._flags,
            "metadata": self._metadata,
            "recommendations": self.get_trading_recommendations(),
        }

    def apply_session_marker_modifier(self, signal: Signal) -> Signal:
        """Применение модификатора маркера сессий к сигналу."""
        if not self.session_marker_result:
            return signal

        try:
            # Создаем копию сигнала для модификации
            modified_signal = signal.copy()

            # Применяем модификаторы на основе контекста сессий
            if (
                self.session_marker_result.primary_session
                and self.session_marker_result.primary_session.is_active
            ):
                # Активная основная сессия - увеличиваем уверенность
                new_confidence = Percentage(
                    min(
                        Decimal("1.0"),
                        Decimal(str(float(modified_signal.confidence))) * Decimal("1.1"),
                    )
                )
                # Создаем новый сигнал с обновленной уверенностью
                modified_signal = Signal(
                    direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                    trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                    signal_type=modified_signal.signal_type,
                    confidence=Decimal(str(new_confidence)),
                    strength=modified_signal.strength,
                    metadata=modified_signal.metadata
                )

                # Проверяем фазу сессии
                phase = self.session_marker_result.primary_session.phase
                if phase and hasattr(phase, "value"):
                    if phase.value == "high_volatility":
                        # Высокая волатильность - увеличиваем размер позиции
                        new_strength = Percentage(
                            min(
                                Decimal("1.0"),
                                Decimal(str(float(modified_signal.strength))) * Decimal("1.15"),
                            )
                        )
                        modified_signal = Signal(
                            direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                            trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                            signal_type=modified_signal.signal_type,
                            confidence=modified_signal.confidence,
                            strength=Decimal(str(new_strength)),
                            metadata=modified_signal.metadata
                        )
                    elif phase.value == "low_volatility":
                        # Низкая волатильность - снижаем размер позиции
                        new_strength = Percentage(
                            max(
                                Decimal("0.1"),
                                Decimal(str(float(modified_signal.strength))) * Decimal("0.9"),
                            )
                        )
                        modified_signal = Signal(
                            direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                            trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                            signal_type=modified_signal.signal_type,
                            confidence=modified_signal.confidence,
                            strength=Decimal(str(new_strength)),
                            metadata=modified_signal.metadata
                        )

                # Проверяем перекрытия сессий
                if (
                    self.session_marker_result.primary_session.overlap_with_other_sessions
                ):
                    overlap_count = len(
                        self.session_marker_result.primary_session.overlap_with_other_sessions
                    )
                    if overlap_count > 0:
                        # Перекрытие сессий - увеличиваем агрессивность
                        new_confidence = Percentage(
                            min(
                                Decimal("1.0"),
                                Decimal(str(float(modified_signal.confidence))) * Decimal(str(overlap_count * 0.05)),
                            )
                        )
                        new_strength = Percentage(
                            min(
                                Decimal("1.0"),
                                Decimal(str(float(modified_signal.strength))) * (Decimal("1.0") + Decimal(str(overlap_count * 0.03))),
                            )
                        )
                        modified_signal = Signal(
                            direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                            trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                            signal_type=modified_signal.signal_type,
                            confidence=Decimal(str(new_confidence)),
                            strength=Decimal(str(new_strength)),
                            metadata=modified_signal.metadata
                        )

                # Добавляем метаданные
                new_metadata = modified_signal.metadata or {}
                new_metadata["session_marker"] = {
                    "primary_session": self.session_marker_result.primary_session.session_type.value,
                    "phase": (
                        self.session_marker_result.primary_session.phase.value
                        if self.session_marker_result.primary_session.phase
                        else None
                    ),
                    "active_sessions_count": len(
                        self.session_marker_result.active_sessions
                    ),
                    "overlap_count": (
                        len(
                            self.session_marker_result.primary_session.overlap_with_other_sessions
                        )
                        if self.session_marker_result.primary_session
                        else 0
                    ),
                }
                modified_signal = Signal(
                    direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                    trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                    signal_type=modified_signal.signal_type,
                    confidence=modified_signal.confidence,
                    strength=modified_signal.strength,
                    metadata=new_metadata
                )

            return modified_signal

        except Exception as e:
            logger.error(f"Error applying session marker modifier: {e}")
            return signal

    def apply_live_adaptation_modifier(self) -> None:
        """Применение модификатора адаптации в реальном времени."""
        try:
            if not self.live_adaptation_result:
                return

            # Применяем модификаторы на основе результатов адаптации
            adaptation_score = self.live_adaptation_result.get("adaptation_score", 0.0)
            confidence_boost = self.live_adaptation_result.get("confidence_boost", 1.0)
            risk_adjustment = self.live_adaptation_result.get("risk_adjustment", 1.0)

            # Модификаторы на основе адаптации
            if adaptation_score > 0.7:
                # Высокая адаптация - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= confidence_boost

            elif adaptation_score < 0.3:
                # Низкая адаптация - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.85
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Применяем корректировку риска
            self.strategy_modifiers.risk_multiplier *= risk_adjustment

            logger.debug(
                f"Applied live adaptation modifier: score={adaptation_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying live adaptation modifier: {e}")

    def apply_decision_reasoning_modifier(self) -> None:
        """Применение модификатора анализа решений."""
        try:
            if not self.decision_reasoning_result:
                return

            # Применяем модификаторы на основе результатов анализа решений
            confidence = self.decision_reasoning_result.confidence
            action = self.decision_reasoning_result.action
            # Исправление: используем правильный атрибут вместо direction
            decision_type = getattr(self.decision_reasoning_result, 'decision_type', 'hold')

            # Модификаторы на основе уверенности в решении
            if confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.15
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе действия
            if action == "hold":
                # Решение воздержаться - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.confidence_multiplier *= 0.7

            # Модификаторы на основе направления
            if decision_type in ["long", "short"]:
                # Четкое направление - умеренно увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.05
                self.strategy_modifiers.position_size_multiplier *= 1.02

            logger.debug(
                f"Applied decision reasoning modifier: confidence={confidence:.3f}, action={action}"
            )

        except Exception as e:
            logger.error(f"Error applying decision reasoning modifier: {e}")

    def apply_evolutionary_transformer_modifier(self) -> None:
        """Применение модификатора эволюционного трансформера."""
        try:
            if not self.evolutionary_transformer_result:
                return

            # Применяем модификаторы на основе результатов эволюционного трансформера
            evolution_score = self.evolutionary_transformer_result.get(
                "evolution_score", 0.0
            )
            fitness_score = self.evolutionary_transformer_result.get(
                "fitness_score", 0.0
            )
            adaptation_rate = self.evolutionary_transformer_result.get(
                "adaptation_rate", 1.0
            )
            generation = self.evolutionary_transformer_result.get("generation", 0)
            best_model_confidence = self.evolutionary_transformer_result.get(
                "best_model_confidence", 0.0
            )

            # Модификаторы на основе эволюционного скора
            if evolution_score > 0.8:
                # Высокий эволюционный скор - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.15

            elif evolution_score < 0.3:
                # Низкий эволюционный скор - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе фитнес-скора
            if fitness_score > 0.9:
                # Очень высокий фитнес - максимальная агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.25
                self.strategy_modifiers.position_size_multiplier *= 1.15
                self.strategy_modifiers.confidence_multiplier *= 1.2

            elif fitness_score < 0.5:
                # Низкий фитнес - снижаем активность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.confidence_multiplier *= 0.75

            # Модификаторы на основе скорости адаптации
            if adaptation_rate > 1.5:
                # Быстрая адаптация - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif adaptation_rate < 0.5:
                # Медленная адаптация - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.85
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Модификаторы на основе поколения
            if generation > 50:
                # Зрелая эволюция - стабильность
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.risk_multiplier *= 0.9

            elif generation < 10:
                # Ранняя эволюция - осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.risk_multiplier *= 1.2

            # Модификаторы на основе уверенности лучшей модели
            if best_model_confidence > 0.95:
                # Очень высокая уверенность - максимальная агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.3
                self.strategy_modifiers.position_size_multiplier *= 1.2
                self.strategy_modifiers.confidence_multiplier *= 1.25

            elif best_model_confidence < 0.6:
                # Низкая уверенность - снижаем активность
                self.strategy_modifiers.order_aggressiveness *= 0.6
                self.strategy_modifiers.position_size_multiplier *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.7

            # Специальные модификаторы для эволюционных состояний
            evolution_state = self.evolutionary_transformer_result.get(
                "evolution_state", "stable"
            )
            if evolution_state == "converging":
                # Сходимость - стабильность
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.risk_multiplier *= 0.9

            elif evolution_state == "exploring":
                # Исследование - осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.risk_multiplier *= 1.3

            elif evolution_state == "stagnant":
                # Застой - снижаем активность
                self.strategy_modifiers.order_aggressiveness *= 0.6
                self.strategy_modifiers.position_size_multiplier *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.7

            logger.debug(
                f"Applied evolutionary transformer modifier: evolution_score={evolution_score:.3f}, fitness={fitness_score:.3f}, generation={generation}"
            )

        except Exception as e:
            logger.error(f"Error applying evolutionary transformer modifier: {e}")

    def get_evolutionary_transformer_status(self) -> Dict[str, Any]:
        """Получение статуса эволюционного трансформера."""
        try:
            if not self.evolutionary_transformer_result:
                return {"evolution_active": False, "status": "unknown"}

            return {
                "evolution_active": True,
                "evolution_score": self.evolutionary_transformer_result.get(
                    "evolution_score", 0.0
                ),
                "fitness_score": self.evolutionary_transformer_result.get(
                    "fitness_score", 0.0
                ),
                "adaptation_rate": self.evolutionary_transformer_result.get(
                    "adaptation_rate", 1.0
                ),
                "generation": self.evolutionary_transformer_result.get("generation", 0),
                "best_model_confidence": self.evolutionary_transformer_result.get(
                    "best_model_confidence", 0.0
                ),
                "evolution_state": self.evolutionary_transformer_result.get(
                    "evolution_state", "unknown"
                ),
                "status": (
                    "high_evolution"
                    if self.evolutionary_transformer_result.get("evolution_score", 0.0)
                    > 0.8
                    else "moderate_evolution"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting evolutionary transformer status: {e}")
            return {"evolution_active": False, "status": "error"}

    def update_evolutionary_transformer_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата эволюционного трансформера."""
        try:
            self.evolutionary_transformer_result = result
            logger.debug(
                f"Updated evolutionary transformer result for {self.symbol}: evolution_score={result.get('evolution_score', 0.0):.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating evolutionary transformer result: {e}")

    def get_evolutionary_transformer_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата эволюционного трансформера."""
        return self.evolutionary_transformer_result

    def apply_pattern_discovery_modifier(self) -> None:
        """Применение модификатора обнаружения паттернов."""
        try:
            if not self.pattern_discovery_result:
                return

            # Применяем модификаторы на основе обнаруженного паттерна
            pattern_type = self.pattern_discovery_result.pattern_type
            confidence = self.pattern_discovery_result.confidence
            support = self.pattern_discovery_result.support

            # Базовые модификаторы на основе уверенности
            if confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.15
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе поддержки паттерна
            if support > 0.3:
                # Высокая поддержка - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05

            elif support < 0.1:
                # Низкая поддержка - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.85

            # Специфичные модификаторы для разных типов паттернов
            if pattern_type == "candle":
                # Свечные паттерны - умеренная агрессивность
                self.strategy_modifiers.confidence_multiplier *= 1.05

            elif pattern_type == "price":
                # Ценовые паттерны - высокая агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif pattern_type == "volume":
                # Объемные паттерны - осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.85
                self.strategy_modifiers.risk_multiplier *= 1.2

            # Модификаторы на основе метаданных паттерна
            metadata = self.pattern_discovery_result.metadata
            if metadata:
                trend = metadata.get("trend")
                if trend == "up":
                    # Восходящий тренд - увеличиваем агрессивность
                    self.strategy_modifiers.order_aggressiveness *= 1.05
                    self.strategy_modifiers.position_size_multiplier *= 1.02

                elif trend == "down":
                    # Нисходящий тренд - снижаем агрессивность
                    self.strategy_modifiers.order_aggressiveness *= 0.95
                    self.strategy_modifiers.position_size_multiplier *= 0.98

                # Модификаторы на основе технических индикаторов
                technical_indicators = metadata.get("technical_indicators", {})
                if technical_indicators:
                    rsi = technical_indicators.get("RSI", 50)
                    if rsi > 70:
                        # Перекупленность - осторожность
                        self.strategy_modifiers.order_aggressiveness *= 0.8
                        self.strategy_modifiers.risk_multiplier *= 1.3

                    elif rsi < 30:
                        # Перепроданность - умеренная агрессивность
                        self.strategy_modifiers.order_aggressiveness *= 1.05
                        self.strategy_modifiers.position_size_multiplier *= 1.02

            logger.debug(
                f"Applied pattern discovery modifier: type={pattern_type}, confidence={confidence:.3f}, support={support:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying pattern discovery modifier: {e}")

    def get_pattern_discovery_status(self) -> Dict[str, Any]:
        """Получение статуса обнаружения паттернов."""
        try:
            if not self.pattern_discovery_result:
                return {"pattern_detected": False, "status": "unknown"}

            return {
                "pattern_detected": True,
                "pattern_type": self.pattern_discovery_result.pattern_type,
                "confidence": self.pattern_discovery_result.confidence,
                "support": self.pattern_discovery_result.support,
                "start_idx": self.pattern_discovery_result.start_idx,
                "end_idx": self.pattern_discovery_result.end_idx,
                "trend": self.pattern_discovery_result.trend,
                "volume_profile": self.pattern_discovery_result.volume_profile,
                "technical_indicators": self.pattern_discovery_result.technical_indicators,
                "status": (
                    "high_confidence"
                    if self.pattern_discovery_result.confidence > 0.8
                    else "moderate_confidence"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting pattern discovery status: {e}")
            return {"pattern_detected": False, "status": "error"}

    def update_pattern_discovery_result(self, result: Pattern) -> None:
        """Обновление результата обнаружения паттернов."""
        try:
            self.pattern_discovery_result = result
            logger.debug(
                f"Updated pattern discovery result for {self.symbol}: type={result.pattern_type}, confidence={result.confidence:.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating pattern discovery result: {e}")

    def get_pattern_discovery_result(self) -> Optional[Pattern]:
        """Получение результата обнаружения паттернов."""
        return self.pattern_discovery_result

    def apply_meta_learning_modifier(self) -> None:
        """Применение модификатора мета-обучения."""
        try:
            if not self.meta_learning_result:
                return

            # Применяем модификаторы на основе результатов мета-обучения
            confidence = self.meta_learning_result.get("confidence", 1.0)
            adaptation_score = self.meta_learning_result.get("adaptation_score", 1.0)
            feature_importance = self.meta_learning_result.get("feature_importance", {})
            meta_boost = self.meta_learning_result.get("meta_boost", 1.0)
            risk_adjustment = self.meta_learning_result.get("risk_adjustment", 1.0)

            # Модификаторы уверенности и агрессивности
            if confidence > 0.8:
                self.strategy_modifiers.confidence_multiplier *= 1.15 * meta_boost
                self.strategy_modifiers.order_aggressiveness *= 1.1 * adaptation_score
            elif confidence < 0.5:
                self.strategy_modifiers.confidence_multiplier *= 0.85
                self.strategy_modifiers.order_aggressiveness *= 0.9

            # Модификаторы на основе важности признаков
            if feature_importance:
                important_features = [
                    k for k, v in feature_importance.items() if v > 0.1
                ]
                if "volatility" in important_features:
                    self.strategy_modifiers.risk_multiplier *= 1.2
                if "momentum" in important_features:
                    self.strategy_modifiers.momentum_enabled = True
                if "volume_ratio" in important_features:
                    self.strategy_modifiers.position_size_multiplier *= 1.05

            # Модификаторы риска
            self.strategy_modifiers.risk_multiplier *= risk_adjustment

            # Интеграция с PatternPredictionContext
            if hasattr(self, "pattern_prediction") and self.pattern_prediction:
                self.pattern_prediction.pattern_confidence_boost *= (
                    confidence * meta_boost
                )
                self.pattern_prediction.pattern_risk_adjustment *= risk_adjustment

            logger.debug(
                f"Applied meta learning modifier: confidence={confidence:.3f}, adaptation_score={adaptation_score:.3f}, meta_boost={meta_boost:.3f}"
            )
        except Exception as e:
            logger.error(f"Error applying meta learning modifier: {e}")

    def apply_whale_analysis_modifier(self) -> None:
        """Применение модификатора анализа китов."""
        try:
            if not self.whale_analysis_result:
                return

            # Применяем модификаторы на основе анализа китов
            whale_activity = self.whale_analysis_result.get("whale_activity", 0.0)
            whale_confidence = self.whale_analysis_result.get("whale_confidence", 0.0)
            whale_impact = self.whale_analysis_result.get("whale_impact", 0.0)
            whale_volume = self.whale_analysis_result.get("whale_volume", 0.0)
            whale_activity_type = self.whale_analysis_result.get("whale_activity_type", "unknown")

            if whale_activity > 0.7:
                # Высокая активность китов - осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.risk_multiplier *= 1.3

            # Модификаторы на основе типа активности
            if whale_activity_type == "accumulation":
                # Накопление - увеличиваем готовность к покупке
                self.strategy_modifiers.order_aggressiveness *= 1.1
            elif whale_activity_type == "distribution":
                # Распределение - осторожность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7

            # Модификаторы на основе объема
            if whale_volume > 0.8:
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 1.5)

            logger.debug(
                f"Applied whale analysis modifier: activity={whale_activity:.3f}, type={whale_activity_type}"
            )

        except Exception as e:
            logger.error(f"Error applying whale analysis modifier: {e}")

    def apply_risk_analysis_modifier(self) -> None:
        """Применение модификаторов анализа рисков."""
        if not self.risk_analysis_result:
            return

        try:
            # Получаем данные анализа рисков
            var_95 = self.risk_analysis_result.get("var_95", 0.0)
            self.risk_analysis_result.get("var_99", 0.0)
            max_drawdown = self.risk_analysis_result.get("max_drawdown", 0.0)
            kelly_criterion = self.risk_analysis_result.get("kelly_criterion", 0.0)
            volatility = self.risk_analysis_result.get("volatility", 0.0)
            exposure_level = self.risk_analysis_result.get("exposure_level", 0.0)
            confidence_score = self.risk_analysis_result.get("confidence_score", 0.0)
            risk_limits_ok = self.risk_analysis_result.get("risk_limits_ok", True)
            position_size_multiplier = self.risk_analysis_result.get(
                "position_size_multiplier", 1.0
            )
            leverage_multiplier = self.risk_analysis_result.get(
                "leverage_multiplier", 1.0
            )
            confidence_multiplier = self.risk_analysis_result.get(
                "confidence_multiplier", 1.0
            )

            # Применяем модификаторы к стратегии
            self.strategy_modifiers.confidence_multiplier *= confidence_multiplier
            self.strategy_modifiers.position_size_multiplier *= position_size_multiplier
            self.strategy_modifiers.risk_multiplier *= leverage_multiplier

            # Дополнительные модификаторы на основе метрик риска
            if not risk_limits_ok:
                # Превышены лимиты риска - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.5
                self.strategy_modifiers.confidence_multiplier *= 0.6

            # Модификаторы на основе VaR
            if var_95 > 0.05:  # Высокий VaR
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8
            elif var_95 < 0.02:  # Низкий VaR
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.05

            # Модификаторы на основе максимальной просадки
            if max_drawdown > 0.1:  # Высокая просадка
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе критерия Келли
            if kelly_criterion > 0.5:  # Высокий критерий Келли
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.05
            elif kelly_criterion < 0.2:  # Низкий критерий Келли
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе волатильности
            if volatility > 0.05:  # Высокая волатильность
                self.strategy_modifiers.risk_multiplier *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8

            # Модификаторы на основе уровня экспозиции
            if exposure_level > 0.8:  # Высокая экспозиция
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6

            # Модификаторы на основе оценки уверенности
            if confidence_score < 0.5:  # Низкая уверенность
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.order_aggressiveness *= 0.8

            # Устанавливаем флаги
            self.set("risk_analysis_applied", True)
            self.set("risk_limits_ok", risk_limits_ok)
            self.set("var_95", var_95)
            self.set("max_drawdown", max_drawdown)
            self.set("kelly_criterion", kelly_criterion)

            logger.debug(
                f"Applied risk analysis modifiers: var_95={var_95:.3f}, "
                f"max_drawdown={max_drawdown:.3f}, kelly={kelly_criterion:.3f}, "
                f"risk_limits_ok={risk_limits_ok}"
            )

        except Exception as e:
            logger.error(f"Error applying risk analysis modifier: {e}")

    def apply_portfolio_analysis_modifier(self) -> None:
        """Применение модификаторов анализа портфеля."""
        if not self.portfolio_analysis_result:
            return

        try:
            # Получаем данные анализа портфеля
            self.portfolio_analysis_result.get("portfolio_weights", {})
            asset_metrics = self.portfolio_analysis_result.get("asset_metrics", {})
            self.portfolio_analysis_result.get("portfolio_metrics", {})
            self.portfolio_analysis_result.get("correlation_matrix", {})
            btc_dominance = self.portfolio_analysis_result.get("btc_dominance", 0.0)
            market_regime = self.portfolio_analysis_result.get(
                "market_regime", "SIDEWAYS"
            )
            regime_confidence = self.portfolio_analysis_result.get(
                "regime_confidence", 0.0
            )
            self.portfolio_analysis_result.get("suggested_trades", [])
            self.portfolio_analysis_result.get("portfolio_state", {})

            # Получаем метрики текущего актива
            current_asset_metrics = asset_metrics.get(self.symbol, {})
            expected_return = current_asset_metrics.get("expected_return", 0.0)
            risk_score = current_asset_metrics.get("risk_score", 1.0)
            liquidity_score = current_asset_metrics.get("liquidity_score", 0.5)
            trend_strength = current_asset_metrics.get("trend_strength", 0.0)
            volume_score = current_asset_metrics.get("volume_score", 0.5)
            correlation_with_btc = current_asset_metrics.get(
                "correlation_with_btc", 0.0
            )
            current_weight = current_asset_metrics.get("current_weight", 0.0)
            target_weight = current_asset_metrics.get("target_weight", 0.0)

            # Применяем модификаторы к стратегии
            # Модификаторы на основе ожидаемой доходности
            if expected_return > 0.05:  # Высокая ожидаемая доходность
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.order_aggressiveness *= 1.05
            elif expected_return < -0.02:  # Отрицательная ожидаемая доходность
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.order_aggressiveness *= 0.7

            # Модификаторы на основе оценки риска
            if risk_score > 0.8:  # Высокий риск
                self.strategy_modifiers.risk_multiplier *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
            elif risk_score < 0.3:  # Низкий риск
                self.strategy_modifiers.risk_multiplier *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05

            # Модификаторы на основе ликвидности
            if liquidity_score > 0.8:  # Высокая ликвидность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms - 50
                )
            elif liquidity_score < 0.3:  # Низкая ликвидность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.execution_delay_ms += 100

            # Модификаторы на основе силы тренда
            if trend_strength > 0.7:  # Сильный тренд
                self.strategy_modifiers.confidence_multiplier *= 1.15
                self.strategy_modifiers.order_aggressiveness *= 1.1
            elif trend_strength < 0.2:  # Слабый тренд
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.order_aggressiveness *= 0.9

            # Модификаторы на основе объема
            if volume_score > 0.8:  # Высокий объем
                self.strategy_modifiers.order_aggressiveness *= 1.05
                self.strategy_modifiers.position_size_multiplier *= 1.1
            elif volume_score < 0.3:  # Низкий объем
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8

            # Модификаторы на основе корреляции с BTC
            if abs(correlation_with_btc) > 0.8:  # Высокая корреляция с BTC
                if btc_dominance > 0.5:  # BTC доминирует
                    self.strategy_modifiers.confidence_multiplier *= 1.1
                else:  # BTC не доминирует
                    self.strategy_modifiers.confidence_multiplier *= 0.9

            # Модификаторы на основе веса в портфеле
            weight_diff = target_weight - current_weight
            if abs(weight_diff) > 0.1:  # Большая разница в весах
                if weight_diff > 0:  # Нужно увеличить позицию
                    self.strategy_modifiers.position_size_multiplier *= 1.2
                    self.strategy_modifiers.order_aggressiveness *= 1.1
                else:  # Нужно уменьшить позицию
                    self.strategy_modifiers.position_size_multiplier *= 0.8
                    self.strategy_modifiers.order_aggressiveness *= 0.9

            # Модификаторы на основе рыночного режима
            if market_regime == "BULL":
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.order_aggressiveness *= 1.05
            elif market_regime == "BEAR":
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.risk_multiplier *= 0.9

            # Модификаторы на основе уверенности в режиме
            if regime_confidence > 0.8:  # Высокая уверенность в режиме
                self.strategy_modifiers.confidence_multiplier *= 1.05
            elif regime_confidence < 0.4:  # Низкая уверенность в режиме
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Устанавливаем флаги
            self.set("portfolio_analysis_applied", True)
            self.set("expected_return", expected_return)
            self.set("risk_score", risk_score)
            self.set("liquidity_score", liquidity_score)
            self.set("trend_strength", trend_strength)
            self.set("volume_score", volume_score)
            self.set("correlation_with_btc", correlation_with_btc)
            self.set("current_weight", current_weight)
            self.set("target_weight", target_weight)
            self.set("market_regime", market_regime)
            self.set("regime_confidence", regime_confidence)
            self.set("btc_dominance", btc_dominance)

            logger.debug(
                f"Applied portfolio analysis modifiers: expected_return={expected_return:.3f}, "
                f"risk_score={risk_score:.3f}, liquidity_score={liquidity_score:.3f}, "
                f"trend_strength={trend_strength:.3f}, market_regime={market_regime}, "
                f"regime_confidence={regime_confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying portfolio analysis modifier: {e}")

    def get_whale_analysis_status(self) -> Dict[str, Any]:
        """Получение статуса анализа активности китов."""
        try:
            if not self.whale_analysis_result:
                return {"whale_activity": False, "status": "unknown"}

            return {
                "whale_activity": self.whale_analysis_result.get(
                    "whale_activity", False
                ),
                "confidence": self.whale_analysis_result.get("confidence", 0.0),
                "impact": self.whale_analysis_result.get("impact", 0.0),
                "volume": self.whale_analysis_result.get("volume", 0.0),
                "activity_type": self.whale_analysis_result.get(
                    "activity_type", "unknown"
                ),
                "status": (
                    "high_activity"
                    if self.whale_analysis_result.get("whale_activity", False)
                    else "normal_activity"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting whale analysis status: {e}")
            return {"whale_activity": False, "status": "error"}

    def update_whale_analysis_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата анализа активности китов."""
        try:
            self.whale_analysis_result = result
            logger.debug(
                f"Updated whale analysis result for {self.symbol}: activity={result.get('whale_activity', False)}, confidence={result.get('confidence', 0.0):.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating whale analysis result: {e}")

    def get_whale_analysis_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата анализа активности китов."""
        return self.whale_analysis_result

    def apply_meta_controller_modifier(self) -> None:
        """Применение модификаторов мета-контроллера агентов."""
        if not self.meta_controller_result:
            return
        try:
            coordination_score = self.meta_controller_result.get(
                "coordination_score", 1.0
            )
            agent_priority = self.meta_controller_result.get("agent_priority", 1.0)
            meta_strategy_boost = self.meta_controller_result.get(
                "meta_strategy_boost", 1.0
            )
            # Применяем модификаторы
            self.strategy_modifiers.confidence_multiplier *= coordination_score
            self.strategy_modifiers.position_size_multiplier *= agent_priority
            self.strategy_modifiers.meta_controller_boost *= meta_strategy_boost
            self.set("meta_controller_applied", True)
            logger.debug(
                f"Applied meta controller modifier: coordination={coordination_score}, priority={agent_priority}, boost={meta_strategy_boost}"
            )
        except Exception as e:
            logger.error(f"Error applying meta controller modifier: {e}")

    def apply_genetic_optimization_modifier(self) -> None:
        """Применение модификаторов генетической оптимизации."""
        if not self.genetic_optimization_result:
            return
        try:
            # Получаем результаты генетической оптимизации
            best_fitness = self.genetic_optimization_result.get("best_fitness", 0.0)
            best_parameters = self.genetic_optimization_result.get(
                "best_parameters", {}
            )
            optimization_type = self.genetic_optimization_result.get(
                "parameter_type", "unknown"
            )
            confidence = self.genetic_optimization_result.get("confidence", 1.0)
            adaptation_score = self.genetic_optimization_result.get(
                "adaptation_score", 1.0
            )

            # Применяем модификаторы на основе фитнеса
            if best_fitness > 0.8:
                # Высокий фитнес - увеличиваем агрессивность
                self.strategy_modifiers.confidence_multiplier *= 1.15
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
            elif best_fitness < 0.4:
                # Низкий фитнес - снижаем агрессивность
                self.strategy_modifiers.confidence_multiplier *= 0.85
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.9

            # Применяем модификаторы на основе типа оптимизации
            if optimization_type == "strategy_optimization":
                # Оптимизация стратегии - применяем параметры
                risk_tolerance = best_parameters.get("risk_tolerance", 0.5)
                profit_target = best_parameters.get("profit_target", 0.02)
                stop_loss = best_parameters.get("stop_loss", 0.01)
                position_size = best_parameters.get("position_size", 0.1)
                best_parameters.get("max_positions", 5)

                # Модификаторы на основе риска
                self.strategy_modifiers.risk_multiplier *= 1.0 + risk_tolerance * 0.2

                # Модификаторы на основе целей
                if profit_target > 0.03:
                    self.strategy_modifiers.confidence_multiplier *= 1.1
                if stop_loss < 0.008:
                    self.strategy_modifiers.risk_multiplier *= 0.9

                # Модификаторы на основе размера позиции
                self.strategy_modifiers.position_size_multiplier *= position_size * 10

            elif optimization_type == "system_configuration":
                # Оптимизация конфигурации системы
                confidence_threshold = best_parameters.get("confidence_threshold", 0.8)
                improvement_threshold = best_parameters.get(
                    "improvement_threshold", 0.05
                )

                # Модификаторы на основе порогов
                if confidence_threshold > 0.9:
                    self.strategy_modifiers.confidence_multiplier *= 1.1
                if improvement_threshold < 0.03:
                    self.strategy_modifiers.order_aggressiveness *= 1.05

            # Применяем модификаторы на основе уверенности
            self.strategy_modifiers.confidence_multiplier *= confidence

            # Применяем модификаторы на основе адаптации
            self.strategy_modifiers.order_aggressiveness *= adaptation_score

            # Устанавливаем флаги
            self.set("genetic_optimization_applied", True)
            self.set("optimization_type", optimization_type)
            self.set("best_fitness", best_fitness)
            self.set("optimization_confidence", confidence)

            logger.debug(
                f"Applied genetic optimization modifier: type={optimization_type}, "
                f"fitness={best_fitness:.3f}, confidence={confidence:.3f}, "
                f"adaptation_score={adaptation_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying genetic optimization modifier: {e}")

    def get_genetic_optimization_status(self) -> Dict[str, Any]:
        """Получение статуса генетической оптимизации."""
        try:
            if not self.genetic_optimization_result:
                return {"optimization_applied": False, "status": "unknown"}

            return {
                "optimization_applied": True,
                "parameter_type": self.genetic_optimization_result.get(
                    "parameter_type", "unknown"
                ),
                "best_fitness": self.genetic_optimization_result.get(
                    "best_fitness", 0.0
                ),
                "best_parameters": self.genetic_optimization_result.get(
                    "best_parameters", {}
                ),
                "confidence": self.genetic_optimization_result.get("confidence", 0.0),
                "adaptation_score": self.genetic_optimization_result.get(
                    "adaptation_score", 0.0
                ),
                "status": (
                    "high_fitness"
                    if self.genetic_optimization_result.get("best_fitness", 0.0) > 0.8
                    else "moderate_fitness"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting genetic optimization status: {e}")
            return {"optimization_applied": False, "status": "error"}

    def update_genetic_optimization_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата генетической оптимизации."""
        try:
            self.genetic_optimization_result = result
            logger.debug(
                f"Updated genetic optimization result for {self.symbol}: type={result.get('parameter_type', 'unknown')}, fitness={result.get('best_fitness', 0.0):.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating genetic optimization result: {e}")

    def get_genetic_optimization_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата генетической оптимизации."""
        return self.genetic_optimization_result

    def apply_dataset_manager_modifier(self) -> None:
        """Применение модификаторов менеджера датасетов."""
        try:
            if not self.dataset_manager_result:
                return

            # Получаем данные менеджера датасетов
            dataset_quality = self.dataset_manager_result.get("dataset_quality", 0.5)
            dataset_size = self.dataset_manager_result.get("dataset_size", 0)
            win_rate = self.dataset_manager_result.get("win_rate", 0.5)
            avg_pnl = self.dataset_manager_result.get("avg_pnl", 0.0)
            avg_drawdown = self.dataset_manager_result.get("avg_drawdown", 0.0)
            data_freshness = self.dataset_manager_result.get("data_freshness", 0.5)
            pattern_diversity = self.dataset_manager_result.get(
                "pattern_diversity", 0.5
            )
            confidence_score = self.dataset_manager_result.get("confidence_score", 0.5)

            # Модификаторы на основе качества датасета
            if dataset_quality > 0.8:
                # Высокое качество данных - увеличиваем уверенность
                self.strategy_modifiers.dataset_manager_confidence_multiplier = 1.15
                self.strategy_modifiers.dataset_manager_strength_multiplier = 1.1
                self.strategy_modifiers.dataset_manager_data_quality_boost = 1.2
            elif dataset_quality < 0.3:
                # Низкое качество данных - снижаем уверенность
                self.strategy_modifiers.dataset_manager_confidence_multiplier = 0.8
                self.strategy_modifiers.dataset_manager_strength_multiplier = 0.7
                self.strategy_modifiers.dataset_manager_data_quality_boost = 0.8

            # Модификаторы на основе размера датасета
            if dataset_size > 1000:
                # Большой датасет - высокая уверенность
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.order_aggressiveness *= 1.05
            elif dataset_size < 100:
                # Маленький датасет - осторожность
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.order_aggressiveness *= 0.7

            # Модификаторы на основе винрейта
            if win_rate > 0.6:
                # Высокий винрейт - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
            elif win_rate < 0.4:
                # Низкий винрейт - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7

            # Модификаторы на основе средней прибыли
            if avg_pnl > 0.02:
                # Высокая прибыльность - увеличиваем размер позиции
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.05
            elif avg_pnl < -0.01:
                # Убыточность - снижаем размер позиции
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе средней просадки
            if avg_drawdown > 0.05:
                # Высокая просадка - увеличиваем осторожность
                self.strategy_modifiers.risk_multiplier *= 1.3
                self.strategy_modifiers.order_aggressiveness *= 0.8
            elif avg_drawdown < 0.01:
                # Низкая просадка - умеренная агрессивность
                self.strategy_modifiers.risk_multiplier *= 0.9
                self.strategy_modifiers.order_aggressiveness *= 1.05

            # Модификаторы на основе свежести данных
            if data_freshness > 0.8:
                # Свежие данные - быстрая реакция
                self.strategy_modifiers.dataset_manager_execution_delay_ms = 0
                self.strategy_modifiers.order_aggressiveness *= 1.05
            elif data_freshness < 0.3:
                # Устаревшие данные - медленная реакция
                self.strategy_modifiers.dataset_manager_execution_delay_ms = 200
                self.strategy_modifiers.order_aggressiveness *= 0.9

            # Модификаторы на основе разнообразия паттернов
            if pattern_diversity > 0.7:
                # Высокое разнообразие - адаптивность
                self.strategy_modifiers.confidence_multiplier *= 1.05
                self.strategy_modifiers.order_aggressiveness *= 1.02
            elif pattern_diversity < 0.3:
                # Низкое разнообразие - осторожность
                self.strategy_modifiers.confidence_multiplier *= 0.9
                self.strategy_modifiers.order_aggressiveness *= 0.85

            # Модификаторы на основе оценки уверенности
            if confidence_score > 0.8:
                # Высокая уверенность в данных
                self.strategy_modifiers.confidence_multiplier *= 1.1
                self.strategy_modifiers.order_aggressiveness *= 1.05
            elif confidence_score < 0.4:
                # Низкая уверенность в данных
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.order_aggressiveness *= 0.8

            # Устанавливаем флаги
            self.set("dataset_manager_applied", True)
            self.set("dataset_quality", dataset_quality)
            self.set("dataset_size", dataset_size)
            self.set("win_rate", win_rate)
            self.set("avg_pnl", avg_pnl)
            self.set("avg_drawdown", avg_drawdown)
            self.set("data_freshness", data_freshness)
            self.set("pattern_diversity", pattern_diversity)
            self.set("confidence_score", confidence_score)

            logger.debug(
                f"Applied dataset manager modifier: quality={dataset_quality:.3f}, "
                f"size={dataset_size}, win_rate={win_rate:.3f}, avg_pnl={avg_pnl:.3f}, "
                f"freshness={data_freshness:.3f}, diversity={pattern_diversity:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying dataset manager modifier: {e}")

    def get_dataset_manager_status(self) -> Dict[str, Any]:
        """Получение статуса менеджера датасетов."""
        try:
            if not self.dataset_manager_result:
                return {"dataset_available": False, "status": "unknown"}

            return {
                "dataset_available": True,
                "dataset_quality": self.dataset_manager_result.get(
                    "dataset_quality", 0.0
                ),
                "dataset_size": self.dataset_manager_result.get("dataset_size", 0),
                "win_rate": self.dataset_manager_result.get("win_rate", 0.0),
                "avg_pnl": self.dataset_manager_result.get("avg_pnl", 0.0),
                "avg_drawdown": self.dataset_manager_result.get("avg_drawdown", 0.0),
                "data_freshness": self.dataset_manager_result.get(
                    "data_freshness", 0.0
                ),
                "pattern_diversity": self.dataset_manager_result.get(
                    "pattern_diversity", 0.0
                ),
                "confidence_score": self.dataset_manager_result.get(
                    "confidence_score", 0.0
                ),
                "status": (
                    "high_quality"
                    if self.dataset_manager_result.get("dataset_quality", 0.0) > 0.8
                    else "moderate_quality"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting dataset manager status: {e}")
            return {"dataset_available": False, "status": "error"}

    def update_dataset_manager_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата менеджера датасетов."""
        try:
            self.dataset_manager_result = result
            logger.debug(
                f"Updated dataset manager result for {self.symbol}: quality={result.get('dataset_quality', 0.0):.3f}, size={result.get('dataset_size', 0)}"
            )
        except Exception as e:
            logger.error(f"Error updating dataset manager result: {e}")

    def get_dataset_manager_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата менеджера датасетов."""
        return self.dataset_manager_result

    def apply_evolvable_decision_reasoner_modifier(self) -> None:
        """Применение модификатора эволюционного анализа решений."""
        try:
            if not self.evolvable_decision_reasoner_result:
                return

            # Применяем модификаторы на основе результатов эволюционного анализа решений
            decision_confidence = self.evolvable_decision_reasoner_result.get(
                "decision_confidence", 0.0
            )
            decision_strength = self.evolvable_decision_reasoner_result.get(
                "decision_strength", 0.0
            )
            decision_execution_delay_ms = self.evolvable_decision_reasoner_result.get(
                "decision_execution_delay_ms", 0
            )

            # Модификаторы на основе уверенности в решении
            if decision_confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.15
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif decision_confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе силы решения
            if decision_strength > 0.8:
                # Сильное решение - усиливаем модификаторы
                self.strategy_modifiers.confidence_multiplier *= 1.2
            elif decision_strength < 0.3:
                # Слабое решение - ослабляем модификаторы
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе задержки исполнения
            if decision_execution_delay_ms > 0:
                # Задержка исполнения - увеличиваем осторожность
                self.strategy_modifiers.execution_delay_ms = max(
                    0,
                    self.strategy_modifiers.execution_delay_ms
                    + decision_execution_delay_ms,
                )

            logger.debug(
                f"Applied evolvable decision reasoner modifier: confidence={decision_confidence:.3f}, strength={decision_strength:.3f}, delay={decision_execution_delay_ms}ms"
            )

        except Exception as e:
            logger.error(f"Error applying evolvable decision reasoner modifier: {e}")

    def get_evolvable_decision_reasoner_status(self) -> Dict[str, Any]:
        """Получение статуса эволюционного анализа решений."""
        try:
            if not self.evolvable_decision_reasoner_result:
                return {"decision_active": False, "status": "unknown"}

            return {
                "decision_active": True,
                "decision_confidence": self.evolvable_decision_reasoner_result.get(
                    "decision_confidence", 0.0
                ),
                "decision_strength": self.evolvable_decision_reasoner_result.get(
                    "decision_strength", 0.0
                ),
                "decision_execution_delay_ms": self.evolvable_decision_reasoner_result.get(
                    "decision_execution_delay_ms", 0
                ),
                "status": (
                    "high_decision"
                    if self.evolvable_decision_reasoner_result.get(
                        "decision_confidence", 0.0
                    )
                    > 0.8
                    else "moderate_decision"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting evolvable decision reasoner status: {e}")
            return {"decision_active": False, "status": "error"}

    def update_evolvable_decision_reasoner_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата эволюционного анализа решений."""
        try:
            self.evolvable_decision_reasoner_result = result
            logger.debug(
                f"Updated evolvable decision reasoner result for {self.symbol}: confidence={result.get('decision_confidence', 0.0):.3f}, strength={result.get('decision_strength', 0.0):.3f}"
            )
        except Exception as e:
            logger.error(f"Error updating evolvable decision reasoner result: {e}")

    def get_evolvable_decision_reasoner_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата эволюционного анализа решений."""
        return self.evolvable_decision_reasoner_result

    def apply_regime_discovery_modifier(self) -> None:
        """Применение модификатора обнаружения режима."""
        try:
            if not self.regime_discovery_result:
                return

            # Применяем модификаторы на основе результатов обнаружения режима
            regime_confidence = self.regime_discovery_result.get(
                "regime_confidence", 0.0
            )
            regime_type = self.regime_discovery_result.get("regime_type", "unknown")

            # Модификаторы уверенности и агрессивности
            if regime_confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.1

            elif regime_confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе типа режима
            if regime_type == "bullish":
                # Бычий режим - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.risk_multiplier *= 1.1

            elif regime_type == "bearish":
                # Медвежий режим - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.risk_multiplier *= 0.9

            # Обновляем контекст рынка
            if self.market_context:
                self.market_context.regime_shift = regime_type != "sideways"
                self.market_context.gravity_bias = (
                    self.strategy_modifiers.order_aggressiveness
                )

            logger.debug(
                f"Applied regime discovery modifier: confidence={regime_confidence:.3f}, regime_type={regime_type}"
            )

        except Exception as e:
            logger.error(f"Error applying regime discovery modifier: {e}")

    def get_regime_discovery_status(self) -> Dict[str, Any]:
        """Получение статуса обнаружения режима."""
        try:
            if not self.regime_discovery_result:
                return {"regime_detected": False, "status": "unknown"}

            return {
                "regime_detected": True,
                "regime_confidence": self.regime_discovery_result.get(
                    "regime_confidence", 0.0
                ),
                "regime_type": self.regime_discovery_result.get(
                    "regime_type", "unknown"
                ),
                "status": (
                    "high_confidence"
                    if self.regime_discovery_result.get("regime_confidence", 0.0) > 0.8
                    else "moderate_confidence"
                ),
            }

        except Exception as e:
            logger.error(f"Error getting regime discovery status: {e}")
            return {"regime_detected": False, "status": "error"}

    def update_regime_discovery_result(self, result: Dict[str, Any]) -> None:
        """Обновление результата обнаружения режима."""
        try:
            self.regime_discovery_result = result
            logger.debug(
                f"Updated regime discovery result for {self.symbol}: confidence={result.get('regime_confidence', 0.0):.3f}, regime_type={result.get('regime_type', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"Error updating regime discovery result: {e}")

    def get_regime_discovery_result(self) -> Optional[Dict[str, Any]]:
        """Получение результата обнаружения режима."""
        return self.regime_discovery_result

    def apply_local_ai_controller_modifier(self) -> None:
        """Применение модификаторов LocalAIController."""
        if self.local_ai_controller_result:
            result = self.local_ai_controller_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов LocalAIController
            if "confidence_boost" in result:
                modifiers.confidence_multiplier *= result["confidence_boost"]
            if "aggressiveness_adjustment" in result:
                modifiers.order_aggressiveness *= result["aggressiveness_adjustment"]
            if "risk_adjustment" in result:
                modifiers.risk_multiplier *= result["risk_adjustment"]

    def apply_analytical_integration_modifier(self) -> None:
        """Применение модификаторов AnalyticalIntegration."""
        if self.analytical_integration_result:
            result = self.analytical_integration_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов аналитической интеграции
            if "trading_recommendations" in result:
                recommendations = result["trading_recommendations"]
                if "aggressiveness_adjustment" in recommendations:
                    modifiers.order_aggressiveness *= recommendations[
                        "aggressiveness_adjustment"
                    ]
                if "position_size_adjustment" in recommendations:
                    modifiers.position_size_multiplier *= recommendations[
                        "position_size_adjustment"
                    ]
                if "confidence_adjustment" in recommendations:
                    modifiers.confidence_multiplier *= recommendations[
                        "confidence_adjustment"
                    ]
                if "price_offset_adjustment" in recommendations:
                    modifiers.price_offset_percent += recommendations[
                        "price_offset_adjustment"
                    ]
                if "risk_adjustment" in recommendations:
                    modifiers.risk_multiplier *= recommendations["risk_adjustment"]

    def apply_entanglement_integration_modifier(self) -> None:
        """Применение модификаторов EntanglementIntegration."""
        if self.entanglement_integration_result:
            result = self.entanglement_integration_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов интеграции запутанности
            if "entanglement_score" in result:
                modifiers.entanglement_confidence_multiplier = result[
                    "entanglement_score"
                ]
            if "entanglement_strength" in result:
                modifiers.entanglement_strength_multiplier = result[
                    "entanglement_strength"
                ]
            if "entanglement_execution_delay_ms" in result:
                modifiers.entanglement_execution_delay_ms = result[
                    "entanglement_execution_delay_ms"
                ]
            # Убираем обращение к несуществующему атрибуту entanglement_risk_multiplier

    def apply_agent_order_executor_modifier(self) -> None:
        """Применение модификаторов AgentOrderExecutor."""
        if self.agent_order_executor_result:
            result = self.agent_order_executor_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов AgentOrderExecutor
            if "order_aggressiveness_adjustment" in result:
                modifiers.order_aggressiveness *= result[
                    "order_aggressiveness_adjustment"
                ]
            if "position_size_multiplier_adjustment" in result:
                modifiers.position_size_multiplier *= result[
                    "position_size_multiplier_adjustment"
                ]
            if "confidence_multiplier_adjustment" in result:
                modifiers.confidence_multiplier *= result[
                    "confidence_multiplier_adjustment"
                ]
            if "price_offset_percent_adjustment" in result:
                modifiers.price_offset_percent += result[
                    "price_offset_percent_adjustment"
                ]
            if "risk_multiplier_adjustment" in result:
                modifiers.risk_multiplier *= result["risk_multiplier_adjustment"]

    def apply_agent_market_regime_modifier(self) -> None:
        """Применение модификаторов AgentMarketRegime."""
        if self.agent_market_regime_result:
            result = self.agent_market_regime_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов AgentMarketRegime
            if "regime_type" in result:
                regime_type = result["regime_type"]
                if regime_type == "TREND":
                    modifiers.order_aggressiveness *= 1.1
                    modifiers.position_size_multiplier *= 1.05
                    modifiers.confidence_multiplier *= 1.1
                elif regime_type == "SIDEWAYS":
                    modifiers.order_aggressiveness *= 0.9
                    modifiers.position_size_multiplier *= 0.8
                    modifiers.confidence_multiplier *= 0.9
                elif regime_type == "REVERSAL":
                    modifiers.order_aggressiveness *= 0.7
                    modifiers.position_size_multiplier *= 0.6
                    modifiers.confidence_multiplier *= 0.8
                elif regime_type == "MANIPULATION":
                    modifiers.order_aggressiveness *= 0.5
                    modifiers.position_size_multiplier *= 0.4
                    modifiers.confidence_multiplier *= 0.6
                    modifiers.risk_multiplier *= 1.5
                elif regime_type == "VOLATILITY":
                    modifiers.order_aggressiveness *= 0.8
                    modifiers.position_size_multiplier *= 0.7
                    modifiers.confidence_multiplier *= 0.85
                    modifiers.price_offset_percent *= 1.2
                elif regime_type == "ANOMALY":
                    modifiers.order_aggressiveness *= 0.3
                    modifiers.position_size_multiplier *= 0.2
                    modifiers.confidence_multiplier *= 0.5
                    modifiers.risk_multiplier *= 2.0

            if "regime_confidence" in result:
                confidence = result["regime_confidence"]
                if confidence > 0.8:
                    modifiers.confidence_multiplier *= 1.1
                elif confidence < 0.5:
                    modifiers.confidence_multiplier *= 0.8

    def apply_agent_market_maker_model_modifier(self) -> None:
        """Применение модификаторов AgentMarketMakerModel."""
        if self.agent_market_maker_model_result:
            result = self.agent_market_maker_model_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов AgentMarketMakerModel
            if "spread_analysis" in result:
                spread_data = result["spread_analysis"]
                if "confidence" in spread_data:
                    modifiers.confidence_multiplier *= spread_data["confidence"]
                if "imbalance" in spread_data:
                    modifiers.order_aggressiveness *= 1 + spread_data["imbalance"] * 0.1

            if "liquidity_analysis" in result:
                liquidity_data = result["liquidity_analysis"]
                if "zone_strength" in liquidity_data:
                    modifiers.position_size_multiplier *= liquidity_data[
                        "zone_strength"
                    ]
                if "fakeout_probability" in liquidity_data:
                    modifiers.risk_multiplier *= (
                        1 + liquidity_data["fakeout_probability"] * 0.2
                    )

            if "prediction_confidence" in result:
                modifiers.confidence_multiplier *= result["prediction_confidence"]

    def apply_sandbox_trainer_modifier(self) -> None:
        """Применение модификаторов SandboxTrainer."""
        if self.sandbox_trainer_result:
            result = self.sandbox_trainer_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов SandboxTrainer
            if "trainer_confidence" in result:
                modifiers.confidence_multiplier *= result["trainer_confidence"]
            if "trainer_strength" in result:
                modifiers.position_size_multiplier *= result["trainer_strength"]
            if "trainer_risk_multiplier" in result:
                modifiers.risk_multiplier *= result["trainer_risk_multiplier"]

    def apply_model_trainer_modifier(self) -> None:
        """Применение модификаторов ModelTrainer."""
        if self.model_trainer_result:
            result = self.model_trainer_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов ModelTrainer
            if "trainer_confidence" in result:
                modifiers.confidence_multiplier *= result["trainer_confidence"]
            if "trainer_strength" in result:
                modifiers.position_size_multiplier *= result["trainer_strength"]
            if "trainer_risk_multiplier" in result:
                modifiers.risk_multiplier *= result["trainer_risk_multiplier"]

    def apply_window_model_trainer_modifier(self) -> None:
        """Применение модификаторов WindowModelTrainer."""
        if self.window_model_trainer_result:
            result = self.window_model_trainer_result
            modifiers = self.strategy_modifiers

            # Применяем модификаторы на основе результатов WindowModelTrainer
            if "trainer_confidence" in result:
                modifiers.confidence_multiplier *= result["trainer_confidence"]
            if "trainer_strength" in result:
                modifiers.position_size_multiplier *= result["trainer_strength"]
            if "trainer_risk_multiplier" in result:
                modifiers.risk_multiplier *= result["trainer_risk_multiplier"]

    def apply_all_modifiers(
        self, priority_order: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Применение всех модификаторов с приоритизацией и кэшированием.

        Args:
            priority_order: Порядок приоритета модификаторов

        Returns:
            Словарь с метриками производительности
        """
        import time

        start_time = time.time()

        # Приоритетный порядок модификаторов (критические -> важные -> вспомогательные)
        if priority_order is None:
            priority_order = [
                # Критические модификаторы (влияют на безопасность)
                "entanglement_monitor",
                "noise_analyzer",
                "risk_analysis",
                "agent_market_regime",
                # Важные модификаторы (влияют на эффективность)
                "market_pattern",
                "live_adaptation",
                "decision_reasoning",
                "evolutionary_transformer",
                "meta_learning",
                "genetic_optimization",
                # Вспомогательные модификаторы (улучшают точность)
                "mirror_map",
                "session_influence",
                "session_marker",
                "whale_analysis",
                "portfolio_analysis",
                "meta_controller",
                "pattern_discovery",
                # ML модификаторы
                "model_selector",
                "advanced_price_predictor",
                "window_optimizer",
                "state_manager",
                "dataset_manager",
                "evolvable_decision_reasoner",
                "regime_discovery",
                # Дополнительные агенты
                "advanced_market_maker",
                "market_memory_integration",
                "market_memory_whale_integration",
                "local_ai_controller",
                "analytical_integration",
                "entanglement_integration",
                "agent_order_executor",
                "agent_market_maker_model",
                # Инструменты обучения
                "sandbox_trainer",
                "model_trainer",
                "window_model_trainer",
            ]

        # Проверяем кэш
        cache_key = self._generate_modifiers_cache_key()
        if self.get("modifiers_cache_key") == cache_key:
            logger.debug(f"Using cached modifiers for {self.symbol}")
            return {
                "execution_time_ms": 0.0,
                "modifiers_applied": 0,
                "cache_hit": True,
                "priority_levels_processed": 0,
            }

        # Счетчики для метрик
        modifiers_applied = 0
        priority_levels_processed = 0

        # Группируем модификаторы по приоритету
        priority_groups = {
            "critical": [
                "entanglement_monitor",
                "noise_analyzer",
                "risk_analysis",
                "agent_market_regime",
            ],
            "important": [
                "market_pattern",
                "live_adaptation",
                "decision_reasoning",
                "evolutionary_transformer",
                "meta_learning",
                "genetic_optimization",
            ],
            "auxiliary": [
                "mirror_map",
                "session_influence",
                "session_marker",
                "whale_analysis",
                "portfolio_analysis",
                "meta_controller",
                "pattern_discovery",
            ],
            "ml": [
                "model_selector",
                "advanced_price_predictor",
                "window_optimizer",
                "state_manager",
                "dataset_manager",
                "evolvable_decision_reasoner",
                "regime_discovery",
            ],
            "agents": [
                "advanced_market_maker",
                "market_memory_integration",
                "market_memory_whale_integration",
                "local_ai_controller",
                "analytical_integration",
                "entanglement_integration",
                "agent_order_executor",
                "agent_market_maker_model",
            ],
            "training": ["sandbox_trainer", "model_trainer", "window_model_trainer"],
        }

        # Применяем модификаторы в порядке приоритета
        for modifier_name in priority_order:
            try:
                # Определяем группу приоритета
                current_priority = None
                for priority, group in priority_groups.items():
                    if modifier_name in group:
                        current_priority = priority
                        break

                # Применяем модификатор
                if modifier_name == "entanglement_monitor" and self.entanglement_result:
                    self.apply_entanglement_monitor_modifier()
                    modifiers_applied += 1
                elif modifier_name == "noise_analyzer" and self.noise_result:
                    self.apply_noise_analyzer_modifier()
                    modifiers_applied += 1
                elif modifier_name == "risk_analysis" and self.risk_analysis_result:
                    self.apply_risk_analysis_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "agent_market_regime"
                    and self.agent_market_regime_result
                ):
                    self.apply_agent_market_regime_modifier()
                    modifiers_applied += 1
                elif modifier_name == "market_pattern" and self.market_pattern_result:
                    self.apply_market_pattern_modifier()
                    modifiers_applied += 1
                elif modifier_name == "live_adaptation" and self.live_adaptation_result:
                    self.apply_live_adaptation_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "decision_reasoning"
                    and self.decision_reasoning_result
                ):
                    self.apply_decision_reasoning_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "evolutionary_transformer"
                    and self.evolutionary_transformer_result
                ):
                    self.apply_evolutionary_transformer_modifier()
                    modifiers_applied += 1
                elif modifier_name == "meta_learning" and self.meta_learning_result:
                    self.apply_meta_learning_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "genetic_optimization"
                    and self.genetic_optimization_result
                ):
                    self.apply_genetic_optimization_modifier()
                    modifiers_applied += 1
                elif modifier_name == "mirror_map" and self.mirror_map:
                    self.apply_mirror_map_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "session_influence"
                    and self.session_influence_result
                ):
                    self.apply_session_influence_modifier()
                    modifiers_applied += 1
                elif modifier_name == "session_marker" and self.session_marker_result:
                    # Применяем к базовому сигналу
                    if hasattr(self, 'base_signal') and self.base_signal is not None:
                        self.base_signal = self.apply_session_marker_modifier(
                            self.base_signal
                        )
                    modifiers_applied += 1
                elif modifier_name == "whale_analysis" and self.whale_analysis_result:
                    self.apply_whale_analysis_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "portfolio_analysis"
                    and self.portfolio_analysis_result
                ):
                    self.apply_portfolio_analysis_modifier()
                    modifiers_applied += 1
                elif modifier_name == "meta_controller" and self.meta_controller_result:
                    self.apply_meta_controller_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "pattern_discovery"
                    and self.pattern_discovery_result
                ):
                    self.apply_pattern_discovery_modifier()
                    modifiers_applied += 1
                elif modifier_name == "model_selector" and self.model_selector_result:
                    self.apply_model_trainer_modifier()  # Используем существующий метод
                    modifiers_applied += 1
                elif (
                    modifier_name == "advanced_price_predictor"
                    and self.advanced_price_predictor_result
                ):
                    # Убираем вызов несуществующего метода
                    modifiers_applied += 1
                elif (
                    modifier_name == "window_optimizer" and self.window_optimizer_result
                ):
                    # Убираем вызов несуществующего метода
                    modifiers_applied += 1
                elif modifier_name == "state_manager" and self.state_manager_result:
                    self.apply_noise_analyzer_modifier()  # Используем существующий метод
                    modifiers_applied += 1
                elif modifier_name == "dataset_manager" and self.dataset_manager_result:
                    self.apply_dataset_manager_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "evolvable_decision_reasoner"
                    and self.evolvable_decision_reasoner_result
                ):
                    self.apply_evolvable_decision_reasoner_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "regime_discovery" and self.regime_discovery_result
                ):
                    self.apply_regime_discovery_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "advanced_market_maker"
                    and self.advanced_market_maker_result
                ):
                    # Убираем вызов несуществующего метода
                    modifiers_applied += 1
                elif (
                    modifier_name == "market_memory_integration"
                    and self.market_memory_integration_result
                ):
                    self.apply_entanglement_integration_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "market_memory_whale_integration"
                    and self.market_memory_whale_integration_result
                ):
                    self.apply_entanglement_integration_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "local_ai_controller"
                    and self.local_ai_controller_result
                ):
                    self.apply_local_ai_controller_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "analytical_integration"
                    and self.analytical_integration_result
                ):
                    self.apply_analytical_integration_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "entanglement_integration"
                    and self.entanglement_integration_result
                ):
                    self.apply_entanglement_integration_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "agent_order_executor"
                    and self.agent_order_executor_result
                ):
                    self.apply_agent_order_executor_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "agent_market_maker_model"
                    and self.agent_market_maker_model_result
                ):
                    self.apply_agent_market_maker_model_modifier()
                    modifiers_applied += 1
                elif modifier_name == "sandbox_trainer" and self.sandbox_trainer_result:
                    self.apply_sandbox_trainer_modifier()
                    modifiers_applied += 1
                elif modifier_name == "model_trainer" and self.model_trainer_result:
                    self.apply_model_trainer_modifier()
                    modifiers_applied += 1
                elif (
                    modifier_name == "window_model_trainer"
                    and self.window_model_trainer_result
                ):
                    self.apply_window_model_trainer_modifier()
                    modifiers_applied += 1

                # Обновляем счетчик уровней приоритета
                if current_priority and current_priority not in self._metadata.get(
                    "processed_priorities", []
                ):
                    if "processed_priorities" not in self._metadata:
                        self._metadata["processed_priorities"] = []
                    self._metadata["processed_priorities"].append(current_priority)
                    priority_levels_processed += 1

            except Exception as e:
                logger.error(f"Error applying modifier {modifier_name}: {e}")
                continue

        # Сохраняем кэш
        self.set("modifiers_cache_key", cache_key)
        self.set("last_modifiers_application", time.time())

        # Вычисляем время выполнения
        execution_time_ms = (time.time() - start_time) * 1000

        # Логируем метрики
        logger.debug(
            f"Applied {modifiers_applied} modifiers for {self.symbol} in {execution_time_ms:.2f}ms, "
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
        import hashlib

        # Создаем строку с состоянием всех результатов
        state_string = ""

        # Добавляем хеши результатов аналитических модулей
        for result_name in [
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
            "model_selector_result",
            "advanced_price_predictor_result",
            "window_optimizer_result",
            "state_manager_result",
            "dataset_manager_result",
            "evolvable_decision_reasoner_result",
            "regime_discovery_result",
            "advanced_market_maker_result",
            "market_memory_integration_result",
            "market_memory_whale_integration_result",
            "local_ai_controller_result",
            "analytical_integration_result",
            "entanglement_integration_result",
            "agent_order_executor_result",
            "agent_market_regime_result",
            "agent_market_maker_model_result",
            "sandbox_trainer_result",
            "model_trainer_result",
            "window_model_trainer_result",
        ]:
            result = getattr(self, result_name, None)
            if result:
                state_string += f"{result_name}:{hash(str(result))}"

        # Добавляем хеш модификаторов стратегий
        state_string += f"modifiers:{hash(str(self.strategy_modifiers))}"

        # Создаем MD5 хеш
        return hashlib.md5(state_string.encode()).hexdigest()

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

        for result_name in [
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
            "model_selector_result",
            "advanced_price_predictor_result",
            "window_optimizer_result",
            "state_manager_result",
            "dataset_manager_result",
            "evolvable_decision_reasoner_result",
            "regime_discovery_result",
            "advanced_market_maker_result",
            "market_memory_integration_result",
            "market_memory_whale_integration_result",
            "local_ai_controller_result",
            "analytical_integration_result",
            "entanglement_integration_result",
            "agent_order_executor_result",
            "agent_market_regime_result",
            "agent_market_maker_model_result",
            "sandbox_trainer_result",
            "model_trainer_result",
            "window_model_trainer_result",
        ]:
            if getattr(self, result_name, None) is not None:
                active_count += 1

        return active_count

    def reset_modifiers_cache(self) -> None:
        """Сброс кэша модификаторов."""
        self.remove("modifiers_cache_key")
        self.remove("last_modifiers_application")
        self._metadata.pop("processed_priorities", None)
        logger.debug(f"Reset modifiers cache for {self.symbol}")

    def validate_modifiers(self) -> Dict[str, Any]:
        """Валидация модификаторов стратегий."""
        validation_result: Dict[str, Any] = {"is_valid": True, "warnings": [], "errors": []}

        modifiers = self.strategy_modifiers

        # Проверяем граничные значения
        if modifiers.order_aggressiveness < 0.1:
            validation_result["warnings"].append("Very low order aggressiveness")
        elif modifiers.order_aggressiveness > 5.0:
            validation_result["warnings"].append("Very high order aggressiveness")

        if modifiers.position_size_multiplier < 0.1:
            validation_result["warnings"].append("Very low position size multiplier")
        elif modifiers.position_size_multiplier > 3.0:
            validation_result["warnings"].append("Very high position size multiplier")

        if modifiers.confidence_multiplier < 0.1:
            validation_result["errors"].append("Invalid confidence multiplier")
            validation_result["is_valid"] = False
        elif modifiers.confidence_multiplier > 2.0:
            validation_result["warnings"].append("Very high confidence multiplier")

        if modifiers.risk_multiplier < 0.1:
            validation_result["errors"].append("Invalid risk multiplier")
            validation_result["is_valid"] = False
        elif modifiers.risk_multiplier > 5.0:
            validation_result["warnings"].append("Very high risk multiplier")

        if modifiers.execution_delay_ms < 0:
            validation_result["errors"].append("Invalid execution delay")
            validation_result["is_valid"] = False
        elif modifiers.execution_delay_ms > 10000:
            validation_result["warnings"].append("Very high execution delay")

        # Проверяем логическую согласованность
        if modifiers.order_aggressiveness > 1.5 and modifiers.risk_multiplier > 1.5:
            validation_result["warnings"].append("High aggressiveness with high risk")

        if (
            modifiers.confidence_multiplier < 0.5
            and modifiers.position_size_multiplier > 1.2
        ):
            validation_result["warnings"].append(
                "Low confidence with high position size"
            )

        return validation_result

    def apply_doass_modifier(self) -> None:
        """Применение модификатора DOASS."""
        try:
            if not self.doass_result:
                return

            # Получаем данные о текущем символе из DOASS
            symbol_data = getattr(self.doass_result, 'detailed_profiles', {}).get(self.symbol)
            if not symbol_data:
                return

            # Применяем модификаторы на основе opportunity score
            opportunity_score = symbol_data.opportunity_score
            confidence = symbol_data.confidence
            market_phase = symbol_data.market_phase

            # Модификаторы на основе opportunity score
            if opportunity_score > 0.9:
                # Очень высокая возможность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.15
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif opportunity_score > 0.8:
                # Высокая возможность - умеренно увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.position_size_multiplier *= 1.05
                self.strategy_modifiers.confidence_multiplier *= 1.05
            elif opportunity_score < 0.6:
                # Низкая возможность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.8
                self.strategy_modifiers.position_size_multiplier *= 0.7
                self.strategy_modifiers.confidence_multiplier *= 0.9

            # Модификаторы на основе уверенности
            if confidence > 0.8:
                # Высокая уверенность - увеличиваем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.confidence_multiplier *= 1.1
            elif confidence < 0.5:
                # Низкая уверенность - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.confidence_multiplier *= 0.8

            # Модификаторы на основе фазы рынка
            if market_phase.value == "breakout_setup":
                # Подготовка к пробою - увеличиваем готовность
                self.strategy_modifiers.order_aggressiveness *= 1.1
                self.strategy_modifiers.execution_delay_ms = max(
                    0, self.strategy_modifiers.execution_delay_ms - 100
                )
            elif market_phase.value == "breakout_active":
                # Активный пробой - максимальная агрессивность
                self.strategy_modifiers.order_aggressiveness *= 1.2
                self.strategy_modifiers.position_size_multiplier *= 1.1
                self.strategy_modifiers.execution_delay_ms = 0
            elif market_phase.value == "exhaustion":
                # Истощение - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.risk_multiplier *= 1.3
            elif market_phase.value == "reversion_potential":
                # Потенциал разворота - осторожная агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.9
                self.strategy_modifiers.position_size_multiplier *= 0.8
                self.strategy_modifiers.price_offset_percent *= 1.2

            # Модификаторы на основе корреляций
            if (
                hasattr(self.doass_result, "correlation_matrix")
                and not self.doass_result.correlation_matrix.empty
            ):
                # Проверяем корреляции с другими символами
                symbol_correlations: Dict[str, float] = (
                    self.doass_result.correlation_matrix.get(self.symbol, {})
                )
                high_correlations = sum(
                    1 for corr in symbol_correlations.values() if abs(corr) > 0.8
                )

                if high_correlations > 3:
                    # Много высоких корреляций - снижаем диверсификацию
                    self.strategy_modifiers.position_size_multiplier *= 0.8
                    self.strategy_modifiers.risk_multiplier *= 1.2

            # Модификаторы на основе энтанглмента
            if (
                hasattr(self.doass_result, "entanglement_groups")
                and self.doass_result.entanglement_groups
            ):
                # Проверяем, находится ли символ в группе энтанглмента
                for group in self.doass_result.entanglement_groups:
                    if self.symbol in group and len(group) > 1:
                        # Символ в группе энтанглмента - снижаем агрессивность
                        self.strategy_modifiers.order_aggressiveness *= 0.8
                        self.strategy_modifiers.position_size_multiplier *= 0.7
                        self.strategy_modifiers.risk_multiplier *= 1.3
                        break

            # Модификаторы на основе памяти паттернов
            if hasattr(self.doass_result, "pattern_memory_insights"):
                pattern_insights = self.doass_result.pattern_memory_insights.get(
                    self.symbol
                )
                if pattern_insights:
                    success_rate = pattern_insights.get("success_rate", 0.5)
                    if success_rate > 0.7:
                        # Высокий успех паттернов - увеличиваем уверенность
                        self.strategy_modifiers.confidence_multiplier *= 1.1
                    elif success_rate < 0.4:
                        # Низкий успех паттернов - снижаем уверенность
                        self.strategy_modifiers.confidence_multiplier *= 0.9

            # Модификаторы на основе ликвидности
            if hasattr(self.doass_result, "liquidity_gravity_scores"):
                liquidity_score = self.doass_result.liquidity_gravity_scores.get(
                    self.symbol, 0.5
                )
                if liquidity_score > 0.8:
                    # Высокая ликвидность - увеличиваем размер позиции
                    self.strategy_modifiers.position_size_multiplier *= 1.1
                elif liquidity_score < 0.3:
                    # Низкая ликвидность - снижаем размер позиции
                    self.strategy_modifiers.position_size_multiplier *= 0.7
                    self.strategy_modifiers.risk_multiplier *= 1.2

            # Модификаторы на основе предсказания разворотов
            if hasattr(self.doass_result, "reversal_probabilities"):
                reversal_prob = self.doass_result.reversal_probabilities.get(
                    self.symbol, 0.0
                )
                if reversal_prob > 0.7:
                    # Высокая вероятность разворота - осторожность
                    self.strategy_modifiers.order_aggressiveness *= 0.8
                    self.strategy_modifiers.position_size_multiplier *= 0.7
                    self.strategy_modifiers.price_offset_percent *= 1.3

            logger.debug(
                f"Applied DOASS modifier for {self.symbol}: opportunity_score={opportunity_score:.3f}, confidence={confidence:.3f}, phase={market_phase.value}"
            )

        except Exception as e:
            logger.error(f"Error applying DOASS modifier: {e}")

    def get_doass_status(self) -> Dict[str, Any]:
        """Получение статуса DOASS."""
        try:
            if not self.doass_result:
                return {"is_analyzed": False, "status": "unknown"}

            # Получаем данные о текущем символе из DOASS
            symbol_data = getattr(self.doass_result, 'detailed_profiles', {}).get(self.symbol)
            if not symbol_data:
                return {"is_analyzed": False, "status": "not_found"}

            return {
                "is_analyzed": True,
                "opportunity_score": getattr(symbol_data, 'opportunity_score', 0.0),
                "confidence": getattr(symbol_data, 'confidence', 0.0),
                "market_phase": getattr(symbol_data, 'market_phase', 'unknown'),
                "total_symbols_analyzed": getattr(self.doass_result, 'total_symbols_analyzed', 0),
                "processing_time_ms": getattr(self.doass_result, 'processing_time_ms', 0),
                "cache_hit_rate": getattr(self.doass_result, 'cache_hit_rate', 0.0),
                "status": "high_opportunity" if getattr(symbol_data, 'opportunity_score', 0.0) > 0.8 else "normal",
            }

        except Exception as e:
            logger.error(f"Error getting DOASS status: {e}")
            return {"is_analyzed": False, "status": "error"}

    def apply_entanglement_risk_modifier(self) -> None:
        """Применение модификатора риска запутанности."""
        try:
            if not self.entanglement_result:
                return

            # Применяем модификаторы на основе риска запутанности
            correlation_score = self.entanglement_result.correlation_score
            confidence = self.entanglement_result.confidence

            if correlation_score > 0.9:
                # Очень высокая корреляция - резко снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.5
                self.strategy_modifiers.position_size_multiplier *= 0.4
                self.strategy_modifiers.confidence_multiplier *= 0.6
                self.strategy_modifiers.risk_multiplier *= 2.0
            elif correlation_score > 0.7:
                # Высокая корреляция - снижаем агрессивность
                self.strategy_modifiers.order_aggressiveness *= 0.7
                self.strategy_modifiers.position_size_multiplier *= 0.6
                self.strategy_modifiers.confidence_multiplier *= 0.8
                self.strategy_modifiers.risk_multiplier *= 1.5

            # Модификаторы на основе уверенности
            if confidence < 0.6:
                self.strategy_modifiers.execution_delay_ms = int(self.strategy_modifiers.execution_delay_ms * 2.0)

            logger.debug(
                f"Applied entanglement risk modifier: correlation={correlation_score:.3f}, confidence={confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Error applying entanglement risk modifier: {e}")

    def apply_signal_modifiers(self, signal: Signal) -> Signal:
        try:
            modified_confidence = min(1.0, float(signal.confidence) * self.strategy_modifiers.confidence_multiplier)
            modified_strength = min(1.0, float(signal.strength) * self.strategy_modifiers.confidence_multiplier)
            modified_signal = Signal(
                direction=signal.direction if isinstance(signal.direction, SignalDirection) else SignalDirection(signal.direction),
                trading_pair=signal.trading_pair if isinstance(signal.trading_pair, TradingPair) else TradingPair(signal.trading_pair),
                signal_type=signal.signal_type,
                confidence=Decimal(str(modified_confidence)),
                strength=Decimal(str(modified_strength)),
                metadata=signal.metadata
            )
            return modified_signal
        except Exception as e:
            logger.error(f"Error applying signal modifiers: {e}")
            return signal


class AgentContextManager:
    """Менеджер контекстов агентов."""

    def __init__(self) -> None:
        self.contexts: Dict[str, AgentContext] = {}

    def get_context(self, symbol: str) -> AgentContext:
        """Получение контекста для символа."""
        if symbol not in self.contexts:
            self.contexts[symbol] = AgentContext(symbol=symbol)
        return self.contexts[symbol]

    def update_context(self, symbol: str, context: AgentContext) -> None:
        """Обновление контекста."""
        self.contexts[symbol] = context

    def clear_context(self, symbol: str) -> None:
        """Очистка контекста."""
        if symbol in self.contexts:
            del self.contexts[symbol]

    def get_context_statistics(self) -> Dict[str, Any]:
        """Получение статистики контекстов."""
        return {
            "total_contexts": len(self.contexts),
            "contexts_with_pattern_prediction": sum(
                1
                for ctx in self.contexts.values()
                if ctx.pattern_prediction.prediction_result is not None
            ),
            "contexts_with_risk_assessment": sum(
                1 for ctx in self.contexts.values() if ctx.risk_assessment is not None
            ),
            "average_confidence": sum(
                ctx.pattern_prediction.prediction_confidence
                for ctx in self.contexts.values()
                if ctx.pattern_prediction.prediction_confidence is not None
            )
            / max(len(self.contexts), 1),
        }
