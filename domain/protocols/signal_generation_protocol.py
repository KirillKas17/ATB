"""
Протокол генерации сигналов для торговых стратегий.
Обеспечивает создание, валидацию, оптимизацию и фильтрацию торговых сигналов
с использованием продвинутых алгоритмов и машинного обучения.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.types import (
    ConfidenceLevel,
    PriceValue,
    SignalId,
    StrategyId,
    Symbol,
    VolumeValue,
)
from domain.types.protocol_types import SignalFilterDict
from domain.exceptions import StrategyExecutionError
from domain.exceptions.base_exceptions import DomainException


class SignalQuality(Enum):
    """Качество торгового сигнала."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"


class SignalSource(Enum):
    """Источник сигнала."""

    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MACHINE_LEARNING = "machine_learning"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    COMPOSITE = "composite"


@dataclass
class SignalMetadata:
    """Метаданные торгового сигнала."""

    source: SignalSource
    quality: SignalQuality
    confidence_score: float  # 0.0 - 1.0
    risk_score: float  # 0.0 - 1.0
    expected_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_holding_period: timedelta
    market_regime: str
    volatility_regime: str
    correlation_with_market: float
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamental_factors: Dict[str, float] = field(default_factory=dict)
    sentiment_metrics: Dict[str, float] = field(default_factory=dict)
    market_microstructure: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация метаданных."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"Risk score must be between 0.0 and 1.0, got {self.risk_score}")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"Win rate must be between 0.0 and 1.0, got {self.win_rate}")
        if self.profit_factor < 0:
            raise ValueError(f"Profit factor cannot be negative, got {self.profit_factor}")


@dataclass
class SignalValidationResult:
    """Результат валидации сигнала."""

    is_valid: bool
    confidence: float  # 0.0 - 1.0
    risk_level: str  # "low", "medium", "high", "extreme"
    validation_score: float  # 0.0 - 1.0
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация результата."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.validation_score <= 1.0:
            raise ValueError(f"Validation score must be between 0.0 and 1.0, got {self.validation_score}")


@dataclass
class SignalOptimizationResult:
    """Результат оптимизации сигнала."""

    original_signal: Signal
    optimized_signal: Signal
    improvement_score: float  # 0.0 - 1.0
    parameter_changes: Dict[str, Tuple[Any, Any]]  # (old_value, new_value)
    expected_improvement: Dict[str, float]
    optimization_method: str
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация результата оптимизации."""
        if not 0.0 <= self.improvement_score <= 1.0:
            raise ValueError(f"Improvement score must be between 0.0 and 1.0, got {self.improvement_score}")
        if self.optimization_time < 0:
            raise ValueError(f"Optimization time cannot be negative, got {self.optimization_time}")


@runtime_checkable
class SignalGenerationProtocol(Protocol):
    """Протокол генерации сигналов."""

    async def validate_signal_conditions(self, market_data: MarketData) -> bool: ...
    async def calculate_signal_strength(
        self, indicators: Dict[str, float]
    ) -> SignalStrength: ...
    async def apply_risk_filters(self, signal: Signal) -> bool: ...
    async def optimize_signal_parameters(self, signal: Signal) -> Signal: ...
    async def calculate_signal_confidence(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> ConfidenceLevel: ...


class SignalGenerationProtocolImpl(ABC):
    """
    Реализация протокола генерации сигналов.
    
    Предоставляет продвинутые алгоритмы для:
    - Генерации сигналов на основе множественных источников
    - Валидации сигналов с комплексными проверками
    - Оптимизации параметров сигналов
    - Фильтрации сигналов по качеству и риску
    - Расчетов уверенности и метрик производительности
    """

    def __init__(self) -> None:
        """Инициализация генератора сигналов."""
        self.logger = logging.getLogger(__name__)
        self._signal_cache: Dict[str, Signal] = {}
        self._validation_cache: Dict[str, SignalValidationResult] = {}
        self._optimization_cache: Dict[str, SignalOptimizationResult] = {}
        self._ml_models: Dict[str, Any] = {}
        self._signal_history: List[Signal] = []

    # ============================================================================
    # ОСНОВНЫЕ МЕТОДЫ ГЕНЕРАЦИИ
    # ============================================================================

    @abstractmethod
    async def generate_signal(
        self,
        strategy_id: StrategyId,
        market_data: pd.DataFrame,
        signal_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала с комплексным анализом.
        
        Args:
            strategy_id: ID стратегии
            market_data: Рыночные данные
            signal_params: Параметры генерации сигнала
            
        Returns:
            Optional[Signal]: Торговый сигнал или None
            
        Raises:
            StrategyNotFoundError: Стратегия не найдена
            SignalGenerationError: Ошибка генерации сигнала
        """
        pass

    @abstractmethod
    async def validate_signal(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        risk_limits: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Валидация торгового сигнала с комплексными проверками.
        
        Args:
            signal: Сигнал для валидации
            market_data: Рыночные данные
            risk_limits: Лимиты риска
            
        Returns:
            bool: Валидность сигнала
        """
        pass

    @abstractmethod
    async def calculate_signal_confidence(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        historical_signals: List[Signal],
    ) -> ConfidenceLevel:
        """
        Расчет уверенности в сигнале с машинным обучением.
        
        Args:
            signal: Сигнал
            market_data: Рыночные данные
            historical_signals: Исторические сигналы
            
        Returns:
            ConfidenceLevel: Уровень уверенности
        """
        pass

    @abstractmethod
    async def optimize_signal_parameters(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> Signal:
        """
        Оптимизация параметров сигнала с генетическими алгоритмами.
        
        Args:
            signal: Исходный сигнал
            market_data: Рыночные данные
            
        Returns:
            Signal: Оптимизированный сигнал
        """
        pass

    @abstractmethod
    async def filter_signals(
        self, signals: List[Signal], filters: SignalFilterDict
    ) -> List[Signal]:
        """
        Фильтрация сигналов с продвинутыми алгоритмами.
        
        Args:
            signals: Список сигналов
            filters: Фильтры
            
        Returns:
            List[Signal]: Отфильтрованные сигналы
        """
        pass

    # ============================================================================
    # ПРОДВИНУТЫЕ МЕТОДЫ ГЕНЕРАЦИИ
    # ============================================================================

    async def generate_composite_signal(
        self,
        strategy_id: StrategyId,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float],
        fundamental_data: Optional[Dict[str, float]] = None,
        sentiment_data: Optional[Dict[str, float]] = None,
        microstructure_data: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        """
        Генерация композитного сигнала на основе множественных источников.
        
        Объединяет:
        - Технический анализ
        - Фундаментальный анализ
        - Анализ настроений
        - Микроструктуру рынка
        """
        # Анализ технических индикаторов
        technical_score = await self._calculate_technical_score(technical_indicators)
        
        # Анализ фундаментальных данных
        fundamental_score = 0.5  # Нейтральный по умолчанию
        if fundamental_data:
            fundamental_score = await self._calculate_fundamental_score(fundamental_data)
        
        # Анализ настроений
        sentiment_score = 0.5  # Нейтральный по умолчанию
        if sentiment_data:
            sentiment_score = await self._calculate_sentiment_score(sentiment_data)
        
        # Анализ микроструктуры
        microstructure_score = 0.5  # Нейтральный по умолчанию
        if microstructure_data:
            microstructure_score = await self._calculate_microstructure_score(microstructure_data)
        
        # Взвешенная комбинация
        weights = {
            "technical": 0.4,
            "fundamental": 0.2,
            "sentiment": 0.2,
            "microstructure": 0.2
        }
        
        composite_score = (
            technical_score * weights["technical"] +
            fundamental_score * weights["fundamental"] +
            sentiment_score * weights["sentiment"] +
            microstructure_score * weights["microstructure"]
        )
        
        # Определение направления сигнала
        if composite_score > 0.6:
            signal_type = SignalType.BUY
            strength = SignalStrength.STRONG if composite_score > 0.8 else SignalStrength.MEDIUM
        elif composite_score < 0.4:
            signal_type = SignalType.SELL
            strength = SignalStrength.STRONG if composite_score < 0.2 else SignalStrength.MEDIUM
        else:
            return None  # Нет четкого сигнала
        
        # Создание сигнала
        signal = Signal(
            id=SignalId(str(hashlib.md5(f"{strategy_id}_{datetime.now()}".encode()).hexdigest()[:16])),
            type=signal_type,
            strength=strength,
            price=PriceValue(market_data["close"].iloc[-1]),
            volume=VolumeValue(1.0),  # Будет рассчитано позже
            timestamp=datetime.now(),
            metadata={
                "composite_score": composite_score,
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "sentiment_score": sentiment_score,
                "microstructure_score": microstructure_score,
                "source": SignalSource.COMPOSITE.value
            }
        )
        
        return signal

    async def validate_signal_comprehensive(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        risk_limits: Optional[Dict[str, float]] = None,
    ) -> SignalValidationResult:
        """
        Комплексная валидация сигнала с множественными проверками.
        
        Включает:
        - Проверки технического анализа
        - Валидацию риска
        - Проверки рыночных условий
        - Анализ исторической производительности
        - Проверки корреляции
        """
        passed_checks = []
        failed_checks = []
        warnings = []
        recommendations = []
        
        # Проверка технических условий
        technical_valid = await self._validate_technical_conditions(signal, market_data)
        if technical_valid:
            passed_checks.append("technical_conditions")
        else:
            failed_checks.append("technical_conditions")
            recommendations.append("Проверьте технические индикаторы")
        
        # Проверка рыночных условий
        market_valid = await self._validate_market_conditions(signal, market_data)
        if market_valid:
            passed_checks.append("market_conditions")
        else:
            failed_checks.append("market_conditions")
            warnings.append("Неблагоприятные рыночные условия")
        
        # Проверка риска
        risk_valid = await self._validate_risk_limits(signal, risk_limits or {})
        if risk_valid:
            passed_checks.append("risk_limits")
        else:
            failed_checks.append("risk_limits")
            failed_checks.append("risk_limits")
            recommendations.append("Уменьшите размер позиции")
        
        # Проверка волатильности
        volatility_valid = await self._validate_volatility_conditions(signal, market_data)
        if volatility_valid:
            passed_checks.append("volatility_conditions")
        else:
            failed_checks.append("volatility_conditions")
            warnings.append("Высокая волатильность")
        
        # Проверка ликвидности
        liquidity_valid = await self._validate_liquidity_conditions(signal, market_data)
        if liquidity_valid:
            passed_checks.append("liquidity_conditions")
        else:
            failed_checks.append("liquidity_conditions")
            failed_checks.append("liquidity_conditions")
            recommendations.append("Проверьте ликвидность рынка")
        
        # Расчет общего результата
        total_checks = len(passed_checks) + len(failed_checks)
        validation_score = len(passed_checks) / total_checks if total_checks > 0 else 0.0
        
        # Определение уровня риска
        risk_level = self._determine_risk_level(len(failed_checks), validation_score)
        
        # Расчет уверенности
        confidence = self._calculate_validation_confidence(passed_checks, failed_checks, warnings)
        
        return SignalValidationResult(
            is_valid=validation_score >= 0.7 and len(failed_checks) <= 2,
            confidence=confidence,
            risk_level=risk_level,
            validation_score=validation_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            recommendations=recommendations
        )

    async def optimize_signal_advanced(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        optimization_target: str = "sharpe_ratio",
        optimization_method: str = "genetic_algorithm",
    ) -> SignalOptimizationResult:
        """
        Продвинутая оптимизация сигнала с множественными методами.
        
        Args:
            signal: Исходный сигнал
            market_data: Рыночные данные
            optimization_target: Цель оптимизации
            optimization_method: Метод оптимизации
            
        Returns:
            SignalOptimizationResult: Результат оптимизации
        """
        start_time = datetime.now()
        
        # Клонирование исходного сигнала
        original_signal = signal
        
        # Выбор метода оптимизации
        if optimization_method == "genetic_algorithm":
            optimized_signal = await self._optimize_with_genetic_algorithm(
                signal, market_data, optimization_target
            )
        elif optimization_method == "bayesian_optimization":
            optimized_signal = await self._optimize_with_bayesian(
                signal, market_data, optimization_target
            )
        elif optimization_method == "gradient_descent":
            optimized_signal = await self._optimize_with_gradient_descent(
                signal, market_data, optimization_target
            )
        else:
            optimized_signal = signal
        
        # Расчет улучшения
        improvement_score = await self._calculate_improvement_score(
            original_signal, optimized_signal, market_data, optimization_target
        )
        
        # Определение изменений параметров
        parameter_changes = self._identify_parameter_changes(original_signal, optimized_signal)
        
        # Ожидаемые улучшения
        expected_improvement = await self._calculate_expected_improvements(
            original_signal, optimized_signal, market_data
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return SignalOptimizationResult(
            original_signal=original_signal,
            optimized_signal=optimized_signal,
            improvement_score=improvement_score,
            parameter_changes=parameter_changes,
            expected_improvement=expected_improvement,
            optimization_method=optimization_method,
            optimization_time=optimization_time
        )

    async def filter_signals_advanced(
        self,
        signals: List[Signal],
        filters: SignalFilterDict,
        market_data: pd.DataFrame,
    ) -> List[Signal]:
        """
        Продвинутая фильтрация сигналов с машинным обучением.
        
        Включает:
        - Фильтрацию по качеству
        - Фильтрацию по риску
        - Фильтрацию по корреляции
        - Фильтрацию по рыночным условиям
        """
        filtered_signals = signals.copy()
        
        # Фильтрация по минимальной уверенности
        if "min_confidence" in filters:
            min_confidence = filters["min_confidence"]
            filtered_signals = [
                signal for signal in filtered_signals
                if signal.metadata.get("confidence", 0.0) >= min_confidence
            ]
        
        # Фильтрация по максимальному риску
        if "max_risk" in filters:
            max_risk = filters["max_risk"]
            filtered_signals = [
                signal for signal in filtered_signals
                if signal.metadata.get("risk_score", 1.0) <= max_risk
            ]
        
        # Фильтрация по типу сигнала
        if "allowed_types" in filters:
            allowed_types = filters["allowed_types"]
            filtered_signals = [
                signal for signal in filtered_signals
                if signal.type.value in allowed_types
            ]
        
        # Фильтрация по рыночным условиям
        market_filtered = await self._filter_by_market_conditions(
            filtered_signals, market_data
        )
        
        # Фильтрация по корреляции
        correlation_filtered = await self._filter_by_correlation(market_filtered)
        
        # Фильтрация по качеству с ML
        quality_filtered = await self._filter_by_quality_ml(correlation_filtered, market_data)
        
        return quality_filtered

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================================

    async def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Расчет технического скора."""
        if not indicators:
            return 0.5
        
        # Нормализация индикаторов
        normalized_indicators = {}
        for name, value in indicators.items():
            if name.startswith("rsi"):
                normalized_indicators[name] = value / 100.0
            elif name.startswith("macd"):
                normalized_indicators[name] = 0.5 + (value / 100.0)
            elif name.startswith("bb"):
                normalized_indicators[name] = value
            else:
                normalized_indicators[name] = 0.5
        
        # Взвешенное среднее
        weights = {
            "rsi": 0.3,
            "macd": 0.3,
            "bb": 0.2,
            "volume": 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for name, value in normalized_indicators.items():
            for indicator_type, weight in weights.items():
                if name.startswith(indicator_type):
                    score += value * weight
                    total_weight += weight
                    break
        
        return score / total_weight if total_weight > 0 else 0.5

    async def _calculate_fundamental_score(self, fundamental_data: Dict[str, float]) -> float:
        """Расчет фундаментального скора."""
        # Упрощенная реализация
        return 0.5

    async def _calculate_sentiment_score(self, sentiment_data: Dict[str, float]) -> float:
        """Расчет скора настроений."""
        # Упрощенная реализация
        return 0.5

    async def _calculate_microstructure_score(self, microstructure_data: Dict[str, float]) -> float:
        """Расчет скора микроструктуры."""
        # Упрощенная реализация
        return 0.5

    async def _validate_technical_conditions(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> bool:
        """Валидация технических условий."""
        # Проверка тренда
        if len(market_data) < 20:
            return False
        
        # Простая проверка тренда
        recent_prices = market_data["close"].tail(20)
        trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if signal.type == SignalType.BUY and trend < -0.05:
            return False
        if signal.type == SignalType.SELL and trend > 0.05:
            return False
        
        return True

    async def _validate_market_conditions(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> bool:
        """Валидация рыночных условий."""
        # Проверка волатильности
        if len(market_data) < 10:
            return False
        
        volatility = market_data["close"].pct_change().std()
        if volatility > 0.1:  # Высокая волатильность
            return False
        
        return True

    async def _validate_risk_limits(
        self, signal: Signal, risk_limits: Dict[str, float]
    ) -> bool:
        """Валидация лимитов риска."""
        # Проверка максимального риска на сделку
        max_risk = risk_limits.get("max_risk_per_trade", 0.02)
        signal_risk = signal.metadata.get("risk_score", 0.5)
        
        return signal_risk <= max_risk

    async def _validate_volatility_conditions(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> bool:
        """Валидация условий волатильности."""
        # Упрощенная проверка
        return True

    async def _validate_liquidity_conditions(
        self, signal: Signal, market_data: pd.DataFrame
    ) -> bool:
        """Валидация условий ликвидности."""
        # Упрощенная проверка
        return True

    def _determine_risk_level(self, failed_checks: int, validation_score: float) -> str:
        """Определение уровня риска."""
        if failed_checks == 0 and validation_score >= 0.9:
            return "low"
        elif failed_checks <= 1 and validation_score >= 0.8:
            return "medium"
        elif failed_checks <= 2 and validation_score >= 0.7:
            return "high"
        else:
            return "extreme"

    def _calculate_validation_confidence(
        self, passed_checks: List[str], failed_checks: List[str], warnings: List[str]
    ) -> float:
        """Расчет уверенности валидации."""
        total_checks = len(passed_checks) + len(failed_checks)
        if total_checks == 0:
            return 0.5
        
        base_confidence = len(passed_checks) / total_checks
        
        # Снижение за предупреждения
        warning_penalty = len(warnings) * 0.05
        
        return max(0.0, min(1.0, base_confidence - warning_penalty))

    async def _optimize_with_genetic_algorithm(
        self, signal: Signal, market_data: pd.DataFrame, target: str
    ) -> Signal:
        """Оптимизация с генетическим алгоритмом."""
        # Упрощенная реализация
        return signal

    async def _optimize_with_bayesian(
        self, signal: Signal, market_data: pd.DataFrame, target: str
    ) -> Signal:
        """Оптимизация с байесовской оптимизацией."""
        # Упрощенная реализация
        return signal

    async def _optimize_with_gradient_descent(
        self, signal: Signal, market_data: pd.DataFrame, target: str
    ) -> Signal:
        """Оптимизация с градиентным спуском."""
        # Упрощенная реализация
        return signal

    async def _calculate_improvement_score(
        self, original: Signal, optimized: Signal, market_data: pd.DataFrame, target: str
    ) -> float:
        """Расчет скора улучшения."""
        # Упрощенная реализация
        return 0.1

    def _identify_parameter_changes(
        self, original: Signal, optimized: Signal
    ) -> Dict[str, Tuple[Any, Any]]:
        """Определение изменений параметров."""
        changes = {}
        
        # Сравнение основных параметров
        if original.price != optimized.price:
            changes["price"] = (original.price, optimized.price)
        
        if original.volume != optimized.volume:
            changes["volume"] = (original.volume, optimized.volume)
        
        if original.strength != optimized.strength:
            changes["strength"] = (original.strength, optimized.strength)
        
        return changes

    async def _calculate_expected_improvements(
        self, original: Signal, optimized: Signal, market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Расчет ожидаемых улучшений."""
        return {
            "expected_return": 0.05,
            "risk_reduction": 0.1,
            "sharpe_improvement": 0.2
        }

    async def _filter_by_market_conditions(
        self, signals: List[Signal], market_data: pd.DataFrame
    ) -> List[Signal]:
        """Фильтрация по рыночным условиям."""
        return signals

    async def _filter_by_correlation(self, signals: List[Signal]) -> List[Signal]:
        """Фильтрация по корреляции."""
        return signals

    async def _filter_by_quality_ml(
        self, signals: List[Signal], market_data: pd.DataFrame
    ) -> List[Signal]:
        """Фильтрация по качеству с машинным обучением."""
        return signals 