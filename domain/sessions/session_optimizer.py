# -*- coding: utf-8 -*-
"""Оптимизатор торговых сессий."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame, Series

from domain.types.session_types import (
    SessionAnalysisResult,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .interfaces import SessionRegistry
from .session_marker import SessionMarker


@dataclass
class OptimizationTarget:
    """Цель оптимизации."""

    # Основные цели
    maximize_volume: bool = True
    minimize_volatility: bool = False
    maximize_momentum: bool = True
    minimize_spread: bool = True
    # Веса целей (0.0 - 1.0)
    volume_weight: float = 0.3
    volatility_weight: float = 0.2
    momentum_weight: float = 0.3
    spread_weight: float = 0.2
    # Ограничения
    min_volume_multiplier: float = 0.5
    max_volatility_multiplier: float = 2.0
    min_momentum_strength: float = 0.1
    max_spread_multiplier: float = 1.5

    def validate(self) -> bool:
        """Валидация параметров оптимизации."""
        weights_sum = (
            self.volume_weight
            + self.volatility_weight
            + self.momentum_weight
            + self.spread_weight
        )
        if not (0.99 <= weights_sum <= 1.01):
            logger.warning(f"Optimization weights sum to {weights_sum}, should be 1.0")
            return False
        if not (0.0 <= self.volume_weight <= 1.0):
            logger.warning(f"Invalid volume_weight: {self.volume_weight}")
            return False
        if not (0.0 <= self.volatility_weight <= 1.0):
            logger.warning(f"Invalid volatility_weight: {self.volatility_weight}")
            return False
        if not (0.0 <= self.momentum_weight <= 1.0):
            logger.warning(f"Invalid momentum_weight: {self.momentum_weight}")
            return False
        if not (0.0 <= self.spread_weight <= 1.0):
            logger.warning(f"Invalid spread_weight: {self.spread_weight}")
            return False
        # Если все проверки пройдены, возвращаем True
        return True

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, Dict[str, float]]]:
        """Преобразование в словарь."""
        return {
            "maximize_volume": self.maximize_volume,
            "minimize_volatility": self.minimize_volatility,
            "maximize_momentum": self.maximize_momentum,
            "minimize_spread": self.minimize_spread,
            "volume_weight": self.volume_weight,
            "volatility_weight": self.volatility_weight,
            "momentum_weight": self.momentum_weight,
            "spread_weight": self.spread_weight,
            "min_volume_multiplier": self.min_volume_multiplier,
            "max_volatility_multiplier": self.max_volatility_multiplier,
            "min_momentum_strength": self.min_momentum_strength,
            "max_spread_multiplier": self.max_spread_multiplier,
        }


@dataclass
class OptimizationResult:
    """Результат оптимизации."""

    # Основные результаты
    optimized_profile: SessionProfile
    optimization_score: float
    improvement_percentage: float
    # Детали оптимизации
    original_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_metrics: Dict[str, float] = field(default_factory=dict)
    parameter_changes: Dict[str, float] = field(default_factory=dict)
    # Метаданные
    optimization_time_ms: float = 0.0
    iterations_count: int = 0
    convergence_achieved: bool = True

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, Dict[str, float]]]:
        """Преобразование в словарь."""
        return {
            "optimization_score": self.optimization_score,
            "improvement_percentage": self.improvement_percentage,
            "original_metrics": self.original_metrics,
            "optimized_metrics": self.optimized_metrics,
            "parameter_changes": self.parameter_changes,
            "optimization_time_ms": self.optimization_time_ms,
            "iterations_count": self.iterations_count,
            "convergence_achieved": self.convergence_achieved,
        }


class SessionOptimizer:
    """Оптимизатор торговых сессий."""

    def __init__(
        self,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
    ) -> None:
        self.registry = registry
        self.session_marker = session_marker
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        logger.info("SessionOptimizer initialized")

    def optimize_session_profile(
        self,
        session_type: SessionType,
        target: OptimizationTarget,
        historical_data: Optional[DataFrame] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[OptimizationResult]:
        """Оптимизация профиля сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Валидируем цель оптимизации
            if not target.validate():
                logger.error("Invalid optimization target")
                return None
            # Получаем текущий профиль
            current_profile = self.registry.get_profile(session_type)
            if not current_profile:
                logger.error(f"Session profile not found for {session_type.value}")
                return None
            # Анализируем текущие метрики
            current_metrics = self._analyze_profile_metrics(
                current_profile, historical_data, timestamp
            )
            # Выполняем оптимизацию
            start_time = time.time()
            optimized_profile, optimization_score, iterations = (
                self._perform_optimization(
                    current_profile, target, historical_data, timestamp
                )
            )
            optimization_time_ms = (time.time() - start_time) * 1000
            # Анализируем оптимизированные метрики
            optimized_metrics = self._analyze_profile_metrics(
                optimized_profile, historical_data, timestamp
            )
            # Рассчитываем улучшение
            improvement_percentage = self._calculate_improvement(
                current_metrics, optimized_metrics, target
            )
            # Определяем изменения параметров
            parameter_changes = self._calculate_parameter_changes(
                current_profile, optimized_profile
            )
            # Проверяем сходимость
            convergence_achieved = iterations < self.max_iterations
            result = OptimizationResult(
                optimized_profile=optimized_profile,
                optimization_score=optimization_score,
                improvement_percentage=improvement_percentage,
                original_metrics=current_metrics,
                optimized_metrics=optimized_metrics,
                parameter_changes=parameter_changes,
                optimization_time_ms=optimization_time_ms,
                iterations_count=iterations,
                convergence_achieved=convergence_achieved,
            )
            logger.info(
                f"Session profile optimization completed for {session_type.value} - "
                f"Score: {optimization_score:.4f}, "
                f"Iterations: {iterations}, "
                f"Time: {optimization_time_ms:.2f}ms"
            )
            return result
        except Exception as e:
            logger.error(
                f"Error optimizing session profile for {session_type.value}: {e}"
            )
            return None

    def optimize_multiple_sessions(
        self,
        session_types: List[SessionType],
        target: OptimizationTarget,
        historical_data: Optional[DataFrame] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Dict[SessionType, OptimizationResult]:
        """Оптимизация нескольких сессий."""
        results: Dict[SessionType, OptimizationResult] = {}
        for session_type in session_types:
            result = self.optimize_session_profile(
                session_type, target, historical_data, timestamp
            )
            if result:
                results[session_type] = result
        logger.info(f"Optimized {len(results)} session profiles")
        return results

    def get_optimization_recommendations(
        self,
        session_type: SessionType,
        historical_data: Optional[DataFrame] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> List[str]:
        """Получение рекомендаций по оптимизации."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            profile = self.registry.get_profile(session_type)
            if not profile:
                return []
            recommendations: List[str] = []
            # Анализируем текущие характеристики
            metrics = self._analyze_profile_metrics(profile, historical_data, timestamp)
            # Рекомендации на основе метрик
            if metrics.get("volume_multiplier", 1.0) < 0.8:
                recommendations.append(
                    "Рассмотрите увеличение типичного множителя объема для повышения ликвидности"
                )
            if metrics.get("volatility_multiplier", 1.0) > 1.3:
                recommendations.append(
                    "Рассмотрите снижение типичного множителя волатильности для стабилизации"
                )
            if metrics.get("spread_multiplier", 1.0) > 1.2:
                recommendations.append(
                    "Рассмотрите оптимизацию спредов для улучшения исполнения ордеров"
                )
            if metrics.get("momentum_strength", 0.0) < 0.3:
                recommendations.append(
                    "Рассмотрите улучшение технических сигналов для повышения импульса"
                )
            # Рекомендации на основе поведения
            if profile.behavior.false_breakout_probability > 0.4:
                recommendations.append(
                    "Высокая вероятность ложных пробоев - рассмотрите улучшение фильтрации сигналов"
                )
            if profile.behavior.reversal_probability > 0.3:
                recommendations.append(
                    "Высокая вероятность разворотов - рассмотрите улучшение определения трендов"
                )
            return recommendations
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return []

    def _analyze_profile_metrics(
        self,
        profile: SessionProfile,
        historical_data: Optional[DataFrame],
        timestamp: Timestamp,
    ) -> Dict[str, float]:
        """Анализ метрик профиля."""
        metrics: Dict[str, float] = {
            "volume_multiplier": profile.typical_volume_multiplier,
            "volatility_multiplier": profile.typical_volatility_multiplier,
            "spread_multiplier": profile.typical_spread_multiplier,
            "momentum_strength": profile.technical_signal_strength,
            "whale_activity": profile.whale_activity_probability,
            "mm_activity": profile.mm_activity_probability,
            "news_sensitivity": profile.news_sensitivity,
            "manipulation_susceptibility": profile.manipulation_susceptibility,
        }
        # Дополнительные метрики на основе исторических данных
        if historical_data is not None and len(historical_data) > 0:
            historical_metrics = self._calculate_historical_metrics(
                profile, historical_data, timestamp
            )
            metrics.update(historical_metrics)
        return metrics

    def _calculate_historical_metrics(
        self,
        profile: SessionProfile,
        historical_data: DataFrame,
        timestamp: Timestamp,
    ) -> Dict[str, float]:
        """Расчет метрик на основе исторических данных."""
        metrics: Dict[str, float] = {}
        try:
            # Анализируем данные для данного типа сессии
            session_context = self.session_marker.get_session_context(timestamp)
            if (
                session_context.primary_session
                and session_context.primary_session.session_type == profile.session_type
            ):
                # Фильтруем данные по времени сессии
                session_data = self._filter_data_by_session(
                    historical_data, profile, timestamp
                )
                if len(session_data) > 0:
                    # Рассчитываем метрики
                    metrics["historical_volume_ratio"] = self._calculate_volume_ratio(
                        session_data
                    )
                    metrics["historical_volatility_ratio"] = (
                        self._calculate_volatility_ratio(session_data)
                    )
                    metrics["historical_momentum_ratio"] = (
                        self._calculate_momentum_ratio(session_data)
                    )
        except Exception as e:
            logger.warning(f"Error calculating historical metrics: {e}")
        return metrics

    def _filter_data_by_session(
        self,
        data: DataFrame,
        profile: SessionProfile,
        timestamp: Timestamp,
    ) -> DataFrame:
        """Фильтрация данных по сессии."""
        if data.empty:
            return data

        def is_session_time(row_timestamp: pd.Timestamp) -> bool:
            hour = row_timestamp.hour
            minute = row_timestamp.minute
            
            # Получаем параметры сессии
            session_start_hour = getattr(profile, 'session_start_hour', 0)
            session_end_hour = getattr(profile, 'session_end_hour', 24)
            
            # Проверяем, что значения являются int
            if not isinstance(hour, int):
                hour = 0
            if not isinstance(minute, int):
                minute = 0
            if not isinstance(session_start_hour, int):
                session_start_hour = 0
            if not isinstance(session_end_hour, int):
                session_end_hour = 24
            
            if session_start_hour <= session_end_hour:
                return session_start_hour <= hour <= session_end_hour
            else:
                return hour >= session_start_hour or hour <= session_end_hour

        # Применяем фильтр
        session_mask = data.index.map(is_session_time)
        return data[session_mask]

    def _calculate_volume_ratio(self, data: DataFrame) -> float:
        """Расчет соотношения объема."""
        if "volume" not in data.columns or len(data) < 2:
            return 1.0
        volume_series: Series = data["volume"]
        recent_volume = volume_series.tail(10).mean()
        historical_volume = volume_series.head(-10).mean()
        if historical_volume == 0:
            return 1.0
        return float(recent_volume / historical_volume)

    def _calculate_volatility_ratio(self, data: DataFrame) -> float:
        """Расчет соотношения волатильности."""
        if "close" not in data.columns or len(data) < 20:
            return 1.0
        close_series: Series = data["close"]
        returns = close_series.pct_change().dropna()
        if len(returns) < 10:
            return 1.0
        recent_volatility = returns.tail(10).std()
        historical_volatility = returns.head(-10).std()
        if historical_volatility == 0:
            return 1.0
        return float(recent_volatility / historical_volatility)

    def _calculate_momentum_ratio(self, data: DataFrame) -> float:
        """Расчет соотношения импульса."""
        if "close" not in data.columns or (hasattr(data.index, '__len__') and len(data.index) < 14):
            return 1.0
        close_series: Series = data["close"]
        # Рассчитываем RSI
        delta = close_series.diff()
        gain = (delta.where(delta.gt(0), 0)).rolling(window=14).mean()
        loss = (delta.where(delta.lt(0), 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) == 0:
            return 1.0
        
        current_rsi = float(rsi.iloc[-1])
        if pd.isna(current_rsi):
            return 1.0
        
        # Нормализуем RSI к диапазону 0-2
        if current_rsi > 70:
            momentum = 1.5 + (current_rsi - 70) / 30
            return float(np.clip(momentum, 0.0, 2.0))
        elif current_rsi < 30:
            momentum = 0.5 - (30 - current_rsi) / 30
            return float(np.clip(momentum, 0.0, 2.0))
        else:
            momentum = 1.0
            return float(np.clip(momentum, 0.0, 2.0))

    def _perform_optimization(
        self,
        current_profile: SessionProfile,
        target: OptimizationTarget,
        historical_data: Optional[DataFrame],
        timestamp: Timestamp,
    ) -> Tuple[SessionProfile, float, int]:
        """Выполнение оптимизации."""
        best_profile = current_profile
        best_score = self._calculate_optimization_score(
            current_profile, target, historical_data, timestamp
        )
        iterations = 0
        improvement_threshold = self.convergence_threshold
        while iterations < self.max_iterations:
            # Генерируем кандидата
            candidate_profile = self._generate_candidate_profile(
                current_profile, target
            )
            # Оцениваем кандидата
            candidate_score = self._calculate_optimization_score(
                candidate_profile, target, historical_data, timestamp
            )
            # Проверяем улучшение
            if candidate_score > best_score + improvement_threshold:
                best_profile = candidate_profile
                best_score = candidate_score
                improvement_threshold = self.convergence_threshold
            else:
                improvement_threshold *= 0.95  # Уменьшаем порог
            iterations += 1
            # Проверяем сходимость
            if improvement_threshold < self.convergence_threshold * 0.1:
                break
        
        return best_profile, best_score, iterations

    def _generate_candidate_profile(
        self,
        current_profile: SessionProfile,
        target: OptimizationTarget,
    ) -> SessionProfile:
        """Генерация кандидата для оптимизации."""
        # Создаем копию профиля
        candidate = current_profile.model_copy()
        # Случайно модифицируем параметры в пределах ограничений
        import random

        # Модифицируем множители
        candidate.typical_volume_multiplier = np.clip(
            candidate.typical_volume_multiplier * random.uniform(0.9, 1.1),
            target.min_volume_multiplier,
            2.0,
        )
        candidate.typical_volatility_multiplier = np.clip(
            candidate.typical_volatility_multiplier * random.uniform(0.9, 1.1),
            0.5,
            target.max_volatility_multiplier,
        )
        candidate.typical_spread_multiplier = np.clip(
            candidate.typical_spread_multiplier * random.uniform(0.9, 1.1),
            0.5,
            target.max_spread_multiplier,
        )
        candidate.technical_signal_strength = np.clip(
            candidate.technical_signal_strength * random.uniform(0.9, 1.1),
            target.min_momentum_strength,
            1.0,
        )
        return candidate

    def _calculate_optimization_score(
        self,
        profile: SessionProfile,
        target: OptimizationTarget,
        historical_data: Optional[DataFrame],
        timestamp: Timestamp,
    ) -> float:
        """Расчет оценки оптимизации."""
        metrics = self._analyze_profile_metrics(profile, historical_data, timestamp)
        score = 0.0
        # Объем
        if target.maximize_volume:
            volume_score = min(metrics.get("volume_multiplier", 1.0) / 2.0, 1.0)
            score += target.volume_weight * volume_score
        else:
            volume_score = max(
                0.0, 1.0 - abs(metrics.get("volume_multiplier", 1.0) - 1.0)
            )
            score += target.volume_weight * volume_score
        # Волатильность
        if target.minimize_volatility:
            volatility_score = max(
                0.0, 1.0 - abs(metrics.get("volatility_multiplier", 1.0) - 1.0)
            )
            score += target.volatility_weight * volatility_score
        else:
            volatility_score = min(metrics.get("volatility_multiplier", 1.0) / 2.0, 1.0)
            score += target.volatility_weight * volatility_score
        # Импульс
        if target.maximize_momentum:
            momentum_score = metrics.get("momentum_strength", 0.0)
            score += target.momentum_weight * momentum_score
        else:
            momentum_score = max(
                0.0, 1.0 - abs(metrics.get("momentum_strength", 0.0) - 0.5)
            )
            score += target.momentum_weight * momentum_score
        # Спред
        if target.minimize_spread:
            spread_score = max(
                0.0, 1.0 - abs(metrics.get("spread_multiplier", 1.0) - 1.0)
            )
            score += target.spread_weight * spread_score
        else:
            spread_score = min(metrics.get("spread_multiplier", 1.0) / 2.0, 1.0)
            score += target.spread_weight * spread_score
        return score

    def _calculate_improvement(
        self,
        current_metrics: Dict[str, float],
        optimized_metrics: Dict[str, float],
        target: OptimizationTarget,
    ) -> float:
        """Расчет процента улучшения."""
        current_score = self._calculate_metrics_score(current_metrics, target)
        optimized_score = self._calculate_metrics_score(optimized_metrics, target)
        if current_score == 0:
            return 0.0
        improvement = (optimized_score - current_score) / current_score * 100
        return float(np.clip(improvement, -100.0, 1000.0))

    def _calculate_metrics_score(
        self,
        metrics: Dict[str, float],
        target: OptimizationTarget,
    ) -> float:
        """Расчет оценки метрик."""
        score = 0.0
        # Упрощенная оценка на основе основных метрик
        volume_score = min(metrics.get("volume_multiplier", 1.0) / 2.0, 1.0)
        volatility_score = max(
            0.0, 1.0 - abs(metrics.get("volatility_multiplier", 1.0) - 1.0)
        )
        momentum_score = metrics.get("momentum_strength", 0.0)
        spread_score = max(0.0, 1.0 - abs(metrics.get("spread_multiplier", 1.0) - 1.0))
        score = (
            target.volume_weight * volume_score
            + target.volatility_weight * volatility_score
            + target.momentum_weight * momentum_score
            + target.spread_weight * spread_score
        )
        return score

    def _calculate_parameter_changes(
        self,
        current_profile: SessionProfile,
        optimized_profile: SessionProfile,
    ) -> Dict[str, float]:
        """Расчет изменений параметров."""
        changes: Dict[str, float] = {}
        # Основные параметры
        changes["volume_multiplier_change"] = (
            optimized_profile.typical_volume_multiplier
            - current_profile.typical_volume_multiplier
        )
        changes["volatility_multiplier_change"] = (
            optimized_profile.typical_volatility_multiplier
            - current_profile.typical_volatility_multiplier
        )
        changes["spread_multiplier_change"] = (
            optimized_profile.typical_spread_multiplier
            - current_profile.typical_spread_multiplier
        )
        changes["technical_signal_strength_change"] = (
            optimized_profile.technical_signal_strength
            - current_profile.technical_signal_strength
        )
        return changes
