"""
Анализатор успешности паттернов маркет-мейкера.
Промышленная реализация с поддержкой:
- Анализа исторической успешности
- Прогнозирования результатов
- Адаптивных порогов
- Статистического анализа
"""

from shared.numpy_utils import np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from domain.market_maker.mm_pattern import PatternMemory, PatternOutcome, PatternResult
from domain.types.market_maker_types import (
    Accuracy,
    AverageReturn,
    MarketMakerPatternType,
    SuccessCount,
    TotalCount,
)

from ..models.analysis_config import AnalysisConfig


@dataclass
class SuccessAnalysisResult:
    """Результат анализа успешности."""

    pattern_type: MarketMakerPatternType
    success_rate: float
    avg_return: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    trend: str  # "improving", "declining", "stable"
    reliability_score: float
    market_conditions: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]


class SuccessRateAnalyzer:
    """
    Промышленный анализатор успешности паттернов.
    Особенности:
    - Статистический анализ успешности
    - Прогнозирование трендов
    - Адаптивные пороги
    - Анализ рыночных условий
    - Генерация рекомендаций
    """

    def __init__(self, config: AnalysisConfig):
        """
        Инициализация анализатора.
        Args:
            config: Конфигурация анализа
        """
        self.config = config
        self._setup_analysis_components()
        logger.info("SuccessRateAnalyzer initialized successfully")

    def _setup_analysis_components(self) -> None:
        """Настройка компонентов анализа."""
        # Кэш для результатов анализа
        self.analysis_cache: Dict[str, SuccessAnalysisResult] = {}
        # Статистика анализа
        self.analysis_stats = {
            "total_analyses": 0,
            "high_reliability_count": 0,
            "improving_trends": 0,
            "declining_trends": 0,
            "stable_trends": 0,
        }
        # Пороги для различных уровней успешности
        self.success_thresholds = {
            "excellent": 0.8,
            "good": 0.7,
            "fair": 0.6,
            "poor": 0.5,
        }

    async def analyze_pattern_success(
        self,
        pattern_type: MarketMakerPatternType,
        historical_patterns: List[PatternMemory],
        time_window_days: int = 30,
    ) -> SuccessAnalysisResult:
        """
        Анализ успешности паттерна.
        Args:
            pattern_type: Тип паттерна
            historical_patterns: Исторические паттерны
            time_window_days: Временное окно анализа
        Returns:
            Результат анализа успешности
        """
        try:
            # Проверяем кэш
            cache_key = f"{pattern_type.value}_{time_window_days}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            # Фильтруем паттерны по типу и времени
            filtered_patterns = self._filter_patterns_by_type_and_time(
                historical_patterns, pattern_type, time_window_days
            )
            if not filtered_patterns:
                return self._create_default_analysis_result(pattern_type)
            # Рассчитываем базовую статистику
            success_rate = self._calculate_success_rate(filtered_patterns)
            avg_return = self._calculate_average_return(filtered_patterns)
            sample_size = len(filtered_patterns)
            # Рассчитываем доверительный интервал
            confidence_interval = self._calculate_confidence_interval(
                success_rate, sample_size
            )
            # Анализируем тренд
            trend = self._analyze_success_trend(filtered_patterns)
            # Рассчитываем оценку надежности
            reliability_score = self._calculate_reliability_score(
                success_rate, sample_size, confidence_interval
            )
            # Анализируем рыночные условия
            market_conditions = self._analyze_market_conditions(filtered_patterns)
            # Генерируем рекомендации
            recommendations = self._generate_recommendations(
                success_rate, avg_return, trend, reliability_score
            )
            # Создаем результат анализа
            result = SuccessAnalysisResult(
                pattern_type=pattern_type,
                success_rate=success_rate,
                avg_return=avg_return,
                confidence_interval=confidence_interval,
                sample_size=sample_size,
                trend=trend,
                reliability_score=reliability_score,
                market_conditions=market_conditions,
                recommendations=recommendations,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "time_window_days": time_window_days,
                    "filtered_patterns_count": len(filtered_patterns),
                },
            )
            # Кэшируем результат
            self.analysis_cache[cache_key] = result
            # Обновляем статистику
            self._update_analysis_stats(result)
            logger.debug(
                f"Success analysis for {pattern_type.value}: "
                f"rate={success_rate:.3f}, return={avg_return:.3f}, "
                f"trend={trend}, reliability={reliability_score:.3f}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to analyze pattern success: {e}")
            return self._create_default_analysis_result(pattern_type)

    def _filter_patterns_by_type_and_time(
        self,
        patterns: List[PatternMemory],
        pattern_type: MarketMakerPatternType,
        time_window_days: int,
    ) -> List[PatternMemory]:
        """Фильтрация паттернов по типу и времени."""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            filtered_patterns = []
            for pattern in patterns:
                # Проверяем тип паттерна
                if pattern.pattern.pattern_type != pattern_type:
                    continue
                # Проверяем время
                if pattern.last_seen and pattern.last_seen < cutoff_date:
                    continue
                # Проверяем наличие результата
                if not pattern.result:
                    continue
                filtered_patterns.append(pattern)
            return filtered_patterns
        except Exception as e:
            logger.error(f"Failed to filter patterns: {e}")
            return []

    def _calculate_success_rate(self, patterns: List[PatternMemory]) -> float:
        """Расчет коэффициента успешности."""
        try:
            if not patterns:
                return 0.0
            successful_count = sum(
                1
                for p in patterns
                if p.result and p.result.outcome == PatternOutcome.SUCCESS
            )
            total_count = len(patterns)
            return successful_count / total_count if total_count > 0 else 0.0
        except Exception as e:
            logger.error(f"Failed to calculate success rate: {e}")
            return 0.0

    def _calculate_average_return(self, patterns: List[PatternMemory]) -> float:
        """Расчет средней доходности."""
        try:
            if not patterns:
                return 0.0
            returns = []
            for pattern in patterns:
                if pattern.result:
                    returns.append(pattern.result.price_change_15min)
            if not returns:
                return 0.0
            return float(np.mean(returns))
        except Exception as e:
            logger.error(f"Failed to calculate average return: {e}")
            return 0.0

    def _calculate_confidence_interval(
        self, success_rate: float, sample_size: int, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Расчет доверительного интервала."""
        try:
            if sample_size == 0:
                return (0.0, 0.0)
            # Используем нормальное приближение для биномиального распределения
            z_score = 1.96  # Для 95% доверительного интервала
            standard_error = np.sqrt(success_rate * (1 - success_rate) / sample_size)
            margin_of_error = z_score * standard_error
            lower_bound = max(0.0, success_rate - margin_of_error)
            upper_bound = min(1.0, success_rate + margin_of_error)
            return (lower_bound, upper_bound)
        except Exception as e:
            logger.error(f"Failed to calculate confidence interval: {e}")
            return (0.0, 0.0)

    def _analyze_success_trend(self, patterns: List[PatternMemory]) -> str:
        """Анализ тренда успешности."""
        try:
            if len(patterns) < 10:
                return "stable"
            # Сортируем по времени
            sorted_patterns = sorted(
                patterns, key=lambda p: p.last_seen or datetime.min
            )
            # Разделяем на две группы
            mid_point = len(sorted_patterns) // 2
            early_patterns = sorted_patterns[:mid_point]
            late_patterns = sorted_patterns[mid_point:]
            # Рассчитываем успешность для каждой группы
            early_success_rate = self._calculate_success_rate(early_patterns)
            late_success_rate = self._calculate_success_rate(late_patterns)
            # Определяем тренд
            difference = late_success_rate - early_success_rate
            if difference > 0.1:
                return "improving"
            elif difference < -0.1:
                return "declining"
            else:
                return "stable"
        except Exception as e:
            logger.error(f"Failed to analyze success trend: {e}")
            return "stable"

    def _calculate_reliability_score(
        self,
        success_rate: float,
        sample_size: int,
        confidence_interval: Tuple[float, float],
    ) -> float:
        """Расчет оценки надежности."""
        try:
            # Факторы надежности
            sample_size_factor = min(
                1.0, sample_size / 50.0
            )  # Больше образцов = выше надежность
            confidence_width = confidence_interval[1] - confidence_interval[0]
            confidence_factor = max(
                0.0, 1.0 - confidence_width
            )  # Уже интервал = выше надежность
            # Комбинированная оценка
            reliability = (
                sample_size_factor * 0.4 + confidence_factor * 0.4 + success_rate * 0.2
            )
            return max(0.0, min(1.0, reliability))
        except Exception as e:
            logger.error(f"Failed to calculate reliability score: {e}")
            return 0.5

    def _analyze_market_conditions(
        self, patterns: List[PatternMemory]
    ) -> Dict[str, float]:
        """Анализ рыночных условий."""
        try:
            if not patterns:
                return {}
            market_conditions = {
                "avg_volume": 0.0,
                "avg_volatility": 0.0,
                "avg_spread": 0.0,
                "bull_market_ratio": 0.0,
                "bear_market_ratio": 0.0,
                "sideways_market_ratio": 0.0,
            }
            volumes = []
            volatilities = []
            spreads = []
            market_phases = []
            for pattern in patterns:
                if pattern.result:
                    # Анализируем объем
                    if hasattr(pattern.pattern.features, "volume_delta"):
                        volumes.append(
                            abs(float(pattern.pattern.features.volume_delta))
                        )
                    # Анализируем волатильность
                    if hasattr(pattern.pattern.features, "price_volatility"):
                        volatilities.append(
                            float(pattern.pattern.features.price_volatility)
                        )
                    # Анализируем спред
                    if hasattr(pattern.pattern.features, "spread_change"):
                        spreads.append(
                            abs(float(pattern.pattern.features.spread_change))
                        )
                    # Анализируем рыночную фазу
                    if pattern.result.price_change_15min > 0.005:
                        market_phases.append("bull")
                    elif pattern.result.price_change_15min < -0.005:
                        market_phases.append("bear")
                    else:
                        market_phases.append("sideways")
            # Рассчитываем средние значения
            if volumes:
                market_conditions["avg_volume"] = float(np.mean(volumes))
            if volatilities:
                market_conditions["avg_volatility"] = float(np.mean(volatilities))
            if spreads:
                market_conditions["avg_spread"] = float(np.mean(spreads))
            # Рассчитываем соотношения рыночных фаз
            if market_phases:
                total_phases = len(market_phases)
                market_conditions["bull_market_ratio"] = (
                    market_phases.count("bull") / total_phases
                )
                market_conditions["bear_market_ratio"] = (
                    market_phases.count("bear") / total_phases
                )
                market_conditions["sideways_market_ratio"] = (
                    market_phases.count("sideways") / total_phases
                )
            return market_conditions
        except Exception as e:
            logger.error(f"Failed to analyze market conditions: {e}")
            return {}

    def _generate_recommendations(
        self,
        success_rate: float,
        avg_return: float,
        trend: str,
        reliability_score: float,
    ) -> List[str]:
        """Генерация рекомендаций."""
        try:
            recommendations = []
            # Рекомендации по успешности
            if success_rate >= self.success_thresholds["excellent"]:
                recommendations.append(
                    "Паттерн показывает отличную успешность. Рекомендуется активное использование."
                )
            elif success_rate >= self.success_thresholds["good"]:
                recommendations.append(
                    "Паттерн показывает хорошую успешность. Рекомендуется умеренное использование."
                )
            elif success_rate >= self.success_thresholds["fair"]:
                recommendations.append(
                    "Паттерн показывает удовлетворительную успешность. Рекомендуется осторожное использование."
                )
            else:
                recommendations.append(
                    "Паттерн показывает низкую успешность. Рекомендуется избегать или использовать с крайней осторожностью."
                )
            # Рекомендации по доходности
            if avg_return > 0.01:
                recommendations.append(
                    "Паттерн показывает высокую доходность. Рассмотрите увеличение размера позиции."
                )
            elif avg_return < -0.01:
                recommendations.append(
                    "Паттерн показывает убыточность. Рассмотрите уменьшение размера позиции или избегание."
                )
            # Рекомендации по тренду
            if trend == "improving":
                recommendations.append(
                    "Тренд успешности улучшается. Рассмотрите увеличение активности."
                )
            elif trend == "declining":
                recommendations.append(
                    "Тренд успешности ухудшается. Рассмотрите уменьшение активности."
                )
            # Рекомендации по надежности
            if reliability_score >= 0.8:
                recommendations.append(
                    "Высокая надежность анализа. Можно полагаться на результаты."
                )
            elif reliability_score < 0.5:
                recommendations.append(
                    "Низкая надежность анализа. Требуется дополнительная валидация."
                )
            return recommendations
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Ошибка при генерации рекомендаций"]

    def _create_default_analysis_result(
        self, pattern_type: MarketMakerPatternType
    ) -> SuccessAnalysisResult:
        """Создание результата анализа по умолчанию."""
        return SuccessAnalysisResult(
            pattern_type=pattern_type,
            success_rate=0.5,
            avg_return=0.0,
            confidence_interval=(0.0, 1.0),
            sample_size=0,
            trend="stable",
            reliability_score=0.0,
            market_conditions={},
            recommendations=["Недостаточно данных для анализа"],
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "default_result": True,
            },
        )

    def _update_analysis_stats(self, result: SuccessAnalysisResult) -> None:
        """Обновление статистики анализа."""
        try:
            self.analysis_stats["total_analyses"] += 1
            if result.reliability_score >= 0.8:
                self.analysis_stats["high_reliability_count"] += 1
            if result.trend == "improving":
                self.analysis_stats["improving_trends"] += 1
            elif result.trend == "declining":
                self.analysis_stats["declining_trends"] += 1
            else:
                self.analysis_stats["stable_trends"] += 1
        except Exception as e:
            logger.error(f"Failed to update analysis stats: {e}")

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Получение статистики анализа."""
        try:
            total_analyses = self.analysis_stats["total_analyses"]
            if total_analyses == 0:
                return {
                    "total_analyses": 0,
                    "high_reliability_rate": 0.0,
                    "improving_trends_rate": 0.0,
                    "declining_trends_rate": 0.0,
                    "stable_trends_rate": 0.0,
                }
            return {
                "total_analyses": total_analyses,
                "high_reliability_rate": self.analysis_stats["high_reliability_count"]
                / total_analyses,
                "improving_trends_rate": self.analysis_stats["improving_trends"]
                / total_analyses,
                "declining_trends_rate": self.analysis_stats["declining_trends"]
                / total_analyses,
                "stable_trends_rate": self.analysis_stats["stable_trends"]
                / total_analyses,
            }
        except Exception as e:
            logger.error(f"Failed to get analysis statistics: {e}")
            return {}

    def clear_cache(self) -> None:
        """Очистка кэша анализа."""
        try:
            self.analysis_cache.clear()
            logger.info("Success rate analysis cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear analysis cache: {e}")

    def get_cache_size(self) -> int:
        """Получение размера кэша."""
        return len(self.analysis_cache)
