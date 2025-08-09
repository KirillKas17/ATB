"""
Анализатор паттернов маркет-мейкинга.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass

import pandas as pd
from shared.numpy_utils import np

from domain.market_maker.mm_pattern import MarketMakerPattern, PatternResult
from domain.market_maker.mm_pattern_memory import PatternMemory
from domain.type_definitions.market_maker_types import (
    PatternOutcome, Confidence, SimilarityScore, PatternContext
)
from domain.interfaces.pattern_analyzer import IPatternAnalyzer
from ..models.analysis_config import AnalysisConfig

# Type aliases for better mypy support
Series = pd.Series
DataFrame = pd.DataFrame

class PatternResultProtocol(Protocol):
    """Протокол для результатов паттернов."""
    outcome: PatternOutcome
    price_change_5min: float
    price_change_15min: float
    price_change_30min: float
    volume_change: float
    volatility_change: float

@dataclass
class PatternAnalysisResult:
    """Результат анализа паттерна."""

    pattern: MarketMakerPattern
    similarity_score: SimilarityScore
    confidence: Confidence
    predicted_outcome: PatternResult
    effectiveness: float
    risk_level: str
    recommendations: List[Dict[str, Any]]
    market_context: Dict[str, Any]
    analysis_metadata: Dict[str, Any]

class PatternAnalyzer(IPatternAnalyzer):
    """Анализатор паттернов маркет-мейкинга."""

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_analysis_components()

    def _setup_analysis_components(self) -> None:
        """Настройка компонентов анализа."""
        # Инициализация компонентов анализа
        self.similarity_threshold = self.config.get_similarity_threshold()
        self.confidence_threshold = self.config.get_confidence_threshold()
        self.effectiveness_threshold = self.config.get_effectiveness_threshold()

    async def analyze_pattern_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> SimilarityScore:
        """
        Анализ схожести двух паттернов.
        Args:
            pattern1: Первый паттерн
            pattern2: Второй паттерн
        Returns:
            Оценка схожести
        """
        try:
            # Базовое сравнение типов паттернов
            if pattern1.pattern_type != pattern2.pattern_type:
                return SimilarityScore(0.0)

            # Сравнение признаков
            feature_similarity = self._calculate_feature_similarity(
                pattern1.features, pattern2.features
            )

            # Сравнение контекста - конвертируем PatternContext в dict
            context1_dict = dict(pattern1.context) if isinstance(pattern1.context, dict) else getattr(pattern1.context, '__dict__', {})
            context2_dict = dict(pattern2.context) if isinstance(pattern2.context, dict) else getattr(pattern2.context, '__dict__', {})
            context_similarity = self._calculate_context_similarity(
                context1_dict, context2_dict
            )

            # Сравнение временных характеристик
            temporal_similarity = self._calculate_temporal_similarity(
                pattern1.timestamp, pattern2.timestamp
            )

            # Взвешенная оценка схожести
            weighted_similarity = (
                feature_similarity * 0.5 +
                context_similarity * 0.3 +
                temporal_similarity * 0.2
            )

            return SimilarityScore(float(weighted_similarity))
        except Exception as e:
            self.logger.error(f"Failed to analyze pattern similarity: {e}")
            return SimilarityScore(0.0)

    async def calculate_pattern_confidence(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> Confidence:
        """
        Расчет уверенности в паттерне.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Уровень уверенности
        """
        try:
            if not historical_patterns:
                return Confidence(0.5)

            # Находим похожие паттерны
            similar_patterns = []
            for hist_pattern in historical_patterns:
                similarity = await self.analyze_pattern_similarity(
                    pattern, hist_pattern.pattern
                )
                if float(similarity) > self.similarity_threshold:
                    similar_patterns.append((hist_pattern, float(similarity)))

            if not similar_patterns:
                return Confidence(0.3)

            # Рассчитываем взвешенную точность
            total_weight = 0.0
            weighted_accuracy = 0.0
            for hist_pattern, similarity in similar_patterns:
                weight = similarity
                accuracy = float(hist_pattern.accuracy)
                weighted_accuracy += accuracy * weight
                total_weight += weight

            if total_weight == 0:
                return Confidence(0.5)

            weighted_similarity_float = weighted_accuracy / total_weight

            # Повышение уверенности на основе дополнительных факторов
            confidence_boost = await self._calculate_confidence_boost(
                pattern, historical_patterns
            )

            # Финальная уверенность
            final_confidence = min(1.0, weighted_similarity_float + confidence_boost)
            self.logger.debug(f"Pattern confidence: {final_confidence:.3f}")
            return Confidence(final_confidence)
        except Exception as e:
            self.logger.error(f"Failed to calculate pattern confidence: {e}")
            return Confidence(0.5)

    async def _calculate_confidence_boost(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> float:
        """Расчет повышения уверенности."""
        boost = 0.0
        # Фактор высокой точности
        high_accuracy_patterns = [p for p in historical_patterns if p.is_reliable()]
        if high_accuracy_patterns:
            boost += self.config.get_confidence_boost_factor("high_accuracy") * 0.3
        # Фактор объема
        if hasattr(pattern.features, "volume_delta"):
            volume_factor = min(1.0, abs(float(pattern.features.volume_delta)) / 0.5)
            boost += (
                self.config.get_confidence_boost_factor("high_volume") * volume_factor
            )
        # Фактор частоты паттерна
        pattern_type_count = len(
            [
                p
                for p in historical_patterns
                if p.pattern.pattern_type == pattern.pattern_type
            ]
        )
        frequency_factor = min(1.0, pattern_type_count / 10.0)
        boost += (
            self.config.get_confidence_boost_factor("pattern_frequency")
            * frequency_factor
        )
        # Фактор соответствия рыночному режиму
        market_regime_match = await self._check_market_regime_match(
            pattern, historical_patterns
        )
        boost += (
            self.config.get_confidence_boost_factor("market_regime_match")
            * market_regime_match
        )
        # Фактор временной актуальности
        recent_patterns = [
            p for p in historical_patterns
            if p.last_seen is not None and (datetime.now() - p.last_seen).days <= 7
        ]
        recency_factor = min(1.0, len(recent_patterns) / 5.0)
        boost += (
            self.config.get_confidence_boost_factor("time_recency") * recency_factor
        )
        return min(0.3, boost)  # Ограничиваем максимальное повышение

    async def _check_market_regime_match(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> float:
        """Проверка соответствия рыночному режиму."""
        try:
            total_contexts = 0
            similar_contexts = 0
            current_context = pattern.context
            current_context_dict = dict(current_context) if isinstance(current_context, dict) else getattr(current_context, '__dict__', {})
            for hist_pattern in historical_patterns:
                hist_context = hist_pattern.pattern.context
                hist_context_dict = dict(hist_context) if isinstance(hist_context, dict) else getattr(hist_context, '__dict__', {})
                context_similarity = self._calculate_context_similarity(
                    current_context_dict, hist_context_dict
                )
                if context_similarity > 0.7:
                    similar_contexts += 1
                total_contexts += 1
            if total_contexts == 0:
                return 0.5
            return similar_contexts / total_contexts
        except Exception as e:
            self.logger.error(f"Failed to check market regime match: {e}")
            return 0.5

    def _calculate_context_similarity(
        self, context1: Dict[str, Any], context2: Dict[str, Any]
    ) -> float:
        """Расчет схожести контекстов."""
        try:
            # Простая метрика схожести на основе общих ключей
            keys1 = set(context1.keys())
            keys2 = set(context2.keys())
            if not keys1 or not keys2:
                return 0.0
            intersection = keys1.intersection(keys2)
            union = keys1.union(keys2)
            if not union:
                return 0.0
            return len(intersection) / len(union)
        except Exception:
            return 0.0

    def _calculate_feature_similarity(
        self, features1: Any, features2: Any
    ) -> float:
        """Расчет схожести признаков."""
        try:
            # Простая метрика схожести на основе числовых значений
            if not hasattr(features1, '__dict__') or not hasattr(features2, '__dict__'):
                return 0.5
            
            # Извлекаем числовые значения признаков
            values1 = []
            values2 = []
            
            # Основные признаки
            for attr in ['book_pressure', 'volume_delta', 'price_reaction', 'spread_change', 
                        'order_imbalance', 'liquidity_depth', 'volume_concentration', 'price_volatility']:
                if hasattr(features1, attr) and hasattr(features2, attr):
                    try:
                        values1.append(float(getattr(features1, attr)))
                        values2.append(float(getattr(features2, attr)))
                    except (ValueError, TypeError):
                        continue
            
            if not values1 or not values2:
                return 0.5
            
            # Рассчитываем евклидово расстояние
            if len(values1) != len(values2):
                return 0.5
            
            distance = sum((a - b) ** 2 for a, b in zip(values1, values2)) ** 0.5
            max_distance = sum(max(a, b) ** 2 for a, b in zip(values1, values2)) ** 0.5
            
            if max_distance == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.5

    def _calculate_temporal_similarity(
        self, timestamp1: datetime, timestamp2: datetime
    ) -> float:
        """Расчет временной схожести."""
        try:
            # Разница во времени в часах
            time_diff = abs((timestamp1 - timestamp2).total_seconds() / 3600)
            
            # Экспоненциальное затухание
            # Чем больше разница во времени, тем меньше схожесть
            similarity = float(np.exp(-time_diff / 24.0))  # 24 часа как базовая единица
            
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.5

    async def predict_pattern_outcome(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> PatternResult:
        """
        Предсказание исхода паттерна на основе исторических данных.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Предсказанный результат
        """
        try:
            if not historical_patterns:
                return self._create_neutral_result()
            # Находим наиболее похожие паттерны
            similar_patterns = await self._find_most_similar_patterns(
                pattern, historical_patterns
            )
            if not similar_patterns:
                return self._create_neutral_result()
            # Анализируем результаты похожих паттернов
            outcomes: List[PatternResult] = []
            weights = []
            for hist_pattern in similar_patterns:
                if hist_pattern.result:
                    # Преобразуем PatternResultProtocol в PatternResult
                    if hasattr(hist_pattern.result, 'outcome'):
                        # Проверяем, что результат является PatternResult
                        if isinstance(hist_pattern.result, PatternResult):
                            outcomes.append(hist_pattern.result)
                        else:
                            # Создаем PatternResult из протокола
                            outcomes.append(PatternResult(
                                outcome=hist_pattern.result.outcome,
                                price_change_5min=hist_pattern.result.price_change_5min,
                                price_change_15min=hist_pattern.result.price_change_15min,
                                price_change_30min=hist_pattern.result.price_change_30min,
                                volume_change=hist_pattern.result.volume_change,
                                volatility_change=hist_pattern.result.volatility_change
                            ))
                    # Вес на основе схожести и точности
                    similarity = await self.analyze_pattern_similarity(
                        pattern, hist_pattern.pattern
                    )
                    weight = float(similarity) * float(hist_pattern.accuracy)
                    weights.append(weight)
            if not outcomes:
                return self._create_neutral_result()
            # Рассчитываем взвешенный результат
            weighted_outcome = self._calculate_weighted_outcome(outcomes, weights)
            # Применяем рыночные корректировки
            adjusted_outcome = await self._apply_market_adjustments(
                pattern, weighted_outcome
            )
            self.logger.debug(f"Predicted outcome: {adjusted_outcome.outcome.value}")
            return adjusted_outcome
        except Exception as e:
            self.logger.error(f"Failed to predict pattern outcome: {e}")
            return self._create_neutral_result()

    async def _find_most_similar_patterns(
        self,
        pattern: MarketMakerPattern,
        historical_patterns: List[PatternMemory],
        max_patterns: int = 10,
    ) -> List[PatternMemory]:
        """Поиск наиболее похожих паттернов."""
        similarities = []
        for hist_pattern in historical_patterns:
            if hist_pattern.result:  # Только паттерны с результатами
                similarity = await self.analyze_pattern_similarity(
                    pattern, hist_pattern.pattern
                )
                similarities.append((hist_pattern, float(similarity)))
        # Сортируем по схожести и берем топ
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in similarities[:max_patterns]]

    def _calculate_weighted_outcome(
        self, outcomes: List[PatternResult], weights: List[float]
    ) -> PatternResult:
        """Расчет взвешенного результата."""
        if not outcomes or not weights:
            return self._create_neutral_result()
        # Нормализуем веса
        total_weight = sum(weights)
        if total_weight == 0:
            return self._create_neutral_result()
        normalized_weights = [w / total_weight for w in weights]
        # Рассчитываем взвешенные значения
        weighted_price_change_5min = sum(
            o.price_change_5min * w for o, w in zip(outcomes, normalized_weights)
        )
        weighted_price_change_15min = sum(
            o.price_change_15min * w for o, w in zip(outcomes, normalized_weights)
        )
        weighted_price_change_30min = sum(
            o.price_change_30min * w for o, w in zip(outcomes, normalized_weights)
        )
        weighted_volume_change = sum(
            o.volume_change * w for o, w in zip(outcomes, normalized_weights)
        )
        weighted_volatility_change = sum(
            o.volatility_change * w for o, w in zip(outcomes, normalized_weights)
        )
        # Определяем исход на основе взвешенной доходности
        if weighted_price_change_15min > 0.005:
            outcome = PatternOutcome.SUCCESS
        elif weighted_price_change_15min < -0.005:
            outcome = PatternOutcome.FAILURE
        else:
            outcome = PatternOutcome.NEUTRAL
        return PatternResult(
            outcome=outcome,
            price_change_5min=weighted_price_change_5min,
            price_change_15min=weighted_price_change_15min,
            price_change_30min=weighted_price_change_30min,
            volume_change=weighted_volume_change,
            volatility_change=weighted_volatility_change,
            market_context={},
        )

    async def _apply_market_adjustments(
        self, pattern: MarketMakerPattern, outcome: PatternResult
    ) -> PatternResult:
        """Применение рыночных корректировок к результату."""
        try:
            # Анализируем текущий рыночный контекст
            market_context = await self.analyze_market_context(
                pattern.symbol, pattern.timestamp
            )
            # Применяем корректировки на основе рыночной фазы
            phase_adjustment = self._get_phase_adjustment(
                market_context.get("market_phase", "transition")
            )
            # Применяем корректировки на основе волатильности
            volatility_adjustment = self._get_volatility_adjustment(
                market_context.get("volatility_regime", "medium")
            )
            # Применяем корректировки на основе ликвидности
            liquidity_adjustment = self._get_liquidity_adjustment(
                market_context.get("liquidity_regime", "medium")
            )
            # Общая корректировка
            total_adjustment = (
                phase_adjustment * volatility_adjustment * liquidity_adjustment
            )
            # Применяем корректировку к доходности
            adjusted_price_change_15min = outcome.price_change_15min * total_adjustment
            # Обновляем исход если необходимо
            if (
                adjusted_price_change_15min > 0.005
                and outcome.outcome != PatternOutcome.SUCCESS
            ):
                outcome = PatternResult(
                    outcome=PatternOutcome.SUCCESS,
                    price_change_5min=outcome.price_change_5min * total_adjustment,
                    price_change_15min=adjusted_price_change_15min,
                    price_change_30min=outcome.price_change_30min * total_adjustment,
                    volume_change=outcome.volume_change,
                    volatility_change=outcome.volatility_change,
                    market_context=outcome.market_context,
                )
            elif (
                adjusted_price_change_15min < -0.005
                and outcome.outcome != PatternOutcome.FAILURE
            ):
                outcome = PatternResult(
                    outcome=PatternOutcome.FAILURE,
                    price_change_5min=outcome.price_change_5min * total_adjustment,
                    price_change_15min=adjusted_price_change_15min,
                    price_change_30min=outcome.price_change_30min * total_adjustment,
                    volume_change=outcome.volume_change,
                    volatility_change=outcome.volatility_change,
                    market_context=outcome.market_context,
                )
            return outcome
        except Exception as e:
            self.logger.error(f"Failed to apply market adjustments: {e}")
            return outcome

    def _get_phase_adjustment(self, phase: str) -> float:
        """Получение корректировки для рыночной фазы."""
        return self.config.get_market_phase_weight(phase)

    def _get_volatility_adjustment(self, regime: str) -> float:
        """Получение корректировки для режима волатильности."""
        return self.config.get_volatility_regime_weight(regime)

    def _get_liquidity_adjustment(self, regime: str) -> float:
        """Получение корректировки для режима ликвидности."""
        return self.config.get_liquidity_regime_weight(regime)

    def _create_neutral_result(self) -> PatternResult:
        """Создание нейтрального результата."""
        return PatternResult(
            outcome=PatternOutcome.NEUTRAL,
            price_change_5min=0.0,
            price_change_15min=0.0,
            price_change_30min=0.0,
            volume_change=0.0,
            volatility_change=0.0,
            market_context={},
        )

    async def analyze_market_context(
        self, symbol: str, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Анализ рыночного контекста.
        Args:
            symbol: Символ торговой пары
            timestamp: Временная метка
        Returns:
            Рыночный контекст
        """
        try:
            # Базовая реализация анализа контекста
            # В реальной системе здесь был бы анализ рыночных данных
            context = {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "market_phase": "transition",  # По умолчанию
                "volatility_regime": "medium",  # По умолчанию
                "liquidity_regime": "medium",  # По умолчанию
                "volume_profile": "normal",
                "price_action": "sideways",
                "order_flow": "balanced",
                "spread_behavior": "normal",
                "imbalance_behavior": "neutral",
                "pressure_behavior": "low",
            }
            # Анализируем время дня
            hour = timestamp.hour
            if 8 <= hour <= 16:
                context["market_phase"] = "active"
            elif 16 < hour <= 20:
                context["market_phase"] = "transition"
            else:
                context["market_phase"] = "quiet"
            self.logger.debug(f"Market context for {symbol}: {context}")
            return context
        except Exception as e:
            self.logger.error(f"Failed to analyze market context: {e}")
            return {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "market_phase": "transition",
                "volatility_regime": "medium",
                "liquidity_regime": "medium",
            }

    async def calculate_pattern_effectiveness(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> float:
        """
        Расчет эффективности паттерна.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Эффективность паттерна
        """
        try:
            if not historical_patterns:
                return 0.5
            # Анализируем похожие паттерны
            similar_patterns = await self._find_most_similar_patterns(
                pattern, historical_patterns
            )
            if not similar_patterns:
                return 0.5
            # Рассчитываем эффективность на основе результатов
            effectiveness_scores = []
            for hist_pattern in similar_patterns:
                if hist_pattern.result:
                    # Эффективность на основе доходности и точности
                    return_score = abs(hist_pattern.result.price_change_15min)
                    accuracy_score = float(hist_pattern.accuracy)
                    # Комбинированная оценка
                    effectiveness = return_score * 0.6 + accuracy_score * 0.4
                    effectiveness_scores.append(effectiveness)
            if not effectiveness_scores:
                return 0.5
            # Средняя эффективность
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            # Нормализуем результат
            normalized_effectiveness = min(1.0, max(0.0, avg_effectiveness))
            self.logger.debug(f"Pattern effectiveness: {normalized_effectiveness:.3f}")
            return normalized_effectiveness
        except Exception as e:
            self.logger.error(f"Failed to calculate pattern effectiveness: {e}")
            return 0.5

    async def get_pattern_recommendations(
        self, symbol: str, current_patterns: List[MarketMakerPattern]
    ) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций по паттернам.
        Args:
            symbol: Символ торговой пары
            current_patterns: Текущие паттерны
        Returns:
            Список рекомендаций
        """
        try:
            recommendations = []
            for pattern in current_patterns:
                # Анализируем каждый паттерн
                confidence = await self.calculate_pattern_confidence(pattern, [])
                predicted_outcome = await self.predict_pattern_outcome(pattern, [])
                effectiveness = await self.calculate_pattern_effectiveness(pattern, [])
                # Определяем уровень риска
                risk_level = self._determine_risk_level(confidence, effectiveness)
                # Генерируем рекомендацию
                recommendation = {
                    "pattern_type": pattern.pattern_type.value,
                    "symbol": symbol,
                    "confidence": float(confidence),
                    "predicted_outcome": predicted_outcome.outcome.value,
                    "expected_return": predicted_outcome.price_change_15min,
                    "effectiveness": effectiveness,
                    "risk_level": risk_level,
                    "recommended_action": self._get_recommended_action(
                        confidence, predicted_outcome
                    ),
                    "position_size": self._calculate_position_size(
                        confidence, risk_level
                    ),
                    "stop_loss": self._calculate_stop_loss(
                        predicted_outcome, risk_level
                    ),
                    "take_profit": self._calculate_take_profit(
                        predicted_outcome, risk_level
                    ),
                    "time_horizon": "15min",
                    "timestamp": pattern.timestamp.isoformat(),
                }
                recommendations.append(recommendation)
            # Сортируем по уверенности
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            self.logger.info(
                f"Generated {len(recommendations)} recommendations for {symbol}"
            )
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to get pattern recommendations: {e}")
            return []

    def _determine_risk_level(
        self, confidence: Confidence, effectiveness: float
    ) -> str:
        """Определение уровня риска."""
        confidence_score = float(confidence)
        combined_score = (confidence_score + effectiveness) / 2
        if combined_score >= 0.8:
            return "low"
        elif combined_score >= 0.6:
            return "medium"
        else:
            return "high"

    def _get_recommended_action(
        self, confidence: Confidence, outcome: PatternResult
    ) -> str:
        """Получение рекомендуемого действия."""
        confidence_score = float(confidence)
        if confidence_score < 0.5:
            return "hold"
        if outcome.outcome == PatternOutcome.SUCCESS:
            return "buy" if outcome.price_change_15min > 0 else "sell"
        elif outcome.outcome == PatternOutcome.FAILURE:
            return "sell" if outcome.price_change_15min > 0 else "buy"
        else:
            return "hold"

    def _calculate_position_size(
        self, confidence: Confidence, risk_level: str
    ) -> float:
        """Расчет размера позиции."""
        base_size = float(confidence)
        if risk_level == "low":
            return base_size
        elif risk_level == "medium":
            return base_size * 0.7
        else:
            return base_size * 0.4

    def _calculate_stop_loss(self, outcome: PatternResult, risk_level: str) -> float:
        """Расчет стоп-лосса."""
        base_loss = abs(outcome.price_change_15min) * 0.5
        if risk_level == "low":
            return base_loss * 0.8
        elif risk_level == "medium":
            return base_loss
        else:
            return base_loss * 1.2

    def _calculate_take_profit(self, outcome: PatternResult, risk_level: str) -> float:
        """Расчет тейк-профита."""
        base_profit = abs(outcome.price_change_15min)
        if risk_level == "low":
            return base_profit * 1.2
        elif risk_level == "medium":
            return base_profit
        else:
            return base_profit * 0.8
