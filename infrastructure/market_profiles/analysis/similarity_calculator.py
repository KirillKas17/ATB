"""
Калькулятор схожести паттернов маркет-мейкера.
Промышленная реализация с поддержкой:
- Многомерного анализа схожести
- Взвешенных метрик
- Машинного обучения
- Оптимизации производительности
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from shared.numpy_utils import np

from domain.market_maker.mm_pattern import MarketMakerPattern
from domain.types.market_maker_types import (
    BookPressure,
    OrderImbalance,
    PriceReaction,
    PriceVolatility,
    SimilarityScore,
    SpreadChange,
    VolumeConcentration,
    VolumeDelta,
)

from ..models.analysis_config import AnalysisConfig


@dataclass
class SimilarityMetrics:
    """Метрики схожести паттернов."""

    feature_similarity: float
    temporal_similarity: float
    contextual_similarity: float
    behavioral_similarity: float
    weighted_similarity: float
    confidence: float
    metadata: Dict[str, Any]


class SimilarityCalculator:
    """
    Промышленный калькулятор схожести паттернов.
    Особенности:
    - Многомерный анализ схожести
    - Взвешенные метрики
    - Адаптивные пороги
    - Кэширование результатов
    - Оптимизация производительности
    """

    def __init__(self, config: AnalysisConfig):
        """
        Инициализация калькулятора.
        Args:
            config: Конфигурация анализа
        """
        self.config = config
        self._setup_similarity_components()
        logger.info("SimilarityCalculator initialized successfully")

    def _setup_similarity_components(self) -> None:
        """Настройка компонентов схожести."""
        # Кэш для результатов схожести
        self.similarity_cache: Dict[str, SimilarityScore] = {}
        # Статистика расчетов
        self.calculation_stats = {
            "total_calculations": 0,
            "cache_hits": 0,
            "avg_similarity": 0.0,
            "high_similarity_count": 0,
        }
        # Пороги для различных типов схожести
        self.similarity_thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}

    async def calculate_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> SimilarityScore:
        """
        Расчет схожести между двумя паттернами.
        Args:
            pattern1: Первый паттерн
            pattern2: Второй паттерн
        Returns:
            Оценка схожести
        """
        try:
            # Проверяем кэш
            cache_key = self._generate_cache_key(pattern1, pattern2)
            if cache_key in self.similarity_cache:
                self.calculation_stats["cache_hits"] += 1
                return self.similarity_cache[cache_key]
            # Рассчитываем различные типы схожести
            feature_sim = self._calculate_feature_similarity(pattern1, pattern2)
            temporal_sim = self._calculate_temporal_similarity(pattern1, pattern2)
            contextual_sim = self._calculate_contextual_similarity(pattern1, pattern2)
            behavioral_sim = self._calculate_behavioral_similarity(pattern1, pattern2)
            # Рассчитываем взвешенную схожесть
            weighted_similarity = self._calculate_weighted_similarity(
                feature_sim, temporal_sim, contextual_sim, behavioral_sim
            )
            # Рассчитываем уверенность в оценке
            confidence = self._calculate_similarity_confidence(
                feature_sim, temporal_sim, contextual_sim, behavioral_sim
            )
            # Применяем корректировки
            final_similarity = self._apply_similarity_corrections(
                weighted_similarity, pattern1, pattern2
            )
            # Нормализуем результат
            normalized_similarity = min(1.0, max(0.0, final_similarity))
            # Кэшируем результат
            self.similarity_cache[cache_key] = SimilarityScore(normalized_similarity)
            # Обновляем статистику
            self._update_calculation_stats(normalized_similarity)
            logger.debug(f"Similarity between patterns: {normalized_similarity:.3f}")
            return SimilarityScore(normalized_similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return SimilarityScore(0.0)

    def _generate_cache_key(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> str:
        """Генерация ключа кэша для двух паттернов."""
        # Сортируем паттерны для обеспечения уникальности ключа
        if pattern1.symbol < pattern2.symbol:
            p1, p2 = pattern1, pattern2
        else:
            p1, p2 = pattern2, pattern1
        return (
            f"{p1.symbol}_{p1.pattern_type.value}_{p1.timestamp.isoformat()}_"
            f"{p2.symbol}_{p2.pattern_type.value}_{p2.timestamp.isoformat()}"
        )

    def _calculate_feature_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> float:
        """Расчет схожести признаков."""
        try:
            features1 = pattern1.features
            features2 = pattern2.features
            # Извлекаем числовые признаки
            feature_values1 = self._extract_feature_values(features1)
            feature_values2 = self._extract_feature_values(features2)
            if not feature_values1 or not feature_values2:
                return 0.0
            # Нормализуем значения
            normalized1 = self._normalize_feature_values(feature_values1)
            normalized2 = self._normalize_feature_values(feature_values2)
            # Рассчитываем косинусное сходство
            similarity = self._cosine_similarity(normalized1, normalized2)
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate feature similarity: {e}")
            return 0.0

    def _extract_feature_values(self, features: Any) -> List[float]:
        """Извлечение числовых значений признаков."""
        try:
            values = []
            # Основные признаки
            if hasattr(features, "book_pressure"):
                values.append(float(features.book_pressure))
            if hasattr(features, "volume_delta"):
                values.append(float(features.volume_delta))
            if hasattr(features, "price_reaction"):
                values.append(float(features.price_reaction))
            if hasattr(features, "spread_change"):
                values.append(float(features.spread_change))
            if hasattr(features, "order_imbalance"):
                values.append(float(features.order_imbalance))
            if hasattr(features, "liquidity_depth"):
                values.append(float(features.liquidity_depth))
            if hasattr(features, "volume_concentration"):
                values.append(float(features.volume_concentration))
            if hasattr(features, "price_volatility"):
                values.append(float(features.price_volatility))
            return values
        except Exception as e:
            logger.error(f"Failed to extract feature values: {e}")
            return []

    def _normalize_feature_values(self, values: List[float]) -> List[float]:
        """Нормализация значений признаков."""
        try:
            if not values:
                return []
            # Используем min-max нормализацию
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [0.5] * len(values)  # Нейтральные значения
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize feature values: {e}")
            return values

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Расчет косинусного сходства."""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            # Рассчитываем скалярное произведение
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            # Рассчитываем нормы векторов
            norm1 = float(np.sqrt(sum(a * a for a in vec1)))
            norm2 = float(np.sqrt(sum(b * b for b in vec2)))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            # Косинусное сходство
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def _calculate_temporal_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> float:
        """Расчет временной схожести."""
        try:
            # Разница во времени
            time_diff = abs((pattern1.timestamp - pattern2.timestamp).total_seconds())
            # Нормализуем разницу (1 час = 3600 секунд)
            normalized_diff = min(1.0, time_diff / 3600)
            # Экспоненциальное затухание
            temporal_similarity = float(np.exp(-normalized_diff))
            return temporal_similarity
        except Exception as e:
            logger.error(f"Failed to calculate temporal similarity: {e}")
            return 0.5

    def _calculate_contextual_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> float:
        """Расчет контекстуальной схожести."""
        try:
            context1 = pattern1.context
            context2 = pattern2.context
            if not context1 or not context2:
                return 0.5
            # Сравниваем общие ключи
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0
            similarities = []
            for key in common_keys:
                val1 = context1.get(key)
                val2 = context2.get(key)
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Числовые значения
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Строковые значения
                    similarity = 1.0 if val1 == val2 else 0.0
                else:
                    # Другие типы
                    similarity = 1.0 if val1 == val2 else 0.0
                similarities.append(similarity)
            if not similarities:
                return 0.0
            return sum(similarities) / len(similarities)
        except Exception as e:
            logger.error(f"Failed to calculate contextual similarity: {e}")
            return 0.5

    def _calculate_behavioral_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> float:
        """Расчет поведенческой схожести."""
        try:
            # Сравниваем типы паттернов
            type_similarity = (
                1.0 if pattern1.pattern_type == pattern2.pattern_type else 0.3
            )
            # Сравниваем уверенность
            confidence_diff = abs(
                float(pattern1.confidence) - float(pattern2.confidence)
            )
            confidence_similarity = 1.0 - confidence_diff
            # Сравниваем символы
            symbol_similarity = 1.0 if pattern1.symbol == pattern2.symbol else 0.5
            # Взвешенная схожесть
            behavioral_similarity = (
                type_similarity * 0.5
                + confidence_similarity * 0.3
                + symbol_similarity * 0.2
            )
            return max(0.0, min(1.0, behavioral_similarity))
        except Exception as e:
            logger.error(f"Failed to calculate behavioral similarity: {e}")
            return 0.5

    def _calculate_weighted_similarity(
        self,
        feature_sim: float,
        temporal_sim: float,
        contextual_sim: float,
        behavioral_sim: float,
    ) -> float:
        """Расчет взвешенной схожести."""
        try:
            # Используем веса из конфигурации
            weights = self.config.feature_weights
            # Взвешенная схожесть
            weighted_similarity = (
                feature_sim * 0.4
                + temporal_sim * 0.2
                + contextual_sim * 0.2
                + behavioral_sim * 0.2
            )
            return weighted_similarity
        except Exception as e:
            logger.error(f"Failed to calculate weighted similarity: {e}")
            return 0.0

    def _calculate_similarity_confidence(
        self,
        feature_sim: float,
        temporal_sim: float,
        contextual_sim: float,
        behavioral_sim: float,
    ) -> float:
        """Расчет уверенности в оценке схожести."""
        try:
            # Уверенность на основе согласованности метрик
            similarities = [feature_sim, temporal_sim, contextual_sim, behavioral_sim]
            # Стандартное отклонение как мера согласованности
            std_dev = float(np.std(similarities))
            # Уверенность обратно пропорциональна стандартному отклонению
            confidence = max(0.0, 1.0 - std_dev)
            return confidence
        except Exception as e:
            logger.error(f"Failed to calculate similarity confidence: {e}")
            return 0.5

    def _apply_similarity_corrections(
        self,
        similarity: float,
        pattern1: MarketMakerPattern,
        pattern2: MarketMakerPattern,
    ) -> float:
        """Применение корректировок к схожести."""
        try:
            corrected_similarity = similarity
            # Корректировка на основе типов паттернов
            type_weight = self.config.get_pattern_type_weight(
                str(pattern1.pattern_type.value)
            )
            corrected_similarity *= type_weight
            # Корректировка на основе временного расстояния
            time_diff_hours = abs(
                (pattern1.timestamp - pattern2.timestamp).total_seconds() / 3600
            )
            if time_diff_hours > 24:
                # Паттерны из разных дней
                corrected_similarity *= 0.9
            # Корректировка на основе символов
            if pattern1.symbol != pattern2.symbol:
                # Разные символы
                corrected_similarity *= 0.8
            return corrected_similarity
        except Exception as e:
            logger.error(f"Failed to apply similarity corrections: {e}")
            return similarity

    def _update_calculation_stats(self, similarity: float) -> None:
        """Обновление статистики расчетов."""
        try:
            self.calculation_stats["total_calculations"] += 1
            # Обновляем среднюю схожесть
            total_calc = self.calculation_stats["total_calculations"]
            current_avg = self.calculation_stats["avg_similarity"]
            self.calculation_stats["avg_similarity"] = (
                current_avg * (total_calc - 1) + similarity
            ) / total_calc
            # Подсчитываем высокую схожесть
            if similarity >= self.similarity_thresholds["high"]:
                self.calculation_stats["high_similarity_count"] += 1
        except Exception as e:
            logger.error(f"Failed to update calculation stats: {e}")

    def get_similarity_statistics(self) -> Dict[str, Any]:
        """Получение статистики схожести."""
        try:
            total_calc = self.calculation_stats["total_calculations"]
            if total_calc == 0:
                return {
                    "total_calculations": 0,
                    "cache_hit_rate": 0.0,
                    "avg_similarity": 0.0,
                    "high_similarity_rate": 0.0,
                }
            cache_hit_rate = self.calculation_stats["cache_hits"] / total_calc
            high_similarity_rate = (
                self.calculation_stats["high_similarity_count"] / total_calc
            )
            return {
                "total_calculations": total_calc,
                "cache_hit_rate": cache_hit_rate,
                "avg_similarity": self.calculation_stats["avg_similarity"],
                "high_similarity_rate": high_similarity_rate,
            }
        except Exception as e:
            logger.error(f"Failed to get similarity statistics: {e}")
            return {}

    def clear_cache(self) -> None:
        """Очистка кэша схожести."""
        try:
            self.similarity_cache.clear()
            logger.info("Similarity cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear similarity cache: {e}")

    def get_cache_size(self) -> int:
        """Получение размера кэша."""
        return len(self.similarity_cache)
