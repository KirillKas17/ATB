"""
Доменный сервис для работы с паттернами торговли.
Устраняет дублирование кода между различными компонентами.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from shared.numpy_utils import np
from scipy.signal import find_peaks

from domain.entities.pattern import Pattern, PatternConfidence
from domain.types.pattern_types import PatternType
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume


@dataclass
class PatternSearchCriteria:
    """Критерии поиска паттернов."""

    pattern_type: Optional[PatternType] = None
    min_confidence: Optional[PatternConfidence] = None
    min_similarity: Optional[float] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    trading_pair_id: Optional[str] = None
    min_volume: Optional[Volume] = None
    max_volume: Optional[Volume] = None
    min_price: Optional[Price] = None
    max_price: Optional[Price] = None


@dataclass
class PatternMatchResult:
    """Результат поиска похожих паттернов."""

    pattern: Pattern
    similarity_score: float
    confidence_score: float
    match_metadata: Dict[str, Any]


class PatternAnalyzer(ABC):
    """Абстрактный анализатор паттернов."""

    @abstractmethod
    def analyze_pattern(self, data: Dict[str, Any]) -> Pattern:
        """Анализ паттерна в данных."""
        pass

    @abstractmethod
    def calculate_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """Вычисление схожести между паттернами."""
        pass


class TechnicalPatternAnalyzer(PatternAnalyzer):
    """Анализатор технических паттернов."""

    def __init__(self) -> None:
        self.similarity_threshold = 0.7
        self.confidence_weights = {
            "volume": 0.3,
            "price_movement": 0.4,
            "time_duration": 0.2,
            "volatility": 0.1,
        }

    def analyze_pattern(self, data: Dict[str, Any]) -> Pattern:
        """Анализ технического паттерна."""
        try:
            # Извлечение данных
            prices = np.array(data.get("prices", []))
            volumes = np.array(data.get("volumes", []))
            timestamps = data.get("timestamps", [])
            if len(prices) < 2:
                raise ValueError("Insufficient data for pattern analysis")
            # Определение типа паттерна
            pattern_type = self._identify_pattern_type(prices, volumes)
            # Вычисление характеристик
            characteristics = self._calculate_characteristics(
                prices, volumes, timestamps
            )
            # Вычисление уверенности
            confidence = self._calculate_confidence(characteristics)
            # Создание паттерна
            pattern = Pattern(
                id=uuid4(),
                pattern_type=pattern_type,
                characteristics=characteristics,
                confidence=confidence,
                trading_pair_id=data.get("trading_pair_id", ""),
                created_at=datetime.now(),
                metadata={
                    "data_points": len(prices),
                    "time_range": {
                        "start": timestamps[0] if timestamps else None,
                        "end": timestamps[-1] if timestamps else None,
                    },
                    "analysis_method": "technical",
                },
            )
            return pattern
        except Exception as e:
            # Создаем паттерн с минимальной уверенностью при ошибке
            return Pattern(
                id=uuid4(),
                pattern_type=PatternType.UNKNOWN,
                characteristics={},
                confidence=PatternConfidence.LOW,
                trading_pair_id=data.get("trading_pair_id", ""),
                created_at=datetime.now(),
                metadata={"error": str(e)},
            )

    def calculate_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """Вычисление схожести между паттернами."""
        try:
            if pattern1.pattern_type != pattern2.pattern_type:
                return 0.0
            # Сравнение характеристик
            similarity_scores = []
            # Сравнение ценовых движений
            if (
                "price_movement" in pattern1.characteristics
                and "price_movement" in pattern2.characteristics
            ):
                price_similarity = self._compare_price_movements(
                    pattern1.characteristics["price_movement"],
                    pattern2.characteristics["price_movement"],
                )
                similarity_scores.append(
                    price_similarity * self.confidence_weights["price_movement"]
                )
            # Сравнение объемов
            if (
                "volume_profile" in pattern1.characteristics
                and "volume_profile" in pattern2.characteristics
            ):
                volume_similarity = self._compare_volume_profiles(
                    pattern1.characteristics["volume_profile"],
                    pattern2.characteristics["volume_profile"],
                )
                similarity_scores.append(
                    volume_similarity * self.confidence_weights["volume"]
                )
            # Сравнение временных характеристик
            if (
                "duration" in pattern1.characteristics
                and "duration" in pattern2.characteristics
            ):
                time_similarity = self._compare_durations(
                    pattern1.characteristics["duration"],
                    pattern2.characteristics["duration"],
                )
                similarity_scores.append(
                    time_similarity * self.confidence_weights["time_duration"]
                )
            # Сравнение волатильности
            if (
                "volatility" in pattern1.characteristics
                and "volatility" in pattern2.characteristics
            ):
                volatility_similarity = self._compare_volatility(
                    pattern1.characteristics["volatility"],
                    pattern2.characteristics["volatility"],
                )
                similarity_scores.append(
                    volatility_similarity * self.confidence_weights["volatility"]
                )
            # Вычисление общего счета схожести
            if similarity_scores:
                return sum(similarity_scores)
            else:
                return 0.0
        except Exception:
            return 0.0

    def _identify_pattern_type(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> PatternType:
        """Определение типа паттерна."""
        try:
            # Вычисление производных
            price_changes = np.diff(prices)
            volume_changes = np.diff(volumes)
            # Анализ тренда
            trend = np.mean(price_changes)
            trend_strength = (
                abs(trend) / np.std(price_changes) if np.std(price_changes) > 0 else 0
            )
            # Анализ объема
            volume_trend = np.mean(volume_changes)
            # Определение паттерна на основе характеристик
            if trend_strength > 1.5:
                if trend > 0:
                    return PatternType.TREND
                else:
                    return PatternType.COUNTER_TREND
            elif self._detect_reversal(prices, volumes):
                return PatternType.REVERSAL
            elif self._detect_consolidation(prices, volumes):
                return PatternType.CONSOLIDATION
            elif self._detect_breakout(prices, volumes):
                return PatternType.BREAKOUT
            else:
                return PatternType.SIDEWAYS
        except Exception:
            return PatternType.UNKNOWN

    def _calculate_characteristics(
        self, prices: np.ndarray, volumes: np.ndarray, timestamps: List
    ) -> Dict[str, Any]:
        """Вычисление характеристик паттерна."""
        characteristics: Dict[str, Any] = {}
        try:
            # Ценовые характеристики
            price_changes = np.diff(prices)
            characteristics["price_movement"] = {
                "trend": float(np.mean(price_changes)),
                "volatility": float(np.std(price_changes)),
                "max_change": float(np.max(np.abs(price_changes))),
                "min_change": float(np.min(price_changes)),
                "range": float(np.max(prices) - np.min(prices)),
            }
            # Объемные характеристики
            characteristics["volume_profile"] = {
                "mean_volume": float(np.mean(volumes)),
                "volume_trend": float(np.mean(np.diff(volumes))),
                "volume_volatility": float(np.std(volumes)),
                "max_volume": float(np.max(volumes)),
                "min_volume": float(np.min(volumes)),
            }
            # Временные характеристики
            if timestamps and len(timestamps) > 1:
                duration = (timestamps[-1] - timestamps[0]).total_seconds()
                characteristics["duration"] = {
                    "seconds": duration,
                    "minutes": duration / 60,
                    "hours": duration / 3600,
                }
            # Волатильность
            characteristics["volatility"] = {
                "price_volatility": (
                    float(np.std(prices) / np.mean(prices))
                    if np.mean(prices) > 0
                    else 0
                ),
                "volume_volatility": (
                    float(np.std(volumes) / np.mean(volumes))
                    if np.mean(volumes) > 0
                    else 0
                ),
            }
            # Дополнительные метрики
            characteristics["statistics"] = {
                "peaks": len(find_peaks(prices)[0]),
                "troughs": len(find_peaks(-prices)[0]),
                "correlation_price_volume": (
                    float(np.corrcoef(prices, volumes)[0, 1]) if len(prices) > 1 else 0
                ),
            }
        except Exception as e:
            characteristics["error"] = str(e)
        return characteristics

    def _calculate_confidence(
        self, characteristics: Dict[str, Any]
    ) -> PatternConfidence:
        """Вычисление уверенности в паттерне."""
        try:
            confidence_score = 0.0
            # Оценка качества данных
            if "price_movement" in characteristics:
                volatility = characteristics["price_movement"].get("volatility", 0)
                if volatility > 0:
                    confidence_score += 0.3
            if "volume_profile" in characteristics:
                volume_volatility = characteristics["volume_profile"].get(
                    "volume_volatility", 0
                )
                if volume_volatility > 0:
                    confidence_score += 0.2
            if "statistics" in characteristics:
                correlation = abs(
                    characteristics["statistics"].get("correlation_price_volume", 0)
                )
                confidence_score += correlation * 0.3
            if "duration" in characteristics:
                duration_hours = characteristics["duration"].get("hours", 0)
                if duration_hours > 0.1:  # Минимум 6 минут
                    confidence_score += 0.2
            # Определение уровня уверенности
            if confidence_score >= 0.8:
                return PatternConfidence.HIGH
            elif confidence_score >= 0.6:
                return PatternConfidence.MEDIUM
            elif confidence_score >= 0.4:
                return PatternConfidence.LOW
            else:
                return PatternConfidence.UNKNOWN
        except Exception:
            return PatternConfidence.UNKNOWN

    def _detect_reversal(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Определение разворота."""
        try:
            if len(prices) < 5:
                return False
            # Анализ последних точек
            recent_prices = prices[-5:]
            recent_volumes = volumes[-5:]
            # Проверка на разворот
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            # Разворот: изменение тренда с увеличением объема
            return bool(abs(price_trend) > 0.1 and volume_trend > 0)
        except Exception:
            return False

    def _detect_consolidation(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Определение консолидации."""
        try:
            if len(prices) < 3:
                return False
            # Низкая волатильность цены
            price_volatility = (
                np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            )
            # Стабильный объем
            volume_volatility = (
                np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            )
            return bool(price_volatility < 0.02 and volume_volatility < 0.5)
        except Exception:
            return False

    def _detect_breakout(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Определение пробоя."""
        try:
            if len(prices) < 10:
                return False
            # Анализ последних точек относительно предыдущих
            early_prices = prices[:-3]
            late_prices = prices[-3:]
            early_mean = np.mean(early_prices)
            late_mean = np.mean(late_prices)
            # Значительное отклонение от среднего
            deviation = (
                abs(late_mean - early_mean) / np.std(early_prices)
                if np.std(early_prices) > 0
                else 0
            )
            # Увеличение объема
            volume_increase = np.mean(volumes[-3:]) > np.mean(volumes[:-3]) * 1.5
            return bool(deviation > 2.0 and volume_increase)
        except Exception:
            return False

    def _compare_price_movements(self, movement1: Dict, movement2: Dict) -> float:
        """Сравнение ценовых движений."""
        try:
            # Сравнение трендов
            trend_diff = abs(movement1.get("trend", 0) - movement2.get("trend", 0))
            trend_similarity = max(
                0,
                1
                - trend_diff
                / max(
                    abs(movement1.get("trend", 0)),
                    abs(movement2.get("trend", 0)),
                    0.001,
                ),
            )
            # Сравнение волатильности
            vol_diff = abs(
                movement1.get("volatility", 0) - movement2.get("volatility", 0)
            )
            vol_similarity = max(
                0,
                1
                - vol_diff
                / max(
                    movement1.get("volatility", 0),
                    movement2.get("volatility", 0),
                    0.001,
                ),
            )
            return float((trend_similarity + vol_similarity) / 2)
        except Exception:
            return 0.0

    def _compare_volume_profiles(self, profile1: Dict, profile2: Dict) -> float:
        """Сравнение профилей объема."""
        try:
            # Сравнение средних объемов
            mean_diff = abs(
                profile1.get("mean_volume", 0) - profile2.get("mean_volume", 0)
            )
            mean_similarity = max(
                0,
                1
                - mean_diff
                / max(
                    profile1.get("mean_volume", 0),
                    profile2.get("mean_volume", 0),
                    0.001,
                ),
            )
            # Сравнение трендов объема
            trend_diff = abs(
                profile1.get("volume_trend", 0) - profile2.get("volume_trend", 0)
            )
            trend_similarity = max(
                0,
                1
                - trend_diff
                / max(
                    abs(profile1.get("volume_trend", 0)),
                    abs(profile2.get("volume_trend", 0)),
                    0.001,
                ),
            )
            return float((mean_similarity + trend_similarity) / 2)
        except Exception:
            return 0.0

    def _compare_durations(self, duration1: Dict, duration2: Dict) -> float:
        """Сравнение длительностей."""
        try:
            hours1 = duration1.get("hours", 0)
            hours2 = duration2.get("hours", 0)
            if hours1 == 0 and hours2 == 0:
                return 1.0
            duration_diff = abs(hours1 - hours2)
            max_duration = max(hours1, hours2)
            return float(max(0, 1 - duration_diff / max_duration))
        except Exception:
            return 0.0

    def _compare_volatility(self, volatility1: Dict, volatility2: Dict) -> float:
        """Сравнение волатильности."""
        try:
            price_vol1 = volatility1.get("price_volatility", 0)
            price_vol2 = volatility2.get("price_volatility", 0)
            vol_diff = abs(price_vol1 - price_vol2)
            max_vol = max(price_vol1, price_vol2)
            return float(max(0, 1 - vol_diff / max_vol)) if max_vol > 0 else 1.0
        except Exception:
            return 0.0


class PatternSearchService:
    """Сервис поиска паттернов."""

    def __init__(self, analyzer: PatternAnalyzer):
        self.analyzer = analyzer
        self.patterns: Dict[UUID, Pattern] = {}

    def add_pattern(self, pattern: Pattern) -> None:
        """Добавление паттерна в хранилище."""
        self.patterns[pattern.id] = pattern

    def find_similar_patterns(
        self, target_pattern: Pattern, criteria: PatternSearchCriteria
    ) -> List[PatternMatchResult]:
        """Поиск похожих паттернов."""
        results = []
        for pattern in self.patterns.values():
            # Применение фильтров
            if not self._matches_criteria(pattern, criteria):
                continue
            # Вычисление схожести
            similarity = self.analyzer.calculate_similarity(target_pattern, pattern)
            if similarity >= (criteria.min_similarity or 0.5):
                results.append(
                    PatternMatchResult(
                        pattern=pattern,
                        similarity_score=similarity,
                        confidence_score=pattern.confidence.value,
                        match_metadata={
                            "analysis_method": "technical",
                            "matched_at": datetime.now().isoformat(),
                        },
                    )
                )
        # Сортировка по схожести
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results

    def _matches_criteria(
        self, pattern: Pattern, criteria: PatternSearchCriteria
    ) -> bool:
        """Проверка соответствия паттерна критериям."""
        # Фильтр по типу паттерна
        if criteria.pattern_type and pattern.pattern_type != criteria.pattern_type:
            return False
        # Фильтр по уверенности
        if (
            criteria.min_confidence
            and pattern.confidence.value < criteria.min_confidence.value
        ):
            return False
        # Фильтр по торговой паре
        if (
            criteria.trading_pair_id
            and pattern.trading_pair_id != criteria.trading_pair_id
        ):
            return False
        # Фильтр по времени
        if criteria.time_range:
            start_time, end_time = criteria.time_range
            if pattern.created_at < start_time or pattern.created_at > end_time:
                return False
        # Фильтр по объему
        if criteria.min_volume or criteria.max_volume:
            volume = pattern.characteristics.get("volume_profile", {}).get(
                "mean_volume", 0
            )
            if criteria.min_volume and volume < criteria.min_volume.value:
                return False
            if criteria.max_volume and volume > criteria.max_volume.value:
                return False
        # Фильтр по цене
        if criteria.min_price or criteria.max_price:
            price_range = pattern.characteristics.get("price_movement", {}).get(
                "range", 0
            )
            if criteria.min_price and price_range < criteria.min_price.value:
                return False
            if criteria.max_price and price_range > criteria.max_price.value:
                return False
        return True
