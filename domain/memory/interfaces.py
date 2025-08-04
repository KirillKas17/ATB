# -*- coding: utf-8 -*-
"""Интерфейсы для модуля памяти паттернов."""
from typing import Any, Dict, List, Optional, Protocol, Tuple
from shared.numpy_utils import np

from domain.memory.entities import PatternOutcome, PatternSnapshot, PredictionResult
from domain.memory.types import (
    MarketFeatures,
    MemoryStatistics,
    SimilarityMetrics,
)
from domain.type_definitions.pattern_types import PatternType


class IPatternMemoryRepository(Protocol):
    """Интерфейс репозитория памяти паттернов."""

    def save_snapshot(self, pattern_id: str, snapshot: PatternSnapshot) -> bool:
        """Сохранение снимка паттерна."""
        ...

    def save_outcome(self, pattern_id: str, outcome: PatternOutcome) -> bool:
        """Сохранение исхода паттерна."""
        ...

    def get_snapshots(
        self,
        symbol: str,
        pattern_type: Optional[PatternType] = None,
        limit: Optional[int] = None,
    ) -> List[PatternSnapshot]:
        """Получение снимков паттернов."""
        ...

    def get_outcomes(self, pattern_ids: List[str]) -> List[PatternOutcome]:
        """Получение исходов паттернов."""
        ...

    def get_statistics(self) -> MemoryStatistics:
        """Получение статистики."""
        ...

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Очистка старых данных."""
        ...


class IPatternMatcher(Protocol):
    """Интерфейс для сопоставления паттернов."""

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Вычисление сходства между векторами."""
        ...

    def find_similar_patterns(
        self,
        current_features: MarketFeatures,
        snapshots: List[PatternSnapshot],
        similarity_threshold: float = 0.9,
        max_results: int = 10,
    ) -> List[Tuple[PatternSnapshot, float]]:
        """Поиск похожих паттернов."""
        ...

    def calculate_confidence_boost(
        self, similarity: float, snapshot: PatternSnapshot
    ) -> float:
        """Вычисление повышения уверенности."""
        ...

    def calculate_signal_strength(self, snapshot: PatternSnapshot) -> float:
        """Вычисление силы сигнала."""
        ...


class IPatternMemoryService(Protocol):
    """Интерфейс сервиса памяти паттернов."""

    def match_snapshot(
        self,
        current_features: MarketFeatures,
        symbol: str,
        pattern_type: Optional[PatternType] = None,
    ) -> Optional[PredictionResult]:
        """Сопоставление снимка с историческими паттернами."""
        ...

    def save_pattern_data(self, pattern_id: str, snapshot: PatternSnapshot) -> bool:
        """Сохранение данных паттерна."""
        ...

    def update_pattern_outcome(self, pattern_id: str, outcome: PatternOutcome) -> bool:
        """Обновление исхода паттерна."""
        ...

    def get_pattern_statistics(
        self, symbol: str, pattern_type: Optional[PatternType] = None
    ) -> Dict[str, Any]:
        """Получение статистики паттернов."""
        ...

    def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """Очистка старых паттернов."""
        ...


class IPatternPredictor(Protocol):
    """Интерфейс для прогнозирования на основе паттернов."""

    def generate_prediction(
        self,
        similar_cases: List[Tuple[PatternSnapshot, float]],
        outcomes: List[PatternOutcome],
        current_features: MarketFeatures,
        symbol: str,
    ) -> Optional[PredictionResult]:
        """Генерация прогноза на основе похожих случаев."""
        ...

    def calculate_prediction_confidence(
        self,
        similar_cases: List[Tuple[PatternSnapshot, float]],
        outcomes: List[PatternOutcome],
    ) -> float:
        """Вычисление уверенности прогноза."""
        ...

    def calculate_predicted_return(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> float:
        """Вычисление прогнозируемой доходности."""
        ...

    def calculate_predicted_duration(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> int:
        """Вычисление прогнозируемой длительности."""
        ...


class IPatternMemoryAnalyzer(Protocol):
    """Интерфейс для анализа памяти паттернов."""

    def analyze_pattern_effectiveness(
        self, symbol: str, pattern_type: PatternType
    ) -> Dict[str, Any]:
        """Анализ эффективности паттернов."""
        ...

    def analyze_market_regime_patterns(self, symbol: str) -> Dict[str, Any]:
        """Анализ паттернов рыночных режимов."""
        ...

    def analyze_volume_profile_patterns(self, symbol: str) -> Dict[str, Any]:
        """Анализ паттернов профилей объема."""
        ...

    def get_pattern_correlation_matrix(self, symbol: str) -> np.ndarray:
        """Получение матрицы корреляций паттернов."""
        ...

    def identify_pattern_clusters(
        self, symbol: str, pattern_type: PatternType
    ) -> List[Dict[str, Any]]:
        """Идентификация кластеров паттернов."""
        ...


class IPatternMemoryOptimizer(Protocol):
    """Интерфейс для оптимизации памяти паттернов."""

    def optimize_similarity_threshold(
        self, symbol: str, pattern_type: PatternType
    ) -> float:
        """Оптимизация порога сходства."""
        ...

    def optimize_feature_weights(
        self, symbol: str, pattern_type: PatternType
    ) -> np.ndarray:
        """Оптимизация весов признаков."""
        ...

    def optimize_prediction_parameters(
        self, symbol: str, pattern_type: PatternType
    ) -> Dict[str, Any]:
        """Оптимизация параметров прогнозирования."""
        ...

    def validate_pattern_quality(self, snapshot: PatternSnapshot) -> bool:
        """Валидация качества паттерна."""
        ...

    def filter_noise_patterns(
        self, snapshots: List[PatternSnapshot]
    ) -> List[PatternSnapshot]:
        """Фильтрация шумовых паттернов."""
        ...
