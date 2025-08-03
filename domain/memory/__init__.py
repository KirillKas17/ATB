# -*- coding: utf-8 -*-
"""Модуль памяти паттернов для рыночной аналитики."""

from .interfaces import (
    IPatternMatcher,
    IPatternMemoryRepository,
    IPatternMemoryService,
)
from .pattern_memory import (
    MarketFeatures,
    OutcomeType,
    PatternMemory,
    PatternOutcome,
    PatternSnapshot,
    PredictionResult,
)
from .types import (
    MemoryStatistics,
    PatternMemoryConfig,
    PredictionMetadata,
    SimilarityMetrics,
)

__all__ = [
    # Основные классы
    "MarketFeatures",
    "OutcomeType",
    "PatternMemory",
    "PatternOutcome",
    "PatternSnapshot",
    "PredictionResult",
    # Интерфейсы
    "IPatternMemoryRepository",
    "IPatternMemoryService",
    "IPatternMatcher",
    # Типы
    "PatternMemoryConfig",
    "MemoryStatistics",
    "SimilarityMetrics",
    "PredictionMetadata",
]
