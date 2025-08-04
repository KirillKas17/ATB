"""
Модуль для работы с профилями рынка и паттернами маркет-мейкера.
Этот модуль предоставляет промышленную реализацию хранения, анализа и управления
паттернами маркет-мейкера с использованием строгой типизации и архитектуры DDD.
Основные возможности:
- Промышленное хранение паттернов с кэшированием и сжатием
- Многомерный анализ схожести паттернов
- Статистический анализ успешности
- Анализ поведения маркет-мейкера
- Генерация торговых рекомендаций
- Валидация и мониторинг данных
Архитектура:
- Строгая типизация с использованием NewType и Protocol
- Асинхронные операции с блокировками
- Модульная структура по принципам DDD
- Конфигурируемые параметры для всех компонентов
- Встроенные метрики производительности
"""

# Re-export domain models for convenience
from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternFeatures,
    PatternMemory,
    PatternResult,
)

# Re-export domain types for convenience
from domain.type_definitions.market_maker_types import (
    Accuracy,
    AverageReturn,
    Confidence,
    MarketMakerPatternType,
    MarketPhase,
    PatternConfidence,
    PatternOutcome,
    SignalStrength,
    SimilarityScore,
    SuccessCount,
    TotalCount,
)

# Interfaces
from .interfaces.storage_interfaces import (
    IBehaviorHistoryStorage,
    IPatternAnalyzer,
    IPatternStorage,
)

# Storage components
# Analysis components
# Configuration models
# Data models
from .models.storage_models import (
    BehaviorRecord,
    PatternMetadata,
    StorageStatistics,
    SuccessMapEntry,
)

__version__ = "1.0.0"
__author__ = "ATB Trading System"
__description__ = (
    "Промышленный модуль для работы с профилями рынка и паттернами маркет-мейкера"
)
__all__ = [
    # Storage
    "MarketMakerStorage",
    "PatternMemoryRepository",
    "BehaviorHistoryRepository",
    # Analysis
    "PatternAnalyzer",
    "SimilarityCalculator",
    "SuccessRateAnalyzer",
    # Configuration
    "StorageConfig",
    "AnalysisConfig",
    # Models
    "StorageStatistics",
    "PatternMetadata",
    "BehaviorRecord",
    "SuccessMapEntry",
    # Interfaces
    "IPatternStorage",
    "IBehaviorHistoryStorage",
    "IPatternAnalyzer",
    # Domain types (re-exported for convenience)
    "MarketMakerPatternType",
    "PatternOutcome",
    "PatternConfidence",
    "MarketPhase",
    "Confidence",
    "SimilarityScore",
    "SignalStrength",
    "Accuracy",
    "AverageReturn",
    "SuccessCount",
    "TotalCount",
    # Domain models (re-exported for convenience)
    "MarketMakerPattern",
    "PatternMemory",
    "PatternResult",
    "PatternFeatures",
]
# Version info
__version_info__ = (1, 0, 0)
# Module metadata
__module_metadata__ = {
    "name": "market_profiles",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "architecture": "DDD + SOLID",
    "features": [
        "Промышленное хранение паттернов",
        "Многомерный анализ схожести",
        "Статистический анализ успешности",
        "Анализ поведения маркет-мейкера",
        "Генерация торговых рекомендаций",
        "Строгая типизация",
        "Асинхронные операции",
        "Кэширование и оптимизация",
        "Валидация и мониторинг",
    ],
    "components": {
        "storage": [
            "MarketMakerStorage",
            "PatternMemoryRepository",
            "BehaviorHistoryRepository",
        ],
        "analysis": ["PatternAnalyzer", "SimilarityCalculator", "SuccessRateAnalyzer"],
        "models": [
            "StorageConfig",
            "AnalysisConfig",
            "StorageStatistics",
            "PatternMetadata",
            "BehaviorRecord",
            "SuccessMapEntry",
        ],
        "interfaces": [
            "IPatternStorage",
            "IBehaviorHistoryStorage",
            "IPatternAnalyzer",
        ],
    },
}
