# -*- coding: utf-8 -*-
"""Модуль для работы с торговыми символами и их анализом."""
from .cache import (
    MarketPhaseCache,
    MemorySymbolCache,
    OpportunityScoreCache,
    SymbolCacheManager,
)
from .market_phase_classifier import MarketPhaseClassifier, PhaseDetectionConfig
from .opportunity_score import OpportunityScoreCalculator
from .symbol_profile import (
    MarketPhase,
    OrderBookMetricsData,
    PatternMetricsData,
    PriceStructure,
    SessionMetricsData,
    SymbolProfile,
    VolumeProfile,
)
from .validators import SymbolValidator, SymbolDataValidator

__all__ = [
    # Основные классы
    "SymbolProfile",
    "MarketPhaseClassifier",
    "OpportunityScoreCalculator",
    # Конфигурации
    "PhaseDetectionConfig",
    "OpportunityScoreConfig",
    # Результаты
    "OpportunityScore",
    # Вспомогательные классы
    "VolumeProfile",
    "PriceStructure",
    "OrderBookMetricsData",
    "PatternMetricsData",
    "SessionMetricsData",
    # Валидаторы
    "SymbolDataValidator",
    "SymbolValidator",
    # Кэширование
    "MemorySymbolCache",
    "MarketPhaseCache",
    "OpportunityScoreCache",
    "SymbolCacheManager",
    # Перечисления
    "MarketPhase",
]
