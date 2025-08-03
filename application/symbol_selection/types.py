"""
Типы и конфигурации для модуля выбора символов.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

import pandas as pd

from domain.symbols import MarketPhase, SymbolProfile


@dataclass
class DOASSConfig:
    """Конфигурация DOASS с продвинутыми параметрами."""

    # Основные параметры
    update_interval_seconds: int = 60  # Обновление каждую минуту
    max_symbols_per_cycle: int = 15
    min_opportunity_score: float = 0.78
    min_confidence_threshold: float = 0.6
    # Продвинутые фильтры
    enable_correlation_filtering: bool = True
    enable_entanglement_analysis: bool = True
    enable_pattern_memory_integration: bool = True
    enable_session_alignment: bool = True
    enable_liquidity_gravity: bool = True
    enable_reversal_prediction: bool = True
    # Параметры корреляции
    max_correlation_threshold: float = 0.85
    min_correlation_diversity: float = 0.3
    # Параметры энтанглмента
    entanglement_confidence_threshold: float = 0.7
    max_entangled_pairs: int = 3
    # Параметры памяти паттернов
    pattern_memory_lookback_days: int = 30
    min_pattern_success_rate: float = 0.65
    # Параметры сессий
    session_alignment_threshold: float = 0.6
    cross_session_correlation_threshold: float = 0.5
    # Параметры ликвидности
    min_liquidity_score: float = 0.5
    gravity_anomaly_threshold: float = 0.7
    # Параметры предсказания разворотов
    reversal_confidence_threshold: float = 0.6
    reversal_timeframe_hours: int = 24
    # Кэширование и производительность
    cache_ttl_seconds: int = 300  # 5 минут
    max_cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 8
    # Мониторинг и метрики
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = True
    metrics_export_interval: int = 300  # 5 минут


@dataclass
class SymbolSelectionResult:
    """Результат выбора символов с продвинутой аналитикой."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
    reversal_probabilities: Dict[str, float] = field(default_factory=dict)
    # Метаданные
    processing_time_ms: float = 0.0
    total_symbols_analyzed: int = 0
    cache_hit_rate: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    # Детальная информация
    detailed_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    # Дополнительные поля для совместимости
    profiles: List[SymbolProfile] = field(default_factory=list)
    total_analyzed: int = 0
    total_filtered: int = 0
    total_selected: int = 0
