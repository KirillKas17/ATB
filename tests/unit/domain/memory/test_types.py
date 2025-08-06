"""
Unit тесты для domain.memory.types

Покрывает:
- OutcomeType, VolumeProfile, MarketRegime, PredictionDirection
- MarketFeatures, PatternMemoryConfig, MemoryStatistics, SimilarityMetrics, PredictionMetadata
- TypedDict структуры
"""
import pytest
import numpy as np
from datetime import datetime
from domain.memory.types import (
    OutcomeType, VolumeProfile, MarketRegime, PredictionDirection,
    MarketFeatures, PatternMemoryConfig, MemoryStatistics, SimilarityMetrics, PredictionMetadata
)
from domain.type_definitions.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp

def test_enum_values():
    assert OutcomeType.PROFITABLE.value == "profitable"
    assert VolumeProfile.INCREASING.value == "increasing"
    assert MarketRegime.TRENDING.value == "trending"
    assert PredictionDirection.UP.value == "up"

def test_market_features_to_from_dict():
    mf = MarketFeatures(
        price=100.0, price_change_1m=0.1, price_change_5m=0.2, price_change_15m=0.3, volatility=0.01,
        volume=1000.0, volume_change_1m=0.05, volume_change_5m=0.1, volume_sma_ratio=1.1,
        spread=0.01, spread_change=0.001, bid_volume=500.0, ask_volume=500.0, order_book_imbalance=0.0,
        depth_absorption=0.2, entropy=0.5, gravity=0.1, latency=0.01, correlation=0.0,
        whale_signal=0.0, mm_signal=0.0, external_sync=False
    )
    d = mf.to_dict()
    mf2 = MarketFeatures.from_dict(d)
    assert mf.price == mf2.price
    assert mf.entropy == mf2.entropy
    assert mf.external_sync == mf2.external_sync

def test_market_features_to_vector():
    mf = MarketFeatures(
        price=100.0, price_change_1m=0.1, price_change_5m=0.2, price_change_15m=0.3, volatility=0.01,
        volume=1000.0, volume_change_1m=0.05, volume_change_5m=0.1, volume_sma_ratio=1.1,
        spread=0.01, spread_change=0.001, bid_volume=500.0, ask_volume=500.0, order_book_imbalance=0.0,
        depth_absorption=0.2, entropy=0.5, gravity=0.1, latency=0.01, correlation=0.0,
        whale_signal=0.0, mm_signal=0.0, external_sync=False
    )
    vec = mf.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] > 0

def test_pattern_memory_config_defaults():
    cfg = PatternMemoryConfig()
    assert cfg.db_path == "data/pattern_memory.db"
    assert cfg.similarity_threshold == 0.9
    assert cfg.max_similar_cases == 10
    assert cfg.days_to_keep == 30

def test_memory_statistics_to_dict():
    ms = MemoryStatistics(
        total_snapshots=10, total_outcomes=5, pattern_type_stats={"CANDLE":2},
        outcome_type_stats={"profitable":1}, symbol_stats={"BTC/USDT":3},
        avg_confidence=0.8, avg_success_rate=0.7, last_cleanup=None
    )
    d = ms.to_dict()
    assert d["total_snapshots"] == 10
    assert d["avg_confidence"] == 0.8

def test_similarity_metrics_to_dict():
    sm = SimilarityMetrics(
        similarity_score=0.95, confidence_boost=0.1, signal_strength=0.2,
        pattern_type=PatternType.BREAKOUT, timestamp=Timestamp(datetime.now()), accuracy=0.8, avg_return=1.0
    )
    d = sm.to_dict()
    assert d["similarity_score"] == 0.95
    assert d["pattern_type"] == "breakout"

def test_prediction_metadata_to_dict():
    pm = PredictionMetadata(
        algorithm_version="v1", processing_time_ms=10.0, data_points_used=100,
        confidence_interval=(0.7, 0.9), model_parameters={"a":1}, quality_metrics={"f1":0.8}
    )
    d = pm.to_dict()
    assert d["algorithm_version"] == "v1"
    assert d["quality_metrics"]["f1"] == 0.8