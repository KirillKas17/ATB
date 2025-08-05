"""
Unit тесты для domain.memory.entities

Покрывает:
- PatternSnapshot
- PatternOutcome
- PredictionResult
- PatternCluster
- PatternAnalysis
- MemoryOptimizationResult
"""
import pytest
from shared.numpy_utils import np
from datetime import datetime
from domain.memory.entities import (
    PatternSnapshot, PatternOutcome, PredictionResult, PatternCluster, PatternAnalysis, MemoryOptimizationResult
)
from domain.memory.types import MarketFeatures, OutcomeType, PredictionDirection
from domain.type_definitions.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp

def make_market_features():
    return MarketFeatures(
        price=100.0, price_change_1m=0.1, price_change_5m=0.2, price_change_15m=0.3, volatility=0.01,
        volume=1000.0, volume_change_1m=0.05, volume_change_5m=0.1, volume_sma_ratio=1.1,
        spread=0.01, spread_change=0.001, bid_volume=500.0, ask_volume=500.0, order_book_imbalance=0.0,
        depth_absorption=0.2, entropy=0.5, gravity=0.1, latency=0.01, correlation=0.0,
        whale_signal=0.0, mm_signal=0.0, external_sync=False
    )

    def test_pattern_snapshot_to_from_dict():
    ts = Timestamp(datetime.now())
    snap = PatternSnapshot(
        pattern_id="pid", timestamp=ts, symbol="BTC/USDT", pattern_type=PatternType.CANDLE,
        confidence=0.9, strength=0.8, direction="up", features=make_market_features(), metadata={"a":1}
    )
    d = snap.to_dict()
    snap2 = PatternSnapshot.from_dict(d)
    assert snap.pattern_id == snap2.pattern_id
    assert snap.symbol == snap2.symbol
    assert snap.pattern_type == snap2.pattern_type
    assert snap.confidence == snap2.confidence
    assert snap.features.to_dict() == snap2.features.to_dict()

    def test_pattern_outcome_to_from_dict():
    ts = Timestamp(datetime.now())
    out = PatternOutcome(
        pattern_id="pid", symbol="BTC/USDT", outcome_type=OutcomeType.PROFITABLE, timestamp=ts,
        price_change_percent=1.0, volume_change_percent=2.0, duration_minutes=5,
        max_profit_percent=1.5, max_loss_percent=-0.5, final_return_percent=1.0,
        volatility_during=0.01, volume_profile="stable", market_regime="trending", metadata={"b":2}
    )
    d = out.to_dict()
    out2 = PatternOutcome.from_dict(d)
    assert out.pattern_id == out2.pattern_id
    assert out.symbol == out2.symbol
    assert out.outcome_type == out2.outcome_type
    assert out.price_change_percent == out2.price_change_percent

    def test_prediction_result_to_from_dict():
    pr = PredictionResult(
        pattern_id="pid", symbol="BTC/USDT", confidence=0.8, predicted_direction="up",
        predicted_return_percent=1.2, predicted_duration_minutes=10, predicted_volatility=0.01,
        similar_cases_count=3, success_rate=0.7, avg_return=1.1, avg_duration=9, metadata={"c":3}
    )
    d = pr.to_dict()
    pr2 = PredictionResult.from_dict(d)
    assert pr.pattern_id == pr2.pattern_id
    assert pr.symbol == pr2.symbol
    assert pr.confidence == pr2.confidence
    assert pr.predicted_direction == pr2.predicted_direction

    def test_pattern_cluster_to_dict():
    mf = make_market_features()
    snap = PatternSnapshot(
        pattern_id="pid", timestamp=Timestamp(datetime.now()), symbol="BTC/USDT", pattern_type=PatternType.CANDLE,
        confidence=0.9, strength=0.8, direction="up", features=mf, metadata={}
    )
    out = PatternOutcome(
        pattern_id="pid", symbol="BTC/USDT", outcome_type=OutcomeType.PROFITABLE, timestamp=Timestamp(datetime.now()),
        price_change_percent=1.0, volume_change_percent=2.0, duration_minutes=5,
        max_profit_percent=1.5, max_loss_percent=-0.5, final_return_percent=1.0,
        volatility_during=0.01, volume_profile="stable", market_regime="trending", metadata={}
    )
    cluster = PatternCluster(
        cluster_id="cid", center_features=mf, patterns=[snap], outcomes=[out],
        avg_similarity=0.95, avg_confidence=0.9, avg_return=1.0, success_rate=0.8, size=1, metadata={}
    )
    d = cluster.to_dict()
    assert d["cluster_id"] == "cid"
    assert isinstance(d["patterns"], list)
    assert isinstance(d["outcomes"], list)

    def test_pattern_analysis_to_dict():
    pa = PatternAnalysis(
        pattern_id="pid", symbol="BTC/USDT", pattern_type=PatternType.CANDLE, timestamp=Timestamp(datetime.now()),
        quality_score=0.9, reliability_score=0.8, uniqueness_score=0.7, market_impact=0.5, volume_significance=0.6,
        price_momentum=0.4, volatility_impact=0.3, cluster_id=None, cluster_similarity=None, metadata={}
    )
    d = pa.to_dict()
    assert d["pattern_id"] == "pid"
    assert d["symbol"] == "BTC/USDT"
    assert d["pattern_type"] == PatternType.CANDLE.value

    def test_memory_optimization_result_to_dict():
    mor = MemoryOptimizationResult(
        optimal_similarity_threshold=0.9, optimal_feature_weights=np.array([1.0,2.0,3.0]),
        optimal_prediction_params={"param":1}, accuracy_improvement=0.1, precision_improvement=0.2,
        recall_improvement=0.3, f1_improvement=0.4, total_patterns_analyzed=100, patterns_removed=10,
        clusters_created=2, optimization_method="auto", processing_time_seconds=1.5, metadata={}
    )
    d = mor.to_dict()
    assert d["optimal_similarity_threshold"] == 0.9
    assert d["optimal_feature_weights"] == [1.0,2.0,3.0]
    assert d["optimization_method"] == "auto"