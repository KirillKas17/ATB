"""
Тесты для SessionDataRepository.
"""
import os
import shutil
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta, timezone
from domain.type_definitions.session_types import SessionType, SessionPhase, SessionAnalysisResult, SessionMetrics, MarketConditions, ConfidenceScore, MarketRegime, SessionIntensity
from domain.value_objects.timestamp import Timestamp
from domain.sessions.repositories import SessionDataRepository

test_storage = "test_data_sessions"

@pytest.fixture(scope="function", autouse=True)
def cleanup() -> Any:
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
    yield
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)

def make_analysis(ts: Timestamp) -> SessionAnalysisResult:
    return SessionAnalysisResult(
        session_type=SessionType.ASIAN,
        session_phase=SessionPhase.OPENING,
        timestamp=ts,
        confidence=ConfidenceScore(0.95),
        metrics=SessionMetrics(
            volume_change_percent=1.0,
            volatility_change_percent=1.0,
            price_direction_bias=0.1,
            momentum_strength=0.5,
            false_breakout_probability=0.2,
            reversal_probability=0.1,
            trend_continuation_probability=0.7,
            influence_duration_minutes=60,
            peak_influence_time_minutes=30,
            spread_impact=0.05,
            liquidity_impact=0.1,
            correlation_with_other_sessions=0.8
        ),
        market_conditions=MarketConditions(
            volatility=1.0,
            volume=1.0,
            spread=0.01,
            liquidity=1.0,
            momentum=0.5,
            trend_strength=0.3,
            market_regime=MarketRegime.RANGING,
            session_intensity=SessionIntensity.NORMAL
        ),
        predictions={"volatility": 1.0, "volume": 1.0},
        risk_factors=["manipulation", "gap"]
    )

def test_save_and_get_analysis() -> None:
    repo = SessionDataRepository(storage_path=test_storage)
    ts = Timestamp(datetime.now(timezone.utc))
    analysis = make_analysis(ts)
    repo.save_session_analysis(analysis)
    result = repo.get_session_analysis(SessionType.ASIAN, ts, ts)
    assert len(result) == 1
    assert result[0].session_type == SessionType.ASIAN
    assert result[0].confidence == 0.95

def test_delete_analysis() -> None:
    repo = SessionDataRepository(storage_path=test_storage)
    ts = Timestamp(datetime.now(timezone.utc))
    analysis = make_analysis(ts)
    repo.save_session_analysis(analysis)
    deleted = repo.delete_session_analysis(SessionType.ASIAN, ts, ts)
    assert deleted == 1
    result = repo.get_session_analysis(SessionType.ASIAN, ts, ts)
    assert len(result) == 0

def test_get_session_statistics() -> None:
    repo = SessionDataRepository(storage_path=test_storage)
    ts = datetime.now(timezone.utc)
    for i in range(3):
        analysis = make_analysis(Timestamp(ts + timedelta(minutes=i)))
        repo.save_session_analysis(analysis)
    start = Timestamp(ts)
    end = Timestamp(ts + timedelta(minutes=2))
    stats = repo.get_session_statistics(SessionType.ASIAN, start_time=start, end_time=end)
    assert stats["total_analyses"] >= 3
    assert stats["avg_confidence"] > 0 
