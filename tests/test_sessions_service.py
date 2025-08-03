"""
Тесты для SessionService.
"""
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from domain.types.session_types import (
    SessionType, SessionPhase, SessionAnalysisResult, SessionMetrics, 
    MarketConditions, ConfidenceScore, MarketRegime, SessionIntensity
)
from domain.value_objects.timestamp import Timestamp
from domain.sessions.services import SessionService, SessionContext, SessionPrediction
from domain.sessions.session_marker import MarketSessionContext, SessionState
@pytest.fixture
def mock_registry() -> Any:
    registry = Mock()
    profile = Mock()
    profile.false_breakout_probability = 0.3
    profile.reversal_probability = 0.2
    profile.continuation_probability = 0.6
    profile.behavior.typical_volatility_spike_minutes = 60
    profile.correlation_breakdown_probability = 0.1
    profile.typical_volatility_multiplier = 1.2
    profile.typical_volume_multiplier = 1.1
    profile.manipulation_susceptibility = 0.3
    profile.whale_activity_probability = 0.1
    profile.mm_activity_probability = 0.3
    profile.time_window.get_phase.return_value = SessionPhase.MID_SESSION
    profile.calculate_session_impact.return_value = SessionMetrics(
        volume_change_percent=0.0,
        volatility_change_percent=0.0,
        price_direction_bias=0.1,
        momentum_strength=0.5,
        false_breakout_probability=0.3,
        reversal_probability=0.2,
        trend_continuation_probability=0.6,
        influence_duration_minutes=60,
        peak_influence_time_minutes=30,
        spread_impact=1.0,
        liquidity_impact=1.0,
        correlation_with_other_sessions=0.9
    )
    registry.get_profile.return_value = profile
    registry.get_session_statistics.return_value = {"total_analyses": 10}
    registry.get_session_recommendations.return_value = ["Test recommendation"]
    registry.get_session_overlap.return_value = 0.5
    registry.get_all_profiles.return_value = {SessionType.ASIAN: profile}
    return registry
@pytest.fixture
def mock_session_marker() -> Any:
    marker = Mock()
    # Создаем SessionState для активной сессии
    session_state = SessionState(
        session_type=SessionType.ASIAN,
        phase=SessionPhase.OPENING,
        is_active=True,
        time_until_open=None,
        time_until_close=None,
        overlap_with_other_sessions={}
    )
    context = MarketSessionContext(
        timestamp=Timestamp(datetime.now(timezone.utc)),
        primary_session=session_state,
        active_sessions=[session_state],
        session_transitions=[],
        market_conditions=None
    )
    marker.get_session_context.return_value = context
    marker.get_next_session_change.return_value = {"time_ahead_hours": 2.0}
    return marker
@pytest.fixture
def mock_influence_analyzer() -> Any:
    analyzer = Mock()
    analysis = SessionAnalysisResult(
        session_type=SessionType.ASIAN,
        session_phase=SessionPhase.OPENING,
        timestamp=Timestamp(datetime.now(timezone.utc)),
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
    analyzer.analyze_session.return_value = analysis
    return analyzer
@pytest.fixture
def mock_transition_manager() -> Any:
    manager = Mock()
    manager.is_transition_period.return_value = False
    manager.get_active_transitions.return_value = []
    return manager
@pytest.fixture
def mock_cache() -> Any:
    cache = Mock()
    cache.get.return_value = None  # По умолчанию кэш пустой
    return cache
@pytest.fixture
def mock_validator() -> Any:
    validator = Mock()
    validator.validate_market_data.return_value = True
    validator.validate_session_analysis.return_value = True
    return validator
@pytest.fixture
def session_service(mock_registry, mock_session_marker, mock_influence_analyzer, 
                   mock_transition_manager, mock_cache, mock_validator) -> Any:
    return SessionService(
        registry=mock_registry,
        session_marker=mock_session_marker,
        influence_analyzer=mock_influence_analyzer,
        transition_manager=mock_transition_manager,
        cache=mock_cache,
        validator=mock_validator
    )
def test_get_current_session_context(session_service, mock_cache) -> None:
    """Тест получения контекста сессии."""
    context = session_service.get_current_session_context()
    assert isinstance(context, dict)
    assert "active_sessions" in context
    assert "primary_session" in context
    assert context["primary_session"] is not None
    assert "phase" in context["primary_session"]
    mock_cache.set.assert_called_once()
def test_analyze_session_influence(session_service, mock_cache) -> None:
    """Тест анализа влияния сессии."""
    market_data = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    analysis = session_service.analyze_session_influence("BTCUSDT", market_data)
    assert analysis is not None
    assert analysis.session_type == SessionType.ASIAN
    assert analysis.confidence == 0.95
    mock_cache.set.assert_called_once()
def test_predict_session_behavior(session_service, mock_cache) -> None:
    """Тест прогноза поведения сессии."""
    market_conditions = MarketConditions(
        volatility=1.0,
        volume=1.0,
        spread=0.01,
        liquidity=1.0,
        momentum=0.5,
        trend_strength=0.3,
        market_regime=MarketRegime.RANGING,
        session_intensity=SessionIntensity.NORMAL
    )
    prediction = session_service.predict_session_behavior(SessionType.ASIAN, market_conditions)
    assert isinstance(prediction, dict)
    assert "predicted_volatility" in prediction
    assert "predicted_volume" in prediction
    assert "reversal_probability" in prediction
    mock_cache.set.assert_called_once()
def test_get_session_recommendations(session_service, mock_cache) -> None:
    """Тест получения рекомендаций."""
    recommendations = session_service.get_session_recommendations("BTCUSDT", SessionType.ASIAN)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    mock_cache.set.assert_called_once()
def test_get_session_statistics(session_service, mock_cache) -> None:
    """Тест получения статистики."""
    stats = session_service.get_session_statistics(SessionType.ASIAN)
    assert isinstance(stats, dict)
    assert "total_analyses" in stats
    mock_cache.set.assert_called_once()
def test_is_transition_period(session_service) -> None:
    """Тест проверки переходного периода."""
    result = session_service.is_transition_period()
    assert isinstance(result, bool)
def test_get_active_transitions(session_service) -> None:
    """Тест получения активных переходов."""
    transitions = session_service.get_active_transitions()
    assert isinstance(transitions, list)
def test_get_session_overlap(session_service) -> None:
    """Тест получения перекрытия сессий."""
    overlap = session_service.get_session_overlap(SessionType.ASIAN, SessionType.LONDON)
    assert isinstance(overlap, float)
    assert 0.0 <= overlap <= 1.0
def test_get_session_phase(session_service) -> None:
    """Тест получения фазы сессии."""
    phase = session_service.get_session_phase(SessionType.ASIAN)
    assert phase is not None
    assert isinstance(phase, str)
def test_get_next_session_change(session_service) -> None:
    """Тест получения следующего изменения сессии."""
    change = session_service.get_next_session_change()
    assert isinstance(change, dict)
    assert "time_ahead_hours" in change
def test_clear_cache(session_service, mock_cache) -> None:
    """Тест очистки кэша."""
    session_service.clear_cache()
    mock_cache.clear.assert_called_once()
def test_get_session_health_check(session_service) -> None:
    """Тест проверки здоровья сервиса."""
    health = session_service.get_session_health_check()
    assert isinstance(health, dict)
    assert "status" in health
    assert "timestamp" in health
    assert "components" in health
    assert "metrics" in health
    assert health["status"] == "healthy"
def test_analyze_session_influence_with_invalid_data(session_service, mock_validator) -> None:
    """Тест анализа с невалидными данными."""
    mock_validator.validate_market_data.return_value = False
    market_data = pd.DataFrame({'invalid': [1, 2, 3]})
    analysis = session_service.analyze_session_influence("BTCUSDT", market_data)
    assert analysis is None
def test_predict_session_behavior_without_profile(session_service, mock_registry) -> None:
    """Тест прогноза без профиля сессии."""
    mock_registry.get_profile.return_value = None
    market_conditions = MarketConditions(
        volatility=1.0,
        volume=1.0,
        spread=0.01,
        liquidity=1.0,
        momentum=0.5,
        trend_strength=0.3,
        market_regime=MarketRegime.RANGING,
        session_intensity=SessionIntensity.NORMAL
    )
    prediction = session_service.predict_session_behavior(SessionType.ASIAN, market_conditions)
    assert isinstance(prediction, dict)
    assert prediction["predicted_volatility"] == 1.0  # Значение по умолчанию 
