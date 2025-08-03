"""
Тесты для SessionConfigurationRepository.
"""
import os
import shutil
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.types.session_types import SessionType, SessionProfile, SessionBehavior, SessionTimeWindow
from domain.sessions.repositories import SessionConfigurationRepository
from datetime import time

test_storage = "test_config_sessions"

@pytest.fixture(scope="function", autouse=True)
def cleanup() -> Any:
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
    yield
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)

def make_profile() -> SessionProfile:
    return SessionProfile(
        session_type=SessionType.ASIAN,
        time_window=SessionTimeWindow(
            start_time=time(0, 0),
            end_time=time(8, 0),
            timezone="UTC"
        ),
        behavior=SessionBehavior(
            typical_volatility_spike_minutes=45,
            volume_peak_hours=[2, 4, 6],
            quiet_hours=[1, 5],
            avg_volume_multiplier=0.8,
            avg_volatility_multiplier=0.9,
            typical_direction_bias=0.1,
            common_patterns=["asian_range"],
            false_breakout_probability=0.4,
            reversal_probability=0.25,
            overlap_impact={"london": 1.2}
        ),
        description="Test Asian Session",
        typical_volume_multiplier=0.8,
        typical_volatility_multiplier=0.9,
        typical_spread_multiplier=1.1,
        typical_direction_bias=0.1,
        whale_activity_probability=0.15,
        mm_activity_probability=0.25,
        news_sensitivity=0.4,
        technical_signal_strength=0.6,
        fundamental_impact_multiplier=0.8,
        correlation_breakdown_probability=0.3,
        gap_probability=0.15,
        reversal_probability=0.25,
        continuation_probability=0.55,
        manipulation_susceptibility=0.35,
        false_breakout_probability=0.4
    )

def test_save_and_get_profile() -> None:
    repo = SessionConfigurationRepository(storage_path=test_storage)
    profile = make_profile()
    repo.save_session_profile(profile)
    loaded = repo.get_session_profile(SessionType.ASIAN)
    assert loaded is not None
    assert loaded.session_type == SessionType.ASIAN
    assert loaded.description == "Test Asian Session"

def test_update_profile() -> None:
    repo = SessionConfigurationRepository(storage_path=test_storage)
    profile = make_profile()
    repo.save_session_profile(profile)
    repo.update_session_profile(SessionType.ASIAN, {"description": "Updated"})
    loaded = repo.get_session_profile(SessionType.ASIAN)
    assert loaded is not None
    assert loaded.description == "Updated"

def test_delete_profile() -> None:
    repo = SessionConfigurationRepository(storage_path=test_storage)
    profile = make_profile()
    repo.save_session_profile(profile)
    assert repo.delete_session_profile(SessionType.ASIAN)
    assert repo.get_session_profile(SessionType.ASIAN) is None

def test_get_all_profiles() -> None:
    repo = SessionConfigurationRepository(storage_path=test_storage)
    profile = make_profile()
    repo.save_session_profile(profile)
    all_profiles = repo.get_all_session_profiles()
    assert SessionType.ASIAN in all_profiles
    assert isinstance(all_profiles[SessionType.ASIAN], SessionProfile) 
