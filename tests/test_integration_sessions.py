"""
Тесты для модуля торговых сессий.
"""
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.type_definitions.session_types import (
    SessionType,
    SessionPhase,
    SessionProfile,
    SessionBehavior,
    SessionTimeWindow,
    MarketConditions,
    MarketRegime,
    SessionIntensity,
)
from domain.sessions.session_profile import SessionProfileRegistry
from domain.sessions.session_marker import SessionMarker
from domain.sessions.implementations import DefaultSessionMetricsAnalyzer
class TestSessionProfileRegistry:
    """Тесты для реестра профилей сессий."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = SessionProfileRegistry()
    def test_get_profile(self) -> None:
        """Тест получения профиля сессии."""
        profile = self.registry.get_profile(SessionType.ASIAN)
        assert profile is not None
        assert profile.session_type == SessionType.ASIAN
        assert profile.description == "Азиатская торговая сессия (Токио)"
    def test_get_all_profiles(self) -> None:
        """Тест получения всех профилей."""
        profiles = self.registry.get_all_profiles()
        assert len(profiles) >= 4  # Минимум 4 профиля по умолчанию
        assert SessionType.ASIAN in profiles
        assert SessionType.LONDON in profiles
        assert SessionType.NEW_YORK in profiles
        assert SessionType.CRYPTO_24H in profiles
    def test_get_session_overlap(self) -> None:
        """Тест расчета перекрытия сессий."""
        overlap = self.registry.get_session_overlap(SessionType.ASIAN, SessionType.LONDON)
        assert isinstance(overlap, float)
        assert 0.0 <= overlap <= 1.0
    def test_get_session_recommendations(self) -> None:
        """Тест получения рекомендаций."""
        recommendations = self.registry.get_session_recommendations(SessionType.ASIAN)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    def test_get_session_statistics(self) -> None:
        """Тест получения статистики."""
        statistics = self.registry.get_session_statistics(SessionType.ASIAN)
        assert isinstance(statistics, dict)
        assert "session_type" in statistics
        assert statistics["session_type"] == "asian"
class TestSessionMarker:
    """Тесты для маркера сессий."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = SessionProfileRegistry()
        self.marker = SessionMarker(self.registry)
    def test_get_session_context(self) -> None:
        """Тест получения контекста сессии."""
        context = self.marker.get_session_context()
        assert context is not None
        assert hasattr(context, 'timestamp')
        assert hasattr(context, 'active_sessions')
        assert hasattr(context, 'primary_session')
    def test_is_session_active(self) -> None:
        """Тест проверки активности сессии."""
        # Тестируем азиатскую сессию
        is_active = self.marker.is_session_active(SessionType.ASIAN)
        assert isinstance(is_active, bool)
    def test_get_session_overlap(self) -> None:
        """Тест получения перекрытия сессий."""
        overlap = self.marker.get_session_overlap(SessionType.ASIAN, SessionType.LONDON)
        assert isinstance(overlap, float)
        assert 0.0 <= overlap <= 1.0
class TestSessionMetricsAnalyzer:
    """Тесты для анализатора метрик сессий."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.analyzer = DefaultSessionMetricsAnalyzer()
        self.session_profile = SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(
                start_time=time(0, 0),
                end_time=time(8, 0)
            ),
            behavior=SessionBehavior(
                typical_volatility_spike_minutes=45,
                volume_peak_hours=[2, 4, 6],
                quiet_hours=[1, 5],
                avg_volume_multiplier=0.8,
                avg_volatility_multiplier=0.9,
                typical_direction_bias=0.1,
                common_patterns=["asian_range", "breakout_failure"],
                false_breakout_probability=0.4,
                reversal_probability=0.25,
                overlap_impact={"london": 1.2, "new_york": 0.9}
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
    def test_calculate_volume_impact(self) -> None:
        """Тест расчета влияния на объем."""
        # Создаем тестовые данные
        market_data = pd.DataFrame({
            'volume': [100, 200, 150, 300, 250],
            'close': [100.0, 101.0, 100.5, 102.0, 101.5]
        })
        impact = self.analyzer.calculate_volume_impact(market_data, self.session_profile)
        assert isinstance(impact, float)
        assert impact > 0
    def test_calculate_volatility_impact(self) -> None:
        """Тест расчета влияния на волатильность."""
        # Создаем тестовые данные
        market_data = pd.DataFrame({
            'close': [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5]
        })
        impact = self.analyzer.calculate_volatility_impact(market_data, self.session_profile)
        assert isinstance(impact, float)
        assert impact > 0
    def test_calculate_direction_bias(self) -> None:
        """Тест расчета смещения направления."""
        # Создаем тестовые данные с трендом
        market_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                     110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0]
        })
        bias = self.analyzer.calculate_direction_bias(market_data, self.session_profile)
        assert isinstance(bias, float)
        assert -1.0 <= bias <= 1.0
    def test_calculate_momentum_strength(self) -> None:
        """Тест расчета силы импульса."""
        # Создаем тестовые данные
        market_data = pd.DataFrame({
            'close': [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0,
                     104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0]
        })
        momentum = self.analyzer.calculate_momentum_strength(market_data, self.session_profile)
        assert isinstance(momentum, float)
        assert 0.0 <= momentum <= 1.0
class TestSessionService:
    """Тесты для сервиса сессий."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        from domain.sessions.factories import get_session_service
        self.service = get_session_service()
    def test_get_current_session_context(self) -> None:
        """Тест получения текущего контекста сессии."""
        context = self.service.get_current_session_context()
        assert isinstance(context, dict)
        assert "timestamp" in context
        assert "active_sessions" in context
    def test_predict_session_behavior(self) -> None:
        """Тест прогнозирования поведения сессии."""
        market_conditions = MarketConditions(
            volatility=1.0,
            volume=1.0,
            spread=1.0,
            liquidity=1.0,
            momentum=0.5,
            trend_strength=0.3,
            market_regime=MarketRegime.RANGING,
            session_intensity=SessionIntensity.NORMAL
        )
        prediction = self.service.predict_session_behavior(SessionType.ASIAN, market_conditions)
        assert isinstance(prediction, dict)
        assert "predicted_volatility" in prediction
        assert "predicted_volume" in prediction
        assert "predicted_direction_bias" in prediction
    def test_get_session_recommendations(self) -> None:
        """Тест получения рекомендаций."""
        recommendations = self.service.get_session_recommendations("BTCUSD", SessionType.ASIAN)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    def test_get_session_statistics(self) -> None:
        """Тест получения статистики."""
        statistics = self.service.get_session_statistics(SessionType.ASIAN)
        assert isinstance(statistics, dict)
        assert "session_type" in statistics
class TestSessionTypes:
    """Тесты для типов сессий."""
    def test_session_type_enum(self) -> None:
        """Тест перечисления типов сессий."""
        assert SessionType.ASIAN.value == "asian"
        assert SessionType.LONDON.value == "london"
        assert SessionType.NEW_YORK.value == "new_york"
        assert SessionType.CRYPTO_24H.value == "crypto_24h"
    def test_session_phase_enum(self) -> None:
        """Тест перечисления фаз сессий."""
        assert SessionPhase.OPENING.value == "opening"
        assert SessionPhase.MID_SESSION.value == "mid_session"
        assert SessionPhase.CLOSING.value == "closing"
    def test_session_profile_creation(self) -> None:
        """Тест создания профиля сессии."""
        profile = SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(
                start_time=time(0, 0),
                end_time=time(8, 0)
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
            description="Test Profile",
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
        assert profile.session_type == SessionType.ASIAN
        assert profile.description == "Test Profile"
        assert profile.typical_volume_multiplier == 0.8
        assert profile.typical_volatility_multiplier == 0.9
if __name__ == "__main__":
    pytest.main([__file__]) 
