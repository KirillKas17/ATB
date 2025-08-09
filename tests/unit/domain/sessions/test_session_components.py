"""
Юнит-тесты для компонентов модуля sessions.
"""

import pandas as pd
from shared.numpy_utils import np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, time
from typing import Dict, Any

from domain.type_definitions.session_types import (
    SessionType,
    SessionPhase,
    SessionProfile,
    SessionBehavior,
    SessionTimeWindow,
    MarketConditions,
    MarketRegime,
    SessionIntensity,
    SessionAnalysisResult,
    SessionMetrics,
    ConfidenceScore,
)
from domain.value_objects.timestamp import Timestamp
from domain.sessions.session_profile import SessionProfileRegistry
from domain.sessions.session_marker import SessionMarker, MarketSessionContext, SessionState
from domain.sessions.session_influence_analyzer import SessionInfluenceAnalyzer
from domain.sessions.session_analyzer import SessionAnalyzer
from domain.sessions.session_manager import SessionManager
from domain.sessions.session_optimizer import SessionOptimizer
from domain.sessions.session_predictor import SessionPredictor
from domain.sessions.session_analyzer_factory import SessionAnalyzerFactory
from domain.sessions.repositories import SessionDataRepository, SessionConfigurationRepository


class TestSessionProfileRegistry:
    """Тесты для SessionProfileRegistry."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = SessionProfileRegistry()

    def test_get_profile_all_types(self: "TestSessionProfileRegistry") -> None:
        """Тест получения профилей всех типов сессий."""
        for session_type in SessionType:
            profile = self.registry.get_profile(session_type)
            assert profile is not None
            assert profile.session_type == session_type
            assert isinstance(profile.description, str)
            assert len(profile.description) > 0

    def test_get_all_profiles_completeness(self: "TestSessionProfileRegistry") -> None:
        """Тест полноты всех профилей."""
        profiles = self.registry.get_all_profiles()
        assert len(profiles) == len(SessionType)
        for session_type in SessionType:
            assert session_type in profiles
            profile = profiles[session_type]
            assert isinstance(profile, SessionProfile)

    def test_get_session_overlap_matrix(self: "TestSessionProfileRegistry") -> None:
        """Тест матрицы перекрытий сессий."""
        session_types = list(SessionType)
        for i, session1 in enumerate(session_types):
            for j, session2 in enumerate(session_types):
                overlap = self.registry.get_session_overlap(session1, session2)
                assert isinstance(overlap, float)
                assert 0.0 <= overlap <= 1.0
                # Перекрытие с самим собой должно быть 1.0
                if i == j:
                    assert overlap == 1.0

    def test_get_session_recommendations_all_types(self: "TestSessionProfileRegistry") -> None:
        """Тест рекомендаций для всех типов сессий."""
        for session_type in SessionType:
            recommendations = self.registry.get_session_recommendations(session_type)
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            for rec in recommendations:
                assert isinstance(rec, str)
                assert len(rec) > 0

    def test_get_session_statistics_all_types(self: "TestSessionProfileRegistry") -> None:
        """Тест статистики для всех типов сессий."""
        for session_type in SessionType:
            statistics = self.registry.get_session_statistics(session_type)
            assert isinstance(statistics, dict)
            assert "session_type" in statistics
            assert statistics["session_type"] == session_type.value

    def test_profile_validation(self: "TestSessionProfileRegistry") -> None:
        """Тест валидации профилей."""
        for session_type in SessionType:
            profile = self.registry.get_profile(session_type)
            # Проверяем обязательные поля
            assert hasattr(profile, "session_type")
            assert hasattr(profile, "time_window")
            assert hasattr(profile, "behavior")
            assert hasattr(profile, "description")
            # Проверяем типы данных
            assert isinstance(profile.session_type, SessionType)
            assert isinstance(profile.time_window, SessionTimeWindow)
            assert isinstance(profile.behavior, SessionBehavior)
            assert isinstance(profile.description, str)


class TestSessionMarker:
    """Тесты для SessionMarker."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = SessionProfileRegistry()
        self.marker = SessionMarker(self.registry)

    def test_get_session_context_structure(self: "TestSessionMarker") -> None:
        """Тест структуры контекста сессии."""
        context = self.marker.get_session_context()
        assert isinstance(context, MarketSessionContext)
        assert hasattr(context, "timestamp")
        assert hasattr(context, "active_sessions")
        assert hasattr(context, "primary_session")
        assert hasattr(context, "session_transitions")
        assert hasattr(context, "market_conditions")
        assert isinstance(context.timestamp, Timestamp)
        assert isinstance(context.active_sessions, list)
        assert isinstance(context.primary_session, SessionState)

    def test_is_session_active_all_types(self: "TestSessionMarker") -> None:
        """Тест активности всех типов сессий."""
        for session_type in SessionType:
            is_active = self.marker.is_session_active(session_type)
            assert isinstance(is_active, bool)

    def test_get_session_overlap_consistency(self: "TestSessionMarker") -> None:
        """Тест консистентности перекрытий сессий."""
        session_types = list(SessionType)
        for session1 in session_types:
            for session2 in session_types:
                overlap1 = self.marker.get_session_overlap(session1, session2)
                overlap2 = self.marker.get_session_overlap(session2, session1)
                # Перекрытие должно быть симметричным
                assert abs(overlap1 - overlap2) < 1e-6

    def test_get_next_session_change(self: "TestSessionMarker") -> None:
        """Тест получения следующего изменения сессии."""
        change_info = self.marker.get_next_session_change()
        assert isinstance(change_info, dict)
        assert "time_ahead_hours" in change_info
        assert isinstance(change_info["time_ahead_hours"], float)
        assert change_info["time_ahead_hours"] >= 0.0

    def test_get_session_phase(self: "TestSessionMarker") -> None:
        """Тест получения фазы сессии."""
        for session_type in SessionType:
            phase = self.marker.get_session_phase(session_type)
            assert isinstance(phase, SessionPhase)

    def test_market_session_context_validation(self: "TestSessionMarker") -> None:
        """Тест валидации контекста рыночной сессии."""
        context = self.marker.get_session_context()
        # Проверяем primary_session
        assert context.primary_session.session_type in SessionType
        assert context.primary_session.phase in SessionPhase
        assert isinstance(context.primary_session.is_active, bool)
        # Проверяем active_sessions
        for session in context.active_sessions:
            assert isinstance(session, SessionState)
            assert session.session_type in SessionType
            assert session.phase in SessionPhase


class TestSessionInfluenceAnalyzer:
    """Тесты для SessionInfluenceAnalyzer."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.analyzer = SessionInfluenceAnalyzer()
        self.session_profile = self._create_test_session_profile()

    def _create_test_session_profile(self) -> SessionProfile:
        """Создает тестовый профиль сессии."""
        return SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(start_time=time(0, 0), end_time=time(8, 0)),
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
                overlap_impact={"london": 1.2, "new_york": 0.9},
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
            false_breakout_probability=0.4,
        )

    def test_analyze_session_influence_structure(self: "TestSessionInfluenceAnalyzer") -> None:
        """Тест структуры анализа влияния сессии."""
        market_data = self._create_test_market_data()
        result = self.analyzer.analyze_session_influence("BTCUSDT", market_data, self.session_profile)
        assert isinstance(result, SessionAnalysisResult)
        assert result.session_type == SessionType.ASIAN
        assert result.session_phase in SessionPhase
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.confidence, ConfidenceScore)
        assert isinstance(result.metrics, SessionMetrics)
        assert isinstance(result.market_conditions, MarketConditions)
        assert isinstance(result.predictions, dict)
        assert isinstance(result.risk_factors, list)

    def test_analyze_session_influence_metrics_validation(self: "TestSessionInfluenceAnalyzer") -> None:
        """Тест валидации метрик анализа."""
        market_data = self._create_test_market_data()
        result = self.analyzer.analyze_session_influence("BTCUSDT", market_data, self.session_profile)
        # Проверяем метрики
        metrics = result.metrics
        assert isinstance(metrics, SessionMetrics)
        assert isinstance(metrics.volume_change_percent, float)
        assert isinstance(metrics.volatility_change_percent, float)
        assert isinstance(metrics.price_direction_bias, float)
        assert isinstance(metrics.momentum_strength, float)
        assert isinstance(metrics.false_breakout_probability, float)
        assert isinstance(metrics.reversal_probability, float)
        assert isinstance(metrics.trend_continuation_probability, float)
        assert isinstance(metrics.influence_duration_minutes, int)
        assert isinstance(metrics.peak_influence_time_minutes, int)
        assert isinstance(metrics.spread_impact, float)
        assert isinstance(metrics.liquidity_impact, float)
        assert isinstance(metrics.correlation_with_other_sessions, float)

    def test_analyze_session_influence_market_conditions(self: "TestSessionInfluenceAnalyzer") -> None:
        """Тест валидации рыночных условий."""
        market_data = self._create_test_market_data()
        result = self.analyzer.analyze_session_influence("BTCUSDT", market_data, self.session_profile)
        # Проверяем рыночные условия
        conditions = result.market_conditions
        assert isinstance(conditions, MarketConditions)
        assert isinstance(conditions.volatility, float)
        assert isinstance(conditions.volume, float)
        assert isinstance(conditions.spread, float)
        assert isinstance(conditions.liquidity, float)
        assert isinstance(conditions.momentum, float)
        assert isinstance(conditions.trend_strength, float)
        assert isinstance(conditions.market_regime, MarketRegime)
        assert isinstance(conditions.session_intensity, SessionIntensity)

    def test_analyze_session_influence_predictions(self: "TestSessionInfluenceAnalyzer") -> None:
        """Тест валидации прогнозов."""
        market_data = self._create_test_market_data()
        result = self.analyzer.analyze_session_influence("BTCUSDT", market_data, self.session_profile)
        # Проверяем прогнозы
        predictions = result.predictions
        assert isinstance(predictions, dict)
        assert "volatility_prediction" in predictions
        assert "volume_prediction" in predictions
        assert "direction_prediction" in predictions
        assert "momentum_prediction" in predictions

    def test_analyze_session_influence_risk_factors(self: "TestSessionInfluenceAnalyzer") -> None:
        """Тест валидации факторов риска."""
        market_data = self._create_test_market_data()
        result = self.analyzer.analyze_session_influence("BTCUSDT", market_data, self.session_profile)
        # Проверяем факторы риска
        risk_factors = result.risk_factors
        assert isinstance(risk_factors, list)
        for risk_factor in risk_factors:
            assert isinstance(risk_factor, dict)
            assert "type" in risk_factor
            assert "severity" in risk_factor
            assert "description" in risk_factor

    def _create_test_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        data = {
            "timestamp": pd.DatetimeIndex(pd.date_range(start="2024-01-01", periods=100, freq="1H")),
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(45000, 55000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "volatility": np.random.uniform(0.01, 0.05, 100),
            "spread": np.random.uniform(0.1, 1.0, 100),
            "liquidity": np.random.uniform(0.5, 1.0, 100),
            "momentum": np.random.uniform(-0.1, 0.1, 100),
            "trend_strength": np.random.uniform(0.0, 1.0, 100),
        }
        return pd.DataFrame(data)


class TestSessionAnalyzer:
    """Тесты для SessionAnalyzer."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.analyzer = SessionAnalyzer()
        self.session_profile = self._create_test_session_profile()

    def _create_test_session_profile(self) -> SessionProfile:
        """Создает тестовый профиль сессии."""
        return SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(start_time=time(0, 0), end_time=time(8, 0)),
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
                overlap_impact={"london": 1.2, "new_york": 0.9},
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
            false_breakout_probability=0.4,
        )

    def test_analyze_session_metrics(self: "TestSessionAnalyzer") -> None:
        """Тест анализа метрик сессии."""
        market_data = self._create_test_market_data()
        metrics = self.analyzer.analyze_session_metrics(market_data, self.session_profile)
        # Проверяем, что это словарь с правильными типами значений
        assert isinstance(metrics, dict)
        assert isinstance(metrics.get("volume_change_percent"), (int, float))
        assert isinstance(metrics.get("volatility_change_percent"), (int, float))
        assert isinstance(metrics.get("price_direction_bias"), (int, float))
        assert isinstance(metrics.get("momentum_strength"), (int, float))

    def test_analyze_market_conditions(self: "TestSessionAnalyzer") -> None:
        """Тест анализа рыночных условий."""
        market_data = self._create_test_market_data()
        conditions = self.analyzer.analyze_market_conditions(market_data, self.session_profile)
        # Проверяем, что это словарь с правильными типами значений
        assert isinstance(conditions, dict)
        assert isinstance(conditions.get("volatility"), (int, float))
        assert isinstance(conditions.get("volume"), (int, float))
        assert isinstance(conditions.get("spread"), (int, float))
        assert isinstance(conditions.get("liquidity"), (int, float))
        assert isinstance(conditions.get("momentum"), (int, float))
        assert isinstance(conditions.get("trend_strength"), (int, float))
        assert isinstance(conditions.get("market_regime"), str)
        assert isinstance(conditions.get("session_intensity"), str)

    def test_generate_predictions(self: "TestSessionAnalyzer") -> None:
        """Тест генерации прогнозов."""
        market_data = self._create_test_market_data()
        predictions = self.analyzer.generate_predictions(market_data, self.session_profile)
        assert isinstance(predictions, dict)
        assert "volatility_prediction" in predictions
        assert "volume_prediction" in predictions
        assert "direction_prediction" in predictions

    def test_identify_risk_factors(self: "TestSessionAnalyzer") -> None:
        """Тест идентификации факторов риска."""
        market_data = self._create_test_market_data()
        risk_factors = self.analyzer.identify_risk_factors(market_data, self.session_profile)
        assert isinstance(risk_factors, list)
        for risk_factor in risk_factors:
            assert isinstance(risk_factor, dict)
            assert "type" in risk_factor
            assert "severity" in risk_factor

    def _create_test_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1H"),
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(45000, 55000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "volatility": np.random.uniform(0.01, 0.05, 100),
            "spread": np.random.uniform(0.1, 1.0, 100),
            "liquidity": np.random.uniform(0.5, 1.0, 100),
            "momentum": np.random.uniform(-0.1, 0.1, 100),
            "trend_strength": np.random.uniform(0.0, 1.0, 100),
        }
        return pd.DataFrame(data)


class TestSessionManager:
    """Тесты для SessionManager."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.manager = SessionManager()

    def test_register_session_analyzer(self: "TestSessionManager") -> None:
        """Тест регистрации анализатора сессии."""
        analyzer = Mock()
        self.manager.register_session_analyzer(SessionType.ASIAN, analyzer)
        registered_analyzer = self.manager.get_session_analyzer(SessionType.ASIAN)
        assert registered_analyzer == analyzer

    def test_get_session_analyzer(self: "TestSessionManager") -> None:
        """Тест получения анализатора сессии."""
        analyzer = Mock()
        self.manager.register_session_analyzer(SessionType.ASIAN, analyzer)
        result = self.manager.get_session_analyzer(SessionType.ASIAN)
        assert result == analyzer

    def test_get_session_analyzer_default(self: "TestSessionManager") -> None:
        """Тест получения анализатора по умолчанию."""
        result = self.manager.get_session_analyzer(SessionType.ASIAN)
        assert result is not None

    def test_analyze_session(self: "TestSessionManager") -> None:
        """Тест анализа сессии."""
        market_data = self._create_test_market_data()
        result = self.manager.analyze_session(SessionType.ASIAN, market_data)
        assert isinstance(result, SessionAnalysisResult)

    def test_get_session_statistics(self: "TestSessionManager") -> None:
        """Тест получения статистики сессии."""
        statistics = self.manager.get_session_statistics(SessionType.ASIAN)
        assert isinstance(statistics, dict)
        assert "session_type" in statistics

    def _create_test_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1H"),
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(45000, 55000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "volatility": np.random.uniform(0.01, 0.05, 100),
            "spread": np.random.uniform(0.1, 1.0, 100),
            "liquidity": np.random.uniform(0.5, 1.0, 100),
            "momentum": np.random.uniform(-0.1, 0.1, 100),
            "trend_strength": np.random.uniform(0.0, 1.0, 100),
        }
        return pd.DataFrame(data)


class TestSessionOptimizer:
    """Тесты для SessionOptimizer."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.optimizer = SessionOptimizer()

    def test_optimize_session_parameters(self: "TestSessionOptimizer") -> None:
        """Тест оптимизации параметров сессии."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        optimized_profile = self.optimizer.optimize_session_parameters(profile, market_conditions)
        assert isinstance(optimized_profile, SessionProfile)
        assert optimized_profile.session_type == profile.session_type

    def test_optimize_trading_strategy(self: "TestSessionOptimizer") -> None:
        """Тест оптимизации торговой стратегии."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        strategy = self.optimizer.optimize_trading_strategy(profile, market_conditions)
        assert isinstance(strategy, dict)
        assert "entry_rules" in strategy
        assert "exit_rules" in strategy
        assert "risk_management" in strategy

    def test_optimize_risk_management(self: "TestSessionOptimizer") -> None:
        """Тест оптимизации управления рисками."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        risk_config = self.optimizer.optimize_risk_management(profile, market_conditions)
        assert isinstance(risk_config, dict)
        assert "position_size" in risk_config
        assert "stop_loss" in risk_config
        assert "take_profit" in risk_config

    def _create_test_session_profile(self) -> SessionProfile:
        """Создает тестовый профиль сессии."""
        return SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(start_time=time(0, 0), end_time=time(8, 0)),
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
                overlap_impact={"london": 1.2, "new_york": 0.9},
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
            false_breakout_probability=0.4,
        )

    def _create_test_market_conditions(self) -> MarketConditions:
        """Создает тестовые рыночные условия."""
        return MarketConditions(
            volatility=0.02,
            volume=3000.0,
            spread=0.5,
            liquidity=0.8,
            momentum=0.05,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING,
            session_intensity=SessionIntensity.HIGH,  # Используем существующий атрибут
        )

    def _create_test_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1H"),
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(45000, 55000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "volatility": np.random.uniform(0.01, 0.05, 100),
            "spread": np.random.uniform(0.1, 1.0, 100),
            "liquidity": np.random.uniform(0.5, 1.0, 100),
            "momentum": np.random.uniform(-0.1, 0.1, 100),
            "trend_strength": np.random.uniform(0.0, 1.0, 100),
        }
        return pd.DataFrame(data)


class TestSessionPredictor:
    """Тесты для SessionPredictor."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.predictor = SessionPredictor()

    def test_predict_session_behavior(self: "TestSessionPredictor") -> None:
        """Тест прогнозирования поведения сессии."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        prediction = self.predictor.predict_session_behavior(profile, market_conditions)
        assert isinstance(prediction, dict)
        assert "predicted_volatility" in prediction
        assert "predicted_volume" in prediction
        assert "predicted_direction_bias" in prediction
        assert "predicted_momentum" in prediction
        assert "reversal_probability" in prediction
        assert "continuation_probability" in prediction
        assert "false_breakout_probability" in prediction
        assert "manipulation_risk" in prediction
        assert "whale_activity_probability" in prediction
        assert "mm_activity_probability" in prediction

    def test_predict_session_transitions(self: "TestSessionPredictor") -> None:
        """Тест прогнозирования переходов сессий."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        transitions = self.predictor.predict_session_transitions(profile, market_conditions)
        assert isinstance(transitions, list)
        for transition in transitions:
            assert isinstance(transition, dict)
            assert "from_session" in transition
            assert "to_session" in transition
            assert "probability" in transition
            assert "expected_time" in transition

    def test_predict_market_regime_changes(self: "TestSessionPredictor") -> None:
        """Тест прогнозирования изменений рыночного режима."""
        profile = self._create_test_session_profile()
        market_conditions = self._create_test_market_conditions()
        regime_changes = self.predictor.predict_market_regime_changes(profile, market_conditions)
        assert isinstance(regime_changes, list)
        for change in regime_changes:
            assert isinstance(change, dict)
            assert "from_regime" in change
            assert "to_regime" in change
            assert "probability" in change
            assert "expected_time" in change

    def _create_test_session_profile(self) -> SessionProfile:
        """Создает тестовый профиль сессии."""
        return SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(start_time=time(0, 0), end_time=time(8, 0)),
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
                overlap_impact={"london": 1.2, "new_york": 0.9},
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
            false_breakout_probability=0.4,
        )

    def _create_test_market_conditions(self) -> MarketConditions:
        """Создает тестовые рыночные условия."""
        return MarketConditions(
            volatility=0.02,
            volume=3000.0,
            spread=0.5,
            liquidity=0.8,
            momentum=0.05,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING,
            session_intensity=SessionIntensity.HIGH,  # Используем существующий атрибут
        )


class TestSessionAnalyzerFactory:
    """Тесты для SessionAnalyzerFactory."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.factory = SessionAnalyzerFactory()

    def test_create_analyzer_for_session_type(self: "TestSessionAnalyzerFactory") -> None:
        """Тест создания анализатора для типа сессии."""
        analyzer = self.factory.create_analyzer_for_session_type(SessionType.ASIAN)
        assert analyzer is not None
        assert hasattr(analyzer, "analyze_session")

    def test_create_analyzer_with_custom_config(self: "TestSessionAnalyzerFactory") -> None:
        """Тест создания анализатора с пользовательской конфигурацией."""
        config = {"custom_param": "value"}
        analyzer = self.factory.create_analyzer_with_custom_config(SessionType.ASIAN, config)
        assert analyzer is not None
        assert hasattr(analyzer, "analyze_session")

    def test_get_available_analyzers(self: "TestSessionAnalyzerFactory") -> None:
        """Тест получения доступных анализаторов."""
        analyzers = self.factory.get_available_analyzers()
        assert isinstance(analyzers, list)
        assert len(analyzers) > 0
        for analyzer_info in analyzers:
            assert isinstance(analyzer_info, dict)
            assert "session_type" in analyzer_info
            assert "description" in analyzer_info


class TestSessionRepositories:
    """Тесты для репозиториев сессий."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.session_repo = SessionDataRepository()
        self.config_repo = SessionConfigurationRepository()

    def test_session_repository_save_and_get(self: "TestSessionRepositories") -> None:
        """Тест сохранения и получения профиля сессии."""
        profile = self._create_test_session_profile()
        self.session_repo.save_session_profile(profile)
        retrieved_profile = self.session_repo.get_session_profile(profile.session_type)
        assert retrieved_profile is not None
        assert retrieved_profile.session_type == profile.session_type

    def test_session_repository_get_all(self: "TestSessionRepositories") -> None:
        """Тест получения всех профилей сессий."""
        profiles = self.session_repo.get_all_session_profiles()
        assert isinstance(profiles, dict)
        assert len(profiles) > 0
        for session_type, profile in profiles.items():
            assert isinstance(session_type, SessionType)
            assert isinstance(profile, SessionProfile)

    def test_analysis_repository_save_and_get(self: "TestSessionRepositories") -> None:
        """Тест сохранения и получения результата анализа."""
        analysis_result = self._create_test_analysis_result()
        self.config_repo.save_session_analysis(analysis_result)
        retrieved_analysis = self.config_repo.get_session_analysis(
            analysis_result.session_type, analysis_result.timestamp, analysis_result.timestamp
        )
        assert isinstance(retrieved_analysis, list)
        assert len(retrieved_analysis) > 0

    def test_analysis_repository_get_statistics(self: "TestSessionRepositories") -> None:
        """Тест получения статистики анализа."""
        statistics = self.config_repo.get_session_statistics(SessionType.ASIAN)
        assert isinstance(statistics, dict)
        assert "session_type" in statistics
        assert "total_analyses" in statistics
        assert "avg_confidence" in statistics
        assert "success_rate" in statistics

    def _create_test_session_profile(self) -> SessionProfile:
        """Создает тестовый профиль сессии."""
        return SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=SessionTimeWindow(start_time=time(0, 0), end_time=time(8, 0)),
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
                overlap_impact={"london": 1.2, "new_york": 0.9},
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
            false_breakout_probability=0.4,
        )

    def _create_test_analysis_result(self) -> SessionAnalysisResult:
        """Создает тестовый результат анализа."""
        return SessionAnalysisResult(
            session_type=SessionType.ASIAN,
            session_phase=SessionPhase.ACTIVE,
            timestamp=Timestamp.now(),
            confidence=ConfidenceScore(0.8),
            metrics=SessionMetrics(
                volume_change_percent=0.1,
                volatility_change_percent=0.05,
                price_direction_bias=0.2,
                momentum_strength=0.6,
                false_breakout_probability=0.3,
                reversal_probability=0.25,
                trend_continuation_probability=0.55,
                influence_duration_minutes=45,
                peak_influence_time_minutes=30,
                spread_impact=0.1,
                liquidity_impact=0.05,
                correlation_with_other_sessions=0.3,
            ),
            market_conditions=MarketConditions(
                volatility=0.02,
                volume=3000.0,
                spread=0.5,
                liquidity=0.8,
                momentum=0.05,
                trend_strength=0.7,
                market_regime=MarketRegime.TRENDING,
                session_intensity=SessionIntensity.HIGH,  # Используем существующий атрибут
            ),
            predictions={},
            risk_factors=[],
        )
