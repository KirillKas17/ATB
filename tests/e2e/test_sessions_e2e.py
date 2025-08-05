"""
E2E тесты для модуля sessions.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from shared.numpy_utils import np
import pandas as pd
from datetime import datetime, timezone, timedelta
from domain.type_definitions.session_types import (
    SessionType, SessionPhase, SessionProfile, SessionBehavior, SessionTimeWindow,
    MarketConditions, MarketRegime, SessionIntensity, SessionAnalysisResult,
    SessionMetrics, ConfidenceScore
)
from domain.value_objects.timestamp import Timestamp
from domain.sessions.session_profile import SessionProfileRegistry
from domain.sessions.session_marker import SessionMarker
from domain.sessions.session_influence_analyzer import SessionInfluenceAnalyzer
from domain.sessions.session_analyzer import SessionAnalyzer
from domain.sessions.session_manager import SessionManager
from domain.sessions.session_optimizer import SessionOptimizer
from domain.sessions.session_predictor import SessionPredictor
from domain.sessions.session_analyzer_factory import SessionAnalyzerFactory
from domain.sessions.factories import get_session_service
from domain.sessions.repositories import SessionDataRepository, SessionConfigurationRepository
class TestSessionsE2E:
    """E2E тесты для модуля sessions."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = SessionProfileRegistry()
        self.marker = SessionMarker(self.registry)
        self.influence_analyzer = SessionInfluenceAnalyzer()
        self.analyzer = SessionAnalyzer()
        self.manager = SessionManager()
        self.optimizer = SessionOptimizer()
        self.predictor = SessionPredictor()
        self.factory = SessionAnalyzerFactory()
        self.service = get_session_service()
        self.data_repo = SessionDataRepository()
        self.config_repo = SessionConfigurationRepository()
    @pytest.mark.e2e
    def test_complete_session_analysis_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест полного рабочего процесса анализа сессии."""
        # Шаг 1: Инициализация системы
        assert self.registry is not None
        assert self.marker is not None
        assert self.service is not None
        # Шаг 2: Получение текущего контекста сессии
        context = self.service.get_current_session_context()
        assert isinstance(context, dict)
        assert "active_sessions" in context
        assert "primary_session" in context
        # Шаг 3: Создание тестовых рыночных данных
        market_data = self._create_realistic_market_data()
        assert len(market_data) > 0
        assert "close" in market_data.columns
        assert "volume" in market_data.columns
        # Шаг 4: Анализ влияния сессии
        analysis = self.service.analyze_session_influence("BTCUSDT", market_data)
        assert isinstance(analysis, SessionAnalysisResult)
        assert analysis.session_type in SessionType
        assert analysis.session_phase in SessionPhase
        assert float(analysis.confidence) > 0.0
        # Шаг 5: Сохранение результатов анализа
        self.data_repo.save_session_analysis(analysis)
        # Шаг 6: Получение статистики
        statistics = self.data_repo.get_session_statistics(
            analysis.session_type, lookback_days=1
        )
        assert isinstance(statistics, dict)
        assert "total_analyses" in statistics
        assert statistics["total_analyses"] > 0
        # Шаг 7: Получение рекомендаций
        recommendations = self.service.get_session_recommendations(analysis.session_type)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Шаг 8: Проверка целостности данных
        retrieved_analyses = self.data_repo.get_session_analysis(
            analysis.session_type,
            Timestamp(analysis.timestamp.to_datetime() - timedelta(hours=1)),
            Timestamp(analysis.timestamp.to_datetime() + timedelta(hours=1))
        )
        assert len(retrieved_analyses) > 0
        assert any(a.timestamp == analysis.timestamp for a in retrieved_analyses)
    @pytest.mark.e2e
    def test_multi_session_analysis_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест анализа множественных сессий."""
        # Шаг 1: Анализ всех типов сессий
        market_data = self._create_realistic_market_data()
        session_results = {}
        for session_type in SessionType:
            session_profile = self.registry.get_profile(session_type)
            result = self.influence_analyzer.analyze_session_influence(
                "BTCUSDT", market_data, session_profile
            )
            session_results[session_type] = result
            # Сохраняем результат
            self.data_repo.save_session_analysis(result)
        # Шаг 2: Проверка результатов
        assert len(session_results) == len(SessionType)
        for session_type, result in session_results.items():
            assert isinstance(result, SessionAnalysisResult)
            assert result.session_type == session_type
            assert float(result.confidence) > 0.0
        # Шаг 3: Сравнительный анализ
        confidences = [float(result.confidence) for result in session_results.values()]
        volatilities = [result.predictions.get("volatility", 0.0) for result in session_results.values()]
        # Проверяем, что есть различия между сессиями
        assert len(set(confidences)) > 1
        assert len(set(volatilities)) > 1
        # Шаг 4: Получение сводной статистики
        for session_type in SessionType:
            summary = self.data_repo.get_session_analysis_summary(session_type, lookback_days=1)
            assert isinstance(summary, dict)
            assert "session_type" in summary
            assert summary["session_type"] == session_type.value
    @pytest.mark.e2e
    def test_session_optimization_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест оптимизации сессий."""
        # Шаг 1: Получение профиля сессии
        session_profile = self.registry.get_profile(SessionType.LONDON)
        assert session_profile is not None
        # Шаг 2: Создание рыночных данных
        market_data = self._create_realistic_market_data()
        market_conditions = self._create_realistic_market_conditions()
        # Шаг 3: Оптимизация параметров сессии
        optimized_profile = self.optimizer.optimize_session_parameters(
            session_profile, market_data
        )
        assert isinstance(optimized_profile, SessionProfile)
        assert optimized_profile.session_type == session_profile.session_type
        # Шаг 4: Оптимизация торговой стратегии
        strategy_params = self.optimizer.optimize_trading_strategy(
            session_profile, market_conditions
        )
        assert isinstance(strategy_params, dict)
        assert len(strategy_params) > 0
        # Шаг 5: Оптимизация управления рисками
        risk_params = self.optimizer.optimize_risk_management(
            session_profile, market_conditions
        )
        assert isinstance(risk_params, dict)
        assert len(risk_params) > 0
        # Шаг 6: Сохранение оптимизированного профиля
        self.config_repo.save_session_profile(optimized_profile)
        # Шаг 7: Проверка сохранения
        retrieved_profile = self.config_repo.get_session_profile(SessionType.LONDON)
        assert retrieved_profile is not None
        assert retrieved_profile.session_type == optimized_profile.session_type
    @pytest.mark.e2e
    def test_session_prediction_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест предсказания сессий."""
        # Шаг 1: Получение текущего контекста
        context = self.service.get_current_session_context()
        current_session = context["primary_session"]["session_type"]
        # Шаг 2: Создание рыночных условий
        market_conditions = self._create_realistic_market_conditions()
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        # Шаг 3: Предсказание поведения сессии
        behavior_prediction = self.predictor.predict_session_behavior(
            session_profile, market_conditions
        )
        assert isinstance(behavior_prediction, dict)
        assert len(behavior_prediction) > 0
        # Шаг 4: Предсказание переходов сессий
        transitions = self.predictor.predict_session_transitions(
            current_session, market_conditions
        )
        assert isinstance(transitions, list)
        for transition in transitions:
            assert "from_session" in transition
            assert "to_session" in transition
            assert "probability" in transition
            assert "time_ahead_hours" in transition
        # Шаг 5: Предсказание изменений рыночного режима
        current_regime = MarketRegime.RANGING
        regime_changes = self.predictor.predict_market_regime_changes(
            current_regime, session_profile
        )
        assert isinstance(regime_changes, list)
        for change in regime_changes:
            assert "from_regime" in change
            assert "to_regime" in change
            assert "probability" in change
            assert "time_ahead_hours" in change
        # Шаг 6: Интеграция предсказаний
        for transition in transitions:
            to_session = transition["to_session"]
            if transition["probability"] > 0.5:  # Высокая вероятность перехода
                # Анализируем влияние будущей сессии
                future_profile = self.registry.get_profile(to_session)
                market_data = self._create_realistic_market_data()
                future_analysis = self.influence_analyzer.analyze_session_influence(
                    "BTCUSDT", market_data, future_profile
                )
                assert isinstance(future_analysis, SessionAnalysisResult)
    @pytest.mark.e2e
    def test_session_manager_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест управления сессиями."""
        # Шаг 1: Регистрация анализаторов
        for session_type in SessionType:
            analyzer = self.factory.create_analyzer(session_type)
            self.manager.register_session_analyzer(session_type, analyzer)
        # Шаг 2: Проверка регистрации
        for session_type in SessionType:
            analyzer = self.manager.get_session_analyzer(session_type)
            assert analyzer is not None
            assert hasattr(analyzer, 'analyze_session_influence')
        # Шаг 3: Анализ сессий через менеджер
        market_data = self._create_realistic_market_data()
        for session_type in SessionType:
            result = self.manager.analyze_session(session_type, "BTCUSDT", market_data)
            assert isinstance(result, SessionAnalysisResult)
            assert result.session_type == session_type
        # Шаг 4: Получение статистики через менеджер
        for session_type in SessionType:
            statistics = self.manager.get_session_statistics(session_type)
            assert isinstance(statistics, dict)
            assert "session_type" in statistics
            assert statistics["session_type"] == session_type.value
    @pytest.mark.e2e
    def test_session_service_complete_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест полного рабочего процесса SessionService."""
        # Шаг 1: Получение текущего контекста
        context = self.service.get_current_session_context()
        assert isinstance(context, dict)
        # Шаг 2: Анализ влияния сессии
        market_data = self._create_realistic_market_data()
        analysis = self.service.analyze_session_influence("BTCUSDT", market_data)
        assert isinstance(analysis, SessionAnalysisResult)
        # Шаг 3: Предсказание поведения
        market_conditions = self._create_realistic_market_conditions()
        prediction = self.service.predict_session_behavior(analysis.session_type, market_conditions)
        assert isinstance(prediction, dict)
        # Шаг 4: Получение рекомендаций
        recommendations = self.service.get_session_recommendations(analysis.session_type)
        assert isinstance(recommendations, list)
        # Шаг 5: Получение статистики
        statistics = self.service.get_session_statistics(analysis.session_type)
        assert isinstance(statistics, dict)
        # Шаг 6: Проверка переходов
        is_transition = self.service.is_transition_period()
        assert isinstance(is_transition, bool)
        # Шаг 7: Получение активных переходов
        active_transitions = self.service.get_active_transitions()
        assert isinstance(active_transitions, list)
        # Шаг 8: Получение перекрытий сессий
        for session_type in SessionType:
            overlap = self.service.get_session_overlap(analysis.session_type, session_type)
            assert isinstance(overlap, float)
            assert 0.0 <= overlap <= 1.0
        # Шаг 9: Получение фазы сессии
        phase = self.service.get_session_phase(analysis.session_type)
        assert isinstance(phase, SessionPhase)
        # Шаг 10: Получение следующего изменения
        next_change = self.service.get_next_session_change()
        assert isinstance(next_change, dict)
        assert "time_ahead_hours" in next_change
        # Шаг 11: Проверка здоровья системы
        health_check = self.service.get_session_health_check()
        assert isinstance(health_check, dict)
        assert "status" in health_check
        assert health_check["status"] in ["healthy", "warning", "error"]
    @pytest.mark.e2e
    def test_session_data_persistence_workflow(self: "TestSessionsE2E") -> None:
        """E2E тест персистентности данных сессий."""
        # Шаг 1: Создание тестовых данных
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        analysis_result = self._create_test_analysis_result()
        # Шаг 2: Сохранение профиля
        self.config_repo.save_session_profile(session_profile)
        # Шаг 3: Проверка сохранения профиля
        retrieved_profile = self.config_repo.get_session_profile(SessionType.ASIAN)
        assert retrieved_profile is not None
        assert retrieved_profile.session_type == session_profile.session_type
        # Шаг 4: Получение всех профилей
        all_profiles = self.config_repo.get_all_session_profiles()
        assert isinstance(all_profiles, dict)
        assert len(all_profiles) > 0
        assert SessionType.ASIAN in all_profiles
        # Шаг 5: Сохранение анализа
        self.data_repo.save_session_analysis(analysis_result)
        # Шаг 6: Проверка сохранения анализа
        start_time = Timestamp(analysis_result.timestamp.to_datetime() - timedelta(hours=1))
        end_time = Timestamp(analysis_result.timestamp.to_datetime() + timedelta(hours=1))
        retrieved_analyses = self.data_repo.get_session_analysis(
            SessionType.ASIAN, start_time, end_time
        )
        assert len(retrieved_analyses) > 0
        # Шаг 7: Получение статистики
        statistics = self.data_repo.get_session_statistics(SessionType.ASIAN, lookback_days=1)
        assert isinstance(statistics, dict)
        assert "total_analyses" in statistics
        assert statistics["total_analyses"] > 0
        # Шаг 8: Получение сводки
        summary = self.data_repo.get_session_analysis_summary(SessionType.ASIAN, lookback_days=1)
        assert isinstance(summary, dict)
        assert "session_type" in summary
        assert summary["session_type"] == SessionType.ASIAN.value
        # Шаг 9: Удаление данных
        deleted_count = self.data_repo.delete_session_analysis(
            SessionType.ASIAN, start_time, end_time
        )
        assert deleted_count > 0
        # Шаг 10: Проверка удаления
        remaining_analyses = self.data_repo.get_session_analysis(
            SessionType.ASIAN, start_time, end_time
        )
        assert len(remaining_analyses) == 0
    def _create_realistic_market_data(self) -> pd.DataFrame:
        """Создает реалистичные рыночные данные."""
        # Создаем временной ряд с реалистичными паттернами
        # Исправляем использование pd.date_range на pd.DatetimeIndex
        timestamps = pd.DatetimeIndex(pd.date_range('2024-01-01', periods=1000, freq='1min'))
        
        # Базовые цены
        base_price = 50000
        price_changes = np.random.normal(0, 100, 1000)  # Нормальное распределение изменений
        prices = base_price + np.cumsum(price_changes)
        
        # Объемы с паттернами
        base_volume = 1000
        volume_multiplier = 1 + 0.5 * np.sin(np.arange(1000) * 2 * np.pi / 1440)  # Дневные паттерны
        volumes = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, 1000)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * np.random.uniform(1.0, 1.02, 1000),
            'low': prices * np.random.uniform(0.98, 1.0, 1000),
            'close': prices,
            'volume': volumes,
            'bid': prices * np.random.uniform(0.999, 1.0, 1000),
            'ask': prices * np.random.uniform(1.0, 1.001, 1000)
        })

    def _create_realistic_market_conditions(self) -> MarketConditions:
        """Создает реалистичные рыночные условия."""
        # Исправляем передачу enum значений в np.random.choice
        market_regimes = [regime.value for regime in MarketRegime]
        session_intensities = [intensity.value for intensity in SessionIntensity]
        
        return MarketConditions(
            volatility=np.random.uniform(0.8, 2.0),
            volume=np.random.uniform(0.5, 2.0),
            spread=np.random.uniform(0.005, 0.02),
            liquidity=np.random.uniform(0.7, 1.5),
            momentum=np.random.uniform(0.0, 1.0),
            trend_strength=np.random.uniform(0.0, 1.0),
            market_regime=MarketRegime(np.random.choice(market_regimes)),
            session_intensity=SessionIntensity(np.random.choice(session_intensities))
        )
    def _create_test_analysis_result(self) -> SessionAnalysisResult:
        """Создает тестовый результат анализа."""
        return SessionAnalysisResult(
            session_type=SessionType.ASIAN,
            session_phase=SessionPhase.MID_SESSION,
            timestamp=Timestamp(datetime.now(timezone.utc)),
            confidence=ConfidenceScore(0.85),
            metrics=SessionMetrics(
                volume_change_percent=5.0,
                volatility_change_percent=3.0,
                price_direction_bias=0.1,
                momentum_strength=0.6,
                false_breakout_probability=0.3,
                reversal_probability=0.2,
                trend_continuation_probability=0.6,
                influence_duration_minutes=60,
                peak_influence_time_minutes=30,
                spread_impact=1.0,
                liquidity_impact=1.0,
                correlation_with_other_sessions=0.8
            ),
            market_conditions=MarketConditions(
                volatility=1.1,
                volume=1.2,
                spread=0.01,
                liquidity=1.1,
                momentum=0.5,
                trend_strength=0.3,
                market_regime=MarketRegime.RANGING,
                session_intensity=SessionIntensity.NORMAL
            ),
            predictions={"volatility": 1.1, "volume": 1.2},
            risk_factors=["manipulation", "gap"]
        ) 
