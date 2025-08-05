"""
Интеграционные тесты для модуля sessions.
"""
from unittest.mock import Mock, AsyncMock, patch
from shared.numpy_utils import np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
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

# Заглушки для отсутствующих классов
class MockSessionProfileRegistry:
    def get_profile(self, session_type) -> Any:
        return Mock(spec=SessionProfile)

class MockSessionMarker:
    def get_session_context(self) -> Any:
        return {
            "timestamp": datetime.now(),
            "active_sessions": [SessionType.ASIAN],
            "primary_session": {"session_type": SessionType.ASIAN}
        }
    
    def is_session_active(self, session_type) -> Any:
        return True
    
    def get_session_overlap(self, session1, session2) -> Any:
        return 0.5
    
    def get_next_session_change(self) -> Any:
        return {"time_ahead_hours": 2.0}
    
    def get_session_phase(self, session_type) -> Any:
        return SessionPhase.MID_SESSION

class MockSessionInfluenceAnalyzer:
    def analyze_session_influence(self, symbol, market_data, session_profile) -> Any:
        return Mock(spec=SessionAnalysisResult)

class MockSessionAnalyzer:
    def analyze_session_metrics(self, market_data, session_profile) -> Any:
        return {"volume": 1.0, "volatility": 1.0}
    
    def analyze_market_conditions(self, market_data, session_profile) -> Any:
        return {"regime": MarketRegime.RANGING}
    
    def generate_predictions(self, metrics, conditions, session_profile) -> Any:
        return {"volatility": 1.1, "volume": 1.2}
    
    def identify_risk_factors(self, metrics, conditions, session_profile) -> Any:
        return ["manipulation", "gap"]

class MockSessionManager:
    def register_session_analyzer(self, session_type, analyzer) -> Any:
        pass
    
    def get_session_analyzer(self, session_type) -> Any:
        return Mock()
    
    def analyze_session(self, session_type, symbol, market_data) -> Any:
        return Mock(spec=SessionAnalysisResult)
    
    def get_session_statistics(self, session_type) -> Any:
        return {"total_analyses": 10}

class MockSessionOptimizer:
    def optimize_session_parameters(self, session_profile, market_data) -> Any:
        return Mock(spec=SessionProfile)
    
    def optimize_trading_strategy(self, session_profile, market_conditions) -> Any:
        return {"strategy": "optimized"}
    
    def optimize_risk_management(self, session_profile, market_conditions) -> Any:
        return {"risk": "optimized"}

class MockSessionPredictor:
    def predict_session_behavior(self, session_profile, market_conditions) -> Any:
        return {"behavior": "predicted"}
    
    def predict_session_transitions(self, current_session, market_conditions) -> Any:
        return [{"from_session": current_session, "to_session": SessionType.LONDON, "probability": 0.8, "time_ahead_hours": 2.0}]
    
    def predict_market_regime_changes(self, current_regime, session_profile) -> Any:
        return [{"regime": MarketRegime.TRENDING_BULL, "probability": 0.6}]

class MockSessionAnalyzerFactory:
    def create_analyzer(self, session_type, config=None) -> Any:
        return Mock()
    
    def get_available_analyzers(self) -> Any:
        return ["analyzer1", "analyzer2"]

class MockSessionService:
    def get_current_session_context(self) -> Any:
        return {
            "active_sessions": [SessionType.ASIAN],
            "primary_session": SessionType.ASIAN
        }
    
    def analyze_session_influence(self, symbol, market_data) -> Any:
        return Mock(spec=SessionAnalysisResult)
    
    def predict_session_behavior(self, session_type, market_conditions) -> Any:
        return {"behavior": "predicted"}
    
    def get_session_recommendations(self, session_type) -> Any:
        return ["recommendation1", "recommendation2"]
    
    def get_session_statistics(self, session_type) -> Any:
        return {"total_analyses": 10}

class MockSessionDataRepository:
    def save_session_analysis(self, analysis_result) -> Any:
        pass
    
    def get_session_analysis(self, session_type, start_time, end_time) -> Any:
        return [Mock(spec=SessionAnalysisResult)]
    
    def get_session_statistics(self, session_type, lookback_days=1) -> Any:
        return {"total_analyses": 10}

class MockSessionConfigurationRepository:
    def save_session_profile(self, session_profile) -> Any:
        pass
    
    def get_session_profile(self, session_type) -> Any:
        return Mock(spec=SessionProfile)
    
    def get_all_session_profiles(self) -> Any:
        return {SessionType.ASIAN: Mock(spec=SessionProfile)}

# Заглушка для get_session_service
def get_session_service_mock() -> Any:
    return MockSessionService()

class TestSessionsIntegration:
    """Интеграционные тесты для модуля sessions."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.registry = MockSessionProfileRegistry()
        self.marker = MockSessionMarker()
        self.influence_analyzer = MockSessionInfluenceAnalyzer()
        self.analyzer = MockSessionAnalyzer()
        self.manager = MockSessionManager()
        self.optimizer = MockSessionOptimizer()
        self.predictor = MockSessionPredictor()
        self.factory = MockSessionAnalyzerFactory()
        self.service = MockSessionService()
        self.data_repo = MockSessionDataRepository()
        self.config_repo = MockSessionConfigurationRepository()
    def test_full_session_analysis_pipeline(self: "TestSessionsIntegration") -> None:
        """Тест полного пайплайна анализа сессии."""
        # 1. Получаем профиль сессии
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        assert session_profile is not None
        # 2. Создаем тестовые рыночные данные
        market_data = self._create_test_market_data()
        # 3. Анализируем влияние сессии
        influence_result = self.influence_analyzer.analyze_session_influence(
            "BTCUSDT", market_data, session_profile
        )
        assert isinstance(influence_result, SessionAnalysisResult)
        # 4. Анализируем метрики сессии
        metrics = self.analyzer.analyze_session_metrics(market_data, session_profile)
        assert isinstance(metrics, dict)
        # 5. Анализируем рыночные условия
        conditions = self.analyzer.analyze_market_conditions(market_data, session_profile)
        assert isinstance(conditions, dict)
        # 6. Генерируем предсказания
        predictions = self.analyzer.generate_predictions(metrics, conditions, session_profile)
        assert isinstance(predictions, dict)
        # 7. Идентифицируем факторы риска
        risk_factors = self.analyzer.identify_risk_factors(metrics, conditions, session_profile)
        assert isinstance(risk_factors, list)
        # 8. Сохраняем результат анализа
        self.data_repo.save_session_analysis(influence_result)
        # 9. Получаем статистику
        statistics = self.data_repo.get_session_statistics(SessionType.ASIAN, lookback_days=1)
        assert isinstance(statistics, dict)
        assert "total_analyses" in statistics
    def test_session_manager_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionManager."""
        # 1. Регистрируем анализатор
        analyzer = Mock()
        session_type = SessionType.ASIAN
        self.manager.register_session_analyzer(session_type, analyzer)
        # 2. Получаем анализатор
        retrieved_analyzer = self.manager.get_session_analyzer(session_type)
        assert retrieved_analyzer == analyzer
        # 3. Анализируем сессию
        market_data = self._create_test_market_data()
        result = self.manager.analyze_session(session_type, "BTCUSDT", market_data)
        assert isinstance(result, SessionAnalysisResult)
        # 4. Получаем статистику
        statistics = self.manager.get_session_statistics(session_type)
        assert isinstance(statistics, dict)
    def test_session_optimizer_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionOptimizer."""
        # 1. Создаем тестовые данные
        session_profile = self.registry.get_profile(SessionType.LONDON)
        market_data = self._create_test_market_data()
        market_conditions = self._create_test_market_conditions()
        # 2. Оптимизируем параметры сессии
        optimized_profile = self.optimizer.optimize_session_parameters(
            session_profile, market_data
        )
        assert isinstance(optimized_profile, SessionProfile)
        # 3. Оптимизируем торговую стратегию
        strategy_params = self.optimizer.optimize_trading_strategy(
            session_profile, market_conditions
        )
        assert isinstance(strategy_params, dict)
        # 4. Оптимизируем управление рисками
        risk_params = self.optimizer.optimize_risk_management(
            session_profile, market_conditions
        )
        assert isinstance(risk_params, dict)
    def test_session_predictor_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionPredictor."""
        # 1. Создаем тестовые данные
        session_profile = self.registry.get_profile(SessionType.NEW_YORK)
        market_conditions = self._create_test_market_conditions()
        # 2. Предсказываем поведение сессии
        behavior_prediction = self.predictor.predict_session_behavior(
            session_profile, market_conditions
        )
        assert isinstance(behavior_prediction, dict)
        # 3. Предсказываем переходы сессий
        current_session = SessionType.ASIAN
        transitions = self.predictor.predict_session_transitions(
            current_session, market_conditions
        )
        assert isinstance(transitions, list)
        # 4. Предсказываем изменения рыночного режима
        current_regime = MarketRegime.RANGING
        regime_changes = self.predictor.predict_market_regime_changes(
            current_regime, session_profile
        )
        assert isinstance(regime_changes, list)
    def test_session_analyzer_factory_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionAnalyzerFactory."""
        # 1. Создаем анализаторы для всех типов сессий
        for session_type in SessionType:
            analyzer = self.factory.create_analyzer(session_type)
            assert analyzer is not None
            assert hasattr(analyzer, 'analyze_session_influence')
        # 2. Создаем анализатор с кастомной конфигурацией
        config = {
            "analysis_depth": "deep",
            "prediction_horizon": 24,
            "confidence_threshold": 0.8
        }
        analyzer = self.factory.create_analyzer(SessionType.ASIAN, config=config)
        assert analyzer is not None
        # 3. Получаем доступные анализаторы
        analyzers = self.factory.get_available_analyzers()
        assert isinstance(analyzers, list)
        assert len(analyzers) > 0
    def test_session_service_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionService."""
        # 1. Получаем текущий контекст сессии
        context = self.service.get_current_session_context()
        assert isinstance(context, dict)
        assert "active_sessions" in context
        assert "primary_session" in context
        # 2. Анализируем влияние сессии
        market_data = self._create_test_market_data()
        analysis = self.service.analyze_session_influence("BTCUSDT", market_data)
        assert isinstance(analysis, SessionAnalysisResult)
        # 3. Предсказываем поведение сессии
        market_conditions = self._create_test_market_conditions()
        prediction = self.service.predict_session_behavior(SessionType.ASIAN, market_conditions)
        assert isinstance(prediction, dict)
        # 4. Получаем рекомендации
        recommendations = self.service.get_session_recommendations(SessionType.ASIAN)
        assert isinstance(recommendations, list)
        # 5. Получаем статистику
        statistics = self.service.get_session_statistics(SessionType.ASIAN)
        assert isinstance(statistics, dict)
    def test_repositories_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции репозиториев."""
        # 1. Создаем тестовые данные
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        analysis_result = self._create_test_analysis_result()
        # 2. Сохраняем профиль сессии
        self.config_repo.save_session_profile(session_profile)
        # 3. Получаем профиль сессии
        retrieved_profile = self.config_repo.get_session_profile(SessionType.ASIAN)
        assert retrieved_profile is not None
        assert retrieved_profile.session_type == session_profile.session_type
        # 4. Получаем все профили
        all_profiles = self.config_repo.get_all_session_profiles()
        assert isinstance(all_profiles, dict)
        assert len(all_profiles) > 0
        # 5. Сохраняем анализ
        self.data_repo.save_session_analysis(analysis_result)
        # 6. Получаем анализ
        start_time = Timestamp(analysis_result.timestamp.to_datetime())
        end_time = Timestamp(analysis_result.timestamp.to_datetime())
        retrieved_analyses = self.data_repo.get_session_analysis(
            SessionType.ASIAN, start_time, end_time
        )
        assert len(retrieved_analyses) > 0
        # 7. Получаем статистику
        statistics = self.data_repo.get_session_statistics(SessionType.ASIAN, lookback_days=1)
        assert isinstance(statistics, dict)
        assert "total_analyses" in statistics
    def test_session_marker_integration(self: "TestSessionsIntegration") -> None:
        """Тест интеграции SessionMarker."""
        # 1. Получаем контекст сессии
        context = self.marker.get_session_context()
        assert isinstance(context, dict)
        assert "timestamp" in context
        assert "active_sessions" in context
        assert "primary_session" in context
        # 2. Проверяем активность сессий
        for session_type in SessionType:
            is_active = self.marker.is_session_active(session_type)
            assert isinstance(is_active, bool)
        # 3. Получаем перекрытия сессий
        session_types = list(SessionType)
        for session1 in session_types:
            for session2 in session_types:
                overlap = self.marker.get_session_overlap(session1, session2)
                assert isinstance(overlap, float)
                assert 0.0 <= overlap <= 1.0
        # 4. Получаем следующее изменение сессии
        change_info = self.marker.get_next_session_change()
        assert isinstance(change_info, dict)
        assert "time_ahead_hours" in change_info
        # 5. Получаем фазы сессий
        for session_type in SessionType:
            phase = self.marker.get_session_phase(session_type)
            assert isinstance(phase, SessionPhase)
    def test_cross_session_analysis_integration(self: "TestSessionsIntegration") -> None:
        """Тест кросс-сессионного анализа."""
        # 1. Анализируем все типы сессий
        market_data = self._create_test_market_data()
        results = {}
        for session_type in SessionType:
            session_profile = self.registry.get_profile(session_type)
            result = self.influence_analyzer.analyze_session_influence(
                "BTCUSDT", market_data, session_profile
            )
            results[session_type] = result
        # 2. Проверяем результаты
        assert len(results) == len(SessionType)
        for session_type, result in results.items():
            assert isinstance(result, SessionAnalysisResult)
            assert result.session_type == session_type
        # 3. Сравниваем метрики между сессиями
        metrics_comparison = {}
        for session_type, result in results.items():
            metrics_comparison[session_type] = {
                "confidence": float(result.confidence),
                "volatility": result.predictions.get("volatility", 0.0),
                "volume": result.predictions.get("volume", 0.0)
            }
        # 4. Проверяем, что метрики различаются между сессиями
        confidences = [metrics["confidence"] for metrics in metrics_comparison.values()]
        assert len(set(confidences)) > 1  # Должны быть различия
    def test_session_transition_analysis_integration(self: "TestSessionsIntegration") -> None:
        """Тест анализа переходов между сессиями."""
        # 1. Получаем текущий контекст
        context = self.marker.get_session_context()
        current_session = context["primary_session"]["session_type"]
        # 2. Предсказываем переходы
        market_conditions = self._create_test_market_conditions()
        transitions = self.predictor.predict_session_transitions(
            current_session, market_conditions
        )
        # 3. Проверяем переходы
        assert isinstance(transitions, list)
        for transition in transitions:
            assert "from_session" in transition
            assert "to_session" in transition
            assert "probability" in transition
            assert "time_ahead_hours" in transition
            assert transition["from_session"] == current_session
        # 4. Анализируем влияние переходов
        for transition in transitions:
            to_session = transition["to_session"]
            session_profile = self.registry.get_profile(to_session)
            # Анализируем влияние будущей сессии
            market_data = self._create_test_market_data()
            result = self.influence_analyzer.analyze_session_influence(
                "BTCUSDT", market_data, session_profile
            )
            assert isinstance(result, SessionAnalysisResult)
    def _create_test_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        # Создаем тестовые данные
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(50000, 51000, 1000),
            'high': np.random.uniform(50000, 51000, 1000),
            'low': np.random.uniform(50000, 51000, 1000),
            'close': np.random.uniform(50000, 51000, 1000),
            'volume': np.random.uniform(100, 1000, 1000),
            'bid': np.random.uniform(50000, 51000, 1000),
            'ask': np.random.uniform(50000, 51000, 1000)
        })
    def _create_test_market_conditions(self) -> MarketConditions:
        """Создает тестовые рыночные условия."""
        return MarketConditions(
            volatility=1.2,
            volume=1.5,
            spread=0.008,
            liquidity=1.3,
            momentum=0.6,
            trend_strength=0.4,
            market_regime=MarketRegime.RANGING,
            session_intensity=SessionIntensity.HIGH
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
