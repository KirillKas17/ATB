"""
Тесты производительности для модуля sessions.
"""
import pytest
import time
import cProfile
import pstats
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
class TestSessionsPerformance:
    """Тесты производительности для модуля sessions."""
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
    @pytest.mark.performance
    def test_session_profile_registry_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionProfileRegistry."""
        # Тест получения профилей
        start_time = time.time()
        for _ in range(1000):
            for session_type in SessionType:
                profile = self.registry.get_profile(session_type)
                assert profile is not None
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем, что выполнение занимает менее 1 секунды
        assert execution_time < 1.0, f"Execution time: {execution_time:.3f}s"
        # Тест получения всех профилей
        start_time = time.time()
        for _ in range(100):
            profiles = self.registry.get_all_profiles()
            assert len(profiles) == len(SessionType)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 0.5, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_session_marker_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionMarker."""
        # Тест получения контекста сессии
        start_time = time.time()
        for _ in range(1000):
            context = self.marker.get_session_context()
            assert context is not None
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Execution time: {execution_time:.3f}s"
        # Тест проверки активности сессий
        start_time = time.time()
        for _ in range(1000):
            for session_type in SessionType:
                is_active = self.marker.is_session_active(session_type)
                assert isinstance(is_active, bool)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_session_influence_analyzer_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionInfluenceAnalyzer."""
        market_data = self._create_large_market_data(10000)
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        # Профилирование анализа
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        for _ in range(10):
            result = self.influence_analyzer.analyze_session_influence(
                "BTCUSDT", market_data, session_profile
            )
            assert isinstance(result, SessionAnalysisResult)
        end_time = time.time()
        profiler.disable()
        execution_time = end_time - start_time
        # Проверяем производительность
        assert execution_time < 5.0, f"Execution time: {execution_time:.3f}s"
        # Анализируем профиль
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        # Проверяем, что нет узких мест
        total_calls = sum(stat[1] for stat in stats.stats.values())
        assert total_calls < 10000, f"Too many function calls: {total_calls}"
    @pytest.mark.performance
    def test_session_analyzer_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionAnalyzer."""
        market_data = self._create_large_market_data(5000)
        session_profile = self.registry.get_profile(SessionType.LONDON)
        start_time = time.time()
        for _ in range(20):
            # Анализ метрик
            metrics = self.analyzer.analyze_session_metrics(market_data, session_profile)
            assert isinstance(metrics, dict)
            # Анализ рыночных условий
            conditions = self.analyzer.analyze_market_conditions(market_data, session_profile)
            assert isinstance(conditions, dict)
            # Генерация предсказаний
            predictions = self.analyzer.generate_predictions(metrics, conditions, session_profile)
            assert isinstance(predictions, dict)
            # Идентификация факторов риска
            risk_factors = self.analyzer.identify_risk_factors(metrics, conditions, session_profile)
            assert isinstance(risk_factors, list)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 3.0, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_session_manager_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionManager."""
        # Регистрация анализаторов
        start_time = time.time()
        for session_type in SessionType:
            analyzer = self.factory.create_analyzer(session_type)
            self.manager.register_session_analyzer(session_type, analyzer)
        end_time = time.time()
        registration_time = end_time - start_time
        assert registration_time < 0.1, f"Registration time: {registration_time:.3f}s"
        # Анализ сессий
        market_data = self._create_large_market_data(1000)
        start_time = time.time()
        for _ in range(10):
            for session_type in SessionType:
                result = self.manager.analyze_session(session_type, "BTCUSDT", market_data)
                assert isinstance(result, SessionAnalysisResult)
        end_time = time.time()
        analysis_time = end_time - start_time
        assert analysis_time < 5.0, f"Analysis time: {analysis_time:.3f}s"
    @pytest.mark.performance
    def test_session_optimizer_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionOptimizer."""
        session_profile = self.registry.get_profile(SessionType.NEW_YORK)
        market_data = self._create_large_market_data(2000)
        market_conditions = self._create_market_conditions()
        start_time = time.time()
        for _ in range(5):
            # Оптимизация параметров сессии
            optimized_profile = self.optimizer.optimize_session_parameters(
                session_profile, market_data
            )
            assert isinstance(optimized_profile, SessionProfile)
            # Оптимизация торговой стратегии
            strategy_params = self.optimizer.optimize_trading_strategy(
                session_profile, market_conditions
            )
            assert isinstance(strategy_params, dict)
            # Оптимизация управления рисками
            risk_params = self.optimizer.optimize_risk_management(
                session_profile, market_conditions
            )
            assert isinstance(risk_params, dict)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 3.0, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_session_predictor_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionPredictor."""
        session_profile = self.registry.get_profile(SessionType.CRYPTO_24H)
        market_conditions = self._create_market_conditions()
        current_session = SessionType.ASIAN
        current_regime = MarketRegime.RANGING
        start_time = time.time()
        for _ in range(50):
            # Предсказание поведения сессии
            behavior_prediction = self.predictor.predict_session_behavior(
                session_profile, market_conditions
            )
            assert isinstance(behavior_prediction, dict)
            # Предсказание переходов сессий
            transitions = self.predictor.predict_session_transitions(
                current_session, market_conditions
            )
            assert isinstance(transitions, list)
            # Предсказание изменений рыночного режима
            regime_changes = self.predictor.predict_market_regime_changes(
                current_regime, session_profile
            )
            assert isinstance(regime_changes, list)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_session_service_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности SessionService."""
        market_data = self._create_large_market_data(1000)
        market_conditions = self._create_market_conditions()
        start_time = time.time()
        for _ in range(10):
            # Получение контекста сессии
            context = self.service.get_current_session_context()
            assert isinstance(context, dict)
            # Анализ влияния сессии
            analysis = self.service.analyze_session_influence("BTCUSDT", market_data)
            assert isinstance(analysis, SessionAnalysisResult)
            # Предсказание поведения
            prediction = self.service.predict_session_behavior(analysis.session_type, market_conditions)
            assert isinstance(prediction, dict)
            # Получение рекомендаций
            recommendations = self.service.get_session_recommendations(analysis.session_type)
            assert isinstance(recommendations, list)
            # Получение статистики
            statistics = self.service.get_session_statistics(analysis.session_type)
            assert isinstance(statistics, dict)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"Execution time: {execution_time:.3f}s"
    @pytest.mark.performance
    def test_repositories_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности репозиториев."""
        session_profile = self.registry.get_profile(SessionType.ASIAN)
        analysis_result = self._create_test_analysis_result()
        # Тест сохранения и получения профилей
        start_time = time.time()
        for _ in range(100):
            self.config_repo.save_session_profile(session_profile)
            retrieved_profile = self.config_repo.get_session_profile(SessionType.ASIAN)
            assert retrieved_profile is not None
        end_time = time.time()
        profile_time = end_time - start_time
        assert profile_time < 1.0, f"Profile operations time: {profile_time:.3f}s"
        # Тест сохранения и получения анализов
        start_time = time.time()
        for _ in range(50):
            self.data_repo.save_session_analysis(analysis_result)
            start_time_range = Timestamp(analysis_result.timestamp.to_datetime() - timedelta(hours=1))
            end_time_range = Timestamp(analysis_result.timestamp.to_datetime() + timedelta(hours=1))
            retrieved_analyses = self.data_repo.get_session_analysis(
                SessionType.ASIAN, start_time_range, end_time_range
            )
            assert len(retrieved_analyses) > 0
        end_time = time.time()
        analysis_time = end_time - start_time
        assert analysis_time < 2.0, f"Analysis operations time: {analysis_time:.3f}s"
    @pytest.mark.performance
    def test_memory_usage_performance(self: "TestSessionsPerformance") -> None:
        """Тест использования памяти."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Выполняем интенсивные операции
        market_data = self._create_large_market_data(5000)
        for _ in range(10):
            for session_type in SessionType:
                session_profile = self.registry.get_profile(session_type)
                result = self.influence_analyzer.analyze_session_influence(
                    "BTCUSDT", market_data, session_profile
                )
                self.data_repo.save_session_analysis(result)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        # Проверяем, что увеличение памяти не превышает 100 MB
        assert memory_increase < 100, f"Memory increase: {memory_increase:.1f} MB"
    @pytest.mark.performance
    def test_concurrent_operations_performance(self: "TestSessionsPerformance") -> None:
        """Тест производительности при конкурентных операциях."""
        import threading
        import queue
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        def analyze_session(session_type: SessionType, market_data: pd.DataFrame) -> Any:
            try:
                session_profile = self.registry.get_profile(session_type)
                result = self.influence_analyzer.analyze_session_influence(
                    "BTCUSDT", market_data, session_profile
                )
                results_queue.put((session_type, result))
            except Exception as e:
                errors_queue.put((session_type, e))
        # Создаем потоки для каждого типа сессии
        market_data = self._create_large_market_data(1000)
        threads = []
        start_time = time.time()
        for session_type in SessionType:
            thread = threading.Thread(
                target=analyze_session,
                args=(session_type, market_data)
            )
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем результаты
        assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
        assert results_queue.qsize() == len(SessionType)
        # Проверяем производительность
        assert execution_time < 3.0, f"Concurrent execution time: {execution_time:.3f}s"
    def _create_large_market_data(self, size: int) -> pd.DataFrame:
        """Создает большие рыночные данные для тестирования производительности."""
        # Создаем тестовые данные
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        # Создаем реалистичные данные с трендами
        base_price = 50000
        trend = np.linspace(0, 1000, size)  # Линейный тренд
        noise = np.random.normal(0, 50, size)  # Шум
        prices = base_price + trend + noise
        volumes = np.random.uniform(500, 2000, size)
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * np.random.uniform(1.0, 1.02, size),
            'low': prices * np.random.uniform(0.98, 1.0, size),
            'close': prices,
            'volume': volumes,
            'bid': prices * np.random.uniform(0.999, 1.0, size),
            'ask': prices * np.random.uniform(1.0, 1.001, size)
        })
    def _create_market_conditions(self) -> MarketConditions:
        """Создает рыночные условия для тестирования."""
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
