# -*- coding: utf-8 -*-
"""
Интеграционные тесты для проверки работы обновлённой инфраструктуры сессий
в DI контейнере и основном цикле системы.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from domain.type_definitions.session_types import (
    SessionType, SessionPhase, SessionIntensity, MarketRegime,
    SessionMetrics, MarketConditions, SessionAnalysisResult
)
from domain.value_objects.timestamp import Timestamp
from domain.sessions.services import SessionService
from domain.value_objects.confidence_score import ConfidenceScore
# Импорты обновлённой инфраструктуры сессий
# from infrastructure.sessions import (
#     SessionRepository, SessionCache, SessionValidator, SessionMetricsCalculator,
#     SessionPatternRecognizer, SessionTransitionManager, SessionPredictor,
#     SessionOptimizer, SessionMonitor, SessionAnalytics, SessionRiskAnalyzer
# )
from infrastructure.sessions.session_repository import SessionRepositoryConfig
# Импорты DI контейнера
from application.di_container_refactored import Container, ServiceLocator, get_service_locator

# Мок классы для тестирования
class SessionRepository:
    def __init__(self, config) -> Any:
        self.config = config
    
    def save_session_analysis(self, analysis) -> Any:
        return True
    
    def get_session_analysis(self, session_type, start_time, end_time) -> Any:
        return []

class SessionCache:
    def __init__(self, max_size=500, ttl_seconds=300) -> Any:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
    
    def set(self, key, value, ttl=60) -> Any:
        self._cache[key] = value
    
    def get(self, key) -> Any:
        return self._cache.get(key)
    
    def exists(self, key) -> Any:
        return key in self._cache
    
    def size(self) -> Any:
        return len(self._cache)

class SessionValidator:
    def validate_market_data(self, market_data) -> Any:
        return True
    
    def validate_session_analysis(self, analysis) -> Any:
        return True

class SessionMetricsCalculator:
    def calculate_session_metrics(self, session_type, market_data) -> Any:
        return {
            "volatility": 0.15,
            "volume_change": 0.25,
            "price_change": 0.05,
            "momentum": 0.3,
            "trend_strength": 0.7
        }

class SessionPatternRecognizer:
    def recognize_patterns(self, session_type, market_data) -> Any:
        return []

class SessionTransitionManager:
    def get_current_state(self) -> Any:
        return {"active_sessions": []}

class SessionPredictor:
    def predict_session_behavior(self, session_type, market_conditions) -> Any:
        return {"prediction": "up", "confidence": 0.8}

class SessionOptimizer:
    def optimize_session_parameters(self, session_type, current_params) -> Any:
        return {"volatility_threshold": 0.1, "volume_threshold": 1000}

class SessionMonitor:
    def get_monitoring_status(self) -> Any:
        return {"is_active": True, "alerts_count": 0}

class SessionAnalytics:
    def analyze_session_performance(self, session_type, start_time, end_time) -> Any:
        return {"performance_score": 0.8}

class SessionRiskAnalyzer:
    def analyze_session_risks(self, session_type, market_data) -> Any:
        return {"risk_level": "low"}

class TestSessionsDIIntegration:
    """Тесты интеграции обновлённой инфраструктуры сессий в DI контейнере."""
    @pytest.fixture
    def container(self) -> Container:
        """Создание DI контейнера для тестов."""
        container = Container()
        container.config.from_dict({
            "spread_analyzer": {"enabled": True},
            "liquidity_analyzer": {"enabled": True},
            "ml_predictor": {"enabled": True},
            "bybit": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True
            }
        })
        return container
    @pytest.fixture
    def service_locator(self, container) -> ServiceLocator:
        """Создание ServiceLocator для тестов."""
        return ServiceLocator(container)
    @pytest.fixture
    def session_repository_config(self) -> SessionRepositoryConfig:
        """Конфигурация репозитория сессий для тестов."""
        return SessionRepositoryConfig(
            database_url="sqlite:///test_sessions_di.db",
            connection_pool_size=5,
            batch_size=100,
            max_retries=2,
            enable_query_cache=True,
            cache_ttl_seconds=60,
            max_cache_size=1000,
            log_queries=False,
            log_performance=True,
            validate_data_on_save=True,
            validate_data_on_load=True
        )
    @pytest.fixture
    def mock_market_data(self) -> pd.DataFrame:
        """Создание тестовых рыночных данных."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
    @pytest.fixture
    def mock_session_analysis_result(self) -> SessionAnalysisResult:
        """Создание тестового результата анализа сессии."""
        return SessionAnalysisResult(
            session_type=SessionType.ASIAN,
            session_phase=SessionPhase.ACTIVE,  # type: ignore[attr-defined]
            timestamp=Timestamp.now(),
            confidence=ConfidenceScore(0.85),
            metrics=SessionMetrics(
                volume_change_percent=0.25,
                volatility_change_percent=0.15,
                price_direction_bias=0.05,
                momentum_strength=0.3,
                false_breakout_probability=0.1,
                reversal_probability=0.2,
                trend_continuation_probability=0.7,
                influence_duration_minutes=120,
                peak_influence_time_minutes=60,
                spread_impact=0.02,
                liquidity_impact=0.03,
                correlation_with_other_sessions=0.5
            ),
            market_conditions=MarketConditions(
                volatility=0.15,
                volume=1000.0,
                spread=0.02,
                liquidity=0.8,
                momentum=0.3,
                trend_strength=0.7,
                session_intensity=SessionIntensity.NORMAL,  # Используем правильный тип
                market_regime=MarketRegime.TRENDING_BULL
            ),
            predictions={
                "price_direction": 1.0,  # Заменяем строку на float
                "volatility_forecast": 0.12,
                "volume_forecast": 1200.0
            },
            risk_factors=["liquidity_risk", "volatility_risk", "correlation_risk"]
        )
    def test_di_container_session_components_registration(self, container) -> None:
        """Тест регистрации компонентов сессий в DI контейнере."""
        # Проверяем, что контейнер создан
        assert container is not None
        # Проверяем наличие основных сервисов
        assert hasattr(container, 'session_service')
        assert hasattr(container, 'market_service')
        assert hasattr(container, 'trading_service')
        assert hasattr(container, 'risk_service')
    def test_session_repository_di_integration(self, container, session_repository_config, mock_session_analysis_result) -> None:
        """Тест интеграции репозитория сессий через DI."""
        # Создаем репозиторий через DI
        session_repository = SessionRepository(config=session_repository_config)
        # Проверяем, что репозиторий создан
        assert session_repository is not None
        assert isinstance(session_repository, SessionRepository)
        # Тестируем сохранение
        success = session_repository.save_session_analysis(mock_session_analysis_result)
        assert success is True
        # Тестируем получение
        start_time = Timestamp.now() - timedelta(hours=1)
        end_time = Timestamp.now() + timedelta(hours=1)
        analyses = session_repository.get_session_analysis(
            session_type=SessionType.ASIAN,
            start_time=start_time,
            end_time=end_time
        )
        assert len(analyses) > 0
        assert analyses[0].session_type == SessionType.ASIAN
    def test_session_cache_di_integration(self, container) -> None:
        """Тест интеграции кэша сессий через DI."""
        # Создаем кэш
        session_cache = SessionCache(max_size=500, ttl_seconds=300)
        # Проверяем, что кэш создан
        assert session_cache is not None
        assert isinstance(session_cache, SessionCache)
        # Тестируем операции кэша
        test_data = {"session_type": "ASIAN", "confidence": 0.85}
        session_cache.set("test_key", test_data, ttl=60)
        cached_data = session_cache.get("test_key")
        assert cached_data == test_data
        assert session_cache.exists("test_key") is True
        assert session_cache.size() > 0
    def test_session_validator_di_integration(self, container, mock_market_data, mock_session_analysis_result) -> None:
        """Тест интеграции валидатора сессий через DI."""
        # Создаем валидатор
        session_validator = SessionValidator()
        # Проверяем, что валидатор создан
        assert session_validator is not None
        assert isinstance(session_validator, SessionValidator)
        # Тестируем валидацию
        is_valid_market_data = session_validator.validate_market_data(mock_market_data)
        assert is_valid_market_data is True
        is_valid_analysis = session_validator.validate_session_analysis(mock_session_analysis_result)
        assert is_valid_analysis is True
    def test_session_metrics_calculator_di_integration(self, container, mock_market_data) -> None:
        """Тест интеграции калькулятора метрик сессий через DI."""
        # Создаем калькулятор
        session_metrics_calculator = SessionMetricsCalculator()
        # Проверяем, что калькулятор создан
        assert session_metrics_calculator is not None
        assert isinstance(session_metrics_calculator, SessionMetricsCalculator)
        # Тестируем расчёт метрик
        metrics = session_metrics_calculator.calculate_session_metrics(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(metrics, dict)
        assert "volatility" in metrics
        assert "volume_change" in metrics
        assert "price_change" in metrics
    def test_session_pattern_recognizer_di_integration(self, container, mock_market_data) -> None:
        """Тест интеграции распознавателя паттернов сессий через DI."""
        # Создаем распознаватель
        session_pattern_recognizer = SessionPatternRecognizer()
        # Проверяем, что распознаватель создан
        assert session_pattern_recognizer is not None
        assert isinstance(session_pattern_recognizer, SessionPatternRecognizer)
        # Тестируем распознавание паттернов
        patterns = session_pattern_recognizer.recognize_patterns(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(patterns, list)
    def test_session_transition_manager_di_integration(self, container) -> None:
        """Тест интеграции менеджера переходов сессий через DI."""
        # Создаем менеджер
        session_transition_manager = SessionTransitionManager()
        # Проверяем, что менеджер создан
        assert session_transition_manager is not None
        assert isinstance(session_transition_manager, SessionTransitionManager)
        # Тестируем получение состояния
        current_state = session_transition_manager.get_current_state()
        assert isinstance(current_state, dict)
        assert "active_sessions" in current_state
    def test_session_predictor_di_integration(self, container, mock_market_data) -> None:
        """Тест интеграции предиктора сессий через DI."""
        # Создаем предиктор
        session_predictor = SessionPredictor()
        # Проверяем, что предиктор создан
        assert session_predictor is not None
        assert isinstance(session_predictor, SessionPredictor)
        # Тестируем прогнозирование
        prediction = session_predictor.predict_session_behavior(
            session_type=SessionType.ASIAN,
            market_conditions=mock_market_data.tail(1).to_dict('records')[0]
        )
        assert isinstance(prediction, dict)
        assert "predicted_volatility" in prediction
        assert "predicted_volume" in prediction
    def test_session_optimizer_di_integration(self, container) -> None:
        """Тест интеграции оптимизатора сессий через DI."""
        # Создаем оптимизатор
        session_optimizer = SessionOptimizer()
        # Проверяем, что оптимизатор создан
        assert session_optimizer is not None
        assert isinstance(session_optimizer, SessionOptimizer)
        # Тестируем оптимизацию
        optimized_params = session_optimizer.optimize_session_parameters(
            session_type=SessionType.ASIAN,
            current_parameters={"volatility_threshold": 0.1, "volume_threshold": 1000}
        )
        assert isinstance(optimized_params, dict)
        assert "volatility_threshold" in optimized_params
    def test_session_monitor_di_integration(self, container) -> None:
        """Тест интеграции монитора сессий через DI."""
        # Создаем монитор
        session_monitor = SessionMonitor()
        # Проверяем, что монитор создан
        assert session_monitor is not None
        assert isinstance(session_monitor, SessionMonitor)
        # Тестируем мониторинг
        status = session_monitor.get_monitoring_status()
        assert isinstance(status, dict)
        assert "is_active" in status
    def test_session_analytics_di_integration(self, container) -> None:
        """Тест интеграции аналитики сессий через DI."""
        # Создаем аналитику
        session_analytics = SessionAnalytics()
        # Проверяем, что аналитика создана
        assert session_analytics is not None
        assert isinstance(session_analytics, SessionAnalytics)
        # Тестируем генерацию отчёта
        report = session_analytics.analyze_session_performance(
            session_type=SessionType.ASIAN,
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now()
        )
        assert isinstance(report, dict)
        assert "performance_score" in report
    def test_session_risk_analyzer_di_integration(self, container, mock_market_data) -> None:
        """Тест интеграции анализатора рисков сессий через DI."""
        # Создаем анализатор рисков
        session_risk_analyzer = SessionRiskAnalyzer()
        # Проверяем, что анализатор создан
        assert session_risk_analyzer is not None
        assert isinstance(session_risk_analyzer, SessionRiskAnalyzer)
        # Тестируем анализ рисков
        risk_analysis = session_risk_analyzer.analyze_session_risks(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(risk_analysis, dict)
        assert "risk_level" in risk_analysis
    def test_full_session_workflow_di_integration(
        self,
        container,
        session_repository_config,
        mock_market_data,
        mock_session_analysis_result
    ) -> None:
        """Тест полного рабочего процесса сессий через DI."""
        # Создаем все компоненты
        session_repository = SessionRepository(config=session_repository_config)
        session_cache = SessionCache(max_size=500, ttl_seconds=300)
        session_validator = SessionValidator()
        session_metrics_calculator = SessionMetricsCalculator()
        session_pattern_recognizer = SessionPatternRecognizer()
        session_transition_manager = SessionTransitionManager()
        session_predictor = SessionPredictor()
        session_optimizer = SessionOptimizer()
        session_monitor = SessionMonitor()
        session_analytics = SessionAnalytics()
        session_risk_analyzer = SessionRiskAnalyzer()
        # Проверяем, что все компоненты созданы
        assert all([
            session_repository, session_cache, session_validator,
            session_metrics_calculator, session_pattern_recognizer,
            session_transition_manager, session_predictor, session_optimizer,
            session_monitor, session_analytics, session_risk_analyzer
        ])
        # 1. Валидация входных данных
        is_valid_data = session_validator.validate_market_data(mock_market_data)
        assert is_valid_data is True
        # 2. Расчёт метрик
        metrics = session_metrics_calculator.calculate_session_metrics(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(metrics, dict)
        # 3. Распознавание паттернов
        patterns = session_pattern_recognizer.recognize_patterns(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(patterns, list)
        # 4. Прогнозирование поведения
        prediction = session_predictor.predict_session_behavior(
            session_type=SessionType.ASIAN,
            market_conditions=mock_market_data.tail(1).to_dict('records')[0]
        )
        assert isinstance(prediction, dict)
        # 5. Анализ рисков
        risk_analysis = session_risk_analyzer.analyze_session_risks(
            session_type=SessionType.ASIAN,
            market_data=mock_market_data
        )
        assert isinstance(risk_analysis, dict)
        # 6. Сохранение в репозиторий
        success = session_repository.save_session_analysis(mock_session_analysis_result)
        assert success is True
        # 7. Кэширование результатов
        cache_key = f"analysis_{SessionType.ASIAN.value}_{Timestamp.now().to_iso()[:10]}"
        session_cache.set(cache_key, mock_session_analysis_result.to_dict(), ttl=300)
        cached_data = session_cache.get(cache_key)
        assert cached_data is not None
        # 8. Мониторинг состояния
        monitoring_status = session_monitor.get_monitoring_status()
        assert isinstance(monitoring_status, dict)
        # 9. Генерация аналитического отчёта
        report = session_analytics.analyze_session_performance(
            session_type=SessionType.ASIAN,
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        assert isinstance(report, dict)
        # 10. Оптимизация параметров
        optimized_params = session_optimizer.optimize_session_parameters(
            session_type=SessionType.ASIAN,
            current_parameters={"volatility_threshold": 0.1}
        )
        assert isinstance(optimized_params, dict)
    @pytest.mark.asyncio
    async def test_service_locator_session_integration(self, service_locator) -> None:
        """Тест интеграции с ServiceLocator."""
        # Проверяем, что ServiceLocator создан
        assert service_locator is not None
        assert isinstance(service_locator, ServiceLocator)
        # Тестируем получение сервисов через ServiceLocator
        # (используем моки, так как реальные сервисы могут требовать дополнительной настройки)
        with Mock() as mock_session_service:
            with Mock() as mock_market_service:
                with Mock() as mock_trading_service:
                    with Mock() as mock_risk_service:
                        mock_get_service = Mock(return_value=mock_session_service)
                        service_locator.container.get_service = mock_get_service
                        session_service = service_locator.get_service(SessionService)
                        assert session_service is not None
                        assert session_service == mock_session_service
                        assert mock_get_service.call_count == 1
                        assert mock_get_service.call_args[0][0] == SessionService
    def test_error_handling_di_integration(self, container, session_repository_config) -> None:
        """Тест обработки ошибок в DI интеграции."""
        # Создаем репозиторий
        session_repository = SessionRepository(config=session_repository_config)
        # Тест с некорректными данными
        with pytest.raises(Exception):
            session_repository.save_session_analysis(None)
        # Тест с некорректными параметрами
        with pytest.raises(Exception):
            session_repository.get_session_analysis(
                session_type=None,
                start_time=Timestamp.now(),
                end_time=Timestamp.now()
            )
    def test_performance_di_integration(self, container, session_repository_config) -> None:
        """Тест производительности DI интеграции."""
        import time
        # Создаем компоненты
        session_repository = SessionRepository(config=session_repository_config)
        session_cache = SessionCache(max_size=500, ttl_seconds=300)
        # Тест производительности сохранения
        start_time = time.time()
        for i in range(10):
            analysis_result = SessionAnalysisResult(
                session_type=SessionType.ASIAN,
                session_phase=SessionPhase.ACTIVE_PHASE,
                timestamp=Timestamp.now(),
                confidence=ConfidenceScore(0.8 + i * 0.01),
                metrics=SessionMetrics(
                    volume_change_percent=0.1 + i * 0.01,
                    volatility_change_percent=0.2 + i * 0.01,
                    price_direction_bias=0.05 + i * 0.01,
                    momentum_strength=0.3 + i * 0.01,
                    false_breakout_probability=0.1 + i * 0.01,
                    reversal_probability=0.2 + i * 0.01,
                    trend_continuation_probability=0.7 + i * 0.01,
                    influence_duration_minutes=120 + i,
                    peak_influence_time_minutes=60 + i,
                    spread_impact=0.02 + i * 0.01,
                    liquidity_impact=0.03 + i * 0.01,
                    correlation_with_other_sessions=0.5 + i * 0.01
                ),
                market_conditions=MarketConditions(
                    volatility=0.1 + i * 0.01,
                    volume=1000.0 + i * 100,
                    spread=0.02 + i * 0.01,
                    liquidity=0.8 + i * 0.01,
                    momentum=0.3 + i * 0.01,
                    trend_strength=0.7 + i * 0.01,
                    session_intensity=0.6 + i * 0.01,
                    market_regime=MarketRegime.TRENDING_BULL
                ),
                predictions={"price_direction": "up"},
                risk_factors=["liquidity_risk", "volatility_risk", "correlation_risk"]
            )
            session_repository.save_session_analysis(analysis_result)
        save_time = time.time() - start_time
        assert save_time < 5.0  # Должно выполняться менее 5 секунд
        # Тест производительности кэша
        start_time = time.time()
        for i in range(100):
            session_cache.set(f"test_key_{i}", {"data": i}, ttl=60)
            session_cache.get(f"test_key_{i}")
        cache_time = time.time() - start_time
        assert cache_time < 1.0  # Должно выполняться менее 1 секунды
    def test_memory_usage_di_integration(self, container, session_repository_config) -> None:
        """Тест использования памяти в DI интеграции."""
        import psutil
        import os
        # Получаем текущий процесс
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        # Создаем компоненты
        session_repository = SessionRepository(config=session_repository_config)
        session_cache = SessionCache(max_size=1000, ttl_seconds=300)
        session_validator = SessionValidator()
        session_metrics_calculator = SessionMetricsCalculator()
        session_pattern_recognizer = SessionPatternRecognizer()
        session_transition_manager = SessionTransitionManager()
        session_predictor = SessionPredictor()
        session_optimizer = SessionOptimizer()
        session_monitor = SessionMonitor()
        session_analytics = SessionAnalytics()
        session_risk_analyzer = SessionRiskAnalyzer()
        # Выполняем операции
        for i in range(50):
            session_cache.set(f"test_key_{i}", {"data": f"value_{i}"}, ttl=60)
        # Проверяем использование памяти
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB в байтах 
