"""
Простой тест интеграции session_service.
"""
from unittest.mock import Mock
from domain.sessions.services import SessionService
from domain.sessions.factories import get_session_service
class TestSessionServiceIntegration:
    """Тесты интеграции SessionService."""
    def test_factory_and_basic_methods(self) -> None:
        service = get_session_service()
        assert isinstance(service, SessionService)
        # Проверяем основные публичные методы
        for method in [
            'get_current_session_context',
            'analyze_session_influence',
            'predict_session_behavior',
            'get_session_recommendations',
            'get_session_statistics',
            'is_transition_period',
            'get_active_transitions',
            'update_session_profile',
            'get_session_overlap',
            'get_session_phase',
            'get_next_session_change',
            'clear_cache',
            'get_session_health_check',
        ]:
            assert hasattr(service, method)
    def test_get_current_session_context(self) -> None:
        service = get_session_service()
        context = service.get_current_session_context()
        assert isinstance(context, dict)
        assert 'active_sessions' in context
        assert 'primary_session' in context
        # Проверяем, что у primary_session есть фаза
        if context['primary_session']:
            assert 'phase' in context['primary_session']
    def test_analyze_session_influence(self) -> None:
        service = get_session_service()
        # market_data должен быть pd.DataFrame, но для smoke-теста можно передать None
        result = service.analyze_session_influence('BTCUSDT', None)
        assert result is None or hasattr(result, 'influence_score') or isinstance(result, dict)
    def test_predict_session_behavior(self) -> None:
        service = get_session_service()
        # Используем SessionType и MarketConditions из domain.types.session_types
        from domain.type_definitions.session_types import SessionType
        market_conditions = {'volatility': 1.0, 'volume': 1000.0, 'market_regime': Mock(value='bull')}
        prediction = service.predict_session_behavior(SessionType.ASIAN, market_conditions)
        assert isinstance(prediction, dict)
        assert 'predicted_volatility' in prediction
    def test_get_session_recommendations(self) -> None:
        service = get_session_service()
        from domain.type_definitions.session_types import SessionType
        recs = service.get_session_recommendations('BTCUSDT', SessionType.ASIAN)
        assert isinstance(recs, list)
    def test_get_session_statistics(self) -> None:
        service = get_session_service()
        from domain.type_definitions.session_types import SessionType
        stats = service.get_session_statistics(SessionType.ASIAN)
        assert isinstance(stats, dict)
    def test_is_transition_period(self) -> None:
        service = get_session_service()
        result = service.is_transition_period()
        assert isinstance(result, bool)
    def test_get_active_transitions(self) -> None:
        service = get_session_service()
        transitions = service.get_active_transitions()
        assert isinstance(transitions, list)
    def test_update_session_profile(self) -> None:
        service = get_session_service()
        from domain.type_definitions.session_types import SessionType
        ok = service.update_session_profile(SessionType.ASIAN, {'foo': 'bar'})
        assert isinstance(ok, bool)
    def test_get_session_overlap(self) -> None:
        service = get_session_service()
        from domain.type_definitions.session_types import SessionType
        overlap = service.get_session_overlap(SessionType.ASIAN, SessionType.LONDON)
        assert isinstance(overlap, float)
    def test_get_session_phase(self) -> None:
        service = get_session_service()
        from domain.type_definitions.session_types import SessionType
        phase = service.get_session_phase(SessionType.ASIAN)
        assert phase is None or isinstance(phase, str)
    def test_get_next_session_change(self) -> None:
        service = get_session_service()
        next_change = service.get_next_session_change()
        assert next_change is None or isinstance(next_change, dict)
    def test_clear_cache(self) -> None:
        service = get_session_service()
        service.clear_cache()  # smoke-test, не должно падать
    def test_get_session_health_check(self) -> None:
        service = get_session_service()
        health = service.get_session_health_check()
        assert isinstance(health, dict)
        assert 'status' in health 
