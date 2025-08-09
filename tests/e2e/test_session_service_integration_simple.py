"""
Упрощенный e2e-тест интеграции SessionService без ML-зависимостей.
"""

import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.sessions.factories import get_session_service
from domain.type_definitions.session_types import SessionType, MarketRegime, SessionIntensity


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_session_service_integration_simple() -> None:
    """
    E2E: SessionService работает корректно в изолированном режиме.
    """
    # Получаем сервис через фабрику
    session_service = get_session_service()
    # Тест 1: Получение контекста сессии
    context = session_service.get_current_session_context()
    assert isinstance(context, dict)
    assert "active_sessions" in context
    assert "primary_session" in context
    # Тест 2: Анализ влияния сессии (с пустыми данными)
    empty_df = pd.DataFrame()
    result = session_service.analyze_session_influence("BTCUSDT", empty_df)
    assert result is None or hasattr(result, "influence_score") or isinstance(result, dict)
    # Тест 3: Прогноз поведения сессии
    market_conditions = {
        "volatility": 1.0,
        "volume": 1000.0,
        "spread": 10.0,
        "liquidity": 100000.0,
        "momentum": 0.5,
        "trend_strength": 0.7,
        "market_regime": MarketRegime.TRENDING_BULL,
        "session_intensity": SessionIntensity.NORMAL,
    }
    prediction = session_service.predict_session_behavior(SessionType.ASIAN, market_conditions)
    assert isinstance(prediction, dict)
    assert "predicted_volatility" in prediction
    # Тест 4: Получение рекомендаций
    recommendations = session_service.get_session_recommendations("BTCUSDT", SessionType.ASIAN)
    assert isinstance(recommendations, list)
    # Тест 5: Получение статистики
    statistics = session_service.get_session_statistics(SessionType.ASIAN)
    assert isinstance(statistics, dict)
    # Тест 6: Проверка переходных периодов
    is_transition = session_service.is_transition_period()
    assert isinstance(is_transition, bool)
    # Тест 7: Получение активных переходов
    transitions = session_service.get_active_transitions()
    assert isinstance(transitions, list)
    # Тест 8: Получение фазы сессии
    phase = session_service.get_session_phase(SessionType.ASIAN)
    assert phase is None or isinstance(phase, str)
    # Тест 9: Получение следующего изменения сессии
    next_change = session_service.get_next_session_change()
    assert next_change is None or isinstance(next_change, dict)
    # Тест 10: Проверка здоровья сервиса
    health = session_service.get_session_health_check()
    assert isinstance(health, dict)
    assert "status" in health
    assert "components" in health
