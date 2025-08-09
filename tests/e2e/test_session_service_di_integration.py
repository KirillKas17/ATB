"""
E2E тест интеграции SessionService через DI-контейнер.
"""

import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.di_container_refactored import get_service_locator
from domain.sessions.services import SessionService
from domain.type_definitions.session_types import SessionType, MarketRegime, SessionIntensity


@pytest.mark.e2e
async def test_session_service_di_integration() -> None:
    """
    E2E: SessionService корректно работает через DI-контейнер.
    """
    # Получаем сервис через DI-контейнер
    locator = get_service_locator()
    session_service = locator.get_service(SessionService)
    # Проверяем, что получили правильный тип
    assert isinstance(session_service, SessionService)
    # Тест 1: Получение контекста сессии через DI
    context = session_service.get_current_session_context()
    assert isinstance(context, dict)
    assert "active_sessions" in context
    assert "primary_session" in context
    # Тест 2: Анализ влияния сессии через DI
    empty_df = pd.DataFrame()
    result = session_service.analyze_session_influence("BTCUSDT", empty_df)
    assert result is None or hasattr(result, "influence_score") or isinstance(result, dict)
    # Тест 3: Прогноз поведения сессии через DI
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
    # Тест 4: Получение рекомендаций через DI
    recommendations = session_service.get_session_recommendations("BTCUSDT", SessionType.ASIAN)
    assert isinstance(recommendations, list)
    # Тест 5: Получение статистики через DI
    statistics = session_service.get_session_statistics(SessionType.ASIAN)
    assert isinstance(statistics, dict)
    # Тест 6: Проверка здоровья сервиса через DI
    health = session_service.get_session_health_check()
    assert isinstance(health, dict)
    assert "status" in health
    assert "components" in health
    # Тест 7: Проверка, что все компоненты работают
    assert session_service.registry is not None
    assert session_service.session_marker is not None
    assert session_service.influence_analyzer is not None
    assert session_service.transition_manager is not None
    assert session_service.cache is not None
    assert session_service.validator is not None
