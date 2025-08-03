from typing import Any
import pytest
from application.di_container_refactored import get_service_locator
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from domain.sessions.services import SessionService
from domain.types.session_types import SessionType, MarketRegime, SessionIntensity
import pandas as pd


class TestOrchestratorSessionIntegration:
    def __init__(self) -> None:
        """Инициализация тестового класса."""
        self.test_data: dict[str, Any] = {}
        self.results: dict[str, Any] = {}

    def setup_method(self) -> None:
        """Настройка тестового метода."""
        # Убираем переопределение переменных
        pass


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_orchestrator_session_service_integration() -> None:
    """
    E2E: orchestrator использует SessionService для получения контекста и анализа влияния сессии.
    """
    locator = get_service_locator()
    orchestrator: DefaultTradingOrchestratorUseCase = locator.get_use_case(DefaultTradingOrchestratorUseCase)
    session_service: SessionService = locator.get_service(SessionService)
    # Получаем контекст сессии через orchestrator
    context = orchestrator.session_service.get_current_session_context()
    assert isinstance(context, dict)
    assert 'active_sessions' in context
    assert 'primary_session' in context
    # Анализ влияния сессии (передаем пустой DataFrame)
    empty_df = pd.DataFrame()
    result = orchestrator.session_service.analyze_session_influence('BTCUSDT', empty_df)
    assert result is None or hasattr(result, 'influence_score') or isinstance(result, dict)
    # Прогноз поведения сессии
    market_conditions = {
        'volatility': 1.0,
        'volume': 1000.0,
        'spread': 10.0,
        'liquidity': 100000.0,
        'momentum': 0.5,
        'trend_strength': 0.7,
        'market_regime': MarketRegime.TRENDING_BULL,
        'session_intensity': SessionIntensity.NORMAL
    }
    prediction = orchestrator.session_service.predict_session_behavior(SessionType.ASIAN, market_conditions)
    assert isinstance(prediction, dict)
    assert 'predicted_volatility' in prediction 
