import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.di_container_refactored import get_service_locator
from domain.services.market_metrics import MarketMetricsService
from domain.services.spread_analyzer import SpreadAnalyzer
from domain.services.liquidity_analyzer import LiquidityAnalyzer
from domain.services.ml_predictor import MLPredictor
from domain.services.technical_analysis import DefaultTechnicalAnalysisService
from domain.services.correlation_chain import CorrelationChain
from domain.services.signal_service import DefaultSignalService
from domain.services.strategy_service import StrategyService
from domain.services.pattern_discovery import PatternDiscovery
from domain.services.risk_analysis import DefaultRiskAnalysisService


@pytest.mark.asyncio
async def test_domain_services_integration() -> None:
    locator = get_service_locator()
    # Проверяем, что все сервисы доступны через DI
    assert isinstance(locator.get_service(MarketMetricsService), MarketMetricsService)
    assert isinstance(locator.get_service(SpreadAnalyzer), SpreadAnalyzer)
    assert isinstance(locator.get_service(LiquidityAnalyzer), LiquidityAnalyzer)
    assert isinstance(locator.get_service(MLPredictor), MLPredictor)
    assert isinstance(locator.get_service(DefaultTechnicalAnalysisService), DefaultTechnicalAnalysisService)
    assert isinstance(locator.get_service(CorrelationChain), CorrelationChain)
    assert isinstance(locator.get_service(DefaultSignalService), DefaultSignalService)
    assert isinstance(locator.get_service(DefaultStrategyService), DefaultStrategyService)
    assert isinstance(locator.get_service(PatternDiscovery), PatternDiscovery)
    assert isinstance(locator.get_service(DefaultRiskAnalysisService), DefaultRiskAnalysisService)

    # Пример вызова метода одного из сервисов (MarketMetricsService)
    service = locator.get_service(MarketMetricsService)
    dummy_data = [{"timestamp": 1, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 100} for _ in range(10)]
    result = service.calculate_metrics(dummy_data)
    assert "volatility" in result
    assert "trend" in result

    # Проверка обработки ошибок
    with pytest.raises(Exception):
        service.calculate_metrics([])  # Пустой вход должен вызвать ошибку или вернуть пустой результат
