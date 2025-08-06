"""
Тесты для technical analysis сервиса.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.external_services.technical_analysis_service import TechnicalAnalysisServiceAdapter
class TestTechnicalAnalysisService:
    """Тесты для TechnicalAnalysisServiceAdapter."""
    @pytest.fixture
    def technical_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра сервиса."""
        return TechnicalAnalysisServiceAdapter()
        # Проверяем логику полос Боллинджера
        # Проверяем наличие основных индикаторов
        # RSI с разными периодами должны давать разные результаты
        # RSI с пустыми данными
        # MACD с пустыми данными
        # RSI с одним значением
        # MACD с одним значением
        # RSI с неверным периодом
        # MACD с неверными периодами
