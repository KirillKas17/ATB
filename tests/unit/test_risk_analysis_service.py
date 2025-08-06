"""
Тесты для risk analysis сервиса.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from unittest.mock import Mock, patch
from infrastructure.external_services.risk_analysis_service import RiskAnalysisServiceAdapter
class TestRiskAnalysisService:
    """Тесты для RiskAnalysisServiceAdapter."""
    @pytest.fixture
    def risk_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра сервиса."""
        return RiskAnalysisServiceAdapter()
        # Проверяем, что значения в разумных пределах
        # Коэффициент Шарпа может быть отрицательным
        # Коэффициент Сортино может быть отрицательным
        # Создаем цены с просадкой
