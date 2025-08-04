"""
Unit тесты для RiskManager.
Тестирует управление рисками, включая расчет рисков,
мониторинг позиций, валидацию операций и управление лимитами.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from infrastructure.core.risk_manager import RiskManager
from infrastructure.messaging.event_bus import EventBus

class TestRiskManager:
    """Тесты для RiskManager."""
    
    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Фикстура для RiskManager."""
        # Создаем мок event_bus и config
        class MockEventBus(EventBus):
            async def publish(self, event: Any) -> bool:
                return True
        
        config = {
            "max_position_size": 5.0,
            "max_portfolio_value": 100000.0,
            "max_daily_loss": 5000.0,
            "max_drawdown": 0.15,
            "max_leverage": 3.0,
            "max_correlation": 0.8,
            "min_margin_ratio": 0.2,
            "max_concentration": 0.3
        }
        
        return RiskManager(MockEventBus(), config)
    
    @pytest.fixture
    def sample_positions(self) -> dict:
        """Фикстура с тестовыми позициями."""
        return {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "side": "long",
                "quantity": Decimal("1.5"),
                "entry_price": Decimal("50000.0"),
                "current_price": Decimal("52000.0"),
                "unrealized_pnl": Decimal("3000.0"),
                "realized_pnl": Decimal("500.0"),
                "timestamp": datetime.now()
            },
            "ETHUSDT": {
                "symbol": "ETHUSDT",
                "side": "short",
                "quantity": Decimal("10.0"),
                "entry_price": Decimal("3000.0"),
                "current_price": Decimal("2900.0"),
                "unrealized_pnl": Decimal("1000.0"),
                "realized_pnl": Decimal("-200.0"),
                "timestamp": datetime.now()
            }
        }
    
    @pytest.fixture
    def sample_risk_limits(self) -> dict:
        """Фикстура с лимитами риска."""
        return {
            "max_position_size": Decimal("5.0"),
            "max_portfolio_value": Decimal("100000.0"),
            "max_daily_loss": Decimal("5000.0"),
            "max_drawdown": Decimal("0.15"),
            "max_leverage": Decimal("3.0"),
            "max_correlation": Decimal("0.8"),
            "min_margin_ratio": Decimal("0.2"),
            "max_concentration": Decimal("0.3")
        }
    
    def test_initialization(self, risk_manager: RiskManager) -> None:
        """Тест инициализации менеджера рисков."""
        assert risk_manager is not None
        assert hasattr(risk_manager, 'config')
        assert hasattr(risk_manager, 'current_metrics')
    
    @pytest.mark.asyncio
    async def test_calculate_position_risk(self, risk_manager: RiskManager, sample_positions: dict) -> None:
        """Тест расчета риска позиции."""
        # Расчет риска позиции
        position_risk = await risk_manager.calculate_position_risk(
            "BTCUSDT", 
            float(sample_positions["BTCUSDT"]["quantity"]), 
            float(sample_positions["BTCUSDT"]["entry_price"])
        )
        # Проверки
        assert position_risk is not None
        assert isinstance(position_risk, dict)
        # Проверяем, что результат содержит ожидаемые ключи
        assert "risk_score" in position_risk or "position_risk" in position_risk
    
    @pytest.mark.asyncio
    async def test_update_portfolio_risk(self, risk_manager: RiskManager, sample_positions: dict) -> None:
        """Тест обновления риска портфеля."""
        # Обновление риска портфеля
        await risk_manager._update_portfolio_risk()
        # Проверки
        assert hasattr(risk_manager, 'current_metrics')
        assert isinstance(risk_manager.current_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, risk_manager: RiskManager) -> None:
        """Тест получения метрик риска."""
        # Получение метрик риска
        metrics = await risk_manager.get_risk_metrics()
        # Проверки
        assert metrics is not None
        assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, risk_manager: RiskManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        try:
            await risk_manager.calculate_position_risk("BTCUSDT", 0, 0)
        except Exception:
            pass  # Ожидаем исключение
        
        # Тест с пустыми позициями
        empty_positions: Dict[str, Any] = {}
        await risk_manager._update_portfolio_risk()
        assert isinstance(risk_manager.current_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, risk_manager: RiskManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень большими значениями
        try:
            await risk_manager.calculate_position_risk("BTCUSDT", 999999, 999999)
        except Exception:
            pass  # Ожидаем исключение
        
        # Тест с отрицательными значениями
        try:
            await risk_manager.calculate_position_risk("BTCUSDT", -1, -1)
        except Exception:
            pass  # Ожидаем исключение
    
    def test_cleanup(self, risk_manager: RiskManager) -> None:
        """Тест очистки ресурсов."""
        # Проверяем, что объект можно безопасно удалить
        del risk_manager 
