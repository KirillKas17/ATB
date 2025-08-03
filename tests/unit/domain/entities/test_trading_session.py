"""
Unit тесты для TradingSession entity.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from domain.entities.trading_session import TradingSession
from domain.exceptions import ValidationError


class TestTradingSession:
    """Тесты для TradingSession entity."""
    
    @pytest.fixture
    def sample_session_data(self) -> Dict[str, Any]:
        """Тестовые данные для торговой сессии."""
        return {
            "session_id": "session_001",
            "account_id": "account_001",
            "strategy_id": "strategy_001",
            "symbol": "BTC/USDT",
            "start_time": "2024-01-01T09:00:00Z",
            "end_time": "2024-01-01T17:00:00Z",
            "status": "active",
            "initial_balance": "10000.00",
            "current_balance": "10500.00",
            "total_pnl": "500.00",
            "total_fees": "25.00"
        }
    
    def test_session_creation(self, sample_session_data: Dict[str, Any]):
        """Тест создания торговой сессии."""
        session = TradingSession(**sample_session_data)
        
        assert session.session_id == "session_001"
        assert session.account_id == "account_001"
        assert session.strategy_id == "strategy_001"
        assert session.symbol == "BTC/USDT"
        assert session.status == "active"
        assert session.initial_balance == Decimal("10000.00")
        assert session.current_balance == Decimal("10500.00")
        assert session.total_pnl == Decimal("500.00")
        assert session.total_fees == Decimal("25.00")
    
    def test_session_validation_empty_session_id(self):
        """Тест валидации пустого session_id."""
        with pytest.raises(ValidationError):
            TradingSession(
                session_id="",
                account_id="account_001",
                strategy_id="strategy_001",
                symbol="BTC/USDT",
                start_time="2024-01-01T09:00:00Z",
                end_time="2024-01-01T17:00:00Z",
                status="active",
                initial_balance="10000.00",
                current_balance="10500.00",
                total_pnl="500.00",
                total_fees="25.00"
            )
    
    def test_session_validation_invalid_status(self):
        """Тест валидации неверного статуса."""
        with pytest.raises(ValidationError):
            TradingSession(
                session_id="session_001",
                account_id="account_001",
                strategy_id="strategy_001",
                symbol="BTC/USDT",
                start_time="2024-01-01T09:00:00Z",
                end_time="2024-01-01T17:00:00Z",
                status="invalid_status",
                initial_balance="10000.00",
                current_balance="10500.00",
                total_pnl="500.00",
                total_fees="25.00"
            )
    
    def test_session_equality(self, sample_session_data: Dict[str, Any]):
        """Тест равенства торговых сессий."""
        session1 = TradingSession(**sample_session_data)
        session2 = TradingSession(**sample_session_data)
        
        assert session1 == session2
    
    def test_session_is_active(self, sample_session_data: Dict[str, Any]):
        """Тест проверки активности сессии."""
        session = TradingSession(**sample_session_data)
        assert session.is_active() is True
        
        completed_session_data = sample_session_data.copy()
        completed_session_data["status"] = "completed"
        completed_session = TradingSession(**completed_session_data)
        assert completed_session.is_active() is False
    
    def test_session_calculate_return_percentage(self, sample_session_data: Dict[str, Any]):
        """Тест расчета процентной доходности."""
        session = TradingSession(**sample_session_data)
        return_pct = session.calculate_return_percentage()
        
        # (10500 - 10000) / 10000 * 100 = 5%
        expected_return = Decimal("5.0")
        assert return_pct == expected_return
    
    def test_session_calculate_net_pnl(self, sample_session_data: Dict[str, Any]):
        """Тест расчета чистого P&L."""
        session = TradingSession(**sample_session_data)
        net_pnl = session.calculate_net_pnl()
        
        # 500 - 25 = 475
        expected_net_pnl = Decimal("475.00")
        assert net_pnl == expected_net_pnl
    
    def test_session_complete(self, sample_session_data: Dict[str, Any]):
        """Тест завершения сессии."""
        session = TradingSession(**sample_session_data)
        assert session.status == "active"
        
        session.complete()
        assert session.status == "completed"
        assert session.end_time is not None 