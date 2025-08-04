"""
Unit тесты для Portfolio.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.portfolio import Portfolio, PortfolioStatus, RiskProfile, PortfolioProtocol
from domain.type_definitions import PortfolioId
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.timestamp import Timestamp
from domain.exceptions.base_exceptions import ValidationError


class TestPortfolio:
    """Тесты для Portfolio."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": PortfolioId(uuid4()),
            "name": "Test Portfolio",
            "status": PortfolioStatus.ACTIVE,
            "total_equity": Money(Decimal("10000"), Currency.USD),
            "free_margin": Money(Decimal("8000"), Currency.USD),
            "used_margin": Money(Decimal("2000"), Currency.USD),
            "risk_profile": RiskProfile.MODERATE,
            "max_leverage": Decimal("10"),
            "metadata": {"strategy": "conservative"}
        }
    
    @pytest.fixture
    def portfolio(self, sample_data) -> Portfolio:
        """Создает тестовый портфель."""
        return Portfolio(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания портфеля."""
        portfolio = Portfolio(**sample_data)
        
        assert portfolio.id == sample_data["id"]
        assert portfolio.name == sample_data["name"]
        assert portfolio.status == sample_data["status"]
        assert portfolio.total_equity == sample_data["total_equity"]
        assert portfolio.free_margin == sample_data["free_margin"]
        assert portfolio.used_margin == sample_data["used_margin"]
        assert portfolio.risk_profile == sample_data["risk_profile"]
        assert portfolio.max_leverage == sample_data["max_leverage"]
        assert portfolio.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания портфеля с дефолтными значениями."""
        portfolio = Portfolio()
        
        assert isinstance(portfolio.id, PortfolioId)
        assert portfolio.name == ""
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert portfolio.total_equity.amount == Decimal("0")
        assert portfolio.free_margin.amount == Decimal("0")
        assert portfolio.used_margin.amount == Decimal("0")
        assert portfolio.risk_profile == RiskProfile.MODERATE
        assert portfolio.max_leverage == Decimal("10")
        assert portfolio.metadata == {}
    
    def test_get_equity(self, portfolio):
        """Тест получения equity."""
        equity = portfolio.get_equity()
        
        assert equity.value == Decimal("10000")
        assert isinstance(equity.value, Decimal)
    
    def test_get_margin_ratio(self, portfolio):
        """Тест расчета margin ratio."""
        margin_ratio = portfolio.get_margin_ratio()
        
        expected_ratio = (Decimal("2000") / Decimal("10000")) * 100
        assert margin_ratio == expected_ratio
        assert margin_ratio == Decimal("20")
    
    def test_get_margin_ratio_zero_equity(self):
        """Тест margin ratio при нулевом equity."""
        portfolio = Portfolio(
            total_equity=Money(Decimal("0"), Currency.USD),
            used_margin=Money(Decimal("1000"), Currency.USD)
        )
        
        margin_ratio = portfolio.get_margin_ratio()
        assert margin_ratio == Decimal("0")
    
    def test_balance_property(self, portfolio):
        """Тест свойства balance."""
        assert portfolio.balance == portfolio.total_equity
        assert portfolio.balance.amount == Decimal("10000")
    
    def test_available_balance_property(self, portfolio):
        """Тест свойства available_balance."""
        assert portfolio.available_balance == portfolio.free_margin
        assert portfolio.available_balance.amount == Decimal("8000")
    
    def test_total_balance_property(self, portfolio):
        """Тест свойства total_balance."""
        assert portfolio.total_balance == portfolio.total_equity
        assert portfolio.total_balance.amount == Decimal("10000")
    
    def test_locked_balance_property(self, portfolio):
        """Тест свойства locked_balance."""
        assert portfolio.locked_balance == portfolio.used_margin
        assert portfolio.locked_balance.amount == Decimal("2000")
    
    def test_unrealized_pnl_property(self, portfolio):
        """Тест свойства unrealized_pnl."""
        pnl = portfolio.unrealized_pnl
        assert pnl.amount == Decimal("0")
        assert pnl.currency == Currency.USD
    
    def test_realized_pnl_property(self, portfolio):
        """Тест свойства realized_pnl."""
        pnl = portfolio.realized_pnl
        assert pnl.amount == Decimal("0")
        assert pnl.currency == Currency.USD
    
    def test_total_pnl_property(self, portfolio):
        """Тест свойства total_pnl."""
        pnl = portfolio.total_pnl
        assert pnl.amount == Decimal("0")
        assert pnl.currency == Currency.USD
    
    def test_margin_balance_property(self, portfolio):
        """Тест свойства margin_balance."""
        assert portfolio.margin_balance == portfolio.total_equity
        assert portfolio.margin_balance.amount == Decimal("10000")
    
    def test_risk_level_property(self, portfolio):
        """Тест свойства risk_level."""
        assert portfolio.risk_level == "moderate"
        
        portfolio.risk_profile = RiskProfile.AGGRESSIVE
        assert portfolio.risk_level == "aggressive"
        
        portfolio.risk_profile = RiskProfile.CONSERVATIVE
        assert portfolio.risk_level == "conservative"
    
    def test_leverage_property(self, portfolio):
        """Тест свойства leverage."""
        assert portfolio.leverage == Decimal("10")
        
        portfolio.max_leverage = Decimal("20")
        assert portfolio.leverage == Decimal("20")
    
    def test_is_active_property(self, portfolio):
        """Тест свойства is_active."""
        assert portfolio.is_active is True
        
        portfolio.status = PortfolioStatus.SUSPENDED
        assert portfolio.is_active is False
        
        portfolio.status = PortfolioStatus.CLOSED
        assert portfolio.is_active is False
    
    def test_is_suspended_property(self, portfolio):
        """Тест свойства is_suspended."""
        assert portfolio.is_suspended is False
        
        portfolio.status = PortfolioStatus.SUSPENDED
        assert portfolio.is_suspended is True
        
        portfolio.status = PortfolioStatus.CLOSED
        assert portfolio.is_suspended is False
    
    def test_is_closed_property(self, portfolio):
        """Тест свойства is_closed."""
        assert portfolio.is_closed is False
        
        portfolio.status = PortfolioStatus.CLOSED
        assert portfolio.is_closed is True
        
        portfolio.status = PortfolioStatus.SUSPENDED
        assert portfolio.is_closed is False
    
    def test_available_margin_property(self, portfolio):
        """Тест свойства available_margin."""
        available_margin = portfolio.available_margin
        assert available_margin.amount == Decimal("8000")
        assert available_margin.currency == Currency.USD
        assert available_margin == portfolio.free_margin
    
    def test_update_equity(self, portfolio):
        """Тест обновления equity."""
        old_updated_at = portfolio.updated_at
        new_equity = Money(Decimal("15000"), Currency.USD)
        
        portfolio.update_equity(new_equity)
        
        assert portfolio.total_equity == new_equity
        assert portfolio.updated_at > old_updated_at
    
    def test_update_margin(self, portfolio):
        """Тест обновления margin."""
        old_updated_at = portfolio.updated_at
        new_free_margin = Money(Decimal("9000"), Currency.USD)
        new_used_margin = Money(Decimal("1000"), Currency.USD)
        
        portfolio.update_margin(new_free_margin, new_used_margin)
        
        assert portfolio.free_margin == new_free_margin
        assert portfolio.used_margin == new_used_margin
        assert portfolio.updated_at > old_updated_at
    
    def test_suspend(self, portfolio):
        """Тест приостановки портфеля."""
        old_updated_at = portfolio.updated_at
        
        portfolio.suspend()
        
        assert portfolio.status == PortfolioStatus.SUSPENDED
        assert portfolio.updated_at > old_updated_at
    
    def test_activate(self, portfolio):
        """Тест активации портфеля."""
        portfolio.status = PortfolioStatus.SUSPENDED
        old_updated_at = portfolio.updated_at
        
        portfolio.activate()
        
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert portfolio.updated_at > old_updated_at
    
    def test_close(self, portfolio):
        """Тест закрытия портфеля."""
        old_updated_at = portfolio.updated_at
        
        portfolio.close()
        
        assert portfolio.status == PortfolioStatus.CLOSED
        assert portfolio.updated_at > old_updated_at
    
    def test_to_dict(self, portfolio):
        """Тест сериализации в словарь."""
        data = portfolio.to_dict()
        
        assert data["id"] == str(portfolio.id)
        assert data["name"] == portfolio.name
        assert data["status"] == portfolio.status.value
        assert data["total_equity"] == str(portfolio.total_equity.amount)
        assert data["free_margin"] == str(portfolio.free_margin.amount)
        assert data["used_margin"] == str(portfolio.used_margin.amount)
        assert data["risk_profile"] == portfolio.risk_profile.value
        assert data["max_leverage"] == str(portfolio.max_leverage)
        assert data["created_at"] == str(portfolio.created_at)
        assert data["updated_at"] == str(portfolio.updated_at)
        assert data["metadata"] == str(portfolio.metadata)
    
    def test_from_dict(self, portfolio):
        """Тест десериализации из словаря."""
        data = portfolio.to_dict()
        new_portfolio = Portfolio.from_dict(data)
        
        assert new_portfolio.id == portfolio.id
        assert new_portfolio.name == portfolio.name
        assert new_portfolio.status == portfolio.status
        assert new_portfolio.total_equity.amount == portfolio.total_equity.amount
        assert new_portfolio.free_margin.amount == portfolio.free_margin.amount
        assert new_portfolio.used_margin.amount == portfolio.used_margin.amount
        assert new_portfolio.risk_profile == portfolio.risk_profile
        assert new_portfolio.max_leverage == portfolio.max_leverage
        assert new_portfolio.metadata == portfolio.metadata
    
    def test_from_dict_without_id(self):
        """Тест десериализации без ID."""
        data = {
            "name": "Test Portfolio",
            "status": PortfolioStatus.ACTIVE.value,
            "total_equity": "10000",
            "free_margin": "8000",
            "used_margin": "2000",
            "risk_profile": RiskProfile.MODERATE.value,
            "max_leverage": "10",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "metadata": "{}"
        }
        
        portfolio = Portfolio.from_dict(data)
        
        assert isinstance(portfolio.id, PortfolioId)
        assert portfolio.name == "Test Portfolio"
        assert portfolio.status == PortfolioStatus.ACTIVE
    
    def test_str_representation(self, portfolio):
        """Тест строкового представления."""
        str_repr = str(portfolio)
        expected = f"Portfolio({portfolio.name}, equity={portfolio.total_equity})"
        
        assert str_repr == expected
    
    def test_repr_representation(self, portfolio):
        """Тест repr представления."""
        repr_str = repr(portfolio)
        
        assert "Portfolio(id=" in repr_str
        assert f"name='{portfolio.name}'" in repr_str
        assert f"status={portfolio.status.value}" in repr_str
        assert f"equity={portfolio.total_equity}" in repr_str
    
    def test_portfolio_protocol_compliance(self, portfolio):
        """Тест соответствия протоколу PortfolioProtocol."""
        assert isinstance(portfolio, PortfolioProtocol)
        
        equity = portfolio.get_equity()
        margin_ratio = portfolio.get_margin_ratio()
        is_active = portfolio.is_active
        
        assert isinstance(equity.value, Decimal)
        assert isinstance(margin_ratio, Decimal)
        assert isinstance(is_active, bool)
    
    def test_portfolio_status_enum(self):
        """Тест enum PortfolioStatus."""
        assert PortfolioStatus.ACTIVE.value == "active"
        assert PortfolioStatus.SUSPENDED.value == "suspended"
        assert PortfolioStatus.CLOSED.value == "closed"
    
    def test_risk_profile_enum(self):
        """Тест enum RiskProfile."""
        assert RiskProfile.CONSERVATIVE.value == "conservative"
        assert RiskProfile.MODERATE.value == "moderate"
        assert RiskProfile.AGGRESSIVE.value == "aggressive"
    
    def test_margin_ratio_calculation_edge_cases(self):
        """Тест граничных случаев расчета margin ratio."""
        # Нулевой equity
        portfolio = Portfolio(
            total_equity=Money(Decimal("0"), Currency.USD),
            used_margin=Money(Decimal("1000"), Currency.USD)
        )
        assert portfolio.get_margin_ratio() == Decimal("0")
        
        # Нулевой used_margin
        portfolio = Portfolio(
            total_equity=Money(Decimal("10000"), Currency.USD),
            used_margin=Money(Decimal("0"), Currency.USD)
        )
        assert portfolio.get_margin_ratio() == Decimal("0")
        
        # Отрицательный equity
        portfolio = Portfolio(
            total_equity=Money(Decimal("-1000"), Currency.USD),
            used_margin=Money(Decimal("1000"), Currency.USD)
        )
        assert portfolio.get_margin_ratio() == Decimal("-100")
    
    def test_status_transitions(self, portfolio):
        """Тест переходов между статусами."""
        # ACTIVE -> SUSPENDED
        portfolio.suspend()
        assert portfolio.status == PortfolioStatus.SUSPENDED
        assert portfolio.is_suspended is True
        
        # SUSPENDED -> ACTIVE
        portfolio.activate()
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert portfolio.is_active is True
        
        # ACTIVE -> CLOSED
        portfolio.close()
        assert portfolio.status == PortfolioStatus.CLOSED
        assert portfolio.is_closed is True
        
        # CLOSED -> ACTIVE (должно работать)
        portfolio.activate()
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert portfolio.is_active is True
    
    def test_metadata_operations(self):
        """Тест операций с метаданными."""
        portfolio = Portfolio(metadata={"strategy": "conservative", "risk_level": "low"})
        
        assert portfolio.metadata["strategy"] == "conservative"
        assert portfolio.metadata["risk_level"] == "low"
        
        # Обновление метаданных
        portfolio.metadata["new_key"] = "new_value"
        assert portfolio.metadata["new_key"] == "new_value"
    
    def test_timestamp_auto_generation(self):
        """Тест автоматической генерации временных меток."""
        portfolio = Portfolio()
        
        assert isinstance(portfolio.created_at, Timestamp)
        assert isinstance(portfolio.updated_at, Timestamp)
        assert portfolio.created_at <= portfolio.updated_at
    
    def test_currency_consistency(self, portfolio):
        """Тест консистентности валют."""
        assert portfolio.total_equity.currency == Currency.USD
        assert portfolio.free_margin.currency == Currency.USD
        assert portfolio.used_margin.currency == Currency.USD
        assert portfolio.unrealized_pnl.currency == Currency.USD
        assert portfolio.realized_pnl.currency == Currency.USD
        assert portfolio.total_pnl.currency == Currency.USD
        assert portfolio.available_margin.currency == Currency.USD 