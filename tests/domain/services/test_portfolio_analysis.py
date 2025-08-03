"""
Unit тесты для domain/services/portfolio_analysis.py.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime, timezone

from domain.services.portfolio_analysis import PortfolioAnalysisService
from domain.entities.portfolio import Portfolio
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions.base_exceptions import ValidationError


class TestPortfolioAnalysisService:
    """Тесты для PortfolioAnalysisService."""
    
    @pytest.fixture
    def service(self):
        """Создание сервиса."""
        return PortfolioAnalysisService()
    
    @pytest.fixture
    def sample_portfolio(self) -> Portfolio:
        """Тестовый портфель."""
        return Portfolio(
            id="portfolio_001",
            name="Test Portfolio",
            currency="USD",
            total_value=Decimal("100000.00"),
            cash_balance=Decimal("20000.00"),
            positions=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_positions(self) -> List[Position]:
        """Тестовые позиции."""
        return [
            Position(
                id="pos_001",
                symbol="BTCUSD",
                side="LONG",
                quantity=Decimal("2.0"),
                average_price=Decimal("45000.00"),
                current_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("10000.00"),
                realized_pnl=Decimal("5000.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            Position(
                id="pos_002",
                symbol="ETHUSD",
                side="LONG",
                quantity=Decimal("10.0"),
                average_price=Decimal("3000.00"),
                current_price=Decimal("3500.00"),
                unrealized_pnl=Decimal("5000.00"),
                realized_pnl=Decimal("2000.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]
    
    @pytest.fixture
    def sample_trades(self) -> List[Trade]:
        """Тестовые сделки."""
        return [
            Trade(
                id="trade_001",
                symbol="BTCUSD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("45000.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            Trade(
                id="trade_002",
                symbol="BTCUSD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]
    
    def test_calculate_total_value(self, service, sample_portfolio, sample_positions):
        """Тест расчета общей стоимости портфеля."""
        sample_portfolio.positions = sample_positions
        
        total_value = service.calculate_total_value(sample_portfolio)
        
        expected_value = Decimal("20000.00") + (Decimal("2.0") * Decimal("50000.00")) + (Decimal("10.0") * Decimal("3500.00"))
        assert total_value == expected_value
    
    def test_calculate_total_pnl(self, service, sample_portfolio, sample_positions):
        """Тест расчета общего P&L."""
        sample_portfolio.positions = sample_positions
        
        total_pnl = service.calculate_total_pnl(sample_portfolio)
        
        expected_pnl = Decimal("10000.00") + Decimal("5000.00") + Decimal("5000.00") + Decimal("2000.00")
        assert total_pnl == expected_pnl
    
    def test_calculate_unrealized_pnl(self, service, sample_portfolio, sample_positions):
        """Тест расчета нереализованного P&L."""
        sample_portfolio.positions = sample_positions
        
        unrealized_pnl = service.calculate_unrealized_pnl(sample_portfolio)
        
        expected_pnl = Decimal("10000.00") + Decimal("5000.00")
        assert unrealized_pnl == expected_pnl
    
    def test_calculate_realized_pnl(self, service, sample_portfolio, sample_positions):
        """Тест расчета реализованного P&L."""
        sample_portfolio.positions = sample_positions
        
        realized_pnl = service.calculate_realized_pnl(sample_portfolio)
        
        expected_pnl = Decimal("5000.00") + Decimal("2000.00")
        assert realized_pnl == expected_pnl
    
    def test_calculate_return_percentage(self, service, sample_portfolio, sample_positions):
        """Тест расчета процентной доходности."""
        sample_portfolio.positions = sample_positions
        
        return_pct = service.calculate_return_percentage(sample_portfolio)
        
        # Ожидаемая доходность: (общий P&L / начальная стоимость) * 100
        total_pnl = Decimal("10000.00") + Decimal("5000.00") + Decimal("5000.00") + Decimal("2000.00")
        initial_value = Decimal("100000.00")
        expected_return = (total_pnl / initial_value) * Decimal("100")
        
        assert return_pct == expected_return
    
    def test_calculate_position_allocation(self, service, sample_portfolio, sample_positions):
        """Тест расчета распределения позиций."""
        sample_portfolio.positions = sample_positions
        
        allocation = service.calculate_position_allocation(sample_portfolio)
        
        # Проверяем, что возвращается словарь с распределением
        assert isinstance(allocation, dict)
        assert "BTCUSD" in allocation
        assert "ETHUSD" in allocation
        
        # Проверяем, что сумма долей равна 100%
        total_allocation = sum(allocation.values())
        assert total_allocation == Decimal("100.0")
    
    def test_calculate_risk_metrics(self, service, sample_portfolio, sample_positions):
        """Тест расчета метрик риска."""
        sample_portfolio.positions = sample_positions
        
        risk_metrics = service.calculate_risk_metrics(sample_portfolio)
        
        # Проверяем, что возвращается словарь с метриками риска
        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert "var_95" in risk_metrics
    
    def test_calculate_correlation_matrix(self, service, sample_portfolio, sample_positions):
        """Тест расчета матрицы корреляций."""
        sample_portfolio.positions = sample_positions
        
        correlation_matrix = service.calculate_correlation_matrix(sample_portfolio)
        
        # Проверяем, что возвращается матрица корреляций
        assert isinstance(correlation_matrix, dict)
        assert "BTCUSD" in correlation_matrix
        assert "ETHUSD" in correlation_matrix
        
        # Проверяем, что корреляция с самим собой равна 1
        assert correlation_matrix["BTCUSD"]["BTCUSD"] == Decimal("1.0")
        assert correlation_matrix["ETHUSD"]["ETHUSD"] == Decimal("1.0")
    
    def test_calculate_portfolio_beta(self, service, sample_portfolio, sample_positions):
        """Тест расчета беты портфеля."""
        sample_portfolio.positions = sample_positions
        
        beta = service.calculate_portfolio_beta(sample_portfolio, "BTCUSD")
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(beta, Decimal)
        assert beta >= Decimal("0")
    
    def test_calculate_alpha(self, service, sample_portfolio, sample_positions):
        """Тест расчета альфы портфеля."""
        sample_portfolio.positions = sample_positions
        
        alpha = service.calculate_alpha(sample_portfolio, Decimal("0.05"), Decimal("0.10"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(alpha, Decimal)
    
    def test_calculate_treynor_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета коэффициента Трейнора."""
        sample_portfolio.positions = sample_positions
        
        treynor_ratio = service.calculate_treynor_ratio(sample_portfolio, Decimal("0.05"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(treynor_ratio, Decimal)
    
    def test_calculate_information_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета информационного коэффициента."""
        sample_portfolio.positions = sample_positions
        
        information_ratio = service.calculate_information_ratio(sample_portfolio, Decimal("0.08"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(information_ratio, Decimal)
    
    def test_calculate_sortino_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета коэффициента Сортино."""
        sample_portfolio.positions = sample_positions
        
        sortino_ratio = service.calculate_sortino_ratio(sample_portfolio, Decimal("0.05"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(sortino_ratio, Decimal)
    
    def test_calculate_calmar_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета коэффициента Кальмара."""
        sample_portfolio.positions = sample_positions
        
        calmar_ratio = service.calculate_calmar_ratio(sample_portfolio)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(calmar_ratio, Decimal)
    
    def test_calculate_omega_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета омега-коэффициента."""
        sample_portfolio.positions = sample_positions
        
        omega_ratio = service.calculate_omega_ratio(sample_portfolio, Decimal("0.05"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(omega_ratio, Decimal)
    
    def test_calculate_ulcer_index(self, service, sample_portfolio, sample_positions):
        """Тест расчета индекса язвы."""
        sample_portfolio.positions = sample_positions
        
        ulcer_index = service.calculate_ulcer_index(sample_portfolio)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(ulcer_index, Decimal)
        assert ulcer_index >= Decimal("0")
    
    def test_calculate_gain_to_pain_ratio(self, service, sample_portfolio, sample_positions):
        """Тест расчета соотношения прибыли к убыткам."""
        sample_portfolio.positions = sample_positions
        
        gain_to_pain_ratio = service.calculate_gain_to_pain_ratio(sample_portfolio)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(gain_to_pain_ratio, Decimal)
    
    def test_calculate_profit_factor(self, service, sample_portfolio, sample_positions):
        """Тест расчета фактора прибыли."""
        sample_portfolio.positions = sample_positions
        
        profit_factor = service.calculate_profit_factor(sample_portfolio)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(profit_factor, Decimal)
        assert profit_factor >= Decimal("0")
    
    def test_calculate_win_rate(self, service, sample_portfolio, sample_trades):
        """Тест расчета процента выигрышных сделок."""
        win_rate = service.calculate_win_rate(sample_trades)
        
        # Проверяем, что возвращается числовое значение между 0 и 100
        assert isinstance(win_rate, Decimal)
        assert Decimal("0") <= win_rate <= Decimal("100")
    
    def test_calculate_average_win(self, service, sample_trades):
        """Тест расчета средней выигрышной сделки."""
        avg_win = service.calculate_average_win(sample_trades)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(avg_win, Decimal)
    
    def test_calculate_average_loss(self, service, sample_trades):
        """Тест расчета средней проигрышной сделки."""
        avg_loss = service.calculate_average_loss(sample_trades)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(avg_loss, Decimal)
    
    def test_calculate_largest_win(self, service, sample_trades):
        """Тест расчета наибольшей выигрышной сделки."""
        largest_win = service.calculate_largest_win(sample_trades)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(largest_win, Decimal)
    
    def test_calculate_largest_loss(self, service, sample_trades):
        """Тест расчета наибольшей проигрышной сделки."""
        largest_loss = service.calculate_largest_loss(sample_trades)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(largest_loss, Decimal)
    
    def test_calculate_consecutive_wins(self, service, sample_trades):
        """Тест расчета последовательных выигрышей."""
        consecutive_wins = service.calculate_consecutive_wins(sample_trades)
        
        # Проверяем, что возвращается целое число
        assert isinstance(consecutive_wins, int)
        assert consecutive_wins >= 0
    
    def test_calculate_consecutive_losses(self, service, sample_trades):
        """Тест расчета последовательных проигрышей."""
        consecutive_losses = service.calculate_consecutive_losses(sample_trades)
        
        # Проверяем, что возвращается целое число
        assert isinstance(consecutive_losses, int)
        assert consecutive_losses >= 0
    
    def test_calculate_max_drawdown(self, service, sample_portfolio, sample_positions):
        """Тест расчета максимальной просадки."""
        sample_portfolio.positions = sample_positions
        
        max_drawdown = service.calculate_max_drawdown(sample_portfolio)
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(max_drawdown, Decimal)
        assert max_drawdown <= Decimal("0")  # Просадка не может быть положительной
    
    def test_calculate_value_at_risk(self, service, sample_portfolio, sample_positions):
        """Тест расчета Value at Risk."""
        sample_portfolio.positions = sample_positions
        
        var_95 = service.calculate_value_at_risk(sample_portfolio, Decimal("0.05"))
        var_99 = service.calculate_value_at_risk(sample_portfolio, Decimal("0.01"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(var_95, Decimal)
        assert isinstance(var_99, Decimal)
        
        # VaR 99% должен быть больше VaR 95%
        assert var_99 <= var_95
    
    def test_calculate_expected_shortfall(self, service, sample_portfolio, sample_positions):
        """Тест расчета Expected Shortfall."""
        sample_portfolio.positions = sample_positions
        
        es = service.calculate_expected_shortfall(sample_portfolio, Decimal("0.05"))
        
        # Проверяем, что возвращается числовое значение
        assert isinstance(es, Decimal)
    
    def test_calculate_portfolio_metrics_summary(self, service, sample_portfolio, sample_positions):
        """Тест расчета сводки метрик портфеля."""
        sample_portfolio.positions = sample_positions
        
        summary = service.calculate_portfolio_metrics_summary(sample_portfolio)
        
        # Проверяем, что возвращается словарь с метриками
        assert isinstance(summary, dict)
        assert "total_value" in summary
        assert "total_pnl" in summary
        assert "return_percentage" in summary
        assert "unrealized_pnl" in summary
        assert "realized_pnl" in summary
        assert "position_allocation" in summary
        assert "risk_metrics" in summary
    
    def test_calculate_portfolio_with_empty_positions(self, service, sample_portfolio):
        """Тест расчета портфеля с пустыми позициями."""
        # Портфель без позиций
        total_value = service.calculate_total_value(sample_portfolio)
        total_pnl = service.calculate_total_pnl(sample_portfolio)
        
        assert total_value == sample_portfolio.cash_balance
        assert total_pnl == Decimal("0")
    
    def test_calculate_portfolio_with_negative_pnl(self, service, sample_portfolio):
        """Тест расчета портфеля с отрицательным P&L."""
        negative_position = Position(
            id="pos_003",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            current_price=Decimal("45000.00"),
            unrealized_pnl=Decimal("-5000.00"),
            realized_pnl=Decimal("-1000.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        sample_portfolio.positions = [negative_position]
        
        total_pnl = service.calculate_total_pnl(sample_portfolio)
        assert total_pnl == Decimal("-6000.00")
    
    def test_calculate_portfolio_with_zero_prices(self, service, sample_portfolio):
        """Тест расчета портфеля с нулевыми ценами."""
        zero_price_position = Position(
            id="pos_004",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            average_price=Decimal("0.00"),
            current_price=Decimal("0.00"),
            unrealized_pnl=Decimal("0.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        sample_portfolio.positions = [zero_price_position]
        
        total_value = service.calculate_total_value(sample_portfolio)
        assert total_value == sample_portfolio.cash_balance
    
    def test_calculate_portfolio_with_high_volatility(self, service, sample_portfolio):
        """Тест расчета портфеля с высокой волатильностью."""
        volatile_position = Position(
            id="pos_005",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            current_price=Decimal("100000.00"),  # 100% рост
            unrealized_pnl=Decimal("50000.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        sample_portfolio.positions = [volatile_position]
        
        return_pct = service.calculate_return_percentage(sample_portfolio)
        assert return_pct > Decimal("50")  # Должна быть высокая доходность 