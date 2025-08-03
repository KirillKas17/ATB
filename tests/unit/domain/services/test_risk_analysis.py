"""
Unit тесты для RiskAnalysis service.

Покрывает:
- Анализ рисков портфеля
- Расчет метрик риска
- Валидацию параметров риска
- Бизнес-логику управления рисками
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from domain.services.risk_analysis import RiskAnalysis
from domain.entities.risk import RiskMetrics
from domain.value_objects.money import Money
from domain.exceptions import ValidationError


class TestRiskAnalysis:
    """Тесты для RiskAnalysis service."""
    
    @pytest.fixture
    def risk_analysis_service(self):
        """Создание экземпляра RiskAnalysis."""
        return RiskAnalysis()
    
    @pytest.fixture
    def sample_portfolio_data(self) -> Dict[str, Any]:
        """Тестовые данные портфеля."""
        return {
            "total_value": "10000.00",
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "size": "1.5",
                    "entry_price": "50000.00",
                    "current_price": "52000.00",
                    "pnl": "3000.00"
                },
                {
                    "symbol": "ETH/USDT",
                    "size": "10.0",
                    "entry_price": "3000.00",
                    "current_price": "3100.00",
                    "pnl": "1000.00"
                }
            ]
        }
    
    @pytest.fixture
    def sample_risk_parameters(self) -> Dict[str, Any]:
        """Тестовые параметры риска."""
        return {
            "max_position_size": "0.1",  # 10% от портфеля
            "max_drawdown": "0.05",      # 5% максимальная просадка
            "stop_loss": "0.02",         # 2% стоп-лосс
            "take_profit": "0.05"        # 5% тейк-профит
        }
    
    def test_risk_analysis_creation(self, risk_analysis_service):
        """Тест создания сервиса анализа рисков."""
        assert risk_analysis_service is not None
        assert isinstance(risk_analysis_service, RiskAnalysis)
    
    def test_calculate_portfolio_risk(self, risk_analysis_service, sample_portfolio_data):
        """Тест расчета риска портфеля."""
        risk_metrics = risk_analysis_service.calculate_portfolio_risk(sample_portfolio_data)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.total_value == Decimal("10000.00")
        assert risk_metrics.total_pnl == Decimal("4000.00")
        assert risk_metrics.return_rate == Decimal("0.4")  # 40%
    
    def test_calculate_position_risk(self, risk_analysis_service):
        """Тест расчета риска позиции."""
        position_data = {
            "symbol": "BTC/USDT",
            "size": "1.5",
            "entry_price": "50000.00",
            "current_price": "52000.00",
            "pnl": "3000.00"
        }
        
        position_risk = risk_analysis_service.calculate_position_risk(position_data)
        
        assert position_risk["symbol"] == "BTC/USDT"
        assert position_risk["pnl"] == Decimal("3000.00")
        assert position_risk["return_rate"] == Decimal("0.04")  # 4%
    
    def test_calculate_var(self, risk_analysis_service):
        """Тест расчета Value at Risk."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01, 0.02]
        confidence_level = 0.95
        
        var = risk_analysis_service.calculate_var(returns, confidence_level)
        
        assert isinstance(var, float)
        assert var > 0
    
    def test_calculate_var_with_empty_returns(self, risk_analysis_service):
        """Тест расчета VaR с пустым списком доходностей."""
        with pytest.raises(ValueError):
            risk_analysis_service.calculate_var([], 0.95)
    
    def test_calculate_var_with_invalid_confidence(self, risk_analysis_service):
        """Тест расчета VaR с неверным уровнем доверия."""
        returns = [0.01, -0.02, 0.03]
        
        with pytest.raises(ValueError):
            risk_analysis_service.calculate_var(returns, 1.5)
        
        with pytest.raises(ValueError):
            risk_analysis_service.calculate_var(returns, -0.1)
    
    def test_calculate_volatility(self, risk_analysis_service):
        """Тест расчета волатильности."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01, 0.02]
        
        volatility = risk_analysis_service.calculate_volatility(returns)
        
        assert isinstance(volatility, float)
        assert volatility > 0
    
    def test_calculate_volatility_with_single_return(self, risk_analysis_service):
        """Тест расчета волатильности с одной доходностью."""
        returns = [0.01]
        
        volatility = risk_analysis_service.calculate_volatility(returns)
        
        assert volatility == 0.0
    
    def test_calculate_sharpe_ratio(self, risk_analysis_service):
        """Тест расчета коэффициента Шарпа."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01, 0.02]
        risk_free_rate = 0.02
        
        sharpe_ratio = risk_analysis_service.calculate_sharpe_ratio(returns, risk_free_rate)
        
        assert isinstance(sharpe_ratio, float)
    
    def test_calculate_sharpe_ratio_with_zero_volatility(self, risk_analysis_service):
        """Тест расчета коэффициента Шарпа с нулевой волатильностью."""
        returns = [0.01, 0.01, 0.01]  # Все доходности одинаковые
        risk_free_rate = 0.02
        
        sharpe_ratio = risk_analysis_service.calculate_sharpe_ratio(returns, risk_free_rate)
        
        assert sharpe_ratio == 0.0
    
    def test_calculate_max_drawdown(self, risk_analysis_service):
        """Тест расчета максимальной просадки."""
        equity_curve = [10000, 10500, 10200, 10800, 10400, 11000, 10600, 11200]
        
        max_drawdown = risk_analysis_service.calculate_max_drawdown(equity_curve)
        
        assert isinstance(max_drawdown, float)
        assert max_drawdown >= 0
        assert max_drawdown <= 1
    
    def test_calculate_max_drawdown_with_increasing_curve(self, risk_analysis_service):
        """Тест расчета максимальной просадки с растущей кривой."""
        equity_curve = [10000, 10500, 11000, 11500, 12000]
        
        max_drawdown = risk_analysis_service.calculate_max_drawdown(equity_curve)
        
        assert max_drawdown == 0.0
    
    def test_calculate_correlation_matrix(self, risk_analysis_service):
        """Тест расчета матрицы корреляций."""
        returns_data = {
            "BTC/USDT": [0.01, -0.02, 0.03, -0.01, 0.02],
            "ETH/USDT": [0.02, -0.01, 0.02, -0.02, 0.01],
            "ADA/USDT": [0.03, -0.03, 0.01, -0.01, 0.03]
        }
        
        correlation_matrix = risk_analysis_service.calculate_correlation_matrix(returns_data)
        
        assert isinstance(correlation_matrix, dict)
        assert "BTC/USDT" in correlation_matrix
        assert "ETH/USDT" in correlation_matrix
        assert "ADA/USDT" in correlation_matrix
        
        # Проверяем, что корреляция с самим собой равна 1
        assert correlation_matrix["BTC/USDT"]["BTC/USDT"] == 1.0
        assert correlation_matrix["ETH/USDT"]["ETH/USDT"] == 1.0
        assert correlation_matrix["ADA/USDT"]["ADA/USDT"] == 1.0
    
    def test_validate_risk_parameters(self, risk_analysis_service, sample_risk_parameters):
        """Тест валидации параметров риска."""
        is_valid = risk_analysis_service.validate_risk_parameters(sample_risk_parameters)
        
        assert is_valid is True
    
    def test_validate_risk_parameters_invalid_max_position_size(self, risk_analysis_service):
        """Тест валидации неверного размера позиции."""
        invalid_params = {
            "max_position_size": "1.5",  # Больше 100%
            "max_drawdown": "0.05",
            "stop_loss": "0.02",
            "take_profit": "0.05"
        }
        
        is_valid = risk_analysis_service.validate_risk_parameters(invalid_params)
        
        assert is_valid is False
    
    def test_validate_risk_parameters_invalid_max_drawdown(self, risk_analysis_service):
        """Тест валидации неверной максимальной просадки."""
        invalid_params = {
            "max_position_size": "0.1",
            "max_drawdown": "1.5",  # Больше 100%
            "stop_loss": "0.02",
            "take_profit": "0.05"
        }
        
        is_valid = risk_analysis_service.validate_risk_parameters(invalid_params)
        
        assert is_valid is False
    
    def test_validate_risk_parameters_negative_values(self, risk_analysis_service):
        """Тест валидации отрицательных значений."""
        invalid_params = {
            "max_position_size": "-0.1",
            "max_drawdown": "0.05",
            "stop_loss": "-0.02",
            "take_profit": "0.05"
        }
        
        is_valid = risk_analysis_service.validate_risk_parameters(invalid_params)
        
        assert is_valid is False
    
    def test_calculate_portfolio_concentration(self, risk_analysis_service, sample_portfolio_data):
        """Тест расчета концентрации портфеля."""
        concentration = risk_analysis_service.calculate_portfolio_concentration(sample_portfolio_data)
        
        assert isinstance(concentration, float)
        assert concentration > 0
        assert concentration <= 1
    
    def test_calculate_portfolio_concentration_single_position(self, risk_analysis_service):
        """Тест расчета концентрации с одной позицией."""
        portfolio_data = {
            "total_value": "10000.00",
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "size": "1.5",
                    "entry_price": "50000.00",
                    "current_price": "52000.00",
                    "pnl": "3000.00"
                }
            ]
        }
        
        concentration = risk_analysis_service.calculate_portfolio_concentration(portfolio_data)
        
        assert concentration == 1.0  # 100% концентрация
    
    def test_calculate_risk_adjusted_return(self, risk_analysis_service):
        """Тест расчета риск-скорректированной доходности."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01, 0.02]
        risk_free_rate = 0.02
        
        risk_adjusted_return = risk_analysis_service.calculate_risk_adjusted_return(returns, risk_free_rate)
        
        assert isinstance(risk_adjusted_return, float)
    
    def test_generate_risk_report(self, risk_analysis_service, sample_portfolio_data, sample_risk_parameters):
        """Тест генерации отчета по рискам."""
        risk_report = risk_analysis_service.generate_risk_report(sample_portfolio_data, sample_risk_parameters)
        
        assert isinstance(risk_report, dict)
        assert "portfolio_metrics" in risk_report
        assert "risk_metrics" in risk_report
        assert "recommendations" in risk_report
        assert "alerts" in risk_report
    
    def test_check_risk_alerts(self, risk_analysis_service, sample_portfolio_data, sample_risk_parameters):
        """Тест проверки предупреждений о рисках."""
        alerts = risk_analysis_service.check_risk_alerts(sample_portfolio_data, sample_risk_parameters)
        
        assert isinstance(alerts, list)
    
    def test_calculate_stress_test_scenarios(self, risk_analysis_service, sample_portfolio_data):
        """Тест расчета стресс-тестов."""
        scenarios = risk_analysis_service.calculate_stress_test_scenarios(sample_portfolio_data)
        
        assert isinstance(scenarios, dict)
        assert "market_crash" in scenarios
        assert "volatility_spike" in scenarios
        assert "correlation_breakdown" in scenarios 