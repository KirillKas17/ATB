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
    def risk_service(self) -> Any:
        """Создание экземпляра сервиса."""
        return RiskAnalysisServiceAdapter()
    @pytest.fixture
    def sample_returns(self) -> Any:
        """Тестовые данные доходностей."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 100))
    @pytest.fixture
    def sample_portfolio(self) -> Any:
        """Тестовый портфель."""
        return {
            'BTC/USDT': 0.5,
            'ETH/USDT': 0.3,
            'ADA/USDT': 0.2
        }
    def test_calculate_risk_metrics(self, risk_service, sample_returns) -> None:
        """Тест расчета метрик риска."""
        metrics = risk_service.calculate_risk_metrics(sample_returns)
        assert isinstance(metrics, dict)
        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        # Проверяем, что значения в разумных пределах
        assert 0 <= metrics['var_95'] <= 1
        assert 0 <= metrics['var_99'] <= 1
        assert 0 <= metrics['volatility'] <= 1
        assert metrics['var_99'] >= metrics['var_95']
    def test_calculate_var(self, risk_service, sample_returns) -> None:
        """Тест расчета VaR."""
        var_95 = risk_service.calculate_var(sample_returns, 0.95)
        var_99 = risk_service.calculate_var(sample_returns, 0.99)
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 <= var_95  # VaR 99% должен быть меньше VaR 95%
    def test_calculate_cvar(self, risk_service, sample_returns) -> None:
        """Тест расчета CVaR."""
        cvar_95 = risk_service.calculate_cvar(sample_returns, 0.95)
        cvar_99 = risk_service.calculate_cvar(sample_returns, 0.99)
        assert isinstance(cvar_95, float)
        assert isinstance(cvar_99, float)
        assert cvar_99 <= cvar_95  # CVaR 99% должен быть меньше CVaR 95%
    def test_calculate_sharpe_ratio(self, risk_service, sample_returns) -> None:
        """Тест расчета коэффициента Шарпа."""
        sharpe = risk_service.calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        # Коэффициент Шарпа может быть отрицательным
    def test_calculate_sortino_ratio(self, risk_service, sample_returns) -> None:
        """Тест расчета коэффициента Сортино."""
        sortino = risk_service.calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
        # Коэффициент Сортино может быть отрицательным
    def test_calculate_max_drawdown(self, risk_service) -> None:
        """Тест расчета максимальной просадки."""
        # Создаем цены с просадкой
        prices = pd.Series([100, 110, 105, 95, 100, 90, 95])
        max_dd = risk_service.calculate_max_drawdown(prices)
        assert isinstance(max_dd, float)
        assert max_dd >= 0
        assert max_dd <= 1
    def test_calculate_beta(self, risk_service, sample_returns) -> None:
        """Тест расчета беты."""
        market_returns = pd.Series(np.random.normal(0.0005, 0.015, 100))
        beta = risk_service.calculate_beta(sample_returns, market_returns)
        assert isinstance(beta, float)
    def test_calculate_correlation_matrix(self, risk_service) -> None:
        """Тест расчета корреляционной матрицы."""
        returns_df = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.02, 100),
            'ETH': np.random.normal(0.001, 0.02, 100),
            'ADA': np.random.normal(0.001, 0.02, 100)
        })
        corr_matrix = risk_service.calculate_correlation_matrix(returns_df)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert (corr_matrix.values.diagonal() == 1.0).all()
    def test_calculate_portfolio_risk(self, risk_service) -> None:
        """Тест расчета риска портфеля."""
        positions = [
            Mock(symbol='BTC', size=0.5, current_price=50000),
            Mock(symbol='ETH', size=0.3, current_price=3000),
            Mock(symbol='ADA', size=0.2, current_price=1.5)
        ]
        market_data = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.02, 100),
            'ETH': np.random.normal(0.001, 0.02, 100),
            'ADA': np.random.normal(0.001, 0.02, 100)
        })
        portfolio_risk = risk_service.calculate_portfolio_risk(positions, market_data)
        assert isinstance(portfolio_risk, dict)
        assert 'total_risk' in portfolio_risk
        assert 'var_95' in portfolio_risk
        assert 'sharpe_ratio' in portfolio_risk
    def test_optimize_portfolio(self, risk_service) -> None:
        """Тест оптимизации портфеля."""
        returns_df = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.02, 100),
            'ETH': np.random.normal(0.001, 0.02, 100),
            'ADA': np.random.normal(0.001, 0.02, 100)
        })
        optimization = risk_service.optimize_portfolio(returns_df, target_return=0.1)
        assert isinstance(optimization, dict)
        assert 'optimal_weights' in optimization
        assert 'expected_return' in optimization
        assert 'expected_risk' in optimization
    def test_analyze_portfolio_risk(self, risk_service, sample_portfolio) -> None:
        """Тест анализа риска портфеля."""
        risk_analysis = risk_service.analyze_portfolio_risk(sample_portfolio)
        assert isinstance(risk_analysis, dict)
        assert 'var_95' in risk_analysis
        assert 'var_99' in risk_analysis
        assert 'expected_shortfall' in risk_analysis
        assert 'concentration_risk' in risk_analysis
        assert 'liquidity_risk' in risk_analysis
    def test_calculate_risk_metrics_empty_data(self, risk_service) -> None:
        """Тест расчета метрик риска с пустыми данными."""
        empty_returns = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            risk_service.calculate_risk_metrics(empty_returns)
    def test_calculate_var_invalid_confidence(self, risk_service, sample_returns) -> None:
        """Тест расчета VaR с неверным уровнем доверия."""
        with pytest.raises(ValueError):
            risk_service.calculate_var(sample_returns, 1.5)
        with pytest.raises(ValueError):
            risk_service.calculate_var(sample_returns, -0.1)
    def test_calculate_beta_different_lengths(self, risk_service) -> None:
        """Тест расчета беты с разными длинами данных."""
        asset_returns = pd.Series([0.01, 0.02, 0.03])
        market_returns = pd.Series([0.005, 0.01])
        with pytest.raises(ValueError):
            risk_service.calculate_beta(asset_returns, market_returns) 
