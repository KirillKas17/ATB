"""
Тесты для доменного сервиса анализа рисков.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import numpy as np
import pandas as pd
from domain.services.risk_analysis import (
    DefaultRiskAnalysisService,
    RiskMetrics,
    PositionRisk,
    RiskMetricType,
    RiskAnalysisError,
)
class TestDefaultRiskAnalysisService:
    """Тесты для сервиса анализа рисков."""
    @pytest.fixture
    def risk_service(self) -> Any:
        """Фикстура сервиса анализа рисков."""
        return DefaultRiskAnalysisService()
    @pytest.fixture
    def sample_returns(self) -> Any:
        """Фикстура с примерными доходностями."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        return pd.Series(returns, name="returns")
    @pytest.fixture
    def sample_prices(self) -> Any:
        """Фикстура с примерными ценами."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = (1 + pd.Series(returns)).cumprod() * 100
        return prices
    @pytest.fixture
    def sample_positions(self) -> Any:
        """Фикстура с примерными позициями."""
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency
        from decimal import Decimal
        
        return [
            PositionRisk(
                position_id="pos1",
                symbol="BTC/USD",
                var_95=Money(Decimal("-1500"), Currency.USD),
                cvar_95=Money(Decimal("-2000"), Currency.USD),
                beta=Decimal("1.2"),
                volatility=Decimal("0.025"),
                correlation_with_portfolio=Decimal("0.8"),
                marginal_contribution=Decimal("0.15"),
                component_var=Decimal("0.12"),
                stress_test_loss=Money(Decimal("-3000"), Currency.USD),
                liquidity_score=Decimal("0.7"),
                concentration_impact=Decimal("0.05"),
            ),
            PositionRisk(
                position_id="pos2",
                symbol="ETH/USD",
                var_95=Money(Decimal("-800"), Currency.USD),
                cvar_95=Money(Decimal("-1200"), Currency.USD),
                beta=Decimal("0.8"),
                volatility=Decimal("0.02"),
                correlation_with_portfolio=Decimal("0.6"),
                marginal_contribution=Decimal("0.1"),
                component_var=Decimal("0.08"),
                stress_test_loss=Money(Decimal("-1500"), Currency.USD),
                liquidity_score=Decimal("0.8"),
                concentration_impact=Decimal("0.03"),
            ),
        ]
    def test_risk_service_initialization(self, risk_service) -> None:
        """Тест инициализации сервиса."""
        assert risk_service.risk_free_rate == 0.02
    def test_calculate_var(self, risk_service, sample_returns) -> None:
        """Тест расчета Value at Risk."""
        var_95 = risk_service.calculate_var(sample_returns, 0.95)
        var_99 = risk_service.calculate_var(sample_returns, 0.99)
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 < var_95  # VaR 99% должен быть меньше VaR 95%
    def test_calculate_var_insufficient_data(self, risk_service) -> None:
        """Тест расчета VaR с недостаточными данными."""
        insufficient_returns = pd.Series([0.01])
        with pytest.raises(RiskAnalysisError):
            risk_service.calculate_var(insufficient_returns, 0.95)
    def test_calculate_cvar(self, risk_service, sample_returns) -> None:
        """Тест расчета Conditional Value at Risk."""
        cvar_95 = risk_service.calculate_cvar(sample_returns, 0.95)
        cvar_99 = risk_service.calculate_cvar(sample_returns, 0.99)
        assert isinstance(cvar_95, float)
        assert isinstance(cvar_99, float)
        assert cvar_99 < cvar_95  # CVaR 99% должен быть меньше CVaR 95%
    def test_calculate_sharpe_ratio(self, risk_service, sample_returns) -> None:
        """Тест расчета коэффициента Шарпа."""
        sharpe = risk_service.calculate_sharpe_ratio(sample_returns, 0.02)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    def test_calculate_sharpe_ratio_zero_volatility(self, risk_service) -> None:
        """Тест расчета коэффициента Шарпа с нулевой волатильностью."""
        constant_returns = pd.Series([0.01] * 100)
        with pytest.raises(RiskAnalysisError):
            risk_service.calculate_sharpe_ratio(constant_returns, 0.02)
    def test_calculate_sortino_ratio(self, risk_service, sample_returns) -> None:
        """Тест расчета коэффициента Сортино."""
        sortino = risk_service.calculate_sortino_ratio(sample_returns, 0.02)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    def test_calculate_max_drawdown(self, risk_service, sample_prices) -> None:
        """Тест расчета максимальной просадки."""
        max_dd = risk_service.calculate_max_drawdown(sample_prices)
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Просадка должна быть отрицательной или нулевой
    def test_calculate_beta(self, risk_service) -> None:
        """Тест расчета беты."""
        asset_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        market_returns = pd.Series(np.random.normal(0.0005, 0.015, 100))
        beta = risk_service.calculate_beta(asset_returns, market_returns)
        assert isinstance(beta, float)
        assert not np.isnan(beta)
    def test_calculate_correlation_matrix(self, risk_service) -> None:
        """Тест расчета корреляционной матрицы."""
        returns_df = pd.DataFrame({
            'asset1': np.random.normal(0.001, 0.02, 100),
            'asset2': np.random.normal(0.001, 0.02, 100),
            'asset3': np.random.normal(0.001, 0.02, 100),
        })
        corr_matrix = risk_service.calculate_correlation_matrix(returns_df)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert all(corr_matrix.diagonal() == 1.0)  # Диагональ должна быть 1
    def test_calculate_portfolio_risk(self, risk_service, sample_positions) -> None:
        """Тест расчета риска портфеля."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        portfolio_risk = risk_service.calculate_portfolio_risk(returns)
        assert isinstance(portfolio_risk, RiskMetrics)
        assert hasattr(portfolio_risk, 'portfolio_volatility')
        assert hasattr(portfolio_risk, 'portfolio_var_95')
        assert hasattr(portfolio_risk, 'max_drawdown')
        assert hasattr(portfolio_risk, 'sharpe_ratio')
    def test_calculate_risk_metrics(self, risk_service, sample_returns) -> None:
        """Тест расчета всех метрик риска."""
        metrics = risk_service.calculate_portfolio_risk(sample_returns)
        assert isinstance(metrics, RiskMetrics)
        assert hasattr(metrics, 'portfolio_volatility')
        assert hasattr(metrics, 'portfolio_var_95')
        assert hasattr(metrics, 'portfolio_cvar_95')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'sortino_ratio')
        assert hasattr(metrics, 'portfolio_beta')
        assert hasattr(metrics, 'avg_correlation')
        assert hasattr(metrics, 'concentration_ratio')
        assert hasattr(metrics, 'liquidity_ratio')
    def test_calculate_risk_metrics_insufficient_data(self, risk_service) -> None:
        """Тест расчета метрик риска с недостаточными данными."""
        insufficient_returns = pd.Series([0.01])
        # Для недостаточных данных должен возвращаться пустой RiskMetrics
        metrics = risk_service.calculate_portfolio_risk(insufficient_returns)
        assert isinstance(metrics, RiskMetrics)
    def test_optimize_portfolio(self, risk_service) -> None:
        """Тест оптимизации портфеля."""
        returns_df = pd.DataFrame({
            'asset1': np.random.normal(0.001, 0.02, 100),
            'asset2': np.random.normal(0.001, 0.02, 100),
            'asset3': np.random.normal(0.001, 0.02, 100),
        })
        weights, metrics = risk_service.optimize_portfolio(returns_df, 0.02)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3
        assert all(0 <= w <= 1 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6  # Сумма весов должна быть 1
        assert isinstance(metrics, RiskMetrics)
    def test_optimize_portfolio_no_target(self, risk_service) -> None:
        """Тест оптимизации портфеля без целевой доходности."""
        returns_df = pd.DataFrame({
            'asset1': np.random.normal(0.001, 0.02, 100),
            'asset2': np.random.normal(0.001, 0.02, 100),
        })
        weights, metrics = risk_service.optimize_portfolio(returns_df, 0.02)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 2
        assert all(0 <= w <= 1 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert isinstance(metrics, RiskMetrics)
class TestRiskMetrics:
    """Тесты для метрик риска."""
    def test_risk_metrics_creation(self) -> None:
        """Тест создания метрик риска."""
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency
        from decimal import Decimal
        
        metrics = RiskMetrics(
            portfolio_volatility=Decimal("0.02"),
            portfolio_var_95=Money(Decimal("-1500"), Currency.USD),
            portfolio_cvar_95=Money(Decimal("-2000"), Currency.USD),
            max_drawdown=Decimal("-0.15"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("1.2"),
            portfolio_beta=Decimal("1.1"),
            avg_correlation=Decimal("0.8"),
            concentration_ratio=Decimal("0.3"),
            liquidity_ratio=Decimal("0.7"),
            stress_test_score=Decimal("0.5"),
            metadata={"test": "value"}
        )
        assert metrics.portfolio_volatility == Decimal("0.02")
        assert metrics.portfolio_var_95.amount == Decimal("-1500")
        assert metrics.max_drawdown == Decimal("-0.15")
        assert metrics.sharpe_ratio == Decimal("1.5")
class TestPositionRisk:
    """Тесты для риска позиции."""
    def test_position_risk_creation(self) -> None:
        """Тест создания риска позиции."""
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency
        from decimal import Decimal
        
        position = PositionRisk(
            position_id="test_pos",
            symbol="BTC/USD",
            var_95=Money(Decimal("-1500"), Currency.USD),
            cvar_95=Money(Decimal("-2000"), Currency.USD),
            beta=Decimal("1.2"),
            volatility=Decimal("0.025"),
            correlation_with_portfolio=Decimal("0.8"),
            marginal_contribution=Decimal("0.15"),
            component_var=Decimal("0.12"),
            stress_test_loss=Money(Decimal("-3000"), Currency.USD),
            liquidity_score=Decimal("0.7"),
            concentration_impact=Decimal("0.05"),
        )
        assert position.position_id == "test_pos"
        assert position.symbol == "BTC/USD"
        assert position.beta == Decimal("1.2")
        assert position.volatility == Decimal("0.025")
class TestRiskMetricType:
    """Тесты для типов метрик риска."""
    def test_risk_metric_type_values(self) -> None:
        """Тест значений типов метрик риска."""
        assert RiskMetricType.VAR.value == "value_at_risk"
        assert RiskMetricType.CVAR.value == "conditional_value_at_risk"
        assert RiskMetricType.SHARPE.value == "sharpe_ratio"
        assert RiskMetricType.SORTINO.value == "sortino_ratio"
        assert RiskMetricType.MAX_DRAWDOWN.value == "max_drawdown"
        assert RiskMetricType.VOLATILITY.value == "volatility"
        assert RiskMetricType.BETA.value == "beta"
        assert RiskMetricType.CORRELATION.value == "correlation" 
