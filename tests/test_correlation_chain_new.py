from datetime import datetime
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.core.correlation_chain import CorrelationChain, CorrelationMetrics

@pytest.fixture
def correlation_chain() -> Any:
    """Фикстура для CorrelationChain"""
    config = {
        "max_lag": 5,
        "min_correlation": 0.7,
        "significance_level": 0.05,
        "regime_window": 20,
        "stability_window": 50,
    }
    return CorrelationChain(config)

@pytest.fixture
def sample_series() -> Any:
    """Фикстура для тестовых временных рядов"""
    # Создаем два коррелированных ряда
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    # Первый ряд
    series1 = pd.Series(
        np.random.normal(0, 1, 100).cumsum() + 100, index=dates, name="BTC/USD"
    )
    # Второй ряд с лагом и корреляцией
    series2 = pd.Series(
        np.random.normal(0, 1, 100).cumsum() + 100, index=dates, name="ETH/USD"
    )
    series2 = series2.shift(2)  # Добавляем лаг
    series2 = series2.bfill()
    return {"BTC/USD": series1, "ETH/USD": series2}

@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    data = {
        "BTC/USDT": pd.DataFrame(
            {
                "open": np.random.normal(100, 1, 100),
                "high": np.random.normal(101, 1, 100),
                "low": np.random.normal(99, 1, 100),
                "close": np.random.normal(100, 1, 100),
                "volume": np.random.normal(1000, 100, 100),
            },
            index=dates,
        ),
        "ETH/USDT": pd.DataFrame(
            {
                "open": np.random.normal(100, 1, 100),
                "high": np.random.normal(101, 1, 100),
                "low": np.random.normal(99, 1, 100),
                "close": np.random.normal(100, 1, 100),
                "volume": np.random.normal(1000, 100, 100),
            },
            index=dates,
        ),
    }
    return data

class TestCorrelationChain:
    """Тесты для CorrelationChain"""

    def test_initialization(self, correlation_chain) -> None:
        """Тест инициализации"""
        assert correlation_chain.max_lag == 5
        assert correlation_chain.min_correlation == 0.7
        assert correlation_chain.significance_level == 0.05
        assert correlation_chain.regime_window == 20
        assert correlation_chain.stability_window == 50

    def test_calculate_correlation(self, correlation_chain, sample_series) -> None:
        """Тест расчета корреляции"""
        metrics = correlation_chain.calculate_correlation(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(metrics, CorrelationMetrics)
        assert metrics.correlation is not None
        assert metrics.lag is not None
        assert metrics.cointegration is not None
        assert metrics.granger_causality is not None
        assert metrics.lead_lag_relationship is not None
        assert metrics.regime_dependency is not None
        assert metrics.stability_score is not None
        assert metrics.breakpoints is not None

    def test_add_metrics(self, correlation_chain, sample_series) -> None:
        """Тест добавления метрик"""
        metrics = correlation_chain.calculate_correlation(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        correlation_chain.add_metrics("BTC/USD", "ETH/USD", metrics)
        assert ("BTC/USD", "ETH/USD") in correlation_chain.metrics

    def test_get_metrics(self, correlation_chain, sample_series) -> None:
        """Тест получения метрик"""
        metrics = correlation_chain.calculate_correlation(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        correlation_chain.add_metrics("BTC/USD", "ETH/USD", metrics)
        retrieved_metrics = correlation_chain.get_metrics("BTC/USD", "ETH/USD")
        assert retrieved_metrics is not None
        assert retrieved_metrics.correlation == metrics.correlation
        assert retrieved_metrics.lag == metrics.lag
    def test_get_correlation_matrix(self, correlation_chain, sample_series) -> None:
        """Тест получения корреляционной матрицы"""
        # Добавляем метрики для нескольких пар
        metrics1 = correlation_chain.calculate_correlation(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        correlation_chain.add_metrics("BTC/USD", "ETH/USD", metrics1)
        matrix = correlation_chain.get_correlation_matrix()
        assert isinstance(matrix, dict)
        assert "BTC/USD" in matrix
        assert "ETH/USD" in matrix["BTC/USD"]
    def test_cointegration_calculation(self, correlation_chain, sample_series) -> None:
        """Тест расчета коинтеграции"""
        cointegration = correlation_chain._calculate_cointegration(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(cointegration, dict)
        assert "is_cointegrated" in cointegration
        assert "p_value" in cointegration
        assert "test_statistic" in cointegration
    def test_granger_causality(self, correlation_chain, sample_series) -> None:
        """Тест теста Грейнджера"""
        causality = correlation_chain._calculate_granger_causality(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(causality, dict)
        assert "forward" in causality
        assert "backward" in causality
        assert "forward_p_value" in causality
        assert "backward_p_value" in causality
    def test_lead_lag_relationship(self, correlation_chain, sample_series) -> None:
        """Тест определения лидирующего/отстающего отношения"""
        relationship = correlation_chain._calculate_lead_lag_relationship(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(relationship, dict)
        assert "lead_lag" in relationship
        assert "correlation" in relationship
        assert "lag" in relationship
    def test_regime_dependency(self, correlation_chain, sample_series) -> None:
        """Тест анализа зависимости от режима"""
        dependency = correlation_chain._calculate_regime_dependency(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(dependency, dict)
        assert "bull_correlation" in dependency
        assert "bear_correlation" in dependency
        assert "regime_dependency" in dependency
    def test_stability_score(self, correlation_chain, sample_series) -> None:
        """Тест расчета показателя стабильности"""
        stability = correlation_chain._calculate_stability_score(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(stability, float)
        assert 0 <= stability <= 1
    def test_breakpoints(self, correlation_chain, sample_series) -> None:
        """Тест определения точек разрыва"""
        breakpoints = correlation_chain._calculate_breakpoints(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        assert isinstance(breakpoints, list)
        if breakpoints:
            assert all(isinstance(bp, datetime) for bp in breakpoints)
    def test_serialization(self, correlation_chain, sample_series) -> None:
        """Тест сериализации/десериализации"""
        metrics = correlation_chain.calculate_correlation(
            sample_series["BTC/USD"], sample_series["ETH/USD"]
        )
        # Сериализация
        metrics_dict = metrics.to_dict()
        # Проверка структуры
        assert "correlation" in metrics_dict
        assert "lag" in metrics_dict
        assert "cointegration" in metrics_dict
        assert "granger_causality" in metrics_dict
        assert "lead_lag_relationship" in metrics_dict
        assert "regime_dependency" in metrics_dict
        assert "stability_score" in metrics_dict
        assert "breakpoints" in metrics_dict
        # Десериализация
        new_metrics = CorrelationMetrics.from_dict(metrics_dict)
        # Проверка равенства
        assert new_metrics.correlation == metrics.correlation
        assert new_metrics.lag == metrics.lag
        assert new_metrics.cointegration == metrics.cointegration
        assert new_metrics.granger_causality == metrics.granger_causality
        assert new_metrics.lead_lag_relationship == metrics.lead_lag_relationship
        assert new_metrics.regime_dependency == metrics.regime_dependency
        assert new_metrics.stability_score == metrics.stability_score
