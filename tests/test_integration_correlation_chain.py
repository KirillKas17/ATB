import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.correlation_chain import CorrelationChain, CorrelationPair
from shared.logging import setup_logger
logger = setup_logger(__name__)
@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
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
        "BNB/USDT": pd.DataFrame(
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
@pytest.fixture
def correlation_config() -> Any:
    """Фикстура с конфигурацией корреляций"""
    return {"min_correlation": 0.7, "max_lag": 5, "window_size": 20}

@pytest.fixture
def correlation_chain(correlation_config, mock_market_data) -> Any:
    """Фикстура с экземпляром CorrelationChain"""
    return CorrelationChain(data=mock_market_data, config=correlation_config)
class TestCorrelationChain:
    def test_initialization(self, correlation_chain, correlation_config) -> None:
        """Тест инициализации"""
        assert (
            correlation_chain.min_correlation == correlation_config["min_correlation"]
        )
        assert correlation_chain.max_lag == correlation_config["max_lag"]
        assert correlation_chain.window_size == correlation_config["window_size"]
        assert isinstance(correlation_chain.pairs, list)
    def test_find_correlations(self, correlation_chain) -> None:
        """Тест поиска корреляций"""
        pairs = correlation_chain.find_correlations()
        assert isinstance(pairs, list)
        assert all(isinstance(pair, CorrelationPair) for pair in pairs)
        assert all(
            pair.correlation >= correlation_chain.min_correlation for pair in pairs
        )
    def test_calculate_correlation(self, correlation_chain) -> None:
        """Тест расчета корреляции"""
        corr, lag = correlation_chain._calculate_correlation(
            correlation_chain.data["BTC/USDT"], correlation_chain.data["ETH/USDT"]
        )
        assert isinstance(corr, float)
        assert -1 <= corr <= 1
        assert isinstance(lag, int)
        assert -correlation_chain.max_lag <= lag <= correlation_chain.max_lag
    def test_get_strongest_pairs(self, correlation_chain) -> None:
        """Тест получения самых сильных корреляций"""
        correlation_chain.find_correlations()
        pairs = correlation_chain.get_strongest_pairs(n=2)
        assert isinstance(pairs, list)
        assert len(pairs) <= 2
        assert all(isinstance(pair, CorrelationPair) for pair in pairs)
    def test_get_correlation_matrix(self, correlation_chain) -> None:
        """Тест получения матрицы корреляций"""
        correlation_chain.find_correlations()
        matrix = correlation_chain.get_correlation_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert not matrix.empty
        assert matrix.shape[0] == matrix.shape[1]
        assert all(-1 <= v <= 1 for v in matrix.values.flatten() if not np.isnan(v))
    def test_update_data(self, correlation_chain, mock_market_data) -> None:
        """Тест обновления данных"""
        correlation_chain.update_data(mock_market_data)
        assert not correlation_chain.correlation_matrix.empty
        assert len(correlation_chain.correlation_matrix) == len(
            mock_market_data.columns
        )
        assert len(correlation_chain.correlation_matrix.columns) == len(
            mock_market_data.columns
        )
    def test_get_correlation_metrics(self, correlation_chain, mock_market_data) -> None:
        """Тест получения метрик корреляции"""
        correlation_chain.update_data(mock_market_data)
        metrics = correlation_chain.get_correlation_metrics("BTC/USDT", "ETH/USDT")
        assert isinstance(metrics, CorrelationMetrics)
        assert hasattr(metrics, "correlation")
        assert hasattr(metrics, "p_value")
        assert hasattr(metrics, "lag")
        assert hasattr(metrics, "strength")
    def test_is_confirmed_by_correlation(self, correlation_chain, mock_market_data) -> None:
        """Тест подтверждения корреляцией"""
        correlation_chain.update_data(mock_market_data)
        confirmed = correlation_chain.is_confirmed_by_correlation(
            "BTC/USDT", "ETH/USDT"
        )
        assert isinstance(confirmed, bool)
    def test_get_leading_indicators(self, correlation_chain, mock_market_data) -> None:
        """Тест получения ведущих индикаторов"""
        correlation_chain.update_data(mock_market_data)
        indicators = correlation_chain.get_leading_indicators("BTC/USDT")
        assert isinstance(indicators, list)
        assert all(isinstance(i, str) for i in indicators)
    def test_get_correlation_chain(self, correlation_chain, mock_market_data) -> None:
        """Тест получения цепочки корреляций"""
        correlation_chain.update_data(mock_market_data)
        chain = correlation_chain.get_correlation_chain("BTC/USDT")
        assert isinstance(chain, list)
        assert all(isinstance(i, str) for i in chain)
    def test_plot_correlation_matrix(self, correlation_chain, mock_market_data) -> None:
        """Тест построения матрицы корреляций"""
        correlation_chain.update_data(mock_market_data)
        fig = correlation_chain.plot_correlation_matrix()
        assert fig is not None
    def test_error_handling(self, correlation_chain) -> None:
        """Тест обработки ошибок"""
        # Тест с пустыми данными
        with pytest.raises(ValueError):
            correlation_chain.update_data(pd.DataFrame())
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            correlation_chain.update_data(pd.DataFrame({"invalid": [1, 2, 3]}))
        # Тест с несуществующими символами
        with pytest.raises(ValueError):
            correlation_chain.calculate_correlation("INVALID", "ALSO_INVALID")
