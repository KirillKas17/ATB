"""
Тесты для доменного сервиса цепочек корреляций.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.correlation_chain import CorrelationChain, ICorrelationChain
class TestCorrelationChain:
    """Тесты для сервиса цепочек корреляций."""
    @pytest.fixture
    def correlation_chain(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура сервиса цепочек корреляций."""
        return CorrelationChain()
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'BTC_close': np.random.uniform(50000, 51000, 100),
            'ETH_close': np.random.uniform(3000, 3100, 100),
            'ADA_close': np.random.uniform(0.5, 0.6, 100),
            'DOT_close': np.random.uniform(7, 8, 100),
            'LINK_close': np.random.uniform(15, 16, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    @pytest.fixture
    def sample_correlation_matrix(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с матрицей корреляций."""
        return pd.DataFrame({
            'BTC': [1.0, 0.8, 0.6, 0.7, 0.5],
            'ETH': [0.8, 1.0, 0.7, 0.8, 0.6],
            'ADA': [0.6, 0.7, 1.0, 0.5, 0.4],
            'DOT': [0.7, 0.8, 0.5, 1.0, 0.7],
            'LINK': [0.5, 0.6, 0.4, 0.7, 1.0]
        }, index=['BTC', 'ETH', 'ADA', 'DOT', 'LINK'])
    def test_correlation_chain_initialization(self, correlation_chain) -> None:
        """Тест инициализации сервиса."""
        assert correlation_chain is not None
        assert isinstance(correlation_chain, ICorrelationChain)
        assert hasattr(correlation_chain, 'config')
        assert isinstance(correlation_chain.config, dict)
    def test_correlation_chain_config_defaults(self, correlation_chain) -> None:
        """Тест конфигурации по умолчанию."""
        config = correlation_chain.config
        assert "correlation_threshold" in config
        assert "chain_length_limit" in config
        assert "lookback_period" in config
        assert "min_correlation" in config
        assert isinstance(config["correlation_threshold"], float)
        assert isinstance(config["chain_length_limit"], int)
    def test_calculate_correlation_matrix(self, correlation_chain, sample_market_data) -> None:
        """Тест расчета матрицы корреляций."""
        correlation_matrix = correlation_chain.calculate_correlation_matrix(sample_market_data)
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert not correlation_matrix.empty
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        # Проверяем, что диагональ равна 1.0
        for i in range(len(correlation_matrix)):
            assert abs(correlation_matrix.iloc[i, i] - 1.0) < 1e-6
        # Проверяем, что все значения корреляции в диапазоне [-1, 1]
        for col in correlation_matrix.columns:
            for value in correlation_matrix[col]:
                assert value >= -1.0 and value <= 1.0
    def test_calculate_correlation_matrix_empty_data(self, correlation_chain) -> None:
        """Тест расчета матрицы корреляций с пустыми данными."""
        empty_data = pd.DataFrame()
        correlation_matrix = correlation_chain.calculate_correlation_matrix(empty_data)
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.empty
    def test_calculate_correlation_matrix_insufficient_data(self, correlation_chain) -> None:
        """Тест расчета матрицы корреляций с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'BTC_close': [50000, 50001],
            'ETH_close': [3000, 3001]
        })
        correlation_matrix = correlation_chain.calculate_correlation_matrix(insufficient_data)
        assert isinstance(correlation_matrix, pd.DataFrame)
        # При недостатке данных матрица должна быть пустой или содержать NaN
        assert correlation_matrix.empty or correlation_matrix.isna().all().all()
    def test_find_correlation_chains(self, correlation_chain, sample_correlation_matrix) -> None:
        """Тест поиска цепочек корреляций."""
        chains = correlation_chain.find_correlation_chains(sample_correlation_matrix)
        assert isinstance(chains, list)
        # Проверяем, что все цепочки имеют правильную структуру
        for chain in chains:
            assert isinstance(chain, CorrelationChainResult)
            assert "assets" in chain
            assert "correlation_strength" in chain
            assert "chain_length" in chain
            assert "confidence" in chain
            assert isinstance(chain["assets"], list)
            assert isinstance(chain["correlation_strength"], float)
            assert isinstance(chain["chain_length"], int)
            assert isinstance(chain["confidence"], float)
            assert len(chain["assets"]) >= 2
            assert chain["correlation_strength"] >= 0.0 and chain["correlation_strength"] <= 1.0
            assert chain["chain_length"] >= 2
            assert chain["confidence"] >= 0.0 and chain["confidence"] <= 1.0
    def test_find_correlation_chains_empty_matrix(self, correlation_chain) -> None:
        """Тест поиска цепочек корреляций с пустой матрицей."""
        empty_matrix = pd.DataFrame()
        chains = correlation_chain.find_correlation_chains(empty_matrix)
        assert isinstance(chains, list)
        assert len(chains) == 0
    def test_find_correlation_chains_low_correlation(self, correlation_chain) -> None:
        """Тест поиска цепочек корреляций с низкой корреляцией."""
        low_correlation_matrix = pd.DataFrame({
            'BTC': [1.0, 0.1, 0.1],
            'ETH': [0.1, 1.0, 0.1],
            'ADA': [0.1, 0.1, 1.0]
        }, index=['BTC', 'ETH', 'ADA'])
        chains = correlation_chain.find_correlation_chains(low_correlation_matrix)
        assert isinstance(chains, list)
        # При низкой корреляции должно быть мало цепочек или их не должно быть
        assert len(chains) <= 1
    def test_analyze_correlation_stability(self, correlation_chain, sample_market_data) -> None:
        """Тест анализа стабильности корреляций."""
        stability = correlation_chain.analyze_correlation_stability(sample_market_data)
        assert isinstance(stability, dict)
        assert "stability_score" in stability
        assert "volatility" in stability
        assert "trend" in stability
        assert "breakdown_points" in stability
        assert isinstance(stability["stability_score"], float)
        assert isinstance(stability["volatility"], float)
        assert isinstance(stability["trend"], str)
        assert isinstance(stability["breakdown_points"], list)
        assert stability["stability_score"] >= 0.0 and stability["stability_score"] <= 1.0
        assert stability["volatility"] >= 0.0
        assert stability["trend"] in ["increasing", "decreasing", "stable"]
    def test_analyze_correlation_stability_empty_data(self, correlation_chain) -> None:
        """Тест анализа стабильности корреляций с пустыми данными."""
        empty_data = pd.DataFrame()
        stability = correlation_chain.analyze_correlation_stability(empty_data)
        assert isinstance(stability, dict)
        assert stability["stability_score"] == 0.0
        assert stability["volatility"] == 0.0
        assert stability["trend"] == "stable"
        assert len(stability["breakdown_points"]) == 0
    def test_detect_correlation_breakdowns(self, correlation_chain, sample_market_data) -> None:
        """Тест обнаружения разрывов корреляций."""
        breakdowns = correlation_chain.detect_correlation_breakdowns(sample_market_data)
        assert isinstance(breakdowns, list)
        # Проверяем, что все разрывы имеют правильную структуру
        for breakdown in breakdowns:
            assert isinstance(breakdown, dict)
            assert "timestamp" in breakdown
            assert "assets" in breakdown
            assert "breakdown_strength" in breakdown
            assert "breakdown_type" in breakdown
            assert isinstance(breakdown["assets"], list)
            assert isinstance(breakdown["breakdown_strength"], float)
            assert isinstance(breakdown["breakdown_type"], str)
            assert len(breakdown["assets"]) >= 2
            assert breakdown["breakdown_strength"] >= 0.0
            assert breakdown["breakdown_type"] in ["sudden", "gradual", "temporary"]
    def test_detect_correlation_breakdowns_empty_data(self, correlation_chain) -> None:
        """Тест обнаружения разрывов корреляций с пустыми данными."""
        empty_data = pd.DataFrame()
        breakdowns = correlation_chain.detect_correlation_breakdowns(empty_data)
        assert isinstance(breakdowns, list)
        assert len(breakdowns) == 0
    def test_calculate_correlation_impact(self, correlation_chain, sample_correlation_matrix) -> None:
        """Тест расчета влияния корреляций."""
        impact = correlation_chain.calculate_correlation_impact(sample_correlation_matrix)
        assert isinstance(impact, dict)
        assert "systemic_risk" in impact
        assert "diversification_benefit" in impact
        assert "portfolio_impact" in impact
        assert "risk_contribution" in impact
        assert isinstance(impact["systemic_risk"], float)
        assert isinstance(impact["diversification_benefit"], float)
        assert isinstance(impact["portfolio_impact"], float)
        assert isinstance(impact["risk_contribution"], dict)
        assert impact["systemic_risk"] >= 0.0 and impact["systemic_risk"] <= 1.0
        assert impact["diversification_benefit"] >= 0.0 and impact["diversification_benefit"] <= 1.0
        assert impact["portfolio_impact"] >= 0.0
    def test_calculate_correlation_impact_empty_matrix(self, correlation_chain) -> None:
        """Тест расчета влияния корреляций с пустой матрицей."""
        empty_matrix = pd.DataFrame()
        impact = correlation_chain.calculate_correlation_impact(empty_matrix)
        assert isinstance(impact, dict)
        assert impact["systemic_risk"] == 0.0
        assert impact["diversification_benefit"] == 0.0
        assert impact["portfolio_impact"] == 0.0
        assert impact["risk_contribution"] == {}
    def test_predict_correlation_changes(self, correlation_chain, sample_market_data) -> None:
        """Тест предсказания изменений корреляций."""
        predictions = correlation_chain.predict_correlation_changes(sample_market_data)
        assert isinstance(predictions, dict)
        assert "predicted_changes" in predictions
        assert "confidence" in predictions
        assert "time_horizon" in predictions
        assert "affected_assets" in predictions
        assert isinstance(predictions["predicted_changes"], list)
        assert isinstance(predictions["confidence"], float)
        assert isinstance(predictions["time_horizon"], int)
        assert isinstance(predictions["affected_assets"], list)
        assert predictions["confidence"] >= 0.0 and predictions["confidence"] <= 1.0
        assert predictions["time_horizon"] >= 0
    def test_predict_correlation_changes_empty_data(self, correlation_chain) -> None:
        """Тест предсказания изменений корреляций с пустыми данными."""
        empty_data = pd.DataFrame()
        predictions = correlation_chain.predict_correlation_changes(empty_data)
        assert isinstance(predictions, dict)
        assert len(predictions["predicted_changes"]) == 0
        assert predictions["confidence"] == 0.0
        assert predictions["time_horizon"] == 0
        assert len(predictions["affected_assets"]) == 0
    def test_get_correlation_statistics(self, correlation_chain, sample_correlation_matrix) -> None:
        """Тест получения статистики корреляций."""
        stats = correlation_chain.get_correlation_statistics(sample_correlation_matrix)
        assert isinstance(stats, dict)
        assert "mean_correlation" in stats
        assert "correlation_std" in stats
        assert "max_correlation" in stats
        assert "min_correlation" in stats
        assert "correlation_distribution" in stats
        assert isinstance(stats["mean_correlation"], float)
        assert isinstance(stats["correlation_std"], float)
        assert isinstance(stats["max_correlation"], float)
        assert isinstance(stats["min_correlation"], float)
        assert isinstance(stats["correlation_distribution"], dict)
        assert stats["max_correlation"] >= stats["min_correlation"]
        assert stats["correlation_std"] >= 0.0
    def test_get_correlation_statistics_empty_matrix(self, correlation_chain) -> None:
        """Тест получения статистики корреляций с пустой матрицей."""
        empty_matrix = pd.DataFrame()
        stats = correlation_chain.get_correlation_statistics(empty_matrix)
        assert isinstance(stats, dict)
        assert stats["mean_correlation"] == 0.0
        assert stats["correlation_std"] == 0.0
        assert stats["max_correlation"] == 0.0
        assert stats["min_correlation"] == 0.0
        assert stats["correlation_distribution"] == {}
    def test_correlation_chain_error_handling(self, correlation_chain) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными
        with pytest.raises(Exception):
            correlation_chain.calculate_correlation_matrix(None)
        # Тест с невалидным типом данных
        with pytest.raises(Exception):
            correlation_chain.find_correlation_chains("invalid_matrix")
        # Тест с невалидной матрицей
        with pytest.raises(Exception):
            correlation_chain.analyze_correlation_stability("invalid_data")
    def test_correlation_chain_performance(self, correlation_chain, sample_market_data) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(10):
            correlation_chain.calculate_correlation_matrix(sample_market_data)
        end_time = time.time()
        # Проверяем, что 10 операций выполняются менее чем за 2 секунды
        assert (end_time - start_time) < 2.0
    def test_correlation_chain_thread_safety(self, correlation_chain, sample_market_data) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue
        results = queue.Queue()
        def calculate_correlation() -> Any:
            try:
                result = correlation_chain.calculate_correlation_matrix(sample_market_data)
                results.put(result)
            except Exception as e:
                results.put(e)
        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=calculate_correlation)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(5):
            result = results.get()
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    def test_correlation_chain_config_customization(self: "TestCorrelationChain") -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "correlation_threshold": 0.8,
            "chain_length_limit": 5,
            "lookback_period": 50,
            "min_correlation": 0.3
        }
        service = CorrelationChain(custom_config)
        assert service.config["correlation_threshold"] == 0.8
        assert service.config["chain_length_limit"] == 5
        assert service.config["lookback_period"] == 50
        assert service.config["min_correlation"] == 0.3
    def test_correlation_chain_integration_with_market_data(self, correlation_chain, sample_market_data) -> None:
        """Тест интеграции с рыночными данными."""
        # Полный анализ корреляций
        correlation_matrix = correlation_chain.calculate_correlation_matrix(sample_market_data)
        chains = correlation_chain.find_correlation_chains(correlation_matrix)
        stability = correlation_chain.analyze_correlation_stability(sample_market_data)
        # Проверяем согласованность результатов
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert isinstance(chains, list)
        assert isinstance(stability, dict)
        # Если есть данные, должны быть результаты
        if not sample_market_data.empty:
            assert not correlation_matrix.empty
            assert len(chains) >= 0  # Может быть 0 цепочек
            assert stability["stability_score"] >= 0.0
    def test_correlation_chain_data_consistency(self, correlation_chain, sample_market_data) -> None:
        """Тест согласованности данных."""
        # Выполняем расчеты несколько раз с одинаковыми данными
        results = []
        for _ in range(3):
            correlation_matrix = correlation_chain.calculate_correlation_matrix(sample_market_data)
            chains = correlation_chain.find_correlation_chains(correlation_matrix)
            results.append((correlation_matrix, chains))
        # Проверяем, что результаты согласованны
        for i in range(1, len(results)):
            # Матрицы корреляций должны быть одинаковыми
            pd.testing.assert_frame_equal(results[i][0], results[0][0])
            # Количество цепочек должно быть одинаковым
            assert len(results[i][1]) == len(results[0][1]) 
