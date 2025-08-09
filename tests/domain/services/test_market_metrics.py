"""
Тесты для доменного сервиса рыночных метрик.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.market_metrics import MarketMetrics, IMarketMetrics


class TestMarketMetrics:
    """Тесты для сервиса рыночных метрик."""

    @pytest.fixture
    def market_metrics(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура сервиса рыночных метрик."""
        return MarketMetrics()

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "open": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(51000, 52000, 100),
                "low": np.random.uniform(49000, 50000, 100),
                "close": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(1000, 5000, 100),
                "vwap": np.random.uniform(50000, 51000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_order_book(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерным ордербуком."""
        return {
            "bids": [
                {"price": "50000.0", "quantity": "1.5"},
                {"price": "49999.0", "quantity": "2.0"},
                {"price": "49998.0", "quantity": "1.0"},
            ],
            "asks": [
                {"price": "50001.0", "quantity": "1.0"},
                {"price": "50002.0", "quantity": "2.5"},
                {"price": "50003.0", "quantity": "1.5"},
            ],
            "timestamp": "2024-01-01T12:00:00Z",
        }

    def test_market_metrics_initialization(self, market_metrics) -> None:
        """Тест инициализации сервиса."""
        assert market_metrics is not None
        assert isinstance(market_metrics, IMarketMetrics)
        assert hasattr(market_metrics, "config")
        assert isinstance(market_metrics.config, dict)

    def test_market_metrics_config_defaults(self, market_metrics) -> None:
        """Тест конфигурации по умолчанию."""
        config = market_metrics.config
        assert "volatility_window" in config
        assert "trend_window" in config
        assert "volume_window" in config
        assert "correlation_window" in config
        assert isinstance(config["volatility_window"], int)
        assert isinstance(config["trend_window"], int)

    def test_calculate_volatility_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик волатильности."""
        volatility = market_metrics.calculate_volatility_metrics(sample_market_data)
        assert isinstance(volatility, dict)
        assert "current_volatility" in volatility
        assert "historical_volatility" in volatility
        assert "volatility_percentile" in volatility
        assert "volatility_trend" in volatility
        assert isinstance(volatility["current_volatility"], float)
        assert isinstance(volatility["historical_volatility"], float)
        assert isinstance(volatility["volatility_percentile"], float)
        assert isinstance(volatility["volatility_trend"], str)
        assert volatility["current_volatility"] >= 0.0
        assert volatility["historical_volatility"] >= 0.0
        assert volatility["volatility_percentile"] >= 0.0 and volatility["volatility_percentile"] <= 1.0
        assert volatility["volatility_trend"] in ["increasing", "decreasing", "stable"]

    def test_calculate_volatility_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик волатильности с пустыми данными."""
        empty_data = pd.DataFrame()
        volatility = market_metrics.calculate_volatility_metrics(empty_data)
        assert isinstance(volatility, dict)
        assert volatility["current_volatility"] == 0.0
        assert volatility["historical_volatility"] == 0.0
        assert volatility["volatility_percentile"] == 0.0
        assert volatility["volatility_trend"] == "stable"

    def test_calculate_trend_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик тренда."""
        trend = market_metrics.calculate_trend_metrics(sample_market_data)
        assert isinstance(trend, dict)
        assert "trend_direction" in trend
        assert "trend_strength" in trend
        assert "trend_duration" in trend
        assert "support_resistance" in trend
        assert isinstance(trend["trend_direction"], str)
        assert isinstance(trend["trend_strength"], float)
        assert isinstance(trend["trend_duration"], int)
        assert isinstance(trend["support_resistance"], dict)
        assert trend["trend_direction"] in ["uptrend", "downtrend", "sideways"]
        assert trend["trend_strength"] >= 0.0 and trend["trend_strength"] <= 1.0
        assert trend["trend_duration"] >= 0

    def test_calculate_trend_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик тренда с пустыми данными."""
        empty_data = pd.DataFrame()
        trend = market_metrics.calculate_trend_metrics(empty_data)
        assert isinstance(trend, dict)
        assert trend["trend_direction"] == "sideways"
        assert trend["trend_strength"] == 0.0
        assert trend["trend_duration"] == 0

    def test_calculate_volume_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик объема."""
        volume_metrics = market_metrics.calculate_volume_metrics(sample_market_data)
        assert isinstance(volume_metrics, dict)
        assert "volume_ma" in volume_metrics
        assert "volume_ratio" in volume_metrics
        assert "volume_trend" in volume_metrics
        assert "volume_profile" in volume_metrics
        assert isinstance(volume_metrics["volume_ma"], float)
        assert isinstance(volume_metrics["volume_ratio"], float)
        assert isinstance(volume_metrics["volume_trend"], str)
        assert isinstance(volume_metrics["volume_profile"], dict)
        assert volume_metrics["volume_ma"] >= 0.0
        assert volume_metrics["volume_ratio"] >= 0.0
        assert volume_metrics["volume_trend"] in ["increasing", "decreasing", "stable"]

    def test_calculate_volume_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик объема с пустыми данными."""
        empty_data = pd.DataFrame()
        volume_metrics = market_metrics.calculate_volume_metrics(empty_data)
        assert isinstance(volume_metrics, dict)
        assert volume_metrics["volume_ma"] == 0.0
        assert volume_metrics["volume_ratio"] == 0.0
        assert volume_metrics["volume_trend"] == "stable"

    def test_calculate_correlation_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик корреляции."""
        # Создаем данные для двух активов
        asset1_data = sample_market_data.copy()
        asset1_data["close"] = np.random.uniform(50000, 51000, 100)
        asset2_data = sample_market_data.copy()
        asset2_data["close"] = np.random.uniform(50000, 51000, 100)
        correlation = market_metrics.calculate_correlation_metrics(asset1_data, asset2_data)
        assert isinstance(correlation, dict)
        assert "correlation_coefficient" in correlation
        assert "correlation_strength" in correlation
        assert "correlation_trend" in correlation
        assert isinstance(correlation["correlation_coefficient"], float)
        assert isinstance(correlation["correlation_strength"], str)
        assert isinstance(correlation["correlation_trend"], str)
        assert correlation["correlation_coefficient"] >= -1.0 and correlation["correlation_coefficient"] <= 1.0
        assert correlation["correlation_strength"] in ["strong", "moderate", "weak"]
        assert correlation["correlation_trend"] in ["increasing", "decreasing", "stable"]

    def test_calculate_correlation_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик корреляции с пустыми данными."""
        empty_data = pd.DataFrame()
        correlation = market_metrics.calculate_correlation_metrics(empty_data, empty_data)
        assert isinstance(correlation, dict)
        assert correlation["correlation_coefficient"] == 0.0
        assert correlation["correlation_strength"] == "weak"
        assert correlation["correlation_trend"] == "stable"

    def test_calculate_market_efficiency_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик эффективности рынка."""
        efficiency = market_metrics.calculate_market_efficiency_metrics(sample_market_data)
        assert isinstance(efficiency, dict)
        assert "efficiency_ratio" in efficiency
        assert "hurst_exponent" in efficiency
        assert "market_regime" in efficiency
        assert isinstance(efficiency["efficiency_ratio"], float)
        assert isinstance(efficiency["hurst_exponent"], float)
        assert isinstance(efficiency["market_regime"], str)
        assert efficiency["efficiency_ratio"] >= 0.0 and efficiency["efficiency_ratio"] <= 1.0
        assert efficiency["hurst_exponent"] >= 0.0
        assert efficiency["market_regime"] in ["efficient", "inefficient", "trending", "mean_reverting"]

    def test_calculate_market_efficiency_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик эффективности рынка с пустыми данными."""
        empty_data = pd.DataFrame()
        efficiency = market_metrics.calculate_market_efficiency_metrics(empty_data)
        assert isinstance(efficiency, dict)
        assert efficiency["efficiency_ratio"] == 0.0
        assert efficiency["hurst_exponent"] == 0.0
        assert efficiency["market_regime"] == "efficient"

    def test_calculate_liquidity_metrics(self, market_metrics, sample_market_data, sample_order_book) -> None:
        """Тест расчета метрик ликвидности."""
        liquidity = market_metrics.calculate_liquidity_metrics(sample_market_data, sample_order_book)
        assert isinstance(liquidity, dict)
        assert "bid_ask_spread" in liquidity
        assert "order_book_depth" in liquidity
        assert "volume_liquidity" in liquidity
        assert "liquidity_score" in liquidity
        assert isinstance(liquidity["bid_ask_spread"], float)
        assert isinstance(liquidity["order_book_depth"], float)
        assert isinstance(liquidity["volume_liquidity"], float)
        assert isinstance(liquidity["liquidity_score"], float)
        assert liquidity["bid_ask_spread"] >= 0.0
        assert liquidity["order_book_depth"] >= 0.0
        assert liquidity["volume_liquidity"] >= 0.0
        assert liquidity["liquidity_score"] >= 0.0 and liquidity["liquidity_score"] <= 1.0

    def test_calculate_liquidity_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик ликвидности с пустыми данными."""
        empty_market_data = pd.DataFrame()
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        liquidity = market_metrics.calculate_liquidity_metrics(empty_market_data, empty_order_book)
        assert isinstance(liquidity, dict)
        assert liquidity["bid_ask_spread"] == 0.0
        assert liquidity["order_book_depth"] == 0.0
        assert liquidity["volume_liquidity"] == 0.0
        assert liquidity["liquidity_score"] == 0.0

    def test_calculate_momentum_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик моментума."""
        momentum = market_metrics.calculate_momentum_metrics(sample_market_data)
        assert isinstance(momentum, dict)
        assert "price_momentum" in momentum
        assert "volume_momentum" in momentum
        assert "momentum_strength" in momentum
        assert "momentum_divergence" in momentum
        assert isinstance(momentum["price_momentum"], float)
        assert isinstance(momentum["volume_momentum"], float)
        assert isinstance(momentum["momentum_strength"], float)
        assert isinstance(momentum["momentum_divergence"], bool)
        assert momentum["momentum_strength"] >= 0.0 and momentum["momentum_strength"] <= 1.0

    def test_calculate_momentum_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик моментума с пустыми данными."""
        empty_data = pd.DataFrame()
        momentum = market_metrics.calculate_momentum_metrics(empty_data)
        assert isinstance(momentum, dict)
        assert momentum["price_momentum"] == 0.0
        assert momentum["volume_momentum"] == 0.0
        assert momentum["momentum_strength"] == 0.0
        assert momentum["momentum_divergence"] == False

    def test_calculate_market_stress_metrics(self, market_metrics, sample_market_data) -> None:
        """Тест расчета метрик рыночного стресса."""
        stress = market_metrics.calculate_market_stress_metrics(sample_market_data)
        assert isinstance(stress, dict)
        assert "stress_level" in stress
        assert "stress_indicators" in stress
        assert "stress_trend" in stress
        assert isinstance(stress["stress_level"], float)
        assert isinstance(stress["stress_indicators"], list)
        assert isinstance(stress["stress_trend"], str)
        assert stress["stress_level"] >= 0.0 and stress["stress_level"] <= 1.0
        assert stress["stress_trend"] in ["increasing", "decreasing", "stable"]

    def test_calculate_market_stress_metrics_empty_data(self, market_metrics) -> None:
        """Тест расчета метрик рыночного стресса с пустыми данными."""
        empty_data = pd.DataFrame()
        stress = market_metrics.calculate_market_stress_metrics(empty_data)
        assert isinstance(stress, dict)
        assert stress["stress_level"] == 0.0
        assert len(stress["stress_indicators"]) == 0
        assert stress["stress_trend"] == "stable"

    def test_get_comprehensive_metrics(self, market_metrics, sample_market_data, sample_order_book) -> None:
        """Тест получения комплексных метрик."""
        metrics = market_metrics.get_comprehensive_metrics(sample_market_data, sample_order_book)
        assert isinstance(metrics, dict)
        assert "volatility" in metrics
        assert "trend" in metrics
        assert "volume" in metrics
        assert "liquidity" in metrics
        assert "momentum" in metrics
        assert "stress" in metrics
        assert "timestamp" in metrics
        assert isinstance(metrics["volatility"], dict)
        assert isinstance(metrics["trend"], dict)
        assert isinstance(metrics["volume"], dict)
        assert isinstance(metrics["liquidity"], dict)
        assert isinstance(metrics["momentum"], dict)
        assert isinstance(metrics["stress"], dict)

    def test_get_comprehensive_metrics_empty_data(self, market_metrics) -> None:
        """Тест получения комплексных метрик с пустыми данными."""
        empty_market_data = pd.DataFrame()
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        metrics = market_metrics.get_comprehensive_metrics(empty_market_data, empty_order_book)
        assert isinstance(metrics, dict)
        assert "volatility" in metrics
        assert "trend" in metrics
        assert "volume" in metrics
        assert "liquidity" in metrics
        assert "momentum" in metrics
        assert "stress" in metrics

    def test_market_metrics_error_handling(self, market_metrics) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными - сервис должен корректно обработать
        volatility = market_metrics.calculate_volatility_metrics(None)
        assert isinstance(volatility, dict)
        assert volatility["current_volatility"] == 0.0
        # Тест с невалидным типом данных - сервис должен корректно обработать
        trend = market_metrics.calculate_trend_metrics("invalid_data")
        assert isinstance(trend, dict)
        assert trend["trend_direction"] == "sideways"
        # Тест с невалидным ордербуком - сервис должен корректно обработать
        liquidity = market_metrics.calculate_liquidity_metrics(pd.DataFrame(), "invalid_order_book")
        assert isinstance(liquidity, dict)
        assert liquidity["bid_ask_spread"] == 0.0

    def test_market_metrics_performance(self, market_metrics, sample_market_data, sample_order_book) -> None:
        """Тест производительности сервиса."""
        import time

        start_time = time.time()
        for _ in range(10):
            market_metrics.get_comprehensive_metrics(sample_market_data, sample_order_book)
        end_time = time.time()
        # Проверяем, что 10 операций выполняются менее чем за 2 секунды
        assert (end_time - start_time) < 2.0

    def test_market_metrics_thread_safety(self, market_metrics, sample_market_data, sample_order_book) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue

        results = queue.Queue()

        def calculate_metrics() -> Any:
            try:
                result = market_metrics.get_comprehensive_metrics(sample_market_data, sample_order_book)
                results.put(result)
            except Exception as e:
                results.put(e)

        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=calculate_metrics)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(5):
            result = results.get()
            assert isinstance(result, dict)
            assert "volatility" in result

    def test_market_metrics_config_customization(self: "TestMarketMetrics") -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {"volatility_window": 30, "trend_window": 50, "volume_window": 20, "correlation_window": 60}
        service = MarketMetrics(custom_config)
        assert service.config["volatility_window"] == 30
        assert service.config["trend_window"] == 50
        assert service.config["volume_window"] == 20
        assert service.config["correlation_window"] == 60

    def test_market_metrics_integration_with_different_data_types(self, market_metrics) -> None:
        """Тест интеграции с различными типами данных."""
        # Тест с данными разных временных интервалов
        hourly_data = pd.DataFrame(
            {"close": np.random.uniform(50000, 51000, 100), "volume": np.random.uniform(1000, 5000, 100)}
        )
        daily_data = pd.DataFrame(
            {"close": np.random.uniform(50000, 51000, 30), "volume": np.random.uniform(1000, 5000, 30)}
        )
        # Проверяем, что сервис работает с разными типами данных
        hourly_metrics = market_metrics.calculate_volatility_metrics(hourly_data)
        daily_metrics = market_metrics.calculate_volatility_metrics(daily_data)
        assert isinstance(hourly_metrics, dict)
        assert isinstance(daily_metrics, dict)
        # Проверяем, что метрики рассчитываются корректно
        assert hourly_metrics["current_volatility"] >= 0.0
        assert daily_metrics["current_volatility"] >= 0.0

    def test_market_metrics_consistency(self, market_metrics, sample_market_data) -> None:
        """Тест согласованности метрик."""
        # Проверяем, что метрики согласованы между собой
        volatility = market_metrics.calculate_volatility_metrics(sample_market_data)
        trend = market_metrics.calculate_trend_metrics(sample_market_data)
        momentum = market_metrics.calculate_momentum_metrics(sample_market_data)
        # Проверяем, что все метрики возвращают корректные типы данных
        assert isinstance(volatility, dict)
        assert isinstance(trend, dict)
        assert isinstance(momentum, dict)
        # Проверяем, что значения находятся в ожидаемых диапазонах
        assert 0.0 <= volatility["current_volatility"] <= 1.0
        assert 0.0 <= trend["trend_strength"] <= 1.0
        assert 0.0 <= momentum["momentum_strength"] <= 1.0
