"""
Тесты для доменного сервиса анализа спредов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.spread_analyzer import SpreadAnalyzer, ISpreadAnalyzer
from domain.types.ml_types import SpreadAnalysisResult, SpreadMovementPrediction
import pandas as pd
import numpy as np

class TestSpreadAnalyzer:
    """Тесты для сервиса анализа спредов."""
    @pytest.fixture
    def spread_analyzer(self) -> Any:
        """Фикстура сервиса анализа спредов."""
        return SpreadAnalyzer()
    @pytest.fixture
    def sample_order_book(self) -> Any:
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
            "timestamp": "2024-01-01T12:00:00Z"
        }
    @pytest.fixture
    def sample_historical_data(self) -> Any:
        """Фикстура с историческими данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'spread': np.random.uniform(0.1, 2.0, 100)
        }, index=dates)
    def test_spread_analyzer_initialization(self, spread_analyzer) -> None:
        """Тест инициализации сервиса."""
        assert spread_analyzer is not None
        assert isinstance(spread_analyzer, ISpreadAnalyzer)
        assert hasattr(spread_analyzer, 'config')
        assert isinstance(spread_analyzer.config, dict)
    def test_spread_analyzer_config_defaults(self, spread_analyzer) -> None:
        """Тест конфигурации по умолчанию."""
        config = spread_analyzer.config
        assert "spread_threshold" in config
        assert "imbalance_threshold" in config
        assert "volume_threshold" in config
        assert "lookback_period" in config
        assert isinstance(config["spread_threshold"], float)
        assert isinstance(config["imbalance_threshold"], float)
    def test_analyze_spread_valid_order_book(self, spread_analyzer, sample_order_book) -> None:
        """Тест анализа спреда с валидным ордербуком."""
        result = spread_analyzer.analyze_spread(sample_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert "spread" in result
        assert "imbalance" in result
        assert "confidence" in result
        assert "best_bid" in result
        assert "best_ask" in result
        assert isinstance(result["spread"], float)
        assert isinstance(result["imbalance"], float)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["best_bid"], float)
        assert isinstance(result["best_ask"], float)
        # Проверяем логику
        assert result["best_bid"] == 50000.0
        assert result["best_ask"] == 50001.0
        assert result["spread"] == 1.0
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
    def test_analyze_spread_empty_order_book(self, spread_analyzer) -> None:
        """Тест анализа спреда с пустым ордербуком."""
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        result = spread_analyzer.analyze_spread(empty_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["spread"] == 0.0
        assert result["imbalance"] == 0.0
        assert result["confidence"] == 0.0
        assert result["best_bid"] == 0.0
        assert result["best_ask"] == 0.0
    def test_analyze_spread_missing_bids(self, spread_analyzer) -> None:
        """Тест анализа спреда без бидов."""
        order_book_no_bids = {
            "bids": [],
            "asks": [{"price": "50001.0", "quantity": "1.0"}],
            "timestamp": "2024-01-01T12:00:00Z"
        }
        result = spread_analyzer.analyze_spread(order_book_no_bids)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["best_bid"] == 0.0
        assert result["best_ask"] == 50001.0
        assert result["spread"] == 0.0
    def test_analyze_spread_missing_asks(self, spread_analyzer) -> None:
        """Тест анализа спреда без асков."""
        order_book_no_asks = {
            "bids": [{"price": "50000.0", "quantity": "1.0"}],
            "asks": [],
            "timestamp": "2024-01-01T12:00:00Z"
        }
        result = spread_analyzer.analyze_spread(order_book_no_asks)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["best_bid"] == 50000.0
        assert result["best_ask"] == 0.0
        assert result["spread"] == 0.0
    def test_analyze_spread_invalid_price_format(self, spread_analyzer) -> None:
        """Тест анализа спреда с невалидным форматом цен."""
        invalid_order_book = {
            "bids": [{"price": "invalid", "quantity": "1.0"}],
            "asks": [{"price": "50001.0", "quantity": "1.0"}],
            "timestamp": "2024-01-01T12:00:00Z"
        }
        result = spread_analyzer.analyze_spread(invalid_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        # Должен обработать ошибку и вернуть безопасные значения
        assert result["best_bid"] == 0.0 or result["best_bid"] == 50001.0
    def test_calculate_spread_imbalance(self, spread_analyzer, sample_order_book) -> None:
        """Тест расчета дисбаланса спреда."""
        result = spread_analyzer.analyze_spread(sample_order_book)
        # Проверяем, что дисбаланс рассчитывается корректно
        assert isinstance(result["imbalance"], float)
        assert result["imbalance"] >= -1.0 and result["imbalance"] <= 1.0
    def test_predict_spread_movement_valid_data(self, spread_analyzer, sample_historical_data) -> None:
        """Тест предсказания движения спреда с валидными данными."""
        result = spread_analyzer.predict_spread_movement(sample_historical_data)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert "direction" in result
        assert "confidence" in result
        assert "predicted_spread" in result
        assert "trend_strength" in result
        assert isinstance(result["direction"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["predicted_spread"], float)
        assert isinstance(result["trend_strength"], float)
        assert result["direction"] in ["increase", "decrease", "stable"]
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
        assert result["predicted_spread"] >= 0.0
    def test_predict_spread_movement_empty_data(self, spread_analyzer) -> None:
        """Тест предсказания движения спреда с пустыми данными."""
        empty_data = pd.DataFrame()
        result = spread_analyzer.predict_spread_movement(empty_data)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["direction"] == "stable"
        assert result["confidence"] == 0.0
        assert result["predicted_spread"] == 0.0
        assert result["trend_strength"] == 0.0
    def test_predict_spread_movement_insufficient_data(self, spread_analyzer) -> None:
        """Тест предсказания движения спреда с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'spread': [1.0, 1.1]  # Только 2 точки данных
        })
        result = spread_analyzer.predict_spread_movement(insufficient_data)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["confidence"] < 0.5  # Низкая уверенность при недостатке данных
    def test_predict_spread_movement_missing_spread_column(self, spread_analyzer) -> None:
        """Тест предсказания движения спреда без колонки spread."""
        data_without_spread = pd.DataFrame({
            'open': [50000, 50001, 50002],
            'close': [50001, 50002, 50003]
        })
        result = spread_analyzer.predict_spread_movement(data_without_spread)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        # Должен обработать отсутствие колонки spread
        assert result["direction"] == "stable"
    def test_get_spread_statistics(self, spread_analyzer, sample_historical_data) -> None:
        """Тест получения статистики спреда."""
        stats = spread_analyzer.get_spread_statistics(sample_historical_data)
        assert isinstance(stats, dict)
        assert "mean_spread" in stats
        assert "std_spread" in stats
        assert "min_spread" in stats
        assert "max_spread" in stats
        assert "spread_volatility" in stats
        assert isinstance(stats["mean_spread"], float)
        assert isinstance(stats["std_spread"], float)
        assert isinstance(stats["min_spread"], float)
        assert isinstance(stats["max_spread"], float)
        assert isinstance(stats["spread_volatility"], float)
        # Проверяем логику статистики
        assert stats["min_spread"] <= stats["mean_spread"] <= stats["max_spread"]
        assert stats["std_spread"] >= 0.0
        assert stats["spread_volatility"] >= 0.0
    def test_get_spread_statistics_empty_data(self, spread_analyzer) -> None:
        """Тест получения статистики спреда с пустыми данными."""
        empty_data = pd.DataFrame()
        stats = spread_analyzer.get_spread_statistics(empty_data)
        assert isinstance(stats, dict)
        assert stats["mean_spread"] == 0.0
        assert stats["std_spread"] == 0.0
        assert stats["min_spread"] == 0.0
        assert stats["max_spread"] == 0.0
        assert stats["spread_volatility"] == 0.0
    def test_detect_spread_anomalies(self, spread_analyzer, sample_historical_data) -> None:
        """Тест обнаружения аномалий спреда."""
        anomalies = spread_analyzer.detect_spread_anomalies(sample_historical_data)
        assert isinstance(anomalies, list)
        # Проверяем, что все аномалии имеют правильную структуру
        for anomaly in anomalies:
            assert isinstance(anomaly, dict)
            assert "timestamp" in anomaly
            assert "spread_value" in anomaly
            assert "anomaly_score" in anomaly
            assert "anomaly_type" in anomaly
            assert isinstance(anomaly["spread_value"], float)
            assert isinstance(anomaly["anomaly_score"], float)
            assert isinstance(anomaly["anomaly_type"], str)
            assert anomaly["anomaly_score"] >= 0.0
    def test_detect_spread_anomalies_empty_data(self, spread_analyzer) -> None:
        """Тест обнаружения аномалий спреда с пустыми данными."""
        empty_data = pd.DataFrame()
        anomalies = spread_analyzer.detect_spread_anomalies(empty_data)
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0
    def test_calculate_spread_correlation(self, spread_analyzer, sample_historical_data) -> None:
        """Тест расчета корреляции спреда."""
        # Создаем данные с двумя активами
        data_asset1 = sample_historical_data.copy()
        data_asset1['spread'] = np.random.uniform(0.1, 2.0, len(data_asset1))
        data_asset2 = sample_historical_data.copy()
        data_asset2['spread'] = np.random.uniform(0.1, 2.0, len(data_asset2))
        correlation = spread_analyzer.calculate_spread_correlation(
            data_asset1, data_asset2
        )
        assert isinstance(correlation, float)
        assert correlation >= -1.0 and correlation <= 1.0
    def test_calculate_spread_correlation_insufficient_data(self, spread_analyzer) -> None:
        """Тест расчета корреляции спреда с недостаточными данными."""
        insufficient_data1 = pd.DataFrame({'spread': [1.0]})
        insufficient_data2 = pd.DataFrame({'spread': [1.1]})
        correlation = spread_analyzer.calculate_spread_correlation(
            insufficient_data1, insufficient_data2
        )
        assert isinstance(correlation, float)
        assert correlation == 0.0  # Должен вернуть 0 при недостатке данных
    def test_get_spread_forecast(self, spread_analyzer, sample_historical_data) -> None:
        """Тест получения прогноза спреда."""
        forecast = spread_analyzer.get_spread_forecast(sample_historical_data, periods=5)
        assert isinstance(forecast, dict)
        assert "forecast_values" in forecast
        assert "confidence_intervals" in forecast
        assert "forecast_horizon" in forecast
        assert isinstance(forecast["forecast_values"], list)
        assert isinstance(forecast["confidence_intervals"], list)
        assert isinstance(forecast["forecast_horizon"], int)
        assert len(forecast["forecast_values"]) == 5
        assert len(forecast["confidence_intervals"]) == 5
        assert forecast["forecast_horizon"] == 5
    def test_get_spread_forecast_empty_data(self, spread_analyzer) -> None:
        """Тест получения прогноза спреда с пустыми данными."""
        empty_data = pd.DataFrame()
        forecast = spread_analyzer.get_spread_forecast(empty_data, periods=3)
        assert isinstance(forecast, dict)
        assert forecast["forecast_values"] == [0.0, 0.0, 0.0]
        assert forecast["confidence_intervals"] == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    def test_analyze_spread_impact_on_volume(self, spread_analyzer, sample_historical_data) -> None:
        """Тест анализа влияния спреда на объем."""
        impact = spread_analyzer.analyze_spread_impact_on_volume(sample_historical_data)
        assert isinstance(impact, dict)
        assert "correlation" in impact
        assert "impact_strength" in impact
        assert "volume_sensitivity" in impact
        assert isinstance(impact["correlation"], float)
        assert isinstance(impact["impact_strength"], float)
        assert isinstance(impact["volume_sensitivity"], float)
        assert impact["correlation"] >= -1.0 and impact["correlation"] <= 1.0
        assert impact["impact_strength"] >= 0.0
    def test_analyze_spread_impact_on_volume_empty_data(self, spread_analyzer) -> None:
        """Тест анализа влияния спреда на объем с пустыми данными."""
        empty_data = pd.DataFrame()
        impact = spread_analyzer.analyze_spread_impact_on_volume(empty_data)
        assert isinstance(impact, dict)
        assert impact["correlation"] == 0.0
        assert impact["impact_strength"] == 0.0
        assert impact["volume_sensitivity"] == 0.0
    def test_get_optimal_spread_levels(self, spread_analyzer, sample_historical_data) -> None:
        """Тест получения оптимальных уровней спреда."""
        levels = spread_analyzer.get_optimal_spread_levels(sample_historical_data)
        assert isinstance(levels, dict)
        assert "min_spread" in levels
        assert "max_spread" in levels
        assert "optimal_spread" in levels
        assert "spread_range" in levels
        assert isinstance(levels["min_spread"], float)
        assert isinstance(levels["max_spread"], float)
        assert isinstance(levels["optimal_spread"], float)
        assert isinstance(levels["spread_range"], float)
        assert levels["min_spread"] <= levels["optimal_spread"] <= levels["max_spread"]
        assert levels["spread_range"] >= 0.0
    def test_get_optimal_spread_levels_empty_data(self, spread_analyzer) -> None:
        """Тест получения оптимальных уровней спреда с пустыми данными."""
        empty_data = pd.DataFrame()
        levels = spread_analyzer.get_optimal_spread_levels(empty_data)
        assert isinstance(levels, dict)
        assert levels["min_spread"] == 0.0
        assert levels["max_spread"] == 0.0
        assert levels["optimal_spread"] == 0.0
        assert levels["spread_range"] == 0.0
    def test_spread_analyzer_error_handling(self, spread_analyzer) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными
        with pytest.raises(Exception):
            spread_analyzer.analyze_spread(None)
        # Тест с невалидным типом данных
        with pytest.raises(Exception):
            spread_analyzer.predict_spread_movement("invalid_data")
        # Тест с невалидным ордербуком
        invalid_order_book = {"invalid": "data"}
        result = spread_analyzer.analyze_spread(invalid_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        # Должен вернуть безопасные значения
    def test_spread_analyzer_performance(self, spread_analyzer, sample_order_book) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(100):
            spread_analyzer.analyze_spread(sample_order_book)
        end_time = time.time()
        # Проверяем, что 100 операций выполняются менее чем за 1 секунду
        assert (end_time - start_time) < 1.0
    def test_spread_analyzer_thread_safety(self, spread_analyzer, sample_order_book) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue
        results = queue.Queue()
        def analyze_spread() -> Any:
            try:
                result = spread_analyzer.analyze_spread(sample_order_book)
                results.put(result)
            except Exception as e:
                results.put(e)
        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=analyze_spread)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(10):
            result = results.get()
            # Проверяем структуру результата вместо isinstance
            assert isinstance(result, dict)
            assert "spread" in result 
