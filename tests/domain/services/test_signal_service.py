"""
Тесты для доменного сервиса сигналов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.signal_service import SignalService, ISignalService
from domain.types.ml_types import SignalResult, SignalStrength, SignalType
import pandas as pd
from shared.numpy_utils import np

class TestSignalService:
    """Тесты для сервиса сигналов."""
    @pytest.fixture
    def signal_service(self) -> Any:
        """Фикстура сервиса сигналов."""
        return SignalService()
    @pytest.fixture
    def sample_market_data(self) -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 10, 100),
            'macd_signal': np.random.normal(0, 10, 100),
            'bollinger_upper': np.random.uniform(51000, 52000, 100),
            'bollinger_lower': np.random.uniform(49000, 50000, 100),
            'sma_20': np.random.uniform(50000, 51000, 100),
            'sma_50': np.random.uniform(50000, 51000, 100)
        }, index=dates)
    @pytest.fixture
    def sample_indicators(self) -> Any:
        """Фикстура с техническими индикаторами."""
        return {
            'rsi': 65.5,
            'macd': 2.5,
            'macd_signal': 1.8,
            'bollinger_position': 0.7,
            'sma_trend': 0.6,
            'volume_ratio': 1.2
        }
    def test_signal_service_initialization(self, signal_service) -> None:
        """Тест инициализации сервиса."""
        assert signal_service is not None
        assert isinstance(signal_service, ISignalService)
        assert hasattr(signal_service, 'config')
        assert isinstance(signal_service.config, dict)
    def test_signal_service_config_defaults(self, signal_service) -> None:
        """Тест конфигурации по умолчанию."""
        config = signal_service.config
        assert "rsi_overbought" in config
        assert "rsi_oversold" in config
        assert "macd_threshold" in config
        assert "volume_threshold" in config
        assert "signal_strength_threshold" in config
        assert isinstance(config["rsi_overbought"], float)
        assert isinstance(config["rsi_oversold"], float)
    def test_generate_signals_valid_data(self, signal_service, sample_market_data) -> None:
        """Тест генерации сигналов с валидными данными."""
        signals = signal_service.generate_signals(sample_market_data)
        assert isinstance(signals, list)
        assert len(signals) > 0
        for signal in signals:
            assert isinstance(signal, dict)
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert "strength" in signal
            assert "direction" in signal
            assert isinstance(signal["timestamp"], pd.Timestamp)
            assert isinstance(signal["signal_type"], str)
            assert isinstance(signal["strength"], float)
            assert isinstance(signal["direction"], str)
            assert signal["strength"] >= 0.0 and signal["strength"] <= 1.0
            assert signal["direction"] in ["buy", "sell", "hold"]
    def test_generate_signals_empty_data(self, signal_service) -> None:
        """Тест генерации сигналов с пустыми данными."""
        empty_data = pd.DataFrame()
        signals = signal_service.generate_signals(empty_data)
        assert isinstance(signals, list)
        assert len(signals) == 0
    def test_generate_signals_single_row(self, signal_service) -> None:
        """Тест генерации сигналов с одной строкой данных."""
        single_row_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 12:00:00')],
            'open': [100.0],
            'high': [110.0],
            'low': [95.0],
            'close': [105.0],
            'volume': [1000.0]
        })
        signals = signal_service.generate_signals(single_row_data)
        assert isinstance(signals, list)
        # Может быть 0 или больше сигналов в зависимости от логики
    def test_filter_signals_by_strength(self, signal_service, sample_market_data) -> None:
        """Тест фильтрации сигналов по силе."""
        all_signals = signal_service.generate_signals(sample_market_data)
        # Фильтруем сигналы с силой больше 0.5
        strong_signals = signal_service.filter_signals_by_strength(all_signals, min_strength=0.5)
        assert isinstance(strong_signals, list)
        for signal in strong_signals:
            assert signal["strength"] >= 0.5
    def test_filter_signals_by_type(self, signal_service, sample_market_data) -> None:
        """Тест фильтрации сигналов по типу."""
        all_signals = signal_service.generate_signals(sample_market_data)
        # Фильтруем только buy сигналы
        buy_signals = signal_service.filter_signals_by_type(all_signals, signal_type="buy")
        assert isinstance(buy_signals, list)
        for signal in buy_signals:
            assert signal["direction"] == "buy"
    def test_aggregate_signals(self, signal_service, sample_market_data) -> None:
        """Тест агрегации сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        aggregated = signal_service.aggregate_signals(signals)
        assert isinstance(aggregated, dict)
        assert "buy_signals" in aggregated
        assert "sell_signals" in aggregated
        assert "hold_signals" in aggregated
        assert "total_signals" in aggregated
        assert "average_strength" in aggregated
        assert isinstance(aggregated["buy_signals"], int)
        assert isinstance(aggregated["sell_signals"], int)
        assert isinstance(aggregated["hold_signals"], int)
        assert isinstance(aggregated["total_signals"], int)
        assert isinstance(aggregated["average_strength"], float)
        assert aggregated["total_signals"] == len(signals)
    def test_calculate_signal_confidence(self, signal_service, sample_market_data) -> None:
        """Тест расчета уверенности сигнала."""
        signals = signal_service.generate_signals(sample_market_data)
        if signals:
            signal = signals[0]
            confidence = signal_service.calculate_signal_confidence(signal)
            assert isinstance(confidence, float)
            assert confidence >= 0.0 and confidence <= 1.0
    def test_validate_signal_consistency(self, signal_service, sample_market_data) -> None:
        """Тест валидации консистентности сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        is_consistent = signal_service.validate_signal_consistency(signals)
        assert isinstance(is_consistent, bool)
    def test_generate_trend_signals(self, signal_service, sample_market_data) -> None:
        """Тест генерации трендовых сигналов."""
        trend_signals = signal_service.generate_trend_signals(sample_market_data)
        assert isinstance(trend_signals, list)
        for signal in trend_signals:
            assert isinstance(signal, dict)
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert signal["signal_type"] == "trend"
    def test_generate_momentum_signals(self, signal_service, sample_market_data) -> None:
        """Тест генерации сигналов импульса."""
        momentum_signals = signal_service.generate_momentum_signals(sample_market_data)
        assert isinstance(momentum_signals, list)
        for signal in momentum_signals:
            assert isinstance(signal, dict)
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert signal["signal_type"] == "momentum"
    def test_generate_volatility_signals(self, signal_service, sample_market_data) -> None:
        """Тест генерации сигналов волатильности."""
        volatility_signals = signal_service.generate_volatility_signals(sample_market_data)
        assert isinstance(volatility_signals, list)
        for signal in volatility_signals:
            assert isinstance(signal, dict)
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert signal["signal_type"] == "volatility"
    def test_generate_volume_signals(self, signal_service, sample_market_data) -> None:
        """Тест генерации сигналов объема."""
        volume_signals = signal_service.generate_volume_signals(sample_market_data)
        assert isinstance(volume_signals, list)
        for signal in volume_signals:
            assert isinstance(signal, dict)
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert signal["signal_type"] == "volume"
    def test_combine_signals(self, signal_service, sample_market_data) -> None:
        """Тест комбинирования сигналов."""
        trend_signals = signal_service.generate_trend_signals(sample_market_data)
        momentum_signals = signal_service.generate_momentum_signals(sample_market_data)
        combined = signal_service.combine_signals([trend_signals, momentum_signals])
        assert isinstance(combined, list)
        assert len(combined) >= max(len(trend_signals), len(momentum_signals))
    def test_signal_service_error_handling(self, signal_service) -> None:
        """Тест обработки ошибок в сервисе."""
        with pytest.raises(Exception):
            signal_service.generate_signals(None)
        with pytest.raises(Exception):
            signal_service.filter_signals_by_strength("invalid_signals", 0.5)
    def test_signal_service_performance(self, signal_service, sample_market_data) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(5):
            signal_service.generate_signals(sample_market_data)
        end_time = time.time()
        assert (end_time - start_time) < 5.0
    def test_signal_service_config_customization(self) -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "signal_types": ["trend", "momentum"],
            "min_strength": 0.3,
            "max_signals": 100,
            "signal_combining": "weighted_average"
        }
        service = SignalService(custom_config)
        assert service.config["signal_types"] == ["trend", "momentum"]
        assert service.config["min_strength"] == 0.3
        assert service.config["max_signals"] == 100
        assert service.config["signal_combining"] == "weighted_average"
    def test_signal_service_signal_validation(self, signal_service, sample_market_data) -> None:
        """Тест валидации сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        if signals:
            signal = signals[0]
            is_valid = signal_service.validate_signal(signal)
            assert isinstance(is_valid, bool)
    def test_signal_service_signal_ranking(self, signal_service, sample_market_data) -> None:
        """Тест ранжирования сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        if len(signals) > 1:
            ranked_signals = signal_service.rank_signals(signals)
            assert isinstance(ranked_signals, list)
            assert len(ranked_signals) == len(signals)
            # Проверяем, что сигналы отсортированы по силе (убывание)
            for i in range(len(ranked_signals) - 1):
                assert ranked_signals[i]["strength"] >= ranked_signals[i + 1]["strength"]
    def test_signal_service_signal_clustering(self, signal_service, sample_market_data) -> None:
        """Тест кластеризации сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        if len(signals) > 1:
            clusters = signal_service.cluster_signals(signals)
            assert isinstance(clusters, list)
            assert len(clusters) > 0
            for cluster in clusters:
                assert isinstance(cluster, list)
                assert len(cluster) > 0
    def test_signal_service_signal_correlation(self, signal_service, sample_market_data) -> None:
        """Тест корреляции сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        if len(signals) > 1:
            correlation_matrix = signal_service.calculate_signal_correlation(signals)
            assert isinstance(correlation_matrix, dict)
            assert "correlation_matrix" in correlation_matrix
            assert "correlation_stats" in correlation_matrix
    def test_signal_service_signal_forecasting(self, signal_service, sample_market_data) -> None:
        """Тест прогнозирования сигналов."""
        signals = signal_service.generate_signals(sample_market_data)
        if len(signals) > 5:
            forecast = signal_service.forecast_signals(signals, horizon=5)
            assert isinstance(forecast, dict)
            assert "forecasted_signals" in forecast
            assert "confidence_intervals" in forecast
            assert "forecast_accuracy" in forecast 
