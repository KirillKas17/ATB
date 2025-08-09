"""
Unit тесты для SignalProcessor.
Тестирует обработку сигналов, включая генерацию, фильтрацию,
агрегацию и анализ торговых сигналов.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from infrastructure.core.signal_processor import SignalProcessor

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


class TestSignalProcessor:
    """Тесты для SignalProcessor."""

    @pytest.fixture
    def signal_processor(self) -> SignalProcessor:
        """Фикстура для SignalProcessor."""
        return SignalProcessor()

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми рыночными данными."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="1H")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 1000),
                "high": np.random.uniform(46000, 56000, 1000),
                "low": np.random.uniform(44000, 54000, 1000),
                "close": np.random.uniform(45000, 55000, 1000),
                "volume": np.random.uniform(1000000, 5000000, 1000),
            },
            index=dates,
        )
        # Создание более реалистичных данных
        data["high"] = data[["open", "close"]].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data["low"] = data[["open", "close"]].min(axis=1) - np.random.uniform(0, 1000, 1000)
        return data

    @pytest.fixture
    def sample_signals(self) -> list:
        """Фикстура с тестовыми сигналами."""
        return [
            {
                "id": "signal_001",
                "symbol": "BTCUSDT",
                "type": "buy",
                "strength": 0.8,
                "source": "rsi",
                "timestamp": datetime.now() - timedelta(hours=1),
                "price": Decimal("50000.0"),
                "confidence": 0.75,
                "metadata": {"rsi_value": 25, "oversold": True},
            },
            {
                "id": "signal_002",
                "symbol": "ETHUSDT",
                "type": "sell",
                "strength": 0.6,
                "source": "macd",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "price": Decimal("3000.0"),
                "confidence": 0.65,
                "metadata": {"macd_signal": "bearish", "divergence": True},
            },
        ]

    def test_initialization(self, signal_processor: SignalProcessor) -> None:
        """Тест инициализации процессора сигналов."""
        assert signal_processor is not None
        assert hasattr(signal_processor, "signal_generators")
        assert hasattr(signal_processor, "signal_filters")
        assert hasattr(signal_processor, "signal_aggregators")

    def test_generate_rsi_signals(self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест генерации RSI сигналов."""
        # Генерация RSI сигналов
        rsi_signals = signal_processor.generate_rsi_signals(sample_market_data, period=14)
        # Проверки
        assert rsi_signals is not None
        assert isinstance(rsi_signals, list)
        assert len(rsi_signals) >= 0
        # Проверка структуры сигналов
        for signal in rsi_signals:
            assert "type" in signal
            assert "strength" in signal
            assert "timestamp" in signal
            assert "price" in signal
            assert "confidence" in signal
            # Проверка типов данных
            assert signal["type"] in ["buy", "sell", "hold"]
            assert isinstance(signal["strength"], float)
            assert isinstance(signal["timestamp"], datetime)
            assert isinstance(signal["price"], Decimal)
            assert isinstance(signal["confidence"], float)
            # Проверка диапазонов
            assert 0.0 <= signal["strength"] <= 1.0
            assert 0.0 <= signal["confidence"] <= 1.0

    def test_generate_macd_signals(self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест генерации MACD сигналов."""
        # Генерация MACD сигналов
        macd_signals = signal_processor.generate_macd_signals(sample_market_data)
        # Проверки
        assert macd_signals is not None
        assert isinstance(macd_signals, list)
        assert len(macd_signals) >= 0
        # Проверка структуры сигналов
        for signal in macd_signals:
            assert "type" in signal
            assert "strength" in signal
            assert "timestamp" in signal
            assert "price" in signal
            assert "confidence" in signal

    def test_generate_moving_average_signals(
        self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест генерации сигналов скользящих средних."""
        # Генерация сигналов скользящих средних
        ma_signals = signal_processor.generate_moving_average_signals(
            sample_market_data, short_period=10, long_period=50
        )
        # Проверки
        assert ma_signals is not None
        assert isinstance(ma_signals, list)
        assert len(ma_signals) >= 0
        # Проверка структуры сигналов
        for signal in ma_signals:
            assert "type" in signal
            assert "strength" in signal
            assert "timestamp" in signal
            assert "price" in signal
            assert "confidence" in signal

    def test_generate_bollinger_bands_signals(
        self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест генерации сигналов полос Боллинджера."""
        # Генерация сигналов полос Боллинджера
        bb_signals = signal_processor.generate_bollinger_bands_signals(sample_market_data)
        # Проверки
        assert bb_signals is not None
        assert isinstance(bb_signals, list)
        assert len(bb_signals) >= 0
        # Проверка структуры сигналов
        for signal in bb_signals:
            assert "type" in signal
            assert "strength" in signal
            assert "timestamp" in signal
            assert "price" in signal
            assert "confidence" in signal

    def test_generate_volume_signals(self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест генерации объемных сигналов."""
        # Генерация объемных сигналов
        volume_signals = signal_processor.generate_volume_signals(sample_market_data)
        # Проверки
        assert volume_signals is not None
        assert isinstance(volume_signals, list)
        assert len(volume_signals) >= 0
        # Проверка структуры сигналов
        for signal in volume_signals:
            assert "type" in signal
            assert "strength" in signal
            assert "timestamp" in signal
            assert "price" in signal
            assert "confidence" in signal

    def test_filter_signals(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест фильтрации сигналов."""
        # Фильтрация сигналов
        filtered_signals = signal_processor.filter_signals(
            sample_signals, filters={"min_strength": 0.5, "min_confidence": 0.6, "signal_types": ["buy", "sell"]}
        )
        # Проверки
        assert filtered_signals is not None
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(sample_signals)
        # Проверка фильтрации
        for signal in filtered_signals:
            assert signal["strength"] >= 0.5
            assert signal["confidence"] >= 0.6
            assert signal["type"] in ["buy", "sell"]

    def test_aggregate_signals(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест агрегации сигналов."""
        # Агрегация сигналов
        aggregated_signals = signal_processor.aggregate_signals(sample_signals)
        # Проверки
        assert aggregated_signals is not None
        assert "aggregated_signal" in aggregated_signals
        assert "signal_consensus" in aggregated_signals
        assert "aggregation_metrics" in aggregated_signals
        # Проверка типов данных
        assert isinstance(aggregated_signals["aggregated_signal"], dict)
        assert isinstance(aggregated_signals["signal_consensus"], str)
        assert isinstance(aggregated_signals["aggregation_metrics"], dict)

    def test_analyze_signal_quality(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест анализа качества сигналов."""
        # Анализ качества сигналов
        quality_analysis = signal_processor.analyze_signal_quality(sample_signals)
        # Проверки
        assert quality_analysis is not None
        assert "overall_quality" in quality_analysis
        assert "signal_reliability" in quality_analysis
        assert "signal_consistency" in quality_analysis
        assert "quality_metrics" in quality_analysis
        # Проверка типов данных
        assert isinstance(quality_analysis["overall_quality"], float)
        assert isinstance(quality_analysis["signal_reliability"], float)
        assert isinstance(quality_analysis["signal_consistency"], float)
        assert isinstance(quality_analysis["quality_metrics"], dict)
        # Проверка диапазонов
        assert 0.0 <= quality_analysis["overall_quality"] <= 1.0
        assert 0.0 <= quality_analysis["signal_reliability"] <= 1.0
        assert 0.0 <= quality_analysis["signal_consistency"] <= 1.0

    def test_validate_signals(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест валидации сигналов."""
        # Валидация сигналов
        validation_result = signal_processor.validate_signals(sample_signals)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "validation_score" in validation_result
        assert "validated_signals" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["validation_score"], float)
        assert isinstance(validation_result["validated_signals"], list)
        # Проверка диапазона
        assert 0.0 <= validation_result["validation_score"] <= 1.0

    def test_calculate_signal_statistics(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест расчета статистики сигналов."""
        # Расчет статистики
        statistics = signal_processor.calculate_signal_statistics(sample_signals)
        # Проверки
        assert statistics is not None
        assert "total_signals" in statistics
        assert "buy_signals" in statistics
        assert "sell_signals" in statistics
        assert "avg_strength" in statistics
        assert "avg_confidence" in statistics
        assert "signal_distribution" in statistics
        # Проверка типов данных
        assert isinstance(statistics["total_signals"], int)
        assert isinstance(statistics["buy_signals"], int)
        assert isinstance(statistics["sell_signals"], int)
        assert isinstance(statistics["avg_strength"], float)
        assert isinstance(statistics["avg_confidence"], float)
        assert isinstance(statistics["signal_distribution"], dict)
        # Проверка логики
        assert statistics["total_signals"] >= 0
        assert statistics["buy_signals"] + statistics["sell_signals"] <= statistics["total_signals"]

    def test_backtest_signals(
        self, signal_processor: SignalProcessor, sample_signals: list, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест бэктестинга сигналов."""
        # Бэктестинг сигналов
        backtest_result = signal_processor.backtest_signals(sample_signals, sample_market_data)
        # Проверки
        assert backtest_result is not None
        assert "total_return" in backtest_result
        assert "signal_accuracy" in backtest_result
        assert "profit_factor" in backtest_result
        assert "win_rate" in backtest_result
        assert "trades" in backtest_result
        # Проверка типов данных
        assert isinstance(backtest_result["total_return"], float)
        assert isinstance(backtest_result["signal_accuracy"], float)
        assert isinstance(backtest_result["profit_factor"], float)
        assert isinstance(backtest_result["win_rate"], float)
        assert isinstance(backtest_result["trades"], list)
        # Проверка диапазонов
        assert 0.0 <= backtest_result["signal_accuracy"] <= 1.0
        assert 0.0 <= backtest_result["win_rate"] <= 1.0

    def test_optimize_signal_parameters(
        self, signal_processor: SignalProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации параметров сигналов."""
        # Параметры для оптимизации
        param_ranges = {"rsi_period": (10, 20), "rsi_overbought": (65, 75), "rsi_oversold": (25, 35)}
        # Оптимизация параметров
        optimization_result = signal_processor.optimize_signal_parameters(sample_market_data)
        # Проверки
        assert optimization_result is not None
        assert "best_parameters" in optimization_result
        assert "best_performance" in optimization_result
        assert "optimization_score" in optimization_result
        # Проверка типов данных
        assert isinstance(optimization_result["best_parameters"], dict)
        assert isinstance(optimization_result["best_performance"], dict)
        assert isinstance(optimization_result["optimization_score"], float)
        # Проверка диапазона
        assert 0.0 <= optimization_result["optimization_score"] <= 1.0

    def test_generate_signal_alerts(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест генерации алертов сигналов."""
        # Генерация алертов
        alerts = signal_processor.generate_signal_alerts(sample_signals)
        # Проверки
        assert alerts is not None
        assert isinstance(alerts, list)
        # Проверка структуры алертов
        for alert in alerts:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert
            assert "timestamp" in alert
            assert "signal_id" in alert
            # Проверка типов данных
            assert alert["type"] in ["signal_generated", "signal_confirmed", "signal_expired"]
            assert alert["severity"] in ["low", "medium", "high", "critical"]
            assert isinstance(alert["message"], str)
            assert isinstance(alert["timestamp"], datetime)
            assert isinstance(alert["signal_id"], str)

    def test_get_signal_history(self, signal_processor: SignalProcessor, sample_signals: list) -> None:
        """Тест получения истории сигналов."""
        # Получение истории
        history = signal_processor.get_signal_history("BTCUSDT", "1h")
        # Проверки
        assert history is not None
        assert isinstance(history, list)
        assert len(history) >= 0

    def test_error_handling(self, signal_processor: SignalProcessor) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            signal_processor.generate_rsi_signals(None)
        with pytest.raises(ValueError):
            signal_processor.filter_signals(None, {})

    def test_edge_cases(self, signal_processor: SignalProcessor) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        # Эти функции должны обрабатывать короткие данные
        rsi_signals = signal_processor.generate_rsi_signals(short_data, period=2)
        assert isinstance(rsi_signals, list)
        # Тест с пустыми сигналами
        empty_signals: list[dict] = []
        filtered_signals = signal_processor.filter_signals(empty_signals, {})
        assert filtered_signals == []

    def test_cleanup(self, signal_processor: SignalProcessor) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        signal_processor.cleanup()
        # Проверка, что ресурсы освобождены
        assert signal_processor.signal_generators == {}
        assert signal_processor.signal_filters == {}
        assert signal_processor.signal_aggregators == {}
