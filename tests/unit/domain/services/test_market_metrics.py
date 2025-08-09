"""
Unit тесты для MarketMetricsService.

Покрывает:
- Основной функционал расчета метрик рынка
- Валидацию данных
- Бизнес-логику анализа метрик
- Обработку ошибок
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from domain.services.market_metrics import MarketMetricsService, _safe_float, _safe_series
from domain.type_definitions.market_metrics_types import (
    VolatilityMetrics,
    TrendMetrics,
    VolumeMetrics,
    LiquidityMetrics,
    MomentumMetrics,
    MarketStressMetrics,
    MarketMetricsResult,
    TrendDirection,
    VolatilityTrend,
    VolumeTrend,
)


class TestSafeFunctions:
    """Тесты для вспомогательных функций."""

    def test_safe_float_valid(self):
        """Тест безопасного преобразования в float с валидными данными."""
        assert _safe_float(10) == 10.0
        assert _safe_float(10.5) == 10.5
        assert _safe_float("10.5") == 10.5
        assert _safe_float(None, default=5.0) == 5.0

    def test_safe_float_invalid(self):
        """Тест безопасного преобразования в float с невалидными данными."""
        assert _safe_float("invalid", default=5.0) == 5.0
        assert _safe_float([], default=3.0) == 3.0
        assert _safe_float({}, default=1.0) == 1.0

    def test_safe_series_valid(self):
        """Тест безопасного преобразования в Series с валидными данными."""
        data = [1, 2, 3, 4, 5]
        series = _safe_series(data)

        assert isinstance(series, pd.Series)
        assert len(series) == 5
        assert series.iloc[0] == 1

    def test_safe_series_invalid(self):
        """Тест безопасного преобразования в Series с невалидными данными."""
        series = _safe_series("invalid")

        assert isinstance(series, pd.Series)
        assert len(series) == 0


class TestMarketMetricsService:
    """Тесты для MarketMetricsService."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Тестовая конфигурация."""
        return {
            "volatility_window": 20,
            "trend_window": 30,
            "volume_window": 20,
            "correlation_window": 30,
        }

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Тестовые данные."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(200, 300, 100),
                "low": np.random.uniform(50, 100, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # Создаем тренд
        data["close"] = data["close"] + np.arange(100) * 0.1
        data["volume"] = data["volume"] + np.random.normal(0, 100, 100)

        return data

    @pytest.fixture
    def order_book(self) -> Dict[str, Any]:
        """Тестовый order book."""
        return {
            "bids": [[100.0, 10.0], [99.9, 15.0], [99.8, 20.0]],
            "asks": [[100.1, 12.0], [100.2, 18.0], [100.3, 25.0]],
        }

    @pytest.fixture
    def service(self, config) -> MarketMetricsService:
        """Экземпляр MarketMetricsService."""
        return MarketMetricsService(config)

    def test_creation_default_config(self):
        """Тест создания сервиса с конфигурацией по умолчанию."""
        service = MarketMetricsService()

        assert service.config["volatility_window"] == 20
        assert service.config["trend_window"] == 30
        assert service.config["volume_window"] == 20
        assert service.config["correlation_window"] == 30
        assert service._lock is not None

    def test_creation_custom_config(self, config):
        """Тест создания сервиса с пользовательской конфигурацией."""
        custom_config = config.copy()
        custom_config["volatility_window"] = 50

        service = MarketMetricsService(custom_config)

        assert service.config["volatility_window"] == 50
        assert service.config["trend_window"] == 30

    def test_calculate_volatility_metrics_none_data(self, service):
        """Тест расчета метрик волатильности с None данными."""
        metrics = service.calculate_volatility_metrics(None)

        assert isinstance(metrics, VolatilityMetrics)
        assert metrics.current_volatility == 0.0
        assert metrics.historical_volatility == 0.0
        assert metrics.volatility_percentile == 0.0
        assert metrics.volatility_trend == VolatilityTrend.STABLE

    def test_calculate_volatility_metrics_empty_data(self, service):
        """Тест расчета метрик волатильности с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_volatility_metrics(empty_data)

        assert isinstance(metrics, VolatilityMetrics)
        assert metrics.current_volatility == 0.0
        assert metrics.historical_volatility == 0.0
        assert metrics.volatility_percentile == 0.0
        assert metrics.volatility_trend == VolatilityTrend.STABLE

    def test_calculate_volatility_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик волатильности."""
        metrics = service.calculate_volatility_metrics(sample_data)

        assert isinstance(metrics, VolatilityMetrics)
        assert metrics.current_volatility >= 0.0
        assert metrics.historical_volatility >= 0.0
        assert 0.0 <= metrics.volatility_percentile <= 1.0
        assert isinstance(metrics.volatility_trend, VolatilityTrend)

    def test_calculate_trend_metrics_none_data(self, service):
        """Тест расчета метрик тренда с None данными."""
        metrics = service.calculate_trend_metrics(None)

        assert isinstance(metrics, dict)
        assert metrics.get("trend_direction") == TrendDirection.SIDEWAYS
        assert metrics.get("trend_strength") == 0.0
        assert metrics.get("trend_confidence") == 0.0

    def test_calculate_trend_metrics_empty_data(self, service):
        """Тест расчета метрик тренда с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_trend_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics.get("trend_direction") == TrendDirection.SIDEWAYS
        assert metrics.get("trend_strength") == 0.0
        assert metrics.get("trend_confidence") == 0.0

    def test_calculate_trend_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик тренда."""
        metrics = service.calculate_trend_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert "trend_direction" in metrics
        assert "trend_strength" in metrics
        assert "trend_confidence" in metrics
        assert isinstance(metrics["trend_direction"], TrendDirection)
        assert 0.0 <= metrics["trend_strength"] <= 1.0
        assert 0.0 <= metrics["trend_confidence"] <= 1.0

    def test_calculate_volume_metrics_none_data(self, service):
        """Тест расчета метрик объема с None данными."""
        metrics = service.calculate_volume_metrics(None)

        assert isinstance(metrics, dict)
        assert metrics.get("current_volume") == 0.0
        assert metrics.get("average_volume") == 0.0
        assert metrics.get("volume_trend") == VolumeTrend.STABLE
        assert metrics.get("volume_ratio") == 0.0
        assert metrics.get("unusual_volume") is False

    def test_calculate_volume_metrics_empty_data(self, service):
        """Тест расчета метрик объема с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_volume_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics.get("current_volume") == 0.0
        assert metrics.get("average_volume") == 0.0
        assert metrics.get("volume_trend") == VolumeTrend.STABLE
        assert metrics.get("volume_ratio") == 0.0
        assert metrics.get("unusual_volume") is False

    def test_calculate_volume_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик объема."""
        metrics = service.calculate_volume_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert "current_volume" in metrics
        assert "average_volume" in metrics
        assert "volume_trend" in metrics
        assert "volume_ratio" in metrics
        assert "unusual_volume" in metrics
        assert isinstance(metrics["volume_trend"], VolumeTrend)
        assert metrics["current_volume"] >= 0.0
        assert metrics["average_volume"] >= 0.0
        assert isinstance(metrics["unusual_volume"], bool)

    def test_calculate_liquidity_metrics_none_data(self, service):
        """Тест расчета метрик ликвидности с None данными."""
        metrics = service.calculate_liquidity_metrics(None, None)

        assert isinstance(metrics, dict)
        assert metrics.get("bid_ask_spread") == 0.0
        assert metrics.get("market_depth") == 0.0
        assert metrics.get("order_book_imbalance") == 0.0
        assert metrics.get("liquidity_score") == 0.0

    def test_calculate_liquidity_metrics_empty_data(self, service):
        """Тест расчета метрик ликвидности с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_liquidity_metrics(empty_data, None)

        assert isinstance(metrics, dict)
        assert metrics.get("bid_ask_spread") == 0.0
        assert metrics.get("market_depth") == 0.0
        assert metrics.get("order_book_imbalance") == 0.0
        assert metrics.get("liquidity_score") == 0.0

    def test_calculate_liquidity_metrics_success(self, service, sample_data, order_book):
        """Тест успешного расчета метрик ликвидности."""
        metrics = service.calculate_liquidity_metrics(sample_data, order_book)

        assert isinstance(metrics, dict)
        assert "bid_ask_spread" in metrics
        assert "market_depth" in metrics
        assert "order_book_imbalance" in metrics
        assert "liquidity_score" in metrics
        assert metrics["bid_ask_spread"] >= 0.0
        assert metrics["market_depth"] >= 0.0
        assert -1.0 <= metrics["order_book_imbalance"] <= 1.0
        assert 0.0 <= metrics["liquidity_score"] <= 1.0

    def test_calculate_momentum_metrics_none_data(self, service):
        """Тест расчета метрик импульса с None данными."""
        metrics = service.calculate_momentum_metrics(None)

        assert isinstance(metrics, dict)
        assert metrics.get("momentum_score") == 0.0
        assert metrics.get("momentum_direction") == "neutral"
        assert metrics.get("momentum_strength") == 0.0

    def test_calculate_momentum_metrics_empty_data(self, service):
        """Тест расчета метрик импульса с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_momentum_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics.get("momentum_score") == 0.0
        assert metrics.get("momentum_direction") == "neutral"
        assert metrics.get("momentum_strength") == 0.0

    def test_calculate_momentum_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик импульса."""
        metrics = service.calculate_momentum_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert "momentum_score" in metrics
        assert "momentum_direction" in metrics
        assert "momentum_strength" in metrics
        assert isinstance(metrics["momentum_score"], float)
        assert metrics["momentum_direction"] in ["bullish", "bearish", "neutral"]
        assert 0.0 <= metrics["momentum_strength"] <= 1.0

    def test_calculate_market_stress_metrics_none_data(self, service):
        """Тест расчета метрик стресса рынка с None данными."""
        metrics = service.calculate_market_stress_metrics(None)

        assert isinstance(metrics, dict)
        assert metrics.get("stress_level") == 0.0
        assert metrics.get("stress_indicator") == "low"
        assert metrics.get("market_regime") == "normal"

    def test_calculate_market_stress_metrics_empty_data(self, service):
        """Тест расчета метрик стресса рынка с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_market_stress_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics.get("stress_level") == 0.0
        assert metrics.get("stress_indicator") == "low"
        assert metrics.get("market_regime") == "normal"

    def test_calculate_market_stress_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик стресса рынка."""
        metrics = service.calculate_market_stress_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert "stress_level" in metrics
        assert "stress_indicator" in metrics
        assert "market_regime" in metrics
        assert 0.0 <= metrics["stress_level"] <= 1.0
        assert metrics["stress_indicator"] in ["low", "medium", "high"]
        assert metrics["market_regime"] in ["normal", "stressed", "crisis"]

    def test_calculate_market_efficiency_metrics_none_data(self, service):
        """Тест расчета метрик эффективности рынка с None данными."""
        metrics = service.calculate_market_efficiency_metrics(None)

        assert isinstance(metrics, dict)
        assert metrics.get("efficiency_ratio") == 0.0
        assert metrics.get("hurst_exponent") == 0.5
        assert metrics.get("market_efficiency") == "inefficient"

    def test_calculate_market_efficiency_metrics_empty_data(self, service):
        """Тест расчета метрик эффективности рынка с пустыми данными."""
        empty_data = pd.DataFrame()
        metrics = service.calculate_market_efficiency_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics.get("efficiency_ratio") == 0.0
        assert metrics.get("hurst_exponent") == 0.5
        assert metrics.get("market_efficiency") == "inefficient"

    def test_calculate_market_efficiency_metrics_success(self, service, sample_data):
        """Тест успешного расчета метрик эффективности рынка."""
        metrics = service.calculate_market_efficiency_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert "efficiency_ratio" in metrics
        assert "hurst_exponent" in metrics
        assert "market_efficiency" in metrics
        assert metrics["efficiency_ratio"] >= 0.0
        assert 0.0 <= metrics["hurst_exponent"] <= 1.0
        assert metrics["market_efficiency"] in ["efficient", "inefficient", "semi_efficient"]

    def test_get_comprehensive_metrics_none_data(self, service):
        """Тест получения комплексных метрик с None данными."""
        result = service.get_comprehensive_metrics(None, None)

        assert isinstance(result, MarketMetricsResult)
        assert result.volatility_metrics is not None
        assert result.trend_metrics is not None
        assert result.volume_metrics is not None
        assert result.liquidity_metrics is not None
        assert result.momentum_metrics is not None
        assert result.stress_metrics is not None

    def test_get_comprehensive_metrics_empty_data(self, service):
        """Тест получения комплексных метрик с пустыми данными."""
        empty_data = pd.DataFrame()
        result = service.get_comprehensive_metrics(empty_data, None)

        assert isinstance(result, MarketMetricsResult)
        assert result.volatility_metrics is not None
        assert result.trend_metrics is not None
        assert result.volume_metrics is not None
        assert result.liquidity_metrics is not None
        assert result.momentum_metrics is not None
        assert result.stress_metrics is not None

    def test_get_comprehensive_metrics_success(self, service, sample_data, order_book):
        """Тест успешного получения комплексных метрик."""
        result = service.get_comprehensive_metrics(sample_data, order_book)

        assert isinstance(result, MarketMetricsResult)
        assert result.volatility_metrics is not None
        assert result.trend_metrics is not None
        assert result.volume_metrics is not None
        assert result.liquidity_metrics is not None
        assert result.momentum_metrics is not None
        assert result.stress_metrics is not None

        # Проверяем типы метрик
        assert isinstance(result.volatility_metrics, VolatilityMetrics)
        assert isinstance(result.trend_metrics, TrendMetrics)
        assert isinstance(result.volume_metrics, VolumeMetrics)
        assert isinstance(result.liquidity_metrics, LiquidityMetrics)
        assert isinstance(result.momentum_metrics, MomentumMetrics)
        assert isinstance(result.stress_metrics, MarketStressMetrics)

    def test_thread_safety(self, service, sample_data):
        """Тест потокобезопасности сервиса."""
        import threading
        import time

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    metrics = service.calculate_volatility_metrics(sample_data)
                    results.append(metrics)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Ошибки в потоках: {errors}"
        assert len(results) == 50  # 5 потоков * 10 итераций

    def test_data_with_missing_columns(self, service):
        """Тест обработки данных с отсутствующими колонками."""
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                # Отсутствуют 'close', 'low', 'volume'
            }
        )

        # Должны обрабатываться без ошибок
        volatility_metrics = service.calculate_volatility_metrics(data)
        trend_metrics = service.calculate_trend_metrics(data)
        volume_metrics = service.calculate_volume_metrics(data)

        assert isinstance(volatility_metrics, VolatilityMetrics)
        assert isinstance(trend_metrics, dict)
        assert isinstance(volume_metrics, dict)

    def test_data_with_invalid_values(self, service):
        """Тест обработки данных с невалидными значениями."""
        data = pd.DataFrame(
            {
                "open": [100, "invalid", 102],
                "high": [105, 106, "invalid"],
                "low": [95, 96, 97],
                "close": [101, 102, 103],
                "volume": [1000, 2000, "invalid"],
            }
        )

        # Должны обрабатываться без ошибок
        volatility_metrics = service.calculate_volatility_metrics(data)
        trend_metrics = service.calculate_trend_metrics(data)
        volume_metrics = service.calculate_volume_metrics(data)

        assert isinstance(volatility_metrics, VolatilityMetrics)
        assert isinstance(trend_metrics, dict)
        assert isinstance(volume_metrics, dict)

    def test_order_book_processing(self, service, sample_data):
        """Тест обработки order book."""
        # Тест с валидным order book
        valid_order_book = {"bids": [[100.0, 10.0], [99.9, 15.0]], "asks": [[100.1, 12.0], [100.2, 18.0]]}

        metrics = service.calculate_liquidity_metrics(sample_data, valid_order_book)

        assert isinstance(metrics, dict)
        assert "bid_ask_spread" in metrics
        assert "market_depth" in metrics
        assert "order_book_imbalance" in metrics
        assert "liquidity_score" in metrics

        # Тест с невалидным order book
        invalid_order_book = {"invalid": "data"}

        metrics = service.calculate_liquidity_metrics(sample_data, invalid_order_book)

        assert isinstance(metrics, dict)
        assert "bid_ask_spread" in metrics
        assert "market_depth" in metrics
        assert "order_book_imbalance" in metrics
        assert "liquidity_score" in metrics
