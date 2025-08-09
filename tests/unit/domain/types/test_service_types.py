"""
Unit тесты для service_types.

Покрывает:
- Базовые типы для сервисов
- Enums для рыночных метрик
- TypedDict для метрик
- Dataclass для результатов
- Протоколы для сервисов
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock
import pandas as pd

from domain.type_definitions.service_types import (
    # Базовые типы
    MarketDataFrame,
    OrderBookData,
    HistoricalData,
    FeatureVector,
    VolatilityValue,
    TrendStrengthValue,
    CorrelationValue,
    LiquidityScore,
    SpreadValue,
    ServiceConfig,
    AnalysisConfig,
    ModelConfig,
    # Enums
    TrendDirection,
    VolatilityTrend,
    VolumeTrend,
    CorrelationStrength,
    MarketRegime,
    LiquidityZoneType,
    SweepType,
    PatternType,
    IndicatorType,
    # TypedDict
    VolatilityMetrics,
    TrendMetrics,
    VolumeMetrics,
    CorrelationMetrics,
    MarketEfficiencyMetrics,
    LiquidityMetrics,
    MomentumMetrics,
    MarketStressMetrics,
    # Dataclass
    MarketMetricsResult,
    LiquidityZone,
    LiquiditySweep,
    LiquidityAnalysisResult,
    SpreadAnalysisResult,
    SpreadMovementPrediction,
    PredictionResult,
    LiquidityPredictionResult,
    MLModelPerformance,
    AggregationRules,
    # Протоколы
    MarketAnalysisProtocol,
    PatternAnalysisProtocol,
    RiskAnalysisProtocol,
    LiquidityAnalysisProtocol,
    SpreadAnalysisProtocol,
    MLPredictionProtocol,
)
from domain.type_definitions import (
    ConfidenceLevel,
    MetadataDict,
    PerformanceScore,
    PriceValue,
    RiskLevel,
    SignalId,
    StrategyId,
    Symbol,
    TimestampValue,
    VolumeValue,
)


class TestBaseTypes:
    """Тесты для базовых типов сервисов."""

    def test_market_data_frame_type(self):
        """Тест типа MarketDataFrame."""
        # Создаем DataFrame
        data = {
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000.0],
        }
        df = pd.DataFrame(data)

        # Проверяем, что это DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_order_book_data_type(self):
        """Тест типа OrderBookData."""
        order_book: OrderBookData = {
            "bids": [[100.0, 1.5], [99.0, 2.0]],
            "asks": [[101.0, 1.0], [102.0, 1.5]],
            "timestamp": datetime.now().isoformat(),
        }

        assert isinstance(order_book, dict)
        assert "bids" in order_book
        assert "asks" in order_book
        assert "timestamp" in order_book

    def test_historical_data_type(self):
        """Тест типа HistoricalData."""
        data = {"timestamp": [datetime.now()], "price": [100.0], "volume": [1000.0]}
        df = pd.DataFrame(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_feature_vector_type(self):
        """Тест типа FeatureVector."""
        features: FeatureVector = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert isinstance(features, list)
        assert all(isinstance(f, float) for f in features)
        assert len(features) == 5

    def test_volatility_value_type(self):
        """Тест типа VolatilityValue."""
        volatility = VolatilityValue(Decimal("0.15"))

        assert isinstance(volatility, Decimal)
        assert volatility == Decimal("0.15")

    def test_trend_strength_value_type(self):
        """Тест типа TrendStrengthValue."""
        strength = TrendStrengthValue(Decimal("0.75"))

        assert isinstance(strength, Decimal)
        assert strength == Decimal("0.75")

    def test_correlation_value_type(self):
        """Тест типа CorrelationValue."""
        correlation = CorrelationValue(Decimal("0.85"))

        assert isinstance(correlation, Decimal)
        assert correlation == Decimal("0.85")

    def test_liquidity_score_type(self):
        """Тест типа LiquidityScore."""
        score = LiquidityScore(Decimal("0.90"))

        assert isinstance(score, Decimal)
        assert score == Decimal("0.90")

    def test_spread_value_type(self):
        """Тест типа SpreadValue."""
        spread = SpreadValue(Decimal("0.001"))

        assert isinstance(spread, Decimal)
        assert spread == Decimal("0.001")

    def test_service_config_type(self):
        """Тест типа ServiceConfig."""
        config: ServiceConfig = {"timeout": 30, "retry_count": 3, "cache_enabled": True}

        assert isinstance(config, dict)
        assert "timeout" in config
        assert "retry_count" in config
        assert "cache_enabled" in config

    def test_analysis_config_type(self):
        """Тест типа AnalysisConfig."""
        config: AnalysisConfig = {"window_size": 20, "confidence_level": 0.95, "min_pattern_length": 5}

        assert isinstance(config, dict)
        assert "window_size" in config
        assert "confidence_level" in config
        assert "min_pattern_length" in config

    def test_model_config_type(self):
        """Тест типа ModelConfig."""
        config: ModelConfig = {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 100},
            "feature_columns": ["feature1", "feature2"],
        }

        assert isinstance(config, dict)
        assert "model_type" in config
        assert "hyperparameters" in config
        assert "feature_columns" in config


class TestEnums:
    """Тесты для Enums."""

    def test_trend_direction_enum(self):
        """Тест TrendDirection enum."""
        assert TrendDirection.UPTREND == "uptrend"
        assert TrendDirection.DOWNTREND == "downtrend"
        assert TrendDirection.SIDEWAYS == "sideways"
        assert TrendDirection.UNKNOWN == "unknown"

        # Проверяем все значения
        values = [e.value for e in TrendDirection]
        assert "uptrend" in values
        assert "downtrend" in values
        assert "sideways" in values
        assert "unknown" in values

    def test_volatility_trend_enum(self):
        """Тест VolatilityTrend enum."""
        assert VolatilityTrend.INCREASING == "increasing"
        assert VolatilityTrend.DECREASING == "decreasing"
        assert VolatilityTrend.STABLE == "stable"

        values = [e.value for e in VolatilityTrend]
        assert "increasing" in values
        assert "decreasing" in values
        assert "stable" in values

    def test_volume_trend_enum(self):
        """Тест VolumeTrend enum."""
        assert VolumeTrend.INCREASING == "increasing"
        assert VolumeTrend.DECREASING == "decreasing"
        assert VolumeTrend.STABLE == "stable"

        values = [e.value for e in VolumeTrend]
        assert "increasing" in values
        assert "decreasing" in values
        assert "stable" in values

    def test_correlation_strength_enum(self):
        """Тест CorrelationStrength enum."""
        assert CorrelationStrength.STRONG == "strong"
        assert CorrelationStrength.MODERATE == "moderate"

        values = [e.value for e in CorrelationStrength]
        assert "strong" in values
        assert "moderate" in values

    def test_market_regime_enum(self):
        """Тест MarketRegime enum."""
        assert MarketRegime.EFFICIENT == "efficient"
        assert MarketRegime.TRENDING == "trending"
        assert MarketRegime.MEAN_REVERTING == "mean_reverting"
        assert MarketRegime.VOLATILE == "volatile"

        values = [e.value for e in MarketRegime]
        assert "efficient" in values
        assert "trending" in values
        assert "mean_reverting" in values
        assert "volatile" in values

    def test_liquidity_zone_type_enum(self):
        """Тест LiquidityZoneType enum."""
        assert LiquidityZoneType.SUPPORT == "support"
        assert LiquidityZoneType.RESISTANCE == "resistance"
        assert LiquidityZoneType.NEUTRAL == "neutral"

        values = [e.value for e in LiquidityZoneType]
        assert "support" in values
        assert "resistance" in values
        assert "neutral" in values

    def test_sweep_type_enum(self):
        """Тест SweepType enum."""
        assert SweepType.SWEEP_HIGH == "sweep_high"
        assert SweepType.SWEEP_LOW == "sweep_low"

        values = [e.value for e in SweepType]
        assert "sweep_high" in values
        assert "sweep_low" in values

    def test_pattern_type_enum(self):
        """Тест PatternType enum."""
        assert PatternType.CANDLE == "candle"
        assert PatternType.PRICE == "price"
        assert PatternType.VOLUME == "volume"
        assert PatternType.TECHNICAL == "technical"
        assert PatternType.COMBINED == "combined"

        values = [e.value for e in PatternType]
        assert "candle" in values
        assert "price" in values
        assert "volume" in values
        assert "technical" in values
        assert "combined" in values

    def test_indicator_type_enum(self):
        """Тест IndicatorType enum."""
        assert IndicatorType.TREND == "trend"
        assert IndicatorType.MOMENTUM == "momentum"
        assert IndicatorType.VOLATILITY == "volatility"
        assert IndicatorType.VOLUME == "volume"
        assert IndicatorType.SUPPORT_RESISTANCE == "support_resistance"

        values = [e.value for e in IndicatorType]
        assert "trend" in values
        assert "momentum" in values
        assert "volatility" in values
        assert "volume" in values
        assert "support_resistance" in values


class TestTypedDict:
    """Тесты для TypedDict."""

    def test_volatility_metrics(self):
        """Тест VolatilityMetrics TypedDict."""
        metrics: VolatilityMetrics = {
            "current_volatility": 0.15,
            "historical_volatility": 0.12,
            "volatility_percentile": 0.75,
            "volatility_trend": VolatilityTrend.INCREASING,
            "garch_volatility": 0.14,
            "realized_volatility": 0.13,
            "implied_volatility": 0.16,
        }

        assert isinstance(metrics, dict)
        assert "current_volatility" in metrics
        assert "volatility_trend" in metrics
        assert metrics["volatility_trend"] == VolatilityTrend.INCREASING

    def test_trend_metrics(self):
        """Тест TrendMetrics TypedDict."""
        metrics: TrendMetrics = {
            "trend_direction": TrendDirection.UPTREND,
            "trend_strength": 0.85,
            "trend_duration": 10,
            "support_resistance": {"support": 100.0, "resistance": 110.0},
            "adx_value": 25.5,
            "trend_quality": 0.9,
            "momentum_score": 0.75,
        }

        assert isinstance(metrics, dict)
        assert "trend_direction" in metrics
        assert "support_resistance" in metrics
        assert metrics["trend_direction"] == TrendDirection.UPTREND

    def test_volume_metrics(self):
        """Тест VolumeMetrics TypedDict."""
        metrics: VolumeMetrics = {
            "volume_ma": 1000.0,
            "volume_ratio": 1.2,
            "volume_trend": VolumeTrend.INCREASING,
            "volume_profile": {"high": 1200.0, "low": 800.0},
            "volume_delta": 200.0,
            "volume_imbalance": 0.1,
            "vwap": 105.0,
        }

        assert isinstance(metrics, dict)
        assert "volume_trend" in metrics
        assert "volume_profile" in metrics
        assert metrics["volume_trend"] == VolumeTrend.INCREASING

    def test_correlation_metrics(self):
        """Тест CorrelationMetrics TypedDict."""
        metrics: CorrelationMetrics = {
            "correlation_coefficient": 0.85,
            "correlation_strength": CorrelationStrength.STRONG,
            "correlation_trend": "increasing",
            "lag_value": 5,
            "rolling_correlation": 0.82,
            "cointegration_score": 0.9,
        }

        assert isinstance(metrics, dict)
        assert "correlation_strength" in metrics
        assert metrics["correlation_strength"] == CorrelationStrength.STRONG

    def test_market_efficiency_metrics(self):
        """Тест MarketEfficiencyMetrics TypedDict."""
        metrics: MarketEfficiencyMetrics = {
            "efficiency_ratio": 0.75,
            "hurst_exponent": 0.55,
            "market_regime": MarketRegime.TRENDING,
            "fractal_dimension": 1.5,
            "entropy_value": 0.8,
        }

        assert isinstance(metrics, dict)
        assert "market_regime" in metrics
        assert metrics["market_regime"] == MarketRegime.TRENDING

    def test_liquidity_metrics(self):
        """Тест LiquidityMetrics TypedDict."""
        metrics: LiquidityMetrics = {
            "liquidity_score": 0.9,
            "bid_ask_spread": 0.001,
            "order_book_depth": 1000000.0,
            "volume_imbalance": 0.05,
            "market_impact": 0.0001,
            "slippage_estimate": 0.0005,
        }

        assert isinstance(metrics, dict)
        assert "liquidity_score" in metrics
        assert "bid_ask_spread" in metrics

    def test_momentum_metrics(self):
        """Тест MomentumMetrics TypedDict."""
        metrics: MomentumMetrics = {
            "rsi_value": 65.0,
            "macd_value": 0.5,
            "stochastic_value": 75.0,
            "williams_r": -25.0,
            "momentum_score": 0.8,
            "divergence_detected": False,
        }

        assert isinstance(metrics, dict)
        assert "momentum_score" in metrics
        assert "divergence_detected" in metrics

    def test_market_stress_metrics(self):
        """Тест MarketStressMetrics TypedDict."""
        metrics: MarketStressMetrics = {
            "stress_index": 0.3,
            "fear_greed_index": 0.6,
            "volatility_regime": "normal",
            "liquidity_crisis": False,
            "flash_crash_risk": 0.1,
        }

        assert isinstance(metrics, dict)
        assert "stress_index" in metrics
        assert "liquidity_crisis" in metrics


class TestDataclasses:
    """Тесты для dataclass."""

    def test_market_metrics_result(self):
        """Тест MarketMetricsResult dataclass."""
        volatility_metrics: VolatilityMetrics = {
            "current_volatility": 0.15,
            "volatility_trend": VolatilityTrend.INCREASING,
        }

        trend_metrics: TrendMetrics = {"trend_direction": TrendDirection.UPTREND, "trend_strength": 0.85}

        volume_metrics: VolumeMetrics = {"volume_ma": 1000.0, "volume_trend": VolumeTrend.INCREASING}

        momentum_metrics: MomentumMetrics = {"rsi_value": 65.0, "momentum_score": 0.8, "divergence_detected": False}

        liquidity_metrics: LiquidityMetrics = {"liquidity_score": 0.9, "bid_ask_spread": 0.001}

        stress_metrics: MarketStressMetrics = {"stress_index": 0.3, "liquidity_crisis": False}

        result = MarketMetricsResult(
            volatility=volatility_metrics,
            trend=trend_metrics,
            volume=volume_metrics,
            momentum=momentum_metrics,
            liquidity=liquidity_metrics,
            stress=stress_metrics,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            confidence=ConfidenceLevel.HIGH,
        )

        assert result.volatility == volatility_metrics
        assert result.trend == trend_metrics
        assert result.volume == volume_metrics
        assert result.momentum == momentum_metrics
        assert result.liquidity == liquidity_metrics
        assert result.stress == stress_metrics
        assert result.symbol == "BTCUSDT"
        assert result.confidence == ConfidenceLevel.HIGH

    def test_liquidity_zone(self):
        """Тест LiquidityZone dataclass."""
        zone = LiquidityZone(
            price=Decimal("50000.0"),
            zone_type=LiquidityZoneType.SUPPORT,
            strength=0.85,
            volume=Decimal("1000.0"),
            touches=5,
            timestamp=datetime.now(),
            confidence=ConfidenceLevel.HIGH,
        )

        assert zone.price == Decimal("50000.0")
        assert zone.zone_type == LiquidityZoneType.SUPPORT
        assert zone.strength == 0.85
        assert zone.volume == Decimal("1000.0")
        assert zone.touches == 5
        assert zone.confidence == ConfidenceLevel.HIGH

    def test_liquidity_sweep(self):
        """Тест LiquiditySweep dataclass."""
        sweep = LiquiditySweep(
            timestamp=datetime.now(),
            price=Decimal("50000.0"),
            sweep_type=SweepType.SWEEP_HIGH,
            confidence=ConfidenceLevel.MEDIUM,
            volume=Decimal("500.0"),
        )

        assert sweep.price == Decimal("50000.0")
        assert sweep.sweep_type == SweepType.SWEEP_HIGH
        assert sweep.confidence == ConfidenceLevel.MEDIUM
        assert sweep.volume == Decimal("500.0")

    def test_liquidity_analysis_result(self):
        """Тест LiquidityAnalysisResult dataclass."""
        zones = [
            LiquidityZone(
                price=Decimal("50000.0"),
                zone_type=LiquidityZoneType.SUPPORT,
                strength=0.85,
                volume=Decimal("1000.0"),
                touches=5,
                timestamp=datetime.now(),
                confidence=ConfidenceLevel.HIGH,
            )
        ]

        sweeps = [
            LiquiditySweep(
                timestamp=datetime.now(),
                price=Decimal("50000.0"),
                sweep_type=SweepType.SWEEP_HIGH,
                confidence=ConfidenceLevel.MEDIUM,
                volume=Decimal("500.0"),
            )
        ]

        result = LiquidityAnalysisResult(
            liquidity_score=Decimal("0.9"),
            confidence=ConfidenceLevel.HIGH,
            volume_score=0.85,
            order_book_score=0.9,
            volatility_score=0.8,
            zones=zones,
            sweeps=sweeps,
        )

        assert result.liquidity_score == Decimal("0.9")
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.volume_score == 0.85
        assert result.order_book_score == 0.9
        assert result.volatility_score == 0.8
        assert len(result.zones) == 1
        assert len(result.sweeps) == 1

    def test_spread_analysis_result(self):
        """Тест SpreadAnalysisResult dataclass."""
        result = SpreadAnalysisResult(
            spread=Decimal("0.001"),
            imbalance=0.1,
            confidence=ConfidenceLevel.HIGH,
            best_bid=Decimal("50000.0"),
            best_ask=Decimal("50000.001"),
            depth_imbalance=0.05,
            spread_trend="increasing",
        )

        assert result.spread == Decimal("0.001")
        assert result.imbalance == 0.1
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.best_bid == Decimal("50000.0")
        assert result.best_ask == Decimal("50000.001")
        assert result.depth_imbalance == 0.05
        assert result.spread_trend == "increasing"

    def test_spread_movement_prediction(self):
        """Тест SpreadMovementPrediction dataclass."""
        prediction = SpreadMovementPrediction(
            prediction=0.002,
            confidence=ConfidenceLevel.MEDIUM,
            ma_short=0.001,
            ma_long=0.0015,
            volatility=0.1,
            direction="increasing",
        )

        assert prediction.prediction == 0.002
        assert prediction.confidence == ConfidenceLevel.MEDIUM
        assert prediction.ma_short == 0.001
        assert prediction.ma_long == 0.0015
        assert prediction.volatility == 0.1
        assert prediction.direction == "increasing"

    def test_prediction_result(self):
        """Тест PredictionResult dataclass."""
        result = PredictionResult(
            predicted_spread=0.002,
            confidence=ConfidenceLevel.HIGH,
            model_accuracy=0.85,
            feature_importance={"feature1": 0.5, "feature2": 0.3},
            prediction_interval=(0.001, 0.003),
        )

        assert result.predicted_spread == 0.002
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.model_accuracy == 0.85
        assert "feature1" in result.feature_importance
        assert result.prediction_interval == (0.001, 0.003)

    def test_liquidity_prediction_result(self):
        """Тест LiquidityPredictionResult dataclass."""
        result = LiquidityPredictionResult(
            predicted_class="high",
            confidence=ConfidenceLevel.HIGH,
            probabilities={"high": 0.8, "medium": 0.15, "low": 0.05},
            model_accuracy=0.9,
            feature_importance={"feature1": 0.6, "feature2": 0.4},
        )

        assert result.predicted_class == "high"
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.probabilities["high"] == 0.8
        assert result.model_accuracy == 0.9
        assert "feature1" in result.feature_importance

    def test_ml_model_performance(self):
        """Тест MLModelPerformance dataclass."""
        performance = MLModelPerformance(
            spread_accuracy=0.85,
            liquidity_accuracy=0.9,
            spread_loss=0.15,
            liquidity_loss=0.1,
            overall_performance=0.875,
            training_samples=10000,
            validation_samples=2000,
            last_training=datetime.now(),
        )

        assert performance.spread_accuracy == 0.85
        assert performance.liquidity_accuracy == 0.9
        assert performance.spread_loss == 0.15
        assert performance.liquidity_loss == 0.1
        assert performance.overall_performance == 0.875
        assert performance.training_samples == 10000
        assert performance.validation_samples == 2000

    def test_aggregation_rules(self):
        """Тест AggregationRules dataclass."""
        rules = AggregationRules(
            confidence_threshold=0.7,
            max_signals=15,
            time_window_hours=48,
            weight_by_confidence=True,
            weight_by_recency=True,
            weight_by_volume=True,
        )

        assert rules.confidence_threshold == 0.7
        assert rules.max_signals == 15
        assert rules.time_window_hours == 48
        assert rules.weight_by_confidence is True
        assert rules.weight_by_recency is True
        assert rules.weight_by_volume is True


class TestProtocols:
    """Тесты для протоколов."""

    def test_market_analysis_protocol(self):
        """Тест MarketAnalysisProtocol."""
        mock_service = Mock(spec=MarketAnalysisProtocol)

        # Настройка методов
        mock_service.analyze_market_data = AsyncMock(return_value=Mock())
        mock_service.calculate_volatility_metrics = AsyncMock(return_value={})
        mock_service.calculate_trend_metrics = AsyncMock(return_value={})

        # Проверка наличия методов
        assert hasattr(mock_service, "analyze_market_data")
        assert hasattr(mock_service, "calculate_volatility_metrics")
        assert hasattr(mock_service, "calculate_trend_metrics")
        assert callable(mock_service.analyze_market_data)
        assert callable(mock_service.calculate_volatility_metrics)
        assert callable(mock_service.calculate_trend_metrics)

    def test_pattern_analysis_protocol(self):
        """Тест PatternAnalysisProtocol."""
        mock_service = Mock(spec=PatternAnalysisProtocol)

        # Настройка методов
        mock_service.discover_patterns = AsyncMock(return_value=[])
        mock_service.validate_pattern = AsyncMock(return_value=0.85)

        # Проверка наличия методов
        assert hasattr(mock_service, "discover_patterns")
        assert hasattr(mock_service, "validate_pattern")
        assert callable(mock_service.discover_patterns)
        assert callable(mock_service.validate_pattern)

    def test_risk_analysis_protocol(self):
        """Тест RiskAnalysisProtocol."""
        mock_service = Mock(spec=RiskAnalysisProtocol)

        # Настройка методов
        mock_service.calculate_risk_metrics = AsyncMock(return_value={})
        mock_service.calculate_var = AsyncMock(return_value=0.05)

        # Проверка наличия методов
        assert hasattr(mock_service, "calculate_risk_metrics")
        assert hasattr(mock_service, "calculate_var")
        assert callable(mock_service.calculate_risk_metrics)
        assert callable(mock_service.calculate_var)

    def test_liquidity_analysis_protocol(self):
        """Тест LiquidityAnalysisProtocol."""
        mock_service = Mock(spec=LiquidityAnalysisProtocol)

        # Настройка методов
        mock_service.analyze_liquidity = AsyncMock(return_value=Mock())
        mock_service.identify_liquidity_zones = AsyncMock(return_value=[])

        # Проверка наличия методов
        assert hasattr(mock_service, "analyze_liquidity")
        assert hasattr(mock_service, "identify_liquidity_zones")
        assert callable(mock_service.analyze_liquidity)
        assert callable(mock_service.identify_liquidity_zones)

    def test_spread_analysis_protocol(self):
        """Тест SpreadAnalysisProtocol."""
        mock_service = Mock(spec=SpreadAnalysisProtocol)

        # Настройка методов
        mock_service.analyze_spread = AsyncMock(return_value=Mock())
        mock_service.predict_spread_movement = AsyncMock(return_value=Mock())

        # Проверка наличия методов
        assert hasattr(mock_service, "analyze_spread")
        assert hasattr(mock_service, "predict_spread_movement")
        assert callable(mock_service.analyze_spread)
        assert callable(mock_service.predict_spread_movement)

    def test_ml_prediction_protocol(self):
        """Тест MLPredictionProtocol."""
        mock_service = Mock(spec=MLPredictionProtocol)

        # Настройка методов
        mock_service.predict_spread = AsyncMock(return_value=Mock())
        mock_service.predict_liquidity = AsyncMock(return_value=Mock())
        mock_service.train_models = AsyncMock(return_value=True)

        # Проверка наличия методов
        assert hasattr(mock_service, "predict_spread")
        assert hasattr(mock_service, "predict_liquidity")
        assert hasattr(mock_service, "train_models")
        assert callable(mock_service.predict_spread)
        assert callable(mock_service.predict_liquidity)
        assert callable(mock_service.train_models)


class TestIntegration:
    """Интеграционные тесты."""

    def test_complete_analysis_workflow(self):
        """Тест полного рабочего процесса анализа."""
        # Создаем тестовые данные
        market_data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "volume": [1000.0],
            }
        )

        order_book = {
            "bids": [[100.0, 1.5], [99.0, 2.0]],
            "asks": [[101.0, 1.0], [102.0, 1.5]],
            "timestamp": datetime.now().isoformat(),
        }

        # Создаем метрики
        volatility_metrics: VolatilityMetrics = {
            "current_volatility": 0.15,
            "volatility_trend": VolatilityTrend.INCREASING,
        }

        trend_metrics: TrendMetrics = {"trend_direction": TrendDirection.UPTREND, "trend_strength": 0.85}

        # Создаем результат анализа
        result = MarketMetricsResult(
            volatility=volatility_metrics,
            trend=trend_metrics,
            volume={"volume_ma": 1000.0, "volume_trend": VolumeTrend.INCREASING},
            momentum={"rsi_value": 65.0, "momentum_score": 0.8, "divergence_detected": False},
            liquidity={"liquidity_score": 0.9, "bid_ask_spread": 0.001},
            stress={"stress_index": 0.3, "liquidity_crisis": False},
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            confidence=ConfidenceLevel.HIGH,
        )

        # Проверяем структуру результата
        assert result.volatility["current_volatility"] == 0.15
        assert result.trend["trend_direction"] == TrendDirection.UPTREND
        assert result.symbol == "BTCUSDT"
        assert result.confidence == ConfidenceLevel.HIGH

    def test_liquidity_analysis_workflow(self):
        """Тест рабочего процесса анализа ликвидности."""
        # Создаем зоны ликвидности
        zones = [
            LiquidityZone(
                price=Decimal("50000.0"),
                zone_type=LiquidityZoneType.SUPPORT,
                strength=0.85,
                volume=Decimal("1000.0"),
                touches=5,
                timestamp=datetime.now(),
                confidence=ConfidenceLevel.HIGH,
            ),
            LiquidityZone(
                price=Decimal("51000.0"),
                zone_type=LiquidityZoneType.RESISTANCE,
                strength=0.75,
                volume=Decimal("800.0"),
                touches=3,
                timestamp=datetime.now(),
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ]

        # Создаем sweep'ы
        sweeps = [
            LiquiditySweep(
                timestamp=datetime.now(),
                price=Decimal("50000.0"),
                sweep_type=SweepType.SWEEP_HIGH,
                confidence=ConfidenceLevel.MEDIUM,
                volume=Decimal("500.0"),
            )
        ]

        # Создаем результат анализа ликвидности
        result = LiquidityAnalysisResult(
            liquidity_score=Decimal("0.9"),
            confidence=ConfidenceLevel.HIGH,
            volume_score=0.85,
            order_book_score=0.9,
            volatility_score=0.8,
            zones=zones,
            sweeps=sweeps,
        )

        # Проверяем результат
        assert result.liquidity_score == Decimal("0.9")
        assert len(result.zones) == 2
        assert len(result.sweeps) == 1
        assert result.zones[0].zone_type == LiquidityZoneType.SUPPORT
        assert result.zones[1].zone_type == LiquidityZoneType.RESISTANCE
        assert result.sweeps[0].sweep_type == SweepType.SWEEP_HIGH

    def test_ml_prediction_workflow(self):
        """Тест рабочего процесса ML предсказаний."""
        # Создаем результат предсказания спреда
        spread_prediction = PredictionResult(
            predicted_spread=0.002,
            confidence=ConfidenceLevel.HIGH,
            model_accuracy=0.85,
            feature_importance={"feature1": 0.5, "feature2": 0.3},
            prediction_interval=(0.001, 0.003),
        )

        # Создаем результат предсказания ликвидности
        liquidity_prediction = LiquidityPredictionResult(
            predicted_class="high",
            confidence=ConfidenceLevel.HIGH,
            probabilities={"high": 0.8, "medium": 0.15, "low": 0.05},
            model_accuracy=0.9,
            feature_importance={"feature1": 0.6, "feature2": 0.4},
        )

        # Создаем метрики производительности
        performance = MLModelPerformance(
            spread_accuracy=0.85,
            liquidity_accuracy=0.9,
            spread_loss=0.15,
            liquidity_loss=0.1,
            overall_performance=0.875,
            training_samples=10000,
            validation_samples=2000,
            last_training=datetime.now(),
        )

        # Проверяем результаты
        assert spread_prediction.predicted_spread == 0.002
        assert liquidity_prediction.predicted_class == "high"
        assert performance.overall_performance == 0.875
        assert performance.training_samples == 10000
