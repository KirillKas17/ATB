"""
Unit тесты для market_metrics_types.

Покрывает:
- Перечисления (TrendDirection, VolatilityTrend, VolumeTrend)
- Датаклассы (VolatilityMetrics, TrendMetrics, VolumeMetrics, MomentumMetrics, LiquidityMetrics, MarketStressMetrics, MarketMetricsResult)
"""

import pytest
from typing import Optional
import pandas as pd

from domain.type_definitions.market_metrics_types import (
    TrendDirection,
    VolatilityTrend,
    VolumeTrend,
    VolatilityMetrics,
    TrendMetrics,
    VolumeMetrics,
    MomentumMetrics,
    LiquidityMetrics,
    MarketStressMetrics,
    MarketMetricsResult,
)


class TestTrendDirection:
    """Тесты для TrendDirection."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        assert TrendDirection.UP == "up"
        assert TrendDirection.DOWN == "down"
        assert TrendDirection.SIDEWAYS == "sideways"

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "up" in TrendDirection
        assert "down" in TrendDirection
        assert "sideways" in TrendDirection

    def test_enum_iteration(self):
        """Тест итерации по перечислению."""
        directions = list(TrendDirection)
        assert len(directions) == 3
        assert all(isinstance(d, TrendDirection) for d in directions)

    def test_enum_comparison(self):
        """Тест сравнения значений перечисления."""
        assert TrendDirection.UP != TrendDirection.DOWN
        assert TrendDirection.UP == TrendDirection.UP


class TestVolatilityTrend:
    """Тесты для VolatilityTrend."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        assert VolatilityTrend.INCREASING == "increasing"
        assert VolatilityTrend.DECREASING == "decreasing"
        assert VolatilityTrend.STABLE == "stable"

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "increasing" in VolatilityTrend
        assert "decreasing" in VolatilityTrend
        assert "stable" in VolatilityTrend

    def test_enum_count(self):
        """Тест количества значений."""
        assert len(VolatilityTrend) == 3


class TestVolumeTrend:
    """Тесты для VolumeTrend."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        assert VolumeTrend.INCREASING == "increasing"
        assert VolumeTrend.DECREASING == "decreasing"
        assert VolumeTrend.STABLE == "stable"

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "increasing" in VolumeTrend
        assert "decreasing" in VolumeTrend
        assert "stable" in VolumeTrend

    def test_enum_count(self):
        """Тест количества значений."""
        assert len(VolumeTrend) == 3


class TestVolatilityMetrics:
    """Тесты для VolatilityMetrics."""

    @pytest.fixture
    def sample_volatility_metrics(self) -> VolatilityMetrics:
        """Тестовые метрики волатильности."""
        return VolatilityMetrics(
            current_volatility=0.25,
            historical_volatility=0.20,
            volatility_percentile=0.75,
            volatility_trend=VolatilityTrend.INCREASING,
        )

    def test_creation(self, sample_volatility_metrics):
        """Тест создания метрик волатильности."""
        assert sample_volatility_metrics.current_volatility == 0.25
        assert sample_volatility_metrics.historical_volatility == 0.20
        assert sample_volatility_metrics.volatility_percentile == 0.75
        assert sample_volatility_metrics.volatility_trend == VolatilityTrend.INCREASING

    def test_volatility_comparison(self, sample_volatility_metrics):
        """Тест сравнения волатильности."""
        # Текущая волатильность выше исторической
        assert sample_volatility_metrics.current_volatility > sample_volatility_metrics.historical_volatility

        # Процентный ранг выше 50%
        assert sample_volatility_metrics.volatility_percentile > 0.5

    def test_volatility_trend_validation(self):
        """Тест валидации тренда волатильности."""
        metrics = VolatilityMetrics(
            current_volatility=0.15,
            historical_volatility=0.20,
            volatility_percentile=0.25,
            volatility_trend=VolatilityTrend.DECREASING,
        )

        assert metrics.volatility_trend == VolatilityTrend.DECREASING
        assert metrics.current_volatility < metrics.historical_volatility


class TestTrendMetrics:
    """Тесты для TrendMetrics."""

    @pytest.fixture
    def sample_trend_metrics(self) -> TrendMetrics:
        """Тестовые метрики тренда."""
        return TrendMetrics(
            trend_direction=TrendDirection.UP,
            trend_strength=0.85,
            trend_confidence=0.9,
            support_level=45000.0,
            resistance_level=52000.0,
        )

    def test_creation(self, sample_trend_metrics):
        """Тест создания метрик тренда."""
        assert sample_trend_metrics.trend_direction == TrendDirection.UP
        assert sample_trend_metrics.trend_strength == 0.85
        assert sample_trend_metrics.trend_confidence == 0.9
        assert sample_trend_metrics.support_level == 45000.0
        assert sample_trend_metrics.resistance_level == 52000.0

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = TrendMetrics(trend_direction=TrendDirection.SIDEWAYS, trend_strength=0.5)

        assert metrics.trend_confidence == 0.0
        assert metrics.support_level is None
        assert metrics.resistance_level is None

    def test_trend_strength_validation(self):
        """Тест валидации силы тренда."""
        # Сильный тренд
        strong_trend = TrendMetrics(trend_direction=TrendDirection.UP, trend_strength=0.9)
        assert strong_trend.trend_strength > 0.8

        # Слабый тренд
        weak_trend = TrendMetrics(trend_direction=TrendDirection.DOWN, trend_strength=0.3)
        assert weak_trend.trend_strength < 0.5

    def test_support_resistance_validation(self, sample_trend_metrics):
        """Тест валидации уровней поддержки и сопротивления."""
        # Уровень сопротивления должен быть выше поддержки
        assert sample_trend_metrics.resistance_level > sample_trend_metrics.support_level


class TestVolumeMetrics:
    """Тесты для VolumeMetrics."""

    @pytest.fixture
    def sample_volume_metrics(self) -> VolumeMetrics:
        """Тестовые метрики объема."""
        return VolumeMetrics(
            current_volume=1500000.0,
            average_volume=1000000.0,
            volume_trend=VolumeTrend.INCREASING,
            volume_ratio=1.5,
            unusual_volume=True,
        )

    def test_creation(self, sample_volume_metrics):
        """Тест создания метрик объема."""
        assert sample_volume_metrics.current_volume == 1500000.0
        assert sample_volume_metrics.average_volume == 1000000.0
        assert sample_volume_metrics.volume_trend == VolumeTrend.INCREASING
        assert sample_volume_metrics.volume_ratio == 1.5
        assert sample_volume_metrics.unusual_volume is True

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = VolumeMetrics(current_volume=1000000.0, average_volume=1000000.0, volume_trend=VolumeTrend.STABLE)

        assert metrics.volume_ratio == 0.0
        assert metrics.unusual_volume is False

    def test_volume_ratio_calculation(self):
        """Тест расчета соотношения объема."""
        # Объем выше среднего
        high_volume = VolumeMetrics(
            current_volume=2000000.0, average_volume=1000000.0, volume_trend=VolumeTrend.INCREASING, volume_ratio=2.0
        )
        assert high_volume.volume_ratio == 2.0
        assert high_volume.current_volume > high_volume.average_volume

        # Объем ниже среднего
        low_volume = VolumeMetrics(
            current_volume=500000.0, average_volume=1000000.0, volume_trend=VolumeTrend.DECREASING, volume_ratio=0.5
        )
        assert low_volume.volume_ratio == 0.5
        assert low_volume.current_volume < low_volume.average_volume

    def test_unusual_volume_detection(self):
        """Тест обнаружения необычного объема."""
        # Необычно высокий объем
        unusual_high = VolumeMetrics(
            current_volume=3000000.0,
            average_volume=1000000.0,
            volume_trend=VolumeTrend.INCREASING,
            volume_ratio=3.0,
            unusual_volume=True,
        )
        assert unusual_high.unusual_volume is True
        assert unusual_high.volume_ratio > 2.0


class TestMomentumMetrics:
    """Тесты для MomentumMetrics."""

    @pytest.fixture
    def sample_momentum_metrics(self) -> MomentumMetrics:
        """Тестовые метрики моментума."""
        return MomentumMetrics(rsi=65.0, macd=0.5, macd_signal=0.3, macd_histogram=0.2, momentum_score=0.75)

    def test_creation(self, sample_momentum_metrics):
        """Тест создания метрик моментума."""
        assert sample_momentum_metrics.rsi == 65.0
        assert sample_momentum_metrics.macd == 0.5
        assert sample_momentum_metrics.macd_signal == 0.3
        assert sample_momentum_metrics.macd_histogram == 0.2
        assert sample_momentum_metrics.momentum_score == 0.75

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = MomentumMetrics(rsi=50.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0)

        assert metrics.momentum_score == 0.0

    def test_rsi_validation(self):
        """Тест валидации RSI."""
        # Перекупленность
        overbought = MomentumMetrics(rsi=80.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0)
        assert overbought.rsi > 70

        # Перепроданность
        oversold = MomentumMetrics(rsi=25.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0)
        assert oversold.rsi < 30

    def test_macd_validation(self, sample_momentum_metrics):
        """Тест валидации MACD."""
        # MACD выше сигнальной линии
        assert sample_momentum_metrics.macd > sample_momentum_metrics.macd_signal

        # Гистограмма положительная
        assert sample_momentum_metrics.macd_histogram > 0


class TestLiquidityMetrics:
    """Тесты для LiquidityMetrics."""

    @pytest.fixture
    def sample_liquidity_metrics(self) -> LiquidityMetrics:
        """Тестовые метрики ликвидности."""
        return LiquidityMetrics(
            bid_ask_spread=0.001, market_depth=1000000.0, order_book_imbalance=0.1, liquidity_score=0.85
        )

    def test_creation(self, sample_liquidity_metrics):
        """Тест создания метрик ликвидности."""
        assert sample_liquidity_metrics.bid_ask_spread == 0.001
        assert sample_liquidity_metrics.market_depth == 1000000.0
        assert sample_liquidity_metrics.order_book_imbalance == 0.1
        assert sample_liquidity_metrics.liquidity_score == 0.85

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = LiquidityMetrics(bid_ask_spread=0.01, market_depth=500000.0, order_book_imbalance=0.0)

        assert metrics.liquidity_score == 0.0

    def test_spread_validation(self):
        """Тест валидации спреда."""
        # Узкий спред - высокая ликвидность
        tight_spread = LiquidityMetrics(bid_ask_spread=0.0005, market_depth=2000000.0, order_book_imbalance=0.05)
        assert tight_spread.bid_ask_spread < 0.001

        # Широкий спред - низкая ликвидность
        wide_spread = LiquidityMetrics(bid_ask_spread=0.01, market_depth=100000.0, order_book_imbalance=0.3)
        assert wide_spread.bid_ask_spread > 0.005

    def test_market_depth_validation(self):
        """Тест валидации глубины рынка."""
        # Высокая глубина рынка
        high_depth = LiquidityMetrics(bid_ask_spread=0.001, market_depth=5000000.0, order_book_imbalance=0.05)
        assert high_depth.market_depth > 1000000.0

        # Низкая глубина рынка
        low_depth = LiquidityMetrics(bid_ask_spread=0.005, market_depth=50000.0, order_book_imbalance=0.2)
        assert low_depth.market_depth < 100000.0


class TestMarketStressMetrics:
    """Тесты для MarketStressMetrics."""

    @pytest.fixture
    def sample_stress_metrics(self) -> MarketStressMetrics:
        """Тестовые метрики стресса."""
        return MarketStressMetrics(stress_index=0.3, fear_greed_index=45.0, market_regime="normal", stress_level="low")

    def test_creation(self, sample_stress_metrics):
        """Тест создания метрик стресса."""
        assert sample_stress_metrics.stress_index == 0.3
        assert sample_stress_metrics.fear_greed_index == 45.0
        assert sample_stress_metrics.market_regime == "normal"
        assert sample_stress_metrics.stress_level == "low"

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = MarketStressMetrics(stress_index=0.5, fear_greed_index=50.0)

        assert metrics.market_regime == "normal"
        assert metrics.stress_level == "low"

    def test_stress_index_validation(self):
        """Тест валидации индекса стресса."""
        # Низкий стресс
        low_stress = MarketStressMetrics(stress_index=0.2, fear_greed_index=60.0, stress_level="low")
        assert low_stress.stress_index < 0.3

        # Высокий стресс
        high_stress = MarketStressMetrics(stress_index=0.8, fear_greed_index=20.0, stress_level="high")
        assert high_stress.stress_index > 0.7

    def test_fear_greed_index_validation(self):
        """Тест валидации индекса страха и жадности."""
        # Страх
        fear = MarketStressMetrics(stress_index=0.7, fear_greed_index=25.0, stress_level="high")
        assert fear.fear_greed_index < 30

        # Жадность
        greed = MarketStressMetrics(stress_index=0.2, fear_greed_index=75.0, stress_level="low")
        assert greed.fear_greed_index > 70


class TestMarketMetricsResult:
    """Тесты для MarketMetricsResult."""

    @pytest.fixture
    def sample_metrics_result(self) -> MarketMetricsResult:
        """Тестовый результат метрик."""
        volatility = VolatilityMetrics(
            current_volatility=0.25,
            historical_volatility=0.20,
            volatility_percentile=0.75,
            volatility_trend=VolatilityTrend.INCREASING,
        )

        trend = TrendMetrics(
            trend_direction=TrendDirection.UP,
            trend_strength=0.85,
            trend_confidence=0.9,
            support_level=45000.0,
            resistance_level=52000.0,
        )

        volume = VolumeMetrics(
            current_volume=1500000.0,
            average_volume=1000000.0,
            volume_trend=VolumeTrend.INCREASING,
            volume_ratio=1.5,
            unusual_volume=True,
        )

        momentum = MomentumMetrics(rsi=65.0, macd=0.5, macd_signal=0.3, macd_histogram=0.2, momentum_score=0.75)

        liquidity = LiquidityMetrics(
            bid_ask_spread=0.001, market_depth=1000000.0, order_book_imbalance=0.1, liquidity_score=0.85
        )

        stress = MarketStressMetrics(
            stress_index=0.3, fear_greed_index=45.0, market_regime="normal", stress_level="low"
        )

        return MarketMetricsResult(
            volatility=volatility,
            trend=trend,
            volume=volume,
            momentum=momentum,
            liquidity=liquidity,
            stress=stress,
            timestamp="2024-01-01T12:00:00Z",
            symbol="BTC/USDT",
        )

    def test_creation(self, sample_metrics_result):
        """Тест создания результата метрик."""
        assert sample_metrics_result.symbol == "BTC/USDT"
        assert sample_metrics_result.timestamp == "2024-01-01T12:00:00Z"
        assert isinstance(sample_metrics_result.volatility, VolatilityMetrics)
        assert isinstance(sample_metrics_result.trend, TrendMetrics)
        assert isinstance(sample_metrics_result.volume, VolumeMetrics)
        assert isinstance(sample_metrics_result.momentum, MomentumMetrics)
        assert isinstance(sample_metrics_result.liquidity, LiquidityMetrics)
        assert isinstance(sample_metrics_result.stress, MarketStressMetrics)

    def test_default_values(self):
        """Тест значений по умолчанию."""
        volatility = VolatilityMetrics(
            current_volatility=0.2,
            historical_volatility=0.2,
            volatility_percentile=0.5,
            volatility_trend=VolatilityTrend.STABLE,
        )

        trend = TrendMetrics(trend_direction=TrendDirection.SIDEWAYS, trend_strength=0.5)

        volume = VolumeMetrics(current_volume=1000000.0, average_volume=1000000.0, volume_trend=VolumeTrend.STABLE)

        momentum = MomentumMetrics(rsi=50.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0)

        liquidity = LiquidityMetrics(bid_ask_spread=0.01, market_depth=500000.0, order_book_imbalance=0.0)

        stress = MarketStressMetrics(stress_index=0.5, fear_greed_index=50.0)

        result = MarketMetricsResult(
            volatility=volatility, trend=trend, volume=volume, momentum=momentum, liquidity=liquidity, stress=stress
        )

        assert result.timestamp is None
        assert result.symbol is None

    def test_comprehensive_analysis(self, sample_metrics_result):
        """Тест комплексного анализа."""
        # Проверка волатильности
        assert (
            sample_metrics_result.volatility.current_volatility > sample_metrics_result.volatility.historical_volatility
        )
        assert sample_metrics_result.volatility.volatility_trend == VolatilityTrend.INCREASING

        # Проверка тренда
        assert sample_metrics_result.trend.trend_direction == TrendDirection.UP
        assert sample_metrics_result.trend.trend_strength > 0.8
        assert sample_metrics_result.trend.resistance_level > sample_metrics_result.trend.support_level

        # Проверка объема
        assert sample_metrics_result.volume.current_volume > sample_metrics_result.volume.average_volume
        assert sample_metrics_result.volume.volume_ratio > 1.0
        assert sample_metrics_result.volume.unusual_volume is True

        # Проверка моментума
        assert sample_metrics_result.momentum.rsi > 50
        assert sample_metrics_result.momentum.macd > sample_metrics_result.momentum.macd_signal
        assert sample_metrics_result.momentum.momentum_score > 0.5

        # Проверка ликвидности
        assert sample_metrics_result.liquidity.bid_ask_spread < 0.005
        assert sample_metrics_result.liquidity.market_depth > 500000.0
        assert sample_metrics_result.liquidity.liquidity_score > 0.5

        # Проверка стресса
        assert sample_metrics_result.stress.stress_index < 0.5
        assert sample_metrics_result.stress.fear_greed_index > 30
        assert sample_metrics_result.stress.stress_level == "low"


class TestIntegration:
    """Интеграционные тесты."""

    def test_market_analysis_workflow(self):
        """Тест workflow анализа рынка."""
        # Создание всех метрик
        volatility = VolatilityMetrics(
            current_volatility=0.3,
            historical_volatility=0.25,
            volatility_percentile=0.8,
            volatility_trend=VolatilityTrend.INCREASING,
        )

        trend = TrendMetrics(
            trend_direction=TrendDirection.UP,
            trend_strength=0.9,
            trend_confidence=0.95,
            support_level=48000.0,
            resistance_level=55000.0,
        )

        volume = VolumeMetrics(
            current_volume=2000000.0,
            average_volume=1200000.0,
            volume_trend=VolumeTrend.INCREASING,
            volume_ratio=1.67,
            unusual_volume=True,
        )

        momentum = MomentumMetrics(rsi=70.0, macd=0.8, macd_signal=0.4, macd_histogram=0.4, momentum_score=0.85)

        liquidity = LiquidityMetrics(
            bid_ask_spread=0.0008, market_depth=2000000.0, order_book_imbalance=0.05, liquidity_score=0.9
        )

        stress = MarketStressMetrics(
            stress_index=0.2, fear_greed_index=65.0, market_regime="bullish", stress_level="low"
        )

        # Создание комплексного результата
        result = MarketMetricsResult(
            volatility=volatility,
            trend=trend,
            volume=volume,
            momentum=momentum,
            liquidity=liquidity,
            stress=stress,
            timestamp="2024-01-01T12:00:00Z",
            symbol="BTC/USDT",
        )

        # Проверка комплексного анализа
        assert result.volatility.volatility_trend == VolatilityTrend.INCREASING
        assert result.trend.trend_direction == TrendDirection.UP
        assert result.volume.unusual_volume is True
        assert result.momentum.rsi > 70  # Перекупленность
        assert result.liquidity.liquidity_score > 0.8  # Высокая ликвидность
        assert result.stress.stress_level == "low"  # Низкий стресс

        # Проверка согласованности данных
        assert result.trend.trend_strength > 0.8  # Сильный тренд
        assert result.momentum.momentum_score > 0.8  # Высокий моментум
        assert result.volume.volume_ratio > 1.5  # Высокий объем

    def test_market_regime_detection(self):
        """Тест определения рыночного режима."""
        # Бычий рынок
        bullish_metrics = MarketMetricsResult(
            volatility=VolatilityMetrics(0.2, 0.2, 0.5, VolatilityTrend.STABLE),
            trend=TrendMetrics(TrendDirection.UP, 0.9, 0.95),
            volume=VolumeMetrics(1500000.0, 1000000.0, VolumeTrend.INCREASING, 1.5, True),
            momentum=MomentumMetrics(70.0, 0.5, 0.3, 0.2, 0.8),
            liquidity=LiquidityMetrics(0.001, 1000000.0, 0.1, 0.85),
            stress=MarketStressMetrics(0.2, 65.0, "bullish", "low"),
            symbol="BTC/USDT",
        )

        assert bullish_metrics.trend.trend_direction == TrendDirection.UP
        assert bullish_metrics.trend.trend_strength > 0.8
        assert bullish_metrics.stress.market_regime == "bullish"

        # Медвежий рынок
        bearish_metrics = MarketMetricsResult(
            volatility=VolatilityMetrics(0.4, 0.3, 0.9, VolatilityTrend.INCREASING),
            trend=TrendMetrics(TrendDirection.DOWN, 0.8, 0.9),
            volume=VolumeMetrics(1800000.0, 1000000.0, VolumeTrend.INCREASING, 1.8, True),
            momentum=MomentumMetrics(25.0, -0.5, -0.2, -0.3, 0.2),
            liquidity=LiquidityMetrics(0.005, 500000.0, 0.3, 0.4),
            stress=MarketStressMetrics(0.8, 25.0, "bearish", "high"),
            symbol="BTC/USDT",
        )

        assert bearish_metrics.trend.trend_direction == TrendDirection.DOWN
        assert bearish_metrics.momentum.rsi < 30  # Перепроданность
        assert bearish_metrics.stress.market_regime == "bearish"
        assert bearish_metrics.stress.stress_level == "high"
