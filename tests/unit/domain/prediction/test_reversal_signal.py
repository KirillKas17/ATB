"""
Unit тесты для reversal_signal.py.

Покрывает:
- PivotPoint - точки разворота
- FibonacciLevel - уровни Фибоначчи
- VolumeProfile - профиль объема
- LiquidityCluster - кластеры ликвидности
- DivergenceSignal - сигналы дивергенции
- CandlestickPattern - свечные паттерны
- MomentumAnalysis - анализ импульса
- MeanReversionBand - полосы возврата к среднему
- ReversalSignal - сигналы разворота
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any

from domain.prediction.reversal_signal import (
    PivotPoint,
    FibonacciLevel,
    VolumeProfile,
    LiquidityCluster,
    DivergenceSignal,
    CandlestickPattern,
    MomentumAnalysis,
    MeanReversionBand,
    ReversalSignal,
)
from domain.type_definitions.prediction_types import (
    ReversalDirection,
    DivergenceType,
    ConfidenceScore,
    SignalStrengthScore,
    SignalStrength,
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp


class TestPivotPoint:
    """Тесты для PivotPoint."""

    def test_creation_valid(self: "TestPivotPoint") -> None:
        """Тест создания с валидными данными."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        pivot = PivotPoint(price=price, timestamp=timestamp, volume=1000.0, pivot_type="high", strength=0.8)

        assert pivot.price == price
        assert pivot.timestamp == timestamp
        assert pivot.volume == 1000.0
        assert pivot.pivot_type == "high"
        assert pivot.strength == 0.8
        assert pivot.confirmation_levels == []
        assert pivot.volume_cluster is None
        assert pivot.fibonacci_levels == []

    def test_creation_with_optional_fields(self: "TestPivotPoint") -> None:
        """Тест создания с опциональными полями."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        pivot = PivotPoint(
            price=price,
            timestamp=timestamp,
            volume=1000.0,
            pivot_type="low",
            strength=0.6,
            confirmation_levels=[0.5, 0.7],
            volume_cluster=500.0,
            fibonacci_levels=[0.382, 0.618],
        )

        assert pivot.confirmation_levels == [0.5, 0.7]
        assert pivot.volume_cluster == 500.0
        assert pivot.fibonacci_levels == [0.382, 0.618]

    def test_validation_strength_too_high(self: "TestPivotPoint") -> None:
        """Тест валидации - сила слишком высокая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Pivot strength must be between 0.0 and 1.0"):
            PivotPoint(price=price, timestamp=timestamp, volume=1000.0, pivot_type="high", strength=1.5)

    def test_validation_strength_too_low(self: "TestPivotPoint") -> None:
        """Тест валидации - сила слишком низкая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Pivot strength must be between 0.0 and 1.0"):
            PivotPoint(price=price, timestamp=timestamp, volume=1000.0, pivot_type="high", strength=-0.1)

    def test_validation_negative_volume(self: "TestPivotPoint") -> None:
        """Тест валидации - отрицательный объем."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Volume cannot be negative"):
            PivotPoint(price=price, timestamp=timestamp, volume=-100.0, pivot_type="high", strength=0.8)


class TestFibonacciLevel:
    """Тесты для FibonacciLevel."""

    def test_creation_valid(self: "TestFibonacciLevel") -> None:
        """Тест создания с валидными данными."""
        price = Price(Decimal("50000"), Currency("USD"))

        fib_level = FibonacciLevel(level=38.2, price=price, strength=0.7)

        assert fib_level.level == 38.2
        assert fib_level.price == price
        assert fib_level.strength == 0.7
        assert fib_level.volume_cluster is None
        assert fib_level.confluence_count == 0

    def test_creation_with_optional_fields(self: "TestFibonacciLevel") -> None:
        """Тест создания с опциональными полями."""
        price = Price(Decimal("50000"), Currency("USD"))

        fib_level = FibonacciLevel(level=61.8, price=price, strength=0.8, volume_cluster=500.0, confluence_count=3)

        assert fib_level.volume_cluster == 500.0
        assert fib_level.confluence_count == 3

    def test_validation_invalid_level(self: "TestFibonacciLevel") -> None:
        """Тест валидации - неверный уровень."""
        price = Price(Decimal("50000"), Currency("USD"))

        with pytest.raises(ValueError, match="Invalid Fibonacci level"):
            FibonacciLevel(level=25.0, price=price, strength=0.7)  # Неверный уровень

    def test_validation_strength_too_high(self: "TestFibonacciLevel") -> None:
        """Тест валидации - сила слишком высокая."""
        price = Price(Decimal("50000"), Currency("USD"))

        with pytest.raises(ValueError, match="Fibonacci strength must be between 0.0 and 1.0"):
            FibonacciLevel(level=38.2, price=price, strength=1.5)

    def test_validation_negative_confluence(self: "TestFibonacciLevel") -> None:
        """Тест валидации - отрицательное количество совпадений."""
        price = Price(Decimal("50000"), Currency("USD"))

        with pytest.raises(ValueError, match="Confluence count cannot be negative"):
            FibonacciLevel(level=38.2, price=price, strength=0.7, confluence_count=-1)

    def test_all_valid_levels(self: "TestFibonacciLevel") -> None:
        """Тест всех валидных уровней Фибоначчи."""
        price = Price(Decimal("50000"), Currency("USD"))
        valid_levels = [23.6, 38.2, 50.0, 61.8, 78.6]

        for level in valid_levels:
            fib_level = FibonacciLevel(level=level, price=price, strength=0.7)
            assert fib_level.level == level


class TestVolumeProfile:
    """Тесты для VolumeProfile."""

    def test_creation_valid(self: "TestVolumeProfile") -> None:
        """Тест создания с валидными данными."""
        price = Price(Decimal("50000"), Currency("USD"))
        poc_price = Price(Decimal("50100"), Currency("USD"))
        vah = Price(Decimal("50200"), Currency("USD"))
        val = Price(Decimal("49900"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        profile = VolumeProfile(
            price_level=price,
            volume_density=0.8,
            poc_price=poc_price,
            value_area_high=vah,
            value_area_low=val,
            timestamp=timestamp,
        )

        assert profile.price_level == price
        assert profile.volume_density == 0.8
        assert profile.poc_price == poc_price
        assert profile.value_area_high == vah
        assert profile.value_area_low == val
        assert profile.timestamp == timestamp
        assert profile.volume_nodes == []
        assert profile.imbalance_ratio == 0.0

    def test_creation_with_optional_fields(self: "TestVolumeProfile") -> None:
        """Тест создания с опциональными полями."""
        price = Price(Decimal("50000"), Currency("USD"))
        poc_price = Price(Decimal("50100"), Currency("USD"))
        vah = Price(Decimal("50200"), Currency("USD"))
        val = Price(Decimal("49900"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        volume_nodes = [{"price": 50000.0, "volume": 1000.0}]

        profile = VolumeProfile(
            price_level=price,
            volume_density=0.8,
            poc_price=poc_price,
            value_area_high=vah,
            value_area_low=val,
            timestamp=timestamp,
            volume_nodes=volume_nodes,
            imbalance_ratio=0.3,
        )

        assert profile.volume_nodes == volume_nodes
        assert profile.imbalance_ratio == 0.3

    def test_validation_negative_volume_density(self: "TestVolumeProfile") -> None:
        """Тест валидации - отрицательная плотность объема."""
        price = Price(Decimal("50000"), Currency("USD"))
        poc_price = Price(Decimal("50100"), Currency("USD"))
        vah = Price(Decimal("50200"), Currency("USD"))
        val = Price(Decimal("49900"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Volume density cannot be negative"):
            VolumeProfile(
                price_level=price,
                volume_density=-0.1,
                poc_price=poc_price,
                value_area_high=vah,
                value_area_low=val,
                timestamp=timestamp,
            )

    def test_validation_imbalance_ratio_too_high(self: "TestVolumeProfile") -> None:
        """Тест валидации - коэффициент дисбаланса слишком высокий."""
        price = Price(Decimal("50000"), Currency("USD"))
        poc_price = Price(Decimal("50100"), Currency("USD"))
        vah = Price(Decimal("50200"), Currency("USD"))
        val = Price(Decimal("49900"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Imbalance ratio must be between -1.0 and 1.0"):
            VolumeProfile(
                price_level=price,
                volume_density=0.8,
                poc_price=poc_price,
                value_area_high=vah,
                value_area_low=val,
                timestamp=timestamp,
                imbalance_ratio=1.5,
            )

    def test_validation_imbalance_ratio_too_low(self: "TestVolumeProfile") -> None:
        """Тест валидации - коэффициент дисбаланса слишком низкий."""
        price = Price(Decimal("50000"), Currency("USD"))
        poc_price = Price(Decimal("50100"), Currency("USD"))
        vah = Price(Decimal("50200"), Currency("USD"))
        val = Price(Decimal("49900"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Imbalance ratio must be between -1.0 and 1.0"):
            VolumeProfile(
                price_level=price,
                volume_density=0.8,
                poc_price=poc_price,
                value_area_high=vah,
                value_area_low=val,
                timestamp=timestamp,
                imbalance_ratio=-1.5,
            )


class TestLiquidityCluster:
    """Тесты для LiquidityCluster."""

    def test_creation_valid(self: "TestLiquidityCluster") -> None:
        """Тест создания с валидными данными."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        cluster = LiquidityCluster(
            price=price, volume=1000.0, side="bid", cluster_size=5, strength=0.8, timestamp=timestamp
        )

        assert cluster.price == price
        assert cluster.volume == 1000.0
        assert cluster.side == "bid"
        assert cluster.cluster_size == 5
        assert cluster.strength == 0.8
        assert cluster.timestamp == timestamp

    def test_validation_negative_volume(self: "TestLiquidityCluster") -> None:
        """Тест валидации - отрицательный объем."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Volume cannot be negative"):
            LiquidityCluster(price=price, volume=-100.0, side="bid", cluster_size=5, strength=0.8, timestamp=timestamp)

    def test_validation_zero_cluster_size(self: "TestLiquidityCluster") -> None:
        """Тест валидации - нулевой размер кластера."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Cluster size must be positive"):
            LiquidityCluster(price=price, volume=1000.0, side="bid", cluster_size=0, strength=0.8, timestamp=timestamp)

    def test_validation_strength_too_high(self: "TestLiquidityCluster") -> None:
        """Тест валидации - сила слишком высокая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Strength must be between 0.0 and 1.0"):
            LiquidityCluster(price=price, volume=1000.0, side="ask", cluster_size=5, strength=1.5, timestamp=timestamp)


class TestDivergenceSignal:
    """Тесты для DivergenceSignal."""

    def test_creation_valid(self: "TestDivergenceSignal") -> None:
        """Тест создания с валидными данными."""
        timestamp = Timestamp(datetime.now())

        signal = DivergenceSignal(
            type=DivergenceType.BEARISH_REGULAR,
            indicator="RSI",
            price_highs=[100.0, 110.0],
            price_lows=[],
            indicator_highs=[70.0, 65.0],
            indicator_lows=[],
            strength=0.8,
            confidence=0.7,
            timestamp=timestamp,
        )

        assert signal.type == DivergenceType.BEARISH_REGULAR
        assert signal.indicator == "RSI"
        assert signal.price_highs == [100.0, 110.0]
        assert signal.price_lows == []
        assert signal.indicator_highs == [70.0, 65.0]
        assert signal.indicator_lows == []
        assert signal.strength == 0.8
        assert signal.confidence == 0.7
        assert signal.timestamp == timestamp

    def test_validation_strength_too_high(self: "TestDivergenceSignal") -> None:
        """Тест валидации - сила слишком высокая."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Strength must be between 0.0 and 1.0"):
            DivergenceSignal(
                type=DivergenceType.BEARISH_REGULAR,
                indicator="RSI",
                price_highs=[100.0, 110.0],
                price_lows=[],
                indicator_highs=[70.0, 65.0],
                indicator_lows=[],
                strength=1.5,
                confidence=0.7,
                timestamp=timestamp,
            )

    def test_validation_confidence_too_low(self: "TestDivergenceSignal") -> None:
        """Тест валидации - уверенность слишком низкая."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DivergenceSignal(
                type=DivergenceType.BEARISH_REGULAR,
                indicator="RSI",
                price_highs=[100.0, 110.0],
                price_lows=[],
                indicator_highs=[70.0, 65.0],
                indicator_lows=[],
                strength=0.8,
                confidence=-0.1,
                timestamp=timestamp,
            )

    def test_validation_empty_indicator(self: "TestDivergenceSignal") -> None:
        """Тест валидации - пустой индикатор."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Indicator cannot be empty"):
            DivergenceSignal(
                type=DivergenceType.BEARISH_REGULAR,
                indicator="",
                price_highs=[100.0, 110.0],
                price_lows=[],
                indicator_highs=[70.0, 65.0],
                indicator_lows=[],
                strength=0.8,
                confidence=0.7,
                timestamp=timestamp,
            )


class TestCandlestickPattern:
    """Тесты для CandlestickPattern."""

    def test_creation_valid(self: "TestCandlestickPattern") -> None:
        """Тест создания с валидными данными."""
        timestamp = Timestamp(datetime.now())

        pattern = CandlestickPattern(
            name="hammer",
            direction=ReversalDirection.BULLISH,
            strength=0.7,
            confirmation_level=0.6,
            volume_confirmation=True,
            timestamp=timestamp,
        )

        assert pattern.name == "hammer"
        assert pattern.direction == ReversalDirection.BULLISH
        assert pattern.strength == 0.7
        assert pattern.confirmation_level == 0.6
        assert pattern.volume_confirmation is True
        assert pattern.timestamp == timestamp
        assert pattern.metadata == {}

    def test_creation_with_metadata(self: "TestCandlestickPattern") -> None:
        """Тест создания с метаданными."""
        timestamp = Timestamp(datetime.now())
        metadata = {"body_ratio": 0.3, "shadow_ratio": 0.7}

        pattern = CandlestickPattern(
            name="doji",
            direction=ReversalDirection.NEUTRAL,
            strength=0.5,
            confirmation_level=0.4,
            volume_confirmation=False,
            timestamp=timestamp,
            metadata=metadata,
        )

        assert pattern.metadata == metadata

    def test_validation_empty_name(self: "TestCandlestickPattern") -> None:
        """Тест валидации - пустое имя."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Pattern name cannot be empty"):
            CandlestickPattern(
                name="",
                direction=ReversalDirection.BULLISH,
                strength=0.7,
                confirmation_level=0.6,
                volume_confirmation=True,
                timestamp=timestamp,
            )

    def test_validation_strength_too_high(self: "TestCandlestickPattern") -> None:
        """Тест валидации - сила слишком высокая."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Strength must be between 0.0 and 1.0"):
            CandlestickPattern(
                name="hammer",
                direction=ReversalDirection.BULLISH,
                strength=1.5,
                confirmation_level=0.6,
                volume_confirmation=True,
                timestamp=timestamp,
            )

    def test_validation_confirmation_level_too_low(self: "TestCandlestickPattern") -> None:
        """Тест валидации - уровень подтверждения слишком низкий."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Confirmation level must be between 0.0 and 1.0"):
            CandlestickPattern(
                name="hammer",
                direction=ReversalDirection.BULLISH,
                strength=0.7,
                confirmation_level=-0.1,
                volume_confirmation=True,
                timestamp=timestamp,
            )


class TestMomentumAnalysis:
    """Тесты для MomentumAnalysis."""

    def test_creation_valid(self: "TestMomentumAnalysis") -> None:
        """Тест создания с валидными данными."""
        timestamp = Timestamp(datetime.now())

        analysis = MomentumAnalysis(
            timestamp=timestamp,
            momentum_loss=0.2,
            velocity_change=0.1,
            acceleration=0.05,
            volume_momentum=0.3,
            price_momentum=0.15,
        )

        assert analysis.timestamp == timestamp
        assert analysis.momentum_loss == 0.2
        assert analysis.velocity_change == 0.1
        assert analysis.acceleration == 0.05
        assert analysis.volume_momentum == 0.3
        assert analysis.price_momentum == 0.15
        assert analysis.momentum_divergence is None

    def test_creation_with_divergence(self: "TestMomentumAnalysis") -> None:
        """Тест создания с дивергенцией."""
        timestamp = Timestamp(datetime.now())

        analysis = MomentumAnalysis(
            timestamp=timestamp,
            momentum_loss=0.2,
            velocity_change=0.1,
            acceleration=0.05,
            volume_momentum=0.3,
            price_momentum=0.15,
            momentum_divergence=0.4,
        )

        assert analysis.momentum_divergence == 0.4

    def test_validation_non_numeric_momentum_loss(self: "TestMomentumAnalysis") -> None:
        """Тест валидации - нечисловое значение потери импульса."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Momentum loss must be numeric"):
            MomentumAnalysis(
                timestamp=timestamp,
                momentum_loss="invalid",
                velocity_change=0.1,
                acceleration=0.05,
                volume_momentum=0.3,
                price_momentum=0.15,
            )

    def test_validation_non_numeric_velocity_change(self: "TestMomentumAnalysis") -> None:
        """Тест валидации - нечисловое изменение скорости."""
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Velocity change must be numeric"):
            MomentumAnalysis(
                timestamp=timestamp,
                momentum_loss=0.2,
                velocity_change="invalid",
                acceleration=0.05,
                volume_momentum=0.3,
                price_momentum=0.15,
            )


class TestMeanReversionBand:
    """Тесты для MeanReversionBand."""

    def test_creation_valid(self: "TestMeanReversionBand") -> None:
        """Тест создания с валидными данными."""
        upper_band = Price(Decimal("51000"), Currency("USD"))
        lower_band = Price(Decimal("49000"), Currency("USD"))
        middle_line = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        band = MeanReversionBand(
            upper_band=upper_band,
            lower_band=lower_band,
            middle_line=middle_line,
            deviation=0.02,
            band_width=2000.0,
            current_position=0.5,
            timestamp=timestamp,
        )

        assert band.upper_band == upper_band
        assert band.lower_band == lower_band
        assert band.middle_line == middle_line
        assert band.deviation == 0.02
        assert band.band_width == 2000.0
        assert band.current_position == 0.5
        assert band.timestamp == timestamp

    def test_validation_negative_deviation(self: "TestMeanReversionBand") -> None:
        """Тест валидации - отрицательное отклонение."""
        upper_band = Price(Decimal("51000"), Currency("USD"))
        lower_band = Price(Decimal("49000"), Currency("USD"))
        middle_line = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Deviation cannot be negative"):
            MeanReversionBand(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                deviation=-0.01,
                band_width=2000.0,
                current_position=0.5,
                timestamp=timestamp,
            )

    def test_validation_negative_band_width(self: "TestMeanReversionBand") -> None:
        """Тест валидации - отрицательная ширина полосы."""
        upper_band = Price(Decimal("51000"), Currency("USD"))
        lower_band = Price(Decimal("49000"), Currency("USD"))
        middle_line = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Band width cannot be negative"):
            MeanReversionBand(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                deviation=0.02,
                band_width=-100.0,
                current_position=0.5,
                timestamp=timestamp,
            )

    def test_validation_current_position_too_high(self: "TestMeanReversionBand") -> None:
        """Тест валидации - текущая позиция слишком высокая."""
        upper_band = Price(Decimal("51000"), Currency("USD"))
        lower_band = Price(Decimal("49000"), Currency("USD"))
        middle_line = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Current position must be between 0.0 and 1.0"):
            MeanReversionBand(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                deviation=0.02,
                band_width=2000.0,
                current_position=1.5,
                timestamp=timestamp,
            )


class TestReversalSignal:
    """Тесты для ReversalSignal."""

    @pytest.fixture
    def sample_signal(self) -> ReversalSignal:
        """Тестовый сигнал разворота."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        return ReversalSignal(
            symbol="BTC/USDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=price,
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.7),
            timestamp=timestamp,
        )

    def test_creation_valid(self, sample_signal: ReversalSignal) -> None:
        """Тест создания с валидными данными."""
        assert sample_signal.symbol == "BTC/USDT"
        assert sample_signal.direction == ReversalDirection.BULLISH
        assert sample_signal.pivot_price.value == Decimal("50000")
        assert sample_signal.confidence == ConfidenceScore(0.8)
        assert sample_signal.horizon == timedelta(hours=4)
        assert sample_signal.signal_strength == SignalStrengthScore(0.7)
        assert sample_signal.is_controversial is False
        assert sample_signal.agreement_score == 0.0
        assert sample_signal.pivot_points == []
        assert sample_signal.fibonacci_levels == []
        assert sample_signal.liquidity_clusters == []
        assert sample_signal.divergence_signals == []
        assert sample_signal.candlestick_patterns == []
        assert sample_signal.controversy_reasons == []
        assert sample_signal.analysis_metadata == {}
        assert sample_signal.risk_metrics == {}

    def test_validation_empty_symbol(self: "TestReversalSignal") -> None:
        """Тест валидации - пустой символ."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            ReversalSignal(
                symbol="",
                direction=ReversalDirection.BULLISH,
                pivot_price=price,
                confidence=ConfidenceScore(0.8),
                horizon=timedelta(hours=4),
                signal_strength=SignalStrengthScore(0.7),
                timestamp=timestamp,
            )

    def test_validation_confidence_too_high(self: "TestReversalSignal") -> None:
        """Тест валидации - уверенность слишком высокая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ReversalSignal(
                symbol="BTC/USDT",
                direction=ReversalDirection.BULLISH,
                pivot_price=price,
                confidence=ConfidenceScore(1.5),
                horizon=timedelta(hours=4),
                signal_strength=SignalStrengthScore(0.7),
                timestamp=timestamp,
            )

    def test_validation_signal_strength_too_low(self: "TestReversalSignal") -> None:
        """Тест валидации - сила сигнала слишком низкая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Signal strength must be between 0.0 and 1.0"):
            ReversalSignal(
                symbol="BTC/USDT",
                direction=ReversalDirection.BULLISH,
                pivot_price=price,
                confidence=ConfidenceScore(0.8),
                horizon=timedelta(hours=4),
                signal_strength=SignalStrengthScore(-0.1),
                timestamp=timestamp,
            )

    def test_validation_agreement_score_too_high(self: "TestReversalSignal") -> None:
        """Тест валидации - оценка согласованности слишком высокая."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Agreement score must be between 0.0 and 1.0"):
            ReversalSignal(
                symbol="BTC/USDT",
                direction=ReversalDirection.BULLISH,
                pivot_price=price,
                confidence=ConfidenceScore(0.8),
                horizon=timedelta(hours=4),
                signal_strength=SignalStrengthScore(0.7),
                timestamp=timestamp,
                agreement_score=1.5,
            )

    def test_validation_zero_horizon(self: "TestReversalSignal") -> None:
        """Тест валидации - нулевой горизонт."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        with pytest.raises(ValueError, match="Horizon must be positive"):
            ReversalSignal(
                symbol="BTC/USDT",
                direction=ReversalDirection.BULLISH,
                pivot_price=price,
                confidence=ConfidenceScore(0.8),
                horizon=timedelta(0),
                signal_strength=SignalStrengthScore(0.7),
                timestamp=timestamp,
            )

    def test_strength_category_weak(self, sample_signal: ReversalSignal) -> None:
        """Тест категории силы сигнала - слабый."""
        sample_signal.signal_strength = SignalStrengthScore(0.2)
        assert sample_signal.strength_category == SignalStrength.WEAK

    def test_strength_category_moderate(self, sample_signal: ReversalSignal) -> None:
        """Тест категории силы сигнала - умеренный."""
        sample_signal.signal_strength = SignalStrengthScore(0.5)
        assert sample_signal.strength_category == SignalStrength.MODERATE

    def test_strength_category_strong(self, sample_signal: ReversalSignal) -> None:
        """Тест категории силы сигнала - сильный."""
        sample_signal.signal_strength = SignalStrengthScore(0.7)
        assert sample_signal.strength_category == SignalStrength.STRONG

    def test_strength_category_very_strong(self, sample_signal: ReversalSignal) -> None:
        """Тест категории силы сигнала - очень сильный."""
        sample_signal.signal_strength = SignalStrengthScore(0.9)
        assert sample_signal.strength_category == SignalStrength.VERY_STRONG

    def test_is_expired_false(self, sample_signal: ReversalSignal) -> None:
        """Тест истечения срока - не истек."""
        assert sample_signal.is_expired is False

    def test_is_expired_true(self: "TestReversalSignal") -> None:
        """Тест истечения срока - истек."""
        price = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now() - timedelta(hours=5))  # 5 часов назад

        signal = ReversalSignal(
            symbol="BTC/USDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=price,
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),  # Горизонт 4 часа
            signal_strength=SignalStrengthScore(0.7),
            timestamp=timestamp,
        )

        assert signal.is_expired is True

    def test_time_to_expiry(self, sample_signal: ReversalSignal) -> None:
        """Тест времени до истечения."""
        time_to_expiry = sample_signal.time_to_expiry
        assert isinstance(time_to_expiry, timedelta)
        assert time_to_expiry.total_seconds() > 0

    def test_risk_level_low(self, sample_signal: ReversalSignal) -> None:
        """Тест уровня риска - низкий."""
        sample_signal.confidence = ConfidenceScore(0.9)
        sample_signal.agreement_score = 0.9
        sample_signal.signal_strength = SignalStrengthScore(0.9)
        assert sample_signal.risk_level == "low"

    def test_risk_level_medium(self, sample_signal: ReversalSignal) -> None:
        """Тест уровня риска - средний."""
        sample_signal.confidence = ConfidenceScore(0.6)
        sample_signal.agreement_score = 0.5
        sample_signal.signal_strength = SignalStrengthScore(0.5)
        assert sample_signal.risk_level == "medium"

    def test_risk_level_high(self, sample_signal: ReversalSignal) -> None:
        """Тест уровня риска - высокий."""
        sample_signal.confidence = ConfidenceScore(0.3)
        sample_signal.agreement_score = 0.2
        sample_signal.signal_strength = SignalStrengthScore(0.2)
        sample_signal.is_controversial = True
        assert sample_signal.risk_level == "high"

    def test_enhance_confidence(self, sample_signal: ReversalSignal) -> None:
        """Тест усиления уверенности."""
        original_confidence = float(sample_signal.confidence)
        sample_signal.enhance_confidence(0.2)

        assert float(sample_signal.confidence) > original_confidence
        assert float(sample_signal.confidence) <= 1.0

    def test_enhance_confidence_invalid_factor(self, sample_signal: ReversalSignal) -> None:
        """Тест усиления уверенности с неверным фактором."""
        with pytest.raises(ValueError, match="Enhancement factor must be between 0.0 and 1.0"):
            sample_signal.enhance_confidence(1.5)

    def test_reduce_confidence(self, sample_signal: ReversalSignal) -> None:
        """Тест снижения уверенности."""
        original_confidence = float(sample_signal.confidence)
        sample_signal.reduce_confidence(0.2)

        assert float(sample_signal.confidence) < original_confidence
        assert float(sample_signal.confidence) >= 0.0

    def test_reduce_confidence_invalid_factor(self, sample_signal: ReversalSignal) -> None:
        """Тест снижения уверенности с неверным фактором."""
        with pytest.raises(ValueError, match="Reduction factor must be between 0.0 and 1.0"):
            sample_signal.reduce_confidence(1.5)

    def test_mark_controversial(self, sample_signal: ReversalSignal) -> None:
        """Тест пометки как спорного."""
        sample_signal.mark_controversial("Conflicting indicators", {"rsi": 30, "macd": 0.5})

        assert sample_signal.is_controversial is True
        assert len(sample_signal.controversy_reasons) == 1
        assert sample_signal.controversy_reasons[0]["reason"] == "Conflicting indicators"
        assert sample_signal.controversy_reasons[0]["details"]["rsi"] == 30

    def test_mark_controversial_empty_reason(self, sample_signal: ReversalSignal) -> None:
        """Тест пометки как спорного с пустой причиной."""
        with pytest.raises(ValueError, match="Controversy reason cannot be empty"):
            sample_signal.mark_controversial("")

    def test_update_agreement_score(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления оценки согласованности."""
        sample_signal.update_agreement_score(0.8)

        assert sample_signal.agreement_score == 0.8

    def test_update_agreement_score_invalid(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления оценки согласованности с неверным значением."""
        with pytest.raises(ValueError, match="Agreement score must be between 0.0 and 1.0"):
            sample_signal.update_agreement_score(1.5)

    def test_add_divergence_signal(self, sample_signal: ReversalSignal) -> None:
        """Тест добавления сигнала дивергенции."""
        timestamp = Timestamp(datetime.now())
        divergence = DivergenceSignal(
            type=DivergenceType.BEARISH_REGULAR,
            indicator="RSI",
            price_highs=[100.0, 110.0],
            price_lows=[],
            indicator_highs=[70.0, 65.0],
            indicator_lows=[],
            strength=0.8,
            confidence=0.7,
            timestamp=timestamp,
        )

        sample_signal.add_divergence_signal(divergence)

        assert len(sample_signal.divergence_signals) == 1
        assert sample_signal.divergence_signals[0] == divergence

    def test_add_divergence_signal_invalid_type(self, sample_signal: ReversalSignal) -> None:
        """Тест добавления сигнала дивергенции с неверным типом."""
        with pytest.raises(TypeError, match="Expected DivergenceSignal"):
            sample_signal.add_divergence_signal("invalid")

    def test_add_candlestick_pattern(self, sample_signal: ReversalSignal) -> None:
        """Тест добавления свечного паттерна."""
        timestamp = Timestamp(datetime.now())
        pattern = CandlestickPattern(
            name="hammer",
            direction=ReversalDirection.BULLISH,
            strength=0.7,
            confirmation_level=0.6,
            volume_confirmation=True,
            timestamp=timestamp,
        )

        sample_signal.add_candlestick_pattern(pattern)

        assert len(sample_signal.candlestick_patterns) == 1
        assert sample_signal.candlestick_patterns[0] == pattern

    def test_add_candlestick_pattern_invalid_type(self, sample_signal: ReversalSignal) -> None:
        """Тест добавления свечного паттерна с неверным типом."""
        with pytest.raises(TypeError, match="Expected CandlestickPattern"):
            sample_signal.add_candlestick_pattern("invalid")

    def test_update_momentum_analysis(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления анализа импульса."""
        timestamp = Timestamp(datetime.now())
        momentum = MomentumAnalysis(
            timestamp=timestamp,
            momentum_loss=0.2,
            velocity_change=0.1,
            acceleration=0.05,
            volume_momentum=0.3,
            price_momentum=0.15,
        )

        sample_signal.update_momentum_analysis(momentum)

        assert sample_signal.momentum_analysis == momentum

    def test_update_momentum_analysis_invalid_type(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления анализа импульса с неверным типом."""
        with pytest.raises(TypeError, match="Expected MomentumAnalysis"):
            sample_signal.update_momentum_analysis("invalid")

    def test_update_mean_reversion_band(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления полосы возврата к среднему."""
        upper_band = Price(Decimal("51000"), Currency("USD"))
        lower_band = Price(Decimal("49000"), Currency("USD"))
        middle_line = Price(Decimal("50000"), Currency("USD"))
        timestamp = Timestamp(datetime.now())

        band = MeanReversionBand(
            upper_band=upper_band,
            lower_band=lower_band,
            middle_line=middle_line,
            deviation=0.02,
            band_width=2000.0,
            current_position=0.5,
            timestamp=timestamp,
        )

        sample_signal.update_mean_reversion_band(band)

        assert sample_signal.mean_reversion_band == band

    def test_update_mean_reversion_band_invalid_type(self, sample_signal: ReversalSignal) -> None:
        """Тест обновления полосы возврата к среднему с неверным типом."""
        with pytest.raises(TypeError, match="Expected MeanReversionBand"):
            sample_signal.update_mean_reversion_band("invalid")

    def test_to_dict(self, sample_signal: ReversalSignal) -> None:
        """Тест преобразования в словарь."""
        result = sample_signal.to_dict()

        assert isinstance(result, dict)
        assert result["symbol"] == "BTC/USDT"
        assert result["direction"] == ReversalDirection.BULLISH.value
        assert result["confidence"] == ConfidenceScore(0.8)
        assert result["signal_strength"] == SignalStrengthScore(0.7)
        assert result["is_controversial"] is False
        assert result["agreement_score"] == 0.0
        assert result["pivot_points_count"] == 0
        assert result["fibonacci_levels_count"] == 0
        assert result["divergence_signals_count"] == 0
        assert result["candlestick_patterns_count"] == 0
        assert result["liquidity_clusters_count"] == 0
        assert result["controversy_reasons_count"] == 0

    def test_str_representation(self, sample_signal: ReversalSignal) -> None:
        """Тест строкового представления."""
        result = str(sample_signal)

        assert "ReversalSignal" in result
        assert "BTC/USDT" in result
        assert "BULLISH" in result
        assert "50000.00" in result

    def test_repr_representation(self, sample_signal: ReversalSignal) -> None:
        """Тест представления для отладки."""
        result = repr(sample_signal)

        assert result == str(sample_signal)
