"""
Unit тесты для domain/types/session_types.py.

Покрывает:
- NewType определения
- Enum классы
- TypedDict классы
- Protocol классы
- Pydantic модели
"""

import pytest
from datetime import time
from unittest.mock import Mock

from domain.types.session_types import (
    SessionId, Symbol, VolumeMultiplier, VolatilityMultiplier,
    ConfidenceScore, CorrelationScore,
    DEFAULT_LOOKBACK_DAYS, DEFAULT_MIN_DATA_POINTS, DEFAULT_CONFIDENCE_THRESHOLD,
    SessionType, SessionPhase, MarketRegime, SessionIntensity,
    LiquidityProfile, InfluenceType, PriceDirection,
    SessionMetrics, MarketConditions, SessionTransition,
    SessionTimeProvider, SessionMetricsCalculator,
    SessionTimeWindow, SessionBehavior, SessionProfile,
    SessionAnalysisResult,
)
from domain.value_objects.timestamp import Timestamp


class TestNewTypeDefinitions:
    """Тесты для NewType определений."""

    def test_session_id_creation(self):
        """Тест создания SessionId."""
        session_str = "session_123"
        session_id = SessionId(session_str)
        assert session_id == session_str
        assert isinstance(session_id, str)

    def test_symbol_creation(self):
        """Тест создания Symbol."""
        symbol_str = "BTCUSDT"
        symbol = Symbol(symbol_str)
        assert symbol == symbol_str
        assert isinstance(symbol, str)

    def test_volume_multiplier_creation(self):
        """Тест создания VolumeMultiplier."""
        multiplier_float = 1.5
        volume_multiplier = VolumeMultiplier(multiplier_float)
        assert volume_multiplier == multiplier_float
        assert isinstance(volume_multiplier, float)

    def test_confidence_score_creation(self):
        """Тест создания ConfidenceScore."""
        score_float = 0.85
        confidence_score = ConfidenceScore(score_float)
        assert confidence_score == score_float
        assert isinstance(confidence_score, float)


class TestConstants:
    """Тесты для констант."""

    def test_default_lookback_days(self):
        """Тест DEFAULT_LOOKBACK_DAYS."""
        assert DEFAULT_LOOKBACK_DAYS == 30
        assert isinstance(DEFAULT_LOOKBACK_DAYS, int)

    def test_default_min_data_points(self):
        """Тест DEFAULT_MIN_DATA_POINTS."""
        assert DEFAULT_MIN_DATA_POINTS == 100
        assert isinstance(DEFAULT_MIN_DATA_POINTS, int)

    def test_default_confidence_threshold(self):
        """Тест DEFAULT_CONFIDENCE_THRESHOLD."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.6
        assert isinstance(DEFAULT_CONFIDENCE_THRESHOLD, float)


class TestEnumClasses:
    """Тесты для Enum классов."""

    def test_session_type_values(self):
        """Тест значений SessionType."""
        assert SessionType.ASIAN.value == "asian"
        assert SessionType.LONDON.value == "london"
        assert SessionType.NEW_YORK.value == "new_york"
        assert SessionType.GLOBAL.value == "global"
        assert SessionType.CRYPTO_24H.value == "crypto_24h"

    def test_session_phase_values(self):
        """Тест значений SessionPhase."""
        assert SessionPhase.PRE_OPENING.value == "pre_opening"
        assert SessionPhase.OPENING.value == "opening"
        assert SessionPhase.EARLY_SESSION.value == "early_session"
        assert SessionPhase.MID_SESSION.value == "mid_session"
        assert SessionPhase.LATE_SESSION.value == "late_session"
        assert SessionPhase.CLOSING.value == "closing"

    def test_market_regime_values(self):
        """Тест значений MarketRegime."""
        assert MarketRegime.TRENDING_BULL.value == "trending_bull"
        assert MarketRegime.TRENDING_BEAR.value == "trending_bear"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.SIDEWAYS.value == "sideways"

    def test_session_intensity_values(self):
        """Тест значений SessionIntensity."""
        assert SessionIntensity.EXTREMELY_LOW.value == "extremely_low"
        assert SessionIntensity.VERY_LOW.value == "very_low"
        assert SessionIntensity.LOW.value == "low"
        assert SessionIntensity.NORMAL.value == "normal"
        assert SessionIntensity.HIGH.value == "high"
        assert SessionIntensity.VERY_HIGH.value == "very_high"
        assert SessionIntensity.EXTREMELY_HIGH.value == "extremely_high"
        assert SessionIntensity.CRITICAL.value == "critical"

    def test_liquidity_profile_values(self):
        """Тест значений LiquidityProfile."""
        assert LiquidityProfile.EXCESSIVE.value == "excessive"
        assert LiquidityProfile.ABUNDANT.value == "abundant"
        assert LiquidityProfile.NORMAL.value == "normal"
        assert LiquidityProfile.TIGHT.value == "tight"
        assert LiquidityProfile.SCARCE.value == "scarce"
        assert LiquidityProfile.CRITICAL.value == "critical"
        assert LiquidityProfile.DRY.value == "dry"
        assert LiquidityProfile.FROZEN.value == "frozen"

    def test_influence_type_values(self):
        """Тест значений InfluenceType."""
        assert InfluenceType.VOLATILITY.value == "volatility"
        assert InfluenceType.VOLUME.value == "volume"
        assert InfluenceType.DIRECTION.value == "direction"
        assert InfluenceType.MOMENTUM.value == "momentum"
        assert InfluenceType.REVERSAL.value == "reversal"
        assert InfluenceType.BREAKOUT.value == "breakout"

    def test_price_direction_values(self):
        """Тест значений PriceDirection."""
        assert PriceDirection.BULLISH.value == "bullish"
        assert PriceDirection.BEARISH.value == "bearish"
        assert PriceDirection.NEUTRAL.value == "neutral"


class TestTypedDictClasses:
    """Тесты для TypedDict классов."""

    def test_session_metrics_creation(self):
        """Тест создания SessionMetrics."""
        metrics = SessionMetrics(
            volume_change_percent=15.5,
            volatility_change_percent=25.0,
            price_direction_bias=0.3,
            momentum_strength=0.8,
            false_breakout_probability=0.2,
            reversal_probability=0.15,
            trend_continuation_probability=0.7,
            influence_duration_minutes=120,
            peak_influence_time_minutes=60,
            spread_impact=0.1,
            liquidity_impact=0.05,
            correlation_with_other_sessions=0.6
        )
        assert metrics["volume_change_percent"] == 15.5
        assert metrics["volatility_change_percent"] == 25.0
        assert metrics["price_direction_bias"] == 0.3
        assert metrics["momentum_strength"] == 0.8
        assert metrics["false_breakout_probability"] == 0.2
        assert metrics["reversal_probability"] == 0.15
        assert metrics["trend_continuation_probability"] == 0.7
        assert metrics["influence_duration_minutes"] == 120
        assert metrics["peak_influence_time_minutes"] == 60
        assert metrics["spread_impact"] == 0.1
        assert metrics["liquidity_impact"] == 0.05
        assert metrics["correlation_with_other_sessions"] == 0.6

    def test_market_conditions_creation(self):
        """Тест создания MarketConditions."""
        conditions = MarketConditions(
            volatility=0.25,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.6,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING_BULL,
            session_intensity=SessionIntensity.HIGH
        )
        assert conditions["volatility"] == 0.25
        assert conditions["volume"] == 1000000.0
        assert conditions["spread"] == 0.001
        assert conditions["liquidity"] == 0.8
        assert conditions["momentum"] == 0.6
        assert conditions["trend_strength"] == 0.7
        assert conditions["market_regime"] == MarketRegime.TRENDING_BULL
        assert conditions["session_intensity"] == SessionIntensity.HIGH

    def test_session_transition_creation(self):
        """Тест создания SessionTransition."""
        transition = SessionTransition(
            from_session=SessionType.ASIAN,
            to_session=SessionType.LONDON,
            transition_duration_minutes=30,
            volume_decay_rate=0.5,
            volatility_spike_probability=0.3,
            gap_probability=0.1,
            correlation_shift_probability=0.2,
            liquidity_drain_rate=0.4,
            manipulation_window_minutes=15
        )
        assert transition["from_session"] == SessionType.ASIAN
        assert transition["to_session"] == SessionType.LONDON
        assert transition["transition_duration_minutes"] == 30
        assert transition["volume_decay_rate"] == 0.5
        assert transition["volatility_spike_probability"] == 0.3
        assert transition["gap_probability"] == 0.1
        assert transition["correlation_shift_probability"] == 0.2
        assert transition["liquidity_drain_rate"] == 0.4
        assert transition["manipulation_window_minutes"] == 15


class TestProtocolClasses:
    """Тесты для Protocol классов."""

    def test_session_time_provider_interface(self):
        """Тест интерфейса SessionTimeProvider."""
        mock_provider = Mock(spec=SessionTimeProvider)
        assert isinstance(mock_provider, SessionTimeProvider)
        assert hasattr(mock_provider, 'is_active')
        assert hasattr(mock_provider, 'get_phase')
        assert hasattr(mock_provider, 'get_duration')

    def test_session_metrics_calculator_interface(self):
        """Тест интерфейса SessionMetricsCalculator."""
        mock_calculator = Mock(spec=SessionMetricsCalculator)
        assert isinstance(mock_calculator, SessionMetricsCalculator)
        assert hasattr(mock_calculator, 'calculate_volume_impact')
        assert hasattr(mock_calculator, 'calculate_volatility_impact')
        assert hasattr(mock_calculator, 'calculate_direction_bias')


class TestPydanticModels:
    """Тесты для Pydantic моделей."""

    def test_session_time_window_creation(self):
        """Тест создания SessionTimeWindow."""
        time_window = SessionTimeWindow(
            start_time=time(9, 0),  # 09:00
            end_time=time(17, 0),   # 17:00
            timezone="UTC"
        )
        assert time_window.start_time == time(9, 0)
        assert time_window.end_time == time(17, 0)
        assert time_window.timezone == "UTC"
        assert time_window.overlap_start is None
        assert time_window.overlap_end is None

    def test_session_time_window_with_overlap(self):
        """Тест создания SessionTimeWindow с перекрытием."""
        time_window = SessionTimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            timezone="UTC",
            overlap_start=time(14, 0),
            overlap_end=time(16, 0)
        )
        assert time_window.overlap_start == time(14, 0)
        assert time_window.overlap_end == time(16, 0)

    def test_session_time_window_validation(self):
        """Тест валидации SessionTimeWindow."""
        with pytest.raises(ValueError, match="End time cannot be equal to start time"):
            SessionTimeWindow(
                start_time=time(9, 0),
                end_time=time(9, 0),
                timezone="UTC"
            )

    def test_session_time_window_is_active(self):
        """Тест метода is_active для SessionTimeWindow."""
        time_window = SessionTimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            timezone="UTC"
        )
        
        # Тест активности в пределах сессии
        assert time_window.is_active(time(10, 0)) is True
        assert time_window.is_active(time(15, 30)) is True
        assert time_window.is_active(time(9, 0)) is True
        assert time_window.is_active(time(17, 0)) is True
        
        # Тест неактивности вне сессии
        assert time_window.is_active(time(8, 0)) is False
        assert time_window.is_active(time(18, 0)) is False

    def test_session_time_window_get_phase(self):
        """Тест метода get_phase для SessionTimeWindow."""
        time_window = SessionTimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            timezone="UTC"
        )
        
        # Тест различных фаз
        assert time_window.get_phase(time(8, 0)) == SessionPhase.PRE_OPENING
        assert time_window.get_phase(time(9, 0)) == SessionPhase.OPENING
        assert time_window.get_phase(time(10, 0)) == SessionPhase.EARLY_SESSION
        assert time_window.get_phase(time(13, 0)) == SessionPhase.MID_SESSION
        assert time_window.get_phase(time(15, 0)) == SessionPhase.LATE_SESSION
        assert time_window.get_phase(time(16, 30)) == SessionPhase.CLOSING
        assert time_window.get_phase(time(17, 30)) == SessionPhase.POST_CLOSING

    def test_session_behavior_creation(self):
        """Тест создания SessionBehavior."""
        behavior = SessionBehavior(
            typical_volatility_spike_minutes=30,
            volume_peak_hours=[2, 4, 6],
            quiet_hours=[1, 5],
            avg_volume_multiplier=1.5,
            avg_volatility_multiplier=1.8,
            typical_direction_bias=0.2,
            common_patterns=["breakout", "reversal"],
            false_breakout_probability=0.3,
            reversal_probability=0.2,
            overlap_impact={"london": 0.8, "new_york": 0.9}
        )
        assert behavior.typical_volatility_spike_minutes == 30
        assert behavior.volume_peak_hours == [2, 4, 6]
        assert behavior.quiet_hours == [1, 5]
        assert behavior.avg_volume_multiplier == 1.5
        assert behavior.avg_volatility_multiplier == 1.8
        assert behavior.typical_direction_bias == 0.2
        assert behavior.common_patterns == ["breakout", "reversal"]
        assert behavior.false_breakout_probability == 0.3
        assert behavior.reversal_probability == 0.2
        assert behavior.overlap_impact == {"london": 0.8, "new_york": 0.9}

    def test_session_behavior_validation(self):
        """Тест валидации SessionBehavior."""
        # Тест валидных часов
        behavior = SessionBehavior(
            volume_peak_hours=[0, 12, 23],
            quiet_hours=[1, 5, 22]
        )
        assert behavior.volume_peak_hours == [0, 12, 23]
        assert behavior.quiet_hours == [1, 5, 22]

        # Тест невалидных часов
        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            SessionBehavior(volume_peak_hours=[25])

        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            SessionBehavior(quiet_hours=[-1])

    def test_session_profile_creation(self):
        """Тест создания SessionProfile."""
        time_window = SessionTimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            timezone="UTC"
        )
        behavior = SessionBehavior()
        
        profile = SessionProfile(
            session_type=SessionType.LONDON,
            time_window=time_window,
            behavior=behavior,
            description="London trading session",
            is_active=True,
            typical_volume_multiplier=1.5,
            typical_volatility_multiplier=1.8,
            typical_spread_multiplier=1.2,
            typical_direction_bias=0.1,
            liquidity_profile=LiquidityProfile.ABUNDANT,
            intensity_profile=SessionIntensity.HIGH,
            market_regime_tendency=MarketRegime.TRENDING_BULL,
            whale_activity_probability=0.2,
            mm_activity_probability=0.4,
            news_sensitivity=0.6,
            technical_signal_strength=0.8,
            fundamental_impact_multiplier=1.2,
            correlation_breakdown_probability=0.15,
            gap_probability=0.1,
            false_breakout_probability=0.25,
            reversal_probability=0.2,
            continuation_probability=0.7,
            manipulation_susceptibility=0.3
        )
        
        assert profile.session_type == SessionType.LONDON
        assert profile.time_window == time_window
        assert profile.behavior == behavior
        assert profile.description == "London trading session"
        assert profile.is_active is True
        assert profile.typical_volume_multiplier == 1.5
        assert profile.typical_volatility_multiplier == 1.8
        assert profile.typical_spread_multiplier == 1.2
        assert profile.typical_direction_bias == 0.1
        assert profile.liquidity_profile == LiquidityProfile.ABUNDANT
        assert profile.intensity_profile == SessionIntensity.HIGH
        assert profile.market_regime_tendency == MarketRegime.TRENDING_BULL
        assert profile.whale_activity_probability == 0.2
        assert profile.mm_activity_probability == 0.4
        assert profile.news_sensitivity == 0.6
        assert profile.technical_signal_strength == 0.8
        assert profile.fundamental_impact_multiplier == 1.2
        assert profile.correlation_breakdown_probability == 0.15
        assert profile.gap_probability == 0.1
        assert profile.false_breakout_probability == 0.25
        assert profile.reversal_probability == 0.2
        assert profile.continuation_probability == 0.7
        assert profile.manipulation_susceptibility == 0.3


class TestDataclassClasses:
    """Тесты для Dataclass классов."""

    def test_session_analysis_result_creation(self):
        """Тест создания SessionAnalysisResult."""
        timestamp = Timestamp.from_iso("2023-01-01T10:00:00Z")
        
        metrics = SessionMetrics(
            volume_change_percent=15.5,
            volatility_change_percent=25.0,
            price_direction_bias=0.3,
            momentum_strength=0.8,
            false_breakout_probability=0.2,
            reversal_probability=0.15,
            trend_continuation_probability=0.7,
            influence_duration_minutes=120,
            peak_influence_time_minutes=60,
            spread_impact=0.1,
            liquidity_impact=0.05,
            correlation_with_other_sessions=0.6
        )
        
        market_conditions = MarketConditions(
            volatility=0.25,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.6,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING_BULL,
            session_intensity=SessionIntensity.HIGH
        )
        
        predictions = {
            "price_movement": 0.6,
            "volume_increase": 0.8,
            "volatility_spike": 0.4
        }
        
        risk_factors = ["low_liquidity", "high_volatility"]
        
        result = SessionAnalysisResult(
            session_type=SessionType.LONDON,
            session_phase=SessionPhase.OPENING,
            timestamp=timestamp,
            confidence=ConfidenceScore(0.85),
            metrics=metrics,
            market_conditions=market_conditions,
            predictions=predictions,
            risk_factors=risk_factors
        )
        
        assert result.session_type == SessionType.LONDON
        assert result.session_phase == SessionPhase.OPENING
        assert result.timestamp == timestamp
        assert result.confidence == ConfidenceScore(0.85)
        assert result.metrics == metrics
        assert result.market_conditions == market_conditions
        assert result.predictions == predictions
        assert result.risk_factors == risk_factors

    def test_session_analysis_result_to_dict(self):
        """Тест метода to_dict для SessionAnalysisResult."""
        timestamp = Timestamp.from_iso("2023-01-01T10:00:00Z")
        
        metrics = SessionMetrics(
            volume_change_percent=15.5,
            volatility_change_percent=25.0,
            price_direction_bias=0.3,
            momentum_strength=0.8,
            false_breakout_probability=0.2,
            reversal_probability=0.15,
            trend_continuation_probability=0.7,
            influence_duration_minutes=120,
            peak_influence_time_minutes=60,
            spread_impact=0.1,
            liquidity_impact=0.05,
            correlation_with_other_sessions=0.6
        )
        
        market_conditions = MarketConditions(
            volatility=0.25,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.6,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING_BULL,
            session_intensity=SessionIntensity.HIGH
        )
        
        predictions = {"price_movement": 0.6}
        risk_factors = ["low_liquidity"]
        
        result = SessionAnalysisResult(
            session_type=SessionType.LONDON,
            session_phase=SessionPhase.OPENING,
            timestamp=timestamp,
            confidence=ConfidenceScore(0.85),
            metrics=metrics,
            market_conditions=market_conditions,
            predictions=predictions,
            risk_factors=risk_factors
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["session_type"] == "london"
        assert result_dict["session_phase"] == "opening"
        assert result_dict["timestamp"] == "2023-01-01T10:00:00Z"
        assert result_dict["confidence"] == 0.85
        assert isinstance(result_dict["metrics"], dict)
        assert isinstance(result_dict["market_conditions"], dict)
        assert result_dict["market_conditions"]["market_regime"] == "trending_bull"
        assert result_dict["market_conditions"]["session_intensity"] == "high"
        assert result_dict["predictions"] == {"price_movement": 0.6}
        assert result_dict["risk_factors"] == ["low_liquidity"]


class TestTypeValidation:
    """Тесты валидации типов."""

    def test_invalid_session_id(self):
        """Тест обработки невалидных SessionId."""
        with pytest.raises(TypeError):
            SessionId(123)
        
        with pytest.raises(TypeError):
            SessionId(None)

    def test_invalid_symbol(self):
        """Тест обработки невалидных Symbol."""
        with pytest.raises(TypeError):
            Symbol(123)
        
        with pytest.raises(TypeError):
            Symbol(None)

    def test_invalid_multipliers(self):
        """Тест обработки невалидных множителей."""
        with pytest.raises(TypeError):
            VolumeMultiplier("invalid")
        
        with pytest.raises(TypeError):
            VolatilityMultiplier(None)

    def test_invalid_scores(self):
        """Тест обработки невалидных оценок."""
        with pytest.raises(TypeError):
            ConfidenceScore("invalid")
        
        with pytest.raises(TypeError):
            CorrelationScore(None)


class TestTypeIntegration:
    """Интеграционные тесты типов."""

    def test_complete_session_analysis_flow(self):
        """Тест полного потока анализа сессии с типами."""
        # Создаем временное окно сессии
        time_window = SessionTimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            timezone="UTC"
        )
        
        # Создаем поведенческие характеристики
        behavior = SessionBehavior(
            typical_volatility_spike_minutes=30,
            volume_peak_hours=[2, 4, 6],
            quiet_hours=[1, 5],
            avg_volume_multiplier=1.5,
            avg_volatility_multiplier=1.8,
            typical_direction_bias=0.2
        )
        
        # Создаем профиль сессии
        profile = SessionProfile(
            session_type=SessionType.LONDON,
            time_window=time_window,
            behavior=behavior,
            description="London trading session",
            typical_volume_multiplier=1.5,
            typical_volatility_multiplier=1.8,
            typical_spread_multiplier=1.2,
            typical_direction_bias=0.1,
            liquidity_profile=LiquidityProfile.ABUNDANT,
            intensity_profile=SessionIntensity.HIGH,
            market_regime_tendency=MarketRegime.TRENDING_BULL
        )
        
        # Создаем базовые метрики
        base_metrics = SessionMetrics(
            volume_change_percent=10.0,
            volatility_change_percent=20.0,
            price_direction_bias=0.0,
            momentum_strength=0.5,
            false_breakout_probability=0.2,
            reversal_probability=0.1,
            trend_continuation_probability=0.6,
            influence_duration_minutes=120,
            peak_influence_time_minutes=60,
            spread_impact=0.05,
            liquidity_impact=0.1,
            correlation_with_other_sessions=0.8
        )
        
        # Создаем рыночные условия
        market_conditions = MarketConditions(
            volatility=0.25,
            volume=1000000.0,
            spread=0.001,
            liquidity=0.8,
            momentum=0.6,
            trend_strength=0.7,
            market_regime=MarketRegime.TRENDING_BULL,
            session_intensity=SessionIntensity.HIGH
        )
        
        # Рассчитываем влияние сессии
        session_impact = profile.calculate_session_impact(
            base_metrics, SessionPhase.OPENING, market_conditions
        )
        
        # Создаем результат анализа
        timestamp = Timestamp.from_iso("2023-01-01T10:00:00Z")
        predictions = {"price_movement": 0.6, "volume_increase": 0.8}
        risk_factors = ["low_liquidity", "high_volatility"]
        
        analysis_result = SessionAnalysisResult(
            session_type=SessionType.LONDON,
            session_phase=SessionPhase.OPENING,
            timestamp=timestamp,
            confidence=ConfidenceScore(0.85),
            metrics=session_impact,
            market_conditions=market_conditions,
            predictions=predictions,
            risk_factors=risk_factors
        )
        
        # Проверяем, что все типы работают корректно
        assert isinstance(profile.session_type, SessionType)
        assert isinstance(profile.time_window, SessionTimeWindow)
        assert isinstance(profile.behavior, SessionBehavior)
        assert isinstance(session_impact, dict)
        assert isinstance(analysis_result.session_type, SessionType)
        assert isinstance(analysis_result.session_phase, SessionPhase)
        assert isinstance(analysis_result.timestamp, Timestamp)
        assert isinstance(analysis_result.confidence, float)
        assert isinstance(analysis_result.metrics, dict)
        assert isinstance(analysis_result.market_conditions, dict)
        assert isinstance(analysis_result.predictions, dict)
        assert isinstance(analysis_result.risk_factors, list)

    def test_protocol_implementation_check(self):
        """Тест проверки реализации протоколов."""
        # Создаем мок объекты, реализующие протоколы
        mock_time_provider = Mock(spec=SessionTimeProvider)
        mock_metrics_calculator = Mock(spec=SessionMetricsCalculator)
        
        # Проверяем, что все объекты реализуют соответствующие протоколы
        assert isinstance(mock_time_provider, SessionTimeProvider)
        assert isinstance(mock_metrics_calculator, SessionMetricsCalculator) 