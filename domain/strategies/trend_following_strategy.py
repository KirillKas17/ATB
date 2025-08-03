from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import StrategyType
from domain.strategies.strategy_interface import (
    StrategyAnalysisResult,
    StrategyInterface,
)
from domain.strategies.strategy_types import TrendFollowingParams
from domain.types import ConfidenceLevel, RiskLevel, StrategyId
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


class TrendFollowingStrategy(StrategyInterface):
    def __init__(
        self,
        strategy_id: StrategyId,
        name: str,
        trading_pairs: List[str],
        parameters: Dict[str, Any],
        risk_level: RiskLevel = RiskLevel(Decimal("0.5")),
        confidence_threshold: ConfidenceLevel = ConfidenceLevel(Decimal("0.6")),
    ):
        super().__init__(
            strategy_id=strategy_id,
            name=name,
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=trading_pairs,
            parameters=parameters,
            risk_level=risk_level,
            confidence_threshold=confidence_threshold,
        )
        self._params = TrendFollowingParams(**parameters)
        self._validate_trend_following_params()

    def _validate_trend_following_params(self) -> None:
        if self._params.short_period >= self._params.long_period:
            raise ValueError("Short period must be less than long period")
        if self._params.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if not (0 <= self._params.rsi_oversold <= 100):
            raise ValueError("RSI oversold must be between 0 and 100")
        if not (0 <= self._params.rsi_overbought <= 100):
            raise ValueError("RSI overbought must be between 0 and 100")

    def _perform_market_analysis(
        self, market_data: MarketData
    ) -> StrategyAnalysisResult:
        price_change = float((market_data.close.value - market_data.open.value) / market_data.open.value)
        trend_direction = (
            "up"
            if price_change > 0.01
            else "down" if price_change < -0.01 else "sideways"
        )
        trend_strength = min(abs(price_change) * 10, 1.0)
        volatility_level = float(
            (market_data.high.value - market_data.low.value) / market_data.open.value
        )
        volume_analysis = {"volume": float(market_data.volume.value)}
        technical_indicators = {
            "sma_short": float(market_data.close.value),
            "sma_long": float(market_data.close.value),
            "rsi": 50.0,
        }
        market_regime = self._determine_market_regime(market_data)
        risk_assessment = self._assess_risk(market_data)
        support_resistance = (None, None)
        momentum_indicators = {"momentum": price_change}
        pattern_recognition: List[str] = []
        market_sentiment = "neutral"
        confidence_score = self._calculate_confidence_score(market_data)
        return StrategyAnalysisResult(
            confidence_score=confidence_score,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            volume_analysis=volume_analysis,
            technical_indicators=technical_indicators,
            market_regime=market_regime,
            risk_assessment=risk_assessment,
            support_resistance=support_resistance,
            momentum_indicators=momentum_indicators,
            pattern_recognition=pattern_recognition,
            market_sentiment=market_sentiment,
        )

    def _generate_signal_by_type(
        self, market_data: MarketData, analysis: StrategyAnalysisResult
    ) -> Optional[Signal]:
        confidence_score = analysis.confidence_score
        trend_direction = analysis.trend_direction
        strength = (
            SignalStrength.VERY_STRONG
            if confidence_score > 0.8
            else (
                SignalStrength.STRONG
                if confidence_score > 0.7
                else (
                    SignalStrength.MEDIUM
                    if confidence_score > 0.6
                    else SignalStrength.WEAK
                )
            )
        )
        if trend_direction == "up" and confidence_score >= float(
            self._confidence_threshold
        ):
            return Signal(
                strategy_id=self._strategy_id,
                trading_pair=str(market_data.symbol),
                signal_type=SignalType.BUY,
                strength=strength,
                confidence=Decimal(str(confidence_score)),
                price=Money(Decimal(str(market_data.close.value)), Currency.USD),
                metadata={
                    "strategy_type": "trend_following",
                    "trend_direction": trend_direction,
                    "short_period": self._params.short_period,
                    "long_period": self._params.long_period,
                    "rsi_period": self._params.rsi_period,
                },
            )
        elif trend_direction == "down" and confidence_score >= float(
            self._confidence_threshold
        ):
            return Signal(
                strategy_id=self._strategy_id,
                trading_pair=str(market_data.symbol),
                signal_type=SignalType.SELL,
                strength=strength,
                confidence=Decimal(str(confidence_score)),
                price=Money(Decimal(str(market_data.close.value)), Currency.USD),
                metadata={
                    "strategy_type": "trend_following",
                    "trend_direction": trend_direction,
                    "short_period": self._params.short_period,
                    "long_period": self._params.long_period,
                    "rsi_period": self._params.rsi_period,
                },
            )
        return None
