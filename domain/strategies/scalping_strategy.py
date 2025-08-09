from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import StrategyType
from domain.strategies.strategy_interface import (
    StrategyAnalysisResult,
    StrategyInterface,
)
from domain.type_definitions.strategy_types import ScalpingParams
from domain.type_definitions import ConfidenceLevel, RiskLevel, StrategyId
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


class ScalpingStrategy(StrategyInterface):
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
            strategy_type=StrategyType.SCALPING,
            trading_pairs=trading_pairs,
            parameters=parameters,
            risk_level=risk_level,
            confidence_threshold=confidence_threshold,
        )
        self._params = ScalpingParams(**parameters)
        self._validate_scalping_params()

    def _validate_scalping_params(self) -> None:
        if self._params.profit_threshold <= 0:
            raise ValueError("Profit threshold must be positive")
        if self._params.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")

    def _perform_market_analysis(
        self, market_data: MarketData
    ) -> StrategyAnalysisResult:
        price_change = float((market_data.close.value - market_data.open.value) / market_data.open.value)
        volatility_level = float(
            (market_data.high.value - market_data.low.value) / market_data.open.value
        )
        confidence_score = self._calculate_confidence_score(market_data)
        trend_direction = (
            "scalp"
            if abs(price_change) < float(self._params.profit_threshold)
            else "wait"
        )
        trend_strength = abs(price_change)
        volume_analysis = {"volume": float(market_data.volume.value)}
        technical_indicators: Dict[str, Any] = {}
        market_regime = self._determine_market_regime(market_data)
        risk_assessment = self._assess_risk(market_data)
        support_resistance = (None, None)
        momentum_indicators = {"momentum": price_change}
        pattern_recognition: List[str] = []
        market_sentiment = "neutral"
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
        price_change = abs((market_data.close.value - market_data.open.value) / market_data.open.value)
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
        if price_change < float(
            self._params.profit_threshold
        ) and confidence_score >= float(self._confidence_threshold):
            return Signal(
                strategy_id=self._strategy_id,
                trading_pair=str(market_data.symbol),
                signal_type=SignalType.BUY,
                strength=strength,
                confidence=Decimal(str(confidence_score)),
                price=Money(market_data.close.value, Currency.USD),
                metadata={
                    "strategy_type": "scalping",
                    "profit_threshold": self._params.profit_threshold,
                },
            )
        return None
