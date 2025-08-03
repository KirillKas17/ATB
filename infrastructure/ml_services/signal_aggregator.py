"""
Signal Aggregator для ML Services в ATB Trading System.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

from domain.types.ml_types import AggregatedSignal, SignalSource, SignalType


class ActionType(Enum):
    """Типы торговых действий."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalAggregator:
    """Агрегатор сигналов из различных источников"""

    def __init__(self, method: str = "weighted_voting") -> None:
        self.method = method
        self.signal_history: List[AggregatedSignal] = []
        self.source_weights: Dict[str, float] = {}
        self.source_performance: Dict[str, float] = {}

    def aggregate_signals(self, signals: List[SignalSource]) -> AggregatedSignal:
        """Агрегация сигналов из различных источников"""
        if not signals:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                risk_score=1.0,
                sources=[],
                timestamp=datetime.now(),
                explanation="No signals available",
            )
        if self.method == "weighted_voting":
            return self._weighted_voting(signals)
        elif self.method == "ensemble":
            return self._ensemble_aggregation(signals)
        elif self.method == "bayesian":
            return self._bayesian_aggregation(signals)
        else:
            return self._weighted_voting(signals)

    def _weighted_voting(self, signals: List[SignalSource]) -> AggregatedSignal:
        """Взвешенное голосование"""
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        for signal in signals:
            weight = signal.weight * signal.confidence
            total_weight += weight
            if signal.signal_type == SignalType.BUY:
                buy_score += weight
            elif signal.signal_type == SignalType.SELL:
                sell_score += weight
            else:
                hold_score += weight
        if total_weight == 0:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                risk_score=1.0,
                sources=signals,
                timestamp=datetime.now(),
                explanation="No valid signals",
            )
        # Нормализация
        buy_score /= total_weight
        sell_score /= total_weight
        hold_score /= total_weight
        # Определение действия
        scores = {"buy": buy_score, "sell": sell_score, "hold": hold_score}
        action_str = max(scores, key=lambda k: scores[k])
        # Преобразование строки в enum
        if action_str == "buy":
            action = ActionType.BUY
        elif action_str == "sell":
            action = ActionType.SELL
        else:
            action = ActionType.HOLD
        confidence = float(scores[action_str])
        # Расчет риска
        risk_score = 1.0 - confidence
        explanation = f"Aggregated from {len(signals)} sources: buy({buy_score:.3f}), sell({sell_score:.3f}), hold({hold_score:.3f})"
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            risk_score=risk_score,
            sources=signals,
            timestamp=datetime.now(),
            explanation=explanation,
        )

    def _ensemble_aggregation(self, signals: List[SignalSource]) -> AggregatedSignal:
        """Ансамблевая агрегация"""
        # Использование ансамбля моделей для агрегации
        predictions = []
        weights = []
        for signal in signals:
            predictions.append(
                1
                if signal.signal_type == SignalType.BUY
                else (-1 if signal.signal_type == SignalType.SELL else 0)
            )
            weights.append(signal.weight * signal.confidence)
        if not weights:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                risk_score=1.0,
                sources=signals,
                timestamp=datetime.now(),
                explanation="No valid signals",
            )
        # Взвешенное среднее
        weighted_pred = np.average(predictions, weights=weights)
        # Определение действия
        if weighted_pred > 0.3:
            action = ActionType.BUY
            confidence = min(float(weighted_pred), 1.0)
        elif weighted_pred < -0.3:
            action = ActionType.SELL
            confidence = min(float(abs(weighted_pred)), 1.0)
        else:
            action = ActionType.HOLD
            confidence = 1.0 - float(abs(weighted_pred))
        risk_score = 1.0 - confidence
        explanation = f"Ensemble prediction: {weighted_pred:.3f} -> {action}"
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            risk_score=risk_score,
            sources=signals,
            timestamp=datetime.now(),
            explanation=explanation,
        )

    def _bayesian_aggregation(self, signals: List[SignalSource]) -> AggregatedSignal:
        """Байесовская агрегация"""
        # Байесовское обновление вероятностей
        prior_buy = 0.33
        prior_sell = 0.33
        prior_hold = 0.34
        for signal in signals:
            likelihood = signal.confidence
            if signal.signal_type == SignalType.BUY:
                prior_buy *= likelihood
            elif signal.signal_type == SignalType.SELL:
                prior_sell *= likelihood
            else:
                prior_hold *= likelihood
        # Нормализация
        total = prior_buy + prior_sell + prior_hold
        if total > 0:
            prior_buy /= total
            prior_sell /= total
            prior_hold /= total
        # Определение действия
        probs = {"buy": prior_buy, "sell": prior_sell, "hold": prior_hold}
        action_str = max(probs, key=lambda k: probs[k])
        # Преобразование строки в enum
        if action_str == "buy":
            action = ActionType.BUY
        elif action_str == "sell":
            action = ActionType.SELL
        else:
            action = ActionType.HOLD
        confidence = float(probs[action_str])
        risk_score = 1.0 - confidence
        explanation = f"Bayesian probabilities: buy({prior_buy:.3f}), sell({prior_sell:.3f}), hold({prior_hold:.3f})"
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            risk_score=risk_score,
            sources=signals,
            timestamp=datetime.now(),
            explanation=explanation,
        )

    def update_source_performance(self, source_name: str, performance: float) -> None:
        """Обновление производительности источника"""
        self.source_performance[source_name] = performance
        # Адаптация весов на основе производительности
        if performance > 0.6:
            self.source_weights[source_name] = min(
                self.source_weights.get(source_name, 1.0) * 1.1, 2.0
            )
        elif performance < 0.4:
            self.source_weights[source_name] = max(
                self.source_weights.get(source_name, 1.0) * 0.9, 0.1
            )
