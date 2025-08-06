"""
Signal Aggregator для ML Services в ATB Trading System.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from shared.numpy_utils import np

from domain.type_definitions.ml_types import ActionType, AggregatedSignal, SignalSource, SignalType, TradingSignal

# Настройка логирования
logger = logging.getLogger(__name__)


class SignalAggregator:
    """Агрегатор сигналов из различных источников"""
    
    def __init__(self) -> None:
        self.signal_weights = {
            SignalSource.TECHNICAL: 0.4,
            SignalSource.SENTIMENT: 0.3,
            SignalSource.NEWS: 0.2,
            SignalSource.VOLUME: 0.1
        }
    
    def aggregate_ensemble_signals(self, ensemble_predictions: List[Dict[str, Any]]) -> AggregatedSignal:
        """Агрегация сигналов от ансамбля моделей"""
        if not ensemble_predictions:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                symbol="UNKNOWN"
            )
        
        # Подсчет голосов
        votes = {"buy": 0, "sell": 0, "hold": 0}
        scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for pred in ensemble_predictions:
            action = pred.get("action", "hold")
            confidence = pred.get("confidence", 0.0)
            votes[action] += 1
            scores[action] += confidence
        
        # Нормализация скоров
        total_votes = sum(votes.values())
        if total_votes > 0:
            for key in scores:
                scores[key] = scores[key] / total_votes
        
        # Определение итогового действия
        action_str = max(scores, key=lambda k: scores[k])
        # Преобразование строки в enum
        if action_str == "buy":
            action = ActionType.BUY
        elif action_str == "sell":
            action = ActionType.SELL
        else:
            action = ActionType.HOLD
        confidence = float(scores[action_str])
        
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=ensemble_predictions[0].get("symbol", "UNKNOWN"),
            consensus_score=confidence
        )
    
    def aggregate_ml_predictions(self, predictions: List[float], weights: List[float]) -> AggregatedSignal:
        """Агрегация численных предсказаний ML моделей"""
        if not predictions or not weights:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                symbol="UNKNOWN"
            )
        
        # Взвешенное усреднение
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
            confidence = 1.0 - min(float(abs(weighted_pred)), 1.0)
        
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol="UNKNOWN"
        )
    
    def aggregate_classification_results(self, class_probs: List[Dict[str, float]]) -> AggregatedSignal:
        """Агрегация результатов классификации"""
        if not class_probs:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                symbol="UNKNOWN"
            )
        
        # Усреднение вероятностей
        avg_probs: Dict[str, List[float]] = {}
        for prob_dict in class_probs:
            for class_name, prob in prob_dict.items():
                if class_name not in avg_probs:
                    avg_probs[class_name] = []
                avg_probs[class_name].append(prob)
        
        # Вычисление средних вероятностей
        probs = {}
        for class_name, prob_list in avg_probs.items():
            probs[class_name] = sum(prob_list) / len(prob_list)
        
        # Определение итогового действия
        action_str = max(probs, key=lambda k: probs[k])
        # Преобразование строки в enum
        if action_str == "buy":
            action = ActionType.BUY
        elif action_str == "sell":
            action = ActionType.SELL
        else:
            action = ActionType.HOLD
        confidence = float(probs[action_str])
        
        return AggregatedSignal(
            action=action,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol="UNKNOWN",
            consensus_score=confidence
        )
