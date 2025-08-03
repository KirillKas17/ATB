"""
Эволюционный решатель решений.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvolvableDecisionReasonerConfig:
    """Конфигурация эволюционного решателя решений."""

    evolution_rate: float = 0.1
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.05
    max_decisions: int = 1000
    confidence_threshold: float = 0.7
    enable_auto_adaptation: bool = True


class EvolvableDecisionReasoner(ABC):
    """Абстрактный эволюционный решатель решений."""

    def __init__(self, config: Optional[EvolvableDecisionReasonerConfig] = None):
        self.config = config or EvolvableDecisionReasonerConfig()
        self.is_active: bool = False
        self.decision_history: List[Dict[str, Any]] = []
        self.adaptation_count: int = 0

    @abstractmethod
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Принятие решения с эволюционными улучшениями."""

    @abstractmethod
    async def adapt_reasoning_logic(self, performance_data: Dict[str, Any]) -> bool:
        """Адаптация логики рассуждений."""

    @abstractmethod
    async def evolve_decision_patterns(
        self, market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция паттернов принятия решений."""


class DefaultEvolvableDecisionReasoner(EvolvableDecisionReasoner):
    """Реализация эволюционного решателя решений по умолчанию."""

    def __init__(self, config: Optional[EvolvableDecisionReasonerConfig] = None):
        super().__init__(config)
        self.decision_patterns: Dict[str, float] = {}
        self.reasoning_weights: Dict[str, float] = {}
        self.market_condition_memory: List[Dict[str, Any]] = []

    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Принятие решения с эволюционными улучшениями."""
        try:
            # Анализ текущих рыночных условий
            market_conditions = self._analyze_market_conditions(context)
            # Адаптация логики рассуждений
            await self.adapt_reasoning_logic(market_conditions)
            # Эволюция паттернов принятия решений
            decision_patterns = await self.evolve_decision_patterns(market_conditions)
            # Принятие решения с учетом эволюционных улучшений
            decision = self._make_enhanced_decision(context, decision_patterns)
            # Обновление истории решений
            self._update_decision_history(decision)
            return {
                "decision": decision,
                "confidence": decision.get("confidence", 0.0),
                "reasoning": decision.get("reasoning", ""),
                "evolution_metrics": {
                    "adaptation_count": self.adaptation_count,
                    "pattern_confidence": decision_patterns.get("confidence", 0.0),
                    "market_conditions": market_conditions.get("summary", "unknown"),
                },
            }
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return {"error": str(e)}

    async def adapt_reasoning_logic(self, performance_data: Dict[str, Any]) -> bool:
        """Адаптация логики рассуждений."""
        try:
            current_performance = performance_data.get("performance", 0.0)
            # Проверяем, нужна ли адаптация
            if len(self.decision_history) > 0:
                recent_decisions = (
                    self.decision_history[-10:]
                    if len(self.decision_history) >= 10
                    else self.decision_history
                )
                avg_performance = sum(
                    d.get("performance", 0.0) for d in recent_decisions
                ) / len(recent_decisions)
                performance_delta = current_performance - avg_performance
                if abs(performance_delta) > self.config.adaptation_threshold:
                    # Выполняем адаптацию
                    self._perform_reasoning_adaptation(performance_delta)
                    self.adaptation_count += 1
                    return True
            return False
        except Exception as e:
            logger.error(f"Error adapting reasoning logic: {e}")
            return False

    async def evolve_decision_patterns(
        self, market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция паттернов принятия решений."""
        try:
            # Анализ текущего рыночного режима
            regime = market_conditions.get("regime", "unknown")
            volatility = market_conditions.get("volatility", 0.0)
            trend_strength = market_conditions.get("trend_strength", 0.0)
            # Эволюция паттернов принятия решений
            evolved_patterns = {
                "risk_assessment": self._evolve_risk_assessment(volatility),
                "position_sizing": self._evolve_position_sizing(trend_strength),
                "entry_timing": self._evolve_entry_timing(regime),
                "exit_strategy": self._evolve_exit_strategy(volatility),
                "confidence_calculation": self._evolve_confidence_calculation(regime),
            }
            return evolved_patterns
        except Exception as e:
            logger.error(f"Error evolving decision patterns: {e}")
            return {"error": str(e)}

    def _analyze_market_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ рыночных условий."""
        try:
            market_data = context.get("market_data", {})
            # Простой анализ рыночного режима
            prices = market_data.get("prices", [])
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                volatility = self._calculate_volatility(recent_prices)
                trend_strength = self._calculate_trend_strength(recent_prices)
                # Определение режима
                if volatility > 0.05:
                    regime = "volatile"
                elif trend_strength > 0.7:
                    regime = "trending"
                else:
                    regime = "ranging"
            else:
                volatility = 0.0
                trend_strength = 0.0
                regime = "unknown"
            return {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "timestamp": context.get("timestamp"),
                "summary": f"{regime}_{volatility:.3f}_{trend_strength:.3f}",
            }
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                "regime": "unknown",
                "volatility": 0.0,
                "trend_strength": 0.0,
                "summary": "error",
            }

    def _make_enhanced_decision(
        self, context: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Принятие улучшенного решения."""
        try:
            # Базовая логика принятия решений
            base_decision = self._make_base_decision(context)
            # Применяем эволюционные паттерны
            enhanced_decision = base_decision.copy()
            # Корректируем риск
            risk_assessment = patterns.get("risk_assessment", 1.0)
            enhanced_decision["risk_level"] = (
                base_decision.get("risk_level", 0.5) * risk_assessment
            )
            # Корректируем размер позиции
            position_sizing = patterns.get("position_sizing", 1.0)
            enhanced_decision["position_size"] = (
                base_decision.get("position_size", 0.1) * position_sizing
            )
            # Корректируем время входа
            entry_timing = patterns.get("entry_timing", "normal")
            enhanced_decision["entry_timing"] = entry_timing
            # Корректируем стратегию выхода
            exit_strategy = patterns.get("exit_strategy", "standard")
            enhanced_decision["exit_strategy"] = exit_strategy
            # Корректируем уверенность
            confidence_calc = patterns.get("confidence_calculation", 1.0)
            enhanced_decision["confidence"] = min(
                1.0, base_decision.get("confidence", 0.5) * confidence_calc
            )
            # Добавляем рассуждения
            enhanced_decision["reasoning"] = self._generate_reasoning(context, patterns)
            return enhanced_decision
        except Exception as e:
            logger.error(f"Error making enhanced decision: {e}")
            return {"error": str(e)}

    def _make_base_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Базовая логика принятия решений."""
        try:
            # Простая базовая логика
            market_data = context.get("market_data", {})
            prices = market_data.get("prices", [])
            if len(prices) < 2:
                return {
                    "action": "hold",
                    "confidence": 0.5,
                    "risk_level": 0.5,
                    "position_size": 0.1,
                    "reasoning": "insufficient_data",
                }
            current_price = prices[-1]
            previous_price = prices[-2]
            price_change = (current_price - previous_price) / previous_price
            if price_change > 0.02:
                action = "buy"
                confidence = min(0.9, 0.5 + abs(price_change) * 10)
            elif price_change < -0.02:
                action = "sell"
                confidence = min(0.9, 0.5 + abs(price_change) * 10)
            else:
                action = "hold"
                confidence = 0.5
            return {
                "action": action,
                "confidence": confidence,
                "risk_level": 0.5,
                "position_size": 0.1,
                "reasoning": f"price_change_{price_change:.3f}",
            }
        except Exception as e:
            logger.error(f"Error making base decision: {e}")
            return {"action": "hold", "confidence": 0.5, "error": str(e)}

    def _generate_reasoning(
        self, context: Dict[str, Any], patterns: Dict[str, Any]
    ) -> str:
        """Генерация рассуждений."""
        try:
            reasoning_parts = []
            # Добавляем анализ рыночных условий
            market_conditions = self._analyze_market_conditions(context)
            reasoning_parts.append(
                f"Market regime: {market_conditions.get('regime', 'unknown')}"
            )
            # Добавляем анализ паттернов
            for pattern_name, pattern_value in patterns.items():
                if isinstance(pattern_value, (int, float)):
                    reasoning_parts.append(f"{pattern_name}: {pattern_value:.3f}")
                else:
                    reasoning_parts.append(f"{pattern_name}: {pattern_value}")
            return "; ".join(reasoning_parts)
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"reasoning_error: {str(e)}"

    def _perform_reasoning_adaptation(self, performance_delta: float) -> None:
        """Выполнение адаптации логики рассуждений."""
        try:
            # Адаптация параметров на основе изменения производительности
            if performance_delta > 0:
                # Улучшение производительности - увеличиваем агрессивность
                self.config.evolution_rate *= 1.1
                self.config.adaptation_threshold *= 0.9
            else:
                # Снижение производительности - уменьшаем агрессивность
                self.config.evolution_rate *= 0.9
                self.config.adaptation_threshold *= 1.1
        except Exception as e:
            logger.error(f"Error performing reasoning adaptation: {e}")

    def _evolve_risk_assessment(self, volatility: float) -> float:
        """Эволюция оценки риска."""
        base_risk = 0.5
        volatility_factor = 1.0 + (
            volatility * 2.0
        )  # Увеличиваем риск при высокой волатильности
        return max(0.1, min(1.0, base_risk * volatility_factor))

    def _evolve_position_sizing(self, trend_strength: float) -> float:
        """Эволюция размера позиций."""
        base_sizing = 0.1
        trend_factor = 1.0 + (
            trend_strength * 0.5
        )  # Увеличиваем размер при сильном тренде
        return max(0.05, min(0.2, base_sizing * trend_factor))

    def _evolve_entry_timing(self, regime: str) -> str:
        """Эволюция времени входа."""
        timing_map = {
            "trending": "aggressive",
            "volatile": "cautious",
            "ranging": "normal",
            "unknown": "conservative",
        }
        return timing_map.get(regime, "conservative")

    def _evolve_exit_strategy(self, volatility: float) -> str:
        """Эволюция стратегии выхода."""
        if volatility > 0.05:
            return "quick_exit"
        else:
            return "trailing_stop"

    def _evolve_confidence_calculation(self, regime: str) -> float:
        """Эволюция расчета уверенности."""
        confidence_map = {
            "trending": 1.1,
            "volatile": 0.8,
            "ranging": 1.0,
            "unknown": 0.9,
        }
        return confidence_map.get(regime, 1.0)

    def _update_decision_history(self, decision: Dict[str, Any]) -> None:
        """Обновление истории решений."""
        try:
            decision_record = {
                "timestamp": datetime.now(),
                "decision": decision,
                "performance": decision.get("confidence", 0.0),
            }
            self.decision_history.append(decision_record)
            # Ограничение размера истории
            if len(self.decision_history) > self.config.max_decisions:
                self.decision_history = self.decision_history[
                    -self.config.max_decisions :
                ]
        except Exception as e:
            logger.error(f"Error updating decision history: {e}")

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Расчет волатильности."""
        if len(prices) < 2:
            return 0.0
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        if returns:
            return float(np.std(returns))
        return 0.0

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Расчет силы тренда."""
        if len(prices) < 2:
            return 0.0
        # Простой расчет силы тренда через линейную регрессию
        x = list(range(len(prices)))
        y = prices
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        if n * sum_x2 - sum_x**2 != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            # Нормализуем наклон
            max_slope = max(abs(prices[-1] - prices[0]) / len(prices), 0.001)
            trend_strength = abs(slope) / max_slope
            return min(1.0, trend_strength)
        return 0.0
