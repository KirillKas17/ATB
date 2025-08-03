"""
Эволюционный мета-контроллер агент.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EvolvableMetaControllerConfig:
    """Конфигурация эволюционного мета-контроллера."""

    evolution_rate: float = 0.1
    adaptation_threshold: float = 0.05
    learning_rate: float = 0.01
    max_strategies: int = 10
    performance_window: int = 100
    enable_auto_adaptation: bool = True


class EvolvableMetaController(ABC):
    """Абстрактный эволюционный мета-контроллер."""

    def __init__(self, config: Optional[EvolvableMetaControllerConfig] = None):
        self.config = config or EvolvableMetaControllerConfig()
        self.is_active: bool = False
        self.performance_history: List[float] = []
        self.adaptation_count: int = 0

    @abstractmethod
    async def coordinate_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Координация агентов с эволюционными улучшениями."""

    @abstractmethod
    async def adapt_strategy(self, performance_data: Dict[str, Any]) -> bool:
        """Адаптация стратегии на основе производительности."""

    @abstractmethod
    async def evolve_decision_logic(
        self, market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция логики принятия решений."""

    @abstractmethod
    async def optimize_agent_weights(
        self, agent_performances: Dict[str, float]
    ) -> Dict[str, float]:
        """Оптимизация весов агентов."""


class DefaultEvolvableMetaController(EvolvableMetaController):
    """Реализация эволюционного мета-контроллера по умолчанию."""

    def __init__(self, config: Optional[EvolvableMetaControllerConfig] = None):
        super().__init__(config)
        self.agent_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, float] = {}
        self.market_regime_memory: List[Dict[str, Any]] = []

    async def coordinate_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Координация агентов с эволюционными улучшениями."""
        try:
            # Анализ текущих рыночных условий
            market_conditions = self._analyze_market_conditions(context)
            # Адаптация к рыночным условиям
            await self.adapt_strategy(market_conditions)
            # Эволюция логики принятия решений
            decision_logic = await self.evolve_decision_logic(market_conditions)
            # Координация агентов с учетом эволюционных улучшений
            agent_results = {}
            for agent_id, agent in context.get("agents", {}).items():
                if hasattr(agent, "process") and callable(agent.process):
                    # Применяем эволюционную логику к каждому агенту
                    enhanced_context = self._enhance_context_with_evolution(
                        context, decision_logic
                    )
                    agent_results[agent_id] = await agent.process(enhanced_context)
                else:
                    agent_results[agent_id] = {"status": "no_process_method"}
            # Агрегация результатов с учетом весов агентов
            aggregated_result = self._aggregate_results_with_weights(agent_results)
            # Обновление истории производительности
            self._update_performance_history(aggregated_result.get("performance", 0.0))
            return {
                "coordinated_results": agent_results,
                "aggregated_result": aggregated_result,
                "evolution_metrics": {
                    "adaptation_count": self.adaptation_count,
                    "performance_trend": self._calculate_performance_trend(),
                    "market_regime": market_conditions.get("regime", "unknown"),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    async def adapt_strategy(self, performance_data: Dict[str, Any]) -> bool:
        """Адаптация стратегии на основе производительности."""
        try:
            current_performance = performance_data.get("performance", 0.0)
            # Проверяем, нужна ли адаптация
            if len(self.performance_history) > 0:
                avg_performance = sum(self.performance_history) / len(
                    self.performance_history
                )
                performance_delta = current_performance - avg_performance
                if abs(performance_delta) > self.config.adaptation_threshold:
                    # Выполняем адаптацию
                    self._perform_strategy_adaptation(performance_delta)
                    self.adaptation_count += 1
                    return True
            return False
        except Exception:
            return False

    async def evolve_decision_logic(
        self, market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция логики принятия решений."""
        try:
            # Анализ текущего рыночного режима
            regime = market_conditions.get("regime", "unknown")
            volatility = market_conditions.get("volatility", 0.0)
            trend_strength = market_conditions.get("trend_strength", 0.0)
            # Эволюция параметров принятия решений
            evolved_logic = {
                "risk_tolerance": self._evolve_risk_tolerance(volatility),
                "position_sizing": self._evolve_position_sizing(trend_strength),
                "entry_timing": self._evolve_entry_timing(regime),
                "exit_strategy": self._evolve_exit_strategy(volatility),
                "agent_prioritization": self._evolve_agent_prioritization(regime),
            }
            return evolved_logic
        except Exception as e:
            return {"error": str(e)}

    async def optimize_agent_weights(
        self, agent_performances: Dict[str, float]
    ) -> Dict[str, float]:
        """Оптимизация весов агентов."""
        try:
            if not agent_performances:
                return {}
            # Простая оптимизация на основе производительности
            total_performance = sum(agent_performances.values())
            if total_performance > 0:
                optimized_weights = {}
                for agent_id, performance in agent_performances.items():
                    weight = performance / total_performance
                    # Применяем сглаживание
                    current_weight = self.agent_weights.get(agent_id, 1.0)
                    smoothed_weight = (current_weight * 0.7) + (weight * 0.3)
                    optimized_weights[agent_id] = smoothed_weight
                self.agent_weights = optimized_weights
                return optimized_weights
            else:
                # Равномерное распределение весов
                equal_weight = 1.0 / len(agent_performances)
                self.agent_weights = {
                    agent_id: equal_weight for agent_id in agent_performances
                }
                return self.agent_weights
        except Exception:
            return {}

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
            }
        except Exception:
            return {"regime": "unknown", "volatility": 0.0, "trend_strength": 0.0}

    def _enhance_context_with_evolution(
        self, context: Dict[str, Any], decision_logic: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Улучшение контекста эволюционной логикой."""
        enhanced_context = context.copy()
        enhanced_context["evolution_logic"] = decision_logic
        enhanced_context["agent_weights"] = self.agent_weights
        return enhanced_context

    def _aggregate_results_with_weights(
        self, agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Агрегация результатов с учетом весов агентов."""
        try:
            aggregated: Dict[str, Any] = {
                "signals": [],
                "confidence": 0.0,
                "performance": 0.0,
                "risk_score": 0.0,
            }
            total_weight = 0.0
            weighted_confidence = 0.0
            weighted_performance = 0.0
            weighted_risk = 0.0
            for agent_id, result in agent_results.items():
                weight = self.agent_weights.get(agent_id, 1.0)
                total_weight += weight
                if isinstance(result, dict):
                    confidence = result.get("confidence", 0.0)
                    performance = result.get("performance", 0.0)
                    risk_score = result.get("risk_score", 0.0)
                    weighted_confidence += confidence * weight
                    weighted_performance += performance * weight
                    weighted_risk += risk_score * weight
                    # Агрегация сигналов
                    if "signals" in result and isinstance(result["signals"], list):
                        signals = result["signals"]
                        for signal in signals:
                            if isinstance(signal, dict):
                                aggregated["signals"].append(signal)
            if total_weight > 0:
                aggregated["confidence"] = weighted_confidence / total_weight
                aggregated["performance"] = weighted_performance / total_weight
                aggregated["risk_score"] = weighted_risk / total_weight
            return aggregated
        except Exception as e:
            return {"error": str(e)}

    def _update_performance_history(self, performance: float) -> None:
        """Обновление истории производительности."""
        self.performance_history.append(performance)
        # Ограничение размера истории
        if len(self.performance_history) > self.config.performance_window:
            self.performance_history = self.performance_history[
                -self.config.performance_window :
            ]

    def _calculate_performance_trend(self) -> float:
        """Расчет тренда производительности."""
        if len(self.performance_history) < 2:
            return 0.0
        recent = (
            self.performance_history[-10:]
            if len(self.performance_history) >= 10
            else self.performance_history
        )
        older = (
            self.performance_history[:-10]
            if len(self.performance_history) >= 10
            else [0.0]
        )
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else 0.0
        return recent_avg - older_avg

    def _perform_strategy_adaptation(self, performance_delta: float) -> None:
        """Выполнение адаптации стратегии."""
        # Адаптация параметров на основе изменения производительности
        if performance_delta > 0:
            # Улучшение производительности - увеличиваем агрессивность
            self.config.evolution_rate *= 1.1
            self.config.adaptation_threshold *= 0.9
        else:
            # Снижение производительности - уменьшаем агрессивность
            self.config.evolution_rate *= 0.9
            self.config.adaptation_threshold *= 1.1

    def _evolve_risk_tolerance(self, volatility: float) -> float:
        """Эволюция толерантности к риску."""
        base_tolerance = 0.5
        volatility_factor = 1.0 - (
            volatility * 2.0
        )  # Уменьшаем толерантность при высокой волатильности
        return max(0.1, min(0.9, base_tolerance * volatility_factor))

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
            "trending": "momentum",
            "volatile": "mean_reversion",
            "ranging": "breakout",
            "unknown": "conservative",
        }
        return timing_map.get(regime, "conservative")

    def _evolve_exit_strategy(self, volatility: float) -> str:
        """Эволюция стратегии выхода."""
        if volatility > 0.05:
            return "quick_exit"
        else:
            return "trailing_stop"

    def _evolve_agent_prioritization(self, regime: str) -> Dict[str, float]:
        """Эволюция приоритизации агентов."""
        prioritization = {
            "risk_agent": 1.0,
            "market_maker_agent": 1.0,
            "portfolio_agent": 1.0,
        }
        if regime == "volatile":
            prioritization["risk_agent"] = 1.5
        elif regime == "trending":
            prioritization["market_maker_agent"] = 1.3
        elif regime == "ranging":
            prioritization["portfolio_agent"] = 1.2
        return prioritization

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
