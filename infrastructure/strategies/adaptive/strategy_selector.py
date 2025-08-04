"""
Селектор стратегий для адаптивных стратегий
"""

from typing import Any, Callable, Dict, List, Optional
from shared.numpy_utils import np

from loguru import logger

from domain.types.strategy_types import MarketRegime


class StrategySelector:
    """Селектор стратегий"""

    def __init__(
        self, backtest_results: Dict[str, Dict], adaptation_threshold: float = 0.7
    ):
        self.backtest_results = backtest_results or {}
        self.adaptation_threshold = adaptation_threshold
        # Маппинг режимов на стратегии
        self.regime_strategies = {
            "trend": ["trend_strategy_ema_macd", "trend_strategy_price_action"],
            "sideways": ["sideways_strategy_bb_rsi", "sideways_strategy_stoch_obv"],
            "reversal": [
                "reversal_strategy_rsi_divergence",
                "reversal_strategy_fibo_pinbar",
            ],
            "volatility": [
                "volatility_strategy_atr_breakout",
                "volatility_strategy_ema_keltner",
            ],
            "manipulation": [
                "manipulation_strategy_stop_hunt",
                "manipulation_strategy_fake_breakout",
            ],
        }

    def select_best_strategy(self, regime: MarketRegime, data: Any) -> Optional[str]:
        """Выбор лучшей стратегии для режима"""
        try:
            regime_strategies = self.regime_strategies.get(regime.value, [])
            if not regime_strategies:
                return None
            best_strategy = None
            best_score = -np.inf
            for strategy_name in regime_strategies:
                if strategy_name in self.backtest_results:
                    score = self._calculate_strategy_score(
                        self.backtest_results[strategy_name]
                    )
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy_name
            return best_strategy if best_score > self.adaptation_threshold else None
        except Exception as e:
            logger.error(f"Error selecting best strategy: {str(e)}")
            return None

    def _calculate_strategy_score(self, results: Dict) -> float:
        """Расчет оценки стратегии"""
        if not results:
            return 0.0
        # Комбинированная оценка на основе метрик (max_drawdown учитывается как штраф)
        score = (
            results.get("win_rate", 0) * 0.3
            + results.get("profit_factor", 0) * 0.3
            + results.get("sharpe_ratio", 0) * 0.2
            - abs(results.get("max_drawdown", 0)) * 0.2
        )
        return float(score)

    def get_strategy_weights(self) -> Dict[str, float]:
        """Получение весов стратегий"""
        return {strategy: 1.0 for strategy in self.regime_strategies.keys()}

    def update_strategy_weights(self, performance: Dict[str, float]) -> None:
        """Обновление весов стратегий на основе производительности"""
        try:
            total_performance = sum(performance.values())
            if total_performance > 0:
                for strategy, perf in performance.items():
                    if strategy in self.regime_strategies:
                        # Обновляем вес стратегии
                        self.regime_strategies[strategy] = perf / total_performance
        except Exception as e:
            logger.error(f"Error updating strategy weights: {str(e)}")
