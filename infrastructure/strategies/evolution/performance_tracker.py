"""
Трекер производительности эволюционных стратегий
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger
from shared.numpy_utils import np
import pandas as pd

from domain.type_definitions.strategy_types import StrategyMetrics


class PerformanceRecord(TypedDict):
    strategy_id: str
    performance: float
    timestamp: datetime


class MetricsRecord(TypedDict):
    strategy_id: str
    metrics: Dict[str, Any]
    timestamp: datetime


class PerformanceTracker:
    """Трекер производительности стратегий"""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.performance_history: List[PerformanceRecord] = []
        self.metrics_history: List[MetricsRecord] = []

    def add_performance_data(
        self, strategy_id: str, performance: float, metrics: Dict[str, Any]
    ) -> None:
        """Добавление данных о производительности"""
        try:
            timestamp = datetime.now()
            self.performance_history.append(
                {
                    "strategy_id": strategy_id,
                    "performance": performance,
                    "timestamp": timestamp,
                }
            )
            self.metrics_history.append(
                {"strategy_id": strategy_id, "metrics": metrics, "timestamp": timestamp}
            )
            # Ограничение размера истории
            if len(self.performance_history) > self.window_size:
                self.performance_history = self.performance_history[-self.window_size :]
                self.metrics_history = self.metrics_history[-self.window_size :]
        except Exception as e:
            logger.error(f"Error adding performance data: {str(e)}")

    def get_performance_trend(
        self, strategy_id: str, days: int = 30
    ) -> Dict[str, float]:
        """Получение тренда производительности"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_performance = [
                p
                for p in self.performance_history
                if p["strategy_id"] == strategy_id and p["timestamp"] > cutoff_date
            ]
            if len(recent_performance) < 2:
                return {"trend": 0.0, "volatility": 0.0, "avg_performance": 0.0}
            performances = [float(p["performance"]) if p["performance"] is not None else 0.0 for p in recent_performance]
            # Расчет тренда
            x = np.arange(len(performances))
            slope = np.polyfit(x, np.array(performances), 1)[0]
            # Расчет волатильности
            volatility = np.std(performances)
            # Средняя производительность
            avg_performance = np.mean(performances)
            return {
                "trend": float(slope) if slope is not None and isinstance(slope, (int, float, np.number)) else 0.0,
                "volatility": float(volatility) if volatility is not None and isinstance(volatility, (int, float, np.number)) else 0.0,
                "avg_performance": float(avg_performance) if avg_performance is not None and isinstance(avg_performance, (int, float, np.number)) else 0.0,
            }
        except Exception as e:
            logger.error(f"Error getting performance trend: {str(e)}")
            return {"trend": 0.0, "volatility": 0.0, "avg_performance": 0.0}

    def get_strategy_ranking(self) -> List[Dict[str, Any]]:
        """Получение рейтинга стратегий"""
        try:
            if not self.performance_history:
                return []
            # Группировка по стратегиям
            strategy_performance: Dict[str, List[float]] = {}
            for record in self.performance_history:
                strategy_id = record["strategy_id"]
                if strategy_id not in strategy_performance:
                    strategy_performance[strategy_id] = []
                strategy_performance[strategy_id].append(record["performance"])
            # Расчет средних показателей
            rankings = []
            for strategy_id, performances in strategy_performance.items():
                avg_performance = np.mean(performances)
                volatility = np.std(performances)
                rankings.append(
                    {
                        "strategy_id": strategy_id,
                        "avg_performance": float(avg_performance) if avg_performance is not None and isinstance(avg_performance, (int, float, np.number)) else 0.0,
                        "volatility": float(volatility) if volatility is not None and isinstance(volatility, (int, float, np.number)) else 0.0,
                        "sample_count": len(performances),
                    }
                )
            # Сортировка по средней производительности
            rankings.sort(key=lambda x: float(x["avg_performance"]) if x["avg_performance"] is not None and isinstance(x["avg_performance"], (int, float, str)) else 0.0, reverse=True)
            return rankings
        except Exception as e:
            logger.error(f"Error getting strategy ranking: {str(e)}")
            return []

    def get_performance_metrics(self, strategy_id: str) -> StrategyMetrics:
        """Получение метрик производительности"""
        try:
            strategy_performance = [
                p for p in self.performance_history if p["strategy_id"] == strategy_id
            ]
            if not strategy_performance:
                return StrategyMetrics()
            performances = [p["performance"] for p in strategy_performance]
            
            # Безопасный расчет win_rate
            win_count = sum(1 for p in performances if p > 0)
            win_rate = float(win_count / len(performances)) if performances else 0.0
            
            # Безопасный расчет expectancy
            expectancy = float(np.mean(performances)) if performances else 0.0
            # Проверяем на NaN или infinity
            if not (isinstance(expectancy, (int, float)) and np.isfinite(expectancy)):
                expectancy = 0.0
            
            # Безопасный расчет volatility
            volatility = float(np.std(performances)) if len(performances) > 1 else 0.0
            
            return StrategyMetrics(
                total_signals=len(performances),
                win_rate=win_rate,
                expectancy=expectancy,
                max_drawdown=self._calculate_max_drawdown(performances),
                sharpe_ratio=self._calculate_sharpe_ratio(performances),
                volatility=volatility,
            )
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return StrategyMetrics()

    def _calculate_max_drawdown(self, performances: List[float]) -> float:
        """Расчет максимальной просадки"""
        try:
            if not performances:
                return 0.0
            cumulative = np.cumsum(performances)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            min_drawdown = np.min(drawdown)
            return float(min_drawdown) if min_drawdown is not None and isinstance(min_drawdown, (int, float, np.number)) else 0.0
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(
        self, performances: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """Расчет коэффициента Шарпа"""
        try:
            if len(performances) < 2:
                return 0.0
            returns = np.array(performances)
            excess_returns = (
                returns - risk_free_rate / 252
            )  # Дневная безрисковая ставка
            std_returns = np.std(excess_returns)
            if std_returns == 0:
                return 0.0
            sharpe = np.mean(excess_returns) / std_returns * np.sqrt(252)
            return float(sharpe) if sharpe is not None and isinstance(sharpe, (int, float, np.number)) else 0.0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def export_performance_data(self, filepath: str) -> bool:
        """Экспорт данных производительности"""
        try:
            df = pd.DataFrame(self.performance_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Performance data exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            return False
