"""Модуль для сбора и анализа метрик системы."""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from domain.types.messaging_types import Event as MessagingEvent, EventPriority as MessagingEventPriority, EventName, EventType
from infrastructure.messaging.event_bus import EventBus


@dataclass
class MetricPoint:
    """Точка метрики."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Серия метрик."""

    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    aggregation_window: int = 300  # 5 минут

    def add_point(
        self, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Добавление точки метрики."""
        point = MetricPoint(
            timestamp=datetime.now(), value=value, metadata=metadata or {}
        )
        self.points.append(point)

    def get_latest(self) -> Optional[float]:
        """Получение последнего значения."""
        if not self.points:
            return None
        latest_value: float = self.points[-1].value
        return latest_value

    def get_average(self, window_minutes: int = 5) -> Optional[float]:
        """Получение среднего значения за период."""
        if not self.points:
            return None
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_points = [p.value for p in self.points if p.timestamp >= cutoff_time]
        if not recent_points:
            return None
        average_value: float = statistics.mean(recent_points)
        return average_value

    def get_statistics(self, window_minutes: int = 60) -> Dict[str, float]:
        """Получение статистики за период."""
        if not self.points:
            return {}
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_points = [p.value for p in self.points if p.timestamp >= cutoff_time]
        if not recent_points:
            return {}
        return {
            "min": min(recent_points),
            "max": max(recent_points),
            "mean": statistics.mean(recent_points),
            "median": statistics.median(recent_points),
            "std": statistics.stdev(recent_points) if len(recent_points) > 1 else 0.0,
            "count": len(recent_points),
        }


class MetricsCollector:
    """
    Сборщик метрик системы.
    Собирает и анализирует метрики:
    - Производительности системы
    - Торговых операций
    - Риска и портфеля
    - Стратегий
    """

    def __init__(self, event_bus: EventBus):
        """Инициализация сборщика метрик."""
        self.event_bus = event_bus
        self.metrics: Dict[str, MetricSeries] = defaultdict(lambda: MetricSeries(""))
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        # Инициализация метрик
        self._initialize_metrics()
        logger.info("MetricsCollector initialized")

    def _initialize_metrics(self) -> None:
        """Инициализация базовых метрик."""
        # Системные метрики
        self.metrics["cpu_usage"] = MetricSeries("cpu_usage")
        self.metrics["memory_usage"] = MetricSeries("memory_usage")
        self.metrics["disk_usage"] = MetricSeries("disk_usage")
        self.metrics["network_latency"] = MetricSeries("network_latency")
        # Торговые метрики
        self.metrics["total_trades"] = MetricSeries("total_trades")
        self.metrics["winning_trades"] = MetricSeries("winning_trades")
        self.metrics["losing_trades"] = MetricSeries("losing_trades")
        self.metrics["win_rate"] = MetricSeries("win_rate")
        self.metrics["profit_factor"] = MetricSeries("profit_factor")
        self.metrics["total_pnl"] = MetricSeries("total_pnl")
        self.metrics["daily_pnl"] = MetricSeries("daily_pnl")
        # Метрики риска
        self.metrics["portfolio_risk"] = MetricSeries("portfolio_risk")
        self.metrics["var_95"] = MetricSeries("var_95")
        self.metrics["max_drawdown"] = MetricSeries("max_drawdown")
        self.metrics["sharpe_ratio"] = MetricSeries("sharpe_ratio")
        self.metrics["sortino_ratio"] = MetricSeries("sortino_ratio")
        # Метрики стратегий
        self.metrics["active_strategies"] = MetricSeries("active_strategies")
        self.metrics["strategy_signals"] = MetricSeries("strategy_signals")
        self.metrics["strategy_performance"] = MetricSeries("strategy_performance")

    async def start_collection(self) -> None:
        """Запуск сбора метрик."""
        if self.is_collecting:
            logger.warning("Metrics collection is already running")
            return
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")

    async def stop_collection(self) -> None:
        """Остановка сбора метрик."""
        if not self.is_collecting:
            return
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")

    async def _collection_loop(self) -> None:
        """Основной цикл сбора метрик."""
        logger.info("Starting metrics collection loop")
        while self.is_collecting:
            try:
                # Сбор системных метрик
                await self._collect_system_metrics()
                # Сбор торговых метрик
                await self._collect_trading_metrics()
                # Сбор метрик риска
                await self._collect_risk_metrics()
                # Сбор метрик стратегий
                await self._collect_strategy_metrics()
                # Публикация агрегированных метрик
                await self._publish_aggregated_metrics()
                # Ожидание следующего сбора
                await asyncio.sleep(30)  # Сбор каждые 30 секунд
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self) -> None:
        """Сбор системных метрик."""
        try:
            import psutil

            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            self.metrics["cpu_usage"].add_point(cpu_usage)
            # Память
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self.metrics["memory_usage"].add_point(memory_usage)
            # Диск
            disk = psutil.disk_usage("/")
            disk_usage = (disk.used / disk.total) * 100
            self.metrics["disk_usage"].add_point(disk_usage)
            # Сеть (заглушка)
            network_latency = 0.1  # Заглушка
            self.metrics["network_latency"].add_point(network_latency)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _collect_trading_metrics(self) -> None:
        """Сбор торговых метрик."""
        try:
            # Здесь должна быть логика получения торговых данных
            # Пока используем заглушки
            # Общее количество сделок
            total_trades = 150  # Заглушка
            self.metrics["total_trades"].add_point(total_trades)
            # Выигрышные сделки
            winning_trades = 90  # Заглушка
            self.metrics["winning_trades"].add_point(winning_trades)
            # Проигрышные сделки
            losing_trades = 60  # Заглушка
            self.metrics["losing_trades"].add_point(losing_trades)
            # Винрейт
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            self.metrics["win_rate"].add_point(win_rate)
            # Profit factor (заглушка)
            profit_factor = 1.5  # Заглушка
            self.metrics["profit_factor"].add_point(profit_factor)
            # PnL
            total_pnl = 1250.50  # Заглушка
            self.metrics["total_pnl"].add_point(total_pnl)
            daily_pnl = 45.20  # Заглушка
            self.metrics["daily_pnl"].add_point(daily_pnl)
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")

    async def _collect_risk_metrics(self) -> None:
        """Сбор метрик риска."""
        try:
            # Portfolio risk (заглушка)
            portfolio_risk = 0.15  # 15% риск
            self.metrics["portfolio_risk"].add_point(portfolio_risk)
            # VaR 95% (заглушка)
            var_95 = 0.08  # 8% VaR
            self.metrics["var_95"].add_point(var_95)
            # Max drawdown (заглушка)
            max_drawdown = 0.12  # 12% максимальная просадка
            self.metrics["max_drawdown"].add_point(max_drawdown)
            # Sharpe ratio (заглушка)
            sharpe_ratio = 1.8  # Заглушка
            self.metrics["sharpe_ratio"].add_point(sharpe_ratio)
            # Sortino ratio (заглушка)
            sortino_ratio = 2.1  # Заглушка
            self.metrics["sortino_ratio"].add_point(sortino_ratio)
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")

    async def _collect_strategy_metrics(self) -> None:
        """Сбор метрик стратегий."""
        try:
            # Активные стратегии (заглушка)
            active_strategies = 3  # Заглушка
            self.metrics["active_strategies"].add_point(active_strategies)
            # Сигналы стратегий (заглушка)
            strategy_signals = 25  # Заглушка
            self.metrics["strategy_signals"].add_point(strategy_signals)
            # Производительность стратегий (заглушка)
            strategy_performance = 0.85  # 85% производительность
            self.metrics["strategy_performance"].add_point(strategy_performance)
        except Exception as e:
            logger.error(f"Error collecting strategy metrics: {e}")

    async def _publish_aggregated_metrics(self) -> None:
        """Публикация агрегированных метрик."""
        try:
            # Создание агрегированного отчета
            aggregated_metrics = {
                "system": await self._get_system_metrics_summary(),
                "trading": await self._get_trading_metrics_summary(),
                "risk": await self._get_risk_metrics_summary(),
                "strategies": await self._get_strategy_metrics_summary(),
                "timestamp": datetime.now().isoformat(),
            }
            # Публикация события
            event = MessagingEvent(
                name=EventName("metrics.aggregated"),  # Добавляю обязательный аргумент
                type=EventType("metrics"),  # Добавляю обязательный аргумент
                data=aggregated_metrics,
                priority=MessagingEventPriority.NORMAL,
            )
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error publishing aggregated metrics: {e}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        try:
            return {
                "cpu_usage": self.metrics["cpu_usage"].get_latest() or 0.0,
                "memory_usage": self.metrics["memory_usage"].get_latest() or 0.0,
                "disk_usage": self.metrics["disk_usage"].get_latest() or 0.0,
                "network_latency": self.metrics["network_latency"].get_latest() or 0.0,
                "active_strategies": self.metrics["active_strategies"].get_latest()
                or 0,
                "total_trades": self.metrics["total_trades"].get_latest() or 0,
                "system_uptime": time.time(),  # Заглушка
                "error_rate": 0.0,  # Заглушка
                "performance_score": self._calculate_performance_score(),
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Получение метрик риска."""
        try:
            return {
                "portfolio_risk": self.metrics["portfolio_risk"].get_latest() or 0.0,
                "var_95": self.metrics["var_95"].get_latest() or 0.0,
                "var_99": 0.12,  # Заглушка
                "max_drawdown": self.metrics["max_drawdown"].get_latest() or 0.0,
                "sharpe_ratio": self.metrics["sharpe_ratio"].get_latest() or 0.0,
                "sortino_ratio": self.metrics["sortino_ratio"].get_latest() or 0.0,
                "calmar_ratio": 1.5,  # Заглушка
                "correlation_matrix": {},  # Заглушка
                "concentration_risk": 0.05,  # Заглушка
                "leverage_ratio": 1.2,  # Заглушка
                "margin_usage": 0.15,  # Заглушка
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}

    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Получение метрик портфеля."""
        try:
            return {
                "total_value": 50000.0,  # Заглушка
                "total_pnl": self.metrics["total_pnl"].get_latest() or 0.0,
                "daily_pnl": self.metrics["daily_pnl"].get_latest() or 0.0,
                "weekly_pnl": 320.50,  # Заглушка
                "monthly_pnl": 1250.75,  # Заглушка
                "total_return": 0.025,  # Заглушка
                "daily_return": 0.001,  # Заглушка
                "volatility": 0.18,  # Заглушка
                "beta": 0.95,  # Заглушка
                "alpha": 0.02,  # Заглушка
                "information_ratio": 1.2,  # Заглушка
                "treynor_ratio": 1.8,  # Заглушка
                "jensen_alpha": 0.015,  # Заглушка
                "positions": {},  # Заглушка
                "cash_balance": 15000.0,  # Заглушка
                "margin_balance": 35000.0,  # Заглушка
            }
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            return {}

    async def get_strategy_metrics(
        self, strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Получение метрик стратегий."""
        try:
            if strategy_name:
                # Метрики конкретной стратегии
                return {
                    "name": strategy_name,
                    "total_trades": 50,  # Заглушка
                    "winning_trades": 35,  # Заглушка
                    "losing_trades": 15,  # Заглушка
                    "win_rate": 0.7,  # Заглушка
                    "profit_factor": 1.8,  # Заглушка
                    "average_profit": 25.0,  # Заглушка
                    "average_loss": -15.0,  # Заглушка
                    "max_drawdown": 0.08,  # Заглушка
                    "sharpe_ratio": 1.6,  # Заглушка
                    "sortino_ratio": 2.0,  # Заглушка
                    "calmar_ratio": 1.4,  # Заглушка
                    "total_return": 0.15,  # Заглушка
                    "daily_return": 0.002,  # Заглушка
                    "volatility": 0.12,  # Заглушка
                    "beta": 0.8,  # Заглушка
                    "alpha": 0.03,  # Заглушка
                    "information_ratio": 1.5,  # Заглушка
                    "start_time": datetime.now() - timedelta(days=30),  # Заглушка
                    "end_time": None,
                    "parameters": {},  # Заглушка
                    "status": "active",
                }
            else:
                # Общие метрики стратегий
                return {
                    "active_strategies": self.metrics["active_strategies"].get_latest()
                    or 0,
                    "total_signals": self.metrics["strategy_signals"].get_latest() or 0,
                    "average_performance": self.metrics[
                        "strategy_performance"
                    ].get_latest()
                    or 0.0,
                    "total_trades": self.metrics["total_trades"].get_latest() or 0,
                    "win_rate": self.metrics["win_rate"].get_latest() or 0.0,
                    "profit_factor": self.metrics["profit_factor"].get_latest() or 0.0,
                }
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            return {}

    async def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки системных метрик."""
        return {
            "cpu_usage": self.metrics["cpu_usage"].get_statistics(60),
            "memory_usage": self.metrics["memory_usage"].get_statistics(60),
            "disk_usage": self.metrics["disk_usage"].get_statistics(60),
            "network_latency": self.metrics["network_latency"].get_statistics(60),
        }

    async def _get_trading_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки торговых метрик."""
        return {
            "total_trades": self.metrics["total_trades"].get_latest() or 0,
            "win_rate": self.metrics["win_rate"].get_latest() or 0.0,
            "profit_factor": self.metrics["profit_factor"].get_latest() or 0.0,
            "total_pnl": self.metrics["total_pnl"].get_latest() or 0.0,
            "daily_pnl": self.metrics["daily_pnl"].get_latest() or 0.0,
        }

    async def _get_risk_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик риска."""
        return {
            "portfolio_risk": self.metrics["portfolio_risk"].get_latest() or 0.0,
            "var_95": self.metrics["var_95"].get_latest() or 0.0,
            "max_drawdown": self.metrics["max_drawdown"].get_latest() or 0.0,
            "sharpe_ratio": self.metrics["sharpe_ratio"].get_latest() or 0.0,
            "sortino_ratio": self.metrics["sortino_ratio"].get_latest() or 0.0,
        }

    async def _get_strategy_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик стратегий."""
        return {
            "active_strategies": self.metrics["active_strategies"].get_latest() or 0,
            "total_signals": self.metrics["strategy_signals"].get_latest() or 0,
            "performance": self.metrics["strategy_performance"].get_latest() or 0.0,
        }

    def _calculate_performance_score(self) -> float:
        """Расчет оценки производительности."""
        try:
            cpu_usage = self.metrics["cpu_usage"].get_latest() or 0.0
            memory_usage = self.metrics["memory_usage"].get_latest() or 0.0
            disk_usage = self.metrics["disk_usage"].get_latest() or 0.0
            network_latency = self.metrics["network_latency"].get_latest() or 0.0
            # Нормализация (0-1, где 1 - лучший результат)
            cpu_score = max(0, 1 - cpu_usage / 100)
            memory_score = max(0, 1 - memory_usage / 100)
            disk_score = max(0, 1 - disk_usage / 100)
            latency_score = max(0, 1 - min(network_latency, 5) / 5)
            # Взвешенная оценка
            score = (
                cpu_score * 0.3
                + memory_score * 0.3
                + disk_score * 0.2
                + latency_score * 0.2
            )
            return round(score, 3)
        except Exception:
            return 0.0

    def add_custom_metric(
        self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Добавление пользовательской метрики."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
        self.metrics[name].add_point(value, metadata)

    def get_metric_history(
        self, metric_name: str, hours: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Получение истории метрики."""
        if metric_name not in self.metrics:
            return []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = []
        for point in self.metrics[metric_name].points:
            if point.timestamp >= cutoff_time:
                history.append((point.timestamp, point.value))
        return history

    def get_metric_statistics(
        self, metric_name: str, hours: int = 24
    ) -> Dict[str, float]:
        """Получение статистики метрики."""
        history = self.get_metric_history(metric_name, hours)
        if not history:
            return {}
        values = [value for _, value in history]
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }
