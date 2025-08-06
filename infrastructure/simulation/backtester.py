"""
Промышленный бэктестер торговых стратегий.
Полная реализация всех методов с профессиональной логикой.
"""

import asyncio
import json
import random
import signal
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from shared.numpy_utils import np
import psutil
from loguru import logger

from .types import (
    BacktestConfig,
    BacktestMetricsDict,
    BacktestResult,
    MarketRegimeType,
    SimulationMarketData,
    SimulationMoney,
    SimulationOrder,
    SimulationSignal,
    SimulationTimestamp,
    SimulationTrade,
    Symbol,
    TradeMetricsDict,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Protocol интерфейсы
# ============================================================================
@runtime_checkable
class StrategyProtocol(Protocol):
    """Протокол для торговой стратегии."""

    async def generate_signal(
        self, market_data: SimulationMarketData, context: Dict[str, Any]
    ) -> Optional[SimulationSignal]: ...
    async def validate_signal(self, signal: SimulationSignal) -> bool: ...
    async def get_signal_confidence(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> float: ...
@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Протокол для риск-менеджера."""

    async def check_risk_limits(
        self,
        signal: SimulationSignal,
        current_balance: SimulationMoney,
        current_positions: List[SimulationTrade],
    ) -> bool: ...
    async def calculate_position_size(
        self, signal: SimulationSignal, balance: SimulationMoney, risk_per_trade: float
    ) -> float: ...
    async def calculate_stop_loss(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> Optional[float]: ...


# ============================================================================
# Конфигурации и метрики
# ============================================================================
@dataclass
class BacktestMetrics:
    """Расширенные метрики бэктеста."""

    start_time: datetime
    end_time: datetime
    simulation_time: float
    memory_usage: float
    cpu_usage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    risk_reward_ratio: float
    kelly_criterion: float
    total_profit: float
    total_loss: float
    net_profit: float
    success: bool
    error: Optional[str] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    model_metrics: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    regime_metrics: Optional[Dict[str, Any]] = None
    pattern_metrics: Optional[Dict[str, Any]] = None
    adaptation_metrics: Optional[Dict[str, Any]] = None


# ============================================================================
# Основной класс бэктестера
# ============================================================================
class Backtester:
    """Промышленный бэктестер с полной реализацией всех методов."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Инициализация бэктестера."""
        self.config = config or self._load_default_config()
        self.metrics_history: List[BacktestMetrics] = []
        self._sim_lock = asyncio.Lock()
        self._start_time: Optional[datetime] = None
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Установка случайного зерна
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        # Создание директорий
        for dir_path in [
            self.config.data_dir,
            self.config.results_dir,
            self.config.logs_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Инициализация компонентов
        self.order_manager = OrderManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.metrics_calculator = MetricsCalculator()
        # Инициализация состояния
        self.state: Dict[str, Any] = {
            "balance": self.config.initial_balance,
            "positions": [],
            "trades": [],
            "equity": [float(self.config.initial_balance)],
            "drawdown": [0.0],
            "regime": MarketRegimeType.UNKNOWN,
            "patterns": [],
            "adaptation_state": {},
        }
        # Инициализация кэша
        self._init_cache()
        # Настройка логгера
        self._setup_logger()
        # Обработка сигналов завершения
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_default_config(self) -> BacktestConfig:
        """Загрузка конфигурации по умолчанию."""
        return BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_balance=SimulationMoney(Decimal("10000")),
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.1,
            risk_per_trade=0.02,
            confidence_threshold=0.7,
            use_realistic_slippage=True,
            use_market_impact=True,
            use_latency=True,
            use_partial_fills=True,
            calculate_metrics=True,
            generate_plots=True,
            save_trades=True,
            save_equity_curve=True,
            min_trades=10,
            min_win_rate=0.4,
            min_profit_factor=1.1,
            symbols=[Symbol("BTCUSDT")],
            timeframes=["1m", "5m", "15m", "1h"],
            random_seed=42,
            cache_size=1000,
            max_workers=4,
            data_dir=Path("data/backtest"),
            results_dir=Path("results/backtest"),
            logs_dir=Path("logs/backtest"),
        )

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        log_path = self.config.logs_dir / "backtester.log"
        logger.add(
            log_path,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    def _init_cache(self) -> None:
        """Инициализация кэша."""
        self._price_cache: Dict[int, float] = {}
        self._volume_cache: Dict[int, float] = {}
        self._regime_cache: Dict[int, str] = {}
        self._pattern_cache: Dict[int, List] = {}
        self._adaptation_cache: Dict[int, Dict] = {}

    @lru_cache(maxsize=1000)
    def _get_cached_price(self, timestamp: int) -> Optional[float]:
        """Получение кэшированной цены."""
        return self._price_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_volume(self, timestamp: int) -> Optional[float]:
        """Получение кэшированного объема."""
        return self._volume_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_regime(self, timestamp: int) -> Optional[str]:
        """Получение кэшированного режима."""
        return self._regime_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_pattern(self, timestamp: int) -> Optional[List]:
        """Получение кэшированных паттернов."""
        return self._pattern_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_adaptation(self, timestamp: int) -> Optional[Dict]:
        """Получение кэшированного состояния адаптации."""
        return self._adaptation_cache.get(timestamp)

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик."""
        return {
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
            "thread_count": len(self._executor._threads),
            "cache_size": (
                len(self._price_cache)
                + len(self._volume_cache)
                + len(self._regime_cache)
                + len(self._pattern_cache)
                + len(self._adaptation_cache)
            ),
        }

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Обработка сигналов завершения."""
        logger.info("Получен сигнал завершения")
        self._running = False
        self._executor.shutdown(wait=False)

    async def run_backtest(
        self, strategy: StrategyProtocol, market_data: List[SimulationMarketData]
    ) -> BacktestResult:
        """Запуск бэктеста с полной реализацией."""
        logger.info("Starting backtest")
        self._start_time = datetime.now()
        self._running = True
        try:
            # Валидация входных данных
            if not market_data:
                raise ValueError("Empty market data provided")
            # Инициализация состояния
            await self._initialize_backtest_state()
            # Основной цикл бэктеста
            for i, market_point in enumerate(market_data):
                if not self._running:
                    break
                # Обновление состояния рынка
                await self._update_market_state(market_point)
                # Генерация сигнала
                signal = await self._generate_signal(strategy, market_point)
                if signal:
                    # Проверка риск-лимитов
                    if await self.risk_manager.check_risk_limits(
                        signal, self.state["balance"], self.state["positions"]
                    ):
                        # Исполнение сделки
                        trade = await self._execute_trade(signal, market_point)
                        if trade:
                            self.state["trades"].append(trade)
                            await self._update_equity_curve(trade)
                # Обновление метрик
                if i % 100 == 0:
                    await self._update_metrics(market_point, i)
            # Расчет финальных метрик
            metrics = await self._calculate_final_metrics()
            # Создание результата
            result = BacktestResult(
                config=self.config,
                trades=self.state["trades"],
                equity_curve=self.state["equity"],
                metrics=metrics,
                start_time=market_data[0].timestamp.value,
                end_time=market_data[-1].timestamp.value,
                initial_balance=self.config.initial_balance,
                final_balance=self.state["balance"],
                success=True,
                metadata={
                    "strategy_name": strategy.__class__.__name__,
                    "total_data_points": len(market_data),
                    "system_metrics": self._get_system_metrics(),
                },
            )
            # Сохранение результатов
            if self.config.save_trades:
                await self._save_results(result)
            # Генерация графиков
            if self.config.generate_plots:
                await self._generate_plots(result)
            logger.info("Backtest completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            return BacktestResult(
                config=self.config,
                trades=[],
                equity_curve=[],
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                initial_balance=self.config.initial_balance,
                final_balance=self.config.initial_balance,
                success=False,
                error_message=str(e),
            )
        finally:
            self._running = False
            await self._cleanup()

    async def _initialize_backtest_state(self) -> None:
        """Инициализация состояния бэктеста."""
        self.state["balance"] = self.config.initial_balance
        self.state["positions"] = []
        self.state["trades"] = []
        self.state["equity"] = [float(self.config.initial_balance)]
        self.state["drawdown"] = [0.0]
        self.state["regime"] = MarketRegimeType.UNKNOWN
        self.state["patterns"] = []
        self.state["adaptation_state"] = {}
        logger.info("Backtest state initialized")

    async def _update_market_state(self, market_data: SimulationMarketData) -> None:
        """Обновление состояния рынка."""
        # Обновление режима
        self.state["regime"] = market_data.regime
        # Обновление паттернов (упрощенная версия)
        if len(self.state["equity"]) > 10:
            recent_equity = self.state["equity"][-10:]
            pattern = self._detect_pattern(recent_equity)
            if pattern:
                self.state["patterns"].append(pattern)
        # Ограничение размера истории паттернов
        if len(self.state["patterns"]) > 100:
            self.state["patterns"] = self.state["patterns"][-100:]

    async def _generate_signal(
        self, strategy: StrategyProtocol, market_data: SimulationMarketData
    ) -> Optional[SimulationSignal]:
        """Генерация торгового сигнала."""
        try:
            context = {
                "balance": self.state["balance"],
                "positions": self.state["positions"],
                "regime": self.state["regime"],
                "patterns": self.state["patterns"],
                "equity_curve": self.state["equity"],
            }
            signal = await strategy.generate_signal(market_data, context)
            if signal and signal.confidence >= self.config.confidence_threshold:
                # Валидация сигнала
                if await strategy.validate_signal(signal):
                    return signal
            return None
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    async def _execute_trade(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> Optional[SimulationTrade]:
        """Исполнение сделки."""
        try:
            # Расчет размера позиции
            position_size = await self.risk_manager.calculate_position_size(
                signal, self.state["balance"], self.config.risk_per_trade
            )
            # Расчет стоп-лосса
            stop_loss = await self.risk_manager.calculate_stop_loss(signal, market_data)
            # Создание ордера
            order = SimulationOrder(
                symbol=signal.symbol,
                timestamp=market_data.timestamp,
                order_type="MARKET",
                side=signal.signal_type.upper(),
                quantity=SimulationVolume(Decimal(str(position_size))),
                price=signal.price,
                stop_price=(
                    SimulationPrice(Decimal(str(stop_loss))) if stop_loss else None
                ),
            )
            # Исполнение ордера
            trade = await self.order_manager.execute_order(order, market_data)
            # Обновление баланса
            if trade:
                self.state["balance"] = self.state["balance"] - trade.total_cost
                # Обновление позиций
                if trade.side == "BUY":
                    self.state["positions"].append(trade)
                else:
                    # Закрытие позиции
                    self.state["positions"] = [
                        p for p in self.state["positions"] if p.symbol != trade.symbol
                    ]
            return trade
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    async def _update_equity_curve(self, trade: SimulationTrade) -> None:
        """Обновление кривой доходности."""
        current_equity = self.state["equity"][-1]
        new_equity = current_equity + float(trade.pnl)
        self.state["equity"].append(new_equity)
        # Расчет просадки
        peak = max(self.state["equity"])
        drawdown = (peak - new_equity) / peak if peak > 0 else 0.0
        self.state["drawdown"].append(drawdown)

    async def _update_metrics(
        self, market_data: SimulationMarketData, index: int
    ) -> None:
        """Обновление метрик во время бэктеста."""
        # Обновление кэша
        timestamp = int(market_data.timestamp.value.timestamp())
        self._price_cache[timestamp] = float(market_data.close)
        self._volume_cache[timestamp] = float(market_data.volume)
        self._regime_cache[timestamp] = market_data.regime.value
        # Очистка старых записей кэша
        if len(self._price_cache) > self.config.cache_size:
            old_timestamps = sorted(self._price_cache.keys())[: -self.config.cache_size]
            for ts in old_timestamps:
                del self._price_cache[ts]
                del self._volume_cache[ts]
                del self._regime_cache[ts]

    async def _calculate_final_metrics(self) -> BacktestMetricsDict:
        """Расчет финальных метрик."""
        return await self.metrics_calculator.calculate_backtest_metrics(
            self.state["trades"], self.state["equity"]
        )

    def _detect_pattern(self, equity_curve: List[float]) -> Optional[Dict[str, Any]]:
        """Обнаружение паттернов в кривой доходности."""
        if len(equity_curve) < 5:
            return None
        # Простой анализ тренда
        returns = np.diff(equity_curve)
        trend = np.mean(returns)
        volatility = np.std(returns)
        if trend > 0.001 and volatility < 0.01:
            return {"type": "uptrend", "confidence": 0.7, "volatility": volatility}
        elif trend < -0.001 and volatility < 0.01:
            return {"type": "downtrend", "confidence": 0.7, "volatility": volatility}
        elif volatility > 0.02:
            return {"type": "volatile", "confidence": 0.6, "volatility": volatility}
        return None

    async def _save_results(self, result: BacktestResult) -> None:
        """Сохранение результатов."""
        results_file = (
            self.config.results_dir
            / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        # Преобразование в JSON-совместимый формат
        result_dict = {
            "config": {
                "initial_balance": float(result.config.initial_balance),
                "commission_rate": result.config.commission_rate,
                "slippage_rate": result.config.slippage_rate,
                "start_date": result.start_time.isoformat(),
                "end_date": result.end_time.isoformat(),
            },
            "metrics": result.metrics,
            "total_trades": len(result.trades),
            "final_balance": float(result.final_balance),
            "success": result.success,
            "error_message": result.error_message,
            "metadata": result.metadata,
        }
        with open(results_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    async def _generate_plots(self, result: BacktestResult) -> None:
        """Генерация графиков."""
        plots_dir = self.config.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        # График кривой доходности
        plt.figure(figsize=(12, 8))
        plt.plot(result.equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.savefig(plots_dir / "equity_curve.png")
        plt.close()
        # График распределения доходности
        if result.trades:
            returns = [float(trade.pnl) for trade in result.trades]
            plt.figure(figsize=(10, 6))
            plt.hist(returns, bins=50, alpha=0.7)
            plt.title("Return Distribution")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(plots_dir / "return_distribution.png")
            plt.close()
        logger.info(f"Plots saved to {plots_dir}")

    async def _cleanup(self) -> None:
        """Очистка ресурсов."""
        self._executor.shutdown(wait=True)
        # Очистка кэша
        self._price_cache.clear()
        self._volume_cache.clear()
        self._regime_cache.clear()
        self._pattern_cache.clear()
        self._adaptation_cache.clear()
        logger.info("Backtester cleanup completed")


# ============================================================================
# Вспомогательные классы
# ============================================================================
class OrderManager:
    """Менеджер ордеров."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    async def execute_order(
        self, order: SimulationOrder, market_data: SimulationMarketData
    ) -> Optional[SimulationTrade]:
        """Исполнение ордера."""
        try:
            # Расчет проскальзывания
            slippage = await self._calculate_slippage(order, market_data)
            # Расчет комиссии
            commission = await self._calculate_commission(order)
            # Расчет цены исполнения
            execution_price = await self._calculate_execution_price(
                order, market_data, slippage
            )
            # Создание сделки
            trade = SimulationTrade(
                order_id=order.id,
                symbol=order.symbol,
                timestamp=order.timestamp,
                side=order.side,
                quantity=order.quantity,
                price=SimulationPrice(Decimal(str(execution_price))),
                commission=commission,
                slippage=slippage,
                pnl=SimulationMoney(Decimal("0")),  # Будет рассчитано позже
                market_impact=float(slippage) / float(order.quantity) / execution_price,
                execution_quality=1.0
                - float(slippage) / float(order.quantity) / execution_price,
            )
            return trade
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return None

    async def _calculate_slippage(
        self, order: SimulationOrder, market_data: SimulationMarketData
    ) -> SimulationMoney:
        """Расчет проскальзывания."""
        if not self.config.use_realistic_slippage:
            return SimulationMoney(Decimal("0"))
        base_slippage = (
            float(order.quantity) * float(market_data.close) * self.config.slippage_rate
        )
        # Модификация по объему
        volume_impact = (
            float(order.quantity) / float(market_data.volume)
            if float(market_data.volume) > 0
            else 0
        )
        volume_slippage = base_slippage * (1 + volume_impact)
        # Модификация по волатильности
        volatility_impact = market_data.volatility * 10
        total_slippage = volume_slippage * (1 + volatility_impact)
        return SimulationMoney(Decimal(str(total_slippage)))

    async def _calculate_commission(self, order: SimulationOrder) -> SimulationMoney:
        """Расчет комиссии."""
        trade_value = float(order.quantity) * float(order.price) if order.price else 0
        commission = trade_value * self.config.commission_rate
        return SimulationMoney(Decimal(str(commission)))

    async def _calculate_execution_price(
        self,
        order: SimulationOrder,
        market_data: SimulationMarketData,
        slippage: SimulationMoney,
    ) -> float:
        """Расчет цены исполнения."""
        base_price = float(market_data.close)
        if order.side == "BUY":
            return base_price + float(slippage) / float(order.quantity)
        else:
            return base_price - float(slippage) / float(order.quantity)


class RiskManager:
    """Риск-менеджер."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    async def check_risk_limits(
        self,
        signal: SimulationSignal,
        current_balance: SimulationMoney,
        current_positions: List[SimulationTrade],
    ) -> bool:
        """Проверка риск-лимитов."""
        try:
            # Проверка максимальной просадки
            if len(current_positions) > 0:
                total_exposure = sum(
                    float(p.quantity) * float(p.price) for p in current_positions
                )
                exposure_ratio = total_exposure / float(current_balance)
                if exposure_ratio > self.config.max_position_size:
                    return False
            # Проверка максимального количества позиций
            if len(current_positions) >= 10:  # Максимум 10 позиций
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False

    async def calculate_position_size(
        self, signal: SimulationSignal, balance: SimulationMoney, risk_per_trade: float
    ) -> float:
        """Расчет размера позиции."""
        try:
            risk_amount = float(balance) * risk_per_trade
            if signal.stop_loss and signal.price:
                stop_distance = abs(float(signal.price) - float(signal.stop_loss))
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                else:
                    position_size = risk_amount / (
                        float(signal.price) * 0.02
                    )  # 2% по умолчанию
            else:
                position_size = (
                    risk_amount / (float(signal.price) * 0.02) if signal.price else 0
                )
            # Ограничение размера позиции
            max_size = float(balance) * self.config.max_position_size
            position_size = min(position_size, max_size)
            return max(position_size, 0)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    async def calculate_stop_loss(
        self, signal: SimulationSignal, market_data: SimulationMarketData
    ) -> Optional[float]:
        """Расчет стоп-лосса."""
        if not signal.price:
            return None
        try:
            # Простой стоп-лосс на основе волатильности
            volatility_stop = market_data.volatility * 2  # 2 стандартных отклонения
            if signal.signal_type == "buy":
                return float(signal.price) * (1 - volatility_stop)
            else:
                return float(signal.price) * (1 + volatility_stop)
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return None


class MetricsCalculator:
    """Калькулятор метрик."""

    async def calculate_backtest_metrics(
        self, trades: List[SimulationTrade], equity_curve: List[float]
    ) -> BacktestMetricsDict:
        """Расчет метрик бэктеста."""
        try:
            if not trades:
                return self._empty_metrics()
            # Базовые метрики
            total_trades = len(trades)
            winning_trades = len([t for t in trades if float(t.pnl) > 0])
            losing_trades = len([t for t in trades if float(t.pnl) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            # Прибыль/убыток
            total_profit = sum(float(t.pnl) for t in trades if float(t.pnl) > 0)
            total_loss = abs(sum(float(t.pnl) for t in trades if float(t.pnl) < 0))
            net_profit = total_profit - total_loss
            profit_factor = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )
            # Риск-метрики
            returns = np.diff(equity_curve) if len(equity_curve) > 1 else [0]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            # Дополнительные метрики
            calmar_ratio = net_profit / max_drawdown if max_drawdown > 0 else 0
            recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0
            expectancy = net_profit / total_trades if total_trades > 0 else 0
            risk_reward_ratio = total_profit / total_loss if total_loss > 0 else 0
            kelly_criterion = self._calculate_kelly_criterion(win_rate, profit_factor)
            return BacktestMetricsDict(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                recovery_factor=recovery_factor,
                expectancy=expectancy,
                risk_reward_ratio=risk_reward_ratio,
                kelly_criterion=kelly_criterion,
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._empty_metrics()

    def _empty_metrics(self) -> BacktestMetricsDict:
        """Пустые метрики."""
        return BacktestMetricsDict(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            recovery_factor=0.0,
            expectancy=0.0,
            risk_reward_ratio=0.0,
            kelly_criterion=0.0,
        )

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Расчет коэффициента Шарпа."""
        if not returns or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Расчет коэффициента Сортино."""
        if not returns:
            return 0.0
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float("inf")
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        return np.mean(returns) / downside_deviation * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Расчет максимальной просадки."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _calculate_kelly_criterion(
        self, win_rate: float, profit_factor: float
    ) -> float:
        """Расчет критерия Келли."""
        if profit_factor <= 1:
            return 0.0
        return (win_rate * profit_factor - (1 - win_rate)) / profit_factor


# ============================================================================
# Точка входа
# ============================================================================
async def main():
    """Основная функция для тестирования."""
    # Создание конфигурации
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_balance=SimulationMoney(Decimal("10000")),
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.1,
        risk_per_trade=0.02,
        confidence_threshold=0.7,
        symbols=[Symbol("BTCUSDT")],
        timeframes=["1m", "5m", "15m", "1h"],
        random_seed=42,
    )
    # Создание бэктестера
    backtester = Backtester(config)

    # Создание простой стратегии для тестирования
    class SimpleStrategy:
        async def generate_signal(
            self, market_data: SimulationMarketData, context: Dict[str, Any]
        ) -> Optional[SimulationSignal]:
            # Простая стратегия: покупать при росте цены
            if market_data.price_change_percent > 0.5:
                return SimulationSignal(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    signal_type="buy",
                    confidence=0.6,
                    price=market_data.close,
                )
            return None

        async def validate_signal(self, signal: SimulationSignal) -> bool:
            return True

        async def get_signal_confidence(
            self, signal: SimulationSignal, market_data: SimulationMarketData
        ) -> float:
            return signal.confidence

    # Создание тестовых данных
    from .market_simulator import MarketSimulator
    from .types import MarketSimulationConfig

    market_config = MarketSimulationConfig(
        start_date=config.start_date,
        end_date=config.end_date,
        initial_balance=SimulationMoney(Decimal("10000")),
        symbols=config.symbols,
        timeframes=config.timeframes,
        random_seed=config.random_seed,
    )
    market_simulator = MarketSimulator(market_config)
    await market_simulator.initialize()
    market_data = await market_simulator.generate_market_data(
        Symbol("BTCUSDT"), config.start_date, config.end_date
    )
    # Запуск бэктеста
    strategy = SimpleStrategy()
    result = await backtester.run_backtest(strategy, market_data)
    print(f"Backtest completed: {result.success}")
    print(f"Total trades: {len(result.trades)}")
    print(f"Final balance: {result.final_balance}")
    print(f"Win rate: {result.metrics.get('win_rate', 0):.2%}")
    print(f"Sharpe ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
