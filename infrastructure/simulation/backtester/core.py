"""
Основной модуль бэктестера с полной реализацией всех методов.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

# Удаляю: from shared.logging import setup_logger

from .data_processor import DataProcessor
from .metrics import MetricsCalculator
from .trade_executor import TradeExecutor
from .types import (
    BacktestConfig,
    BacktestResult,
    MarketData,
    RiskInfo,
    Signal,
    Tag,
    Trade,
    TradeAction,
    TradeDirection,
    TradeError,
    TradeEvent,
    TradeStatus,
)

# Удаляю: logger = setup_logger(__name__)


# ============================================================================
# Protocol интерфейсы
# ============================================================================
@runtime_checkable
class StrategyProtocol(Protocol):
    """Протокол для торговой стратегии."""

    async def generate_signal(
        self, market_data: MarketData, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]: ...
    async def validate_signal(self, signal: Signal) -> bool: ...
    async def get_signal_confidence(
        self, signal: Signal, market_data: MarketData
    ) -> float: ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Протокол для риск-менеджера."""

    async def check_risk_limits(
        self, signal: Signal, current_balance: float, current_positions: List[Trade]
    ) -> bool: ...
    async def calculate_position_size(
        self, signal: Signal, balance: float, risk_per_trade: float
    ) -> float: ...
    async def calculate_stop_loss(
        self, signal: Signal, market_data: MarketData
    ) -> Optional[float]: ...


# ============================================================================
# Конфигурации и метрики
# ============================================================================
@dataclass
class BacktesterConfig:
    """Расширенная конфигурация бэктестера."""

    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.2
    risk_per_trade: float = 0.02
    confidence_threshold: float = 0.7
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    log_dir: str = "logs"
    max_trades: Optional[int] = None
    random_seed: Optional[int] = None
    tags: List[Tag] = field(default_factory=list)
    notes: Optional[str] = None
    use_realistic_slippage: bool = True
    use_market_impact: bool = True
    use_latency: bool = True
    use_partial_fills: bool = True
    calculate_metrics: bool = True
    generate_plots: bool = True
    save_trades: bool = True
    save_equity_curve: bool = True
    min_trades: int = 10
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.1


@dataclass
class BacktesterMetrics:
    """Расширенные метрики бэктестера."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    events: List[TradeEvent] = field(default_factory=list)
    tags: List[Tag] = field(default_factory=list)
    risk: Optional[RiskInfo] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    simulation_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class MarketState:
    """Состояние рынка."""

    current_price: float = 0.0
    current_volume: float = 0.0
    current_regime: str = "unknown"
    volatility: float = 0.0
    trend_strength: float = 0.0
    momentum: float = 0.0
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    regime_history: List[str] = field(default_factory=list)

    def update(self, market_data: MarketData) -> None:
        """Обновление состояния рынка."""
        self.current_price = market_data.close.value if hasattr(market_data.close, 'value') else market_data.close
        self.current_volume = market_data.volume.value if hasattr(market_data.volume, 'value') else market_data.volume
        self.current_regime = market_data.metadata.get("regime", "unknown")
        self.volatility = market_data.metadata.get("volatility", 0.0)
        self.trend_strength = market_data.metadata.get("trend_strength", 0.0)
        self.momentum = market_data.metadata.get("momentum", 0.0)
        # Обновление истории
        self.price_history.append(self.current_price)
        self.volume_history.append(self.current_volume)
        self.regime_history.append(self.current_regime)
        # Ограничение размера истории
        max_history = 1000
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.regime_history = self.regime_history[-max_history:]


@dataclass
class SignalProcessor:
    """Обработчик сигналов."""

    async def process_signals(
        self, symbol: str, strategy: str, market_state: MarketState
    ) -> Optional[Dict[str, Any]]:
        """Обработка сигналов."""
        try:
            # Простая логика генерации сигналов
            if len(market_state.price_history) < 10:
                return None
            # Анализ тренда
            recent_prices = market_state.price_history[-10:]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            # Генерация сигнала на основе тренда
            if price_change > 0.01:  # Рост более 1%
                return {
                    "direction": "long",
                    "confidence": 0.7,
                    "entry_price": market_state.current_price,
                    "stop_loss": market_state.current_price * 0.98,
                    "take_profit": market_state.current_price * 1.03,
                    "volume": 0.1,
                    "leverage": 1.0,
                    "metadata": {"trend": "up", "price_change": price_change},
                    "tags": [],
                    "timeframe": "1m",
                    "risk": None,
                }
            elif price_change < -0.01:  # Падение более 1%
                return {
                    "direction": "short",
                    "confidence": 0.6,
                    "entry_price": market_state.current_price,
                    "stop_loss": market_state.current_price * 1.02,
                    "take_profit": market_state.current_price * 0.97,
                    "volume": 0.1,
                    "leverage": 1.0,
                    "metadata": {"trend": "down", "price_change": price_change},
                    "tags": [],
                    "timeframe": "1m",
                    "risk": None,
                }
            return None
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
            return None


# ============================================================================
# Основной класс бэктестера
# ============================================================================
class Backtester:
    """Промышленный бэктестер торговых стратегий с полной реализацией."""

    def __init__(self, config: Optional[BacktesterConfig] = None):
        """Инициализация бэктестера."""
        self.config = config or BacktesterConfig()
        self.metrics = BacktesterMetrics()
        self.market_state = MarketState()
        self.signal_processor = SignalProcessor()
        self.trade_executor = TradeExecutor(self.config.__dict__)
        self.data_processor = DataProcessor(self.config.__dict__)
        self.metrics_calculator = MetricsCalculator()
        self.results: Dict[str, Any] = {"trades": [], "equity_curve": [], "metrics": {}}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        log_path = Path(self.config.log_dir) / "backtester.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    async def run_backtest(
        self, symbol: str, strategy: StrategyProtocol, data: Any
    ) -> BacktestResult:
        """Запуск бэктеста с полной реализацией."""
        try:
            logger.info(
                f"Starting backtest for {symbol} with strategy {strategy.__class__.__name__}"
            )
            # Валидация данных
            is_valid, error = self.data_processor.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")
            # Обработка данных
            processed_data = self.data_processor.preprocess_data(data)
            market_data = self.data_processor.process_market_data(
                processed_data, symbol=symbol
            )
            # Инициализация состояния
            balance = self.config.initial_balance
            position = None
            trades = []
            equity_curve = [balance]
            self.metrics.start_time = datetime.now()
            # Основной цикл бэктеста
            for i, market_point in enumerate(market_data):
                # Обновление состояния рынка
                # Преобразуем domain MarketData в infrastructure MarketData
                infrastructure_market_data = MarketData(
                    symbol=market_point.symbol,
                    timestamp=market_point.timestamp,
                    timeframe="1m",  # Добавляем недостающий параметр timeframe
                    open=float(market_point.open.value) if hasattr(market_point.open, 'value') else float(market_point.open),
                    high=float(market_point.high.value) if hasattr(market_point.high, 'value') else float(market_point.high),
                    low=float(market_point.low.value) if hasattr(market_point.low, 'value') else float(market_point.low),
                    close=float(market_point.close.value) if hasattr(market_point.close, 'value') else float(market_point.close),
                    volume=float(market_point.volume.value) if hasattr(market_point.volume, 'value') else float(market_point.volume),
                    metadata=market_point.metadata
                )
                self.market_state.update(infrastructure_market_data)
                # Генерация сигнала
                signal = await self._get_signal(strategy, infrastructure_market_data)
                if signal:
                    # Преобразуем infrastructure Signal в domain Signal
                    from domain.entities.trading import Signal as DomainSignal
                    domain_signal = DomainSignal(
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        confidence=signal.confidence,
                    )
                    # Преобразуем infrastructure MarketData в domain MarketData
                    from domain.entities.market import MarketData as DomainMarketData
                    domain_market_data = DomainMarketData(
                        symbol=infrastructure_market_data.symbol,
                        timestamp=infrastructure_market_data.timestamp,
                        open=infrastructure_market_data.open,
                        high=infrastructure_market_data.high,
                        low=infrastructure_market_data.low,
                        close=infrastructure_market_data.close,
                        volume=infrastructure_market_data.volume,
                    )
                    # Исполнение сделки
                    trade_result = await self.trade_executor.execute_trade(
                        signal=domain_signal,
                        market_data=domain_market_data,
                        balance=balance,
                        position=None,  # Исправление: передаем None вместо Trade
                    )
                    trade, new_balance = trade_result
                    if trade:
                        # Преобразуем domain Trade в infrastructure Trade
                        from infrastructure.simulation.backtester.types import Trade as InfraTrade
                        infra_trade = InfraTrade(
                            id=trade.id,
                            symbol=trade.symbol,
                            side=trade.side,
                            order_type=trade.order_type,
                            quantity=trade.quantity,
                            price=trade.price,
                            status=trade.status,
                            created_at=trade.created_at,
                            updated_at=trade.updated_at,
                        )
                        trades.append(infra_trade)
                        balance = new_balance
                        # Проверяем тип trade и его атрибуты
                        if hasattr(trade, 'action') and hasattr(trade.action, 'name'):
                            position = infra_trade if trade.action.name == 'OPEN' else None
                        else:
                            position = None
                        # Обновление метрик
                        self.metrics.total_trades += 1
                        if hasattr(trade, 'pnl') and hasattr(trade.pnl, 'value') and trade.pnl.value > 0:
                            self.metrics.winning_trades += 1
                        elif hasattr(trade, 'pnl') and hasattr(trade.pnl, 'value') and trade.pnl.value < 0:
                            self.metrics.losing_trades += 1
                        # Добавление события
                        self.metrics.events.append(
                            TradeEvent(
                                event_type="trade_executed",
                                timestamp=infrastructure_market_data.timestamp,
                                details={"trade": trade},
                            )
                        )
                equity_curve.append(balance)
                # Проверка остановки
                if self.config.max_trades and len(trades) >= self.config.max_trades:
                    break
            self.metrics.end_time = datetime.now()
            self.metrics.simulation_time = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()
            # Расчет метрик
            # Используем infrastructure Trade для calculate_metrics
            metrics = self.metrics_calculator.calculate_metrics(trades, equity_curve)
            # Создание результата
            result = BacktestResult(
                trades=trades,  # Используем infrastructure trades для BacktestResult
                equity_curve=equity_curve,
                metrics=metrics,
                config=BacktestConfig(**self.config.__dict__),
                start_time=market_data[0].timestamp,
                end_time=market_data[-1].timestamp,
                initial_balance=self.config.initial_balance,
                final_balance=balance,
                total_trades=len(trades),
                winning_trades=len([t for t in trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value') and t.pnl.value > 0]),
                losing_trades=len([t for t in trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value') and t.pnl.value < 0]),
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                max_drawdown=metrics["max_drawdown"],
                sharpe_ratio=metrics["sharpe_ratio"],
                sortino_ratio=metrics["sortino_ratio"],
                calmar_ratio=metrics["calmar_ratio"],
                recovery_factor=metrics["recovery_factor"],
                expectancy=metrics["expectancy"],
                risk_reward_ratio=metrics["risk_reward_ratio"],
                kelly_criterion=metrics["kelly_criterion"],
                tags=self.config.tags,
                errors=[
                    TradeError(code=0, message=e, timestamp=datetime.now())
                    for e in (
                        [self.metrics.last_error] if self.metrics.last_error else []
                    )
                ],
                events=self.metrics.events,
                risk=self.metrics.risk,
            )
            logger.info(f"Backtest completed for {symbol}")
            return result
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error running backtest: {str(e)}")
            raise

    async def run_portfolio_backtest(
        self, symbols: List[str], strategy: StrategyProtocol, data_dict: Dict[str, Any]
    ) -> List[BacktestResult]:
        """Портфельный бэктест по нескольким активам."""
        results = []
        for symbol in symbols:
            if symbol in data_dict:
                data = data_dict[symbol]
                result = await self.run_backtest(symbol, strategy, data)
                results.append(result)
            else:
                logger.warning(f"No data found for symbol {symbol}")
        return results

    async def _get_signal(
        self, strategy: StrategyProtocol, market_data: MarketData
    ) -> Optional[Signal]:
        """Получение сигнала от стратегии."""
        try:
            context = {
                "balance": self.config.initial_balance,
                "market_state": self.market_state,
                "equity_curve": self.results.get("equity_curve", []),
            }
            signal_data = await strategy.generate_signal(market_data, context)
            if not signal_data:
                return None
            return Signal(
                symbol=market_data.symbol,
                direction=(
                    TradeDirection.LONG
                    if signal_data["direction"] == "long"
                    else TradeDirection.SHORT
                ),
                confidence=signal_data["confidence"],
                entry_price=signal_data["entry_price"],
                stop_loss=signal_data.get("stop_loss"),
                take_profit=signal_data.get("take_profit"),
                volume=signal_data.get("volume"),
                leverage=signal_data.get("leverage", 1.0),
                metadata=signal_data.get("metadata", {}),
                tags=signal_data.get("tags", []),
                timeframe=signal_data.get("timeframe"),
                risk=signal_data.get("risk"),
            )
        except Exception as e:
            logger.error(f"Error getting signal: {str(e)}")
            return None

    def save_results(self, result: BacktestResult, path: str) -> None:
        """Сохранение результатов."""
        try:
            results_path = Path(path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            # Преобразование в JSON-совместимый формат
            result_dict = {
                "config": {
                    "initial_balance": result.config.initial_balance,
                    "commission": result.config.commission,
                    "slippage": result.config.slippage,
                    "max_position_size": result.config.max_position_size,
                    "risk_per_trade": result.config.risk_per_trade,
                    "confidence_threshold": result.config.confidence_threshold,
                    "symbols": result.config.symbols,
                    "timeframes": result.config.timeframes,
                    "random_seed": result.config.random_seed,
                },
                "trades": [
                    {
                        "symbol": trade.symbol if hasattr(trade, 'symbol') else str(trade),
                        "action": trade.action.name if hasattr(trade, 'action') and hasattr(trade.action, 'name') else "UNKNOWN",
                        "direction": trade.direction.name if hasattr(trade, 'direction') and hasattr(trade.direction, 'name') else "UNKNOWN",
                        "volume": trade.volume if hasattr(trade, 'volume') else 0.0,
                        "price": trade.price if hasattr(trade, 'price') else 0.0,
                        "commission": trade.commission if hasattr(trade, 'commission') else 0.0,
                        "pnl": trade.pnl if hasattr(trade, 'pnl') else 0.0,
                        "timestamp": trade.timestamp.isoformat() if hasattr(trade, 'timestamp') and hasattr(trade.timestamp, 'isoformat') else datetime.now().isoformat(),
                        "status": trade.status.name if hasattr(trade, 'status') and hasattr(trade.status, 'name') else "UNKNOWN",
                    }
                    for trade in result.trades
                ],
                "equity_curve": result.equity_curve,
                "metrics": result.metrics,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "recovery_factor": result.recovery_factor,
                "expectancy": result.expectancy,
                "risk_reward_ratio": result.risk_reward_ratio,
                "kelly_criterion": result.kelly_criterion,
                "tags": [{"name": tag.name, "value": tag.value} for tag in result.tags],
                "errors": [
                    {
                        "code": error.code,
                        "message": error.message,
                        "timestamp": error.timestamp.isoformat(),
                    }
                    for error in result.errors
                ],
                "events": [
                    {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "details": event.details,
                    }
                    for event in result.events
                ],
            }
            with open(results_path, "w") as f:
                json.dump(result_dict, f, indent=2)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def load_results(self, path: str) -> BacktestResult:
        """Загрузка результатов."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Восстановление конфигурации
            config = BacktestConfig(**data["config"])
            # Восстановление сделок
            trades = []
            for trade_data in data["trades"]:
                trade = Trade(
                    symbol=trade_data["symbol"],
                    action=TradeAction[trade_data["action"]],
                    direction=TradeDirection[trade_data["direction"]],
                    volume=trade_data["volume"],
                    price=trade_data["price"],
                    commission=trade_data["commission"],
                    pnl=trade_data["pnl"],
                    timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                    status=TradeStatus[trade_data["status"]],
                )
                trades.append(trade)
            # Восстановление событий
            events = []
            for event_data in data["events"]:
                event = TradeEvent(
                    event_type=event_data["event_type"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    details=event_data["details"],
                )
                events.append(event)
            # Восстановление ошибок
            errors = []
            for error_data in data["errors"]:
                error = TradeError(
                    code=error_data["code"],
                    message=error_data["message"],
                    timestamp=datetime.fromisoformat(error_data["timestamp"]),
                )
                errors.append(error)
            # Восстановление тегов
            tags = []
            for tag_data in data["tags"]:
                tag = Tag(name=tag_data["name"], value=tag_data["value"])
                tags.append(tag)
            # Создание результата
            result = BacktestResult(
                trades=trades,
                equity_curve=data["equity_curve"],
                metrics=data["metrics"],
                config=config,
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]),
                initial_balance=data["initial_balance"],
                final_balance=data["final_balance"],
                total_trades=data["total_trades"],
                winning_trades=data["winning_trades"],
                losing_trades=data["losing_trades"],
                win_rate=data["win_rate"],
                profit_factor=data["profit_factor"],
                max_drawdown=data["max_drawdown"],
                sharpe_ratio=data["sharpe_ratio"],
                sortino_ratio=data["sortino_ratio"],
                calmar_ratio=data["calmar_ratio"],
                recovery_factor=data["recovery_factor"],
                expectancy=data["expectancy"],
                risk_reward_ratio=data["risk_reward_ratio"],
                kelly_criterion=data["kelly_criterion"],
                tags=tags,
                errors=errors,
                events=events,
            )
            logger.info(f"Results loaded from {path}")
            return result
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            raise
