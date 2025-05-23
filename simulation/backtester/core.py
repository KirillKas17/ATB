import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.strategy import Signal
from utils.logger import setup_logger

from .data_processor import DataProcessor
from .metrics import MetricsCalculator
from .trade_executor import TradeExecutor
from .types import (
    BacktestConfig,
    BacktestResult,
    RiskInfo,
    Tag,
    TradeAction,
    TradeDirection,
    TradeError,
    TradeEvent,
)

logger = setup_logger(__name__)


@dataclass
class BacktesterConfig:
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.001
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    confidence_threshold: float = 0.7
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    log_dir: str = "logs"
    max_trades: Optional[int] = None
    random_seed: Optional[int] = None
    tags: List[Tag] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class BacktesterMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    events: List[TradeEvent] = field(default_factory=list)
    tags: List[Tag] = field(default_factory=list)
    risk: Optional[RiskInfo] = None


class Backtester:
    """Бэктестер торговых стратегий (расширенный)"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = (
            BacktesterConfig(**config)
            if config and not isinstance(config, BacktesterConfig)
            else (config or BacktesterConfig())
        )
        self.metrics = BacktesterMetrics()
        self.market_state = MarketState()
        self.signal_processor = SignalProcessor()
        self.trade_executor = TradeExecutor(self.config.__dict__)
        self.data_processor = DataProcessor(self.config.__dict__)
        self.metrics_calculator = MetricsCalculator()
        self.results = {"trades": [], "equity_curve": [], "metrics": {}}
        self._setup_logger()

    def _setup_logger(self):
        logger.add(
            f"{self.config.log_dir}/backtester_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    async def run_backtest(
        self, symbol: str, strategy: str, data: Any
    ) -> BacktestResult:
        """
        Запуск бэктеста (расширенный, поддержка событий, тегов, ошибок, рисков)
        """
        try:
            logger.info(f"Starting backtest for {symbol} with strategy {strategy}")
            is_valid, error = self.data_processor.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")
            processed_data = self.data_processor.preprocess_data(data)
            market_data = self.data_processor.process_market_data(
                processed_data, symbol=symbol
            )
            balance = self.config.initial_balance
            position = None
            trades = []
            equity_curve = [balance]
            for market_point in market_data:
                self.market_state.update(market_point)
                signal = await self._get_signal(symbol, strategy)
                if signal:
                    trade, new_balance = self.trade_executor.execute_trade(
                        signal=signal,
                        market_data=market_point,
                        balance=balance,
                        position=position,
                    )
                    if trade:
                        trades.append(trade)
                        balance = new_balance
                        position = trade if trade.action == TradeAction.OPEN else None
                        self.metrics.total_trades += 1
                        if trade.pnl > 0:
                            self.metrics.winning_trades += 1
                        elif trade.pnl < 0:
                            self.metrics.losing_trades += 1
                        self.metrics.events.append(
                            TradeEvent(
                                event_type="trade_executed",
                                timestamp=market_point.timestamp,
                                details={"trade": trade},
                            )
                        )
                equity_curve.append(balance)
            metrics = self.metrics_calculator.calculate_metrics(trades, equity_curve)
            result = BacktestResult(
                trades=trades,
                equity_curve=equity_curve,
                metrics=metrics,
                config=BacktestConfig(**self.config.__dict__),
                start_time=market_data[0].timestamp,
                end_time=market_data[-1].timestamp,
                initial_balance=self.config.initial_balance,
                final_balance=balance,
                total_trades=len(trades),
                winning_trades=len([t for t in trades if t.pnl > 0]),
                losing_trades=len([t for t in trades if t.pnl < 0]),
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
        self, symbols: List[str], strategy: str, data_dict: Dict[str, Any]
    ) -> List[BacktestResult]:
        """
        Портфельный бэктест по нескольким активам
        """
        results = []
        for symbol in symbols:
            data = data_dict[symbol]
            result = await self.run_backtest(symbol, strategy, data)
            results.append(result)
        return results

    async def _get_signal(self, symbol: str, strategy: str) -> Optional[Signal]:
        try:
            signal_data = await self.signal_processor.process_signals(
                symbol=symbol, strategy=strategy, market_state=self.market_state
            )
            if not signal_data:
                return None
            return Signal(
                symbol=symbol,
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
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error getting signal: {str(e)}")
            return None

    def save_results(self, result: BacktestResult, path: str):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            result_dict = {
                "trades": [
                    {
                        "symbol": t.symbol,
                        "action": t.action.name,
                        "direction": t.direction.name,
                        "volume": t.volume,
                        "price": t.price,
                        "commission": t.commission,
                        "pnl": t.pnl,
                        "timestamp": t.timestamp.isoformat(),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "stop_loss": t.stop_loss,
                        "take_profit": t.take_profit,
                        "tags": [tag.name for tag in getattr(t, "tags", [])],
                        "metadata": t.metadata,
                    }
                    for t in result.trades
                ],
                "equity_curve": result.equity_curve,
                "metrics": result.metrics,
                "config": result.config.__dict__,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "tags": [tag.name for tag in getattr(result, "tags", [])],
                "events": [e.event_type for e in getattr(result, "events", [])],
                "errors": [e.message for e in getattr(result, "errors", [])],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def load_results(self, path: str) -> BacktestResult:
        try:
            with open(path, "r", encoding="utf-8") as f:
                result_dict = json.load(f)
            # Здесь можно добавить восстановление объектов из словаря
            return result_dict
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None
