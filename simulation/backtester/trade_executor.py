from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .types import (
    MarketData,
    RiskInfo,
    Signal,
    Tag,
    Trade,
    TradeAction,
    TradeDirection,
    TradeEvent,
)


@dataclass
class TradeExecutorConfig:
    commission: float = 0.001
    slippage: float = 0.0005
    risk_per_trade: float = 0.01
    max_position_size: float = 0.2
    confidence_threshold: float = 0.7
    partial_close: bool = True
    trailing_stop: bool = True
    trailing_step: float = 0.002
    max_trades: Optional[int] = None
    log_dir: str = "logs"


@dataclass
class TradeExecutorMetrics:
    executed_trades: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    events: List[TradeEvent] = field(default_factory=list)
    tags: List[Tag] = field(default_factory=list)
    risk: Optional[RiskInfo] = None


class TradeExecutor:
    """Исполнитель сделок (расширенный)"""

    def __init__(self, config: Dict):
        self.config = (
            TradeExecutorConfig(**config)
            if not isinstance(config, TradeExecutorConfig)
            else config
        )
        self.metrics = TradeExecutorMetrics()
        self._setup_logger()

    def _setup_logger(self):
        logger.add(
            f"{self.config.log_dir}/trade_executor_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def calculate_position_size(
        self, balance: float, risk: float, stop_loss: float, entry: float
    ) -> float:
        """
        Расчет размера позиции с учетом риска и ограничений
        """
        try:
            risk_amount = balance * risk
            position_size = risk_amount / max(abs(entry - stop_loss), 1e-8)
            max_size = balance * self.config.max_position_size
            position_size = min(position_size, max_size)
            return position_size
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def execute_trade(
        self,
        signal: Signal,
        market_data: MarketData,
        balance: float,
        position: Optional[Trade] = None,
    ) -> Tuple[Optional[Trade], float]:
        """
        Исполнение сделки с поддержкой частичного закрытия, трейлинг-стопа, событий и тегов
        """
        try:
            if position:
                if self._should_close_position(signal, position, market_data):
                    return self._close_position(position, market_data, balance)
                if self.config.trailing_stop:
                    self._update_trailing_stop(position, market_data)
                return position, balance
            if signal.confidence >= self.config.confidence_threshold:
                return self._open_position(signal, market_data, balance)
            return None, balance
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error executing trade: {str(e)}")
            return None, balance

    def _should_close_position(
        self, signal: Signal, position: Trade, market_data: MarketData
    ) -> bool:
        """
        Проверка необходимости закрытия позиции (расширенная)
        """
        try:
            if position.direction == TradeDirection.LONG:
                if (
                    market_data.close is not None
                    and position.stop_loss is not None
                    and market_data.close <= position.stop_loss
                ):
                    return True
                if (
                    market_data.close is not None
                    and position.take_profit is not None
                    and market_data.close >= position.take_profit
                ):
                    return True
            else:
                if (
                    market_data.close is not None
                    and position.stop_loss is not None
                    and market_data.close >= position.stop_loss
                ):
                    return True
                if (
                    market_data.close is not None
                    and position.take_profit is not None
                    and market_data.close <= position.take_profit
                ):
                    return True
            if signal.direction != position.direction:
                return True
            return False
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error checking position closure: {str(e)}")
            return False

    def _open_position(
        self, signal: Signal, market_data: MarketData, balance: float
    ) -> Tuple[Optional[Trade], float]:
        """
        Открытие позиции с поддержкой тегов, событий, рисков
        """
        try:
            volume = self.calculate_position_size(
                balance=balance,
                risk=self.config.risk_per_trade,
                stop_loss=signal.stop_loss,
                entry=signal.entry_price,
            )
            if signal.direction == TradeDirection.LONG:
                price = market_data.close * (1 + self.config.slippage)
            else:
                price = market_data.close * (1 - self.config.slippage)
            commission = price * volume * self.config.commission
            trade = Trade(
                symbol=signal.symbol,
                action=TradeAction.OPEN,
                direction=signal.direction,
                volume=volume,
                price=price,
                commission=commission,
                pnl=0.0,
                timestamp=market_data.timestamp,
                entry_price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                tags=signal.tags,
                metadata=signal.metadata,
                timeframe=signal.timeframe,
                risk=signal.risk,
            )
            self.metrics.executed_trades += 1
            self.metrics.events.append(
                TradeEvent(
                    event_type="open_position",
                    timestamp=market_data.timestamp,
                    details={"trade": trade},
                )
            )
            new_balance = balance - commission
            return trade, new_balance
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error opening position: {str(e)}")
            return None, balance

    def _close_position(
        self, position: Trade, market_data: MarketData, balance: float
    ) -> Tuple[Optional[Trade], float]:
        """
        Закрытие позиции с поддержкой тегов, событий, рисков
        """
        try:
            if position.direction == TradeDirection.LONG:
                price = market_data.close * (1 - self.config.slippage)
            else:
                price = market_data.close * (1 + self.config.slippage)
            commission = price * position.volume * self.config.commission
            if position.direction == TradeDirection.LONG:
                pnl = (price - position.entry_price) * position.volume
            else:
                pnl = (position.entry_price - price) * position.volume
            pnl -= commission
            trade = Trade(
                symbol=position.symbol,
                action=TradeAction.CLOSE,
                direction=position.direction,
                volume=position.volume,
                price=price,
                commission=commission,
                pnl=pnl,
                timestamp=market_data.timestamp,
                entry_price=position.entry_price,
                exit_price=price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                tags=position.tags,
                metadata=position.metadata,
                timeframe=position.timeframe,
                risk=position.risk,
            )
            self.metrics.executed_trades += 1
            self.metrics.events.append(
                TradeEvent(
                    event_type="close_position",
                    timestamp=market_data.timestamp,
                    details={"trade": trade},
                )
            )
            new_balance = balance + pnl
            return trade, new_balance
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error closing position: {str(e)}")
            return None, balance

    def _update_trailing_stop(self, position: Trade, market_data: MarketData):
        """
        Обновление трейлинг-стопа для позиции
        """
        try:
            if not self.config.trailing_stop or not position.stop_loss:
                return
            if position.direction == TradeDirection.LONG:
                new_stop = max(
                    position.stop_loss, market_data.close - self.config.trailing_step
                )
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = min(
                    position.stop_loss, market_data.close + self.config.trailing_step
                )
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
            self.metrics.events.append(
                TradeEvent(
                    event_type="update_trailing_stop",
                    timestamp=market_data.timestamp,
                    details={"position": position},
                )
            )
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error updating trailing stop: {str(e)}")
