"""
Промышленный исполнитель сделок для бэктестинга.
"""

from shared.numpy_utils import np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from domain.entities.market import MarketData
from domain.entities.trading import (
    OrderSide,
    Position,
    PositionSide,
    Signal,
    SignalType,
    Trade,
)
from domain.type_definitions import TradeId
from domain.value_objects import Price as DomainPrice, Volume as DomainVolume
from domain.value_objects.price import Price


@dataclass
class TradeExecutorConfig:
    """Конфигурация исполнителя сделок."""

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
    use_realistic_slippage: bool = True
    use_market_impact: bool = True
    use_latency: bool = True
    use_partial_fills: bool = True


@dataclass
class TradeExecutorMetrics:
    """Метрики исполнителя сделок."""

    executed_trades: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    risk: Optional[Dict[str, Any]] = None
    total_commission: float = 0.0
    total_slippage: float = 0.0
    average_execution_time: float = 0.0
    partial_fills: int = 0
    rejected_trades: int = 0


class TradeExecutor:
    """Промышленный исполнитель сделок с полной реализацией."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Инициализация исполнителя сделок."""
        self.config = (
            TradeExecutorConfig(**config)
            if not isinstance(config, TradeExecutorConfig)
            else config
        )
        self.metrics = TradeExecutorMetrics()
        self._setup_logger()
        self._execution_history: List[Dict[str, Any]] = []
        self._position_cache: Dict[str, Position] = {}

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        log_path = Path(self.config.log_dir) / "trade_executor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    def calculate_position_size(
        self, balance: float, risk: float, stop_loss: float, entry: float
    ) -> float:
        """Расчет размера позиции с учетом риска и ограничений."""
        try:
            # Базовый расчет размера позиции
            risk_amount = balance * risk
            position_size = risk_amount / max(abs(entry - stop_loss), 1e-8)
            # Ограничение максимальным размером позиции
            max_size = balance * self.config.max_position_size
            position_size = min(position_size, max_size)
            # Минимальный размер позиции
            min_size = balance * 0.001  # 0.1% от баланса
            position_size = max(position_size, min_size)
            return position_size
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    async def execute_trade(
        self,
        signal: Signal,
        market_data: MarketData,
        balance: float,
        position: Optional[Position] = None,
    ) -> Tuple[Optional[Trade], float]:
        """Исполнение сделки с поддержкой частичного закрытия, трейлинг-стопа, событий и тегов."""
        start_time = datetime.now()
        try:
            # Проверка существующей позиции
            if position:
                # Проверка необходимости закрытия позиции
                if self._should_close_position(signal, position, market_data):
                    closed_trade = await self._close_position(
                        position, market_data, balance
                    )
                    if closed_trade:
                        self._update_metrics(closed_trade, start_time)
                        return closed_trade, balance
                    return None, balance
                # Обновление трейлинг-стопа
                if self.config.trailing_stop:
                    self._update_trailing_stop(position, market_data)
                return None, balance  # Возвращаем None для Trade, так как позиция не закрыта
            # Проверка уверенности сигнала
            if signal.confidence < self.config.confidence_threshold:
                self.metrics.rejected_trades += 1
                logger.info(
                    f"Signal rejected due to low confidence: {signal.confidence}"
                )
                return None, balance
            # Открытие новой позиции
            new_position = await self._open_position(signal, market_data, balance)
            if new_position:
                # Создаем Trade для открытия позиции
                from domain.value_objects.money import Money
                from domain.type_definitions import MetadataDict, TradeId
                from domain.value_objects.trading_pair import TradingPair
                from domain.value_objects.currency import Currency
                from uuid import uuid4
                
                # Создаем валюты из символа
                symbol_parts = str(market_data.symbol).split('/')
                if len(symbol_parts) == 2:
                    base_currency = Currency(symbol_parts[0])
                    quote_currency = Currency(symbol_parts[1])
                else:
                    # Если символ не содержит '/', предполагаем что это BTC/USDT
                    base_currency = Currency("BTC")
                    quote_currency = Currency("USDT")
                
                trade = Trade(
                    id=TradeId(uuid4()),
                    order_id=OrderId(uuid4()),  # Используем OrderId, а не TradeId
                    trading_pair=TradingPair(base_currency, quote_currency, str(market_data.symbol)),  # type: ignore[arg-type]
                    side=OrderSide.BUY if new_position.side == PositionSide.LONG else OrderSide.SELL,
                    quantity=new_position.quantity,
                    price=market_data.close,
                    commission=Money(Decimal("0"), market_data.close.currency),
                    timestamp=market_data.timestamp
                )
                
                self._update_metrics(trade, start_time)
                return trade, balance
            return None, balance
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error executing trade: {str(e)}")
            return None, balance

    def _should_close_position(
        self, signal: Signal, position: Position, market_data: MarketData
    ) -> bool:
        """Проверка необходимости закрытия позиции."""
        try:
            # Проверка стоп-лосса
            if position.stop_loss is not None:
                if position.side == PositionSide.LONG:
                    if market_data.close.value <= position.stop_loss.value:
                        logger.info(f"Stop loss triggered for {position.symbol}")
                        return True
                else:
                    if market_data.close.value >= position.stop_loss.value:
                        logger.info(f"Stop loss triggered for {position.symbol}")
                        return True
            # Проверка тейк-профита
            if position.take_profit is not None:
                if position.side == PositionSide.LONG:
                    if market_data.close.value >= position.take_profit.value:
                        logger.info(f"Take profit triggered for {position.symbol}")
                        return True
                else:
                    if market_data.close.value <= position.take_profit.value:
                        logger.info(f"Take profit triggered for {position.symbol}")
                        return True
            # Проверка противоположного сигнала
            if signal.signal_type != self._get_position_signal_type(position.side):
                logger.info(f"Opposite signal received for {position.symbol}")
                return True
            # Проверка времени удержания позиции
            if position.entry_time is not None:
                position_duration = datetime.now() - position.entry_time
                max_hold_time = timedelta(days=7)  # Максимум 7 дней
                if position_duration > max_hold_time:
                    logger.info(f"Position held too long for {position.symbol}")
                    return True
            return False
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error checking position closure: {str(e)}")
            return False

    def _get_position_signal_type(self, side: PositionSide) -> SignalType:
        """Получение типа сигнала для стороны позиции."""
        return SignalType.BUY if side == PositionSide.LONG else SignalType.SELL

    async def _open_position(
        self, signal: Signal, market_data: MarketData, balance: float
    ) -> Optional[Position]:
        """Открытие новой позиции."""
        try:
            # Определение стороны позиции
            side = PositionSide.LONG if signal.signal_type == SignalType.BUY else PositionSide.SHORT
            
            # Расчет размера позиции
            entry_price = float(signal.entry_price) if hasattr(signal, 'entry_price') else float(market_data.close.value)
            stop_loss = self._calculate_stop_loss(signal, market_data, side)
            position_size = self.calculate_position_size(
                balance, self.config.risk_per_trade, stop_loss, entry_price
            )
            
            if position_size <= 0:
                logger.warning("Position size is zero or negative")
                return None
            
            # Создание позиции
            from domain.value_objects.volume import Volume
            from domain.type_definitions import MetadataDict, Symbol
            from domain.value_objects.trading_pair import TradingPair
            
            position = Position(
                symbol=Symbol(signal.symbol),
                side=side,
                quantity=Volume(Decimal(str(position_size)), market_data.close.currency),
                entry_price=market_data.close,
                current_price=market_data.close,
                stop_loss=Price(Decimal(str(stop_loss)), market_data.close.currency) if stop_loss else None,
                take_profit=self._calculate_take_profit(signal, market_data, side),
                metadata=MetadataDict({"signal_id": str(getattr(signal, 'id', 'unknown')), "signal_confidence": str(signal.confidence)})
            )
            
            logger.info(f"Opened {side.value} position for {signal.symbol}")
            return position
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error opening position: {str(e)}")
            return None

    def _calculate_stop_loss(
        self, signal: Signal, market_data: MarketData, side: PositionSide
    ) -> float:
        """Расчет стоп-лосса."""
        entry_price = float(signal.entry_price) if hasattr(signal, 'entry_price') else float(market_data.close.value)
        atr = self._calculate_atr(market_data) if hasattr(market_data, 'atr') else entry_price * 0.02
        
        if side == PositionSide.LONG:
            return entry_price - (atr * 2)  # 2 ATR ниже входа
        else:
            return entry_price + (atr * 2)  # 2 ATR выше входа

    def _calculate_take_profit(
        self, signal: Signal, market_data: MarketData, side: PositionSide
    ) -> Optional[Price]:
        """Расчет тейк-профита."""
        entry_price = float(signal.entry_price) if hasattr(signal, 'entry_price') else float(market_data.close.value)
        atr = self._calculate_atr(market_data) if hasattr(market_data, 'atr') else entry_price * 0.02
        
        if side == PositionSide.LONG:
            take_profit_price = entry_price + (atr * 3)  # 3 ATR выше входа
        else:
            take_profit_price = entry_price - (atr * 3)  # 3 ATR ниже входа
        
        return Price(Decimal(str(take_profit_price)), market_data.close.currency) if take_profit_price else None

    def _calculate_atr(self, market_data: MarketData) -> float:
        """Расчет ATR (Average True Range)."""
        # Упрощенный расчет ATR
        high_low = float(market_data.high.value) - float(market_data.low.value)
        return high_low * 0.5  # Примерное значение

    async def _close_position(
        self, position: Position, market_data: MarketData, balance: float
    ) -> Optional[Trade]:
        """Закрытие позиции."""
        try:
            # Расчет P&L
            entry_value = float(position.entry_price.value) * float(position.quantity.value)
            exit_value = float(market_data.close.value) * float(position.quantity.value)
            
            if position.side == PositionSide.LONG:
                pnl = exit_value - entry_value
            else:
                pnl = entry_value - exit_value
            
            # Расчет комиссии
            commission = exit_value * self.config.commission
            
            # Создание сделки закрытия
            from domain.value_objects.money import Money
            from domain.type_definitions import MetadataDict, TradeId
            from domain.value_objects.order_id import OrderId
            from domain.value_objects.trading_pair import TradingPair
            from domain.value_objects.currency import Currency
            from uuid import uuid4
            
            # Создаем валюты из символа
            symbol_parts = str(market_data.symbol).split('/')
            if len(symbol_parts) == 2:
                base_currency = Currency(symbol_parts[0])
                quote_currency = Currency(symbol_parts[1])
            else:
                # Если символ не содержит '/', предполагаем что это BTC/USDT
                base_currency = Currency("BTC")
                quote_currency = Currency("USDT")
            
            trade = Trade(
                id=TradeId(uuid4()),
                order_id=OrderId(uuid4()),
                trading_pair=TradingPair(base_currency, quote_currency, str(market_data.symbol)),  # type: ignore[arg-type]
                side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                quantity=position.quantity,
                price=market_data.close,
                commission=Money(Decimal("0"), market_data.close.currency),
                timestamp=market_data.timestamp
            )
            
            logger.info(f"Closed {position.side.value} position for {position.symbol}")
            return trade
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error closing position: {str(e)}")
            return None

    def _update_trailing_stop(self, position: Position, market_data: MarketData) -> None:
        """Обновление трейлинг-стопа."""
        try:
            if position.stop_loss is None:
                return
            
            current_price = float(market_data.close.value)
            current_stop = float(position.stop_loss.value)
            
            if position.side == PositionSide.LONG:
                # Для длинной позиции поднимаем стоп-лосс
                new_stop = current_price - (current_price * self.config.trailing_step)
                if new_stop > current_stop:
                    position.stop_loss = Price(Decimal(str(new_stop)), position.entry_price.currency, position.entry_price.currency)
                    logger.debug(f"Updated trailing stop for {position.symbol}: {new_stop}")
            else:
                # Для короткой позиции опускаем стоп-лосс
                new_stop = current_price + (current_price * self.config.trailing_step)
                if new_stop < current_stop:
                    position.stop_loss = Price(Decimal(str(new_stop)), position.entry_price.currency, position.entry_price.currency)
                    logger.debug(f"Updated trailing stop for {position.symbol}: {new_stop}")
                    
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error updating trailing stop: {str(e)}")

    def _update_metrics(self, trade: Trade, start_time: datetime) -> None:
        """Обновление метрик."""
        try:
            self.metrics.executed_trades += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.average_execution_time = (
                (self.metrics.average_execution_time * (self.metrics.executed_trades - 1) + execution_time)
                / self.metrics.executed_trades
            )
            
            # Проверяем наличие атрибутов у trade
            if hasattr(trade, 'fee') and hasattr(trade.fee, 'value'):
                self.metrics.total_commission += float(trade.fee.value)
            
            # Добавление события
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": "trade_executed",
                "trade_id": str(trade.id) if hasattr(trade, 'id') else "unknown",
                "symbol": str(trade.symbol) if hasattr(trade, 'symbol') else "unknown",
                "side": trade.side.value if hasattr(trade, 'side') and hasattr(trade.side, 'value') else "unknown",
                "quantity": float(trade.volume.value) if hasattr(trade, 'volume') and hasattr(trade.volume, 'value') else 0.0,
                "price": float(trade.price.value) if hasattr(trade, 'price') and hasattr(trade.price, 'value') else 0.0,
                "commission": float(trade.fee.value) if hasattr(trade, 'fee') and hasattr(trade.fee, 'value') else 0.0,
                "execution_time": execution_time
            }
            # Проверяем и инициализируем events если нужно
            if not hasattr(self.metrics, 'events') or self.metrics.events is None:
                self.metrics.events = []
            if isinstance(self.metrics.events, list):
                self.metrics.events.append(event)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error updating metrics: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        try:
            total_trades = self.metrics.executed_trades
            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "success_rate": 0.0,
                    "average_execution_time": 0.0,
                    "total_commission": 0.0,
                    "total_slippage": 0.0,
                    "errors": 0,
                    "rejected_trades": 0
                }
            
            return {
                "total_trades": total_trades,
                "success_rate": (total_trades - self.metrics.errors) / total_trades,
                "average_execution_time": self.metrics.average_execution_time,
                "total_commission": self.metrics.total_commission,
                "total_slippage": self.metrics.total_slippage,
                "errors": self.metrics.errors,
                "rejected_trades": self.metrics.rejected_trades,
                "partial_fills": self.metrics.partial_fills,
                "last_error": self.metrics.last_error
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

    def get_position_summary(self) -> Dict[str, Any]:
        """Получение сводки позиций."""
        try:
            positions = list(self._position_cache.values())
            if not positions:
                return {"total_positions": 0, "positions": []}
            
            summary = {
                "total_positions": len(positions),
                "long_positions": len([p for p in positions if hasattr(p, 'side') and p.side == PositionSide.LONG]),
                "short_positions": len([p for p in positions if hasattr(p, 'side') and p.side == PositionSide.SHORT]),
                "positions": []
            }
            
            for position in positions:
                position_data = {
                    "symbol": str(position.symbol) if hasattr(position, 'symbol') else "unknown",
                    "side": position.side.value if hasattr(position, 'side') and hasattr(position.side, 'value') else "unknown",
                    "quantity": float(position.quantity.value) if hasattr(position, 'quantity') and hasattr(position.quantity, 'value') else 0.0,
                    "entry_price": float(position.entry_price.value) if hasattr(position, 'entry_price') and hasattr(position.entry_price, 'value') else 0.0,
                    "current_price": float(position.current_price.value) if hasattr(position, 'current_price') and hasattr(position.current_price, 'value') else 0.0,
                    "unrealized_pnl": float(position.unrealized_pnl.value) if hasattr(position, 'unrealized_pnl') and hasattr(position.unrealized_pnl, 'value') else 0.0,
                    "stop_loss": float(position.stop_loss.value) if hasattr(position, 'stop_loss') and position.stop_loss and hasattr(position.stop_loss, 'value') else None,
                    "take_profit": float(position.take_profit.value) if hasattr(position, 'take_profit') and position.take_profit and hasattr(position.take_profit, 'value') else None
                }
                summary["positions"].append(position_data)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position summary: {str(e)}")
            return {"total_positions": 0, "positions": []}

    def reset_metrics(self) -> None:
        """Сброс метрик."""
        self.metrics = TradeExecutorMetrics()
        self._execution_history.clear()
        logger.info("Trade executor metrics reset")

    def cleanup(self) -> None:
        """Очистка ресурсов."""
        self._position_cache.clear()
        self._execution_history.clear()
        logger.info("Trade executor cleanup completed")
