import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.controllers.market_controller import MarketController
from core.controllers.order_controller import OrderController
from core.controllers.position_controller import PositionController
from core.controllers.risk_controller import RiskController
from core.logger import Logger

from ..models import MarketData, Order, Position
from ..signal_processor import SignalProcessor
from ..strategy import Strategy
from ..types import TradingMode, TradingPair
from .base import BaseController

logger = Logger()


class TradingController(BaseController):
    """Основной контроллер для торговли"""

    def __init__(self, exchange, config: Dict[str, Any]):
        super().__init__()
        self.exchange = exchange
        self.config = config
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.mode = TradingMode.PAUSED
        self.trading_pairs: Dict[str, TradingPair] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.market_data: dict[str, MarketData] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: dict[str, List[Order]] = {}

        # Инициализация контроллеров
        self.order_controller = OrderController(exchange, config)
        self.position_controller = PositionController(exchange, self.order_controller)
        self.market_controller = MarketController(exchange)
        self.risk_controller = RiskController(config)
        self.signal_processor = SignalProcessor(config)
        self.timeframe = config.get("timeframe", "1h")
        self.strategies: List[Strategy] = []
        self.symbol = config.get("symbol", "BTCUSDT")

        # Интервалы обновления
        self.market_update_interval: int = int(config.get("market_update_interval", 60))
        self.position_update_interval: int = int(
            config.get("position_update_interval", 60)
        )
        self.order_update_interval: int = int(config.get("order_update_interval", 60))

        # Мониторинг
        self.market_monitor_task: Optional[asyncio.Task] = None
        self.position_monitor_task: Optional[asyncio.Task] = None
        self.order_monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Запуск контроллера"""
        try:
            # Запуск базового контроллера
            await super().start()

            # Загрузка конфигурации
            await self._load_config()

            # Инициализация торговых пар
            await self._init_trading_pairs()

            # Запуск мониторинга
            await self._start_monitoring()

            # Инициализация стратегий
            for strategy_config in self.config.get("strategies", []):
                strategy = self._create_strategy(strategy_config)
                if strategy:
                    self.strategies.append(strategy)

            logger.info("Trading controller started")

            # Запуск обработки сигналов
            while True:
                await self._process_signals()
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error starting trading controller: {str(e)}")
            raise

    async def stop(self) -> None:
        """Остановка контроллера"""
        try:
            # Остановка мониторинга
            await self._stop_monitoring()

            # Закрытие всех позиций
            await self.close_all_positions()

            # Отмена всех ордеров
            await self.cancel_all_orders()

            # Остановка базового контроллера
            await super().stop()

            logger.info("Trading controller stopped")

        except Exception as e:
            logger.error(f"Error stopping trading controller: {str(e)}")
            raise

    async def _load_config(self) -> None:
        """Загрузка конфигурации"""
        try:
            if not self._validate_config():
                raise ValueError("Invalid configuration")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    async def _init_trading_pairs(self) -> None:
        """Инициализация торговых пар"""
        try:
            trading_pairs = self.config.get("trading_pairs", [])
            if hasattr(self.exchange, "fetch_markets"):
                markets = await self.exchange.fetch_markets()
                for market in markets:
                    if market["active"] and market["symbol"] in trading_pairs:
                        symbol = str(market["symbol"])
                        self.trading_pairs[symbol] = TradingPair(
                            base=str(market["base"]),
                            quote=str(market["quote"]),
                            symbol=symbol,
                            active=True,
                        )
            else:
                for pair in trading_pairs:
                    base, quote = pair.split("/")
                    symbol = str(pair)
                    self.trading_pairs[symbol] = TradingPair(
                        base=base, quote=quote, symbol=symbol, active=True
                    )
        except Exception as e:
            logger.error(f"Error initializing trading pairs: {str(e)}")
            raise

    async def _start_monitoring(self) -> None:
        """Запуск мониторинга"""
        try:
            # Запуск мониторинга рынка
            self.monitoring_tasks.append(asyncio.create_task(self._monitor_market()))

            # Запуск мониторинга позиций
            self.monitoring_tasks.append(asyncio.create_task(self._monitor_positions()))

            # Запуск мониторинга ордеров
            self.monitoring_tasks.append(asyncio.create_task(self._monitor_orders()))

        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            raise

    async def _stop_monitoring(self) -> None:
        """Остановка мониторинга"""
        try:
            for task in self.monitoring_tasks:
                task.cancel()
            self.monitoring_tasks.clear()

        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            raise

    async def _monitor_market(self) -> None:
        """Мониторинг рынка"""
        try:
            while self.state.is_running:
                try:
                    for pair in self.trading_pairs:
                        data = await self.exchange.get_market_data(pair, "1m")
                        if data is not None:
                            self.market_data = data
                    await asyncio.sleep(self.market_update_interval)
                except Exception as e:
                    logger.error(f"Error monitoring market data: {str(e)}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def _monitor_positions(self) -> None:
        """Мониторинг позиций."""
        while True:
            try:
                for symbol, pair in self.trading_pairs.items():
                    position = self.position_controller.get_position(symbol)
                    if position:
                        self.positions[symbol] = position
            except Exception as e:
                logger.error(f"Ошибка при мониторинге позиций: {e}")
            await asyncio.sleep(self.config.get("position_update_interval", 60))

    async def _monitor_orders(self) -> None:
        """Мониторинг ордеров."""
        while True:
            try:
                for symbol, pair in self.trading_pairs.items():
                    orders = await self.order_controller.get_open_orders()
                    if orders:
                        self.orders[symbol] = orders
            except Exception as e:
                logger.error(f"Ошибка при мониторинге ордеров: {e}")
            await asyncio.sleep(self.config.get("order_update_interval", 60))

    def _validate_config(self) -> bool:
        """
        Валидация конфигурации.

        Returns:
            bool: True если конфигурация валидна
        """
        try:
            # Проверяем обязательные поля
            required_fields = [
                "trading_pairs",
                "market_update_interval",
                "position_update_interval",
                "order_update_interval",
            ]

            for field in required_fields:
                if field not in self.config:
                    logger.error(f"Missing required field in config: {field}")
                    return False

            # Проверяем типы данных
            if not isinstance(self.config["trading_pairs"], list):
                logger.error("trading_pairs must be a list")
                return False

            if not all(
                isinstance(interval, (int, float))
                for interval in [
                    self.config["market_update_interval"],
                    self.config["position_update_interval"],
                    self.config["order_update_interval"],
                ]
            ):
                logger.error("Update intervals must be numbers")
                return False

            # Проверяем значения
            if not self.config["trading_pairs"]:
                logger.error("trading_pairs list cannot be empty")
                return False

            if any(
                interval <= 0
                for interval in [
                    self.config["market_update_interval"],
                    self.config["position_update_interval"],
                    self.config["order_update_interval"],
                ]
            ):
                logger.error("Update intervals must be positive")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating config: {str(e)}")
            return False

    async def close_all_positions(self) -> None:
        """Закрытие всех позиций"""
        try:
            positions = self.position_controller.get_all_positions()
            for position in positions:
                result = self.position_controller.close_position(position.pair)
                if asyncio.iscoroutine(result):
                    result = await result
                if not result or getattr(result, "status", None) != "closed":
                    logger.error(f"Failed to close position: {position.pair}")
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")
            raise

    async def cancel_all_orders(self) -> None:
        """Отмена всех ордеров"""
        try:
            for order_id in list(self.order_controller.active_orders.keys()):
                await self.order_controller.cancel_order(order_id)
                if order_id in self.order_controller.active_orders:
                    del self.order_controller.active_orders[order_id]

            logger.info("All orders canceled")

        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            raise

    async def set_mode(self, mode: TradingMode) -> None:
        """Установка режима работы"""
        self.mode = mode
        if mode == TradingMode.TRADING:
            await self._start_trading_mode()
        elif mode == TradingMode.PAUSED:
            await self._stop_trading_mode()

    async def _start_trading_mode(self) -> None:
        """Запуск торгового режима"""
        self.is_running = True
        await self._start_monitoring()

    async def _stop_trading_mode(self) -> None:
        """Остановка торгового режима"""
        self.is_running = False
        await self._stop_monitoring()

    def _load_trading_pairs(self) -> None:
        """Загрузка торговых пар."""
        try:
            for pair in self.config["trading_pairs"]:
                base, quote = pair.split("/")
                trading_pair = TradingPair(
                    base=base, quote=quote, symbol=pair, active=True
                )
                self.trading_pairs[pair] = trading_pair
        except Exception as e:
            logger.error(f"Error loading trading pairs: {e}")

    def _collect_signals(self) -> List[Dict[str, Any]]:
        """Сбор сигналов от всех стратегий."""
        signals = []
        for pair in self.trading_pairs.values():
            market_data = self.exchange.get_market_data(pair.symbol, self.timeframe)
            if market_data:
                for strategy in self.strategies:
                    signals.extend(strategy.analyze(market_data))
        return signals

    def _analyze_market_data(self, market_data: Dict) -> Optional[Dict]:
        """Анализ рыночных данных"""
        # Здесь должна быть логика анализа
        return {"action": "buy", "confidence": 0.8}

    def _calculate_levels(
        self, symbol: str, direction: str, entry_price: float, confidence: float
    ) -> tuple:
        """Расчет уровней входа и выхода"""
        if direction == "long":
            stop_loss = entry_price * (1 - 0.02 * confidence)
            take_profit = entry_price * (1 + 0.04 * confidence)
        else:
            stop_loss = entry_price * (1 + 0.02 * confidence)
            take_profit = entry_price * (1 - 0.04 * confidence)
        return stop_loss, take_profit

    async def _process_signals(self) -> None:
        """Обработка сигналов."""
        try:
            # Получение рыночных данных
            market_data = self.market_controller.get_market_data(self.symbol)

            if market_data is None:
                logger.warning(f"No market data available for {self.symbol}")
                return

            # Анализ данных стратегиями
            signals = []
            for strategy in self.strategies:
                if strategy is None:
                    continue
                df = pd.DataFrame([m.__dict__ for m in market_data])
                strategy_signals = strategy.analyze(data=df)
                if strategy_signals:
                    signals.extend(strategy_signals)

            if not signals:
                logger.debug("No signals generated")
                return

            # Обработка сигналов
            processed_signals = await self.signal_processor.process_signals(signals)

            # Применение сигналов
            for signal in processed_signals:
                if signal is None:
                    continue

                # Проверка рисков
                if not self.risk_controller.validate_signal(signal):
                    logger.warning(f"Signal rejected by risk controller: {signal}")
                    continue

                # Принятие решения
                decision = await self.decide_action(
                    symbol=self.symbol, signal=signal, market_context=market_data
                )

                if decision["action"] == "open":
                    # Открытие позиции
                    position = Position(
                        pair=self.symbol,
                        side=decision["direction"],
                        size=decision["volume"],
                        entry_price=market_data[0].close,
                        current_price=market_data[0].close,
                        pnl=0.0,
                        entry_time=datetime.now(),
                        stop_loss=decision["stop_loss"],
                        take_profit=decision["take_profit"],
                    )
                    result = await self.position_controller.open_position(position)
                    if result:
                        logger.info(f"Position opened: {self.symbol}")
                elif decision["action"] == "close":
                    # Закрытие позиции
                    result = self.position_controller.close_position(self.symbol)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result:
                        logger.info(f"Position closed: {self.symbol}")

        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")

    async def decide_action(
        self, symbol: str, signal: Any, market_context: List[MarketData]
    ) -> Dict[str, Any]:
        """
        Принятие решения на основе сигнала и контекста рынка.

        Args:
            symbol: Торговая пара
            signal: Сигнал
            market_context: Контекст рынка

        Returns:
            Dict[str, Any]: Решение
        """
        try:
            # Получение текущих позиций
            positions = await self.position_controller.get_positions(symbol)

            # Анализ сигнала
            if signal.type == "buy" and not positions:
                # Открытие длинной позиции
                volume = self.risk_controller.calculate_position_size(
                    symbol,
                    market_context[0].close,
                    self.config.get("risk_per_trade", 0.02),
                )
                stop_loss = self.risk_controller.calculate_stop_loss(
                    Position(
                        pair=symbol,
                        side="long",
                        size=volume,
                        entry_price=market_context[0].close,
                        current_price=market_context[0].close,
                        pnl=0.0,
                        entry_time=datetime.now(),
                    ),
                    market_context[0].close * 0.02,  # ATR как 2% от цены
                )
                take_profit = self.risk_controller.calculate_take_profit(
                    Position(
                        pair=symbol,
                        side="long",
                        size=volume,
                        entry_price=market_context[0].close,
                        current_price=market_context[0].close,
                        pnl=0.0,
                        entry_time=datetime.now(),
                    ),
                    market_context[0].close * 0.02,  # ATR как 2% от цены
                )
                return {
                    "action": "open",
                    "direction": "long",
                    "volume": volume,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            elif signal.type == "sell" and positions:
                # Закрытие позиции
                return {"action": "close"}

            return {"action": "none"}

        except Exception as e:
            logger.error(f"Error deciding action: {str(e)}")
            return {"action": "none"}

    def _create_strategy(self, config: Dict) -> Optional[Strategy]:
        """
        Создание стратегии на основе конфигурации.

        Args:
            config: Конфигурация стратегии

        Returns:
            Optional[Strategy]: Созданная стратегия или None
        """
        try:
            strategy_type = config.get("type")
            if not strategy_type:
                return None

            # Создаем копию конфигурации и добавляем символ
            strategy_config = config.copy()
            strategy_config["symbol"] = self.symbol
            strategy = Strategy(strategy_config)
            return strategy

        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            return None

    async def _process_market_data(self, symbol: str, market_data: MarketData) -> None:
        """Обработка рыночных данных."""
        try:
            self.market_data[symbol] = market_data
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    async def _process_positions(self, positions: List[Position]) -> None:
        """Обработка позиций."""
        try:
            for position in positions:
                self.positions[position.pair] = position
        except Exception as e:
            logger.error(f"Error processing positions: {str(e)}")

    async def _process_orders(self, orders: List[Order]) -> None:
        """Обработка ордеров."""
        try:
            # Группируем ордера по парам
            orders_by_pair: Dict[str, List[Order]] = {}
            for order in orders:
                if order.pair not in orders_by_pair:
                    orders_by_pair[order.pair] = []
                orders_by_pair[order.pair].append(order)

            # Обновляем словарь ордеров
            for pair, pair_orders in orders_by_pair.items():
                self.orders[pair] = pair_orders
        except Exception as e:
            logger.error(f"Error processing orders: {str(e)}")

    async def _update_market_data(self) -> None:
        """Обновление рыночных данных."""
        while True:
            try:
                for symbol, pair in self.trading_pairs.items():
                    market_data = await self.exchange.get_market_data(symbol)
                    if market_data:
                        self.market_data[symbol] = market_data
            except Exception as e:
                logger.error(f"Ошибка при обновлении рыночных данных: {e}")
            await asyncio.sleep(self.config.get("market_update_interval", 60))
