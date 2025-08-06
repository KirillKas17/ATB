# -*- coding: utf-8 -*-
"""Stream manager for entanglement detection with multiple exchanges."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from domain.intelligence.entanglement_detector import EntanglementDetector
from domain.protocols.exchange_protocols import (
    MarketDataConnectorProtocol,
    MarketStreamAggregatorProtocol,
    WebSocketClientProtocol,
)
from shared.models.orderbook import OrderBookUpdate


class StreamManager:
    """Менеджер потоков данных для обнаружения запутанности."""

    def __init__(
        self,
        aggregator: MarketStreamAggregatorProtocol,
        detector: EntanglementDetector,
        detection_interval: float = 0.1,
    ):
        """Инициализация StreamManager."""
        self.aggregator = aggregator
        self.detector = detector
        self.detection_interval = detection_interval
        self.is_running = False
        self.monitored_symbols: set = set()
        self.entanglement_callbacks: List[Callable] = []
        self.debug_mode = False
        self._last_sequence_ids: Dict[str, int] = {}
        self.stats = {
            "total_detections": 0,
            "entangled_detections": 0,
            "exchanges_connected": 0,
            "start_time": time.time(),
        }

    async def initialize_exchanges(
        self, symbols: List[str], exchanges: Dict[str, WebSocketClientProtocol]
    ):
        """Инициализация подключений к биржам."""
        try:
            logger.info("Initializing exchange connections...")

            # Добавляем источники в агрегатор
            for exchange_name, client in exchanges.items():
                success = self.aggregator.add_source(exchange_name, client)
                if success:
                    self.stats["exchanges_connected"] = int(self.stats.get("exchanges_connected", 0)) + 1
                    logger.info(f"Added {exchange_name} to aggregator")
                else:
                    logger.error(f"Failed to add {exchange_name} to aggregator")

            # Подписываемся на символы
            for symbol in symbols:
                await self.subscribe_symbol(symbol)

            # Добавляем callback для обработки обновлений
            self.aggregator.add_callback(self._handle_order_book_update)

            logger.info(f"Initialized {self.stats['exchanges_connected']} exchanges")

        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")

    async def subscribe_symbol(self, symbol: str) -> bool:
        """Подписка на символ во всех биржах."""
        try:
            success = await self.aggregator.subscribe_symbol(symbol)
            if success:
                self.monitored_symbols.add(symbol)
                logger.info(f"Subscribed to {symbol} on all exchanges")
            return success
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Отписка от символа."""
        try:
            success = await self.aggregator.unsubscribe_symbol(symbol)
            if success:
                self.monitored_symbols.discard(symbol)
                logger.info(f"Unsubscribed from {symbol}")
            return success
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {symbol}: {e}")
            return False

    async def start_monitoring(self) -> Any:
        """Запуск мониторинга запутанности."""
        if self.is_running:
            logger.warning("Stream manager is already running")
            return

        self.is_running = True
        logger.info("Starting entanglement monitoring")

        # Запускаем агрегатор
        aggregator_task = asyncio.create_task(self.aggregator.start())

        # Запускаем детектор запутанности
        detector_task = asyncio.create_task(self._run_entanglement_detector())

        try:
            await asyncio.gather(aggregator_task, detector_task, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in stream manager: {e}")
        finally:
            self.is_running = False

    async def stop_monitoring(self) -> Any:
        """Остановка мониторинга."""
        self.is_running = False
        await self.aggregator.stop()
        logger.info("Stopped entanglement monitoring")

    async def _handle_order_book_update(self, *args, **kwargs) -> Any:
        """Обработка обновления ордербука."""
        try:
            # Валидация обновления
            if not update or not update.symbol:
                logger.warning("Invalid order book update received")
                return

            # Проверка последовательности
            if hasattr(update, "sequence_id") and update.sequence_id:
                self._validate_sequence(update.symbol, update.sequence_id)

            # Логирование для отладки
            if self.debug_mode:
                bids_count = len(update.bids) if hasattr(update.bids, '__len__') and not callable(update.bids) else 0
                asks_count = len(update.asks) if hasattr(update.asks, '__len__') and not callable(update.asks) else 0
                logger.debug(
                    f"Processing order book update for {update.symbol}: {bids_count} bids, {asks_count} asks"
                )

            # Дополнительная обработка может быть добавлена здесь
            # Например, фильтрация, нормализация, агрегация и т.д.

        except Exception as e:
            logger.error(f"Error handling order book update: {e}")

    def _validate_sequence(self, symbol: str, sequence_id: int) -> None:
        """Валидация последовательности обновлений."""
        if symbol not in self._last_sequence_ids:
            self._last_sequence_ids[symbol] = sequence_id
            return

        expected_sequence = self._last_sequence_ids[symbol] + 1
        if sequence_id != expected_sequence:
            logger.warning(
                f"Sequence gap detected for {symbol}: expected {expected_sequence}, got {sequence_id}"
            )

        self._last_sequence_ids[symbol] = sequence_id

    async def _run_entanglement_detector(self) -> Any:
        """Запуск детектора запутанности."""
        while self.is_running:
            try:
                # Получаем синхронизированные обновления
                synchronized_updates = self.aggregator.get_synchronized_updates()

                if synchronized_updates:
                    # Обрабатываем обновления через детектор
                    await self._process_order_book_updates(synchronized_updates)

                await asyncio.sleep(self.detection_interval)

            except Exception as e:
                logger.error(f"Error in entanglement detector: {e}")
                await asyncio.sleep(1.0)

    async def _process_order_book_updates(self, *args, **kwargs) -> Any:
        """Обработка списка обновлений ордербука."""
        if not updates:
            return

        # Группируем обновления по символам
        symbol_updates: Dict[str, Dict[str, OrderBookUpdate]] = {}
        for update in updates:
            if update.symbol not in symbol_updates:
                symbol_updates[update.symbol] = {}
            symbol_updates[update.symbol][update.exchange] = update

        # Анализируем каждый символ
        for symbol, exchange_data in symbol_updates.items():
            if symbol not in self.monitored_symbols:
                continue

            try:
                # Преобразуем данные для детектора
                detector_data = {}
                for exchange, update in exchange_data.items():
                    detector_data[exchange] = {
                        "symbol": update.symbol,
                        "bids": update.bids,
                        "asks": update.asks,
                        "timestamp": update.timestamp.to_iso(),
                        "metadata": {"sequence_id": update.sequence_id},
                    }

                # Запускаем детекцию
                result = self.detector.detect_entanglement(symbol, detector_data)

                # Обновляем статистику
                self.stats["total_detections"] += 1
                if result.is_entangled:
                    self.stats["entangled_detections"] += 1

                # Вызываем callbacks
                await self._handle_entanglement_result(result)

            except Exception as e:
                logger.error(f"Error processing updates for {symbol}: {e}")

    async def _handle_entanglement_result(self, *args, **kwargs) -> Any:
        """Обработка результата обнаружения запутанности."""
        try:
            # Обновляем статистику
            self.stats["total_detections"] = int(self.stats.get("total_detections", 0)) + 1
            
            if result and hasattr(result, "is_entangled") and result.is_entangled:
                self.stats["entangled_detections"] = int(self.stats.get("entangled_detections", 0)) + 1
                logger.info(f"Entanglement detected: {result}")

            # Вызываем callbacks
            for callback in self.entanglement_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.error(f"Error in entanglement callback: {e}")

        except Exception as e:
            logger.error(f"Error handling entanglement result: {e}")

    def add_entanglement_callback(self, *args, **kwargs) -> Any:
        """Добавление callback для обработки результатов запутанности."""
        self.entanglement_callbacks.append(callback)
        callback_count = len(self.entanglement_callbacks) if hasattr(self.entanglement_callbacks, '__len__') else 0
        logger.info(
            f"Added entanglement callback, total: {callback_count}"
        )

    def remove_entanglement_callback(self, *args, **kwargs) -> Any:
        """Удаление callback."""
        if callback in self.entanglement_callbacks:
            self.entanglement_callbacks.remove(callback)
            callback_count = len(self.entanglement_callbacks) if hasattr(self.entanglement_callbacks, '__len__') else 0
            logger.info(
                f"Removed entanglement callback, total: {callback_count}"
            )

    def get_status(self) -> Dict:
        """Получение статуса системы."""
        return {
            "is_running": self.is_running,
            "monitored_symbols": list(self.monitored_symbols),
            "aggregator_stats": self.aggregator.get_aggregator_stats(),
            "source_status": self.aggregator.get_source_status(),
            "detector_stats": self.detector.get_detector_statistics(),
            "entanglement_stats": self.stats,
            "uptime": time.time() - self.stats["start_time"],
        }

    def get_entanglement_stats(self) -> Dict:
        """Получение статистики обнаружения запутанности."""
        return {
            **self.stats,
            "detection_rate": (
                self.stats["total_detections"]
                / max(time.time() - self.stats["start_time"], 1.0)
            ),
            "entanglement_rate": (
                self.stats["entangled_detections"]
                / max(self.stats["total_detections"], 1)
            ),
        }

    def reset_stats(self) -> None:
        """Сброс статистики."""
        self.stats = {
            "total_detections": 0,
            "entangled_detections": 0,
            "last_detection_time": 0.0,
            "start_time": time.time(),
            "exchanges_connected": self.stats["exchanges_connected"],
        }
        self.detector.reset_statistics()
        self.aggregator.reset_stats()
        logger.info("Reset stream manager statistics")

    def get_buffer_status(self) -> Dict[str, Any]:
        """Получение статуса буферов."""
        return {
            "aggregator_buffer_size": len(self.aggregator.update_buffer) if hasattr(self.aggregator.update_buffer, '__len__') else 0,
            "monitored_symbols": list(self.monitored_symbols),
            "detector_stats": self.detector.get_detector_statistics(),
        }

    def clear_buffers(self) -> None:
        """Очистка буферов."""
        self.aggregator.clear_buffer()
        self.detector.reset_statistics()
        logger.info("Cleared all buffers")
