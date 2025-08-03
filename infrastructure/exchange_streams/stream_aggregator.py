# -*- coding: utf-8 -*-
"""Market stream aggregator for combining data from multiple exchanges."""
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from loguru import logger

from shared.models.orderbook import OrderBookUpdate


@dataclass
class StreamSource:
    """Источник потока данных."""

    name: str
    client: Any
    is_active: bool = True
    last_update: float = 0.0
    update_count: int = 0
    error_count: int = 0


class MarketStreamAggregator:
    """Агрегатор потоков данных с разных бирж."""

    def __init__(self) -> None:
        self.sources: Dict[str, StreamSource] = {}
        self.symbols: Set[str] = set()
        self.callbacks: List[Callable[[OrderBookUpdate], Union[None, Any]]] = []
        self.update_buffer: List[OrderBookUpdate] = []
        self.buffer_size: int = 1000
        self.sync_tolerance_ms: float = 100.0  # 100ms tolerance
        self.is_running: bool = False
        self.stats: Dict[str, Any] = {
            "total_updates": 0,
            "last_update_time": 0.0,
            "exchanges": {},
        }

    def add_source(self, name: str, client: Any) -> bool:
        """Добавление источника данных."""
        if name in self.sources:
            logger.warning(f"Source {name} already exists")
            return False
        self.sources[name] = StreamSource(name=name, client=client)
        logger.info(f"Added source: {name}")
        return True

    def remove_source(self, name: str) -> bool:
        """Удаление источника данных."""
        try:
            if name in self.sources:
                source = self.sources[name]
                if hasattr(source.client, "disconnect"):
                    asyncio.create_task(source.client.disconnect())
                del self.sources[name]
                logger.info(f"Removed stream source: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove source {name}: {e}")
            return False

    def add_callback(self, callback: Callable[[OrderBookUpdate], Union[None, Any]]) -> None:
        """Добавление callback для обработки обновлений."""
        self.callbacks.append(callback)
        logger.info(f"Added callback, total callbacks: {len(self.callbacks)}")

    def remove_callback(self, callback: Callable[[OrderBookUpdate], Union[None, Any]]) -> None:
        """Удаление callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Removed callback, total callbacks: {len(self.callbacks)}")

    async def subscribe_symbol(self, symbol: str) -> bool:
        """Подписка на символ во всех активных источниках."""
        try:
            self.symbols.add(symbol)
            success_count = 0
            for source_name, source in self.sources.items():
                if not source.is_active:
                    continue
                try:
                    if hasattr(source.client, "subscribe"):
                        success = await source.client.subscribe(symbol)
                        if success:
                            success_count += 1
                            logger.info(f"Subscribed to {symbol} on {source_name}")
                        else:
                            logger.warning(
                                f"Failed to subscribe to {symbol} on {source_name}"
                            )
                    else:
                        logger.warning(
                            f"Client {source_name} doesn't support subscription"
                        )
                except Exception as e:
                    logger.error(f"Error subscribing to {symbol} on {source_name}: {e}")
                    source.error_count += 1
            logger.info(
                f"Subscribed to {symbol} on {success_count}/{len(self.sources)} sources"
            )
            return success_count > 0
        except Exception as e:
            logger.error(f"Failed to subscribe to symbol {symbol}: {e}")
            return False

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Отписка от символа во всех источниках."""
        try:
            self.symbols.discard(symbol)
            success_count = 0
            for source_name, source in self.sources.items():
                if not source.is_active:
                    continue
                try:
                    if hasattr(source.client, "unsubscribe"):
                        success = await source.client.unsubscribe(symbol)
                        if success:
                            success_count += 1
                            logger.info(f"Unsubscribed from {symbol} on {source_name}")
                        else:
                            logger.warning(
                                f"Failed to unsubscribe from {symbol} on {source_name}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error unsubscribing from {symbol} on {source_name}: {e}"
                    )
                    source.error_count += 1
            logger.info(
                f"Unsubscribed from {symbol} on {success_count}/{len(self.sources)} sources"
            )
            return success_count > 0
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbol {symbol}: {e}")
            return False

    async def start(self) -> None:
        """Запуск агрегатора."""
        if self.is_running:
            logger.warning("Aggregator is already running")
            return
        self.is_running = True
        logger.info("Starting market stream aggregator")
        # Запускаем прослушивание для всех источников
        tasks = []
        for source_name, source in self.sources.items():
            if source.is_active and hasattr(source.client, "listen"):
                task = asyncio.create_task(self._listen_source(source_name, source))
                tasks.append(task)
        # Ждем завершения всех задач
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No active sources to listen to")

    async def stop(self) -> None:
        """Остановка агрегатора."""
        self.is_running = False
        logger.info("Stopping market stream aggregator")
        # Отключаем все источники
        for source_name, source in self.sources.items():
            try:
                if hasattr(source.client, "disconnect"):
                    await source.client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {source_name}: {e}")

    async def _listen_source(self, source_name: str, source: StreamSource) -> None:
        """Прослушивание конкретного источника."""
        try:
            # Создаем callback для обработки обновлений от этого источника
            async def source_callback(update: OrderBookUpdate) -> None:
                await self._handle_update(source_name, source, update)

            # Запускаем прослушивание
            await source.client.listen(source_callback)
        except Exception as e:
            logger.error(f"Error listening to source {source_name}: {e}")
            source.error_count += 1
            source.is_active = False

    async def _handle_update(
        self, source_name: str, source: StreamSource, update: OrderBookUpdate
    ) -> None:
        """Обработка обновления от источника."""
        try:
            # Обновляем статистику источника
            source.last_update = time.time()
            source.update_count += 1
            # Добавляем в буфер
            self.update_buffer.append(update)
            # Ограничиваем размер буфера
            if len(self.update_buffer) > self.buffer_size:
                self.update_buffer.pop(0)
            # Обновляем общую статистику
            if isinstance(self.stats["total_updates"], int):
                self.stats["total_updates"] = self.stats["total_updates"] + 1
            else:
                self.stats["total_updates"] = 1
            self.stats["last_update_time"] = time.time()
            # Обновляем статистику по биржам
            exchanges_data = self.stats.get("exchanges", {})
            if not isinstance(exchanges_data, dict):
                exchanges_data = {}
                self.stats["exchanges"] = exchanges_data
            if update.exchange not in exchanges_data:
                exchanges_data[update.exchange] = {
                    "update_count": 0,
                    "last_update": 0.0,
                }
            if update.exchange in exchanges_data:
                exchange_data = exchanges_data[update.exchange]
                if isinstance(exchange_data, dict):
                    exchange_data["update_count"] = exchange_data.get("update_count", 0) + 1
                    exchange_data["last_update"] = time.time()
            # Вызываем все callbacks
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
        except Exception as e:
            logger.error(f"Error handling update from {source_name}: {e}")
            source.error_count += 1

    def get_synchronized_updates(
        self, tolerance_ms: Optional[float] = None
    ) -> List[OrderBookUpdate]:
        """Получение синхронизированных обновлений."""
        if tolerance_ms is None:
            tolerance_ms = self.sync_tolerance_ms
        if not self.update_buffer:
            return []
        # Группируем обновления по времени
        current_time = time.time()
        synchronized_updates = []
        for update in self.update_buffer:
            update_time = update.timestamp.to_unix()
            time_diff = abs(current_time - update_time) * 1000  # в миллисекундах
            if time_diff <= tolerance_ms:
                synchronized_updates.append(update)
        return synchronized_updates

    def get_source_status(self) -> Dict[str, Any]:
        """Получение статуса всех источников."""
        status = {}
        for source_name, source in self.sources.items():
            status[source_name] = {
                "is_active": source.is_active,
                "last_update": source.last_update,
                "update_count": source.update_count,
                "error_count": source.error_count,
                "uptime": (
                    time.time() - source.last_update if source.last_update > 0 else 0
                ),
            }
            # Добавляем статус клиента если доступен
            if hasattr(source.client, "get_status"):
                try:
                    status[source_name]["client_status"] = source.client.get_status()
                except Exception as e:
                    status[source_name]["client_status"] = {"error": str(e)}  # type: ignore
        return status

    def get_aggregator_stats(self) -> Dict[str, Any]:
        """Получение статистики агрегатора."""
        stats_copy = self.stats.copy()
        last_update_time = self.stats.get("last_update_time", 0.0)
        if isinstance(last_update_time, (int, float)):
            uptime = time.time() - last_update_time
        else:
            uptime = 0.0
        stats_copy.update({
            "uptime": uptime,
            "active_sources": len([s for s in self.sources.values() if s.is_active]),
            "total_sources": len(self.sources),
            "buffer_size": len(self.update_buffer),
            "subscribed_symbols": list(self.symbols),
        })
        return stats_copy

    def clear_buffer(self) -> None:
        """Очистка буфера обновлений."""
        self.update_buffer.clear()
        logger.info("Cleared update buffer")

    def reset_stats(self) -> None:
        """Сброс статистики."""
        self.stats = {"total_updates": 0, "last_update_time": 0.0, "exchanges": {}}
        for source in self.sources.values():
            source.update_count = 0
            source.error_count = 0
        logger.info("Reset aggregator statistics")
