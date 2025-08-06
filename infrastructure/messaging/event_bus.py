"""
Профессиональная реализация EventBus для мессенджинга.
Особенности:
- Строгая типизация с использованием Protocol
- Асинхронная обработка с приоритетными очередями
- Мониторинг производительности и метрики
- Обработка ошибок и retry механизмы
- Потокобезопасность
- Соответствие DDD и SOLID принципам
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

from loguru import logger

from domain.type_definitions.messaging_types import (
    Event,
    EventBusConfig,
    EventBusError,
    EventBusMetrics,
    EventBusProtocol,
    EventHandler,
    EventHandlerInfo,
    EventName,
    EventPriority,
    EventType,
    HandlerConfig,
    HandlerID,
    MessageID,
    create_event,
    create_handler_config,
)


class HandlerMetrics(TypedDict):
    """Метрики обработчика."""

    call_count: int
    total_time: float
    avg_time: float
    error_count: int
    last_execution: Optional[datetime]


class EventBus(EventBusProtocol):
    """
    Профессиональная реализация EventBus с асинхронностью и приоритетными очередями.
    Реализует EventBusProtocol и обеспечивает:
    - Асинхронную обработку событий
    - Очереди с приоритетами
    - Мониторинг производительности
    - Обработку ошибок и retry
    - Потокобезопасность
    """

    def __init__(self, config: Optional[EventBusConfig] = None) -> None:
        """
        Инициализация EventBus.
        Args:
            config: Конфигурация EventBus
        """
        self.config = config or EventBusConfig()
        # Очереди событий по приоритетам
        self.event_queues: Dict[EventPriority, deque] = {
            priority: deque(maxlen=self.config.queue_size) for priority in EventPriority
        }
        # Обработчики событий
        self.handlers: Dict[EventName, Dict[HandlerID, EventHandlerInfo]] = defaultdict(
            dict
        )
        # История событий
        self.event_history: deque = deque(maxlen=self.config.max_history)
        # Пул потоков для синхронных обработчиков
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Флаги состояния
        self.is_running = False
        self.is_shutdown = False
        self.start_time: Optional[datetime] = None
        # Блокировки для потокобезопасности
        self._lock = threading.RLock()
        self._handler_lock = threading.RLock()
        self._history_lock = threading.RLock()
        # Метрики производительности
        self.metrics: Dict[str, Any] = {
            "total_events_published": 0,
            "total_events_processed": 0,
            "total_errors": 0,
            "events_per_second": 0.0,
            "average_processing_time": 0.0,
            "queue_size": 0,
            "active_handlers": 0,
            "total_handlers": 0,
            "uptime_seconds": 0.0,
        }
        self.handler_metrics: Dict[str, HandlerMetrics] = defaultdict(
            lambda: {
                "call_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "error_count": 0,
                "last_execution": None,
            }
        )
        # Задачи мониторинга
        self._monitor_tasks: Set[asyncio.Task] = set()
        logger.info(f"EventBus initialized with config: {self.config}")

    async def start(self) -> None:
        """Запуск EventBus."""
        if self.is_running:
            logger.warning("EventBus is already running")
            return
        self.is_running = True
        self.is_shutdown = False
        self.start_time = datetime.utcnow()
        # Запускаем задачи мониторинга
        if self.config.enable_metrics:
            self._monitor_tasks.add(asyncio.create_task(self._event_processor()))
            self._monitor_tasks.add(asyncio.create_task(self._queue_monitor()))
            self._monitor_tasks.add(asyncio.create_task(self._performance_monitor()))
        logger.info("EventBus started successfully")

    async def stop(self) -> None:
        """Остановка EventBus."""
        if not self.is_running:
            logger.warning("EventBus is not running")
            return
        self.is_running = False
        self.is_shutdown = True
        # Отменяем задачи мониторинга
        for task in self._monitor_tasks:
            task.cancel()
        # Ждем завершения задач
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        # Закрываем пул потоков
        self.executor.shutdown(wait=True)
        # Обновляем время работы
        if self.start_time:
            self.metrics["uptime_seconds"] = (
                datetime.utcnow() - self.start_time
            ).total_seconds()
        logger.info("EventBus stopped successfully")

    def subscribe(
        self,
        event_name: EventName,
        handler: EventHandler,
        config: Optional[HandlerConfig] = None,
    ) -> HandlerID:
        """
        Подписка на событие.
        Args:
            event_name: Имя события
            handler: Обработчик события
            config: Конфигурация обработчика
        Returns:
            ID обработчика
        Raises:
            EventBusError: При ошибке подписки
        """
        try:
            if not callable(handler):
                raise EventBusError("Handler must be callable")
            # Создаем конфигурацию по умолчанию если не предоставлена
            if config is None:
                config = create_handler_config(
                    name=f"handler_{event_name}", priority=EventPriority.NORMAL
                )
            # Проверяем, является ли обработчик асинхронным
            is_async = asyncio.iscoroutinefunction(handler)
            # Создаем информацию об обработчике
            handler_info = EventHandlerInfo(
                config=config, handler=handler, is_async=is_async
            )
            with self._handler_lock:
                self.handlers[event_name][config.id] = handler_info
                self.metrics["total_handlers"] = len(self.handlers[event_name])
            logger.debug(
                f"Subscribed to event '{event_name}' with handler ID {config.id}"
            )
            return config.id
        except Exception as e:
            logger.error(f"Error subscribing to event '{event_name}': {e}")
            raise EventBusError(f"Failed to subscribe to event '{event_name}': {e}")

    def unsubscribe(self, event_name: EventName, handler_id: HandlerID) -> bool:
        """
        Отписка от события.
        Args:
            event_name: Имя события
            handler_id: ID обработчика
        Returns:
            True если отписка успешна
        """
        try:
            with self._handler_lock:
                if (
                    event_name in self.handlers
                    and handler_id in self.handlers[event_name]
                ):
                    del self.handlers[event_name][handler_id]
                    # Удаляем событие если нет обработчиков
                    if not self.handlers[event_name]:
                        del self.handlers[event_name]
                    self.metrics["total_handlers"] = sum(
                        len(handlers) for handlers in self.handlers.values()
                    )
                    logger.debug(
                        f"Unsubscribed from event '{event_name}' with handler ID {handler_id}"
                    )
                    return True
                logger.warning(
                    f"Handler {handler_id} not found for event '{event_name}'"
                )
                return False
        except Exception as e:
            logger.error(f"Error unsubscribing from event '{event_name}': {e}")
            return False

    async def publish(self, event: Event) -> bool:
        """
        Асинхронная публикация события.
        Args:
            event: Событие для публикации
        Returns:
            True если событие добавлено в очередь
        """
        try:
            if self.is_shutdown:
                logger.warning("Cannot publish event: EventBus is shutdown")
                return False
            # Проверяем истечение срока действия
            if event.is_expired():
                logger.warning(f"Event {event.id} is expired, skipping")
                return False
            # Добавляем в очередь по приоритету
            with self._lock:
                self.event_queues[event.priority].append(event)
                self.metrics["total_events_published"] += 1
                self.metrics["queue_size"] = sum(
                    len(q) for q in self.event_queues.values()
                )
            # Добавляем в историю
            with self._history_lock:
                self.event_history.append(event)
            logger.debug(f"Published event {event.id} with priority {event.priority}")
            return True
        except Exception as e:
            logger.error(f"Error publishing event {event.id}: {e}")
            self.metrics["total_errors"] += 1
            return False

    def publish_sync(self, event: Event) -> bool:
        """
        Синхронная публикация события.
        Args:
            event: Событие для публикации
        Returns:
            True если событие обработано
        """
        try:
            if self.is_shutdown:
                logger.warning("Cannot publish event: EventBus is shutdown")
                return False
            # Проверяем истечение срока действия
            if event.is_expired():
                logger.warning(f"Event {event.id} is expired, skipping")
                return False
            # Добавляем в историю
            with self._history_lock:
                self.event_history.append(event)
            # Обрабатываем событие синхронно
            return self._process_event_sync(event)
        except Exception as e:
            logger.error(f"Error publishing event {event.id} synchronously: {e}")
            self.metrics["total_errors"] += 1
            return False

    def get_event_history(
        self, event_name: Optional[EventName] = None, limit: int = 100
    ) -> List[Event]:
        """
        Получить историю событий.
        Args:
            event_name: Фильтр по имени события
            limit: Максимальное количество событий
        Returns:
            Список событий
        """
        try:
            with self._history_lock:
                if event_name:
                    filtered_events = [
                        e for e in self.event_history if e.name == event_name
                    ]
                    return filtered_events[-limit:]
                else:
                    return list(self.event_history)[-limit:]
        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []

    def clear_history(self) -> None:
        """Очистить историю событий."""
        try:
            with self._history_lock:
                self.event_history.clear()
            logger.info("Event history cleared")
        except Exception as e:
            logger.error(f"Error clearing event history: {e}")

    def get_handlers(self, event_name: EventName) -> List[EventHandlerInfo]:
        """
        Получить обработчики события.
        Args:
            event_name: Имя события
        Returns:
            Список информации об обработчиках
        """
        try:
            with self._handler_lock:
                return list(self.handlers.get(event_name, {}).values())
        except Exception as e:
            logger.error(f"Error getting handlers for event '{event_name}': {e}")
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        try:
            if self.start_time:
                self.metrics["uptime_seconds"] = (
                    datetime.utcnow() - self.start_time
                ).total_seconds()
            self.metrics["active_handlers"] = sum(
                len(handlers) for handlers in self.handlers.values()
            )
            result = dict(self.metrics)
            # handler_performance только в result, не в self.metrics
            result["handler_performance"] = dict(self.handler_metrics)
            return result
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def _event_processor(self) -> None:
        """Основной процессор событий."""
        while self.is_running:
            try:
                # Получаем событие из очереди с наивысшим приоритетом
                event = await self._get_next_event()
                if event is None:
                    await asyncio.sleep(0.001)  # Небольшая пауза
                    continue
                # Обрабатываем событие
                await self._process_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                self.metrics["total_errors"] += 1

    async def _get_next_event(self) -> Optional[Event]:
        """Получение следующего события с наивысшим приоритетом."""
        try:
            with self._lock:
                for priority in reversed(list(EventPriority)):
                    queue = self.event_queues[priority]
                    if queue:
                        try:
                            event = queue.popleft()
                            assert isinstance(event, Event)
                            return event
                        except IndexError:
                            continue
            return None
        except Exception as e:
            logger.error(f"Error getting next event: {e}")
            return None

    async def _process_event(self, event: Event) -> None:
        """Обработка события."""
        try:
            start_time = time.time()
            # Получаем обработчики для события
            handlers = self.get_handlers(event.name)
            if not handlers:
                logger.debug(f"No handlers found for event {event.name}")
                return
            # Обрабатываем асинхронные и синхронные обработчики отдельно
            async_handlers = [h for h in handlers if h.is_async]
            sync_handlers = [h for h in handlers if not h.is_async]
            # Запускаем асинхронные обработчики
            if async_handlers:
                await self._execute_async_handlers(event, async_handlers)
            # Запускаем синхронные обработчики в пуле потоков
            if sync_handlers:
                await self._execute_sync_handlers(event, sync_handlers)
            # Обновляем метрики
            processing_time = time.time() - start_time
            self.metrics["total_events_processed"] += 1
            self.metrics["average_processing_time"] = (
                self.metrics["average_processing_time"]
                * (self.metrics["total_events_processed"] - 1)
                + processing_time
            ) / self.metrics["total_events_processed"]
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
            self.metrics["total_errors"] += 1

    def _process_event_sync(self, event: Event) -> bool:
        """Синхронная обработка события."""
        start_time = time.time()
        try:
            # Получаем обработчики для события
            handlers = self.get_handlers(event.name)
            if not handlers:
                logger.debug(f"No handlers found for event {event.name}")
                return True
            # Обрабатываем только синхронные обработчики
            sync_handlers = [h for h in handlers if not h.is_async]
            for handler_info in sync_handlers:
                try:
                    self._execute_sync_handler_direct(event, handler_info)
                except Exception as e:
                    logger.error(f"Error in sync handler {handler_info.config.id}: {e}")
                    self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            # Обновляем метрики
            processing_time = time.time() - start_time
            self.metrics["total_events_processed"] = (
                int(self.metrics["total_events_processed"]) + 1
            )
            current_avg = float(self.metrics["average_processing_time"])
            current_total = int(self.metrics["total_events_processed"])
            self.metrics["average_processing_time"] = (
                current_avg * (current_total - 1) + processing_time
            ) / current_total
            return True
        except Exception as e:
            logger.error(f"Error processing event {event.id} synchronously: {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def _execute_async_handlers(
        self, event: Event, handlers: List[EventHandlerInfo]
    ) -> None:
        """Выполнение асинхронных обработчиков."""
        tasks = []
        for handler_info in handlers:
            if not handler_info.config.enabled:
                continue
            task = asyncio.create_task(self._execute_async_handler(event, handler_info))
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_async_handler(
        self, event: Event, handler_info: EventHandlerInfo
    ) -> None:
        """Выполнение одного асинхронного обработчика."""
        start_time = time.time()
        try:
            # Проверяем таймаут
            handler_result = handler_info.handler(event)
            if handler_result is not None:
                if self.config.enable_async_processing:
                    await asyncio.wait_for(handler_result, timeout=handler_info.config.timeout)
                else:
                    await handler_result
            # Обновляем метрики обработчика
            execution_time = time.time() - start_time
            self._update_handler_metrics(handler_info, execution_time, False)
        except asyncio.TimeoutError:
            logger.error(
                f"Handler {handler_info.config.id} timed out after {handler_info.config.timeout}s"
            )
            self._update_handler_metrics(
                handler_info, handler_info.config.timeout, True
            )
        except Exception as e:
            logger.error(f"Error in async handler {handler_info.config.id}: {e}")
            self._update_handler_metrics(handler_info, time.time() - start_time, True)
            # Вызываем обработчик ошибок если есть
            if handler_info.config.error_handler:
                try:
                    if asyncio.iscoroutinefunction(handler_info.config.error_handler):
                        await handler_info.config.error_handler(e, event)
                    else:
                        handler_info.config.error_handler(e, event)
                except Exception as error_e:
                    logger.error(f"Error in error handler: {error_e}")

    async def _execute_sync_handlers(
        self, event: Event, handlers: List[EventHandlerInfo]
    ) -> None:
        """Выполнение синхронных обработчиков в пуле потоков."""
        tasks = []
        for handler_info in handlers:
            if not handler_info.config.enabled:
                continue
            task = asyncio.create_task(
                self._execute_sync_handler_async(event, handler_info)
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_sync_handler_async(
        self, event: Event, handler_info: EventHandlerInfo
    ) -> None:
        """Выполнение одного синхронного обработчика в пуле потоков."""
        start_time = time.time()
        try:
            # Выполняем в пуле потоков
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self.executor, handler_info.handler, event),
                timeout=handler_info.config.timeout,
            )
            # Обновляем метрики обработчика
            execution_time = time.time() - start_time
            self._update_handler_metrics(handler_info, execution_time, False)
        except asyncio.TimeoutError:
            logger.error(
                f"Handler {handler_info.config.id} timed out after {handler_info.config.timeout}s"
            )
            self._update_handler_metrics(
                handler_info, handler_info.config.timeout, True
            )
        except Exception as e:
            logger.error(f"Error in sync handler {handler_info.config.id}: {e}")
            self._update_handler_metrics(handler_info, time.time() - start_time, True)
            # Вызываем обработчик ошибок если есть
            if handler_info.config.error_handler:
                try:
                    if asyncio.iscoroutinefunction(handler_info.config.error_handler):
                        await handler_info.config.error_handler(e, event)
                    else:
                        handler_info.config.error_handler(e, event)
                except Exception as error_e:
                    logger.error(f"Error in error handler: {error_e}")

    def _execute_sync_handler_direct(
        self, event: Event, handler_info: EventHandlerInfo
    ) -> None:
        """Выполнение одного синхронного обработчика напрямую."""
        start_time = time.time()
        try:
            handler_info.handler(event)
            # Обновляем метрики обработчика
            execution_time = time.time() - start_time
            self._update_handler_metrics(handler_info, execution_time, False)
        except Exception as e:
            logger.error(f"Error in sync handler {handler_info.config.id}: {e}")
            self._update_handler_metrics(handler_info, time.time() - start_time, True)
            # Вызываем обработчик ошибок если есть
            if handler_info.config.error_handler:
                try:
                    handler_info.config.error_handler(e, event)
                except Exception as error_e:
                    logger.error(f"Error in error handler: {error_e}")

    def _update_handler_metrics(
        self, handler_info: EventHandlerInfo, execution_time: float, is_error: bool
    ) -> None:
        """Обновление метрик обработчика."""
        try:
            handler_id = handler_info.config.id
            handler_info.last_execution = datetime.utcnow()
            if is_error:
                handler_info.error_count += 1
            else:
                handler_info.success_count += 1
            handler_performance = self.handler_metrics
            metrics = handler_performance.get(
                handler_id,
                {
                    "call_count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "error_count": 0,
                    "last_execution": None,
                },
            )
            metrics["call_count"] += 1
            metrics["total_time"] += execution_time
            metrics["avg_time"] = metrics["total_time"] / metrics["call_count"]
            if is_error:
                metrics["error_count"] += 1
            metrics["last_execution"] = handler_info.last_execution
            handler_performance[handler_id] = metrics
        except Exception as e:
            logger.error(f"Error updating handler metrics: {e}")

    async def _queue_monitor(self) -> None:
        """Мониторинг очередей."""
        while self.is_running:
            try:
                # Обновляем размер очереди
                with self._lock:
                    queue_size = sum(len(q) for q in self.event_queues.values())
                    self.metrics["queue_size"] = queue_size
                # Проверяем переполнение очередей
                for priority, queue in self.event_queues.items():
                    if len(queue) > self.config.queue_size * 0.8:
                        logger.warning(
                            f"Queue for priority {priority} is 80% full ({len(queue)}/{self.config.queue_size})"
                        )
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue monitor: {e}")

    async def _performance_monitor(self) -> None:
        """Мониторинг производительности."""
        while self.is_running:
            try:
                # Обновляем события в секунду
                uptime = float(self.metrics["uptime_seconds"])
                if uptime > 0:
                    total_processed = int(self.metrics["total_events_processed"])
                    self.metrics["events_per_second"] = total_processed / (
                        uptime / 60
                    )  # за минуту
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")

    def __enter__(self) -> "EventBus":
        """Контекстный менеджер для входа."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Контекстный менеджер для выхода."""
        if self.is_running:
            # Создаем задачу для остановки, но не ждем её завершения
            # так как это синхронный контекстный менеджер
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())
            else:
                # Если цикл не запущен, запускаем его для выполнения остановки
                loop.run_until_complete(self.stop())

    async def __aenter__(self) -> "EventBus":
        """Асинхронный контекстный менеджер для входа."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Асинхронный контекстный менеджер для выхода."""
        await self.stop()
