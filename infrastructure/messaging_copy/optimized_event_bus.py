"""
Оптимизированный обработчик событий с асинхронностью и очередями.
Включает:
- Асинхронную обработку событий
- Очереди с приоритетами
- Мониторинг производительности
- Балансировку нагрузки
- Обработку ошибок
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    TypedDict,
    Union,
)

from loguru import logger


class EventPriority(Enum):
    """Приоритеты событий."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Событие с метаданными."""

    name: str
    data: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class EventHandler:
    """Обработчик события."""

    callback: Callable
    priority: EventPriority = EventPriority.NORMAL
    async_handler: bool = False
    timeout: float = 30.0
    error_handler: Optional[Callable] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class HandlerMetrics(TypedDict):
    call_count: int
    total_time: float
    avg_time: float
    error_count: int


class SlowHandlerEntry(TypedDict):
    handler: str
    execution_time: float
    timestamp: float


class OptimizedEventBus:
    """
    Оптимизированный обработчик событий с асинхронностью и очередями.
    """

    def __init__(self, max_workers: int = 10, queue_size: int = 1000) -> None:
        """
        Инициализация оптимизированного обработчика событий.
        Args:
            max_workers: Максимальное количество рабочих потоков
            queue_size: Размер очереди событий
        """
        self.max_workers = max_workers
        self.queue_size = queue_size
        # Очереди событий по приоритетам
        self.event_queues: Dict[EventPriority, deque] = {
            priority: deque(maxlen=queue_size) for priority in EventPriority
        }
        # Обработчики событий
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        # Асинхронные обработчики
        self.async_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        # Пул потоков для синхронных обработчиков
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Флаги состояния
        self.is_running = False
        self.is_shutdown = False
        # Очередь для обработки
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        # Метрики производительности
        self.performance_metrics: Dict[str, Any] = {
            "total_events_processed": 0,
            "total_processing_time": 0.0,
            "events_per_second": 0.0,
            "error_count": 0,
            "retry_count": 0,
            "queue_size": 0,
            "active_workers": 0,
            "slow_handlers": [],
        }
        # Метрики обработчиков
        self.handler_metrics: Dict[str, HandlerMetrics] = defaultdict(
            lambda: {
                "call_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "error_count": 0,
            }
        )
        # Блокировки для потокобезопасности
        self._lock = threading.RLock()
        self._handler_lock = threading.RLock()
        logger.info(
            f"OptimizedEventBus initialized with {max_workers} workers and queue size {queue_size}"
        )

    async def start(self) -> None:
        """Запуск обработчика событий."""
        if self.is_running:
            return
        self.is_running = True
        self.is_shutdown = False
        # Запускаем обработчики событий
        asyncio.create_task(self._event_processor())
        asyncio.create_task(self._queue_monitor())
        asyncio.create_task(self._performance_monitor())
        logger.info("Optimized event bus started")

    async def stop(self) -> None:
        """Остановка обработчика событий."""
        if not self.is_running:
            return
        self.is_running = False
        self.is_shutdown = True
        # Ждем завершения обработки
        await self.processing_queue.join()
        # Закрываем пул потоков
        self.executor.shutdown(wait=True)
        logger.info("Optimized event bus stopped")

    def subscribe(
        self,
        event_name: str,
        handler: Callable,
        priority: EventPriority = EventPriority.NORMAL,
        async_handler: bool = False,
        timeout: float = 30.0,
        error_handler: Optional[Callable] = None,
    ) -> None:
        """
        Подписка на событие.
        Args:
            event_name: Имя события
            handler: Обработчик события
            priority: Приоритет обработчика
            async_handler: Асинхронный ли обработчик
            timeout: Таймаут обработки
            error_handler: Обработчик ошибок
        """
        with self._handler_lock:
            event_handler = EventHandler(
                callback=handler,
                priority=priority,
                async_handler=async_handler,
                timeout=timeout,
                error_handler=error_handler,
            )
            if async_handler:
                self.async_handlers[event_name].append(event_handler)
            else:
                self.handlers[event_name].append(event_handler)
            # Сортируем по приоритету
            self.handlers[event_name].sort(key=lambda h: h.priority.value, reverse=True)
            self.async_handlers[event_name].sort(
                key=lambda h: h.priority.value, reverse=True
            )
            logger.debug(f"Subscribed to event '{event_name}' with priority {priority}")

    def unsubscribe(self, event_name: str, handler: Callable) -> bool:
        """
        Отписка от события.
        Args:
            event_name: Имя события
            handler: Обработчик события
        Returns:
            True если отписка успешна
        """
        with self._handler_lock:
            # Удаляем из синхронных обработчиков
            self.handlers[event_name] = [
                h for h in self.handlers[event_name] if h.callback != handler
            ]
            # Удаляем из асинхронных обработчиков
            self.async_handlers[event_name] = [
                h for h in self.async_handlers[event_name] if h.callback != handler
            ]
            return True

    def publish(
        self,
        event_name: str,
        data: Any,
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Публикация события.
        Args:
            event_name: Имя события
            data: Данные события
            priority: Приоритет события
            source: Источник события
            correlation_id: ID корреляции
        Returns:
            True если событие добавлено в очередь
        """
        if self.is_shutdown:
            return False
        event = Event(
            name=event_name,
            data=data,
            priority=priority,
            source=source,
            correlation_id=correlation_id,
        )
        # Добавляем в очередь по приоритету
        try:
            self.event_queues[priority].append(event)
            self.performance_metrics["queue_size"] = sum(
                len(q) for q in self.event_queues.values()
            )
            return True
        except IndexError:
            logger.warning(f"Event queue for priority {priority} is full")
            return False

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

    async def _get_next_event(self) -> Optional[Event]:
        """Получение следующего события с наивысшим приоритетом."""
        for priority in reversed(list(EventPriority)):
            queue = self.event_queues[priority]
            if queue:
                try:
                    event = queue.popleft()
                    return event
                except IndexError:
                    continue
        return None

    async def _process_event(self, event: Event) -> None:
        """Обработка события."""
        start_time = time.time()
        try:
            # Обрабатываем асинхронные обработчики
            async_handlers = self.async_handlers.get(event.name, [])
            for handler in async_handlers:
                await self._execute_async_handler(event, handler)
            # Обрабатываем синхронные обработчики
            sync_handlers = self.handlers.get(event.name, [])
            if sync_handlers:
                await self._execute_sync_handlers(event, sync_handlers)
            # Обновляем метрики
            processing_time = time.time() - start_time
            self.performance_metrics["total_events_processed"] = (
                int(self.performance_metrics["total_events_processed"]) + 1
            )
            self.performance_metrics["total_processing_time"] = (
                float(self.performance_metrics["total_processing_time"])
                + processing_time
            )
        except Exception as e:
            logger.error(f"Error processing event {event.name}: {e}")
            self.performance_metrics["error_count"] = (
                int(self.performance_metrics["error_count"]) + 1
            )
            # Повторная попытка обработки
            if event.retry_count < event.max_retries:
                event.retry_count += 1
                self.performance_metrics["retry_count"] = (
                    int(self.performance_metrics["retry_count"]) + 1
                )
                await asyncio.sleep(1)  # Пауза перед повтором
                await self._process_event(event)

    async def _execute_async_handler(self, event: Event, handler: EventHandler) -> None:
        """Выполнение асинхронного обработчика."""
        start_time = time.time()
        try:
            # Выполняем с таймаутом
            await asyncio.wait_for(handler.callback(event), timeout=handler.timeout)
            # Обновляем метрики обработчика
            execution_time = time.time() - start_time
            self._update_handler_metrics(handler, execution_time, False)
        except asyncio.TimeoutError:
            logger.warning(f"Async handler timeout for event {event.name}")
            self._update_handler_metrics(handler, handler.timeout, True)
        except Exception as e:
            logger.error(f"Error in async handler for event {event.name}: {e}")
            self._update_handler_metrics(handler, time.time() - start_time, True)
            # Вызываем обработчик ошибок
            if handler.error_handler:
                try:
                    await handler.error_handler(event, e)
                except Exception as error_e:
                    logger.error(f"Error in error handler: {error_e}")

    async def _execute_sync_handlers(
        self, event: Event, handlers: List[EventHandler]
    ) -> None:
        """Выполнение синхронных обработчиков."""
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._execute_sync_handler(event, handler))
            tasks.append(task)
        # Выполняем все обработчики параллельно
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_sync_handler(self, event: Event, handler: EventHandler) -> None:
        """Выполнение синхронного обработчика."""
        start_time = time.time()
        try:
            # Выполняем в пуле потоков
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self.executor, handler.callback, event),
                timeout=handler.timeout,
            )
            # Обновляем метрики обработчика
            execution_time = time.time() - start_time
            self._update_handler_metrics(handler, execution_time, False)
        except asyncio.TimeoutError:
            logger.warning(f"Sync handler timeout for event {event.name}")
            self._update_handler_metrics(handler, handler.timeout, True)
        except Exception as e:
            logger.error(f"Error in sync handler for event {event.name}: {e}")
            self._update_handler_metrics(handler, time.time() - start_time, True)
            # Вызываем обработчик ошибок
            if handler.error_handler:
                try:
                    await handler.error_handler(event, e)
                except Exception as error_e:
                    logger.error(f"Error in error handler: {error_e}")

    def _update_handler_metrics(
        self, handler: EventHandler, execution_time: float, is_error: bool
    ) -> None:
        """Обновление метрик обработчика."""
        handler_name = handler.callback.__name__
        metrics = self.handler_metrics[handler_name]
        metrics["call_count"] = int(metrics["call_count"]) + 1
        metrics["total_time"] = float(metrics["total_time"]) + execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["call_count"]
        if is_error:
            metrics["error_count"] = int(metrics["error_count"]) + 1
        # Логируем медленные обработчики (>1 секунды)
        if execution_time > 1.0:
            slow_handlers = self.performance_metrics["slow_handlers"]
            if isinstance(slow_handlers, list):
                slow_handlers.append(
                    {
                        "handler": handler_name,
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                    }
                )
                # Ограничиваем количество записей
                if len(slow_handlers) > 100:
                    self.performance_metrics["slow_handlers"] = slow_handlers[-50:]

    async def _queue_monitor(self) -> None:
        """Мониторинг очередей."""
        while self.is_running:
            try:
                # Обновляем размер очереди
                self.performance_metrics["queue_size"] = sum(
                    len(q) for q in self.event_queues.values()
                )
                # Проверяем переполнение очередей
                for priority, queue in self.event_queues.items():
                    if len(queue) > self.queue_size * 0.8:
                        logger.warning(f"Queue for priority {priority} is 80% full")
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
                total_processing_time = float(
                    self.performance_metrics["total_processing_time"]
                )
                if total_processing_time > 0:
                    total_events = int(
                        self.performance_metrics["total_events_processed"]
                    )
                    self.performance_metrics["events_per_second"] = total_events / (
                        total_processing_time / 60
                    )  # за минуту
                # Обновляем количество активных рабочих потоков
                self.performance_metrics["active_workers"] = len(self.executor._threads)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        return self.performance_metrics.copy()

    def get_queue_status(self) -> Dict[str, Any]:
        """Получение статуса очередей."""
        return {
            priority.name: {"size": len(queue), "max_size": queue.maxlen}
            for priority, queue in self.event_queues.items()
        }

    def get_handler_stats(self) -> Dict[str, Any]:
        """Получение статистики обработчиков."""
        return dict(self.handler_metrics)

    def clear_metrics(self) -> None:
        """Очистка метрик."""
        self.performance_metrics = {
            "total_events_processed": 0,
            "total_processing_time": 0.0,
            "events_per_second": 0.0,
            "error_count": 0,
            "retry_count": 0,
            "queue_size": 0,
            "active_workers": 0,
            "slow_handlers": [],
        }
        self.handler_metrics.clear()
