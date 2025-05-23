import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

from loguru import logger

T = TypeVar("T")


class EventPriority(Enum):
    """Приоритеты событий"""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class Event(Generic[T]):
    """Структура события"""

    type: str
    data: T
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL


class EventCallback(Protocol):
    """Протокол для callback-функций"""

    async def __call__(self, event: Event[Any]) -> None: ...


@dataclass
class EventBusMetrics:
    published_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    queue_size: int = 0


class EventBus:
    """Класс для управления событиями в системе (thread-safe)"""

    def __init__(
        self,
        max_queue_size: int = 1000,
        max_subscribers: int = 100,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
    ):
        """
        Инициализация шины событий

        Args:
            max_queue_size: Максимальный размер очереди
            max_subscribers: Максимальное количество подписчиков на тип события
            batch_size: Размер батча для обработки событий
            batch_timeout: Таймаут для формирования батча (в секундах)
        """
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._max_subscribers = max_subscribers
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._metrics = EventBusMetrics()
        self._lock = threading.Lock()

    async def start(self) -> None:
        """Запуск обработчика событий"""
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._process_events())
            logger.info("Event bus started")

    async def stop(self) -> None:
        """Остановка обработчика событий (TODO: дождаться обработки очереди)"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
        # TODO: дождаться обработки всех событий из очереди (graceful shutdown)

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Подписка на событие с потокобезопасностью и защитой от дублирования"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if callback in self._subscribers[event_type]:
                logger.warning(f"Callback already subscribed for event: {event_type}")
                return
            if len(self._subscribers[event_type]) >= self._max_subscribers:
                raise ValueError(
                    f"Maximum number of subscribers ({self._max_subscribers}) exceeded for event type: {event_type}"
                )
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Отписка от события с обработкой ошибок"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from event: {event_type}")
                except ValueError:
                    logger.warning(f"Callback not found for event: {event_type}")

    async def publish(
        self,
        event_type: str,
        data: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Публикация события с обработкой переполнения очереди"""
        event = Event(
            type=event_type, data=data, timestamp=datetime.now(), priority=priority
        )

        try:
            await self._queue.put(event)
            self._metrics.published_events += 1
            self._metrics.queue_size = self._queue.qsize()
            logger.debug(f"Published event: {event_type}")
        except asyncio.QueueFull:
            logger.error(f"Event queue is full, dropping event: {event_type}")
            # TODO: реализовать fallback-очередь для дропнутых событий
            raise

    def get_metrics(self) -> Dict[str, int]:
        """Получение метрик шины событий"""
        return self._metrics.__dict__.copy()

    async def _process_events(self) -> None:
        """Обработка событий из очереди батчами"""
        batch: List[Event] = []
        while self._is_running:
            try:
                while len(batch) < self._batch_size:
                    try:
                        event = await asyncio.wait_for(
                            self._queue.get(), timeout=self._batch_timeout
                        )
                        batch.append(event)
                    except asyncio.TimeoutError:
                        break

                if not batch:
                    continue

                for event in batch:
                    callbacks = []
                    with self._lock:
                        callbacks = list(self._subscribers.get(event.type, []))
                    for callback in callbacks:
                        try:
                            await callback(event)
                            self._metrics.processed_events += 1
                        except Exception as e:
                            self._metrics.failed_events += 1
                            logger.error(
                                f"Error processing event {event.type}: {str(e)}"
                            )

                    self._queue.task_done()

                self._metrics.queue_size = self._queue.qsize()
                batch.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {str(e)}")
                await asyncio.sleep(1)
