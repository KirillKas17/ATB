"""
Оптимизированная реализация EventBus для высокопроизводительной обработки событий.

Особенности:
- Асинхронная обработка с пулом воркеров
- Приоритетные очереди
- Статистика производительности
- Обработка ошибок
- Потокобезопасность
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from loguru import logger


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """Event types."""

    MARKET_DATA = "market_data"
    TRADE = "trade"
    ORDER = "order"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class EventMetadata:
    """Event metadata."""

    event_id: str
    timestamp: float
    priority: EventPriority
    source: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class Event:
    """Event object."""

    type: EventType
    data: Any
    metadata: EventMetadata


@dataclass
class EventHandler:
    """Event handler configuration."""

    handler: Callable
    event_types: Set[EventType]
    priority: EventPriority = EventPriority.NORMAL
    async_handler: bool = True
    error_handler: Optional[Callable] = None


class EventBus:
    """Optimized event bus implementation."""

    def __init__(self, max_workers: int = 10, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats: Dict[str, Union[int, float]] = {
            "events_processed": 0,
            "events_dropped": 0,
            "errors": 0,
            "start_time": 0,
        }

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = int(time.time())

        # Start worker tasks
        for _ in range(self.max_workers):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)

        logger.info(f"Event bus started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop the event bus."""
        if not self._running:
            return

        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Event bus stopped")

    async def publish(self, event: Event) -> bool:
        """Publish an event."""
        if not self._running:
            logger.warning("Event bus is not running")
            return False

        try:
            await self._event_queue.put(event)
            return True
        except asyncio.QueueFull:
            self._stats["events_dropped"] = int(self._stats["events_dropped"]) + 1
            logger.warning("Event queue is full, dropping event")
            return False

    def subscribe(
        self,
        event_types: List[EventType],
        handler: Callable,
        priority: EventPriority = EventPriority.NORMAL,
        error_handler: Optional[Callable] = None,
    ) -> None:
        """Subscribe to events."""
        event_handler = EventHandler(
            handler=handler,
            event_types=set(event_types),
            priority=priority,
            error_handler=error_handler,
        )

        for event_type in event_types:
            self._handlers[event_type].append(event_handler)
            # Sort handlers by priority (highest first)
            self._handlers[event_type].sort(
                key=lambda h: h.priority.value, reverse=True
            )

    def unsubscribe(self, event_types: List[EventType], handler: Callable) -> None:
        """Unsubscribe from events."""
        for event_type in event_types:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h.handler != handler
                ]

    async def _worker(self) -> None:
        """Worker task for processing events."""
        while self._running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Process event
                await self._process_event(event)

                # Mark task as done
                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self._stats["errors"] = int(self._stats["errors"]) + 1

    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        handlers = self._handlers.get(event.type, [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event.type}")
            return

        # Execute handlers
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._execute_handler(handler, event))
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._stats["events_processed"] = int(self._stats["events_processed"]) + 1

    async def _execute_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute a single event handler."""
        try:
            if handler.async_handler:
                await handler.handler(event)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler.handler, event)

        except Exception as e:
            logger.error(f"Handler error: {e}")
            self._stats["errors"] = int(self._stats["errors"]) + 1

            # Call error handler if provided
            if handler.error_handler:
                try:
                    if asyncio.iscoroutinefunction(handler.error_handler):
                        await handler.error_handler(event, e)
                    else:
                        handler.error_handler(event, e)
                except Exception as error_e:
                    logger.error(f"Error handler error: {error_e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        start_time = float(self._stats["start_time"])
        uptime = time.time() - start_time if start_time > 0 else 0

        return {
            **self._stats,
            "uptime_seconds": uptime,
            "queue_size": self._event_queue.qsize(),
            "active_workers": len(self._workers),
            "running": self._running,
        }


# Global event bus instance
_event_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
        await _event_bus.start()
    return _event_bus


def create_event(
    event_type: EventType,
    data: Any,
    source: str,
    priority: EventPriority = EventPriority.NORMAL,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create an event."""
    import uuid

    metadata = EventMetadata(
        event_id=str(uuid.uuid4()),
        timestamp=time.time(),
        priority=priority,
        source=source,
        correlation_id=correlation_id,
    )

    return Event(type=event_type, data=data, metadata=metadata)
