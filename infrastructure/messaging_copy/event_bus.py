"""
Реализация EventBus для мессенджинга.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventPriority(Enum):
    """Приоритеты событий."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Event:
    """Событие."""

    def __init__(
        self,
        name: str,
        data: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        self.name = name
        self.data = data
        self.priority = priority
        self.timestamp = datetime.now()
        self.id = f"{name}_{self.timestamp.timestamp()}"


class EventBus:
    """EventBus для обработки событий."""

    def __init__(self) -> None:
        self.handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000

    def subscribe(self, event_name: str, handler: Callable) -> None:
        """Подписаться на событие."""
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)
        return None

    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Отписаться от события."""
        if event_name in self.handlers:
            if handler in self.handlers[event_name]:
                self.handlers[event_name].remove(handler)
        return None

    async def publish(self, event: Event) -> None:
        """Опубликовать событие."""
        # Добавить в историю
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Вызвать обработчики
        if event.name in self.handlers:
            for handler in self.handlers[event.name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Ошибка в обработчике события {event.name}: {e}")
        return None

    def publish_sync(self, event: Event) -> None:
        """Опубликовать событие синхронно."""
        # Добавить в историю
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Вызвать обработчики
        if event.name in self.handlers:
            for handler in self.handlers[event.name]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Ошибка в обработчике события {event.name}: {e}")
        return None

    def get_event_history(
        self, event_name: Optional[str] = None, limit: int = 100
    ) -> List[Event]:
        """Получить историю событий."""
        if event_name:
            filtered_events = [e for e in self.event_history if e.name == event_name]
            return filtered_events[-limit:]
        return self.event_history[-limit:]

    def clear_history(self) -> None:
        """Очистить историю событий."""
        self.event_history.clear()
        return None
