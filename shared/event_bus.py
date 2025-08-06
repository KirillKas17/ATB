"""
Event Bus для системы обмена событиями.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Приоритеты событий."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    """Базовое событие."""
    event_type: str
    data: Any
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None

class EventBus:
    """Шина событий для асинхронного обмена сообщениями."""
    
    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Подписка на событие."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        
    async def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Отписка от события."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass
                
    async def publish(self, event: Event) -> None:
        """Публикация события."""
        if not self._running:
            return
            
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error handling event {event.event_type}: {e}")
                
    async def start(self) -> None:
        """Запуск шины событий."""
        self._running = True
        
    async def stop(self) -> None:
        """Остановка шины событий."""
        self._running = False

# Глобальная шина событий
event_bus = EventBus()