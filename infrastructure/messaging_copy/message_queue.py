"""
Реализация MessageQueue для мессенджинга.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class Message:
    """Сообщение."""

    def __init__(self, topic: str, data: Any, priority: int = 1) -> None:
        self.topic = topic
        self.data = data
        self.priority = priority
        self.timestamp = datetime.now()
        self.id = f"{topic}_{self.timestamp.timestamp()}"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": self.id,
            "topic": self.topic,
            "data": self.data,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Создать из словаря."""
        message = cls(data["topic"], data["data"], data["priority"])
        message.id = data["id"]
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message


class MessageQueue:
    """Очередь сообщений."""

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.queues: Dict[str, List[Message]] = {}
        self.subscribers: Dict[str, List[Callable[[Message], Any]]] = {}
        self.running = False

    async def publish(self, topic: str, data: Any, priority: int = 1) -> bool:
        """Опубликовать сообщение."""
        try:
            message = Message(topic, data, priority)

            if topic not in self.queues:
                self.queues[topic] = []

            # Добавить сообщение в очередь
            self.queues[topic].append(message)

            # Сортировать по приоритету
            self.queues[topic].sort(key=lambda x: x.priority, reverse=True)

            # Ограничить размер очереди
            if len(self.queues[topic]) > self.max_size:
                self.queues[topic] = self.queues[topic][: self.max_size]

            # Уведомить подписчиков
            if topic in self.subscribers:
                for subscriber in self.subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(message)
                        else:
                            subscriber(message)
                    except Exception as e:
                        print(f"Ошибка в подписчике {topic}: {e}")

            return True
        except Exception as e:
            print(f"Ошибка публикации сообщения: {e}")
            return False

    async def subscribe(self, topic: str, handler: Callable[[Message], Any]) -> None:
        """Подписаться на тему."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

    async def unsubscribe(self, topic: str, handler: Callable[[Message], Any]) -> None:
        """Отписаться от темы."""
        if topic in self.subscribers:
            if handler in self.subscribers[topic]:
                self.subscribers[topic].remove(handler)

    async def get_message(self, topic: str) -> Optional[Message]:
        """Получить сообщение из очереди."""
        if topic in self.queues and self.queues[topic]:
            return self.queues[topic].pop(0)
        return None

    async def peek_message(self, topic: str) -> Optional[Message]:
        """Посмотреть сообщение без извлечения."""
        if topic in self.queues and self.queues[topic]:
            return self.queues[topic][0]
        return None

    def get_queue_size(self, topic: str) -> int:
        """Получить размер очереди."""
        return len(self.queues.get(topic, []))

    def get_all_topics(self) -> List[str]:
        """Получить все темы."""
        return list(self.queues.keys())

    async def start(self) -> None:
        """Запустить обработку сообщений."""
        self.running = True
        while self.running:
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Остановить обработку сообщений."""
        self.running = False
