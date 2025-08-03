"""
Реализация WebSocket сервиса.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List


class WebSocketConnection:
    """WebSocket соединение."""

    def __init__(self, connection_id: str, websocket: Any) -> None:
        self.connection_id = connection_id
        self.websocket = websocket
        self.subscriptions: List[str] = []
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()

    async def send(self, message: str) -> bool:
        """Отправить сообщение."""
        try:
            await self.websocket.send_text(message)
            self.last_activity = datetime.now()
            return True
        except Exception as e:
            print(f"Ошибка отправки сообщения: {e}")
            return False

    async def close(self) -> None:
        """Закрыть соединение."""
        try:
            await self.websocket.close()
        except Exception as e:
            print(f"Ошибка закрытия соединения: {e}")
        return None


class WebSocketService:
    """WebSocket сервис."""

    def __init__(self) -> None:
        self.connections: Dict[str, WebSocketConnection] = {}
        self.subscribers: Dict[str, List[str]] = {}  # topic -> connection_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False

    async def add_connection(
        self, connection_id: str, websocket: Any
    ) -> WebSocketConnection:
        """Добавить соединение."""
        connection = WebSocketConnection(connection_id, websocket)
        self.connections[connection_id] = connection
        return connection

    async def remove_connection(self, connection_id: str) -> None:
        """Удалить соединение."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]

            # Удалить из подписок
            for topic in connection.subscriptions:
                if topic in self.subscribers:
                    if connection_id in self.subscribers[topic]:
                        self.subscribers[topic].remove(connection_id)

            # Закрыть соединение
            await connection.close()
            del self.connections[connection_id]
        return None

    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Подписаться на тему."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        connection.subscriptions.append(topic)

        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(connection_id)

        return True

    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Отписаться от темы."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        if topic in connection.subscriptions:
            connection.subscriptions.remove(topic)

        if topic in self.subscribers:
            if connection_id in self.subscribers[topic]:
                self.subscribers[topic].remove(connection_id)

        return True

    async def broadcast(self, topic: str, message: Any) -> int:
        """Отправить сообщение всем подписчикам темы."""
        if topic not in self.subscribers:
            return 0

        message_str = json.dumps(
            {"topic": topic, "data": message, "timestamp": datetime.now().isoformat()}
        )

        sent_count = 0
        for connection_id in self.subscribers[topic]:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                if await connection.send(message_str):
                    sent_count += 1

        return sent_count

    async def send_to_connection(self, connection_id: str, message: Any) -> bool:
        """Отправить сообщение конкретному соединению."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        message_str = json.dumps(
            {"data": message, "timestamp": datetime.now().isoformat()}
        )

        return await connection.send(message_str)

    def get_connection_count(self) -> int:
        """Получить количество соединений."""
        return len(self.connections)

    def get_subscriber_count(self, topic: str) -> int:
        """Получить количество подписчиков темы."""
        return len(self.subscribers.get(topic, []))

    def get_all_topics(self) -> List[str]:
        """Получить все темы."""
        return list(self.subscribers.keys())

    async def start(self) -> None:
        """Запустить сервис."""
        self.running = True
        while self.running:
            await asyncio.sleep(1)
        return None

    async def stop(self) -> None:
        """Остановить сервис."""
        self.running = False

        # Закрыть все соединения
        for connection_id in list(self.connections.keys()):
            await self.remove_connection(connection_id)
        return None
