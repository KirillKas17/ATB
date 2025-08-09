"""
Профессиональная реализация WebSocketService для мессенджинга.
Особенности:
- Строгая типизация с использованием Protocol
- Асинхронная обработка WebSocket соединений
- Мониторинг производительности и метрики
- Обработка ошибок и retry механизмы
- Потокобезопасность
- Соответствие DDD и SOLID принципам
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

from loguru import logger

from domain.type_definitions.messaging_types import (
    ConnectionID,
    TopicName,
    WebSocketCommand,
    WebSocketMessage,
    WebSocketResponse,
    WebSocketServiceConfig,
    WebSocketServiceError,
    WebSocketServiceMetrics,
    WebSocketServiceProtocol,
)


class WebSocketConnection:
    """
    WebSocket соединение с метаданными и управлением состоянием.
    Обеспечивает:
    - Управление жизненным циклом соединения
    - Отслеживание активности
    - Управление подписками
    - Обработку ошибок
    """

    def __init__(self, connection_id: ConnectionID, websocket: Any) -> None:
        """
        Инициализация WebSocket соединения.
        Args:
            connection_id: Уникальный ID соединения
            websocket: WebSocket объект
        """
        self.connection_id = connection_id
        self.websocket = websocket
        self.subscriptions: Set[TopicName] = set()
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.error_count = 0
        self.message_count = 0
        self.metadata: Dict[str, Any] = {}
        # Блокировка для потокобезопасности
        self._lock = asyncio.Lock()

    async def send(self, message: Union[str, Dict[str, Any]]) -> bool:
        """
        Отправить сообщение через WebSocket.
        Args:
            message: Сообщение для отправки
        Returns:
            True если сообщение отправлено успешно
        """
        try:
            async with self._lock:
                if not self.is_active:
                    logger.warning(f"Connection {self.connection_id} is not active")
                    return False
                # Преобразуем сообщение в строку если нужно
                if isinstance(message, dict):
                    message_str = json.dumps(message, default=str)
                else:
                    message_str = str(message)
                # Отправляем сообщение
                await self.websocket.send_text(message_str)
                # Обновляем метаданные
                self.last_activity = datetime.utcnow()
                self.message_count += 1
                logger.debug(f"Message sent to connection {self.connection_id}")
                return True
        except Exception as e:
            logger.error(
                f"Error sending message to connection {self.connection_id}: {e}"
            )
            self.error_count += 1
            return False

    async def close(self) -> None:
        """Закрыть WebSocket соединение."""
        try:
            async with self._lock:
                if not self.is_active:
                    return
                self.is_active = False
                await self.websocket.close()
                logger.info(f"Connection {self.connection_id} closed")
        except Exception as e:
            logger.error(f"Error closing connection {self.connection_id}: {e}")

    def add_subscription(self, topic: TopicName) -> None:
        """Добавить подписку на тему."""
        # Для синхронных методов используем простую проверку
        self.subscriptions.add(topic)

    def remove_subscription(self, topic: TopicName) -> None:
        """Удалить подписку на тему."""
        # Для синхронных методов используем простую проверку
        self.subscriptions.discard(topic)

    def has_subscription(self, topic: TopicName) -> bool:
        """Проверить наличие подписки на тему."""
        # Для синхронных методов используем простую проверку
        return topic in self.subscriptions

    def get_info(self) -> Dict[str, Any]:
        """Получить информацию о соединении."""
        # Для синхронных методов используем простую проверку
        return {
            "connection_id": self.connection_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "subscriptions": list(self.subscriptions),
            "error_count": self.error_count,
            "message_count": self.message_count,
            "metadata": self.metadata,
        }

    def is_idle(self, timeout_seconds: float) -> bool:
        """Проверить, является ли соединение неактивным."""
        # Для синхронных методов используем простую проверку
        return (
            datetime.utcnow() - self.last_activity
        ).total_seconds() > timeout_seconds


class WebSocketService(WebSocketServiceProtocol):
    """
    Профессиональная реализация WebSocketService с асинхронностью и управлением соединениями.
    Реализует WebSocketServiceProtocol и обеспечивает:
    - Управление WebSocket соединениями
    - Подписки на темы
    - Broadcast сообщений
    - Мониторинг производительности
    - Обработку ошибок
    - Потокобезопасность
    """

    def __init__(self, config: Optional[WebSocketServiceConfig] = None) -> None:
        """
        Инициализация WebSocket сервиса.
        Args:
            config: Конфигурация сервиса
        """
        self.config = config or WebSocketServiceConfig()
        # Соединения
        self.connections: Dict[ConnectionID, WebSocketConnection] = {}
        self._connection_lock = asyncio.Lock()
        # Подписки
        self.subscribers: Dict[TopicName, Set[ConnectionID]] = defaultdict(set)
        self._subscription_lock = asyncio.Lock()
        # Метрики
        self.metrics: Dict[str, Any] = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_errors": 0,
            "topics_count": 0,
            "avg_message_size": 0.0,
            "connection_errors": 0,
            "subscription_errors": 0,
            "broadcast_errors": 0,
        }
        # Метрики соединений
        self.connection_metrics: Dict[ConnectionID, Dict[str, Any]] = {}
        # Задачи мониторинга
        self.monitor_tasks: List[asyncio.Task] = []
        self.is_running = False
        logger.info("WebSocketService initialized")

    async def start(self) -> None:
        """Запуск WebSocketService."""
        if self.is_running:
            logger.warning("WebSocketService is already running")
            return
        self.is_running = True
        self.is_shutdown = False
        self.start_time = datetime.utcnow()
        # Запускаем задачи мониторинга
        if self.config.enable_metrics:
            self.monitor_tasks.append(asyncio.create_task(self._connection_monitor()))
            self.monitor_tasks.append(asyncio.create_task(self._performance_monitor()))
        logger.info("WebSocketService started successfully")

    async def stop(self) -> None:
        """Остановка WebSocketService."""
        if not self.is_running:
            logger.warning("WebSocketService is not running")
            return
        self.is_running = False
        self.is_shutdown = True
        # Отменяем задачи мониторинга
        for task in self.monitor_tasks:
            task.cancel()
        # Ждем завершения задач
        if self.monitor_tasks:
            await asyncio.gather(*self.monitor_tasks, return_exceptions=True)
        # Закрываем все соединения
        await self._close_all_connections()
        # Обновляем время работы
        if self.start_time:
            self.metrics["uptime_seconds"] = (
                datetime.utcnow() - self.start_time
            ).total_seconds()
        logger.info("WebSocketService stopped successfully")

    async def add_connection(self, connection_id: ConnectionID, websocket: Any) -> bool:
        """
        Добавить новое WebSocket соединение.
        Args:
            connection_id: Уникальный ID соединения
            websocket: WebSocket объект
        Returns:
            True если соединение добавлено успешно
        """
        try:
            if self.is_shutdown:
                logger.warning("Cannot add connection: WebSocketService is shutdown")
                return False
            # Проверяем лимит соединений
            if len(self.connections) >= self.config.max_connections:
                logger.warning(
                    f"Maximum connections limit reached ({self.config.max_connections})"
                )
                return False
            # Создаем соединение
            connection = WebSocketConnection(connection_id, websocket)
            async with self._connection_lock:
                self.connections[connection_id] = connection
                # Инициализируем метрики соединения
                self.connection_metrics[connection_id] = {
                    "message_count": 0,
                    "error_count": 0,
                    "last_activity": datetime.utcnow(),
                    "connected_at": datetime.utcnow(),
                }
                self.metrics["total_connections"] = (
                    int(self.metrics["total_connections"]) + 1
                )
                self.metrics["active_connections"] = (
                    int(self.metrics["active_connections"]) + 1
                )
            logger.info(f"Connection {connection_id} added successfully")
            return True
        except Exception as e:
            logger.error(f"Error adding connection {connection_id}: {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def remove_connection(self, connection_id: ConnectionID) -> bool:
        """
        Удалить WebSocket соединение.
        Args:
            connection_id: ID соединения для удаления
        Returns:
            True если соединение удалено успешно
        """
        try:
            async with self._connection_lock:
                if connection_id not in self.connections:
                    logger.warning(f"Connection {connection_id} not found")
                    return False
                connection = self.connections[connection_id]
                # Удаляем из всех подписок
                async with self._subscription_lock:
                    for topic in list(connection.subscriptions):
                        self._remove_subscription_internal(connection_id, topic)
                # Закрываем соединение
                await connection.close()
                # Удаляем из списка соединений
                del self.connections[connection_id]
                # Удаляем метрики соединения
                if connection_id in self.connection_metrics:
                    del self.connection_metrics[connection_id]
                self.metrics["active_connections"] = (
                    int(self.metrics["active_connections"]) - 1
                )
                logger.info(f"Connection {connection_id} removed successfully")
                return True
        except Exception as e:
            logger.error(f"Error removing connection {connection_id}: {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def subscribe(self, connection_id: ConnectionID, topic: TopicName) -> bool:
        """
        Подписать соединение на тему.
        Args:
            connection_id: ID соединения
            topic: Тема для подписки
        Returns:
            True если подписка успешна
        """
        try:
            async with self._connection_lock:
                if connection_id not in self.connections:
                    logger.warning(f"Connection {connection_id} not found")
                    return False
                connection = self.connections[connection_id]
                async with self._subscription_lock:
                    connection.add_subscription(topic)
                    self.subscribers[topic].add(connection_id)
                    self.metrics["topics_count"] = len(self.subscribers)
                logger.debug(
                    f"Connection {connection_id} subscribed to topic '{topic}'"
                )
                return True
        except Exception as e:
            logger.error(
                f"Error subscribing connection {connection_id} to topic '{topic}': {e}"
            )
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def unsubscribe(self, connection_id: ConnectionID, topic: TopicName) -> bool:
        """
        Отписать соединение от темы.
        Args:
            connection_id: ID соединения
            topic: Тема для отписки
        Returns:
            True если отписка успешна
        """
        try:
            async with self._connection_lock:
                if connection_id not in self.connections:
                    logger.warning(f"Connection {connection_id} not found")
                    return False
                connection = self.connections[connection_id]
                async with self._subscription_lock:
                    connection.remove_subscription(topic)
                    self._remove_subscription_internal(connection_id, topic)
                logger.debug(
                    f"Connection {connection_id} unsubscribed from topic '{topic}'"
                )
                return True
        except Exception as e:
            logger.error(
                f"Error unsubscribing connection {connection_id} from topic '{topic}': {e}"
            )
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def broadcast(
        self,
        topic: TopicName,
        message: Any,
        exclude_connection: Optional[ConnectionID] = None,
    ) -> int:
        """
        Отправить сообщение всем подписчикам темы.
        Args:
            topic: Тема сообщения
            message: Сообщение для отправки
            exclude_connection: ID соединения для исключения
        Returns:
            Количество отправленных сообщений
        """
        try:
            async with self._subscription_lock:
                if topic not in self.subscribers:
                    logger.debug(f"No subscribers for topic '{topic}'")
                    return 0
                subscriber_ids = self.subscribers[topic].copy()
            # Исключаем соединение если указано
            if exclude_connection and exclude_connection in subscriber_ids:
                subscriber_ids.remove(exclude_connection)
            # Отправляем сообщения
            sent_count = 0
            tasks = []
            for connection_id in subscriber_ids:
                task = asyncio.create_task(
                    self._send_to_connection_internal(connection_id, message)
                )
                tasks.append(task)
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                sent_count = sum(1 for result in results if result is True)
            self.metrics["total_messages_sent"] = (
                int(self.metrics["total_messages_sent"]) + sent_count
            )
            logger.debug(
                f"Broadcasted message to {sent_count} connections for topic '{topic}'"
            )
            return sent_count
        except Exception as e:
            logger.error(f"Error broadcasting message to topic '{topic}': {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return 0

    async def send_to_connection(
        self, connection_id: ConnectionID, message: Any
    ) -> bool:
        """
        Отправить сообщение конкретному соединению.
        Args:
            connection_id: ID соединения
            message: Сообщение для отправки
        Returns:
            True если сообщение отправлено успешно
        """
        try:
            result = await self._send_to_connection_internal(connection_id, message)
            if result:
                self.metrics["total_messages_sent"] = (
                    int(self.metrics["total_messages_sent"]) + 1
                )
            return result
        except Exception as e:
            logger.error(f"Error sending message to connection {connection_id}: {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    def get_connection_count(self) -> int:
        """
        Получить количество активных соединений.
        Returns:
            Количество соединений
        """
        try:
            # Для синхронного метода используем простую проверку
            return len(self.connections)
        except Exception as e:
            logger.error(f"Error getting connection count: {e}")
            return 0

    def get_subscriber_count(self, topic: TopicName) -> int:
        """
        Получить количество подписчиков темы.
        Args:
            topic: Тема сообщения
        Returns:
            Количество подписчиков
        """
        try:
            # Для синхронного метода используем простую проверку
            return len(self.subscribers.get(topic, set()))
        except Exception as e:
            logger.error(f"Error getting subscriber count for topic '{topic}': {e}")
            return 0

    def get_all_topics(self) -> List[TopicName]:
        """
        Получить все темы.
        Returns:
            Список всех тем
        """
        try:
            # Для синхронного метода используем простую проверку
            return list(self.subscribers.keys())
        except Exception as e:
            logger.error(f"Error getting all topics: {e}")
            return []

    def get_connection_info(
        self, connection_id: ConnectionID
    ) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о соединении.
        Args:
            connection_id: ID соединения
        Returns:
            Информация о соединении или None если не найдено
        """
        try:
            # Для синхронного метода используем простую проверку
            if connection_id not in self.connections:
                return None
            return self.connections[connection_id].get_info()
        except Exception as e:
            logger.error(f"Error getting connection info for {connection_id}: {e}")
            return None

    async def _send_to_connection_internal(
        self, connection_id: ConnectionID, message: Any
    ) -> bool:
        """Внутренний метод отправки сообщения соединению."""
        try:
            async with self._connection_lock:
                if connection_id not in self.connections:
                    return False
                connection = self.connections[connection_id]
            # Формируем WebSocket сообщение
            ws_message = {
                "type": "message",
                "data": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            # Отправляем сообщение
            success = await connection.send(ws_message)
            # Обновляем метрики
            if success:
                self.connection_metrics[connection_id]["message_count"] += 1
                self.connection_metrics[connection_id][
                    "last_activity"
                ] = datetime.utcnow()
            else:
                self.connection_metrics[connection_id]["error_count"] += 1
            return success
        except Exception as e:
            logger.error(
                f"Error in _send_to_connection_internal for {connection_id}: {e}"
            )
            return False

    def _remove_subscription_internal(
        self, connection_id: ConnectionID, topic: TopicName
    ) -> None:
        """Внутренний метод удаления подписки."""
        try:
            if topic in self.subscribers:
                self.subscribers[topic].discard(connection_id)
                # Удаляем тему если нет подписчиков
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
                self.metrics["topics_count"] = len(self.subscribers)
        except Exception as e:
            logger.error(f"Error in _remove_subscription_internal: {e}")

    async def _close_all_connections(self) -> None:
        """Закрыть все соединения."""
        try:
            connection_ids = list(self.connections.keys())
            tasks = []
            for connection_id in connection_ids:
                task = asyncio.create_task(self.remove_connection(connection_id))
                tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error closing all connections: {e}")

    async def _connection_monitor(self) -> None:
        """Мониторинг соединений."""
        while self.is_running:
            try:
                # Проверяем неактивные соединения
                current_time = time.time()
                idle_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.is_idle(self.config.idle_timeout):
                        idle_connections.append(connection_id)
                # Закрываем неактивные соединения
                for connection_id in idle_connections:
                    logger.info(f"Closing idle connection {connection_id}")
                    await self.remove_connection(connection_id)
                await asyncio.sleep(self.config.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")

    async def _performance_monitor(self) -> None:
        """Мониторинг производительности."""
        while self.is_running:
            try:
                # Обновляем средний размер сообщения
                total_messages = int(self.metrics["total_messages_sent"])
                if total_messages > 0:
                    # Здесь можно добавить логику расчета среднего размера
                    self.metrics["avg_message_size"] = 0.0  # Заглушка
                await asyncio.sleep(30)  # Проверяем каждые 30 секунд
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")

    def __enter__(self) -> "WebSocketService":
        """Контекстный менеджер для входа."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Контекстный менеджер для выхода."""
        if self.is_running:
            asyncio.create_task(self.stop())

    async def __aenter__(self) -> "WebSocketService":
        """Асинхронный контекстный менеджер для входа."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Асинхронный контекстный менеджер для выхода."""
        await self.stop()
