"""
Профессиональная реализация MessageQueue для мессенджинга.
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
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

from loguru import logger

from domain.types.messaging_types import (
    EventMetadata,
    EventPriority,
    HandlerConfig,
    HandlerID,
    Message,
    MessageHandler,
    MessageHandlerInfo,
    MessageID,
    MessagePriority,
    MessageQueueConfig,
    MessageQueueError,
    MessageQueueMetrics,
    MessageQueueProtocol,
    TopicName,
    create_handler_config,
    create_message,
)


class MessageQueue(MessageQueueProtocol):
    """
    Профессиональная реализация MessageQueue с асинхронностью и приоритетными очередями.
    Реализует MessageQueueProtocol и обеспечивает:
    - Асинхронную обработку сообщений
    - Очереди с приоритетами
    - Мониторинг производительности
    - Обработку ошибок и retry
    - Потокобезопасность
    """

    def __init__(self, config: Optional[MessageQueueConfig] = None):
        """
        Инициализация MessageQueue.
        Args:
            config: Конфигурация MessageQueue
        """
        self.config = config or MessageQueueConfig()
        # Очереди сообщений по темам и приоритетам
        self.queues: Dict[TopicName, Dict[MessagePriority, deque]] = defaultdict(
            lambda: {
                priority: deque(maxlen=self.config.max_size)
                for priority in MessagePriority
            }
        )
        # Обработчики сообщений
        self.handlers: Dict[TopicName, Dict[HandlerID, MessageHandlerInfo]] = (
            defaultdict(dict)
        )
        # Пул потоков для синхронных обработчиков
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Флаги состояния
        self.is_running = False
        self.is_shutdown = False
        self.start_time: Optional[datetime] = None
        # Блокировки для потокобезопасности
        self._lock = threading.RLock()
        self._handler_lock = threading.RLock()
        self._queue_lock = threading.RLock()
        # Метрики производительности
        self.metrics: Dict[str, Union[int, float]] = {
            "total_messages_published": 0,
            "total_messages_processed": 0,
            "total_errors": 0,
            "messages_per_second": 0.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "total_queue_size": 0,
            "active_handlers": 0,
            "total_handlers": 0,
            "topics_count": 0,
            "uptime_seconds": 0.0,
        }
        # Дополнительные метрики производительности обработчиков
        self.handler_metrics: Dict[
            str, Dict[str, Union[int, float, Optional[datetime]]]
        ] = defaultdict(
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
        logger.info(f"MessageQueue initialized with config: {self.config}")

    async def start(self) -> None:
        """Запуск MessageQueue."""
        if self.is_running:
            logger.warning("MessageQueue is already running")
            return
        self.is_running = True
        self.is_shutdown = False
        self.start_time = datetime.utcnow()
        # Запускаем задачи мониторинга
        if self.config.enable_metrics:
            self._monitor_tasks.add(asyncio.create_task(self._message_processor()))
            self._monitor_tasks.add(asyncio.create_task(self._queue_monitor()))
            self._monitor_tasks.add(asyncio.create_task(self._performance_monitor()))
        logger.info("MessageQueue started successfully")

    async def stop(self) -> None:
        """Остановка MessageQueue."""
        if not self.is_running:
            logger.warning("MessageQueue is not running")
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
        logger.info("MessageQueue stopped successfully")

    async def publish(
        self,
        topic: TopicName,
        data: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[EventMetadata] = None,
    ) -> bool:
        """
        Публикация сообщения в очередь.
        Args:
            topic: Тема сообщения
            data: Данные сообщения
            priority: Приоритет сообщения
            metadata: Метаданные события
        Returns:
            True если сообщение опубликовано успешно
        """
        try:
            # Создаем сообщение
            message = Message(
                id=MessageID(str(uuid.uuid4())),
                topic=topic,
                data=data,
                priority=priority,
                metadata=metadata or EventMetadata(source="message_queue"),
                timestamp=datetime.utcnow(),
            )
            # Проверяем истечение срока действия
            if message.is_expired():
                logger.warning(f"Message {message.id} is expired, skipping")
                return False
            # Добавляем в очередь по приоритету
            with self._queue_lock:
                self.queues[topic][priority].append(message)
                self.metrics["total_messages_published"] = (
                    int(self.metrics["total_messages_published"]) + 1
                )
                self._update_queue_metrics()
            logger.debug(
                f"Published message {message.id} to topic '{topic}' with priority {priority}"
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing message to topic '{topic}': {e}")
            self.metrics["total_errors"] = int(self.metrics["total_errors"]) + 1
            return False

    async def subscribe(
        self,
        topic: TopicName,
        handler: MessageHandler,
        config: Optional[HandlerConfig] = None,
    ) -> HandlerID:
        """
        Подписка на тему.
        Args:
            topic: Тема для подписки
            handler: Обработчик сообщений
            config: Конфигурация обработчика
        Returns:
            ID обработчика
        Raises:
            MessageQueueError: При ошибке подписки
        """
        try:
            if not callable(handler):
                raise MessageQueueError("Handler must be callable")
            # Создаем конфигурацию по умолчанию если не предоставлена
            if config is None:
                config = create_handler_config(
                    name=f"handler_{topic}", priority=EventPriority.NORMAL
                )
            # Проверяем, является ли обработчик асинхронным
            is_async = asyncio.iscoroutinefunction(handler)
            # Создаем информацию об обработчике
            handler_info = MessageHandlerInfo(
                config=config, handler=handler, is_async=is_async
            )
            with self._handler_lock:
                self.handlers[topic][config.id] = handler_info
                self.metrics["total_handlers"] = sum(
                    len(handlers) for handlers in self.handlers.values()
                )
                self.metrics["topics_count"] = len(self.handlers)
            logger.debug(f"Subscribed to topic '{topic}' with handler ID {config.id}")
            return config.id
        except Exception as e:
            logger.error(f"Error subscribing to topic '{topic}': {e}")
            raise MessageQueueError(f"Failed to subscribe to topic '{topic}': {e}")

    async def unsubscribe(self, topic: TopicName, handler_id: HandlerID) -> bool:
        """
        Отписка от темы.
        Args:
            topic: Тема для отписки
            handler_id: ID обработчика
        Returns:
            True если отписка успешна
        """
        try:
            with self._handler_lock:
                if topic in self.handlers and handler_id in self.handlers[topic]:
                    del self.handlers[topic][handler_id]
                    # Удаляем тему если нет обработчиков
                    if not self.handlers[topic]:
                        del self.handlers[topic]
                    self.metrics["total_handlers"] = sum(
                        len(handlers) for handlers in self.handlers.values()
                    )
                    self.metrics["topics_count"] = len(self.handlers)
                    logger.debug(
                        f"Unsubscribed from topic '{topic}' with handler ID {handler_id}"
                    )
                    return True
                logger.warning(f"Handler {handler_id} not found for topic '{topic}'")
                return False
        except Exception as e:
            logger.error(f"Error unsubscribing from topic '{topic}': {e}")
            return False

    async def get_message(self, topic: TopicName) -> Optional[Message]:
        """
        Получить сообщение из очереди.
        Args:
            topic: Тема сообщения
        Returns:
            Сообщение или None если очередь пуста
        """
        try:
            with self._queue_lock:
                if topic not in self.queues:
                    return None
                # Получаем сообщение с наивысшим приоритетом
                for priority in reversed(list(MessagePriority)):
                    queue = self.queues[topic][priority]
                    if queue:
                        try:
                            message = queue.popleft()
                            self._update_queue_metrics()
                            assert isinstance(message, Message)
                            return message
                        except IndexError:
                            continue
                return None
        except Exception as e:
            logger.error(f"Error getting message from topic '{topic}': {e}")
            return None

    async def peek_message(self, topic: TopicName) -> Optional[Message]:
        """
        Посмотреть сообщение без извлечения.
        Args:
            topic: Тема сообщения
        Returns:
            Сообщение или None если очередь пуста
        """
        try:
            with self._queue_lock:
                if topic not in self.queues:
                    return None
                # Получаем сообщение с наивысшим приоритетом
                for priority in reversed(list(MessagePriority)):
                    queue = self.queues[topic][priority]
                    if queue:
                        try:
                            message = queue[0]
                            assert isinstance(message, Message)
                            return message
                        except IndexError:
                            continue
                return None
        except Exception as e:
            logger.error(f"Error peeking message from topic '{topic}': {e}")
            return None

    def get_queue_size(self, topic: TopicName) -> int:
        """
        Получить размер очереди для темы.
        Args:
            topic: Тема сообщения
        Returns:
            Размер очереди
        """
        try:
            with self._queue_lock:
                if topic not in self.queues:
                    return 0
                return sum(len(queue) for queue in self.queues[topic].values())
        except Exception as e:
            logger.error(f"Error getting queue size for topic '{topic}': {e}")
            return 0

    def get_all_topics(self) -> List[TopicName]:
        """
        Получить все темы.
        Returns:
            Список всех тем
        """
        try:
            with self._queue_lock:
                return list(self.queues.keys())
        except Exception as e:
            logger.error(f"Error getting all topics: {e}")
            return []

    def get_handlers(self, topic: TopicName) -> List[MessageHandlerInfo]:
        """
        Получить обработчики темы.
        Args:
            topic: Тема сообщения
        Returns:
            Список информации об обработчиках
        """
        try:
            with self._handler_lock:
                return list(self.handlers.get(topic, {}).values())
        except Exception as e:
            logger.error(f"Error getting handlers for topic '{topic}': {e}")
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Получить метрики производительности.
        Returns:
            Словарь с метриками
        """
        try:
            # Обновляем время работы
            if self.start_time:
                self.metrics["uptime_seconds"] = (
                    datetime.utcnow() - self.start_time
                ).total_seconds()
            # Обновляем количество активных обработчиков
            self.metrics["active_handlers"] = sum(
                len(handlers) for handlers in self.handlers.values()
            )
            return dict(self.metrics)
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def _message_processor(self) -> None:
        """Основной процессор сообщений."""
        while self.is_running:
            try:
                # Обрабатываем сообщения для всех тем
                for topic in list(self.handlers.keys()):
                    message = await self.get_message(topic)
                    if message is not None:
                        await self._process_message(topic, message)
                await asyncio.sleep(0.001)  # Небольшая пауза
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                self.metrics["total_errors"] += 1

    async def _process_message(self, topic: TopicName, message: Message) -> None:
        """Обработка сообщения."""
        try:
            start_time = time.time()
            # Получаем обработчики для темы
            handlers = self.get_handlers(topic)
            if not handlers:
                logger.debug(f"No handlers found for topic '{topic}'")
                return
            # Обрабатываем асинхронные и синхронные обработчики отдельно
            async_handlers = [h for h in handlers if h.is_async]
            sync_handlers = [h for h in handlers if not h.is_async]
            # Запускаем асинхронные обработчики
            if async_handlers:
                await self._execute_async_handlers(message, async_handlers)
            # Запускаем синхронные обработчики в пуле потоков
            if sync_handlers:
                await self._execute_sync_handlers(message, sync_handlers)
            # Обновляем метрики
            processing_time = time.time() - start_time
            self.metrics["total_messages_processed"] += 1
            self.metrics["total_processing_time"] += processing_time
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / self.metrics["total_messages_processed"]
            )
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            self.metrics["total_errors"] += 1

    async def _execute_async_handlers(
        self, message: Message, handlers: List[MessageHandlerInfo]
    ) -> None:
        """Выполнение асинхронных обработчиков."""
        tasks = []
        for handler_info in handlers:
            if not handler_info.config.enabled:
                continue
            task = asyncio.create_task(
                self._execute_async_handler(message, handler_info)
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_async_handler(
        self, message: Message, handler_info: MessageHandlerInfo
    ) -> None:
        """Выполнение одного асинхронного обработчика."""
        start_time = time.time()
        try:
            # Проверяем таймаут
            handler_result = handler_info.handler(message)
            if handler_result is not None:
                await asyncio.wait_for(handler_result, timeout=handler_info.config.timeout)
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

    async def _execute_sync_handlers(
        self, message: Message, handlers: List[MessageHandlerInfo]
    ) -> None:
        """Выполнение синхронных обработчиков в пуле потоков."""
        tasks = []
        for handler_info in handlers:
            if not handler_info.config.enabled:
                continue
            task = asyncio.create_task(
                self._execute_sync_handler(message, handler_info)
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_sync_handler(
        self, message: Message, handler_info: MessageHandlerInfo
    ) -> None:
        """Выполнение одного синхронного обработчика в пуле потоков."""
        start_time = time.time()
        try:
            # Выполняем в пуле потоков
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self.executor, handler_info.handler, message),
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

    def _update_handler_metrics(
        self, handler_info: MessageHandlerInfo, execution_time: float, is_error: bool
    ) -> None:
        """Обновление метрик обработчика."""
        try:
            handler_id = handler_info.config.id
            # Обновляем метрики обработчика
            handler_info.last_execution = datetime.utcnow()
            if is_error:
                handler_info.error_count += 1
            else:
                handler_info.success_count += 1
            # Обновляем глобальные метрики
            metrics = self.handler_metrics[str(handler_id)]
            call_count = metrics["call_count"]
            if isinstance(call_count, (int, float)):
                metrics["call_count"] = call_count + 1
            else:
                metrics["call_count"] = 1
            
            total_time = metrics["total_time"]
            if isinstance(total_time, (int, float)):
                metrics["total_time"] = total_time + execution_time
            else:
                metrics["total_time"] = execution_time
            
            # Обновляем среднее время
            new_call_count = metrics["call_count"]
            new_total_time = metrics["total_time"]
            if isinstance(new_call_count, (int, float)) and isinstance(new_total_time, (int, float)) and new_call_count > 0:
                metrics["avg_time"] = new_total_time / new_call_count
            
            if is_error:
                error_count = metrics["error_count"]
                if isinstance(error_count, (int, float)):
                    metrics["error_count"] = error_count + 1
                else:
                    metrics["error_count"] = 1
            
            metrics["last_execution"] = handler_info.last_execution
        except Exception as e:
            logger.error(f"Error updating handler metrics: {e}")

    def _update_queue_metrics(self) -> None:
        """Обновление метрик очереди."""
        try:
            total_size = 0
            for topic_queues in self.queues.values():
                for queue in topic_queues.values():
                    total_size += len(queue)
            self.metrics["total_queue_size"] = total_size
        except Exception as e:
            logger.error(f"Error updating queue metrics: {e}")

    async def _queue_monitor(self) -> None:
        """Мониторинг очередей."""
        while self.is_running:
            try:
                # Обновляем размеры очередей
                total_size = 0
                for topic_queues in self.queues.values():
                    for queue in topic_queues.values():
                        total_size += len(queue)
                self.metrics["total_queue_size"] = total_size
                # Проверяем переполнение очередей
                for topic, topic_queues in self.queues.items():
                    topic_size = sum(len(queue) for queue in topic_queues.values())
                    if topic_size > self.config.max_size * 0.8:
                        logger.warning(
                            f"Queue for topic '{topic}' is 80% full ({topic_size}/{self.config.max_size})"
                        )
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue monitor: {e}")

    async def _performance_monitor(self) -> None:
        """Мониторинг производительности."""
        while self.is_running:
            try:
                # Обновляем сообщения в секунду
                total_processing_time = self.metrics.get("total_processing_time", 0.0)
                if isinstance(total_processing_time, (int, float)) and total_processing_time > 0:
                    total_messages = self.metrics.get("total_messages_processed", 0)
                    if isinstance(total_messages, (int, float)):
                        self.metrics["messages_per_second"] = total_messages / (
                            total_processing_time / 60
                        )  # за минуту
                # Обновляем среднее время обработки
                total_messages = self.metrics.get("total_messages_processed", 0)
                if isinstance(total_messages, (int, float)) and total_messages > 0:
                    total_processing_time = self.metrics.get("total_processing_time", 0.0)
                    if isinstance(total_processing_time, (int, float)):
                        self.metrics["avg_processing_time"] = total_processing_time / total_messages
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")

    def __enter__(self) -> "MessageQueue":
        """Контекстный менеджер для входа."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Контекстный менеджер для выхода."""
        if self.is_running:
            asyncio.create_task(self.stop())

    async def __aenter__(self) -> "MessageQueue":
        """Асинхронный контекстный менеджер для входа."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Асинхронный контекстный менеджер для выхода."""
        await self.stop()
