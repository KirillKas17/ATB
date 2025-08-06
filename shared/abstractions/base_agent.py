"""
Базовые абстракции для агентов.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class AgentConfig:
    """Конфигурация агента."""

    name: str
    agent_type: str
    enabled: bool = True
    priority: int = 0
    update_interval: float = 1.0  # секунды
    max_concurrent_tasks: int = 10
    timeout: float = 30.0  # секунды
    retry_attempts: int = 3
    retry_delay: float = 1.0  # секунды
    log_level: str = "INFO"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Состояние агента."""

    is_initialized: bool = False
    is_running: bool = False
    is_healthy: bool = True
    last_update: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_tasks: int = 0
    queue_size: int = 0
    uptime: float = 0.0  # Добавляем атрибут uptime


@dataclass
class AgentMetrics:
    """Метрики агента."""

    total_decisions: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    uptime: float = 0.0
    last_decision_time: Optional[datetime] = None
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


class AgentProtocol(Protocol):
    """Протокол для всех агентов."""

    async def initialize(self) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def process(self, data: Any) -> Any: ...
    async def adapt(self) -> bool: ...
    async def learn(self) -> bool: ...
    async def evolve(self) -> bool: ...
    def get_config(self) -> AgentConfig: ...
    def get_state(self) -> AgentState: ...
    def get_metrics(self) -> AgentMetrics: ...
    def is_healthy(self) -> bool: ...
    def reset(self) -> None: ...
class BaseAgent(ABC):
    """
    Базовый класс для всех агентов.
    Предоставляет общую функциональность для всех агентов:
    - Инициализация и управление жизненным циклом
    - Обработка ошибок и восстановление
    - Сбор метрик и мониторинг
    - Асинхронная обработка задач
    - Конфигурация и состояние
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Инициализация базового агента.
        Args:
            config: Конфигурация агента
        """
        self.config = config
        self.state = AgentState()
        self.metrics = AgentMetrics()
        # Асинхронные компоненты
        self._lock = asyncio.Lock()
        self._task_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_concurrent_tasks)
        self._running_tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Время начала работы
        self._start_time = datetime.now()
        # Логгер будет настроен в подклассах
        self.logger = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        Инициализация агента.
        Должен быть реализован в подклассах для специфичной инициализации.
        """
        pass

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        Обработка данных.
        Args:
            data: Входные данные для обработки
        Returns:
            Any: Результат обработки
        Raises:
            Exception: При ошибке обработки
        """
        pass

    async def start(self) -> None:
        """Запуск агента."""
        try:
            async with self._lock:
                if self.state.is_running:
                    return
                # Инициализация если еще не выполнена
                if not self.state.is_initialized:
                    await self.initialize()
                    self.state.is_initialized = True
                # Сброс состояния
                self._stop_event.clear()
                self.state.is_running = True
                self.state.last_update = datetime.now()
                # Запуск обработчика задач
                asyncio.create_task(self._task_processor())
                if self.logger:
                    self.logger.info(f"Agent {self.config.name} started")
        except Exception as e:
            self.state.is_healthy = False
            self.state.last_error = str(e)
            self.state.error_count += 1
            if self.logger:
                self.logger.error(f"Failed to start agent {self.config.name}: {e}")
            raise

    async def stop(self) -> None:
        """Остановка агента."""
        try:
            async with self._lock:
                if not self.state.is_running:
                    return
                # Сигнал остановки
                self._stop_event.set()
                self.state.is_running = False
                # Ожидание завершения всех задач
                if self._running_tasks:
                    await asyncio.gather(*self._running_tasks, return_exceptions=True)
                    self._running_tasks.clear()
                # Очистка очереди
                while not self._task_queue.empty():
                    try:
                        self._task_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if self.logger:
                    self.logger.info(f"Agent {self.config.name} stopped")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping agent {self.config.name}: {e}")
            raise

    async def _task_processor(self) -> None:
        """Обработчик задач из очереди."""
        while not self._stop_event.is_set():
            try:
                # Получение задачи из очереди с таймаутом
                try:
                    task_data = await asyncio.wait_for(
                        self._task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                # Создание задачи обработки
                task = asyncio.create_task(self._process_task(task_data))
                self._running_tasks.append(task)
                # Очистка завершенных задач
                self._running_tasks = [t for t in self._running_tasks if not t.done()]
                # Обновление состояния
                self.state.active_tasks = len(self._running_tasks)
                self.state.queue_size = self._task_queue.qsize()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(0.1)

    async def _process_task(self, data: Any) -> None:
        """Обработка отдельной задачи."""
        start_time = datetime.now()
        try:
            # Обработка данных
            result = await self.process(data)
            # Обновление метрик
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_success_metrics(processing_time)
            if self.logger:
                self.logger.debug(
                    f"Task processed successfully in {processing_time:.3f}s"
                )
        except Exception as e:
            # Обновление метрик ошибок
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_error_metrics(str(e), processing_time)
            if self.logger:
                self.logger.error(f"Task processing failed: {e}")
            # Повторные попытки если настроено
            if self.config.retry_attempts > 0:
                await self._retry_task(data)

    async def _retry_task(self, data: Any) -> None:
        """Повторная попытка обработки задачи."""
        for attempt in range(self.config.retry_attempts):
            try:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                await self.process(data)
                if self.logger:
                    self.logger.info(f"Task retry successful on attempt {attempt + 1}")
                return
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Task retry {attempt + 1} failed: {e}")
        if self.logger:
            self.logger.error(f"All retry attempts failed for task")

    def _update_success_metrics(self, processing_time: float) -> None:
        """Обновление метрик успешной обработки."""
        self.metrics.successful_decisions += 1
        self.metrics.total_decisions += 1
        self.metrics.last_decision_time = datetime.now()
        self.metrics.average_processing_time += processing_time
        self.metrics.average_processing_time = (
            self.metrics.average_processing_time / self.metrics.total_decisions
        )
        self.state.success_count += 1
        self.state.last_update = datetime.now()
        self.state.is_healthy = True

    def _update_error_metrics(self, error: str, processing_time: float) -> None:
        """Обновление метрик ошибок."""
        self.metrics.failed_decisions += 1
        self.metrics.total_decisions += 1
        self.metrics.average_processing_time += processing_time
        self.state.error_count += 1
        self.state.last_error = error
        self.state.last_update = datetime.now()
        # Проверка здоровья агента
        if self.state.error_count > 10:
            self.state.is_healthy = False

    async def adapt(self) -> bool:
        """
        Адаптация агента к изменениям.
        Returns:
            bool: True если адаптация прошла успешно
        """
        try:
            # Базовая реализация - может быть переопределена в подклассах
            if self.logger:
                self.logger.info(f"Agent {self.config.name} adapting...")
            # Сброс метрик для новой адаптации
            self.metrics = AgentMetrics()
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Adaptation failed for agent {self.config.name}: {e}"
                )
            return False

    async def learn(self) -> bool:
        """
        Обучение агента.
        Returns:
            bool: True если обучение прошло успешно
        """
        try:
            # Базовая реализация - должна быть переопределена в подклассах
            if self.logger:
                self.logger.info(f"Agent {self.config.name} learning...")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Learning failed for agent {self.config.name}: {e}")
            return False

    async def evolve(self) -> bool:
        """
        Эволюция агента.
        Returns:
            bool: True если эволюция прошла успешно
        """
        try:
            # Базовая реализация - должна быть переопределена в подклассах
            if self.logger:
                self.logger.info(f"Agent {self.config.name} evolving...")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Evolution failed for agent {self.config.name}: {e}")
            return False

    def get_config(self) -> AgentConfig:
        """Получение конфигурации агента."""
        return self.config

    def get_state(self) -> AgentState:
        """Получение состояния агента."""
        # Обновление времени работы
        if self._start_time:
            self.state.uptime = (datetime.now() - self._start_time).total_seconds()
        return self.state

    def get_metrics(self) -> AgentMetrics:
        """Получение метрик агента."""
        # Обновление времени работы
        if self._start_time:
            self.metrics.uptime = (datetime.now() - self._start_time).total_seconds()
        return self.metrics

    def is_healthy(self) -> bool:
        """Проверка здоровья агента."""
        return (
            self.state.is_healthy
            and self.state.is_running
            and self.state.error_count < 10
            and self.metrics.failed_decisions / max(self.metrics.total_decisions, 1)
            < 0.5
        )

    def reset(self) -> None:
        """Сброс состояния агента."""
        self.state = AgentState()
        self.metrics = AgentMetrics()
        self._start_time = datetime.now()
        if self.logger:
            self.logger.info(f"Agent {self.config.name} reset")

    async def submit_task(self, data: Any) -> None:
        """
        Отправка задачи в очередь обработки.
        Args:
            data: Данные для обработки
        """
        try:
            await self._task_queue.put(data)
        except asyncio.QueueFull:
            if self.logger:
                self.logger.warning(f"Task queue full for agent {self.config.name}")
            raise

    def __str__(self) -> str:
        """Строковое представление."""
        return f"BaseAgent({self.config.name}, type={self.config.agent_type})"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return (
            f"BaseAgent(name='{self.config.name}', type='{self.config.agent_type}', "
            f"enabled={self.config.enabled}, running={self.state.is_running}, "
            f"healthy={self.state.is_healthy})"
        )
