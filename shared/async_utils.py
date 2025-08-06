"""
Утилиты для устранения race conditions в асинхронном коде.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class AsyncLockInfo:
    """Информация о блокировке."""

    resource_id: str
    acquired_at: datetime
    timeout: timedelta
    owner: str


class AsyncResourceManager:
    """Менеджер асинхронных ресурсов для предотвращения race conditions."""

    def __init__(self) -> None:
        """Инициализация менеджера ресурсов."""
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_info: Dict[str, AsyncLockInfo] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._logger = logger.bind(module="AsyncResourceManager")

    async def acquire_lock(
        self,
        resource_id: str,
        owner: str = "",
        timeout: timedelta = timedelta(seconds=30),
    ) -> bool:
        """
        Приобретение блокировки ресурса.
        Args:
            resource_id: Идентификатор ресурса
            owner: Владелец блокировки
            timeout: Таймаут блокировки
        Returns:
            bool: True если блокировка приобретена успешно
        """
        if resource_id not in self._locks:
            self._locks[resource_id] = asyncio.Lock()
        lock = self._locks[resource_id]
        try:
            # Попытка приобретения блокировки с таймаутом
            await asyncio.wait_for(lock.acquire(), timeout=timeout.total_seconds())
            # Запись информации о блокировке
            self._lock_info[resource_id] = AsyncLockInfo(
                resource_id=resource_id,
                acquired_at=datetime.now(),
                timeout=timeout,
                owner=owner,
            )
            self._logger.debug(
                f"Lock acquired for resource '{resource_id}' by '{owner}'"
            )
            return True
        except asyncio.TimeoutError:
            self._logger.warning(
                f"Failed to acquire lock for resource '{resource_id}' by '{owner}' (timeout)"
            )
            return False
        except Exception as e:
            self._logger.error(
                f"Error acquiring lock for resource '{resource_id}': {e}"
            )
            return False

    async def release_lock(self, resource_id: str, owner: str = "") -> bool:
        """
        Освобождение блокировки ресурса.
        Args:
            resource_id: Идентификатор ресурса
            owner: Владелец блокировки
        Returns:
            bool: True если блокировка освобождена успешно
        """
        if resource_id not in self._locks:
            self._logger.warning(f"Lock for resource '{resource_id}' does not exist")
            return False
        lock = self._locks[resource_id]
        # Проверка владельца блокировки
        if resource_id in self._lock_info:
            lock_info = self._lock_info[resource_id]
            if lock_info.owner != owner:
                self._logger.warning(
                    f"Attempt to release lock for resource '{resource_id}' by '{owner}', "
                    f"but owned by '{lock_info.owner}'"
                )
                return False
        try:
            lock.release()
            if resource_id in self._lock_info:
                del self._lock_info[resource_id]
            self._logger.debug(
                f"Lock released for resource '{resource_id}' by '{owner}'"
            )
            return True
        except Exception as e:
            self._logger.error(
                f"Error releasing lock for resource '{resource_id}': {e}"
            )
            return False

    async def acquire_semaphore(
        self, resource_id: str, max_concurrent: int = 1
    ) -> bool:
        """
        Приобретение семафора для ограничения параллельного доступа.
        Args:
            resource_id: Идентификатор ресурса
            max_concurrent: Максимальное количество параллельных операций
        Returns:
            bool: True если семафор приобретен успешно
        """
        if resource_id not in self._semaphores:
            self._semaphores[resource_id] = asyncio.Semaphore(max_concurrent)
        semaphore = self._semaphores[resource_id]
        try:
            await semaphore.acquire()
            self._logger.debug(f"Semaphore acquired for resource '{resource_id}'")
            return True
        except Exception as e:
            self._logger.error(
                f"Error acquiring semaphore for resource '{resource_id}': {e}"
            )
            return False

    async def release_semaphore(self, resource_id: str) -> bool:
        """
        Освобождение семафора.
        Args:
            resource_id: Идентификатор ресурса
        Returns:
            bool: True если семафор освобожден успешно
        """
        if resource_id not in self._semaphores:
            self._logger.warning(
                f"Semaphore for resource '{resource_id}' does not exist"
            )
            return False
        semaphore = self._semaphores[resource_id]
        try:
            semaphore.release()
            self._logger.debug(f"Semaphore released for resource '{resource_id}'")
            return True
        except Exception as e:
            self._logger.error(
                f"Error releasing semaphore for resource '{resource_id}': {e}"
            )
            return False

    @asynccontextmanager
    async def resource_lock(
        self,
        resource_id: str,
        owner: str = "",
        timeout: timedelta = timedelta(seconds=30),
    ):
        """
        Контекстный менеджер для блокировки ресурса.
        Args:
            resource_id: Идентификатор ресурса
            owner: Владелец блокировки
            timeout: Таймаут блокировки
        """
        acquired = await self.acquire_lock(resource_id, owner, timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire lock for resource '{resource_id}'")
        try:
            yield
        finally:
            await self.release_lock(resource_id, owner)

    @asynccontextmanager
    async def resource_semaphore(self, resource_id: str, max_concurrent: int = 1) -> None:
        """
        Контекстный менеджер для семафора ресурса.
        Args:
            resource_id: Идентификатор ресурса
            max_concurrent: Максимальное количество параллельных операций
        """
        acquired = await self.acquire_semaphore(resource_id, max_concurrent)
        if not acquired:
            raise RuntimeError(
                f"Failed to acquire semaphore for resource '{resource_id}'"
            )
        try:
            yield
        finally:
            await self.release_semaphore(resource_id)

    async def cleanup_expired_locks(self) -> None:
        """Очистка истекших блокировок."""
        now = datetime.now()
        expired_resources = []
        for resource_id, lock_info in self._lock_info.items():
            if now - lock_info.acquired_at > lock_info.timeout:
                expired_resources.append(resource_id)
        for resource_id in expired_resources:
            self._logger.warning(
                f"Cleaning up expired lock for resource '{resource_id}'"
            )
            await self.release_lock(resource_id, self._lock_info[resource_id].owner)

    def get_lock_status(self) -> Dict[str, Any]:
        """Получение статуса блокировок."""
        return {
            "active_locks": len(self._lock_info),
            "active_semaphores": len(self._semaphores),
            "lock_details": [
                {
                    "resource_id": info.resource_id,
                    "owner": info.owner,
                    "acquired_at": info.acquired_at.isoformat(),
                    "timeout": info.timeout.total_seconds(),
                }
                for info in self._lock_info.values()
            ],
        }


class AsyncTaskManager:
    """Менеджер асинхронных задач для предотвращения race conditions."""

    def __init__(self) -> None:
        """Инициализация менеджера задач."""
        self._tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Any] = {}
        self._task_errors: Dict[str, Exception] = {}
        self._logger = logger.bind(module="AsyncTaskManager")

    async def run_task(self, task_id: str, coro: Callable, *args, **kwargs) -> Any:
        """
        Запуск задачи с предотвращением дублирования.
        Args:
            task_id: Идентификатор задачи
            coro: Корутина для выполнения
            *args: Аргументы корутины
            **kwargs: Ключевые аргументы корутины
        Returns:
            Any: Результат выполнения задачи
        """
        # Проверка, не выполняется ли уже задача
        if task_id in self._tasks and not self._tasks[task_id].done():
            self._logger.warning(f"Task '{task_id}' is already running")
            return await self._tasks[task_id]
        # Проверка, есть ли уже результат
        if task_id in self._task_results:
            self._logger.debug(f"Returning cached result for task '{task_id}'")
            return self._task_results[task_id]
        # Проверка, была ли ошибка
        if task_id in self._task_errors:
            self._logger.warning(
                f"Task '{task_id}' previously failed: {self._task_errors[task_id]}"
            )
            raise self._task_errors[task_id]
        # Создание и запуск задачи
        task = asyncio.create_task(coro(*args, **kwargs))
        self._tasks[task_id] = task
        try:
            result = await task
            self._task_results[task_id] = result
            self._logger.debug(f"Task '{task_id}' completed successfully")
            return result
        except Exception as e:
            self._task_errors[task_id] = e
            self._logger.error(f"Task '{task_id}' failed: {e}")
            raise
        finally:
            # Очистка завершенной задачи
            if task_id in self._tasks:
                del self._tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """
        Отмена задачи.
        Args:
            task_id: Идентификатор задачи
        Returns:
            bool: True если задача отменена успешно
        """
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        del self._tasks[task_id]
        self._logger.info(f"Task '{task_id}' cancelled")
        return True

    def get_task_status(self) -> Dict[str, Any]:
        """Получение статуса задач."""
        return {
            "active_tasks": len(self._tasks),
            "completed_tasks": len(self._task_results),
            "failed_tasks": len(self._task_errors),
            "task_details": [
                {
                    "task_id": task_id,
                    "status": "running" if not task.done() else "completed",
                    "cancelled": task.cancelled(),
                }
                for task_id, task in self._tasks.items()
            ],
        }


class AsyncCache:
    """Асинхронный кэш с предотвращением race conditions."""

    def __init__(self, max_size: int = 1000, ttl: timedelta = timedelta(minutes=5)) -> None:
        """Инициализация кэша."""
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._lock = asyncio.Lock()
        self._logger = logger.bind(module="AsyncCache")

    async def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.
        Args:
            key: Ключ кэша
        Returns:
            Optional[Any]: Значение или None
        """
        async with self._lock:
            if key not in self._cache:
                return None
            # Проверка TTL
            if datetime.now() - self._timestamps[key] > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            return self._cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Установка значения в кэш.
        Args:
            key: Ключ кэша
            value: Значение
            ttl: Время жизни (если None, используется значение по умолчанию)
        """
        async with self._lock:
            # Очистка при превышении размера
            if len(self._cache) >= self._max_size:
                self._evict_oldest()
            self._cache[key] = value
            self._timestamps[key] = datetime.now()

    async def delete(self, key: str) -> bool:
        """
        Удаление значения из кэша.
        Args:
            key: Ключ кэша
        Returns:
            bool: True если значение удалено
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False

    async def clear(self) -> None:
        """Очистка кэша."""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def _evict_oldest(self) -> None:
        """Вытеснение самого старого элемента."""
        if not self._timestamps:
            return
        oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        del self._cache[oldest_key]
        del self._timestamps[oldest_key]

    async def cleanup_expired(self) -> int:
        """
        Очистка истекших элементов.
        Returns:
            int: Количество удаленных элементов
        """
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key
                for key, timestamp in self._timestamps.items()
                if now - timestamp > self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl.total_seconds(),
        }


# Глобальные экземпляры для использования в приложении
resource_manager = AsyncResourceManager()
task_manager = AsyncTaskManager()
async_cache = AsyncCache()
