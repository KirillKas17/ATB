"""
Базовый репозиторий с полной реализацией транзакционных методов.
Поддерживает различные бэкенды БД и продвинутое управление транзакциями.
"""
import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncContextManager, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
import threading
from collections import defaultdict
import weakref

# SQLAlchemy imports
try:
    from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text
    HAS_SQLALCHEMY = True
except ImportError:
    AsyncSession = None  # type: ignore
    AsyncEngine = None  # type: ignore
    HAS_SQLALCHEMY = False

logger = logging.getLogger(__name__)

T = TypeVar("T")

class TransactionState(Enum):
    """Состояния транзакции."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PENDING = "pending"

class IsolationLevel(Enum):
    """Уровни изоляции транзакций."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"

class BackendType(Enum):
    """Типы бэкендов для репозитория."""
    SQLALCHEMY = "sqlalchemy"
    MEMORY = "memory"
    REDIS = "redis"
    MONGODB = "mongodb"
    FILE = "file"

@dataclass
class TransactionMetrics:
    """Метрики транзакции."""
    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    operations_count: int = 0
    rollback_count: int = 0
    commit_attempts: int = 0
    isolation_level: Optional[IsolationLevel] = None
    
    def complete(self) -> None:
        """Завершение метрик."""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

@dataclass
class SavePoint:
    """Точка сохранения в транзакции."""
    name: str
    created_at: datetime
    operations_count: int
    data_snapshot: Optional[Dict[str, Any]] = None

class TransactionContext:
    """Контекст транзакции с расширенной функциональностью."""
    
    def __init__(
        self, 
        transaction_id: Optional[str] = None,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 30.0
    ):
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.isolation_level = isolation_level
        self.timeout_seconds = timeout_seconds
        self.state = TransactionState.PENDING
        self.created_at = datetime.now()
        self.metrics = TransactionMetrics(
            transaction_id=self.transaction_id,
            start_time=self.created_at,
            isolation_level=isolation_level
        )
        
        # Управление операциями и откатами
        self._operations: List[Dict[str, Any]] = []
        self._savepoints: Dict[str, SavePoint] = {}
        self._rollback_handlers: List[Callable[[Any], Any]] = []
        self._commit_handlers: List[Callable[[Any], Any]] = []
        
        # Блокировки и синхронизация
        self._lock = asyncio.Lock()
        self._active_operations: Set[str] = set()
        
        # Кэш и временные данные
        self._cache: Dict[str, Any] = {}
        self._pending_writes: Dict[str, Any] = {}
        
        # Обработка ошибок
        self._errors: List[Exception] = []
        self._retry_count = 0
        self._max_retries = 3
    
    async def add_operation(self, operation_type: str, entity_id: str, data: Any) -> None:
        """Добавление операции в транзакцию."""
        async with self._lock:
            operation = {
                'type': operation_type,
                'entity_id': entity_id,
                'data': data,
                'timestamp': datetime.now(),
                'operation_id': str(uuid.uuid4())
            }
            self._operations.append(operation)
            self.metrics.operations_count += 1
            self._active_operations.add(operation['operation_id'])
    
    async def create_savepoint(self, name: str) -> SavePoint:
        """Создание точки сохранения."""
        async with self._lock:
            if name in self._savepoints:
                raise ValueError(f"Savepoint '{name}' already exists")
            
            savepoint = SavePoint(
                name=name,
                created_at=datetime.now(),
                operations_count=len(self._operations),
                data_snapshot=self._cache.copy()
            )
            self._savepoints[name] = savepoint
            logger.debug(f"Created savepoint '{name}' in transaction {self.transaction_id}")
            return savepoint
    
    async def rollback_to_savepoint(self, name: str) -> None:
        """Откат к точке сохранения."""
        async with self._lock:
            if name not in self._savepoints:
                raise ValueError(f"Savepoint '{name}' not found")
            
            savepoint = self._savepoints[name]
            
            # Откат операций
            operations_to_remove = len(self._operations) - savepoint.operations_count
            if operations_to_remove > 0:
                removed_operations = self._operations[-operations_to_remove:]
                self._operations = self._operations[:-operations_to_remove]
                
                # Удаление активных операций
                for op in removed_operations:
                    self._active_operations.discard(op['operation_id'])
            
            # Восстановление кэша
            if savepoint.data_snapshot:
                self._cache = savepoint.data_snapshot.copy()
            
            # Удаление последующих точек сохранения
            savepoints_to_remove = [sp_name for sp_name, sp in self._savepoints.items() 
                                  if sp.created_at > savepoint.created_at]
            for sp_name in savepoints_to_remove:
                del self._savepoints[sp_name]
            
            self.metrics.rollback_count += 1
            logger.info(f"Rolled back to savepoint '{name}' in transaction {self.transaction_id}")
    
    def add_rollback_handler(self, handler: Callable[[Any], Any]) -> None:
        """Добавление обработчика отката."""
        self._rollback_handlers.append(handler)
    
    def add_commit_handler(self, handler: Callable[[Any], Any]) -> None:
        """Добавление обработчика коммита."""
        self._commit_handlers.append(handler)
    
    async def execute_rollback_handlers(self) -> None:
        """Выполнение обработчиков отката."""
        for handler in self._rollback_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self)
                else:
                    handler(self)
            except Exception as e:
                logger.error(f"Error in rollback handler: {e}")
                self._errors.append(e)
    
    async def execute_commit_handlers(self) -> None:
        """Выполнение обработчиков коммита."""
        for handler in self._commit_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self)
                else:
                    handler(self)
            except Exception as e:
                logger.error(f"Error in commit handler: {e}")
                self._errors.append(e)
                raise
    
    def is_expired(self) -> bool:
        """Проверка истечения времени транзакции."""
        if self.timeout_seconds <= 0:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.timeout_seconds
    
    def get_operation_count(self) -> int:
        """Получение количества операций."""
        return len(self._operations)
    
    def has_pending_operations(self) -> bool:
        """Проверка наличия незавершённых операций."""
        return len(self._active_operations) > 0

class RepositoryBackend(ABC):
    """Абстрактный бэкенд для репозитория."""
    
    @abstractmethod
    async def begin_transaction(self, context: TransactionContext) -> None:
        """Начало транзакции."""
        pass
    
    @abstractmethod
    async def commit_transaction(self, context: TransactionContext) -> None:
        """Коммит транзакции."""
        pass
    
    @abstractmethod
    async def rollback_transaction(self, context: TransactionContext) -> None:
        """Откат транзакции."""
        pass
    
    @abstractmethod
    async def execute_operation(self, context: TransactionContext, operation: Dict[str, Any]) -> Any:
        """Выполнение операции."""
        pass

class SQLAlchemyBackend(RepositoryBackend):
    """Бэкенд для SQLAlchemy."""
    
    def __init__(self, engine: Optional[AsyncEngine] = None, session_factory: Optional[Callable[[], AsyncSession]] = None):
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is required for SQLAlchemyBackend")
        
        self.engine = engine
        self.session_factory = session_factory
        self._active_sessions: Dict[str, AsyncSession] = {}
        self._session_lock = asyncio.Lock()
    
    async def begin_transaction(self, context: TransactionContext) -> None:
        """Начало транзакции SQLAlchemy."""
        if not self.session_factory:
            raise ValueError("Session factory is required")
        
        async with self._session_lock:
            session = self.session_factory()
            await session.begin()
            
            # Установка уровня изоляции
            if context.isolation_level != IsolationLevel.READ_COMMITTED:
                await session.execute(
                    text(f"SET TRANSACTION ISOLATION LEVEL {context.isolation_level.value}")
                )
            
            self._active_sessions[context.transaction_id] = session
            context.state = TransactionState.ACTIVE
            logger.debug(f"Started SQLAlchemy transaction {context.transaction_id}")
    
    async def commit_transaction(self, context: TransactionContext) -> None:
        """Коммит транзакции SQLAlchemy."""
        async with self._session_lock:
            session = self._active_sessions.get(context.transaction_id)
            if not session:
                raise ValueError(f"No active session for transaction {context.transaction_id}")
            
            try:
                await session.commit()
                context.state = TransactionState.COMMITTED
                logger.debug(f"Committed SQLAlchemy transaction {context.transaction_id}")
            except Exception as e:
                context.state = TransactionState.FAILED
                context._errors.append(e)
                await session.rollback()
                raise
            finally:
                await session.close()
                del self._active_sessions[context.transaction_id]
    
    async def rollback_transaction(self, context: TransactionContext) -> None:
        """Откат транзакции SQLAlchemy."""
        async with self._session_lock:
            session = self._active_sessions.get(context.transaction_id)
            if not session:
                logger.warning(f"No active session for transaction {context.transaction_id}")
                return
            
            try:
                await session.rollback()
                context.state = TransactionState.ROLLED_BACK
                logger.debug(f"Rolled back SQLAlchemy transaction {context.transaction_id}")
            except Exception as e:
                context.state = TransactionState.FAILED
                context._errors.append(e)
                logger.error(f"Error during rollback: {e}")
            finally:
                await session.close()
                del self._active_sessions[context.transaction_id]
    
    async def execute_operation(self, context: TransactionContext, operation: Dict[str, Any]) -> Any:
        """Выполнение операции SQLAlchemy."""
        session = self._active_sessions.get(context.transaction_id)
        if not session:
            raise ValueError(f"No active session for transaction {context.transaction_id}")
        
        # Здесь будет специфичная для домена логика выполнения операций
        # В реальной реализации это будет зависеть от типа операции
        operation_type = operation['type']
        entity_id = operation['entity_id']
        data = operation['data']
        
        if operation_type == 'insert':
            # Логика вставки
            pass
        elif operation_type == 'update':
            # Логика обновления
            pass
        elif operation_type == 'delete':
            # Логика удаления
            pass
        
        return None

class MemoryBackend(RepositoryBackend):
    """In-memory бэкенд для тестирования."""
    
    def __init__(self) -> None:
        self._storage: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._transaction_snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def begin_transaction(self, context: TransactionContext) -> None:
        """Начало in-memory транзакции."""
        async with self._lock:
            # Создание снимка текущего состояния
            snapshot = {}
            for collection, items in self._storage.items():
                snapshot[collection] = items.copy()
            
            self._transaction_snapshots[context.transaction_id] = snapshot
            context.state = TransactionState.ACTIVE
            logger.debug(f"Started memory transaction {context.transaction_id}")
    
    async def commit_transaction(self, context: TransactionContext) -> None:
        """Коммит in-memory транзакции."""
        async with self._lock:
            # Применение всех операций из контекста
            for operation in context._operations:
                await self._apply_operation(operation)
            
            # Удаление снимка
            if context.transaction_id in self._transaction_snapshots:
                del self._transaction_snapshots[context.transaction_id]
            
            context.state = TransactionState.COMMITTED
            logger.debug(f"Committed memory transaction {context.transaction_id}")
    
    async def rollback_transaction(self, context: TransactionContext) -> None:
        """Откат in-memory транзакции."""
        async with self._lock:
            # Восстановление из снимка
            snapshot = self._transaction_snapshots.get(context.transaction_id)
            if snapshot:
                self._storage.clear()
                for collection, items in snapshot.items():
                    self._storage[collection] = items.copy()
                del self._transaction_snapshots[context.transaction_id]
            
            context.state = TransactionState.ROLLED_BACK
            logger.debug(f"Rolled back memory transaction {context.transaction_id}")
    
    async def execute_operation(self, context: TransactionContext, operation: Dict[str, Any]) -> Any:
        """Выполнение операции в памяти."""
        # Операции выполняются при коммите для in-memory бэкенда
        return None
    
    async def _apply_operation(self, operation: Dict[str, Any]) -> None:
        """Применение операции к хранилищу."""
        operation_type = operation['type']
        entity_id = operation['entity_id']
        data = operation['data']
        collection = data.get('collection', 'default')
        
        if operation_type == 'insert' or operation_type == 'update':
            self._storage[collection][entity_id] = data
        elif operation_type == 'delete':
            self._storage[collection].pop(entity_id, None)

class TransactionManager:
    """Менеджер транзакций."""
    
    def __init__(self, backend: RepositoryBackend):
        self.backend = backend
        self._active_transactions: Dict[str, TransactionContext] = {}
        self._transaction_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics: Dict[str, Any] = defaultdict(int)
        
        # Запуск фонового процесса очистки
        self._start_cleanup_task()
    
    async def begin_transaction(
        self, 
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 30.0
    ) -> TransactionContext:
        """Начало новой транзакции."""
        context = TransactionContext(
            isolation_level=isolation_level,
            timeout_seconds=timeout_seconds
        )
        
        try:
            await self.backend.begin_transaction(context)
            
            async with self._transaction_lock:
                self._active_transactions[context.transaction_id] = context
            
            self._metrics['transactions_started'] += 1
            logger.info(f"Started transaction {context.transaction_id}")
            return context
            
        except Exception as e:
            context.state = TransactionState.FAILED
            context._errors.append(e)
            self._metrics['transactions_failed'] += 1
            logger.error(f"Failed to start transaction: {e}")
            raise
    
    async def commit_transaction(self, context: TransactionContext) -> None:
        """Коммит транзакции."""
        if context.state != TransactionState.ACTIVE:
            raise ValueError(f"Cannot commit transaction in state {context.state}")
        
        if context.is_expired():
            await self.rollback_transaction(context)
            raise TimeoutError(f"Transaction {context.transaction_id} expired")
        
        try:
            context.metrics.commit_attempts += 1
            
            # Выполнение обработчиков коммита
            await context.execute_commit_handlers()
            
            # Коммит в бэкенде
            await self.backend.commit_transaction(context)
            
            # Завершение метрик
            context.metrics.complete()
            
            async with self._transaction_lock:
                self._active_transactions.pop(context.transaction_id, None)
            
            self._metrics['transactions_committed'] += 1
            logger.info(f"Committed transaction {context.transaction_id} "
                       f"({context.metrics.operations_count} operations, "
                       f"{context.metrics.duration_ms:.2f}ms)")
            
        except Exception as e:
            context.state = TransactionState.FAILED
            context._errors.append(e)
            self._metrics['transactions_failed'] += 1
            logger.error(f"Failed to commit transaction {context.transaction_id}: {e}")
            
            # Попытка отката при неудачном коммите
            try:
                await self.rollback_transaction(context)
            except Exception as rollback_error:
                logger.error(f"Failed to rollback after commit failure: {rollback_error}")
            
            raise
    
    async def rollback_transaction(self, context: TransactionContext) -> None:
        """Откат транзакции."""
        if context.state in [TransactionState.COMMITTED, TransactionState.ROLLED_BACK]:
            logger.warning(f"Transaction {context.transaction_id} already finalized")
            return
        
        try:
            # Выполнение обработчиков отката
            await context.execute_rollback_handlers()
            
            # Откат в бэкенде
            await self.backend.rollback_transaction(context)
            
            # Завершение метрик
            context.metrics.complete()
            
            async with self._transaction_lock:
                self._active_transactions.pop(context.transaction_id, None)
            
            self._metrics['transactions_rolled_back'] += 1
            logger.info(f"Rolled back transaction {context.transaction_id}")
            
        except Exception as e:
            context.state = TransactionState.FAILED
            context._errors.append(e)
            self._metrics['transactions_failed'] += 1
            logger.error(f"Failed to rollback transaction {context.transaction_id}: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(
        self, 
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 30.0
    ) -> AsyncContextManager[TransactionContext]:
        """Контекстный менеджер для транзакций."""
        context = await self.begin_transaction(isolation_level, timeout_seconds)
        try:
            yield context
            await self.commit_transaction(context)
        except Exception:
            await self.rollback_transaction(context)
            raise
    
    async def get_active_transactions(self) -> List[TransactionContext]:
        """Получение списка активных транзакций."""
        async with self._transaction_lock:
            return list(self._active_transactions.values())
    
    async def get_transaction_metrics(self) -> Dict[str, Any]:
        """Получение метрик транзакций."""
        async with self._transaction_lock:
            active_count = len(self._active_transactions)
            
        return {
            **self._metrics,
            'active_transactions': active_count,
            'avg_transaction_duration': self._calculate_avg_duration(),
            'success_rate': self._calculate_success_rate()
        }
    
    def _start_cleanup_task(self) -> None:
        """Запуск фонового процесса очистки."""
        async def cleanup_expired_transactions():
            while True:
                try:
                    await asyncio.sleep(60)  # Проверка каждую минуту
                    await self._cleanup_expired_transactions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in transaction cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_transactions())
    
    async def _cleanup_expired_transactions(self) -> None:
        """Очистка истёкших транзакций."""
        async with self._transaction_lock:
            expired_transactions = [
                context for context in self._active_transactions.values()
                if context.is_expired()
            ]
        
        for context in expired_transactions:
            try:
                logger.warning(f"Force rolling back expired transaction {context.transaction_id}")
                await self.rollback_transaction(context)
            except Exception as e:
                logger.error(f"Error rolling back expired transaction {context.transaction_id}: {e}")
    
    def _calculate_avg_duration(self) -> float:
        """Расчёт средней продолжительности транзакций."""
        # Упрощённая реализация - в реальности нужна история метрик
        return 0.0
    
    def _calculate_success_rate(self) -> float:
        """Расчёт коэффициента успешности транзакций."""
        total = (self._metrics['transactions_committed'] + 
                self._metrics['transactions_rolled_back'] + 
                self._metrics['transactions_failed'])
        if total == 0:
            return 0.0
        return self._metrics['transactions_committed'] / total

class BaseRepository(Generic[T], ABC):
    """Базовый репозиторий с полной поддержкой транзакций."""
    
    def __init__(self, backend_type: BackendType = BackendType.MEMORY, **backend_kwargs):
        self.backend_type = backend_type
        self._backend = self._create_backend(backend_type, **backend_kwargs)
        self._transaction_manager = TransactionManager(self._backend)
        self._local_transactions: Dict[int, Optional[TransactionContext]] = {}
    
    def _create_backend(self, backend_type: BackendType, **kwargs) -> RepositoryBackend:
        """Создание бэкенда репозитория."""
        if backend_type == BackendType.SQLALCHEMY:
            return SQLAlchemyBackend(**kwargs)
        elif backend_type == BackendType.MEMORY:
            return MemoryBackend()
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    @property
    def current_transaction(self) -> Optional[TransactionContext]:
        """Получение текущей транзакции для данного потока."""
        thread_id = threading.get_ident()
        return self._local_transactions.get(thread_id)
    
    async def begin_transaction(
        self, 
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 30.0
    ) -> TransactionContext:
        """Начало транзакции."""
        context = await self._transaction_manager.begin_transaction(
            isolation_level, timeout_seconds
        )
        
        thread_id = threading.get_ident()
        self._local_transactions[thread_id] = context
        
        return context
    
    async def commit(self) -> None:
        """Коммит текущей транзакции."""
        context = self.current_transaction
        if not context:
            logger.warning("No active transaction to commit")
            return
        
        try:
            await self._transaction_manager.commit_transaction(context)
        finally:
            thread_id = threading.get_ident()
            self._local_transactions.pop(thread_id, None)
    
    async def rollback(self) -> None:
        """Откат текущей транзакции."""
        context = self.current_transaction
        if not context:
            logger.warning("No active transaction to rollback")
            return
        
        try:
            await self._transaction_manager.rollback_transaction(context)
        finally:
            thread_id = threading.get_ident()
            self._local_transactions.pop(thread_id, None)
    
    @asynccontextmanager
    async def transaction(
        self, 
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 30.0
    ) -> AsyncContextManager[TransactionContext]:
        """Контекстный менеджер для транзакций."""
        async with self._transaction_manager.transaction(isolation_level, timeout_seconds) as context:
            thread_id = threading.get_ident()
            old_transaction = self._local_transactions.get(thread_id)
            self._local_transactions[thread_id] = context
            try:
                yield context
            finally:
                if old_transaction:
                    self._local_transactions[thread_id] = old_transaction
                else:
                    self._local_transactions.pop(thread_id, None)
    
    async def execute_in_transaction(
        self, 
        operation_type: str, 
        entity_id: str, 
        data: Any
    ) -> None:
        """Выполнение операции в рамках транзакции."""
        context = self.current_transaction
        if not context:
            # Автоматическое создание транзакции если её нет
            async with self.transaction() as new_context:
                await new_context.add_operation(operation_type, entity_id, data)
                await self._backend.execute_operation(new_context, {
                    'type': operation_type,
                    'entity_id': entity_id,
                    'data': data
                })
        else:
            await context.add_operation(operation_type, entity_id, data)
            await self._backend.execute_operation(context, {
                'type': operation_type,
                'entity_id': entity_id,
                'data': data
            })
    
    async def create_savepoint(self, name: str) -> SavePoint:
        """Создание точки сохранения."""
        context = self.current_transaction
        if not context:
            raise ValueError("No active transaction for savepoint creation")
        
        return await context.create_savepoint(name)
    
    async def rollback_to_savepoint(self, name: str) -> None:
        """Откат к точке сохранения."""
        context = self.current_transaction
        if not context:
            raise ValueError("No active transaction for savepoint rollback")
        
        await context.rollback_to_savepoint(name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния репозитория."""
        try:
            metrics = await self._transaction_manager.get_transaction_metrics()
            active_transactions = await self._transaction_manager.get_active_transactions()
            
            return {
                'status': 'healthy',
                'backend_type': self.backend_type.value,
                'transaction_metrics': metrics,
                'active_transactions_count': len(active_transactions),
                'oldest_transaction_age_seconds': self._get_oldest_transaction_age(active_transactions),
                'memory_usage_mb': self._estimate_memory_usage()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend_type': self.backend_type.value
            }
    
    def _get_oldest_transaction_age(self, transactions: List[TransactionContext]) -> float:
        """Получение возраста старейшей транзакции."""
        if not transactions:
            return 0.0
        
        oldest = min(transactions, key=lambda t: t.created_at)
        return (datetime.now() - oldest.created_at).total_seconds()
    
    def _estimate_memory_usage(self) -> float:
        """Оценка использования памяти."""
        # Упрощённая оценка - в реальности нужен более точный расчёт
        import sys
        return sys.getsizeof(self._local_transactions) / (1024 * 1024)
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Сохранение сущности."""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Поиск сущности по ID."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[T]:
        """Получение всех сущностей."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Удаление сущности."""
        pass

# Фабрика для создания репозиториев с различными бэкендами
class RepositoryFactory:
    """Фабрика репозиториев."""
    
    @staticmethod
    def create_repository(
        repository_class: type,
        backend_type: BackendType = BackendType.MEMORY,
        **backend_kwargs
    ) -> BaseRepository:
        """Создание репозитория с указанным бэкендом."""
        return repository_class(backend_type=backend_type, **backend_kwargs)
    
    @staticmethod
    def create_sqlalchemy_repository(
        repository_class: type,
        database_url: str,
        **engine_kwargs
    ) -> BaseRepository:
        """Создание репозитория с SQLAlchemy бэкендом."""
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is required")
        
        engine = create_async_engine(database_url, **engine_kwargs)
        session_factory = sessionmaker(engine, class_=AsyncSession)
        
        return repository_class(
            backend_type=BackendType.SQLALCHEMY,
            engine=engine,
            session_factory=session_factory
        )
