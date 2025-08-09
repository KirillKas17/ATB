"""
Базовая реализация репозитория в domain слое.
Использует интерфейсы для соблюдения принципов DDD.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import uuid4

from .interfaces import IRepository, IBackend, MemoryBackend, ISessionRepository, ISessionConfigRepository

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepositoryImpl(IRepository[T], Generic[T]):
    """Базовая реализация репозитория в domain слое."""
    
    def __init__(self, backend: Optional[IBackend] = None):
        self.backend = backend or MemoryBackend()
        self._entities: Dict[str, T] = {}
        self._entity_type: Optional[str] = None
    
    async def save(self, entity: T) -> T:
        """Сохранение сущности."""
        try:
            entity_id = self._get_entity_id(entity)
            if not entity_id:
                entity_id = str(uuid4())
                self._set_entity_id(entity, entity_id)
            
            # Сохраняем в локальное хранилище
            self._entities[entity_id] = entity
            
            # Сохраняем в бэкенд
            await self.backend.save(f"{self._get_entity_type()}:{entity_id}", entity)
            
            logger.debug(f"Saved entity {entity_id} of type {self._get_entity_type()}")
            return entity
            
        except Exception as e:
            logger.error(f"Error saving entity: {e}")
            raise
    
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Поиск сущности по ID."""
        try:
            # Сначала ищем в локальном хранилище
            if entity_id in self._entities:
                return self._entities[entity_id]
            
            # Затем ищем в бэкенде
            entity = await self.backend.get(f"{self._get_entity_type()}:{entity_id}")
            if entity:
                self._entities[entity_id] = entity
                return entity
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding entity {entity_id}: {e}")
            return None
    
    async def find_all(self) -> List[T]:
        """Получение всех сущностей."""
        try:
            # Получаем все ключи для данного типа сущности
            pattern = f"{self._get_entity_type()}:*"
            keys = await self.backend.list_keys(pattern)
            
            entities = []
            for key in keys:
                entity_id = key.split(":", 1)[1]
                entity = await self.find_by_id(entity_id)
                if entity:
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error finding all entities: {e}")
            return []
    
    async def delete(self, entity_id: str) -> bool:
        """Удаление сущности по ID."""
        try:
            # Удаляем из локального хранилища
            if entity_id in self._entities:
                del self._entities[entity_id]
            
            # Удаляем из бэкенда
            success = await self.backend.delete(f"{self._get_entity_type()}:{entity_id}")
            
            if success:
                logger.debug(f"Deleted entity {entity_id} of type {self._get_entity_type()}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting entity {entity_id}: {e}")
            return False
    
    async def exists(self, entity_id: str) -> bool:
        """Проверка существования сущности."""
        try:
            # Проверяем в локальном хранилище
            if entity_id in self._entities:
                return True
            
            # Проверяем в бэкенде
            return await self.backend.exists(f"{self._get_entity_type()}:{entity_id}")
            
        except Exception as e:
            logger.error(f"Error checking existence of entity {entity_id}: {e}")
            return False
    
    async def count(self) -> int:
        """Подсчет количества сущностей."""
        try:
            pattern = f"{self._get_entity_type()}:*"
            keys = await self.backend.list_keys(pattern)
            return len(keys)
            
        except Exception as e:
            logger.error(f"Error counting entities: {e}")
            return 0
    
    @asynccontextmanager
    async def transaction(self):
        """Контекстный менеджер для транзакций."""
        # Простая реализация транзакций
        try:
            # Создаем снапшот текущего состояния
            snapshot = self._entities.copy()
            yield self
        except Exception as e:
            # Откатываем изменения
            self._entities = snapshot
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            # Фиксируем изменения
            pass
    
    def _get_entity_id(self, entity: T) -> Optional[str]:
        """Получение ID сущности."""
        if hasattr(entity, 'id'):
            return str(getattr(entity, 'id'))
        elif hasattr(entity, 'entity_id'):
            return str(getattr(entity, 'entity_id'))
        return None
    
    def _set_entity_id(self, entity: T, entity_id: str) -> None:
        """Установка ID сущности."""
        if hasattr(entity, 'id'):
            setattr(entity, 'id', entity_id)
        elif hasattr(entity, 'entity_id'):
            setattr(entity, 'entity_id', entity_id)
    
    def _get_entity_type(self) -> str:
        """Получение типа сущности."""
        if self._entity_type:
            return self._entity_type
        
        # Определяем тип по имени класса
        entity_class = type(self).__orig_bases__[0].__args__[0]
        if hasattr(entity_class, '__name__'):
            self._entity_type = entity_class.__name__.lower()
        else:
            self._entity_type = 'entity'
        
        return self._entity_type


class SessionRepositoryImpl(BaseRepositoryImpl[Any], ISessionRepository):
    """Реализация репозитория сессий в domain слое."""
    
    def __init__(self, backend: Optional[IBackend] = None):
        super().__init__(backend)
        self._entity_type = 'session'
    
    async def find_by_session_type(self, session_type: str) -> List[Any]:
        """Поиск сессий по типу."""
        try:
            all_sessions = await self.find_all()
            return [
                session for session in all_sessions
                if hasattr(session, 'session_type') and getattr(session, 'session_type') == session_type
            ]
        except Exception as e:
            logger.error(f"Error finding sessions by type {session_type}: {e}")
            return []
    
    async def find_active_sessions(self, timestamp: Any) -> List[Any]:
        """Поиск активных сессий."""
        try:
            all_sessions = await self.find_all()
            active_sessions = []
            
            for session in all_sessions:
                if hasattr(session, 'is_active'):
                    if callable(getattr(session, 'is_active')):
                        if getattr(session, 'is_active')(timestamp):
                            active_sessions.append(session)
                    elif getattr(session, 'is_active'):
                        active_sessions.append(session)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Error finding active sessions: {e}")
            return []


class SessionConfigRepositoryImpl(BaseRepositoryImpl[Any], ISessionConfigRepository):
    """Реализация репозитория конфигураций сессий в domain слое."""
    
    def __init__(self, backend: Optional[IBackend] = None):
        super().__init__(backend)
        self._entity_type = 'session_config'
    
    async def find_by_config_type(self, config_type: str) -> List[Any]:
        """Поиск конфигураций по типу."""
        try:
            all_configs = await self.find_all()
            return [
                config for config in all_configs
                if hasattr(config, 'config_type') and getattr(config, 'config_type') == config_type
            ]
        except Exception as e:
            logger.error(f"Error finding configs by type {config_type}: {e}")
            return []
    
    async def find_by_session_type(self, session_type: str) -> Optional[Any]:
        """Поиск конфигурации по типу сессии."""
        try:
            all_configs = await self.find_all()
            for config in all_configs:
                if hasattr(config, 'session_type') and getattr(config, 'session_type') == session_type:
                    return config
            return None
            
        except Exception as e:
            logger.error(f"Error finding config by session type {session_type}: {e}")
            return None
