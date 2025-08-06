"""
Dependency Injection Container для приложения.
"""

from typing import Any, Callable, Dict, TypeVar, Type, Optional, cast
import logging

T = TypeVar('T')

logger: logging.Logger = logging.getLogger(__name__)


class DIContainer:
    """Контейнер для dependency injection."""
    
    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Регистрация singleton сервиса."""
        key = self._get_key(service_type)
        self._singletons[key] = instance
        logger.debug(f"Registered singleton: {key}")
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Регистрация factory для сервиса."""
        key = self._get_key(service_type)
        self._factories[key] = factory
        logger.debug(f"Registered factory: {key}")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Регистрация конкретного экземпляра."""
        key = self._get_key(service_type)
        self._services[key] = instance
        logger.debug(f"Registered instance: {key}")
    
    def get(self, service_type: Type[T]) -> Optional[T]:
        """Получение сервиса из контейнера."""
        key = self._get_key(service_type)
        
        # Проверяем singleton
        if key in self._singletons:
            return cast(T, self._singletons[key])
        
        # Проверяем зарегистрированные экземпляры
        if key in self._services:
            return cast(T, self._services[key])
        
        # Проверяем factories
        if key in self._factories:
            try:
                instance = self._factories[key]()
                # Сохраняем как singleton если это factory
                self._singletons[key] = instance
                return cast(T, instance)
            except Exception as e:
                logger.error(f"Error creating instance for {key}: {e}")
                return None
        
        logger.warning(f"Service not found: {key}")
        return None
    
    def resolve(self, service_type: Type[T]) -> T:
        """Получение сервиса с обязательным результатом."""
        service = self.get(service_type)
        if service is None:
            raise ValueError(f"Service not registered: {service_type}")
        return service
    
    def _get_key(self, service_type: Type) -> str:
        """Получение ключа для типа сервиса."""
        return f"{service_type.__module__}.{service_type.__name__}"
    
    def clear(self) -> None:
        """Очистка контейнера."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        logger.debug("DI Container cleared")


# Глобальный экземпляр контейнера
container: DIContainer = DIContainer()


def get_container() -> DIContainer:
    """Получение глобального контейнера."""
    return container