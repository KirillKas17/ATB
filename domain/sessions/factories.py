# -*- coding: utf-8 -*-
"""
Фабрики для создания компонентов торговых сессий.
"""

from typing import Any, Dict, List, Optional, Union, Type
from typing import cast
from .interfaces import SessionAnalyzerFactory

from loguru import logger

from domain.types.session_types import SessionType
from domain.value_objects.timestamp import Timestamp

from .interfaces import (
    BaseSessionAnalyzer,
    BaseSessionPredictor,
    BaseSessionTransitionManager,
    SessionCache,
    SessionDataValidator,
    SessionMetricsAnalyzer,
    SessionPatternRecognizer,
    SessionRegistry,
    SessionTimeCalculator,
    SessionAnalyzerFactory,
)
from .session_marker import SessionMarker
from .session_profile import SessionProfileRegistry
from .session_influence_analyzer import SessionInfluenceAnalyzer
from .implementations import (
    DefaultSessionDataValidator,
    DefaultSessionMetricsAnalyzer,
    DefaultSessionPatternRecognizer,
    DefaultSessionTimeCalculator,
    DefaultSessionTransitionManager,
    InMemorySessionCache,
    SessionPredictor,
)

import uuid


class SessionProfileRegistryAdapter(SessionRegistry):
    """Адаптер для SessionProfileRegistry к интерфейсу SessionRegistry."""

    def __init__(self, registry: SessionProfileRegistry) -> None:
        self._registry = registry

    def get_profile(self, session_type: SessionType) -> Optional[Any]:
        """Получение профиля сессии."""
        return self._registry.get_profile(session_type)

    def get_all_profiles(self) -> Dict[SessionType, Any]:
        """Получение всех профилей."""
        return self._registry.get_all_profiles()

    def register_profile(self, profile: Any) -> None:
        """Регистрация профиля."""
        self._registry.register_profile(profile)

    def get_active_sessions(self, timestamp: Timestamp) -> List[Any]:
        """Получение активных сессий."""
        return self._registry.get_active_sessions(timestamp)

    def get_primary_session(self, timestamp: Timestamp) -> Optional[Any]:
        """Получение основной сессии."""
        return self._registry.get_primary_session(timestamp)

    def get_session_overlap(self, session1: SessionType, session2: SessionType) -> float:
        """Получение перекрытия сессий."""
        return self._registry.get_session_overlap(session1, session2)

    def get_session_recommendations(self, session_type: SessionType) -> List[str]:
        """Получение рекомендаций для сессии."""
        return self._registry.get_session_recommendations(session_type)

    def get_session_statistics(self, session_type: SessionType) -> Dict[str, float]:
        """Получение статистики сессии."""
        # Привести все значения к float
        stats = self._registry.get_session_statistics(session_type)
        return {k: float(v) for k, v in stats.items()}

    def update_profile(self, session_type: SessionType, updates: Dict[str, Union[str, float, int, bool]]) -> bool:
        return self._registry.update_profile(session_type, updates)


class SessionComponentFactory:
    """Фабрика для создания компонентов сессий."""

    def __init__(self) -> None:
        self._registry: Optional[SessionRegistry] = None
        self._cache: Optional[SessionCache] = None
        self._validator: Optional[SessionDataValidator] = None

    def create_registry(self) -> SessionRegistry:
        """Создание реестра сессий."""
        if self._registry is None:
            from .session_profile import SessionProfileRegistry
            self._registry = SessionProfileRegistryAdapter(SessionProfileRegistry())
            logger.info("Created session registry")
        return self._registry

    def create_cache(self, cache_type: str = "memory") -> SessionCache:
        """Создание кэша."""
        if self._cache is None:
            if cache_type == "memory":
                self._cache = InMemorySessionCache()
            else:
                raise ValueError(f"Unsupported cache type: {cache_type}")
            logger.info(f"Created {cache_type} session cache")
        return self._cache

    def create_validator(self) -> SessionDataValidator:
        """Создание валидатора."""
        if self._validator is None:
            self._validator = DefaultSessionDataValidator()
            logger.info("Created session data validator")
        return self._validator

    def create_metrics_analyzer(self) -> SessionMetricsAnalyzer:
        """Создание анализатора метрик."""
        return DefaultSessionMetricsAnalyzer()

    def create_pattern_recognizer(self) -> SessionPatternRecognizer:
        """Создание распознавателя паттернов."""
        return DefaultSessionPatternRecognizer()

    def create_time_calculator(
        self, registry: Optional[SessionRegistry] = None
    ) -> SessionTimeCalculator:
        """Создание калькулятора времени."""
        if registry is None:
            registry = self.create_registry()
        return DefaultSessionTimeCalculator(registry)

    def create_session_marker(
        self, registry: Optional[SessionRegistry] = None
    ) -> SessionMarker:
        """Создание маркера сессий."""
        if registry is None:
            registry = self.create_registry()
        # Если registry является адаптером, получаем оригинальный реестр
        if isinstance(registry, SessionProfileRegistryAdapter):
            profile_registry = registry._registry
        else:
            # Создаем новый SessionProfileRegistry если тип не подходит
            from .session_profile import SessionProfileRegistry
            profile_registry = SessionProfileRegistry()
        return SessionMarker(profile_registry)

    def create_influence_analyzer(
        self,
        registry: Optional[SessionRegistry] = None,
        session_marker: Optional[SessionMarker] = None,
    ) -> SessionInfluenceAnalyzer:
        """Создание анализатора влияния сессий."""
        if registry is None:
            registry = self.create_registry()
        if session_marker is None:
            session_marker = self.create_session_marker(registry)
        return SessionInfluenceAnalyzer(registry, session_marker)

    def create_transition_manager(
        self, registry: Optional[SessionRegistry] = None
    ) -> BaseSessionTransitionManager:
        """Создание менеджера переходов."""
        if registry is None:
            registry = self.create_registry()
        return DefaultSessionTransitionManager(registry)

    def create_session_predictor(
        self,
        registry: Optional[SessionRegistry] = None,
        analyzer: Optional[SessionMetricsAnalyzer] = None,
    ) -> BaseSessionPredictor:
        """Создание предиктора сессий."""
        if registry is None:
            registry = self.create_registry()
        if analyzer is None:
            analyzer = self.create_metrics_analyzer()
        return SessionPredictor(registry, analyzer)

    def create_analyzer_factory(self) -> SessionAnalyzerFactory:
        """Создание фабрики анализаторов."""
        return DefaultSessionAnalyzerFactory(self)


class DefaultSessionAnalyzerFactory(SessionAnalyzerFactory):
    """Реализация фабрики анализаторов сессий."""

    def __init__(self, component_factory: SessionComponentFactory) -> None:
        self.component_factory = component_factory

    def create_analyzer(
        self,
        name: str,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        config: Optional[Any] = None,
        force_recreate: bool = False
    ) -> Optional[BaseSessionAnalyzer]:
        """Создание анализатора."""
        if name == "influence":
            return self.component_factory.create_influence_analyzer(registry, session_marker)
        else:
            raise ValueError(f"Unknown analyzer type: {name}")

    def create_predictor(self, predictor_type: str) -> BaseSessionPredictor:
        """Создание предиктора."""
        if predictor_type == "session":
            return self.component_factory.create_session_predictor()
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    def create_transition_manager(self) -> BaseSessionTransitionManager:
        """Создание менеджера переходов."""
        return self.component_factory.create_transition_manager()

    def register_analyzer(
        self,
        name: str,
        analyzer_class: Type[BaseSessionAnalyzer],
        config: Optional[Any] = None,
    ) -> None:
        """Регистрация нового анализатора."""
        # Реализация для DefaultSessionAnalyzerFactory
        pass

    def get_analyzer(self, name: str) -> Optional[BaseSessionAnalyzer]:
        """Получение анализатора из кэша."""
        return None

    def get_available_analyzers(self) -> List[str]:
        """Получение списка доступных анализаторов."""
        return ["influence"]

    def get_analyzer_config(self, name: str) -> Optional[Any]:
        """Получение конфигурации анализатора."""
        return None

    def update_analyzer_config(
        self,
        name: str,
        config: Any,
    ) -> bool:
        """Обновление конфигурации анализатора."""
        return True

    def remove_analyzer(self, name: str) -> bool:
        """Удаление анализатора."""
        return True

    def create_analyzer_for_session_type(
        self,
        session_type: SessionType,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        config: Optional[Any] = None,
    ) -> Optional[BaseSessionAnalyzer]:
        """Создание анализатора для конкретного типа сессии."""
        return self.create_analyzer("influence", registry, session_marker, config)

    def create_multi_analyzer(
        self,
        analyzer_names: List[str],
        registry: SessionRegistry,
        session_marker: SessionMarker,
        configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, BaseSessionAnalyzer]:
        """Создание нескольких анализаторов."""
        analyzers: Dict[str, BaseSessionAnalyzer] = {}
        for name in analyzer_names:
            config = configs.get(name) if configs else None
            analyzer = self.create_analyzer(name, registry, session_marker, config)
            if analyzer:
                analyzers[name] = analyzer
        return analyzers

    def get_analyzer_statistics(self) -> Dict[str, Union[str, int, bool, List[str]]]:
        """Получение статистики анализаторов."""
        return {"available_analyzers": ["influence"]}

    def clear_cache(self) -> None:
        """Очистка кэша."""
        pass

    def validate_analyzer(self, name: str) -> bool:
        """Валидация анализатора."""
        return name == "influence"


class SessionServiceFactory:
    """Фабрика для создания сервисов сессий."""

    def __init__(self, component_factory: Optional[SessionComponentFactory] = None) -> None:
        self.component_factory = component_factory or SessionComponentFactory()

    def create_session_service(self) -> Any:
        """Создание сервиса сессий."""
        from .session_service import SessionService
        from .session_marker import SessionMarker
        from .session_influence_analyzer import SessionInfluenceAnalyzer
        
        registry = self.component_factory.create_registry()
        session_marker = self.component_factory.create_session_marker(registry)
        influence_analyzer = self.component_factory.create_influence_analyzer(registry, session_marker)
        
        # Создаем фабрику анализаторов
        from .session_analyzer_factory import SessionAnalyzerFactory
        analyzer_factory = SessionAnalyzerFactory()
        
        return SessionService(
            registry=registry,
            session_marker=session_marker,
            analyzer_factory=analyzer_factory,
        )

    def create_session_repository(
        self, storage_path: Optional[str] = None
    ) -> Any:
        """Создание репозитория сессий."""
        from infrastructure.repositories.base_repository import BaseRepository, MemoryBackend
        
        class SessionRepository(BaseRepository):
            def __init__(self, storage_path: Optional[str] = None):
                backend = MemoryBackend()
                super().__init__(backend)
                self._storage_path = storage_path
                self._sessions = {}
            
            async def save(self, entity):
                async with self.transaction():
                    session_id = getattr(entity, 'id', str(uuid.uuid4()))
                    self._sessions[session_id] = entity
                    return entity
            
            async def find_by_id(self, entity_id):
                return self._sessions.get(entity_id)
            
            async def find_all(self):
                return list(self._sessions.values())
            
            async def delete(self, entity_id):
                if entity_id in self._sessions:
                    del self._sessions[entity_id]
                    return True
                return False
        
        return SessionRepository(storage_path)

    def create_session_config_repository(
        self, storage_path: Optional[str] = None
    ) -> Any:
        """Создание репозитория конфигураций сессий."""
        from infrastructure.repositories.base_repository import BaseRepository, MemoryBackend
        
        class SessionConfigRepository(BaseRepository):
            def __init__(self, storage_path: Optional[str] = None):
                backend = MemoryBackend()
                super().__init__(backend)
                self._storage_path = storage_path
                self._configs = {}
            
            async def save(self, entity):
                async with self.transaction():
                    config_id = getattr(entity, 'id', str(uuid.uuid4()))
                    self._configs[config_id] = entity
                    return entity
            
            async def find_by_id(self, entity_id):
                return self._configs.get(entity_id)
            
            async def find_all(self):
                return list(self._configs.values())
            
            async def delete(self, entity_id):
                if entity_id in self._configs:
                    del self._configs[entity_id]
                    return True
                return False
        
        return SessionConfigRepository(storage_path)


# Глобальные функции для удобства
def get_session_service() -> Any:
    """Получение сервиса сессий."""
    factory = SessionServiceFactory()
    return factory.create_session_service()


def get_session_analyzer() -> SessionInfluenceAnalyzer:
    """Получение анализатора сессий."""
    factory = SessionComponentFactory()
    return factory.create_influence_analyzer()


def get_session_marker() -> SessionMarker:
    """Получение маркера сессий."""
    factory = SessionComponentFactory()
    return factory.create_session_marker()


def get_session_registry() -> SessionRegistry:
    """Получение реестра сессий."""
    factory = SessionComponentFactory()
    return factory.create_registry()
