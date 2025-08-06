#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для DI Container Application Layer.
"""

import pytest
from typing import Any, Dict, Protocol
from unittest.mock import Mock, AsyncMock

from application.di_container import DIContainer
from domain.exceptions import ValidationError


class TestService(Protocol):
    """Тестовый протокол сервиса."""
    
    def process(self, data: str) -> str:
        ...


class MockTestService:
    """Мок-реализация тестового сервиса."""
    
    def process(self, data: str) -> str:
        return f"processed: {data}"


class MockDependentService:
    """Сервис с зависимостью."""
    
    def __init__(self, test_service: TestService):
        self.test_service = test_service
    
    def execute(self, data: str) -> str:
        return f"executed: {self.test_service.process(data)}"


class TestDIContainer:
    """Тесты для DI Container."""

    @pytest.fixture
    def container(self) -> DIContainer:
        """Фикстура DI контейнера."""
        return DIContainer()

    @pytest.fixture
    def mock_service(self) -> MockTestService:
        """Фикстура мок-сервиса."""
        return MockTestService()

    def test_container_initialization(self, container: DIContainer) -> None:
        """Тест инициализации контейнера."""
        assert container is not None
        assert hasattr(container, '_services')
        assert hasattr(container, '_singletons')
        assert len(container._services) == 0
        assert len(container._singletons) == 0

    def test_register_service_transient(self, container: DIContainer) -> None:
        """Тест регистрации транзиентного сервиса."""
        container.register(TestService, MockTestService, singleton=False)
        
        # Проверяем, что сервис зарегистрирован
        assert TestService in container._services
        assert container._services[TestService]['factory'] == MockTestService
        assert container._services[TestService]['singleton'] is False

    def test_register_service_singleton(self, container: DIContainer) -> None:
        """Тест регистрации singleton сервиса."""
        container.register(TestService, MockTestService, singleton=True)
        
        # Проверяем, что сервис зарегистрирован как singleton
        assert TestService in container._services
        assert container._services[TestService]['singleton'] is True

    def test_register_service_with_factory(self, container: DIContainer) -> None:
        """Тест регистрации сервиса с фабрикой."""
        def service_factory():
            return MockTestService()
        
        container.register(TestService, service_factory, singleton=False)
        
        assert TestService in container._services
        assert container._services[TestService]['factory'] == service_factory

    def test_register_service_duplicate(self, container: DIContainer) -> None:
        """Тест регистрации дублирующегося сервиса."""
        container.register(TestService, MockTestService)
        
        # Повторная регистрация должна перезаписать
        container.register(TestService, MockTestService, singleton=True)
        
        assert container._services[TestService]['singleton'] is True

    def test_resolve_transient_service(self, container: DIContainer) -> None:
        """Тест разрешения транзиентного сервиса."""
        container.register(TestService, MockTestService, singleton=False)
        
        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)
        
        # Должны быть разные экземпляры
        assert isinstance(service1, MockTestService)
        assert isinstance(service2, MockTestService)
        assert service1 is not service2

    def test_resolve_singleton_service(self, container: DIContainer) -> None:
        """Тест разрешения singleton сервиса."""
        container.register(TestService, MockTestService, singleton=True)
        
        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)
        
        # Должен быть один и тот же экземпляр
        assert isinstance(service1, MockTestService)
        assert service1 is service2

    def test_resolve_unregistered_service(self, container: DIContainer) -> None:
        """Тест разрешения незарегистрированного сервиса."""
        with pytest.raises(ValidationError, match="Service .* not registered"):
            container.resolve(TestService)

    def test_resolve_with_dependencies(self, container: DIContainer) -> None:
        """Тест разрешения сервиса с зависимостями."""
        # Регистрируем зависимость
        container.register(TestService, MockTestService, singleton=True)
        
        # Регистрируем сервис с зависимостью
        container.register(MockDependentService, MockDependentService, singleton=False)
        
        # Разрешаем сервис
        dependent_service = container.resolve(MockDependentService)
        
        assert isinstance(dependent_service, MockDependentService)
        assert isinstance(dependent_service.test_service, MockTestService)

    def test_resolve_circular_dependency(self, container: DIContainer) -> None:
        """Тест обнаружения циклических зависимостей."""
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
        
        container.register(ServiceA, ServiceA)
        container.register(ServiceB, ServiceB)
        
        with pytest.raises(ValidationError, match="Circular dependency detected"):
            container.resolve(ServiceA)

    def test_register_instance(self, container: DIContainer, mock_service: MockTestService) -> None:
        """Тест регистрации готового экземпляра."""
        container.register_instance(TestService, mock_service)
        
        resolved_service = container.resolve(TestService)
        
        # Должен вернуть тот же экземпляр
        assert resolved_service is mock_service

    def test_is_registered(self, container: DIContainer) -> None:
        """Тест проверки регистрации сервиса."""
        assert not container.is_registered(TestService)
        
        container.register(TestService, MockTestService)
        
        assert container.is_registered(TestService)

    def test_unregister_service(self, container: DIContainer) -> None:
        """Тест отмены регистрации сервиса."""
        container.register(TestService, MockTestService)
        assert container.is_registered(TestService)
        
        container.unregister(TestService)
        assert not container.is_registered(TestService)

    def test_clear_container(self, container: DIContainer) -> None:
        """Тест очистки контейнера."""
        container.register(TestService, MockTestService)
        container.register(MockDependentService, MockDependentService)
        
        assert len(container._services) == 2
        
        container.clear()
        
        assert len(container._services) == 0
        assert len(container._singletons) == 0

    def test_get_registered_services(self, container: DIContainer) -> None:
        """Тест получения списка зарегистрированных сервисов."""
        container.register(TestService, MockTestService)
        container.register(MockDependentService, MockDependentService)
        
        services = container.get_registered_services()
        
        assert TestService in services
        assert MockDependentService in services
        assert len(services) == 2

    def test_container_context_manager(self, container: DIContainer) -> None:
        """Тест использования контейнера как контекстного менеджера."""
        with container:
            container.register(TestService, MockTestService)
            service = container.resolve(TestService)
            assert isinstance(service, MockTestService)
        
        # После выхода из контекста сервисы должны быть очищены
        assert len(container._services) == 0

    def test_lazy_initialization(self, container: DIContainer) -> None:
        """Тест ленивой инициализации."""
        init_count = 0
        
        class LazyService:
            def __init__(self):
                nonlocal init_count
                init_count += 1
        
        container.register(LazyService, LazyService, singleton=True)
        
        # Инициализация не должна произойти до первого обращения
        assert init_count == 0
        
        service1 = container.resolve(LazyService)
        assert init_count == 1
        
        service2 = container.resolve(LazyService)
        assert init_count == 1  # Повторной инициализации не должно быть
        assert service1 is service2

    def test_factory_with_parameters(self, container: DIContainer) -> None:
        """Тест фабрики с параметрами."""
        def create_service(name: str):
            service = MockTestService()
            service.name = name
            return service
        
        container.register_factory(TestService, create_service, name="test_service")
        
        service = container.resolve(TestService)
        assert isinstance(service, MockTestService)
        assert hasattr(service, 'name')
        assert service.name == "test_service"

    def test_conditional_registration(self, container: DIContainer) -> None:
        """Тест условной регистрации."""
        def condition():
            return True
        
        container.register_conditional(TestService, MockTestService, condition)
        
        service = container.resolve(TestService)
        assert isinstance(service, MockTestService)

    def test_conditional_registration_false(self, container: DIContainer) -> None:
        """Тест условной регистрации с ложным условием."""
        def condition():
            return False
        
        container.register_conditional(TestService, MockTestService, condition)
        
        with pytest.raises(ValidationError):
            container.resolve(TestService)

    def test_decorator_registration(self, container: DIContainer) -> None:
        """Тест регистрации через декоратор."""
        @container.service(TestService, singleton=True)
        class DecoratedService:
            def process(self, data: str) -> str:
                return f"decorated: {data}"
        
        service = container.resolve(TestService)
        assert isinstance(service, DecoratedService)
        assert service.process("test") == "decorated: test"

    def test_async_service_resolution(self, container: DIContainer) -> None:
        """Тест разрешения асинхронных сервисов."""
        class AsyncService:
            async def process_async(self, data: str) -> str:
                return f"async: {data}"
        
        container.register(AsyncService, AsyncService)
        
        service = container.resolve(AsyncService)
        assert isinstance(service, AsyncService)

    def test_service_lifecycle_events(self, container: DIContainer) -> None:
        """Тест событий жизненного цикла сервиса."""
        created_services = []
        disposed_services = []
        
        def on_service_created(service_type, service_instance):
            created_services.append((service_type, service_instance))
        
        def on_service_disposed(service_type, service_instance):
            disposed_services.append((service_type, service_instance))
        
        container.on_service_created = on_service_created
        container.on_service_disposed = on_service_disposed
        
        container.register(TestService, MockTestService)
        service = container.resolve(TestService)
        
        assert len(created_services) == 1
        assert created_services[0][0] == TestService
        assert isinstance(created_services[0][1], MockTestService)
        
        container.dispose_service(TestService, service)
        
        assert len(disposed_services) == 1

    def test_nested_dependencies(self, container: DIContainer) -> None:
        """Тест вложенных зависимостей."""
        class Level1Service:
            pass
        
        class Level2Service:
            def __init__(self, level1: Level1Service):
                self.level1 = level1
        
        class Level3Service:
            def __init__(self, level2: Level2Service):
                self.level2 = level2
        
        container.register(Level1Service, Level1Service)
        container.register(Level2Service, Level2Service)
        container.register(Level3Service, Level3Service)
        
        service = container.resolve(Level3Service)
        
        assert isinstance(service, Level3Service)
        assert isinstance(service.level2, Level2Service)
        assert isinstance(service.level2.level1, Level1Service)

    def test_optional_dependencies(self, container: DIContainer) -> None:
        """Тест опциональных зависимостей."""
        from typing import Optional
        
        class OptionalService:
            def __init__(self, optional_dep: Optional[TestService] = None):
                self.optional_dep = optional_dep
        
        container.register(OptionalService, OptionalService)
        
        # Без регистрации опциональной зависимости
        service = container.resolve(OptionalService)
        assert isinstance(service, OptionalService)
        assert service.optional_dep is None
        
        # С регистрацией опциональной зависимости
        container.register(TestService, MockTestService)
        service = container.resolve(OptionalService)
        assert isinstance(service.optional_dep, MockTestService)

    def test_multiple_implementations(self, container: DIContainer) -> None:
        """Тест множественных реализаций."""
        class Implementation1(MockTestService):
            def process(self, data: str) -> str:
                return f"impl1: {data}"
        
        class Implementation2(MockTestService):
            def process(self, data: str) -> str:
                return f"impl2: {data}"
        
        container.register_multiple(TestService, [Implementation1, Implementation2])
        
        implementations = container.resolve_all(TestService)
        
        assert len(implementations) == 2
        assert any(isinstance(impl, Implementation1) for impl in implementations)
        assert any(isinstance(impl, Implementation2) for impl in implementations)

    def test_scoped_services(self, container: DIContainer) -> None:
        """Тест сервисов с областью видимости."""
        container.register_scoped(TestService, MockTestService)
        
        with container.create_scope() as scope:
            service1 = scope.resolve(TestService)
            service2 = scope.resolve(TestService)
            
            # В рамках одной области видимости - один экземпляр
            assert service1 is service2
        
        with container.create_scope() as scope2:
            service3 = scope2.resolve(TestService)
            
            # В новой области видимости - новый экземпляр
            assert service3 is not service1

    def test_service_validation(self, container: DIContainer) -> None:
        """Тест валидации сервисов."""
        # Регистрация с неправильным типом
        with pytest.raises(ValidationError):
            container.register(TestService, str)  # str не реализует TestService
        
        # Регистрация None
        with pytest.raises(ValidationError):
            container.register(TestService, None)

    def test_container_thread_safety(self, container: DIContainer) -> None:
        """Тест потокобезопасности контейнера."""
        import threading
        import time
        
        results = []
        
        def resolve_service():
            service = container.resolve(TestService)
            results.append(service)
            time.sleep(0.01)  # Небольшая задержка
        
        container.register(TestService, MockTestService, singleton=True)
        
        threads = [threading.Thread(target=resolve_service) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Все результаты должны быть одинаковыми (singleton)
        assert len(results) == 10
        assert all(service is results[0] for service in results)

    def test_performance_resolution(self, container: DIContainer) -> None:
        """Тест производительности разрешения сервисов."""
        container.register(TestService, MockTestService, singleton=True)
        
        # Множественное разрешение должно быть быстрым
        import time
        start_time = time.time()
        
        for _ in range(1000):
            service = container.resolve(TestService)
        
        end_time = time.time()
        
        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0

    def test_memory_management(self, container: DIContainer) -> None:
        """Тест управления памятью."""
        import gc
        import weakref
        
        container.register(TestService, MockTestService, singleton=False)
        
        service = container.resolve(TestService)
        weak_ref = weakref.ref(service)
        
        del service
        gc.collect()
        
        # Объект должен быть удален из памяти
        assert weak_ref() is None

    def test_container_serialization(self, container: DIContainer) -> None:
        """Тест сериализации состояния контейнера."""
        container.register(TestService, MockTestService)
        
        # Экспорт конфигурации
        config = container.export_configuration()
        
        # Создание нового контейнера
        new_container = DIContainer()
        new_container.import_configuration(config)
        
        # Проверка, что конфигурация восстановлена
        assert new_container.is_registered(TestService)
        
        service = new_container.resolve(TestService)
        assert isinstance(service, MockTestService)