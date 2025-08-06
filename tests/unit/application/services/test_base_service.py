"""
Unit тесты для application.services.base_service

Покрывает:
- Создание и инициализацию BaseService
- Логирование
- Обработка ошибок
- Метрики и мониторинг
"""
import pytest
from unittest.mock import Mock, patch
from application.services.base_service import BaseService


class TestBaseService:
    """Тесты для BaseService"""

    def test_base_service_creation(self):
        """Тест создания BaseService"""
        service = BaseService(name="test_service")
        assert service.name == "test_service"
        assert service.logger is not None
        assert service.metrics is not None

    def test_base_service_logging(self):
        """Тест логирования"""
        with patch('application.services.base_service.logging') as mock_logging:
            service = BaseService(name="test_service")
            
            service.log_info("Test info message")
            service.log_warning("Test warning message")
            service.log_error("Test error message")
            service.log_debug("Test debug message")
            
            # Проверяем, что методы логирования были вызваны
            assert mock_logging.getLogger.called

    def test_base_service_metrics(self):
        """Тест метрик"""
        service = BaseService(name="test_service")
        
        # Тест инкремента счетчика
        service.increment_counter("test_counter")
        service.increment_counter("test_counter", 5)
        
        # Тест записи значения
        service.record_value("test_value", 100.5)
        
        # Тест записи времени выполнения
        service.record_execution_time("test_operation", 0.5)

    def test_base_service_error_handling(self):
        """Тест обработки ошибок"""
        service = BaseService(name="test_service")
        
        # Тест обработки исключения
        with pytest.raises(ValueError):
            service.handle_error(ValueError("Test error"), "Test operation")
        
        # Тест логирования ошибки
        with patch.object(service, 'log_error') as mock_log_error:
            try:
                raise ValueError("Test error")
            except Exception as e:
                service.handle_error(e, "Test operation")
            
            mock_log_error.assert_called_once()

    def test_base_service_health_check(self):
        """Тест проверки здоровья сервиса"""
        service = BaseService(name="test_service")
        
        health_status = service.health_check()
        assert health_status["service_name"] == "test_service"
        assert health_status["status"] == "healthy"
        assert "uptime" in health_status
        assert "version" in health_status

    def test_base_service_get_service_info(self):
        """Тест получения информации о сервисе"""
        service = BaseService(name="test_service")
        
        service_info = service.get_service_info()
        assert service_info["name"] == "test_service"
        assert service_info["type"] == "BaseService"
        assert "created_at" in service_info

    def test_base_service_cleanup(self):
        """Тест очистки ресурсов"""
        service = BaseService(name="test_service")
        
        # Тест успешной очистки
        result = service.cleanup()
        assert result is True

    def test_base_service_validate_input(self):
        """Тест валидации входных данных"""
        service = BaseService(name="test_service")
        
        # Тест валидных данных
        assert service.validate_input({"test": "data"}) is True
        
        # Тест невалидных данных
        with pytest.raises(ValueError):
            service.validate_input(None)

    def test_base_service_validate_output(self):
        """Тест валидации выходных данных"""
        service = BaseService(name="test_service")
        
        # Тест валидных данных
        assert service.validate_output({"result": "success"}) is True
        
        # Тест невалидных данных
        with pytest.raises(ValueError):
            service.validate_output(None)

    def test_base_service_async_operation(self):
        """Тест асинхронных операций"""
        import asyncio
        
        service = BaseService(name="test_service")
        
        async def test_async_func():
            return "async_result"
        
        # Тест выполнения асинхронной операции
        result = asyncio.run(service.execute_async(test_async_func()))
        assert result == "async_result"

    def test_base_service_retry_mechanism(self):
        """Тест механизма повторных попыток"""
        service = BaseService(name="test_service")
        
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        # Тест успешного выполнения после повторных попыток
        result = service.retry_operation(failing_function, max_retries=3)
        assert result == "success"
        assert call_count == 3

    def test_base_service_cache_operations(self):
        """Тест операций с кешем"""
        service = BaseService(name="test_service")
        
        # Тест записи в кеш
        service.cache_set("test_key", "test_value", ttl=60)
        
        # Тест чтения из кеша
        value = service.cache_get("test_key")
        assert value == "test_value"
        
        # Тест удаления из кеша
        service.cache_delete("test_key")
        value = service.cache_get("test_key")
        assert value is None

    def test_base_service_configuration(self):
        """Тест конфигурации сервиса"""
        service = BaseService(name="test_service")
        
        # Тест установки конфигурации
        config = {"timeout": 30, "retries": 3}
        service.set_configuration(config)
        
        # Тест получения конфигурации
        retrieved_config = service.get_configuration()
        assert retrieved_config["timeout"] == 30
        assert retrieved_config["retries"] == 3

    def test_base_service_state_management(self):
        """Тест управления состоянием"""
        service = BaseService(name="test_service")
        
        # Тест установки состояния
        service.set_state("running")
        assert service.get_state() == "running"
        
        # Тест изменения состояния
        service.set_state("stopped")
        assert service.get_state() == "stopped"

    def test_base_service_performance_monitoring(self):
        """Тест мониторинга производительности"""
        service = BaseService(name="test_service")
        
        # Тест начала мониторинга
        service.start_performance_monitoring("test_operation")
        
        # Тест остановки мониторинга
        performance_data = service.stop_performance_monitoring("test_operation")
        assert "duration" in performance_data
        assert "start_time" in performance_data
        assert "end_time" in performance_data

    def test_base_service_dependency_injection(self):
        """Тест внедрения зависимостей"""
        service = BaseService(name="test_service")
        
        # Тест установки зависимости
        mock_dependency = Mock()
        service.set_dependency("test_dep", mock_dependency)
        
        # Тест получения зависимости
        retrieved_dependency = service.get_dependency("test_dep")
        assert retrieved_dependency == mock_dependency

    def test_base_service_event_handling(self):
        """Тест обработки событий"""
        service = BaseService(name="test_service")
        
        # Тест подписки на событие
        event_handler = Mock()
        service.subscribe_to_event("test_event", event_handler)
        
        # Тест публикации события
        service.publish_event("test_event", {"data": "test"})
        event_handler.assert_called_once_with({"data": "test"})

    def test_base_service_serialization(self):
        """Тест сериализации сервиса"""
        service = BaseService(name="test_service")
        
        # Тест сериализации в словарь
        data = service.to_dict()
        assert data["name"] == "test_service"
        assert data["type"] == "BaseService"
        
        # Тест десериализации из словаря
        new_service = BaseService.from_dict(data)
        assert new_service.name == "test_service"

    def test_base_service_equality(self):
        """Тест равенства сервисов"""
        service1 = BaseService(name="test_service")
        service2 = BaseService(name="test_service")
        service3 = BaseService(name="different_service")
        
        assert service1 == service2
        assert service1 != service3
        assert hash(service1) == hash(service2)
        assert hash(service1) != hash(service3)

    def test_base_service_repr(self):
        """Тест строкового представления"""
        service = BaseService(name="test_service")
        
        repr_str = repr(service)
        assert "BaseService" in repr_str
        assert "test_service" in repr_str

    def test_base_service_str(self):
        """Тест строкового представления для пользователя"""
        service = BaseService(name="test_service")
        
        str_repr = str(service)
        assert "test_service" in str_repr
        assert "BaseService" in str_repr 