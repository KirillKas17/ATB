"""
Unit тесты для base_service.

Покрывает:
- BaseServiceProtocol протокол
- BaseService базовый класс
- Все методы и их поведение
- Обработку ошибок и метрик
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch
import time

from domain.interfaces.base_service import BaseServiceProtocol, BaseService


class TestBaseServiceProtocol:
    """Тесты для BaseServiceProtocol."""

    def test_protocol_implementation(self):
        """Тест реализации протокола."""
        class MockService:
            def get_service_name(self) -> str:
                return "TestService"
            
            def get_service_version(self) -> str:
                return "1.0.0"
            
            def is_healthy(self) -> bool:
                return True
            
            def get_metrics(self) -> Dict[str, Any]:
                return {"requests": 100}
            
            def initialize(self) -> None:
                pass
            
            def shutdown(self) -> None:
                pass
            
            def get_last_error(self) -> Optional[str]:
                return None
            
            def get_start_time(self) -> datetime:
                return datetime.now()
            
            def get_uptime(self) -> float:
                return 3600.0

        service = MockService()
        assert isinstance(service, BaseServiceProtocol)

    def test_protocol_methods_exist(self):
        """Тест наличия всех методов протокола."""
        protocol_methods = [
            'get_service_name',
            'get_service_version', 
            'is_healthy',
            'get_metrics',
            'initialize',
            'shutdown',
            'get_last_error',
            'get_start_time',
            'get_uptime'
        ]
        
        for method_name in protocol_methods:
            assert hasattr(BaseServiceProtocol, method_name)


class TestBaseService:
    """Тесты для BaseService."""

    @pytest.fixture
    def base_service(self) -> BaseService:
        """Тестовый базовый сервис."""
        return BaseService()

    def test_initialization(self, base_service):
        """Тест инициализации сервиса."""
        assert base_service.get_service_name() == "BaseService"
        assert base_service.get_service_version() == "1.0.0"
        assert base_service.is_healthy() is True
        assert isinstance(base_service.get_start_time(), datetime)
        assert base_service.get_last_error() is None
        assert isinstance(base_service.get_metrics(), dict)

    def test_service_name(self, base_service):
        """Тест имени сервиса."""
        assert base_service.get_service_name() == "BaseService"
        assert isinstance(base_service.get_service_name(), str)

    def test_service_version(self, base_service):
        """Тест версии сервиса."""
        assert base_service.get_service_version() == "1.0.0"
        assert isinstance(base_service.get_service_version(), str)

    def test_health_status(self, base_service):
        """Тест статуса здоровья."""
        assert base_service.is_healthy() is True
        assert isinstance(base_service.is_healthy(), bool)

    def test_metrics_initialization(self, base_service):
        """Тест инициализации метрик."""
        metrics = base_service.get_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # Изначально пустой словарь

    def test_start_time(self, base_service):
        """Тест времени запуска."""
        start_time = base_service.get_start_time()
        assert isinstance(start_time, datetime)
        
        # Время запуска должно быть близко к текущему времени
        now = datetime.now()
        time_diff = abs((now - start_time).total_seconds())
        assert time_diff < 5  # Разница не более 5 секунд

    def test_uptime_calculation(self, base_service):
        """Тест расчета времени работы."""
        # Даем время на инициализацию
        time.sleep(0.1)
        
        uptime = base_service.get_uptime()
        assert isinstance(uptime, float)
        assert uptime >= 0.0
        assert uptime < 10.0  # Не должно быть больше 10 секунд

    def test_last_error_initial(self, base_service):
        """Тест последней ошибки при инициализации."""
        assert base_service.get_last_error() is None

    def test_initialize_method(self, base_service):
        """Тест метода initialize."""
        # Изменяем состояние сервиса
        base_service._healthy = False
        base_service._last_error = "Test error"
        
        # Вызываем initialize
        base_service.initialize()
        
        # Проверяем, что состояние сброшено
        assert base_service.is_healthy() is True
        assert base_service.get_last_error() is None
        
        # Время запуска должно обновиться
        start_time = base_service.get_start_time()
        now = datetime.now()
        time_diff = abs((now - start_time).total_seconds())
        assert time_diff < 5

    def test_shutdown_method(self, base_service):
        """Тест метода shutdown."""
        assert base_service.is_healthy() is True
        
        base_service.shutdown()
        
        assert base_service.is_healthy() is False

    def test_protocol_compliance(self, base_service):
        """Тест соответствия протоколу."""
        assert isinstance(base_service, BaseServiceProtocol)

    def test_metrics_modification(self, base_service):
        """Тест модификации метрик."""
        # Добавляем метрики
        base_service._metrics["requests"] = 100
        base_service._metrics["errors"] = 5
        base_service._metrics["response_time"] = 0.5
        
        metrics = base_service.get_metrics()
        
        assert metrics["requests"] == 100
        assert metrics["errors"] == 5
        assert metrics["response_time"] == 0.5
        assert len(metrics) == 3

    def test_error_handling(self, base_service):
        """Тест обработки ошибок."""
        # Устанавливаем ошибку
        test_error = "Database connection failed"
        base_service._last_error = test_error
        
        assert base_service.get_last_error() == test_error

    def test_service_name_customization(self):
        """Тест кастомизации имени сервиса."""
        class CustomService(BaseService):
            def __init__(self):
                super().__init__()
                self._service_name = "CustomService"
                self._service_version = "2.0.0"

        custom_service = CustomService()
        
        assert custom_service.get_service_name() == "CustomService"
        assert custom_service.get_service_version() == "2.0.0"

    def test_health_status_transitions(self, base_service):
        """Тест переходов статуса здоровья."""
        # Изначально здоров
        assert base_service.is_healthy() is True
        
        # Устанавливаем нездоровое состояние
        base_service._healthy = False
        assert base_service.is_healthy() is False
        
        # Инициализируем снова
        base_service.initialize()
        assert base_service.is_healthy() is True

    def test_uptime_accuracy(self, base_service):
        """Тест точности расчета времени работы."""
        # Запоминаем время запуска
        start_time = base_service.get_start_time()
        
        # Ждем немного
        time.sleep(0.1)
        
        # Проверяем uptime
        uptime = base_service.get_uptime()
        expected_uptime = (datetime.now() - start_time).total_seconds()
        
        # Разница должна быть небольшой
        assert abs(uptime - expected_uptime) < 0.1

    def test_metrics_persistence(self, base_service):
        """Тест сохранения метрик."""
        # Добавляем метрики
        base_service._metrics["test_metric"] = 42
        
        # Проверяем, что метрики сохранились
        metrics = base_service.get_metrics()
        assert metrics["test_metric"] == 42
        
        # Проверяем, что возвращается тот же объект
        metrics2 = base_service.get_metrics()
        assert metrics is metrics2

    def test_error_clearing(self, base_service):
        """Тест очистки ошибок."""
        # Устанавливаем ошибку
        base_service._last_error = "Test error"
        assert base_service.get_last_error() == "Test error"
        
        # Очищаем ошибку
        base_service._last_error = None
        assert base_service.get_last_error() is None

    def test_service_lifecycle(self, base_service):
        """Тест жизненного цикла сервиса."""
        # Начальное состояние
        assert base_service.is_healthy() is True
        assert base_service.get_last_error() is None
        
        # Симулируем ошибку
        base_service._last_error = "Service error"
        base_service._healthy = False
        
        assert base_service.is_healthy() is False
        assert base_service.get_last_error() == "Service error"
        
        # Перезапуск сервиса
        base_service.initialize()
        
        assert base_service.is_healthy() is True
        assert base_service.get_last_error() is None

    def test_metrics_types(self, base_service):
        """Тест типов метрик."""
        # Добавляем метрики разных типов
        base_service._metrics = {
            "string_metric": "test_value",
            "int_metric": 42,
            "float_metric": 3.14,
            "bool_metric": True,
            "list_metric": [1, 2, 3],
            "dict_metric": {"key": "value"}
        }
        
        metrics = base_service.get_metrics()
        
        assert isinstance(metrics["string_metric"], str)
        assert isinstance(metrics["int_metric"], int)
        assert isinstance(metrics["float_metric"], float)
        assert isinstance(metrics["bool_metric"], bool)
        assert isinstance(metrics["list_metric"], list)
        assert isinstance(metrics["dict_metric"], dict)

    def test_concurrent_metrics_access(self, base_service):
        """Тест конкурентного доступа к метрикам."""
        import threading
        
        def update_metrics():
            for i in range(100):
                base_service._metrics[f"metric_{i}"] = i
        
        def read_metrics():
            for i in range(100):
                metrics = base_service.get_metrics()
                assert isinstance(metrics, dict)
        
        # Запускаем потоки
        writer = threading.Thread(target=update_metrics)
        reader = threading.Thread(target=read_metrics)
        
        writer.start()
        reader.start()
        
        writer.join()
        reader.join()
        
        # Проверяем результат
        metrics = base_service.get_metrics()
        assert len(metrics) == 100
        for i in range(100):
            assert metrics[f"metric_{i}"] == i


class TestBaseServiceIntegration:
    """Интеграционные тесты для BaseService."""

    def test_full_service_lifecycle(self):
        """Тест полного жизненного цикла сервиса."""
        service = BaseService()
        
        # Проверяем начальное состояние
        assert service.get_service_name() == "BaseService"
        assert service.get_service_version() == "1.0.0"
        assert service.is_healthy() is True
        assert service.get_last_error() is None
        assert len(service.get_metrics()) == 0
        
        # Добавляем метрики
        service._metrics["requests"] = 100
        service._metrics["errors"] = 5
        
        # Проверяем метрики
        metrics = service.get_metrics()
        assert metrics["requests"] == 100
        assert metrics["errors"] == 5
        
        # Симулируем ошибку
        service._last_error = "Connection timeout"
        service._healthy = False
        
        assert service.is_healthy() is False
        assert service.get_last_error() == "Connection timeout"
        
        # Перезапускаем сервис
        service.initialize()
        
        assert service.is_healthy() is True
        assert service.get_last_error() is None
        # Метрики должны сохраниться
        assert service.get_metrics()["requests"] == 100
        
        # Завершаем работу
        service.shutdown()
        assert service.is_healthy() is False

    def test_service_monitoring(self):
        """Тест мониторинга сервиса."""
        service = BaseService()
        
        # Запоминаем время запуска
        start_time = service.get_start_time()
        
        # Симулируем работу сервиса
        time.sleep(0.1)
        
        # Проверяем uptime
        uptime = service.get_uptime()
        assert uptime > 0
        
        # Проверяем, что uptime соответствует времени работы
        expected_uptime = (datetime.now() - start_time).total_seconds()
        assert abs(uptime - expected_uptime) < 0.1

    def test_service_health_monitoring(self):
        """Тест мониторинга здоровья сервиса."""
        service = BaseService()
        
        # Изначально здоров
        assert service.is_healthy() is True
        
        # Симулируем проблемы
        service._healthy = False
        service._last_error = "Service unavailable"
        
        # Проверяем состояние
        assert service.is_healthy() is False
        assert service.get_last_error() == "Service unavailable"
        
        # Восстанавливаем сервис
        service.initialize()
        assert service.is_healthy() is True
        assert service.get_last_error() is None

    def test_metrics_tracking(self):
        """Тест отслеживания метрик."""
        service = BaseService()
        
        # Добавляем различные метрики
        service._metrics.update({
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "average_response_time": 0.15,
            "peak_response_time": 2.5,
            "active_connections": 25,
            "memory_usage_mb": 128.5,
            "cpu_usage_percent": 45.2
        })
        
        metrics = service.get_metrics()
        
        # Проверяем все метрики
        assert metrics["total_requests"] == 1000
        assert metrics["successful_requests"] == 950
        assert metrics["failed_requests"] == 50
        assert metrics["average_response_time"] == 0.15
        assert metrics["peak_response_time"] == 2.5
        assert metrics["active_connections"] == 25
        assert metrics["memory_usage_mb"] == 128.5
        assert metrics["cpu_usage_percent"] == 45.2
        
        # Проверяем вычисляемые метрики
        success_rate = metrics["successful_requests"] / metrics["total_requests"]
        assert success_rate == 0.95 