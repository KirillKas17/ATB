"""
Unit тесты для модуля monitoring_alerts.
Тестирует:
- AlertManager
- Систему алертов
- Обработчики алертов
- Правила алертов
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from infrastructure.monitoring.monitoring_alerts import (
    AlertManager,
    get_alert_manager,
    create_alert,
    add_alert_handler,
    resolve_alert,
    get_alerts,
    AlertRule,
    AlertHandler,
    Alert,
    AlertSeverity
)
try:
    from infrastructure.monitoring.monitoring_alerts import AlertRule, AlertHandler
except ImportError:
    class MockAlertRule: pass
    class MockAlertHandler: pass
class TestAlertManager:
    """Тесты для AlertManager."""
    def test_init_default(self) -> None:
        """Тест инициализации с параметрами по умолчанию."""
        manager = AlertManager()
        assert manager.name == "default"
        assert manager.alerts == []
        assert manager.rules == []
        assert manager.handlers == {}
        assert manager.is_running is False
        assert manager.evaluation_interval == 30.0
    def test_init_custom(self) -> None:
        """Тест инициализации с пользовательскими параметрами."""
        manager = AlertManager(
            name="custom_alerts",
            evaluation_interval=60.0
        )
        assert manager.name == "custom_alerts"
        assert manager.evaluation_interval == 60.0
    def test_create_alert(self) -> None:
        """Тест создания алерта."""
        manager = AlertManager()
        alert = manager.create_alert(
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            metadata={"test_key": "test_value"}
        )
        assert alert.message == "Test alert message"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "test_source"
        assert alert.metadata["test_key"] == "test_value"
        assert alert.timestamp is not None
        assert alert.alert_id is not None
        assert alert.acknowledged is False
        assert alert.resolved is False
    def test_create_alert_with_exception(self) -> None:
        """Тест создания алерта с исключением."""
        manager = AlertManager()
        exception = ValueError("Test error")
        alert = manager.create_alert(
            message="Test error alert",
            severity=AlertSeverity.ERROR,
            source="test_source",
            exception=exception
        )
        assert alert.exception == exception
        assert alert.severity == AlertSeverity.ERROR
    def test_add_alert_rule(self) -> None:
        """Тест добавления правила алерта."""
        manager = AlertManager()
        # Mock rule
        rule = Mock()
        rule.name = "test_rule"
        rule.evaluate = Mock(return_value=True)
        manager.add_alert_rule(rule)
        assert rule in manager.rules
    def test_add_alert_handler(self) -> None:
        """Тест добавления обработчика алертов."""
        manager = AlertManager()
        # Mock handler
        handler = Mock()
        handler.name = "test_handler"
        handler.handle = Mock()
        manager.add_alert_handler(AlertSeverity.WARNING, handler)
        assert AlertSeverity.WARNING in manager.handlers
        assert handler in manager.handlers[AlertSeverity.WARNING]
    def test_get_alerts(self) -> None:
        """Тест получения алертов."""
        manager = AlertManager()
        # Создаем несколько алертов
        alert1 = manager.create_alert("Alert 1", AlertSeverity.INFO, "source1")
        alert2 = manager.create_alert("Alert 2", AlertSeverity.WARNING, "source2")
        alert3 = manager.create_alert("Alert 3", AlertSeverity.ERROR, "source3")
        # Получаем все алерты
        all_alerts = manager.get_alerts()
        assert len(all_alerts) == 3
        # Получаем алерты по уровню
        warning_alerts = manager.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].severity == AlertSeverity.WARNING
        # Получаем неподтвержденные алерты
        unacknowledged_alerts = manager.get_alerts(acknowledged=False)
        assert len(unacknowledged_alerts) == 3
        # Получаем алерты с лимитом
        limited_alerts = manager.get_alerts(limit=2)
        assert len(limited_alerts) == 2
    def test_acknowledge_alert(self) -> None:
        """Тест подтверждения алерта."""
        manager = AlertManager()
        alert = manager.create_alert("Test alert", AlertSeverity.WARNING, "test_source")
        # Подтверждаем алерт
        result = manager.acknowledge_alert(alert.alert_id)
        assert result is True
        # Проверяем, что алерт подтвержден
        updated_alert = manager.get_alerts(alert_id=alert.alert_id)[0]
        assert updated_alert.acknowledged is True
        assert updated_alert.acknowledged_at is not None
    def test_resolve_alert(self) -> None:
        """Тест разрешения алерта."""
        manager = AlertManager()
        alert = manager.create_alert("Test alert", AlertSeverity.WARNING, "test_source")
        # Разрешаем алерт
        result = manager.resolve_alert(alert.alert_id)
        assert result is True
        # Проверяем, что алерт разрешен
        updated_alert = manager.get_alerts(alert_id=alert.alert_id)[0]
        assert updated_alert.resolved is True
        assert updated_alert.resolved_at is not None
    def test_acknowledge_nonexistent_alert(self) -> None:
        """Тест подтверждения несуществующего алерта."""
        manager = AlertManager()
        result = manager.acknowledge_alert("nonexistent-id")
        assert result is False
    def test_resolve_nonexistent_alert(self) -> None:
        """Тест разрешения несуществующего алерта."""
        manager = AlertManager()
        result = manager.resolve_alert("nonexistent-id")
        assert result is False
    async def test_start_evaluation(self) -> None:
        """Тест запуска оценки алертов."""
        manager = AlertManager()
        await manager.start_evaluation()
        assert manager.is_running is True
    async def test_stop_evaluation(self) -> None:
        """Тест остановки оценки алертов."""
        manager = AlertManager()
        await manager.start_evaluation()
        await manager.stop_evaluation()
        assert manager.is_running is False
    async def test_evaluation_loop(self) -> None:
        """Тест цикла оценки алертов."""
        manager = AlertManager(evaluation_interval=0.1)  # Быстрый интервал для тестов
        # Mock rule that creates an alert
        rule = Mock()
        rule.name = "test_rule"
        rule.evaluate = Mock(return_value=True)
        rule.create_alert = Mock(return_value=manager.create_alert(
            "Rule triggered alert", AlertSeverity.WARNING, "test_rule"
        ))
        manager.add_alert_rule(rule)
        # Запускаем оценку
        await manager.start_evaluation()
        # Ждем немного для выполнения оценки
        await asyncio.sleep(0.2)
        # Останавливаем оценку
        await manager.stop_evaluation()
        # Проверяем, что правило было оценено
        assert rule.evaluate.called
    async def test_handler_notification(self) -> None:
        """Тест уведомления обработчиков."""
        manager = AlertManager()
        # Mock handler
        handler = Mock()
        handler.name = "test_handler"
        handler.handle = AsyncMock()
        manager.add_alert_handler(AlertSeverity.WARNING, handler)
        # Создаем алерт
        alert = manager.create_alert("Test alert", AlertSeverity.WARNING, "test_source")
        # Уведомляем обработчики
        await manager._notify_handlers(alert)
        # Проверяем, что обработчик был вызван
        handler.handle.assert_called_once_with(alert)
    def test_alert_statistics(self) -> None:
        """Тест статистики алертов."""
        manager = AlertManager()
        # Создаем алерты разных уровней
        manager.create_alert("Info alert", AlertSeverity.INFO, "source1")
        manager.create_alert("Warning alert", AlertSeverity.WARNING, "source2")
        manager.create_alert("Error alert", AlertSeverity.ERROR, "source3")
        manager.create_alert("Critical alert", AlertSeverity.CRITICAL, "source4")
        # Подтверждаем один алерт
        alerts = manager.get_alerts()
        manager.acknowledge_alert(alerts[0].alert_id)
        # Разрешаем один алерт
        manager.resolve_alert(alerts[1].alert_id)
        stats = manager.get_alert_statistics()
        assert stats["total_alerts"] == 4
        assert stats["acknowledged_alerts"] == 1
        assert stats["resolved_alerts"] == 1
        assert stats["unacknowledged_alerts"] == 3
        assert stats["unresolved_alerts"] == 3
        assert stats["alerts_by_severity"]["INFO"] == 1
        assert stats["alerts_by_severity"]["WARNING"] == 1
        assert stats["alerts_by_severity"]["ERROR"] == 1
        assert stats["alerts_by_severity"]["CRITICAL"] == 1
    def test_cleanup_old_alerts(self) -> None:
        """Тест очистки старых алертов."""
        manager = AlertManager()
        # Создаем старый алерт
        old_alert = manager.create_alert("Old alert", AlertSeverity.INFO, "source1")
        old_alert.timestamp = datetime.now() - timedelta(days=31)  # 31 день назад
        # Создаем новый алерт
        new_alert = manager.create_alert("New alert", AlertSeverity.INFO, "source2")
        # Очищаем старые алерты (старше 30 дней)
        manager.cleanup_old_alerts(days=30)
        # Проверяем, что старый алерт удален, а новый остался
        remaining_alerts = manager.get_alerts()
        assert len(remaining_alerts) == 1
        assert remaining_alerts[0].alert_id == new_alert.alert_id
    def test_alert_deduplication(self) -> None:
        """Тест дедупликации алертов."""
        manager = AlertManager()
        # Создаем одинаковые алерты
        alert1 = manager.create_alert("Duplicate alert", AlertSeverity.WARNING, "source1")
        alert2 = manager.create_alert("Duplicate alert", AlertSeverity.WARNING, "source1")
        # Проверяем, что создались два алерта (без дедупликации)
        assert len(manager.get_alerts()) == 2
        # Включаем дедупликацию
        manager.enable_deduplication = True
        # Создаем еще один одинаковый алерт
        alert3 = manager.create_alert("Duplicate alert", AlertSeverity.WARNING, "source1")
        # Проверяем, что третий алерт не создался
        assert len(manager.get_alerts()) == 2
    def test_alert_escalation(self) -> None:
        """Тест эскалации алертов."""
        manager = AlertManager()
        # Создаем алерт
        alert = manager.create_alert("Test alert", AlertSeverity.WARNING, "source1")
        # Настраиваем эскалацию через 5 минут
        manager.set_escalation_time(AlertSeverity.WARNING, timedelta(minutes=5))
        # Устанавливаем время создания алерта на 6 минут назад
        alert.timestamp = datetime.now() - timedelta(minutes=6)
        # Проверяем эскалацию
        escalated_alerts = manager.get_escalated_alerts()
        assert len(escalated_alerts) == 1
        assert escalated_alerts[0].alert_id == alert.alert_id
    def test_alert_grouping(self) -> None:
        """Тест группировки алертов."""
        manager = AlertManager()
        # Создаем алерты с одинаковым источником
        manager.create_alert("Alert 1", AlertSeverity.WARNING, "source1")
        manager.create_alert("Alert 2", AlertSeverity.ERROR, "source1")
        manager.create_alert("Alert 3", AlertSeverity.WARNING, "source2")
        # Группируем по источнику
        grouped_alerts = manager.get_alerts_grouped_by_source()
        assert "source1" in grouped_alerts
        assert "source2" in grouped_alerts
        assert len(grouped_alerts["source1"]) == 2
        assert len(grouped_alerts["source2"]) == 1
    def test_alert_search(self) -> None:
        """Тест поиска алертов."""
        manager = AlertManager()
        # Создаем алерты с разными сообщениями
        manager.create_alert("Database connection failed", AlertSeverity.ERROR, "database")
        manager.create_alert("API timeout", AlertSeverity.WARNING, "api")
        manager.create_alert("Memory usage high", AlertSeverity.WARNING, "system")
        # Ищем алерты по ключевому слову
        database_alerts = manager.search_alerts("database")
        assert len(database_alerts) == 1
        assert "Database" in database_alerts[0].message
        # Ищем алерты по источнику
        api_alerts = manager.search_alerts("api", search_in="source")
        assert len(api_alerts) == 1
        assert api_alerts[0].source == "api"
    def test_alert_export(self) -> None:
        """Тест экспорта алертов."""
        manager = AlertManager()
        # Создаем алерты
        manager.create_alert("Test alert 1", AlertSeverity.WARNING, "source1")
        manager.create_alert("Test alert 2", AlertSeverity.ERROR, "source2")
        # Экспортируем в JSON
        json_data = manager.export_alerts(format="json")
        assert isinstance(json_data, str)
        assert "Test alert 1" in json_data
        assert "Test alert 2" in json_data
        # Экспортируем в CSV
        csv_data = manager.export_alerts(format="csv")
        assert isinstance(csv_data, str)
        assert "Test alert 1" in csv_data
        assert "Test alert 2" in csv_data
    def test_performance_with_many_alerts(self) -> None:
        """Тест производительности с большим количеством алертов."""
        manager = AlertManager()
        import time
        start_time = time.time()
        # Создаем много алертов
        for i in range(1000):
            manager.create_alert(f"Alert {i}", AlertSeverity.INFO, f"source{i}")
        end_time = time.time()
        duration = end_time - start_time
        # Создание 1000 алертов должно занимать менее 1 секунды
        assert duration < 1.0
        assert len(manager.get_alerts()) == 1000
    def test_memory_usage_with_alerts(self) -> None:
        """Тест использования памяти с алертами."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        manager = AlertManager()
        # Создаем много алертов с метаданными
        for i in range(10000):
            manager.create_alert(
                f"Alert {i}",
                AlertSeverity.INFO,
                f"source{i}",
                metadata={"index": i, "data": "test" * 100}
            )
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024
class TestAlertRule:
    """Тесты для AlertRule."""
    def test_alert_rule_init(self) -> None:
        """Тест инициализации AlertRule."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            message="Test rule message",
            source="test_source"
        )
        assert rule.name == "test_rule"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.message == "Test rule message"
        assert rule.source == "test_source"
    def test_alert_rule_evaluate_true(self) -> None:
        """Тест оценки правила, возвращающего True."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            message="Test rule message",
            source="test_source"
        )
        result = rule.evaluate()
        assert result is True
    def test_alert_rule_evaluate_false(self) -> None:
        """Тест оценки правила, возвращающего False."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda: False,
            severity=AlertSeverity.WARNING,
            message="Test rule message",
            source="test_source"
        )
        result = rule.evaluate()
        assert result is False
    def test_alert_rule_with_parameters(self) -> None:
        """Тест правила с параметрами."""
        def condition_with_params(param1: int, param2: str) -> bool:
            return param1 > 10 and param2 == "test"
        rule = AlertRule(
            name="test_rule",
            condition=condition_with_params,
            severity=AlertSeverity.WARNING,
            message="Test rule message",
            source="test_source"
        )
        # Оцениваем с параметрами
        result = rule.evaluate(15, "test")
        assert result is True
        result = rule.evaluate(5, "test")
        assert result is False
    def test_alert_rule_get_metadata(self) -> None:
        """Тест получения метаданных правила."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            message="Test rule message",
            source="test_source"
        )
        metadata = rule.get_metadata()
        assert metadata["name"] == "test_rule"
        assert metadata["severity"] == "WARNING"
        assert metadata["message"] == "Test rule message"
        assert metadata["source"] == "test_source"
class TestAlertHandler:
    """Тесты для AlertHandler."""
    def test_alert_handler_init(self) -> None:
        """Тест инициализации AlertHandler."""
        handler = AlertHandler(
            name="test_handler",
            handle_func=lambda alert: None
        )
        assert handler.name == "test_handler"
        assert handler.handle_func is not None
    async def test_alert_handler_handle(self) -> None:
        """Тест обработки алерта."""
        handled_alerts = []
        def handle_func(alert) -> Any:
            handled_alerts.append(alert)
        handler = AlertHandler(
            name="test_handler",
            handle_func=handle_func
        )
        # Создаем тестовый алерт
        test_alert = Alert(
            alert_id="test-id",
            message="Test alert",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        # Обрабатываем алерт
        await handler.handle(test_alert)
        assert len(handled_alerts) == 1
        assert handled_alerts[0] == test_alert
    async def test_alert_handler_async_handle(self) -> None:
        """Тест асинхронной обработки алерта."""
        handled_alerts = []
        async def async_handle_func(alert) -> Any:
            handled_alerts.append(alert)
            await asyncio.sleep(0.01)  # Имитируем асинхронную работу
        handler = AlertHandler(
            name="test_handler",
            handle_func=async_handle_func
        )
        # Создаем тестовый алерт
        test_alert = Alert(
            alert_id="test-id",
            message="Test alert",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        # Обрабатываем алерт
        await handler.handle(test_alert)
        assert len(handled_alerts) == 1
        assert handled_alerts[0] == test_alert
class TestAlert:
    """Тесты для Alert."""
    def test_alert_init(self) -> None:
        """Тест инициализации Alert."""
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        assert alert.alert_id == "test-id"
        assert alert.message == "Test alert message"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "test_source"
        assert alert.timestamp is not None
        assert alert.acknowledged is False
        assert alert.resolved is False
    def test_alert_with_metadata(self) -> None:
        """Тест Alert с метаданными."""
        metadata = {"test_key": "test_value", "number": 42}
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now(),
            metadata=metadata
        )
        assert alert.metadata["test_key"] == "test_value"
        assert alert.metadata["number"] == 42
    def test_alert_with_exception(self) -> None:
        """Тест Alert с исключением."""
        exception = ValueError("Test error")
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.ERROR,
            source="test_source",
            timestamp=datetime.now(),
            exception=exception
        )
        assert alert.exception == exception
    def test_alert_acknowledge(self) -> None:
        """Тест подтверждения Alert."""
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        alert.acknowledge()
        assert alert.acknowledged is True
        assert alert.acknowledged_at is not None
    def test_alert_resolve(self) -> None:
        """Тест разрешения Alert."""
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        alert.resolve()
        assert alert.resolved is True
        assert alert.resolved_at is not None
    def test_alert_to_dict(self) -> None:
        """Тест преобразования Alert в словарь."""
        alert = Alert(
            alert_id="test-id",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now(),
            metadata={"test": "data"}
        )
        alert_dict = alert.to_dict()
        assert alert_dict["alert_id"] == "test-id"
        assert alert_dict["message"] == "Test alert message"
        assert alert_dict["severity"] == "WARNING"
        assert alert_dict["source"] == "test_source"
        assert "test" in alert_dict["metadata"]
class TestAlertSeverity:
    """Тесты для AlertSeverity."""
    def test_alert_severity_values(self) -> None:
        """Тест значений AlertSeverity."""
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.ERROR.value == "ERROR"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"
    def test_alert_severity_comparison(self) -> None:
        """Тест сравнения AlertSeverity."""
        assert AlertSeverity.INFO < AlertSeverity.WARNING
        assert AlertSeverity.WARNING < AlertSeverity.ERROR
        assert AlertSeverity.ERROR < AlertSeverity.CRITICAL
    def test_alert_severity_from_string(self) -> None:
        """Тест создания AlertSeverity из строки."""
        assert AlertSeverity.from_string("INFO") == AlertSeverity.INFO
        assert AlertSeverity.from_string("WARNING") == AlertSeverity.WARNING
        assert AlertSeverity.from_string("ERROR") == AlertSeverity.ERROR
        assert AlertSeverity.from_string("CRITICAL") == AlertSeverity.CRITICAL
    def test_alert_severity_invalid_string(self) -> None:
        """Тест обработки неверной строки для AlertSeverity."""
        with pytest.raises(ValueError):
            AlertSeverity.from_string("INVALID")
class TestGetAlertManager:
    """Тесты для функции get_alert_manager."""
    def test_get_alert_manager_default(self) -> None:
        """Тест получения менеджера алертов по умолчанию."""
        manager = get_alert_manager()
        assert isinstance(manager, AlertManager)
        assert manager.name == "default"
    def test_get_alert_manager_custom_name(self) -> None:
        """Тест получения менеджера алертов с пользовательским именем."""
        manager = get_alert_manager("custom_alerts")
        assert isinstance(manager, AlertManager)
        assert manager.name == "custom_alerts"
    def test_get_alert_manager_singleton(self) -> None:
        """Тест, что get_alert_manager возвращает тот же экземпляр для одного имени."""
        manager1 = get_alert_manager("singleton_test")
        manager2 = get_alert_manager("singleton_test")
        assert manager1 is manager2
    def test_get_alert_manager_different_names(self) -> None:
        """Тест, что разные имена возвращают разные экземпляры."""
        manager1 = get_alert_manager("manager1")
        manager2 = get_alert_manager("manager2")
        assert manager1 is not manager2
class TestAlertFunctions:
    """Тесты для функций работы с алертами."""
    @patch('infrastructure.monitoring.monitoring_alerts.get_alert_manager')
    def test_create_alert(self, mock_get_manager) -> None:
        """Тест функции create_alert."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_alert = Mock()
        mock_manager.create_alert.return_value = mock_alert
        alert = create_alert(
            message="Test alert",
            severity="WARNING",
            source="test_source"
        )
        mock_manager.create_alert.assert_called_once_with(
            message="Test alert",
            severity=AlertSeverity.WARNING,
            source="test_source"
        )
        assert alert == mock_alert
    @patch('infrastructure.monitoring.monitoring_alerts.get_alert_manager')
    def test_add_alert_handler(self, mock_get_manager) -> None:
        """Тест функции add_alert_handler."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        handler = Mock()
        add_alert_handler("WARNING", handler)
        mock_manager.add_alert_handler.assert_called_once_with(AlertSeverity.WARNING, handler)
    @patch('infrastructure.monitoring.monitoring_alerts.get_alert_manager')
    def test_resolve_alert(self, mock_get_manager) -> None:
        """Тест функции resolve_alert."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        resolve_alert("test-alert-id")
        mock_manager.resolve_alert.assert_called_once_with("test-alert-id")
    @patch('infrastructure.monitoring.monitoring_alerts.get_alert_manager')
    def test_get_alerts(self, mock_get_manager) -> None:
        """Тест функции get_alerts."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_alerts = [Mock(), Mock()]
        mock_manager.get_alerts.return_value = mock_alerts
        alerts = get_alerts(severity="WARNING", limit=10)
        mock_manager.get_alerts.assert_called_once_with(severity=AlertSeverity.WARNING, limit=10)
        assert alerts == mock_alerts
class TestAlertHandlerProtocol:
    """Тесты для протокола AlertHandlerProtocol."""
    def test_alert_manager_implements_protocol(self) -> None:
        """Тест, что AlertManager реализует AlertHandlerProtocol."""
        manager = AlertManager()
        # Проверяем наличие всех методов протокола
        assert hasattr(manager, 'create_alert')
        assert hasattr(manager, 'add_alert_rule')
        assert hasattr(manager, 'add_alert_handler')
        assert hasattr(manager, 'get_alerts')
        assert hasattr(manager, 'acknowledge_alert')
        assert hasattr(manager, 'resolve_alert')
        assert hasattr(manager, 'start_evaluation')
        assert hasattr(manager, 'stop_evaluation')
if __name__ == "__main__":
    pytest.main([__file__]) 
