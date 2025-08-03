"""
Unit тесты для EventManager.
Тестирует управление событиями, включая публикацию, подписку,
обработку и маршрутизацию событий в системе.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from infrastructure.core.event_manager import EventManager

class TestEventManager:
    """Тесты для EventManager."""
    @pytest.fixture
    def event_manager(self) -> EventManager:
        """Фикстура для EventManager."""
        return EventManager()
    @pytest.fixture
    def sample_event(self) -> dict:
        """Фикстура с тестовым событием."""
        return {
            "id": "event_001",
            "type": "order_created",
            "source": "order_manager",
            "timestamp": datetime.now(),
            "data": {
                "order_id": "order_123",
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": Decimal("0.1"),
                "price": Decimal("50000.0")
            },
            "priority": "normal",
            "metadata": {
                "user_id": "user_001",
                "session_id": "session_123"
            }
        }
    @pytest.fixture
    def sample_events_list(self) -> list:
        """Фикстура со списком тестовых событий."""
        return [
            {
                "id": "event_001",
                "type": "order_created",
                "source": "order_manager",
                "timestamp": datetime.now() - timedelta(minutes=5),
                "data": {"order_id": "order_123"},
                "priority": "normal"
            },
            {
                "id": "event_002",
                "type": "position_opened",
                "source": "position_manager",
                "timestamp": datetime.now() - timedelta(minutes=3),
                "data": {"position_id": "pos_456"},
                "priority": "high"
            },
            {
                "id": "event_003",
                "type": "signal_generated",
                "source": "signal_processor",
                "timestamp": datetime.now() - timedelta(minutes=1),
                "data": {"signal_id": "signal_789"},
                "priority": "normal"
            }
        ]
    def test_initialization(self, event_manager: EventManager) -> None:
        """Тест инициализации менеджера событий."""
        assert event_manager is not None
        assert hasattr(event_manager, 'event_bus')
        assert hasattr(event_manager, 'event_handlers')
        assert hasattr(event_manager, 'event_filters')
    @pytest.mark.asyncio
    async def test_publish_event(self, event_manager: EventManager, sample_event: dict) -> None:
        """Тест публикации события."""
        # Публикация события
        publish_result = await event_manager.publish_event(sample_event)
        # Проверки
        assert publish_result is not None
        assert "success" in publish_result
        assert "event_id" in publish_result
        assert "publish_time" in publish_result
        assert "subscribers_notified" in publish_result
        # Проверка типов данных
        assert isinstance(publish_result["success"], bool)
        assert isinstance(publish_result["event_id"], str)
        assert isinstance(publish_result["publish_time"], datetime)
        assert isinstance(publish_result["subscribers_notified"], int)
    @pytest.mark.asyncio
    async def test_subscribe_to_event(self, event_manager: EventManager) -> None:
        """Тест подписки на событие."""
        # Мок обработчика событий
        handler = AsyncMock()
        # Подписка на событие
        subscription_result = await event_manager.subscribe_to_event(
            "order_created", handler
        )
        # Проверки
        assert subscription_result is not None
        assert "subscription_id" in subscription_result
        assert "event_type" in subscription_result
        assert "handler" in subscription_result
        assert "subscribe_time" in subscription_result
        # Проверка типов данных
        assert isinstance(subscription_result["subscription_id"], str)
        assert isinstance(subscription_result["event_type"], str)
        assert subscription_result["handler"] == handler
        assert isinstance(subscription_result["subscribe_time"], datetime)
    @pytest.mark.asyncio
    async def test_unsubscribe_from_event(self, event_manager: EventManager) -> None:
        """Тест отписки от события."""
        # Мок обработчика событий
        handler = AsyncMock()
        # Подписка на событие
        subscription = await event_manager.subscribe_to_event("order_created", handler)
        # Отписка от события
        unsubscribe_result = await event_manager.unsubscribe_from_event(
            subscription["subscription_id"]
        )
        # Проверки
        assert unsubscribe_result is not None
        assert "success" in unsubscribe_result
        assert "unsubscribe_time" in unsubscribe_result
        assert "subscription_id" in unsubscribe_result
        # Проверка типов данных
        assert isinstance(unsubscribe_result["success"], bool)
        assert isinstance(unsubscribe_result["unsubscribe_time"], datetime)
        assert isinstance(unsubscribe_result["subscription_id"], str)
    @pytest.mark.asyncio
    async def test_handle_event(self, event_manager: EventManager, sample_event: dict) -> None:
        """Тест обработки события."""
        # Мок обработчика событий
        handler = AsyncMock()
        # Подписка на событие
        await event_manager.subscribe_to_event("order_created", handler)
        # Публикация события
        await event_manager.publish_event(sample_event)
        # Проверка, что обработчик был вызван
        await asyncio.sleep(0.1)  # Небольшая задержка для асинхронной обработки
        handler.assert_called_once()
    def test_filter_events(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест фильтрации событий."""
        # Фильтрация событий
        filtered_events = event_manager.filter_events(
            sample_events_list,
            filters={
                "type": "order_created",
                "priority": "high",
                "source": "order_manager"
            }
        )
        # Проверки
        assert filtered_events is not None
        assert isinstance(filtered_events, list)
        assert len(filtered_events) <= len(sample_events_list)
    def test_get_event_statistics(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест получения статистики событий."""
        # Получение статистики
        statistics = event_manager.get_event_statistics(sample_events_list)
        # Проверки
        assert statistics is not None
        assert "total_events" in statistics
        assert "events_by_type" in statistics
        assert "events_by_source" in statistics
        assert "events_by_priority" in statistics
        assert "avg_events_per_minute" in statistics
        # Проверка типов данных
        assert isinstance(statistics["total_events"], int)
        assert isinstance(statistics["events_by_type"], dict)
        assert isinstance(statistics["events_by_source"], dict)
        assert isinstance(statistics["events_by_priority"], dict)
        assert isinstance(statistics["avg_events_per_minute"], float)
        # Проверка логики
        assert statistics["total_events"] == len(sample_events_list)
    def test_validate_event(self, event_manager: EventManager, sample_event: dict) -> None:
        """Тест валидации события."""
        # Валидация события
        validation_result = event_manager.validate_event(sample_event)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "validation_score" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["validation_score"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["validation_score"] <= 1.0
    def test_prioritize_events(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест приоритизации событий."""
        # Приоритизация событий
        prioritized_events = event_manager.prioritize_events(sample_events_list)
        # Проверки
        assert prioritized_events is not None
        assert isinstance(prioritized_events, list)
        assert len(prioritized_events) == len(sample_events_list)
        # Проверка, что события отсортированы по приоритету
        priorities = ["critical", "high", "normal", "low"]
        current_priority_index = 0
        for event in prioritized_events:
            event_priority_index = priorities.index(event["priority"])
            assert event_priority_index >= current_priority_index
            current_priority_index = event_priority_index
    def test_deduplicate_events(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест дедупликации событий."""
        # Добавление дубликатов
        events_with_duplicates = sample_events_list + [sample_events_list[0]]
        # Дедупликация событий
        deduplicated_events = event_manager.deduplicate_events(events_with_duplicates)
        # Проверки
        assert deduplicated_events is not None
        assert isinstance(deduplicated_events, list)
        assert len(deduplicated_events) <= len(events_with_duplicates)
        # Проверка, что дубликаты удалены
        event_ids = [event["id"] for event in deduplicated_events]
        assert len(event_ids) == len(set(event_ids))
    def test_get_event_history(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест получения истории событий."""
        # Получение истории событий
        history = event_manager.get_event_history(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        # Проверки
        assert history is not None
        assert isinstance(history, list)
        assert len(history) >= 0
    def test_analyze_event_patterns(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест анализа паттернов событий."""
        # Анализ паттернов
        pattern_analysis = event_manager.analyze_event_patterns(sample_events_list)
        # Проверки
        assert pattern_analysis is not None
        assert "event_patterns" in pattern_analysis
        assert "correlation_analysis" in pattern_analysis
        assert "anomaly_detection" in pattern_analysis
        assert "pattern_recommendations" in pattern_analysis
        # Проверка типов данных
        assert isinstance(pattern_analysis["event_patterns"], dict)
        assert isinstance(pattern_analysis["correlation_analysis"], dict)
        assert isinstance(pattern_analysis["anomaly_detection"], dict)
        assert isinstance(pattern_analysis["pattern_recommendations"], list)
    def test_route_event(self, event_manager: EventManager, sample_event: dict) -> None:
        """Тест маршрутизации события."""
        # Маршрутизация события
        routing_result = event_manager.route_event(sample_event)
        # Проверки
        assert routing_result is not None
        assert "routing_path" in routing_result
        assert "target_handlers" in routing_result
        assert "routing_time" in routing_result
        assert "routing_success" in routing_result
        # Проверка типов данных
        assert isinstance(routing_result["routing_path"], list)
        assert isinstance(routing_result["target_handlers"], list)
        assert isinstance(routing_result["routing_time"], datetime)
        assert isinstance(routing_result["routing_success"], bool)
    def test_batch_process_events(self, event_manager: EventManager, sample_events_list: list) -> None:
        """Тест пакетной обработки событий."""
        # Пакетная обработка событий
        batch_result = event_manager.batch_process_events(sample_events_list)
        # Проверки
        assert batch_result is not None
        assert "processed_events" in batch_result
        assert "processing_time" in batch_result
        assert "success_rate" in batch_result
        assert "errors" in batch_result
        # Проверка типов данных
        assert isinstance(batch_result["processed_events"], int)
        assert isinstance(batch_result["processing_time"], float)
        assert isinstance(batch_result["success_rate"], float)
        assert isinstance(batch_result["errors"], list)
        # Проверка диапазона
        assert 0.0 <= batch_result["success_rate"] <= 1.0
    def test_error_handling(self, event_manager: EventManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            event_manager.validate_event(None)
        with pytest.raises(ValueError):
            event_manager.filter_events(None, {})
    def test_edge_cases(self, event_manager: EventManager) -> None:
        """Тест граничных случаев."""
        # Тест с пустыми событиями
        empty_events: list = []
        filtered_events = event_manager.filter_events(empty_events, {})
        assert filtered_events == []
        # Тест с очень большим событием
        large_event = {
            "id": "large_event",
            "type": "test",
            "source": "test",
            "timestamp": datetime.now(),
            "data": {"large_field": "x" * 10000},
            "priority": "normal"
        }
        validation_result = event_manager.validate_event(large_event)
        assert validation_result["is_valid"] is True
    def test_cleanup(self, event_manager: EventManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        event_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert event_manager.event_bus == {}
        assert event_manager.event_handlers == {}
        assert event_manager.event_filters == {} 
