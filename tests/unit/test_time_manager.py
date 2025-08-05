"""
Unit тесты для TimeManager.
Тестирует управление временем, включая синхронизацию,
форматирование, расчеты и обработку временных меток.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
from datetime import datetime, timedelta, timezone
from infrastructure.core.time_manager import TimeManager
class TestTimeManager:
    """Тесты для TimeManager."""
    @pytest.fixture
    def time_manager(self) -> TimeManager:
        """Фикстура для TimeManager."""
        return TimeManager()
    @pytest.fixture
    def sample_timestamp(self) -> datetime:
        """Фикстура с тестовой временной меткой."""
        return datetime.now(timezone.utc)
    def test_initialization(self, time_manager: TimeManager) -> None:
        """Тест инициализации менеджера времени."""
        assert time_manager is not None
        assert hasattr(time_manager, 'time_sources')
        assert hasattr(time_manager, 'time_formatters')
        assert hasattr(time_manager, 'time_validators')
    def test_get_current_time(self, time_manager: TimeManager) -> None:
        """Тест получения текущего времени."""
        # Получение текущего времени
        current_time = time_manager.get_current_time()
        # Проверки
        assert current_time is not None
        assert "timestamp" in current_time
        assert "timezone" in current_time
        assert "time_source" in current_time
        assert "accuracy" in current_time
        # Проверка типов данных
        assert isinstance(current_time["timestamp"], datetime)
        assert isinstance(current_time["timezone"], str)
        assert isinstance(current_time["time_source"], str)
        assert isinstance(current_time["accuracy"], float)
        # Проверка диапазона
        assert current_time["accuracy"] >= 0.0
    def test_synchronize_time(self, time_manager: TimeManager) -> None:
        """Тест синхронизации времени."""
        # Синхронизация времени
        sync_result = time_manager.synchronize_time()
        # Проверки
        assert sync_result is not None
        assert "success" in sync_result
        assert "sync_time" in sync_result
        assert "time_offset" in sync_result
        assert "sync_source" in sync_result
        # Проверка типов данных
        assert isinstance(sync_result["success"], bool)
        assert isinstance(sync_result["sync_time"], datetime)
        assert isinstance(sync_result["time_offset"], float)
        assert isinstance(sync_result["sync_source"], str)
    def test_format_timestamp(self, time_manager: TimeManager, sample_timestamp: datetime) -> None:
        """Тест форматирования временной метки."""
        # Форматирование временной метки
        format_result = time_manager.format_timestamp(
            sample_timestamp,
            format_type="iso"
        )
        # Проверки
        assert format_result is not None
        assert "formatted_time" in format_result
        assert "format_type" in format_result
        assert "original_timestamp" in format_result
        # Проверка типов данных
        assert isinstance(format_result["formatted_time"], str)
        assert isinstance(format_result["format_type"], str)
        assert isinstance(format_result["original_timestamp"], datetime)
    def test_parse_timestamp(self, time_manager: TimeManager) -> None:
        """Тест парсинга временной метки."""
        # Тестовая строка времени
        time_string = "2023-12-01T10:30:00Z"
        # Парсинг временной метки
        parse_result = time_manager.parse_timestamp(time_string)
        # Проверки
        assert parse_result is not None
        assert "timestamp" in parse_result
        assert "timezone" in parse_result
        assert "parse_success" in parse_result
        # Проверка типов данных
        assert isinstance(parse_result["timestamp"], datetime)
        assert isinstance(parse_result["timezone"], str)
        assert isinstance(parse_result["parse_success"], bool)
    def test_calculate_time_difference(self, time_manager: TimeManager) -> None:
        """Тест расчета разности времени."""
        # Создание двух временных меток
        time1 = datetime.now(timezone.utc)
        time.sleep(0.1)  # Небольшая задержка
        time2 = datetime.now(timezone.utc)
        # Расчет разности времени
        diff_result = time_manager.calculate_time_difference(time1, time2)
        # Проверки
        assert diff_result is not None
        assert "time_difference" in diff_result
        assert "difference_seconds" in diff_result
        assert "difference_minutes" in diff_result
        assert "difference_hours" in diff_result
        # Проверка типов данных
        assert isinstance(diff_result["time_difference"], timedelta)
        assert isinstance(diff_result["difference_seconds"], float)
        assert isinstance(diff_result["difference_minutes"], float)
        assert isinstance(diff_result["difference_hours"], float)
        # Проверка логики
        assert diff_result["difference_seconds"] > 0
    def test_add_time_duration(self, time_manager: TimeManager, sample_timestamp: datetime) -> None:
        """Тест добавления длительности времени."""
        # Добавление длительности
        duration = timedelta(hours=2, minutes=30)
        add_result = time_manager.add_time_duration(sample_timestamp, duration)
        # Проверки
        assert add_result is not None
        assert "result_timestamp" in add_result
        assert "original_timestamp" in add_result
        assert "added_duration" in add_result
        # Проверка типов данных
        assert isinstance(add_result["result_timestamp"], datetime)
        assert isinstance(add_result["original_timestamp"], datetime)
        assert isinstance(add_result["added_duration"], timedelta)
        # Проверка логики
        expected_time = sample_timestamp + duration
        assert add_result["result_timestamp"] == expected_time
    def test_subtract_time_duration(self, time_manager: TimeManager, sample_timestamp: datetime) -> None:
        """Тест вычитания длительности времени."""
        # Вычитание длительности
        duration = timedelta(hours=1, minutes=15)
        subtract_result = time_manager.subtract_time_duration(sample_timestamp, duration)
        # Проверки
        assert subtract_result is not None
        assert "result_timestamp" in subtract_result
        assert "original_timestamp" in subtract_result
        assert "subtracted_duration" in subtract_result
        # Проверка типов данных
        assert isinstance(subtract_result["result_timestamp"], datetime)
        assert isinstance(subtract_result["original_timestamp"], datetime)
        assert isinstance(subtract_result["subtracted_duration"], timedelta)
        # Проверка логики
        expected_time = sample_timestamp - duration
        assert subtract_result["result_timestamp"] == expected_time
    def test_convert_timezone(self, time_manager: TimeManager, sample_timestamp: datetime) -> None:
        """Тест конвертации часового пояса."""
        # Конвертация часового пояса
        convert_result = time_manager.convert_timezone(
            sample_timestamp,
            target_timezone="America/New_York"
        )
        # Проверки
        assert convert_result is not None
        assert "converted_timestamp" in convert_result
        assert "original_timestamp" in convert_result
        assert "source_timezone" in convert_result
        assert "target_timezone" in convert_result
        # Проверка типов данных
        assert isinstance(convert_result["converted_timestamp"], datetime)
        assert isinstance(convert_result["original_timestamp"], datetime)
        assert isinstance(convert_result["source_timezone"], str)
        assert isinstance(convert_result["target_timezone"], str)
    def test_validate_timestamp(self, time_manager: TimeManager, sample_timestamp: datetime) -> None:
        """Тест валидации временной метки."""
        # Валидация временной метки
        validation_result = time_manager.validate_timestamp(sample_timestamp)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "timestamp_info" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["timestamp_info"], dict)
        # Проверка логики
        assert validation_result["is_valid"] is True
    def test_get_time_statistics(self, time_manager: TimeManager) -> None:
        """Тест получения статистики времени."""
        # Получение статистики времени
        statistics = time_manager.get_time_statistics()
        # Проверки
        assert statistics is not None
        assert "current_time" in statistics
        assert "time_sources" in statistics
        assert "time_accuracy" in statistics
        assert "timezone_info" in statistics
        # Проверка типов данных
        assert isinstance(statistics["current_time"], datetime)
        assert isinstance(statistics["time_sources"], list)
        assert isinstance(statistics["time_accuracy"], float)
        assert isinstance(statistics["timezone_info"], dict)
    def test_calculate_business_days(self, time_manager: TimeManager) -> None:
        """Тест расчета рабочих дней."""
        # Создание диапазона дат
        start_date = datetime(2023, 12, 1)
        end_date = datetime(2023, 12, 31)
        # Расчет рабочих дней
        business_days_result = time_manager.calculate_business_days(start_date, end_date)
        # Проверки
        assert business_days_result is not None
        assert "business_days" in business_days_result
        assert "total_days" in business_days_result
        assert "weekends" in business_days_result
        assert "holidays" in business_days_result
        # Проверка типов данных
        assert isinstance(business_days_result["business_days"], int)
        assert isinstance(business_days_result["total_days"], int)
        assert isinstance(business_days_result["weekends"], int)
        assert isinstance(business_days_result["holidays"], int)
        # Проверка логики
        assert business_days_result["business_days"] <= business_days_result["total_days"]
    def test_get_market_hours(self, time_manager: TimeManager) -> None:
        """Тест получения часов работы рынка."""
        # Получение часов работы рынка
        market_hours = time_manager.get_market_hours("crypto")
        # Проверки
        assert market_hours is not None
        assert "market_type" in market_hours
        assert "is_open" in market_hours
        assert "open_time" in market_hours
        assert "close_time" in market_hours
        assert "timezone" in market_hours
        # Проверка типов данных
        assert isinstance(market_hours["market_type"], str)
        assert isinstance(market_hours["is_open"], bool)
        assert isinstance(market_hours["open_time"], str)
        assert isinstance(market_hours["close_time"], str)
        assert isinstance(market_hours["timezone"], str)
    def test_calculate_execution_time(self, time_manager: TimeManager) -> None:
        """Тест расчета времени выполнения."""
        # Создание функции для тестирования
    def test_function() -> None:
            time.sleep(0.1)
            return "test_result"
        # Расчет времени выполнения
        execution_result = time_manager.calculate_execution_time(test_function)
        # Проверки
        assert execution_result is not None
        assert "execution_time" in execution_result
        assert "function_result" in execution_result
        assert "start_time" in execution_result
        assert "end_time" in execution_result
        # Проверка типов данных
        assert isinstance(execution_result["execution_time"], float)
        assert execution_result["function_result"] == "test_result"
        assert isinstance(execution_result["start_time"], datetime)
        assert isinstance(execution_result["end_time"], datetime)
        # Проверка логики
        assert execution_result["execution_time"] >= 0.1
    def test_schedule_task(self, time_manager: TimeManager) -> None:
        """Тест планирования задачи."""
        # Планирование задачи
        task_time = datetime.now(timezone.utc) + timedelta(seconds=1)
        schedule_result = time_manager.schedule_task(
            task_time,
            "test_task",
            {"task_data": "test_value"}
        )
        # Проверки
        assert schedule_result is not None
        assert "success" in schedule_result
        assert "task_id" in schedule_result
        assert "scheduled_time" in schedule_result
        assert "task_info" in schedule_result
        # Проверка типов данных
        assert isinstance(schedule_result["success"], bool)
        assert isinstance(schedule_result["task_id"], str)
        assert isinstance(schedule_result["scheduled_time"], datetime)
        assert isinstance(schedule_result["task_info"], dict)
    def test_error_handling(self, time_manager: TimeManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            time_manager.format_timestamp(None, "iso")
        with pytest.raises(ValueError):
            time_manager.calculate_time_difference(None, None)
    def test_edge_cases(self, time_manager: TimeManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень старым временем
        old_time = datetime(1900, 1, 1)
        validation_result = time_manager.validate_timestamp(old_time)
        assert validation_result["is_valid"] is True
        # Тест с очень будущим временем
        future_time = datetime(2100, 12, 31)
        validation_result = time_manager.validate_timestamp(future_time)
        assert validation_result["is_valid"] is True
        # Тест с нулевой длительностью
        zero_duration = timedelta(0)
        add_result = time_manager.add_time_duration(datetime.now(), zero_duration)
        assert add_result["result_timestamp"] == add_result["original_timestamp"]
    def test_cleanup(self, time_manager: TimeManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        time_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert time_manager.time_sources == {}
        assert time_manager.time_formatters == {}
        assert time_manager.time_validators == {} 
