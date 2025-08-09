"""
Unit тесты для timestamp.py.

Покрывает:
- Основной функционал Timestamp
- Валидацию данных
- Бизнес-логику временных операций
- Обработку ошибок
- Анализ временных интервалов
- Сериализацию и десериализацию
"""

import pytest
import dataclasses
from typing import Dict, Any, Union
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from domain.value_objects.timestamp import Timestamp


class TestTimestamp:
    """Тесты для Timestamp."""

    @pytest.fixture
    def sample_timestamp(self) -> Timestamp:
        """Тестовая временная метка."""
        return Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    @pytest.fixture
    def future_timestamp(self) -> Timestamp:
        """Будущая временная метка."""
        return Timestamp(value=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    @pytest.fixture
    def past_timestamp(self) -> Timestamp:
        """Прошлая временная метка."""
        return Timestamp(value=datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_timestamp_creation_from_datetime(self, sample_timestamp):
        """Тест создания временной метки из datetime."""
        assert sample_timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_creation_from_string(self):
        """Тест создания временной метки из строки."""
        timestamp = Timestamp(value="2023-01-01T12:00:00+00:00")
        assert timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_creation_from_unix_timestamp(self):
        """Тест создания временной метки из Unix timestamp."""
        unix_time = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        timestamp = Timestamp(value=unix_time)
        assert timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_creation_from_unix_ms(self):
        """Тест создания временной метки из Unix timestamp в миллисекундах."""
        unix_ms = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        timestamp = Timestamp(value=unix_ms)
        assert timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_creation_from_float(self):
        """Тест создания временной метки из float."""
        unix_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        timestamp = Timestamp(value=unix_time)
        assert timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_creation_without_timezone(self):
        """Тест создания временной метки без timezone."""
        dt = datetime(2023, 1, 1, 12, 0, 0)  # Без timezone
        timestamp = Timestamp(value=dt)
        assert timestamp.value.tzinfo == timezone.utc  # Должен быть установлен UTC

    def test_timestamp_validation_invalid_type(self):
        """Тест валидации неверного типа."""
        with pytest.raises(ValueError, match="Unsupported timestamp format"):
            Timestamp(value={"invalid": "type"})

    def test_timestamp_properties(self, sample_timestamp):
        """Тест свойств временной метки."""
        assert isinstance(sample_timestamp.value, datetime)
        assert sample_timestamp.value.tzinfo == timezone.utc

    def test_timestamp_equality(self, sample_timestamp):
        """Тест равенства временных меток."""
        same_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        different_timestamp = Timestamp(value=datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc))

        assert sample_timestamp == same_timestamp
        assert sample_timestamp != different_timestamp
        assert sample_timestamp != "not a timestamp"

    def test_timestamp_hash_equality(self, sample_timestamp):
        """Тест хеширования для равенства."""
        same_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        assert hash(sample_timestamp) == hash(same_timestamp)

    def test_timestamp_comparison_operations(self, sample_timestamp, future_timestamp, past_timestamp):
        """Тест операций сравнения."""
        # Меньше
        assert past_timestamp < sample_timestamp
        assert sample_timestamp < future_timestamp

        # Больше
        assert future_timestamp > sample_timestamp
        assert sample_timestamp > past_timestamp

        # Меньше или равно
        assert past_timestamp <= sample_timestamp
        assert sample_timestamp <= sample_timestamp

        # Больше или равно
        assert future_timestamp >= sample_timestamp
        assert sample_timestamp >= sample_timestamp

    def test_timestamp_comparison_errors(self, sample_timestamp):
        """Тест ошибок сравнения."""
        with pytest.raises(TypeError, match="Can only compare Timestamp with Timestamp"):
            sample_timestamp < "2023-01-01"

    def test_timestamp_conversion_methods(self, sample_timestamp):
        """Тест методов конвертации."""
        # To ISO
        iso_string = sample_timestamp.to_iso()
        assert iso_string == "2023-01-01T12:00:00+00:00"

        # To Unix
        unix_time = sample_timestamp.to_unix()
        expected_unix = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert unix_time == expected_unix

        # To Unix MS
        unix_ms = sample_timestamp.to_unix_ms()
        expected_unix_ms = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert unix_ms == expected_unix_ms

        # To datetime
        dt = sample_timestamp.to_datetime()
        assert dt == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_time_checks(self, sample_timestamp, future_timestamp, past_timestamp):
        """Тест проверок времени."""
        # Будущее время
        assert future_timestamp.is_future() is True
        assert sample_timestamp.is_future() is False
        assert past_timestamp.is_future() is False

        # Прошлое время
        assert past_timestamp.is_past() is True
        assert sample_timestamp.is_past() is False
        assert future_timestamp.is_past() is False

        # Текущее время
        now_timestamp = Timestamp.now()
        assert now_timestamp.is_now() is True
        assert now_timestamp.is_now(tolerance_seconds=1) is True

    def test_timestamp_arithmetic_operations(self, sample_timestamp):
        """Тест арифметических операций."""
        # Добавление времени
        result = sample_timestamp.add_seconds(3600)  # +1 час
        expected = datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.add_minutes(30)  # +30 минут
        expected = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.add_hours(2)  # +2 часа
        expected = datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.add_days(1)  # +1 день
        expected = datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

        # Вычитание времени
        result = sample_timestamp.subtract_seconds(3600)  # -1 час
        expected = datetime(2023, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.subtract_minutes(30)  # -30 минут
        expected = datetime(2023, 1, 1, 11, 30, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.subtract_hours(2)  # -2 часа
        expected = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

        result = sample_timestamp.subtract_days(1)  # -1 день
        expected = datetime(2022, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
        assert result.value == expected

    def test_timestamp_time_difference(self, sample_timestamp):
        """Тест вычисления разности времени."""
        later_timestamp = Timestamp(value=datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc))

        # Разность в секундах
        diff_seconds = later_timestamp.time_difference(sample_timestamp)
        assert diff_seconds == 3600.0  # 1 час = 3600 секунд

        # Разность в минутах
        diff_minutes = later_timestamp.time_difference_minutes(sample_timestamp)
        assert diff_minutes == 60.0  # 60 минут

        # Разность в часах
        diff_hours = later_timestamp.time_difference_hours(sample_timestamp)
        assert diff_hours == 1.0  # 1 час

        # Разность в днях
        next_day_timestamp = Timestamp(value=datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc))
        diff_days = next_day_timestamp.time_difference_days(sample_timestamp)
        assert diff_days == 1.0  # 1 день

    def test_timestamp_same_time_checks(self, sample_timestamp):
        """Тест проверок одинакового времени."""
        same_day_timestamp = Timestamp(value=datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc))
        same_hour_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc))
        same_minute_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 0, 30, tzinfo=timezone.utc))

        # Одинаковый день
        assert sample_timestamp.is_same_day(same_day_timestamp) is True

        # Одинаковый час
        assert sample_timestamp.is_same_hour(same_hour_timestamp) is True

        # Одинаковая минута
        assert sample_timestamp.is_same_minute(same_minute_timestamp) is True

    def test_timestamp_weekend_weekday_checks(self, sample_timestamp):
        """Тест проверок выходных и рабочих дней."""
        # 1 января 2023 - воскресенье
        assert sample_timestamp.is_weekend() is True
        assert sample_timestamp.is_weekday() is False

        # 2 января 2023 - понедельник
        monday_timestamp = Timestamp(value=datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc))
        assert monday_timestamp.is_weekend() is False
        assert monday_timestamp.is_weekday() is True

    def test_timestamp_trading_hours(self, sample_timestamp):
        """Тест проверок торговых часов."""
        # 12:00 - в торговых часах
        assert sample_timestamp.is_trading_hours() is True

        # 18:00 - вне торговых часов
        evening_timestamp = Timestamp(value=datetime(2023, 1, 1, 18, 0, 0, tzinfo=timezone.utc))
        assert evening_timestamp.is_trading_hours() is False

        # Пользовательские торговые часы
        assert sample_timestamp.is_trading_hours(start_hour=10, end_hour=14) is True
        assert sample_timestamp.is_trading_hours(start_hour=14, end_hour=16) is False

    def test_timestamp_rounding(self, sample_timestamp):
        """Тест округления временных меток."""
        # Округление до минуты
        rounded_minute = sample_timestamp.round_to_minute()
        assert rounded_minute.value.second == 0

        # Округление до часа
        rounded_hour = sample_timestamp.round_to_hour()
        assert rounded_hour.value.minute == 0
        assert rounded_hour.value.second == 0

        # Округление до дня
        rounded_day = sample_timestamp.round_to_day()
        assert rounded_day.value.hour == 0
        assert rounded_day.value.minute == 0
        assert rounded_day.value.second == 0

    def test_timestamp_min_max(self, sample_timestamp, future_timestamp, past_timestamp):
        """Тест методов min и max."""
        min_timestamp = sample_timestamp.min(past_timestamp)
        assert min_timestamp == past_timestamp

        max_timestamp = sample_timestamp.max(future_timestamp)
        assert max_timestamp == future_timestamp

    def test_timestamp_between_check(self, sample_timestamp, future_timestamp, past_timestamp):
        """Тест проверки нахождения между двумя временными метками."""
        assert sample_timestamp.is_between(past_timestamp, future_timestamp) is True
        assert past_timestamp.is_between(sample_timestamp, future_timestamp) is False

    def test_timestamp_trading_session(self, sample_timestamp):
        """Тест определения торговой сессии."""
        session = sample_timestamp.get_trading_session()
        assert isinstance(session, str)
        assert len(session) > 0

    def test_timestamp_market_status(self, sample_timestamp):
        """Тест статуса рынка."""
        # Рынок открыт в торговые часы
        assert sample_timestamp.is_market_open() is True

        # Время до открытия рынка
        time_until_open = sample_timestamp.get_time_until_market_open()
        assert time_until_open is not None

        # Время до закрытия рынка
        time_until_close = sample_timestamp.get_time_until_market_close()
        assert time_until_close is not None

    def test_timestamp_age_checks(self, sample_timestamp):
        """Тест проверок возраста."""
        # Возраст в секундах
        age_seconds = sample_timestamp.get_age_in_seconds()
        assert age_seconds > 0

        # Недавняя временная метка
        recent_timestamp = Timestamp.now()
        assert recent_timestamp.is_recent(max_age_seconds=300) is True

        # Устаревшая временная метка
        assert sample_timestamp.is_expired(max_age_seconds=3600) is True

    def test_timestamp_to_dict(self, sample_timestamp):
        """Тест сериализации в словарь."""
        result = sample_timestamp.to_dict()

        assert result["value"] == "2023-01-01T12:00:00+00:00"
        assert result["type"] == "Timestamp"

    def test_timestamp_from_dict(self, sample_timestamp):
        """Тест десериализации из словаря."""
        data = {"value": "2023-01-01T12:00:00+00:00", "type": "Timestamp"}

        timestamp = Timestamp.from_dict(data)
        assert timestamp.value == sample_timestamp.value

    def test_timestamp_factory_methods(self):
        """Тест фабричных методов."""
        # Now
        now_timestamp = Timestamp.now()
        assert isinstance(now_timestamp.value, datetime)
        assert now_timestamp.value.tzinfo == timezone.utc

        # From ISO
        iso_timestamp = Timestamp.from_iso("2023-01-01T12:00:00+00:00")
        assert iso_timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # From Unix
        unix_time = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        unix_timestamp = Timestamp.from_unix(unix_time)
        assert unix_timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # From Unix MS
        unix_ms = int(datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        unix_ms_timestamp = Timestamp.from_unix_ms(unix_ms)
        assert unix_ms_timestamp.value == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # From datetime
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        datetime_timestamp = Timestamp.from_datetime(dt)
        assert datetime_timestamp.value == dt

    def test_timestamp_copy(self, sample_timestamp):
        """Тест копирования временной метки."""
        copied_timestamp = sample_timestamp.copy()
        assert copied_timestamp == sample_timestamp
        assert copied_timestamp is not sample_timestamp

    def test_timestamp_str_representation(self, sample_timestamp):
        """Тест строкового представления."""
        result = str(sample_timestamp)
        assert "2023-01-01T12:00:00+00:00" in result

    def test_timestamp_repr_representation(self, sample_timestamp):
        """Тест repr представления."""
        result = repr(sample_timestamp)
        assert "Timestamp" in result
        assert "2023-01-01T12:00:00+00:00" in result

    def test_timestamp_hash(self, sample_timestamp):
        """Тест хеширования временной метки."""
        hash_value = sample_timestamp.hash
        assert len(hash_value) == 32  # MD5 hex digest length
        assert isinstance(hash_value, str)

    def test_timestamp_validation(self, sample_timestamp):
        """Тест валидации временной метки."""
        assert sample_timestamp.validate() is True


class TestTimestampOperations:
    """Тесты операций с временными метками."""

    def test_timestamp_precision_handling(self):
        """Тест обработки точности."""
        # Временная метка с микросекундами
        dt = datetime(2023, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        timestamp = Timestamp(value=dt)
        assert timestamp.value == dt

    def test_timestamp_timezone_handling(self):
        """Тест обработки временных зон."""
        # Разные временные зоны
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        utc_timestamp = Timestamp(value=utc_dt)

        # Должен сохранить timezone
        assert utc_timestamp.value.tzinfo == timezone.utc

    def test_timestamp_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        original_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

        # Сериализация
        data = original_timestamp.to_dict()

        # Десериализация
        restored_timestamp = Timestamp.from_dict(data)

        # Проверка равенства
        assert restored_timestamp == original_timestamp
        assert restored_timestamp.value == original_timestamp.value


class TestTimestampTradingAnalysis:
    """Тесты анализа торговых сессий."""

    def test_trading_session_detection(self):
        """Тест определения торговых сессий."""
        # Утренняя сессия
        morning_timestamp = Timestamp(value=datetime(2023, 1, 2, 9, 30, 0, tzinfo=timezone.utc))
        session = morning_timestamp.get_trading_session()
        assert "morning" in session.lower() or "open" in session.lower()

        # Дневная сессия
        afternoon_timestamp = Timestamp(value=datetime(2023, 1, 2, 14, 30, 0, tzinfo=timezone.utc))
        session = afternoon_timestamp.get_trading_session()
        assert "afternoon" in session.lower() or "open" in session.lower()

        # Вечерняя сессия
        evening_timestamp = Timestamp(value=datetime(2023, 1, 2, 18, 30, 0, tzinfo=timezone.utc))
        session = evening_timestamp.get_trading_session()
        assert "closed" in session.lower() or "after" in session.lower()

    def test_market_open_status(self):
        """Тест статуса открытия рынка."""
        # Рынок открыт
        open_timestamp = Timestamp(value=datetime(2023, 1, 2, 10, 0, 0, tzinfo=timezone.utc))
        assert open_timestamp.is_market_open() is True

        # Рынок закрыт (выходной)
        weekend_timestamp = Timestamp(value=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc))
        assert weekend_timestamp.is_market_open() is False

        # Рынок закрыт (вне часов)
        closed_timestamp = Timestamp(value=datetime(2023, 1, 2, 18, 0, 0, tzinfo=timezone.utc))
        assert closed_timestamp.is_market_open() is False


class TestTimestampEdgeCases:
    """Тесты граничных случаев для временных меток."""

    def test_timestamp_minimum_values(self):
        """Тест минимальных значений."""
        # Unix timestamp 0
        min_timestamp = Timestamp.from_unix(0)
        assert min_timestamp.value == datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_maximum_values(self):
        """Тест максимальных значений."""
        # Большой Unix timestamp
        max_timestamp = Timestamp.from_unix(9999999999)
        assert max_timestamp.value > datetime(2000, 1, 1, tzinfo=timezone.utc)

    def test_timestamp_invalid_string_format(self):
        """Тест неверного формата строки."""
        with pytest.raises(ValueError):
            Timestamp(value="invalid-date-format")

    def test_timestamp_negative_unix(self):
        """Тест отрицательного Unix timestamp."""
        # Отрицательный Unix timestamp (до 1970 года)
        negative_timestamp = Timestamp.from_unix(-1000000)
        assert negative_timestamp.value < datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_timestamp_time_difference_edge_cases(self):
        """Тест граничных случаев разности времени."""
        # Одинаковые временные метки
        same_timestamp = Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        diff = same_timestamp.time_difference(same_timestamp)
        assert diff == 0.0

        # Очень большая разность
        future_timestamp = Timestamp(value=datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        past_timestamp = Timestamp(value=datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        diff = future_timestamp.time_difference(past_timestamp)
        assert diff > 0

    def test_timestamp_hash_collision_resistance(self):
        """Тест устойчивости к коллизиям хешей."""
        timestamps = [
            Timestamp(value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
            Timestamp(value=datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)),
            Timestamp(value=datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)),
            Timestamp(value=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
        ]

        hashes = [timestamp.hash for timestamp in timestamps]
        assert len(hashes) == len(set(hashes))  # Все хеши должны быть уникальными

    def test_timestamp_performance(self):
        """Тест производительности операций с временными метками."""
        import time

        timestamp = Timestamp.now()

        # Тест скорости арифметических операций
        start_time = time.time()
        for _ in range(1000):
            result = timestamp.add_seconds(1)
        end_time = time.time()

        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций
