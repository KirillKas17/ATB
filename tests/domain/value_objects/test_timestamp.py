"""Тесты для Timestamp value object."""

import dataclasses
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from domain.value_objects.timestamp import Timestamp


class TestTimestamp:
    """Тесты для класса Timestamp."""

    def test_timestamp_creation(self: "TestTimestamp") -> None:
        """Тест создания временной метки."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)
        assert ts.value == dt

    def test_timestamp_creation_without_timezone(self: "TestTimestamp") -> None:
        """Тест создания временной метки без timezone."""
        dt = datetime(2024, 1, 1, 12, 0, 0)  # Без timezone
        ts = Timestamp(dt)
        assert ts.value.tzinfo == timezone.utc
        assert ts.value.replace(tzinfo=None) == dt

    def test_timestamp_invalid_creation(self: "TestTimestamp") -> None:
        """Тест создания с невалидным типом."""
        with pytest.raises(ValueError, match="Timestamp value must be a datetime object"):
            Timestamp("2024-01-01")

    def test_timestamp_immutability(self: "TestTimestamp") -> None:
        """Тест неизменяемости."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)
        original_value = ts.value

        # Попытка изменения должна вызвать ошибку
        with pytest.raises(dataclasses.FrozenInstanceError):
            ts.value = datetime.now(timezone.utc)

        assert ts.value == original_value

    def test_timestamp_comparison(self: "TestTimestamp") -> None:
        """Тест сравнения временных меток."""
        ts1 = Timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        ts2 = Timestamp(datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc))
        ts3 = Timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

        assert ts1 < ts2
        assert ts2 > ts1
        assert ts1 == ts3
        assert ts1 <= ts3
        assert ts1 >= ts3

    def test_timestamp_comparison_with_non_timestamp(self: "TestTimestamp") -> None:
        """Тест сравнения с не-временными метками."""
        ts = Timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

        with pytest.raises(TypeError, match="Can only compare Timestamp with Timestamp"):
            _ = ts < datetime.now()

    def test_timestamp_string_representation(self: "TestTimestamp") -> None:
        """Тест строкового представления."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)
        assert str(ts) == dt.isoformat()

    def test_timestamp_repr_representation(self: "TestTimestamp") -> None:
        """Тест представления для отладки."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)
        assert repr(ts) == f"Timestamp({dt})"

    def test_timestamp_equality(self: "TestTimestamp") -> None:
        """Тест равенства."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        ts1 = Timestamp(dt1)
        ts2 = Timestamp(dt1)
        ts3 = Timestamp(dt2)

        assert ts1 == ts2
        assert ts1 != ts3
        assert ts1 != "2024-01-01"

    def test_timestamp_conversion_methods(self: "TestTimestamp") -> None:
        """Тест методов преобразования."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)

        assert ts.to_iso() == dt.isoformat()
        assert ts.to_unix() == int(dt.timestamp())
        assert ts.to_unix_ms() == int(dt.timestamp() * 1000)
        assert ts.to_datetime() == dt
        assert ts.to_float() == dt.timestamp()
        assert ts.to_decimal() == Decimal(str(dt.timestamp()))

    def test_timestamp_time_checks(self: "TestTimestamp") -> None:
        """Тест проверок времени."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        ts_past = Timestamp(past)
        ts_future = Timestamp(future)
        ts_now = Timestamp(now)

        assert ts_past.is_past()
        assert not ts_past.is_future()

        assert ts_future.is_future()
        assert not ts_future.is_past()

        assert ts_now.is_now()
        assert ts_now.is_now(tolerance_seconds=5)

    def test_timestamp_addition_methods(self: "TestTimestamp") -> None:
        """Тест методов добавления времени."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)

        # Добавление секунд
        result = ts.add_seconds(60)
        expected = dt + timedelta(seconds=60)
        assert result.value == expected

        # Добавление минут
        result = ts.add_minutes(30)
        expected = dt + timedelta(minutes=30)
        assert result.value == expected

        # Добавление часов
        result = ts.add_hours(2)
        expected = dt + timedelta(hours=2)
        assert result.value == expected

        # Добавление дней
        result = ts.add_days(1)
        expected = dt + timedelta(days=1)
        assert result.value == expected

    def test_timestamp_subtraction_methods(self: "TestTimestamp") -> None:
        """Тест методов вычитания времени."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)

        # Вычитание секунд
        result = ts.subtract_seconds(60)
        expected = dt - timedelta(seconds=60)
        assert result.value == expected

        # Вычитание минут
        result = ts.subtract_minutes(30)
        expected = dt - timedelta(minutes=30)
        assert result.value == expected

        # Вычитание часов
        result = ts.subtract_hours(2)
        expected = dt - timedelta(hours=2)
        assert result.value == expected

        # Вычитание дней
        result = ts.subtract_days(1)
        expected = dt - timedelta(days=1)
        assert result.value == expected

    def test_timestamp_difference_methods(self: "TestTimestamp") -> None:
        """Тест методов расчета разности времени."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        ts1 = Timestamp(dt1)
        ts2 = Timestamp(dt2)

        # Разность в секундах
        diff_seconds = ts2.time_difference(ts1)
        assert diff_seconds == 3600.0

        # Разность в минутах
        diff_minutes = ts2.time_difference_minutes(ts1)
        assert diff_minutes == 60.0

        # Разность в часах
        diff_hours = ts2.time_difference_hours(ts1)
        assert diff_hours == 1.0

        # Разность в днях
        diff_days = ts2.time_difference_days(ts1)
        assert diff_days == 1.0 / 24.0

    def test_timestamp_difference_with_non_timestamp(self: "TestTimestamp") -> None:
        """Тест расчета разности с не-временной меткой."""
        ts = Timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

        with pytest.raises(TypeError, match="Can only calculate time difference with Timestamp"):
            ts.time_difference("2024-01-01")

    def test_timestamp_serialization(self: "TestTimestamp") -> None:
        """Тест сериализации."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamp(dt)

        data = ts.to_dict()
        assert data == {"value": dt.isoformat()}

        restored = Timestamp.from_dict(data)
        assert restored == ts

    def test_timestamp_class_methods(self: "TestTimestamp") -> None:
        """Тест классовых методов."""
        # Создание текущего времени
        now_ts = Timestamp.now()
        assert isinstance(now_ts.value, datetime)
        assert now_ts.value.tzinfo == timezone.utc

        # Создание из ISO строки
        iso_string = "2024-01-01T12:00:00+00:00"
        ts1 = Timestamp.from_iso(iso_string)
        assert ts1.value.isoformat() == iso_string

        # Создание из Unix timestamp
        unix_ts = 1704110400  # 2024-01-01 12:00:00 UTC
        ts2 = Timestamp.from_unix(unix_ts)
        assert ts2.to_unix() == unix_ts

        # Создание из Unix timestamp в миллисекундах
        unix_ms = 1704110400000
        ts3 = Timestamp.from_unix_ms(unix_ms)
        assert ts3.to_unix_ms() == unix_ms

        # Создание из datetime объекта
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts4 = Timestamp.from_datetime(dt)
        assert ts4.value == dt

    def test_timestamp_hash(self: "TestTimestamp") -> None:
        """Тест хеширования."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        ts1 = Timestamp(dt1)
        ts2 = Timestamp(dt1)
        ts3 = Timestamp(dt2)

        timestamp_set = {ts1, ts2, ts3}
        assert len(timestamp_set) == 2  # ts1 и ts2 одинаковые
        assert ts1 in timestamp_set
