# -*- coding: utf-8 -*-
"""Точные часы UTC для работы с временем в торговых сессиях."""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import ntplib
from loguru import logger

from domain.value_objects.timestamp import Timestamp


@dataclass
class TimeSyncInfo:
    """Информация о синхронизации времени."""

    is_synchronized: bool
    offset_ms: float
    last_sync_time: datetime
    ntp_server: str
    sync_interval_seconds: int
    drift_rate_ms_per_hour: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "is_synchronized": self.is_synchronized,
            "offset_ms": self.offset_ms,
            "last_sync_time": self.last_sync_time.isoformat(),
            "ntp_server": self.ntp_server,
            "sync_interval_seconds": self.sync_interval_seconds,
            "drift_rate_ms_per_hour": self.drift_rate_ms_per_hour,
        }


class UTCClock:
    """Точные часы UTC с синхронизацией по NTP."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "ntp_servers": ["pool.ntp.org", "time.nist.gov", "time.google.com"],
            "sync_interval_seconds": 3600,  # 1 час
            "max_offset_ms": 100,  # Максимальное допустимое смещение
            "enable_auto_sync": True,
            "timeout_seconds": 5,
        }

        # Состояние синхронизации
        self.sync_info = TimeSyncInfo(
            is_synchronized=False,
            offset_ms=0.0,
            last_sync_time=datetime.now(timezone.utc),
            ntp_server="",
            sync_interval_seconds=self.config["sync_interval_seconds"],
        )

        # NTP клиент
        self.ntp_client = ntplib.NTPClient()

        # Время последней синхронизации
        self._last_sync_attempt = datetime.now(timezone.utc)

        logger.info("UTCClock initialized")

    def now(self) -> Timestamp:
        """Получение текущего времени UTC с учетом смещения."""
        current_time = datetime.now(timezone.utc)

        # Применяем смещение если синхронизация активна
        if self.sync_info.is_synchronized:
            adjusted_time = current_time + timedelta(
                milliseconds=self.sync_info.offset_ms
            )
        else:
            adjusted_time = current_time

        return Timestamp(adjusted_time)

    def now_datetime(self) -> datetime:
        """Получение текущего времени как datetime."""
        return self.now().to_datetime()

    def now_unix(self) -> int:
        """Получение текущего времени как Unix timestamp."""
        return self.now().to_unix()

    def now_unix_ms(self) -> int:
        """Получение текущего времени как Unix timestamp в миллисекундах."""
        return self.now().to_unix_ms()

    def sync_with_ntp(self, force: bool = False) -> bool:
        """
        Синхронизация с NTP сервером.

        Args:
            force: Принудительная синхронизация

        Returns:
            True если синхронизация успешна
        """
        try:
            current_time = datetime.now(timezone.utc)

            # Проверяем необходимость синхронизации
            if not force:
                time_since_last_sync = (
                    current_time - self._last_sync_attempt
                ).total_seconds()
                if time_since_last_sync < self.config["sync_interval_seconds"]:
                    return self.sync_info.is_synchronized

            self._last_sync_attempt = current_time

            # Пробуем синхронизироваться с разными серверами
            for ntp_server in self.config["ntp_servers"]:
                try:
                    logger.debug(f"Attempting NTP sync with {ntp_server}")

                    response = self.ntp_client.request(
                        ntp_server, version=3, timeout=self.config["timeout_seconds"]
                    )

                    if response.offset is not None:
                        offset_ms = (
                            response.offset * 1000
                        )  # Конвертируем в миллисекунды

                        # Проверяем допустимость смещения
                        if abs(offset_ms) <= self.config["max_offset_ms"]:
                            self.sync_info.offset_ms = offset_ms
                            self.sync_info.is_synchronized = True
                            self.sync_info.last_sync_time = current_time
                            self.sync_info.ntp_server = ntp_server

                            logger.info(
                                f"NTP sync successful with {ntp_server}: "
                                f"offset={offset_ms:.2f}ms"
                            )
                            return True
                        else:
                            logger.warning(
                                f"NTP offset too large from {ntp_server}: "
                                f"{offset_ms:.2f}ms > {self.config['max_offset_ms']}ms"
                            )

                except Exception as e:
                    logger.debug(f"NTP sync failed with {ntp_server}: {e}")
                    continue

            # Если все серверы недоступны
            if not self.sync_info.is_synchronized:
                logger.warning("All NTP servers failed, using local time")
                self.sync_info.offset_ms = 0.0
                self.sync_info.is_synchronized = False

            return self.sync_info.is_synchronized

        except Exception as e:
            logger.error(f"Error during NTP sync: {e}")
            return False

    def get_sync_info(self) -> TimeSyncInfo:
        """Получение информации о синхронизации."""
        return self.sync_info

    def is_time_accurate(self) -> bool:
        """Проверка точности времени."""
        return (
            self.sync_info.is_synchronized
            and abs(self.sync_info.offset_ms) <= self.config["max_offset_ms"]
        )

    def get_time_drift(self) -> float:
        """Получение дрейфа времени в миллисекундах в час."""
        return self.sync_info.drift_rate_ms_per_hour

    def calculate_drift_rate(self) -> float:
        """Вычисление скорости дрейфа времени."""
        try:
            if not self.sync_info.is_synchronized:
                return 0.0

            # Вычисляем дрейф на основе времени с последней синхронизации
            current_time = datetime.now(timezone.utc)
            time_since_sync = (
                current_time - self.sync_info.last_sync_time
            ).total_seconds() / 3600  # в часах

            if time_since_sync <= 0:
                return 0.0

            # Предполагаем, что смещение увеличивается линейно
            drift_rate = self.sync_info.offset_ms / time_since_sync
            self.sync_info.drift_rate_ms_per_hour = drift_rate

            return drift_rate

        except Exception as e:
            logger.error(f"Error calculating drift rate: {e}")
            return 0.0

    def wait_until(self, target_time: Timestamp) -> None:
        """
        Ожидание до указанного времени.

        Args:
            target_time: Целевое время
        """
        while self.now() < target_time:
            time.sleep(0.001)  # Ждем 1 миллисекунду

    def wait_for_seconds(self, seconds: float) -> None:
        """
        Ожидание указанное количество секунд.

        Args:
            seconds: Количество секунд для ожидания
        """
        time.sleep(seconds)

    def format_time(
        self, timestamp: Timestamp, format_str: str = "%Y-%m-%d %H:%M:%S UTC"
    ) -> str:
        """
        Форматирование времени.

        Args:
            timestamp: Временная метка
            format_str: Строка формата

        Returns:
            Отформатированное время
        """
        return timestamp.to_datetime().strftime(format_str)

    def get_timezone_info(self) -> Dict[str, Any]:
        """Получение информации о временной зоне."""
        current_time = datetime.now(timezone.utc)

        return {
            "current_timezone": "UTC",
            "utc_offset": "+00:00",
            "is_dst": False,
            "timezone_name": "Coordinated Universal Time",
            "current_time": current_time.isoformat(),
            "timestamp": current_time.timestamp(),
        }

    def validate_timestamp(self, timestamp: Timestamp) -> bool:
        """
        Валидация временной метки.

        Args:
            timestamp: Временная метка для проверки

        Returns:
            True если метка валидна
        """
        try:
            # Проверяем, что время не в далеком будущем или прошлом
            current_time = self.now()
            time_diff_hours = abs(current_time.time_difference_hours(timestamp))

            # Допускаем разницу до 24 часов
            return time_diff_hours <= 24

        except Exception:
            return False

    def get_time_until_session(self, session_open_time: str) -> Optional[timedelta]:
        """
        Вычисление времени до открытия сессии.

        Args:
            session_open_time: Время открытия сессии в формате "HH:MM"

        Returns:
            Время до открытия сессии
        """
        try:
            from datetime import time as dt_time

            # Парсим время открытия
            hour, minute = map(int, session_open_time.split(":"))
            session_time = dt_time(hour, minute)

            current_time = self.now().to_datetime()
            current_time_only = current_time.time()

            # Вычисляем время до следующего открытия
            today = current_time.date()
            session_datetime = datetime.combine(
                today, session_time, tzinfo=timezone.utc
            )

            # Если время сессии уже прошло сегодня, берем завтра
            if current_time_only >= session_time:
                session_datetime = datetime.combine(
                    today + timedelta(days=1), session_time, tzinfo=timezone.utc
                )

            return session_datetime - current_time

        except Exception as e:
            logger.error(f"Error calculating time until session: {e}")
            return None


# Глобальный экземпляр часов
utc_clock = UTCClock()
