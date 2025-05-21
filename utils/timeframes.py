from enum import Enum
from typing import Any, Dict


class TimeFrame(Enum):
    """Временные фреймы для торговли"""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN = "1M"

    @classmethod
    def get_seconds(cls, timeframe: "TimeFrame") -> int:
        """Получение количества секунд во фрейме"""
        seconds_map = {
            cls.M1: 60,
            cls.M5: 300,
            cls.M15: 900,
            cls.M30: 1800,
            cls.H1: 3600,
            cls.H4: 14400,
            cls.D1: 86400,
            cls.W1: 604800,
            cls.MN: 2592000,
        }
        return seconds_map[timeframe]

    @classmethod
    def get_milliseconds(cls, timeframe: "TimeFrame") -> int:
        """Получение количества миллисекунд во фрейме"""
        return cls.get_seconds(timeframe) * 1000

    @classmethod
    def from_string(cls, timeframe_str: str) -> "TimeFrame":
        """Создание TimeFrame из строки"""
        try:
            return cls(timeframe_str)
        except ValueError:
            raise ValueError(f"Invalid timeframe: {timeframe_str}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "value": self.value,
            "seconds": self.get_seconds(self),
            "milliseconds": self.get_milliseconds(self),
        }
