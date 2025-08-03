# Миграция из utils/timeframes.py
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def is_valid_timeframe(tf: str) -> bool:
    return tf in TIMEFRAMES
