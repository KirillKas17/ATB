# Миграция из utils/fibonacci_tools.py
def calculate_fibonacci_levels(high: float, low: float) -> list:
    diff = high - low
    return [
        high,
        high - 0.236 * diff,
        high - 0.382 * diff,
        high - 0.5 * diff,
        high - 0.618 * diff,
        high - 0.786 * diff,
        low,
    ]
