# Миграция из utils/fibonacci_tools.py
from typing import List

def calculate_fibonacci_levels(high: float, low: float) -> List[float]:
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
