from typing import List


def detect_market_regime(prices: List[float], window: int = 20) -> str:
    """Определение рыночного режима: тренд, флэт, высокая волатильность"""
    if len(prices) < window:
        return "unknown"
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    avg_diff = sum(diffs[-window:]) / window
    if avg_diff > 2:
        return "volatile"
    elif prices[-1] > prices[-window]:
        return "uptrend"
    elif prices[-1] < prices[-window]:
        return "downtrend"
    else:
        return "sideways"
