from typing import Any, Dict


def generate_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Генерация признаков из рыночных данных

    Args:
        data: Словарь с рыночными данными

    Returns:
        Dict[str, float]: Словарь с признаками
    """
    features = {}

    # Базовые признаки
    if "close" in data:
        features["close"] = float(data["close"])
    if "volume" in data:
        features["volume"] = float(data["volume"])
    if "high" in data:
        features["high"] = float(data["high"])
    if "low" in data:
        features["low"] = float(data["low"])

    # Производные признаки
    if all(k in data for k in ["high", "low", "close"]):
        features["body"] = data["close"] - data["open"]
        features["upper_shadow"] = data["high"] - max(data["open"], data["close"])
        features["lower_shadow"] = min(data["open"], data["close"]) - data["low"]

    # Волатильность
    if "volatility" in data:
        features["volatility"] = float(data["volatility"])

    # Тренд
    if "trend" in data:
        features["trend"] = 1.0 if data["trend"] == "up" else -1.0

    # Режим рынка
    if "market_regime" in data:
        features["market_regime"] = 1.0 if data["market_regime"] == "bull" else -1.0

    return features
