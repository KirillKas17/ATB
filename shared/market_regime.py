from typing import List


def detect_market_regime(prices: List[float], window: int = 20) -> str:
    """
    Улучшенное определение рыночного режима с нормализацией и анализом промежуточных движений.
    
    Args:
        prices: Список цен
        window: Окно для анализа
        
    Returns:
        str: Режим рынка ('volatile', 'uptrend', 'downtrend', 'sideways', 'unknown')
    """
    if len(prices) < window:
        return "unknown"
    
    # Получаем цены для анализа
    recent_prices = prices[-window:]
    
    # Нормализуем волатильность к среднему уровню цен
    avg_price = sum(recent_prices) / len(recent_prices)
    price_changes = [abs(recent_prices[i] - recent_prices[i - 1]) for i in range(1, len(recent_prices))]
    avg_change = sum(price_changes) / len(price_changes)
    
    # Адаптивный порог волатильности (2% от средней цены)
    volatility_threshold = avg_price * 0.02
    
    if avg_change > volatility_threshold:
        return "volatile"
    
    # Анализ тренда с учетом промежуточных движений
    start_price = recent_prices[0]
    end_price = recent_prices[-1]
    
    # Считаем процент движений в сторону тренда
    trend_moves = 0
    total_moves = len(recent_prices) - 1
    
    if end_price > start_price:
        # Проверяем восходящий тренд
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                trend_moves += 1
        trend_consistency = trend_moves / total_moves
        
        # Требуем минимум 60% движений в сторону тренда и значимое изменение цены
        price_change_percent = (end_price - start_price) / start_price
        if trend_consistency >= 0.6 and price_change_percent > 0.01:  # 1% минимальное движение
            return "uptrend"
            
    elif end_price < start_price:
        # Проверяем нисходящий тренд
        for i in range(1, len(recent_prices)):
            if recent_prices[i] < recent_prices[i-1]:
                trend_moves += 1
        trend_consistency = trend_moves / total_moves
        
        # Требуем минимум 60% движений в сторону тренда и значимое изменение цены
        price_change_percent = abs(end_price - start_price) / start_price
        if trend_consistency >= 0.6 and price_change_percent > 0.01:  # 1% минимальное движение
            return "downtrend"
    
    # Если не определили четкий тренд или волатильность - это боковик
    return "sideways"
