"""
Утилиты кэширования для повышения производительности финансовых расчетов.
"""

import hashlib
import time
from decimal import Decimal
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union


class PerformanceCache:
    """Высокопроизводительный кэш для финансовых расчетов."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300) -> None:
        """
        Инициализация кэша.
        
        Args:
            max_size: Максимальный размер кэша
            ttl_seconds: Время жизни записи в секундах
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Статистика
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Генерация ключа кэша."""
        # Преобразуем Decimal в строку для хэширования
        serializable_args: List[Any] = []
        for arg in args:
            if isinstance(arg, Decimal):
                serializable_args.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                serializable_args.append(tuple(str(x) if isinstance(x, Decimal) else x for x in arg))
            else:
                serializable_args.append(arg)
        
        # Создаем уникальный ключ
        key_data = f"{func_name}:{serializable_args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Получение значения из кэша."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                return True, value
            else:
                # Запись устарела
                del self._cache[key]
        
        self.misses += 1
        return False, None
    
    def set(self, key: str, value: Any) -> None:
        """Сохранение значения в кэше."""
        # Очистка устаревших записей при превышении размера
        if len(self._cache) >= self.max_size:
            self._cleanup_expired()
            
            # Если всё ещё превышен размер, удаляем самые старые
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())
    
    def _cleanup_expired(self) -> None:
        """Очистка устаревших записей."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def clear(self) -> None:
        """Очистка всего кэша."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Получение статистики кэша."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }


# Глобальный экземпляр кэша
_global_cache = PerformanceCache()


def cached_calculation(ttl_seconds: int = 300, cache_instance: PerformanceCache = None) -> Callable:
    """
    Декоратор для кэширования результатов финансовых расчетов.
    
    Args:
        ttl_seconds: Время жизни кэша в секундах
        cache_instance: Экземпляр кэша (по умолчанию глобальный)
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _global_cache
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            # Генерируем ключ кэша
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # Проверяем кэш
            found, cached_result = cache.get(cache_key)
            if found:
                return cached_result
            
            # Вычисляем результат
            result = func(*args, **kwargs)
            
            # Сохраняем в кэш
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


@cached_calculation(ttl_seconds=600)  # Кэшируем на 10 минут
def cached_fibonacci_levels(high: Decimal, low: Decimal) -> Tuple[Decimal, ...]:
    """Кэшированный расчет уровней Фибоначчи."""
    from shared.math_utils import calculate_fibonacci_levels
    return tuple(calculate_fibonacci_levels(high, low))


@cached_calculation(ttl_seconds=300)  # Кэшируем на 5 минут
def cached_liquidity_impact(order_size: Decimal, orderbook_data: Tuple, side: str = "buy") -> Dict[str, Decimal]:
    """Кэшированный расчет влияния на ликвидность."""
    from shared.math_utils import calculate_liquidity_impact
    return calculate_liquidity_impact(order_size, list(orderbook_data), side)


def get_cache_stats() -> Dict[str, Union[int, float]]:
    """Получение статистики глобального кэша."""
    return _global_cache.get_stats()


def clear_global_cache() -> None:
    """Очистка глобального кэша."""
    _global_cache.clear()