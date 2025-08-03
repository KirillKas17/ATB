import asyncio
import pandas as pd
from typing import Dict, List


class CacheService:
    """Сервис кэширования для оптимизации производительности."""

    def __init__(self) -> None:
        self.volume_profiles: Dict[str, pd.DataFrame] = {}
        self.fractal_levels: Dict[str, Dict[str, List[float]]] = {}
        self.liquidity_zones: Dict[str, List[Dict]] = {}
        self.timestamps: Dict[str, float] = {}
        self.cache_ttl = 300.0  # 5 минут

    def clear(self) -> None:
        """Очистка кэша."""
        self.volume_profiles.clear()
        self.fractal_levels.clear()
        self.liquidity_zones.clear()
        self.timestamps.clear()

    def is_valid(self, key: str) -> bool:
        """Проверка валидности кэша."""
        if key not in self.timestamps:
            return False
        return asyncio.get_event_loop().time() - self.timestamps[key] < self.cache_ttl

    def update_timestamp(self, key: str) -> None:
        """Обновление временной метки."""
        self.timestamps[key] = asyncio.get_event_loop().time()
