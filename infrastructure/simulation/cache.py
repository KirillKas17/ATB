from typing import Any, Dict


class SimulationCache:
    """Кэш для хранения промежуточных данных симуляции и бэктеста."""

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: Dict[Any, Any] = {}
        self._max_size = max_size

    def get(self, key: Any) -> Any:
        return self._cache.get(key)

    def set(self, key: Any, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
