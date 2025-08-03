"""
Реализация сервиса метрик.
"""

from typing import Dict, List, Optional

from domain.exceptions import NetworkError


class PrometheusMetricsService:
    """Реализация сервиса метрик с Prometheus."""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        # Здесь будет инициализация подключения к Prometheus

    async def record_counter(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать счетчик."""
        try:
            # Реализация записи счетчика в Prometheus
            return True
        except Exception as e:
            raise NetworkError(f"Ошибка записи счетчика в Prometheus: {e}")

    async def record_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать gauge."""
        try:
            # Реализация записи gauge в Prometheus
            return True
        except Exception as e:
            raise NetworkError(f"Ошибка записи gauge в Prometheus: {e}")

    async def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать гистограмму."""
        try:
            # Реализация записи гистограммы в Prometheus
            return True
        except Exception as e:
            raise NetworkError(f"Ошибка записи гистограммы в Prometheus: {e}")

    async def get_metric(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Получить значение метрики."""
        try:
            # Реализация получения метрики из Prometheus
            return None
        except Exception as e:
            raise NetworkError(f"Ошибка получения метрики из Prometheus: {e}")


class InMemoryMetricsService:
    """In-memory реализация сервиса метрик."""

    def __init__(self) -> None:
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}

    async def record_counter(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать счетчик."""
        key = self._build_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
        return True

    async def record_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать gauge."""
        key = self._build_key(name, labels)
        self.gauges[key] = value
        return True

    async def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Записать гистограмму."""
        key = self._build_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        return True

    async def get_metric(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Получить значение метрики."""
        key = self._build_key(name, labels)

        if key in self.counters:
            return self.counters[key]
        elif key in self.gauges:
            return self.gauges[key]
        elif key in self.histograms:
            values = self.histograms[key]
            return sum(values) / len(values) if values else None

        return None

    def _build_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Построить ключ для метрики."""
        if labels:
            label_str = ",".join([f"{k}={v}" for k, v in labels.items()])
            return f"{name}_{label_str}"
        return name
