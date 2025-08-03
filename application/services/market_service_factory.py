"""
Фабрика для создания специализированных рыночных сервисов.
"""

from typing import Dict, Any

from domain.repositories.market_repository import MarketRepository

from .market_analysis_service import MarketAnalysisService
from .market_data_service import MarketDataService
from .market_service import MarketService
from .technical_analysis_service import TechnicalAnalysisService


class MarketServiceFactory:
    """Фабрика для создания рыночных сервисов."""

    def __init__(self, market_repository: MarketRepository):
        self.market_repository = market_repository
        self._services_cache: Dict[str, Any] = {}

    def create_market_service(self) -> MarketService:
        """Создает основной рыночный сервис."""
        if "market_service" not in self._services_cache:
            technical_analysis_service = self.create_technical_analysis_service()
            self._services_cache["market_service"] = MarketService(
                self.market_repository,
                technical_analysis_service
            )
        return self._services_cache["market_service"]

    def create_market_data_service(self) -> MarketDataService:
        """Создает сервис для работы с рыночными данными."""
        if "market_data_service" not in self._services_cache:
            self._services_cache["market_data_service"] = MarketDataService(
                self.market_repository
            )
        return self._services_cache["market_data_service"]

    def create_technical_analysis_service(self) -> TechnicalAnalysisService:
        """Создает сервис для технического анализа."""
        if "technical_analysis_service" not in self._services_cache:
            self._services_cache["technical_analysis_service"] = (
                TechnicalAnalysisService(self.market_repository)
            )
        return self._services_cache["technical_analysis_service"]

    def create_market_analysis_service(self) -> MarketAnalysisService:
        """Создает сервис для анализа рынка."""
        if "market_analysis_service" not in self._services_cache:
            self._services_cache["market_analysis_service"] = MarketAnalysisService(
                self.market_repository
            )
        return self._services_cache["market_analysis_service"]

    def get_service(self, service_type: str):
        """Получает сервис по типу."""
        service_map = {
            "market": self.create_market_service,
            "data": self.create_market_data_service,
            "technical": self.create_technical_analysis_service,
            "analysis": self.create_market_analysis_service,
        }
        if service_type not in service_map:
            raise ValueError(f"Unknown service type: {service_type}")
        return service_map[service_type]()

    def clear_cache(self):
        """Очищает кэш сервисов."""
        self._services_cache.clear()
