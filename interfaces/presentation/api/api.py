"""
Основной API класс для торговой системы.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.services.trading_service import TradingService
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.protocols.repository_protocol import (PortfolioRepositoryProtocol,
                                                  TradingRepositoryProtocol)


class TradingAPI:
    """API для торговой системы."""

    def __init__(
        self,
        trading_service: TradingService,
        trading_repository: TradingRepositoryProtocol,
        portfolio_repository: PortfolioRepositoryProtocol,
        exchange_service: ExchangeProtocol,
    ):
        self.trading_service = trading_service
        self.trading_repository = trading_repository
        self.portfolio_repository = portfolio_repository
        self.exchange_service = exchange_service

        self.app = FastAPI(
            title="Syntra API",
            description="API для автоматической торговой системы",
            version="1.0.0",
        )

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Настройка middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Настройка маршрутов."""
        from .routes import portfolio_router, strategy_router, trading_router

        self.app.include_router(
            trading_router, prefix="/api/v1/trading", tags=["trading"]
        )
        self.app.include_router(
            portfolio_router, prefix="/api/v1/portfolio", tags=["portfolio"]
        )
        self.app.include_router(
            strategy_router, prefix="/api/v1/strategy", tags=["strategy"]
        )

        @self.app.get("/")
        async def root():
            return {"message": "Syntra API", "version": "1.0.0"}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

    def get_app(self) -> FastAPI:
        """Получить FastAPI приложение."""
        return self.app

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Запустить API сервер."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
