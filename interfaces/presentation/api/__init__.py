"""
API интерфейс для торговой системы.
"""

from .api import TradingAPI
from .routes import portfolio_router, strategy_router, trading_router

__all__ = [
    "TradingAPI",
    "trading_router",
    "portfolio_router",
    "strategy_router",
]
