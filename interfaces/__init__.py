"""
Interfaces Layer - интерфейсы пользователя.
"""

# Безопасные импорты - импортируем только при необходимости
__all__ = [
    "EntityDashboard",
    "TradingAPI", 
    "TradingCLI",
    "get_trading_api",
    "get_trading_cli", 
    "get_entity_dashboard",
]

def get_trading_api():
    """Безопасный импорт TradingAPI."""
    try:
        from .presentation.api.api import TradingAPI
        return TradingAPI
    except ImportError as e:
        print(f"Warning: TradingAPI not available: {e}")
        return None

def get_trading_cli():
    """Безопасный импорт TradingCLI."""
    try:
        from .presentation.cli.cli import TradingCLI
        return TradingCLI
    except ImportError as e:
        print(f"Warning: TradingCLI not available: {e}")
        return None

def get_entity_dashboard():
    """Безопасный импорт EntityDashboard."""
    try:
        from .presentation.dashboard.app import EntityDashboard
        return EntityDashboard
    except ImportError as e:
        print(f"Warning: EntityDashboard not available: {e}")
        return None

# Для обратной совместимости - попытаемся импортировать если возможно
try:
    from .presentation.api.api import TradingAPI
except ImportError:
    TradingAPI = None

try:
    from .presentation.cli.cli import TradingCLI
except ImportError:
    TradingCLI = None

try:
    from .presentation.dashboard.app import EntityDashboard
except ImportError:
    EntityDashboard = None
