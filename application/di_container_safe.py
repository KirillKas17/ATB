"""
Safe dependency injection container with protected imports
"""

from safe_import_wrapper import safe_import

# Core components that should work
from dependency_injector import containers, providers

# Safe imports for potentially problematic components
TrendStrategy = safe_import("infrastructure.strategies.trend_strategy", "TrendStrategy")
SidewaysStrategy = safe_import("infrastructure.strategies.sideways_strategy", "SidewaysStrategy") 
AdaptiveStrategyGenerator = safe_import("infrastructure.strategies.adaptive.adaptive_strategy_generator", "AdaptiveStrategyGenerator")
VolatilityStrategy = safe_import("infrastructure.strategies.volatility_strategy", "VolatilityStrategy")
PairsTradingStrategy = safe_import("infrastructure.strategies.pairs_trading_strategy", "PairsTradingStrategy")

# Service locator with safe defaults
class SafeContainer(containers.DeclarativeContainer):
    """Safe container that handles missing components gracefully"""
    
    # Configuration
    config = providers.Configuration()
    
    # Core services - use safe implementations for testing
    from application.safe_services import SafeTradingService, SafeRiskService, SafeMarketService
    trading_service = providers.Singleton(SafeTradingService)
    
    risk_service = providers.Singleton(SafeRiskService)
    market_service = providers.Singleton(SafeMarketService)


def get_safe_service_locator():
    """Get a safe service locator that won't crash on missing imports"""
    container = SafeContainer()
    container.config.from_dict({
        "trading": {
            "enabled": True,
            "test_mode": True
        },
        "risk": {
            "max_exposure": 0.1
        }
    })
    return container