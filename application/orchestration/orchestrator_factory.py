"""
Factory для создания TradingOrchestrator с необходимыми зависимостями.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
from decimal import Decimal

from application.orchestration.trading_orchestrator import TradingOrchestrator
from shared.models.config import ApplicationConfig, create_default_config
from domain.interfaces.strategy_registry import StrategyRegistryProtocol
from domain.interfaces.risk_manager import RiskManagerProtocol
from domain.interfaces.market_data import MarketDataProtocol
from domain.interfaces.sentiment_analyzer import SentimentAnalyzerProtocol
from domain.interfaces.portfolio_manager import PortfolioManagerProtocol
from domain.interfaces.evolution_manager import EvolutionManagerProtocol


class MockStrategyRegistry:
    """Mock реализация StrategyRegistryProtocol."""
    
    async def initialize(self) -> None:
        """Инициализация реестра стратегий."""
        pass
    
    async def register_strategy(self, strategy_id: str, strategy_class, config: Dict[str, Any]) -> bool:
        return True
    
    async def get_active_strategies(self) -> List[Any]:
        return []
    
    async def get_all_strategies(self) -> Dict[str, Any]:
        return {}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья реестра стратегий."""
        return {"status": "healthy", "active_strategies": 0}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass


class MockRiskManager:
    """Mock реализация RiskManagerProtocol."""
    
    async def initialize(self) -> None:
        """Инициализация риск-менеджера."""
        pass
    
    async def assess_trade_risk(self, symbol: str, quantity: Decimal, side: str, price: Optional[Decimal] = None) -> Dict[str, Any]:
        return {"risk_level": "LOW", "approved": True}
    
    async def check_position_limits(self, symbol: str, quantity: Decimal) -> bool:
        return True
    
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        return {"var_95": 0.05, "max_drawdown": 0.02}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья риск-менеджера."""
        return {"status": "healthy", "risk_level": "LOW"}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass


class MockMarketData:
    """Mock реализация MarketDataProtocol."""
    
    async def initialize(self) -> None:
        """Инициализация провайдера рыночных данных."""
        pass
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": "50000.00",
            "volume": "1000.0",
            "timestamp": "2025-01-01T00:00:00Z"
        }
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return {
            "bids": [["49999.0", "1.0"], ["49998.0", "2.0"]],
            "asks": [["50001.0", "1.0"], ["50002.0", "2.0"]]
        }
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Получение рыночных данных для символа."""
        return {
            "symbol": symbol,
            "price": "50000.00",
            "bid": "49999.00",
            "ask": "50001.00",
            "volume": "1000.0",
            "change_24h": "2.5",
            "timestamp": "2025-08-03T16:58:00Z"
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья провайдера рыночных данных."""
        return {"status": "healthy", "connection": "active"}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass


class MockSentimentAnalyzer:
    """Mock реализация SentimentAnalyzerProtocol."""
    
    async def initialize(self) -> None:
        """Инициализация анализатора настроений."""
        pass
    
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.7}
    
    async def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Анализ настроений для символа."""
        return {
            "symbol": symbol,
            "sentiment": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.75,
            "sources": ["news", "social"],
            "timestamp": "2025-08-03T16:58:00Z"
        }
    
    async def get_fear_greed_index(self) -> float:
        return 50.0  # Нейтральный
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья анализатора настроений."""
        return {"status": "healthy", "data_sources": "active"}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass


class MockPortfolioManager:
    """Mock реализация PortfolioManagerProtocol."""
    
    async def initialize(self) -> None:
        """Инициализация портфолио менеджера."""
        pass
    
    async def get_total_balance(self) -> Dict[str, Decimal]:
        return {"USDT": Decimal("10000.0"), "BTC": Decimal("0.5")}
    
    async def get_portfolio_value(self, base_currency: str = "USDT") -> Decimal:
        return Decimal("35000.0")
    
    async def rebalance_portfolio(self, target_allocation: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Ребалансировка портфеля."""
        return {"status": "completed", "changes": {}}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья портфолио менеджера."""
        return {"status": "healthy", "portfolio_value": "35000.0"}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass


class MockEvolutionManager:
    """Mock реализация EvolutionManagerProtocol."""
    
    async def initialize(self) -> None:
        pass
    
    async def evolve_strategies(self, generation_count: int = 10, population_size: int = 50) -> List[Dict[str, Any]]:
        return []
    
    async def perform_evolution_cycle(self) -> Dict[str, Any]:
        """Выполнение цикла эволюции."""
        return {"generation": 1, "best_fitness": 0.5, "population_size": 10}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья эволюционного менеджера."""
        return {"status": "healthy", "generation": 1, "population_size": 10}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        pass
    
    @property
    def is_running(self) -> bool:
        return False


class MockOrderManagement:
    """Mock реализация для order management use case."""
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"order_id": "mock_order_123", "status": "FILLED"}


class MockPositionManagement:
    """Mock реализация для position management use case."""
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        return []


class MockRiskManagement:
    """Mock реализация для risk management use case."""
    
    async def assess_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"risk_approved": True}


class MockTradingPairManagement:
    """Mock реализация для trading pair management use case."""
    
    async def get_active_pairs(self) -> List[str]:
        return ["BTCUSDT", "ETHUSDT"]
    
    async def get_trading_pairs(self, request) -> Any:
        """Получение торговых пар."""
        # Mock response объект
        class MockTradingPair:
            def __init__(self, symbol: str):
                self.symbol = symbol
                self.base_asset = symbol[:-4] if len(symbol) > 4 else symbol[:3]
                self.quote_asset = symbol[-4:] if len(symbol) > 4 else "USDT"
                self.is_active = True
        
        class MockResponse:
            def __init__(self):
                self.success = True
                self.trading_pairs = [
                    MockTradingPair("BTCUSDT"),
                    MockTradingPair("ETHUSDT"), 
                    MockTradingPair("ADAUSDT"),
                    MockTradingPair("DOTUSDT")
                ]
        
        return MockResponse()


def create_trading_orchestrator(config: Optional[ApplicationConfig] = None) -> TradingOrchestrator:
    """
    Создает TradingOrchestrator с mock зависимостями.
    
    Args:
        config: Конфигурация приложения (опционально)
        
    Returns:
        Настроенный TradingOrchestrator
    """
    if config is None:
        config = create_default_config()
    
    # Создаем mock зависимости
    strategy_registry = MockStrategyRegistry()
    risk_manager = MockRiskManager()
    market_data = MockMarketData()
    sentiment_analyzer = MockSentimentAnalyzer()
    portfolio_manager = MockPortfolioManager()
    evolution_manager = MockEvolutionManager()
    
    # Use cases
    order_use_case = MockOrderManagement()
    position_use_case = MockPositionManagement()
    risk_use_case = MockRiskManagement()
    trading_pair_use_case = MockTradingPairManagement()
    
    return TradingOrchestrator(
        config=config,
        strategy_registry=strategy_registry,
        risk_manager=risk_manager,
        market_data=market_data,
        sentiment_analyzer=sentiment_analyzer,
        portfolio_manager=portfolio_manager,
        evolution_manager=evolution_manager,
        order_use_case=order_use_case,
        position_use_case=position_use_case,
        risk_use_case=risk_use_case,
        trading_pair_use_case=trading_pair_use_case
    )