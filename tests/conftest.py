#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Центральная конфигурация pytest и общие фикстуры для всех тестов.
"""

import pytest
import asyncio
import sys
from decimal import Decimal
from typing import Dict, Any, Generator, AsyncGenerator, List
from unittest.mock import AsyncMock, Mock
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.position import Position, PositionSide
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy, StrategyType
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.entities.trading_pair import TradingPair
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.protocols.ml_protocol import MLProtocol
from domain.protocols.repository_protocol import RepositoryProtocol
from infrastructure.market_profiles.storage.market_maker_storage import MarketMakerStorage
from infrastructure.market_profiles.storage.pattern_memory_repository import PatternMemoryRepository
from infrastructure.market_profiles.storage.behavior_history_repository import BehaviorHistoryRepository
from infrastructure.market_profiles.analysis.pattern_analyzer import PatternAnalyzer
from infrastructure.market_profiles.analysis.similarity_calculator import SimilarityCalculator
from infrastructure.market_profiles.analysis.success_rate_analyzer import SuccessRateAnalyzer
from infrastructure.market_profiles.models.storage_config import StorageConfig
from infrastructure.market_profiles.models.analysis_config import AnalysisConfig
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType,
    PatternResult, PatternOutcome, PatternMemory
)
from domain.type_definitions.market_maker_types import (
    BookPressure, VolumeDelta, PriceReaction, SpreadChange,
    OrderImbalance, LiquidityDepth, TimeDuration, VolumeConcentration,
    PriceVolatility, MarketMicrostructure, Confidence, Accuracy,
    AverageReturn, SuccessCount, TotalCount
)
from infrastructure.external_services.bybit_client import BybitClient


# ======================= PYTEST CONFIGURATION =======================

def pytest_configure(config):
    """Конфигурация pytest."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "asyncio: marks tests as async")


@pytest.fixture(scope="session")
def event_loop():
    """Создаем event loop для всей сессии тестов."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ======================= COMMON VALUE OBJECTS =======================

@pytest.fixture
def usd_currency() -> Currency:
    """Фикстура USD валюты."""
    return Currency("USD")


@pytest.fixture
def btc_currency() -> Currency:
    """Фикстура BTC валюты."""
    return Currency("BTC")


@pytest.fixture
def eth_currency() -> Currency:
    """Фикстура ETH валюты."""
    return Currency("ETH")


@pytest.fixture
def sample_price(usd_currency: Currency) -> Price:
    """Фикстура базовой цены BTC."""
    return Price(value=Decimal("45000.00"), currency=usd_currency)


@pytest.fixture
def sample_volume(btc_currency: Currency) -> Volume:
    """Фикстура базового объема BTC."""
    return Volume(value=Decimal("0.001"), currency=btc_currency)


@pytest.fixture
def sample_timestamp() -> Timestamp:
    """Фикстура базового timestamp."""
    return Timestamp.now()


# ======================= MOCK EXCHANGE CLIENTS =======================

@pytest.fixture
def mock_bybit_client() -> AsyncMock:
    """Мок Bybit клиента с базовыми методами."""
    client = AsyncMock(spec=BybitClient)
    
    # Настраиваем стандартные ответы
    client.get_account_balance.return_value = {
        "USDT": {"total": Decimal("10000.00"), "available": Decimal("9500.00")},
        "BTC": {"total": Decimal("0.1"), "available": Decimal("0.1")},
        "ETH": {"total": Decimal("3.0"), "available": Decimal("3.0")}
    }
    
    client.get_ticker.return_value = {
        "symbol": "BTCUSDT",
        "price": Decimal("45000.00"),
        "volume": Decimal("1000.0"),
        "bid": Decimal("44999.50"),
        "ask": Decimal("45000.50"),
        "timestamp": 1640995200000
    }
    
    client.place_limit_order.return_value = {
        "order_id": "test_order_123",
        "status": "pending",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": Decimal("0.001"),
        "price": Decimal("45000.00")
    }
    
    client.place_market_order.return_value = {
        "order_id": "test_market_order_456",
        "status": "pending",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": Decimal("0.001")
    }
    
    client.get_order_status.return_value = {
        "order_id": "test_order_123",
        "status": "FILLED",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "filled_quantity": Decimal("0.001"),
        "average_price": Decimal("45000.50")
    }
    
    client.get_positions.return_value = []
    
    client.cancel_order.return_value = {
        "order_id": "test_order_123",
        "status": "CANCELLED"
    }
    
    return client


@pytest.fixture
def mock_binance_client() -> AsyncMock:
    """Мок Binance клиента."""
    client = AsyncMock(spec=BybitClient)
    
    # Slightly different prices for arbitrage testing
    client.get_ticker.return_value = {
        "symbol": "BTCUSDT",
        "price": Decimal("44980.00"),
        "volume": Decimal("1200.0"),
        "bid": Decimal("44979.50"),
        "ask": Decimal("44980.50"),
        "timestamp": 1640995200000
    }
    
    client.place_limit_order.return_value = {
        "order_id": "binance_order_789",
        "status": "pending"
    }
    
    return client


@pytest.fixture
def mock_okx_client() -> AsyncMock:
    """Мок OKX клиента."""
    client = AsyncMock(spec=BybitClient)
    
    client.get_ticker.return_value = {
        "symbol": "BTCUSDT",
        "price": Decimal("45020.00"),
        "volume": Decimal("800.0"),
        "bid": Decimal("45019.50"),
        "ask": Decimal("45020.50"),
        "timestamp": 1640995200000
    }
    
    client.place_limit_order.return_value = {
        "order_id": "okx_order_101",
        "status": "pending"
    }
    
    return client


# ======================= ORDER FIXTURES =======================

@pytest.fixture
def sample_buy_order_data() -> Dict[str, Any]:
    """Фикстура данных для buy ордера."""
    return {
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("0.001"),
        "price": Decimal("45000.00"),
        "strategy_id": "test_strategy",
        "portfolio_id": "test_portfolio"
    }


@pytest.fixture
def sample_sell_order_data() -> Dict[str, Any]:
    """Фикстура данных для sell ордера."""
    return {
        "symbol": "BTCUSDT",
        "side": OrderSide.SELL,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("0.001"),
        "price": Decimal("45100.00"),
        "strategy_id": "test_strategy",
        "portfolio_id": "test_portfolio"
    }


@pytest.fixture
def sample_market_order_data() -> Dict[str, Any]:
    """Фикстура данных для market ордера."""
    return {
        "symbol": "ETHUSDT",
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("0.1"),
        "strategy_id": "test_strategy",
        "portfolio_id": "test_portfolio"
    }


@pytest.fixture
def sample_order(sample_buy_order_data: Dict[str, Any]) -> Order:
    """Фикстура готового ордера."""
    return Order(**sample_buy_order_data)


# ======================= MARKET DATA FIXTURES =======================

@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Фикстура рыночных данных."""
    return {
        "symbol": "BTCUSDT",
        "price": Decimal("45000.00"),
        "volume": Decimal("1000.0"),
        "timestamp": 1640995200000,
        "bid": Decimal("44999.50"),
        "ask": Decimal("45000.50"),
        "high_24h": Decimal("46000.00"),
        "low_24h": Decimal("44000.00"),
        "change_24h": Decimal("2.27")
    }


@pytest.fixture
def sample_orderbook() -> Dict[str, Any]:
    """Фикстура orderbook данных."""
    return {
        "symbol": "BTCUSDT",
        "bids": [
            [Decimal("44999.50"), Decimal("0.5")],
            [Decimal("44999.00"), Decimal("1.0")],
            [Decimal("44998.50"), Decimal("0.8")],
            [Decimal("44998.00"), Decimal("1.2")],
            [Decimal("44997.50"), Decimal("0.3")]
        ],
        "asks": [
            [Decimal("45000.50"), Decimal("0.7")],
            [Decimal("45001.00"), Decimal("0.9")],
            [Decimal("45001.50"), Decimal("1.1")],
            [Decimal("45002.00"), Decimal("0.6")],
            [Decimal("45002.50"), Decimal("1.5")]
        ],
        "timestamp": 1640995200000
    }


# ======================= STRATEGY FIXTURES =======================

@pytest.fixture
def sample_strategy_config() -> Dict[str, Any]:
    """Фикстура конфигурации стратегии."""
    return {
        "name": "test_strategy",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "exchanges": ["bybit", "binance"],
        "max_position_size": Decimal("1.0"),
        "risk_multiplier": Decimal("1.5"),
        "min_profit_threshold": Decimal("0.002")
    }


# ======================= PORTFOLIO FIXTURES =======================

@pytest.fixture
def sample_portfolio_data() -> Dict[str, Any]:
    """Фикстура данных портфеля."""
    return {
        "total_balance": Decimal("50000.00"),
        "available_balance": Decimal("45000.00"),
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "size": Decimal("0.5"),
                "entry_price": Decimal("44000.00"),
                "mark_price": Decimal("45000.00"),
                "unrealized_pnl": Decimal("500.00")
            }
        ],
        "total_unrealized_pnl": Decimal("500.00"),
        "equity": Decimal("50500.00")
    }


# ======================= ASYNC GENERATORS =======================

@pytest.fixture
async def market_data_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Асинхронный генератор рыночных данных."""
    base_price = Decimal("45000.00")
    
    for i in range(100):
        price_change = Decimal(str((i % 20 - 10) * 0.001))
        current_price = base_price * (Decimal("1") + price_change)
        
        yield {
            "symbol": "BTCUSDT",
            "price": current_price,
            "volume": Decimal(f"{1000 + (i % 100)}.0"),
            "timestamp": 1640995200000 + i * 1000,
            "bid": current_price - Decimal("0.50"),
            "ask": current_price + Decimal("0.50")
        }
        
        await asyncio.sleep(0.01)


# ======================= PERFORMANCE FIXTURES =======================

@pytest.fixture
def performance_test_config() -> Dict[str, Any]:
    """Конфигурация для performance тестов."""
    return {
        "max_latency_ms": 5.0,
        "min_throughput_ops_sec": 100,
        "max_memory_usage_mb": 100,
        "max_cpu_usage_percent": 80,
        "test_duration_seconds": 30
    }


# ======================= DATABASE FIXTURES =======================

@pytest.fixture(scope="function")
def mock_database():
    """Мок базы данных для тестирования."""
    db_mock = Mock()
    
    # Настраиваем стандартные методы
    db_mock.save.return_value = True
    db_mock.find_by_id.return_value = None
    db_mock.find_all.return_value = []
    db_mock.update.return_value = True
    db_mock.delete.return_value = True
    
    return db_mock


# ======================= UTILITY FIXTURES =======================

@pytest.fixture
def test_data_generator():
    """Генератор тестовых данных."""
    
    def generate_orders(count: int = 10) -> List[Dict[str, Any]]:
        """Генерирует список тестовых ордеров."""
        orders = []
        for i in range(count):
            orders.append({
                "symbol": "BTCUSDT",
                "side": OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                "order_type": OrderType.LIMIT,
                "quantity": Decimal(f"0.00{i+1}"),
                "price": Decimal(f"45{i:03d}.00"),
                "strategy_id": f"test_strategy_{i}"
            })
        return orders
    
    def generate_price_series(count: int = 100, base_price: Decimal = Decimal("45000.00")) -> List[Decimal]:
        """Генерирует серию цен."""
        import random
        prices = [base_price]
        
        for _ in range(count - 1):
            change_percent = Decimal(str(random.uniform(-0.02, 0.02)))  # ±2%
            new_price = prices[-1] * (Decimal("1") + change_percent)
            prices.append(new_price)
        
        return prices
    
    return {
        "orders": generate_orders,
        "prices": generate_price_series
    }


# ======================= CLEANUP FIXTURES =======================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Автоматическая очистка после каждого теста."""
    yield
    # Здесь можно добавить код очистки
    # Например, очистка временных файлов, сброс моков и т.д.


# ======================= TEST MARKERS =======================

# Автоматически применяем asyncio marker для async функций
def pytest_collection_modifyitems(config, items):
    """Автоматически добавляем маркеры для тестов."""
    for item in items:
        # Добавляем asyncio marker для async тестов
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Добавляем unit marker для unit тестов
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Добавляем integration marker для integration тестов
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Добавляем e2e marker для e2e тестов
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Добавляем performance marker для performance тестов
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ======================= CUSTOM ASSERTIONS =======================

def assert_decimal_equal(actual: Decimal, expected: Decimal, tolerance: Decimal = Decimal("0.000001")):
    """Проверка равенства Decimal с допуском."""
    assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual}, tolerance {tolerance}"


def assert_order_valid(order: Order):
    """Проверка валидности ордера."""
    assert order.order_id is not None
    assert order.symbol is not None
    assert order.side in [OrderSide.BUY, OrderSide.SELL]
    assert order.order_type in [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP]
    assert order.quantity > Decimal("0")
    if order.order_type in [OrderType.LIMIT, OrderType.STOP]:
        assert order.price is not None
        assert order.price > Decimal("0")


def assert_price_valid(price: Price):
    """Проверка валидности цены."""
    assert price.value > Decimal("0")
    assert price.currency is not None
    assert isinstance(price.currency, Currency)


# Добавляем кастомные assertions в pytest
pytest.assert_decimal_equal = assert_decimal_equal
pytest.assert_order_valid = assert_order_valid
pytest.assert_price_valid = assert_price_valid


# ======================= TEST CONFIGURATION =======================

# Настройки для pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# Настройки для production-ready тестирования
PRODUCTION_TEST_CONFIG = {
    "strict_mode": True,
    "fail_fast": False,
    "detailed_errors": True,
    "performance_monitoring": True,
    "memory_profiling": True,
    "coverage_threshold": 0.95
} 
