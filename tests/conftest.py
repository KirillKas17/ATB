#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Общие фикстуры для всех тестов Syntra.
"""
import asyncio
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
import pytest
from unittest.mock import Mock, patch
import pandas as pd

# Импорты доменных сущностей - отложенные импорты для избежания циклических зависимостей
# from domain.value_objects.currency import Currency
# from domain.value_objects.money import Money
# from domain.value_objects.price import Price
# from domain.value_objects.volume import Volume
# from domain.value_objects.timestamp import Timestamp
# from domain.value_objects.percentage import Percentage
# Импорты сущностей
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.position import Position, PositionSide
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy, StrategyType
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.entities.trading_pair import TradingPair

# Импорты протоколов
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
    MarketMakerPattern,
    PatternFeatures,
    MarketMakerPatternType,
    PatternResult,
    PatternOutcome,
    PatternMemory,
)
from domain.type_definitions.market_maker_types import (
    BookPressure,
    VolumeDelta,
    PriceReaction,
    SpreadChange,
    OrderImbalance,
    LiquidityDepth,
    TimeDuration,
    VolumeConcentration,
    PriceVolatility,
    MarketMicrostructure,
    Confidence,
    Accuracy,
    AverageReturn,
    SuccessCount,
    TotalCount,
)


@pytest.fixture(scope="session")
def event_loop() -> Any:
    """Создает event loop для асинхронных тестов."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Фикстура с тестовыми рыночными данными."""
    return {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 1000.0,
        "timestamp": datetime.now().isoformat(),
        "open": 49900.0,
        "high": 50100.0,
        "low": 49800.0,
        "close": 50000.0,
        "bid": 49999.0,
        "ask": 50001.0,
        "spread": 2.0,
        "bid_volume": 500.0,
        "ask_volume": 500.0,
    }


@pytest.fixture
def sample_orderbook_data() -> Dict[str, Any]:
    """Фикстура с данными ордербука."""
    return {
        "symbol": "BTCUSDT",
        "timestamp": datetime.now().isoformat(),
        "bids": [
            [49999.0, 1.5],
            [49998.0, 2.0],
            [49997.0, 1.0],
        ],
        "asks": [
            [50001.0, 1.5],
            [50002.0, 2.0],
            [50003.0, 1.0],
        ],
        "last_update_id": 123456789,
    }


@pytest.fixture
def sample_candles_data() -> pd.DataFrame:
    """Фикстура с данными свечей."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    data = {
        "open": [50000 + i * 0.1 for i in range(100)],
        "high": [50000 + i * 0.1 + 10 for i in range(100)],
        "low": [50000 + i * 0.1 - 10 for i in range(100)],
        "close": [50000 + i * 0.1 + 5 for i in range(100)],
        "volume": [1000000 + i * 1000 for i in range(100)],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_exchange() -> Mock:
    """Фикстура для мока биржи."""
    exchange = Mock(spec=ExchangeProtocol)
    # Настройка методов
    exchange.create_order = AsyncMock(return_value={"id": "test_order_123", "status": "pending"})
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.fetch_order = AsyncMock(return_value={"id": "test_order_123", "status": "filled"})
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_balance = AsyncMock(return_value={"BTC": {"free": 1.0, "used": 0.0}})
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0, "bid": 49999.0, "ask": 50001.0})
    exchange.fetch_order_book = AsyncMock(return_value=sample_orderbook_data())
    return exchange


@pytest.fixture
def mock_ml_service() -> Mock:
    """Фикстура для мока ML сервиса."""
    ml_service = Mock(spec=MLProtocol)
    # Настройка методов
    ml_service.predict = AsyncMock(return_value={"prediction": 0.75, "confidence": 0.85})
    ml_service.train = AsyncMock(return_value={"accuracy": 0.82, "loss": 0.15})
    ml_service.evaluate = AsyncMock(return_value={"precision": 0.78, "recall": 0.81})
    ml_service.get_model_info = AsyncMock(return_value={"version": "1.0", "type": "transformer"})
    return ml_service


@pytest.fixture
def mock_repository() -> Mock:
    """Фикстура для мока репозитория."""
    repository = Mock(spec=RepositoryProtocol)
    # Настройка методов
    repository.save = AsyncMock(return_value=True)
    repository.get_by_id = AsyncMock(return_value=None)
    repository.get_all = AsyncMock(return_value=[])
    repository.delete = AsyncMock(return_value=True)
    repository.update = AsyncMock(return_value=True)
    return repository


@pytest.fixture
def sample_order() -> Order:
    """Фикстура с тестовым ордером."""
    # Импорты внутри функции для избежания циклических зависимостей
    from domain.value_objects.currency import Currency
    from domain.value_objects.price import Price
    from domain.type_definitions import OrderId, VolumeValue, TradingPair, Symbol

    return Order(
        id=OrderId(uuid4()),
        trading_pair=TradingPair(Symbol("BTC/USDT")),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=VolumeValue(Decimal("0.001")),
        price=Price(Decimal("50000"), Currency.USDT),
    )


@pytest.fixture
def sample_position() -> Position:
    """Фикстура с тестовой позицией."""
    # Импорты внутри функции для избежания циклических зависимостей
    from domain.value_objects.currency import Currency
    from domain.value_objects.price import Price
    from domain.value_objects.volume import Volume
    from domain.type_definitions import PositionId, PortfolioId, Symbol

    pair = TradingPair(symbol=Symbol("BTC/USDT"), base_currency=Currency.BTC, quote_currency=Currency.USDT)
    return Position(
        id=PositionId(uuid4()),
        portfolio_id=PortfolioId(uuid4()),
        trading_pair=pair,
        side=PositionSide.LONG,
        volume=Volume(Decimal("0.001"), Currency.BTC),
        entry_price=Price(Decimal("50000"), Currency.USDT),
        current_price=Price(Decimal("51000"), Currency.USDT),
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Фикстура с тестовым портфелем."""
    # Импорты внутри функции для избежания циклических зависимостей
    from domain.value_objects.currency import Currency
    from domain.value_objects.money import Money

    return Portfolio(
        total_equity=Money(Decimal("10000"), Currency.USD),
        free_margin=Money(Decimal("10000"), Currency.USD),
    )


@pytest.fixture
def sample_strategy() -> Strategy:
    """Фикстура с тестовой стратегией."""
    return Strategy(
        name="Test Strategy",
        description="Test Description",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=["BTCUSDT", "ETHUSDT"],
    )


@pytest.fixture
def sample_signal() -> Signal:
    """Фикстура с тестовым сигналом."""
    from domain.value_objects.currency import Currency
    from domain.value_objects.money import Money

    return Signal(
        strategy_id=uuid4(),
        trading_pair="BTCUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("0.8"),
        price=Money(Decimal("50000"), Currency.USDT),
        quantity=Decimal("0.001"),
    )


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Фикстура с тестовой конфигурацией."""
    return {
        "exchange": {
            "name": "testnet",
            "api_key": "test_key",
            "secret": "test_secret",
            "testnet": True,
        },
        "risk": {
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_weekly_loss": 0.15,
        },
        "portfolio": {
            "max_position_size": 0.20,
            "min_position_size": 0.01,
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "test",
            "password": "test",
            "database": "test_db",
        },
        "ml": {
            "model_path": "models/",
            "data_path": "data/",
            "features": ["close", "volume", "rsi", "macd"],
            "target": "returns",
        },
    }


@pytest.fixture
def temp_file_path() -> Generator[Path, None, None]:
    """Фикстура для временного файла."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        temp_path = Path(f.name)
    yield temp_path
    # Очистка после теста
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_dir_path() -> Generator[Path, None, None]:
    """Фикстура для временной директории."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Очистка после теста
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_entanglement_result() -> Dict[str, Any]:
    """Фикстура с результатом обнаружения запутанности."""
    return {
        "is_entangled": True,
        "correlation_score": 0.98,
        "confidence": 0.95,
        "exchange_pair": ("binance", "bybit"),
        "symbol": "BTCUSDT",
        "lag_ms": 1.5,
        "metadata": {"test": "data"},
    }


@pytest.fixture
def sample_noise_analysis_result() -> Dict[str, Any]:
    """Фикстура с результатом анализа шума."""
    return {
        "is_synthetic": True,
        "noise_intensity": 0.85,
        "confidence": 0.92,
        "noise_pattern": "artificial_clustering",
        "metadata": {"test": "data"},
    }


@pytest.fixture
def sample_mirror_signal() -> Dict[str, Any]:
    """Фикстура с зеркальным сигналом."""
    return {
        "is_mirror": True,
        "leader_asset": "ETHUSDT",
        "follower_asset": "BTCUSDT",
        "correlation": 0.92,
        "lag_periods": 3,
        "confidence": 0.88,
        "metadata": {"test": "data"},
    }


@pytest.fixture
def sample_liquidity_gravity_result() -> Dict[str, Any]:
    """Фикстура с результатом гравитации ликвидности."""
    return {
        "total_gravity": 0.75,
        "bid_ask_forces": [0.3, 0.45],
        "gravity_distribution": {"bids": 0.4, "asks": 0.6},
        "risk_level": "medium",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"test": "data"},
    }


@pytest.fixture
def mock_event_bus() -> Mock:
    """Фикстура для мока шины событий."""
    event_bus = Mock()
    event_bus.publish = AsyncMock()
    event_bus.subscribe = AsyncMock()
    event_bus.unsubscribe = AsyncMock()
    return event_bus


@pytest.fixture
def mock_logger() -> Mock:
    """Фикстура для мока логгера."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_metrics_collector() -> Mock:
    """Фикстура для мока сборщика метрик."""
    metrics = Mock()
    metrics.record_trade = AsyncMock()
    metrics.record_signal = AsyncMock()
    metrics.get_performance_metrics = AsyncMock(return_value={"total_trades": 10})
    return metrics


@pytest.fixture
def sample_trade_data() -> Dict[str, Any]:
    """Фикстура с данными сделки."""
    return {
        "trade_id": "test_trade_123",
        "symbol": "BTCUSDT",
        "strategy": "test_strategy",
        "pnl": 100.0,
        "side": "buy",
        "quantity": 0.001,
        "price": 50000.0,
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_risk_assessment() -> Dict[str, Any]:
    """Фикстура с оценкой риска."""
    return {
        "is_allowed": True,
        "risk_score": 0.3,
        "position_size": 0.01,
        "max_loss": 50.0,
        "confidence": 0.85,
        "warnings": [],
    }


@pytest.fixture
def mock_health_checker() -> Mock:
    """Фикстура для мока проверки здоровья."""
    health_checker = Mock()
    health_checker.check_all_services = AsyncMock(return_value={"overall_healthy": True})
    health_checker.get_health_status = AsyncMock(return_value={"status": "healthy"})
    return health_checker


@pytest.fixture
def sample_portfolio_metrics() -> Dict[str, Any]:
    """Фикстура с метриками портфеля."""
    return {
        "total_equity": 10000.0,
        "free_margin": 8000.0,
        "used_margin": 2000.0,
        "position_count": 2,
        "total_pnl": 500.0,
        "daily_pnl": 100.0,
        "weekly_pnl": 300.0,
        "weights": {"BTCUSDT": 0.6, "ETHUSDT": 0.4},
    }


@pytest.fixture
def mock_cache_manager() -> Mock:
    """Фикстура для мока менеджера кэша."""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    cache.clear = AsyncMock()
    return cache


@pytest.fixture
def sample_strategy_config() -> Dict[str, Any]:
    """Фикстура с конфигурацией стратегии."""
    return {
        "name": "Test Strategy",
        "type": "trend_following",
        "parameters": {
            "window": 20,
            "threshold": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10,
        },
        "risk_limits": {
            "max_position_size": 0.1,
            "max_daily_trades": 10,
            "max_drawdown": 0.15,
        },
        "enabled": True,
    }


@pytest.fixture
def mock_data_loader() -> Mock:
    """Фикстура для мока загрузчика данных."""
    loader = Mock()
    loader.load_market_data = AsyncMock(return_value=sample_candles_data())
    loader.save_market_data = AsyncMock()
    loader.get_available_symbols = AsyncMock(return_value=["BTCUSDT", "ETHUSDT"])
    return loader


@pytest.fixture
def sample_backtest_result() -> Dict[str, Any]:
    """Фикстура с результатом бэктеста."""
    return {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "win_rate": 0.65,
        "total_trades": 50,
        "profit_factor": 1.8,
        "equity_curve": pd.Series([10000, 10100, 10200, 10150, 10300]),
        "trades": [
            {"entry": 50000, "exit": 51000, "pnl": 100, "side": "long"},
            {"entry": 51000, "exit": 50500, "pnl": -50, "side": "short"},
        ],
    }


@pytest.fixture
def mock_optimizer() -> Mock:
    """Фикстура для мока оптимизатора."""
    optimizer = Mock()
    optimizer.optimize = AsyncMock(return_value={"best_params": {"window": 25, "threshold": 0.03}})
    optimizer.evaluate = AsyncMock(return_value={"score": 0.85})
    return optimizer


@pytest.fixture
def sample_ml_prediction() -> Dict[str, Any]:
    """Фикстура с ML предсказанием."""
    return {
        "prediction": 0.75,
        "confidence": 0.85,
        "model_type": "transformer",
        "features_used": ["close", "volume", "rsi"],
        "timestamp": datetime.now().isoformat(),
        "metadata": {"model_version": "1.0"},
    }


@pytest.fixture
def mock_webhook_sender() -> Mock:
    """Фикстура для мока отправителя вебхуков."""
    sender = Mock()
    sender.send_notification = AsyncMock(return_value=True)
    sender.send_alert = AsyncMock(return_value=True)
    return sender


@pytest.fixture
def sample_api_response() -> Dict[str, Any]:
    """Фикстура с API ответом."""
    return {
        "status": "success",
        "data": {"result": "test_data"},
        "timestamp": datetime.now().isoformat(),
        "request_id": "req_123",
    }


@pytest.fixture
def mock_database_connection() -> Mock:
    """Фикстура для мока подключения к БД."""
    connection = Mock()
    connection.execute = AsyncMock()
    connection.fetch_one = AsyncMock(return_value=None)
    connection.fetch_all = AsyncMock(return_value=[])
    connection.commit = AsyncMock()
    connection.rollback = AsyncMock()
    return connection


@pytest.fixture
def sample_error_report() -> Dict[str, Any]:
    """Фикстура с отчетом об ошибке."""
    return {
        "error_id": "err_123",
        "error_type": "validation_error",
        "message": "Invalid input data",
        "stack_trace": "Traceback...",
        "timestamp": datetime.now().isoformat(),
        "context": {"user_id": "user_123", "action": "create_order"},
    }


@pytest.fixture
def mock_file_system() -> Mock:
    """Фикстура для мока файловой системы."""
    fs = Mock()
    fs.write_file = AsyncMock()
    fs.read_file = AsyncMock(return_value="test content")
    fs.file_exists = AsyncMock(return_value=True)
    fs.delete_file = AsyncMock()
    return fs


@pytest.fixture
def sample_performance_metrics() -> Dict[str, Any]:
    """Фикстура с метриками производительности."""
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "response_time": 0.15,
        "throughput": 1000,
        "error_rate": 0.01,
        "uptime": 99.9,
    }


@pytest.fixture
def temp_dir() -> Any:
    """Временная директория для тестов."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_config(temp_dir) -> Any:
    """Конфигурация хранилища для тестов."""
    return StorageConfig(
        base_path=temp_dir,
        compression_enabled=True,
        max_workers=2,
        cache_size=1000,
        backup_enabled=True,
        backup_interval_hours=1,
        cleanup_enabled=True,
        cleanup_interval_days=1,
    )


@pytest.fixture
def analysis_config() -> Any:
    """Конфигурация анализа для тестов."""
    return AnalysisConfig(
        min_confidence=Confidence(0.6),
        similarity_threshold=0.8,
        accuracy_threshold=0.7,
        volume_threshold=1000.0,
        spread_threshold=0.001,
        time_window_seconds=300,
        min_trades_count=10,
        max_history_size=1000,
    )


@pytest.fixture
def market_profiles_components(storage_config, analysis_config) -> Any:
    """Компоненты market_profiles для тестов."""
    storage = MarketMakerStorage(storage_config)
    pattern_repo = PatternMemoryRepository()
    behavior_repo = BehaviorHistoryRepository()
    analyzer = PatternAnalyzer(analysis_config)
    similarity_calc = SimilarityCalculator()
    success_analyzer = SuccessRateAnalyzer()
    return {
        "storage": storage,
        "pattern_repo": pattern_repo,
        "behavior_repo": behavior_repo,
        "analyzer": analyzer,
        "similarity_calc": similarity_calc,
        "success_analyzer": success_analyzer,
    }


@pytest.fixture
def sample_pattern() -> Any:
    """Образец паттерна для тестов."""
    features = PatternFeatures(
        book_pressure=BookPressure(0.7),
        volume_delta=VolumeDelta(0.15),
        price_reaction=PriceReaction(0.02),
        spread_change=SpreadChange(0.05),
        order_imbalance=OrderImbalance(0.6),
        liquidity_depth=LiquidityDepth(0.8),
        time_duration=TimeDuration(300),
        volume_concentration=VolumeConcentration(0.75),
        price_volatility=PriceVolatility(0.03),
        market_microstructure=MarketMicrostructure({"depth_imbalance": 0.4, "flow_imbalance": 0.6}),
    )
    return MarketMakerPattern(
        pattern_type=MarketMakerPatternType.ACCUMULATION,
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        features=features,
        confidence=Confidence(0.85),
        context={"market_regime": "trending", "session": "asian"},
    )


@pytest.fixture
def sample_patterns() -> Any:
    """Образцы паттернов для тестов."""
    patterns = []
    pattern_types = [
        (MarketMakerPatternType.ACCUMULATION, 0.85, "trending"),
        (MarketMakerPatternType.EXIT, 0.75, "trending"),
        (MarketMakerPatternType.ABSORPTION, 0.65, "sideways"),
        (MarketMakerPatternType.DISTRIBUTION, 0.70, "trending"),
        (MarketMakerPatternType.MARKUP, 0.80, "trending"),
    ]
    for i, (pattern_type, confidence, market_regime) in enumerate(pattern_types):
        features = PatternFeatures(
            book_pressure=BookPressure(0.6 + i * 0.1),
            volume_delta=VolumeDelta(0.1 + i * 0.05),
            price_reaction=PriceReaction(0.01 + i * 0.005),
            spread_change=SpreadChange(0.03 + i * 0.01),
            order_imbalance=OrderImbalance(0.5 + i * 0.1),
            liquidity_depth=LiquidityDepth(0.7 + i * 0.05),
            time_duration=TimeDuration(200 + i * 50),
            volume_concentration=VolumeConcentration(0.6 + i * 0.05),
            price_volatility=PriceVolatility(0.02 + i * 0.005),
            market_microstructure=MarketMicrostructure(
                {
                    "depth_imbalance": 0.3 + i * 0.1,
                    "flow_imbalance": 0.5 + i * 0.1,
                    "order_flow": 0.4 + i * 0.1,
                    "liquidity_imbalance": 0.2 + i * 0.1,
                }
            ),
        )
        pattern = MarketMakerPattern(
            pattern_type=pattern_type,
            symbol="BTCUSDT",
            timestamp=datetime.now() + timedelta(minutes=i * 10),
            features=features,
            confidence=Confidence(confidence),
            context={
                "market_regime": market_regime,
                "session": "asian" if i % 2 == 0 else "european",
                "volatility": "medium",
                "volume_profile": "normal",
                "price_action": "trending" if market_regime == "trending" else "sideways",
            },
        )
        patterns.append(pattern)
    return patterns


@pytest.fixture
def sample_pattern_memories(sample_patterns) -> Any:
    """Образцы памяти паттернов для тестов."""
    memories = []
    for i, pattern in enumerate(sample_patterns):
        # Создаем результат
        if i < 3:  # Успешные
            result = PatternResult(
                outcome=PatternOutcome.SUCCESS,
                price_change_15min=0.02 + i * 0.005,
                price_change_1h=0.05 + i * 0.01,
                volume_change=0.1 + i * 0.02,
                execution_time=300 + i * 30,
                confidence=Confidence(0.8 + i * 0.02),
            )
        elif i == 3:  # Частично успешный
            result = PatternResult(
                outcome=PatternOutcome.PARTIAL,
                price_change_15min=0.005,
                price_change_1h=0.01,
                volume_change=0.02,
                execution_time=300,
                confidence=Confidence(0.7),
            )
        else:  # Неуспешный
            result = PatternResult(
                outcome=PatternOutcome.FAILURE,
                price_change_15min=-0.01,
                price_change_1h=-0.02,
                volume_change=-0.05,
                execution_time=300,
                confidence=Confidence(0.6),
            )
        memory = PatternMemory(
            pattern=pattern,
            result=result,
            accuracy=Accuracy(0.7 + i * 0.05),
            avg_return=AverageReturn(0.01 + i * 0.005),
            success_count=SuccessCount(8 if i < 3 else 2),
            total_count=TotalCount(10),
            last_seen=datetime.now(),
        )
        memories.append(memory)
    return memories


@pytest.fixture
def mock_storage() -> Any:
    """Мок хранилища для тестов."""
    mock = AsyncMock()
    mock.save_pattern.return_value = True
    mock.get_patterns_by_symbol.return_value = []
    mock.update_pattern_result.return_value = True
    mock.get_storage_statistics.return_value = Mock()
    mock.find_similar_patterns.return_value = []
    mock.validate_data_integrity.return_value = True
    mock.backup_data.return_value = True
    mock.cleanup_old_data.return_value = 0
    return mock


@pytest.fixture
def mock_analyzer() -> Any:
    """Мок анализатора для тестов."""
    mock = AsyncMock()
    mock.analyze_pattern.return_value = {
        "confidence": 0.8,
        "similarity_score": 0.85,
        "success_probability": 0.75,
        "market_context": {},
        "risk_assessment": {},
        "recommendations": [],
    }
    mock.analyze_market_context.return_value = {
        "market_phase": "trending",
        "volatility_regime": "medium",
        "liquidity_regime": "high",
        "volume_profile": "normal",
        "price_action": "trending",
        "order_flow": "positive",
    }
    return mock


@pytest.fixture
def mock_similarity_calculator() -> Any:
    """Мок калькулятора схожести для тестов."""
    mock = AsyncMock()
    mock.calculate_similarity.return_value = 0.85
    return mock


@pytest.fixture
def mock_success_analyzer() -> Any:
    """Мок анализатора успешности для тестов."""
    mock = AsyncMock()
    mock.calculate_success_rate.return_value = 0.75
    mock.analyze_success_trends.return_value = {
        "trend_direction": "up",
        "trend_strength": 0.7,
        "confidence": 0.8,
        "periods": 5,
    }
    mock.calculate_accuracy_metrics.return_value = {
        "avg_accuracy": 0.8,
        "accuracy_std": 0.1,
        "min_accuracy": 0.6,
        "max_accuracy": 0.9,
        "accuracy_trend": "stable",
    }
    mock.calculate_return_metrics.return_value = {
        "avg_return": 0.02,
        "return_std": 0.01,
        "min_return": 0.01,
        "max_return": 0.03,
        "return_trend": "positive",
    }
    mock.generate_recommendations.return_value = [
        "Увеличить уверенность для паттернов накопления",
        "Снизить риск для паттернов выхода",
    ]
    return mock


@pytest.fixture
def mock_behavior_repository() -> Any:
    """Мок репозитория поведения для тестов."""
    mock = AsyncMock()
    mock.save_behavior_record.return_value = True
    mock.get_behavior_history.return_value = []
    mock.get_statistics.return_value = {
        "total_records": 0,
        "avg_volume": 0.0,
        "avg_spread": 0.0,
        "avg_imbalance": 0.0,
        "avg_pressure": 0.0,
    }
    return mock


@pytest.fixture
def mock_pattern_repository() -> Any:
    """Мок репозитория паттернов для тестов."""
    mock = AsyncMock()
    mock.save_pattern.return_value = True
    mock.get_patterns_by_symbol.return_value = []
    mock.update_pattern_result.return_value = True
    mock.get_storage_statistics.return_value = {
        "total_patterns": 0,
        "total_symbols": 0,
        "total_successful_patterns": 0,
        "total_storage_size_bytes": 0,
        "avg_pattern_size_bytes": 0,
        "compression_ratio": 1.0,
        "cache_hit_ratio": 0.0,
        "avg_read_time_ms": 0.0,
        "avg_write_time_ms": 0.0,
        "error_count": 0,
        "warning_count": 0,
    }
    return mock


# Хуки для настройки тестов
def pytest_configure(config) -> Any:
    """Конфигурация pytest."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items) -> Any:
    """Модификация коллекции тестов."""
    for item in items:
        # Добавляем маркеры по умолчанию
        if "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        # Маркируем медленные тесты
        if "test_high_load" in item.nodeid or "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# Утилиты для тестов
class TestUtils:
    """Утилиты для тестов."""

    @staticmethod
    def create_test_pattern(pattern_type: MarketMakerPatternType, confidence: float = 0.8) -> Any:
        """Создает тестовый паттерн."""
        from domain.type_definitions.market_maker_types import Symbol

        features = PatternFeatures(
            book_pressure=BookPressure(0.6),
            volume_delta=VolumeDelta(0.1),
            price_reaction=PriceReaction(0.01),
            spread_change=SpreadChange(0.03),
            order_imbalance=OrderImbalance(0.5),
            liquidity_depth=LiquidityDepth(0.7),
            time_duration=TimeDuration(200),
            volume_concentration=VolumeConcentration(0.6),
            price_volatility=PriceVolatility(0.02),
            market_microstructure=MarketMicrostructure({}),
        )
        return MarketMakerPattern(
            pattern_type=pattern_type,
            symbol=Symbol("BTCUSDT"),
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(confidence),
            context={},
        )

    @staticmethod
    def create_test_result(outcome: PatternOutcome, confidence: float = 0.8) -> Any:
        """Создает тестовый результат."""
        return PatternResult(
            outcome=outcome,
            price_change_5min=0.01 if outcome == PatternOutcome.SUCCESS else -0.005,
            price_change_15min=0.02 if outcome == PatternOutcome.SUCCESS else -0.01,
            price_change_30min=0.03 if outcome == PatternOutcome.SUCCESS else -0.015,
            volume_change=0.1 if outcome == PatternOutcome.SUCCESS else -0.05,
            volatility_change=0.02 if outcome == PatternOutcome.SUCCESS else -0.01,
        )

    @staticmethod
    def create_test_behavior_data(symbol: str = "BTCUSDT", pattern_type: str = "accumulation") -> Any:
        """Создает тестовые данные поведения."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "pattern_type": pattern_type,
            "volume": 1000.0,
            "spread": 0.001,
            "imbalance": 0.3,
            "pressure": 0.4,
            "confidence": 0.8,
            "market_phase": "trending",
            "volatility_regime": "medium",
            "liquidity_regime": "high",
        }


# Экспортируем утилиты


@pytest.fixture
def test_utils() -> None:
    """Утилиты для тестов."""
    return TestUtils
