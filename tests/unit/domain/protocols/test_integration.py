"""
Unit тесты для domain/protocols/integration.py.

Покрывает:
- SimpleMonitor
- IntegrationTestExchangeProtocol
- IntegrationTestMLProtocol
- IntegrationTestStrategyProtocol
- IntegrationTestRepositoryProtocol
- TradingSystemIntegration
- Интеграционные тесты
- Обработку ошибок
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4
from decimal import Decimal

import pandas as pd

from domain.protocols.integration import (
    SimpleMonitor,
    IntegrationTestExchangeProtocol,
    IntegrationTestMLProtocol,
    IntegrationTestStrategyProtocol,
    IntegrationTestRepositoryProtocol,
    TradingSystemIntegration,
    test_exchange_protocol_integration,
    test_ml_protocol_integration,
    test_strategy_protocol_integration,
    test_repository_protocol_integration,
    test_trading_system_integration,
    test_protocol_decorators_integration,
    test_protocol_validators_integration,
    test_protocol_monitoring_integration,
    test_protocol_performance_integration,
    run_all_integration_tests,
)
from domain.entities.market import MarketData
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.ml import Model, ModelType, ModelStatus, Prediction, PredictionType
from domain.entities.strategy import StrategyType as DomainStrategyType
from domain.entities.signal import Signal
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.type_definitions import (
    Symbol,
    OrderId,
    TradingPair,
    TimestampValue,
    MetadataDict,
    ModelId,
    PredictionId,
    StrategyId,
    PriceValue,
    VolumeValue,
    QueryFilter,
    QueryOptions,
)
from domain.protocols.strategy_protocol import MarketRegime
from domain.type_definitions.protocol_types import (
    PatternDetectionResult,
    SignalFilterDict,
    StrategyAdaptationRules,
    StrategyErrorContext,
    RepositoryResponse as ProtocolRepositoryResponse,
    PerformanceMetricsDict as ProtocolPerformanceMetricsDict,
    HealthCheckDict as ProtocolHealthCheckDict,
    MarketAnalysisResult,
)
from domain.protocols.ml_protocol import (
    ModelConfig,
    TrainingConfig,
    PredictionConfig,
    ModelMetrics,
    ModelState,
    OptimizationMethod,
    ConfidenceLevel,
)
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    TransactionalProtocol,
    TransactionId,
    TransactionStatus,
    RepositoryResponse,
    TransactionProtocol,
)
from domain.protocols.strategy_protocol import PerformanceMetrics


class TestSimpleMonitor:
    """Тесты для SimpleMonitor."""

    @pytest.fixture
    def simple_monitor(self):
        """Фикстура простого монитора."""
        return SimpleMonitor()

    def test_initialization(self, simple_monitor):
        """Тест инициализации."""
        assert simple_monitor.running is False
        assert simple_monitor.metrics == {}

    async def test_start_stop(self, simple_monitor):
        """Тест запуска и остановки."""
        await simple_monitor.start()
        assert simple_monitor.running is True
        
        await simple_monitor.stop()
        assert simple_monitor.running is False

    def test_is_running(self, simple_monitor):
        """Тест проверки состояния."""
        assert simple_monitor.is_running() is False
        
        simple_monitor.running = True
        assert simple_monitor.is_running() is True

    def test_record_metric(self, simple_monitor):
        """Тест записи метрики."""
        simple_monitor.record_metric("test_metric", 42.5)
        assert simple_monitor.metrics["test_metric"] == 42.5

    def test_get_metrics(self, simple_monitor):
        """Тест получения метрик."""
        simple_monitor.record_metric("test_metric", 42.5)
        metrics = simple_monitor.get_metrics()
        assert metrics["test_metric"] == 42.5


class TestIntegrationTestExchangeProtocol:
    """Тесты для IntegrationTestExchangeProtocol."""

    @pytest.fixture
    def exchange_protocol(self):
        """Фикстура протокола биржи."""
        return IntegrationTestExchangeProtocol()

    async def test_initialization(self, exchange_protocol):
        """Тест инициализации."""
        assert exchange_protocol.connected is False
        assert exchange_protocol.orders == {}
        assert exchange_protocol.balances == {}

    async def test_connect_disconnect(self, exchange_protocol):
        """Тест подключения и отключения."""
        result = await exchange_protocol.connect()
        assert result is True
        assert exchange_protocol.connected is True
        
        await exchange_protocol.disconnect()
        assert exchange_protocol.connected is False

    async def test_is_connected(self, exchange_protocol):
        """Тест проверки подключения."""
        assert await exchange_protocol.is_connected() is False
        
        await exchange_protocol.connect()
        assert await exchange_protocol.is_connected() is True

    async def test_get_market_data(self, exchange_protocol):
        """Тест получения рыночных данных."""
        market_data = await exchange_protocol.get_market_data("BTC/USD")
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "BTC/USD"

    async def test_create_order(self, exchange_protocol):
        """Тест создания ордера."""
        order = Order(
            id=OrderId("test_order"),
            symbol=Symbol("BTC/USD"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=Volume(Decimal("1.0")),
            price=Price(Decimal("50000.0")),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        result = await exchange_protocol.create_order(order)
        assert isinstance(result, dict)
        assert "order_id" in result

    async def test_get_order_status(self, exchange_protocol):
        """Тест получения статуса ордера."""
        order = Order(
            id=OrderId("test_order"),
            symbol=Symbol("BTC/USD"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=Volume(Decimal("1.0")),
            price=Price(Decimal("50000.0")),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        await exchange_protocol.create_order(order)
        status = await exchange_protocol.get_order_status("test_order")
        assert isinstance(status, Order)

    async def test_cancel_order(self, exchange_protocol):
        """Тест отмены ордера."""
        order = Order(
            id=OrderId("test_order"),
            symbol=Symbol("BTC/USD"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=Volume(Decimal("1.0")),
            price=Price(Decimal("50000.0")),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        await exchange_protocol.create_order(order)
        result = await exchange_protocol.cancel_order("test_order")
        assert result is True

    async def test_fetch_order(self, exchange_protocol):
        """Тест получения ордера."""
        order_data = await exchange_protocol.fetch_order("test_order")
        assert isinstance(order_data, dict)

    async def test_fetch_open_orders(self, exchange_protocol):
        """Тест получения открытых ордеров."""
        orders = await exchange_protocol.fetch_open_orders()
        assert isinstance(orders, list)

    async def test_fetch_balance(self, exchange_protocol):
        """Тест получения баланса."""
        balance = await exchange_protocol.fetch_balance()
        assert isinstance(balance, dict)

    async def test_fetch_ticker(self, exchange_protocol):
        """Тест получения тикера."""
        ticker = await exchange_protocol.fetch_ticker("BTC/USD")
        assert isinstance(ticker, dict)
        assert "symbol" in ticker

    async def test_fetch_order_book(self, exchange_protocol):
        """Тест получения ордербука."""
        order_book = await exchange_protocol.fetch_order_book("BTC/USD")
        assert isinstance(order_book, dict)
        assert "bids" in order_book
        assert "asks" in order_book


class TestIntegrationTestMLProtocol:
    """Тесты для IntegrationTestMLProtocol."""

    @pytest.fixture
    def ml_protocol(self):
        """Фикстура ML протокола."""
        return IntegrationTestMLProtocol()

    def test_initialization(self, ml_protocol):
        """Тест инициализации."""
        assert ml_protocol.models == {}
        assert ml_protocol.predictions == {}

    async def test_create_model(self, ml_protocol):
        """Тест создания модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target",
            hyperparameters={"learning_rate": 0.01}
        )
        
        model = await ml_protocol.create_model(config)
        assert isinstance(model, Model)
        assert model.name == "test_model"
        assert model.model_type == ModelType.REGRESSION

    async def test_train_model(self, ml_protocol):
        """Тест обучения модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        training_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        })
        
        training_config = TrainingConfig(
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        
        trained_model = await ml_protocol.train_model(
            model.id, training_data, training_config
        )
        assert isinstance(trained_model, Model)
        assert trained_model.status == ModelStatus.TRAINED

    async def test_predict(self, ml_protocol):
        """Тест предсказания."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        features = {"feature1": 1.0, "feature2": 2.0}
        
        prediction = await ml_protocol.predict(model.id, features)
        assert isinstance(prediction, Prediction)
        assert prediction.model_id == model.id

    async def test_batch_predict(self, ml_protocol):
        """Тест пакетного предсказания."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        features_batch = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
        
        predictions = await ml_protocol.batch_predict(model.id, features_batch)
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        assert all(isinstance(p, Prediction) for p in predictions)

    async def test_evaluate_model(self, ml_protocol):
        """Тест оценки модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        test_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        })
        
        metrics = await ml_protocol.evaluate_model(model.id, test_data)
        assert isinstance(metrics, ModelMetrics)

    async def test_get_model_status(self, ml_protocol):
        """Тест получения статуса модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        status = await ml_protocol.get_model_status(model.id)
        assert isinstance(status, ModelStatus)

    async def test_activate_deactivate_model(self, ml_protocol):
        """Тест активации и деактивации модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.REGRESSION,
            features=["feature1", "feature2"],
            target="target"
        )
        
        model = await ml_protocol.create_model(config)
        
        result = await ml_protocol.activate_model(model.id)
        assert result is True
        
        result = await ml_protocol.deactivate_model(model.id)
        assert result is True


class TestIntegrationTestStrategyProtocol:
    """Тесты для IntegrationTestStrategyProtocol."""

    @pytest.fixture
    def strategy_protocol(self):
        """Фикстура протокола стратегии."""
        return IntegrationTestStrategyProtocol()

    async def test_initialization(self, strategy_protocol):
        """Тест инициализации."""
        assert strategy_protocol.strategies == {}
        assert strategy_protocol.signals == {}

    async def test_analyze_market(self, strategy_protocol):
        """Тест анализа рынка."""
        market_data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        analysis = await strategy_protocol.analyze_market(
            market_data, DomainStrategyType.MOMENTUM
        )
        assert isinstance(analysis, MarketAnalysisResult)

    async def test_generate_signals(self, strategy_protocol):
        """Тест генерации сигналов."""
        analysis = MarketAnalysisResult(
            trend="bullish",
            strength=0.8,
            confidence=0.9,
            indicators={"rsi": 65.0, "macd": 0.5},
            patterns=["double_bottom"],
            support_level=Decimal("49000.0"),
            resistance_level=Decimal("51000.0"),
            volatility=Decimal("0.02"),
            volume_profile={"high": 1000.0, "low": 500.0},
            market_regime=MarketRegime.TRENDING,
            timestamp=datetime.now()
        )
        
        signals = await strategy_protocol.generate_signals(analysis)
        assert isinstance(signals, list)
        assert len(signals) > 0

    async def test_execute_signal(self, strategy_protocol):
        """Тест выполнения сигнала."""
        signal = {
            "type": "BUY",
            "symbol": "BTC/USD",
            "price": 50000.0,
            "volume": 1.0,
            "confidence": 0.8
        }
        
        result = await strategy_protocol.execute_signal(signal)
        assert isinstance(result, dict)
        assert "executed" in result

    async def test_calculate_technical_indicators(self, strategy_protocol):
        """Тест расчета технических индикаторов."""
        data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        indicators = ["rsi", "macd", "bollinger_bands"]
        result = await strategy_protocol.calculate_technical_indicators(data, indicators)
        assert isinstance(result, dict)
        assert "rsi" in result

    async def test_detect_market_patterns(self, strategy_protocol):
        """Тест обнаружения рыночных паттернов."""
        data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        patterns = await strategy_protocol.detect_market_patterns(data)
        assert isinstance(patterns, list)

    async def test_analyze_market_regime(self, strategy_protocol):
        """Тест анализа рыночного режима."""
        data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        regime = await strategy_protocol.analyze_market_regime(data)
        assert isinstance(regime, MarketRegime)

    async def test_calculate_volatility(self, strategy_protocol):
        """Тест расчета волатильности."""
        data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        volatility = await strategy_protocol.calculate_volatility(data)
        assert isinstance(volatility, Decimal)

    async def test_generate_signal(self, strategy_protocol):
        """Тест генерации сигнала."""
        market_data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        })
        
        signal = await strategy_protocol.generate_signal(
            StrategyId("test_strategy"), market_data
        )
        assert signal is None or isinstance(signal, Signal)

    async def test_validate_signal(self, strategy_protocol):
        """Тест валидации сигнала."""
        signal = Signal(
            id="test_signal",
            type="BUY",
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1.0")),
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        market_data = MarketData(
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1000.0")),
            timestamp=datetime.now()
        )
        
        is_valid = await strategy_protocol.validate_signal(signal, market_data)
        assert isinstance(is_valid, bool)

    async def test_calculate_signal_confidence(self, strategy_protocol):
        """Тест расчета уверенности сигнала."""
        signal = Signal(
            id="test_signal",
            type="BUY",
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1.0")),
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        market_data = MarketData(
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1000.0")),
            timestamp=datetime.now()
        )
        
        confidence = await strategy_protocol.calculate_signal_confidence(
            signal, market_data, []
        )
        assert isinstance(confidence, ConfidenceLevel)

    async def test_create_order_from_signal(self, strategy_protocol):
        """Тест создания ордера из сигнала."""
        signal = Signal(
            id="test_signal",
            type="BUY",
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1.0")),
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        account_balance = Decimal("100000.0")
        risk_params = {"max_risk_per_trade": 0.02}
        
        order = await strategy_protocol.create_order_from_signal(
            signal, account_balance, risk_params
        )
        assert isinstance(order, Order)

    async def test_calculate_position_size(self, strategy_protocol):
        """Тест расчета размера позиции."""
        signal = Signal(
            id="test_signal",
            type="BUY",
            symbol=Symbol("BTC/USD"),
            price=Price(Decimal("50000.0")),
            volume=Volume(Decimal("1.0")),
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        account_balance = Decimal("100000.0")
        risk_per_trade = Decimal("0.02")
        
        position_size = await strategy_protocol.calculate_position_size(
            signal, account_balance, risk_per_trade
        )
        assert isinstance(position_size, VolumeValue)

    async def test_get_strategy_performance(self, strategy_protocol):
        """Тест получения производительности стратегии."""
        performance = await strategy_protocol.get_strategy_performance(
            StrategyId("test_strategy")
        )
        assert isinstance(performance, PerformanceMetrics)


class TestIntegrationTestRepositoryProtocol:
    """Тесты для IntegrationTestRepositoryProtocol."""

    @pytest.fixture
    def repository_protocol(self):
        """Фикстура протокола репозитория."""
        return IntegrationTestRepositoryProtocol()

    def test_initialization(self, repository_protocol):
        """Тест инициализации."""
        assert repository_protocol.data == {}
        assert repository_protocol.cache == {}

    async def test_create(self, repository_protocol):
        """Тест создания записи."""
        data = {"name": "test", "value": 42}
        result = await repository_protocol.create("test_collection", data)
        assert isinstance(result, dict)
        assert "id" in result

    async def test_read(self, repository_protocol):
        """Тест чтения записи."""
        data = {"name": "test", "value": 42}
        created = await repository_protocol.create("test_collection", data)
        
        result = await repository_protocol.read("test_collection", created["id"])
        assert result is not None
        assert result["name"] == "test"

    async def test_update(self, repository_protocol):
        """Тест обновления записи."""
        data = {"name": "test", "value": 42}
        created = await repository_protocol.create("test_collection", data)
        
        updated_data = {"name": "updated", "value": 100}
        result = await repository_protocol.update(updated_data)
        assert isinstance(result, dict)

    async def test_delete(self, repository_protocol):
        """Тест удаления записи."""
        data = {"name": "test", "value": 42}
        created = await repository_protocol.create("test_collection", data)
        
        result = await repository_protocol.delete(created["id"])
        assert result is True

    async def test_save(self, repository_protocol):
        """Тест сохранения записи."""
        data = {"name": "test", "value": 42}
        result = await repository_protocol.save(data)
        assert isinstance(result, dict)

    async def test_get_by_id(self, repository_protocol):
        """Тест получения по ID."""
        data = {"name": "test", "value": 42}
        created = await repository_protocol.create("test_collection", data)
        
        result = await repository_protocol.get_by_id(created["id"])
        assert result is not None

    async def test_exists(self, repository_protocol):
        """Тест проверки существования."""
        data = {"name": "test", "value": 42}
        created = await repository_protocol.create("test_collection", data)
        
        exists = await repository_protocol.exists(created["id"])
        assert exists is True

    async def test_bulk_save(self, repository_protocol):
        """Тест пакетного сохранения."""
        entities = [
            {"name": "test1", "value": 1},
            {"name": "test2", "value": 2}
        ]
        
        result = await repository_protocol.bulk_save(entities)
        assert isinstance(result, BulkOperationResult)

    async def test_find_by(self, repository_protocol):
        """Тест поиска по фильтрам."""
        filters = [{"field": "name", "operator": "eq", "value": "test"}]
        result = await repository_protocol.find_by(filters)
        assert isinstance(result, list)

    async def test_count(self, repository_protocol):
        """Тест подсчета записей."""
        count = await repository_protocol.count()
        assert isinstance(count, int)

    async def test_get_all(self, repository_protocol):
        """Тест получения всех записей."""
        all_records = await repository_protocol.get_all()
        assert isinstance(all_records, list)

    async def test_transaction(self, repository_protocol):
        """Тест транзакций."""
        async with repository_protocol.transaction() as transaction:
            assert transaction.is_active() is True
            await transaction.commit()


class TestTradingSystemIntegration:
    """Тесты для TradingSystemIntegration."""

    @pytest.fixture
    def trading_system(self):
        """Фикстура торговой системы."""
        return TradingSystemIntegration()

    def test_initialization(self, trading_system):
        """Тест инициализации."""
        assert trading_system.exchange_protocol is not None
        assert trading_system.ml_protocol is not None
        assert trading_system.strategy_protocol is not None
        assert trading_system.repository_protocol is not None
        assert trading_system.security_manager is not None
        assert trading_system.monitor is not None
        assert trading_system.running is False

    async def test_initialize(self, trading_system):
        """Тест инициализации системы."""
        result = await trading_system.initialize()
        assert result is True

    async def test_start_stop(self, trading_system):
        """Тест запуска и остановки."""
        await trading_system.initialize()
        
        result = await trading_system.start()
        assert result is True
        assert trading_system.running is True
        
        result = await trading_system.stop()
        assert result is True
        assert trading_system.running is False

    async def test_get_status(self, trading_system):
        """Тест получения статуса."""
        status = await trading_system.get_status()
        assert isinstance(status, dict)
        assert "running" in status
        assert "exchange_connected" in status
        assert "active_strategies" in status

    async def test_run_trading_cycle(self, trading_system):
        """Тест выполнения торгового цикла."""
        await trading_system.initialize()
        await trading_system.start()
        
        cycle_result = await trading_system.run_trading_cycle()
        assert isinstance(cycle_result, dict)
        assert "signals_generated" in cycle_result
        assert "orders_executed" in cycle_result
        assert "performance_metrics" in cycle_result


class TestIntegrationTests:
    """Тесты интеграционных функций."""

    @pytest.mark.asyncio
    async def test_exchange_protocol_integration(self):
        """Тест интеграции протокола биржи."""
        await test_exchange_protocol_integration()

    @pytest.mark.asyncio
    async def test_ml_protocol_integration(self):
        """Тест интеграции ML протокола."""
        await test_ml_protocol_integration()

    @pytest.mark.asyncio
    async def test_strategy_protocol_integration(self):
        """Тест интеграции протокола стратегии."""
        await test_strategy_protocol_integration()

    @pytest.mark.asyncio
    async def test_repository_protocol_integration(self):
        """Тест интеграции протокола репозитория."""
        await test_repository_protocol_integration()

    @pytest.mark.asyncio
    async def test_trading_system_integration(self):
        """Тест интеграции торговой системы."""
        await test_trading_system_integration()

    @pytest.mark.asyncio
    async def test_protocol_decorators_integration(self):
        """Тест интеграции декораторов протоколов."""
        await test_protocol_decorators_integration()

    @pytest.mark.asyncio
    async def test_protocol_validators_integration(self):
        """Тест интеграции валидаторов протоколов."""
        await test_protocol_validators_integration()

    @pytest.mark.asyncio
    async def test_protocol_monitoring_integration(self):
        """Тест интеграции мониторинга протоколов."""
        await test_protocol_monitoring_integration()

    @pytest.mark.asyncio
    async def test_protocol_performance_integration(self):
        """Тест интеграции производительности протоколов."""
        await test_protocol_performance_integration()


class TestIntegrationUtilities:
    """Тесты утилит интеграции."""

    async def test_run_all_integration_tests(self):
        """Тест запуска всех интеграционных тестов."""
        results = await run_all_integration_tests()
        assert isinstance(results, list)


class TestIntegrationErrorHandling:
    """Тесты обработки ошибок интеграции."""

    async def test_exchange_protocol_error_handling(self):
        """Тест обработки ошибок протокола биржи."""
        exchange_protocol = IntegrationTestExchangeProtocol()
        
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await exchange_protocol.get_market_data("")  # Пустой символ

    async def test_ml_protocol_error_handling(self):
        """Тест обработки ошибок ML протокола."""
        ml_protocol = IntegrationTestMLProtocol()
        
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await ml_protocol.predict(uuid4(), {})  # Несуществующая модель

    async def test_strategy_protocol_error_handling(self):
        """Тест обработки ошибок протокола стратегии."""
        strategy_protocol = IntegrationTestStrategyProtocol()
        
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await strategy_protocol.analyze_market(
                pd.DataFrame(), DomainStrategyType.MOMENTUM
            )  # Пустые данные

    async def test_repository_protocol_error_handling(self):
        """Тест обработки ошибок протокола репозитория."""
        repository_protocol = IntegrationTestRepositoryProtocol()
        
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await repository_protocol.read("", "")  # Пустые параметры


class TestIntegrationPerformance:
    """Тесты производительности интеграции."""

    async def test_trading_cycle_performance(self):
        """Тест производительности торгового цикла."""
        trading_system = TradingSystemIntegration()
        await trading_system.initialize()
        await trading_system.start()
        
        start_time = datetime.now()
        await trading_system.run_trading_cycle()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        assert duration < 5.0  # Цикл должен выполняться менее 5 секунд

    async def test_bulk_operations_performance(self):
        """Тест производительности пакетных операций."""
        repository_protocol = IntegrationTestRepositoryProtocol()
        
        # Создаем большое количество записей
        entities = [
            {"name": f"test_{i}", "value": i}
            for i in range(1000)
        ]
        
        start_time = datetime.now()
        result = await repository_protocol.bulk_save(entities)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        assert duration < 10.0  # Пакетная операция должна выполняться менее 10 секунд
        assert result.success_count == 1000 