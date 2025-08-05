"""
Тесты для промышленных протоколов домена.
Проверяют корректность работы всех протоколов и их реализаций.
"""
import pytest
import asyncio
from typing import Any
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import List
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.strategy import Signal, SignalType, SignalStrength
from domain.entities.ml import Model, ModelType, Prediction, PredictionType, ModelStatus
from domain.entities.market import MarketData
from domain.entities.position import Position
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.protocols.ml_protocol import MLProtocol
from domain.protocols.strategy_protocol import StrategyProtocol
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.type_definitions import (
    OrderId, TradeId, PositionId, StrategyId, ModelId, PredictionId,
    PortfolioId, RiskProfileId, Symbol, TradingPair, PriceValue, VolumeValue, TimestampValue
)
from domain.value_objects import Money, Price, Volume
from domain.value_objects.currency import Currency
from domain.exceptions.protocol_exceptions import (
    ExchangeConnectionError, ModelNotFoundError, StrategyNotFoundError,
    EntityNotFoundError, ValidationError, TimeoutError, RetryExhaustedError
)
from domain.protocols.examples import ExampleExchangeProtocol, ExampleMLProtocol
from domain.protocols.utils import (
    retry_on_error, timeout, validate_symbol, ProtocolCache, ProtocolMetrics,
    log_operation, log_error
)
from domain.value_objects.timestamp import Timestamp
# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_order() -> Order:
    """Фикстура для тестового ордера."""
    return Order(
        id=OrderId(uuid4()),
        trading_pair=TradingPair("BTCUSDT"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=VolumeValue(Decimal("0.001")),
        price=Price(Decimal("50000"), Currency.USD),
        status=OrderStatus.PENDING
    )


@pytest.fixture
def sample_signal() -> Signal:
    """Фикстура для тестового сигнала."""
    return Signal(
        id=uuid4(),
        strategy_id=uuid4(),
        trading_pair="BTCUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("0.8"),
        price=Money(Decimal("50000"), Currency.USD),
        quantity=Decimal("0.001")
    )


@pytest.fixture
def sample_model() -> Model:
    """Фикстура для тестовой модели."""
    return Model(
        id=uuid4(),
        name="Test Model",
        model_type=ModelType.RANDOM_FOREST,
        trading_pair="BTCUSDT",
        prediction_type=PredictionType.PRICE,
        hyperparameters={"n_estimators": 100},
        features=["price", "volume"],
        target="next_price"
    )


@pytest.fixture
def sample_market_data() -> List[MarketData]:
    """Фикстура для тестовых рыночных данных."""
    return [
        MarketData(
            symbol=Symbol("BTCUSDT"),
            timestamp=TimestampValue(Timestamp.now().value - timedelta(minutes=i)),
            open=Price(Decimal("50000") + Decimal(str(i)), Currency.USDT),
            high=Price(Decimal("50100") + Decimal(str(i)), Currency.USDT),
            low=Price(Decimal("49900") + Decimal(str(i)), Currency.USDT),
            close=Price(Decimal("50050") + Decimal(str(i)), Currency.USDT),
            volume=Volume(Decimal("1000") + Decimal(str(i * 10)), Currency.USDT)
        )
        for i in range(10)
    ]
# ============================================================================
# EXCHANGE PROTOCOL TESTS
# ============================================================================
class TestExchangeProtocol:
    """Тесты для протокола биржи."""
    
    async def test_initialization(self: "TestExchangeProtocol") -> None:
        """Тест инициализации биржи."""
        exchange = ExampleExchangeProtocol("binance")
        config = {"api_key": "test", "api_secret": "test"}
        result = await exchange.initialize(config)
        assert result is True
        assert exchange.is_connected is True
    
    @pytest.mark.asyncio
    async def test_connection(self: "TestExchangeProtocol") -> None:
        """Тест подключения к бирже."""
        exchange = ExampleExchangeProtocol("binance")
        result = await exchange.connect()
        assert result is True
        assert exchange.is_connected is True
    
    @pytest.mark.asyncio
    async def test_disconnection(self: "TestExchangeProtocol") -> None:
        """Тест отключения от биржи."""
        exchange = ExampleExchangeProtocol("binance")
        await exchange.connect()
        result = await exchange.disconnect()
        assert result is True
        assert exchange.is_connected is False
    @pytest.mark.asyncio

    async def test_get_market_data(self: "TestExchangeProtocol") -> None:
        """Тест получения рыночных данных."""
        exchange = ExampleExchangeProtocol("binance")
        await exchange.initialize({})
        symbol = Symbol("BTCUSDT")
        market_data = await exchange.get_market_data(symbol, "1h", 10)
        assert len(market_data) == 10
        assert all(isinstance(data, MarketData) for data in market_data)
        assert all(data.symbol == symbol for data in market_data)
    @pytest.mark.asyncio

    async def test_create_order(self: "TestExchangeProtocol") -> None:
        """Тест создания ордера."""
        exchange = ExampleExchangeProtocol("binance")
        symbol = Symbol("BTCUSDT")
        order = await exchange.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=VolumeValue(Decimal("0.001")),
            price=PriceValue(Decimal("50000"))
        )
        assert isinstance(order, Order)
        assert order.trading_pair == TradingPair(str(symbol))
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.PENDING
    @pytest.mark.asyncio

    async def test_create_order_without_price_for_limit(self: "TestExchangeProtocol") -> None:
        """Тест создания лимитного ордера без цены."""
        exchange = ExampleExchangeProtocol("binance")
        with pytest.raises(ValidationError):
            await exchange.create_order(
                symbol=Symbol("BTCUSDT"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=VolumeValue(Decimal("0.001"))
                # price отсутствует
            )
    @pytest.mark.asyncio
    async def test_place_order(self, sample_order) -> None:
        """Тест размещения ордера."""
        exchange = ExampleExchangeProtocol("binance")
        placed_order = await exchange.place_order(sample_order)
        assert placed_order.status == OrderStatus.OPEN
        assert placed_order.id == sample_order.id
    @pytest.mark.asyncio

    async def test_cancel_order(self: "TestExchangeProtocol") -> None:
        """Тест отмены ордера."""
        exchange = ExampleExchangeProtocol("binance")
        order_id = OrderId(uuid4())
        result = await exchange.cancel_order(order_id)
        assert result is True
    @pytest.mark.asyncio

    async def test_get_balance(self: "TestExchangeProtocol") -> None:
        """Тест получения баланса."""
        exchange = ExampleExchangeProtocol("binance")
        balance = await exchange.get_balance()
        assert isinstance(balance, dict)
        assert Currency.USD in balance
        assert Currency.BTC in balance
        assert all(isinstance(money, Money) for money in balance.values())
    @pytest.mark.asyncio

    async def test_get_positions(self: "TestExchangeProtocol") -> None:
        """Тест получения позиций."""
        exchange = ExampleExchangeProtocol("binance")
        positions = await exchange.get_positions()
        assert isinstance(positions, list)
        assert all(isinstance(pos, Position) for pos in positions)
# ============================================================================
# ML PROTOCOL TESTS
# ============================================================================
class TestMLProtocol:
    """Тесты для ML протокола."""
    @pytest.mark.asyncio

    async def test_create_model(self: "TestMLProtocol") -> None:
        """Тест создания модели."""
        ml_protocol = ExampleMLProtocol()
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        assert isinstance(model, Model)
        assert model.name == "Test Model"
        assert model.model_type == ModelType.RANDOM_FOREST
        assert model.trading_pair == "BTCUSDT"
    @pytest.mark.asyncio

    async def test_train_model(self: "TestMLProtocol") -> None:
        """Тест обучения модели."""
        ml_protocol = ExampleMLProtocol()
        # Создаем модель
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        # Обучаем модель
        training_data = {
            "features": [[50000, 1000]],
            "targets": [50100]
        }
        trained_model = await ml_protocol.train_model(model.id, training_data)
        assert trained_model.status == ModelStatus.TRAINED
        assert trained_model.accuracy > 0
        assert trained_model.precision > 0
        assert trained_model.recall > 0
        assert trained_model.f1_score > 0
    @pytest.mark.asyncio

    async def test_train_nonexistent_model(self: "TestMLProtocol") -> None:
        """Тест обучения несуществующей модели."""
        ml_protocol = ExampleMLProtocol()
        nonexistent_id = ModelId(uuid4())
        with pytest.raises(ModelNotFoundError):
            await ml_protocol.train_model(nonexistent_id, {})
    @pytest.mark.asyncio

    async def test_predict(self: "TestMLProtocol") -> None:
        """Тест предсказания."""
        ml_protocol = ExampleMLProtocol()
        # Создаем и обучаем модель
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        await ml_protocol.train_model(model.id, {"features": [[50000, 1000]], "targets": [50100]})
        await ml_protocol.activate_model(model.id)
        # Делаем предсказание
        features = {"price": 50000, "volume": 1000}
        prediction = await ml_protocol.predict(model.id, features)
        assert prediction is not None
        assert isinstance(prediction, Prediction)
        assert prediction.model_id == model.id
        assert prediction.confidence > 0
    @pytest.mark.asyncio

    async def test_predict_with_inactive_model(self: "TestMLProtocol") -> None:
        """Тест предсказания с неактивной моделью."""
        ml_protocol = ExampleMLProtocol()
        # Создаем модель без активации
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        with pytest.raises(ModelNotReadyError):
            await ml_protocol.predict(model.id, {"price": 50000, "volume": 1000})
    @pytest.mark.asyncio

    async def test_predict_with_confidence_threshold(self: "TestMLProtocol") -> None:
        """Тест предсказания с порогом уверенности."""
        ml_protocol = ExampleMLProtocol()
        # Создаем и активируем модель
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        await ml_protocol.train_model(model.id, {"features": [[50000, 1000]], "targets": [50100]})
        await ml_protocol.activate_model(model.id)
        # Предсказание с высоким порогом уверенности
        features = {"price": 50000, "volume": 1000}
        prediction = await ml_protocol.predict(
            model.id, 
            features, 
            confidence_threshold=ConfidenceLevel(Decimal("0.9"))
        )
        # Должно вернуть None из-за высокого порога
        assert prediction is None
    @pytest.mark.asyncio

    async def test_batch_predict(self: "TestMLProtocol") -> None:
        """Тест пакетного предсказания."""
        ml_protocol = ExampleMLProtocol()
        # Создаем и активируем модель
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        await ml_protocol.train_model(model.id, {"features": [[50000, 1000]], "targets": [50100]})
        await ml_protocol.activate_model(model.id)
        # Пакетное предсказание
        features_batch = [
            {"price": 50000, "volume": 1000},
            {"price": 50100, "volume": 1100},
            {"price": 50200, "volume": 1200}
        ]
        predictions = await ml_protocol.batch_predict(model.id, features_batch)
        assert len(predictions) == 3
        assert all(isinstance(pred, Prediction) for pred in predictions)
    @pytest.mark.asyncio

    async def test_evaluate_model(self: "TestMLProtocol") -> None:
        """Тест оценки модели."""
        ml_protocol = ExampleMLProtocol()
        # Создаем модель
        model = await ml_protocol.create_model(
            name="Test Model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        # Оцениваем модель
        test_data = {"features": [[50000, 1000]], "targets": [50100]}
        metrics = await ml_protocol.evaluate_model(model.id, test_data)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(isinstance(value, float) for value in metrics.values())
# ============================================================================
# UTILITY TESTS
# ============================================================================
class TestProtocolUtils:
    """Тесты для утилит протоколов."""


def test_validate_symbol_valid(self: "TestProtocolUtils") -> None:
        """Тест валидации корректного символа."""
        symbol_str = "BTCUSDT"
        symbol = validate_symbol(symbol_str)
        # Проверяем, что это Symbol, но не используем isinstance с NewType
        assert str(symbol) == symbol_str


def test_validate_symbol_invalid(self: "TestProtocolUtils") -> None:
        """Тест валидации некорректного символа."""
        with pytest.raises(ValidationError):
            validate_symbol("")  # Пустой символ
        with pytest.raises(ValidationError):
            validate_symbol("A" * 25)  # Слишком длинный символ


def test_validate_symbol_wrong_type(self: "TestProtocolUtils") -> None:
        """Тест валидации символа неправильного типа."""
        with pytest.raises(ValidationError):
            validate_symbol(123)  # Не строка


def test_protocol_cache(self: "TestProtocolUtils") -> None:
        """Тест кэша протоколов."""
        cache = ProtocolCache(ttl_seconds=1)
        # Установка значения
        cache.set("test_key", "test_value")
        # Получение значения
        value = cache.get("test_key")
        assert value == "test_value"
        # Проверка истечения
        import time
        time.sleep(1.1)
        expired_value = cache.get("test_key")
        assert expired_value is None


def test_protocol_metrics(self: "TestProtocolUtils") -> None:
        """Тест метрик протоколов."""
        metrics = ProtocolMetrics()
        # Запись метрик
        metrics.record_operation("test_op", 1.5, success=True)
        metrics.record_operation("test_op", 2.0, success=False, error_type="TimeoutError")
        # Получение статистики
        stats = metrics.get_operation_stats("test_op")
        assert stats["count"] == 2
        assert stats["avg_time"] == 1.75
        assert stats["min_time"] == 1.5
        assert stats["max_time"] == 2.0
        assert stats["error_count"] == 1
# ============================================================================
# DECORATOR TESTS
# ============================================================================
class TestProtocolDecorators:
    """Тесты для декораторов протоколов."""
    @pytest.mark.asyncio

    async def test_retry_on_error_success(self: "TestProtocolDecorators") -> None:
        """Тест декоратора retry_on_error при успехе."""
        call_count = 0
        @retry_on_error(max_retries=3, delay=0.1)
        async def successful_operation() -> Any:
            nonlocal call_count
            call_count += 1
            return "success"
        result = await successful_operation()
        assert result == "success"
        assert call_count == 1
    @pytest.mark.asyncio

    async def test_retry_on_error_failure(self: "TestProtocolDecorators") -> None:
        """Тест декоратора retry_on_error при неудаче."""
        call_count = 0
        @retry_on_error(max_retries=2, delay=0.1)
        async def failing_operation() -> Any:
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")
        with pytest.raises(RetryExhaustedError):
            await failing_operation()
        assert call_count == 3  # 1 + 2 retries
    @pytest.mark.asyncio

    async def test_timeout_decorator_success(self: "TestProtocolDecorators") -> None:
        """Тест декоратора timeout при успехе."""
        @timeout(1.0)
        async def fast_operation() -> Any:
            await asyncio.sleep(0.1)
            return "success"
        result = await fast_operation()
        assert result == "success"
    @pytest.mark.asyncio

    async def test_timeout_decorator_failure(self: "TestProtocolDecorators") -> None:
        """Тест декоратора timeout при превышении времени."""
        @timeout(0.1)
        async def slow_operation() -> Any:
            await asyncio.sleep(1.0)
            return "success"
        with pytest.raises(TimeoutError):
            await slow_operation()
# ============================================================================
# INTEGRATION TESTS
# ============================================================================
class TestProtocolIntegration:
    """Интеграционные тесты протоколов."""
    @pytest.mark.asyncio

    async def test_exchange_ml_integration(self: "TestProtocolIntegration") -> None:
        """Тест интеграции биржи и ML протоколов."""
        exchange = ExampleExchangeProtocol("binance")
        ml_protocol = ExampleMLProtocol()
        # Инициализация биржи
        await exchange.initialize({})
        # Получение рыночных данных
        symbol = Symbol("BTCUSDT")
        market_data = await exchange.get_market_data(symbol, "1h", 10)
        # Создание и обучение модели
        model = await ml_protocol.create_model(
            name="BTC Predictor",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=symbol,
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="next_price"
        )
        # Подготовка данных для обучения
        training_data = {
            "features": [[float(data.close), float(data.volume)] for data in market_data],
            "targets": [float(data.close) + 100 for data in market_data]  # Простая цель
        }
        # Обучение модели
        trained_model = await ml_protocol.train_model(model.id, training_data)
        await ml_protocol.activate_model(model.id)
        # Предсказание на основе последних данных
        latest_data = market_data[0]
        features = {
            "price": float(latest_data.close),
            "volume": float(latest_data.volume)
        }
        prediction = await ml_protocol.predict(model.id, features)
        # Проверки
        assert trained_model.status == ModelStatus.TRAINED
        assert prediction is not None
        assert prediction.model_id == model.id
        assert prediction.confidence > 0
    @pytest.mark.asyncio

    async def test_error_handling_integration(self: "TestProtocolIntegration") -> None:
        """Тест обработки ошибок в интеграции."""
        exchange = ExampleExchangeProtocol("binance")
        # Тест обработки ошибки подключения
        try:
            await exchange.initialize({"invalid": "config"})
        except Exception as e:
            # Ошибка должна быть обработана
            handled = await exchange.handle_error(e)
            assert handled is True
# ============================================================================
# PERFORMANCE TESTS
# ============================================================================
class TestProtocolPerformance:
    """Тесты производительности протоколов."""
    @pytest.mark.asyncio

    async def test_cache_performance(self: "TestProtocolPerformance") -> None:
        """Тест производительности кэша."""
        cache = ProtocolCache(ttl_seconds=60)
        # Заполнение кэша
        start_time = datetime.now()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        # Чтение из кэша
        for i in range(1000):
            value = cache.get(f"key_{i}")
            assert value == f"value_{i}"
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        # Операции должны выполняться быстро
        assert duration < 1.0  # Менее 1 секунды
    @pytest.mark.asyncio

    async def test_metrics_performance(self: "TestProtocolPerformance") -> None:
        """Тест производительности метрик."""
        metrics = ProtocolMetrics()
        # Запись большого количества метрик
        start_time = datetime.now()
        for i in range(10000):
            metrics.record_operation(f"op_{i % 10}", i / 1000.0, success=True)
        # Получение статистики
        for i in range(10):
            stats = metrics.get_operation_stats(f"op_{i}")
            assert stats["count"] > 0
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        # Операции должны выполняться быстро
        assert duration < 1.0  # Менее 1 секунды
# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
