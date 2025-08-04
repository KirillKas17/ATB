"""
Примеры использования промышленных протоколов домена.
Демонстрирует правильные паттерны работы с протоколами.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from domain.entities.market import MarketData
from domain.entities.ml import Model, ModelType, Prediction, PredictionType, ModelStatus
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.position import Position
from domain.exceptions.protocol_exceptions import (
    EntityNotFoundError,
    ExchangeConnectionError,
    ModelNotFoundError,
    ModelNotReadyError,
    StrategyNotFoundError,
    ValidationError,
)
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.protocols.ml_protocol import MLProtocol, ModelConfig, TrainingConfig, PredictionConfig
from domain.protocols.repository_protocol import (
    MarketRepositoryProtocol,
    MLRepositoryProtocol,
    PortfolioRepositoryProtocol,
    RepositoryProtocol,
    RiskRepositoryProtocol,
    StrategyRepositoryProtocol,
    TradingRepositoryProtocol,
)
from domain.protocols.utils import (
    ProtocolCache,
    ProtocolMetrics,
    log_error,
    log_operation,
    retry_on_error,
    timeout,
    validate_symbol,
)
from domain.type_definitions import (
    ModelId,
    OrderId,
    PortfolioId,
    PositionId,
    PredictionId,
    PriceValue,
    RiskProfileId,
    StrategyId,
    Symbol,
    TradeId,
    TradingPair,
    VolumeValue,
    ConfidenceLevel,
    TimestampValue,
)
from domain.value_objects import Money, Price, Volume
from domain.value_objects.currency import Currency


# ============================================================================
# EXAMPLE IMPLEMENTATIONS
# ============================================================================
class ExampleExchangeProtocol(ExchangeProtocol):
    """Пример реализации протокола биржи."""

    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self._is_connected = False
        self.cache = ProtocolCache(ttl_seconds=60)
        self.metrics = ProtocolMetrics()

    @retry_on_error(max_retries=3, delay=1.0)
    @timeout(30.0)
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Инициализация подключения к бирже."""
        try:
            log_operation(
                "initialize", "exchange", extra_data={"exchange": self.exchange_name}
            )
            start_time = datetime.now()
            # Симуляция инициализации
            await asyncio.sleep(0.1)
            self._is_connected = True
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation("initialize", duration, success=True)
            return True
        except Exception as e:
            log_error(
                e, "initialize", "exchange", extra_data={"exchange": self.exchange_name}
            )
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation(
                "initialize", duration, success=False, error_type=type(e).__name__
            )
            raise

    @retry_on_error(max_retries=2, delay=0.5)
    async def connect(self) -> bool:
        """Установка соединения с биржей."""
        if self._is_connected:
            return True
        # Симуляция подключения
        await asyncio.sleep(0.05)
        self._is_connected = True
        return True

    async def disconnect(self) -> bool:
        """Отключение от биржи."""
        self._is_connected = False
        return True

    async def is_connected(self) -> bool:
        """Проверка состояния соединения."""
        return self._is_connected

    async def get_market_data(
        self, symbol: Symbol, timeframe: str = "1m", limit: int = 100
    ) -> List[MarketData]:
        """Получение рыночных данных."""
        # Проверяем кэш
        cache_key = f"market_data_{symbol}_{timeframe}_{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        # Симуляция получения данных
        market_data = []
        for i in range(limit):
            data = MarketData(
                symbol=symbol,
                timestamp=TimestampValue(datetime.now() - timedelta(minutes=i)),
                open=Price(Decimal("50000") + Decimal(i), Currency.USD),
                high=Price(Decimal("50100") + Decimal(i), Currency.USD),
                low=Price(Decimal("49900") + Decimal(i), Currency.USD),
                close=Price(Decimal("50050") + Decimal(i), Currency.USD),
                volume=Volume(Decimal("1000") + Decimal(i * 10)),
            )
            market_data.append(data)
        # Сохраняем в кэш
        self.cache.set(cache_key, market_data, ttl_seconds=30)
        return market_data

    async def create_order(self, order: Order) -> Dict[str, Any]:
        """Создание торгового ордера."""
        # Валидация
        validate_symbol(str(order.trading_pair))
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            raise ValidationError(
                "Price is required for limit orders",
                field_name="price",
                field_value=order.price,
                validation_rule="required_for_limit",
            )
        # Симуляция создания ордера
        await asyncio.sleep(0.1)
        return {
            "order_id": str(order.id),
            "status": "created",
            "symbol": str(order.trading_pair),
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price.amount) if order.price else None,
        }

    async def place_order(self, order: Order) -> Order:
        """Размещение ордера на бирже."""
        # Симуляция размещения
        await asyncio.sleep(0.1)
        order.status = OrderStatus.OPEN
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        # Симуляция отмены
        await asyncio.sleep(0.05)
        return True

    async def get_order_status(self, order_id: OrderId) -> OrderStatus:
        """Получение статуса ордера."""
        # Симуляция получения статуса
        return OrderStatus.OPEN

    async def get_balance(self) -> Dict[Currency, Money]:
        """Получение баланса."""
        return {
            Currency.USD: Money(Decimal("10000"), Currency.USD),
            Currency.BTC: Money(Decimal("1.5"), Currency.BTC),
        }

    async def get_positions(self) -> List[Position]:
        """Получение позиций."""
        return []

    # Реализация остальных методов...
    async def get_ticker(self, symbol: Symbol) -> Dict[str, Any]:
        return {"symbol": str(symbol), "price": "50000", "volume": "1000"}

    async def get_orderbook(
        self, symbol: Symbol, depth: int = 20
    ) -> Dict[str, List[Dict[str, Union[PriceValue, VolumeValue]]]]:
        return {"bids": [], "asks": []}

    async def get_recent_trades(
        self, symbol: Symbol, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return []

    async def get_account_info(self) -> Dict[str, Any]:
        return {"account_type": "spot", "permissions": ["spot"]}

    async def get_trading_fees(self) -> Dict[str, Decimal]:
        return {"maker": Decimal("0.001"), "taker": Decimal("0.001")}

    async def get_exchange_info(self) -> Dict[str, Any]:
        return {"timezone": "UTC", "serverTime": datetime.now().timestamp()}

    async def get_server_time(self) -> datetime:
        return datetime.now()

    async def test_connection(self) -> bool:
        return self._is_connected

    async def get_rate_limits(self) -> List[Dict[str, Any]]:
        return [{"rateLimitType": "REQUEST_WEIGHT", "limit": 1200}]

    async def handle_error(self, error: Exception) -> bool:
        log_error(error, "handle_error", "exchange")
        return True

    async def retry_operation(
        self, operation: Callable[[], Any], max_retries: int = 3, delay: float = 1.0
    ) -> Any:
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(delay)
        return None

    # Методы протокола ExchangeProtocol
    async def fetch_order(self, order_id: str) -> Dict[str, Any]:
        """Получение информации об ордере."""
        return {"order_id": order_id, "status": "open"}

    async def fetch_open_orders(self) -> List[Dict[str, Any]]:
        """Получение открытых ордеров."""
        return []

    async def fetch_balance(self) -> Dict[str, Any]:
        """Получение баланса."""
        return {"USD": "10000", "BTC": "1.5"}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера."""
        return {"symbol": symbol, "price": "50000", "volume": "1000"}

    async def fetch_order_book(self, symbol: str) -> Dict[str, Any]:
        """Получение ордербука."""
        return {"symbol": symbol, "bids": [], "asks": []}


class ExampleMLProtocol(MLProtocol):
    """Пример реализации ML протокола."""

    def __init__(self) -> None:
        """Инициализация примера протокола."""
        self.name = "Example Protocol"
        self.version = "1.0.0"
        self.models: Dict[ModelId, Model] = {}
        self.predictions: Dict[PredictionId, Prediction] = {}

    async def create_model(self, config: ModelConfig) -> Model:
        """Создание новой модели."""
        model_id = ModelId(uuid4())
        model = Model(
            id=model_id,
            name=config.name,
            model_type=config.model_type,
            trading_pair=config.trading_pair,
            prediction_type=config.prediction_type,
            hyperparameters=config.hyperparameters,
            features=config.features,
            target=config.target,
            description=config.description,
        )
        self.models[model_id] = model
        return model

    async def train_model(
        self,
        model_id: ModelId,
        training_data: Any,  # DataFrame в реальной реализации
        config: TrainingConfig,
        validation_data: Optional[Any] = None,  # DataFrame в реальной реализации
    ) -> Model:
        """Обучение модели."""
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        
        model = self.models[model_id]
        model.status = ModelStatus.TRAINING
        # Симуляция обучения
        await asyncio.sleep(0.1)
        model.status = ModelStatus.TRAINED
        return model

    async def predict(
        self,
        model_id: ModelId,
        features: Dict[str, Any],
        config: Optional[PredictionConfig] = None,
    ) -> Optional[Prediction]:
        """Предсказание с помощью модели."""
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        
        model = self.models[model_id]
        if model.status != ModelStatus.ACTIVE:
            raise ModelNotReadyError(
                f"Model {model_id} is not ready for prediction",
                model_id=model_id,
                current_status=model.status.value,
                required_status="active",
            )
        
        # Симуляция предсказания
        prediction_id = PredictionId(uuid4())
        prediction = Prediction(
            id=prediction_id,
            model_id=model_id,
            value=Decimal("0.75"),
            confidence=ConfidenceLevel(Decimal("0.85")),
            features=features,
            timestamp=TimestampValue(datetime.now()),
        )
        self.predictions[prediction_id] = prediction
        return prediction

    async def batch_predict(
        self,
        model_id: ModelId,
        features_batch: List[Dict[str, Any]],
        config: Optional[PredictionConfig] = None,
    ) -> List[Prediction]:
        """Пакетное предсказание."""
        predictions = []
        for features in features_batch:
            prediction = await self.predict(model_id, features, config)
            if prediction:
                predictions.append(prediction)
        return predictions

    async def evaluate_model(
        self,
        model_id: ModelId,
        test_data: Any,  # DataFrame в реальной реализации
        metrics: Optional[List[str]] = None,
    ) -> Any:  # ModelMetrics в реальной реализации
        """Оценка модели."""
        return {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}

    async def cross_validate(
        self,
        model_id: ModelId,
        data: Any,  # DataFrame в реальной реализации
        cv_folds: int = 5,
        cv_method: str = "time_series",
    ) -> Dict[str, List[float]]:
        """Кросс-валидация модели."""
        return {
            "accuracy": [0.85, 0.87, 0.83, 0.86, 0.84],
            "precision": [0.82, 0.84, 0.80, 0.83, 0.81],
        }

    async def backtest_model(
        self,
        model_id: ModelId,
        historical_data: Any,  # DataFrame в реальной реализации
        initial_capital: Decimal = Decimal("10000"),
        transaction_cost: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0001"),
    ) -> Dict[str, Any]:
        """Бэктестинг модели."""
        return {
            "total_return": Decimal("0.15"),
            "sharpe_ratio": 1.2,
            "max_drawdown": Decimal("0.05"),
        }

    async def optimize_hyperparameters(
        self,
        model_id: ModelId,
        training_data: Any,  # DataFrame в реальной реализации
        validation_data: Any,  # DataFrame в реальной реализации
        param_grid: Dict[str, List[Any]],
        optimization_method: Any = None,  # OptimizationMethod в реальной реализации
        n_trials: int = 100,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Оптимизация гиперпараметров."""
        return {"best_params": {"learning_rate": 0.01}, "best_score": 0.87}

    async def feature_selection(
        self,
        model_id: ModelId,
        data: Any,  # DataFrame в реальной реализации
        method: str = "mutual_info",
        n_features: Optional[int] = None,
        threshold: float = 0.01,
    ) -> List[str]:
        """Выбор признаков."""
        return ["feature1", "feature2", "feature3"]

    async def calculate_feature_importance(
        self, model_id: ModelId, data: Any  # DataFrame в реальной реализации
    ) -> Dict[str, float]:
        """Расчет важности признаков."""
        return {"feature1": 0.3, "feature2": 0.5, "feature3": 0.2}

    async def save_model(self, model_id: ModelId, path: str) -> bool:
        """Сохранение модели."""
        return True

    async def load_model(self, model_id: ModelId, path: str) -> Model:
        """Загрузка модели."""
        return self.models[model_id]

    async def export_model_metadata(self, model_id: ModelId, path: str) -> bool:
        """Экспорт метаданных модели."""
        return True

    async def import_model_metadata(self, path: str) -> ModelConfig:
        """Импорт метаданных модели."""
        return ModelConfig(
            name="Imported Model",
            model_type=ModelType.LSTM,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={},
            features=[],
            target="price",
        )

    async def get_model_status(self, model_id: ModelId) -> ModelStatus:
        """Получение статуса модели."""
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        return self.models[model_id].status

    async def activate_model(self, model_id: ModelId) -> bool:
        """Активация модели."""
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        self.models[model_id].status = ModelStatus.ACTIVE
        return True

    async def deactivate_model(self, model_id: ModelId) -> bool:
        """Деактивация модели."""
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        self.models[model_id].status = ModelStatus.INACTIVE
        return True

    async def delete_model(self, model_id: ModelId) -> bool:
        """Удаление модели."""
        if model_id in self.models:
            del self.models[model_id]
        return True

    async def archive_model(self, model_id: ModelId) -> bool:
        """Архивирование модели."""
        if model_id in self.models:
            self.models[model_id].status = ModelStatus.DEPRECATED
        return True

    async def get_model_performance_history(
        self,
        model_id: ModelId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Получение истории производительности модели."""
        return [{"date": datetime.now(), "accuracy": 0.85}]

    async def get_prediction_history(
        self, 
        model_id: ModelId, 
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Prediction]:
        """Получение истории предсказаний."""
        return list(self.predictions.values())[:limit]

    async def monitor_model_drift(
        self,
        model_id: ModelId,
        recent_data: Any,  # DataFrame в реальной реализации
        drift_threshold: float = 0.1,
        drift_method: str = "ks_test",
    ) -> Dict[str, Any]:
        """Мониторинг дрифта модели."""
        return {"drift_detected": False, "drift_score": 0.05}

    async def calculate_model_confidence(
        self, model_id: ModelId, prediction: Prediction
    ) -> ConfidenceLevel:
        """Расчет уверенности модели."""
        return ConfidenceLevel(Decimal("0.85"))

    async def create_ensemble(
        self,
        name: str,
        models: List[ModelId],
        weights: Optional[List[float]] = None,
        voting_method: str = "weighted_average",
        meta_learner: Optional[str] = None,
    ) -> ModelId:
        """Создание ансамбля моделей."""
        ensemble_id = ModelId(uuid4())
        # Создаем модель-ансамбль
        ensemble_model = Model(
            id=ensemble_id,
            name=name,
            model_type=ModelType.ENSEMBLE,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"voting_method": voting_method, "weights": weights},
            features=[],
            target="price",
        )
        self.models[ensemble_id] = ensemble_model
        return ensemble_id

    async def ensemble_predict(
        self, 
        ensemble_id: ModelId, 
        features: Dict[str, Any],
        config: Optional[PredictionConfig] = None,
    ) -> Prediction:
        """Предсказание ансамблем."""
        # Предсказание ансамблем
        prediction_id = PredictionId(uuid4())
        prediction = Prediction(
            id=prediction_id,
            model_id=ensemble_id,
            value=Decimal("0.78"),
            confidence=ConfidenceLevel(Decimal("0.90")),
            features=features,
            timestamp=TimestampValue(datetime.now()),
        )
        self.predictions[prediction_id] = prediction
        return prediction

    async def update_ensemble_weights(
        self, ensemble_id: ModelId, new_weights: List[float]
    ) -> bool:
        """Обновление весов ансамбля."""
        return True

    async def online_learning(
        self,
        model_id: ModelId,
        new_data: Any,  # DataFrame в реальной реализации
        learning_rate: float = 0.01,
        batch_size: int = 1,
    ) -> bool:
        """Онлайн обучение."""
        return True

    async def transfer_learning(
        self,
        source_model_id: ModelId,
        target_config: ModelConfig,
        fine_tune_layers: Optional[List[str]] = None,
    ) -> ModelId:
        """Перенос обучения."""
        return ModelId(uuid4())

    async def adaptive_learning(
        self, model_id: ModelId, market_regime: str, adaptation_rules: Dict[str, Any]
    ) -> bool:
        """Адаптивное обучение."""
        return True

    async def handle_model_error(self, model_id: ModelId, error: Exception) -> bool:
        """Обработка ошибки модели."""
        return True

    async def retry_prediction(
        self, 
        model_id: ModelId, 
        features: Dict[str, Any], 
        max_retries: int = 3,
        fallback_model_id: Optional[ModelId] = None,
    ) -> Optional[Prediction]:
        """Повторная попытка предсказания."""
        for attempt in range(max_retries):
            try:
                return await self.predict(model_id, features)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.1)
        return None

    async def validate_model_integrity(self, model_id: ModelId) -> bool:
        """Проверка целостности модели."""
        return True

    async def recover_model_state(
        self, model_id: ModelId, recovery_point: Optional[datetime] = None
    ) -> bool:
        """Восстановление состояния модели."""
        return True


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
async def example_exchange_usage() -> None:
    """Пример использования протокола биржи."""
    exchange = ExampleExchangeProtocol("Binance")
    
    # Инициализация
    await exchange.initialize({"api_key": "test", "api_secret": "test"})
    
    # Получение рыночных данных
    market_data = await exchange.get_market_data(Symbol("BTCUSDT"), "1m", 10)
    print(f"Получено {len(market_data)} записей рыночных данных")
    
    # Создание и размещение ордера
    order = Order(
        id=OrderId(uuid4()),
        trading_pair=TradingPair("BTCUSDT"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=VolumeValue(Decimal("0.001")),
        status=OrderStatus.PENDING,
    )
    order_result = await exchange.create_order(order)
    print(f"Ордер создан: {order_result}")


async def example_ml_usage() -> None:
    """Пример использования ML протокола."""
    ml_protocol = ExampleMLProtocol()
    
    # Создание модели
    config = ModelConfig(
        name="Test Model",
        model_type=ModelType.LSTM,
        trading_pair=Symbol("BTCUSDT"),
        prediction_type=PredictionType.PRICE,
        hyperparameters={"layers": 2, "units": 64},
        features=["price", "volume", "rsi"],
        target="price_change",
    )
    model = await ml_protocol.create_model(config)
    
    # Обучение модели
    training_config = TrainingConfig(
        validation_split=0.2,
        epochs=100,
        batch_size=32,
    )
    trained_model = await ml_protocol.train_model(
        model_id=ModelId(model.id),
        training_data={},  # DataFrame в реальной реализации
        config=training_config,
    )
    
    # Активация модели
    await ml_protocol.activate_model(ModelId(model.id))
    
    # Предсказание
    prediction_config = PredictionConfig(
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    prediction = await ml_protocol.predict(
        model_id=ModelId(model.id),
        features={"price": 50000, "volume": 1000, "rsi": 65},
        config=prediction_config,
    )
    if prediction is not None:
        print(f"Предсказание: {prediction.value} с уверенностью {prediction.confidence}")
    else:
        print("Предсказание не получено")


async def example_strategy_usage() -> None:
    """Пример использования протокола стратегий."""
    # Здесь будет пример использования стратегий
    pass


async def example_repository_usage() -> None:
    """Пример использования протоколов репозиториев."""
    # Здесь будет пример использования репозиториев
    pass


async def main() -> None:
    """Основная функция для демонстрации."""
    print("=== Примеры использования протоколов ===")
    
    print("\n1. Пример использования протокола биржи:")
    await example_exchange_usage()
    
    print("\n2. Пример использования ML протокола:")
    await example_ml_usage()
    
    print("\n3. Пример использования протокола стратегий:")
    await example_strategy_usage()
    
    print("\n4. Пример использования протоколов репозиториев:")
    await example_repository_usage()
    
    print("\n=== Демонстрация завершена ===")


if __name__ == "__main__":
    asyncio.run(main())
