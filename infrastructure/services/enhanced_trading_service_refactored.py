"""
Рефакторенный Enhanced Trading Service с декомпозицией и оптимизацией.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.entities.order import OrderSide, OrderStatus, OrderType
from infrastructure.shared.exceptions import (
    PerformanceError,
    ServiceError,
    ValidationError,
)
from infrastructure.shared.logging import (
    ServiceLogger,
    log_errors_decorator,
    log_performance_decorator,
)


class ExecutionAlgorithm(Enum):
    """Алгоритмы исполнения."""

    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    BRACKET = "bracket"


class StrategyType(Enum):
    """Типы стратегий."""

    SCALPING = "scalping"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    CUSTOM = "custom"


@dataclass
class OrderParameters:
    """Параметры ордера."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    post_only: bool = False

    def validate(self) -> None:
        """Валидация параметров ордера."""
        if not self.symbol:
            raise ValidationError("Symbol is required")
        if self.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if self.order_type == OrderType.LIMIT and not self.price:
            raise ValidationError("Price is required for limit order")
        if self.order_type == OrderType.STOP and not self.stop_price:
            raise ValidationError("Stop price is required for stop order")
        if self.order_type == OrderType.STOP_LIMIT and (
            not self.stop_price or not self.limit_price
        ):
            raise ValidationError(
                "Stop price and limit price are required for stop-limit order"
            )


@dataclass
class ExecutionParameters:
    """Параметры исполнения."""

    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    duration_minutes: int = 30
    max_slippage: Decimal = Decimal("0.01")
    target_vwap: Optional[Decimal] = None
    max_deviation: Decimal = Decimal("0.01")
    visible_quantity: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None


@dataclass
class SentimentAnalysis:
    """Анализ настроений."""

    market_sentiment: float = 0.0
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    combined_sentiment: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_combined_sentiment(self, weights: Dict[str, float]) -> None:
        """Расчет комбинированного настроения."""
        self.combined_sentiment = (
            self.market_sentiment * weights.get("market", 0.5)
            + self.news_sentiment * weights.get("news", 0.3)
            + self.social_sentiment * weights.get("social", 0.2)
        )


@dataclass
class PerformanceMetrics:
    """Метрики производительности."""

    execution_time_ms: float = 0.0
    slippage: Decimal = Decimal("0")
    fill_rate: float = 0.0
    avg_fill_price: Optional[Decimal] = None
    total_cost: Decimal = Decimal("0")
    success: bool = True
    error_message: Optional[str] = None


@runtime_checkable
class OrderCreator(Protocol):
    """Протокол для создания ордеров."""

    def create_order(self, params: OrderParameters) -> Dict[str, Any]:
        """Создание ордера."""
        ...


@runtime_checkable
class StrategyExecutor(Protocol):
    """Протокол для исполнения стратегий."""

    def execute_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        market_data: pd.DataFrame,
        capital: Decimal,
    ) -> Dict[str, Any]:
        """Исполнение стратегии."""
        ...


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Протокол для анализа настроений."""

    def analyze_sentiment(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[List[Dict[str, Any]]] = None,
        social_data: Optional[List[Dict[str, Any]]] = None,
    ) -> SentimentAnalysis:
        """Анализ настроений."""
        ...


class BaseOrderCreator:
    """Базовый создатель ордеров."""

    def __init__(self, logger: ServiceLogger) -> None:
        self.logger = logger

    def create_order(self, params: OrderParameters) -> Dict[str, Any]:
        """Создание базового ордера."""
        params.validate()
        order = {
            "symbol": params.symbol,
            "side": params.side,
            "quantity": params.quantity,
            "order_type": params.order_type,
            "time_in_force": params.time_in_force,
            "reduce_only": params.reduce_only,
            "post_only": params.post_only,
            "timestamp": datetime.now(),
            "status": OrderStatus.PENDING,
        }
        if params.price:
            order["price"] = params.price
        if params.stop_price:
            order["stop_price"] = params.stop_price
        if params.limit_price:
            order["limit_price"] = params.limit_price
        self.logger.log_service_call(
            "create_order", {"symbol": params.symbol, "type": params.order_type}
        )
        return order


class AdvancedOrderCreator(BaseOrderCreator):
    """Продвинутый создатель ордеров."""

    def create_twap_order(
        self, symbol: str, side: OrderSide, quantity: Decimal, duration_minutes: int
    ) -> Dict[str, Any]:
        """Создание TWAP ордера."""
        params = OrderParameters(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            time_in_force="GTC",
        )
        order = self.create_order(params)
        order.update(
            {
                "execution_algorithm": ExecutionAlgorithm.TWAP,
                "duration_minutes": duration_minutes,
                "chunks": self._calculate_twap_chunks(quantity, duration_minutes),
            }
        )
        return order

    def create_vwap_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        target_vwap: Decimal,
        max_deviation: Decimal,
    ) -> Dict[str, Any]:
        """Создание VWAP ордера."""
        params = OrderParameters(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            time_in_force="GTC",
        )
        order = self.create_order(params)
        order.update(
            {
                "execution_algorithm": ExecutionAlgorithm.VWAP,
                "target_vwap": target_vwap,
                "max_deviation": max_deviation,
            }
        )
        return order

    def create_iceberg_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        visible_quantity: Decimal,
        price: Decimal,
    ) -> Dict[str, Any]:
        """Создание айсберг ордера."""
        params = OrderParameters(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price,
            time_in_force="GTC",
        )
        order = self.create_order(params)
        order.update(
            {
                "execution_algorithm": ExecutionAlgorithm.ICEBERG,
                "visible_quantity": visible_quantity,
            }
        )
        return order

    def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        entry_price: Decimal,
        take_profit: Decimal,
        stop_loss: Decimal,
    ) -> Dict[str, Any]:
        """Создание брекет ордера."""
        params = OrderParameters(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=entry_price,
            time_in_force="GTC",
        )
        order = self.create_order(params)
        order.update(
            {
                "execution_algorithm": ExecutionAlgorithm.BRACKET,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
            }
        )
        return order

    def _calculate_twap_chunks(
        self, quantity: Decimal, duration_minutes: int
    ) -> List[Dict[str, Any]]:
        """Расчет чанков для TWAP."""
        chunk_count = max(1, duration_minutes // 5)  # Чанк каждые 5 минут
        chunk_size = quantity / chunk_count
        chunks = []
        for i in range(chunk_count):
            chunks.append(
                {
                    "chunk_id": i + 1,
                    "quantity": chunk_size,
                    "execution_time": i * 5 * 60,  # секунды
                }
            )
        return chunks


class BaseStrategyExecutor:
    """Базовый исполнитель стратегий."""

    def __init__(self, logger: ServiceLogger) -> None:
        self.logger = logger

    def execute_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        market_data: pd.DataFrame,
        capital: Decimal,
    ) -> Dict[str, Any]:
        """Базовое исполнение стратегии."""
        start_time = datetime.now()
        try:
            # Валидация параметров
            self._validate_strategy_parameters(parameters)
            # Расчет размера позиции
            position_size = self._calculate_position_size(parameters, capital)
            # Создание ордеров
            orders = self._create_strategy_orders(
                strategy_type, parameters, position_size
            )
            # Расчет метрик
            performance_metrics = self._calculate_performance_metrics(
                orders, market_data, start_time
            )
            result = {
                "strategy_type": strategy_type,
                "orders": orders,
                "position_size": position_size,
                "performance_metrics": performance_metrics,
                "execution_time": (datetime.now() - start_time).total_seconds(),
            }
            self.logger.log_service_call(
                "execute_strategy",
                {"strategy_type": strategy_type.value, "orders_count": len(orders)},
            )
            return result
        except Exception as e:
            self.logger.log_error(e, "execute_strategy")
            raise ServiceError(
                "EnhancedTradingService", f"Strategy execution failed: {str(e)}"
            )

    def _validate_strategy_parameters(self, parameters: Dict[str, Any]) -> None:
        """Валидация параметров стратегии."""
        required_params = ["risk_per_trade", "max_position_size"]
        for param in required_params:
            if param not in parameters:
                raise ValidationError(f"Required parameter missing: {param}")

    def _calculate_position_size(
        self, parameters: Dict[str, Any], capital: Decimal
    ) -> Decimal:
        """Расчет размера позиции."""
        risk_per_trade = Decimal(str(parameters.get("risk_per_trade", 0.02)))
        max_position_size = Decimal(str(parameters.get("max_position_size", 0.1)))
        # Базовый размер на основе риска
        risk_based_size = capital * risk_per_trade
        # Ограничение максимальным размером
        max_size = capital * max_position_size
        return min(risk_based_size, max_size)

    def _create_strategy_orders(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        position_size: Decimal,
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для стратегии."""
        orders = []
        if strategy_type == StrategyType.SCALPING:
            orders = self._create_scalping_orders(parameters, position_size)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            orders = self._create_mean_reversion_orders(parameters, position_size)
        elif strategy_type == StrategyType.MOMENTUM:
            orders = self._create_momentum_orders(parameters, position_size)
        elif strategy_type == StrategyType.ARBITRAGE:
            orders = self._create_arbitrage_orders(parameters, position_size)
        elif strategy_type == StrategyType.GRID:
            orders = self._create_grid_orders(parameters, position_size)
        else:
            raise ValidationError(f"Unsupported strategy type: {strategy_type}")
        return orders

    def _create_scalping_orders(
        self, parameters: Dict[str, Any], position_size: Decimal
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для скальпинга."""
        # Упрощенная реализация
        return [
            {
                "order_type": OrderType.MARKET,
                "side": OrderSide.BUY,
                "quantity": position_size,
                "parameters": parameters,
            }
        ]

    def _create_mean_reversion_orders(
        self, parameters: Dict[str, Any], position_size: Decimal
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для mean reversion."""
        return [
            {
                "order_type": OrderType.LIMIT,
                "side": OrderSide.BUY,
                "quantity": position_size,
                "parameters": parameters,
            }
        ]

    def _create_momentum_orders(
        self, parameters: Dict[str, Any], position_size: Decimal
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для momentum."""
        return [
            {
                "order_type": OrderType.MARKET,
                "side": OrderSide.BUY,
                "quantity": position_size,
                "parameters": parameters,
            }
        ]

    def _create_arbitrage_orders(
        self, parameters: Dict[str, Any], position_size: Decimal
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для арбитража."""
        return [
            {
                "order_type": OrderType.MARKET,
                "side": OrderSide.BUY,
                "quantity": position_size,
                "parameters": parameters,
            }
        ]

    def _create_grid_orders(
        self, parameters: Dict[str, Any], position_size: Decimal
    ) -> List[Dict[str, Any]]:
        """Создание ордеров для grid стратегии."""
        grid_levels = parameters.get("grid_levels", 5)
        orders = []
        for i in range(grid_levels):
            orders.append(
                {
                    "order_type": OrderType.LIMIT,
                    "side": OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    "quantity": position_size / grid_levels,
                    "parameters": parameters,
                }
            )
        return orders

    def _calculate_performance_metrics(
        self,
        orders: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        start_time: datetime,
    ) -> PerformanceMetrics:
        """Расчет метрик производительности."""
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return PerformanceMetrics(execution_time_ms=execution_time_ms, success=True)


class BaseSentimentAnalyzer:
    """Базовый анализатор настроений."""

    def __init__(self, logger: ServiceLogger) -> None:
        self.logger = logger

    def analyze_sentiment(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[List[Dict[str, Any]]] = None,
        social_data: Optional[List[Dict[str, Any]]] = None,
    ) -> SentimentAnalysis:
        """Базовый анализ настроений."""
        sentiment = SentimentAnalysis()
        # Анализ рыночных настроений
        if not market_data.empty:
            sentiment.market_sentiment = self._analyze_market_sentiment(market_data)
        # Анализ новостных настроений
        if news_data:
            sentiment.news_sentiment = self._analyze_news_sentiment(news_data)
        # Анализ социальных настроений
        if social_data:
            sentiment.social_sentiment = self._analyze_social_sentiment(social_data)
        # Расчет комбинированного настроения
        weights = {"market": 0.5, "news": 0.3, "social": 0.2}
        sentiment.calculate_combined_sentiment(weights)
        # Расчет уверенности
        sentiment.confidence = self._calculate_sentiment_confidence(
            sentiment.market_sentiment,
            sentiment.news_sentiment,
            sentiment.social_sentiment,
        )
        self.logger.log_service_call(
            "analyze_sentiment",
            {
                "market_sentiment": sentiment.market_sentiment,
                "combined_sentiment": sentiment.combined_sentiment,
                "confidence": sentiment.confidence,
            },
        )
        return sentiment

    def _analyze_market_sentiment(self, market_data: pd.DataFrame) -> float:
        """Анализ рыночных настроений."""
        if market_data.empty:
            return 0.0
        # Простой анализ на основе изменения цены
        if len(market_data) > 1:
            price_change = (
                market_data["close"].iloc[-1] - market_data["close"].iloc[0]
            ) / market_data["close"].iloc[0]
            return float(np.tanh(price_change * 10))  # Нормализация в диапазон [-1, 1]
        return 0.0

    def _analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> float:
        """Анализ новостных настроений."""
        if not news_data:
            return 0.0
        # Простой анализ на основе количества позитивных/негативных новостей
        positive_count = sum(1 for news in news_data if news.get("sentiment", 0) > 0)
        negative_count = sum(1 for news in news_data if news.get("sentiment", 0) < 0)
        if positive_count + negative_count == 0:
            return 0.0
        return (positive_count - negative_count) / (positive_count + negative_count)

    def _analyze_social_sentiment(self, social_data: List[Dict[str, Any]]) -> float:
        """Анализ социальных настроений."""
        if not social_data:
            return 0.0
        # Простой анализ на основе социальных метрик
        total_sentiment = sum(post.get("sentiment", 0) for post in social_data)
        return float(np.tanh(total_sentiment / len(social_data)))

    def _calculate_sentiment_confidence(
        self, market_sentiment: float, news_sentiment: float, social_sentiment: float
    ) -> float:
        """Расчет уверенности в настроениях."""
        # Простая метрика на основе согласованности
        sentiments = [market_sentiment, news_sentiment, social_sentiment]
        non_zero_sentiments = [s for s in sentiments if s != 0]
        if not non_zero_sentiments:
            return 0.0
        # Уверенность на основе согласованности и величины
        consistency = float(1.0 - np.std(non_zero_sentiments))
        magnitude = float(np.mean([abs(s) for s in non_zero_sentiments]))
        return min(1.0, consistency * magnitude)


class EnhancedTradingService:
    """Рефакторенный Enhanced Trading Service."""

    def __init__(
        self,
        order_creator: Optional[OrderCreator] = None,
        strategy_executor: Optional[StrategyExecutor] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        enable_cache: bool = True,
        enable_metrics: bool = True,
    ):
        self.logger = ServiceLogger("enhanced_trading_service")
        # Инициализация компонентов
        self.order_creator = order_creator or AdvancedOrderCreator(self.logger)
        self.strategy_executor = strategy_executor or BaseStrategyExecutor(self.logger)
        self.sentiment_analyzer = sentiment_analyzer or BaseSentimentAnalyzer(
            self.logger
        )
        # Настройки
        self.enable_cache = enable_cache
        self.enable_metrics = enable_metrics
        # Кэш для метрик
        self._performance_cache: Dict[str, PerformanceMetrics] = {}

    @log_performance_decorator("create_order")
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[Decimal] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Создание торгового ордера."""
        try:
            params = OrderParameters(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                **kwargs,
            )
            return self.order_creator.create_order(params)
        except Exception as e:
            self.logger.log_error(e, "create_order")
            raise ServiceError(
                "EnhancedTradingService", f"Failed to create order: {str(e)}"
            )

    @log_performance_decorator("create_advanced_order")
    def create_advanced_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: str,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Создание продвинутого ордера."""
        try:
            if not self.order_creator:
                raise ServiceError(
                    "EnhancedTradingService", "Advanced order creator not available"
                )
            # Создаём параметры ордера
            params = OrderParameters(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType(order_type),
                **kwargs,
            )
            # Создаём ордер
            order = self.order_creator.create_order(params)
            # Оптимизируем исполнение если есть рыночные данные
            if not market_data.empty:
                execution_params = ExecutionParameters(**kwargs)
                order = self.optimize_order_execution(
                    order, market_data, execution_params
                )
            return order
        except Exception as e:
            self.logger.log_error(e, "create_advanced_order")
            raise ServiceError(
                "EnhancedTradingService", f"Failed to create advanced order: {str(e)}"
            )

    @log_performance_decorator("execute_strategy")
    def execute_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        market_data: pd.DataFrame,
        capital: Decimal,
    ) -> Dict[str, Any]:
        """Исполнение торговой стратегии."""
        try:
            if not self.strategy_executor:
                raise ServiceError(
                    "EnhancedTradingService", "Strategy executor not available"
                )
            return self.strategy_executor.execute_strategy(
                strategy_type, parameters, market_data, capital
            )
        except Exception as e:
            self.logger.log_error(e, "execute_strategy")
            raise ServiceError(
                "EnhancedTradingService", f"Strategy execution failed: {str(e)}"
            )

    @log_performance_decorator("analyze_sentiment")
    def analyze_sentiment(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[List[Dict[str, Any]]] = None,
        social_data: Optional[List[Dict[str, Any]]] = None,
    ) -> SentimentAnalysis:
        """Анализ настроений."""
        try:
            if not self.sentiment_analyzer:
                return SentimentAnalysis()
            return self.sentiment_analyzer.analyze_sentiment(
                market_data, news_data, social_data
            )
        except Exception as e:
            self.logger.log_error(e, "analyze_sentiment")
            raise ServiceError(
                "EnhancedTradingService", f"Failed to analyze sentiment: {str(e)}"
            )

    @log_performance_decorator("optimize_order_execution")
    def optimize_order_execution(
        self,
        order: Dict[str, Any],
        market_data: pd.DataFrame,
        execution_parameters: Optional[ExecutionParameters] = None,
    ) -> Dict[str, Any]:
        """Оптимизация исполнения ордера."""
        try:
            # Простая оптимизация на основе рыночных данных
            if not market_data.empty and "close" in market_data.columns:
                current_price = market_data["close"].iloc[-1]
                # Корректируем цену если это лимитный ордер
                if order.get("order_type") == OrderType.LIMIT:
                    if order.get("side") == OrderSide.BUY:
                        # Для покупки устанавливаем цену немного выше текущей
                        order["price"] = current_price * Decimal("1.001")
                    else:
                        # Для продажи устанавливаем цену немного ниже текущей
                        order["price"] = current_price * Decimal("0.999")
            return order
        except Exception as e:
            self.logger.log_error(e, "optimize_order_execution")
            raise ServiceError(
                "EnhancedTradingService",
                f"Failed to optimize order execution: {str(e)}",
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        return {
            "cache_enabled": self.enable_cache,
            "metrics_enabled": self.enable_metrics,
            "performance_cache_size": len(self._performance_cache),
            "service_uptime": "N/A",  # Можно добавить реальное время работы
        }

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._performance_cache.clear()
        self.logger.info("Cleared performance cache")

    def validate_order(self, order: Dict[str, Any]) -> bool:
        """Валидация ордера."""
        try:
            required_fields = ["symbol", "side", "quantity", "order_type"]
            for field in required_fields:
                if field not in order:
                    return False
            if order["quantity"] <= 0:
                return False
            return True
        except Exception:
            return False

    def validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """Валидация рыночных данных."""
        try:
            if market_data.empty:
                return False
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for column in required_columns:
                if column not in market_data.columns:
                    return False
            return True
        except Exception:
            return False
