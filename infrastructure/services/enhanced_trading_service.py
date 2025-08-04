"""
Промышленная реализация enhanced trading сервиса.
Фасад для декомпозированных модулей enhanced trading.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd

from domain.entities.order import OrderSide, OrderStatus, OrderType
from domain.exceptions import TradingError

# from domain.services.enhanced_trading import EnhancedTradingService
from domain.value_objects import Currency, Money

# from domain.type_definitions.trading_types import (
#     StrategyType, ExecutionAlgorithm
# )
# Импорты из декомпозированных модулей
from .enhanced_trading import (  # Order creation; Strategy execution; Sentiment adjustment; Utils
    EnhancedTradingCache,
    adjust_trading_parameters,
    analyze_market_sentiment,
    analyze_news_sentiment,
    analyze_social_sentiment,
    apply_risk_management,
    calculate_optimal_order_size,
    calculate_order_timing,
    calculate_performance_metrics,
    calculate_position_size,
    calculate_risk_metrics,
    calculate_sentiment_score,
    calculate_slippage_estimate,
    clean_cache,
    combine_sentiment_sources,
    convert_to_decimal,
    convert_to_money,
    create_bracket_order,
    create_empty_execution_plan,
    create_empty_order,
    create_empty_sentiment_analysis,
    create_execution_plan,
    create_iceberg_order,
    create_limit_order,
    create_market_order,
    create_smart_order,
    create_stop_limit_order,
    create_stop_order,
    create_twap_order,
    create_vwap_order,
    detect_sentiment_shifts,
    execute_algorithm,
    generate_sentiment_alerts,
    monitor_execution,
    normalize_data,
    optimize_order_execution,
    optimize_strategy_parameters,
    validate_market_data,
    validate_order_data,
    validate_order_parameters,
    validate_sentiment_data,
    validate_strategy_data,
    validate_strategy_parameters,
    validate_trading_parameters,
)
from .enhanced_trading.strategy_execution import ExecutionAlgorithm, StrategyType


@dataclass
class EnhancedTradingConfig:
    """Конфигурация enhanced trading сервиса."""

    # Параметры ордеров
    default_order_type: OrderType = OrderType.LIMIT
    max_order_size: Decimal = Decimal("1000")
    min_order_size: Decimal = Decimal("0.01")
    # Параметры исполнения
    default_execution_algorithm: ExecutionAlgorithm = "twap"
    execution_timeout_minutes: int = 30
    max_slippage: Decimal = Decimal("0.01")
    # Параметры риск-менеджмента
    max_position_size: Decimal = Decimal("0.1")  # 10% капитала
    max_drawdown: Decimal = Decimal("0.05")  # 5%
    risk_per_trade: Decimal = Decimal("0.02")  # 2%
    # Параметры настроений
    enable_sentiment_analysis: bool = True
    sentiment_weight: Decimal = Decimal("0.3")
    news_weight: Decimal = Decimal("0.2")
    social_weight: Decimal = Decimal("0.1")


@runtime_checkable
class EnhancedTradingService(Protocol):
    """Протокол для enhanced trading сервиса."""

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: Optional[OrderType] = None,
        price: Optional[Decimal] = None,
        **kwargs,
    ) -> Dict[str, Any]: ...
    def create_advanced_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: str,
        market_data: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]: ...
    def execute_strategy(
        self,
        strategy_type: Any,
        parameters: Dict[str, Any],
        market_data: pd.DataFrame,
        capital: Decimal,
    ) -> Dict[str, Any]: ...
    def analyze_sentiment(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[List[Dict[str, Any]]] = None,
        social_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]: ...
class EnhancedTradingError(TradingError):
    """Ошибка enhanced trading сервиса."""

    pass


class EnhancedTradingServiceImpl(EnhancedTradingService):
    """Промышленная реализация enhanced trading сервиса."""

    def __init__(self, config: Optional[EnhancedTradingConfig] = None):
        """Инициализация сервиса."""
        self.config = config or EnhancedTradingConfig()
        self.logger = logging.getLogger(__name__)
        self.cache = EnhancedTradingCache(ttl_hours=1)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: Optional[OrderType] = None,
        price: Optional[Decimal] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Создание торгового ордера."""
        try:
            order_type = order_type or self.config.default_order_type
            # Валидируем параметры
            if not validate_order_parameters(symbol, side, quantity, price, order_type):
                raise ValueError("Invalid order parameters")
            # Создаём базовый ордер
            if order_type == OrderType.MARKET:
                order = create_market_order(symbol, side, quantity)
            elif order_type == OrderType.LIMIT:
                if not price:
                    raise ValueError("Price required for limit order")
                order = create_limit_order(symbol, side, quantity, price)
            elif order_type == OrderType.STOP:
                stop_price = kwargs.get("stop_price")
                if not stop_price:
                    raise ValueError("Stop price required for stop order")
                order = create_stop_order(symbol, side, quantity, stop_price)
            elif order_type == OrderType.STOP_LIMIT:
                stop_price = kwargs.get("stop_price")
                limit_price = kwargs.get("limit_price", price)
                if not stop_price or not limit_price:
                    raise ValueError(
                        "Stop price and limit price required for stop-limit order"
                    )
                order = create_stop_limit_order(
                    symbol, side, quantity, stop_price, limit_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            # Оптимизируем исполнение
            market_data = kwargs.get("market_data")
            if market_data is not None:
                order = optimize_order_execution(order, market_data)
            return order
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise EnhancedTradingError(f"Failed to create order: {str(e)}")

    def create_advanced_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: str,
        market_data: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """Создание продвинутого ордера."""
        try:
            if not validate_market_data(market_data):
                raise ValueError("Invalid market data")
            if order_type == "twap":
                duration = kwargs.get("duration_minutes", 30)
                order = create_twap_order(symbol, side, quantity, duration)
            elif order_type == "vwap":
                target_vwap = kwargs.get("target_vwap")
                max_deviation = kwargs.get("max_deviation", Decimal("0.01"))
                order = create_vwap_order(
                    symbol, side, quantity, target_vwap, max_deviation
                )
            elif order_type == "iceberg":
                visible_quantity = kwargs.get(
                    "visible_quantity", quantity * Decimal("0.1")
                )
                price = kwargs.get("price")
                if not price:
                    raise ValueError("Price required for iceberg order")
                order = create_iceberg_order(
                    symbol, side, quantity, visible_quantity, price
                )
            elif order_type == "bracket":
                entry_price = kwargs.get("entry_price")
                take_profit = kwargs.get("take_profit")
                stop_loss = kwargs.get("stop_loss")
                if not all([entry_price, take_profit, stop_loss]):
                    raise ValueError(
                        "Entry price, take profit and stop loss required for bracket order"
                    )
                # Преобразуем в Decimal для безопасности типов
                entry_price_decimal = Decimal(str(entry_price)) if entry_price is not None else Decimal("0")
                take_profit_decimal = Decimal(str(take_profit)) if take_profit is not None else Decimal("0")
                stop_loss_decimal = Decimal(str(stop_loss)) if stop_loss is not None else Decimal("0")
                order = create_bracket_order(
                    symbol, side, quantity, entry_price_decimal, take_profit_decimal, stop_loss_decimal
                )
            else:
                raise ValueError(f"Unsupported advanced order type: {order_type}")
            # Оптимизируем исполнение
            order = optimize_order_execution(order, market_data, kwargs)
            return order
        except Exception as e:
            self.logger.error(f"Error creating advanced order: {e}")
            raise EnhancedTradingError(f"Failed to create advanced order: {str(e)}")

    def execute_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        market_data: pd.DataFrame,
        capital: Decimal,
    ) -> Dict[str, Any]:
        """Исполнение торговой стратегии."""
        try:
            if not validate_strategy_parameters(strategy_type, parameters):
                raise ValueError("Invalid strategy parameters")
            if not validate_market_data(market_data):
                raise ValueError("Invalid market data")
            # Создаём план исполнения
            execution_plan = create_execution_plan(
                strategy_type, parameters, market_data, capital
            )
            # Рассчитываем размер позиции
            entry_price = (
                market_data["close"].iloc[-1] if not market_data.empty else Decimal("0")
            )
            stop_loss_price = parameters.get("stop_loss")
            position_size = calculate_position_size(
                capital, self.config.risk_per_trade, entry_price, stop_loss_price
            )
            # Применяем анализ настроений
            if self.config.enable_sentiment_analysis:
                sentiment_analysis = analyze_market_sentiment(market_data)
                parameters = adjust_trading_parameters(parameters, sentiment_analysis)
            # Рассчитываем метрики риска
            risk_metrics = calculate_risk_metrics([], market_data, capital)
            # Применяем риск-менеджмент
            risk_limits = {
                "max_exposure": self.config.max_position_size,
                "max_drawdown": self.config.max_drawdown,
                "max_order_size": self.config.max_order_size,
            }
            managed_positions = apply_risk_management([], risk_limits, market_data)
            return {
                "execution_plan": execution_plan,
                "position_size": position_size,
                "risk_metrics": risk_metrics,
                "managed_positions": managed_positions,
                "sentiment_analysis": (
                    sentiment_analysis
                    if self.config.enable_sentiment_analysis
                    else None
                ),
                "execution_timestamp": datetime.now(),
            }
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            raise EnhancedTradingError(f"Failed to execute strategy: {str(e)}")

    def analyze_sentiment(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[List[Dict[str, Any]]] = None,
        social_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Анализ настроений рынка."""
        try:
            if not validate_market_data(market_data):
                return create_empty_sentiment_analysis()
            # Анализируем рыночные настроения
            market_sentiment = analyze_market_sentiment(market_data)
            # Анализируем новостные настроения
            news_sentiment = None
            if news_data and self.config.enable_sentiment_analysis:
                news_sentiment = analyze_news_sentiment(news_data, market_data)
            # Анализируем социальные настроения
            social_sentiment = None
            if social_data and self.config.enable_sentiment_analysis:
                social_sentiment = analyze_social_sentiment(social_data)
            # Рассчитываем общий показатель настроений
            overall_sentiment_score = calculate_sentiment_score(
                market_sentiment, news_sentiment, social_sentiment
            )
            # Обнаруживаем сдвиги настроений
            sentiment_shifts = detect_sentiment_shifts(market_sentiment, pd.DataFrame())
            # Генерируем алерты
            sentiment_alerts = generate_sentiment_alerts(
                market_sentiment, sentiment_shifts
            )
            return {
                "market_sentiment": market_sentiment,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "overall_sentiment_score": overall_sentiment_score,
                "sentiment_shifts": sentiment_shifts,
                "alerts": sentiment_alerts,
                "analysis_timestamp": datetime.now(),
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return create_empty_sentiment_analysis()

    def optimize_order_execution(
        self,
        order: Dict[str, Any],
        market_data: pd.DataFrame,
        execution_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Оптимизация исполнения ордера."""
        try:
            if not validate_order_data(order):
                raise ValueError("Invalid order data")
            if not validate_market_data(market_data):
                raise ValueError("Invalid market data")
            return optimize_order_execution(order, market_data, execution_parameters)
        except Exception as e:
            self.logger.error(f"Error optimizing order execution: {e}")
            raise EnhancedTradingError(f"Failed to optimize order execution: {str(e)}")

    def monitor_execution(
        self,
        execution_plan: Dict[str, Any],
        market_data: pd.DataFrame,
        current_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Мониторинг исполнения."""
        try:
            if not validate_market_data(market_data):
                return {"error": "Invalid market data"}
            return monitor_execution(execution_plan, market_data, current_positions)
        except Exception as e:
            self.logger.error(f"Error monitoring execution: {e}")
            return {"error": str(e)}

    def calculate_performance_metrics(
        self, orders: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Расчёт метрик производительности."""
        try:
            return calculate_performance_metrics(orders, market_data)  # Убираю лишний аргумент
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "fill_rate": Decimal("0"),
                "average_execution_time": timedelta(0),
                "total_commission": Money(Decimal("0"), Currency.USD),
                "total_pnl": Money(Decimal("0"), Currency.USD),
            }

    def create_smart_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
        market_data: pd.DataFrame,
        execution_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Создание умного ордера."""
        try:
            if not validate_market_data(market_data):
                raise ValueError("Invalid market data")
            return create_smart_order(
                symbol, side, quantity, order_type, market_data, execution_parameters
            )
        except Exception as e:
            self.logger.error(f"Error creating smart order: {e}")
            raise EnhancedTradingError(f"Failed to create smart order: {str(e)}")

    def execute_algorithm(
        self,
        algorithm: ExecutionAlgorithm,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Исполнение торгового алгоритма."""
        try:
            if not validate_market_data(market_data):
                raise ValueError("Invalid market data")
            return execute_algorithm(algorithm, market_data, parameters)
        except Exception as e:
            self.logger.error(f"Error executing algorithm: {e}")
            raise EnhancedTradingError(f"Failed to execute algorithm: {str(e)}")

    def adjust_trading_parameters(
        self, base_parameters: Dict[str, Any], sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Корректировка торговых параметров на основе настроений."""
        try:
            return adjust_trading_parameters(base_parameters, sentiment_analysis)
        except Exception as e:
            self.logger.error(f"Error adjusting trading parameters: {e}")
            return base_parameters

    def validate_order(self, order: Dict[str, Any]) -> bool:
        """Валидация ордера."""
        return validate_order_data(order)

    def validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """Валидация рыночных данных."""
        return validate_market_data(market_data)

    def validate_strategy(self, strategy_data: Dict[str, Any]) -> bool:
        """Валидация стратегии."""
        return validate_strategy_data(strategy_data)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        return {
            "cache_size": len(self.cache.cache),
            "cache_hits": 0,  # Упрощённо
            "cache_misses": 0,  # Упрощённо
            "last_cleanup": datetime.now(),
        }

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self.cache.clear()

    def cleanup_expired_cache(self) -> None:
        """Очистка истёкшего кэша."""
        clean_cache(self.cache)
