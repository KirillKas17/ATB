"""
Протоколы для сервисов приложения.
"""

from abc import abstractmethod
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from domain.entities.market import MarketData
from domain.entities.order import Order, OrderStatus
from domain.entities.portfolio import Portfolio
from domain.entities.signal import Signal
from domain.entities.strategy import Strategy
from domain.entities.trade import Trade
from domain.entities.trading_pair import TradingPair
from domain.type_definitions import (
    OrderId,
    PortfolioId,
    PriceValue,
    StrategyId,
    Symbol,
    TimestampValue,
    VolumeValue,
    RiskMetrics,
    ConfidenceLevel,
    MetadataDict,
)
from domain.value_objects import Money, Price, Volume

from .use_case_protocols import CreateOrderRequest

# Типы для аналитики
class TechnicalIndicators:
    """Технические индикаторы."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class SocialSentiment:
    """Социальные настроения."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class MLPrediction:
    """ML предсказание."""

    def __init__(
        self,
        model_id: str,
        symbol: Symbol,
        prediction_type: str,
        predicted_value: Decimal,
        confidence: ConfidenceLevel,
        timestamp: TimestampValue,
        features: Dict[str, Any],
        metadata: MetadataDict,
    ):
        self.model_id = model_id
        self.symbol = symbol
        self.prediction_type = prediction_type
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.timestamp = timestamp
        self.features = features
        self.metadata = metadata


class PatternDetection:
    """Обнаружение паттерна."""

    def __init__(
        self,
        pattern_id: str,
        symbol: Symbol,
        pattern_type: str,
        confidence: ConfidenceLevel,
        start_time: TimestampValue,
        end_time: TimestampValue,
        price_levels: List[Decimal],
        volume_profile: Dict[str, Any],
        metadata: MetadataDict,
    ):
        self.pattern_id = pattern_id
        self.symbol = symbol
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.price_levels = price_levels
        self.volume_profile = volume_profile
        self.metadata = metadata


class EntanglementResult:
    """Результат анализа запутанности."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class LiquidityGravityResult:
    """Результат расчета гравитации ликвидности."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class MarketAnalysis:
    """Анализ рынка."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class StrategyPerformance:
    """Производительность стратегии."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class PortfolioMetrics:
    """Метрики портфеля."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        total_pnl: Money,
        max_drawdown: Decimal,
        volatility: Decimal,
        sharpe_ratio: Decimal,
        total_trades: int,
        win_rate: Decimal,
        profit_factor: Decimal,
        timestamp: TimestampValue,
    ):
        self.portfolio_id = portfolio_id
        self.total_pnl = total_pnl
        self.max_drawdown = max_drawdown
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.total_trades = total_trades
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.timestamp = timestamp


class PerformanceMetrics:
    """Метрики производительности."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        period: str,
        total_return: Decimal,
        daily_return: Decimal,
        volatility: Decimal,
        sharpe_ratio: Decimal,
        max_drawdown: Decimal,
        win_rate: Decimal,
        profit_factor: Decimal,
        total_trades: int,
        timestamp: TimestampValue,
    ):
        self.portfolio_id = portfolio_id
        self.period = period
        self.total_return = total_return
        self.daily_return = daily_return
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.total_trades = total_trades
        self.timestamp = timestamp


class EvolutionResult:
    """Результат эволюции."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


class SymbolSelectionResult:
    """Результат выбора символов."""

    def __init__(self, *args, **kwargs) -> Any:
        self.data = data


# Протоколы сервисов
@runtime_checkable
class MarketService(Protocol):
    """Протокол для сервиса рыночных данных."""

    @abstractmethod
    async def get_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Получение рыночных данных."""
        ...

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: Symbol,
        start_time: TimestampValue,
        end_time: TimestampValue,
        interval: str,
    ) -> List[Dict[str, Any]]:
        """Получение исторических данных."""
        ...

    @abstractmethod
    async def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Получение текущей цены."""
        ...

    @abstractmethod
    async def get_order_book(
        self, symbol: Symbol, depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Получение стакана заявок."""
        ...

    @abstractmethod
    async def get_market_metrics(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """Получение рыночных метрик."""
        ...

    @abstractmethod
    async def subscribe_to_updates(
        self, symbol: Symbol, callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """Подписка на обновления."""
        ...

    @abstractmethod
    async def analyze_market(self, symbol: Symbol) -> MarketAnalysis:
        """Анализ рынка."""
        ...

    @abstractmethod
    async def get_technical_indicators(self, symbol: Symbol) -> TechnicalIndicators:
        """Получение технических индикаторов."""
        ...


@runtime_checkable
class MLService(Protocol):
    """Протокол для ML сервиса."""

    @abstractmethod
    async def predict_price(
        self, symbol: Symbol, features: Dict[str, Any]
    ) -> Optional[MLPrediction]:
        """Предсказание цены."""
        ...

    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ настроений."""
        ...

    @abstractmethod
    async def detect_patterns(self, market_data: MarketData) -> List[PatternDetection]:
        """Обнаружение паттернов."""
        ...

    @abstractmethod
    async def calculate_risk_metrics(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Расчет метрик риска."""
        ...

    @abstractmethod
    async def train_model(
        self, model_id: str, training_data: List[Dict[str, Any]]
    ) -> bool:
        """Обучение модели."""
        ...

    @abstractmethod
    async def evaluate_model(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Оценка модели."""
        ...


@runtime_checkable
class TradingService(Protocol):
    """Протокол для торгового сервиса."""

    @abstractmethod
    async def place_order(self, order: Order) -> bool:
        """Размещение ордера."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: OrderId) -> bool:
        """Отмена ордера."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: OrderId) -> Optional[OrderStatus]:
        """Получение статуса ордера."""
        ...

    @abstractmethod
    async def get_account_balance(self) -> Dict[str, Money]:
        """Получение баланса аккаунта."""
        ...

    @abstractmethod
    async def get_trade_history(self, symbol: Symbol, limit: int = 100) -> List[Trade]:
        """Получение истории сделок."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """Получение открытых ордеров."""
        ...


@runtime_checkable
class StrategyService(Protocol):
    """Протокол для сервиса стратегий."""

    @abstractmethod
    async def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """Генерация сигналов."""
        ...

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Валидация сигнала."""
        ...

    @abstractmethod
    async def execute_signal(
        self, signal: Signal, portfolio_id: PortfolioId
    ) -> Optional[Order]:
        """Исполнение сигнала."""
        ...

    @abstractmethod
    async def backtest_strategy(
        self, strategy_id: StrategyId, historical_data: List[Dict[str, Any]]
    ) -> StrategyPerformance:
        """Бэктестинг стратегии."""
        ...

    @abstractmethod
    async def optimize_strategy(
        self, strategy_id: StrategyId, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Оптимизация стратегии."""
        ...

    @abstractmethod
    async def get_strategy_performance(
        self, strategy_id: StrategyId
    ) -> StrategyPerformance:
        """Получение производительности стратегии."""
        ...


@runtime_checkable
class PortfolioService(Protocol):
    """Протокол для сервиса портфеля."""

    @abstractmethod
    async def calculate_weights(
        self, portfolio_id: PortfolioId
    ) -> Dict[Symbol, Decimal]:
        """Расчет весов портфеля."""
        ...

    @abstractmethod
    async def rebalance_portfolio(
        self, portfolio_id: PortfolioId, target_weights: Dict[Symbol, Decimal]
    ) -> bool:
        """Ребалансировка портфеля."""
        ...

    @abstractmethod
    async def calculate_pnl(self, portfolio_id: PortfolioId) -> Money:
        """Расчет P&L."""
        ...

    @abstractmethod
    async def get_portfolio_metrics(
        self, portfolio_id: PortfolioId
    ) -> PortfolioMetrics:
        """Получение метрик портфеля."""
        ...

    @abstractmethod
    async def get_portfolio_balance(
        self, portfolio_id: PortfolioId
    ) -> Dict[str, Money]:
        """Получение баланса портфеля."""
        ...

    @abstractmethod
    async def update_portfolio_balance(
        self, portfolio_id: PortfolioId, changes: Dict[str, Money]
    ) -> bool:
        """Обновление баланса портфеля."""
        ...

    @abstractmethod
    async def get_portfolio_performance(
        self, portfolio_id: PortfolioId, period: str = "1d"
    ) -> PerformanceMetrics:
        """Получение отчета о производительности."""
        ...

    @abstractmethod
    async def validate_portfolio_constraints(
        self, portfolio_id: PortfolioId, order_request: CreateOrderRequest
    ) -> tuple[bool, List[str]]:
        """Валидация ограничений портфеля."""
        ...


@runtime_checkable
class RiskService(Protocol):
    """Протокол для сервиса рисков."""

    @abstractmethod
    async def assess_portfolio_risk(self, portfolio_id: PortfolioId) -> RiskMetrics:
        """Оценка риска портфеля."""
        ...

    @abstractmethod
    async def calculate_var(
        self, portfolio_id: PortfolioId, confidence_level: Decimal = Decimal("0.95")
    ) -> Money:
        """Расчет VaR."""
        ...

    @abstractmethod
    async def calculate_max_drawdown(self, portfolio_id: PortfolioId) -> Decimal:
        """Расчет максимальной просадки."""
        ...

    @abstractmethod
    async def validate_risk_limits(
        self, portfolio_id: PortfolioId, order_request: CreateOrderRequest
    ) -> tuple[bool, List[str]]:
        """Валидация лимитов риска."""
        ...

    @abstractmethod
    async def get_risk_alerts(self, portfolio_id: PortfolioId) -> List[Dict[str, Any]]:
        """Получение алертов риска."""
        ...


@runtime_checkable
class CacheService(Protocol):
    """Протокол для сервиса кэширования."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        ...

    @abstractmethod
    async def clear(self, pattern: str = "*") -> bool:
        """Очистка кэша по паттерну."""
        ...

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        ...

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Установка времени жизни ключа."""
        ...


@runtime_checkable
class NotificationService(Protocol):
    """Протокол для сервиса уведомлений."""

    @abstractmethod
    async def send_notification(self, message: str, level: str = "info") -> bool:
        """Отправка уведомления."""
        ...

    @abstractmethod
    async def send_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Отправка алерта."""
        ...

    @abstractmethod
    async def subscribe_to_alerts(self, user_id: str, alert_types: List[str]) -> bool:
        """Подписка на алерты."""
        ...

    @abstractmethod
    async def send_trade_notification(self, trade: Trade) -> bool:
        """Отправка уведомления о сделке."""
        ...

    @abstractmethod
    async def send_risk_alert(
        self, portfolio_id: PortfolioId, risk_level: str, details: Dict[str, Any]
    ) -> bool:
        """Отправка алерта о риске."""
        ...

    @abstractmethod
    async def send_performance_report(
        self, portfolio_id: PortfolioId, metrics: PerformanceMetrics
    ) -> bool:
        """Отправка отчета о производительности."""
        ...


@runtime_checkable
class AnalyticsService(Protocol):
    """Протокол для аналитического сервиса."""

    @abstractmethod
    async def calculate_technical_indicators(
        self, symbol: Symbol, timeframe: str
    ) -> TechnicalIndicators:
        """Расчет технических индикаторов."""
        ...

    @abstractmethod
    async def analyze_market_sentiment(self, symbol: Symbol) -> SocialSentiment:
        """Анализ рыночных настроений."""
        ...

    @abstractmethod
    async def predict_price_movement(
        self, symbol: Symbol, horizon: str = "1h"
    ) -> MLPrediction:
        """Предсказание движения цены."""
        ...

    @abstractmethod
    async def generate_trading_signals(self, symbol: Symbol) -> List[Signal]:
        """Генерация торговых сигналов."""
        ...

    @abstractmethod
    async def detect_market_patterns(self, symbol: Symbol) -> List[PatternDetection]:
        """Обнаружение рыночных паттернов."""
        ...

    @abstractmethod
    async def analyze_entanglement(
        self, symbol: Symbol, exchange_pair: str
    ) -> EntanglementResult:
        """Анализ запутанности."""
        ...

    @abstractmethod
    async def calculate_liquidity_gravity(
        self, symbol: Symbol
    ) -> LiquidityGravityResult:
        """Расчет гравитации ликвидности."""
        ...


@runtime_checkable
class MetricsService(Protocol):
    """Протокол для сервиса метрик."""

    @abstractmethod
    async def record_trade(self, trade: Trade) -> None:
        """Запись метрики сделки."""
        ...

    @abstractmethod
    async def record_order(self, order: Order) -> None:
        """Запись метрики ордера."""
        ...

    @abstractmethod
    async def record_portfolio_performance(
        self, portfolio_id: PortfolioId, metrics: PerformanceMetrics
    ) -> None:
        """Запись метрик производительности портфеля."""
        ...

    @abstractmethod
    async def record_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Запись системных метрик."""
        ...

    @abstractmethod
    async def get_trading_metrics(self, portfolio_id: PortfolioId) -> Dict[str, Any]:
        """Получение торговых метрик."""
        ...

    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Получение состояния системы."""
        ...


@runtime_checkable
class EvolutionService(Protocol):
    """Протокол для сервиса эволюции стратегий."""

    @abstractmethod
    async def evolve_strategy(
        self, strategy_id: StrategyId, generations: int = 10
    ) -> EvolutionResult:
        """Эволюция стратегии."""
        ...

    @abstractmethod
    async def evaluate_fitness(self, strategy_id: StrategyId) -> Decimal:
        """Оценка приспособленности стратегии."""
        ...

    @abstractmethod
    async def optimize_parameters(self, strategy_id: StrategyId) -> Dict[str, Any]:
        """Оптимизация параметров стратегии."""
        ...

    @abstractmethod
    async def get_evolution_history(
        self, strategy_id: StrategyId
    ) -> List[EvolutionResult]:
        """Получение истории эволюции."""
        ...


@runtime_checkable
class SymbolSelectionService(Protocol):
    """Протокол для сервиса выбора символов."""

    @abstractmethod
    async def select_symbols(self, criteria: Dict[str, Any]) -> SymbolSelectionResult:
        """Выбор символов по критериям."""
        ...

    @abstractmethod
    async def analyze_opportunities(
        self, symbols: List[Symbol]
    ) -> Dict[Symbol, Decimal]:
        """Анализ возможностей."""
        ...

    @abstractmethod
    async def get_market_phases(self, symbols: List[Symbol]) -> Dict[Symbol, str]:
        """Получение фаз рынка."""
        ...

    @abstractmethod
    async def filter_symbols(
        self, symbols: List[Symbol], filters: Dict[str, Any]
    ) -> List[Symbol]:
        """Фильтрация символов."""
        ...
