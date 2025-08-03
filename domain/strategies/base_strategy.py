"""
Доменная абстракция базовой стратегии.
Этот модуль определяет интерфейсы и абстракции для стратегий
в доменном слое, обеспечивая независимость от инфраструктуры.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.entities.market import MarketData
from domain.entities.signal import Signal
from domain.entities.strategy import StrategyType
from domain.types import StrategyId, TradingPair


@dataclass
class StrategyMetrics:
    """Метрики стратегии в доменном слое"""

    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    sortino_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    recovery_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    expectancy: Decimal = field(default_factory=lambda: Decimal("0"))
    risk_reward_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    kelly_criterion: Decimal = field(default_factory=lambda: Decimal("0"))
    volatility: Decimal = field(default_factory=lambda: Decimal("0"))
    mar_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    ulcer_index: Decimal = field(default_factory=lambda: Decimal("0"))
    omega_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    gini_coefficient: Decimal = field(default_factory=lambda: Decimal("0"))
    tail_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    skewness: Decimal = field(default_factory=lambda: Decimal("0"))
    kurtosis: Decimal = field(default_factory=lambda: Decimal("0"))
    var_95: Decimal = field(default_factory=lambda: Decimal("0"))
    cvar_95: Decimal = field(default_factory=lambda: Decimal("0"))
    drawdown_duration: int = 0
    max_equity: Decimal = field(default_factory=lambda: Decimal("0"))
    min_equity: Decimal = field(default_factory=lambda: Decimal("0"))
    median_trade: Decimal = field(default_factory=lambda: Decimal("0"))
    median_duration: int = 0
    profit_streak: int = 0
    loss_streak: int = 0
    stability: Decimal = field(default_factory=lambda: Decimal("0"))
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureSignal:
    """Адаптер для сигналов инфраструктурных стратегий"""

    direction: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Доменная абстракция базовой стратегии"""

    def __init__(
        self, strategy_id: StrategyId, config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            strategy_id: Уникальный идентификатор стратегии
            config: Конфигурация стратегии
        """
        self.strategy_id = strategy_id
        self.config = config or {}
        self.metrics = StrategyMetrics()
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Инициализация параметров стратегии"""
        self.required_columns = ["open", "high", "low", "close", "volume"]
        self.timeframes = self.config.get("timeframes", ["1h"])
        self.symbols = self.config.get("symbols", [])
        self.risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))
        self.max_position_size = Decimal(str(self.config.get("max_position_size", 0.1)))
        self.confidence_threshold = Decimal(
            str(self.config.get("confidence_threshold", 0.7))
        )
        self.use_stop_loss = self.config.get("use_stop_loss", True)
        self.use_take_profit = self.config.get("use_take_profit", True)
        self.trailing_stop = self.config.get("trailing_stop", False)
        self.trailing_stop_activation = Decimal(
            str(self.config.get("trailing_stop_activation", 0.02))
        )
        self.trailing_stop_distance = Decimal(
            str(self.config.get("trailing_stop_distance", 0.01))
        )

    @abstractmethod
    def analyze(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Анализ рыночных данных.
        Args:
            market_data: Рыночные данные
        Returns:
            Dict с результатами анализа
        """
        pass

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            market_data: Рыночные данные
        Returns:
            Optional[Signal] с сигналом или None
        """
        pass

    def validate_data(self, market_data: MarketData) -> tuple[bool, Optional[str]]:
        try:
            if market_data is None:
                return False, "Empty market data"
            elif not hasattr(market_data, "data") or market_data.data is None:
                return False, "No market data available"
            else:
                return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def calculate_position_size(
        self, signal: Signal, account_balance: Decimal
    ) -> Decimal:
        """
        Расчёт размера позиции.
        Args:
            signal: Торговый сигнал
            account_balance: Баланс аккаунта
        Returns:
            Размер позиции
        """
        if signal.quantity:
            return Decimal(str(signal.quantity))
        # Базовый расчёт на основе риска
        risk_amount = account_balance * self.risk_per_trade
        return min(risk_amount, account_balance * self.max_position_size)

    def update_metrics(self, signal: Signal, result: Dict[str, Any]) -> None:
        """
        Обновление метрик стратегии.
        Args:
            signal: Торговый сигнал
            result: Результат торговли
        """
        self.metrics.total_signals += 1
        profit = Decimal(str(result.get("profit", 0)))
        if profit > 0:
            self.metrics.successful_signals += 1
            self.metrics.avg_profit = (
                self.metrics.avg_profit * (self.metrics.successful_signals - 1) + profit
            ) / self.metrics.successful_signals
        else:
            self.metrics.failed_signals += 1
            self.metrics.avg_loss = (
                self.metrics.avg_loss * (self.metrics.failed_signals - 1) + abs(profit)
            ) / self.metrics.failed_signals
        # Обновляем win rate
        if self.metrics.total_signals > 0:
            self.metrics.win_rate = Decimal(
                str(self.metrics.successful_signals)
            ) / Decimal(str(self.metrics.total_signals))
        # Обновляем profit factor
        if self.metrics.avg_loss > 0:
            self.metrics.profit_factor = self.metrics.avg_profit / self.metrics.avg_loss

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик стратегии.
        Returns:
            Dict с метриками
        """
        return {
            "total_signals": self.metrics.total_signals,
            "successful_signals": self.metrics.successful_signals,
            "failed_signals": self.metrics.failed_signals,
            "avg_profit": float(self.metrics.avg_profit),
            "avg_loss": float(self.metrics.avg_loss),
            "win_rate": float(self.metrics.win_rate),
            "profit_factor": float(self.metrics.profit_factor),
            "sharpe_ratio": float(self.metrics.sharpe_ratio),
            "sortino_ratio": float(self.metrics.sortino_ratio),
            "max_drawdown": float(self.metrics.max_drawdown),
            "recovery_factor": float(self.metrics.recovery_factor),
            "expectancy": float(self.metrics.expectancy),
            "risk_reward_ratio": float(self.metrics.risk_reward_ratio),
            "kelly_criterion": float(self.metrics.kelly_criterion),
            "volatility": float(self.metrics.volatility),
            "mar_ratio": float(self.metrics.mar_ratio),
            "ulcer_index": float(self.metrics.ulcer_index),
            "omega_ratio": float(self.metrics.omega_ratio),
            "gini_coefficient": float(self.metrics.gini_coefficient),
            "tail_ratio": float(self.metrics.tail_ratio),
            "skewness": float(self.metrics.skewness),
            "kurtosis": float(self.metrics.kurtosis),
            "var_95": float(self.metrics.var_95),
            "cvar_95": float(self.metrics.cvar_95),
            "drawdown_duration": self.metrics.drawdown_duration,
            "max_equity": float(self.metrics.max_equity),
            "min_equity": float(self.metrics.min_equity),
            "median_trade": float(self.metrics.median_trade),
            "median_duration": self.metrics.median_duration,
            "profit_streak": self.metrics.profit_streak,
            "loss_streak": self.metrics.loss_streak,
            "stability": float(self.metrics.stability),
            "additional": self.metrics.additional,
        }

    def get_strategy_id(self) -> StrategyId:
        """Получить ID стратегии."""
        return self.strategy_id

    def get_strategy_type(self) -> StrategyType:
        """Получить тип стратегии."""
        return StrategyType.TREND_FOLLOWING  # По умолчанию

    def get_trading_pairs(self) -> List[TradingPair]:
        """Получить торговые пары стратегии."""
        return [TradingPair(symbol) for symbol in self.symbols]

    def get_config(self) -> Dict[str, Any]:
        """Получить конфигурацию стратегии."""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Обновить конфигурацию стратегии."""
        self.config.update(new_config)
        self._initialize_parameters()
