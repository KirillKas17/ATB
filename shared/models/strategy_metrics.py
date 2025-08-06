"""
Единая модель метрик стратегии.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage


@dataclass
class StrategyMetrics:
    """
    Единая модель метрик стратегии для всего проекта.

    Этот класс заменяет все дублирующиеся определения StrategyMetrics
    в различных модулях проекта.
    """

    # Основные метрики производительности
    win_rate: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    avg_trade: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Дополнительные метрики
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    total_commission: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    best_trade: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    worst_trade: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )

    # Метрики уверенности и качества
    confidence: Decimal = Decimal("0")
    accuracy: Decimal = Decimal("0")
    precision: Decimal = Decimal("0")
    recall: Decimal = Decimal("0")
    f1_score: Decimal = Decimal("0")

    # Временные метрики
    start_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    last_trade_date: Optional[datetime] = None

    # Метаданные
    strategy_id: str = ""
    strategy_name: str = ""
    trading_pair: str = ""
    timeframe: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Пост-инициализация с валидацией типов."""
        # Конвертация типов для совместимости
        if isinstance(self.win_rate, (int, float, str)):
            self.win_rate = Percentage(Decimal(str(self.win_rate)))

        if isinstance(self.profit_factor, (int, float, str)):
            self.profit_factor = Decimal(str(self.profit_factor))

        if isinstance(self.sharpe_ratio, (int, float, str)):
            self.sharpe_ratio = Decimal(str(self.sharpe_ratio))

        if isinstance(self.max_drawdown, (int, float, str)):
            self.max_drawdown = Money(Decimal(str(self.max_drawdown)), Currency.USD)

        if isinstance(self.avg_trade, (int, float, str)):
            self.avg_trade = Money(Decimal(str(self.avg_trade)), Currency.USD)

        if isinstance(self.total_pnl, (int, float, str)):
            self.total_pnl = Money(Decimal(str(self.total_pnl)), Currency.USD)

        if isinstance(self.total_commission, (int, float, str)):
            self.total_commission = Money(
                Decimal(str(self.total_commission)), Currency.USD
            )

        if isinstance(self.best_trade, (int, float, str)):
            self.best_trade = Money(Decimal(str(self.best_trade)), Currency.USD)

        if isinstance(self.worst_trade, (int, float, str)):
            self.worst_trade = Money(Decimal(str(self.worst_trade)), Currency.USD)

        if isinstance(self.confidence, (int, float, str)):
            self.confidence = Decimal(str(self.confidence))

        if isinstance(self.accuracy, (int, float, str)):
            self.accuracy = Decimal(str(self.accuracy))

        if isinstance(self.precision, (int, float, str)):
            self.precision = Decimal(str(self.precision))

        if isinstance(self.recall, (int, float, str)):
            self.recall = Decimal(str(self.recall))

        if isinstance(self.f1_score, (int, float, str)):
            self.f1_score = Decimal(str(self.f1_score))

    def add_trade(self, pnl: Money, commission: Money, is_winning: bool = True) -> None:
        """
        Добавить сделку к метрикам.

        Args:
            pnl: Прибыль/убыток от сделки
            commission: Комиссия по сделке
            is_winning: Является ли сделка прибыльной
        """
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_commission += commission
        self.last_trade_date = datetime.now()

        if is_winning:
            self.winning_trades += 1
            if pnl > self.best_trade:
                self.best_trade = pnl
        else:
            self.losing_trades += 1
            if pnl < self.worst_trade:
                self.worst_trade = pnl

        self._recalculate_metrics()
        self.last_update = datetime.now()

    def _recalculate_metrics(self) -> None:
        """Пересчет метрик на основе текущих данных."""
        if self.total_trades > 0:
            # Пересчет win rate
            win_rate_value = (self.winning_trades / self.total_trades) * 100
            self.win_rate = Percentage(Decimal(str(win_rate_value)))

            # Пересчет средней сделки
            self.avg_trade = self.total_pnl / self.total_trades

            # Пересчет profit factor
            if self.losing_trades > 0:
                total_wins = sum(1 for _ in range(self.winning_trades))  # Упрощенно
                total_losses = sum(1 for _ in range(self.losing_trades))  # Упрощенно
                if total_losses > 0:
                    profit_factor_value = total_wins / total_losses
                    self.profit_factor = Decimal(str(profit_factor_value))

    def get_fitness_score(self) -> Decimal:
        """
        Получить общий фитнес-скор стратегии.

        Returns:
            Decimal: Фитнес-скор от 0 до 1
        """
        # Взвешенная формула фитнес-скора
        win_rate_score = self.win_rate.value / Decimal("100")
        profit_factor_score = min(self.profit_factor / Decimal("2"), Decimal("1"))
        sharpe_score = min(self.sharpe_ratio / Decimal("2"), Decimal("1"))
        drawdown_score = max(Decimal("0"), Decimal("1") - self.max_drawdown.value)

        # Взвешенная сумма
        fitness = (
            win_rate_score * Decimal("0.3")
            + profit_factor_score * Decimal("0.3")
            + sharpe_score * Decimal("0.2")
            + drawdown_score * Decimal("0.2")
        )

        return min(max(fitness, Decimal("0")), Decimal("1"))

    def is_profitable(self) -> bool:
        """Проверка прибыльности стратегии."""
        return self.total_pnl.value > Decimal("0")

    def is_acceptable(
        self,
        min_win_rate: Decimal = Decimal("0.55"),
        min_profit_factor: Decimal = Decimal("1.5"),
        min_sharpe: Decimal = Decimal("1.0"),
        max_drawdown: Decimal = Decimal("0.15"),
    ) -> bool:
        """
        Проверка приемлемости метрик стратегии.

        Args:
            min_win_rate: Минимальный процент выигрышных сделок
            min_profit_factor: Минимальный фактор прибыли
            min_sharpe: Минимальный коэффициент Шарпа
            max_drawdown: Максимальная просадка

        Returns:
            bool: True если метрики приемлемы
        """
        return (
            self.win_rate.value >= min_win_rate * Decimal("100")
            and self.profit_factor >= min_profit_factor
            and self.sharpe_ratio >= min_sharpe
            and self.max_drawdown.value <= max_drawdown
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "win_rate": str(self.win_rate.value),
            "profit_factor": str(self.profit_factor),
            "sharpe_ratio": str(self.sharpe_ratio),
            "max_drawdown": str(self.max_drawdown.value),
            "avg_trade": str(self.avg_trade.value),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl.value),
            "total_commission": str(self.total_commission.value),
            "best_trade": str(self.best_trade.value),
            "worst_trade": str(self.worst_trade.value),
            "confidence": str(self.confidence),
            "accuracy": str(self.accuracy),
            "precision": str(self.precision),
            "recall": str(self.recall),
            "f1_score": str(self.f1_score),
            "start_date": self.start_date.isoformat(),
            "last_update": self.last_update.isoformat(),
            "last_trade_date": (
                self.last_trade_date.isoformat() if self.last_trade_date else None
            ),
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "fitness_score": str(self.get_fitness_score()),
            "is_profitable": self.is_profitable(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyMetrics":
        """Создание из словаря."""
        return cls(
            win_rate=Percentage(Decimal(data["win_rate"])),
            profit_factor=Decimal(data["profit_factor"]),
            sharpe_ratio=Decimal(data["sharpe_ratio"]),
            max_drawdown=Money(Decimal(data["max_drawdown"]), Currency.USD),
            avg_trade=Money(Decimal(data["avg_trade"]), Currency.USD),
            total_trades=data["total_trades"],
            winning_trades=data["winning_trades"],
            losing_trades=data["losing_trades"],
            total_pnl=Money(Decimal(data["total_pnl"]), Currency.USD),
            total_commission=Money(Decimal(data["total_commission"]), Currency.USD),
            best_trade=Money(Decimal(data["best_trade"]), Currency.USD),
            worst_trade=Money(Decimal(data["worst_trade"]), Currency.USD),
            confidence=Decimal(data["confidence"]),
            accuracy=Decimal(data["accuracy"]),
            precision=Decimal(data["precision"]),
            recall=Decimal(data["recall"]),
            f1_score=Decimal(data["f1_score"]),
            start_date=datetime.fromisoformat(data["start_date"]),
            last_update=datetime.fromisoformat(data["last_update"]),
            last_trade_date=(
                datetime.fromisoformat(data["last_trade_date"])
                if data.get("last_trade_date")
                else None
            ),
            strategy_id=data["strategy_id"],
            strategy_name=data["strategy_name"],
            trading_pair=data["trading_pair"],
            timeframe=data["timeframe"],
            metadata=data.get("metadata", {}),
        )
