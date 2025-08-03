"""
Доменная сущность производительности стратегии.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Union
from uuid import UUID, uuid4

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage


@dataclass
class StrategyPerformance:
    """Производительность стратегии"""

    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    total_commission: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    max_drawdown: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    sharpe_ratio: Decimal = Decimal("0")
    win_rate: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    profit_factor: Decimal = Decimal("0")
    average_trade: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    best_trade: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    worst_trade: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    start_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        # Валидация и нормализация полей при необходимости
        pass

    def add_trade(self, pnl: Money, commission: Money) -> None:
        """Добавить сделку"""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_commission += commission

        if pnl.value > 0:
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
        """Пересчитать метрики"""
        if self.total_trades > 0:
            win_rate_value = (self.winning_trades / self.total_trades) * 100
            self.win_rate = Percentage(Decimal(str(win_rate_value)))
            self.average_trade = self.total_pnl / self.total_trades

        # Profit factor
        if self.losing_trades > 0:
            total_wins = sum(1 for _ in range(self.winning_trades))  # Упрощенно
            total_losses = sum(1 for _ in range(self.losing_trades))  # Упрощенно
            if total_losses > 0:
                profit_factor_value = total_wins / total_losses
                self.profit_factor = Decimal(str(profit_factor_value))

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            str,
            int,
            float,
            Decimal,
            bool,
            List[str],
            Dict[str, Union[str, int, float, Decimal, bool]],
        ],
    ]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl.value),
            "total_commission": str(self.total_commission.value),
            "max_drawdown": str(self.max_drawdown.value),
            "sharpe_ratio": str(self.sharpe_ratio),
            "win_rate": str(self.win_rate.value),
            "profit_factor": str(self.profit_factor),
            "average_trade": str(self.average_trade.value),
            "best_trade": str(self.best_trade.value),
            "worst_trade": str(self.worst_trade.value),
            "start_date": self.start_date.isoformat(),
            "last_update": self.last_update.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[
            str,
            Union[
                str,
                int,
                float,
                Decimal,
                bool,
                List[str],
                Dict[str, Union[str, int, float, Decimal, bool]],
            ],
        ],
    ) -> "StrategyPerformance":
        """Создание из словаря."""
        # Безопасное извлечение и преобразование данных
        id_value = data.get("id", "")
        strategy_id_value = data.get("strategy_id", "")
        total_trades = data.get("total_trades", 0)
        winning_trades = data.get("winning_trades", 0)
        losing_trades = data.get("losing_trades", 0)
        total_pnl_value = data.get("total_pnl", "0")
        total_commission_value = data.get("total_commission", "0")
        max_drawdown_value = data.get("max_drawdown", "0")
        sharpe_ratio_value = data.get("sharpe_ratio", "0")
        win_rate_value = data.get("win_rate", "0")
        profit_factor_value = data.get("profit_factor", "0")
        average_trade_value = data.get("average_trade", "0")
        best_trade_value = data.get("best_trade", "0")
        worst_trade_value = data.get("worst_trade", "0")
        start_date_value = data.get("start_date", "")
        last_update_value = data.get("last_update", "")

        # Преобразование UUID
        try:
            id_uuid = UUID(str(id_value)) if id_value else uuid4()
        except ValueError:
            id_uuid = uuid4()

        try:
            strategy_id_uuid = (
                UUID(str(strategy_id_value)) if strategy_id_value else uuid4()
            )
        except ValueError:
            strategy_id_uuid = uuid4()

        # Преобразование числовых значений
        try:
            total_trades_int = (
                int(total_trades) if isinstance(total_trades, (int, float, str)) else 0
            )
        except (ValueError, TypeError):
            total_trades_int = 0

        try:
            winning_trades_int = (
                int(winning_trades)
                if isinstance(winning_trades, (int, float, str))
                else 0
            )
        except (ValueError, TypeError):
            winning_trades_int = 0

        try:
            losing_trades_int = (
                int(losing_trades)
                if isinstance(losing_trades, (int, float, str))
                else 0
            )
        except (ValueError, TypeError):
            losing_trades_int = 0

        # Преобразование денежных значений
        try:
            total_pnl = Money(Decimal(str(total_pnl_value)), Currency.USD)
        except (ValueError, TypeError):
            total_pnl = Money(Decimal("0"), Currency.USD)

        try:
            total_commission = Money(Decimal(str(total_commission_value)), Currency.USD)
        except (ValueError, TypeError):
            total_commission = Money(Decimal("0"), Currency.USD)

        try:
            max_drawdown = Money(Decimal(str(max_drawdown_value)), Currency.USD)
        except (ValueError, TypeError):
            max_drawdown = Money(Decimal("0"), Currency.USD)

        try:
            sharpe_ratio = Decimal(str(sharpe_ratio_value))
        except (ValueError, TypeError):
            sharpe_ratio = Decimal("0")

        try:
            win_rate = Percentage(Decimal(str(win_rate_value)))
        except (ValueError, TypeError):
            win_rate = Percentage(Decimal("0"))

        try:
            profit_factor = Decimal(str(profit_factor_value))
        except (ValueError, TypeError):
            profit_factor = Decimal("0")

        try:
            average_trade = Money(Decimal(str(average_trade_value)), Currency.USD)
        except (ValueError, TypeError):
            average_trade = Money(Decimal("0"), Currency.USD)

        try:
            best_trade = Money(Decimal(str(best_trade_value)), Currency.USD)
        except (ValueError, TypeError):
            best_trade = Money(Decimal("0"), Currency.USD)

        try:
            worst_trade = Money(Decimal(str(worst_trade_value)), Currency.USD)
        except (ValueError, TypeError):
            worst_trade = Money(Decimal("0"), Currency.USD)

        # Преобразование дат
        try:
            start_date = (
                datetime.fromisoformat(str(start_date_value))
                if start_date_value
                else datetime.now()
            )
        except (ValueError, TypeError):
            start_date = datetime.now()

        try:
            last_update = (
                datetime.fromisoformat(str(last_update_value))
                if last_update_value
                else datetime.now()
            )
        except (ValueError, TypeError):
            last_update = datetime.now()

        return cls(
            id=id_uuid,
            strategy_id=strategy_id_uuid,
            total_trades=total_trades_int,
            winning_trades=winning_trades_int,
            losing_trades=losing_trades_int,
            total_pnl=total_pnl,
            total_commission=total_commission,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade=average_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
            start_date=start_date,
            last_update=last_update,
        )
