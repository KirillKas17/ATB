"""
Оценка эффективности стратегий.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, cast, Sequence
from uuid import UUID, uuid4

from shared.numpy_utils import np
import pandas as pd

from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate,
)
from domain.type_definitions.evolution_types import (
    AccuracyScore,
    ConsistencyScore,
    EntryCondition,
    ExitCondition,
    FitnessScore,
    FitnessWeights,
    ProfitabilityScore,
    RiskScore,
    StrategyPerformance,
    TradePosition,
)


@dataclass
class TradeResult:
    """Результат торговой сделки."""

    id: UUID = field(default_factory=uuid4)
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime = field(default_factory=datetime.now)
    entry_price: Decimal = Decimal("0")
    exit_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    pnl: Decimal = Decimal("0")
    pnl_pct: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    signal_type: str = "buy"
    holding_time: int = 0  # секунды
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_pnl(self) -> None:
        """Рассчитать P&L сделки."""
        if self.entry_price > 0 and self.exit_price > 0 and self.quantity > 0:
            # Базовый P&L
            if self.signal_type.lower() == "buy":
                self.pnl = (self.exit_price - self.entry_price) * self.quantity
            else:  # sell
                self.pnl = (self.entry_price - self.exit_price) * self.quantity
            # Процентный P&L
            if self.entry_price > 0:
                self.pnl_pct = self.pnl / (self.entry_price * self.quantity)
            # Учет комиссии
            self.pnl -= self.commission
            # Определение успешности
            self.success = self.pnl > 0

    def get_roi(self) -> Decimal:
        """Получить ROI сделки."""
        if self.entry_price > 0 and self.quantity > 0:
            initial_investment = self.entry_price * self.quantity
            if initial_investment > 0:
                return self.pnl / initial_investment
        return Decimal("0")

    def get_risk_metrics(self) -> Dict[str, float]:
        """Получить метрики риска для сделки."""
        return {
            "pnl": float(self.pnl),
            "pnl_pct": float(self.pnl_pct),
            "roi": float(self.get_roi()),
            "holding_time": self.holding_time,
            "success": self.success,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "quantity": str(self.quantity),
            "pnl": str(self.pnl),
            "pnl_pct": str(self.pnl_pct),
            "commission": str(self.commission),
            "signal_type": self.signal_type,
            "holding_time": self.holding_time,
            "success": self.success,
            "roi": str(self.get_roi()),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeResult":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            entry_price=Decimal(data["entry_price"]),
            exit_price=Decimal(data["exit_price"]),
            quantity=Decimal(data["quantity"]),
            pnl=Decimal(data["pnl"]),
            pnl_pct=Decimal(data["pnl_pct"]),
            commission=Decimal(data["commission"]),
            signal_type=data["signal_type"],
            holding_time=data["holding_time"],
            success=data["success"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StrategyEvaluationResult:
    """Результат оценки стратегии."""

    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    # Основные метрики
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    accuracy: Decimal = Decimal("0")
    # Финансовые метрики
    total_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    profitability: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    # Риск-метрики
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    # Дополнительные метрики
    average_trade: Decimal = Decimal("0")
    best_trade: Decimal = Decimal("0")
    worst_trade: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    # Временные метрики
    average_holding_time: int = 0
    total_trading_time: int = 0
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)
    # Детализация
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    # Статус оценки
    is_approved: bool = False
    approval_reason: str = ""
    evaluation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_trade(self, trade: TradeResult) -> None:
        """Добавить сделку."""
        self.trades.append(trade)
        self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Пересчитать метрики на основе сделок."""
        if not self.trades:
            return
        # Базовые метрики
        self.total_trades = len(self.trades)
        self.winning_trades = len([t for t in self.trades if t.pnl > 0])
        self.losing_trades = len([t for t in self.trades if t.pnl < 0])
        if self.total_trades > 0:
            self.win_rate = Decimal(str(self.winning_trades / self.total_trades))
            self.accuracy = self.win_rate
        # Финансовые метрики
        total_pnl_sum = sum(trade.pnl for trade in self.trades)
        total_commission_sum = sum(trade.commission for trade in self.trades)
        self.total_pnl = Decimal(str(total_pnl_sum))
        self.total_commission = Decimal(str(total_commission_sum))
        self.net_pnl = self.total_pnl - self.total_commission
        # Прибыльность (отношение к начальному капиталу)
        if self.total_trades > 0:
            initial_capital = sum(
                trade.entry_price * trade.quantity for trade in self.trades
            )
            if initial_capital > 0:
                self.profitability = self.net_pnl / initial_capital
        # Profit factor
        total_wins: Decimal = Decimal(str(sum(trade.pnl for trade in self.trades if trade.pnl > 0)))
        total_losses: Decimal = Decimal(str(abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))))
        if total_losses > Decimal("0"):
            self.profit_factor = total_wins / total_losses
        else:
            self.profit_factor = Decimal("0")
        # Средние значения
        if self.total_trades > 0:
            self.average_trade = self.net_pnl / Decimal(str(self.total_trades))
        if self.winning_trades > 0:
            avg_win = (
                sum(trade.pnl for trade in self.trades if trade.pnl > 0)
                / self.winning_trades
            )
            self.average_win = Decimal(str(avg_win))
        if self.losing_trades > 0:
            avg_loss = (
                sum(trade.pnl for trade in self.trades if trade.pnl < 0)
                / self.losing_trades
            )
            self.average_loss = Decimal(str(avg_loss))
        # Лучшие и худшие сделки
        if self.trades:
            self.best_trade = max(trade.pnl for trade in self.trades)
            self.worst_trade = min(trade.pnl for trade in self.trades)
            winning_trades = [trade for trade in self.trades if trade.pnl > 0]
            losing_trades = [trade for trade in self.trades if trade.pnl < 0]
            if winning_trades:
                self.largest_win = max(trade.pnl for trade in winning_trades)
            if losing_trades:
                self.largest_loss = min(trade.pnl for trade in losing_trades)
        # Временные метрики
        if self.trades:
            total_holding_time = sum(trade.holding_time for trade in self.trades)
            self.average_holding_time = int(total_holding_time / self.total_trades)
            self.start_date = min(trade.entry_time for trade in self.trades)
            self.end_date = max(trade.exit_time for trade in self.trades)
            time_diff = self.end_date - self.start_date
            self.total_trading_time = int(time_diff.total_seconds())
        # Риск-метрики
        self._calculate_risk_metrics()

    def _calculate_risk_metrics(self) -> None:
        """Рассчитать риск-метрики."""
        if not self.trades:
            return
        # Equity curve
        equity = Decimal("0")
        peak_equity = Decimal("0")
        max_dd = Decimal("0")
        returns = []
        for trade in sorted(self.trades, key=lambda x: x.entry_time):
            equity += trade.pnl - trade.commission
            returns.append(float(trade.pnl - trade.commission))
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (
                (peak_equity - equity) / peak_equity
                if peak_equity > 0
                else Decimal("0")
            )
            if drawdown > max_dd:
                max_dd = drawdown
            self.equity_curve.append(
                {
                    "timestamp": trade.exit_time.isoformat(),
                    "equity": float(equity),
                    "drawdown": float(drawdown),
                }
            )
        self.max_drawdown = peak_equity - equity
        self.max_drawdown_pct = max_dd
        # Sharpe ratio
        if len(returns) > 1:
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 0:
                self.sharpe_ratio = Decimal(str(mean_return / std_return))
        # Sortino ratio
        if len(returns) > 1:
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    self.sortino_ratio = Decimal(str(mean_return / downside_std))
        # Calmar ratio
        if self.max_drawdown_pct > 0:
            self.calmar_ratio = self.profitability / self.max_drawdown_pct

    def check_approval_criteria(self, context: EvolutionContext) -> bool:
        """Проверить критерии одобрения."""
        criteria_met = True
        reasons = []
        if self.accuracy < context.min_accuracy:
            criteria_met = False
            reasons.append(f"Accuracy {self.accuracy} < {context.min_accuracy}")
        if self.profitability < context.min_profitability:
            criteria_met = False
            reasons.append(
                f"Profitability {self.profitability} < {context.min_profitability}"
            )
        if self.max_drawdown_pct > context.max_drawdown:
            criteria_met = False
            reasons.append(
                f"Max drawdown {self.max_drawdown_pct} > {context.max_drawdown}"
            )
        if self.sharpe_ratio < context.min_sharpe:
            criteria_met = False
            reasons.append(f"Sharpe ratio {self.sharpe_ratio} < {context.min_sharpe}")
        self.is_approved = criteria_met
        self.approval_reason = "; ".join(reasons) if reasons else "All criteria met"
        return criteria_met

    def get_fitness_score(
        self, weights: Optional[FitnessWeights] = None
    ) -> FitnessScore:
        """Получить общий fitness score."""
        if self.total_trades == 0:
            return FitnessScore(Decimal("0"))
        # Использовать переданные веса или значения по умолчанию
        if weights is None:
            weights = FitnessWeights()
        # Нормализованные оценки (0-1)
        accuracy_score = AccuracyScore(min(self.accuracy, Decimal("1.0")))
        profitability_score = ProfitabilityScore(
            min(max(self.profitability, Decimal("0")), Decimal("1.0"))
        )
        # Риск-скор (обратная зависимость от drawdown)
        risk_score = RiskScore(
            max(Decimal("0"), Decimal("1.0") - self.max_drawdown_pct)
        )
        # Консистентность (Sharpe ratio)
        consistency_score = ConsistencyScore(
            min(max(self.sharpe_ratio / Decimal("3.0"), Decimal("0")), Decimal("1.0"))
        )
        # Общий score
        fitness = (
            accuracy_score * weights.accuracy
            + profitability_score * weights.profitability
            + risk_score * weights.risk
            + consistency_score * weights.consistency
        )
        return FitnessScore(fitness)

    def get_risk_metrics(self) -> Dict[str, float]:
        """Получить метрики риска."""
        return {
            "max_drawdown": float(self.max_drawdown_pct),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "profit_factor": float(self.profit_factor),
            "win_rate": float(self.win_rate),
        }

    def get_performance_summary(self) -> StrategyPerformance:
        """Получить сводку производительности."""
        return StrategyPerformance(
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=float(self.win_rate),
            profit_factor=float(self.profit_factor),
            sharpe_ratio=float(self.sharpe_ratio),
            sortino_ratio=float(self.sortino_ratio),
            calmar_ratio=float(self.calmar_ratio),
            max_drawdown=float(self.max_drawdown_pct),
            total_pnl=float(self.total_pnl),
            net_pnl=float(self.net_pnl),
            average_trade=float(self.average_trade),
            best_trade=float(self.best_trade),
            worst_trade=float(self.worst_trade),
            average_win=float(self.average_win),
            average_loss=float(self.average_loss),
            largest_win=float(self.largest_win),
            largest_loss=float(self.largest_loss),
        )

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Получить анализ сделок."""
        if not self.trades:
            return {}
        # Анализ по типам сигналов
        buy_trades = [t for t in self.trades if t.signal_type.lower() == "buy"]
        sell_trades = [t for t in self.trades if t.signal_type.lower() == "sell"]
        # Анализ по времени удержания
        short_trades = [t for t in self.trades if t.holding_time < 3600]  # < 1 час
        medium_trades = [
            t for t in self.trades if 3600 <= t.holding_time < 86400
        ]  # 1 час - 1 день
        long_trades = [t for t in self.trades if t.holding_time >= 86400]  # >= 1 день
        return {
            "signal_analysis": {
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "buy_win_rate": (
                    len([t for t in buy_trades if t.success]) / len(buy_trades)
                    if buy_trades
                    else 0
                ),
                "sell_win_rate": (
                    len([t for t in sell_trades if t.success]) / len(sell_trades)
                    if sell_trades
                    else 0
                ),
            },
            "holding_time_analysis": {
                "short_trades": len(short_trades),
                "medium_trades": len(medium_trades),
                "long_trades": len(long_trades),
                "short_win_rate": (
                    len([t for t in short_trades if t.success]) / len(short_trades)
                    if short_trades
                    else 0
                ),
                "medium_win_rate": (
                    len([t for t in medium_trades if t.success]) / len(medium_trades)
                    if medium_trades
                    else 0
                ),
                "long_win_rate": (
                    len([t for t in long_trades if t.success]) / len(long_trades)
                    if long_trades
                    else 0
                ),
            },
            "monthly_performance": self._get_monthly_performance(),
        }

    def _get_monthly_performance(self) -> Dict[str, float]:
        """Получить месячную производительность."""
        monthly_pnl = {}
        for trade in self.trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0.0
            monthly_pnl[month_key] += float(trade.pnl)
        return monthly_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.win_rate),
            "accuracy": str(self.accuracy),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "profitability": str(self.profitability),
            "profit_factor": str(self.profit_factor),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "average_trade": str(self.average_trade),
            "best_trade": str(self.best_trade),
            "worst_trade": str(self.worst_trade),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "average_holding_time": self.average_holding_time,
            "total_trading_time": self.total_trading_time,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "is_approved": self.is_approved,
            "approval_reason": self.approval_reason,
            "fitness_score": str(self.get_fitness_score()),
            "evaluation_time": self.evaluation_time.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyEvaluationResult":
        """Создать из словаря."""
        result = cls(
            id=UUID(data["id"]),
            strategy_id=UUID(data["strategy_id"]),
            total_trades=data["total_trades"],
            winning_trades=data["winning_trades"],
            losing_trades=data["losing_trades"],
            win_rate=Decimal(data["win_rate"]),
            accuracy=Decimal(data["accuracy"]),
            total_pnl=Decimal(data["total_pnl"]),
            net_pnl=Decimal(data["net_pnl"]),
            profitability=Decimal(data["profitability"]),
            profit_factor=Decimal(data["profit_factor"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            max_drawdown_pct=Decimal(data["max_drawdown_pct"]),
            sharpe_ratio=Decimal(data["sharpe_ratio"]),
            sortino_ratio=Decimal(data["sortino_ratio"]),
            calmar_ratio=Decimal(data["calmar_ratio"]),
            average_trade=Decimal(data["average_trade"]),
            best_trade=Decimal(data["best_trade"]),
            worst_trade=Decimal(data["worst_trade"]),
            average_win=Decimal(data["average_win"]),
            average_loss=Decimal(data["average_loss"]),
            largest_win=Decimal(data["largest_win"]),
            largest_loss=Decimal(data["largest_loss"]),
            average_holding_time=data["average_holding_time"],
            total_trading_time=data["total_trading_time"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            is_approved=data["is_approved"],
            approval_reason=data["approval_reason"],
            evaluation_time=datetime.fromisoformat(data["evaluation_time"]),
            metadata=data["metadata"],
        )
        # Восстановить сделки
        for trade_data in data.get("trades", []):
            trade = TradeResult.from_dict(trade_data)
            result.trades.append(trade)
        # Восстановить equity curve
        result.equity_curve = data.get("equity_curve", [])
        return result


class StrategyFitnessEvaluator:
    """Оценщик эффективности стратегий."""

    def __init__(self, weights: Optional[FitnessWeights] = None):
        self.weights = weights or FitnessWeights()
        self.evaluation_results: Dict[UUID, StrategyEvaluationResult] = {}
        self.current_data: pd.DataFrame = pd.DataFrame()

    def evaluate_strategy(
        self,
        candidate: StrategyCandidate,
        historical_data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
    ) -> StrategyEvaluationResult:
        """Оценить стратегию."""
        print(f"Оцениваю стратегию: {candidate.name}")
        # Создать результат оценки
        evaluation = StrategyEvaluationResult(
            strategy_id=candidate.id,
            evaluation_time=datetime.now(),
        )
        # Симулировать торговлю
        trades = self._simulate_trading(candidate, historical_data, initial_capital)
        # Добавить сделки в оценку
        for trade in trades:
            evaluation.add_trade(trade)
        # Сохранить результат
        self.evaluation_results[candidate.id] = evaluation
        print(f"Оценка завершена: {len(trades)} сделок, P&L: {evaluation.net_pnl}")
        return evaluation

    def _simulate_trading(
        self, candidate: StrategyCandidate, data: pd.DataFrame, initial_capital: Decimal
    ) -> List[TradeResult]:
        """Симулировать торговлю."""
        trades = []
        positions: List[TradePosition] = []
        equity = initial_capital
        for i in range(len(data)):
            current_data = data.iloc[i : i + 1]
            # Проверить сигналы входа
            entry_signal = self._check_entry_signals(candidate, current_data, i, data)
            if entry_signal and len(positions) < candidate.max_positions:
                # Открыть позицию
                position = self._open_position(
                    candidate, current_data, entry_signal, equity
                )
                positions.append(position)
            # Проверить сигналы выхода для существующих позиций
            for j, position in enumerate(positions):
                exit_signal = self._check_exit_signals(
                    candidate, current_data, position, i, data
                )
                if exit_signal:
                    # Закрыть позицию
                    trade = self._close_position(position, current_data, exit_signal)
                    trades.append(trade)
                    equity += trade.pnl
                    positions.pop(j)
        # Закрыть оставшиеся позиции
        for position in positions:
            trade = self._close_position(position, data.iloc[-1:], "end_of_data")
            trades.append(trade)
        return trades

    def _check_entry_signals(
        self,
        candidate: StrategyCandidate,
        current_data: pd.DataFrame,
        current_index: int,
        full_data: pd.DataFrame,
    ) -> Optional[str]:
        """Проверить сигналы входа."""
        self.current_data = current_data
        for rule in candidate.get_active_entry_rules():
            if self._evaluate_conditions(rule.conditions):
                return rule.signal_type.value
        return None

    def _check_exit_signals(
        self,
        candidate: StrategyCandidate,
        current_data: pd.DataFrame,
        position: TradePosition,
        current_index: int,
        full_data: pd.DataFrame,
    ) -> Optional[str]:
        """Проверить сигналы выхода."""
        self.current_data = current_data
        current_price = Decimal(str(current_data.iloc[0]["close"]))
        # Проверить stop loss и take profit
        if position["stop_loss"] and current_price <= position["stop_loss"]:
            return "stop_loss"
        if position["take_profit"] and current_price >= position["take_profit"]:
            return "take_profit"
        # Проверить правила выхода
        for rule in candidate.get_active_exit_rules():
            if self._evaluate_conditions(rule.conditions):
                return rule.signal_type.value
        return None

    def _evaluate_conditions(
        self, conditions: Sequence[Union[Dict[str, Any], ExitCondition, EntryCondition]]
    ) -> bool:
        """Оценка условий входа/выхода"""
        for condition in conditions:
            if not self._evaluate_single_condition(condition):
                return False
        return True

    def _evaluate_single_condition(
        self, condition: Union[Dict[str, Any], ExitCondition, EntryCondition]
    ) -> bool:
        """Оценка одного условия"""
        try:
            # Попробуем обработать как dict
            if isinstance(condition, dict):
                return self._evaluate_dict_condition(cast(Dict[str, Any], condition))
        except (TypeError, AttributeError):
            pass
        
        try:
            # Попробуем обработать как ExitCondition
            if hasattr(condition, 'stop_loss_pct') or hasattr(condition, 'take_profit_pct'):
                return self._evaluate_exit_condition(cast(ExitCondition, condition), self.current_data)
        except (TypeError, AttributeError):
            pass
        
        try:
            # Попробуем обработать как EntryCondition
            return self._evaluate_entry_condition(cast(EntryCondition, condition), self.current_data)
        except (TypeError, AttributeError):
            pass
        
        # Если тип неизвестен, возвращаем False
        return False

    def _evaluate_dict_condition(self, condition: Dict[str, Any]) -> bool:
        """Оценить условие, представленное словарем."""
        # Простая реализация - в реальности здесь была бы сложная логика
        # оценки технических индикаторов
        return True

    def _evaluate_exit_condition(
        self, condition: ExitCondition, current_data: pd.DataFrame
    ) -> bool:
        """Оценить условие выхода."""
        # Простая реализация для ExitCondition
        return True

    def _evaluate_entry_condition(
        self, condition: EntryCondition, current_data: pd.DataFrame
    ) -> bool:
        """Оценить условие входа."""
        # Простая реализация для EntryCondition
        return True

    def _open_position(
        self,
        candidate: StrategyCandidate,
        current_data: pd.DataFrame,
        signal_type: str,
        equity: Decimal,
    ) -> TradePosition:
        """Открыть позицию."""
        current_price = Decimal(str(current_data.iloc[0]["close"]))
        quantity = equity * candidate.position_size_pct / current_price
        entry_time = current_data.index[0].to_pydatetime()
        
        # Создаем TradePosition с обязательными полями
        position: TradePosition = {
            "id": str(uuid4()),
            "entry_time": entry_time,
            "entry_price": current_price,
            "quantity": quantity,
            "signal_type": signal_type,
            "stop_loss": (
                current_price * (1 - candidate.exit_rules[0].stop_loss_pct)
                if candidate.exit_rules
                else None
            ),
            "take_profit": (
                current_price * (1 + candidate.exit_rules[0].take_profit_pct)
                if candidate.exit_rules
                else None
            ),
            "trailing_stop": (
                candidate.exit_rules[0].trailing_stop if candidate.exit_rules else False
            ),
            "trailing_distance": (
                candidate.exit_rules[0].trailing_distance
                if candidate.exit_rules
                else None
            ),
            "current_price": current_price,
            "unrealized_pnl": Decimal("0"),
            "holding_time": 0,
        }
        return position

    def _close_position(
        self, position: TradePosition, current_data: pd.DataFrame, exit_reason: str
    ) -> TradeResult:
        """Закрыть позицию."""
        current_price = Decimal(str(current_data.iloc[0]["close"]))
        exit_time = current_data.index[0].to_pydatetime()
        # Рассчитать P&L
        if position["signal_type"].lower() == "buy":
            pnl = (current_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - current_price) * position["quantity"]
        # Комиссия (упрощенно)
        commission = position["entry_price"] * position["quantity"] * Decimal("0.001")
        trade = TradeResult(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            entry_price=position["entry_price"],
            exit_price=current_price,
            quantity=position["quantity"],
            pnl=pnl,
            commission=commission,
            signal_type=position["signal_type"],
            holding_time=int((exit_time - position["entry_time"]).total_seconds()),
        )
        trade.calculate_pnl()
        return trade

    def get_evaluation_result(
        self, strategy_id: UUID
    ) -> Optional[StrategyEvaluationResult]:
        """Получить результат оценки."""
        return self.evaluation_results.get(strategy_id)

    def get_all_results(self) -> List[StrategyEvaluationResult]:
        """Получить все результаты."""
        return list(self.evaluation_results.values())

    def get_approved_strategies(self) -> List[StrategyEvaluationResult]:
        """Получить одобренные стратегии."""
        return [r for r in self.evaluation_results.values() if r.is_approved]

    def get_top_strategies(self, n: int = 10) -> List[StrategyEvaluationResult]:
        """Получить топ стратегии."""
        sorted_results = sorted(
            self.evaluation_results.values(),
            key=lambda x: x.get_fitness_score(),
            reverse=True,
        )
        return sorted_results[:n]

    def clear_results(self) -> None:
        """Очистить результаты."""
        self.evaluation_results.clear()

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Получить статистику оценок."""
        if not self.evaluation_results:
            return {}
        results = list(self.evaluation_results.values())
        return {
            "total_evaluations": len(results),
            "approved_count": len([r for r in results if r.is_approved]),
            "average_fitness": float(
                sum(r.get_fitness_score() for r in results) / len(results)
            ),
            "best_fitness": float(max(r.get_fitness_score() for r in results)),
            "worst_fitness": float(min(r.get_fitness_score() for r in results)),
            "average_trades": sum(r.total_trades for r in results) / len(results),
            "average_win_rate": float(sum(r.win_rate for r in results) / len(results)),
            "average_profitability": float(
                sum(r.profitability for r in results) / len(results)
            ),
        }
