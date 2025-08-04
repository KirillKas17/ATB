"""
Калькулятор метрик с полной реализацией всех методов.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger

from shared.numpy_utils import np

from .types import Trade, TradeDirection


@dataclass
class MetricsConfig:
    """Конфигурация калькулятора метрик."""

    risk_free_rate: float = 0.02  # 2% годовых
    trading_days_per_year: int = 252
    min_trades_for_metrics: int = 10
    max_drawdown_threshold: float = 0.5  # 50%
    sharpe_ratio_threshold: float = 1.0
    sortino_ratio_threshold: float = 1.0
    calmar_ratio_threshold: float = 1.0
    profit_factor_threshold: float = 1.5
    win_rate_threshold: float = 0.5  # 50%


@dataclass
class MetricsResult:
    """Результаты расчета метрик."""

    # Базовые метрики
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    # Риск-метрики
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    # Дополнительные метрики
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    average_trade: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    # Временные метрики
    total_duration: timedelta = timedelta(0)
    average_trade_duration: timedelta = timedelta(0)
    longest_winning_streak: int = 0
    longest_losing_streak: int = 0
    current_streak: int = 0
    # Качество исполнения
    average_slippage: float = 0.0
    average_commission: float = 0.0
    execution_quality: float = 0.0
    # Валидация
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100, где 100 - максимальный риск


class MetricsCalculator:
    """Промышленный калькулятор метрик с полной реализацией."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        """Инициализация калькулятора метрик."""
        self.config = config or MetricsConfig()
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        logger.add(
            "logs/metrics_calculator.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    def calculate_metrics(
        self, trades: List[Trade], equity_curve: List[float]
    ) -> Dict[str, float]:
        """Расчет всех метрик бэктеста."""
        try:
            if not trades:
                logger.warning("No trades provided for metrics calculation")
                return self._empty_metrics()
            # Создание результата
            result = MetricsResult()
            # Базовые метрики
            self._calculate_basic_metrics(trades, result)
            # Риск-метрики
            self._calculate_risk_metrics(trades, equity_curve, result)
            # Дополнительные метрики
            self._calculate_advanced_metrics(trades, result)
            # Временные метрики
            self._calculate_time_metrics(trades, result)
            # Качество исполнения
            self._calculate_execution_metrics(trades, result)
            # Валидация результатов
            self._validate_metrics(result)
            # Расчет общего рискового скора
            result.risk_score = self._calculate_risk_score(result)
            # Преобразование в словарь
            return self._result_to_dict(result)
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._empty_metrics()

    def _calculate_basic_metrics(
        self, trades: List[Trade], result: MetricsResult
    ) -> None:
        """Расчет базовых метрик."""
        try:
            result.total_trades = len(trades)
            # Разделение на прибыльные и убыточные сделки
            winning_trades = [t for t in trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value') and t.pnl.value > 0]
            losing_trades = [t for t in trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value') and t.pnl.value < 0]
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            # Винрейт
            result.win_rate = (
                result.winning_trades / result.total_trades
                if result.total_trades > 0
                else 0.0
            )
            # Прибыль/убыток
            result.total_profit = sum(t.pnl.value for t in winning_trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value'))
            result.total_loss = abs(sum(t.pnl.value for t in losing_trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value')))
            result.net_profit = result.total_profit - result.total_loss
            # Profit Factor
            result.profit_factor = (
                result.total_profit / result.total_loss
                if result.total_loss > 0
                else float("inf")
            )
            # Средние значения
            result.average_trade = (
                result.net_profit / result.total_trades
                if result.total_trades > 0
                else 0.0
            )
            result.average_win = (
                result.total_profit / result.winning_trades
                if result.winning_trades > 0
                else 0.0
            )
            result.average_loss = (
                result.total_loss / result.losing_trades
                if result.losing_trades > 0
                else 0.0
            )
            # Максимальные значения
            result.largest_win = max((t.pnl.value for t in winning_trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value')), default=0.0)
            result.largest_loss = min((t.pnl.value for t in losing_trades if hasattr(t, 'pnl') and hasattr(t.pnl, 'value')), default=0.0)
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")

    def _calculate_risk_metrics(
        self, trades: List[Trade], equity_curve: List[float], result: MetricsResult
    ) -> None:
        """Расчет риск-метрик."""
        try:
            if len(equity_curve) < 2:
                return
            # Расчет доходностей
            returns = np.diff(equity_curve)
            # Коэффициент Шарпа
            result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            # Коэффициент Сортино
            result.sortino_ratio = self._calculate_sortino_ratio(returns)
            # Максимальная просадка
            result.max_drawdown = self._calculate_max_drawdown(equity_curve)
            # Коэффициент Кальмара
            result.calmar_ratio = self._calculate_calmar_ratio(
                result.net_profit, result.max_drawdown
            )
            # Recovery Factor
            result.recovery_factor = self._calculate_recovery_factor(
                result.net_profit, result.max_drawdown
            )
            # Value at Risk и Conditional Value at Risk
            result.var_95, result.cvar_95 = self._calculate_var_cvar(returns)
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")

    def _calculate_advanced_metrics(
        self, trades: List[Trade], result: MetricsResult
    ) -> None:
        """Расчет дополнительных метрик."""
        try:
            # Expectancy
            result.expectancy = (result.win_rate * result.average_win) - (
                (1 - result.win_rate) * result.average_loss
            )
            # Risk/Reward Ratio
            result.risk_reward_ratio = (
                result.average_win / result.average_loss
                if result.average_loss > 0
                else 0.0
            )
            # Kelly Criterion
            result.kelly_criterion = self._calculate_kelly_criterion(
                result.win_rate, result.profit_factor
            )
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")

    def _calculate_time_metrics(
        self, trades: List[Trade], result: MetricsResult
    ) -> None:
        """Расчет временных метрик."""
        try:
            if not trades:
                return
            # Общая продолжительность
            start_time = min(t.timestamp for t in trades)
            end_time = max(t.timestamp for t in trades)
            result.total_duration = end_time - start_time
            # Средняя продолжительность сделки
            trade_durations = []
            for i in range(1, len(trades)):
                duration = trades[i].timestamp - trades[i - 1].timestamp
                trade_durations.append(duration)
            if trade_durations:
                avg_duration = sum(trade_durations, timedelta(0)) / len(trade_durations)
                result.average_trade_duration = avg_duration
            # Серии побед и поражений
            (
                result.longest_winning_streak,
                result.longest_losing_streak,
                result.current_streak,
            ) = self._calculate_streaks(trades)
        except Exception as e:
            logger.error(f"Error calculating time metrics: {str(e)}")

    def _calculate_execution_metrics(
        self, trades: List[Trade], result: MetricsResult
    ) -> None:
        """Расчет метрик качества исполнения."""
        try:
            if not trades:
                return
            # Средняя комиссия
            total_commission = sum(t.commission for t in trades if hasattr(t, 'commission'))
            result.average_commission = total_commission / len(trades) if trades else 0.0
            # Среднее проскальзывание (если доступно)
            slippages = []
            for trade in trades:
                if hasattr(trade, "slippage") and trade.slippage is not None:
                    slippages.append(trade.slippage)
            if slippages:
                result.average_slippage = sum(slippages) / len(slippages)
            # Качество исполнения (общая оценка)
            result.execution_quality = self._calculate_execution_quality(trades)
        except Exception as e:
            logger.error(f"Error calculating execution metrics: {str(e)}")

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Расчет коэффициента Шарпа."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - (
            self.config.risk_free_rate / self.config.trading_days_per_year
        )
        result = (
            np.mean(excess_returns)
            / np.std(returns)
            * np.sqrt(self.config.trading_days_per_year)
        )
        return float(result)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Расчет коэффициента Сортино."""
        if len(returns) == 0:
            return 0.0
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float("inf")
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        excess_returns = returns - (
            self.config.risk_free_rate / self.config.trading_days_per_year
        )
        result = (
            np.mean(excess_returns)
            / downside_deviation
            * np.sqrt(self.config.trading_days_per_year)
        )
        return float(result)

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Расчет максимальной просадки."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _calculate_calmar_ratio(self, net_profit: float, max_drawdown: float) -> float:
        """Расчет коэффициента Кальмара."""
        if max_drawdown == 0:
            return 0.0
        return net_profit / max_drawdown

    def _calculate_recovery_factor(
        self, net_profit: float, max_drawdown: float
    ) -> float:
        """Расчет фактора восстановления."""
        if max_drawdown == 0:
            return 0.0
        return net_profit / max_drawdown

    def _calculate_var_cvar(self, returns: np.ndarray) -> Tuple[float, float]:
        """Расчет VaR и CVaR."""
        if len(returns) == 0:
            return 0.0, 0.0
        # VaR 95%
        var_95 = np.percentile(returns, 5)
        # CVaR 95%
        cvar_95 = np.mean(returns[returns <= var_95])
        return var_95, cvar_95

    def _calculate_kelly_criterion(
        self, win_rate: float, profit_factor: float
    ) -> float:
        """Расчет критерия Келли."""
        if profit_factor <= 1:
            return 0.0
        return (win_rate * profit_factor - (1 - win_rate)) / profit_factor

    def _calculate_streaks(self, trades: List[Trade]) -> Tuple[int, int, int]:
        """Расчет серий побед и поражений."""
        if not trades:
            return 0, 0, 0
        longest_winning = 0
        longest_losing = 0
        current_streak = 0
        current_streak_type = None
        max_winning = 0
        max_losing = 0
        for trade in trades:
            if hasattr(trade, 'pnl') and hasattr(trade.pnl, 'value') and trade.pnl.value > 0:  # Победа
                if current_streak_type == "win":
                    current_streak += 1
                else:
                    if current_streak_type == "loss":
                        max_losing = max(max_losing, current_streak)
                    current_streak = 1
                    current_streak_type = "win"
            else:  # Поражение
                if current_streak_type == "loss":
                    current_streak += 1
                else:
                    if current_streak_type == "win":
                        max_winning = max(max_winning, current_streak)
                    current_streak = 1
                    current_streak_type = "loss"
        # Обновление максимальных значений для последней серии
        if current_streak_type == "win":
            max_winning = max(max_winning, current_streak)
        else:
            max_losing = max(max_losing, current_streak)
        return max_winning, max_losing, current_streak

    def _calculate_execution_quality(self, trades: List[Trade]) -> float:
        """Расчет качества исполнения."""
        try:
            if not trades:
                return 0.0
            quality_scores = []
            for trade in trades:
                score = 1.0
                # Штраф за высокую комиссию
                if trade.commission > 0:
                    commission_ratio = trade.commission / (trade.volume * trade.price)
                    if commission_ratio > 0.01:  # Более 1%
                        score -= 0.2
                # Штраф за проскальзывание (если доступно)
                if hasattr(trade, "slippage") and trade.slippage is not None:
                    slippage_ratio = trade.slippage / (trade.volume * trade.price)
                    if slippage_ratio > 0.005:  # Более 0.5%
                        score -= 0.3
                # Бонус за прибыльную сделку
                if hasattr(trade, 'pnl') and hasattr(trade.pnl, 'value') and trade.pnl.value > 0:
                    score += 0.1
                quality_scores.append(max(0.0, min(1.0, score)))
            return float(np.mean(quality_scores))
        except Exception as e:
            logger.error(f"Error calculating execution quality: {str(e)}")
            return 0.0

    def _validate_metrics(self, result: MetricsResult) -> None:
        """Валидация метрик."""
        try:
            errors = []
            # Проверка минимального количества сделок
            if result.total_trades < self.config.min_trades_for_metrics:
                errors.append(
                    f"Insufficient trades: {result.total_trades} < {self.config.min_trades_for_metrics}"
                )
            # Проверка винрейта
            if result.win_rate < self.config.win_rate_threshold:
                errors.append(
                    f"Low win rate: {result.win_rate:.2%} < {self.config.win_rate_threshold:.2%}"
                )
            # Проверка profit factor
            if result.profit_factor < self.config.profit_factor_threshold:
                errors.append(
                    f"Low profit factor: {result.profit_factor:.2f} < {self.config.profit_factor_threshold:.2f}"
                )
            # Проверка максимальной просадки
            if result.max_drawdown > self.config.max_drawdown_threshold:
                errors.append(
                    f"High max drawdown: {result.max_drawdown:.2%} > {self.config.max_drawdown_threshold:.2%}"
                )
            # Проверка коэффициента Шарпа
            if result.sharpe_ratio < self.config.sharpe_ratio_threshold:
                errors.append(
                    f"Low Sharpe ratio: {result.sharpe_ratio:.2f} < {self.config.sharpe_ratio_threshold:.2f}"
                )
            result.validation_errors = errors
            result.is_valid = len(errors) == 0
        except Exception as e:
            logger.error(f"Error validating metrics: {str(e)}")
            result.is_valid = False
            result.validation_errors = [f"Validation error: {str(e)}"]

    def _calculate_risk_score(self, result: MetricsResult) -> float:
        """Расчет общего рискового скора (0-100)."""
        try:
            score = 0.0
            # Факторы риска
            risk_factors = [
                (result.max_drawdown, 30),  # Максимальная просадка - 30%
                (1 - result.win_rate, 20),  # Процент проигрышных сделок - 20%
                (1 / max(result.profit_factor, 1), 15),  # Обратный profit factor - 15%
                (max(0, 1 - result.sharpe_ratio), 15),  # Низкий Sharpe ratio - 15%
                (result.var_95, 10),  # VaR - 10%
                (result.cvar_95, 10),  # CVaR - 10%
            ]
            for factor, weight in risk_factors:
                score += min(1.0, max(0.0, factor)) * weight
            return min(100.0, score)
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 50.0  # Средний риск по умолчанию

    def _result_to_dict(self, result: MetricsResult) -> Dict[str, float]:
        """Преобразование результата в словарь."""
        return {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_profit": result.total_profit,
            "total_loss": result.total_loss,
            "net_profit": result.net_profit,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "recovery_factor": result.recovery_factor,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "expectancy": result.expectancy,
            "risk_reward_ratio": result.risk_reward_ratio,
            "kelly_criterion": result.kelly_criterion,
            "average_trade": result.average_trade,
            "average_win": result.average_win,
            "average_loss": result.average_loss,
            "largest_win": result.largest_win,
            "largest_loss": result.largest_loss,
            "longest_winning_streak": result.longest_winning_streak,
            "longest_losing_streak": result.longest_losing_streak,
            "current_streak": result.current_streak,
            "average_slippage": result.average_slippage,
            "average_commission": result.average_commission,
            "execution_quality": result.execution_quality,
            "risk_score": result.risk_score,
            "is_valid": float(result.is_valid),
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """Пустые метрики."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "net_profit": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "recovery_factor": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "expectancy": 0.0,
            "risk_reward_ratio": 0.0,
            "kelly_criterion": 0.0,
            "average_trade": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "longest_winning_streak": 0,
            "longest_losing_streak": 0,
            "current_streak": 0,
            "average_slippage": 0.0,
            "average_commission": 0.0,
            "execution_quality": 0.0,
            "risk_score": 0.0,
            "is_valid": 0.0,
        }

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Генерация отчета по метрикам."""
        try:
            report = []
            report.append("=" * 60)
            report.append("BACKTEST METRICS REPORT")
            report.append("=" * 60)
            # Базовые метрики
            report.append("\n📊 BASIC METRICS:")
            report.append(f"Total Trades: {metrics['total_trades']}")
            report.append(f"Winning Trades: {metrics['winning_trades']}")
            report.append(f"Losing Trades: {metrics['losing_trades']}")
            report.append(f"Win Rate: {metrics['win_rate']:.2%}")
            report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            report.append(f"Net Profit: ${metrics['net_profit']:.2f}")
            # Риск-метрики
            report.append("\n⚠️ RISK METRICS:")
            report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            report.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
            report.append(f"VaR (95%): {metrics['var_95']:.4f}")
            report.append(f"CVaR (95%): {metrics['cvar_95']:.4f}")
            # Дополнительные метрики
            report.append("\n📈 ADVANCED METRICS:")
            report.append(f"Expectancy: {metrics['expectancy']:.2f}")
            report.append(f"Risk/Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
            report.append(f"Kelly Criterion: {metrics['kelly_criterion']:.2f}")
            report.append(f"Average Trade: ${metrics['average_trade']:.2f}")
            report.append(f"Largest Win: ${metrics['largest_win']:.2f}")
            report.append(f"Largest Loss: ${metrics['largest_loss']:.2f}")
            # Качество исполнения
            report.append("\n⚡ EXECUTION QUALITY:")
            report.append(f"Average Commission: ${metrics['average_commission']:.4f}")
            report.append(f"Average Slippage: ${metrics['average_slippage']:.4f}")
            report.append(f"Execution Quality: {metrics['execution_quality']:.2%}")
            # Общий риск
            report.append(f"\n🎯 RISK SCORE: {metrics['risk_score']:.1f}/100")
            report.append(
                f"VALIDATION: {'✅ PASSED' if metrics['is_valid'] else '❌ FAILED'}"
            )
            report.append("\n" + "=" * 60)
            return "\n".join(report)
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "Error generating metrics report"
