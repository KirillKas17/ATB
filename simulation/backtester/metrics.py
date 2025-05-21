from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .types import BacktestResult, Trade


@dataclass
class ExtendedMetrics:
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    volatility: float = 0.0
    mar_ratio: float = 0.0
    ulcer_index: float = 0.0
    omega_ratio: float = 0.0
    gini_coefficient: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    drawdown_duration: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    median_trade: float = 0.0
    median_duration: float = 0.0
    profit_streak: int = 0
    loss_streak: int = 0
    stability: float = 0.0
    additional: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """Калькулятор метрик бэктеста (расширенный)"""

    @staticmethod
    def calculate_metrics(trades: List[Trade], equity_curve: List[float]) -> Dict[str, float]:
        """
        Расчет расширенных метрик бэктеста
        """
        if not trades or not equity_curve:
            return {}

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        returns = np.diff(equity_curve)
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 1
        sharpe_ratio = float(mean_return / std_return) if std_return > 0 else 0
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            float(mean_return / np.std(downside_returns)) if len(downside_returns) > 0 else 0
        )
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0
        calmar_ratio = float(mean_return / max_drawdown) if max_drawdown > 0 else 0
        # MAR (Mean Annual Return / Max Drawdown)
        mar_ratio = float(np.sum(returns) / max_drawdown) if max_drawdown > 0 else 0
        # Ulcer Index
        ulcer_index = float(np.sqrt(np.mean(drawdown**2))) if len(drawdown) > 0 else 0
        # Omega Ratio
        threshold = 0
        omega_ratio = (
            float(np.sum(returns[returns > threshold]) / abs(np.sum(returns[returns < threshold])))
            if np.sum(returns[returns < threshold]) != 0
            else 0
        )
        # Gini coefficient
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        gini = (
            float(
                (2 * np.sum((np.arange(1, n + 1) * sorted_returns)) / (n * np.sum(sorted_returns)))
                - (n + 1) / n
            )
            if n > 0 and np.sum(sorted_returns) != 0
            else 0
        )
        # Tail Ratio
        tail_ratio = (
            float(np.percentile(returns, 95) / abs(np.percentile(returns, 5)))
            if np.percentile(returns, 5) != 0
            else 0
        )
        # Skewness & Kurtosis
        skewness = (
            float((np.mean((returns - mean_return) ** 3)) / (std_return**3))
            if std_return > 0
            else 0
        )
        kurtosis = (
            float((np.mean((returns - mean_return) ** 4)) / (std_return**4))
            if std_return > 0
            else 0
        )
        # Value at Risk (VaR) 95%
        var_95 = float(np.percentile(returns, 5))
        # Conditional VaR (CVaR) 95%
        cvar_95 = (
            float(np.mean(returns[returns <= var_95])) if len(returns[returns <= var_95]) > 0 else 0
        )
        # Drawdown duration
        drawdown_duration = float(np.argmax(drawdown)) if len(drawdown) > 0 else 0
        # Max/Min equity
        max_equity = float(np.max(equity_curve)) if len(equity_curve) > 0 else 0
        min_equity = float(np.min(equity_curve)) if len(equity_curve) > 0 else 0
        # Median trade
        median_trade = float(np.median([t.pnl for t in trades])) if trades else 0
        # Median duration
        median_duration = (
            float(
                np.median(
                    [
                        (t.exit_price or t.price) - (t.entry_price or t.price)
                        for t in trades
                        if t.exit_price and t.entry_price
                    ]
                )
            )
            if trades
            else 0
        )
        # Profit/loss streaks
        profit_streak = MetricsCalculator._max_streak([t.pnl > 0 for t in trades])
        loss_streak = MetricsCalculator._max_streak([t.pnl < 0 for t in trades])
        # Stability (R^2 of equity curve)
        x = np.arange(len(equity_curve))
        if len(equity_curve) > 1:
            coeffs = np.polyfit(x, equity_curve, 1)
            fit = np.polyval(coeffs, x)
            stability = float(
                1
                - np.sum((equity_curve - fit) ** 2)
                / np.sum((equity_curve - np.mean(equity_curve)) ** 2)
            )
        else:
            stability = 0
        # Базовые метрики
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        avg_profit = (
            float(np.mean([t.pnl for t in trades if t.pnl > 0])) if winning_trades > 0 else 0
        )
        avg_loss = (
            float(np.mean([t.pnl for t in trades if t.pnl < 0]))
            if (total_trades - winning_trades) > 0
            else 0
        )
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)
        recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0
        risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        kelly_criterion = (
            (win_rate * (profit_factor + 1) - 1) / profit_factor if profit_factor > 0 else 0
        )
        metrics = ExtendedMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            recovery_factor=recovery_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            expectancy=expectancy,
            risk_reward_ratio=risk_reward_ratio,
            kelly_criterion=kelly_criterion,
            volatility=std_return,
            mar_ratio=mar_ratio,
            ulcer_index=ulcer_index,
            omega_ratio=omega_ratio,
            gini_coefficient=gini,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            drawdown_duration=drawdown_duration,
            max_equity=max_equity,
            min_equity=min_equity,
            median_trade=median_trade,
            median_duration=median_duration,
            profit_streak=profit_streak,
            loss_streak=loss_streak,
            stability=stability,
        )
        return metrics.__dict__

    @staticmethod
    def _max_streak(bools: List[bool]) -> int:
        max_streak = streak = 0
        for b in bools:
            if b:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def calculate_trade_metrics(trade: Trade) -> Dict[str, float]:
        """
        Расчет метрик для отдельной сделки
        """
        if not trade.entry_price or not trade.exit_price:
            return {}
        pnl = trade.pnl
        pnl_percent = (
            (pnl / (trade.entry_price * trade.volume)) * 100
            if trade.entry_price and trade.volume
            else 0
        )
        duration = (trade.timestamp - trade.timestamp).total_seconds() / 3600  # в часах (заглушка)
        risk = abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0
        reward = abs(trade.take_profit - trade.entry_price) if trade.take_profit else 0
        risk_reward = reward / risk if risk > 0 else 0
        return {
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "duration": duration,
            "risk": risk,
            "reward": reward,
            "risk_reward": risk_reward,
        }

    @staticmethod
    def calculate_portfolio_metrics(results: List[BacktestResult]) -> Dict[str, float]:
        """
        Расчет метрик для портфеля стратегий
        """
        if not results:
            return {}
        all_trades = []
        for result in results:
            all_trades.extend(result.trades)
        all_equity = []
        for result in results:
            all_equity.extend(result.equity_curve)
        portfolio_metrics = MetricsCalculator.calculate_metrics(all_trades, all_equity)
        correlations = {}
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i + 1 :], i + 1):
                if len(result1.equity_curve) == len(result2.equity_curve):
                    corr = np.corrcoef(result1.equity_curve, result2.equity_curve)[0, 1]
                else:
                    corr = 0
                correlations[f"correlation_{i}_{j}"] = float(corr)
        portfolio_metrics.update(correlations)
        return portfolio_metrics
