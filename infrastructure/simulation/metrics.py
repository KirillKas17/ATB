import pandas as pd
from typing import List

from .types import (
    BacktestConfig,
    BacktestMetricsDict,
    BaseSimulationComponent,
    ExecutionType,
    MarketImpactType,
    MarketMetricsDict,
    MarketRegimeType,
    MarketSimulationConfig,
    RiskLevelType,
    SignalStrengthType,
    SimulationConfig,
    SimulationConstants,
    SimulationMoney,
    SimulationPrice,
    SimulationTrade,
    SimulationUtils,
    SimulationVolume,
    TradeMetricsDict,
)


class MarketMetricsCalculator:
    """Калькулятор метрик рынка и эффективности симуляции."""

    @staticmethod
    def calculate_market_metrics(data: pd.DataFrame) -> MarketMetricsDict:
        """Расчёт метрик рынка."""
        if data.empty:
            return {
                "volatility": 0.0,
                "trend_strength": 0.0,
                "volume_trend": 0.0,
                "momentum": 0.0,
                "regime_score": 0.0,
                "liquidity_score": 0.0,
                "sentiment_score": 0.0,
            }

        # Расчёт волатильности
        returns = data["close"].pct_change().dropna()
        volatility = returns.std() * (252**0.5) if len(returns) > 1 else 0.0

        # Расчёт силы тренда
        trend_strength = (
            abs(data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0]
            if len(data) > 1
            else 0.0
        )

        # Расчёт тренда объёма
        volume_trend = data["volume"].pct_change().mean() if len(data) > 1 else 0.0

        # Расчёт моментума
        momentum = (
            (data["close"].iloc[-1] / data["close"].iloc[-20] - 1)
            if len(data) >= 20
            else 0.0
        )

        # Расчёт режима рынка
        regime_score = SimulationUtils.detect_market_regime(
            data["close"].tolist(), volatility, trend_strength
        ).value

        # Расчёт ликвидности
        liquidity_score = (
            (data["volume"].mean() / data["volume"].std())
            if data["volume"].std() > 0
            else 0.0
        )

        # Расчёт настроений (упрощённо)
        sentiment_score = 0.5  # По умолчанию нейтральные настроения

        return {
            "volatility": float(volatility),
            "trend_strength": float(trend_strength),
            "volume_trend": float(volume_trend),
            "momentum": float(momentum),
            "regime_score": float(regime_score),
            "liquidity_score": float(liquidity_score),
            "sentiment_score": float(sentiment_score),
        }

    @staticmethod
    def calculate_backtest_metrics(
        trades: List[SimulationTrade], equity_curve: List[float]
    ) -> BacktestMetricsDict:
        """Расчёт метрик бэктеста."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "recovery_factor": 0.0,
                "expectancy": 0.0,
                "risk_reward_ratio": 0.0,
                "kelly_criterion": 0.0,
            }

        # Базовые метрики
        total_trades = len(trades)
        winning_trades = len([t for t in trades if float(t.pnl) > 0])
        losing_trades = len([t for t in trades if float(t.pnl) < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Расчёт прибыли/убытков
        total_profit = sum(float(t.pnl) for t in trades if float(t.pnl) > 0)
        total_loss = abs(sum(float(t.pnl) for t in trades if float(t.pnl) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Расчёт доходности
        returns = [float(t.pnl) for t in trades]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        return_std = (
            (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            if returns
            else 0.0
        )

        # Коэффициент Шарпа
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0.0

        # Коэффициент Сортино
        downside_returns = [r for r in returns if r < 0]
        downside_std = (
            (sum(r**2 for r in downside_returns) / len(downside_returns)) ** 0.5
            if downside_returns
            else 0.0
        )
        sortino_ratio = avg_return / downside_std if downside_std > 0 else 0.0

        # Максимальная просадка
        max_drawdown = 0.0
        if equity_curve:
            peak = equity_curve[0]
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

        # Коэффициент Кальмара
        calmar_ratio = avg_return / max_drawdown if max_drawdown > 0 else 0.0

        # Фактор восстановления
        recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0.0

        # Ожидаемая доходность
        expectancy = win_rate * (
            total_profit / winning_trades if winning_trades > 0 else 0.0
        ) - (1 - win_rate) * (total_loss / losing_trades if losing_trades > 0 else 0.0)

        # Соотношение риск/доходность
        risk_reward_ratio = (
            total_profit / winning_trades if winning_trades > 0 else 0.0
        ) / (total_loss / losing_trades if losing_trades > 0 else 1.0)

        # Критерий Келли
        kelly_criterion = (
            (win_rate * risk_reward_ratio - (1 - win_rate)) / risk_reward_ratio
            if risk_reward_ratio > 0
            else 0.0
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "recovery_factor": recovery_factor,
            "expectancy": expectancy,
            "risk_reward_ratio": risk_reward_ratio,
            "kelly_criterion": kelly_criterion,
        }
