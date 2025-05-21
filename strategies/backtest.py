from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from .base_strategy import BaseStrategy


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""

    # Параметры бэктеста
    initial_capital: float = 10000.0  # Начальный капитал
    commission: float = 0.001  # Комиссия (0.1%)
    slippage: float = 0.0005  # Проскальзывание (0.05%)
    position_size: float = 1.0  # Размер позиции (1.0 = 100%)

    # Параметры визуализации
    plot_results: bool = True  # Строить графики
    save_plots: bool = True  # Сохранять графики
    plot_dir: str = "plots"  # Директория для графиков

    # Параметры логирования
    log_dir: str = "logs"  # Директория для логов
    save_results: bool = True  # Сохранять результаты


class Backtest:
    """Бэктестер для торговых стратегий"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация бэктестера.

        Args:
            config: Словарь с параметрами бэктеста
        """
        self.config = BacktestConfig(**config) if config else BacktestConfig()
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def run(self, strategy: BaseStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Запуск бэктеста стратегии.

        Args:
            strategy: Стратегия для тестирования
            data: DataFrame с данными

        Returns:
            Dict с результатами бэктеста
        """
        try:
            # Инициализация
            capital = self.config.initial_capital
            position = None
            entry_price = None
            entry_time = None
            trades = []
            equity_curve = [capital]

            # Проходим по данным
            for i in range(len(data)):
                # Генерируем сигнал
                signal = strategy.generate_signal(data.iloc[: i + 1])

                if signal:
                    current_price = data["close"].iloc[i]

                    # Учитываем проскальзывание
                    if signal.direction == "long":
                        current_price *= 1 + self.config.slippage
                    elif signal.direction == "short":
                        current_price *= 1 - self.config.slippage

                    # Учитываем комиссию
                    commission = current_price * self.config.commission

                    if signal.direction in ["long", "short"]:
                        # Закрываем предыдущую позицию
                        if position:
                            exit_price = current_price
                            if position == "long":
                                profit = (exit_price - entry_price) * self.config.position_size
                            else:
                                profit = (entry_price - exit_price) * self.config.position_size

                            profit -= commission * 2  # Комиссия за вход и выход
                            capital += profit

                            trades.append(
                                {
                                    "entry_time": entry_time,
                                    "exit_time": data.index[i],
                                    "position": position,
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "profit": profit,
                                    "capital": capital,
                                }
                            )

                        # Открываем новую позицию
                        position = signal.direction
                        entry_price = current_price
                        entry_time = data.index[i]
                        capital -= commission

                    elif signal.direction == "close" and position:
                        # Закрываем позицию
                        exit_price = current_price
                        if position == "long":
                            profit = (exit_price - entry_price) * self.config.position_size
                        else:
                            profit = (entry_price - exit_price) * self.config.position_size

                        profit -= commission * 2  # Комиссия за вход и выход
                        capital += profit

                        trades.append(
                            {
                                "entry_time": entry_time,
                                "exit_time": data.index[i],
                                "position": position,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "profit": profit,
                                "capital": capital,
                            }
                        )

                        position = None
                        entry_price = None
                        entry_time = None

                equity_curve.append(capital)

            # Закрываем последнюю позицию
            if position:
                current_price = data["close"].iloc[-1]
                if position == "long":
                    profit = (current_price - entry_price) * self.config.position_size
                else:
                    profit = (entry_price - current_price) * self.config.position_size

                profit -= self.config.commission * 2
                capital += profit

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": data.index[-1],
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit": profit,
                        "capital": capital,
                    }
                )

            # Рассчитываем метрики
            results = self._calculate_metrics(trades, equity_curve)

            # Визуализируем результаты
            if self.config.plot_results:
                self._plot_results(data, trades, equity_curve, results)

            return results

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}

    def _calculate_metrics(
        self, trades: List[Dict[str, Any]], equity_curve: List[float]
    ) -> Dict[str, Any]:
        """
        Расчет метрик бэктеста.

        Args:
            trades: Список сделок
            equity_curve: Кривая капитала

        Returns:
            Dict с метриками
        """
        try:
            if not trades:
                return {
                    "n_trades": 0,
                    "total_profit": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "avg_trade": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "max_win": 0.0,
                    "max_loss": 0.0,
                    "avg_holding_time": 0.0,
                }

            # Базовые метрики
            profits = [trade["profit"] for trade in trades]
            returns = pd.Series(profits)
            equity = pd.Series(equity_curve)

            # Время в позиции
            holding_times = [
                (trade["exit_time"] - trade["entry_time"]).total_seconds() / 3600
                for trade in trades
            ]

            # Выигрышные и проигрышные сделки
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            return {
                "n_trades": len(trades),
                "total_profit": sum(profits),
                "win_rate": len(winning_trades) / len(trades),
                "profit_factor": (
                    abs(sum(winning_trades) / sum(losing_trades))
                    if sum(losing_trades) != 0
                    else float("inf")
                ),
                "sharpe_ratio": returns.mean() / returns.std() if returns.std() != 0 else 0.0,
                "sortino_ratio": (
                    returns.mean() / returns[returns < 0].std()
                    if returns[returns < 0].std() != 0
                    else 0.0
                ),
                "max_drawdown": self._calculate_max_drawdown(equity),
                "avg_trade": np.mean(profits),
                "avg_win": np.mean(winning_trades) if winning_trades else 0.0,
                "avg_loss": np.mean(losing_trades) if losing_trades else 0.0,
                "max_win": max(profits),
                "max_loss": min(profits),
                "avg_holding_time": np.mean(holding_times),
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """
        Расчет максимальной просадки.

        Args:
            equity: Series с капиталом

        Returns:
            float: Максимальная просадка
        """
        try:
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            return abs(drawdown.min())

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _plot_results(
        self,
        data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        results: Dict[str, Any],
    ):
        """
        Визуализация результатов бэктеста.

        Args:
            data: DataFrame с данными
            trades: Список сделок
            equity_curve: Кривая капитала
            results: Результаты бэктеста
        """
        try:
            # Создаем фигуру
            fig = plt.figure(figsize=(15, 10))

            # График цены и сделок
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(data.index, data["close"], label="Price", color="blue", alpha=0.5)

            # Отмечаем сделки
            for trade in trades:
                if trade["position"] == "long":
                    ax1.scatter(
                        trade["entry_time"], trade["entry_price"], color="green", marker="^"
                    )
                    ax1.scatter(trade["exit_time"], trade["exit_price"], color="red", marker="v")
                else:
                    ax1.scatter(trade["entry_time"], trade["entry_price"], color="red", marker="v")
                    ax1.scatter(trade["exit_time"], trade["exit_price"], color="green", marker="^")

            ax1.set_title("Price and Trades")
            ax1.legend()

            # График капитала
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(data.index, equity_curve, label="Equity", color="green")
            ax2.set_title("Equity Curve")
            ax2.legend()

            # Добавляем метрики
            metrics_text = f"""
            Total Profit: {results['total_profit']:.2f}
            Win Rate: {results['win_rate']:.2%}
            Profit Factor: {results['profit_factor']:.2f}
            Sharpe Ratio: {results['sharpe_ratio']:.2f}
            Sortino Ratio: {results['sortino_ratio']:.2f}
            Max Drawdown: {results['max_drawdown']:.2%}
            """
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10)

            # Сохраняем график
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")

    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Сохранение результатов бэктеста.

        Args:
            results: Результаты бэктеста
            filename: Имя файла
        """
        try:
            if self.config.save_results:
                pd.DataFrame([results]).to_csv(f"{self.config.log_dir}/{filename}.csv", index=False)

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
