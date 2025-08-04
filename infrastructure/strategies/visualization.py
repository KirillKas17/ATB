# -*- coding: utf-8 -*-
"""Модуль визуализации для стратегий."""
import pandas as pd
from shared.numpy_utils import np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""

    # Параметры визуализации
    plot_dir: str = "plots"  # Директория для графиков
    save_plots: bool = True  # Сохранять графики
    style: str = "seaborn"  # Стиль графиков
    dpi: int = 300  # Разрешение графиков
    figsize: Tuple[int, int] = (15, 10)  # Размер графиков
    # Параметры логирования
    log_dir: str = "logs"  # Директория для логов


class Visualizer:
    """Визуализатор для стратегий"""

    def __init__(self, config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None):
        """
        Инициализация визуализатора.
        Args:
            config: Словарь с параметрами визуализации или объект VisualizationConfig
        """
        if isinstance(config, VisualizationConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = VisualizationConfig(**config)
        else:
            self.config = VisualizationConfig()
        self._setup_logger()
        self._setup_style()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_style(self) -> None:
        """Настройка стиля графиков"""
        plt.style.use(self.config.style)

    def plot_price_and_trades(
        self,
        data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        title: str = "Price and Trades",
    ) -> None:
        """
        Построение графика цены и сделок.
        Args:
            data: DataFrame с данными
            trades: Список сделок
            title: Заголовок графика
        """
        try:
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            # График цены
            plt.plot(data.index, data["close"], label="Price", color="blue", alpha=0.5)
            # Отмечаем сделки
            for trade in trades:
                if trade["position"] == "long":
                    plt.scatter(
                        trade["entry_time"],
                        trade["entry_price"],
                        color="green",
                        marker="^",
                        label="Long Entry",
                    )
                    plt.scatter(
                        trade["exit_time"],
                        trade["exit_price"],
                        color="red",
                        marker="v",
                        label="Long Exit",
                    )
                else:
                    plt.scatter(
                        trade["entry_time"],
                        trade["entry_price"],
                        color="red",
                        marker="v",
                        label="Short Entry",
                    )
                    plt.scatter(
                        trade["exit_time"],
                        trade["exit_price"],
                        color="green",
                        marker="^",
                        label="Short Exit",
                    )
            plt.title(title)
            plt.legend()
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/price_and_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting price and trades: {str(e)}")

    def plot_equity_curve(self, equity_curve: List[float], title: str = "Equity Curve") -> None:
        """
        Построение графика кривой капитала.
        Args:
            equity_curve: Список значений капитала
            title: Заголовок графика
        """
        try:
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            plt.plot(equity_curve, label="Equity", color="green")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")

    def plot_drawdown(self, equity_curve: List[float], title: str = "Drawdown") -> None:
        """
        Построение графика просадок.
        Args:
            equity_curve: Список значений капитала
            title: Заголовок графика
        """
        try:
            equity = pd.Series(equity_curve)
            # Исправление 147-150: используем правильные методы pandas
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            plt.plot(drawdown.to_numpy(), label="Drawdown", color="red")  # Исправление: .to_numpy()
            plt.title(title)
            plt.legend()
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting drawdown: {str(e)}")

    def plot_monthly_returns(
        self, trades: List[Dict[str, Any]], title: str = "Monthly Returns"
    ) -> None:
        """
        Построение графика месячной доходности.
        Args:
            trades: Список сделок
            title: Заголовок графика
        """
        try:
            # Создаем DataFrame с датами и прибылями
            df = pd.DataFrame(trades)
            df["exit_time"] = pd.to_datetime(df["exit_time"])
            df["month"] = df["exit_time"].dt.to_period("M")
            # Группируем по месяцам
            monthly_returns = df.groupby("month")["profit"].sum()
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            monthly_returns.plot(kind="bar", color="green")
            plt.title(title)
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/monthly_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting monthly returns: {str(e)}")

    def plot_trade_distribution(
        self, trades: List[Dict[str, Any]], title: str = "Trade Distribution"
    ) -> None:
        """
        Построение графика распределения сделок.
        Args:
            trades: Список сделок
            title: Заголовок графика
        """
        try:
            profits = [trade["profit"] for trade in trades]
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            sns.histplot(profits, kde=True)
            plt.title(title)
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/trade_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting trade distribution: {str(e)}")

    def plot_metrics(self, results: Dict[str, Any], title: str = "Trading Metrics") -> None:
        """
        Построение графика метрик.
        Args:
            results: Словарь с метриками
            title: Заголовок графика
        """
        try:
            # Выбираем метрики для отображения
            metrics = {
                "Win Rate": results["win_rate"],
                "Profit Factor": results["profit_factor"],
                "Sharpe Ratio": results["sharpe_ratio"],
                "Sortino Ratio": results["sortino_ratio"],
                "Max Drawdown": results["max_drawdown"],
            }
            plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            plt.bar(list(metrics.keys()), list(metrics.values()))
            plt.title(title)
            plt.grid(True)
            if self.config.save_plots:
                plt.savefig(
                    f"{self.config.plot_dir}/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")

    def plot_all(
        self,
        data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        results: Dict[str, Any],
    ) -> None:
        """
        Построение всех графиков.
        Args:
            data: DataFrame с данными
            trades: Список сделок
            equity_curve: Кривая капитала
            results: Результаты торговли
        """
        try:
            self.plot_price_and_trades(data, trades)
            self.plot_equity_curve(equity_curve)
            self.plot_drawdown(equity_curve)
            self.plot_monthly_returns(trades)
            self.plot_trade_distribution(trades)
            self.plot_metrics(results)
        except Exception as e:
            logger.error(f"Error plotting all: {str(e)}")

    def create_report(
        self,
        data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        results: Dict[str, Any],
        filename: str,
    ) -> None:
        """
        Создание отчета с графиками.
        Args:
            data: DataFrame с данными
            trades: Список сделок
            equity_curve: Кривая капитала
            results: Результаты торговли
            filename: Имя файла
        """
        try:
            # Создаем директорию для графиков
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)
            # Строим все графики
            self.plot_all(data, trades, equity_curve, results)
            # Сохраняем результаты
            pd.DataFrame([results]).to_csv(
                f"{self.config.log_dir}/{filename}.csv", index=False
            )
        except Exception as e:
            logger.error(f"Error creating report: {str(e)}")
