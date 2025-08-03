"""
Модуль для визуализации данных и результатов анализа.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series

# Настройка логирования
logger = logging.getLogger(__name__)
# Настройка стилей для matplotlib
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TradingVisualizer:
    """Класс для визуализации торговых данных и результатов анализа"""

    def __init__(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        self.figsize = figsize
        self.colors = {
            "buy": "#2E8B57",
            "sell": "#DC143C",
            "hold": "#FFD700",
            "profit": "#32CD32",
            "loss": "#FF6347",
            "neutral": "#87CEEB",
        }

    def plot_candlestick_with_signals(
        self, data: DataFrame, signals: List[Dict], save_path: Optional[str] = None
    ) -> None:
        """Построение графика свечей с сигналами"""
        try:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize, gridspec_kw={"height_ratios": [3, 1]}
            )
            # График свечей
            self._plot_candlesticks(ax1, data)
            self._plot_signals(ax1, signals)
            # График объема
            self._plot_volume(ax2, data)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при построении графика свечей: {e}")

    def plot_technical_indicators(
        self,
        data: DataFrame,
        indicators: Dict[str, Series],
        save_path: Optional[str] = None,
    ) -> None:
        """Построение графика технических индикаторов"""
        try:
            fig, axes = plt.subplots(4, 1, figsize=self.figsize)
            # RSI
            if "rsi" in indicators:
                axes[0].plot(indicators["rsi"].values, label="RSI", color="purple")
                axes[0].axhline(y=70, color="r", linestyle="--", alpha=0.7)
                axes[0].axhline(y=30, color="g", linestyle="--", alpha=0.7)
                axes[0].set_ylabel("RSI")
                axes[0].legend()
                axes[0].grid(True)
            # MACD
            if "macd" in indicators and "macd_signal" in indicators:
                axes[1].plot(indicators["macd"].values, label="MACD", color="blue")
                axes[1].plot(indicators["macd_signal"].values, label="Signal", color="red")
                axes[1].bar(
                    range(len(indicators["macd_histogram"])),
                    indicators["macd_histogram"].values.astype(float),
                    label="Histogram",
                    alpha=0.3,
                )
                axes[1].set_ylabel("MACD")
                axes[1].legend()
                axes[1].grid(True)
            # Bollinger Bands
            if "bb_upper" in indicators and "bb_lower" in indicators:
                axes[2].plot(data['close'].values, label="Price", color="black")
                axes[2].plot(
                    indicators["bb_upper"].values, label="Upper BB", color="red", alpha=0.7
                )
                axes[2].plot(
                    indicators["bb_lower"].values, label="Lower BB", color="red", alpha=0.7
                )
                axes[2].fill_between(
                    range(len(data)),
                    indicators["bb_upper"].values,
                    indicators["bb_lower"].values,
                    alpha=0.1,
                    color="red",
                )
                axes[2].set_ylabel("Price")
                axes[2].legend()
                axes[2].grid(True)
            # Volume
            axes[3].bar(range(len(data)), data['volume'].values, alpha=0.7, color="gray")
            axes[3].set_ylabel("Volume")
            axes[3].grid(True)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при построении графика индикаторов: {e}")

    def plot_performance_metrics(
        self, metrics: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """Построение графика метрик производительности"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            metrics_list = list(metrics.keys())
            for i, metric in enumerate(metrics_list):
                if i < 4:  # Максимум 4 графика
                    row, col = i // 2, i % 2
                    axes[row, col].plot(metrics[metric], label=metric.upper())
                    axes[row, col].set_title(f"{metric.upper()} Over Time")
                    axes[row, col].set_xlabel("Time")
                    axes[row, col].set_ylabel(metric.upper())
                    axes[row, col].legend()
                    axes[row, col].grid(True)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при построении графика метрик: {e}")

    def plot_decision_heatmap(
        self, decisions: List[Dict], save_path: Optional[str] = None
    ) -> None:
        """Построение тепловой карты решений"""
        try:
            # Подготовка данных
            decision_matrix = []
            for decision in decisions:
                row = [
                    decision.get("confidence", 0),
                    decision.get("risk_score", 0),
                    decision.get("signal_strength", 0),
                    decision.get("market_regime_score", 0),
                ]
                decision_matrix.append(row)
            if not decision_matrix:
                logger.warning("Нет данных для построения тепловой карты")
                return
            # Создание тепловой карты
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                decision_matrix,
                annot=True,
                cmap="RdYlGn",
                center=0.5,
                xticklabels=["Confidence", "Risk", "Signal", "Regime"],
                yticklabels=range(len(decision_matrix)),
                ax=ax,
            )
            ax.set_title("Decision Heatmap")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Decisions")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при построении тепловой карты: {e}")

    def plot_feature_importance(
        self, feature_importance: Dict[str, float], save_path: Optional[str] = None
    ) -> None:
        """Построение графика важности признаков"""
        try:
            # Сортировка по важности
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            features, importance = zip(*sorted_features[:20])  # Топ-20 признаков
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(features)), importance, color="skyblue")
            # Добавление значений на столбцы
            for i, (bar, imp) in enumerate(zip(bars, importance)):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{imp:.3f}",
                    va="center",
                )
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top 20 Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при построении графика важности признаков: {e}")

    def _plot_candlesticks(self, ax: plt.Axes, data: DataFrame) -> None:
        """Построение графика свечей"""
        try:
            width = 0.6
            width2 = width * 0.8
            up = data[data['close'] >= data['open']]
            down = data[data['close'] < data['open']]
            # Бычьи свечи
            ax.bar(
                up.index,
                (up['close'] - up['open']).values.astype(float),
                width,
                bottom=up['open'].values.astype(float),
                color=self.colors["buy"],
                alpha=0.7,
            )
            ax.bar(
                up.index,
                (up['high'] - up['close']).values.astype(float),
                width2,
                bottom=up['close'].values.astype(float),
                color=self.colors["buy"],
            )
            ax.bar(
                up.index,
                (up['low'] - up['open']).values.astype(float),
                width2,
                bottom=up['open'].values.astype(float),
                color=self.colors["buy"],
            )
            # Медвежьи свечи
            ax.bar(
                down.index,
                (down['close'] - down['open']).values.astype(float),
                width,
                bottom=down['open'].values.astype(float),
                color=self.colors["sell"],
                alpha=0.7,
            )
            ax.bar(
                down.index,
                (down['high'] - down['open']).values.astype(float),
                width2,
                bottom=down['open'].values.astype(float),
                color=self.colors["sell"],
            )
            ax.bar(
                down.index,
                (down['low'] - down['close']).values.astype(float),
                width2,
                bottom=down['close'].values.astype(float),
                color=self.colors["sell"],
            )
            ax.set_title("Candlestick Chart")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Ошибка при построении свечей: {e}")

    def _plot_signals(self, ax: plt.Axes, signals: List[Dict]) -> None:
        """Добавление сигналов на график"""
        try:
            for signal in signals:
                if "timestamp" in signal and "action" in signal:
                    # Определение цвета сигнала
                    if signal["action"] == "buy":
                        color = self.colors["buy"]
                        marker = "^"
                    elif signal["action"] == "sell":
                        color = self.colors["sell"]
                        marker = "v"
                    else:
                        color = self.colors["hold"]
                        marker = "o"
                    # Добавление маркера сигнала
                    ax.scatter(
                        signal["timestamp"],
                        signal.get("price", 0),
                        color=color,
                        marker=marker,
                        s=100,
                        zorder=5,
                    )
        except Exception as e:
            logger.error(f"Ошибка при добавлении сигналов: {e}")

    def _plot_volume(self, ax: plt.Axes, data: DataFrame) -> None:
        """Построение графика объема"""
        try:
            ax.bar(range(len(data)), data['volume'].values.astype(float), alpha=0.7, color="gray")
            ax.set_ylabel("Volume")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Ошибка при построении графика объема: {e}")

    def create_dashboard(
        self,
        data: DataFrame,
        signals: List[Dict],
        indicators: Dict[str, Series],
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Создание комплексной панели мониторинга"""
        try:
            fig = plt.figure(figsize=(20, 16))
            # Создание сетки
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            # График свечей
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            self._plot_candlesticks(ax1, data)
            self._plot_signals(ax1, signals)
            # Объем
            ax2 = fig.add_subplot(gs[2, 0:2])
            self._plot_volume(ax2, data)
            # RSI
            ax3 = fig.add_subplot(gs[0, 2])
            if "rsi" in indicators:
                ax3.plot(indicators["rsi"].to_numpy(), color="purple")  # Исправление 334: используем to_numpy()
                ax3.axhline(y=70, color="r", linestyle="--", alpha=0.7)
                ax3.axhline(y=30, color="g", linestyle="--", alpha=0.7)
                ax3.set_title("RSI")
                ax3.grid(True)
            # MACD
            ax4 = fig.add_subplot(gs[1, 2])
            if "macd" in indicators and "macd_signal" in indicators:
                ax4.plot(indicators["macd"].to_numpy(), label="MACD", color="blue")  # Исправление 342: используем to_numpy()
                ax4.plot(indicators["macd_signal"].to_numpy(), label="Signal", color="red")  # Исправление 343: используем to_numpy()
                ax4.set_title("MACD")
                ax4.legend()
                ax4.grid(True)
            # Метрики производительности
            ax5 = fig.add_subplot(gs[3, :])
            for metric, values in metrics.items():
                if values:
                    ax5.plot(values, label=metric.upper(), alpha=0.7)
            ax5.set_title("Performance Metrics")
            ax5.legend()
            ax5.grid(True)
            plt.suptitle("Trading Dashboard", fontsize=16)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Ошибка при создании панели мониторинга: {e}")
