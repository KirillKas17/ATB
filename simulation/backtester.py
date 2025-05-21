import asyncio
import json
import signal
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import talib
import yaml
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from core.controllers.trading_controller import TradingController
from core.models import MarketData, Order, Position, Trade
from core.strategy import Signal
from exchange.bybit_client import BybitClient
from ml.dataset_manager import DatasetManager
from ml.live_adaptation import LiveAdaptation
from ml.model_selector import ModelSelector
from ml.pattern_discovery import PatternDiscovery
from strategies.base_strategy import BaseStrategy
from utils.data_loader import DataLoader
from utils.logger import setup_logger
from utils.market_regime import MarketRegime

logger = setup_logger(__name__)


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""

    start_date: datetime
    end_date: datetime
    initial_balance: float
    position_size: float
    max_positions: int
    stop_loss: float
    take_profit: float
    commission: float
    slippage: float
    data_dir: Path = Path("data/backtest")
    models_dir: Path = Path("models/backtest")
    log_dir: str = "logs/backtest"
    backup_dir: str = "backups/backtest"
    metrics_window: int = 100
    min_samples: int = 1000
    max_samples: int = 100000
    num_threads: int = 4
    cache_size: int = 1000
    save_interval: int = 1000
    validation_split: float = 0.2
    random_seed: int = 42
    use_patterns: bool = True
    use_adaptation: bool = True
    use_regime_detection: bool = True

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "BacktestConfig":
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            return cls(
                start_date=datetime.fromisoformat(config["backtest"]["start_date"]),
                end_date=datetime.fromisoformat(config["backtest"]["end_date"]),
                initial_balance=config["backtest"]["initial_balance"],
                position_size=config["backtest"]["position_size"],
                max_positions=config["backtest"]["max_positions"],
                stop_loss=config["backtest"]["stop_loss"],
                take_profit=config["backtest"]["take_profit"],
                commission=config["backtest"]["commission"],
                slippage=config["backtest"]["slippage"],
                data_dir=Path(config["backtest"].get("data_dir", "data/backtest")),
                models_dir=Path(config["backtest"].get("models_dir", "models/backtest")),
                log_dir=config["backtest"].get("log_dir", "logs/backtest"),
                backup_dir=config["backtest"].get("backup_dir", "backups/backtest"),
                metrics_window=config["backtest"].get("metrics_window", 100),
                min_samples=config["backtest"].get("min_samples", 1000),
                max_samples=config["backtest"].get("max_samples", 100000),
                num_threads=config["backtest"].get("num_threads", 4),
                cache_size=config["backtest"].get("cache_size", 1000),
                save_interval=config["backtest"].get("save_interval", 1000),
                validation_split=config["backtest"].get("validation_split", 0.2),
                random_seed=config["backtest"].get("random_seed", 42),
                use_patterns=config["backtest"].get("use_patterns", True),
                use_adaptation=config["backtest"].get("use_adaptation", True),
                use_regime_detection=config["backtest"].get("use_regime_detection", True),
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise


@dataclass
class BacktestMetrics:
    """Метрики бэктеста"""

    start_time: datetime
    end_time: datetime
    simulation_time: float
    memory_usage: float
    cpu_usage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_profit: float
    total_loss: float
    net_profit: float
    success: bool
    error: Optional[str] = None
    validation_metrics: Optional[Dict] = None
    model_metrics: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    regime_metrics: Optional[Dict] = None
    pattern_metrics: Optional[Dict] = None
    adaptation_metrics: Optional[Dict] = None


class Backtester:
    """Бэктестер"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Инициализация бэктестера"""
        self.config = config or BacktestConfig.from_yaml()
        self.metrics_history = []
        self._sim_lock = asyncio.Lock()
        self._start_time = None
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_threads)

        # Установка случайного зерна
        np.random.seed(self.config.random_seed)

        # Создание директорий
        for dir_path in [
            self.config.data_dir,
            self.config.models_dir,
            self.config.log_dir,
            self.config.backup_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Инициализация компонентов
        self.order_manager = OrderManager()
        self.model_selector = ModelSelector()
        self.pattern_discovery = PatternDiscovery()
        self.live_adaptation = LiveAdaptation()

        # Инициализация состояния
        self.state = {
            "balance": self.config.initial_balance,
            "positions": [],
            "trades": [],
            "equity": [self.config.initial_balance],
            "drawdown": [0.0],
            "regime": "normal",
            "patterns": [],
            "adaptation_state": {},
        }

        # Инициализация кэша
        self._init_cache()

    def _init_cache(self):
        """Инициализация кэша"""
        self._price_cache = {}
        self._volume_cache = {}
        self._regime_cache = {}
        self._pattern_cache = {}
        self._adaptation_cache = {}

    @lru_cache(maxsize=1000)
    def _get_cached_price(self, timestamp: int) -> float:
        """Получение кэшированной цены"""
        return self._price_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_volume(self, timestamp: int) -> float:
        """Получение кэшированного объема"""
        return self._volume_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_regime(self, timestamp: int) -> str:
        """Получение кэшированного режима"""
        return self._regime_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_pattern(self, timestamp: int) -> List:
        """Получение кэшированных паттернов"""
        return self._pattern_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_adaptation(self, timestamp: int) -> Dict:
        """Получение кэшированного состояния адаптации"""
        return self._adaptation_cache.get(timestamp)

    def _get_system_metrics(self) -> Dict:
        """Получение системных метрик"""
        return {
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
            "thread_count": len(self._executor._threads),
            "cache_size": len(self._price_cache)
            + len(self._volume_cache)
            + len(self._regime_cache)
            + len(self._pattern_cache)
            + len(self._adaptation_cache),
        }

    def _handle_shutdown(self, signum, frame):
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения")
        self._running = False
        self._executor.shutdown(wait=False)

    def _calculate_position_size(self, price: float) -> float:
        """Расчет размера позиции"""
        try:
            # Базовый размер
            size = self.config.position_size

            # Модификация по режиму
            if self.state["regime"] == "trend":
                size *= 1.5
            elif self.state["regime"] == "mean_reversion":
                size *= 0.8
            elif self.state["regime"] == "volatile":
                size *= 0.5

            # Модификация по паттернам
            if self.config.use_patterns and self.state["patterns"]:
                pattern_confidence = max(p["confidence"] for p in self.state["patterns"])
                size *= 1 + pattern_confidence

            # Модификация по адаптации
            if self.config.use_adaptation and self.state["adaptation_state"]:
                adaptation_factor = self.state["adaptation_state"].get("position_size_factor", 1.0)
                size *= adaptation_factor

            # Ограничение размера
            size = min(size, self.config.max_positions)

            return size

        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {str(e)}")
            return self.config.position_size

    def _calculate_stop_loss(self, price: float, side: str) -> float:
        """Расчет стоп-лосса"""
        try:
            # Базовый стоп-лосс
            stop_loss = self.config.stop_loss

            # Модификация по режиму
            if self.state["regime"] == "trend":
                stop_loss *= 1.5
            elif self.state["regime"] == "mean_reversion":
                stop_loss *= 0.8
            elif self.state["regime"] == "volatile":
                stop_loss *= 2.0

            # Модификация по паттернам
            if self.config.use_patterns and self.state["patterns"]:
                pattern_volatility = max(p["volatility"] for p in self.state["patterns"])
                stop_loss *= 1 + pattern_volatility

            # Модификация по адаптации
            if self.config.use_adaptation and self.state["adaptation_state"]:
                adaptation_factor = self.state["adaptation_state"].get("stop_loss_factor", 1.0)
                stop_loss *= adaptation_factor

            # Применение к цене
            if side == "long":
                return price * (1 - stop_loss)
            else:
                return price * (1 + stop_loss)

        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса: {str(e)}")
            return (
                price * (1 - self.config.stop_loss)
                if side == "long"
                else price * (1 + self.config.stop_loss)
            )

    def _calculate_take_profit(self, price: float, side: str) -> float:
        """Расчет тейк-профита"""
        try:
            # Базовый тейк-профит
            take_profit = self.config.take_profit

            # Модификация по режиму
            if self.state["regime"] == "trend":
                take_profit *= 2.0
            elif self.state["regime"] == "mean_reversion":
                take_profit *= 0.5
            elif self.state["regime"] == "volatile":
                take_profit *= 1.5

            # Модификация по паттернам
            if self.config.use_patterns and self.state["patterns"]:
                pattern_potential = max(p["potential"] for p in self.state["patterns"])
                take_profit *= 1 + pattern_potential

            # Модификация по адаптации
            if self.config.use_adaptation and self.state["adaptation_state"]:
                adaptation_factor = self.state["adaptation_state"].get("take_profit_factor", 1.0)
                take_profit *= adaptation_factor

            # Применение к цене
            if side == "long":
                return price * (1 + take_profit)
            else:
                return price * (1 - take_profit)

        except Exception as e:
            logger.error(f"Ошибка расчета тейк-профита: {str(e)}")
            return (
                price * (1 + self.config.take_profit)
                if side == "long"
                else price * (1 - self.config.take_profit)
            )

    def _calculate_commission(self, size: float, price: float) -> float:
        """Расчет комиссии"""
        try:
            # Базовая комиссия
            commission = size * price * self.config.commission

            # Модификация по режиму
            if self.state["regime"] == "volatile":
                commission *= 1.5

            # Модификация по адаптации
            if self.config.use_adaptation and self.state["adaptation_state"]:
                adaptation_factor = self.state["adaptation_state"].get("commission_factor", 1.0)
                commission *= adaptation_factor

            return commission

        except Exception as e:
            logger.error(f"Ошибка расчета комиссии: {str(e)}")
            return size * price * self.config.commission

    def _calculate_slippage(self, size: float, price: float) -> float:
        """Расчет проскальзывания"""
        try:
            # Базовое проскальзывание
            slippage = size * price * self.config.slippage

            # Модификация по режиму
            if self.state["regime"] == "volatile":
                slippage *= 2.0
            elif self.state["regime"] == "trend":
                slippage *= 1.5

            # Модификация по адаптации
            if self.config.use_adaptation and self.state["adaptation_state"]:
                adaptation_factor = self.state["adaptation_state"].get("slippage_factor", 1.0)
                slippage *= adaptation_factor

            return slippage

        except Exception as e:
            logger.error(f"Ошибка расчета проскальзывания: {str(e)}")
            return size * price * self.config.slippage

    def _update_state(self, data: pd.DataFrame, i: int):
        """Обновление состояния"""
        try:
            # Обновление режима
            if self.config.use_regime_detection:
                self.state["regime"] = data["regime"].iloc[i]

            # Обновление паттернов
            if self.config.use_patterns:
                self.state["patterns"] = self.pattern_discovery.find_patterns(data.iloc[: i + 1])

            # Обновление адаптации
            if self.config.use_adaptation:
                self.state["adaptation_state"] = self.live_adaptation.update_state(
                    data.iloc[: i + 1], self.state["trades"]
                )

        except Exception as e:
            logger.error(f"Ошибка обновления состояния: {str(e)}")

    def _update_metrics(self, data: pd.DataFrame, i: int):
        """Обновление метрик"""
        try:
            # Расчет метрик
            metrics = {
                "total_trades": len(self.state["trades"]),
                "winning_trades": len([t for t in self.state["trades"] if t["profit"] > 0]),
                "losing_trades": len([t for t in self.state["trades"] if t["profit"] < 0]),
                "win_rate": (
                    len([t for t in self.state["trades"] if t["profit"] > 0])
                    / len(self.state["trades"])
                    if self.state["trades"]
                    else 0
                ),
                "profit_factor": (
                    abs(
                        sum(t["profit"] for t in self.state["trades"] if t["profit"] > 0)
                        / sum(t["profit"] for t in self.state["trades"] if t["profit"] < 0)
                    )
                    if any(t["profit"] < 0 for t in self.state["trades"])
                    else float("inf")
                ),
                "sharpe_ratio": self._calculate_sharpe_ratio(),
                "max_drawdown": max(self.state["drawdown"]),
                "total_profit": sum(t["profit"] for t in self.state["trades"] if t["profit"] > 0),
                "total_loss": sum(t["profit"] for t in self.state["trades"] if t["profit"] < 0),
                "net_profit": sum(t["profit"] for t in self.state["trades"]),
            }

            # Добавление метрик в историю
            self.metrics_history.append(metrics)

            # Сохранение метрик
            if i % self.config.save_interval == 0:
                metrics_file = self.config.data_dir / "backtest_metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {str(e)}")

    def _calculate_sharpe_ratio(self) -> float:
        """Расчет коэффициента Шарпа"""
        try:
            if not self.state["trades"]:
                return 0.0

            # Расчет доходностей
            returns = [t["profit"] / t["entry_price"] for t in self.state["trades"]]

            # Расчет коэффициента Шарпа
            if len(returns) < 2:
                return 0.0

            return np.mean(returns) / np.std(returns) * np.sqrt(252)

        except Exception as e:
            logger.error(f"Ошибка расчета коэффициента Шарпа: {str(e)}")
            return 0.0

    def _plot_backtest_analysis(self, data: pd.DataFrame, save_path: Path):
        """Построение анализа бэктеста"""
        try:
            # Создание фигуры
            fig = plt.figure(figsize=(20, 15))

            # График цены
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(data.index, data["close"])
            ax1.set_title("Price")
            ax1.grid(True)

            # График эквити
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(data.index, self.state["equity"])
            ax2.set_title("Equity")
            ax2.grid(True)

            # График просадки
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(data.index, self.state["drawdown"])
            ax3.set_title("Drawdown")
            ax3.grid(True)

            # График сделок
            ax4 = plt.subplot(3, 2, 4)
            for trade in self.state["trades"]:
                if trade["side"] == "long":
                    ax4.scatter(
                        trade["entry_time"], trade["entry_price"], color="green", marker="^"
                    )
                    ax4.scatter(trade["exit_time"], trade["exit_price"], color="red", marker="v")
                else:
                    ax4.scatter(trade["entry_time"], trade["entry_price"], color="red", marker="v")
                    ax4.scatter(trade["exit_time"], trade["exit_price"], color="green", marker="^")
            ax4.set_title("Trades")
            ax4.grid(True)

            # График режимов
            ax5 = plt.subplot(3, 2, 5)
            sns.scatterplot(data=data, x="volatility", y="trend", hue="regime", ax=ax5)
            ax5.set_title("Market Regimes")
            ax5.grid(True)

            # График метрик
            ax6 = plt.subplot(3, 2, 6)
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_df.plot(ax=ax6)
            ax6.set_title("Metrics")
            ax6.grid(True)

            # Сохранение
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            logger.error(f"Ошибка построения анализа: {str(e)}")

    async def run_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Запуск бэктеста"""
        try:
            # Проверка наличия данных
            if data is None or len(data) < self.config.min_samples:
                raise ValueError(
                    f"Недостаточно данных для бэктеста. Требуется минимум {self.config.min_samples} свечей"
                )

            # Проверка наличия необходимых колонок
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")

            # Инициализация состояния
            self._start_time = datetime.now()
            self._running = True

            # Регистрация обработчика сигналов
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            # Основной цикл бэктеста
            for i in range(len(data)):
                if not self._running:
                    break

                # Обновление состояния
                self._update_state(data, i)

                # Обновление метрик
                self._update_metrics(data, i)

                # Сохранение промежуточных результатов
                if i % self.config.save_interval == 0:
                    await self._save_backup(data, i)

            # Расчет финальных метрик
            metrics = self._calculate_final_metrics(data)

            # Сохранение результатов
            await self._save_results(data, metrics)

            return data

        except Exception as e:
            logger.error(f"Ошибка при выполнении бэктеста: {str(e)}")
            raise
        finally:
            self._running = False
            self._executor.shutdown()


async def main():
    """Основная функция"""
    try:
        # Инициализация бэктестера
        backtester = Backtester()

        # Загрузка данных
        data = pd.read_csv(
            backtester.config.data_dir / "market_data.csv", index_col=0, parse_dates=True
        )

        # Запуск бэктеста
        data = await backtester.run_backtest(data)

        logger.info("Бэктест успешно завершен")

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add("logs/backtest_{time}.log", rotation="1 day", retention="7 days", level="INFO")

    # Запуск асинхронного main
    asyncio.run(main())
