import asyncio
import json
import signal
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiofiles
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import seaborn as sns
import shap
import talib
import yaml
from loguru import logger
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from ml.live_adaptation import LiveAdaptation
from ml.model_selector import ModelSelector
from ml.pattern_discovery import PatternDiscovery
from utils.market_regime import MarketRegime


@dataclass
class ExplainerConfig:
    """Конфигурация анализатора"""

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
    use_shap: bool = True
    use_lime: bool = True
    use_permutation: bool = True
    use_correlation: bool = True
    use_causality: bool = True

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "ExplainerConfig":
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            return cls(
                data_dir=Path(config["explainer"].get("data_dir", "data/backtest")),
                models_dir=Path(config["explainer"].get("models_dir", "models/backtest")),
                log_dir=config["explainer"].get("log_dir", "logs/backtest"),
                backup_dir=config["explainer"].get("backup_dir", "backups/backtest"),
                metrics_window=config["explainer"].get("metrics_window", 100),
                min_samples=config["explainer"].get("min_samples", 1000),
                max_samples=config["explainer"].get("max_samples", 100000),
                num_threads=config["explainer"].get("num_threads", 4),
                cache_size=config["explainer"].get("cache_size", 1000),
                save_interval=config["explainer"].get("save_interval", 1000),
                validation_split=config["explainer"].get("validation_split", 0.2),
                random_seed=config["explainer"].get("random_seed", 42),
                use_patterns=config["explainer"].get("use_patterns", True),
                use_adaptation=config["explainer"].get("use_adaptation", True),
                use_regime_detection=config["explainer"].get("use_regime_detection", True),
                use_shap=config["explainer"].get("use_shap", True),
                use_lime=config["explainer"].get("use_lime", True),
                use_permutation=config["explainer"].get("use_permutation", True),
                use_correlation=config["explainer"].get("use_correlation", True),
                use_causality=config["explainer"].get("use_causality", True),
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise


@dataclass
class ExplainerMetrics:
    """Метрики анализатора"""

    start_time: datetime
    end_time: datetime
    analysis_time: float
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
    shap_metrics: Optional[Dict] = None
    lime_metrics: Optional[Dict] = None
    permutation_metrics: Optional[Dict] = None
    correlation_metrics: Optional[Dict] = None
    causality_metrics: Optional[Dict] = None


class TradeAnalyzer:
    """Анализатор сделок"""

    def __init__(self):
        self.feature_importance = {}
        self.shap_values = None
        self.explainer = None

    def analyze_trade(self, trade: Dict, market_data: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Анализ отдельной сделки.

        Args:
            trade: Данные сделки
            market_data: Рыночные данные
            indicators: Индикаторы

        Returns:
            Dict: Результаты анализа
        """
        try:
            # Получение данных на момент входа
            entry_time = trade["timestamp"]
            entry_data = market_data.loc[entry_time]

            # Анализ условий входа
            entry_conditions = self._analyze_entry_conditions(
                trade=trade, market_data=entry_data, indicators=indicators
            )

            # Анализ рыночного режима
            regime_analysis = self._analyze_market_regime(
                market_data=market_data, entry_time=entry_time
            )

            # Анализ факторов успеха/неудачи
            success_factors = self._analyze_success_factors(
                trade=trade, entry_conditions=entry_conditions, regime_analysis=regime_analysis
            )

            return {
                "entry_conditions": entry_conditions,
                "regime_analysis": regime_analysis,
                "success_factors": success_factors,
            }

        except Exception as e:
            logger.error(f"Error analyzing trade: {str(e)}")
            raise

    def _analyze_entry_conditions(
        self, trade: Dict, market_data: pd.Series, indicators: Dict
    ) -> Dict:
        """Анализ условий входа"""
        conditions = {"price_action": {}, "indicators": {}, "volume": {}, "volatility": {}}

        # Анализ price action
        conditions["price_action"] = {
            "trend": self._detect_trend(market_data),
            "support_resistance": self._check_support_resistance(market_data),
            "candlestick_pattern": self._detect_candlestick_pattern(market_data),
        }

        # Анализ индикаторов
        for name, value in indicators.items():
            conditions["indicators"][name] = {
                "value": value,
                "signal": self._analyze_indicator_signal(name, value, market_data),
            }

        # Анализ объема
        conditions["volume"] = {
            "relative_volume": market_data["volume"] / market_data["volume"].mean(),
            "volume_trend": self._analyze_volume_trend(market_data),
        }

        # Анализ волатильности
        conditions["volatility"] = {
            "current": market_data["volatility"],
            "trend": self._analyze_volatility_trend(market_data),
        }

        return conditions

    def _analyze_market_regime(self, market_data: pd.DataFrame, entry_time: datetime) -> Dict:
        """Анализ рыночного режима"""
        # Получение данных до входа
        pre_entry_data = market_data[market_data.index < entry_time]

        return {
            "regime": pre_entry_data["regime"].iloc[-1],
            "regime_strength": self._calculate_regime_strength(pre_entry_data),
            "regime_duration": self._calculate_regime_duration(pre_entry_data),
            "regime_transition": self._detect_regime_transition(pre_entry_data),
        }

    def _analyze_success_factors(
        self, trade: Dict, entry_conditions: Dict, regime_analysis: Dict
    ) -> Dict:
        """Анализ факторов успеха/неудачи"""
        # Расчет прибыли/убытка
        pnl = trade["capital"] - trade["initial_capital"]
        is_successful = pnl > 0

        # Определение ключевых факторов
        factors = {"positive": [], "negative": []}

        # Анализ условий входа
        if entry_conditions["price_action"]["trend"] == "strong":
            factors["positive" if is_successful else "negative"].append("Strong trend")

        if entry_conditions["volume"]["relative_volume"] > 1.5:
            factors["positive" if is_successful else "negative"].append("High volume")

        # Анализ рыночного режима
        if regime_analysis["regime_strength"] > 0.7:
            factors["positive" if is_successful else "negative"].append("Strong regime")

        if regime_analysis["regime_transition"]:
            factors["negative"].append("Regime transition")

        return factors


class BacktestExplainer:
    """Анализатор бэктеста"""

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """Инициализация анализатора"""
        self.config = config or ExplainerConfig.from_yaml()
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
        self.model_selector = ModelSelector()
        self.pattern_discovery = PatternDiscovery()
        self.live_adaptation = LiveAdaptation()

        # Инициализация кэша
        self._init_cache()

    def _init_cache(self):
        """Инициализация кэша"""
        self._price_cache = {}
        self._volume_cache = {}
        self._regime_cache = {}
        self._pattern_cache = {}
        self._adaptation_cache = {}
        self._shap_cache = {}
        self._lime_cache = {}
        self._permutation_cache = {}
        self._correlation_cache = {}
        self._causality_cache = {}

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

    @lru_cache(maxsize=1000)
    def _get_cached_shap(self, timestamp: int) -> Dict:
        """Получение кэшированных значений SHAP"""
        return self._shap_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_lime(self, timestamp: int) -> Dict:
        """Получение кэшированных значений LIME"""
        return self._lime_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_permutation(self, timestamp: int) -> Dict:
        """Получение кэшированных значений перестановочного теста"""
        return self._permutation_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_correlation(self, timestamp: int) -> Dict:
        """Получение кэшированных значений корреляции"""
        return self._correlation_cache.get(timestamp)

    @lru_cache(maxsize=1000)
    def _get_cached_causality(self, timestamp: int) -> Dict:
        """Получение кэшированных значений причинности"""
        return self._causality_cache.get(timestamp)

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
            + len(self._adaptation_cache)
            + len(self._shap_cache)
            + len(self._lime_cache)
            + len(self._permutation_cache)
            + len(self._correlation_cache)
            + len(self._causality_cache),
        }

    def _handle_shutdown(self, signum, frame):
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения")
        self._running = False
        self._executor.shutdown(wait=False)

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Расчет метрик сделок"""
        try:
            if not trades:
                return {}

            # Базовые метрики
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t["profit"] > 0])
            losing_trades = len([t for t in trades if t["profit"] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Прибыль
            total_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
            total_loss = sum(t["profit"] for t in trades if t["profit"] < 0)
            net_profit = total_profit + total_loss
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float("inf")

            # Риск
            returns = [t["profit"] / t["entry_price"] for t in trades]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            )

            # Просадка
            equity = np.cumsum([t["profit"] for t in trades])
            max_equity = np.maximum.accumulate(equity)
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": net_profit,
            }

        except Exception as e:
            logger.error(f"Ошибка расчета метрик сделок: {str(e)}")
            return {}

    def _calculate_regime_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик режимов"""
        try:
            if "regime" not in data.columns:
                return {}

            # Статистика по режимам
            regime_stats = data.groupby("regime").agg(
                {
                    "close": ["mean", "std", "min", "max"],
                    "volume": ["mean", "std", "min", "max"],
                    "volatility": ["mean", "std", "min", "max"],
                }
            )

            # Длительность режимов
            regime_durations = data.groupby("regime").size()

            # Переходы между режимами
            regime_transitions = pd.crosstab(data["regime"].shift(), data["regime"])

            return {
                "regime_stats": regime_stats.to_dict(),
                "regime_durations": regime_durations.to_dict(),
                "regime_transitions": regime_transitions.to_dict(),
            }

        except Exception as e:
            logger.error(f"Ошибка расчета метрик режимов: {str(e)}")
            return {}

    def _calculate_pattern_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик паттернов"""
        try:
            if not self.config.use_patterns:
                return {}

            # Поиск паттернов
            patterns = self.pattern_discovery.find_patterns(data)

            # Статистика по паттернам
            pattern_stats = {
                "total_patterns": len(patterns),
                "pattern_types": {},
                "pattern_accuracy": {},
                "pattern_profitability": {},
            }

            for pattern in patterns:
                pattern_type = pattern["type"]
                if pattern_type not in pattern_stats["pattern_types"]:
                    pattern_stats["pattern_types"][pattern_type] = 0
                pattern_stats["pattern_types"][pattern_type] += 1

                if "accuracy" in pattern:
                    if pattern_type not in pattern_stats["pattern_accuracy"]:
                        pattern_stats["pattern_accuracy"][pattern_type] = []
                    pattern_stats["pattern_accuracy"][pattern_type].append(pattern["accuracy"])

                if "profitability" in pattern:
                    if pattern_type not in pattern_stats["pattern_profitability"]:
                        pattern_stats["pattern_profitability"][pattern_type] = []
                    pattern_stats["pattern_profitability"][pattern_type].append(
                        pattern["profitability"]
                    )

            # Усреднение метрик
            for pattern_type in pattern_stats["pattern_accuracy"]:
                pattern_stats["pattern_accuracy"][pattern_type] = np.mean(
                    pattern_stats["pattern_accuracy"][pattern_type]
                )

            for pattern_type in pattern_stats["pattern_profitability"]:
                pattern_stats["pattern_profitability"][pattern_type] = np.mean(
                    pattern_stats["pattern_profitability"][pattern_type]
                )

            return pattern_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик паттернов: {str(e)}")
            return {}

    def _calculate_adaptation_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик адаптации"""
        try:
            if not self.config.use_adaptation:
                return {}

            # Обновление состояния адаптации
            adaptation_state = self.live_adaptation.update_state(data)

            # Статистика по адаптации
            adaptation_stats = {
                "adaptation_factors": adaptation_state.get("factors", {}),
                "adaptation_metrics": adaptation_state.get("metrics", {}),
                "adaptation_performance": adaptation_state.get("performance", {}),
            }

            return adaptation_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик адаптации: {str(e)}")
            return {}

    def _calculate_shap_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик SHAP"""
        try:
            if not self.config.use_shap:
                return {}

            # Расчет значений SHAP
            shap_values = self.model_selector.explain_predictions(data)

            # Статистика по SHAP
            shap_stats = {
                "feature_importance": shap_values.get("feature_importance", {}),
                "feature_interactions": shap_values.get("feature_interactions", {}),
                "prediction_contributions": shap_values.get("prediction_contributions", {}),
            }

            return shap_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик SHAP: {str(e)}")
            return {}

    def _calculate_lime_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик LIME"""
        try:
            if not self.config.use_lime:
                return {}

            # Расчет значений LIME
            lime_values = self.model_selector.explain_local_predictions(data)

            # Статистика по LIME
            lime_stats = {
                "local_importance": lime_values.get("local_importance", {}),
                "local_interactions": lime_values.get("local_interactions", {}),
                "local_contributions": lime_values.get("local_contributions", {}),
            }

            return lime_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик LIME: {str(e)}")
            return {}

    def _calculate_permutation_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик перестановочного теста"""
        try:
            if not self.config.use_permutation:
                return {}

            # Расчет значений перестановочного теста
            permutation_values = self.model_selector.permutation_importance(data)

            # Статистика по перестановочному тесту
            permutation_stats = {
                "feature_importance": permutation_values.get("feature_importance", {}),
                "feature_interactions": permutation_values.get("feature_interactions", {}),
                "prediction_contributions": permutation_values.get("prediction_contributions", {}),
            }

            return permutation_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик перестановочного теста: {str(e)}")
            return {}

    def _calculate_correlation_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик корреляции"""
        try:
            if not self.config.use_correlation:
                return {}

            # Расчет корреляций
            correlations = data.corr()

            # Статистика по корреляциям
            correlation_stats = {
                "feature_correlations": correlations.to_dict(),
                "target_correlations": (
                    correlations["profit"].to_dict() if "profit" in correlations else {}
                ),
                "feature_interactions": {},
            }

            # Расчет взаимодействий
            for col1 in data.columns:
                for col2 in data.columns:
                    if col1 < col2:
                        interaction = data[col1] * data[col2]
                        correlation = interaction.corr(data["profit"]) if "profit" in data else 0
                        correlation_stats["feature_interactions"][f"{col1}_{col2}"] = correlation

            return correlation_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик корреляции: {str(e)}")
            return {}

    def _calculate_causality_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик причинности"""
        try:
            if not self.config.use_causality:
                return {}

            # Расчет причинности
            causality_values = self.model_selector.calculate_causality(data)

            # Статистика по причинности
            causality_stats = {
                "feature_causality": causality_values.get("feature_causality", {}),
                "feature_interactions": causality_values.get("feature_interactions", {}),
                "prediction_contributions": causality_values.get("prediction_contributions", {}),
            }

            return causality_stats

        except Exception as e:
            logger.error(f"Ошибка расчета метрик причинности: {str(e)}")
            return {}

    def _plot_analysis(self, data: pd.DataFrame, metrics: Dict, save_path: Path):
        """Построение анализа"""
        try:
            # Создание фигуры
            fig = plt.figure(figsize=(20, 15))

            # График цены
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(data.index, data["close"])
            ax1.set_title("Price")
            ax1.grid(True)

            # График объема
            ax2 = plt.subplot(3, 2, 2)
            ax2.bar(data.index, data["volume"])
            ax2.set_title("Volume")
            ax2.grid(True)

            # График волатильности
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(data.index, data["volatility"])
            ax3.set_title("Volatility")
            ax3.grid(True)

            # График режимов
            ax4 = plt.subplot(3, 2, 4)
            sns.scatterplot(data=data, x="volatility", y="trend", hue="regime", ax=ax4)
            ax4.set_title("Market Regimes")
            ax4.grid(True)

            # График метрик
            ax5 = plt.subplot(3, 2, 5)
            metrics_df = pd.DataFrame(metrics)
            metrics_df.plot(ax=ax5)
            ax5.set_title("Metrics")
            ax5.grid(True)

            # График корреляций
            ax6 = plt.subplot(3, 2, 6)
            sns.heatmap(data.corr(), ax=ax6)
            ax6.set_title("Correlations")

            # Сохранение
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            logger.error(f"Ошибка построения анализа: {str(e)}")

    async def analyze_backtest(self, data: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Анализ бэктеста"""
        try:
            # Инициализация
            self._start_time = datetime.now()
            self._running = True

            # Расчет метрик
            trade_metrics = self._calculate_trade_metrics(trades)
            regime_metrics = self._calculate_regime_metrics(data)
            pattern_metrics = self._calculate_pattern_metrics(data)
            adaptation_metrics = self._calculate_adaptation_metrics(data)
            shap_metrics = self._calculate_shap_metrics(data)
            lime_metrics = self._calculate_lime_metrics(data)
            permutation_metrics = self._calculate_permutation_metrics(data)
            correlation_metrics = self._calculate_correlation_metrics(data)
            causality_metrics = self._calculate_causality_metrics(data)

            # Объединение метрик
            metrics = {
                **trade_metrics,
                "regime_metrics": regime_metrics,
                "pattern_metrics": pattern_metrics,
                "adaptation_metrics": adaptation_metrics,
                "shap_metrics": shap_metrics,
                "lime_metrics": lime_metrics,
                "permutation_metrics": permutation_metrics,
                "correlation_metrics": correlation_metrics,
                "causality_metrics": causality_metrics,
            }

            # Построение анализа
            self._plot_analysis(data, metrics, self.config.data_dir / "backtest_analysis.png")

            # Сохранение метрик
            metrics_file = self.config.data_dir / "backtest_metrics.json"
            async with aiofiles.open(metrics_file, "w") as f:
                await f.write(json.dumps(metrics, indent=2))

            return metrics

        except Exception as e:
            logger.error(f"Ошибка анализа: {str(e)}")
            raise
        finally:
            self._running = False
            self._executor.shutdown(wait=True)


async def main():
    """Основная функция"""
    try:
        # Инициализация анализатора
        explainer = BacktestExplainer()

        # Загрузка данных
        data = pd.read_csv(
            explainer.config.data_dir / "backtest_data.csv", index_col=0, parse_dates=True
        )
        trades = json.load(open(explainer.config.data_dir / "backtest_trades.json"))

        # Анализ бэктеста
        metrics = await explainer.analyze_backtest(data, trades)

        logger.info("Анализ успешно завершен")

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add("logs/backtest_{time}.log", rotation="1 day", retention="7 days", level="INFO")

    # Запуск асинхронного main
    asyncio.run(main())
