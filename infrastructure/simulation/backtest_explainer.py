import asyncio
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiofiles
import matplotlib.pyplot as plt
from shared.numpy_utils import np
import pandas as pd
# from pandas import crosstab  # Исправлено: убираем неиспользуемый импорт
import psutil  # type: ignore
import seaborn as sns  # type: ignore
import yaml
from loguru import logger

warnings.filterwarnings("ignore")
# from ml.live_adaptation import LiveAdaptation
# from ml.model_selector import ModelSelector
# from ml.pattern_discovery import PatternDiscovery


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
                models_dir=Path(
                    config["explainer"].get("models_dir", "models/backtest")
                ),
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
                use_regime_detection=config["explainer"].get(
                    "use_regime_detection", True
                ),
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

    def __init__(self) -> None:
        self.feature_importance: Dict[str, float] = {}
        self.shap_values = None
        self.explainer = None

    def analyze_trade(
        self, trade: Dict, market_data: pd.DataFrame, indicators: Dict
    ) -> Dict:
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
            # Безопасное получение данных
            try:
                if entry_time in market_data.index:
                    entry_data: pd.Series = market_data.loc[entry_time]
                else:
                    entry_data_alt: pd.Series = market_data.iloc[0] if len(market_data) > 0 else pd.Series()
            except (KeyError, IndexError):
                entry_data_fallback: pd.Series = market_data.iloc[0] if len(market_data) > 0 else pd.Series()
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
                trade=trade,
                entry_conditions=entry_conditions,
                regime_analysis=regime_analysis,
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
        conditions: Dict[str, Dict] = {
            "price_action": {},
            "indicators": {},
            "volume": {},
            "volatility": {},
        }
        # Анализ price action
        conditions["price_action"] = {
            "trend": "unknown",  # Временно заменяем несуществующий метод
            "support_resistance": "unknown",  # Временно заменяем несуществующий метод
            "candlestick_pattern": "unknown",  # Временно заменяем несуществующий метод
        }
        # Анализ индикаторов
        for name, value in indicators.items():
            if isinstance(value, dict):
                conditions["indicators"][name] = {
                    "value": value,
                    "signal": "unknown",  # Временно заменяем несуществующий метод
                }
            else:
                conditions["indicators"][name] = {
                    "value": value,
                    "signal": "unknown",  # Временно заменяем несуществующий метод
                }
        # Анализ объема
        conditions["volume"] = {
            "relative_volume": market_data["volume"] / market_data["volume"].mean(),
            "volume_trend": "unknown",  # Временно заменяем несуществующий метод
        }
        # Анализ волатильности
        conditions["volatility"] = {
            "current": market_data.get("volatility", 0.0),
            "trend": "unknown",  # Временно заменяем несуществующий метод
        }
        return conditions

    def _analyze_market_regime(
        self, market_data: pd.DataFrame, entry_time: datetime
    ) -> Dict:
        """Анализ рыночного режима"""
        # Получение данных до входа
        pre_entry_data = market_data[market_data.index < entry_time]
        return {
            "regime": pre_entry_data.get("regime", pd.Series()).iloc[-1] if len(pre_entry_data) > 0 else "unknown",
            "regime_strength": 0.5,  # Временно заменяем несуществующий метод
            "regime_duration": 0,  # Временно заменяем несуществующий метод
            "regime_transition": False,  # Временно заменяем несуществующий метод
        }

    def _analyze_success_factors(
        self, trade: Dict, entry_conditions: Dict, regime_analysis: Dict
    ) -> Dict:
        """Анализ факторов успеха/неудачи"""
        # Расчет прибыли/убытка
        pnl = trade["capital"] - trade["initial_capital"]
        is_successful = pnl > 0
        # Определение ключевых факторов
        factors: Dict[str, List[str]] = {"positive": [], "negative": []}
        # Анализ условий входа
        if entry_conditions.get("price_action", {}).get("trend") == "strong":
            factors["positive" if is_successful else "negative"].append("Strong trend")
        if entry_conditions.get("volume", {}).get("relative_volume", 0) > 1.5:
            factors["positive" if is_successful else "negative"].append("High volume")
        # Анализ рыночного режима
        if regime_analysis.get("regime_strength", 0) > 0.7:
            factors["positive" if is_successful else "negative"].append("Strong regime")
        if regime_analysis.get("regime_transition", False):
            factors["negative"].append("Regime transition")
        return factors


class BacktestExplainer:
    """Анализатор бэктеста"""

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """Инициализация анализатора"""
        self.config = config or ExplainerConfig.from_yaml()
        self.metrics_history: List[Dict] = []
        self._sim_lock = asyncio.Lock()
        self._start_time: Optional[datetime] = None
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
            Path(str(dir_path)).mkdir(parents=True, exist_ok=True)
        # Инициализация компонентов
        # Заглушки для неопределенных классов
        class ModelSelector:
            def explain_predictions(self, data: Any) -> Any:
                return {}

            def explain_local_predictions(self, data: Any) -> Any:
                return {}

            def permutation_importance(self, data: Any) -> Any:
                return {}

            def calculate_causality(self, data: Any) -> Any:
                return {}

        class PatternDiscovery:
            def find_patterns(self, data: Any) -> Any:
                return {}

        class LiveAdaptation:
            def update_state(self, data: Any) -> None:
                pass
        
        self.model_selector = ModelSelector()
        self.pattern_discovery = PatternDiscovery()
        self.live_adaptation = LiveAdaptation()
        # Инициализация кэша
        self._init_cache()

    def _init_cache(self) -> None:
        """Инициализация кэша"""
        self._price_cache: Dict[int, float] = {}
        self._volume_cache: Dict[int, float] = {}
        self._regime_cache: Dict[int, str] = {}
        self._pattern_cache: Dict[int, List[Dict[str, Any]]] = {}
        self._adaptation_cache: Dict[int, Dict[str, Any]] = {}
        self._shap_cache: Dict[int, Dict[str, Any]] = {}
        self._lime_cache: Dict[int, Dict[str, Any]] = {}
        self._permutation_cache: Dict[int, Dict[str, Any]] = {}
        self._correlation_cache: Dict[int, Dict[str, Any]] = {}
        self._causality_cache: Dict[int, Dict[str, Any]] = {}

    @lru_cache(maxsize=1000)
    def _get_cached_price(self, timestamp: int) -> float:
        """Получение кэшированной цены"""
        return self._price_cache.get(timestamp, 0.0)

    @lru_cache(maxsize=1000)
    def _get_cached_volume(self, timestamp: int) -> float:
        """Получение кэшированного объема"""
        return self._volume_cache.get(timestamp, 0.0)

    @lru_cache(maxsize=1000)
    def _get_cached_regime(self, timestamp: int) -> str:
        """Получение кэшированного режима"""
        return self._regime_cache.get(timestamp, "unknown")

    @lru_cache(maxsize=1000)
    def _get_cached_pattern(self, timestamp: int) -> List[Dict[str, Any]]:
        """Получение кэшированных паттернов"""
        return self._pattern_cache.get(timestamp, [])

    @lru_cache(maxsize=1000)
    def _get_cached_adaptation(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированного состояния адаптации"""
        return self._adaptation_cache.get(timestamp, {})

    @lru_cache(maxsize=1000)
    def _get_cached_shap(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированных значений SHAP"""
        return self._shap_cache.get(timestamp, {})

    @lru_cache(maxsize=1000)
    def _get_cached_lime(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированных значений LIME"""
        return self._lime_cache.get(timestamp, {})

    @lru_cache(maxsize=1000)
    def _get_cached_permutation(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированных значений permutation importance"""
        return self._permutation_cache.get(timestamp, {})

    @lru_cache(maxsize=1000)
    def _get_cached_correlation(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированных значений корреляции"""
        return self._correlation_cache.get(timestamp, {})

    @lru_cache(maxsize=1000)
    def _get_cached_causality(self, timestamp: int) -> Dict[str, Any]:
        """Получение кэшированных значений причинности"""
        return self._causality_cache.get(timestamp, {})

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

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Обработка сигнала завершения"""
        logger.info("Получен сигнал завершения, останавливаем анализ...")
        self._running = False

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Расчет метрик сделок"""
        try:
            if not trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_profit": 0.0,
                    "total_loss": 0.0,
                    "net_profit": 0.0,
                }
            # Базовые метрики
            profits = [trade["profit"] for trade in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            total_profit = sum(winning_trades)
            total_loss = abs(sum(losing_trades))
            return {
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(trades),
                "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": sum(profits),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета метрик сделок: {str(e)}")
            return {}

    def _calculate_regime_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик режимов"""
        try:
            if not self.config.use_regime_detection:
                return {}
            # Статистика по режимам
            regime_stats = data.groupby("regime").agg(
                {
                    "close": ["mean", "std", "min", "max"],
                    "volume": ["mean", "std"],
                    "volatility": ["mean", "std"],
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
            pattern_stats: Dict[str, Dict] = {
                "pattern_types": {},
                "pattern_accuracy": {},
                "pattern_profitability": {},
            }
            # Исправление: безопасная обработка patterns
            if patterns is not None and hasattr(patterns, '__iter__') and not isinstance(patterns, (str, bytes)):
                for pattern in patterns:
                    if isinstance(pattern, dict):
                        pattern_type = pattern.get("type", "unknown")
                        if pattern_type not in pattern_stats["pattern_types"]:
                            pattern_stats["pattern_types"][pattern_type] = 0
                        pattern_stats["pattern_types"][pattern_type] += 1
                        if "accuracy" in pattern:
                            if pattern_type not in pattern_stats["pattern_accuracy"]:
                                pattern_stats["pattern_accuracy"][pattern_type] = []
                            pattern_stats["pattern_accuracy"][pattern_type].append(
                                pattern["accuracy"]
                            )
                        if "profitability" in pattern:
                            if pattern_type not in pattern_stats["pattern_profitability"]:
                                pattern_stats["pattern_profitability"][pattern_type] = []
                            pattern_stats["pattern_profitability"][pattern_type].append(
                                pattern["profitability"]
                            )
            # Усреднение метрик
            for pattern_type in pattern_stats["pattern_accuracy"]:
                if pattern_stats["pattern_accuracy"][pattern_type]:
                    pattern_stats["pattern_accuracy"][pattern_type] = np.mean(
                        pattern_stats["pattern_accuracy"][pattern_type]
                    )
            for pattern_type in pattern_stats["pattern_profitability"]:
                if pattern_stats["pattern_profitability"][pattern_type]:
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
                "prediction_contributions": shap_values.get(
                    "prediction_contributions", {}
                ),
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
                "feature_interactions": permutation_values.get(
                    "feature_interactions", {}
                ),
                "prediction_contributions": permutation_values.get(
                    "prediction_contributions", {}
                ),
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
                        correlation = (
                            interaction.corr(data["profit"]) if "profit" in data else 0
                        )
                        correlation_stats["feature_interactions"][
                            f"{col1}_{col2}"
                        ] = correlation
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
                "feature_interactions": causality_values.get(
                    "feature_interactions", {}
                ),
                "prediction_contributions": causality_values.get(
                    "prediction_contributions", {}
                ),
            }
            return causality_stats
        except Exception as e:
            logger.error(f"Ошибка расчета метрик причинности: {str(e)}")
            return {}

    def _plot_analysis(self, data: pd.DataFrame, metrics: Dict, save_path: Path) -> None:
        """Построение анализа"""
        try:
            # Создание фигуры
            fig = plt.figure(figsize=(20, 15))
            # График цены
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(data.index, data["close"])
            ax1.set_title("Price")
            # График объема
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(data.index, data["volume"])
            ax2.set_title("Volume")
            # График волатильности
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(data.index, data["volatility"])
            ax3.set_title("Volatility")
            # График режимов
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(data.index, data["regime"])
            ax4.set_title("Regime")
            # График метрик
            ax5 = plt.subplot(3, 2, 5)
            metrics_data = {
                "Win Rate": metrics["win_rate"],
                "Profit Factor": metrics["profit_factor"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Max Drawdown": metrics["max_drawdown"],
            }
            ax5.bar(list(metrics_data.keys()), list(metrics_data.values()))
            ax5.set_title("Metrics")
            # График корреляций
            ax6 = plt.subplot(3, 2, 6)
            if hasattr(data, 'corr'):
                correlations = data.corr()
                im = ax6.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
                ax6.set_title("Correlations")
                plt.colorbar(im, ax=ax6)
            else:
                # Fallback если метод corr() недоступен
                ax6.text(0.5, 0.5, "Correlations not available", ha='center', va='center')
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
            start_time = datetime.now()
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
            # Системные метрики
            system_metrics = self._get_system_metrics()
            end_time = datetime.now()
            # Создание отчета
            report = {
                "start_time": start_time,
                "end_time": end_time,
                "analysis_time": (end_time - start_time).total_seconds(),
                "memory_usage": system_metrics["memory_usage"],
                "cpu_usage": system_metrics["cpu_usage"],
                "success": True,
                "trade_metrics": trade_metrics,
                "regime_metrics": regime_metrics,
                "pattern_metrics": pattern_metrics,
                "adaptation_metrics": adaptation_metrics,
                "shap_metrics": shap_metrics,
                "lime_metrics": lime_metrics,
                "permutation_metrics": permutation_metrics,
                "correlation_metrics": correlation_metrics,
                "causality_metrics": causality_metrics,
            }
            # Сохранение отчета
            report_path = self.config.data_dir / f"report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            async with aiofiles.open(report_path, "w") as f:
                await f.write(json.dumps(report, default=str, indent=2))
            # Построение графиков
            plot_path = self.config.data_dir / f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
            self._plot_analysis(data, trade_metrics, plot_path)
            return report
        except Exception as e:
            logger.error(f"Ошибка анализа бэктеста: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "start_time": datetime.now(),
                "end_time": datetime.now(),
            }


async def main() -> None:
    """Основная функция"""
    try:
        # Создание анализатора
        explainer = BacktestExplainer()
        # Загрузка данных (пример)
        data = pd.DataFrame({
            "close": np.random.randn(1000).cumsum(),
            "volume": np.random.randint(100, 1000, 1000),
            "volatility": np.random.uniform(0.01, 0.05, 1000),
            "regime": np.random.choice(["trending", "ranging", "volatile"], 1000),
            "profit": np.random.randn(1000),
        })
        trades = [
            {
                "timestamp": datetime.now(),
                "capital": 10000,
                "initial_capital": 10000,
                "profit": 100,
            }
        ]
        # Анализ
        report = await explainer.analyze_backtest(data, trades)
        print(f"Анализ завершен: {report['success']}")
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
