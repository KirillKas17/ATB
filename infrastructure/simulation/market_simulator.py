import asyncio
import random
import signal
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import joblib  # type: ignore
import psutil  # type: ignore
import talib  # type: ignore
import yaml
from loguru import logger
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from shared.numpy_utils import np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

warnings.filterwarnings("ignore")


@dataclass
class SimulationConfig:
    """Конфигурация симуляции"""

    start_date: datetime
    end_date: datetime
    initial_price: float
    volatility: float
    trend_strength: float
    mean_reversion: float
    noise_level: float
    volume_scale: float
    market_impact: float
    liquidity_factor: float
    regime_switching: bool = True
    regime_probability: float = 0.1
    regime_duration: int = 100
    data_dir: Path = Path("data/simulation")
    models_dir: Path = Path("models/simulation")
    log_dir: Path = Path("logs/simulation")
    backup_dir: Path = Path("backups/simulation")
    metrics_window: int = 100
    min_samples: int = 1000
    max_samples: int = 100000
    num_threads: int = 4
    cache_size: int = 1000
    save_interval: int = 1000
    validation_split: float = 0.2
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "SimulationConfig":
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return cls(
                start_date=datetime.fromisoformat(config["simulation"]["start_date"]),
                end_date=datetime.fromisoformat(config["simulation"]["end_date"]),
                initial_price=config["simulation"]["initial_price"],
                volatility=config["simulation"]["volatility"],
                trend_strength=config["simulation"]["trend_strength"],
                mean_reversion=config["simulation"]["mean_reversion"],
                noise_level=config["simulation"]["noise_level"],
                volume_scale=config["simulation"]["volume_scale"],
                market_impact=config["simulation"]["market_impact"],
                liquidity_factor=config["simulation"]["liquidity_factor"],
                regime_switching=config["simulation"].get("regime_switching", True),
                regime_probability=config["simulation"].get("regime_probability", 0.1),
                regime_duration=config["simulation"].get("regime_duration", 100),
                data_dir=Path(config["simulation"].get("data_dir", "data/simulation")),
                models_dir=Path(
                    config["simulation"].get("models_dir", "models/simulation")
                ),
                log_dir=Path(config["simulation"].get("log_dir", "logs/simulation")),
                backup_dir=Path(config["simulation"].get("backup_dir", "backups/simulation")),
                metrics_window=config["simulation"].get("metrics_window", 100),
                min_samples=config["simulation"].get("min_samples", 1000),
                max_samples=config["simulation"].get("max_samples", 100000),
                num_threads=config["simulation"].get("num_threads", 4),
                cache_size=config["simulation"].get("cache_size", 1000),
                save_interval=config["simulation"].get("save_interval", 1000),
                validation_split=config["simulation"].get("validation_split", 0.2),
                random_seed=config["simulation"].get("random_seed", 42),
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise


@dataclass
class SimulationMetrics:
    """Метрики симуляции"""

    start_time: datetime
    end_time: datetime
    simulation_time: float
    memory_usage: float
    cpu_usage: float
    total_ticks: int
    regime_changes: int
    price_range: Tuple[float, float]
    volume_range: Tuple[float, float]
    volatility_range: Tuple[float, float]
    trend_strength_range: Tuple[float, float]
    mean_reversion_range: Tuple[float, float]
    noise_level_range: Tuple[float, float]
    market_impact_range: Tuple[float, float]
    liquidity_factor_range: Tuple[float, float]
    success: bool
    error: Optional[str] = None
    validation_metrics: Optional[Dict] = None
    model_metrics: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None


class MarketImpact:
    """Модель влияния на рынок"""

    def __init__(
        self,
        base_impact: float = 0.0001,
        volume_factor: float = 0.5,
        volatility_factor: float = 0.3,
    ):
        self.base_impact = base_impact
        self.volume_factor = volume_factor
        self.volatility_factor = volatility_factor

    def calculate_impact(
        self, order_size: float, market_volume: float, volatility: float
    ) -> float:
        """
        Расчет влияния на рынок.
        Args:
            order_size: Размер ордера
            market_volume: Объем рынка
            volatility: Волатильность
        Returns:
            float: Процент влияния
        """
        # Базовое влияние
        impact = self.base_impact
        # Влияние объема
        volume_ratio = order_size / market_volume
        impact += volume_ratio * self.volume_factor
        # Влияние волатильности
        impact += volatility * self.volatility_factor
        return min(impact, 0.01)  # Максимум 1%


class LatencyModel:
    """Модель задержек"""

    def __init__(
        self,
        base_latency: float = 0.1,
        jitter: float = 0.05,
        network_factor: float = 0.2,
    ):
        self.base_latency = base_latency
        self.jitter = jitter
        self.network_factor = network_factor

    def calculate_latency(self, order_size: float) -> float:
        """
        Расчет задержки.
        Args:
            order_size: Размер ордера
        Returns:
            float: Задержка в секундах
        """
        # Базовая задержка
        latency = self.base_latency
        # Случайные колебания
        latency += random.uniform(-self.jitter, self.jitter)
        # Влияние размера ордера
        latency += order_size * self.network_factor
        return max(latency, 0.05)  # Минимум 50мс


class MarketSimulator:
    """Симулятор рынка"""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Инициализация симулятора"""
        self.config = config or SimulationConfig.from_yaml()
        self.metrics_history: list[SimulationMetrics] = []
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
            dir_path.mkdir(parents=True, exist_ok=True)
        # Инициализация компонентов
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.regime_model = None
        self.market_state = {
            "price": self.config.initial_price,
            "volume": 0.0,
            "volatility": self.config.volatility,
            "trend_strength": self.config.trend_strength,
            "mean_reversion": self.config.mean_reversion,
            "noise_level": self.config.noise_level,
            "market_impact": self.config.market_impact,
            "liquidity_factor": self.config.liquidity_factor,
            "regime": "normal",
        }
        # Инициализация кэша
        self._init_cache()

    def _init_cache(self) -> None:
        """Инициализация кэша"""
        self._price_cache = {}
        self._volume_cache = {}
        self._regime_cache = {}

    @lru_cache(maxsize=1000)
    def _get_cached_price(self, timestamp: int) -> float:
        """Получение кэшированной цены"""
        result = self._price_cache.get(timestamp)
        return float(result) if result is not None else 0.0

    @lru_cache(maxsize=1000)
    def _get_cached_volume(self, timestamp: int) -> float:
        """Получение кэшированного объема"""
        result = self._volume_cache.get(timestamp)
        return float(result) if result is not None else 0.0

    @lru_cache(maxsize=1000)
    def _get_cached_regime(self, timestamp: int) -> str:
        """Получение кэшированного режима"""
        result = self._regime_cache.get(timestamp)
        return str(result) if result is not None else "normal"

    def _get_system_metrics(self) -> Dict:
        """Получение системных метрик"""
        return {
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
            "thread_count": len(self._executor._threads),
            "cache_size": len(self._price_cache)
            + len(self._volume_cache)
            + len(self._regime_cache),
        }

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения")
        self._running = False
        self._executor.shutdown(wait=False)

    def _generate_price_movement(self) -> float:
        """Генерация движения цены"""
        try:
            # Базовое движение
            trend_strength_val = self.market_state.get("trend_strength", 0.0)
            trend_strength = float(trend_strength_val) if isinstance(trend_strength_val, (int, float, str)) else 0.0
            trend = trend_strength * np.random.normal(0, 1)
            mean_rev_strength_val = self.market_state.get("mean_reversion", 0.0)
            mean_rev_strength = float(mean_rev_strength_val) if isinstance(mean_rev_strength_val, (int, float, str)) else 0.0
            current_price_val = self.market_state.get("price", self.config.initial_price)
            current_price = float(current_price_val) if isinstance(current_price_val, (int, float, str)) else self.config.initial_price
            mean_rev = mean_rev_strength * (
                self.config.initial_price - current_price
            )
            noise_level_val = self.market_state.get("noise_level", 0.0)
            noise_level = float(noise_level_val) if isinstance(noise_level_val, (int, float, str)) else 0.0
            noise = noise_level * np.random.normal(0, 1)
            # Режим рынка
            if self.market_state["regime"] == "trend":
                trend *= 2
                mean_rev *= 0.5
            elif self.market_state["regime"] == "mean_reversion":
                trend *= 0.5
                mean_rev *= 2
            elif self.market_state["regime"] == "volatile":
                noise *= 2
            # Общее движение
            movement = trend + mean_rev + noise
            # Применение волатильности
            volatility_val = self.market_state.get("volatility", 0.0)
            volatility = float(volatility_val) if isinstance(volatility_val, (int, float, str)) else 0.02
            movement *= volatility
            return float(movement)
        except Exception as e:
            logger.error(f"Ошибка генерации движения цены: {str(e)}")
            return 0.0

    def _generate_volume(self) -> float:
        """Генерация объема"""
        try:
            # Базовый объем
            base_volume = np.random.lognormal(0, 1)
            # Модификация объема
            volume = base_volume * self.config.volume_scale
            # Учет ликвидности
            liquidity_factor_val = self.market_state.get("liquidity_factor", 1.0)
            liquidity_factor = float(liquidity_factor_val) if isinstance(liquidity_factor_val, (int, float, str)) else 1.0
            volume *= liquidity_factor
            # Учет рыночного воздействия
            market_impact_val = self.market_state.get("market_impact", 0.0)
            market_impact = float(market_impact_val) if isinstance(market_impact_val, (int, float, str)) else 0.0
            volume *= 1 - market_impact
            return float(volume)
        except Exception as e:
            logger.error(f"Ошибка генерации объема: {str(e)}")
            return 0.0

    def _update_market_state(self) -> None:
        """Обновление состояния рынка"""
        try:
            # Обновление режима
            if self.config.regime_switching:
                if np.random.random() < self.config.regime_probability:
                    regimes = ["normal", "trend", "mean_reversion", "volatile"]
                    self.market_state["regime"] = np.random.choice(regimes)
            # Обновление параметров
            volatility_val = self.market_state.get("volatility", 0.02)
            volatility = float(volatility_val) if isinstance(volatility_val, (int, float, str)) else 0.02
            self.market_state["volatility"] = volatility * (1 + np.random.normal(0, 0.1))
            
            trend_strength_val = self.market_state.get("trend_strength", 0.1)
            trend_strength = float(trend_strength_val) if isinstance(trend_strength_val, (int, float, str)) else 0.1
            self.market_state["trend_strength"] = trend_strength * (1 + np.random.normal(0, 0.1))
            
            mean_reversion_val = self.market_state.get("mean_reversion", 0.05)
            mean_reversion = float(mean_reversion_val) if isinstance(mean_reversion_val, (int, float, str)) else 0.05
            self.market_state["mean_reversion"] = mean_reversion * (1 + np.random.normal(0, 0.1))
            
            noise_level_val = self.market_state.get("noise_level", 0.01)
            noise_level = float(noise_level_val) if isinstance(noise_level_val, (int, float, str)) else 0.01
            self.market_state["noise_level"] = noise_level * (1 + np.random.normal(0, 0.1))
            
            market_impact_val = self.market_state.get("market_impact", 0.0001)
            market_impact = float(market_impact_val) if isinstance(market_impact_val, (int, float, str)) else 0.0001
            self.market_state["market_impact"] = market_impact * (1 + np.random.normal(0, 0.1))
            
            liquidity_factor_val = self.market_state.get("liquidity_factor", 1.0)
            liquidity_factor = float(liquidity_factor_val) if isinstance(liquidity_factor_val, (int, float, str)) else 1.0
            self.market_state["liquidity_factor"] = liquidity_factor * (1 + np.random.normal(0, 0.1))
            # Нормализация параметров
            for key in [
                "volatility",
                "trend_strength",
                "mean_reversion",
                "noise_level",
                "market_impact",
                "liquidity_factor",
            ]:
                self.market_state[key] = max(0.1, min(2.0, float(self.market_state[key])))
        except Exception as e:
            logger.error(f"Ошибка обновления состояния рынка: {str(e)}")

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
        try:
            # Базовые индикаторы
            data["SMA_20"] = talib.SMA(data["close"], timeperiod=20)
            data["SMA_50"] = talib.SMA(data["close"], timeperiod=50)
            data["SMA_200"] = talib.SMA(data["close"], timeperiod=200)
            data["RSI"] = talib.RSI(data["close"], timeperiod=14)
            data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = talib.MACD(
                data["close"]
            )
            data["BBANDS_UPPER"], data["BBANDS_MIDDLE"], data["BBANDS_LOWER"] = (
                talib.BBANDS(data["close"])
            )
            data["ATR"] = talib.ATR(
                data["high"], data["low"], data["close"], timeperiod=14
            )
            # Объемные индикаторы
            data["OBV"] = talib.OBV(data["close"], data["volume"])
            data["AD"] = talib.AD(
                data["high"], data["low"], data["close"], data["volume"]
            )
            # Моментум индикаторы
            data["MOM"] = talib.MOM(data["close"], timeperiod=10)
            data["ROC"] = talib.ROC(data["close"], timeperiod=10)
            # Волатильность
            data["NATR"] = talib.NATR(
                data["high"], data["low"], data["close"], timeperiod=14
            )
            data["TRANGE"] = talib.TRANGE(data["high"], data["low"], data["close"])
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {str(e)}")
            return data

    def _calculate_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет режима рынка"""
        try:
            # Признаки режима
            data["volatility"] = data["close"].pct_change().rolling(window=20).std()
            data["trend"] = (data["close"] - data["close"].shift(20)) / data[
                "close"
            ].shift(20)
            data["mean_reversion"] = (data["close"] - data["SMA_20"]) / data["SMA_20"]
            # Классификация режима
            conditions = [
                (data["volatility"] > data["volatility"].quantile(0.8)),
                (data["trend"].abs() > data["trend"].abs().quantile(0.8)),
                (
                    data["mean_reversion"].abs()
                    > data["mean_reversion"].abs().quantile(0.8)
                ),
            ]
            choices = ["volatile", "trend", "mean_reversion"]
            data["regime"] = np.select(conditions, choices, default="normal")
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета режима рынка: {str(e)}")
            return data

    def _calculate_market_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет качества рынка"""
        try:
            # Ликвидность
            data["liquidity"] = data["volume"] * data["close"]
            data["spread"] = (data["high"] - data["low"]) / data["close"]
            # Волатильность
            data["volatility"] = data["close"].pct_change().rolling(window=20).std()
            # Тренд
            data["trend"] = (data["close"] - data["close"].shift(20)) / data[
                "close"
            ].shift(20)
            # Качество рынка
            data["market_quality"] = (
                (1 / data["spread"])  # Узкий спред
                * (1 / data["volatility"])  # Низкая волатильность
                * (1 / data["trend"].abs())  # Отсутствие сильного тренда
            )
            # Нормализация
            data["market_quality"] = (
                data["market_quality"] - data["market_quality"].min()
            ) / (data["market_quality"].max() - data["market_quality"].min())
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета качества рынка: {str(e)}")
            return data

    def _calculate_market_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет рыночного воздействия"""
        try:
            # Базовое воздействие
            data["base_impact"] = data["volume"] * data["close"] / data["liquidity"]
            # Модификация по режиму
            regime_impact = {
                "normal": 1.0,
                "trend": 1.5,
                "mean_reversion": 0.8,
                "volatile": 2.0,
            }
            data["regime_impact"] = data["regime"].map(regime_impact)
            # Общее воздействие
            data["market_impact"] = data["base_impact"] * data["regime_impact"]
            # Нормализация
            data["market_impact"] = (
                data["market_impact"] - data["market_impact"].min()
            ) / (data["market_impact"].max() - data["market_impact"].min())
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета рыночного воздействия: {str(e)}")
            return data

    def _calculate_market_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет эффективности рынка"""
        try:
            # Автокорреляция
            data["autocorr"] = (
                data["close"]
                .pct_change()
                .rolling(window=20)
                .apply(lambda x: x.autocorr())
            )
            # Тест на случайное блуждание
            data["random_walk"] = (
                data["close"]
                .pct_change()
                .rolling(window=20)
                .apply(lambda x: np.abs(x).sum() / len(x))
            )
            # Эффективность
            data["efficiency"] = 1 - data["autocorr"].abs()
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета эффективности рынка: {str(e)}")
            return data

    def _calculate_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет признаков режима рынка"""
        try:
            # Волатильность
            data["volatility"] = data["close"].pct_change().rolling(window=20).std()
            # Тренд
            data["trend"] = (data["close"] - data["close"].shift(20)) / data[
                "close"
            ].shift(20)
            # Среднее возвращение
            data["mean_reversion"] = (data["close"] - data["SMA_20"]) / data["SMA_20"]
            # Шум
            data["noise"] = data["close"].pct_change().rolling(window=5).std()
            # Ликвидность
            data["liquidity"] = data["volume"] * data["close"]
            # Рыночное воздействие
            data["market_impact"] = data["volume"] / data["liquidity"]
            return data
        except Exception as e:
            logger.error(f"Ошибка расчета признаков режима: {str(e)}")
            return data

    def _train_regime_model(self, data: pd.DataFrame) -> None:
        """Обучение модели режима"""
        try:
            if self.regime_model is not None:
                return
            # Подготовка данных
            features = [
                "volatility",
                "trend",
                "mean_reversion",
                "noise",
                "liquidity",
                "market_impact",
            ]
            X = data[features].to_numpy()
            # Нормализация
            X = self.scaler.fit_transform(X)
            # PCA
            X = self.pca.fit_transform(X)
            # Кластеризация
            from sklearn.cluster import KMeans

            self.regime_model = KMeans(n_clusters=4, random_state=42)
            self.regime_model.fit(X)
            # Сохранение модели
            joblib.dump(
                self.regime_model, self.config.models_dir / "regime_model.joblib"
            )
            joblib.dump(self.scaler, self.config.models_dir / "scaler.joblib")
            joblib.dump(self.pca, self.config.models_dir / "pca.joblib")
        except Exception as e:
            logger.error(f"Ошибка обучения модели режима: {str(e)}")

    def _predict_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Предсказание режима"""
        try:
            if self.regime_model is None:
                return data
            # Подготовка данных
            features = [
                "volatility",
                "trend",
                "mean_reversion",
                "noise",
                "liquidity",
                "market_impact",
            ]
            X = data[features].to_numpy()
            # Нормализация
            X = self.scaler.transform(X)
            # PCA
            X = self.pca.transform(X)
            # Предсказание
            data["predicted_regime"] = self.regime_model.predict(X)
            # Маппинг режимов
            regime_map = {0: "normal", 1: "trend", 2: "mean_reversion", 3: "volatile"}
            data["predicted_regime"] = data["predicted_regime"].map(regime_map)
            return data
        except Exception as e:
            logger.error(f"Ошибка предсказания режима: {str(e)}")
            return data

    def _calculate_market_metrics(self, data: pd.DataFrame) -> Dict:
        """Расчет метрик рынка"""
        try:
            metrics = {
                "price_range": (float(data["close"].min()), float(data["close"].max())),
                "volume_range": (float(data["volume"].min()), float(data["volume"].max())),
                "volatility_range": (
                    float(data["volatility"].min()),
                    float(data["volatility"].max()),
                ),
                "trend_strength_range": (float(data["trend"].min()), float(data["trend"].max())),
                "mean_reversion_range": (
                    float(data["mean_reversion"].min()),
                    float(data["mean_reversion"].max()),
                ),
                "noise_level_range": (float(data["noise"].min()), float(data["noise"].max())),
                "market_impact_range": (
                    float(data["market_impact"].min()),
                    float(data["market_impact"].max()),
                ),
                "liquidity_factor_range": (
                    float(data["liquidity"].min()),
                    float(data["liquidity"].max()),
                ),
            }
            return metrics
        except Exception as e:
            logger.error(f"Ошибка расчета метрик рынка: {str(e)}")
            return {}

    def _plot_market_analysis(self, data: pd.DataFrame, save_path: Path) -> None:
        """Построение анализа рынка"""
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
            # График тренда
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(data.index, data["trend"])
            ax4.set_title("Trend")
            ax4.grid(True)
            # График режимов
            ax5 = plt.subplot(3, 2, 5)
            sns.scatterplot(data=data, x="volatility", y="trend", hue="regime", ax=ax5)
            ax5.set_title("Market Regimes")
            ax5.grid(True)
            # График качества рынка
            ax6 = plt.subplot(3, 2, 6)
            ax6.plot(data.index, data["market_quality"])
            ax6.set_title("Market Quality")
            ax6.grid(True)
            # Сохранение
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            logger.error(f"Ошибка построения анализа: {str(e)}")

    def _save_backup(self, data: pd.DataFrame, path: Path) -> bool:
        """Сохраняет резервную копию данных."""
        try:
            if hasattr(data, 'to_csv'):
                data.to_csv(path, index=False)
            else:
                # Альтернативный способ сохранения
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    if not data.empty:
                        writer = csv.writer(f)
                        writer.writerow(data.columns.tolist())
                        for _, row in data.iterrows():
                            if hasattr(row, 'values'):
                                row_values = row.to_numpy() if hasattr(row, 'to_numpy') else np.asarray(row)
                                writer.writerow(row_values.tolist())
                            else:
                                row_array = np.asarray(row) if hasattr(row, '__iter__') else [row]
                                writer.writerow(row_array.tolist() if hasattr(row_array, 'tolist') else list(row_array))
        except Exception as e:
            logger.error(f"Ошибка сохранения резервной копии: {str(e)}")
            return False
        return True

    def _save_results(self, results: dict, path: Path) -> None:
        """Сохранение результатов"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.dump(results, f)
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {str(e)}")

    async def generate_market_data(self) -> pd.DataFrame:
        """Генерация рыночных данных"""
        try:
            # Проверка конфигурации
            if self.config.start_date >= self.config.end_date:
                raise ValueError("Дата начала должна быть раньше даты окончания")
            # Расчет количества свечей
            total_candles = int(
                (self.config.end_date - self.config.start_date).total_seconds() / 60
            )
            if total_candles < self.config.min_samples:
                raise ValueError(
                    f"Недостаточно свечей для симуляции. Требуется минимум {self.config.min_samples}"
                )
            # Инициализация состояния
            self._start_time = datetime.now()
            self._running = True
            # Регистрация обработчика сигналов
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            # Генерация данных
            data = []
            current_price = self.config.initial_price
            current_volume = self.config.volume_scale
            current_regime = "normal"
            for i in range(total_candles):
                if not self._running:
                    break
                # Генерация движения цены
                price_movement = self._generate_price_movement()
                current_price *= 1 + price_movement
                # Генерация объема
                volume = self._generate_volume()
                current_volume = volume
                # Обновление режима
                if (
                    self.config.regime_switching
                    and random.random() < self.config.regime_probability
                ):
                    current_regime = random.choice(["trend", "range", "volatile"])
                # Добавление свечи
                candle = {
                    "timestamp": self.config.start_date + timedelta(minutes=i),
                    "open": current_price,
                    "high": current_price * (1 + abs(price_movement)),
                    "low": current_price * (1 - abs(price_movement)),
                    "close": current_price,
                    "volume": current_volume,
                    "regime": current_regime,
                }
                data.append(candle)
                # Сохранение промежуточных результатов
                if i % self.config.save_interval == 0:
                    self._save_backup(pd.DataFrame(data), self.config.backup_dir / f"backup_{i}.csv")
            # Преобразование в DataFrame
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            # Расчет технических индикаторов
            df = self._calculate_technical_indicators(df)
            # Расчет режима рынка
            df = self._calculate_market_regime(df)
            # Расчет качества рынка
            df = self._calculate_market_quality(df)
            # Расчет влияния на рынок
            df = self._calculate_market_impact(df)
            # Расчет эффективности рынка
            df = self._calculate_market_efficiency(df)
            # Расчет метрик
            metrics = self._calculate_market_metrics(df)
            # Сохранение результатов
            self._save_results(metrics, self.config.data_dir / "market_metrics.json")
            # Построение графиков
            self._plot_market_analysis(df, self.config.data_dir / "market_analysis.png")
            # Обновление метрик
            end_time = datetime.now()
            start_time_metrics = self._start_time if self._start_time is not None else datetime.now()
            self.metrics = SimulationMetrics(
                start_time=start_time_metrics,
                end_time=end_time,
                simulation_time=(end_time - start_time_metrics).total_seconds(),
                memory_usage=float(self._get_system_metrics()["memory_usage"]),
                cpu_usage=float(self._get_system_metrics()["cpu_usage"]),
                total_ticks=len(df),
                regime_changes=len(df["regime"].unique()),
                price_range=(float(metrics["price_range"][0]), float(metrics["price_range"][1])),
                volume_range=(float(metrics["volume_range"][0]), float(metrics["volume_range"][1])),
                volatility_range=(float(metrics["volatility_range"][0]), float(metrics["volatility_range"][1])),
                trend_strength_range=(float(metrics["trend_strength_range"][0]), float(metrics["trend_strength_range"][1])),
                mean_reversion_range=(float(metrics["mean_reversion_range"][0]), float(metrics["mean_reversion_range"][1])),
                noise_level_range=(float(metrics["noise_level_range"][0]), float(metrics["noise_level_range"][1])),
                market_impact_range=(float(metrics["market_impact_range"][0]), float(metrics["market_impact_range"][1])),
                liquidity_factor_range=(float(metrics["liquidity_factor_range"][0]), float(metrics["liquidity_factor_range"][1])),
                success=True,
            )
            logger.info("Генерация рыночных данных завершена успешно")
            return df
        except Exception as e:
            logger.error(f"Ошибка генерации рыночных данных: {str(e)}")
            end_time = datetime.now()
            start_time_error = self._start_time if self._start_time is not None else datetime.now()
            self.metrics = SimulationMetrics(
                start_time=start_time_error,
                end_time=end_time,
                simulation_time=(end_time - start_time_error).total_seconds(),
                memory_usage=0.0,
                cpu_usage=0.0,
                total_ticks=0,
                regime_changes=0,
                price_range=(0.0, 0.0),
                volume_range=(0.0, 0.0),
                volatility_range=(0.0, 0.0),
                trend_strength_range=(0.0, 0.0),
                mean_reversion_range=(0.0, 0.0),
                noise_level_range=(0.0, 0.0),
                market_impact_range=(0.0, 0.0),
                liquidity_factor_range=(0.0, 0.0),
                success=False,
                error=str(e),
            )
            raise
        finally:
            self._running = False
            self._executor.shutdown()


async def main():
    """Основная функция"""
    try:
        # Инициализация симулятора
        simulator = MarketSimulator()
        # Генерация данных
        await simulator.generate_market_data()
        logger.info("Симуляция успешно завершена")
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/simulation_{time}.log", rotation="1 day", retention="7 days", level="INFO"
    )
    # Запуск асинхронного main
    asyncio.run(main())
