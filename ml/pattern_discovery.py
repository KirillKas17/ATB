import asyncio
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib
import umap
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_rsi,
    calculate_volume_acceleration,
    calculate_vwap,
)


@dataclass
class Pattern:
    """Структура паттерна"""

    name: str  # название паттерна
    type: str  # 'bullish' или 'bearish'
    start_time: datetime
    end_time: datetime
    confidence: float
    price_levels: Dict[str, float]
    volume_profile: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


@dataclass
class PatternConfig:
    """Конфигурация обнаружения паттернов"""

    min_pattern_length: int = 5
    max_pattern_length: int = 50
    min_support: float = 0.1
    min_confidence: float = 0.6
    max_patterns: int = 100
    cluster_eps: float = 0.1
    min_samples: int = 5
    zscore_threshold: float = 2.0
    peak_distance: int = 10
    peak_prominence: float = 0.1
    update_interval: int = 24  # часов
    cache_size: int = 1000
    compression: bool = True


@dataclass
class PatternMetrics:
    """Метрики паттернов"""

    total_patterns: int
    pattern_lengths: Dict[int, int]
    pattern_frequencies: Dict[str, int]
    pattern_returns: Dict[str, float]
    pattern_win_rates: Dict[str, float]
    pattern_sharpe: Dict[str, float]
    pattern_drawdown: Dict[str, float]
    last_update: datetime
    confidence: float


class PatternDiscovery:
    def __init__(self, min_pattern_length=5, max_pattern_length=20, **kwargs):
        """Инициализация обнаружения паттернов"""
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.config = kwargs.get("config", PatternConfig())
        self.patterns_dir = Path("patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # Паттерны
        self.patterns = {}
        self.metrics = {}

        # Кэш
        self._pattern_cache = {}
        self._feature_cache = {}

        # Загрузка паттернов
        self._load_patterns()

    def _load_patterns(self):
        """Загрузка паттернов"""
        try:
            for file in self.patterns_dir.glob("*.json"):
                pattern_type = file.stem
                with open(file, "r") as f:
                    data = json.load(f)
                    self.patterns[pattern_type] = data["patterns"]
                    self.metrics[pattern_type] = data["metrics"]
        except Exception as e:
            logger.error(f"Ошибка загрузки паттернов: {e}")

    def _save_patterns(self, pattern_type: str):
        """Сохранение паттернов"""
        try:
            file_path = self.patterns_dir / f"{pattern_type}.json"
            data = {"patterns": self.patterns[pattern_type], "metrics": self.metrics[pattern_type]}
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения паттернов: {e}")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков"""
        try:
            features = pd.DataFrame()

            # Ценовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log1p(features["returns"])
            features["volatility"] = features["returns"].rolling(20).std()

            # Технические индикаторы
            features["rsi"] = talib.RSI(df["close"])
            features["macd"], features["macd_signal"], _ = talib.MACD(df["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = talib.BBANDS(
                df["close"]
            )
            features["atr"] = talib.ATR(df["high"], df["low"], df["close"])
            features["adx"] = talib.ADX(df["high"], df["low"], df["close"])

            # Объемные признаки
            features["volume_ma"] = df["volume"].rolling(20).mean()
            features["volume_std"] = df["volume"].rolling(20).std()
            features["volume_ratio"] = df["volume"] / features["volume_ma"]

            # Моментум
            features["momentum"] = talib.MOM(df["close"], timeperiod=10)
            features["roc"] = talib.ROC(df["close"], timeperiod=10)

            # Волатильность
            features["high_low_ratio"] = df["high"] / df["low"]
            features["close_open_ratio"] = df["close"] / df["open"]

            # Тренд
            features["trend"] = talib.ADX(df["high"], df["low"], df["close"])
            features["trend_strength"] = abs(features["trend"])

            # Нормализация
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features.fillna(0)),
                columns=features.columns,
                index=features.index,
            )

            return features_scaled

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return pd.DataFrame()

    def _find_peaks_valleys(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Поиск пиков и впадин в ряде"""
        try:
            # Проверка типа входных данных
            if not isinstance(series, pd.Series):
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]  # Берем первый столбец
                else:
                    raise TypeError("Ожидается pd.Series или pd.DataFrame")

            # Нормализация данных
            series = (series - series.mean()) / series.std()

            # Поиск пиков и впадин
            peaks, _ = find_peaks(
                series.values,
                distance=self.config.peak_distance,
                prominence=self.config.peak_prominence,
            )

            valleys, _ = find_peaks(
                -series.values,
                distance=self.config.peak_distance,
                prominence=self.config.peak_prominence,
            )

            return peaks, valleys

        except Exception as e:
            logger.error(f"Ошибка поиска пиков и впадин: {e}")
            return np.array([]), np.array([])

    def _cluster_features(self, features: pd.DataFrame) -> Dict:
        """Кластеризация признаков"""
        try:
            # Нормализация данных
            scaler = StandardScaler()
            X = scaler.fit_transform(features)

            # Кластеризация
            clustering = DBSCAN(
                eps=self.config.cluster_eps, min_samples=self.config.min_samples
            ).fit(X)

            # Анализ кластеров
            cluster_analysis = self._analyze_clusters(features, clustering.labels_)

            return {"labels": clustering.labels_, "analysis": cluster_analysis}

        except Exception as e:
            logger.error(f"Ошибка кластеризации признаков: {e}")
            return {"labels": np.array([]), "analysis": {}}

    def _calculate_pattern_metrics(self, pattern: np.ndarray, df: pd.DataFrame) -> Dict:
        """Расчет метрик паттерна"""
        try:
            # Поиск вхождений паттерна
            pattern_length = len(pattern)
            matches = []

            for i in range(len(df) - pattern_length + 1):
                window = df["close"].iloc[i : i + pattern_length].values
                if np.allclose(window, pattern, rtol=0.1):
                    matches.append(i)

            if not matches:
                return {}

            # Расчет метрик
            returns = []
            for match in matches:
                if match + pattern_length < len(df):
                    future_return = (
                        df["close"].iloc[match + pattern_length] / df["close"].iloc[match] - 1
                    )
                    returns.append(future_return)

            returns = np.array(returns)

            return {
                "frequency": len(matches),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "win_rate": float(np.mean(returns > 0)),
                "sharpe": float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
                "max_drawdown": float(np.min(returns)),
            }

        except Exception as e:
            logger.error(f"Ошибка расчета метрик паттерна: {e}")
            return {}

    def discover_patterns(self, df: pd.DataFrame):
        """Обнаружение паттернов"""
        try:
            start_time = datetime.now()

            # Извлечение признаков
            features = self._extract_features(df)

            # Поиск паттернов для каждого признака
            for column in features.columns:
                series = features[column]

                # Поиск пиков и впадин
                peaks, valleys = self._find_peaks_valleys(series)

                # Извлечение паттернов
                patterns = []
                for i in range(len(peaks) - 1):
                    pattern = series.iloc[peaks[i] : peaks[i + 1]].values
                    if self.min_pattern_length <= len(pattern) <= self.max_pattern_length:
                        patterns.append(pattern)

                # Кластеризация паттернов
                centroids = self._cluster_features(features)

                # Расчет метрик
                pattern_metrics = {}
                for i, pattern in enumerate(centroids["analysis"]):
                    metrics = self._calculate_pattern_metrics(pattern, df)
                    if metrics:
                        pattern_metrics[f"pattern_{i}"] = {
                            "pattern": pattern.tolist(),
                            "metrics": metrics,
                        }

                # Фильтрация по метрикам
                filtered_patterns = {
                    k: v
                    for k, v in pattern_metrics.items()
                    if v["metrics"]["frequency"] >= self.config.min_support * len(df)
                    and v["metrics"]["win_rate"] >= self.config.min_confidence
                }

                # Сортировка по частоте
                sorted_patterns = dict(
                    sorted(
                        filtered_patterns.items(),
                        key=lambda x: x[1]["metrics"]["frequency"],
                        reverse=True,
                    )[: self.config.max_patterns]
                )

                # Сохранение
                self.patterns[column] = sorted_patterns

                # Обновление метрик
                self.metrics[column] = PatternMetrics(
                    total_patterns=len(sorted_patterns),
                    pattern_lengths={
                        len(v["pattern"]): sum(
                            1
                            for p in sorted_patterns.values()
                            if len(p["pattern"]) == len(v["pattern"])
                        )
                        for v in sorted_patterns.values()
                    },
                    pattern_frequencies={
                        k: v["metrics"]["frequency"] for k, v in sorted_patterns.items()
                    },
                    pattern_returns={
                        k: v["metrics"]["mean_return"] for k, v in sorted_patterns.items()
                    },
                    pattern_win_rates={
                        k: v["metrics"]["win_rate"] for k, v in sorted_patterns.items()
                    },
                    pattern_sharpe={k: v["metrics"]["sharpe"] for k, v in sorted_patterns.items()},
                    pattern_drawdown={
                        k: v["metrics"]["max_drawdown"] for k, v in sorted_patterns.items()
                    },
                    last_update=datetime.now(),
                    confidence=1.0,
                ).__dict__

                # Сохранение в файл
                self._save_patterns(column)

                logger.info(f"Обнаружено {len(sorted_patterns)} паттернов для {column}")

        except Exception as e:
            logger.error(f"Ошибка обнаружения паттернов: {e}")
            raise

    def find_patterns(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Поиск паттернов в данных"""
        try:
            results = {}

            for column, patterns in self.patterns.items():
                if column not in df.columns:
                    continue

                series = df[column]
                matches = []

                for pattern_id, pattern_data in patterns.items():
                    pattern = np.array(pattern_data["pattern"])
                    pattern_length = len(pattern)

                    for i in range(len(series) - pattern_length + 1):
                        window = series.iloc[i : i + pattern_length].values
                        if np.allclose(window, pattern, rtol=0.1):
                            matches.append(
                                {
                                    "pattern_id": pattern_id,
                                    "start_index": i,
                                    "end_index": i + pattern_length,
                                    "confidence": 1.0 - np.mean(np.abs(window - pattern) / pattern),
                                    "metrics": pattern_data["metrics"],
                                }
                            )

                results[column] = matches

            return results

        except Exception as e:
            logger.error(f"Ошибка поиска паттернов: {e}")
            return {}

    def get_patterns(self, pair: str, timeframe: str) -> List[Pattern]:
        """Получение паттернов для пары и таймфрейма"""
        try:
            key = f"{pair}_{timeframe}"
            patterns = self.patterns.get(key, [])

            if not patterns:
                logger.warning(f"Паттерны не найдены для {key}")
                return []

            return patterns

        except Exception as e:
            logger.error(f"Ошибка получения паттернов: {e}")
            return []

    def get_metrics(self, column: Optional[str] = None) -> Dict:
        """Получение метрик"""
        if column:
            return self.metrics.get(column, {})
        return self.metrics

    def reset_patterns(self):
        """Сброс паттернов"""
        self.patterns = {}
        self.metrics = {}
        self._pattern_cache.clear()
        self._feature_cache.clear()

    def _init_models(self):
        """Инициализация моделей"""
        try:
            # Инициализация кластеризации
            self.dbscan = DBSCAN(
                eps=self.config["cluster_eps"], min_samples=self.config["min_samples"]
            )
            self.kmeans = KMeans(n_clusters=self.config["n_clusters"])

            # Инициализация снижения размерности
            self.pca = PCA(n_components=self.config["n_components"])
            self.umap = umap.UMAP(
                n_neighbors=self.config["n_neighbors"], min_dist=self.config["min_dist"]
            )

            # Инициализация нормализации
            self.scaler = StandardScaler()

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")

    def discover_patterns(self, pair: str, timeframe: str, data: pd.DataFrame) -> List[Pattern]:
        """
        Обнаружение паттернов.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            data: Данные

        Returns:
            List[Pattern]: Список обнаруженных паттернов
        """
        try:
            # Подготовка данных
            features = self._prepare_features(data)

            # Поиск ассоциаций
            associations = self._find_associations(features)

            # Кластеризация паттернов
            clusters = self._cluster_features(features)

            # Поиск свечных паттернов
            candle_patterns = self._find_candle_patterns(data)

            # Объединение паттернов
            patterns = self._combine_patterns(associations, clusters, candle_patterns)

            # Сохранение паттернов
            self._save_patterns(pair, timeframe, patterns)

            return patterns

        except Exception as e:
            logger.error(f"Error discovering patterns: {str(e)}")
            return []

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков"""
        try:
            features = pd.DataFrame()

            # Технические индикаторы
            features["rsi"] = ta.rsi(data["close"])
            features["macd"], features["macd_signal"], _ = ta.macd(data["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = ta.bbands(
                data["close"]
            )
            features["atr"] = ta.atr(data["high"], data["low"], data["close"])

            # Свечные характеристики
            features["body_size"] = abs(data["close"] - data["open"])
            features["upper_shadow"] = data["high"] - data[["open", "close"]].max(axis=1)
            features["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]
            features["is_bullish"] = (data["close"] > data["open"]).astype(int)

            # Объемные характеристики
            features["volume_ma"] = ta.sma(data["volume"], timeperiod=20)
            features["volume_ratio"] = data["volume"] / features["volume_ma"]

            # Волатильность
            features["volatility"] = data["close"].pct_change().rolling(window=20).std()

            # Тренд
            features["trend"] = ta.adx(data["high"], data["low"], data["close"])

            # Нормализация
            features = pd.DataFrame(
                self.scaler.fit_transform(features), columns=features.columns, index=features.index
            )

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _find_associations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Поиск ассоциаций"""
        try:
            # Дискретизация данных
            transactions = []
            for _, row in features.iterrows():
                transaction = []
                for col in features.columns:
                    if row[col] > 0.5:
                        transaction.append(f"{col}_high")
                    elif row[col] < -0.5:
                        transaction.append(f"{col}_low")
                    else:
                        transaction.append(f"{col}_normal")
                transactions.append(transaction)

            # Кодирование транзакций
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            # Поиск частых наборов
            frequent_itemsets = apriori(
                df, min_support=self.config["min_support"], use_colnames=True
            )

            # Поиск правил
            rules = association_rules(
                frequent_itemsets, metric="lift", min_threshold=self.config["min_lift"]
            )

            return rules

        except Exception as e:
            logger.error(f"Error finding associations: {str(e)}")
            return pd.DataFrame()

    def _analyze_clusters(self, features: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Анализ кластеров"""
        try:
            clusters = {}

            for label in set(labels):
                if label == -1:  # шум
                    continue

                # Получение данных кластера
                cluster_data = features[labels == label]

                # Расчет характеристик
                clusters[label] = {
                    "size": len(cluster_data),
                    "center": cluster_data.mean(),
                    "std": cluster_data.std(),
                    "features": self._get_important_features(cluster_data),
                }

            return clusters

        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}")
            return {}

    def _get_important_features(self, data: pd.DataFrame) -> List[str]:
        """Получение важных признаков"""
        try:
            # Расчет важности признаков
            importance = data.std() / data.mean()

            # Выбор важных признаков
            important_features = importance[
                importance > self.config["feature_importance_threshold"]
            ].index.tolist()

            return important_features

        except Exception as e:
            logger.error(f"Error getting important features: {str(e)}")
            return []

    def _find_candle_patterns(self, data: pd.DataFrame) -> Dict:
        """Поиск свечных паттернов"""
        try:
            patterns = {}

            for pattern in self.config["candle_patterns"]:
                # Получение функции паттерна
                pattern_func = getattr(ta, pattern)

                # Поиск паттернов
                result = pattern_func(data["open"], data["high"], data["low"], data["close"])

                if result is not None:
                    patterns[pattern] = {
                        "indices": np.where(result != 0)[0],
                        "values": result[result != 0],
                    }

            return patterns

        except Exception as e:
            logger.error(f"Error finding candle patterns: {str(e)}")
            return {}

    def _combine_patterns(
        self, associations: pd.DataFrame, clusters: Dict, candle_patterns: Dict
    ) -> List[Pattern]:
        """Объединение паттернов"""
        try:
            patterns = []

            # Проверка входных данных
            if associations is None or clusters is None or candle_patterns is None:
                logger.warning("Отсутствуют данные для объединения паттернов")
                return []

            # Ассоциативные правила
            if not associations.empty:
                for _, rule in associations.iterrows():
                    pattern = Pattern(
                        name=f"association_{len(patterns)}",
                        type="association",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        confidence=rule.get("confidence", 0.0),
                        price_levels={
                            "support": rule.get("support", 0.0),
                            "resistance": rule.get("resistance", 0.0),
                        },
                        metadata={
                            "antecedents": rule.get("antecedents", []),
                            "consequents": rule.get("consequents", []),
                        },
                    )
                    patterns.append(pattern)

            # Кластеры
            if clusters:
                for cluster_id, cluster_data in clusters.items():
                    pattern = Pattern(
                        name=f"cluster_{cluster_id}",
                        type="cluster",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        confidence=cluster_data.get("confidence", 0.0),
                        price_levels=cluster_data.get("price_levels", {}),
                        metadata={
                            "center": cluster_data.get("center", []),
                            "size": cluster_data.get("size", 0),
                        },
                    )
                    patterns.append(pattern)

            # Свечные паттерны
            if candle_patterns:
                for pattern_name, pattern_data in candle_patterns.items():
                    pattern = Pattern(
                        name=pattern_name,
                        type="candle",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        confidence=pattern_data.get("confidence", 0.0),
                        price_levels=pattern_data.get("price_levels", {}),
                        metadata=pattern_data.get("metadata", {}),
                    )
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Ошибка объединения паттернов: {e}")
            return []

    def _save_patterns(self, pair: str, timeframe: str, patterns: List[Pattern]):
        """Сохранение паттернов"""
        try:
            # Создание директории
            pattern_path = os.path.join(self.config["pattern_dir"], pair, timeframe)
            os.makedirs(pattern_path, exist_ok=True)

            # Сохранение паттернов
            joblib.dump(patterns, os.path.join(pattern_path, "patterns.joblib"))

        except Exception as e:
            logger.error(f"Error saving patterns: {str(e)}")

    def evaluate_pattern(self, pattern: Pattern, data: pd.DataFrame) -> float:
        """Оценка паттерна"""
        try:
            if pattern is None or data is None or data.empty:
                logger.warning("Некорректные данные для оценки паттерна")
                return 0.0

            # Оценка в зависимости от типа паттерна
            if pattern.type == "association":
                return self._evaluate_association(pattern, data)
            elif pattern.type == "cluster":
                return self._evaluate_cluster(pattern, data)
            elif pattern.type == "candle":
                return self._evaluate_candle_pattern(pattern, data)
            else:
                logger.warning(f"Неизвестный тип паттерна: {pattern.type}")
                return 0.0

        except Exception as e:
            logger.error(f"Ошибка оценки паттерна: {e}")
            return 0.0

    def _evaluate_association(self, pattern: Pattern, features: pd.DataFrame) -> float:
        """Оценка ассоциации"""
        try:
            # Проверка условий
            antecedents = pattern.conditions["antecedents"]
            consequents = pattern.conditions["consequents"]

            # Расчет поддержки
            support = 0.0
            for _, row in features.iterrows():
                if all(feature in row for feature in antecedents):
                    support += 1
            support /= len(features)

            return support

        except Exception as e:
            logger.error(f"Error evaluating association: {str(e)}")
            return 0.0

    def _evaluate_cluster(self, pattern: Pattern, features: pd.DataFrame) -> float:
        """Оценка кластера"""
        try:
            # Получение центра кластера
            center = pd.Series(pattern.conditions["center"])
            std = pd.Series(pattern.conditions["std"])

            # Расчет расстояний
            distances = cdist(
                features[pattern.features], [center[pattern.features]], metric="euclidean"
            )

            # Нормализация расстояний
            distances = distances / std[pattern.features].mean()

            # Расчет оценки
            score = np.exp(-distances).mean()

            return score

        except Exception as e:
            logger.error(f"Error evaluating cluster: {str(e)}")
            return 0.0

    def _evaluate_candle_pattern(self, pattern: Pattern, data: pd.DataFrame) -> float:
        """Оценка свечного паттерна"""
        try:
            if data.empty or not pattern.metadata:
                return 0.0

            # Получение метаданных с безопасными значениями по умолчанию
            frequency = pattern.metadata.get("frequency", 0)
            signal_strength = pattern.metadata.get("signal_strength", 0.0)

            if frequency == 0 or signal_strength == 0.0:
                return 0.0

            # Расчет оценки
            score = (frequency * signal_strength) / len(data)
            return float(score)

        except Exception as e:
            logger.error(f"Ошибка оценки свечного паттерна: {e}")
            return 0.0

    def generate_multi_tf_features(
        self, symbol: str, base_timeframe: str = "1h", higher_tf: str = "4h", lower_tf: str = "15m"
    ) -> pd.DataFrame:
        """Generate features across multiple timeframes.

        Args:
            symbol: Trading pair symbol
            base_timeframe: Base timeframe for analysis
            higher_tf: Higher timeframe for trend confirmation
            lower_tf: Lower timeframe for entry signals

        Returns:
            DataFrame with aggregated features
        """
        try:
            # Load data for all timeframes
            data = self._load_multi_tf_data(symbol, [lower_tf, base_timeframe, higher_tf])

            # Generate features for each timeframe
            features = {}
            for tf in [lower_tf, base_timeframe, higher_tf]:
                tf_data = data[tf]
                features[tf] = self._generate_tf_features(tf_data, tf)

            # Combine features
            combined_features = self._combine_features(features, base_timeframe)

            # Cache results
            self.feature_cache[symbol] = combined_features

            return combined_features

        except Exception as e:
            logger.error(f"Error generating multi-tf features for {symbol}: {str(e)}")
            return pd.DataFrame()

    def rank_features_by_correlation(
        self, target: pd.Series, method: str = "all"
    ) -> Dict[str, float]:
        """Rank features by their correlation with the target.

        Args:
            target: Target variable (e.g., returns)
            method: Correlation method ('spearman', 'mutual_info', 'permutation', 'all')

        Returns:
            Dictionary of feature importance scores
        """
        try:
            if method not in ["spearman", "mutual_info", "permutation", "all"]:
                raise ValueError(f"Invalid method: {method}")

            # Get features from cache
            features = self.feature_cache.get(symbol, pd.DataFrame())
            if features.empty:
                return {}

            importance_scores = {}

            if method in ["spearman", "all"]:
                # Spearman correlation
                spearman_scores = {}
                for col in features.columns:
                    correlation, _ = stats.spearmanr(features[col], target)
                    spearman_scores[col] = abs(correlation)
                importance_scores["spearman"] = spearman_scores

            if method in ["mutual_info", "all"]:
                # Mutual Information
                mi_scores = mutual_info_regression(features, target)
                importance_scores["mutual_info"] = dict(zip(features.columns, mi_scores))

            if method in ["permutation", "all"]:
                # Permutation Importance
                model = self._get_base_model()  # Implement this method
                perm_scores = permutation_importance(model, features, target, n_repeats=10)
                importance_scores["permutation"] = dict(
                    zip(features.columns, perm_scores.importances_mean)
                )

            # Cache results
            self.importance_cache[symbol] = importance_scores

            return importance_scores

        except Exception as e:
            logger.error(f"Error ranking features: {str(e)}")
            return {}

    def _load_multi_tf_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Load market data for multiple timeframes.

        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to load

        Returns:
            Dictionary of DataFrames for each timeframe
        """
        try:
            data = {}
            for tf in timeframes:
                # Implement data loading logic here
                # This should connect to your data source
                data[tf] = pd.DataFrame()  # Placeholder
            return data

        except Exception as e:
            logger.error(f"Error loading multi-tf data for {symbol}: {str(e)}")
            return {}

    def _generate_tf_features(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Generate features for a single timeframe.

        Args:
            data: Market data for the timeframe
            timeframe: Timeframe identifier

        Returns:
            DataFrame with generated features
        """
        try:
            features = pd.DataFrame(index=data.index)

            # Basic indicators
            features[f"rsi_{timeframe}"] = ta.rsi(data["close"])
            features[f"ema_50_{timeframe}"] = ta.sma(data["close"], timeperiod=50)
            features[f"ema_200_{timeframe}"] = ta.sma(data["close"], timeperiod=200)
            features[f"atr_{timeframe}"] = ta.atr(data["high"], data["low"], data["close"])

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.bbands(data["close"])
            features[f"bb_upper_{timeframe}"] = bb_upper
            features[f"bb_middle_{timeframe}"] = bb_middle
            features[f"bb_lower_{timeframe}"] = bb_lower

            # VWAP and Volume
            features[f"vwap_{timeframe}"] = ta.vwap(data)
            features[f"volume_acc_{timeframe}"] = ta.volume_acceleration(data)

            # Derived features
            features[f"price_velocity_{timeframe}"] = data["close"].pct_change()
            features[f"rsi_roc_{timeframe}"] = features[f"rsi_{timeframe}"].pct_change()
            features[f"bb_width_{timeframe}"] = (bb_upper - bb_lower) / bb_middle
            features[f"volume_ratio_{timeframe}"] = (
                data["volume"] / data["volume"].rolling(20).mean()
            )

            return features

        except Exception as e:
            logger.error(f"Error generating features for {timeframe}: {str(e)}")
            return pd.DataFrame()

    def _combine_features(
        self, features: Dict[str, pd.DataFrame], base_timeframe: str
    ) -> pd.DataFrame:
        """Combine features from different timeframes.

        Args:
            features: Dictionary of feature DataFrames
            base_timeframe: Base timeframe for alignment

        Returns:
            Combined DataFrame of features
        """
        try:
            # Align all features to base timeframe
            base_features = features[base_timeframe]
            combined = base_features.copy()

            # Add higher timeframe features
            for tf, tf_features in features.items():
                if tf != base_timeframe:
                    # Resample to base timeframe
                    resampled = tf_features.resample(base_timeframe).last()
                    # Add prefix to column names
                    resampled.columns = [f"{col}_{tf}" for col in resampled.columns]
                    combined = combined.join(resampled)

            return combined

        except Exception as e:
            logger.error(f"Error combining features: {str(e)}")
            return pd.DataFrame()

    def _get_base_model(self):
        """Get base model for permutation importance calculation."""
        # Implement this method to return your base model
        # This could be a simple model like RandomForest or your custom model
        pass
