import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import joblib  # type: ignore
import numpy as np
import pandas as pd
import ta  # type: ignore
import umap  # type: ignore
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules  # type: ignore
from mlxtend.preprocessing import TransactionEncoder  # type: ignore
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from shared.models.ml_metrics import PatternMetrics

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class Pattern:
    """Структура паттерна"""

    name: str
    type: str  # 'bullish' или 'bearish'
    start_time: datetime
    end_time: datetime
    confidence: float
    price_levels: Dict[str, float]
    volume_profile: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    conditions: Optional[Dict[str, Any]] = None
    features: Optional[List[str]] = None


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
    update_interval: int = 24
    cache_size: int = 1000
    compression: bool = True
    pattern_dir: str = "patterns"
    n_clusters: int = 10
    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.1


class PatternDiscovery:
    def __init__(
        self, min_pattern_length: int = 5, max_pattern_length: int = 20, **kwargs: Any
    ) -> None:
        """Инициализация обнаружения паттернов"""
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.config = kwargs.get("config", PatternConfig())
        self.patterns_dir = Path("patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        # Паттерны
        self.patterns: Dict[str, List[Pattern]] = {}
        self.metrics: Dict[str, PatternMetrics] = {}
        # Кэш
        self._pattern_cache: Dict[str, List[Pattern]] = {}
        self._feature_cache: Dict[str, DataFrame] = {}
        self.importance_cache: Dict[str, Dict[str, float]] = {}
        # Загрузка паттернов
        self._load_patterns()

    def _load_patterns(self) -> None:
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

    def _save_patterns(self, pattern_type: str) -> None:
        """Сохранение паттернов"""
        try:
            file_path = self.patterns_dir / f"{pattern_type}.json"
            data = {
                "patterns": self.patterns[pattern_type],
                "metrics": self.metrics[pattern_type],
            }
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения паттернов: {e}")

    def _extract_features(self, df: DataFrame) -> DataFrame:
        """Извлечение признаков из данных"""
        try:
            if df.empty:
                return pd.DataFrame()
            features = pd.DataFrame(index=df.index)
            # Базовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            # Технические индикаторы
            features["rsi"] = ta.rsi(df["close"])
            features["macd"], features["macd_signal"], _ = ta.macd(df["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                ta.bbands(df["close"])
            )
            features["atr"] = ta.atr(df["high"], df["low"], df["close"])
            # Волатильность
            features["high_low_ratio"] = df["high"] / df["low"]
            features["close_open_ratio"] = df["close"] / df["open"]
            # Тренд
            features["trend"] = ta.adx(df["high"], df["low"], df["close"])
            features["trend_strength"] = abs(features["trend"])
            # Нормализация
            scaler = StandardScaler()
            # Обработка пропущенных значений
            if isinstance(features, pd.DataFrame):
                # Для pandas DataFrame используем fillna напрямую
                if hasattr(features, 'fillna'):
                    features_filled = features.fillna(0)
                else:
                    features_filled = features
            else:
                # Для других типов данных создаем DataFrame
                features_df = pd.DataFrame(features)
                if hasattr(features_df, 'fillna'):
                    features_filled = features_df.fillna(0)
                else:
                    features_filled = features_df
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features_filled),
                columns=features.columns,
                index=features.index,
            )
            return features_scaled
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return pd.DataFrame()

    def _find_peaks_valleys(self, series: Series) -> Tuple[np.ndarray, np.ndarray]:
        """Поиск пиков и впадин в ряде"""
        try:
            # Проверка типа входных данных
            if not isinstance(series, pd.Series):
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]  # Берем первый столбец
                else:
                    raise TypeError("Ожидается pd.Series или pd.DataFrame")
            # Нормализация данных
            series_array = series.to_numpy() if hasattr(series, 'to_numpy') else np.asarray(series)
            series_mean = float(np.mean(series_array))
            series_std = float(np.std(series_array))
            series_normalized = (series_array - series_mean) / series_std
            # Поиск пиков и впадин
            peaks, _ = find_peaks(
                series_normalized,
                distance=self.config.peak_distance,
                prominence=self.config.peak_prominence,
            )
            valleys, _ = find_peaks(
                -series_normalized,
                distance=self.config.peak_distance,
                prominence=self.config.peak_prominence,
            )
            return peaks, valleys
        except Exception as e:
            logger.error(f"Ошибка поиска пиков и впадин: {e}")
            return np.array([]), np.array([])

    def _cluster_features(self, features: DataFrame) -> Dict[str, Any]:
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

    def _calculate_pattern_metrics(self, pattern: np.ndarray, df: DataFrame) -> Dict[str, Any]:
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
                        df["close"].iloc[match + pattern_length] - df["close"].iloc[match]
                    ) / df["close"].iloc[match]
                    returns.append(future_return)
            if not returns:
                return {}
            # Статистики
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
            max_drawdown = min(returns) if returns else 0
            return {
                "frequency": len(matches),
                "mean_return": mean_return,
                "std_return": std_return,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
            }
        except Exception as e:
            logger.error(f"Ошибка расчета метрик паттерна: {e}")
            return {}

    def discover_patterns(self, df: DataFrame) -> None:
        """Обнаружение паттернов в данных"""
        try:
            for column in df.columns:
                if column in ["timestamp", "datetime"]:
                    continue
                # Извлечение признаков
                features = self._extract_features(df)
                if features.empty:
                    continue
                # Поиск паттернов
                patterns = {}
                for length in range(
                    self.config.min_pattern_length, self.config.max_pattern_length + 1
                ):
                    for i in range(len(features) - length + 1):
                        if hasattr(features, 'iloc'):
                            pattern = features.iloc[i : i + length][column].to_numpy() if hasattr(features.iloc[i : i + length][column], 'to_numpy') else np.asarray(features.iloc[i : i + length][column])
                        else:
                            pattern = features[i : i + length][column].to_numpy() if hasattr(features[i : i + length][column], 'to_numpy') else np.asarray(features[i : i + length][column])
                        if len(pattern) < 2:
                            continue
                        # Нормализация паттерна
                        pattern_array = pattern if isinstance(pattern, np.ndarray) else np.asarray(pattern)
                        pattern_norm = (pattern_array - np.mean(pattern_array)) / np.std(pattern_array)
                        # Расчет метрик
                        pattern_metrics = self._calculate_pattern_metrics(pattern_norm, df)
                        if pattern_metrics:
                            pattern_id = f"{column}_pattern_{i}_{length}"
                            patterns[pattern_id] = {
                                "pattern": pattern_norm.tolist(),
                                "metrics": pattern_metrics,
                            }
                # Сортировка по частоте
                sorted_patterns_list = list(
                    sorted(
                        patterns.items(),
                        key=lambda x: x[1]["metrics"]["frequency"],
                        reverse=True,
                    )[: self.config.max_patterns]
                )
                # Сохранение
                self.patterns[column] = dict(sorted_patterns_list)
                # Обновление метрик
                pattern_metrics_obj = PatternMetrics(
                    total_patterns=len(sorted_patterns_list),
                    pattern_lengths={
                        len(v[1]["pattern"]): sum(
                            1
                            for p in sorted_patterns_list
                            if len(p[1]["pattern"]) == len(v[1]["pattern"])
                        )
                        for v in sorted_patterns_list
                    },
                    pattern_frequencies={
                        str(k): v[1]["metrics"]["frequency"] for k, v in sorted_patterns_list.items()
                    },
                    pattern_returns={
                        str(k): v[1]["metrics"].get("mean_return", 0.0) for k, v in sorted_patterns_list.items()
                    },
                    pattern_win_rates={
                        str(k): v[1]["metrics"].get("win_rate", 0.0) for k, v in sorted_patterns_list.items()
                    },
                    pattern_sharpe={
                        str(k): v[1]["metrics"].get("sharpe", 0.0) for k, v in sorted_patterns_list.items()
                    },
                    pattern_drawdown={
                        str(k): v[1]["metrics"].get("max_drawdown", 0.0) for k, v in sorted_patterns_list.items()
                    },
                    last_update=datetime.now(),
                    confidence=0.8  # Default confidence
                )
                self.metrics[column] = pattern_metrics_obj
        except Exception as e:
            logger.error(f"Ошибка обнаружения паттернов: {e}")

    def find_patterns(self, df: DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Поиск паттернов в данных"""
        try:
            results = {}
            for column, patterns_dict in self.patterns.items():
                if column not in df.columns:
                    continue
                series = df[column]
                matches: List[Dict[str, Any]] = []
                if isinstance(patterns_dict, dict):
                    for pattern_id, pattern_data in patterns_dict.items():
                        pattern = np.array(pattern_data["pattern"])
                        pattern_length = len(pattern)
                        for i in range(len(series) - pattern_length + 1):
                            if hasattr(series, 'iloc'):
                                window = series.iloc[i : i + pattern_length].values
                            else:
                                window = series[i : i + pattern_length].values
                            if np.allclose(window, pattern, rtol=0.1):
                                matches.append(
                                    {
                                        "pattern_id": pattern_id,
                                        "start_index": i,
                                        "end_index": i + pattern_length,
                                        "confidence": 1.0
                                        - np.mean(np.abs(window - pattern) / pattern),
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

    def get_metrics(self, column: Optional[str] = None) -> Dict[str, Any]:
        """Получение метрик"""
        if column:
            metrics_result: Dict[str, Any] = self.metrics.get(column, {})
            return metrics_result if isinstance(metrics_result, dict) else {}
        return self.metrics

    def reset_patterns(self) -> None:
        """Сброс паттернов"""
        self.patterns = {}
        self.metrics: Dict[str, Any] = {}
        self._pattern_cache.clear()
        self._feature_cache.clear()

    def _init_models(self) -> None:
        """Инициализация моделей"""
        try:
            # Инициализация кластеризации
            self.dbscan = DBSCAN(
                eps=self.config.cluster_eps, min_samples=self.config.min_samples
            )
            self.kmeans = KMeans(n_clusters=self.config.n_clusters)
            # Инициализация снижения размерности
            self.pca = PCA(n_components=self.config.n_components)
            self.umap = umap.UMAP(
                n_neighbors=self.config.n_neighbors, min_dist=self.config.min_dist
            )
            # Инициализация нормализации
            self.scaler = StandardScaler()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")

    def discover_patterns_multi(
        self, pair: str, timeframe: str, data: DataFrame
    ) -> List[Pattern]:
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
            self._save_patterns_multi(pair, timeframe, patterns)
            return patterns
        except Exception as e:
            logger.error(f"Error discovering patterns: {str(e)}")
            return []

    def _prepare_features(self, data: DataFrame) -> DataFrame:
        """Подготовка признаков"""
        try:
            features = pd.DataFrame()
            # Технические индикаторы
            features["rsi"] = ta.rsi(data["close"])
            features["macd"], features["macd_signal"], _ = ta.macd(data["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                ta.bbands(data["close"])
            )
            features["atr"] = ta.atr(data["high"], data["low"], data["close"])
            # Свечные характеристики
            features["body_size"] = abs(data["close"] - data["open"])
            features["upper_shadow"] = data["high"] - data[["open", "close"]].max(
                axis=1
            )
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
                self.scaler.fit_transform(features),
                columns=features.columns,
                index=features.index,
            )
            return features
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _find_associations(self, features: DataFrame) -> DataFrame:
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
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            # Поиск частых наборов
            frequent_itemsets = apriori(
                df_encoded,
                min_support=self.config.min_support,
                use_colnames=True,
            )
            # Поиск правил ассоциации
            if not frequent_itemsets.empty:
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=self.config.min_confidence,
                )
                return rules
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error finding associations: {str(e)}")
            return pd.DataFrame()

    def _analyze_clusters(self, features: DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Анализ кластеров"""
        try:
            analysis = {}
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:  # Шум
                    continue
                cluster_mask = labels == label
                cluster_data = features[cluster_mask]
                analysis[f"cluster_{label}"] = {
                    "size": len(cluster_data),
                    "center": cluster_data.mean().to_dict(),
                    "std": cluster_data.std().to_dict(),
                    "features": cluster_data.columns.tolist(),
                }
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}")
            return {}

    def _get_important_features(self, data: DataFrame) -> List[str]:
        """Получение важных признаков"""
        try:
            # Расчет корреляций
            if hasattr(data, 'corr'):
                correlations = data.corr().abs()
            else:
                # Если это не pandas DataFrame, возвращаем пустой список
                return []
            # Выбор признаков с высокой корреляцией
            important_features = []
            for col in correlations.columns:
                if correlations[col].mean() > 0.3:
                    important_features.append(col)
            return important_features
        except Exception as e:
            logger.error(f"Error getting important features: {str(e)}")
            return []

    def _find_candle_patterns(self, data: DataFrame) -> Dict[str, Any]:
        """Поиск свечных паттернов"""
        try:
            patterns = {}
            # Doji
            doji_threshold = 0.1
            doji_mask = abs(data["close"] - data["open"]) <= (
                (data["high"] - data["low"]) * doji_threshold
            )
            if doji_mask.any():
                patterns["doji"] = {
                    "frequency": doji_mask.sum(),
                    "signal_strength": 0.5,
                }
            # Hammer
            body = abs(data["close"] - data["open"])
            lower_shadow = data[["open", "close"]].min(axis=1) - data["low"]
            hammer_mask = (lower_shadow > 2 * body) & (data["close"] > data["open"])
            if hammer_mask.any():
                patterns["hammer"] = {
                    "frequency": hammer_mask.sum(),
                    "signal_strength": 0.7,
                }
            return patterns
        except Exception as e:
            logger.error(f"Error finding candle patterns: {str(e)}")
            return {}

    def _combine_patterns(
        self, associations: DataFrame, clusters: Dict[str, Any], candle_patterns: Dict[str, Any]
    ) -> List[Pattern]:
        """Объединение паттернов"""
        try:
            patterns = []
            # Обработка ассоциаций
            if not associations.empty:
                for _, row in associations.iterrows():
                    pattern = Pattern(
                        name=f"association_{row.name}",
                        type="bullish" if row.get("confidence", 0) > 0.5 else "bearish",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        confidence=float(row.get("confidence", 0)),
                        price_levels={"support": 0.0, "resistance": 0.0},
                        metadata={"type": "association", "support": float(row.get("support", 0))},
                    )
                    patterns.append(pattern)
            # Обработка кластеров
            if clusters and "analysis" in clusters:
                cluster_analysis = clusters["analysis"]
                if isinstance(cluster_analysis, dict):
                    for cluster_id, cluster_data in cluster_analysis.items():
                        if isinstance(cluster_data, dict):
                            pattern = Pattern(
                                name=f"cluster_{cluster_id}",
                                type="bullish" if cluster_data.get("mean_return", 0) > 0 else "bearish",
                                start_time=datetime.now(),
                                end_time=datetime.now(),
                                confidence=float(cluster_data.get("confidence", 0)),
                                price_levels={"support": 0.0, "resistance": 0.0},
                                metadata={"type": "cluster", "size": int(cluster_data.get("size", 0))},
                            )
                            patterns.append(pattern)
            # Обработка свечных паттернов
            if candle_patterns and isinstance(candle_patterns, dict):
                for pattern_name, pattern_data in candle_patterns.items():
                    if isinstance(pattern_data, dict):
                        pattern = Pattern(
                            name=f"candle_{pattern_name}",
                            type="bullish" if pattern_data.get("signal", "") == "bullish" else "bearish",
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            confidence=float(pattern_data.get("confidence", 0)),
                            price_levels={"support": 0.0, "resistance": 0.0},
                            metadata={"type": "candle", "frequency": int(pattern_data.get("frequency", 0))},
                        )
                        patterns.append(pattern)
            return patterns
        except Exception as e:
            logger.error(f"Ошибка объединения паттернов: {e}")
            return []

    def _save_patterns_multi(self, pair: str, timeframe: str, patterns: List[Pattern]) -> None:
        """Сохранение паттернов"""
        try:
            # Создание директории
            pattern_path = os.path.join(self.config.pattern_dir, pair, timeframe)
            os.makedirs(pattern_path, exist_ok=True)
            # Сохранение паттернов
            joblib.dump(patterns, os.path.join(pattern_path, "patterns.joblib"))
        except Exception as e:
            logger.error(f"Error saving patterns: {str(e)}")

    def evaluate_pattern(self, pattern: Pattern, data: DataFrame) -> float:
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

    def _evaluate_association(self, pattern: Pattern, features: DataFrame) -> float:
        """Оценка ассоциации"""
        try:
            # Проверка условий
            if not pattern.conditions:
                return 0.0
            antecedents = pattern.conditions.get("antecedents", [])
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

    def _evaluate_cluster(self, pattern: Pattern, features: DataFrame) -> float:
        """Оценка кластера"""
        try:
            # Получение центра кластера
            if not pattern.conditions or not pattern.features:
                return 0.0
            center = pd.Series(pattern.conditions.get("center", {}))
            std = pd.Series(pattern.conditions.get("std", {}))
            # Расчет расстояний
            feature_data = features[pattern.features].values
            center_data = center[pattern.features].values.reshape(1, -1)
            distances = cdist(feature_data, center_data, metric="euclidean")
            # Нормализация расстояний
            std_mean = std[pattern.features].mean()
            if std_mean > 0:
                distances = distances / std_mean
            # Расчет оценки
            score = np.exp(-distances).mean()
            return score
        except Exception as e:
            logger.error(f"Error evaluating cluster: {str(e)}")
            return 0.0

    def _evaluate_candle_pattern(self, pattern: Pattern, data: DataFrame) -> float:
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
        self,
        symbol: str,
        base_timeframe: str = "1h",
        higher_tf: str = "4h",
        lower_tf: str = "15m",
    ) -> DataFrame:
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
            data = self._load_multi_tf_data(
                symbol, [lower_tf, base_timeframe, higher_tf]
            )
            # Generate features for each timeframe
            features = {}
            for tf in [lower_tf, base_timeframe, higher_tf]:
                tf_data = data[tf]
                features[tf] = self._generate_tf_features(tf_data, tf)
            # Combine features
            combined_features = self._combine_features(features, base_timeframe)
            # Cache results
            self._feature_cache[symbol] = combined_features
            return combined_features
        except Exception as e:
            logger.error(f"Error generating multi-tf features for {symbol}: {str(e)}")
            return pd.DataFrame()

    def rank_features_by_correlation(
        self, target: Series, method: str = "all", symbol: str = ""
    ) -> Dict[str, float]:
        """Rank features by their correlation with the target.
        Args:
            target: Target variable (e.g., returns)
            method: Correlation method ('spearman', 'mutual_info', 'permutation', 'all')
            symbol: Symbol for cache lookup
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if method not in ["spearman", "mutual_info", "permutation", "all"]:
                raise ValueError(f"Invalid method: {method}")
            # Get features from cache
            features = self._feature_cache.get(symbol, pd.DataFrame())
            if features.empty:
                return {}
            importance_scores: Dict[str, float] = {}
            if method in ["spearman", "all"]:
                # Spearman Correlation
                spearman_scores = {}
                for col in features.columns:
                    try:
                        # Преобразуем в numpy массивы для совместимости
                        feature_values = features[col].to_numpy() if hasattr(features[col], 'to_numpy') else np.asarray(features[col])
                        target_values = target.to_numpy() if hasattr(target, 'to_numpy') else np.asarray(target)
                        correlation, _ = stats.spearmanr(feature_values, target_values)
                        spearman_scores[col] = abs(correlation) if not np.isnan(correlation) else 0.0
                    except Exception as e:
                        logger.warning(f"Error calculating spearman correlation for {col}: {e}")
                        spearman_scores[col] = 0.0
                importance_scores["spearman"] = spearman_scores
            if method in ["mutual_info", "all"]:
                # Mutual Information
                mi_scores = mutual_info_regression(features, target)
                importance_scores["mutual_info"] = dict(
                    zip(features.columns, mi_scores)
                )
            if method in ["permutation", "all"]:
                # Permutation Importance
                base_model = self._get_base_model()
                base_model.fit(features, target)
                perm_importance = permutation_importance(base_model, features, target)
                importance_scores["permutation"] = dict(
                    zip(features.columns, perm_importance.importances_mean)
                )
            # Cache results
            self.importance_cache[symbol] = importance_scores
            return importance_scores
        except Exception as e:
            logger.error(f"Error ranking features: {str(e)}")
            return {}

    def _load_multi_tf_data(
        self, symbol: str, timeframes: List[str]
    ) -> Dict[str, DataFrame]:
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

    def _generate_tf_features(self, data: DataFrame, timeframe: str) -> DataFrame:
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
            features[f"atr_{timeframe}"] = ta.atr(
                data["high"], data["low"], data["close"]
            )
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
        self, features: Dict[str, DataFrame], base_timeframe: str
    ) -> DataFrame:
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
                    if hasattr(tf_features, 'resample'):
                        resampled = tf_features.resample(base_timeframe).last()
                    else:
                        # Если это не pandas DataFrame, пропускаем
                        continue
                    # Add prefix to column names
                    resampled.columns = [f"{col}_{tf}" for col in resampled.columns]
                    if hasattr(combined, 'join'):
                        combined = combined.join(resampled)
                    else:
                        # Если нет метода join, используем merge
                        combined = pd.concat([combined, resampled], axis=1)
            return combined
        except Exception as e:
            logger.error(f"Error combining features: {str(e)}")
            return pd.DataFrame()

    def _get_base_model(self) -> RandomForestRegressor:
        """Get base model for permutation importance calculation."""
        # Implement this method to return your base model
        # This could be a simple model like RandomForest or your custom model
        return RandomForestRegressor(n_estimators=10, random_state=42)
