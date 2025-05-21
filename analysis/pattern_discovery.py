import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    min_pattern_length: int
    max_pattern_length: int
    min_confidence: float
    min_support: float
    max_patterns: int
    clustering_method: str
    min_cluster_size: int
    pattern_types: List[str]
    feature_columns: List[str]
    window_sizes: List[int]
    similarity_threshold: float
    technical_indicators: List[str] = None
    volume_threshold: float = 1.5
    price_threshold: float = 0.02
    trend_window: int = 20


class Pattern:
    def __init__(
        self,
        pattern_type: str,
        start_idx: int,
        end_idx: int,
        features: np.ndarray,
        confidence: float,
        support: float,
        metadata: Dict[str, Any],
    ):
        self.pattern_type = pattern_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.features = features
        self.confidence = confidence
        self.support = support
        self.metadata = metadata
        self.trend = None
        self.volume_profile = None
        self.technical_indicators = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "features": self.features.tolist(),
            "confidence": self.confidence,
            "support": self.support,
            "metadata": self.metadata,
            "trend": self.trend,
            "volume_profile": self.volume_profile,
            "technical_indicators": self.technical_indicators,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        pattern = cls(
            pattern_type=data["pattern_type"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            features=np.array(data["features"]),
            confidence=data["confidence"],
            support=data["support"],
            metadata=data["metadata"],
        )
        pattern.trend = data.get("trend")
        pattern.volume_profile = data.get("volume_profile")
        pattern.technical_indicators = data.get("technical_indicators", {})
        return pattern


class PatternDiscovery:
    def __init__(self, config: PatternConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._validate_config()

    def _validate_config(self):
        """Проверка корректности конфигурации"""
        if self.config.min_pattern_length >= self.config.max_pattern_length:
            raise ValueError("min_pattern_length must be less than max_pattern_length")
        if not self.config.pattern_types:
            raise ValueError("pattern_types cannot be empty")
        if not self.config.feature_columns:
            raise ValueError("feature_columns cannot be empty")

    def discover_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Обнаружение паттернов в данных"""
        if data.empty:
            raise ValueError("Empty data provided")

        if not all(col in data.columns for col in self.config.feature_columns):
            raise ValueError("Missing required feature columns")

        patterns = []
        for pattern_type in self.config.pattern_types:
            if pattern_type == "candle":
                patterns.extend(self.find_candle_patterns(data))
            elif pattern_type == "price":
                patterns.extend(self.find_price_patterns(data))
            elif pattern_type == "volume":
                patterns.extend(self.find_volume_patterns(data))

        # Фильтрация и ранжирование паттернов
        patterns = self._filter_patterns(patterns)
        patterns = self._rank_patterns(patterns)

        return patterns[: self.config.max_patterns]

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для анализа"""
        features = data[self.config.feature_columns].copy()

        # Добавление технических индикаторов
        if self.config.technical_indicators:
            features = self._add_technical_indicators(features, data)

        # Нормализация данных
        features = pd.DataFrame(
            self.scaler.fit_transform(features), columns=features.columns, index=features.index
        )

        return features

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        if "RSI" in self.config.technical_indicators:
            features["RSI"] = talib.RSI(data["close"])
        if "MACD" in self.config.technical_indicators:
            macd, signal, hist = talib.MACD(data["close"])
            features["MACD"] = macd
            features["MACD_signal"] = signal
            features["MACD_hist"] = hist
        if "BB" in self.config.technical_indicators:
            upper, middle, lower = talib.BBANDS(data["close"])
            features["BB_upper"] = upper
            features["BB_middle"] = middle
            features["BB_lower"] = lower
        return features

    def cluster_patterns(self, patterns: List[Pattern]) -> Dict[str, List[Pattern]]:
        """Кластеризация паттернов"""
        if not patterns:
            return {}

        # Подготовка данных для кластеризации
        features = np.array([p.features.flatten() for p in patterns])
        features = self.scaler.fit_transform(features)

        # Применение DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=self.config.min_cluster_size).fit(features)

        # Группировка паттернов по кластерам
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(patterns[i])

        return clusters

    def find_candle_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Поиск свечных паттернов"""
        patterns = []

        # Поиск стандартных свечных паттернов
        for pattern_name in talib.get_function_groups()["Pattern Recognition"]:
            pattern_func = getattr(talib, pattern_name)
            result = pattern_func(data["open"], data["high"], data["low"], data["close"])

            # Находим индексы, где паттерн обнаружен
            pattern_indices = np.where(result != 0)[0]

            for idx in pattern_indices:
                if idx + self.config.min_pattern_length <= len(data):
                    pattern = Pattern(
                        pattern_type="candle",
                        start_idx=idx,
                        end_idx=idx + self.config.min_pattern_length,
                        features=data.iloc[idx : idx + self.config.min_pattern_length].values,
                        confidence=abs(result[idx]) / 100,
                        support=self._calculate_support(
                            data, idx, idx + self.config.min_pattern_length
                        ),
                        metadata={"pattern_name": pattern_name},
                    )
                    patterns.append(pattern)

        return patterns

    def find_price_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Поиск ценовых паттернов"""
        patterns = []

        # Поиск локальных максимумов и минимумов
        peaks, _ = find_peaks(data["close"].values, distance=self.config.min_pattern_length)
        troughs, _ = find_peaks(-data["close"].values, distance=self.config.min_pattern_length)

        # Анализ трендов
        for i in range(len(peaks) - 1):
            if peaks[i + 1] - peaks[i] <= self.config.max_pattern_length:
                pattern = self._analyze_price_pattern(data, peaks[i], peaks[i + 1])
                if pattern:
                    patterns.append(pattern)

        return patterns

    def find_volume_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Поиск паттернов объема"""
        patterns = []

        # Расчет среднего объема
        avg_volume = data["volume"].rolling(window=self.config.trend_window).mean()

        # Поиск аномальных объемов
        volume_peaks = data[data["volume"] > avg_volume * self.config.volume_threshold].index

        for i in range(len(volume_peaks) - 1):
            if volume_peaks[i + 1] - volume_peaks[i] <= self.config.max_pattern_length:
                pattern = self._analyze_volume_pattern(data, volume_peaks[i], volume_peaks[i + 1])
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _analyze_price_pattern(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Pattern]:
        """Анализ ценового паттерна"""
        price_change = (data["close"].iloc[end_idx] - data["close"].iloc[start_idx]) / data[
            "close"
        ].iloc[start_idx]

        if abs(price_change) >= self.config.price_threshold:
            return Pattern(
                pattern_type="price",
                start_idx=start_idx,
                end_idx=end_idx,
                features=data.iloc[start_idx:end_idx].values,
                confidence=abs(price_change),
                support=self._calculate_support(data, start_idx, end_idx),
                metadata={"price_change": price_change},
            )
        return None

    def _analyze_volume_pattern(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Pattern]:
        """Анализ паттерна объема"""
        volume_change = (data["volume"].iloc[end_idx] - data["volume"].iloc[start_idx]) / data[
            "volume"
        ].iloc[start_idx]

        if volume_change >= self.config.volume_threshold:
            return Pattern(
                pattern_type="volume",
                start_idx=start_idx,
                end_idx=end_idx,
                features=data.iloc[start_idx:end_idx].values,
                confidence=volume_change,
                support=self._calculate_support(data, start_idx, end_idx),
                metadata={"volume_change": volume_change},
            )
        return None

    def _calculate_support(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Расчет поддержки паттерна"""
        pattern_data = data.iloc[start_idx:end_idx]
        return len(pattern_data) / len(data)

    def _filter_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Фильтрация паттернов по критериям"""
        return [
            p
            for p in patterns
            if p.confidence >= self.config.min_confidence and p.support >= self.config.min_support
        ]

    def _rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Ранжирование паттернов по значимости"""
        return sorted(patterns, key=lambda p: (p.confidence * p.support), reverse=True)

    def evaluate_pattern(self, pattern: Pattern, data: pd.DataFrame) -> float:
        """Оценка качества паттерна"""
        # Расчет тренда
        pattern_data = data.iloc[pattern.start_idx : pattern.end_idx]
        trend = self._calculate_trend(pattern_data)

        # Расчет профиля объема
        volume_profile = self._calculate_volume_profile(pattern_data)

        # Расчет технических индикаторов
        technical_indicators = self._calculate_technical_indicators(pattern_data)

        # Обновление паттерна
        pattern.trend = trend
        pattern.volume_profile = volume_profile
        pattern.technical_indicators = technical_indicators

        # Расчет итоговой оценки
        score = (
            pattern.confidence * 0.4
            + pattern.support * 0.3
            + abs(trend) * 0.2
            + volume_profile * 0.1
        )

        return min(max(score, 0), 1)

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """Расчет тренда"""
        returns = data["close"].pct_change()
        return returns.mean() / returns.std()

    def _calculate_volume_profile(self, data: pd.DataFrame) -> float:
        """Расчет профиля объема"""
        return data["volume"].mean() / data["volume"].std()

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Расчет технических индикаторов"""
        indicators = {}

        if "RSI" in self.config.technical_indicators:
            indicators["RSI"] = talib.RSI(data["close"])[-1]
        if "MACD" in self.config.technical_indicators:
            macd, signal, _ = talib.MACD(data["close"])
            indicators["MACD"] = macd[-1]
            indicators["MACD_signal"] = signal[-1]
        if "BB" in self.config.technical_indicators:
            upper, middle, lower = talib.BBANDS(data["close"])
            indicators["BB_position"] = (data["close"][-1] - lower[-1]) / (upper[-1] - lower[-1])

        return indicators

    def save_patterns(self, patterns: List[Pattern], file_path: str):
        """Сохранение паттернов в файл"""
        patterns_data = [p.to_dict() for p in patterns]

        with open(file_path, "w") as f:
            json.dump(patterns_data, f)

    def load_patterns(self, file_path: str) -> List[Pattern]:
        """Загрузка паттернов из файла"""
        with open(file_path, "r") as f:
            patterns_data = json.load(f)

        return [Pattern.from_dict(p_data) for p_data in patterns_data]

    def combine_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Комбинирование похожих паттернов"""
        if not patterns:
            return []

        # Группировка паттернов по типу
        grouped_patterns = {}
        for pattern in patterns:
            if pattern.pattern_type not in grouped_patterns:
                grouped_patterns[pattern.pattern_type] = []
            grouped_patterns[pattern.pattern_type].append(pattern)

        # Комбинирование паттернов внутри каждой группы
        combined_patterns = []
        for pattern_type, type_patterns in grouped_patterns.items():
            clusters = self.cluster_patterns(type_patterns)

            for cluster in clusters.values():
                if len(cluster) > 1:
                    # Создание комбинированного паттерна
                    combined_pattern = self._create_combined_pattern(cluster)
                    combined_patterns.append(combined_pattern)
                else:
                    combined_patterns.extend(cluster)

        return combined_patterns

    def _create_combined_pattern(self, patterns: List[Pattern]) -> Pattern:
        """Создание комбинированного паттерна из группы похожих паттернов"""
        # Использование паттерна с наивысшей оценкой как базового
        base_pattern = max(patterns, key=lambda p: p.confidence)

        # Усреднение признаков
        combined_features = np.mean([p.features for p in patterns], axis=0)

        # Расчет средних значений confidence и support
        avg_confidence = np.mean([p.confidence for p in patterns])
        avg_support = np.mean([p.support for p in patterns])

        return Pattern(
            pattern_type=base_pattern.pattern_type,
            start_idx=base_pattern.start_idx,
            end_idx=base_pattern.end_idx,
            features=combined_features,
            confidence=avg_confidence,
            support=avg_support,
            metadata={
                "combined_from": len(patterns),
                "original_patterns": [p.metadata for p in patterns],
            },
        )
