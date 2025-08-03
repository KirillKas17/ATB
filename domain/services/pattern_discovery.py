"""
Промышленный сервис для обнаружения паттернов (строгая типизация, DDD, SOLID).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from domain.types.service_types import (
    AnalysisConfig,
    IndicatorType,
    MarketDataFrame,
    PatternAnalysisProtocol,
    PatternType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternConfig:
    min_pattern_length: int
    max_pattern_length: int
    min_confidence: float
    min_support: float
    max_patterns: int
    clustering_method: str
    min_cluster_size: int
    pattern_types: List[PatternType]
    feature_columns: List[str]
    window_sizes: List[int]
    similarity_threshold: float
    technical_indicators: Optional[List[IndicatorType]] = None
    volume_threshold: float = 1.5
    price_threshold: float = 0.02
    trend_window: int = 20


@dataclass
class Pattern:
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    features: np.ndarray
    confidence: float
    support: float
    metadata: Dict[str, Any]
    trend: Optional[float] = None
    volume_profile: Optional[float] = None
    technical_indicators: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["features"] = self.features.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        return cls(
            pattern_type=PatternType(data["pattern_type"]),
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            features=np.array(data["features"]),
            confidence=data["confidence"],
            support=data["support"],
            metadata=data["metadata"],
            trend=data.get("trend"),
            volume_profile=data.get("volume_profile"),
            technical_indicators=data.get("technical_indicators", {}),
        )


class PatternDiscovery(PatternAnalysisProtocol):
    def __init__(self, config: PatternConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.min_pattern_length >= self.config.max_pattern_length:
            raise ValueError("min_pattern_length must be less than max_pattern_length")
        if not self.config.pattern_types:
            raise ValueError("pattern_types cannot be empty")
        if not self.config.feature_columns:
            raise ValueError("feature_columns cannot be empty")

    async def discover_patterns(
        self, data: MarketDataFrame, config: AnalysisConfig
    ) -> List[Pattern]:
        if data.empty:
            raise ValueError("Empty data provided")
        if not all(col in data.columns for col in self.config.feature_columns):
            raise ValueError("Missing required feature columns")
        patterns: List[Pattern] = []
        for pattern_type in self.config.pattern_types:
            if pattern_type == PatternType.CANDLE:
                patterns.extend(self.find_candle_patterns(data))
            elif pattern_type == PatternType.PRICE:
                patterns.extend(self.find_price_patterns(data))
            elif pattern_type == PatternType.VOLUME:
                patterns.extend(self.find_volume_patterns(data))
        patterns = self.cluster_patterns(patterns)
        patterns = self._rank_patterns(patterns)
        return patterns[: self.config.max_patterns]

    async def validate_pattern(self, pattern: Pattern, data: MarketDataFrame) -> float:
        """Валидация паттерна на исторических данных."""
        if pattern.start_idx >= len(data) or pattern.end_idx >= len(data):
            return 0.0
        # Проверяем, насколько хорошо паттерн предсказывает будущее движение цены
        # Исправление: безопасное обращение к данным
        if hasattr(data, 'iloc'):
            pattern_data = data.iloc[pattern.start_idx : pattern.end_idx + 1]
        else:
            pattern_data = data[pattern.start_idx : pattern.end_idx + 1]
        
        if len(pattern_data) < 2:
            return 0.0
        
        # Простая валидация на основе движения цены после паттерна
        # Исправление: безопасное обращение к данным
        if hasattr(pattern_data, 'iloc'):
            pattern_end_price = float(pattern_data.iloc[-1]["close"])
        else:
            pattern_end_price = float(pattern_data[-1]["close"]) if hasattr(pattern_data, "__getitem__") else 0.0
        
        future_window = min(10, len(data) - pattern.end_idx - 1)
        if future_window <= 0:
            return 0.0
        
        # Исправление: безопасное обращение к данным
        if hasattr(data, 'iloc'):
            future_data = data.iloc[pattern.end_idx + 1 : pattern.end_idx + 1 + future_window]
        else:
            future_data = data[pattern.end_idx + 1 : pattern.end_idx + 1 + future_window]
        
        if hasattr(future_data, 'empty') and future_data.empty:
            return 0.0
        
        # Исправление: безопасное обращение к данным
        if hasattr(future_data, 'iloc'):
            future_price = float(future_data.iloc[-1]["close"])
        else:
            future_price = float(future_data[-1]["close"]) if hasattr(future_data, "__getitem__") else 0.0
        
        price_change = (future_price - pattern_end_price) / pattern_end_price
        # Возвращаем абсолютное значение изменения цены как меру валидности
        return abs(price_change)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data[self.config.feature_columns].copy()
        if self.config.technical_indicators:
            features = self._add_technical_indicators(features, data)
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index,
        )
        return features

    def _add_technical_indicators(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        if self.config.technical_indicators:
            for ind in self.config.technical_indicators:
                if ind == IndicatorType.MOMENTUM:
                    features["RSI"] = self._calculate_rsi(data["close"])
                if ind == IndicatorType.TREND:
                    macd, signal, hist = self._calculate_macd(data["close"])
                    features["MACD"] = macd
                    features["MACD_signal"] = signal
                    features["MACD_hist"] = hist
                if ind == IndicatorType.VOLATILITY:
                    upper, middle, lower = self._calculate_bollinger_bands(data["close"])
                    features["BB_upper"] = upper
                    features["BB_middle"] = middle
                    features["BB_lower"] = lower
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta.gt(0.0), 0.0)).rolling(window=period).mean()
        loss = (delta.where(delta.lt(0.0), 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def cluster_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        if not patterns:
            return []
        features = np.array([p.features.flatten() for p in patterns])
        features = self.scaler.fit_transform(features)
        clustering = DBSCAN(eps=0.3, min_samples=self.config.min_cluster_size).fit(
            features
        )
        clusters: Dict[int, List[Pattern]] = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(patterns[i])
        # Возвращаем все паттерны из кластеров
        result: List[Pattern] = []
        for cluster in clusters.values():
            result.extend(cluster)
        return result

    def find_candle_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        patterns: List[Pattern] = []
        if not isinstance(data, pd.DataFrame) or data.empty:
            return patterns
        
        for i in range(2, len(data.index) if hasattr(data.index, '__len__') and hasattr(data.index, '__iter__') else 0):
                candle_data: pd.Series = data.iloc[i]
                if not isinstance(candle_data, pd.Series):
                    continue
                if self._is_doji(candle_data):
                    patterns.append(
                        Pattern(
                            pattern_type=PatternType.CANDLE,
                            start_idx=i - 1,
                            end_idx=i,
                            features=np.array([candle_data["close"], candle_data["open"]]),
                            confidence=0.7,
                            support=0.1,
                            metadata={"pattern": "doji", "position": i},
                        )
                    )
                if self._is_hammer(candle_data):
                    patterns.append(
                        Pattern(
                            pattern_type=PatternType.CANDLE,
                            start_idx=i - 1,
                            end_idx=i,
                            features=np.array([candle_data["close"], candle_data["low"]]),
                            confidence=0.8,
                            support=0.15,
                            metadata={"pattern": "hammer", "position": i},
                        )
                    )
                if self._is_shooting_star(candle_data):
                    patterns.append(
                        Pattern(
                            pattern_type=PatternType.CANDLE,
                            start_idx=i - 1,
                            end_idx=i,
                            features=np.array([candle_data["close"], candle_data["high"]]),
                            confidence=0.75,
                            support=0.12,
                            metadata={"pattern": "shooting_star", "position": i},
                        )
                    )
        return patterns

    def _is_doji(self, candle: pd.Series) -> bool:
        body_size = abs(float(candle["close"]) - float(candle["open"]))
        total_range = float(candle["high"]) - float(candle["low"])
        if total_range == 0:
            return False
        return body_size / total_range < 0.1

    def _is_hammer(self, candle: pd.Series) -> bool:
        body_size = abs(float(candle["close"]) - float(candle["open"]))
        total_range = float(candle["high"]) - float(candle["low"])
        if total_range == 0:
            return False
        lower_shadow = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        upper_shadow = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        return (
            lower_shadow > 2 * body_size
            and upper_shadow < body_size
            and body_size / total_range > 0.1
        )

    def _is_shooting_star(self, candle: pd.Series) -> bool:
        body_size = abs(float(candle["close"]) - float(candle["open"]))
        total_range = float(candle["high"]) - float(candle["low"])
        if total_range == 0:
            return False
        lower_shadow = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        upper_shadow = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        return (
            upper_shadow > 2 * body_size
            and lower_shadow < body_size
            and body_size / total_range > 0.1
        )

    def find_price_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        patterns: List[Pattern] = []
        # Безопасно приводим к numpy array для find_peaks
        close_values = data["close"].to_numpy()
        peaks, _ = find_peaks(close_values, distance=self.config.min_pattern_length)
        # Безопасно применяем унарный оператор
        negative_close_values = -close_values
        troughs, _ = find_peaks(
            negative_close_values, distance=self.config.min_pattern_length
        )
        for i in range(len(peaks) - 1):
            # Правильное сравнение индексов - приводим к int
            if int(peaks[i + 1]) - int(peaks[i]) <= self.config.max_pattern_length:
                pattern = self._analyze_price_pattern(data, int(peaks[i]), int(peaks[i + 1]))
                if pattern:
                    patterns.append(pattern)
        return patterns

    def find_volume_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        patterns: List[Pattern] = []
        avg_volume = data["volume"].rolling(window=self.config.trend_window).mean()
        volume_peaks = data[data["volume"] > avg_volume * self.config.volume_threshold].index
        for i in range(len(volume_peaks) - 1):
            # Правильное обращение к индексам - приводим к int
            if int(volume_peaks[i + 1]) - int(volume_peaks[i]) <= self.config.max_pattern_length:
                pattern = self._analyze_volume_pattern(
                    data, int(volume_peaks[i]), int(volume_peaks[i + 1])
                )
                if pattern:
                    patterns.append(pattern)
        return patterns

    def _analyze_price_pattern(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Pattern]:
        # Безопасное обращение к данным
        close_series = data["close"]
        if hasattr(close_series, 'iloc') and callable(close_series.iloc):
            start_price = close_series.iloc[start_idx]
            end_price = close_series.iloc[end_idx]
            price_change = (end_price - start_price) / start_price
        else:
            price_change = 0.0
            
        if abs(price_change) >= self.config.price_threshold:
            # Безопасное получение features
            if hasattr(data, 'iloc') and callable(data.iloc):
                features_data = data.iloc[start_idx:end_idx]
                features = features_data.values if hasattr(features_data, 'values') else np.array([])
            else:
                features = np.array([])
                
            return Pattern(
                pattern_type=PatternType.PRICE,
                start_idx=start_idx,
                end_idx=end_idx,
                features=features,
                confidence=float(abs(price_change)),
                support=float(self._calculate_support(data, start_idx, end_idx)),
                metadata={"price_change": price_change},
            )
        return None

    def _analyze_volume_pattern(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Pattern]:
        # Безопасное обращение к данным
        volume_series = data["volume"]
        if hasattr(volume_series, 'iloc') and callable(volume_series.iloc):
            start_volume = volume_series.iloc[start_idx]
            end_volume = volume_series.iloc[end_idx]
            volume_change = (end_volume - start_volume) / start_volume
        else:
            volume_change = 0.0
            
        if volume_change >= self.config.volume_threshold:
            # Безопасное получение features
            if hasattr(data, 'iloc') and callable(data.iloc):
                features_data = data.iloc[start_idx:end_idx]
                features = features_data.values if hasattr(features_data, 'values') else np.array([])
            else:
                features = np.array([])
                
            return Pattern(
                pattern_type=PatternType.VOLUME,
                start_idx=start_idx,
                end_idx=end_idx,
                features=features,
                confidence=float(volume_change),
                support=float(self._calculate_support(data, start_idx, end_idx)),
                metadata={"volume_change": volume_change},
            )
        return None

    def _calculate_support(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> float:
        # Безопасное обращение к данным
        if hasattr(data, 'iloc') and callable(data.iloc):
            pattern_data = data.iloc[start_idx:end_idx]
            pattern_len = len(pattern_data) if hasattr(pattern_data, '__len__') else 0
        else:
            pattern_len = 0
        data_len = len(data) if hasattr(data, '__len__') else 1
        return pattern_len / data_len

    def _rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        return sorted(patterns, key=lambda p: (p.confidence * p.support), reverse=True)

    def evaluate_pattern(self, pattern: Pattern, data: pd.DataFrame) -> float:
        # Безопасное обращение к данным
        if hasattr(data, 'iloc') and callable(data.iloc):
            pattern_data = data.iloc[pattern.start_idx : pattern.end_idx]
        else:
            pattern_data = data
            
        trend = self._calculate_trend(pattern_data)
        volume_profile = self._calculate_volume_profile(pattern_data)
        technical_indicators = self._calculate_technical_indicators(pattern_data)
        pattern.trend = trend
        pattern.volume_profile = volume_profile
        pattern.technical_indicators = technical_indicators
        score = (
            pattern.confidence * 0.4
            + pattern.support * 0.3
            + abs(trend) * 0.2
            + volume_profile * 0.1
        )
        return min(max(score, 0), 1)

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        returns = data["close"].pct_change()
        return float(returns.mean()) / float(returns.std()) if returns.std() != 0 else 0.0

    def _calculate_volume_profile(self, data: pd.DataFrame) -> float:
        return float(data["volume"].mean()) / float(data["volume"].std()) if data["volume"].std() != 0 else 0.0

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        indicators = {}
        if self.config.technical_indicators:
            if IndicatorType.MOMENTUM in self.config.technical_indicators:
                rsi_series = self._calculate_rsi(data["close"])
                # Безопасное обращение к данным
                if hasattr(rsi_series, 'iloc') and callable(rsi_series.iloc) and len(rsi_series) > 0:
                    indicators["RSI"] = float(rsi_series.iloc[-1])
                else:
                    indicators["RSI"] = 0.0
                    
            if IndicatorType.TREND in self.config.technical_indicators:
                macd, signal, _ = self._calculate_macd(data["close"])
                # Безопасное обращение к данным
                if hasattr(macd, 'iloc') and callable(macd.iloc) and len(macd) > 0:
                    indicators["MACD"] = float(macd.iloc[-1])
                else:
                    indicators["MACD"] = 0.0
                    
                if hasattr(signal, 'iloc') and callable(signal.iloc) and len(signal) > 0:
                    indicators["MACD_signal"] = float(signal.iloc[-1])
                else:
                    indicators["MACD_signal"] = 0.0
                    
            if IndicatorType.VOLATILITY in self.config.technical_indicators:
                upper, middle, lower = self._calculate_bollinger_bands(data["close"])
                # Безопасное обращение к данным
                if (hasattr(upper, 'iloc') and callable(upper.iloc) and len(upper) > 0 and
                    hasattr(lower, 'iloc') and callable(lower.iloc) and len(lower) > 0):
                    
                    close_series = data["close"]
                    if hasattr(close_series, 'iloc') and callable(close_series.iloc) and len(close_series) > 0:
                        current_price = float(close_series.iloc[-1])
                    else:
                        current_price = 0.0
                        
                    lower_val = float(lower.iloc[-1])
                    upper_val = float(upper.iloc[-1])
                    
                    if upper_val > lower_val:
                        indicators["BB_position"] = (current_price - lower_val) / (upper_val - lower_val)
                    else:
                        indicators["BB_position"] = 0.5
                else:
                    indicators["BB_position"] = 0.5
        return indicators

    def save_patterns(self, patterns: List[Pattern], file_path: str) -> None:
        patterns_data = [p.to_dict() for p in patterns]
        with open(file_path, "w") as f:
            json.dump(patterns_data, f)

    def load_patterns(self, file_path: str) -> List[Pattern]:
        with open(file_path, "r") as f:
            patterns_data = json.load(f)
        return [Pattern.from_dict(p_data) for p_data in patterns_data]

    def combine_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        if not patterns:
            return []
        grouped_patterns: Dict[PatternType, List[Pattern]] = {}
        for pattern in patterns:
            if pattern.pattern_type not in grouped_patterns:
                grouped_patterns[pattern.pattern_type] = []
            grouped_patterns[pattern.pattern_type].append(pattern)
        combined_patterns: List[Pattern] = []
        for pattern_type, type_patterns in grouped_patterns.items():
            clusters = self.cluster_patterns(type_patterns)
            # Группируем паттерны по кластерам
            cluster_groups: Dict[int, List[Pattern]] = {}
            for i, pattern in enumerate(clusters):
                cluster_id = i // self.config.min_cluster_size if self.config.min_cluster_size > 0 else 0
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(pattern)
            
            for cluster_patterns_list in cluster_groups.values():
                if len(cluster_patterns_list) > 1:
                    combined_pattern = self._create_combined_pattern(cluster_patterns_list)
                    combined_patterns.append(combined_pattern)
                else:
                    combined_patterns.extend(cluster_patterns_list)
        return combined_patterns

    def _create_combined_pattern(self, patterns: List[Pattern]) -> Pattern:
        base_pattern = max(patterns, key=lambda p: p.confidence)
        combined_features = np.mean([p.features for p in patterns], axis=0)
        avg_confidence = float(np.mean([p.confidence for p in patterns]))
        avg_support = float(np.mean([p.support for p in patterns]))
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


# Экспорт интерфейса для обратной совместимости
IPatternDiscovery = PatternDiscovery
