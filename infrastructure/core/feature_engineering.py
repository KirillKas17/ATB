"""
Модуль для инженерии признаков.
Предоставляет функции для генерации и обработки признаков из рыночных данных.
Включает технические индикаторы, статистические признаки и машинное обучение.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, cast

from shared.numpy_utils import np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import RobustScaler, StandardScaler

from infrastructure.core.technical import (
    adx,
    atr,
    bollinger_bands,
    cci,
    ema,
    macd,
    rsi,
    sma,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Конфигурация генерации признаков."""

    # Технические индикаторы
    use_technical_indicators: bool = True
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    macd_params: Dict[str, int] = field(
        default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9}
    )
    bb_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    # Статистические признаки
    use_statistical_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    # Временные признаки
    use_time_features: bool = True
    use_cyclical_features: bool = True
    # Дополнительные признаки
    use_volume_features: bool = True
    use_price_patterns: bool = True
    use_market_microstructure: bool = True
    # Предобработка
    normalize_features: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    # Селекция признаков
    feature_selection_method: str = (
        "mutual_info"  # "mutual_info", "f_regression", "pca"
    )
    n_features: int = 50


class FeatureEngineer:
    """
    Продвинутый инженер признаков для торговых систем.
    Генерирует технические индикаторы, статистические признаки,
    временные паттерны и рыночные микроструктурные признаки.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """Инициализация инженера признаков."""
        self.config = config or FeatureConfig()
        self.scaler = StandardScaler()
        self._features_cache: Dict[str, pd.DataFrame] = {}
        self.feature_selector = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        logger.info("Feature Engineer initialized")

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация полного набора признаков из рыночных данных (оптимизировано).
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            DataFrame с признаками
        """
        try:
            if data.empty:
                logger.warning("Empty data provided for feature generation")
                return pd.DataFrame()
            
            # Кэширование для больших наборов данных
            cache_key = f"features_{hash(tuple(data['close'].tail(20)))}"
            if hasattr(self, '_features_cache') and cache_key in self._features_cache:
                cached_result = self._features_cache[cache_key]
                logger.info(f"Using cached features: {len(cached_result.columns)} features")
                return cached_result
            
            if not hasattr(self, '_features_cache'):
                self._features_cache = {}
            
            features = pd.DataFrame(index=data.index)
            
            # Пакетное добавление признаков для оптимизации
            feature_batches = []
            
            # Базовые признаки
            batch_features = self._add_basic_features(pd.DataFrame(index=data.index), data)
            feature_batches.append(batch_features)
            
            # Технические индикаторы
            if self.config.use_technical_indicators:
                batch_features = self._add_technical_indicators(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Статистические признаки
            if self.config.use_statistical_features:
                batch_features = self._add_statistical_features(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Временные признаки
            if self.config.use_time_features:
                batch_features = self._add_time_features(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Объемные признаки
            if self.config.use_volume_features:
                batch_features = self._add_volume_features(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Паттерны цен
            if self.config.use_price_patterns:
                batch_features = self._add_price_patterns(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Микроструктурные признаки
            if self.config.use_market_microstructure:
                batch_features = self._add_microstructure_features(pd.DataFrame(index=data.index), data)
                feature_batches.append(batch_features)
            
            # Эффективное объединение всех batch'ей
            features = pd.concat(feature_batches, axis=1)
            
            # Предобработка
            features = self._preprocess_features(features)
            
            # Кэширование результата с ограничением размера
            if len(self._features_cache) > 20:
                oldest_keys = list(self._features_cache.keys())[:10]
                for key in oldest_keys:
                    del self._features_cache[key]
            
            self._features_cache[cache_key] = features
            
            # Сохранение имен признаков
            self.feature_names = features.columns.tolist()
            logger.info(f"Generated {len(features.columns)} features")
            return features
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return pd.DataFrame()

    def _add_basic_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление базовых признаков."""
        try:
            # Ценовые признаки
            features["open"] = data["open"]
            features["high"] = data["high"]
            features["low"] = data["low"]
            features["close"] = data["close"]
            features["volume"] = data["volume"]
            
            # Производные признаки
            features["price_range"] = data["high"] - data["low"]
            features["body_size"] = abs(data["close"] - data["open"])
            features["upper_shadow"] = data["high"] - np.maximum(
                data["open"], data["close"]
            )
            features["lower_shadow"] = (
                np.minimum(data["open"], data["close"]) - data["low"]
            )
            
            # Относительные изменения
            features["price_change"] = data["close"].pct_change()
            features["price_change_abs"] = abs(features["price_change"])
            features["volume_change"] = data["volume"].pct_change()
            
            # Логарифмические изменения
            features["log_return"] = np.log(data["close"] / data["close"].shift(1))
            
            return features
        except Exception as e:
            logger.error(f"Error adding basic features: {e}")
            return features

    def _add_technical_indicators(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление технических индикаторов."""
        try:
            close = data["close"]
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            
            # Векторизованное вычисление EMA для всех периодов одновременно
            ema_data = {}
            for period in self.config.ema_periods:
                ema_data[f"ema_{period}"] = close.ewm(span=period).mean()
            
            # Добавляем все EMA индикаторы и их отношения в batch режиме
            for period in self.config.ema_periods:
                ema_col = f"ema_{period}"
                features[ema_col] = ema_data[ema_col]
                features[f"ema_ratio_{period}"] = close / ema_data[ema_col]
            
            # Векторизованное вычисление RSI для всех периодов одновременно
            rsi_data = {}
            for period in self.config.rsi_periods:
                rsi_result = rsi(close, period)
                # Безопасное извлечение значений
                if callable(rsi_result):
                    rsi_values = rsi_result()
                else:
                    rsi_values = rsi_result
                rsi_data[f"rsi_{period}"] = rsi_values
            
            # Добавляем все RSI индикаторы в batch режиме
            for period in self.config.rsi_periods:
                features[f"rsi_{period}"] = rsi_data[f"rsi_{period}"]
            
            # MACD
            macd_result = macd(close, **self.config.macd_params)
            # Безопасное извлечение значений
            if callable(macd_result):
                macd_values = macd_result()
            else:
                macd_values = macd_result
            
            features["macd"] = macd_values.macd
            features["macd_signal"] = macd_values.signal
            features["macd_histogram"] = macd_values.histogram
            
            # Безопасное сравнение для MACD cross
            if hasattr(macd_values.macd, 'values'):
                macd_series = macd_values.macd.values
                signal_series = macd_values.signal.values
            else:
                macd_series = macd_values.macd
                signal_series = macd_values.signal
            
            features["macd_cross"] = np.where(macd_series > signal_series, 1, -1)
            
            # Bollinger Bands
            for period in self.config.bb_periods:
                bb_lower, bb_middle, bb_upper = bollinger_bands(close, period)
                features[f"bb_lower_{period}"] = bb_lower
                features[f"bb_middle_{period}"] = bb_middle
                features[f"bb_upper_{period}"] = bb_upper
                features[f"bb_width_{period}"] = (bb_upper - bb_lower) / bb_middle
                features[f"bb_position_{period}"] = (close - bb_lower) / (
                    bb_upper - bb_lower
                )
            
            # ATR
            features["atr"] = atr(high, low, close)
            features["atr_ratio"] = features["atr"] / close
            
            # ADX
            features["adx"] = adx(high, low, close)
            
            # CCI
            features["cci"] = cci(high, low, close)
            
            return features
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return features

    def _add_statistical_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление статистических признаков."""
        try:
            close = data["close"]
            volume = data["volume"]
            
            # Волатильность
            for window in self.config.volatility_windows:
                features[f"volatility_{window}"] = close.rolling(window=window).std()
                features[f"volatility_ratio_{window}"] = (
                    features[f"volatility_{window}"] / close
                )
            
            # Скользящие статистики
            for window in self.config.rolling_windows:
                features[f"mean_{window}"] = close.rolling(window=window).mean()
                features[f"median_{window}"] = close.rolling(window=window).median()
                features[f"std_{window}"] = close.rolling(window=window).std()
                features[f"skew_{window}"] = close.rolling(window=window).skew()
                features[f"kurt_{window}"] = close.rolling(window=window).kurt()
                
                # Z-score
                features[f"zscore_{window}"] = (
                    close - features[f"mean_{window}"]
                ) / features[f"std_{window}"]
            
            # Объемные статистики
            for window in self.config.rolling_windows:
                features[f"volume_mean_{window}"] = volume.rolling(window=window).mean()
                features[f"volume_std_{window}"] = volume.rolling(window=window).std()
                features[f"volume_ratio_{window}"] = volume / features[f"volume_mean_{window}"]
            
            # Моментум
            for window in self.config.rolling_windows:
                features[f"momentum_{window}"] = close / close.shift(window) - 1
                features[f"momentum_abs_{window}"] = abs(features[f"momentum_{window}"])
            
            return features
        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return features

    def _add_time_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление временных признаков."""
        try:
            # Проверяем, что индекс является DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("Data index is not DatetimeIndex, skipping time features")
                return features
            
            # Базовые временные признаки
            features["hour"] = data.index.hour
            features["day_of_week"] = data.index.dayofweek
            features["day_of_month"] = data.index.day
            features["month"] = data.index.month
            features["quarter"] = data.index.quarter
            features["year"] = data.index.year
            
            # Циклические признаки
            if self.config.use_cyclical_features:
                features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
                features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
                features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
                features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
                features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
                features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
            
            # Временные паттерны
            features["is_monday"] = (features["day_of_week"] == 0).astype(int)
            features["is_friday"] = (features["day_of_week"] == 4).astype(int)
            features["is_month_start"] = (features["day_of_month"] <= 3).astype(int)
            features["is_month_end"] = (features["day_of_month"] >= 28).astype(int)
            
            return features
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return features

    def _add_volume_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление объемных признаков."""
        try:
            volume = data["volume"]
            close = data["close"]
            
            # Базовые объемные признаки
            features["volume_sma_ratio"] = volume / volume.rolling(window=20).mean()
            features["volume_ema_ratio"] = volume / volume.ewm(span=20).mean()
            
            # VWAP
            features["vwap"] = self._calculate_vwap(data)
            features["price_vwap_ratio"] = close / features["vwap"]
            
            # OBV
            features["obv"] = self._calculate_obv(close, volume)
            features["obv_change"] = features["obv"].pct_change()
            
            # Объемные индикаторы
            features["volume_price_trend"] = (
                volume * (close - close.shift(1))
            ).cumsum()
            
            # Дисбаланс объема
            features["volume_imbalance"] = (
                volume * np.where(close > close.shift(1), 1, -1)
            ).rolling(window=20).sum()
            
            return features
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return features

    def _add_price_patterns(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление паттернов цен."""
        try:
            # Свечные паттерны
            features["is_hammer"] = self._is_hammer(data)
            features["is_shooting_star"] = self._is_shooting_star(data)
            
            # Уровни поддержки и сопротивления
            features["support_level"] = self._find_support_level(data["low"])
            features["resistance_level"] = self._find_resistance_level(data["high"])
            
            # Дивергенция моментума
            features["momentum_divergence"] = self._calculate_momentum_divergence(data["close"])
            
            # Эффективность рынка
            features["market_efficiency"] = self._calculate_market_efficiency(data)
            
            # Асимметрия объема
            features["volume_asymmetry"] = self._calculate_volume_asymmetry(data)
            
            return features
        except Exception as e:
            logger.error(f"Error adding price patterns: {e}")
            return features

    def _add_microstructure_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Добавление микроструктурных признаков."""
        try:
            close = data["close"]
            volume = data["volume"]
            
            # Ликвидность
            features["liquidity_ratio"] = volume / close.rolling(window=20).std()
            
            # Внутридневная волатильность
            features["intraday_volatility"] = (
                data["high"] - data["low"]
            ) / data["close"]
            
            # Моментум цены
            features["price_momentum"] = close.pct_change(periods=5)
            
            # Микроструктурные индикаторы
            features["price_efficiency"] = abs(close - close.shift(20)) / (
                close.rolling(window=20).apply(lambda x: float(sum(abs(x.diff().dropna()))))
            )
            
            return features
        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return features

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Предобработка признаков."""
        try:
            # Удаление выбросов
            if self.config.remove_outliers:
                features = self._remove_outliers(features)
            
            # Нормализация
            if self.config.normalize_features:
                features = self._normalize_features(features)
            
            # Заполнение пропущенных значений
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return features

    def _remove_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов."""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            for column in numeric_features.columns:
                Q1 = numeric_features[column].quantile(0.25)
                Q3 = numeric_features[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                features[column] = features[column].clip(lower_bound, upper_bound)
            return features
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return features

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Нормализация признаков."""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                if not self.is_fitted:
                    self.scaler.fit(numeric_features)
                    self.is_fitted = True
                features[numeric_features.columns] = self.scaler.transform(numeric_features)
            return features
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features

    def select_features(
        self, features: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """Селекция признаков."""
        try:
            # Выравнивание индексов
            common_index = features.index.intersection(target.index)
            features_aligned: pd.DataFrame = features.loc[common_index]
            target_aligned: pd.Series = target.loc[common_index]
            
            # Удаление пропущенных значений
            mask = ~(features_aligned.isna().any(axis=1) | target_aligned.isna())
            features_clean = features_aligned[mask]
            target_clean = target_aligned[mask]
            
            if len(features_clean) < 10:
                logger.warning("Insufficient data for feature selection")
                return features
            
            # Селекция признаков
            if self.config.feature_selection_method == "mutual_info":
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_regression, k=min(self.config.n_features, len(features_clean.columns))
                )
            elif self.config.feature_selection_method == "f_regression":
                self.feature_selector = SelectKBest(
                    score_func=f_regression, k=min(self.config.n_features, len(features_clean.columns))
                )
            elif self.config.feature_selection_method == "pca":
                self.feature_selector = PCA(n_components=min(self.config.n_features, len(features_clean.columns)))
            else:
                logger.warning(f"Unknown feature selection method: {self.config.feature_selection_method}")
                return features
            
            # Применение селекции
            if self.feature_selector is None:
                logger.error("Feature selector not initialized")
                return features
            selected_features_result = self.feature_selector.fit_transform(features_clean, target_clean)
            
            # Проверяем, что результат не является функцией
            if callable(selected_features_result):
                selected_features = selected_features_result()
            else:
                selected_features = selected_features_result
            
            if self.config.feature_selection_method == "pca":
                # Для PCA создаем новые имена признаков
                feature_names = [f"pca_{i}" for i in range(selected_features.shape[1])]
            else:
                # Для других методов используем исходные имена
                if hasattr(self.feature_selector, 'get_support') and self.feature_selector is not None:
                    support_mask = self.feature_selector.get_support()
                    if isinstance(support_mask, (list, tuple, np.ndarray)):
                        feature_names = features_clean.columns[support_mask].tolist()
                    else:
                        feature_names = features_clean.columns.tolist()
                else:
                    feature_names = features_clean.columns.tolist()
            
            # Создаем новый DataFrame
            result = pd.DataFrame(selected_features, index=features_clean.index, columns=feature_names)
            
            logger.info(f"Selected {len(feature_names)} features")
            return result
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return features

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Расчет On-Balance Volume."""
        obv: pd.Series = pd.Series(index=close.index, dtype=float)
        
        # Инициализация первого значения
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            # Получение текущих и предыдущих значений
            current_close: float = close.iloc[i]
            prev_close: float = close.iloc[i-1]
            current_volume: float = volume.iloc[i]
            prev_obv: float = obv.iloc[i-1]
            
            if current_close > prev_close:
                new_obv = prev_obv + current_volume
            elif current_close < prev_close:
                new_obv = prev_obv - current_volume
            else:
                new_obv = prev_obv
            
            # Присваивание нового значения
            obv.iloc[i] = new_obv
        
        return obv

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Расчет VWAP."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        return (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

    def _is_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Определение паттерна молот."""
        body = abs(data["close"] - data["open"])
        lower_shadow = np.minimum(data["open"], data["close"]) - data["low"]
        upper_shadow = data["high"] - np.maximum(data["open"], data["close"])
        return ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)

    def _is_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Определение паттерна падающая звезда."""
        body = abs(data["close"] - data["open"])
        lower_shadow = np.minimum(data["open"], data["close"]) - data["low"]
        upper_shadow = data["high"] - np.maximum(data["open"], data["close"])
        return ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int)

    def _find_support_level(self, low: pd.Series) -> pd.Series:
        """Поиск уровня поддержки."""
        return low.rolling(window=20).min()

    def _find_resistance_level(self, high: pd.Series) -> pd.Series:
        """Поиск уровня сопротивления."""
        return high.rolling(window=20).max()

    def _calculate_momentum_divergence(self, close: pd.Series) -> pd.Series:
        """Расчет дивергенции моментума."""
        momentum = close.pct_change(periods=5)
        return momentum - momentum.rolling(window=20).mean()

    def _calculate_market_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """Расчет эффективности рынка."""
        close = data["close"]
        return abs(close - close.shift(20)) / close.rolling(window=20).apply(
            lambda x: float(sum(abs(x.diff().dropna())))
        )

    def _calculate_volume_asymmetry(self, data: pd.DataFrame) -> pd.Series:
        """Расчет асимметрии объема."""
        volume = data["volume"]
        close = data["close"]
        
        # Объем при росте цены
        up_volume = volume.where(close > close.shift(1), 0)
        down_volume = volume.where(close < close.shift(1), 0)
        
        # Асимметрия
        asymmetry = (up_volume.rolling(window=20).sum() - down_volume.rolling(window=20).sum()) / \
                   (up_volume.rolling(window=20).sum() + down_volume.rolling(window=20).sum())
        
        return asymmetry

    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков."""
        try:
            if self.feature_selector is None:
                return {}
            
            if hasattr(self.feature_selector, 'scores_'):
                # Для SelectKBest
                feature_scores = self.feature_selector.scores_
                feature_names = self.feature_names
                if hasattr(self.feature_selector, 'get_support'):
                    selected_features = self.feature_selector.get_support()
                    feature_names = [name for name, selected in zip(feature_names, selected_features) if selected]
                
                return dict(zip(feature_names, feature_scores))
            elif hasattr(self.feature_selector, 'explained_variance_ratio_'):
                # Для PCA
                return {f"pca_{i}": ratio for i, ratio in enumerate(self.feature_selector.explained_variance_ratio_)}
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def save_features(self, features: pd.DataFrame, path: str) -> None:
        """Сохранение признаков."""
        try:
            features.to_csv(path)
            logger.info(f"Features saved to {path}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")

    def load_features(self, path: str) -> pd.DataFrame:
        """Загрузка признаков."""
        try:
            features = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(f"Features loaded from {path}")
            return features
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return pd.DataFrame()
