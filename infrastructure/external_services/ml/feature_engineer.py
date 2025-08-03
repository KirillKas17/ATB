"""
Инженер признаков для машинного обучения.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas import Series, DataFrame

class FeatureEngineer:
    """Инженер признаков для финансовых данных."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.is_fitted = False

    def engineer_features(self, data: DataFrame) -> DataFrame:
        """Создание технических признаков."""
        if data.empty:
            return DataFrame()
        
        # Создаем копию данных
        features = data.copy()
        
        # Ценовые признаки
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['close'].shift(1)
        features['price_change_pct'] = data['close'].pct_change() * 100
        
        # Волатильность
        features['volatility_5'] = features['returns'].rolling(window=5).std()
        features['volatility_10'] = features['returns'].rolling(window=10).std()
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        
        # Скользящие средние
        features['sma_5'] = data['close'].rolling(window=5).mean()
        features['sma_10'] = data['close'].rolling(window=10).mean()
        features['sma_20'] = data['close'].rolling(window=20).mean()
        features['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Экспоненциальные скользящие средние
        features['ema_5'] = data['close'].ewm(span=5).mean()
        features['ema_10'] = data['close'].ewm(span=10).mean()
        features['ema_20'] = data['close'].ewm(span=20).mean()
        
        # RSI
        features['rsi'] = self._calculate_rsi(data['close'])
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(data['close'])
        
        # Bollinger Bands
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = self._calculate_bollinger_bands(data['close'])
        
        # Объемные признаки (если есть)
        if 'volume' in data.columns:
            features['volume_sma_5'] = data['volume'].rolling(window=5).mean()
            features['volume_sma_10'] = data['volume'].rolling(window=10).mean()
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
            features['volume_change'] = data['volume'].pct_change()
        
        # Временные признаки
        if hasattr(data.index, 'dt'):
            features['hour'] = data.index.dt.hour
            features['day_of_week'] = data.index.dt.dayofweek
            features['month'] = data.index.dt.month
            features['quarter'] = data.index.dt.quarter
        
        # Дополнительные признаки
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['body_size'] = (data['close'] - data['open']).abs()
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        
        # Удаляем NaN значения
        features = features.dropna()
        
        return features

    def _calculate_rsi(self, prices: Series, period: int = 14) -> Series:
        """Расчет RSI."""
        delta = prices.diff()
        gain = delta.where(delta.gt(0), 0).rolling(window=period).mean()
        loss = delta.where(delta.lt(0), 0).abs().rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Series, Series]:
        """Расчет MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(self, prices: Series, period: int = 20, std_dev: float = 2) -> Tuple[Series, Series, Series]:
        """Расчет полос Боллинджера."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def scale_features(self, features: DataFrame) -> DataFrame:
        """Масштабирование признаков."""
        if features.empty:
            return DataFrame()
        
        # Выбираем числовые колонки
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return features
        
        # Масштабируем
        scaled_features = features.copy()
        scaled_features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])
        self.is_fitted = True
        
        return scaled_features

    def reduce_dimensions(self, features: DataFrame) -> DataFrame:
        """Уменьшение размерности с помощью PCA."""
        if features.empty or not self.is_fitted:
            return features
        
        # Выбираем числовые колонки
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return features
        
        # Применяем PCA
        pca_features = self.pca.fit_transform(features[numeric_columns])
        pca_df = DataFrame(
            pca_features,
            columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
            index=features.index
        )
        
        return pca_df

    def select_features(self, features: DataFrame, target: Series, method: str = 'correlation', threshold: float = 0.1) -> DataFrame:
        """Выбор признаков."""
        if features.empty or target.empty:
            return DataFrame()
        
        if method == 'correlation':
            # Выбор по корреляции с целевой переменной
            if hasattr(features, 'corrwith'):
                correlations = features.corrwith(target)
                if hasattr(correlations, 'abs'):
                    correlations = correlations.abs()
                    if hasattr(correlations, '__gt__'):
                        selected_features = correlations[correlations > threshold].index
                        return features[selected_features]
            return features
        
        elif method == 'variance':
            # Выбор по дисперсии
            variances = features.var()
            selected_features = variances[variances > threshold].index
            return features[selected_features]
        
        else:
            return features
