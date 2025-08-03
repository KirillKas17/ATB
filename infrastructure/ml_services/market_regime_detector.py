"""
Детектор рыночных режимов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas import Series, DataFrame

class MarketRegimeDetector:
    """Детектор рыночных режимов на основе кластеризации."""

    def __init__(self, n_regimes: int = 3) -> None:
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.is_fitted = False

    def extract_features(self, data: DataFrame) -> DataFrame:
        """Извлечение признаков для детекции режимов."""
        # Проверяем, что data является DataFrame
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame()
        
        # Проверяем наличие необходимых колонок
        if 'close' not in data.columns:
            return pd.DataFrame()
        
        features = pd.DataFrame()
        
        # Волатильность
        if hasattr(data['close'], 'pct_change') and hasattr(data['close'].pct_change(), 'rolling'):
            features['volatility'] = data['close'].pct_change().rolling(window=20).std()
        else:
            features['volatility'] = pd.Series([0.0] * len(data))
        
        # Тренд
        if hasattr(data['close'], 'rolling'):
            features['trend'] = data['close'].rolling(window=20).mean().pct_change()
        else:
            features['trend'] = pd.Series([0.0] * len(data))
        
        # Объем
        if 'volume' in data.columns:
            if hasattr(data['volume'], 'rolling'):
                features['volume_ratio'] = data['volume'].rolling(window=20).mean() / data['volume'].rolling(window=60).mean()
            else:
                features['volume_ratio'] = pd.Series([1.0] * len(data))
        else:
            features['volume_ratio'] = pd.Series([1.0] * len(data))
        
        # Моментум
        if hasattr(data['close'], 'pct_change'):
            features['momentum'] = data['close'].pct_change(periods=5)
        else:
            features['momentum'] = pd.Series([0.0] * len(data))
        
        # Диапазон
        if 'high' in data.columns and 'low' in data.columns:
            features['range'] = (data['high'] - data['low']) / data['close']
        else:
            if hasattr(data['close'], 'rolling'):
                features['range'] = data['close'].rolling(window=5).max() - data['close'].rolling(window=5).min()
            else:
                features['range'] = pd.Series([0.0] * len(data))
        
        # Проверяем наличие метода dropna у DataFrame
        if hasattr(features, 'dropna'):
            return features.dropna()
        else:
            return features

    def fit(self, data: DataFrame) -> None:
        """Обучение детектора режимов."""
        features = self.extract_features(data)
        
        # Масштабирование
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA для уменьшения размерности
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Кластеризация
        self.kmeans.fit(features_pca)
        self.is_fitted = True

    def predict(self, data: DataFrame) -> np.ndarray:
        """Предсказание режимов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        features = self.extract_features(data)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        predictions = self.kmeans.predict(features_pca)
        return predictions.astype(np.int64)

    def get_regime_characteristics(self, data: DataFrame) -> Dict[str, Dict[str, float]]:
        """Получение характеристик режимов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")
        
        features = self.extract_features(data)
        regimes = self.predict(data)
        
        characteristics: Dict[str, Dict[str, float]] = {}
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            if regime_mask.sum() > 0:
                # Проверяем наличие метода iloc у DataFrame
                if isinstance(features, pd.DataFrame) and len(features) > 0:
                    regime_features = features.iloc[regime_mask]
                else:
                    # Альтернативный способ доступа к данным
                    regime_features = features[regime_mask]
                
                characteristics[f"regime_{regime}"] = {
                    'volatility_mean': float(regime_features['volatility'].mean()),
                    'trend_mean': float(regime_features['trend'].mean()),
                    'volume_ratio_mean': float(regime_features['volume_ratio'].mean()),
                    'momentum_mean': float(regime_features['momentum'].mean()),
                    'range_mean': float(regime_features['range'].mean()),
                    'frequency': float(regime_mask.sum() / len(regimes))
                }
        
        return characteristics
