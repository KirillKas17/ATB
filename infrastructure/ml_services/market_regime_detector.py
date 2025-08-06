"""
Детектор рыночных режимов.
"""

import logging
from typing import Any, Dict
import pandas as pd
from shared.numpy_utils import np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DataFrame = pd.DataFrame

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
        
        # Заполнение NaN значений
        features = features.fillna(0.0)
        
        return features

    def fit(self, data: DataFrame) -> None:
        """Обучение детектора режимов."""
        features = self.extract_features(data)
        if features is None or features.empty:
            logging.warning("Нет признаков для обучения детектора режимов")
            return
        
        # Масштабирование и снижение размерности
        features_scaled = self.scaler.fit_transform(features)
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Кластеризация
        self.kmeans.fit(features_pca)
        self.is_fitted = True

    def predict(self, data: DataFrame) -> np.ndarray:
        """Предсказание режима для новых данных."""
        if not self.is_fitted:
            self.fit(data)
        
        features = self.extract_features(data)
        if features is None or features.empty:
            return np.array([0])
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        return self.kmeans.predict(features_pca)

    def detect_regime(self, data: DataFrame) -> Dict[str, Any]:
        """
        Определяет рыночный режим и возвращает словарь с информацией.
        
        Returns:
            Dict содержащий:
            - trend: 'bullish', 'bearish', или 'sideways'
            - confidence: уровень уверенности (0.0-1.0)
            - volatility: уровень волатильности
        """
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return {
                    'trend': 'sideways',
                    'confidence': 0.0,
                    'volatility': 0.0
                }

            # Получаем признаки
            features = self.extract_features(data)
            if features is None or features.empty:
                return {
                    'trend': 'sideways', 
                    'confidence': 0.0,
                    'volatility': 0.0
                }

            # Анализ тренда по скользящей средней
            if len(data) >= 20:
                current_price = data['close'].iloc[-1]
                ma20 = data['close'].rolling(20).mean().iloc[-1]
                
                # Определяем направление тренда
                if current_price > ma20 * 1.01:  # 1% буфер
                    trend = 'bullish'
                    confidence = min((current_price - ma20) / ma20 * 10, 1.0)
                elif current_price < ma20 * 0.99:  # 1% буфер
                    trend = 'bearish'
                    confidence = min((ma20 - current_price) / ma20 * 10, 1.0)
                else:
                    trend = 'sideways'
                    confidence = 0.5
            else:
                # Недостаточно данных для анализа
                trend = 'sideways'
                confidence = 0.0

            # Расчет волатильности
            if len(data) >= 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
                if pd.isna(volatility):
                    volatility = 0.0
            else:
                volatility = 0.0

            # Нормализация значений
            confidence = max(0.0, min(1.0, confidence))
            volatility = max(0.0, min(1.0, volatility * 100))  # Масштабируем

            return {
                'trend': trend,
                'confidence': confidence,
                'volatility': volatility
            }

        except Exception as e:
            logging.error(f"Ошибка в определении рыночного режима: {e}")
            return {
                'trend': 'sideways',
                'confidence': 0.0,
                'volatility': 0.0
            }

    def get_regime_characteristics(self, data: DataFrame) -> Dict[str, Dict[str, float]]:
        """Получить характеристики режимов."""
        if not self.is_fitted:
            self.fit(data)
        
        features = self.extract_features(data)
        if features is None or features.empty:
            return {}
        
        regimes = self.predict(data)
        
        characteristics = {}
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() > 0:
                regime_features = features[mask]
                characteristics[f"regime_{regime_id}"] = {
                    'avg_volatility': float(regime_features['volatility'].mean()),
                    'avg_trend': float(regime_features['trend'].mean()),
                    'avg_volume': float(regime_features['volume_ratio'].mean())
                }
        
        return characteristics
