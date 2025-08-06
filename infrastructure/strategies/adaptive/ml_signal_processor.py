"""
Обработчик ML сигналов для адаптивных стратегий
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from shared.numpy_utils import np

from infrastructure.core.technical_analysis import calculate_rsi
# from infrastructure.ml_services.advanced_price_predictor import MetaLearner


class MLSignalProcessor:
    """Обработчик ML сигналов"""

    def __init__(self, meta_learner: Optional[Any] = None) -> None:
        self.meta_learner: Optional[Any] = meta_learner

    def get_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Получение ML предсказаний
        """
        try:
            if self.meta_learner:
                features: Dict[str, float] = self._extract_features(data)
                # Исправление: если meta_learner имеет метод predict, используем его, иначе вызываем как torch-модель
                if hasattr(self.meta_learner, 'predict'):
                    predictions = self.meta_learner.predict(features)
                else:
                    try:
                        import torch
                        features_list: List[float] = list(features.values())
                        features_tensor = torch.tensor(features_list, dtype=torch.float32).unsqueeze(0)
                        if hasattr(self.meta_learner, 'forward'):
                            import inspect
                            sig = inspect.signature(self.meta_learner.forward)
                            if 'task_embedding' in sig.parameters:
                                task_embedding = torch.zeros(1, 64)
                                output = self.meta_learner.forward(features_tensor, task_embedding=task_embedding)
                            else:
                                output = self.meta_learner.forward(features_tensor)
                        elif callable(self.meta_learner):
                            output = self.meta_learner(features_tensor)
                        else:
                            output = torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32)
                        import torch.nn.functional as F
                        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
                        predictions = {
                            'buy_probability': float(probs[0]),
                            'hold_probability': float(probs[1]), 
                            'sell_probability': float(probs[2])
                        }
                    except ImportError:
                        # Torch недоступен, используем базовые расчеты
                        predictions = self._calculate_basic_predictions(features)
                    
                return predictions
            else:
                # Если meta_learner недоступен, используем базовые расчеты
                features = self._extract_features(data)
                return self._calculate_basic_predictions(features)
        
        except Exception as e:
            logger.warning(f"Ошибка при получении ML предсказаний: {e}")
            return self._get_default_predictions()

    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Извлечение признаков из данных"""
        try:
            if len(data) < 20:
                return self._get_default_features()
            
            features: Dict[str, float] = {}
            
            # Технические индикаторы
            close_prices = data['close'].astype(float)
            
            # RSI
            rsi_values = calculate_rsi(close_prices)
            features['rsi'] = float(rsi_values.iloc[-1]) if len(rsi_values) > 0 else 50.0
            
            # Скользящие средние
            features['sma_20'] = float(close_prices.rolling(20).mean().iloc[-1])
            features['sma_50'] = float(close_prices.rolling(50).mean().iloc[-1]) if len(close_prices) >= 50 else features['sma_20']
            
            # Волатильность
            features['volatility'] = float(close_prices.pct_change().rolling(20).std().iloc[-1] * 100)
            
            # Объем (если доступен)
            if 'volume' in data.columns:
                volume_series = data['volume'].astype(float)
                features['volume_ratio'] = float(volume_series.iloc[-1] / volume_series.rolling(20).mean().iloc[-1])
            else:
                features['volume_ratio'] = 1.0
            
            # Ценовая динамика
            features['price_change'] = float((close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20] * 100)
            
            return features
            
        except Exception as e:
            logger.warning(f"Ошибка при извлечении признаков: {e}")
            return self._get_default_features()

    def _calculate_basic_predictions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Базовые расчеты предсказаний без ML"""
        try:
            rsi = features.get('rsi', 50.0)
            price_change = features.get('price_change', 0.0)
            volatility = features.get('volatility', 1.0)
            volume_ratio = features.get('volume_ratio', 1.0)
            
            # Простая логика на основе технических индикаторов
            buy_score = 0.0
            sell_score = 0.0
            
            # RSI анализ
            if rsi < 30:
                buy_score += 0.3
            elif rsi > 70:
                sell_score += 0.3
            
            # Ценовая динамика
            if price_change > 2:
                buy_score += 0.2
            elif price_change < -2:
                sell_score += 0.2
            
            # Объем
            if volume_ratio > 1.5:
                if price_change > 0:
                    buy_score += 0.2
                else:
                    sell_score += 0.2
            
            # Нормализация
            total_score = buy_score + sell_score
            if total_score > 0:
                buy_probability = buy_score / total_score
                sell_probability = sell_score / total_score
            else:
                buy_probability = 0.33
                sell_probability = 0.33
            
            hold_probability = 1.0 - buy_probability - sell_probability
            
            return {
                'buy_probability': max(0.0, min(1.0, buy_probability)),
                'hold_probability': max(0.0, min(1.0, hold_probability)),
                'sell_probability': max(0.0, min(1.0, sell_probability))
            }
            
        except Exception as e:
            logger.warning(f"Ошибка в базовых расчетах: {e}")
            return self._get_default_predictions()

    def _get_default_features(self) -> Dict[str, float]:
        """Признаки по умолчанию"""
        return {
            'rsi': 50.0,
            'sma_20': 100.0,
            'sma_50': 100.0,
            'volatility': 1.0,
            'volume_ratio': 1.0,
            'price_change': 0.0
        }

    def _get_default_predictions(self) -> Dict[str, float]:
        """Предсказания по умолчанию"""
        return {
            'buy_probability': 0.33,
            'hold_probability': 0.34,
            'sell_probability': 0.33
        }

    def get_signal_strength(self, predictions: Dict[str, float]) -> float:
        """Получение силы сигнала"""
        try:
            buy_prob = predictions.get('buy_probability', 0.33)
            sell_prob = predictions.get('sell_probability', 0.33)
            hold_prob = predictions.get('hold_probability', 0.34)
            
            max_prob = max(buy_prob, sell_prob, hold_prob)
            
            # Сила сигнала - отклонение от нейтрального состояния
            strength = (max_prob - 0.33) / 0.67
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logger.warning(f"Ошибка при расчете силы сигнала: {e}")
            return 0.0

    def get_signal_direction(self, predictions: Dict[str, float]) -> str:
        """Получение направления сигнала"""
        try:
            buy_prob = predictions.get('buy_probability', 0.33)
            sell_prob = predictions.get('sell_probability', 0.33)
            hold_prob = predictions.get('hold_probability', 0.34)
            
            max_prob = max(buy_prob, sell_prob, hold_prob)
            
            if max_prob == buy_prob:
                return 'buy'
            elif max_prob == sell_prob:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.warning(f"Ошибка при определении направления сигнала: {e}")
            return 'hold'
