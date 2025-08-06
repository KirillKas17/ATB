"""
Обработчик ML сигналов для адаптивных стратегий
"""

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from shared.numpy_utils import np

from infrastructure.core.technical_analysis import calculate_rsi
# from infrastructure.ml_services.advanced_price_predictor import MetaLearner


class MLSignalProcessor:
    """Обработчик ML сигналов"""

    def __init__(self, meta_learner: Optional[Any] = None) -> None:
        self.meta_learner = meta_learner

    def get_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Получение ML предсказаний
        """
        try:
            if self.meta_learner:
                features = self._extract_features(data)
                # Исправление: если meta_learner имеет метод predict, используем его, иначе вызываем как torch-модель
                if hasattr(self.meta_learner, 'predict'):
                    predictions = self.meta_learner.predict(features)
                else:
                    try:
                        import torch
                        features_list = list(features.values())
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
                            'buy_prob': float(probs[0]),
                            'sell_prob': float(probs[1]) if len(probs) > 1 else 0.0,
                            'hold_prob': float(probs[2]) if len(probs) > 2 else 0.0,
                            'confidence': float(max(probs)),
                        }
                    except Exception as e:
                        logger.error(f"Error in torch processing: {str(e)}")
                        predictions = {
                            'buy_prob': 0.33,
                            'sell_prob': 0.33,
                            'hold_prob': 0.34,
                            'confidence': 0.34,
                        }
                return predictions
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting ML predictions: {str(e)}")
            return {}

    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Извлечение признаков для ML"""
        try:
            features = {}
            
            # Безопасная проверка данных
            if len(data) < 50:
                return {"rsi": 50.0, "volatility": 0.0, "momentum": 0.0, "volume_ratio": 1.0, "trend_strength": 0.0}
            
            # Технические индикаторы - RSI
            try:
                rsi_series = calculate_rsi(data["close"], 14)
                # Исправление: добавляем явную аннотацию типа
                rsi_series_alt: pd.Series = rsi_series if hasattr(rsi_series, 'iloc') else pd.Series([50.0])
                if hasattr(rsi_series, 'iloc') and callable(rsi_series.iloc):
                    features["rsi"] = float(rsi_series.iloc[-1]) if not rsi_series.empty and len(rsi_series) > 0 else 50.0
                else:
                    features["rsi"] = 50.0
            except (IndexError, TypeError, AttributeError):
                features["rsi"] = 50.0
            
            # Волатильность
            try:
                if len(data) >= 20:
                    pct_change = data["close"].pct_change()
                    if len(pct_change) >= 20:
                        rolling_std = pct_change.rolling(20).std()
                        if len(rolling_std) > 0:
                            features["volatility"] = float(rolling_std.iloc[-1])
                        else:
                            features["volatility"] = 0.0
                    else:
                        features["volatility"] = 0.0
                else:
                    features["volatility"] = 0.0
            except (IndexError, TypeError):
                features["volatility"] = 0.0
            
            # Моментум
            try:
                if len(data) >= 10:
                    pct_change_10 = data["close"].pct_change(10)
                    if len(pct_change_10) > 0:
                        features["momentum"] = float(pct_change_10.iloc[-1])
                    else:
                        features["momentum"] = 0.0
                else:
                    features["momentum"] = 0.0
            except (IndexError, TypeError):
                features["momentum"] = 0.0
            
            # Объемные метрики
            try:
                if len(data) >= 20 and "volume" in data.columns:
                    current_volume = data["volume"].iloc[-1]
                    volume_rolling = data["volume"].rolling(20).mean()
                    if len(volume_rolling) > 0:
                        avg_volume = volume_rolling.iloc[-1]
                        if avg_volume > 0:
                            features["volume_ratio"] = float(current_volume / avg_volume)
                        else:
                            features["volume_ratio"] = 1.0
                    else:
                        features["volume_ratio"] = 1.0
                else:
                    features["volume_ratio"] = 1.0
            except (IndexError, TypeError, ZeroDivisionError):
                features["volume_ratio"] = 1.0
            
            # Трендовые метрики
            try:
                ema_20 = data["close"].ewm(span=20).mean()
                ema_50 = data["close"].ewm(span=50).mean()
                if len(ema_20) > 0 and len(ema_50) > 0:
                    ema_20_val = ema_20.iloc[-1]
                    ema_50_val = ema_50.iloc[-1]
                    if ema_50_val > 0:
                        features["trend_strength"] = float(abs(ema_20_val - ema_50_val) / ema_50_val)
                    else:
                        features["trend_strength"] = 0.0
                else:
                    features["trend_strength"] = 0.0
            except (IndexError, TypeError, ZeroDivisionError):
                features["trend_strength"] = 0.0
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {"rsi": 50.0, "volatility": 0.0, "momentum": 0.0, "volume_ratio": 1.0, "trend_strength": 0.0}

    def adapt_signal(
        self, base_signal: Any, ml_predictions: Dict[str, float], regime: str
    ) -> Any:
        """Адаптация сигнала на основе ML предсказаний"""
        try:
            # Адаптация направления
            if "direction_confidence" in ml_predictions:
                ml_confidence = ml_predictions["direction_confidence"]
                base_signal.confidence = (base_signal.confidence + ml_confidence) / 2
            # Адаптация размера позиции
            if "position_size" in ml_predictions:
                base_signal.position_size = ml_predictions["position_size"]
            # Адаптация стоп-лосса
            if "stop_loss_adjustment" in ml_predictions:
                adjustment = ml_predictions["stop_loss_adjustment"]
                if base_signal.stop_loss:
                    base_signal.stop_loss *= 1 + adjustment
            # Адаптация тейк-профита
            if "take_profit_adjustment" in ml_predictions:
                adjustment = ml_predictions["take_profit_adjustment"]
                if base_signal.take_profit:
                    base_signal.take_profit *= 1 + adjustment
            return base_signal
        except Exception as e:
            logger.error(f"Error adapting signal: {str(e)}")
            return base_signal

    def calculate_ml_confidence(
        self, data: pd.DataFrame, ml_predictions: Dict[str, float]
    ) -> float:
        """Расчет уверенности ML модели"""
        try:
            ml_confidence = ml_predictions.get("confidence", 0.5)
            # Оценка качества данных
            data_quality = 0.9
            try:
                if not data.empty and hasattr(data, 'isnull'):
                    if hasattr(data.isnull(), 'to_numpy'):
                        if np.any(data.isnull().to_numpy()):  # type: ignore[attr-defined]
                            data_quality = 0.6
            except (AttributeError, TypeError):
                data_quality = 0.9
            confidence = (ml_confidence + data_quality) / 2
            return max(0.1, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error calculating ML confidence: {str(e)}")
            return 0.5
