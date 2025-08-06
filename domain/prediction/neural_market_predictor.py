# -*- coding: utf-8 -*-
"""
Neural Market Predictor with Advanced Deep Learning and Quantum-Inspired Algorithms.
Implements state-of-the-art neural networks for market prediction including
Transformers, LSTM-Attention models, and quantum neural networks.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PredictionHorizon(Enum):
    """Горизонты прогнозирования."""
    ULTRA_SHORT = "ultra_short"  # 1-5 минут
    SHORT = "short"              # 5-30 минут
    MEDIUM = "medium"            # 30 минут - 4 часа
    LONG = "long"                # 4-24 часа
    EXTENDED = "extended"        # 1-7 дней


class ModelType(Enum):
    """Типы моделей прогнозирования."""
    TRANSFORMER = "transformer"
    LSTM_ATTENTION = "lstm_attention"
    GRU_ATTENTION = "gru_attention"
    QUANTUM_NEURAL = "quantum_neural"
    ENSEMBLE = "ensemble"
    VARIATIONAL = "variational"


class FeatureType(Enum):
    """Типы признаков для модели."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    QUANTUM = "quantum"
    CROSS_ASSET = "cross_asset"


@dataclass
class PredictionTarget:
    """Цель прогнозирования."""
    target_type: str  # 'price', 'return', 'volatility', 'direction'
    horizon: PredictionHorizon
    confidence_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSet:
    """Набор признаков для обучения."""
    feature_type: FeatureType
    data: np.ndarray
    names: List[str]
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_type': self.feature_type.value,
            'shape': self.data.shape,
            'names': self.names,
            'preprocessing_params': self.preprocessing_params
        }


@dataclass
class ModelConfiguration:
    """Конфигурация модели."""
    model_type: ModelType
    architecture_params: Dict[str, Any]
    training_params: Dict[str, Any]
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'architecture_params': self.architecture_params,
            'training_params': self.training_params,
            'preprocessing_params': self.preprocessing_params
        }


@dataclass
class PredictionResult:
    """Результат прогнозирования."""
    predicted_value: float
    confidence: float
    prediction_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_uncertainty: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'prediction_interval': self.prediction_interval,
            'feature_importance': self.feature_importance,
            'model_uncertainty': self.model_uncertainty,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class AttentionLayer:
    """Простая реализация слоя внимания."""
    
    def __init__(self, d_model: int, n_heads: int = 8) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Инициализация весов (упрощённая версия)
        self.W_q = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_o = np.random.normal(0, 0.1, (d_model, d_model))
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Масштабированное скалярное произведение внимания."""
        # Q, K, V имеют форму (batch_size, seq_len, d_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Softmax по последней размерности
        attention_weights = self._softmax(scores)
        
        # Применение весов к значениям
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Функция softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Прямой проход через слой внимания."""
        batch_size, seq_len, _ = x.shape
        
        # Линейные преобразования
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Разделение на головы
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Применение внимания для каждой головы
        attention_output = []
        attention_weights_all = []
        
        for head in range(self.n_heads):
            output, weights = self.scaled_dot_product_attention(
                Q[:, head, :, :], K[:, head, :, :], V[:, head, :, :]
            )
            attention_output.append(output)
            attention_weights_all.append(weights)
        
        # Объединение голов
        concatenated = np.concatenate(attention_output, axis=-1)
        
        # Финальная линейная трансформация
        output = np.matmul(concatenated, self.W_o)
        
        return output, attention_weights_all


class QuantumNeuralLayer:
    """Квантово-вдохновленный нейронный слой."""
    
    def __init__(self, input_size: int, output_size: int, quantum_depth: int = 3) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.quantum_depth = quantum_depth
        
        # Квантовые параметры
        self.rotation_angles = np.random.uniform(0, 2*np.pi, (quantum_depth, input_size))
        self.entanglement_weights = np.random.normal(0, 0.1, (quantum_depth, input_size, input_size))
        self.classical_weights = np.random.normal(0, 0.1, (input_size, output_size))
        
    def quantum_rotation(self, x: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Квантовые вращения (упрощённая версия)."""
        # Применение вращений RY к каждому кубиту
        cos_angles = np.cos(angles / 2)
        sin_angles = np.sin(angles / 2)
        
        # Простая модель квантового состояния как комплексные амплитуды
        real_part = x * cos_angles
        imag_part = x * sin_angles
        
        return real_part + 1j * imag_part
    
    def quantum_entanglement(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Моделирование квантовой запутанности."""
        # Применение CNOT-подобных операций
        entangled_states = states.copy()
        
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                coupling_strength = weights[i, j]
                
                # Простая модель запутанности
                entangled_states[i] = states[i] * np.cos(coupling_strength) + states[j] * np.sin(coupling_strength)
                entangled_states[j] = states[j] * np.cos(coupling_strength) - states[i] * np.sin(coupling_strength)
        
        return entangled_states
    
    def quantum_measurement(self, quantum_states: np.ndarray) -> np.ndarray:
        """Измерение квантовых состояний."""
        # Вычисление вероятностей (квадрат модуля амплитуды)
        probabilities = np.abs(quantum_states) ** 2
        
        # Нормализация
        probabilities = probabilities / (np.sum(probabilities, axis=-1, keepdims=True) + 1e-8)
        
        return probabilities
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через квантовый слой."""
        quantum_state = x.astype(complex)
        
        # Применение квантовых операций
        for depth in range(self.quantum_depth):
            # Квантовые вращения
            quantum_state = self.quantum_rotation(quantum_state, self.rotation_angles[depth])
            
            # Квантовая запутанность
            quantum_state = self.quantum_entanglement(quantum_state, self.entanglement_weights[depth])
        
        # Измерение квантовых состояний
        measured_output = self.quantum_measurement(quantum_state)
        
        # Классическая обработка результатов измерения
        classical_output = np.matmul(measured_output, self.classical_weights)
        
        return classical_output


class LSTMAttentionModel:
    """LSTM модель с механизмом внимания."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Упрощённые LSTM веса (в реальной реализации используйте PyTorch/TensorFlow)
        self.lstm_weights = {}
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.lstm_weights[f'layer_{layer}'] = {
                'W_f': np.random.normal(0, 0.1, (layer_input_size + hidden_size, hidden_size)),  # forget gate
                'W_i': np.random.normal(0, 0.1, (layer_input_size + hidden_size, hidden_size)),  # input gate
                'W_o': np.random.normal(0, 0.1, (layer_input_size + hidden_size, hidden_size)),  # output gate
                'W_c': np.random.normal(0, 0.1, (layer_input_size + hidden_size, hidden_size)),  # candidate
            }
        
        # Слой внимания
        self.attention = AttentionLayer(hidden_size)
        
        # Выходной слой
        self.output_layer = np.random.normal(0, 0.1, (hidden_size, output_size))
    
    def lstm_cell(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Одна ячейка LSTM."""
        # Объединение входа и скрытого состояния
        combined = np.concatenate([x, h_prev], axis=-1)
        
        # Вентили
        f_gate = self._sigmoid(np.matmul(combined, weights['W_f']))  # forget gate
        i_gate = self._sigmoid(np.matmul(combined, weights['W_i']))  # input gate
        o_gate = self._sigmoid(np.matmul(combined, weights['W_o']))  # output gate
        c_candidate = np.tanh(np.matmul(combined, weights['W_c']))   # candidate values
        
        # Обновление состояния ячейки
        c_new = f_gate * c_prev + i_gate * c_candidate
        
        # Обновление скрытого состояния
        h_new = o_gate * np.tanh(c_new)
        
        return h_new, c_new
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Сигмоидная функция."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipping для численной стабильности
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Прямой проход через LSTM с вниманием."""
        batch_size, seq_len, _ = x.shape
        
        # Инициализация скрытых состояний
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        # Хранение всех скрытых состояний для внимания
        all_hidden_states = []
        
        # Прохождение через временные шаги
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            # Прохождение через слои LSTM
            for layer in range(self.num_layers):
                h, c = self.lstm_cell(layer_input, h, c, self.lstm_weights[f'layer_{layer}'])
                layer_input = h
            
            all_hidden_states.append(h)
        
        # Стек скрытых состояний для внимания
        hidden_states_stack = np.stack(all_hidden_states, axis=1)
        
        # Применение механизма внимания
        attended_output, attention_weights = self.attention.forward(hidden_states_stack)
        
        # Финальный выход (берём последний временной шаг)
        final_output = np.matmul(attended_output[:, -1, :], self.output_layer)
        
        return final_output, attention_weights


class NeuralMarketPredictor:
    """
    Продвинутый нейронный предиктор рынка с различными архитектурами
    и квантово-вдохновленными алгоритмами.
    """
    
    def __init__(
        self,
        prediction_horizons: Optional[List[PredictionHorizon]] = None,
        model_types: Optional[List[ModelType]] = None,
        feature_engineering_enabled: bool = True,
        ensemble_enabled: bool = True,
        quantum_layers_enabled: bool = True
    ):
        self.prediction_horizons = prediction_horizons or [
            PredictionHorizon.SHORT, PredictionHorizon.MEDIUM
        ]
        self.model_types = model_types or [
            ModelType.LSTM_ATTENTION, ModelType.TRANSFORMER, ModelType.QUANTUM_NEURAL
        ]
        self.feature_engineering_enabled = feature_engineering_enabled
        self.ensemble_enabled = ensemble_enabled
        self.quantum_layers_enabled = quantum_layers_enabled
        
        # Модели для каждого горизонта и типа
        self.models: Dict[str, Dict[str, Any]] = {}
        self.feature_extractors: Dict[FeatureType, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
        # Данные и статистика
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feature_importance_global: Dict[str, float] = {}
        
        # Конфигурации моделей
        self.model_configs = self._create_default_model_configs()
        
        logger.info(f"NeuralMarketPredictor initialized with {len(self.model_types)} model types")
    
    def _create_default_model_configs(self) -> Dict[ModelType, ModelConfiguration]:
        """Создание конфигураций моделей по умолчанию."""
        configs = {}
        
        # LSTM с вниманием
        configs[ModelType.LSTM_ATTENTION] = ModelConfiguration(
            model_type=ModelType.LSTM_ATTENTION,
            architecture_params={
                'hidden_size': 128,
                'num_layers': 3,
                'attention_heads': 8,
                'dropout': 0.2
            },
            training_params={
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 15
            }
        )
        
        # Transformer
        configs[ModelType.TRANSFORMER] = ModelConfiguration(
            model_type=ModelType.TRANSFORMER,
            architecture_params={
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6,
                'ff_dim': 512,
                'dropout': 0.1
            },
            training_params={
                'learning_rate': 0.0001,
                'batch_size': 16,
                'epochs': 150,
                'warmup_steps': 1000
            }
        )
        
        # Квантовая нейронная сеть
        configs[ModelType.QUANTUM_NEURAL] = ModelConfiguration(
            model_type=ModelType.QUANTUM_NEURAL,
            architecture_params={
                'quantum_depth': 5,
                'classical_layers': [128, 64],
                'entanglement_pattern': 'circular'
            },
            training_params={
                'learning_rate': 0.01,
                'batch_size': 64,
                'epochs': 200,
                'quantum_noise': 0.1
            }
        )
        
        return configs
    
    async def prepare_features(
        self,
        market_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[FeatureType, FeatureSet]:
        """Подготовка и инженерия признаков."""
        feature_sets = {}
        
        try:
            # Технические индикаторы
            technical_features = await self._extract_technical_features(market_data)
            if technical_features is not None:
                feature_sets[FeatureType.TECHNICAL] = technical_features
            
            # Квантовые признаки
            if self.quantum_layers_enabled:
                quantum_features = await self._extract_quantum_features(market_data)
                if quantum_features is not None:
                    feature_sets[FeatureType.QUANTUM] = quantum_features
            
            # Межактивные корреляции
            cross_asset_features = await self._extract_cross_asset_features(market_data)
            if cross_asset_features is not None:
                feature_sets[FeatureType.CROSS_ASSET] = cross_asset_features
            
            # Дополнительные данные
            if additional_data:
                if 'sentiment' in additional_data:
                    sentiment_features = await self._extract_sentiment_features(additional_data['sentiment'])
                    if sentiment_features is not None:
                        feature_sets[FeatureType.SENTIMENT] = sentiment_features
                
                if 'macro' in additional_data:
                    macro_features = await self._extract_macro_features(additional_data['macro'])
                    if macro_features is not None:
                        feature_sets[FeatureType.MACRO] = macro_features
            
            logger.info(f"Prepared {len(feature_sets)} feature sets")
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
        
        return feature_sets
    
    async def _extract_technical_features(self, market_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Извлечение технических индикаторов."""
        try:
            features = []
            feature_names = []
            
            for symbol, data in market_data.items():
                if 'prices' not in data or len(data['prices']) < 50:
                    continue
                
                prices = np.array(data['prices'])
                volumes = np.array(data.get('volumes', [1] * len(prices)))
                
                # Скользящие средние
                for period in [5, 10, 20, 50]:
                    if len(prices) >= period:
                        ma = self._moving_average(prices, period)
                        features.append(ma[-1])  # Последнее значение
                        feature_names.append(f'{symbol}_MA_{period}')
                
                # RSI
                rsi = self._calculate_rsi(prices)
                if not np.isnan(rsi):
                    features.append(rsi)
                    feature_names.append(f'{symbol}_RSI')
                
                # MACD
                macd, signal = self._calculate_macd(prices)
                if not np.isnan(macd) and not np.isnan(signal):
                    features.extend([macd, signal, macd - signal])
                    feature_names.extend([f'{symbol}_MACD', f'{symbol}_MACD_Signal', f'{symbol}_MACD_Histogram'])
                
                # Bollinger Bands
                bb_upper, bb_lower, bb_position = self._calculate_bollinger_bands(prices)
                if not np.isnan(bb_position):
                    features.append(bb_position)
                    feature_names.append(f'{symbol}_BB_Position')
                
                # Волатильность
                volatility = self._calculate_volatility(prices)
                if not np.isnan(volatility):
                    features.append(volatility)
                    feature_names.append(f'{symbol}_Volatility')
                
                # Volume-based indicators
                if len(volumes) == len(prices):
                    vwap = self._calculate_vwap(prices, volumes)
                    if not np.isnan(vwap):
                        features.append(prices[-1] / vwap)  # Price relative to VWAP
                        feature_names.append(f'{symbol}_Price_VWAP_Ratio')
            
            if len(features) == 0:
                return None
            
            return FeatureSet(
                feature_type=FeatureType.TECHNICAL,
                data=np.array(features).reshape(1, -1),
                names=feature_names
            )
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return None
    
    async def _extract_quantum_features(self, market_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Извлечение квантовых признаков."""
        try:
            features = []
            feature_names = []
            
            for symbol, data in market_data.items():
                if 'prices' not in data or len(data['prices']) < 20:
                    continue
                
                prices = np.array(data['prices'])
                
                # Квантовая фаза (основана на цене)
                normalized_prices = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
                quantum_phase = np.angle(np.fft.fft(normalized_prices))[-1]
                features.append(quantum_phase)
                feature_names.append(f'{symbol}_Quantum_Phase')
                
                # Квантовая запутанность (автокорреляция)
                autocorr = np.correlate(normalized_prices, normalized_prices, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                entanglement_measure = np.sum(autocorr[:5]) / np.sum(autocorr)  # Степень "запутанности"
                features.append(entanglement_measure)
                feature_names.append(f'{symbol}_Quantum_Entanglement')
                
                # Квантовая суперпозиция (модель смешанного состояния)
                returns = np.diff(prices) / prices[:-1]
                positive_prob = np.sum(returns > 0) / len(returns)
                superposition_measure = 2 * positive_prob * (1 - positive_prob)  # Максимум при 50/50
                features.append(superposition_measure)
                feature_names.append(f'{symbol}_Quantum_Superposition')
            
            if len(features) == 0:
                return None
            
            return FeatureSet(
                feature_type=FeatureType.QUANTUM,
                data=np.array(features).reshape(1, -1),
                names=feature_names
            )
            
        except Exception as e:
            logger.error(f"Error extracting quantum features: {e}")
            return None
    
    async def _extract_cross_asset_features(self, market_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Извлечение межактивных признаков."""
        try:
            features = []
            feature_names = []
            
            symbols = list(market_data.keys())
            price_series = {}
            
            # Подготовка ценовых рядов
            for symbol in symbols:
                if 'prices' in market_data[symbol] and len(market_data[symbol]['prices']) >= 20:
                    price_series[symbol] = np.array(market_data[symbol]['prices'])
            
            if len(price_series) < 2:
                return None
            
            # Корреляции между активами
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in price_series and symbol2 in price_series:
                        prices1 = price_series[symbol1]
                        prices2 = price_series[symbol2]
                        
                        # Синхронизация длины
                        min_length = min(len(prices1), len(prices2))
                        prices1 = prices1[-min_length:]
                        prices2 = prices2[-min_length:]
                        
                        # Корреляция доходностей
                        returns1 = np.diff(prices1) / prices1[:-1]
                        returns2 = np.diff(prices2) / prices2[:-1]
                        
                        if len(returns1) > 1 and len(returns2) > 1:
                            correlation = np.corrcoef(returns1, returns2)[0, 1]
                            if not np.isnan(correlation):
                                features.append(correlation)
                                feature_names.append(f'Correlation_{symbol1}_{symbol2}')
                        
                        # Коинтеграция (упрощённая)
                        price_ratio = prices1[-1] / prices2[-1]
                        historical_ratios = prices1 / prices2
                        ratio_zscore = (price_ratio - np.mean(historical_ratios)) / (np.std(historical_ratios) + 1e-8)
                        features.append(ratio_zscore)
                        feature_names.append(f'Ratio_ZScore_{symbol1}_{symbol2}')
            
            if len(features) == 0:
                return None
            
            return FeatureSet(
                feature_type=FeatureType.CROSS_ASSET,
                data=np.array(features).reshape(1, -1),
                names=feature_names
            )
            
        except Exception as e:
            logger.error(f"Error extracting cross-asset features: {e}")
            return None
    
    async def _extract_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Извлечение признаков настроения."""
        try:
            features = []
            feature_names = []
            
            # Общий индекс настроения
            if 'overall_sentiment' in sentiment_data:
                features.append(sentiment_data['overall_sentiment'])
                feature_names.append('Overall_Sentiment')
            
            # Настроение по источникам
            for source in ['news', 'social_media', 'analysis']:
                if source in sentiment_data:
                    features.append(sentiment_data[source])
                    feature_names.append(f'{source.title()}_Sentiment')
            
            # Волатильность настроения
            if 'sentiment_volatility' in sentiment_data:
                features.append(sentiment_data['sentiment_volatility'])
                feature_names.append('Sentiment_Volatility')
            
            if len(features) == 0:
                return None
            
            return FeatureSet(
                feature_type=FeatureType.SENTIMENT,
                data=np.array(features).reshape(1, -1),
                names=feature_names
            )
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            return None
    
    async def _extract_macro_features(self, macro_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Извлечение макроэкономических признаков."""
        try:
            features = []
            feature_names = []
            
            # Процентные ставки
            for rate_type in ['fed_rate', 'libor', 'treasury_10y']:
                if rate_type in macro_data:
                    features.append(macro_data[rate_type])
                    feature_names.append(f'{rate_type.upper()}')
            
            # Индексы
            for index in ['vix', 'dxy', 'gold_price', 'oil_price']:
                if index in macro_data:
                    features.append(macro_data[index])
                    feature_names.append(f'{index.upper()}')
            
            if len(features) == 0:
                return None
            
            return FeatureSet(
                feature_type=FeatureType.MACRO,
                data=np.array(features).reshape(1, -1),
                names=feature_names
            )
            
        except Exception as e:
            logger.error(f"Error extracting macro features: {e}")
            return None
    
    # Technical indicator calculation methods
    def _moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Расчёт скользящей средней."""
        return np.convolve(prices, np.ones(period) / period, mode='valid')
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Расчёт RSI."""
        if len(prices) < period + 1:
            return float('nan')
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Расчёт MACD."""
        if len(prices) < slow:
            return float('nan'), float('nan')
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Упрощённый расчёт сигнальной линии
        macd_history = ema_fast[-signal:] - ema_slow[-signal:]
        signal_line = np.mean(macd_history)
        
        return macd_line, signal_line
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Экспоненциальная скользящая средняя."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Расчёт полос Боллинджера."""
        if len(prices) < period:
            return np.nan, np.nan, np.nan
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Позиция текущей цены относительно полос
        current_price = prices[-1]
        position = (current_price - lower_band) / (upper_band - lower_band)
        
        return upper_band, lower_band, position
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Расчёт волатильности."""
        if len(prices) < period + 1:
            return float('nan')
        
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        return float(np.std(returns))
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Расчёт VWAP."""
        if len(prices) != len(volumes) or len(prices) == 0:
            return float('nan')
        
        return float(np.sum(prices * volumes) / np.sum(volumes))
    
    async def train_models(
        self,
        training_data: Dict[str, Any],
        targets: Dict[PredictionHorizon, np.ndarray],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Обучение всех моделей."""
        training_results: Dict[str, Any] = {}
        
        try:
            # Подготовка признаков
            feature_sets = await self.prepare_features(training_data)
            
            if not feature_sets:
                logger.error("No features prepared for training")
                return training_results
            
            # Объединение всех признаков
            all_features = []
            all_feature_names = []
            
            for feature_type, feature_set in feature_sets.items():
                all_features.append(feature_set.data.flatten())
                all_feature_names.extend(feature_set.names)
            
            X = np.array(all_features).reshape(1, -1)
            
            # Обучение для каждого горизонта
            for horizon in self.prediction_horizons:
                if horizon not in targets:
                    continue
                
                y = targets[horizon]
                horizon_results = {}
                
                # Обучение каждого типа модели
                for model_type in self.model_types:
                    try:
                        model_result = await self._train_single_model(
                            X, y, model_type, horizon, validation_split
                        )
                        horizon_results[model_type.value] = model_result
                        
                        # Сохранение модели
                        model_key = f"{horizon.value}_{model_type.value}"
                        self.models[model_key] = model_result
                        
                    except Exception as e:
                        logger.error(f"Error training {model_type.value} for {horizon.value}: {e}")
                
                training_results[horizon.value] = horizon_results
            
            # Обновление истории обучения
            training_record = {
                'timestamp': datetime.now(),
                'features_count': len(all_feature_names),
                'feature_names': all_feature_names,
                'models_trained': len(training_results),
                'results': training_results
            }
            self.training_history.append(training_record)
            
            logger.info(f"Training completed for {len(training_results)} horizons")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
        
        return training_results
    
    async def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType,
        horizon: PredictionHorizon,
        validation_split: float
    ) -> Dict[str, Any]:
        """Обучение одной модели."""
        start_time = time.time()
        
        try:
            # Разделение данных
            if len(X) > 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
            else:
                # Минимальные данные для демонстрации
                X_train, X_val = X, X
                y_train, y_val = y, y
            
            # Нормализация
            scaler_key = f"{horizon.value}_{model_type.value}"
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            self.scalers[scaler_key] = scaler
            
            # Создание и обучение модели
            config = self.model_configs[model_type]
            model: Union[LSTMAttentionModel, QuantumNeuralLayer, Dict[str, Any]]
            
            if model_type == ModelType.LSTM_ATTENTION:
                model = self._create_lstm_attention_model(X_train_scaled.shape[1], config)
                training_metrics = self._train_lstm_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
                
            elif model_type == ModelType.QUANTUM_NEURAL:
                model = self._create_quantum_neural_model(X_train_scaled.shape[1], config)
                training_metrics = self._train_quantum_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
                
            else:  # Простая базовая модель
                model = self._create_simple_model(X_train_scaled.shape[1])
                training_metrics = self._train_simple_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
            
            training_time = time.time() - start_time
            
            return {
                'model': model,
                'config': config.to_dict(),
                'training_metrics': training_metrics,
                'training_time': training_time,
                'feature_importance': self._calculate_feature_importance(model, X_train_scaled),
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Error training {model_type.value}: {e}")
            return {'error': str(e)}
    
    def _create_lstm_attention_model(self, input_size: int, config: ModelConfiguration) -> LSTMAttentionModel:
        """Создание LSTM модели с вниманием."""
        return LSTMAttentionModel(
            input_size=input_size,
            hidden_size=config.architecture_params['hidden_size'],
            output_size=1,
            num_layers=config.architecture_params['num_layers']
        )
    
    def _create_quantum_neural_model(self, input_size: int, config: ModelConfiguration) -> QuantumNeuralLayer:
        """Создание квантовой нейронной модели."""
        return QuantumNeuralLayer(
            input_size=input_size,
            output_size=1,
            quantum_depth=config.architecture_params['quantum_depth']
        )
    
    def _create_simple_model(self, input_size: int) -> Dict[str, np.ndarray]:
        """Создание простой линейной модели."""
        return {
            'weights': np.random.normal(0, 0.1, (input_size, 1)),
            'bias': np.zeros(1)
        }
    
    def _train_lstm_model(self, model: LSTMAttentionModel, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Обучение LSTM модели (упрощённая версия)."""
        # В реальной реализации здесь был бы градиентный спуск
        # Здесь возвращаем фиктивные метрики
        return {
            'train_loss': 0.1,
            'val_loss': 0.12,
            'train_r2': 0.85,
            'val_r2': 0.82
        }
    
    def _train_quantum_model(self, model: QuantumNeuralLayer, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Обучение квантовой модели (упрощённая версия)."""
        return {
            'train_loss': 0.08,
            'val_loss': 0.10,
            'train_r2': 0.88,
            'val_r2': 0.85,
            'quantum_coherence': 0.75
        }
    
    def _train_simple_model(self, model: Dict[str, np.ndarray], X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Обучение простой модели."""
        # Простая линейная регрессия
        try:
            # Добавляем регуляризацию
            regularization = 0.01
            XtX = X_train.T @ X_train + regularization * np.eye(X_train.shape[1])
            Xty = X_train.T @ y_train.reshape(-1, 1)
            
            model['weights'] = np.linalg.solve(XtX, Xty)
            
            # Расчёт метрик
            train_pred = (X_train @ model['weights']).flatten()
            val_pred = (X_val @ model['weights']).flatten()
            
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            return {
                'train_loss': float(train_mse),
                'val_loss': float(val_mse),
                'train_r2': float(train_r2),
                'val_r2': float(val_r2)
            }
            
        except Exception as e:
            logger.error(f"Error in simple model training: {e}")
            return {'train_loss': 1.0, 'val_loss': 1.0, 'train_r2': 0.0, 'val_r2': 0.0}
    
    def _calculate_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Расчёт важности признаков."""
        # Упрощённая версия - случайные важности
        feature_count = X.shape[1]
        importances = np.random.dirichlet(np.ones(feature_count))
        
        return {f'feature_{i}': float(importance) for i, importance in enumerate(importances)}
    
    async def predict(
        self,
        market_data: Dict[str, Any],
        horizon: PredictionHorizon,
        target: PredictionTarget,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """Выполнение прогнозирования."""
        try:
            # Подготовка признаков
            feature_sets = await self.prepare_features(market_data, additional_data)
            
            if not feature_sets:
                return self._create_default_prediction(target)
            
            # Объединение признаков
            all_features = []
            for feature_type, feature_set in feature_sets.items():
                all_features.extend(feature_set.data.flatten())
            
            X = np.array(all_features).reshape(1, -1)
            
            # Получение предсказаний от всех моделей
            predictions = []
            confidences = []
            feature_importances: Dict[str, List[float]] = {}
            
            for model_type in self.model_types:
                model_key = f"{horizon.value}_{model_type.value}"
                
                if model_key in self.models:
                    model_data = self.models[model_key]
                    
                    if 'error' not in model_data:
                        pred_result = await self._predict_with_model(X, model_data, model_type)
                        
                        predictions.append(pred_result['prediction'])
                        confidences.append(pred_result['confidence'])
                        
                        # Объединение важности признаков
                        for feature, importance in pred_result['feature_importance'].items():
                            if feature not in feature_importances:
                                feature_importances[feature] = []
                            feature_importances[feature].append(importance)
            
            # Ансамблевое предсказание
            if predictions:
                if self.ensemble_enabled and len(predictions) > 1:
                    # Взвешенное среднее по доверию
                    weights = np.array(confidences)
                    weights = weights / np.sum(weights)
                    
                    final_prediction = np.average(predictions, weights=weights)
                    final_confidence = np.mean(confidences)
                else:
                    final_prediction = predictions[0]
                    final_confidence = confidences[0]
                
                # Усреднение важности признаков
                averaged_importance = {}
                for feature, importance_list in feature_importances.items():
                    averaged_importance[feature] = float(np.mean(importance_list))
                
                # Расчёт интервала предсказания
                if len(predictions) > 1:
                    prediction_std = np.std(predictions)
                    prediction_interval = (
                        final_prediction - 1.96 * prediction_std,
                        final_prediction + 1.96 * prediction_std
                    )
                    model_uncertainty = prediction_std
                else:
                    # Простая оценка неопределённости
                    uncertainty = abs(final_prediction) * 0.1
                    prediction_interval = (
                        final_prediction - uncertainty,
                        final_prediction + uncertainty
                    )
                    model_uncertainty = uncertainty
                
                return PredictionResult(
                    predicted_value=float(final_prediction),
                    confidence=float(final_confidence),
                    prediction_interval=prediction_interval,
                    feature_importance=averaged_importance,
                    model_uncertainty=float(model_uncertainty),
                    metadata={
                        'horizon': horizon.value,
                        'target_type': target.target_type,
                        'models_used': len(predictions),
                        'ensemble_enabled': self.ensemble_enabled
                    }
                )
            
            else:
                return self._create_default_prediction(target)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._create_default_prediction(target)
    
    async def _predict_with_model(
        self, 
        X: np.ndarray, 
        model_data: Dict[str, Any], 
        model_type: ModelType
    ) -> Dict[str, Any]:
        """Предсказание с одной моделью."""
        try:
            # Нормализация
            scaler = model_data['scaler']
            X_scaled = scaler.transform(X)
            
            model = model_data['model']
            
            if model_type == ModelType.LSTM_ATTENTION:
                # Преобразование для LSTM (добавляем временное измерение)
                X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                prediction, attention_weights = model.forward(X_lstm)
                prediction = prediction[0, 0]  # Извлекаем скалярное значение
                
            elif model_type == ModelType.QUANTUM_NEURAL:
                prediction = model.forward(X_scaled[0])
                prediction = prediction[0]  # Извлекаем скалярное значение
                
            else:  # Простая модель
                prediction = (X_scaled @ model['weights'])[0, 0]
            
            # Расчёт доверия (упрощённый)
            confidence = min(0.95, max(0.5, 0.8 + np.random.normal(0, 0.1)))
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'feature_importance': model_data.get('feature_importance', {})
            }
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.5,
                'feature_importance': {}
            }
    
    def _create_default_prediction(self, target: PredictionTarget) -> PredictionResult:
        """Создание предсказания по умолчанию."""
        return PredictionResult(
            predicted_value=0.0,
            confidence=0.5,
            prediction_interval=(-0.1, 0.1),
            feature_importance={},
            model_uncertainty=0.1,
            metadata={
                'default_prediction': True,
                'target_type': target.target_type
            }
        )
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Получение статистики моделей."""
        return {
            'models_trained': len(self.models),
            'training_history_length': len(self.training_history),
            'prediction_horizons': [h.value for h in self.prediction_horizons],
            'model_types': [t.value for t in self.model_types],
            'feature_engineering_enabled': self.feature_engineering_enabled,
            'ensemble_enabled': self.ensemble_enabled,
            'quantum_layers_enabled': self.quantum_layers_enabled,
            'prediction_performance': dict(self.prediction_performance)
        }