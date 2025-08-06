# -*- coding: utf-8 -*-
"""
Quantum-Enhanced Pattern Analyzer for Advanced Market Intelligence.
Implements quantum algorithms for pattern recognition, entanglement detection,
and probabilistic market state analysis.
"""

import asyncio
import cmath
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from typing import Complex
except ImportError:
    from numbers import Complex
from enum import Enum
import concurrent.futures

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats, signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

from domain.type_definitions.intelligence_types import (
    AnalysisMetadata,
    CorrelationMatrix,
    PatternComplexity,
    QuantumState,
    EntanglementStrength,
)
from domain.value_objects.timestamp import Timestamp


class QuantumOperator(Enum):
    """Квантовые операторы для анализа рынка."""
    HADAMARD = "hadamard"
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    PHASE = "phase"
    CNOT = "cnot"
    TOFFOLI = "toffoli"


class PatternDimension(Enum):
    """Измерения паттернов для многомерного анализа."""
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    AMPLITUDE = "amplitude"
    PHASE = "phase"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    ENTROPY = "entropy"


@dataclass
class QuantumPatternState:
    """Квантовое состояние рыночного паттерна."""
    amplitude: Complex
    phase: float
    probability: float
    coherence: float
    entanglement_degree: float
    measurement_basis: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'amplitude': {'real': self.amplitude.real, 'imag': self.amplitude.imag},
            'phase': self.phase,
            'probability': self.probability,
            'coherence': self.coherence,
            'entanglement_degree': self.entanglement_degree,
            'measurement_basis': self.measurement_basis,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MultidimensionalPattern:
    """Многомерный рыночный паттерн."""
    pattern_id: str
    dimensions: Dict[PatternDimension, np.ndarray]
    quantum_states: List[QuantumPatternState]
    complexity_score: float
    confidence: float
    discovery_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_pattern_energy(self) -> float:
        """Расчёт энергии паттерна в квантовом представлении."""
        total_energy = 0.0
        for state in self.quantum_states:
            energy = abs(state.amplitude) ** 2 * state.coherence
            total_energy += energy
        return total_energy
    
    def get_dominant_frequency(self) -> Optional[float]:
        """Определение доминирующей частоты паттерна."""
        if PatternDimension.FREQUENCY not in self.dimensions:
            return None
        
        freq_data = self.dimensions[PatternDimension.FREQUENCY]
        if len(freq_data) == 0:
            return None
            
        # FFT анализ для поиска доминирующей частоты
        fft_result = fft(freq_data)
        frequencies = fftfreq(len(freq_data))
        power_spectrum = np.abs(fft_result) ** 2
        
        # Находим частоту с максимальной мощностью
        max_power_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        return float(frequencies[max_power_idx])


class QuantumPatternAnalyzer:
    """
    Квантовый анализатор паттернов с продвинутыми алгоритмами
    машинного обучения и квантовой механики.
    """
    
    def __init__(
        self,
        quantum_precision: int = 1000,
        max_pattern_history: int = 10000,
        enable_parallel_processing: bool = True,
        quantum_coherence_threshold: float = 0.7
    ):
        self.quantum_precision = quantum_precision
        self.max_pattern_history = max_pattern_history
        self.enable_parallel_processing = enable_parallel_processing
        self.quantum_coherence_threshold = quantum_coherence_threshold
        
        # Хранилища паттернов и состояний
        self.pattern_database: Dict[str, MultidimensionalPattern] = {}
        self.quantum_memory: deque = deque(maxlen=max_pattern_history)
        self.entanglement_matrix: Optional[np.ndarray] = None
        
        # Модели машинного обучения
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        
        # Статистика и метрики
        self.statistics = {
            'total_patterns_analyzed': 0,
            'quantum_states_created': 0,
            'entangled_patterns_found': 0,
            'average_pattern_complexity': 0.0,
            'processing_time_history': deque(maxlen=1000)
        }
        
        logger.info(f"QuantumPatternAnalyzer initialized with precision={quantum_precision}")
    
    async def analyze_market_data(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        timestamp_data: np.ndarray,
        additional_features: Optional[Dict[str, np.ndarray]] = None
    ) -> List[MultidimensionalPattern]:
        """
        Комплексный анализ рыночных данных с квантовыми алгоритмами.
        
        Args:
            price_data: Временной ряд цен
            volume_data: Временной ряд объёмов
            timestamp_data: Временные метки
            additional_features: Дополнительные признаки
            
        Returns:
            Список обнаруженных многомерных паттернов
        """
        start_time = time.time()
        
        try:
            # Подготовка многомерных данных
            dimensions = await self._prepare_multidimensional_data(
                price_data, volume_data, additional_features
            )
            
            # Квантовая обработка каждого измерения
            quantum_states = await self._create_quantum_states(dimensions)
            
            # Поиск паттернов с использованием квантовых алгоритмов
            patterns = await self._detect_quantum_patterns(dimensions, quantum_states)
            
            # Анализ запутанности между паттернами
            await self._analyze_pattern_entanglement(patterns)
            
            # Обновление статистики
            processing_time = (time.time() - start_time) * 1000
            self.statistics['processing_time_history'].append(processing_time)
            self.statistics['total_patterns_analyzed'] += len(patterns)
            
            logger.info(f"Analyzed {len(patterns)} patterns in {processing_time:.2f}ms")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in quantum pattern analysis: {e}")
            return []
    
    async def _prepare_multidimensional_data(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        additional_features: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[PatternDimension, np.ndarray]:
        """Подготовка многомерных данных для анализа."""
        dimensions = {}
        
        # Временное измерение
        dimensions[PatternDimension.TEMPORAL] = price_data
        
        # Частотное измерение (FFT анализ)
        fft_result = fft(price_data)
        dimensions[PatternDimension.FREQUENCY] = np.abs(fft_result)
        
        # Амплитудное измерение
        dimensions[PatternDimension.AMPLITUDE] = np.abs(price_data - np.mean(price_data))
        
        # Фазовое измерение
        dimensions[PatternDimension.PHASE] = np.angle(fft_result)
        
        # Объёмное измерение
        dimensions[PatternDimension.VOLUME] = volume_data
        
        # Волатильность
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.zeros_like(price_data)
        volatility[1:] = np.abs(returns)
        dimensions[PatternDimension.VOLATILITY] = volatility
        
        # Корреляционное измерение
        correlation_window = min(50, len(price_data) // 4)
        correlations = np.zeros_like(price_data)
        for i in range(correlation_window, len(price_data)):
            window_prices = price_data[i-correlation_window:i]
            window_volumes = volume_data[i-correlation_window:i]
            if len(window_prices) > 1 and len(window_volumes) > 1:
                correlations[i] = np.corrcoef(window_prices, window_volumes)[0, 1]
        dimensions[PatternDimension.CORRELATION] = correlations
        
        # Энтропийное измерение
        entropy_window = min(20, len(price_data) // 8)
        entropies = np.zeros_like(price_data)
        for i in range(entropy_window, len(price_data)):
            window_data = price_data[i-entropy_window:i]
            # Вычисляем Шэнноновскую энтропию
            hist, _ = np.histogram(window_data, bins=10)
            hist = hist[hist > 0]  # Убираем нулевые значения
            if len(hist) > 0:
                probs = hist / np.sum(hist)
                entropies[i] = -np.sum(probs * np.log2(probs))
        dimensions[PatternDimension.ENTROPY] = entropies
        
        # Добавляем дополнительные признаки
        if additional_features:
            for feature_name, feature_data in additional_features.items():
                # Создаём кастомные измерения для дополнительных признаков
                custom_dimension = f"custom_{feature_name}"
                dimensions[custom_dimension] = feature_data
        
        return dimensions
    
    async def _create_quantum_states(
        self,
        dimensions: Dict[PatternDimension, np.ndarray]
    ) -> List[QuantumPatternState]:
        """Создание квантовых состояний для каждого измерения."""
        quantum_states = []
        
        for dimension, data in dimensions.items():
            if len(data) == 0:
                continue
                
            # Нормализация данных
            normalized_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Создание комплексной амплитуды
            real_part = np.mean(normalized_data)
            imag_part = np.std(normalized_data)
            amplitude = complex(real_part, imag_part)
            
            # Расчёт фазы
            phase = np.angle(amplitude)
            
            # Вероятность наблюдения состояния
            probability = abs(amplitude) ** 2
            probability = min(probability, 1.0)  # Нормализация
            
            # Когерентность на основе автокорреляции
            if len(data) > 1:
                autocorr = np.correlate(normalized_data, normalized_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                coherence = np.mean(autocorr[:min(10, len(autocorr))])
                coherence = abs(coherence)
            else:
                coherence = 0.0
            
            # Степень запутанности (на основе взаимной информации с другими измерениями)
            entanglement_degree = await self._calculate_entanglement_degree(
                data, dimensions, dimension
            )
            
            quantum_state = QuantumPatternState(
                amplitude=amplitude,
                phase=phase,
                probability=probability,
                coherence=coherence,
                entanglement_degree=entanglement_degree,
                measurement_basis=str(dimension.value)
            )
            
            quantum_states.append(quantum_state)
            self.statistics['quantum_states_created'] += 1
        
        return quantum_states
    
    async def _calculate_entanglement_degree(
        self,
        current_data: np.ndarray,
        all_dimensions: Dict[PatternDimension, np.ndarray],
        current_dimension: PatternDimension
    ) -> float:
        """Расчёт степени запутанности с другими измерениями."""
        entanglement_sum = 0.0
        dimension_count = 0
        
        for dim, data in all_dimensions.items():
            if dim == current_dimension or len(data) == 0:
                continue
                
            # Обеспечиваем одинаковую длину массивов
            min_length = min(len(current_data), len(data))
            if min_length < 2:
                continue
                
            current_slice = current_data[:min_length]
            other_slice = data[:min_length]
            
            # Расчёт взаимной информации
            try:
                # Дискретизация для расчёта взаимной информации
                bins = min(10, min_length // 2)
                current_discrete = np.digitize(current_slice, np.linspace(
                    np.min(current_slice), np.max(current_slice), bins
                ))
                other_discrete = np.digitize(other_slice, np.linspace(
                    np.min(other_slice), np.max(other_slice), bins
                ))
                
                # Совместное распределение
                joint_hist = np.histogram2d(current_discrete, other_discrete, bins=bins)[0]
                joint_hist = joint_hist + 1e-10  # Избегаем деления на ноль
                joint_prob = joint_hist / np.sum(joint_hist)
                
                # Маргинальные распределения
                marginal_current = np.sum(joint_prob, axis=1)
                marginal_other = np.sum(joint_prob, axis=0)
                
                # Взаимная информация
                mutual_info = 0.0
                for i in range(len(marginal_current)):
                    for j in range(len(marginal_other)):
                        if joint_prob[i, j] > 0:
                            mutual_info += joint_prob[i, j] * np.log2(
                                joint_prob[i, j] / (marginal_current[i] * marginal_other[j])
                            )
                
                entanglement_sum += abs(mutual_info)
                dimension_count += 1
                
            except Exception as e:
                logger.debug(f"Error calculating mutual information: {e}")
                continue
        
        if dimension_count > 0:
            average_entanglement = entanglement_sum / dimension_count
            # Нормализация к диапазону [0, 1]
            return min(average_entanglement / 2.0, 1.0)
        
        return 0.0
    
    async def _detect_quantum_patterns(
        self,
        dimensions: Dict[PatternDimension, np.ndarray],
        quantum_states: List[QuantumPatternState]
    ) -> List[MultidimensionalPattern]:
        """Обнаружение паттернов с использованием квантовых алгоритмов."""
        patterns = []
        
        # Квантовый поиск аномалий
        anomaly_patterns = await self._quantum_anomaly_detection(dimensions, quantum_states)
        patterns.extend(anomaly_patterns)
        
        # Квантовая кластеризация
        cluster_patterns = await self._quantum_clustering(dimensions, quantum_states)
        patterns.extend(cluster_patterns)
        
        # Квантовый анализ периодичности
        periodicity_patterns = await self._quantum_periodicity_analysis(dimensions, quantum_states)
        patterns.extend(periodicity_patterns)
        
        # Квантовое обнаружение трендов
        trend_patterns = await self._quantum_trend_detection(dimensions, quantum_states)
        patterns.extend(trend_patterns)
        
        return patterns
    
    async def _quantum_anomaly_detection(
        self,
        dimensions: Dict[PatternDimension, np.ndarray],
        quantum_states: List[QuantumPatternState]
    ) -> List[MultidimensionalPattern]:
        """Квантовое обнаружение аномалий."""
        patterns = []
        
        # Поиск квантовых состояний с высокой когерентностью
        high_coherence_states = [
            state for state in quantum_states 
            if state.coherence > self.quantum_coherence_threshold
        ]
        
        if not high_coherence_states:
            return patterns
        
        # Анализ аномалий в каждом измерении
        for dimension, data in dimensions.items():
            if len(data) < 10:  # Минимальный размер для анализа
                continue
                
            # Z-score анализ
            z_scores = np.abs(stats.zscore(data))
            anomaly_indices = np.where(z_scores > 3.0)[0]  # 3-sigma правило
            
            if len(anomaly_indices) > 0:
                # Создаём паттерн аномалии
                pattern_id = f"anomaly_{dimension.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Выбираем соответствующие квантовые состояния
                relevant_states = [
                    state for state in high_coherence_states
                    if state.measurement_basis == dimension.value
                ]
                
                # Расчёт сложности паттерна
                complexity = self._calculate_pattern_complexity(data[anomaly_indices])
                
                # Доверительный интервал на основе квантовых состояний
                confidence = np.mean([state.probability for state in relevant_states]) if relevant_states else 0.5
                
                pattern = MultidimensionalPattern(
                    pattern_id=pattern_id,
                    dimensions={dimension: data[anomaly_indices]},
                    quantum_states=relevant_states,
                    complexity_score=complexity,
                    confidence=confidence,
                    discovery_timestamp=datetime.now(),
                    metadata={
                        'pattern_type': 'quantum_anomaly',
                        'anomaly_indices': anomaly_indices.tolist(),
                        'z_scores': z_scores[anomaly_indices].tolist(),
                        'dimension': dimension.value
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
    
    async def _quantum_clustering(
        self,
        dimensions: Dict[PatternDimension, np.ndarray],
        quantum_states: List[QuantumPatternState]
    ) -> List[MultidimensionalPattern]:
        """Квантовая кластеризация паттернов."""
        patterns = []
        
        # Подготовка данных для кластеризации
        feature_matrix = []
        dimension_names = []
        
        min_length = min(len(data) for data in dimensions.values() if len(data) > 0)
        if min_length < 5:
            return patterns
        
        for dimension, data in dimensions.items():
            if len(data) >= min_length:
                # Обрезаем до минимальной длины и нормализуем
                normalized_data = (data[:min_length] - np.mean(data[:min_length])) / (np.std(data[:min_length]) + 1e-8)
                feature_matrix.append(normalized_data)
                dimension_names.append(dimension)
        
        if len(feature_matrix) < 2:
            return patterns
        
        feature_matrix = np.array(feature_matrix).T  # Транспонируем для правильной формы
        
        try:
            # Применяем PCA для снижения размерности
            if feature_matrix.shape[1] > 10:
                feature_matrix_reduced = self.pca.fit_transform(feature_matrix)
            else:
                feature_matrix_reduced = feature_matrix
            
            # DBSCAN кластеризация
            cluster_labels = self.clustering_model.fit_predict(feature_matrix_reduced)
            unique_labels = set(cluster_labels)
            
            # Создаём паттерны для каждого кластера
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Шум в DBSCAN
                    continue
                    
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) < 3:  # Минимальный размер кластера
                    continue
                
                # Создаём паттерн кластера
                pattern_id = f"cluster_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Извлекаем данные кластера по всем измерениям
                cluster_dimensions = {}
                for i, dimension in enumerate(dimension_names):
                    original_data = dimensions[dimension][:min_length]
                    cluster_dimensions[dimension] = original_data[cluster_indices]
                
                # Выбираем квантовые состояния для этого кластера
                cluster_states = []
                for dimension in dimension_names:
                    relevant_states = [
                        state for state in quantum_states
                        if state.measurement_basis == dimension.value
                    ]
                    cluster_states.extend(relevant_states)
                
                # Расчёт метрик
                complexity = np.mean([
                    self._calculate_pattern_complexity(data) 
                    for data in cluster_dimensions.values()
                ])
                
                confidence = np.mean([state.probability for state in cluster_states]) if cluster_states else 0.5
                
                pattern = MultidimensionalPattern(
                    pattern_id=pattern_id,
                    dimensions=cluster_dimensions,
                    quantum_states=cluster_states,
                    complexity_score=complexity,
                    confidence=confidence,
                    discovery_timestamp=datetime.now(),
                    metadata={
                        'pattern_type': 'quantum_cluster',
                        'cluster_id': int(cluster_id),
                        'cluster_size': len(cluster_indices),
                        'cluster_indices': cluster_indices.tolist()
                    }
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error in quantum clustering: {e}")
        
        return patterns
    
    async def _quantum_periodicity_analysis(
        self,
        dimensions: Dict[PatternDimension, np.ndarray],
        quantum_states: List[QuantumPatternState]
    ) -> List[MultidimensionalPattern]:
        """Квантовый анализ периодичности."""
        patterns = []
        
        for dimension, data in dimensions.items():
            if len(data) < 20:  # Минимум для анализа периодичности
                continue
            
            try:
                # FFT анализ для поиска доминирующих частот
                fft_result = fft(data)
                frequencies = fftfreq(len(data))
                power_spectrum = np.abs(fft_result) ** 2
                
                # Находим пики в спектре мощности
                peaks, properties = signal.find_peaks(
                    power_spectrum[:len(power_spectrum)//2],
                    height=np.max(power_spectrum) * 0.1,  # Минимум 10% от максимума
                    distance=len(power_spectrum) // 20   # Минимальное расстояние между пиками
                )
                
                if len(peaks) > 0:
                    # Для каждого значимого пика создаём паттерн
                    for peak_idx in peaks[:5]:  # Ограничиваем количество паттернов
                        dominant_freq = frequencies[peak_idx]
                        if abs(dominant_freq) < 1e-10:  # Избегаем нулевую частоту
                            continue
                        
                        period = 1.0 / abs(dominant_freq)
                        
                        pattern_id = f"periodic_{dimension.value}_{period:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Извлекаем периодическую компоненту
                        periodic_component = np.real(ifft(
                            np.where(np.abs(frequencies - dominant_freq) < 0.01, fft_result, 0)
                        ))
                        
                        # Соответствующие квантовые состояния
                        relevant_states = [
                            state for state in quantum_states
                            if state.measurement_basis == dimension.value
                        ]
                        
                        complexity = self._calculate_pattern_complexity(periodic_component)
                        confidence = power_spectrum[peak_idx] / np.max(power_spectrum)
                        
                        pattern = MultidimensionalPattern(
                            pattern_id=pattern_id,
                            dimensions={dimension: periodic_component},
                            quantum_states=relevant_states,
                            complexity_score=complexity,
                            confidence=float(confidence),
                            discovery_timestamp=datetime.now(),
                            metadata={
                                'pattern_type': 'quantum_periodicity',
                                'dominant_frequency': float(dominant_freq),
                                'period': float(period),
                                'peak_power': float(power_spectrum[peak_idx]),
                                'dimension': dimension.value
                            }
                        )
                        
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.error(f"Error in periodicity analysis for {dimension}: {e}")
        
        return patterns
    
    async def _quantum_trend_detection(
        self,
        dimensions: Dict[PatternDimension, np.ndarray],
        quantum_states: List[QuantumPatternState]
    ) -> List[MultidimensionalPattern]:
        """Квантовое обнаружение трендов."""
        patterns = []
        
        for dimension, data in dimensions.items():
            if len(data) < 10:
                continue
            
            try:
                # Линейная регрессия для определения тренда
                x = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                
                # Квантовая оценка силы тренда
                trend_strength = abs(r_value)
                trend_significance = 1.0 - p_value if p_value < 0.05 else 0.0
                
                # Квантовая интерпретация тренда
                quantum_trend_amplitude = complex(slope, std_err)
                trend_phase = np.angle(quantum_trend_amplitude)
                trend_probability = trend_strength * trend_significance
                
                if trend_strength > 0.5 and trend_significance > 0.5:  # Значимый тренд
                    pattern_id = f"trend_{dimension.value}_{slope:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Создаём квантовое состояние тренда
                    trend_quantum_state = QuantumPatternState(
                        amplitude=quantum_trend_amplitude,
                        phase=trend_phase,
                        probability=trend_probability,
                        coherence=trend_strength,
                        entanglement_degree=0.0,  # Будет рассчитано позже
                        measurement_basis=f"trend_{dimension.value}"
                    )
                    
                    # Детрендированные данные для анализа остатков
                    trend_line = slope * x + intercept
                    residuals = data - trend_line
                    
                    complexity = self._calculate_pattern_complexity(residuals)
                    
                    pattern = MultidimensionalPattern(
                        pattern_id=pattern_id,
                        dimensions={dimension: trend_line, f"{dimension}_residuals": residuals},
                        quantum_states=[trend_quantum_state],
                        complexity_score=complexity,
                        confidence=trend_probability,
                        discovery_timestamp=datetime.now(),
                        metadata={
                            'pattern_type': 'quantum_trend',
                            'slope': float(slope),
                            'intercept': float(intercept),
                            'r_squared': float(r_value ** 2),
                            'p_value': float(p_value),
                            'trend_direction': 'upward' if slope > 0 else 'downward',
                            'trend_strength': float(trend_strength),
                            'dimension': dimension.value
                        }
                    )
                    
                    patterns.append(pattern)
                    
            except Exception as e:
                logger.error(f"Error in trend detection for {dimension}: {e}")
        
        return patterns
    
    def _calculate_pattern_complexity(self, data: np.ndarray) -> float:
        """Расчёт сложности паттерна на основе энтропии и фрактальной размерности."""
        if len(data) < 2:
            return 0.0
        
        try:
            # Шэнноновская энтропия
            hist, _ = np.histogram(data, bins=min(10, len(data) // 2))
            hist = hist[hist > 0]
            if len(hist) > 0:
                probs = hist / np.sum(hist)
                shannon_entropy = -np.sum(probs * np.log2(probs))
            else:
                shannon_entropy = 0.0
            
            # Приближённая фрактальная размерность (Box-counting)
            data_range = np.max(data) - np.min(data)
            if data_range == 0:
                fractal_dimension = 1.0
            else:
                # Простая оценка фрактальной размерности
                normalized_data = (data - np.min(data)) / data_range
                box_sizes = [0.1, 0.2, 0.5]
                box_counts = []
                
                for box_size in box_sizes:
                    bins = int(1.0 / box_size)
                    hist, _ = np.histogram(normalized_data, bins=bins, range=(0, 1))
                    non_empty_boxes = np.sum(hist > 0)
                    box_counts.append(non_empty_boxes)
                
                if len(set(box_counts)) > 1:  # Есть вариация
                    # Линейная регрессия log(count) vs log(1/size)
                    log_sizes = np.log([1/size for size in box_sizes])
                    log_counts = np.log(box_counts)
                    slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
                    fractal_dimension = abs(slope)
                else:
                    fractal_dimension = 1.0
            
            # Комбинированная оценка сложности
            complexity = (shannon_entropy / np.log2(len(data)) + fractal_dimension / 3.0) / 2.0
            return min(complexity, 1.0)  # Нормализация к [0, 1]
            
        except Exception as e:
            logger.debug(f"Error calculating pattern complexity: {e}")
            return 0.5  # Средняя сложность по умолчанию
    
    async def _analyze_pattern_entanglement(self, patterns: List[MultidimensionalPattern]) -> None:
        """Анализ запутанности между обнаруженными паттернами."""
        if len(patterns) < 2:
            return
        
        # Создаём матрицу запутанности
        n_patterns = len(patterns)
        entanglement_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                pattern_a = patterns[i]
                pattern_b = patterns[j]
                
                # Расчёт запутанности между паттернами
                entanglement_strength = await self._calculate_pattern_entanglement(pattern_a, pattern_b)
                entanglement_matrix[i, j] = entanglement_strength
                entanglement_matrix[j, i] = entanglement_strength
                
                # Обновляем метаданные паттернов
                if entanglement_strength > 0.5:  # Значимая запутанность
                    pattern_a.metadata.setdefault('entangled_patterns', []).append({
                        'pattern_id': pattern_b.pattern_id,
                        'entanglement_strength': entanglement_strength
                    })
                    pattern_b.metadata.setdefault('entangled_patterns', []).append({
                        'pattern_id': pattern_a.pattern_id,
                        'entanglement_strength': entanglement_strength
                    })
                    
                    self.statistics['entangled_patterns_found'] += 1
        
        self.entanglement_matrix = entanglement_matrix
    
    async def _calculate_pattern_entanglement(
        self, 
        pattern_a: MultidimensionalPattern, 
        pattern_b: MultidimensionalPattern
    ) -> float:
        """Расчёт квантовой запутанности между двумя паттернами."""
        try:
            # Анализ перекрытия в квантовых состояниях
            state_overlap = 0.0
            state_count = 0
            
            for state_a in pattern_a.quantum_states:
                for state_b in pattern_b.quantum_states:
                    # Скалярное произведение квантовых состояний
                    overlap = abs(state_a.amplitude * np.conj(state_b.amplitude))
                    
                    # Учитываем разность фаз
                    phase_factor = np.cos(state_a.phase - state_b.phase)
                    
                    # Когерентность влияет на запутанность
                    coherence_factor = (state_a.coherence + state_b.coherence) / 2.0
                    
                    state_overlap += overlap * phase_factor * coherence_factor
                    state_count += 1
            
            if state_count > 0:
                state_entanglement = state_overlap / state_count
            else:
                state_entanglement = 0.0
            
            # Анализ корреляций в данных
            data_entanglement = 0.0
            dimension_count = 0
            
            # Находим общие измерения
            common_dimensions = set(pattern_a.dimensions.keys()) & set(pattern_b.dimensions.keys())
            
            for dimension in common_dimensions:
                data_a = pattern_a.dimensions[dimension]
                data_b = pattern_b.dimensions[dimension]
                
                # Обеспечиваем одинаковую длину
                min_length = min(len(data_a), len(data_b))
                if min_length < 2:
                    continue
                
                data_a_slice = data_a[:min_length]
                data_b_slice = data_b[:min_length]
                
                # Расчёт корреляции
                correlation = np.corrcoef(data_a_slice, data_b_slice)[0, 1]
                if not np.isnan(correlation):
                    data_entanglement += abs(correlation)
                    dimension_count += 1
            
            if dimension_count > 0:
                data_entanglement /= dimension_count
            
            # Комбинированная оценка запутанности
            total_entanglement = (state_entanglement + data_entanglement) / 2.0
            return min(total_entanglement, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating pattern entanglement: {e}")
            return 0.0
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Получение статистики анализа."""
        avg_processing_time = (
            np.mean(list(self.statistics['processing_time_history']))
            if self.statistics['processing_time_history']
            else 0.0
        )
        
        return {
            'total_patterns_analyzed': self.statistics['total_patterns_analyzed'],
            'quantum_states_created': self.statistics['quantum_states_created'],
            'entangled_patterns_found': self.statistics['entangled_patterns_found'],
            'average_processing_time_ms': float(avg_processing_time),
            'pattern_database_size': len(self.pattern_database),
            'quantum_memory_size': len(self.quantum_memory),
            'quantum_precision': self.quantum_precision,
            'coherence_threshold': self.quantum_coherence_threshold
        }
    
    async def save_pattern_to_database(self, pattern: MultidimensionalPattern) -> bool:
        """Сохранение паттерна в базу данных."""
        try:
            self.pattern_database[pattern.pattern_id] = pattern
            self.quantum_memory.append({
                'pattern_id': pattern.pattern_id,
                'timestamp': pattern.discovery_timestamp,
                'complexity': pattern.complexity_score,
                'confidence': pattern.confidence
            })
            return True
        except Exception as e:
            logger.error(f"Error saving pattern to database: {e}")
            return False
    
    async def search_similar_patterns(
        self, 
        target_pattern: MultidimensionalPattern,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[MultidimensionalPattern, float]]:
        """Поиск похожих паттернов в базе данных."""
        similar_patterns = []
        
        for stored_pattern in self.pattern_database.values():
            if stored_pattern.pattern_id == target_pattern.pattern_id:
                continue
            
            # Расчёт сходства
            similarity = await self._calculate_pattern_similarity(target_pattern, stored_pattern)
            
            if similarity >= similarity_threshold:
                similar_patterns.append((stored_pattern, similarity))
        
        # Сортировка по убыванию сходства
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    async def _calculate_pattern_similarity(
        self,
        pattern_a: MultidimensionalPattern,
        pattern_b: MultidimensionalPattern
    ) -> float:
        """Расчёт сходства между паттернами."""
        # Используем расчёт запутанности как метрику сходства
        return await self._calculate_pattern_entanglement(pattern_a, pattern_b)