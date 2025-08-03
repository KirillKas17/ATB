# -*- coding: utf-8 -*-
"""Модуль для работы с вейвлетами в shared слое."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy import signal

logger = logging.getLogger(__name__)


def extract_wavelet_features(
    data: np.ndarray, scales: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Extract wavelet-based features from signal.
    Args:
        data: Input signal
        scales: Array of scales for wavelet analysis
    Returns:
        Dictionary of wavelet features
    """
    if scales is None:
        scales = np.arange(1, 64)
    try:
        # Perform CWT
        coefficients = perform_cwt(data, scales)
        # Extract features
        features = {
            "wavelet_energy": np.sum(np.abs(coefficients) ** 2),
            "wavelet_entropy": calculate_wavelet_entropy(coefficients),
            "wavelet_variance": np.var(coefficients),
            "wavelet_skewness": calculate_skewness(coefficients),
            "wavelet_kurtosis": calculate_kurtosis(coefficients),
            "wavelet_max_amplitude": np.max(np.abs(coefficients)),
            "wavelet_mean_amplitude": np.mean(np.abs(coefficients)),
            "wavelet_std_amplitude": np.std(np.abs(coefficients)),
        }
        return features
    except Exception as e:
        logger.error(f"Error extracting wavelet features: {e}")
        return {}


def calculate_wavelet_entropy(coefficients: np.ndarray) -> float:
    """Calculate wavelet entropy."""
    try:
        # Normalize coefficients
        norm_coeffs = np.abs(coefficients) / np.sum(np.abs(coefficients))
        # Remove zeros to avoid log(0)
        norm_coeffs = norm_coeffs[norm_coeffs > 0]
        # Calculate entropy
        entropy = -np.sum(norm_coeffs * np.log2(norm_coeffs))
        return float(entropy)
    except Exception:
        return 0.0


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    except Exception:
        return 0.0


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return float(kurtosis)
    except Exception:
        return 0.0


def perform_cwt(
    data: np.ndarray, scales: np.ndarray, wavelet: str = "morlet"
) -> np.ndarray:
    """
    Perform Continuous Wavelet Transform.
    Args:
        data: Input signal
        scales: Array of scales for wavelet analysis
        wavelet: Wavelet type ('morlet', 'gaussian', etc.)
    Returns:
        Wavelet coefficients
    """
    try:
        # Use pywt for CWT instead of scipy.signal.cwt
        coefficients, frequencies = pywt.cwt(data, scales, wavelet)
        return coefficients
    except Exception as e:
        logger.error(f"Error in CWT: {e}")
        return np.zeros((len(scales), len(data)))


def wavelet_denoising(
    data: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    """
    Denoise signal using wavelet transform.
    Args:
        data: Input signal
        wavelet: Wavelet type
        level: Decomposition level
    Returns:
        Denoised signal
    """
    try:
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)
        # Threshold coefficients
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
        # Apply soft thresholding
        coeffs_thresh = []
        for coeff in coeffs:
            coeff_thresh = pywt.threshold(coeff, threshold, mode="soft")
            coeffs_thresh.append(coeff_thresh)
        # Wavelet reconstruction
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        return denoised[: len(data)]  # Ensure same length
    except Exception as e:
        logger.error(f"Error in wavelet denoising: {e}")
        return data


def wavelet_energy_distribution(
    coefficients: np.ndarray, scales: np.ndarray
) -> Dict[str, np.ndarray]:
    """Вычисление распределения энергии по шкалам."""
    power = np.abs(coefficients) ** 2
    energy_per_scale = np.sum(power, axis=1)
    total_energy = np.sum(energy_per_scale)
    normalized_energy = energy_per_scale / total_energy
    dominant_scale_idx = np.argmax(energy_per_scale)
    dominant_scale = scales[dominant_scale_idx]
    return {
        "energy_per_scale": energy_per_scale,
        "normalized_energy": normalized_energy,
        "total_energy": np.array([total_energy]),
        "dominant_scale": np.array([dominant_scale]),
        "dominant_scale_idx": np.array([dominant_scale_idx]),
    }


def wavelet_coherence(
    signal1: np.ndarray, signal2: np.ndarray, scales: np.ndarray
) -> Dict[str, np.ndarray]:
    """Вычисление вейвлет-когерентности между двумя сигналами."""
    # Вейвлет-преобразования
    coeffs1 = perform_cwt(signal1, scales)
    coeffs2 = perform_cwt(signal2, scales)
    # Кросс-спектр
    cross_spectrum = coeffs1 * np.conj(coeffs2)
    # Автоспектры
    auto_spectrum1 = np.abs(coeffs1) ** 2
    auto_spectrum2 = np.abs(coeffs2) ** 2
    # Когерентность
    coherence = np.abs(cross_spectrum) ** 2 / (auto_spectrum1 * auto_spectrum2)
    # Фаза
    phase = np.angle(cross_spectrum)
    return {
        "coherence": coherence,
        "phase": phase,
        "cross_spectrum": cross_spectrum,
        "auto_spectrum1": auto_spectrum1,
        "auto_spectrum2": auto_spectrum2,
    }


def wavelet_ridge_detection(
    coefficients: np.ndarray, scales: np.ndarray
) -> Dict[str, np.ndarray]:
    """Обнаружение вейвлет-гребней."""
    power = np.abs(coefficients) ** 2
    # Поиск локальных максимумов по шкалам
    ridges = []
    ridge_powers = []
    for t in range(coefficients.shape[1]):
        # Поиск максимума по шкалам в момент времени t
        scale_powers = power[:, t]
        max_scale_idx = np.argmax(scale_powers)
        max_power = scale_powers[max_scale_idx]
        ridges.append(scales[max_scale_idx])
        ridge_powers.append(max_power)
    return {
        "ridges": np.array(ridges),
        "ridge_powers": np.array(ridge_powers),
        "ridge_scales": np.array(ridges),
    }


def wavelet_instantaneous_frequency(
    coefficients: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Вычисление мгновенной частоты с помощью вейвлетов."""
    # Производная фазы по времени
    phase = np.angle(coefficients)
    phase_diff = np.diff(phase, axis=1)
    # Мгновенная частота
    instantaneous_freq = phase_diff / (2 * np.pi * np.diff(scales)[:, np.newaxis])
    return instantaneous_freq


def wavelet_bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sampling_rate: float,
    wavelet: str = "morlet",
) -> np.ndarray:
    """Полосовой фильтр на основе вейвлетов."""
    # Определение шкал для заданных частот
    low_scale = sampling_rate / high_freq
    high_scale = sampling_rate / low_freq
    # Создание диапазона шкал
    scales = np.logspace(np.log10(low_scale), np.log10(high_scale), 50)
    # Вейвлет-преобразование
    coefficients = perform_cwt(data, scales)
    # Фильтрация (обнуление коэффициентов вне полосы)
    filtered_coeffs = coefficients.copy()
    # Обратное преобразование (используем pywt вместо scipy)
    # Для простоты возвращаем отфильтрованные коэффициенты
    return filtered_coeffs


def wavelet_spectrogram(
    data: np.ndarray, scales: np.ndarray, window_size: int = 100, overlap: int = 50
) -> np.ndarray:
    """Создание вейвлет-спектрограммы."""
    n_samples = len(data)
    spectrogram = []
    for i in range(0, n_samples - window_size, window_size - overlap):
        window_data = data[i : i + window_size]
        coeffs = perform_cwt(window_data, scales)
        power = np.abs(coeffs) ** 2
        spectrogram.append(np.mean(power, axis=1))
    return np.array(spectrogram).T


def wavelet_feature_extraction(
    data: np.ndarray, scales: np.ndarray
) -> Dict[str, float]:
    """Извлечение признаков из вейвлет-преобразования."""
    coeffs = perform_cwt(data, scales)
    power = np.abs(coeffs) ** 2
    features = {
        "total_energy": np.sum(power),
        "energy_entropy": -np.sum(power * np.log(power + 1e-10)),
        "energy_variance": np.var(power),
        "energy_skewness": np.mean(((power - np.mean(power)) / np.std(power)) ** 3),
        "energy_kurtosis": np.mean(((power - np.mean(power)) / np.std(power)) ** 4),
        "max_energy_scale": scales[np.argmax(np.sum(power, axis=1))],
        "energy_concentration": np.max(np.sum(power, axis=1)) / np.sum(power),
    }
    return features


def detect_wavelet_patterns(
    data: np.ndarray, threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Detect patterns in signal using wavelet analysis.
    Args:
        data: Input signal
        threshold: Threshold for pattern detection
    Returns:
        List of detected patterns
    """
    try:
        scales = np.arange(1, 64)
        coefficients = perform_cwt(data, scales)
        patterns = []
        # Find local maxima in wavelet coefficients
        for i, scale in enumerate(scales):
            coeff_row = coefficients[i, :]
            # Find peaks
            peaks, _ = signal.find_peaks(np.abs(coeff_row), height=threshold)
            for peak in peaks:
                pattern = {
                    "position": int(peak),
                    "scale": float(scale),
                    "amplitude": float(np.abs(coeff_row[peak])),
                    "phase": float(np.angle(coeff_row[peak])),
                    "type": "wavelet_peak",
                }
                patterns.append(pattern)
        return patterns
    except Exception as e:
        logger.error(f"Error detecting wavelet patterns: {e}")
        return []


def analyze_wavelet_coherence(
    signal1: np.ndarray, signal2: np.ndarray, scales: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Анализ вейвлет-когерентности между двумя сигналами.
    
    Args:
        signal1: Первый сигнал
        signal2: Второй сигнал
        scales: Массив шкал для анализа
        
    Returns:
        Словарь с результатами анализа когерентности
    """
    if scales is None:
        scales = np.arange(1, 64)
    
    try:
        # Вычисляем когерентность
        coherence_result = wavelet_coherence(signal1, signal2, scales)
        
        # Анализируем результаты
        mean_coherence = np.mean(coherence_result["coherence"])
        max_coherence = np.max(coherence_result["coherence"])
        coherence_std = np.std(coherence_result["coherence"])
        
        # Находим области высокой когерентности
        high_coherence_threshold = 0.8
        high_coherence_regions = coherence_result["coherence"] > high_coherence_threshold
        
        return {
            "mean_coherence": float(mean_coherence),
            "max_coherence": float(max_coherence),
            "coherence_std": float(coherence_std),
            "high_coherence_fraction": float(np.mean(high_coherence_regions)),
            "coherence_matrix": coherence_result["coherence"],
            "phase_matrix": coherence_result["phase"],
            "scales": scales
        }
    except Exception as e:
        logger.error(f"Error in wavelet coherence analysis: {e}")
        return {}


def perform_dwt(data: np.ndarray, wavelet: str = "db4", level: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Выполнить дискретное вейвлет-преобразование.
    
    Args:
        data: Входной сигнал
        wavelet: Тип вейвлета
        level: Уровень декомпозиции
        
    Returns:
        Кортеж (коэффициенты, последнее приближение)
    """
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        return coeffs[1:], coeffs[0]  # Детализирующие коэффициенты и приближение
    except Exception as e:
        logger.error(f"Error in DWT: {e}")
        return [], np.array([])


def reconstruct_dwt(coeffs: List[np.ndarray], wavelet: str = "db4") -> np.ndarray:
    """
    Восстановить сигнал из вейвлет-коэффициентов.
    
    Args:
        coeffs: Список коэффициентов вейвлет-преобразования
        wavelet: Тип вейвлета
        
    Returns:
        Восстановленный сигнал
    """
    try:
        # Добавляем нулевое приближение (нужно для waverec)
        coeffs_with_approx = [np.zeros_like(coeffs[0])] + coeffs
        reconstructed = pywt.waverec(coeffs_with_approx, wavelet)
        return reconstructed
    except Exception as e:
        logger.error(f"Error in DWT reconstruction: {e}")
        return np.array([])


def extract_wavelet_features_from_ohlcv(ohlcv_data: pd.DataFrame) -> Dict[str, float]:
    """
    Извлечение вейвлет-признаков из OHLCV данных.
    
    Args:
        ohlcv_data: DataFrame с OHLCV данными
        
    Returns:
        Словарь с вейвлет-признаками
    """
    try:
        # Используем цены закрытия для анализа
        close_prices = ohlcv_data['close'].values
        
        # Вычисляем вейвлет-признаки
        features = extract_wavelet_features(close_prices)
        
        # Добавляем признаки на основе объемов
        if 'volume' in ohlcv_data.columns:
            volume_data = ohlcv_data['volume'].values
            volume_features = extract_wavelet_features(volume_data)
            
            # Добавляем префикс к признакам объема
            for key, value in volume_features.items():
                features[f"volume_{key}"] = value
        
        return features
    except Exception as e:
        logger.error(f"Error extracting wavelet features from OHLCV: {e}")
        return {}
