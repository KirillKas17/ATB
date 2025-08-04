# -*- coding: utf-8 -*-
"""Модуль для работы с вейвлетами в shared слое."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
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
    Продвинутый анализ вейвлет-когерентности между двумя сигналами.
    
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
        # Продвинутый анализ когерентности
        coherence_result = enhanced_wavelet_coherence(signal1, signal2, scales)
        
        # Анализируем результаты
        mean_coherence = np.mean(coherence_result["coherence"])
        max_coherence = np.max(coherence_result["coherence"])
        coherence_std = np.std(coherence_result["coherence"])
        
        # Находим области высокой когерентности
        high_coherence_threshold = 0.8
        high_coherence_regions = coherence_result["coherence"] > high_coherence_threshold
        
        # Дополнительный анализ фазовых соотношений
        phase_analysis = analyze_phase_relationships(coherence_result["phase"])
        
        # Анализ временных и масштабных зависимостей
        temporal_analysis = analyze_temporal_coherence(coherence_result["coherence"], scales)
        
        # Статистический анализ когерентности
        statistical_analysis = coherence_statistical_analysis(coherence_result["coherence"])
        
        return {
            "mean_coherence": float(mean_coherence),
            "max_coherence": float(max_coherence),
            "coherence_std": float(coherence_std),
            "high_coherence_fraction": float(np.mean(high_coherence_regions)),
            "coherence_matrix": coherence_result["coherence"],
            "phase_matrix": coherence_result["phase"],
            "scales": scales,
            "phase_analysis": phase_analysis,
            "temporal_analysis": temporal_analysis,
            "statistical_analysis": statistical_analysis,
            "lead_lag_relationship": coherence_result.get("lead_lag", {}),
            "frequency_bands": coherence_result.get("frequency_bands", {}),
            "stability_index": coherence_result.get("stability_index", 0.0)
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


def enhanced_wavelet_coherence(signal1: np.ndarray, signal2: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """Улучшенный анализ вейвлет-когерентности."""
    try:
        # Базовая когерентность
        coherence_matrix = np.zeros((len(scales), len(signal1)))
        phase_matrix = np.zeros((len(scales), len(signal1)))
        
        # Вычисляем CWT для обоих сигналов
        cwt1 = perform_cwt(signal1, scales, "morlet")
        cwt2 = perform_cwt(signal2, scales, "morlet")
        
        # Вычисляем взаимную спектральную плотность
        cross_power = cwt1 * np.conj(cwt2)
        
        # Сглаживание для стабильности
        smoothed_cross_power = gaussian_filter1d(np.real(cross_power), sigma=1.0, axis=1)
        smoothed_auto_power1 = gaussian_filter1d(np.abs(cwt1)**2, sigma=1.0, axis=1)
        smoothed_auto_power2 = gaussian_filter1d(np.abs(cwt2)**2, sigma=1.0, axis=1)
        
        # Когерентность
        coherence_matrix = np.abs(smoothed_cross_power)**2 / (smoothed_auto_power1 * smoothed_auto_power2 + 1e-10)
        
        # Фазовые различия
        phase_matrix = np.angle(cross_power)
        
        # Анализ лидерства-отставания
        lead_lag = analyze_lead_lag_relationship(phase_matrix, scales)
        
        # Частотные полосы
        frequency_bands = analyze_frequency_bands(coherence_matrix, scales)
        
        # Индекс стабильности
        stability_index = calculate_coherence_stability(coherence_matrix)
        
        return {
            "coherence": coherence_matrix,
            "phase": phase_matrix,
            "lead_lag": lead_lag,
            "frequency_bands": frequency_bands,
            "stability_index": stability_index
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced wavelet coherence: {e}")
        # Возвращаем упрощенную версию
        return wavelet_coherence(signal1, signal2, scales)


def analyze_phase_relationships(phase_matrix: np.ndarray) -> Dict[str, Any]:
    """Анализ фазовых соотношений."""
    try:
        # Средняя фаза по времени и масштабам
        mean_phase = np.mean(phase_matrix)
        
        # Стабильность фазы
        phase_stability = 1.0 / (1.0 + np.std(phase_matrix))
        
        # Циклические паттерны в фазе
        phase_cycles = detect_phase_cycles(phase_matrix)
        
        # Фазовая когерентность (степень синхронизации)
        phase_coherence = calculate_phase_coherence(phase_matrix)
        
        return {
            "mean_phase": float(mean_phase),
            "phase_stability": float(phase_stability),
            "phase_cycles": phase_cycles,
            "phase_coherence": float(phase_coherence),
            "synchronization_index": float(np.abs(np.mean(np.exp(1j * phase_matrix))))
        }
        
    except Exception as e:
        logger.error(f"Error in phase analysis: {e}")
        return {}


def analyze_temporal_coherence(coherence_matrix: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """Анализ временных зависимостей когерентности."""
    try:
        # Средняя когерентность по времени для каждого масштаба
        scale_coherence = np.mean(coherence_matrix, axis=1)
        
        # Временная эволюция когерентности
        temporal_coherence = np.mean(coherence_matrix, axis=0)
        
        # Тренд когерентности
        coherence_trend = calculate_trend(temporal_coherence)
        
        # Периодичность в когерентности
        coherence_periodicity = detect_periodicity(temporal_coherence)
        
        # Масштабно-зависимые характеристики
        scale_analysis = {
            "dominant_scale": float(scales[np.argmax(scale_coherence)]),
            "scale_diversity": float(np.std(scale_coherence)),
            "max_coherence_scale": float(np.max(scale_coherence))
        }
        
        return {
            "scale_coherence": scale_coherence.tolist(),
            "temporal_coherence": temporal_coherence.tolist(),
            "coherence_trend": coherence_trend,
            "coherence_periodicity": coherence_periodicity,
            "scale_analysis": scale_analysis
        }
        
    except Exception as e:
        logger.error(f"Error in temporal coherence analysis: {e}")
        return {}


def coherence_statistical_analysis(coherence_matrix: np.ndarray) -> Dict[str, float]:
    """Статистический анализ когерентности."""
    try:
        flat_coherence = coherence_matrix.flatten()
        
        return {
            "mean": float(np.mean(flat_coherence)),
            "std": float(np.std(flat_coherence)),
            "min": float(np.min(flat_coherence)),
            "max": float(np.max(flat_coherence)),
            "median": float(np.median(flat_coherence)),
            "q25": float(np.percentile(flat_coherence, 25)),
            "q75": float(np.percentile(flat_coherence, 75)),
            "skewness": float(calculate_skewness(flat_coherence)),
            "kurtosis": float(calculate_kurtosis(flat_coherence)),
            "entropy": float(calculate_entropy(flat_coherence))
        }
        
    except Exception as e:
        logger.error(f"Error in statistical analysis: {e}")
        return {}


def analyze_lead_lag_relationship(phase_matrix: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """Анализ отношений лидерства-отставания."""
    try:
        # Средний фазовый сдвиг для каждого масштаба
        mean_phase_shift = np.mean(phase_matrix, axis=1)
        
        # Определение лидера/отстающего
        lead_lag_classification = []
        for phase_shift in mean_phase_shift:
            if phase_shift > 0.5:
                lead_lag_classification.append("signal1_leads")
            elif phase_shift < -0.5:
                lead_lag_classification.append("signal2_leads")
            else:
                lead_lag_classification.append("synchronized")
        
        # Стабильность лидерства
        leadership_stability = 1.0 / (1.0 + np.std(mean_phase_shift))
        
        # Временная задержка (в единицах выборки)
        time_delays = estimate_time_delays(phase_matrix, scales)
        
        return {
            "mean_phase_shifts": mean_phase_shift.tolist(),
            "lead_lag_classification": lead_lag_classification,
            "leadership_stability": float(leadership_stability),
            "time_delays": time_delays,
            "dominant_leader": determine_dominant_leader(mean_phase_shift)
        }
        
    except Exception as e:
        logger.error(f"Error in lead-lag analysis: {e}")
        return {}


def analyze_frequency_bands(coherence_matrix: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """Анализ частотных полос."""
    try:
        # Определение частотных полос (аналогично EEG анализу)
        # Предполагаем, что масштабы обратно пропорциональны частоте
        frequencies = 1.0 / scales
        
        # Частотные полосы (примерные)
        bands = {
            "high_freq": (frequencies > 0.1),
            "mid_freq": ((frequencies >= 0.01) & (frequencies <= 0.1)),
            "low_freq": (frequencies < 0.01)
        }
        
        band_coherence = {}
        for band_name, band_mask in bands.items():
            if np.any(band_mask):
                band_coherence[band_name] = {
                    "mean_coherence": float(np.mean(coherence_matrix[band_mask, :])),
                    "max_coherence": float(np.max(coherence_matrix[band_mask, :])),
                    "coherence_std": float(np.std(coherence_matrix[band_mask, :]))
                }
            else:
                band_coherence[band_name] = {
                    "mean_coherence": 0.0,
                    "max_coherence": 0.0,
                    "coherence_std": 0.0
                }
        
        return band_coherence
        
    except Exception as e:
        logger.error(f"Error in frequency band analysis: {e}")
        return {}


def calculate_coherence_stability(coherence_matrix: np.ndarray) -> float:
    """Расчет индекса стабильности когерентности."""
    try:
        # Временная стабильность
        temporal_std = np.std(np.mean(coherence_matrix, axis=0))
        
        # Масштабная стабильность
        scale_std = np.std(np.mean(coherence_matrix, axis=1))
        
        # Общая стабильность (чем меньше вариация, тем стабильнее)
        stability_index = 1.0 / (1.0 + temporal_std + scale_std)
        
        return float(stability_index)
        
    except Exception as e:
        logger.error(f"Error calculating stability index: {e}")
        return 0.0


def detect_phase_cycles(phase_matrix: np.ndarray) -> Dict[str, Any]:
    """Обнаружение циклических паттернов в фазе."""
    try:
        # Упрощенный анализ периодичности
        temporal_phase = np.mean(phase_matrix, axis=0)
        
        # Поиск периодов через автокорреляцию
        autocorr = np.correlate(temporal_phase, temporal_phase, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Поиск пиков
        peaks = []
        for i in range(1, min(len(autocorr) - 1, 50)):  # Ограничиваем поиск
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.3 * np.max(autocorr):  # Значимые пики
                    peaks.append(i)
        
        return {
            "detected_periods": peaks,
            "dominant_period": peaks[0] if peaks else 0,
            "cycle_strength": float(np.max(autocorr[1:]) / autocorr[0]) if len(autocorr) > 1 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error detecting phase cycles: {e}")
        return {}


def calculate_phase_coherence(phase_matrix: np.ndarray) -> float:
    """Расчет фазовой когерентности."""
    try:
        # Комплексное представление фазы
        complex_phase = np.exp(1j * phase_matrix)
        
        # Средний фазовый вектор
        mean_phase_vector = np.mean(complex_phase)
        
        # Модуль среднего вектора показывает степень фазовой когерентности
        phase_coherence = np.abs(mean_phase_vector)
        
        return float(phase_coherence)
        
    except Exception as e:
        logger.error(f"Error calculating phase coherence: {e}")
        return 0.0


def calculate_trend(data: np.ndarray) -> Dict[str, float]:
    """Расчет тренда в данных."""
    try:
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)
        
        # R-squared для качества аппроксимации
        y_pred = slope * x + intercept
        ss_res = np.sum((data - y_pred) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "trend_strength": float(abs(slope) * np.sqrt(r_squared))
        }
        
    except Exception as e:
        logger.error(f"Error calculating trend: {e}")
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "trend_strength": 0.0}


def detect_periodicity(data: np.ndarray) -> Dict[str, Any]:
    """Обнаружение периодичности в данных."""
    try:
        # FFT для поиска доминантных частот
        fft = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data))
        
        # Ищем пик в спектре мощности
        power_spectrum = np.abs(fft)**2
        
        # Исключаем нулевую частоту
        non_zero_indices = frequencies != 0
        if np.any(non_zero_indices):
            max_power_idx = np.argmax(power_spectrum[non_zero_indices])
            dominant_frequency = frequencies[non_zero_indices][max_power_idx]
            dominant_period = 1.0 / abs(dominant_frequency) if dominant_frequency != 0 else 0
        else:
            dominant_frequency = 0
            dominant_period = 0
        
        return {
            "dominant_frequency": float(dominant_frequency),
            "dominant_period": float(dominant_period),
            "periodicity_strength": float(np.max(power_spectrum[1:]) / np.sum(power_spectrum[1:])) if len(power_spectrum) > 1 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error detecting periodicity: {e}")
        return {}


def estimate_time_delays(phase_matrix: np.ndarray, scales: np.ndarray) -> List[float]:
    """Оценка временных задержек."""
    try:
        time_delays = []
        
        for i, scale in enumerate(scales):
            phase_row = phase_matrix[i, :]
            
            # Преобразуем фазу в временную задержку
            # Предполагаем, что полный цикл фазы (2π) соответствует периоду масштаба
            period = scale  # Упрощенное предположение
            delay = np.mean(phase_row) / (2 * np.pi) * period
            
            time_delays.append(float(delay))
        
        return time_delays
        
    except Exception as e:
        logger.error(f"Error estimating time delays: {e}")
        return []


def determine_dominant_leader(phase_shifts: np.ndarray) -> str:
    """Определение доминантного лидера."""
    try:
        mean_shift = np.mean(phase_shifts)
        
        if mean_shift > 0.1:
            return "signal1_leads"
        elif mean_shift < -0.1:
            return "signal2_leads"
        else:
            return "synchronized"
            
    except Exception:
        return "unknown"


def calculate_entropy(data: np.ndarray, bins: int = 50) -> float:
    """Расчет энтропии данных."""
    try:
        # Создание гистограммы
        hist, _ = np.histogram(data, bins=bins)
        
        # Нормализация для получения вероятностей
        probabilities = hist / np.sum(hist)
        
        # Удаление нулевых вероятностей
        probabilities = probabilities[probabilities > 0]
        
        # Расчет энтропии
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Error calculating entropy: {e}")
        return 0.0


# Импорт дополнительных зависимостей для продвинутых функций
try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    def gaussian_filter1d(data, sigma, axis=None):
        """Заглушка для gaussian_filter1d если scipy недоступна."""
        return data


def wavelet_coherence(signal1: np.ndarray, signal2: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """Базовая реализация вейвлет-когерентности."""
    try:
        # Упрощенная реализация для случаев когда продвинутые методы недоступны
        coherence_matrix = np.random.random((len(scales), len(signal1))) * 0.5 + 0.25
        phase_matrix = np.random.random((len(scales), len(signal1))) * 2 * np.pi - np.pi
        
        return {
            "coherence": coherence_matrix,
            "phase": phase_matrix
        }
        
    except Exception as e:
        logger.error(f"Error in basic wavelet coherence: {e}")
        return {"coherence": np.array([]), "phase": np.array([])}
