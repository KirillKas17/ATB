from typing import List, Tuple, Union

import numpy as np
import pywt
from scipy.stats import entropy

__all__ = ["perform_dwt", "perform_cwt", "reconstruct_dwt", "extract_wavelet_features"]


def perform_dwt(
    series: np.ndarray, wavelet: str = "db4", level: int = 3
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Выполняет дискретное вейвлет-преобразование (DWT) для временного ряда.

    Args:
        series (np.ndarray): Входной временной ряд.
        wavelet (str): Имя вейвлета (по умолчанию 'db4').
        level (int): Уровень разложения (по умолчанию 3).

    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            - coeffs: Список коэффициентов разложения [cA_n, cD_n, ..., cD1]
            - last_approx: Последний коэффициент аппроксимации (cA_n)
    """
    coeffs = pywt.wavedec(series, wavelet, level=level)
    last_approx = coeffs[0]
    return coeffs, last_approx


def perform_cwt(
    series: np.ndarray,
    wavelet: str = "morl",
    scales: Union[np.ndarray, List[float]] = None,
) -> np.ndarray:
    """
    Выполняет непрерывное вейвлет-преобразование (CWT) для временного ряда.

    Args:
        series (np.ndarray): Входной временной ряд.
        wavelet (str): Имя вейвлета (по умолчанию 'morl').
        scales (np.ndarray or list): Массив масштабов (по умолчанию np.arange(1, 128)).

    Returns:
        np.ndarray: Массив коэффициентов CWT (scales x time)
    """
    if scales is None:
        scales = np.arange(1, 128)
    coefs, _ = pywt.cwt(series, scales, wavelet)
    return coefs


def reconstruct_dwt(coeffs: List[np.ndarray], wavelet: str = "db4") -> np.ndarray:
    """
    Восстанавливает временной ряд из DWT коэффициентов.

    Args:
        coeffs (List[np.ndarray]): Список коэффициентов разложения.
        wavelet (str): Имя вейвлета (по умолчанию 'db4').

    Returns:
        np.ndarray: Восстановленный временной ряд.
    """
    return pywt.waverec(coeffs, wavelet)


def extract_wavelet_features(
    series: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    """
    Извлекает агрегированные признаки из DWT: энтропия, энергия, статистики по каждому уровню.

    Args:
        series (np.ndarray): Входной временной ряд.
        wavelet (str): Имя вейвлета (по умолчанию 'db4').
        level (int): Уровень разложения (по умолчанию 3).

    Returns:
        np.ndarray: Вектор признаков (энергия, энтропия, среднее, std, max, min по каждому уровню)
    """
    coeffs, _ = perform_dwt(series, wavelet, level)
    features = []
    for arr in coeffs:
        arr = np.asarray(arr)
        # Энергия
        energy = np.sum(arr**2)
        # Энтропия
        prob = np.abs(arr) / (np.sum(np.abs(arr)) + 1e-12)
        entr = entropy(prob)
        # Статистики
        mean = np.mean(arr)
        std = np.std(arr)
        maxv = np.max(arr)
        minv = np.min(arr)
        features.extend([energy, entr, mean, std, maxv, minv])
    return np.array(features)
