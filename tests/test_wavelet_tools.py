from shared.numpy_utils import np
from shared.wavelet_tools import (extract_wavelet_features, perform_cwt,
                                  perform_dwt, reconstruct_dwt)
def test_perform_dwt_and_reconstruct() -> None:
    series = np.sin(np.linspace(0, 8 * np.pi, 256)) + np.random.normal(0, 0.1, 256)
    coeffs, last_approx = perform_dwt(series, wavelet="db4", level=3)
    assert isinstance(coeffs, list)
    assert isinstance(last_approx, np.ndarray)
    # Восстановление
    reconstructed = reconstruct_dwt(coeffs, wavelet="db4")
    # Длина может отличаться из-за паддинга, сравниваем только общую форму
    min_len = min(len(series), len(reconstructed))
    assert np.allclose(series[:min_len], reconstructed[:min_len], atol=0.2)


def test_perform_cwt() -> None:
    series = np.cos(np.linspace(0, 4 * np.pi, 128))
    coefs = perform_cwt(series, wavelet="morl", scales=np.arange(1, 32))
    assert isinstance(coefs, np.ndarray)
    assert coefs.shape[1] == len(series)


def test_extract_wavelet_features() -> None:
    series = np.random.normal(0, 1, 256)
    features = extract_wavelet_features(series, wavelet="db4", level=3)
    # 4 уровней: cA3, cD3, cD2, cD1, по 6 признаков на уровень
    assert features.shape[0] == 4 * 6
    assert np.all(np.isfinite(features))


def test_wavelet_stability_timeframes() -> None:
    # 1h, 4h, 15m (разная длина)
    for length in [128, 256, 512]:
        series = np.random.normal(0, 1, length)
        coeffs, _ = perform_dwt(series, level=3)
        features = extract_wavelet_features(series, level=3)
        assert isinstance(coeffs, list)
        assert features.shape[0] == 4 * 6


def test_docstrings() -> None:
    assert perform_dwt.__doc__
    assert perform_cwt.__doc__
    assert reconstruct_dwt.__doc__
    assert extract_wavelet_features.__doc__
