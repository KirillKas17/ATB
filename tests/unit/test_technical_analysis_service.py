"""
Тесты для technical analysis сервиса.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.external_services.technical_analysis_service import TechnicalAnalysisServiceAdapter


class TestTechnicalAnalysisService:
    """Тесты для TechnicalAnalysisServiceAdapter."""

    @pytest.fixture
    def technical_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра сервиса."""
        return TechnicalAnalysisServiceAdapter()

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Тестовые рыночные данные."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "close": np.random.normal(100, 5, 100),
                "high": np.random.normal(105, 5, 100),
                "low": np.random.normal(95, 5, 100),
                "volume": np.random.normal(1000000, 200000, 100),
            },
            index=dates,
        )

    def test_calculate_rsi(self, technical_service, sample_market_data) -> None:
        """Тест расчета RSI."""
        rsi = technical_service.calculate_rsi(sample_market_data["close"])
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_market_data)
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_calculate_macd(self, technical_service, sample_market_data) -> None:
        """Тест расчета MACD."""
        macd_result = technical_service.calculate_macd(sample_market_data["close"])
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result
        assert "signal" in macd_result
        assert "histogram" in macd_result
        assert isinstance(macd_result["macd"], pd.Series)
        assert isinstance(macd_result["signal"], pd.Series)
        assert isinstance(macd_result["histogram"], pd.Series)

    def test_calculate_bollinger_bands(self, technical_service, sample_market_data) -> None:
        """Тест расчета полос Боллинджера."""
        bb_result = technical_service.calculate_bollinger_bands(sample_market_data["close"])
        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result
        assert isinstance(bb_result["upper"], pd.Series)
        assert isinstance(bb_result["middle"], pd.Series)
        assert isinstance(bb_result["lower"], pd.Series)
        # Проверяем логику полос Боллинджера
        assert (bb_result["upper"] >= bb_result["middle"]).all()
        assert (bb_result["middle"] >= bb_result["lower"]).all()

    def test_calculate_atr(self, technical_service, sample_market_data) -> None:
        """Тест расчета ATR."""
        atr = technical_service.calculate_atr(
            sample_market_data["high"], sample_market_data["low"], sample_market_data["close"]
        )
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_market_data)
        assert (atr >= 0).all()

    def test_calculate_ema(self, technical_service, sample_market_data) -> None:
        """Тест расчета EMA."""
        ema = technical_service.calculate_ema(sample_market_data["close"], period=20)
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_market_data)

    def test_calculate_sma(self, technical_service, sample_market_data) -> None:
        """Тест расчета SMA."""
        sma = technical_service.calculate_sma(sample_market_data["close"], period=20)
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_market_data)

    def test_calculate_stochastic(self, technical_service, sample_market_data) -> None:
        """Тест расчета стохастического осциллятора."""
        stoch_result = technical_service.calculate_stochastic(
            sample_market_data["high"], sample_market_data["low"], sample_market_data["close"]
        )
        assert isinstance(stoch_result, dict)
        assert "k" in stoch_result
        assert "d" in stoch_result
        assert isinstance(stoch_result["k"], pd.Series)
        assert isinstance(stoch_result["d"], pd.Series)
        assert (stoch_result["k"] >= 0).all()
        assert (stoch_result["k"] <= 100).all()

    def test_calculate_williams_r(self, technical_service, sample_market_data) -> None:
        """Тест расчета Williams %R."""
        williams_r = technical_service.calculate_williams_r(
            sample_market_data["high"], sample_market_data["low"], sample_market_data["close"]
        )
        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(sample_market_data)
        assert (williams_r >= -100).all()
        assert (williams_r <= 0).all()

    def test_calculate_volume_profile(self, technical_service, sample_market_data) -> None:
        """Тест расчета профиля объема."""
        volume_profile = technical_service.calculate_volume_profile(sample_market_data)
        assert isinstance(volume_profile, dict)
        assert "price_levels" in volume_profile
        assert "volumes" in volume_profile
        assert "poc" in volume_profile  # Point of Control

    def test_calculate_market_structure(self, technical_service, sample_market_data) -> None:
        """Тест расчета структуры рынка."""
        market_structure = technical_service.calculate_market_structure(sample_market_data)
        assert isinstance(market_structure, dict)
        assert "support_levels" in market_structure
        assert "resistance_levels" in market_structure
        assert "trend" in market_structure

    def test_generate_signals(self, technical_service, sample_market_data) -> None:
        """Тест генерации торговых сигналов."""
        signals = technical_service.generate_signals(sample_market_data)
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, dict)
            assert "name" in signal
            assert "value" in signal
            assert "signal" in signal
            assert "strength" in signal
            assert signal["signal"] in ["BUY", "SELL", "HOLD"]

    def test_calculate_indicators(self, technical_service, sample_market_data) -> None:
        """Тест расчета всех индикаторов."""
        indicators = technical_service.calculate_indicators(sample_market_data)
        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) == len(sample_market_data)
        # Проверяем наличие основных индикаторов
        expected_columns = ["rsi", "macd", "bb_upper", "bb_lower", "atr"]
        for col in expected_columns:
            if col in indicators.columns:
                assert not indicators[col].isna().all()

    def test_calculate_rsi_custom_period(self, technical_service, sample_market_data) -> None:
        """Тест расчета RSI с кастомным периодом."""
        rsi_10 = technical_service.calculate_rsi(sample_market_data["close"], period=10)
        rsi_20 = technical_service.calculate_rsi(sample_market_data["close"], period=20)
        assert len(rsi_10) == len(rsi_20)
        # RSI с разными периодами должны давать разные результаты
        assert not (rsi_10 == rsi_20).all()

    def test_calculate_macd_custom_periods(self, technical_service, sample_market_data) -> None:
        """Тест расчета MACD с кастомными периодами."""
        macd_result = technical_service.calculate_macd(
            sample_market_data["close"], fast_period=8, slow_period=21, signal_period=5
        )
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result
        assert "signal" in macd_result
        assert "histogram" in macd_result

    def test_calculate_bollinger_bands_custom_params(self, technical_service, sample_market_data) -> None:
        """Тест расчета полос Боллинджера с кастомными параметрами."""
        bb_result = technical_service.calculate_bollinger_bands(sample_market_data["close"], period=10, std_dev=1.5)
        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result

    def test_empty_data_handling(self, technical_service) -> None:
        """Тест обработки пустых данных."""
        empty_data = pd.DataFrame()
        # RSI с пустыми данными
        rsi = technical_service.calculate_rsi(pd.Series(dtype=float))
        assert isinstance(rsi, pd.Series)
        # MACD с пустыми данными
        macd_result = technical_service.calculate_macd(pd.Series(dtype=float))
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result

    def test_single_value_data_handling(self, technical_service) -> None:
        """Тест обработки данных с одним значением."""
        single_data = pd.Series([100])
        # RSI с одним значением
        rsi = technical_service.calculate_rsi(single_data)
        assert isinstance(rsi, pd.Series)
        # MACD с одним значением
        macd_result = technical_service.calculate_macd(single_data)
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result

    def test_invalid_periods(self, technical_service, sample_market_data) -> None:
        """Тест обработки неверных периодов."""
        # RSI с неверным периодом
        with pytest.raises(ValueError):
            technical_service.calculate_rsi(sample_market_data["close"], period=0)
        # MACD с неверными периодами
        with pytest.raises(ValueError):
            technical_service.calculate_macd(
                sample_market_data["close"], fast_period=0, slow_period=26, signal_period=9
            )
