"""
Unit тесты для reversal_predictor.py.

Покрывает:
- ReversalPredictor - прогнозатор разворотов цены
- Анализ дивергенций RSI и MACD
- Анализ свечных паттернов
- Анализ импульса и среднего возврата
- Расчет уровней разворота и уверенности
- Вспомогательные методы
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from domain.prediction.reversal_predictor import ReversalPredictor
from domain.prediction.reversal_signal import (
    ReversalSignal,
    DivergenceSignal,
    CandlestickPattern,
    MomentumAnalysis,
    MeanReversionBand,
)
from domain.type_definitions.prediction_types import (
    PredictionConfig,
    ReversalDirection,
    DivergenceType,
    ConfidenceScore,
    SignalStrengthScore,
    OHLCVData,
    OrderBookData,
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume_profile import VolumeProfile


class TestReversalPredictor:
    """Тесты для ReversalPredictor."""

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """Тестовые OHLCV данные."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
        np.random.seed(42)

        # Создаем реалистичные данные с трендом
        base_price = 50000.0
        trend = np.linspace(0, 1000, 100)
        noise = np.random.normal(0, 100, 100)
        prices = base_price + trend + noise

        return pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 50, 100),
                "low": prices - np.random.uniform(0, 50, 100),
                "close": prices + np.random.normal(0, 20, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_order_book(self) -> OrderBookData:
        """Тестовые данные ордербука."""
        return {
            "bids": [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 1.5]],
            "asks": [[50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.1]],
            "timestamp": datetime.now(),
        }

    @pytest.fixture
    def predictor(self) -> ReversalPredictor:
        """Экземпляр ReversalPredictor."""
        config: PredictionConfig = {
            "lookback_period": 50,
            "min_confidence": 0.3,
            "min_signal_strength": 0.4,
            "prediction_horizon": timedelta(hours=4),
        }
        return ReversalPredictor(config)

    def test_initialization(self, predictor: ReversalPredictor) -> None:
        """Тест инициализации."""
        assert predictor.config is not None
        assert predictor.config["lookback_period"] == 50
        assert predictor.config["min_confidence"] == 0.3
        assert predictor.config["min_signal_strength"] == 0.4
        assert predictor.pattern_extractor is not None

    def test_initialization_default_config(self: "TestReversalPredictor") -> None:
        """Тест инициализации с дефолтной конфигурацией."""
        predictor = ReversalPredictor()
        assert predictor.config == {}
        assert predictor.pattern_extractor is not None

    @patch("domain.prediction.reversal_predictor.PricePatternExtractor")
    def test_predict_reversal_success(
        self,
        mock_extractor: Mock,
        predictor: ReversalPredictor,
        sample_ohlcv_data: pd.DataFrame,
        sample_order_book: OrderBookData,
    ) -> None:
        """Тест успешного прогнозирования разворота."""
        # Настройка моков
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance

        # Мокаем методы pattern_extractor
        mock_extractor_instance.extract_pivot_points.return_value = ([10, 30, 50], [20, 40, 60])
        mock_extractor_instance.extract_volume_profile.return_value = Mock(spec=VolumeProfile)
        mock_extractor_instance.extract_liquidity_clusters.return_value = []

        # Мокаем внутренние методы анализа
        with patch.object(predictor, "_analyze_divergences") as mock_divergences, patch.object(
            predictor, "_analyze_candlestick_patterns"
        ) as mock_patterns, patch.object(predictor, "_analyze_momentum") as mock_momentum, patch.object(
            predictor, "_analyze_mean_reversion"
        ) as mock_mean_rev, patch.object(
            predictor, "_determine_reversal_direction"
        ) as mock_direction, patch.object(
            predictor, "_calculate_reversal_level"
        ) as mock_level, patch.object(
            predictor, "_calculate_confidence"
        ) as mock_confidence, patch.object(
            predictor, "_calculate_signal_strength"
        ) as mock_strength:

            mock_divergences.return_value = []
            mock_patterns.return_value = []
            mock_momentum.return_value = Mock(spec=MomentumAnalysis)
            mock_mean_rev.return_value = Mock(spec=MeanReversionBand)
            mock_direction.return_value = ReversalDirection.BULLISH
            mock_level.return_value = Price(Decimal("50000"), Currency("USD"))
            mock_confidence.return_value = ConfidenceScore(0.8)
            mock_strength.return_value = SignalStrengthScore(0.7)

            # Выполняем тест
            result = predictor.predict_reversal("BTC/USDT", sample_ohlcv_data, sample_order_book)

            # Проверяем результат
            assert result is not None
            assert isinstance(result, ReversalSignal)
            assert result.symbol == "BTC/USDT"
            assert result.direction == ReversalDirection.BULLISH
            assert result.confidence == ConfidenceScore(0.8)
            assert result.signal_strength == SignalStrengthScore(0.7)

    def test_predict_reversal_insufficient_data(self, predictor: ReversalPredictor) -> None:
        """Тест с недостаточными данными."""
        small_data = pd.DataFrame(
            {
                "open": [50000] * 10,
                "high": [50100] * 10,
                "low": [49900] * 10,
                "close": [50050] * 10,
                "volume": [1000] * 10,
            }
        )

        result = predictor.predict_reversal("BTC/USDT", small_data)
        assert result is None

    def test_predict_reversal_neutral_direction(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест с нейтральным направлением."""
        with patch.object(predictor, "_determine_reversal_direction") as mock_direction:
            mock_direction.return_value = ReversalDirection.NEUTRAL

            result = predictor.predict_reversal("BTC/USDT", sample_ohlcv_data)
            assert result is None

    def test_predict_reversal_low_confidence(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест с низкой уверенностью."""
        with patch.object(predictor, "_determine_reversal_direction") as mock_direction, patch.object(
            predictor, "_calculate_confidence"
        ) as mock_confidence:

            mock_direction.return_value = ReversalDirection.BULLISH
            mock_confidence.return_value = ConfidenceScore(0.2)  # Ниже min_confidence

            result = predictor.predict_reversal("BTC/USDT", sample_ohlcv_data)
            assert result is None

    def test_predict_reversal_low_signal_strength(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест с низкой силой сигнала."""
        with patch.object(predictor, "_determine_reversal_direction") as mock_direction, patch.object(
            predictor, "_calculate_confidence"
        ) as mock_confidence, patch.object(predictor, "_calculate_signal_strength") as mock_strength:

            mock_direction.return_value = ReversalDirection.BULLISH
            mock_confidence.return_value = ConfidenceScore(0.8)
            mock_strength.return_value = SignalStrengthScore(0.2)  # Ниже min_signal_strength

            result = predictor.predict_reversal("BTC/USDT", sample_ohlcv_data)
            assert result is None

    def test_predict_reversal_exception_handling(self, predictor: ReversalPredictor) -> None:
        """Тест обработки исключений."""
        invalid_data = "invalid_data"

        result = predictor.predict_reversal("BTC/USDT", invalid_data)
        assert result is None

    def test_analyze_divergences(self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame) -> None:
        """Тест анализа дивергенций."""
        with patch.object(predictor, "_detect_rsi_divergences") as mock_rsi, patch.object(
            predictor, "_detect_macd_divergences"
        ) as mock_macd:

            mock_rsi.return_value = [Mock(spec=DivergenceSignal)]
            mock_macd.return_value = [Mock(spec=DivergenceSignal)]

            result = predictor._analyze_divergences(sample_ohlcv_data)

            assert len(result) == 2
            mock_rsi.assert_called_once_with(sample_ohlcv_data)
            mock_macd.assert_called_once_with(sample_ohlcv_data)

    def test_analyze_divergences_exception(self, predictor: ReversalPredictor) -> None:
        """Тест анализа дивергенций с исключением."""
        result = predictor._analyze_divergences("invalid_data")
        assert result == []

    def test_detect_rsi_divergences_bearish(self, predictor: ReversalPredictor) -> None:
        """Тест обнаружения медвежьих дивергенций RSI."""
        # Создаем данные с медвежьей дивергенцией
        data = pd.DataFrame(
            {
                "high": [100, 110, 120, 130, 140],  # Растущие максимумы
                "low": [90, 95, 100, 105, 110],
                "close": [95, 105, 115, 125, 135],
            }
        )

        with patch.object(predictor, "_calculate_rsi") as mock_rsi, patch.object(
            predictor, "_find_peaks"
        ) as mock_peaks:

            # Мокаем RSI с падающими максимумами
            mock_rsi.return_value = pd.Series([70, 65, 60, 55, 50])
            mock_peaks.side_effect = [
                [1, 3],  # price_highs
                [0, 2, 4],  # price_lows
                [1, 3],  # rsi_highs
                [0, 2, 4],  # rsi_lows
            ]

            result = predictor._detect_rsi_divergences(data)

            assert len(result) > 0
            assert result[0].type == DivergenceType.BEARISH_REGULAR
            assert result[0].indicator == "RSI"

    def test_detect_rsi_divergences_bullish(self, predictor: ReversalPredictor) -> None:
        """Тест обнаружения бычьих дивергенций RSI."""
        data = pd.DataFrame(
            {
                "high": [140, 130, 120, 110, 100],  # Падающие максимумы
                "low": [110, 105, 100, 95, 90],  # Падающие минимумы
                "close": [135, 125, 115, 105, 95],
            }
        )

        with patch.object(predictor, "_calculate_rsi") as mock_rsi, patch.object(
            predictor, "_find_peaks"
        ) as mock_peaks:

            # Мокаем RSI с растущими минимумами
            mock_rsi.return_value = pd.Series([30, 35, 40, 45, 50])
            mock_peaks.side_effect = [
                [0, 2, 4],  # price_highs
                [1, 3],  # price_lows
                [0, 2, 4],  # rsi_highs
                [1, 3],  # rsi_lows
            ]

            result = predictor._detect_rsi_divergences(data)

            assert len(result) > 0
            assert result[0].type == DivergenceType.BULLISH_REGULAR
            assert result[0].indicator == "RSI"

    def test_detect_macd_divergences(self, predictor: ReversalPredictor) -> None:
        """Тест обнаружения дивергенций MACD."""
        data = pd.DataFrame(
            {"high": [100, 110, 120, 130, 140], "low": [90, 95, 100, 105, 110], "close": [95, 105, 115, 125, 135]}
        )

        with patch.object(predictor, "_calculate_macd") as mock_macd, patch.object(
            predictor, "_find_peaks"
        ) as mock_peaks:

            # Мокаем MACD
            mock_macd.return_value = (pd.Series([1, 2, 1.5, 1, 0.5]), pd.Series([0.5, 1, 1.2, 1, 0.8]))
            mock_peaks.side_effect = [
                [1, 3],  # price_highs
                [0, 2, 4],  # price_lows
                [1, 3],  # macd_highs
                [0, 2, 4],  # macd_lows
            ]

            result = predictor._detect_macd_divergences(data)

            assert isinstance(result, list)
            # Может быть пустым или содержать дивергенции в зависимости от данных

    def test_analyze_candlestick_patterns(self, predictor: ReversalPredictor) -> None:
        """Тест анализа свечных паттернов."""
        # Создаем данные с разными паттернами
        data = pd.DataFrame(
            {"open": [100, 100, 100], "high": [105, 110, 105], "low": [95, 90, 95], "close": [100, 100, 100]}
        )

        with patch.object(predictor, "_is_doji") as mock_doji, patch.object(
            predictor, "_is_hammer"
        ) as mock_hammer, patch.object(predictor, "_is_shooting_star") as mock_shooting:

            mock_doji.return_value = True
            mock_hammer.return_value = False
            mock_shooting.return_value = False

            result = predictor._analyze_candlestick_patterns(data)

            assert len(result) == 3  # По одному паттерну на каждую свечу
            assert all(isinstance(pattern, CandlestickPattern) for pattern in result)

    def test_analyze_momentum(self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame) -> None:
        """Тест анализа импульса."""
        result = predictor._analyze_momentum(sample_ohlcv_data)

        assert result is not None
        assert isinstance(result, MomentumAnalysis)
        assert result.timestamp is not None
        assert isinstance(result.momentum_loss, float)
        assert isinstance(result.velocity_change, float)
        assert isinstance(result.acceleration, float)

    def test_analyze_momentum_insufficient_data(self, predictor: ReversalPredictor) -> None:
        """Тест анализа импульса с недостаточными данными."""
        small_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # Меньше 20 точек

        result = predictor._analyze_momentum(small_data)
        assert result is None

    def test_analyze_mean_reversion(self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame) -> None:
        """Тест анализа среднего возврата."""
        result = predictor._analyze_mean_reversion(sample_ohlcv_data)

        assert result is not None
        assert isinstance(result, MeanReversionBand)
        assert result.upper_band is not None
        assert result.lower_band is not None
        assert result.middle_line is not None
        assert isinstance(result.deviation, float)
        assert isinstance(result.band_width, float)
        assert isinstance(result.current_position, float)

    def test_analyze_mean_reversion_insufficient_data(self, predictor: ReversalPredictor) -> None:
        """Тест анализа среднего возврата с недостаточными данными."""
        small_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # Меньше 20 точек

        result = predictor._analyze_mean_reversion(small_data)
        assert result is None

    def test_determine_reversal_direction_bullish(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест определения бычьего направления."""
        divergence_signals = [Mock(spec=DivergenceSignal, type=DivergenceType.BULLISH_REGULAR)]
        candlestick_patterns = [Mock(spec=CandlestickPattern, name="hammer")]

        result = predictor._determine_reversal_direction(
            sample_ohlcv_data, [1, 2], [3, 4], divergence_signals, candlestick_patterns
        )

        assert result == ReversalDirection.BULLISH

    def test_determine_reversal_direction_bearish(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест определения медвежьего направления."""
        divergence_signals = [Mock(spec=DivergenceSignal, type=DivergenceType.BEARISH_REGULAR)]
        candlestick_patterns = [Mock(spec=CandlestickPattern, name="shooting_star")]

        result = predictor._determine_reversal_direction(
            sample_ohlcv_data, [1, 2, 3], [4], divergence_signals, candlestick_patterns
        )

        assert result == ReversalDirection.BEARISH

    def test_determine_reversal_direction_neutral(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест определения нейтрального направления."""
        result = predictor._determine_reversal_direction(sample_ohlcv_data, [1, 2], [3, 4], [], [])

        assert result == ReversalDirection.NEUTRAL

    def test_calculate_reversal_level_bullish(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест расчета уровня разворота для бычьего тренда."""
        low_pivots = [45000, 46000, 47000]

        result = predictor._calculate_reversal_level(ReversalDirection.BULLISH, sample_ohlcv_data, [], low_pivots, None)

        assert isinstance(result, Price)
        assert result.value == Decimal("45000")  # Минимальный из low_pivots

    def test_calculate_reversal_level_bearish(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест расчета уровня разворота для медвежьего тренда."""
        high_pivots = [55000, 56000, 57000]

        result = predictor._calculate_reversal_level(
            ReversalDirection.BEARISH, sample_ohlcv_data, high_pivots, [], None
        )

        assert isinstance(result, Price)
        assert result.value == Decimal("57000")  # Максимальный из high_pivots

    def test_calculate_reversal_level_neutral(
        self, predictor: ReversalPredictor, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Тест расчета уровня разворота для нейтрального тренда."""
        current_price = float(sample_ohlcv_data["close"].iloc[-1])

        result = predictor._calculate_reversal_level(ReversalDirection.NEUTRAL, sample_ohlcv_data, [], [], None)

        assert isinstance(result, Price)
        assert abs(float(result.value) - current_price) < 1.0

    def test_calculate_confidence(self, predictor: ReversalPredictor) -> None:
        """Тест расчета уверенности."""
        divergence_signals = [Mock(spec=DivergenceSignal, confidence=ConfidenceScore(0.8))]
        candlestick_patterns = [Mock(spec=CandlestickPattern, confirmation_level=0.7)]
        momentum_analysis = Mock(spec=MomentumAnalysis, momentum_loss=0.2)

        result = predictor._calculate_confidence(
            ReversalDirection.BEARISH, divergence_signals, candlestick_patterns, momentum_analysis
        )

        assert isinstance(result, ConfidenceScore)
        assert 0.0 <= float(result) <= 1.0

    def test_calculate_signal_strength(self, predictor: ReversalPredictor) -> None:
        """Тест расчета силы сигнала."""
        confidence = ConfidenceScore(0.8)
        divergence_signals = [Mock(spec=DivergenceSignal, strength=0.7)]
        candlestick_patterns = [Mock(spec=CandlestickPattern, strength=0.6)]
        momentum_analysis = Mock(spec=MomentumAnalysis, momentum_loss=0.3)

        result = predictor._calculate_signal_strength(
            confidence, divergence_signals, candlestick_patterns, momentum_analysis
        )

        assert isinstance(result, SignalStrengthScore)
        assert 0.0 <= float(result) <= 1.0

    def test_calculate_rsi(self, predictor: ReversalPredictor) -> None:
        """Тест расчета RSI."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90])

        result = predictor._calculate_rsi(prices, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert all(0 <= rsi <= 100 for rsi in result if not pd.isna(rsi))

    def test_calculate_rsi_insufficient_data(self, predictor: ReversalPredictor) -> None:
        """Тест расчета RSI с недостаточными данными."""
        prices = pd.Series([100, 101, 102])  # Меньше периода

        result = predictor._calculate_rsi(prices, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert all(rsi == 50.0 for rsi in result)

    def test_calculate_macd(self, predictor: ReversalPredictor) -> None:
        """Тест расчета MACD."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

        macd, signal = predictor._calculate_macd(prices)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)

    def test_find_peaks_high(self, predictor: ReversalPredictor) -> None:
        """Тест поиска пиков максимумов."""
        arr = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1])

        result = predictor._find_peaks(arr, "high")

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_find_peaks_low(self, predictor: ReversalPredictor) -> None:
        """Тест поиска пиков минимумов."""
        arr = np.array([3, 2, 1, 2, 3, 2, 1, 2, 3])

        result = predictor._find_peaks(arr, "low")

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_find_peaks_empty_array(self, predictor: ReversalPredictor) -> None:
        """Тест поиска пиков в пустом массиве."""
        arr = np.array([])

        result = predictor._find_peaks(arr, "high")

        assert result == []

    def test_is_doji_true(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Doji - истинный случай."""
        candle = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.1})  # Очень маленькое тело

        result = predictor._is_doji(candle)
        assert result is True

    def test_is_doji_false(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Doji - ложный случай."""
        candle = pd.Series({"open": 100.0, "high": 105.0, "low": 95.0, "close": 104.0})  # Большое тело

        result = predictor._is_doji(candle)
        assert result is False

    def test_is_hammer_true(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Hammer - истинный случай."""
        candle = pd.Series({"open": 100.0, "high": 101.0, "low": 90.0, "close": 100.5})  # Длинная нижняя тень

        result = predictor._is_hammer(candle)
        assert result is True

    def test_is_hammer_false(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Hammer - ложный случай."""
        candle = pd.Series({"open": 100.0, "high": 110.0, "low": 99.0, "close": 100.5})  # Длинная верхняя тень

        result = predictor._is_hammer(candle)
        assert result is False

    def test_is_shooting_star_true(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Shooting Star - истинный случай."""
        candle = pd.Series({"open": 100.0, "high": 110.0, "low": 99.0, "close": 100.5})  # Длинная верхняя тень

        result = predictor._is_shooting_star(candle)
        assert result is True

    def test_is_shooting_star_false(self, predictor: ReversalPredictor) -> None:
        """Тест проверки паттерна Shooting Star - ложный случай."""
        candle = pd.Series({"open": 100.0, "high": 101.0, "low": 90.0, "close": 100.5})  # Длинная нижняя тень

        result = predictor._is_shooting_star(candle)
        assert result is False


class TestReversalPredictorIntegration:
    """Интеграционные тесты для ReversalPredictor."""

    @pytest.fixture
    def predictor(self) -> ReversalPredictor:
        """Экземпляр ReversalPredictor для интеграционных тестов."""
        return ReversalPredictor()

    def test_full_prediction_workflow(self, predictor: ReversalPredictor) -> None:
        """Тест полного рабочего процесса прогнозирования."""
        # Создаем реалистичные данные с трендом
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
        np.random.seed(42)

        # Создаем данные с явным трендом вверх, затем разворотом
        trend_up = np.linspace(50000, 55000, 70)
        trend_down = np.linspace(55000, 52000, 30)
        prices = np.concatenate([trend_up, trend_down])

        # Добавляем шум
        noise = np.random.normal(0, 50, 100)
        prices += noise

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 30, 100),
                "low": prices - np.random.uniform(0, 30, 100),
                "close": prices + np.random.normal(0, 10, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # Выполняем прогнозирование
        result = predictor.predict_reversal("BTC/USDT", data)

        # Проверяем, что результат может быть None или ReversalSignal
        if result is not None:
            assert isinstance(result, ReversalSignal)
            assert result.symbol == "BTC/USDT"
            assert result.direction in [ReversalDirection.BULLISH, ReversalDirection.BEARISH, ReversalDirection.NEUTRAL]
            assert result.pivot_price is not None
            assert 0.0 <= float(result.confidence) <= 1.0
            assert 0.0 <= float(result.signal_strength) <= 1.0

    def test_prediction_with_order_book(self, predictor: ReversalPredictor) -> None:
        """Тест прогнозирования с данными ордербука."""
        # Создаем данные
        data = pd.DataFrame(
            {
                "open": [50000] * 100,
                "high": [50100] * 100,
                "low": [49900] * 100,
                "close": [50050] * 100,
                "volume": [1000] * 100,
            }
        )

        order_book = {
            "bids": [[50000, 1.0], [49999, 2.0]],
            "asks": [[50001, 1.2], [50002, 1.8]],
            "timestamp": datetime.now(),
        }

        result = predictor.predict_reversal("BTC/USDT", data, order_book)

        # Проверяем, что результат может быть None или ReversalSignal
        if result is not None:
            assert isinstance(result, ReversalSignal)
            assert result.liquidity_clusters is not None

    def test_prediction_error_handling(self, predictor: ReversalPredictor) -> None:
        """Тест обработки ошибок в прогнозировании."""
        # Тестируем с некорректными данными
        invalid_data = "invalid_data"

        result = predictor.predict_reversal("BTC/USDT", invalid_data)
        assert result is None

        # Тестируем с пустыми данными
        empty_data = pd.DataFrame()
        result = predictor.predict_reversal("BTC/USDT", empty_data)
        assert result is None
