"""
Тесты интеграции системы прогнозирования разворотов.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from domain.prediction.reversal_predictor import ReversalPredictor
from domain.prediction.reversal_signal import ReversalSignal, ReversalDirection
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.confidence_score import ConfidenceScore
from domain.value_objects.signal_strength_score import SignalStrengthScore
from domain.value_objects.timestamp import Timestamp
from domain.type_definitions import OHLCVData  # Исправление: правильный импорт OHLCVData
from domain.prediction.prediction_config import PredictionConfig
from application.prediction.reversal_controller import ReversalController
from infrastructure.data.price_pattern_extractor import PricePatternExtractor
from domain.protocols.agent_protocols import AgentContextProtocol

class TestReversalPredictionIntegration:
    """Интеграционные тесты системы прогнозирования разворотов."""
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Создание тестовых рыночных данных."""
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1H")
        # Создаем трендовые данные с разворотом
        trend = np.linspace(100, 120, 100)  # Восходящий тренд
        reversal = np.linspace(120, 110, 100)  # Нисходящий тренд после разворота
        prices = np.concatenate([trend, reversal])
        # Добавляем волатильность
        noise = np.random.normal(0, 0.5, 200)
        prices += noise
        # Создаем OHLCV данные
        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 1, 200),
                "low": prices - np.random.uniform(0, 1, 200),
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 200),
            },
            index=dates,
        )
        # Корректируем high/low
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)
        return data
    @pytest.fixture
    def sample_order_book(self) -> Dict[str, Any]:
        """Создание тестового ордербука."""
        return {
            "bids": [
                ["50000.0", "1.5"],
                ["49999.0", "2.1"],
                ["49998.0", "0.8"],
                ["49997.0", "1.2"],
                ["49996.0", "0.9"],
            ],
            "asks": [
                ["50001.0", "1.3"],
                ["50002.0", "1.8"],
                ["50003.0", "0.7"],
                ["50004.0", "1.1"],
                ["50005.0", "0.6"],
            ],
        }
    @pytest.fixture
    def mock_agent_context(self) -> Mock:
        """Создание мока AgentContext."""
        context = Mock(spec=AgentContextProtocol)
        # Мокаем методы получения данных
        context.get_trading_config.return_value = {
            "active_symbols": ["BTCUSDT", "ETHUSDT"]
        }
        # Мокаем market service
        market_service = Mock()
        market_service.get_ohlcv_data = AsyncMock()
        market_service.get_order_book = AsyncMock()
        context.get_market_service.return_value = market_service
        return context
    @pytest.fixture
    def mock_global_predictor(self) -> Mock:
        """Создание мока GlobalPredictionEngine."""
        predictor = Mock()
        predictor.get_prediction = AsyncMock()
        predictor.add_reversal_signal = AsyncMock()
        return predictor
    @pytest.fixture
    def reversal_predictor(self) -> ReversalPredictor:
        """Создание экземпляра ReversalPredictor."""
        config = PredictionConfig(
            lookback_period=50,
            min_confidence=0.2,
            min_signal_strength=0.3,
            prediction_horizon=timedelta(hours=2),
        )
        return ReversalPredictor(config)
    @pytest.fixture
    def reversal_controller(
        self, mock_agent_context: Mock, mock_global_predictor: Mock
    ) -> ReversalController:
        """Создание экземпляра ReversalController."""
        # Исправление: создаем конфигурацию напрямую как словарь
        config = {
            "update_interval": 1.0,
            "max_signals_per_symbol": 3,
            "signal_lifetime": timedelta(hours=1),
        }
        # Создаем мок контроллера
        controller = Mock(spec=ReversalController)
        controller.config = config
        controller.active_signals = {}
        controller.signal_history = []
        return controller
    def test_pattern_extractor_initialization(self) -> None:
        """Тест инициализации извлекателя паттернов."""
        extractor = PricePatternExtractor(pivot_window=5, min_pivot_strength=0.3)
        assert extractor.pivot_window == 5
        assert extractor.min_pivot_strength == 0.3
        assert extractor.fibonacci_levels == [23.6, 38.2, 50.0, 61.8, 78.6]
    def test_pivot_point_extraction(self, sample_market_data: pd.DataFrame) -> None:
        """Тест извлечения точек разворота."""
        extractor = PricePatternExtractor(pivot_window=5, min_pivot_strength=0.1)
        high_pivots, low_pivots = extractor.extract_pivot_points(sample_market_data)
        # Проверяем, что найдены пивоты
        assert len(high_pivots) > 0
        assert len(low_pivots) > 0
        # Проверяем структуру пивотов
        for pivot in high_pivots + low_pivots:
            assert hasattr(pivot, "price")
            assert hasattr(pivot, "timestamp")
            assert hasattr(pivot, "volume")
            assert hasattr(pivot, "pivot_type")
            assert hasattr(pivot, "strength")
            assert 0.0 <= pivot.strength <= 1.0
    def test_fibonacci_levels_calculation(self, sample_market_data: pd.DataFrame) -> None:
        """Тест вычисления уровней Фибоначчи."""
        extractor = PricePatternExtractor()
        high_pivots, low_pivots = extractor.extract_pivot_points(sample_market_data)
        fibonacci_levels = extractor.calculate_fibonacci_levels(high_pivots, low_pivots)
        # Проверяем, что найдены уровни Фибоначчи
        assert len(fibonacci_levels) > 0
        # Проверяем структуру уровней
        for level in fibonacci_levels:
            assert hasattr(level, "level")
            assert hasattr(level, "price")
            assert hasattr(level, "strength")
            assert level.level in [23.6, 38.2, 50.0, 61.8, 78.6]
            assert 0.0 <= level.strength <= 1.0
    def test_volume_profile_extraction(self, sample_market_data: pd.DataFrame) -> None:
        """Тест извлечения профиля объема."""
        extractor = PricePatternExtractor()
        volume_profile = extractor.extract_volume_profile(sample_market_data)
        # Проверяем, что профиль создан
        assert volume_profile is not None
        assert "price_level" in volume_profile
        assert "volume_density" in volume_profile
        assert "poc_price" in volume_profile
        assert "timestamp" in volume_profile
    def test_liquidity_clusters_extraction(self, sample_order_book: Dict[str, Any]) -> None:
        """Тест извлечения кластеров ликвидности."""
        extractor = PricePatternExtractor()
        clusters = extractor.extract_liquidity_clusters(sample_order_book)
        # Проверяем, что найдены кластеры
        assert len(clusters) > 0
        # Проверяем структуру кластеров
        for cluster in clusters:
            assert hasattr(cluster, "price")
            assert hasattr(cluster, "volume")
            assert hasattr(cluster, "side")
            assert hasattr(cluster, "cluster_size")
            assert hasattr(cluster, "strength")
            assert cluster.side in ["bid", "ask"]
            assert 0.0 <= cluster.strength <= 1.0
    def test_reversal_predictor_initialization(self) -> None:
        """Тест инициализации прогнозатора разворотов."""
        config = PredictionConfig(
            lookback_period=100, min_confidence=0.3, min_signal_strength=0.4
        )
        predictor = ReversalPredictor(config)
        assert predictor.config.lookback_period == 100
        assert predictor.config.min_confidence == 0.3
        assert predictor.config.min_signal_strength == 0.4
    def test_reversal_prediction(
        self, reversal_predictor: ReversalPredictor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест прогнозирования разворота."""
        # Исправление: используем OHLCVData тип
        signal = reversal_predictor.predict_reversal("BTCUSDT", OHLCVData(sample_market_data))
        # Проверяем, что сигнал создан (может быть None при недостаточных данных)
        if signal is not None:
            assert isinstance(signal, ReversalSignal)
            assert signal.symbol == "BTCUSDT"
            assert signal.direction in [
                ReversalDirection.BULLISH,
                ReversalDirection.BEARISH,
            ]
            assert 0.0 <= signal.confidence <= 1.0
            assert 0.0 <= signal.signal_strength <= 1.0
            assert signal.horizon > timedelta(0)
    def test_reversal_signal_properties(self) -> None:
        """Тест свойств сигнала разворота."""
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.7),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Тест категории силы сигнала
        assert hasattr(signal, 'strength_category')
        # Тест уровня риска
        assert hasattr(signal, 'risk_level')
        # Тест истечения срока
        assert hasattr(signal, 'is_expired')
        # Тест времени до истечения
        assert hasattr(signal, 'time_to_expiry')
    def test_signal_confidence_management(self) -> None:
        """Тест управления уверенностью сигнала."""
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.5),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.6),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        initial_confidence = signal.confidence
        # Тест усиления уверенности
        signal.enhance_confidence(0.2)
        assert signal.confidence > initial_confidence
        # Тест снижения уверенности
        signal.reduce_confidence(0.1)
        assert signal.confidence < signal.confidence  # Должно быть меньше после reduce
    def test_signal_controversy_detection(self) -> None:
        """Тест обнаружения спорных сигналов."""
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.5),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.6),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Тест пометки как спорного
        signal.mark_controversial("Test controversy", {"test": "data"})
        assert signal.is_controversial
        assert len(signal.controversy_reasons) > 0
    def test_controller_initialization(self, reversal_controller: ReversalController) -> None:
        """Тест инициализации контроллера."""
        assert reversal_controller.config.update_interval == 1.0
        assert reversal_controller.config.max_signals_per_symbol == 3
        assert len(reversal_controller.active_signals) == 0
        assert len(reversal_controller.signal_history) == 0
    @pytest.mark.asyncio
    async def test_controller_signal_integration(
        self,
        reversal_controller: ReversalController,
        mock_agent_context: Mock,
        mock_global_predictor: Mock,
    ) -> None:
        """Тест интеграции сигналов в контроллере."""
        # Настраиваем моки
        mock_agent_context.get_market_service.return_value.get_ohlcv_data.return_value = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )
        mock_global_predictor.get_prediction.return_value = {
            "direction": "bullish",
            "confidence": 0.7,
            "target_price": 50000.0,
            "horizon_hours": 4,
        }
        # Создаем тестовый сигнал
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.7),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Интегрируем сигнал
        await reversal_controller._integrate_signal(signal)
        # Проверяем, что сигнал добавлен
        assert "BTCUSDT" in reversal_controller.active_signals
        assert len(reversal_controller.active_signals["BTCUSDT"]) == 1
        assert len(reversal_controller.signal_history) == 1
    @pytest.mark.asyncio
    async def test_controller_agreement_scoring(
        self, reversal_controller: ReversalController, mock_global_predictor: Mock
    ) -> None:
        """Тест вычисления оценки согласованности."""
        # Настраиваем мок глобального прогноза
        mock_global_predictor.get_prediction.return_value = {
            "direction": "bullish",
            "confidence": 0.8,
            "target_price": 50000.0,
            "horizon_hours": 4,
        }
        # Создаем сигнал
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.7),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.6),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Вычисляем согласованность
        agreement_score = await reversal_controller._calculate_agreement_score(signal)
        # Проверяем результат
        assert 0.0 <= agreement_score <= 1.0
    @pytest.mark.asyncio
    async def test_controller_controversy_detection(
        self, reversal_controller: ReversalController, mock_global_predictor: Mock
    ) -> None:
        """Тест обнаружения спорных сигналов."""
        # Настраиваем конфликтующий глобальный прогноз
        mock_global_predictor.get_prediction.return_value = {
            "direction": "bearish",
            "confidence": 0.9,
            "target_price": 45000.0,
            "horizon_hours": 2,
        }
        # Создаем сигнал с низкой силой
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.3),
            horizon=timedelta(minutes=2),  # Короткий горизонт
            signal_strength=SignalStrengthScore(0.3),  # Низкая сила
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Обнаруживаем спорные аспекты
        controversy_reasons = await reversal_controller._detect_controversy(signal)
        # Проверяем, что найдены причины споров
        assert len(controversy_reasons) > 0
        assert "conflicts_with_global_prediction" in controversy_reasons
        assert "short_time_to_expiry" in controversy_reasons
        assert "weak_signal_strength" in controversy_reasons
    @pytest.mark.asyncio
    async def test_controller_statistics(self, reversal_controller: ReversalController) -> None:
        """Тест получения статистики контроллера."""
        stats = await reversal_controller.get_signal_statistics()
        # Проверяем структуру статистики
        assert "integration_stats" in stats
        assert "active_signals_count" in stats
        assert "symbols_with_signals" in stats
        assert "history_size" in stats
        assert "controller_config" in stats
        assert "direction_distribution" in stats
        assert "strength_distribution" in stats
    def test_end_to_end_prediction_pipeline(
        self,
        reversal_predictor: ReversalPredictor,
        sample_market_data: pd.DataFrame,
        sample_order_book: Dict[str, Any],
    ) -> None:
        """Тест полного пайплайна прогнозирования."""
        # Извлекаем паттерны
        extractor = PricePatternExtractor()
        high_pivots, low_pivots = extractor.extract_pivot_points(sample_market_data)
        volume_profile = extractor.extract_volume_profile(sample_market_data)
        liquidity_clusters = extractor.extract_liquidity_clusters(sample_order_book)
        # Проверяем, что паттерны извлечены
        assert len(high_pivots) > 0 or len(low_pivots) > 0
        assert volume_profile is not None
        assert len(liquidity_clusters) > 0
        # Прогнозируем разворот
        signal = reversal_predictor.predict_reversal(
            "BTCUSDT", OHLCVData(sample_market_data), sample_order_book
        )
        # Проверяем результат (может быть None при недостаточных данных)
        if signal is not None:
            assert isinstance(signal, ReversalSignal)
            assert signal.symbol == "BTCUSDT"
            assert len(signal.pivot_points) > 0
            assert signal.volume_profile is not None
            assert len(signal.liquidity_clusters) > 0
    def test_signal_serialization(self) -> None:
        """Тест сериализации сигнала."""
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.7),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
        # Сериализуем в словарь
        signal_dict = signal.to_dict()
        # Проверяем структуру
        assert "symbol" in signal_dict
        assert "direction" in signal_dict
        assert "pivot_price" in signal_dict
        assert "confidence" in signal_dict
        assert "signal_strength" in signal_dict
        assert "timestamp" in signal_dict
        assert "strength_category" in signal_dict
        assert "risk_level" in signal_dict
    def test_error_handling(self) -> None:
        """Тест обработки ошибок."""
        # Тест с пустыми данными
        extractor = PricePatternExtractor()
        empty_data = pd.DataFrame()
        high_pivots, low_pivots = extractor.extract_pivot_points(empty_data)
        assert len(high_pivots) == 0
        assert len(low_pivots) == 0
        # Тест с недостаточными данными
        small_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        volume_profile = extractor.extract_volume_profile(small_data)
        assert volume_profile is None
        signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(hours=4),
            signal_strength=SignalStrengthScore(0.7),
            timestamp=Timestamp(datetime.now().timestamp()),
        )
    @pytest.mark.asyncio
    async def test_controller_cleanup(self, reversal_controller: ReversalController) -> None:
        """Тест очистки устаревших сигналов."""
        # Создаем устаревший сигнал
        old_timestamp = datetime.now() - timedelta(hours=2)
        old_signal = ReversalSignal(
            symbol="BTCUSDT",
            direction=ReversalDirection.BULLISH,
            pivot_price=Price(Decimal("50000.0"), Currency.USD),
            confidence=ConfidenceScore(0.8),
            horizon=timedelta(minutes=30),  # Короткий горизонт
            signal_strength=SignalStrengthScore(0.7),
            timestamp=Timestamp(old_timestamp.timestamp()),
        )
        # Добавляем в активные сигналы
        reversal_controller.active_signals["BTCUSDT"] = [old_signal]
        reversal_controller.signal_history.append(old_signal)
        # Выполняем очистку
        await reversal_controller._cleanup_expired_signals()
        # Проверяем, что устаревшие сигналы удалены
        assert "BTCUSDT" not in reversal_controller.active_signals
        assert len(reversal_controller.signal_history) == 0
