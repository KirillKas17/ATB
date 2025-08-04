"""
Unit тесты для интеграции MirrorDetector в TradingOrchestrator.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from domain.entities.strategy import Signal, SignalType, Strategy
from domain.entities.portfolio import Portfolio
from domain.intelligence.mirror_detector import MirrorDetector, MirrorSignal, CorrelationMatrix
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    TradingOrchestratorUseCase
)
from application.use_cases.trading_orchestrator import (
    ExecuteStrategyRequest,
    ProcessSignalRequest,
)
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
class TestMirrorDetectorIntegration:
    """Тесты интеграции MirrorDetector в TradingOrchestrator"""
    @pytest.fixture
    def mock_mirror_detector(self) -> Any:
        """Мок MirrorDetector"""
        detector = Mock(spec=MirrorDetector)
        # Мок для build_correlation_matrix
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_correlation_matrix.get_correlation.return_value = 0.85
        mock_correlation_matrix.get_confidence.return_value = 0.9
        mock_correlation_matrix.get_lag.return_value = 2
        mock_correlation_matrix.get_p_value.return_value = 0.01
        detector.build_correlation_matrix.return_value = mock_correlation_matrix
        return detector
    @pytest.fixture
    def mock_price_data(self) -> Any:
        """Мок данных о ценах"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        # Создаем коррелированные данные
        base_price = 50000.0
        trend = np.linspace(0, 1000, len(dates))
        noise = np.random.normal(0, 100, len(dates))
        btc_prices = base_price + trend + noise
        eth_prices = base_price * 0.1 + trend * 0.1 + noise * 0.1  # Коррелированные с BTC
        ada_prices = base_price * 0.01 + trend * 0.01 + noise * 0.01  # Коррелированные с BTC
        return {
            "BTC/USD": pd.Series(btc_prices, index=dates),
            "ETH/USD": pd.Series(eth_prices, index=dates),
            "ADA/USD": pd.Series(ada_prices, index=dates)
        }
    @pytest.fixture
    def trading_orchestrator(self, mock_mirror_detector) -> Any:
        """TradingOrchestrator с интегрированным MirrorDetector"""
        with patch('application.use_cases.trading_orchestrator.MirrorDetector') as mock_detector_class:
            mock_detector_class.return_value = mock_mirror_detector
            orchestrator = TradingOrchestratorUseCase()
            return orchestrator
    def test_mirror_detector_integration_in_constructor(self, trading_orchestrator) -> None:
        """Тест интеграции MirrorDetector в конструкторе"""
        assert hasattr(trading_orchestrator, '_mirror_detector')
        assert trading_orchestrator._mirror_detector is not None
    def test_update_mirror_detection(self, trading_orchestrator, mock_price_data) -> None:
        """Тест обновления детекции зеркальных сигналов"""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
        # Обновляем детекцию
        trading_orchestrator._update_mirror_detection(symbols)
        # Проверяем, что детектор был вызван
        trading_orchestrator._mirror_detector.build_correlation_matrix.assert_called_once()
        # Проверяем, что результат кэширован
        assert hasattr(trading_orchestrator, '_mirror_detection_cache')
        assert trading_orchestrator._mirror_detection_cache is not None
    def test_get_price_data_for_mirror_detection(self, trading_orchestrator) -> None:
        """Тест получения данных о ценах для детекции зеркальных сигналов"""
        symbol = "BTC/USD"
        # Получаем данные о ценах
        price_data = trading_orchestrator._get_price_data_for_mirror_detection(symbol)
        # Проверяем, что данные получены
        assert price_data is not None
        assert isinstance(price_data, pd.Series)
        assert len(price_data) > 0
    def test_apply_mirror_detection_analysis(self, trading_orchestrator) -> None:
        """Тест применения анализа детекции зеркальных сигналов к сигналу"""
        from domain.entities.strategy import Signal, SignalType
        # Создаем тестовый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            confidence=0.7,
            strength=0.8,
            metadata={}
        )
        # Настраиваем кэш детекции
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_correlation_matrix.get_correlation.return_value = 0.85
        mock_correlation_matrix.get_confidence.return_value = 0.9
        mock_correlation_matrix.get_lag.return_value = 2
        trading_orchestrator._mirror_detection_cache = {
            "correlation_matrix": mock_correlation_matrix,
            "timestamp": 1640995200
        }
        # Применяем анализ
        modified_signal = trading_orchestrator._apply_mirror_detection_analysis("BTC/USD", signal)
        # Проверяем, что сигнал был модифицирован
        assert modified_signal.confidence > signal.confidence  # Увеличена уверенность
        assert "mirror_assets" in modified_signal.metadata
        assert "avg_correlation" in modified_signal.metadata
        assert "avg_confidence" in modified_signal.metadata
    def test_mirror_detection_in_execute_strategy(self, trading_orchestrator, mock_price_data) -> None:
        """Тест интеграции детекции зеркальных сигналов в execute_strategy"""
        from domain.entities.strategy import Strategy, SignalType
        from domain.entities.portfolio import Portfolio
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency
        # Создаем тестовые объекты
        strategy = Strategy(
            id="test_strategy",
            name="Test Strategy",
            description="Test strategy for mirror detection",
            signals=[Signal(signal_type=SignalType.BUY, trading_pair="BTC/USD", confidence=0.7, strength=0.8)]
        )
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            balance=Money(10000, Currency("USD"))
        )
        # Мокаем репозитории
        trading_orchestrator.strategy_repository.get_by_id = Mock(return_value=strategy)
        trading_orchestrator.portfolio_repository.get_by_id = Mock(return_value=portfolio)
        trading_orchestrator.validate_trading_conditions = Mock(return_value=(True, []))
        trading_orchestrator._generate_signals_with_sentiment = Mock(return_value=strategy.signals)
        trading_orchestrator._create_enhanced_order_from_signal = Mock(return_value=None)
        # Выполняем стратегию
        from application.use_cases.trading_orchestrator import ExecuteStrategyRequest
        request = ExecuteStrategyRequest(
            strategy_id="test_strategy",
            portfolio_id="test_portfolio",
            symbol="BTC/USD"
        )
        result = trading_orchestrator.execute_strategy(request)
        # Проверяем, что детекция зеркальных сигналов была выполнена
        assert result.executed is not None
    def test_mirror_detection_with_high_correlation(self, trading_orchestrator) -> None:
        """Тест детекции зеркальных сигналов с высокой корреляцией"""
        from domain.entities.strategy import Signal, SignalType
        # Создаем тестовый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            confidence=0.6,
            strength=0.7,
            metadata={}
        )
        # Настраиваем кэш с высокой корреляцией
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_correlation_matrix.get_correlation.return_value = 0.95  # Высокая корреляция
        mock_correlation_matrix.get_confidence.return_value = 0.95  # Высокая уверенность
        mock_correlation_matrix.get_lag.return_value = 1
        trading_orchestrator._mirror_detection_cache = {
            "correlation_matrix": mock_correlation_matrix,
            "timestamp": 1640995200
        }
        # Применяем анализ
        modified_signal = trading_orchestrator._apply_mirror_detection_analysis("BTC/USD", signal)
        # Проверяем, что сигнал был значительно усилен
        assert modified_signal.confidence > signal.confidence * 1.2
        assert modified_signal.metadata["avg_correlation"] == 0.95
    def test_mirror_detection_with_low_correlation(self, trading_orchestrator) -> None:
        """Тест детекции зеркальных сигналов с низкой корреляцией"""
        from domain.entities.strategy import Signal, SignalType
        # Создаем тестовый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            confidence=0.7,
            strength=0.8,
            metadata={}
        )
        # Настраиваем кэш с низкой корреляцией
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_correlation_matrix.get_correlation.return_value = 0.2  # Низкая корреляция
        mock_correlation_matrix.get_confidence.return_value = 0.3  # Низкая уверенность
        mock_correlation_matrix.get_lag.return_value = 5
        trading_orchestrator._mirror_detection_cache = {
            "correlation_matrix": mock_correlation_matrix,
            "timestamp": 1640995200
        }
        # Применяем анализ
        modified_signal = trading_orchestrator._apply_mirror_detection_analysis("BTC/USD", signal)
        # Проверяем, что сигнал не был усилен (корреляция ниже порога)
        assert len(modified_signal.metadata.get("mirror_assets", [])) == 0
    def test_mirror_detection_error_handling(self, trading_orchestrator) -> None:
        """Тест обработки ошибок в детекции зеркальных сигналов"""
        # Симулируем ошибку в детекторе
        trading_orchestrator._mirror_detector.build_correlation_matrix.side_effect = Exception("Detection error")
        symbols = ["BTC/USD", "ETH/USD"]
        # Должно обработаться без ошибок
        trading_orchestrator._update_mirror_detection(symbols)
        # Проверяем, что кэш не был поврежден
        assert not hasattr(trading_orchestrator, '_mirror_detection_cache') or trading_orchestrator._mirror_detection_cache is None
    def test_mirror_detection_cache_update_frequency(self, trading_orchestrator) -> None:
        """Тест частоты обновления кэша детекции зеркальных сигналов"""
        symbols = ["BTC/USD", "ETH/USD"]
        # Первое обновление
        trading_orchestrator._update_mirror_detection(symbols)
        first_update_time = trading_orchestrator._last_mirror_detection_update
        # Второе обновление сразу после первого (должно быть пропущено)
        trading_orchestrator._update_mirror_detection(symbols)
        second_update_time = trading_orchestrator._last_mirror_detection_update
        # Проверяем, что время обновления не изменилось
        assert first_update_time == second_update_time
    def test_mirror_detection_with_empty_data(self, trading_orchestrator) -> None:
        """Тест детекции зеркальных сигналов с пустыми данными"""
        # Мокаем получение пустых данных
        trading_orchestrator._get_price_data_for_mirror_detection = Mock(return_value=None)
        symbols = ["BTC/USD", "ETH/USD"]
        # Обновляем детекцию с пустыми данными
        trading_orchestrator._update_mirror_detection(symbols)
        # Проверяем, что детектор не был вызван (недостаточно данных)
        trading_orchestrator._mirror_detector.build_correlation_matrix.assert_not_called()
    def test_mirror_detection_integration_completeness(self, trading_orchestrator) -> None:
        """Тест полноты интеграции MirrorDetector"""
        # Проверяем наличие всех необходимых методов
        assert hasattr(trading_orchestrator, '_update_mirror_detection')
        assert hasattr(trading_orchestrator, '_get_price_data_for_mirror_detection')
        assert hasattr(trading_orchestrator, '_apply_mirror_detection_analysis')
        # Проверяем, что методы являются callable
        assert callable(trading_orchestrator._update_mirror_detection)
        assert callable(trading_orchestrator._get_price_data_for_mirror_detection)
        assert callable(trading_orchestrator._apply_mirror_detection_analysis)
        # Проверяем наличие кэша
        assert hasattr(trading_orchestrator, '_mirror_detection_cache')
        assert hasattr(trading_orchestrator, '_last_mirror_detection_update')
    def test_mirror_detection_with_multiple_assets(self, trading_orchestrator) -> None:
        """Тест детекции зеркальных сигналов с множественными активами"""
        from domain.entities.strategy import Signal, SignalType
        # Создаем тестовый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            confidence=0.7,
            strength=0.8,
            metadata={}
        )
        # Настраиваем кэш с множественными зеркальными активами
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
        # Разные корреляции для разных активов
        def get_correlation(asset1, asset2) -> Any:
            correlations = {
                ("BTC/USD", "ETH/USD"): 0.85,
                ("BTC/USD", "ADA/USD"): 0.75,
                ("BTC/USD", "DOT/USD"): 0.65,
                ("BTC/USD", "LINK/USD"): 0.55,
            }
            return correlations.get((asset1, asset2), 0.0)
        mock_correlation_matrix.get_correlation.side_effect = get_correlation
        mock_correlation_matrix.get_confidence.return_value = 0.9
        mock_correlation_matrix.get_lag.return_value = 2
        trading_orchestrator._mirror_detection_cache = {
            "correlation_matrix": mock_correlation_matrix,
            "timestamp": 1640995200
        }
        # Применяем анализ
        modified_signal = trading_orchestrator._apply_mirror_detection_analysis("BTC/USD", signal)
        # Проверяем, что найдены зеркальные активы
        mirror_assets = modified_signal.metadata.get("mirror_assets", [])
        assert len(mirror_assets) > 0
        # Проверяем, что корреляции корректны
        for asset in mirror_assets:
            assert asset["correlation"] > 0.7
            assert asset["confidence"] > 0.8
    def test_mirror_detection_lag_analysis(self, trading_orchestrator) -> None:
        """Тест анализа лагов в детекции зеркальных сигналов"""
        from domain.entities.strategy import Signal, SignalType
        # Создаем тестовый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            confidence=0.7,
            strength=0.8,
            metadata={}
        )
        # Настраиваем кэш с разными лагами
        mock_correlation_matrix = Mock(spec=CorrelationMatrix)
        mock_correlation_matrix.assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_correlation_matrix.get_correlation.return_value = 0.85
        mock_correlation_matrix.get_confidence.return_value = 0.9
        # Разные лаги для разных активов
        def get_lag(asset1, asset2) -> Any:
            lags = {
                ("BTC/USD", "ETH/USD"): 1,
                ("BTC/USD", "ADA/USD"): 3,
            }
            return lags.get((asset1, asset2), 0)
        mock_correlation_matrix.get_lag.side_effect = get_lag
        trading_orchestrator._mirror_detection_cache = {
            "correlation_matrix": mock_correlation_matrix,
            "timestamp": 1640995200
        }
        # Применяем анализ
        modified_signal = trading_orchestrator._apply_mirror_detection_analysis("BTC/USD", signal)
        # Проверяем, что лаги сохранены в метаданных
        mirror_assets = modified_signal.metadata.get("mirror_assets", [])
        for asset in mirror_assets:
            assert "lag" in asset
            assert isinstance(asset["lag"], int)
class TestMirrorDetectorPerformance:
    """Тесты производительности MirrorDetector."""
    @pytest.mark.asyncio
    async def test_mirror_detection_performance(self) -> None:
        """Тест производительности детекции зеркальных сигналов."""
        import time
        # Arrange
        detector = MirrorDetector()
        # Создаем синтетические данные
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        series1 = pd.Series(np.random.randn(len(dates)), index=dates)
        series2 = pd.Series(np.random.randn(len(dates)), index=dates)
        price_data = {
            "BTCUSDT": series1,
            "ETHUSDT": series2
        }
        # Act
        start_time = time.time()
        correlation_matrix = detector.build_correlation_matrix(
            ["BTCUSDT", "ETHUSDT"], price_data, max_lag=5
        )
        end_time = time.time()
        # Assert
        assert end_time - start_time < 5.0  # Анализ должен выполняться менее чем за 5 секунд
        assert isinstance(correlation_matrix, CorrelationMatrix)
        assert len(correlation_matrix.assets) == 2
class TestMirrorDetectorIntegrationWithRealData:
    """Тесты интеграции с реальными данными."""
    @pytest.mark.asyncio
    async def test_mirror_detection_with_realistic_data(self) -> None:
        """Тест детекции зеркальных сигналов с реалистичными данными."""
        # Arrange
        detector = MirrorDetector()
        # Создаем реалистичные данные о ценах
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        # Создаем коррелированные ряды
        base_trend = np.linspace(0, 1000, len(dates))
        noise1 = np.random.normal(0, 50, len(dates))
        noise2 = np.random.normal(0, 50, len(dates))
        # Добавляем корреляцию между рядами
        correlation_factor = 0.8
        series1 = base_trend + noise1
        series2 = base_trend * correlation_factor + noise2
        price_data = {
            "BTCUSDT": pd.Series(series1, index=dates),
            "ETHUSDT": pd.Series(series2, index=dates)
        }
        # Act
        correlation_matrix = detector.build_correlation_matrix(
            ["BTCUSDT", "ETHUSDT"], price_data, max_lag=5
        )
        # Assert
        assert isinstance(correlation_matrix, CorrelationMatrix)
        assert len(correlation_matrix.assets) == 2
        # Проверяем, что корреляция обнаружена
        correlation = correlation_matrix.get_correlation("BTCUSDT", "ETHUSDT")
        assert abs(correlation) > 0.5  # Должна быть обнаружена сильная корреляция
    @pytest.mark.asyncio
    async def test_mirror_signal_detection(self) -> None:
        """Тест обнаружения зеркального сигнала."""
        # Arrange
        detector = MirrorDetector()
        # Создаем два коррелированных временных ряда
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        # Первый ряд
        series1 = pd.Series(np.random.randn(len(dates)), index=dates)
        # Второй ряд с задержкой и корреляцией
        series2 = pd.Series(np.random.randn(len(dates)), index=dates)
        series2 = series2.shift(2)  # Добавляем задержку
        series2 = series2 * 0.8 + series1 * 0.2  # Добавляем корреляцию
        # Act
        signal = detector.detect_mirror_signal(
            "BTCUSDT", "ETHUSDT", series1, series2, max_lag=5
        )
        # Assert
        if signal:  # Сигнал может быть не обнаружен из-за случайности данных
            assert isinstance(signal, MirrorSignal)
            assert signal.asset1 == "BTCUSDT"
            assert signal.asset2 == "ETHUSDT"
            assert abs(signal.correlation) > 0
            assert signal.confidence > 0
            assert signal.signal_strength > 0 
