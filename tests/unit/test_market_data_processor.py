"""
Unit тесты для MarketDataProcessor.
Тестирует обработку рыночных данных, включая очистку, нормализацию,
агрегацию и анализ данных.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from infrastructure.core.market_data_processor import MarketDataProcessor
class TestMarketDataProcessor:
    """Тесты для MarketDataProcessor."""
    @pytest.fixture
    def market_data_processor(self) -> MarketDataProcessor:
        """Фикстура для MarketDataProcessor."""
        return MarketDataProcessor()
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми рыночными данными."""
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 1000),
            'high': np.random.uniform(46000, 56000, 1000),
            'low': np.random.uniform(44000, 54000, 1000),
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(1000000, 5000000, 1000)
        }, index=dates)
        # Создание более реалистичных данных
        data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 1000, 1000)
        return data
    @pytest.fixture
    def sample_orderbook_data(self) -> dict:
        """Фикстура с данными ордербука."""
        return {
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(),
            "bids": [
                [Decimal("50000.0"), Decimal("1.5")],
                [Decimal("49999.0"), Decimal("2.0")],
                [Decimal("49998.0"), Decimal("3.0")]
            ],
            "asks": [
                [Decimal("50001.0"), Decimal("1.0")],
                [Decimal("50002.0"), Decimal("2.5")],
                [Decimal("50003.0"), Decimal("1.8")]
            ]
        }
    def test_initialization(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест инициализации процессора рыночных данных."""
        assert market_data_processor is not None
        assert hasattr(market_data_processor, 'data_processors')
        assert hasattr(market_data_processor, 'data_validators')
        assert hasattr(market_data_processor, 'data_analyzers')
    def test_process_ohlcv_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест обработки OHLCV данных."""
        # Обработка OHLCV данных
        processed_data = market_data_processor.process_ohlcv_data(sample_market_data)
        # Проверки
        assert processed_data is not None
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(sample_market_data)
        assert len(processed_data.columns) >= len(sample_market_data.columns)
        # Проверка наличия дополнительных столбцов
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in expected_columns:
            assert column in processed_data.columns
    def test_process_orderbook_data(self, market_data_processor: MarketDataProcessor, sample_orderbook_data: dict) -> None:
        """Тест обработки данных ордербука."""
        # Обработка данных ордербука
        processed_orderbook = market_data_processor.process_orderbook_data(sample_orderbook_data)
        # Проверки
        assert processed_orderbook is not None
        assert "processed_bids" in processed_orderbook
        assert "processed_asks" in processed_orderbook
        assert "orderbook_metrics" in processed_orderbook
        assert "liquidity_analysis" in processed_orderbook
        # Проверка типов данных
        assert isinstance(processed_orderbook["processed_bids"], list)
        assert isinstance(processed_orderbook["processed_asks"], list)
        assert isinstance(processed_orderbook["orderbook_metrics"], dict)
        assert isinstance(processed_orderbook["liquidity_analysis"], dict)
    def test_process_trade_data(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест обработки данных о сделках."""
        # Мок данных о сделках
        trade_data = [
            {
                "id": "trade_001",
                "symbol": "BTCUSDT",
                "price": Decimal("50000.0"),
                "quantity": Decimal("0.1"),
                "side": "buy",
                "timestamp": datetime.now()
            },
            {
                "id": "trade_002",
                "symbol": "BTCUSDT",
                "price": Decimal("50001.0"),
                "quantity": Decimal("0.2"),
                "side": "sell",
                "timestamp": datetime.now()
            }
        ]
        # Обработка данных о сделках
        processed_trades = market_data_processor.process_trade_data(trade_data)
        # Проверки
        assert processed_trades is not None
        assert "processed_trades" in processed_trades
        assert "trade_metrics" in processed_trades
        assert "volume_profile" in processed_trades
        # Проверка типов данных
        assert isinstance(processed_trades["processed_trades"], list)
        assert isinstance(processed_trades["trade_metrics"], dict)
        assert isinstance(processed_trades["volume_profile"], dict)
    def test_normalize_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест нормализации рыночных данных."""
        # Нормализация данных
        normalized_data = market_data_processor.normalize_market_data(sample_market_data)
        # Проверки
        assert normalized_data is not None
        assert isinstance(normalized_data, pd.DataFrame)
        assert len(normalized_data) == len(sample_market_data)
        assert len(normalized_data.columns) == len(sample_market_data.columns)
        # Проверка, что нормализованные данные в разумных пределах
        for column in normalized_data.columns:
            if normalized_data[column].dtype in ['float64', 'float32']:
                assert normalized_data[column].min() >= -10
                assert normalized_data[column].max() <= 10
    def test_filter_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест фильтрации рыночных данных."""
        # Фильтрация данных
        filtered_data = market_data_processor.filter_market_data(
            sample_market_data,
            filters={
                "volume": lambda x: x > 2000000,
                "price_change": lambda x: abs(x) < 0.1
            }
        )
        # Проверки
        assert filtered_data is not None
        assert isinstance(filtered_data, pd.DataFrame)
        assert len(filtered_data) <= len(sample_market_data)
    def test_aggregate_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест агрегации рыночных данных."""
        # Агрегация данных
        aggregated_data = market_data_processor.aggregate_market_data(
            sample_market_data,
            freq='1D',
            agg_functions={
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        )
        # Проверки
        assert aggregated_data is not None
        assert isinstance(aggregated_data, pd.DataFrame)
        assert len(aggregated_data) < len(sample_market_data)
        # Проверка наличия ожидаемых столбцов
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in expected_columns:
            assert column in aggregated_data.columns
    def test_calculate_market_metrics(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета рыночных метрик."""
        # Расчет метрик
        market_metrics = market_data_processor.calculate_market_metrics(sample_market_data)
        # Проверки
        assert market_metrics is not None
        assert "volatility" in market_metrics
        assert "volume_profile" in market_metrics
        assert "price_momentum" in market_metrics
        assert "market_regime" in market_metrics
        assert "liquidity_metrics" in market_metrics
        # Проверка типов данных
        assert isinstance(market_metrics["volatility"], float)
        assert isinstance(market_metrics["volume_profile"], dict)
        assert isinstance(market_metrics["price_momentum"], dict)
        assert isinstance(market_metrics["market_regime"], str)
        assert isinstance(market_metrics["liquidity_metrics"], dict)
    def test_detect_market_anomalies(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест обнаружения рыночных аномалий."""
        # Добавление аномалии
        data_with_anomaly = sample_market_data.copy(deep=True)  # type: ignore[no-redef]
        # Подготовка данных с аномалией
        if callable(data_with_anomaly):
            data_with_anomaly: pd.DataFrame = data_with_anomaly()  # type: ignore[no-redef]
        if hasattr(data_with_anomaly, 'loc') and hasattr(data_with_anomaly.index, '__getitem__') and len(data_with_anomaly.index) > 500:
            data_with_anomaly.loc[data_with_anomaly.index[500], 'close'] = 100000
        # Обнаружение аномалий
        anomalies = market_data_processor.detect_market_anomalies(data_with_anomaly)
        # Проверки
        assert anomalies is not None
        assert "anomaly_points" in anomalies
        assert "anomaly_scores" in anomalies
        assert "anomaly_types" in anomalies
        # Проверка типов данных
        assert isinstance(anomalies["anomaly_points"], list)
        assert isinstance(anomalies["anomaly_scores"], dict)
        assert isinstance(anomalies["anomaly_types"], dict)
    def test_analyze_market_liquidity(self, market_data_processor: MarketDataProcessor, sample_orderbook_data: dict) -> None:
        """Тест анализа рыночной ликвидности."""
        # Анализ ликвидности
        liquidity_analysis = market_data_processor.analyze_market_liquidity(sample_orderbook_data)
        # Проверки
        assert liquidity_analysis is not None
        assert "bid_liquidity" in liquidity_analysis
        assert "ask_liquidity" in liquidity_analysis
        assert "spread_analysis" in liquidity_analysis
        assert "depth_analysis" in liquidity_analysis
        assert "liquidity_score" in liquidity_analysis
        # Проверка типов данных
        assert isinstance(liquidity_analysis["bid_liquidity"], dict)
        assert isinstance(liquidity_analysis["ask_liquidity"], dict)
        assert isinstance(liquidity_analysis["spread_analysis"], dict)
        assert isinstance(liquidity_analysis["depth_analysis"], dict)
        assert isinstance(liquidity_analysis["liquidity_score"], float)
        # Проверка диапазона
        assert 0.0 <= liquidity_analysis["liquidity_score"] <= 1.0
    def test_analyze_market_volatility(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа рыночной волатильности."""
        # Анализ волатильности
        volatility_analysis = market_data_processor.analyze_market_volatility(sample_market_data)
        # Проверки
        assert volatility_analysis is not None
        assert "historical_volatility" in volatility_analysis
        assert "implied_volatility" in volatility_analysis
        assert "volatility_regime" in volatility_analysis
        assert "volatility_forecast" in volatility_analysis
        # Проверка типов данных
        assert isinstance(volatility_analysis["historical_volatility"], float)
        assert isinstance(volatility_analysis["implied_volatility"], float)
        assert isinstance(volatility_analysis["volatility_regime"], str)
        assert isinstance(volatility_analysis["volatility_forecast"], dict)
    def test_analyze_market_microstructure(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа рыночной микроструктуры."""
        # Анализ микроструктуры
        microstructure_analysis = market_data_processor.analyze_market_microstructure(sample_market_data)
        # Проверки
        assert microstructure_analysis is not None
        assert "bid_ask_spread" in microstructure_analysis
        assert "order_flow" in microstructure_analysis
        assert "market_impact" in microstructure_analysis
        assert "price_discovery" in microstructure_analysis
        # Проверка типов данных
        assert isinstance(microstructure_analysis["bid_ask_spread"], dict)
        assert isinstance(microstructure_analysis["order_flow"], dict)
        assert isinstance(microstructure_analysis["market_impact"], dict)
        assert isinstance(microstructure_analysis["price_discovery"], dict)
    def test_calculate_market_regime(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета рыночного режима."""
        # Расчет рыночного режима
        market_regime = market_data_processor.calculate_market_regime(sample_market_data)
        # Проверки
        assert market_regime is not None
        assert "regime_type" in market_regime
        assert "regime_confidence" in market_regime
        assert "regime_metrics" in market_regime
        assert "regime_transitions" in market_regime
        # Проверка типов данных
        assert market_regime["regime_type"] in ["trending", "ranging", "volatile", "stable"]
        assert isinstance(market_regime["regime_confidence"], float)
        assert isinstance(market_regime["regime_metrics"], dict)
        assert isinstance(market_regime["regime_transitions"], list)
        # Проверка диапазона
        assert 0.0 <= market_regime["regime_confidence"] <= 1.0
    def test_validate_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест валидации рыночных данных."""
        # Валидация данных
        validation_result = market_data_processor.validate_market_data(sample_market_data)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "data_quality_score" in validation_result
        assert "validation_errors" in validation_result
        assert "data_completeness" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["data_quality_score"], float)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["data_completeness"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["data_quality_score"] <= 1.0
        assert 0.0 <= validation_result["data_completeness"] <= 1.0
    def test_clean_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест очистки рыночных данных."""
        # Добавление грязных данных
        dirty_data: pd.DataFrame = sample_market_data.copy(deep=True)
        if callable(dirty_data):
            dirty_data = dirty_data()
        if hasattr(dirty_data, 'loc') and hasattr(dirty_data.index, '__getitem__') and len(dirty_data.index) > 1:
            dirty_data.loc[dirty_data.index[0], 'close'] = np.nan
            dirty_data.loc[dirty_data.index[1], 'volume'] = -1000
        # Очистка данных
        cleaned_data = market_data_processor.clean_market_data(dirty_data)
        # Проверки
        assert cleaned_data is not None
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) <= len(dirty_data)
        # Проверка, что нет некорректных значений
        assert not cleaned_data.isna().any().any()
        assert (cleaned_data['volume'] >= 0).all()
    def test_resample_market_data(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест передискретизации рыночных данных."""
        # Передискретизация данных
        resampled_data = market_data_processor.resample_market_data(sample_market_data, freq='4H')
        # Проверки
        assert resampled_data is not None
        assert isinstance(resampled_data, pd.DataFrame)
        assert len(resampled_data) != len(sample_market_data)
        # Проверка, что индекс имеет правильную частоту
        time_diff = resampled_data.index[1] - resampled_data.index[0]
        assert time_diff == timedelta(hours=4)
    def test_get_market_statistics(self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест получения рыночной статистики."""
        # Получение статистики
        statistics = market_data_processor.get_market_statistics(sample_market_data)
        # Проверки
        assert statistics is not None
        assert isinstance(statistics, dict)
        assert len(statistics) > 0
        # Проверка наличия основных статистик
        expected_stats = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
        for stat in expected_stats:
            assert stat in statistics
    def test_error_handling(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест обработки ошибок."""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            market_data_processor.process_ohlcv_data(empty_data)
        with pytest.raises(ValueError):
            market_data_processor.normalize_market_data(empty_data)
    def test_edge_cases(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        # Эти функции должны обрабатывать короткие данные
        processed_data = market_data_processor.process_ohlcv_data(short_data)
        assert len(processed_data) == len(short_data)
        # Тест с NaN значениями
        data_with_nan: pd.DataFrame = short_data.copy(deep=True)
        if callable(data_with_nan):
            data_with_nan = data_with_nan()
        if hasattr(data_with_nan, 'loc') and hasattr(data_with_nan.index, '__getitem__') and len(data_with_nan.index) > 1:
            data_with_nan.loc[data_with_nan.index[1], 'close'] = np.nan
        # Функции должны обрабатывать NaN значения
        cleaned_data = market_data_processor.clean_market_data(data_with_nan)
        assert len(cleaned_data) <= len(data_with_nan)
    def test_cleanup(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        market_data_processor.cleanup()
        # Проверка, что ресурсы освобождены
        assert market_data_processor.data_processors == {}
        assert market_data_processor.data_validators == {}
        assert market_data_processor.data_analyzers == {} 
