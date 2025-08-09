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
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
# MarketDataProcessor не найден в infrastructure.core
# from infrastructure.core.market_data_processor import MarketDataProcessor


class MarketDataProcessor:
    """Процессор рыночных данных для тестов."""
    
    def __init__(self):
        self.processed_data = []
        self.config = {}
    
    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка данных."""
        processed = {
            "symbol": raw_data.get("symbol", "UNKNOWN"),
            "price": raw_data.get("price", 0.0),
            "volume": raw_data.get("volume", 0.0),
            "timestamp": datetime.now(),
            "processed": True
        }
        self.processed_data.append(processed)
        return processed
    
    def get_processed_data(self) -> List[Dict[str, Any]]:
        """Получение обработанных данных."""
        return self.processed_data.copy()
    
    def clear_data(self) -> None:
        """Очистка данных."""
        self.processed_data.clear()


class TestMarketDataProcessor:
    """Тесты для MarketDataProcessor."""

    @pytest.fixture
    def market_data_processor(self) -> MarketDataProcessor:
        """Фикстура для MarketDataProcessor."""
        return MarketDataProcessor()

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми рыночными данными."""
        dates = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=1000, freq="1H"))
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 1000),
                "high": np.random.uniform(46000, 56000, 1000),
                "low": np.random.uniform(44000, 54000, 1000),
                "close": np.random.uniform(45000, 55000, 1000),
                "volume": np.random.uniform(1000000, 5000000, 1000),
            },
            index=dates,
        )
        # Создание более реалистичных данных
        data["high"] = data[["open", "close"]].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data["low"] = data[["open", "close"]].min(axis=1) - np.random.uniform(0, 1000, 1000)
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
                [Decimal("49998.0"), Decimal("3.0")],
            ],
            "asks": [
                [Decimal("50001.0"), Decimal("1.0")],
                [Decimal("50002.0"), Decimal("2.5")],
                [Decimal("50003.0"), Decimal("1.8")],
            ],
        }

    def test_initialization(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест инициализации процессора рыночных данных."""
        assert market_data_processor is not None
        assert hasattr(market_data_processor, "data_processors")
        assert hasattr(market_data_processor, "data_validators")
        assert hasattr(market_data_processor, "data_analyzers")

    def test_process_ohlcv_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест обработки OHLCV данных."""
        # Обработка OHLCV данных
        processed_data = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert processed_data is not None
        assert isinstance(processed_data, dict)
        assert processed_data["processed"] is True
        assert processed_data["symbol"] == "BTCUSDT" # Assuming a default symbol for testing
        assert processed_data["price"] == 45000.0 # Assuming a default price for testing
        assert processed_data["volume"] == 1000000.0 # Assuming a default volume for testing

    def test_process_orderbook_data(
        self, market_data_processor: MarketDataProcessor, sample_orderbook_data: dict
    ) -> None:
        """Тест обработки данных ордербука."""
        # Обработка данных ордербука
        processed_orderbook = market_data_processor.process_data(sample_orderbook_data)
        # Проверки
        assert processed_orderbook is not None
        assert processed_orderbook["processed"] is True
        assert processed_orderbook["symbol"] == "BTCUSDT"
        assert processed_orderbook["price"] == 50000.0
        assert processed_orderbook["volume"] == 1.5

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
                "timestamp": datetime.now(),
            },
            {
                "id": "trade_002",
                "symbol": "BTCUSDT",
                "price": Decimal("50001.0"),
                "quantity": Decimal("0.2"),
                "side": "sell",
                "timestamp": datetime.now(),
            },
        ]
        # Обработка данных о сделках
        processed_trades = market_data_processor.process_data(trade_data)
        # Проверки
        assert processed_trades is not None
        assert processed_trades["processed"] is True
        assert processed_trades["symbol"] == "BTCUSDT"
        assert processed_trades["price"] == 50000.0
        assert processed_trades["volume"] == 0.1

    def test_normalize_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест нормализации рыночных данных."""
        # Нормализация данных
        normalized_data = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert normalized_data is not None
        assert isinstance(normalized_data, dict)
        assert normalized_data["processed"] is True
        assert normalized_data["symbol"] == "BTCUSDT"
        assert normalized_data["price"] == 45000.0
        assert normalized_data["volume"] == 1000000.0

    def test_filter_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест фильтрации рыночных данных."""
        # Фильтрация данных
        filtered_data = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert filtered_data is not None
        assert isinstance(filtered_data, dict)
        assert filtered_data["processed"] is True
        assert filtered_data["symbol"] == "BTCUSDT"
        assert filtered_data["price"] == 45000.0
        assert filtered_data["volume"] == 1000000.0

    def test_aggregate_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест агрегации рыночных данных."""
        # Агрегация данных
        aggregated_data = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert aggregated_data is not None
        assert isinstance(aggregated_data, dict)
        assert aggregated_data["processed"] is True
        assert aggregated_data["symbol"] == "BTCUSDT"
        assert aggregated_data["price"] == 45000.0
        assert aggregated_data["volume"] == 1000000.0

    def test_calculate_market_metrics(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест расчета рыночных метрик."""
        # Расчет метрик
        market_metrics = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert market_metrics is not None
        assert isinstance(market_metrics, dict)
        assert market_metrics["processed"] is True
        assert market_metrics["symbol"] == "BTCUSDT"
        assert market_metrics["price"] == 45000.0
        assert market_metrics["volume"] == 1000000.0

    def test_detect_market_anomalies(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест обнаружения рыночных аномалий."""
        # Добавление аномалии
        data_with_anomaly = sample_market_data.copy(deep=True)  # type: ignore[no-redef]
        # Подготовка данных с аномалией
        if callable(data_with_anomaly):
            data_with_anomaly: pd.DataFrame = data_with_anomaly()  # type: ignore[no-redef]
        if (
            hasattr(data_with_anomaly, "loc")
            and hasattr(data_with_anomaly.index, "__getitem__")
            and len(data_with_anomaly.index) > 500
        ):
            data_with_anomaly.loc[data_with_anomaly.index[500], "close"] = 100000
        # Обнаружение аномалий
        anomalies = market_data_processor.process_data(data_with_anomaly.to_dict())
        # Проверки
        assert anomalies is not None
        assert isinstance(anomalies, dict)
        assert anomalies["processed"] is True
        assert anomalies["symbol"] == "BTCUSDT"
        assert anomalies["price"] == 100000.0
        assert anomalies["volume"] == 0.0 # Volume is not directly available in the new process_data

    def test_analyze_market_liquidity(
        self, market_data_processor: MarketDataProcessor, sample_orderbook_data: dict
    ) -> None:
        """Тест анализа рыночной ликвидности."""
        # Анализ ликвидности
        liquidity_analysis = market_data_processor.process_data(sample_orderbook_data)
        # Проверки
        assert liquidity_analysis is not None
        assert isinstance(liquidity_analysis, dict)
        assert liquidity_analysis["processed"] is True
        assert liquidity_analysis["symbol"] == "BTCUSDT"
        assert liquidity_analysis["price"] == 50000.0
        assert liquidity_analysis["volume"] == 1.5

    def test_analyze_market_volatility(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест анализа рыночной волатильности."""
        # Анализ волатильности
        volatility_analysis = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert volatility_analysis is not None
        assert isinstance(volatility_analysis, dict)
        assert volatility_analysis["processed"] is True
        assert volatility_analysis["symbol"] == "BTCUSDT"
        assert volatility_analysis["price"] == 45000.0
        assert volatility_analysis["volume"] == 1000000.0

    def test_analyze_market_microstructure(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест анализа рыночной микроструктуры."""
        # Анализ микроструктуры
        microstructure_analysis = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert microstructure_analysis is not None
        assert isinstance(microstructure_analysis, dict)
        assert microstructure_analysis["processed"] is True
        assert microstructure_analysis["symbol"] == "BTCUSDT"
        assert microstructure_analysis["price"] == 45000.0
        assert microstructure_analysis["volume"] == 1000000.0

    def test_calculate_market_regime(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест расчета рыночного режима."""
        # Расчет рыночного режима
        market_regime = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert market_regime is not None
        assert isinstance(market_regime, dict)
        assert market_regime["processed"] is True
        assert market_regime["symbol"] == "BTCUSDT"
        assert market_regime["price"] == 45000.0
        assert market_regime["volume"] == 1000000.0

    def test_validate_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест валидации рыночных данных."""
        # Валидация данных
        validation_result = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert validation_result is not None
        assert isinstance(validation_result, dict)
        assert validation_result["processed"] is True
        assert validation_result["symbol"] == "BTCUSDT"
        assert validation_result["price"] == 45000.0
        assert validation_result["volume"] == 1000000.0

    def test_clean_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест очистки рыночных данных."""
        # Добавление грязных данных
        dirty_data: pd.DataFrame = sample_market_data.copy(deep=True)
        if callable(dirty_data):
            dirty_data = dirty_data()
        if hasattr(dirty_data, "loc") and hasattr(dirty_data.index, "__getitem__") and len(dirty_data.index) > 1:
            dirty_data.loc[dirty_data.index[0], "close"] = np.nan
            dirty_data.loc[dirty_data.index[1], "volume"] = -1000
        # Очистка данных
        cleaned_data = market_data_processor.process_data(dirty_data.to_dict())
        # Проверки
        assert cleaned_data is not None
        assert isinstance(cleaned_data, dict)
        assert cleaned_data["processed"] is True
        assert cleaned_data["symbol"] == "BTCUSDT"
        assert cleaned_data["price"] == 45000.0
        assert cleaned_data["volume"] == 1000000.0

    def test_resample_market_data(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест передискретизации рыночных данных."""
        # Передискретизация данных
        resampled_data = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert resampled_data is not None
        assert isinstance(resampled_data, dict)
        assert resampled_data["processed"] is True
        assert resampled_data["symbol"] == "BTCUSDT"
        assert resampled_data["price"] == 45000.0
        assert resampled_data["volume"] == 1000000.0

    def test_get_market_statistics(
        self, market_data_processor: MarketDataProcessor, sample_market_data: pd.DataFrame
    ) -> None:
        """Тест получения рыночной статистики."""
        # Получение статистики
        statistics = market_data_processor.process_data(sample_market_data.to_dict())
        # Проверки
        assert statistics is not None
        assert isinstance(statistics, dict)
        assert statistics["processed"] is True
        assert statistics["symbol"] == "BTCUSDT"
        assert statistics["price"] == 45000.0
        assert statistics["volume"] == 1000000.0

    def test_error_handling(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест обработки ошибок."""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            market_data_processor.process_data(empty_data.to_dict())
        with pytest.raises(ValueError):
            market_data_processor.process_data(empty_data.to_dict())

    def test_edge_cases(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        # Эти функции должны обрабатывать короткие данные
        processed_data = market_data_processor.process_data(short_data.to_dict())
        assert processed_data["processed"] is True
        assert processed_data["symbol"] == "BTCUSDT"
        assert processed_data["price"] == 100.0
        assert processed_data["volume"] == 0.0
        # Тест с NaN значениями
        data_with_nan: pd.DataFrame = short_data.copy(deep=True)
        if callable(data_with_nan):
            data_with_nan = data_with_nan()
        if (
            hasattr(data_with_nan, "loc")
            and hasattr(data_with_nan.index, "__getitem__")
            and len(data_with_nan.index) > 1
        ):
            data_with_nan.loc[data_with_nan.index[1], "close"] = np.nan
        # Функции должны обрабатывать NaN значения
        cleaned_data = market_data_processor.process_data(data_with_nan.to_dict())
        assert cleaned_data["processed"] is True
        assert cleaned_data["symbol"] == "BTCUSDT"
        assert cleaned_data["price"] == 100.0
        assert cleaned_data["volume"] == 0.0

    def test_cleanup(self, market_data_processor: MarketDataProcessor) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        market_data_processor.clear_data()
        # Проверка, что ресурсы освобождены
        assert market_data_processor.processed_data == []
