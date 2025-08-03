"""
Unit тесты для TechnicalAnalysisService.

Покрывает:
- Основной функционал технического анализа
- Расчет индикаторов
- Анализ рыночных данных
- Обработку ошибок
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from domain.services.technical_analysis import (
    TechnicalAnalysisService,
    DefaultTechnicalAnalysisService,
    IndicatorConfig,
    TechnicalAnalysisResult,
    MACDData,
    VolumeProfileData,
    MarketStructureData,
    IndicatorType
)
from domain.entities.market import MarketData, TechnicalIndicator
from domain.types.technical_types import (
    BollingerBandsResult,
    MarketStructure,
    MarketStructureResult,
    TrendStrength,
    VolumeProfileResult
)


class TestIndicatorConfig:
    """Тесты для IndicatorConfig."""
    
    def test_creation_default(self):
        """Тест создания с значениями по умолчанию."""
        config = IndicatorConfig()
        
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.atr_period == 14
        assert config.volume_ma_period == 20
        assert config.fractal_period == 2
        assert config.cluster_window == 20
        assert config.volatility_window == 20
        assert config.volume_profile_bins == 24
        assert config.use_cache is True
        assert config.parallel_processing is True
        assert config.max_workers == 4
    
    def test_creation_custom(self):
        """Тест создания с пользовательскими значениями."""
        config = IndicatorConfig(
            rsi_period=21,
            macd_fast=8,
            macd_slow=21,
            bb_period=30,
            bb_std=2.5,
            use_cache=False,
            parallel_processing=False
        )
        
        assert config.rsi_period == 21
        assert config.macd_fast == 8
        assert config.macd_slow == 21
        assert config.bb_period == 30
        assert config.bb_std == 2.5
        assert config.use_cache is False
        assert config.parallel_processing is False


class TestTechnicalAnalysisService:
    """Тесты для TechnicalAnalysisService."""
    
    @pytest.fixture
    def service(self) -> TechnicalAnalysisService:
        """Экземпляр TechnicalAnalysisService."""
        return TechnicalAnalysisService()
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Тестовые данные."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 300, 100),
            'low': np.random.uniform(50, 100, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Создаем тренд
        data['close'] = data['close'] + np.arange(100) * 0.1
        data['high'] = data['high'] + np.arange(100) * 0.1
        data['low'] = data['low'] + np.arange(100) * 0.1
        return data
    
    def test_creation(self):
        """Тест создания сервиса."""
        service = TechnicalAnalysisService()
        
        assert service is not None
        assert hasattr(service, 'calculate_sma')
        assert hasattr(service, 'calculate_ema')
        assert hasattr(service, 'calculate_rsi')
        assert hasattr(service, 'calculate_macd')
    
    def test_calculate_sma(self, service, sample_data):
        """Тест расчета Simple Moving Average."""
        prices = sample_data['close']
        period = 20
        
        sma = service.calculate_sma(prices, period)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(prices)
        assert sma.index.equals(prices.index)
        assert not sma.isna().all()  # Не все значения NaN
    
    def test_calculate_sma_short_period(self, service, sample_data):
        """Тест расчета SMA с коротким периодом."""
        prices = sample_data['close']
        period = 5
        
        sma = service.calculate_sma(prices, period)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(prices)
        assert not sma.isna().all()
    
    def test_calculate_sma_long_period(self, service, sample_data):
        """Тест расчета SMA с длинным периодом."""
        prices = sample_data['close']
        period = 50
        
        sma = service.calculate_sma(prices, period)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(prices)
        # Больше NaN значений в начале из-за длинного периода
        assert sma.isna().sum() > 0
    
    def test_calculate_ema(self, service, sample_data):
        """Тест расчета Exponential Moving Average."""
        prices = sample_data['close']
        period = 20
        
        ema = service.calculate_ema(prices, period)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(prices)
        assert ema.index.equals(prices.index)
        assert not ema.isna().all()
    
    def test_calculate_rsi(self, service, sample_data):
        """Тест расчета RSI."""
        prices = sample_data['close']
        period = 14
        
        rsi = service.calculate_rsi(prices, period)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        assert rsi.index.equals(prices.index)
        # RSI должен быть в диапазоне [0, 100]
        assert rsi.min() >= 0
        assert rsi.max() <= 100
    
    def test_calculate_rsi_default_period(self, service, sample_data):
        """Тест расчета RSI с периодом по умолчанию."""
        prices = sample_data['close']
        
        rsi = service.calculate_rsi(prices)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        assert rsi.min() >= 0
        assert rsi.max() <= 100
    
    def test_calculate_macd(self, service, sample_data):
        """Тест расчета MACD."""
        prices = sample_data['close']
        fast = 12
        slow = 26
        signal = 9
        
        macd_result = service.calculate_macd(prices, fast, slow, signal)
        
        assert isinstance(macd_result, dict)
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        macd = macd_result['macd']
        signal_line = macd_result['signal']
        histogram = macd_result['histogram']
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)
    
    def test_calculate_macd_default_params(self, service, sample_data):
        """Тест расчета MACD с параметрами по умолчанию."""
        prices = sample_data['close']
        
        macd_result = service.calculate_macd(prices)
        
        assert isinstance(macd_result, dict)
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
    
    def test_calculate_bollinger_bands(self, service, sample_data):
        """Тест расчета полос Боллинджера."""
        prices = sample_data['close']
        period = 20
        std_dev = 2.0
        
        bb_result = service.calculate_bollinger_bands(prices, period, std_dev)
        
        assert isinstance(bb_result, BollingerBandsResult)
        assert isinstance(bb_result.upper, pd.Series)
        assert isinstance(bb_result.middle, pd.Series)
        assert isinstance(bb_result.lower, pd.Series)
        assert len(bb_result.upper) == len(prices)
        assert len(bb_result.middle) == len(prices)
        assert len(bb_result.lower) == len(prices)
        
        # Проверяем логику полос Боллинджера
        assert (bb_result.upper >= bb_result.middle).all()
        assert (bb_result.middle >= bb_result.lower).all()
    
    def test_calculate_bollinger_bands_default_params(self, service, sample_data):
        """Тест расчета полос Боллинджера с параметрами по умолчанию."""
        prices = sample_data['close']
        
        bb_result = service.calculate_bollinger_bands(prices)
        
        assert isinstance(bb_result, BollingerBandsResult)
        assert isinstance(bb_result.upper, pd.Series)
        assert isinstance(bb_result.middle, pd.Series)
        assert isinstance(bb_result.lower, pd.Series)
    
    def test_calculate_atr(self, service, sample_data):
        """Тест расчета Average True Range."""
        period = 14
        
        atr = service.calculate_atr(sample_data, period)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)
        assert atr.index.equals(sample_data.index)
        assert (atr >= 0).all()  # ATR всегда положительный
    
    def test_calculate_atr_default_period(self, service, sample_data):
        """Тест расчета ATR с периодом по умолчанию."""
        atr = service.calculate_atr(sample_data)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)
        assert (atr >= 0).all()
    
    def test_calculate_stochastic(self, service, sample_data):
        """Тест расчета Stochastic Oscillator."""
        k_period = 14
        d_period = 3
        
        stoch_result = service.calculate_stochastic(sample_data, k_period, d_period)
        
        assert isinstance(stoch_result, dict)
        assert 'k' in stoch_result
        assert 'd' in stoch_result
        
        k = stoch_result['k']
        d = stoch_result['d']
        
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert len(k) == len(sample_data)
        assert len(d) == len(sample_data)
        # Stochastic должен быть в диапазоне [0, 100]
        assert k.min() >= 0
        assert k.max() <= 100
        assert d.min() >= 0
        assert d.max() <= 100
    
    def test_calculate_stochastic_default_params(self, service, sample_data):
        """Тест расчета Stochastic с параметрами по умолчанию."""
        stoch_result = service.calculate_stochastic(sample_data)
        
        assert isinstance(stoch_result, dict)
        assert 'k' in stoch_result
        assert 'd' in stoch_result
    
    def test_calculate_williams_r(self, service, sample_data):
        """Тест расчета Williams %R."""
        period = 14
        
        williams_r = service.calculate_williams_r(sample_data, period)
        
        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(sample_data)
        assert williams_r.index.equals(sample_data.index)
        # Williams %R должен быть в диапазоне [-100, 0]
        assert williams_r.min() >= -100
        assert williams_r.max() <= 0
    
    def test_calculate_williams_r_default_period(self, service, sample_data):
        """Тест расчета Williams %R с периодом по умолчанию."""
        williams_r = service.calculate_williams_r(sample_data)
        
        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(sample_data)
        assert williams_r.min() >= -100
        assert williams_r.max() <= 0
    
    def test_calculate_cci(self, service, sample_data):
        """Тест расчета Commodity Channel Index."""
        period = 20
        
        cci = service.calculate_cci(sample_data, period)
        
        assert isinstance(cci, pd.Series)
        assert len(cci) == len(sample_data)
        assert cci.index.equals(sample_data.index)
    
    def test_calculate_cci_default_period(self, service, sample_data):
        """Тест расчета CCI с периодом по умолчанию."""
        cci = service.calculate_cci(sample_data)
        
        assert isinstance(cci, pd.Series)
        assert len(cci) == len(sample_data)
    
    def test_analyze_market_data(self, service):
        """Тест анализа рыночных данных."""
        # Создаем тестовые MarketData
        market_data = [
            MarketData(
                symbol="BTC/USD",
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0
            ),
            MarketData(
                symbol="BTC/USD",
                timestamp=datetime.now(),
                open=102.0,
                high=108.0,
                low=100.0,
                close=106.0,
                volume=1200.0
            )
        ]
        
        indicators = ["RSI", "MACD", "BB"]
        
        result = service.analyze_market_data(market_data, indicators)
        
        assert isinstance(result, TechnicalAnalysisResult)
        assert result.symbol == "BTC/USD"
        assert isinstance(result.indicators, dict)
        assert len(result.indicators) > 0
    
    def test_extract_numeric_value(self, service):
        """Тест извлечения числового значения."""
        # Тест с float
        assert service._extract_numeric_value(10.5) == 10.5
        
        # Тест с int
        assert service._extract_numeric_value(10) == 10.0
        
        # Тест с Decimal
        from decimal import Decimal
        assert service._extract_numeric_value(Decimal("10.5")) == 10.5
        
        # Тест с None
        assert service._extract_numeric_value(None) == 0.0
        
        # Тест с невалидным значением
        assert service._extract_numeric_value("invalid") == 0.0


class TestDefaultTechnicalAnalysisService:
    """Тесты для DefaultTechnicalAnalysisService."""
    
    @pytest.fixture
    def config(self) -> IndicatorConfig:
        """Тестовая конфигурация."""
        return IndicatorConfig(
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            bb_period=20,
            use_cache=True
        )
    
    @pytest.fixture
    def service(self, config) -> DefaultTechnicalAnalysisService:
        """Экземпляр DefaultTechnicalAnalysisService."""
        return DefaultTechnicalAnalysisService(config)
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Тестовые данные."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 300, 100),
            'low': np.random.uniform(50, 100, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Создаем тренд
        data['close'] = data['close'] + np.arange(100) * 0.1
        data['high'] = data['high'] + np.arange(100) * 0.1
        data['low'] = data['low'] + np.arange(100) * 0.1
        return data
    
    def test_creation(self, config):
        """Тест создания сервиса."""
        service = DefaultTechnicalAnalysisService(config)
        
        assert service.config == config
        assert service._cache is not None
    
    def test_creation_default_config(self):
        """Тест создания сервиса с конфигурацией по умолчанию."""
        service = DefaultTechnicalAnalysisService()
        
        assert service.config is not None
        assert isinstance(service.config, IndicatorConfig)
        assert service._cache is not None
    
    def test_validate_market_data_valid(self, service, sample_data):
        """Тест валидации валидных рыночных данных."""
        # Должно выполняться без ошибок
        service._validate_market_data(sample_data)
    
    def test_validate_market_data_empty(self, service):
        """Тест валидации пустых данных."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Empty market data"):
            service._validate_market_data(empty_data)
    
    def test_validate_market_data_missing_columns(self, service):
        """Тест валидации данных с отсутствующими колонками."""
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107]
            # Отсутствуют 'low', 'close', 'volume'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            service._validate_market_data(invalid_data)
    
    def test_validate_market_data_invalid_values(self, service):
        """Тест валидации данных с невалидными значениями."""
        invalid_data = pd.DataFrame({
            'open': [100, -50, 102],  # Отрицательное значение
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000, 2000, 3000]
        })
        
        with pytest.raises(ValueError, match="Invalid values in market data"):
            service._validate_market_data(invalid_data)
    
    def test_calculate_indicators(self, service, sample_data):
        """Тест расчета индикаторов."""
        result = service.calculate_indicators(sample_data)
        
        assert isinstance(result, TechnicalAnalysisResult)
        assert result.symbol == "UNKNOWN"
        assert result.timeframe == "1H"
        assert isinstance(result.indicators, dict)
        assert len(result.indicators) > 0
        assert isinstance(result.timestamp, datetime)
    
    def test_calculate_volume_profile(self, service, sample_data):
        """Тест расчета профиля объема."""
        bins = 24
        
        result = service.calculate_volume_profile(sample_data, bins)
        
        assert isinstance(result, VolumeProfileResult)
        assert result.poc is not None
        assert isinstance(result.value_area, list)
        assert isinstance(result.histogram, list)
        assert isinstance(result.bins, list)
        assert len(result.bins) == bins
    
    def test_calculate_volume_profile_default_bins(self, service, sample_data):
        """Тест расчета профиля объема с количеством бинов по умолчанию."""
        result = service.calculate_volume_profile(sample_data)
        
        assert isinstance(result, VolumeProfileResult)
        assert result.poc is not None
        assert len(result.bins) == service.config.volume_profile_bins
    
    def test_calculate_market_structure(self, service, sample_data):
        """Тест расчета структуры рынка."""
        result = service.calculate_market_structure(sample_data)
        
        assert isinstance(result, MarketStructureResult)
        assert isinstance(result.structure, MarketStructure)
        assert isinstance(result.trend_strength, float)
        assert isinstance(result.volatility, float)
        assert isinstance(result.adx, float)
        assert isinstance(result.rsi, float)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_find_support_levels(self, service, sample_data):
        """Тест поиска уровней поддержки."""
        levels = service._find_support_levels(sample_data)
        
        assert isinstance(levels, list)
        assert all(isinstance(level, float) for level in levels)
        assert all(level >= 0 for level in levels)
    
    def test_find_resistance_levels(self, service, sample_data):
        """Тест поиска уровней сопротивления."""
        levels = service._find_resistance_levels(sample_data)
        
        assert isinstance(levels, list)
        assert all(isinstance(level, float) for level in levels)
        assert all(level >= 0 for level in levels)
    
    def test_calculate_trend_indicators(self, service, sample_data):
        """Тест расчета трендовых индикаторов."""
        result = service._calculate_trend_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) > 0
    
    def test_calculate_volatility_momentum(self, service, sample_data):
        """Тест расчета индикаторов волатильности и импульса."""
        result = service._calculate_volatility_momentum(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) > 0
    
    def test_calculate_volume_indicators(self, service, sample_data):
        """Тест расчета объемных индикаторов."""
        result = service._calculate_volume_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) > 0
    
    def test_calculate_structure_indicators(self, service, sample_data):
        """Тест расчета структурных индикаторов."""
        result = service._calculate_structure_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) > 0
    
    def test_error_handling_none_data(self, service):
        """Тест обработки ошибок с None данными."""
        with pytest.raises(ValueError):
            service._validate_market_data(None)
    
    def test_error_handling_invalid_dataframe(self, service):
        """Тест обработки ошибок с невалидным DataFrame."""
        invalid_data = "not a dataframe"
        
        with pytest.raises(ValueError):
            service._validate_market_data(invalid_data)
    
    def test_concurrent_indicator_calculation(self, service, sample_data):
        """Тест конкурентного расчета индикаторов."""
        import threading
        import time
        
        results = []
        errors = []
        
        def calculate_indicators():
            try:
                result = service.calculate_indicators(sample_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=calculate_indicators) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Ошибки в потоках: {errors}"
        assert len(results) == 5
    
    def test_cache_functionality(self, service, sample_data):
        """Тест функциональности кэширования."""
        # Первый вызов
        result1 = service.calculate_indicators(sample_data)
        
        # Второй вызов (должен использовать кэш)
        result2 = service.calculate_indicators(sample_data)
        
        assert isinstance(result1, TechnicalAnalysisResult)
        assert isinstance(result2, TechnicalAnalysisResult)
        assert result1.symbol == result2.symbol
        assert result1.timeframe == result2.timeframe
    
    def test_different_timeframes(self, service):
        """Тест работы с разными таймфреймами."""
        # Создаем данные с разными таймфреймами
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for timeframe in timeframes:
            dates = pd.date_range('2023-01-01', periods=50, freq=timeframe)
            data = pd.DataFrame({
                'open': np.random.uniform(100, 200, 50),
                'high': np.random.uniform(200, 300, 50),
                'low': np.random.uniform(50, 100, 50),
                'close': np.random.uniform(100, 200, 50),
                'volume': np.random.uniform(1000, 10000, 50)
            }, index=dates)
            
            result = service.calculate_indicators(data)
            
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.timeframe == timeframe 