"""
Тесты для MarketConditionsAnalyzer.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from infrastructure.services.market_conditions_analyzer import (
    MarketConditionsAnalyzer,
    MarketConditionsConfig,
    MarketConditionType
)
from domain.entities.market import MarketData
from domain.value_objects.currency import Price, Volume
from domain.type_definitions.base_types import Symbol


class TestMarketConditionsAnalyzer:
    """Тесты для анализатора рыночных условий."""

    @pytest.fixture
    def mock_market_repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок репозитория рыночных данных."""
        repository = Mock()
        repository.get_market_data = AsyncMock()
        return repository

    @pytest.fixture
    def mock_technical_analysis_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок сервиса технического анализа."""
        service = Mock()
        service.perform_complete_analysis.return_value = {
            "indicators": {"rsi": 50.0, "macd": 0.0},
            "signals": []
        }
        service.analyze_market_structure.return_value = {
            "trend_strength": 0.5,
            "trend_direction": "neutral",
            "volatility": 0.02
        }
        return service

    @pytest.fixture
    def analyzer(self, mock_market_repository, mock_technical_analysis_service) -> Any:
        """Создание анализатора для тестов."""
        config = MarketConditionsConfig(
            short_window=5,
            medium_window=10,
            long_window=20,
            volatility_window=5,
            trend_window=10
        )
        return MarketConditionsAnalyzer(
            market_repository=mock_market_repository,
            technical_analysis_service=mock_technical_analysis_service,
            config=config
        )

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестовых рыночных данных."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
        np.random.seed(42)
        
        # Создаем реалистичные данные с трендом
        base_price = 50000.0
        trend = np.linspace(0, 0.1, 50)  # Восходящий тренд
        noise = np.random.normal(0, 0.01, 50)
        prices = base_price * (1 + trend + noise)
        
        data = {
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 50)
        }
        
        return pd.DataFrame(data).set_index('timestamp')

    def test_analyzer_initialization(self, analyzer) -> None:
        """Тест инициализации анализатора."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.market_repository is not None
        assert analyzer.technical_analysis_service is not None

    @pytest.mark.asyncio
    async def test_calculate_market_score_with_valid_data(
        self, analyzer, mock_market_repository, sample_market_data
    ) -> None:
        """Тест расчета скора с валидными данными."""
        # Настраиваем мок для возврата тестовых данных
        mock_market_repository.get_market_data.return_value = [
            self._create_market_data_entity(row, i) 
            for i, row in sample_market_data.iterrows()
        ]
        
        # Выполняем анализ
        result = await analyzer.calculate_market_score(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback_periods=50
        )
        
        # Проверяем результат
        assert result is not None
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'volatility_score')
        assert hasattr(result, 'trend_score')
        assert hasattr(result, 'volume_score')
        assert hasattr(result, 'momentum_score')
        assert hasattr(result, 'regime_score')
        assert hasattr(result, 'condition_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'timestamp')
        
        # Проверяем диапазоны значений
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.volatility_score <= 1.0
        assert 0.0 <= result.trend_score <= 1.0
        assert 0.0 <= result.volume_score <= 1.0
        assert 0.0 <= result.momentum_score <= 1.0
        assert 0.0 <= result.regime_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Проверяем тип условия
        assert isinstance(result.condition_type, MarketConditionType)

    @pytest.mark.asyncio
    async def test_calculate_market_score_with_empty_data(self, analyzer) -> None:
        """Тест расчета скора с пустыми данными."""
        # Настраиваем мок для возврата пустых данных
        analyzer.market_repository.get_market_data.return_value = []
        
        # Выполняем анализ
        result = await analyzer.calculate_market_score(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback_periods=50
        )
        
        # Проверяем, что возвращается скор по умолчанию
        assert result is not None
        assert result.overall_score == 0.5
        assert result.condition_type == MarketConditionType.NO_STRUCTURE

    @pytest.mark.asyncio
    async def test_analyze_volatility(self, analyzer, sample_market_data) -> None:
        """Тест анализа волатильности."""
        result = await analyzer._analyze_volatility(sample_market_data)
        
        assert result is not None
        assert "score" in result
        assert "volatility" in result
        assert "regime" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["volatility"] >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_trend(self, analyzer, sample_market_data) -> None:
        """Тест анализа тренда."""
        result = await analyzer._analyze_trend(sample_market_data)
        
        assert result is not None
        assert "score" in result
        assert "direction" in result
        assert "strength" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["direction"] in ["up", "down", "neutral"]

    @pytest.mark.asyncio
    async def test_analyze_volume(self, analyzer, sample_market_data) -> None:
        """Тест анализа объема."""
        result = await analyzer._analyze_volume(sample_market_data)
        
        assert result is not None
        assert "score" in result
        assert "profile" in result
        assert "trend" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["profile"] in ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_analyze_momentum(self, analyzer, sample_market_data) -> None:
        """Тест анализа моментума."""
        result = await analyzer._analyze_momentum(sample_market_data)
        
        assert result is not None
        assert "score" in result
        assert "direction" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["direction"] in ["positive", "negative", "mixed"]

    @pytest.mark.asyncio
    async def test_analyze_market_regime(self, analyzer, sample_market_data) -> None:
        """Тест анализа режима рынка."""
        result = await analyzer._analyze_market_regime(sample_market_data)
        
        assert result is not None
        assert "score" in result
        assert "regime" in result
        assert "stability" in result
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["stability"] <= 1.0

    def test_calculate_regime_stability(self, analyzer, sample_market_data) -> None:
        """Тест расчета стабильности режима."""
        result = analyzer._calculate_regime_stability(sample_market_data)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_determine_condition_type(self, analyzer) -> None:
        """Тест определения типа рыночных условий."""
        # Тестируем различные комбинации
        volatility_analysis = {"regime": "normal"}
        volume_analysis = {"profile": "normal"}
        regime_analysis = {"regime": "sideways_stable"}
        
        # Тест с восходящим трендом
        trend_analysis = {"direction": "up", "strength": 0.8}
        result = analyzer._determine_condition_type(
            volatility_analysis, trend_analysis, volume_analysis, regime_analysis
        )
        assert result in [MarketConditionType.BULL_TRENDING, MarketConditionType.BREAKOUT_UP]
        
        # Тест с нисходящим трендом
        trend_analysis = {"direction": "down", "strength": 0.8}
        result = analyzer._determine_condition_type(
            volatility_analysis, trend_analysis, volume_analysis, regime_analysis
        )
        assert result in [MarketConditionType.BEAR_TRENDING, MarketConditionType.BREAKOUT_DOWN]

    def test_calculate_confidence(self, analyzer) -> None:
        """Тест расчета уверенности."""
        volatility_analysis = {"stability_factor": 0.8}
        trend_analysis = {"r_squared": 0.7}
        volume_analysis = {}
        momentum_analysis = {}
        regime_analysis = {"stability": 0.6}
        
        result = analyzer._calculate_confidence(
            volatility_analysis, trend_analysis, volume_analysis, 
            momentum_analysis, regime_analysis
        )
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_cache_functionality(self, analyzer) -> None:
        """Тест функциональности кэширования."""
        # Проверяем, что кэш пуст изначально
        assert len(analyzer._score_cache) == 0
        assert len(analyzer._market_data_cache) == 0
        
        # Очищаем кэш
        analyzer.clear_cache()
        assert len(analyzer._score_cache) == 0
        assert len(analyzer._market_data_cache) == 0

    def test_get_analysis_stats(self, analyzer) -> None:
        """Тест получения статистики анализа."""
        stats = analyzer.get_analysis_stats()
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "avg_processing_time" in stats
        
        assert stats["total_analyses"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["avg_processing_time"] == 0.0

    def _create_market_data_entity(self, row, timestamp) -> Any:
        """Создание сущности рыночных данных для тестов."""
        return MarketData(
            timestamp=timestamp,
            open_price=Price(row['open']),
            high_price=Price(row['high']),
            low_price=Price(row['low']),
            close_price=Price(row['close']),
            volume=Volume(row['volume'])
        ) 