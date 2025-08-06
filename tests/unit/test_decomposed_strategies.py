"""
Тесты для декомпозированных стратегий
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from unittest.mock import patch
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy
from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
from infrastructure.strategies.evolution.evolvable_base_strategy import EvolvableBaseStrategy
from domain.type_definitions.strategy_types import StrategyType, MarketRegime
class TestDecomposedStrategies:
    """Тесты для декомпозированных стратегий"""
    @pytest.fixture
    def sample_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        return None
        """Создание корректных тестовых данных"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        # Создание корректных OHLCV данных
        base_price = 100
        data = []
        for i in range(100):
            # Создание корректных цен (high >= max(open, close), low <= min(open, close))
            open_price = base_price + np.random.uniform(-5, 5)
            close_price = base_price + np.random.uniform(-5, 5)
            high_price = max(open_price, close_price) + np.random.uniform(0, 3)
            low_price = min(open_price, close_price) - np.random.uniform(0, 3)
            volume = np.random.uniform(1000, 10000)
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            base_price = close_price
        return pd.DataFrame(data, index=dates)
        # Создаем очень простые данные для теста - только проверяем, что стратегия инициализируется
        # Проверяем, что стратегия может валидировать данные
        # Сигнал может быть None или объектом Signal
        # Сигнал может быть None или объектом Signal
        # Настройка моков
        # Настройка моков
        # Сигнал может быть None или объектом Signal
        # Сигнал может быть None или объектом Signal
        # Пустые данные
        # Данные без необходимых колонок
        # Проверяем, что все стратегии могут выполнить анализ
        # Проверяем консистентность метаданных
        # Тест с None данными
        # Тест с некорректными данными
