"""
Интеграционные тесты для стратегий.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from domain.entities.market import MarketData, MarketState
from domain.type_definitions import StrategyId, TradingPair, ConfidenceLevel, RiskLevel
from domain.strategies.strategy_factory import StrategyFactory, get_strategy_factory
from domain.strategies.strategy_registry import StrategyRegistry, get_strategy_registry
from domain.strategies.base_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    ArbitrageStrategy
)
from domain.strategies.strategy_types import (
    StrategyCategory, RiskProfile, Timeframe, StrategyConfig,
    TrendFollowingParams, MeanReversionParams, BreakoutParams,
    ScalpingParams, ArbitrageParams
)
from domain.strategies.exceptions import (
    StrategyFactoryError, StrategyCreationError, StrategyValidationError,
    StrategyRegistryError, StrategyNotFoundError, StrategyDuplicateError
)
from domain.strategies.utils import StrategyUtils
from domain.strategies.validators import StrategyValidator
class TestStrategyFactoryIntegration:
    """Интеграционные тесты для фабрики стратегий."""
    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return StrategyFactory()
        # Регистрируем стратегии
        # Создаем стратегию
        # Попытка создания с некорректными параметрами
        # Регистрируем несколько стратегий
class TestStrategyRegistryIntegration:
    """Интеграционные тесты для реестра стратегий."""
    @pytest.fixture
    def registry(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать реестр стратегий."""
        return StrategyRegistry()
        # Регистрируем стратегию
        # Получаем стратегию
        # Получаем метаданные
        # Создаем и регистрируем несколько стратегий
        # Активируем некоторые стратегии
        # Тестируем поиск по типу
        # Тестируем поиск по статусу
        # Тестируем поиск по торговой паре
        # Тестируем поиск по тегу
        # Тестируем комплексный поиск
        # Обновляем производительность
        # Проверяем статистику
        # Создаем и регистрируем стратегии с разными статусами
            # Устанавливаем разные статусы
class TestStrategyWorkflowIntegration:
    """Интеграционные тесты для полного workflow стратегий."""
    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return get_strategy_factory()
        # 1. Регистрируем стратегию в фабрике
        # 2. Создаем стратегию через фабрику
        # 3. Регистрируем стратегию в реестре
        # 4. Активируем стратегию
        # 5. Обрабатываем рыночные данные и генерируем сигналы
                    # Обновляем производительность
        # 6. Проверяем результаты
        # 7. Получаем статистику
        # Регистрируем разные типы стратегий
            # Регистрируем в фабрике
            # Создаем стратегию
            # Регистрируем в реестре
        # Активируем все стратегии
        # Тестируем генерацию сигналов для всех стратегий
            # Берем последние данные для тестирования
                # Обновляем производительность
        # Проверяем статистику
        # Проверяем распределение по типам
        # Создаем стратегию с некорректными параметрами
        # Попытка создания с ошибкой
        # Создаем корректную стратегию
        # Симулируем ошибку выполнения
        # Проверяем, что ошибка зафиксирована
        # Проверяем статистику ошибок
class TestStrategyUtilsIntegration:
    """Интеграционные тесты для утилит стратегий."""
    @pytest.fixture
    def utils(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать утилиты стратегий."""
        return StrategyUtils()
        # Тестируем валидацию конфигурации
        # Тестируем валидацию с ошибками
        # Создаем тестовые данные
        # Тестируем расчет индикаторов
        # Тестируем анализ тренда
        # Тестируем анализ волатильности
