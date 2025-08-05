"""
Тесты для советника стратегий application слоя.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Any, Dict, List, Tuple
import pandas as pd

from application.strategy_advisor.mirror_map_builder import MirrorMapBuilder

# Определяем недостающие классы для тестов
class MirrorMap:
    def __init__(self, correlations: Dict, lags: Dict, strengths: Dict, recommendations: List) -> Any:
        self.correlations = correlations
        self.lags = lags
        self.strengths = strengths
        self.recommendations = recommendations

class TestMirrorMapBuilder:
    """Тесты для MirrorMapBuilder."""
    
    @pytest.fixture
    def mock_repositories(self) -> Tuple[Mock, Mock, Mock]:
        """Создает mock репозитории."""
        market_repo = Mock()
        pattern_repo = Mock()
        strategy_repo = Mock()
        market_repo.get_market_data = AsyncMock()
        market_repo.get_correlation_data = AsyncMock()
        pattern_repo.get_patterns = AsyncMock()
        pattern_repo.get_pattern_history = AsyncMock()
        strategy_repo.get_strategies = AsyncMock()
        strategy_repo.get_strategy_performance = AsyncMock()
        return market_repo, pattern_repo, strategy_repo
    
    @pytest.fixture
    def builder(self, mock_repositories: Tuple[Mock, Mock, Mock]) -> MirrorMapBuilder:
        """Создает экземпляр построителя."""
        market_repo, pattern_repo, strategy_repo = mock_repositories
        return MirrorMapBuilder()
    
    @pytest.fixture
    def sample_market_data(self) -> List[Dict[str, Any]]:
        """Создает образец рыночных данных."""
        return [
            {"timestamp": "2024-01-01T00:00:00", "close": "50000", "volume": "1000"},
            {"timestamp": "2024-01-01T01:00:00", "close": "51000", "volume": "1200"},
            {"timestamp": "2024-01-01T02:00:00", "close": "52000", "volume": "1100"}
        ]
    
    def test_mirror_map_builder_creation(self: "TestMirrorMapBuilder") -> None:
        """Тест создания MirrorMapBuilder."""
        builder = MirrorMapBuilder()
        assert builder is not None
        assert isinstance(builder, MirrorMapBuilder)

    def test_build_mirror_map(self: "TestMirrorMapBuilder") -> None:
        """Тест построения карты зеркальных зависимостей."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2020, 2040, 2060, 2080])
        }
        
        # Строим карту
        mirror_map = builder.build_mirror_map(price_data)
        
        assert mirror_map is not None
        assert isinstance(mirror_map, MirrorMap)

    def test_analyze_correlations(self: "TestMirrorMapBuilder") -> None:
        """Тест анализа корреляций."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2020, 2040, 2060, 2080])
        }
        
        # Анализируем корреляции
        correlations = builder.analyze_correlations(price_data)
        
        assert correlations is not None
        assert isinstance(correlations, dict)

    def test_analyze_pattern_mirrors(self: "TestMirrorMapBuilder") -> None:
        """Тест анализа паттернов зеркальных зависимостей."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2020, 2040, 2060, 2080])
        }
        
        # Анализируем паттерны
        patterns = builder.analyze_pattern_mirrors(price_data)
        
        assert patterns is not None
        assert isinstance(patterns, dict)

    def test_analyze_strategy_mirrors(self: "TestMirrorMapBuilder") -> None:
        """Тест анализа стратегических зеркальных зависимостей."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2020, 2040, 2060, 2080])
        }
        
        # Анализируем стратегии
        strategies = builder.analyze_strategy_mirrors(price_data)
        
        assert strategies is not None
        assert isinstance(strategies, dict)

    def test_group_mirrors(self: "TestMirrorMapBuilder") -> None:
        """Тест группировки зеркальных зависимостей."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        correlations = {
            ("BTC", "ETH"): 0.8,
            ("BTC", "ADA"): 0.6,
            ("ETH", "ADA"): 0.7
        }
        
        # Группируем зеркала
        groups = builder.group_mirrors(correlations)
        
        assert groups is not None
        assert isinstance(groups, dict)

    def test_generate_recommendations(self: "TestMirrorMapBuilder") -> None:
        """Тест генерации рекомендаций."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        mirror_map = MirrorMap(
            correlations={},
            lags={},
            strengths={},
            recommendations=[]
        )
        
        # Генерируем рекомендации
        recommendations = builder.generate_recommendations(mirror_map)
        
        assert recommendations is not None
        assert isinstance(recommendations, list)

    def test_calculate_mirror_strength(self: "TestMirrorMapBuilder") -> None:
        """Тест расчета силы зеркальной зависимости."""
        builder = MirrorMapBuilder()
        
        # Тестируем расчет силы
        strength = builder.calculate_mirror_strength(0.8, 2)
        
        assert strength is not None
        assert isinstance(strength, float)
        assert 0 <= strength <= 1

    def test_find_strong_correlations(self: "TestMirrorMapBuilder") -> None:
        """Тест поиска сильных корреляций."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        correlations = {
            ("BTC", "ETH"): 0.8,
            ("BTC", "ADA"): 0.6,
            ("ETH", "ADA"): 0.7
        }
        
        # Ищем сильные корреляции
        strong_correlations = builder.find_strong_correlations(correlations, threshold=0.7)
        
        assert strong_correlations is not None
        assert isinstance(strong_correlations, dict)

    def test_calculate_overall_strength(self: "TestMirrorMapBuilder") -> None:
        """Тест расчета общей силы."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        correlations = {
            ("BTC", "ETH"): 0.8,
            ("BTC", "ADA"): 0.6
        }
        
        # Рассчитываем общую силу
        overall_strength = builder.calculate_overall_strength(correlations)
        
        assert overall_strength is not None
        assert isinstance(overall_strength, float)

    def test_get_mirror_map_statistics(self: "TestMirrorMapBuilder") -> None:
        """Тест получения статистики карты зеркальных зависимостей."""
        builder = MirrorMapBuilder()
        
        # Создаем тестовые данные
        price_data = {
            "BTC": pd.Series([100, 101, 102, 103, 104]),
            "ETH": pd.Series([2000, 2020, 2040, 2060, 2080])
        }
        
        # Получаем статистику
        stats = builder.get_mirror_map_statistics(price_data)
        
        assert stats is not None
        assert isinstance(stats, dict)

    def test_validate_symbols(self: "TestMirrorMapBuilder") -> None:
        """Тест валидации символов."""
        builder = MirrorMapBuilder()
        
        # Тестируем валидные символы
        valid_symbols = ["BTC", "ETH", "ADA"]
        empty_symbols: List[str] = []
        
        # Валидируем символы
        result1 = builder.validate_symbols(valid_symbols)
        result2 = builder.validate_symbols(empty_symbols)
        
        assert result1 is True
        assert result2 is False

    def test_calculate_confidence_score(self: "TestMirrorMapBuilder") -> None:
        """Тест расчета оценки уверенности."""
        builder = MirrorMapBuilder()
        
        # Рассчитываем оценку уверенности
        confidence = builder.calculate_confidence_score(0.8, 100)
        
        assert confidence is not None
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1 
