"""
Тесты для выбора символов.
"""

import pytest
from unittest.mock import Mock
from typing import Any

from application.symbol_selection.analytics import SymbolAnalytics
from application.symbol_selection.cache import SymbolCache
from application.symbol_selection.types import DOASSConfig


class TestSymbolAnalytics:
    """Тесты для SymbolAnalytics."""

    @pytest.fixture
    def analytics(self) -> SymbolAnalytics:
        """Создает экземпляр аналитики."""
        return SymbolAnalytics()

    @pytest.fixture
    def sample_market_data(self) -> list[dict[str, Any]]:
        """Создает образец рыночных данных."""
        return [
            {"timestamp": "2024-01-01T00:00:00", "close": "50000", "volume": "1000"},
            {"timestamp": "2024-01-01T01:00:00", "close": "51000", "volume": "1200"},
            {"timestamp": "2024-01-01T02:00:00", "close": "52000", "volume": "1100"}
        ]

    def test_add_advanced_analytics(self, analytics: SymbolAnalytics) -> None:
        """Тест добавления продвинутой аналитики."""
        from application.symbol_selection.types import SymbolSelectionResult
        from domain.symbols import SymbolProfile
        from domain.type_definitions import Symbol
        
        # Создаем тестовые данные
        result = SymbolSelectionResult()
        profiles = [
            SymbolProfile(symbol=Symbol("BTC/USDT")),
            SymbolProfile(symbol=Symbol("ETH/USDT")),
            SymbolProfile(symbol=Symbol("ADA/USDT"))
        ]
        
        analytics.add_advanced_analytics(result, profiles)
        
        assert hasattr(result, 'correlation_matrix')
        assert hasattr(result, 'entanglement_groups')
        assert hasattr(result, 'pattern_memory_insights')
        assert hasattr(result, 'session_alignment_scores')
        assert hasattr(result, 'liquidity_gravity_scores')
        assert hasattr(result, 'reversal_probabilities')

    def test_calculate_opportunity_score_from_metrics(self, analytics: SymbolAnalytics) -> None:
        """Тест расчета opportunity score из метрик."""
        from domain.type_definitions.symbol_types import MarketPhase
        
        metrics = {
            "volatility": 0.02,
            "volume": 1000000,
            "spread": 0.001
        }
        
        score = analytics.calculate_opportunity_score_from_metrics(metrics, MarketPhase.ACCUMULATION)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_get_market_data_for_phase(self, analytics: SymbolAnalytics) -> None:
        """Тест получения рыночных данных для анализа фазы."""
        import pandas as pd
        
        data = analytics.get_market_data_for_phase("BTC/USDT")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "timestamp" in data.columns
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns


class TestSymbolCache:
    """Тесты для SymbolCache."""

    @pytest.fixture
    def config(self) -> DOASSConfig:
        """Создает конфигурацию для кэша."""
        return DOASSConfig(
            update_interval_seconds=300,
            cache_ttl_seconds=3600,
            max_cache_size=1000
        )

    @pytest.fixture
    def cache(self, config: DOASSConfig) -> SymbolCache:
        """Создает экземпляр кэша."""
        return SymbolCache(config)

    def test_should_update(self, cache: SymbolCache) -> None:
        """Тест проверки необходимости обновления."""
        # Первый вызов должен вернуть True
        assert cache.should_update() is True

    def test_update_cache(self, cache: SymbolCache) -> None:
        """Тест обновления кэша."""
        from domain.symbols import SymbolProfile
        from domain.type_definitions import Symbol
        
        profiles = {
            "BTC/USDT": SymbolProfile(symbol=Symbol("BTC/USDT")),
            "ETH/USDT": SymbolProfile(symbol=Symbol("ETH/USDT"))
        }
        
        cache.update_cache(profiles)
        assert not cache.should_update()

    def test_get_cached_profile(self, cache: SymbolCache) -> None:
        """Тест получения кэшированного профиля."""
        from domain.symbols import SymbolProfile
        from domain.type_definitions import Symbol
        
        symbol = "BTC/USDT"
        profile = SymbolProfile(symbol=Symbol(symbol))
        profiles = {symbol: profile}
        
        cache.update_cache(profiles)
        cached_profile = cache.get_cached_profile(symbol)
        assert cached_profile is not None
        assert cached_profile.symbol == Symbol(symbol)

    def test_calculate_cache_hit_rate(self, cache: SymbolCache) -> None:
        """Тест расчета hit rate кэша."""
        hit_rate = cache.calculate_cache_hit_rate()
        assert isinstance(hit_rate, float)
        assert 0 <= hit_rate <= 1

    def test_update_performance_metrics(self, cache: SymbolCache) -> None:
        """Тест обновления метрик производительности."""
        cache.update_performance_metrics(1.5, 10)
        metrics = cache.get_performance_metrics()
        assert "last_processing_time" in metrics
        assert "last_symbols_count" in metrics
        assert "avg_processing_time" in metrics
        assert "total_processing_count" in metrics

    def test_get_cache_stats(self, cache: SymbolCache) -> None:
        """Тест получения статистики кэша."""
        stats = cache.get_cache_stats()
        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert "hit_rate" in stats
        assert "cache_size_mb" in stats

    def test_get_cached_profiles(self, cache: SymbolCache) -> None:
        """Тест получения всех кэшированных профилей."""
        from domain.symbols import SymbolProfile
        from domain.type_definitions import Symbol
        
        profiles = {
            "BTC/USDT": SymbolProfile(symbol=Symbol("BTC/USDT")),
            "ETH/USDT": SymbolProfile(symbol=Symbol("ETH/USDT"))
        }
        
        cache.update_cache(profiles)
        cached_profiles = cache.get_cached_profiles()
        assert isinstance(cached_profiles, dict)
        assert len(cached_profiles) == 2

    def test_get_hit_rate(self, cache: SymbolCache) -> None:
        """Тест получения hit rate кэша."""
        hit_rate = cache.get_hit_rate()
        assert isinstance(hit_rate, float)
        assert 0 <= hit_rate <= 1

    def test_get_stats(self, cache: SymbolCache) -> None:
        """Тест получения статистики кэша."""
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats 
