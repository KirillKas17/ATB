"""Тесты для кэша символов."""
from typing import cast
from domain.symbols.cache import MemorySymbolCache
from domain.symbols.symbol_profile import SymbolProfile, VolumeProfile, PriceStructure, OrderBookMetricsData, PatternMetricsData, SessionMetricsData
from domain.type_definitions import MarketPhase, VolumeValue, PriceValue, SpreadValue, ConfidenceValue
def test_memory_symbol_cache_basic() -> None:
    cache = MemorySymbolCache(default_ttl=1)
    profile = SymbolProfile(
        symbol="BTCUSDT",
        opportunity_score=0.8,
        market_phase=MarketPhase.ACCUMULATION,
        confidence=ConfidenceValue(0.9),
        volume_profile=VolumeProfile(current_volume=VolumeValue(1000.0)),
        price_structure=PriceStructure(current_price=PriceValue(50000.0)),
        order_book_metrics=OrderBookMetricsData(bid_ask_spread=SpreadValue(0.001)),
        pattern_metrics=PatternMetricsData(),
        session_metrics=SessionMetricsData()
    )
    cache.set_profile("BTCUSDT", profile)
    retrieved = cache.get_profile("BTCUSDT")
    assert retrieved is not None
    retrieved_profile = cast(SymbolProfile, retrieved)
    assert retrieved_profile.symbol == "BTCUSDT"
    assert retrieved_profile.opportunity_score == 0.8
def test_memory_symbol_cache_expiry() -> None:
    import time
    cache = MemorySymbolCache(default_ttl=1)
    profile = SymbolProfile(
        symbol="ETHUSDT",
        opportunity_score=0.5,
        market_phase=MarketPhase.BREAKOUT_ACTIVE,
        confidence=ConfidenceValue(0.5),
        volume_profile=VolumeProfile(current_volume=VolumeValue(500.0)),
        price_structure=PriceStructure(current_price=PriceValue(3000.0)),
        order_book_metrics=OrderBookMetricsData(bid_ask_spread=SpreadValue(0.002)),
        pattern_metrics=PatternMetricsData(),
        session_metrics=SessionMetricsData()
    )
    cache.set_profile("ETHUSDT", profile)
    time.sleep(1.1)  # Ждем истечения TTL
    retrieved = cache.get_profile("ETHUSDT")
    assert retrieved is None
def test_memory_symbol_cache_invalid_type() -> None:
    cache = MemorySymbolCache()
    cache.set_profile("BTCUSDT", "invalid_profile")
    retrieved = cache.get_profile("BTCUSDT")
    assert retrieved is None 
