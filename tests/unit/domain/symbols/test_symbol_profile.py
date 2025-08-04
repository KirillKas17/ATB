"""Тесты для профиля символа."""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.symbols.symbol_profile import SymbolProfile, VolumeProfile, PriceStructure, OrderBookMetricsData, PatternMetricsData, SessionMetricsData
from domain.types import MarketPhase, VolumeValue, PriceValue, SpreadValue, ConfidenceValue
def test_symbol_profile_creation() -> None:
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
    assert profile.symbol == "BTCUSDT"
    assert profile.opportunity_score == 0.8
    assert profile.market_phase == MarketPhase.ACCUMULATION
    assert profile.confidence == ConfidenceValue(0.9)
def test_symbol_profile_edge_cases() -> None:
    # Минимальные значения
    profile = SymbolProfile(
        symbol="ETHUSDT",
        opportunity_score=0.0,
        market_phase=MarketPhase.NO_STRUCTURE,
        confidence=ConfidenceValue(0.0),
        volume_profile=VolumeProfile(current_volume=VolumeValue(0.0)),
        price_structure=PriceStructure(current_price=PriceValue(0.0)),
        order_book_metrics=OrderBookMetricsData(bid_ask_spread=SpreadValue(0.0)),
        pattern_metrics=PatternMetricsData(),
        session_metrics=SessionMetricsData()
    )
    assert profile.symbol == "ETHUSDT"
    assert profile.opportunity_score == 0.0
    assert profile.market_phase == MarketPhase.NO_STRUCTURE
    assert profile.confidence == ConfidenceValue(0.0)
def test_symbol_profile_invalid_types() -> None:
    with pytest.raises(Exception):
        SymbolProfile(
            symbol="",  # Пустой символ
            opportunity_score=0.8,
            market_phase=MarketPhase.ACCUMULATION,
            confidence=ConfidenceValue(0.9)
        )
    with pytest.raises(Exception):
        SymbolProfile(
            symbol="BTCUSDT",
            opportunity_score=1.5,  # > 1.0
            market_phase=MarketPhase.ACCUMULATION,
            confidence=ConfidenceValue(0.9)
        ) 
