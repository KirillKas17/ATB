"""Тесты для классификатора фаз рынка."""
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.symbols.market_phase_classifier import MarketPhaseClassifier, PhaseDetectionConfig
from domain.types import MarketPhase, ValidationError, MarketDataFrame
def test_market_phase_classifier_init() -> None:
    clf = MarketPhaseClassifier()
    assert clf is not None
    assert clf.config is not None
def test_market_phase_classifier_classify_minimal() -> None:
    clf = MarketPhaseClassifier()
    # Минимальный DataFrame с нужными колонками
    df = pd.DataFrame({
        'open': [1, 2, 3, 4, 5]*5,
        'high': [2, 3, 4, 5, 6]*5,
        'low': [0, 1, 2, 3, 4]*5,
        'close': [1.5, 2.5, 3.5, 4.5, 5.5]*5,
        'volume': [10, 20, 30, 40, 50]*5
    })
    market_data = cast(MarketDataFrame, df)
    result = clf.classify_market_phase(market_data)
    assert result['phase'] is not None
    assert 'confidence' in result
    assert 'indicators' in result
    assert 'metadata' in result
def test_market_phase_classifier_invalid_data() -> None:
    clf = MarketPhaseClassifier()
    # DataFrame без нужных колонок
    df = pd.DataFrame({'foo': [1, 2, 3]})
    market_data = cast(MarketDataFrame, df)
    result = clf.classify_market_phase(market_data)
    # Должен вернуть результат с ошибкой, а не выбросить исключение
    assert result['phase'] == MarketPhase.NO_STRUCTURE
    assert result['confidence'] == 0.0
    assert 'error' in result['metadata']
def test_phase_detection_config_validation() -> None:
    # Валидная конфигурация
    config = PhaseDetectionConfig(
        atr_period=14,
        vwap_period=20,
        volume_period=20,
        entropy_period=50
    )
    assert config.atr_period == 14
    # Невалидная конфигурация
    with pytest.raises(Exception):
        PhaseDetectionConfig(atr_period=-1) 
