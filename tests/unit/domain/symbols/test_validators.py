import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.symbols import SymbolValidator
    def test_symbol_validator_init() -> None:
    v = SymbolValidator()
    assert isinstance(v, SymbolValidator)
    assert hasattr(v, 'validate_symbol')
    def test_symbol_validator_valid_symbol() -> None:
    v = SymbolValidator()
    assert v.validate_symbol("BTCUSDT") is True
    assert v.validate_symbol("ETH/USDT") is True
    def test_symbol_validator_invalid_symbol() -> None:
    v = SymbolValidator()
    with pytest.raises(Exception):
        v.validate_symbol(123)
    with pytest.raises(Exception):
        v.validate_symbol("")
    with pytest.raises(Exception):
        v.validate_symbol("@BTC!")
    def test_symbol_validator_validate_ohlcv_data() -> None:
    v = SymbolValidator()
    df = pd.DataFrame({
        'open': [1, 2, 3, 4, 5]*5,
        'high': [2, 3, 4, 5, 6]*5,
        'low': [0, 1, 2, 3, 4]*5,
        'close': [1.5, 2.5, 3.5, 4.5, 5.5]*5,
        'volume': [10, 20, 30, 40, 50]*5
    })
    assert v.validate_ohlcv_data(df) is True
    def test_symbol_validator_validate_ohlcv_data_invalid() -> None:
    v = SymbolValidator()
    df = pd.DataFrame({'foo': [1, 2, 3]})
    with pytest.raises(Exception):
        v.validate_ohlcv_data(df) 
