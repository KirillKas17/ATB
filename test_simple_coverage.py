#!/usr/bin/env python3
"""
Простой тест для анализа покрытия базовых модулей
"""
import pytest
from decimal import Decimal
from datetime import datetime
from typing import List


def test_basic_types() -> None:
    """Тест базовых типов данных"""
    price: Decimal = Decimal('50000.00')
    assert price > 0
    
    volume: Decimal = Decimal('1.5')
    assert volume > 0
    
    timestamp: datetime = datetime.now()
    assert timestamp is not None


def test_basic_calculations() -> None:
    """Тест базовых вычислений"""
    price: Decimal = Decimal('50000.00')
    volume: Decimal = Decimal('0.1')
    total: Decimal = price * volume
    
    assert total == Decimal('5000.00')


def test_string_operations() -> None:
    """Тест строковых операций"""
    symbol: str = "BTC/USDT"
    assert "/" in symbol
    assert symbol.startswith("BTC")
    assert symbol.endswith("USDT")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])