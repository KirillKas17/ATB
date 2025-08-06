#!/usr/bin/env python3
"""
Простой тест для анализа покрытия базовых модулей
"""
import pytest
from decimal import Decimal
from datetime import datetime


def test_basic_types():
    """Тест базовых типов данных"""
    price = Decimal('50000.00')
    assert price > 0
    
    volume = Decimal('1.5')
    assert volume > 0
    
    timestamp = datetime.now()
    assert timestamp is not None


def test_basic_calculations():
    """Тест базовых вычислений"""
    price = Decimal('50000.00')
    volume = Decimal('0.1')
    total = price * volume
    
    assert total == Decimal('5000.00')


def test_string_operations():
    """Тест строковых операций"""
    symbol = "BTC/USDT"
    assert "/" in symbol
    assert symbol.startswith("BTC")
    assert symbol.endswith("USDT")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])