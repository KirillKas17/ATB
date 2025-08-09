#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Минимальные фикстуры для тестов - без проблемных импортов
"""
import pytest
from unittest.mock import Mock
from decimal import Decimal
from datetime import datetime


@pytest.fixture
def mock_exchange():
    """Мок биржи"""
    exchange = Mock()
    exchange.fetch_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49999.0,
        "ask": 50001.0,
        "volume": 1000.0,
        "timestamp": datetime.now().timestamp(),
    }
    return exchange


@pytest.fixture
def sample_price():
    """Образец цены"""
    return Decimal("50000.00")


@pytest.fixture
def sample_volume():
    """Образец объема"""
    return Decimal("1.5")


@pytest.fixture
def mock_repository():
    """Мок репозитория"""
    repo = Mock()
    repo.save.return_value = True
    repo.find_by_id.return_value = None
    return repo
