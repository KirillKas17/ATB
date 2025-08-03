#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Минимальные фикстуры для запуска unit тестов.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
import pytest


@pytest.fixture
def mock_account_data():
    """Мок данных для счета."""
    return {
        'account_id': 'test_account_001',
        'balance': Decimal('10000.00'),
        'currency': 'USDT',
        'created_at': datetime.now()
    }


@pytest.fixture
def mock_order_data():
    """Мок данных для ордера."""
    return {
        'order_id': str(uuid4()),
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'order_type': 'LIMIT',
        'quantity': Decimal('0.1'),
        'price': Decimal('50000.00'),
        'status': 'NEW'
    }


@pytest.fixture
def mock_position_data():
    """Мок данных для позиции."""
    return {
        'position_id': str(uuid4()),
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'size': Decimal('0.1'),
        'entry_price': Decimal('50000.00'),
        'unrealized_pnl': Decimal('0.00')
    }


@pytest.fixture
def event_loop():
    """Фикстура event loop для асинхронных тестов."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_exchange():
    """Мок биржи."""
    exchange = Mock()
    exchange.get_account_info = AsyncMock(return_value={'balance': '10000.00'})
    exchange.place_order = AsyncMock(return_value={'order_id': 'test_order_001'})
    exchange.cancel_order = AsyncMock(return_value=True)
    return exchange


@pytest.fixture
def mock_repository():
    """Мок репозитория."""
    repo = Mock()
    repo.save = AsyncMock(return_value=True)
    repo.find_by_id = AsyncMock(return_value=None)
    repo.find_all = AsyncMock(return_value=[])
    repo.delete = AsyncMock(return_value=True)
    return repo