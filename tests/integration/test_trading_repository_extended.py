"""
Интеграционные тесты для расширенного функционала торгового репозитория.
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from domain.entities.order import Order, OrderId, OrderSide, OrderStatus, OrderType
from domain.entities.trading import Trade
from domain.entities.account import Account, Balance
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.types import Symbol
from infrastructure.repositories.trading import InMemoryTradingRepository


class TestTradingRepositoryExtended:
    """Тесты расширенного функционала торгового репозитория."""

    @pytest.fixture
    async def repository(self) -> Any:
        """Создание репозитория для тестов."""
        repo = InMemoryTradingRepository()
        yield repo
        await repo.clear_all_data()

    @pytest.fixture
    def sample_order(self) -> Any:
        """Создание тестового ордера."""
        return Order(
            id=OrderId(UUID("12345678-1234-5678-9abc-123456789abc")),
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.1")),
            price=Price(Decimal("50000")),
            status=OrderStatus.FILLED
        )

    @pytest.fixture
    def sample_trade(self, sample_order) -> Any:
        """Создание тестовой сделки."""
        return Trade(
            id=UUID("87654321-4321-8765-cba9-987654321cba"),
            order_id=sample_order.id,
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.BUY,
            quantity=Volume(Decimal("0.1")),
            price=Price(Decimal("50000")),
            timestamp=Timestamp(datetime.now())
        )

    @pytest.fixture
    def sample_account(self) -> Any:
        """Создание тестового аккаунта."""
        balances = [
            Balance("USDT", Decimal("10000"), Decimal("10000")),
            Balance("BTC", Decimal("1.5"), Decimal("1.5")),
            Balance("ETH", Decimal("10"), Decimal("10"))
        ]
        return Account(
            account_id="test_account_001",
            balances=balances
        )

    async def test_save_and_get_trade(self, repository, sample_trade) -> None:
        """Тест сохранения и получения сделки."""
        # Сохраняем сделку
        saved_trade = await repository.save_trade(sample_trade)
        assert saved_trade.id == sample_trade.id
        
        # Получаем сделку по ID
        retrieved_trade = repository._trades.get(str(sample_trade.id))
        assert retrieved_trade is not None
        assert retrieved_trade.symbol == sample_trade.symbol

    async def test_get_trades_by_order(self, repository, sample_order, sample_trade) -> None:
        """Тест получения сделок по ордеру."""
        # Сохраняем ордер и сделку
        await repository.save_order(sample_order)
        await repository.save_trade(sample_trade)
        
        # Получаем сделки по ордеру
        trades = await repository.get_trades_by_order(sample_order.id)
        assert len(trades) == 1
        assert trades[0].order_id == sample_order.id

    async def test_get_trades_by_symbol(self, repository, sample_trade) -> None:
        """Тест получения сделок по символу."""
        # Сохраняем сделку
        await repository.save_trade(sample_trade)
        
        # Получаем сделки по символу
        trades = await repository.get_trades_by_symbol(Symbol("BTCUSDT"))
        assert len(trades) == 1
        assert trades[0].symbol == Symbol("BTCUSDT")

    async def test_get_trade_with_filters(self, repository, sample_trade) -> None:
        """Тест получения сделок с фильтрами."""
        # Сохраняем сделку
        await repository.save_trade(sample_trade)
        
        # Получаем сделки с фильтром по символу
        trades = await repository.get_trade(symbol=Symbol("BTCUSDT"), limit=10)
        assert len(trades) == 1
        assert trades[0].symbol == Symbol("BTCUSDT")
        
        # Получаем все сделки с лимитом
        all_trades = await repository.get_trade(limit=5)
        assert len(all_trades) == 1

    async def test_save_and_get_account(self, repository, sample_account) -> None:
        """Тест сохранения и получения аккаунта."""
        # Сохраняем аккаунт
        saved_account = await repository.save_account(sample_account)
        assert saved_account.account_id == sample_account.account_id
        
        # Получаем аккаунт по ID
        retrieved_account = await repository.get_account(sample_account.account_id)
        assert retrieved_account is not None
        assert len(retrieved_account.balances) == 3

    async def test_get_balance(self, repository, sample_account) -> None:
        """Тест получения баланса аккаунта."""
        # Сохраняем аккаунт
        await repository.save_account(sample_account)
        
        # Получаем баланс
        balance = await repository.get_balance(sample_account.account_id)
        assert "USDT" in balance
        assert "BTC" in balance
        assert "ETH" in balance
        assert balance["USDT"].value == Decimal("10000")

    async def test_update_account_balance(self, repository, sample_account) -> None:
        """Тест обновления баланса аккаунта."""
        # Сохраняем аккаунт
        await repository.save_account(sample_account)
        
        # Обновляем баланс
        new_balance = Money(Decimal("15000"), Currency.USDT)
        success = await repository.update_account_balance(
            sample_account.account_id, "USDT", new_balance
        )
        assert success is True
        
        # Проверяем обновленный баланс
        balance = await repository.get_balance(sample_account.account_id)
        assert balance["USDT"].value == Decimal("15000")

    async def test_get_balance_default_account(self, repository) -> None:
        """Тест получения баланса по умолчанию."""
        # Получаем баланс без указания аккаунта
        balance = await repository.get_balance()
        assert "USDT" in balance
        assert "BTC" in balance
        assert "ETH" in balance
        # Все балансы должны быть нулевыми для нового аккаунта
        assert all(money.value == Decimal("0") for money in balance.values())

    async def test_trading_metrics_with_trades(self, repository, sample_trade) -> None:
        """Тест получения торговых метрик с учетом сделок."""
        # Сохраняем сделку
        await repository.save_trade(sample_trade)
        
        # Получаем метрики
        metrics_result = await repository.get_trading_metrics()
        assert metrics_result.success is True
        metrics = metrics_result.data
        
        assert metrics["total_trades"] == 1
        assert metrics["total_trade_volume"] > 0
        assert "trade_currencies" in metrics

    async def test_clear_all_data_includes_trades(self, repository, sample_trade, sample_account) -> None:
        """Тест очистки всех данных включая сделки и аккаунты."""
        # Сохраняем данные
        await repository.save_trade(sample_trade)
        await repository.save_account(sample_account)
        
        # Проверяем, что данные сохранены
        assert len(repository._trades) == 1
        assert len(repository._accounts) == 1
        
        # Очищаем все данные
        result = await repository.clear_all_data()
        assert result.success is True
        
        # Проверяем, что все данные очищены
        assert len(repository._trades) == 0
        assert len(repository._accounts) == 0
        assert len(repository._trades_by_symbol) == 0
        assert len(repository._trades_by_order) == 0
        assert len(repository._balance_cache) == 0

    async def test_multiple_trades_same_symbol(self, repository, sample_trade) -> None:
        """Тест множественных сделок по одному символу."""
        # Создаем вторую сделку
        second_trade = Trade(
            id=UUID("11111111-2222-3333-4444-555555555555"),
            order_id=sample_trade.order_id,
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.SELL,
            quantity=Volume(Decimal("0.05")),
            price=Price(Decimal("51000")),
            timestamp=Timestamp(datetime.now() + timedelta(hours=1))
        )
        
        # Сохраняем обе сделки
        await repository.save_trade(sample_trade)
        await repository.save_trade(second_trade)
        
        # Получаем сделки по символу
        trades = await repository.get_trades_by_symbol(Symbol("BTCUSDT"))
        assert len(trades) == 2
        
        # Проверяем сортировку по времени (новые сначала)
        assert trades[0].timestamp.value > trades[1].timestamp.value

    async def test_balance_cache_invalidation(self, repository, sample_account) -> None:
        """Тест инвалидации кэша баланса."""
        # Сохраняем аккаунт
        await repository.save_account(sample_account)
        
        # Получаем баланс (должен закэшироваться)
        balance1 = await repository.get_balance(sample_account.account_id)
        assert "USDT" in balance1
        
        # Обновляем баланс
        new_balance = Money(Decimal("20000"), Currency.USDT)
        await repository.update_account_balance(
            sample_account.account_id, "USDT", new_balance
        )
        
        # Получаем баланс снова (должен быть обновлен)
        balance2 = await repository.get_balance(sample_account.account_id)
        assert balance2["USDT"].value == Decimal("20000") 
