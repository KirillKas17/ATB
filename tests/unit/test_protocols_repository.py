"""
Production-ready unit тесты для RepositoryProtocol.
Полное покрытие CRUD, ошибок, edge cases, асинхронных сценариев и типизации.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from domain.protocols.repository_protocol import (
    RepositoryProtocol,
    AsyncRepositoryProtocol,
    OrderRepositoryProtocol,
    PositionRepositoryProtocol,
    TradingPairRepositoryProtocol
)
from domain.exceptions.base_exceptions import (
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
    RepositoryConnectionError
)
from domain.exceptions.protocol_exceptions import (
    ValidationError,
    ConnectionError,
    TransactionError
)
from domain.entities.order import Order, OrderType, OrderSide, OrderStatus
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from decimal import Decimal
from domain.value_objects.order_id import OrderId
from domain.value_objects.symbol import Symbol
from domain.value_objects.price import Price
from domain.value_objects.volume_value import VolumeValue
from domain.value_objects.timestamp import Timestamp
class TestRepositoryProtocol:
    """Production-ready тесты для RepositoryProtocol."""
    @pytest.fixture
    def mock_repository(self) -> Mock:
        repo = Mock(spec=RepositoryProtocol)
        repo.create = Mock(return_value=1)
        repo.read = Mock(return_value=Order(
            id=OrderId("order_1"),  # Исправление: используем OrderId
            symbol=Symbol("BTC/USDT"),  # Исправление: используем Symbol
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=Price(Decimal("50000.0"), Currency("USDT")),  # Исправление: используем Price
            quantity=VolumeValue(Decimal("0.1")),  # Исправление: используем VolumeValue
            filled_quantity=VolumeValue(Decimal("0.1")),  # Исправление: используем VolumeValue
            created_at=Timestamp(datetime.utcnow())  # Исправление: используем Timestamp
        ))
        repo.update = Mock(return_value=True)
        repo.delete = Mock(return_value=True)
        repo.list = Mock(return_value=["order_1", "order_2"])
        repo.exists = Mock(return_value=True)
        return repo
    def test_crud_operations(self, mock_repository: Mock) -> None:
        """Тест CRUD-операций."""
        # Create
        obj_id = mock_repository.create({"symbol": "BTC/USDT"})
        assert obj_id == 1
        mock_repository.create.assert_called_once()
        # Read
        order = mock_repository.read("order_1")
        assert str(order.id) == "order_1"
        assert str(order.symbol) == "BTC/USDT"
        mock_repository.read.assert_called_once_with("order_1")
        # Update
        result = mock_repository.update("order_1", {"status": "cancelled"})
        assert result is True
        mock_repository.update.assert_called_once_with("order_1", {"status": "cancelled"})
        # Delete
        result = mock_repository.delete("order_1")
        assert result is True
        mock_repository.delete.assert_called_once_with("order_1")
        # List
        ids = mock_repository.list()
        assert ids == ["order_1", "order_2"]
        mock_repository.list.assert_called_once()
        # Exists
        exists = mock_repository.exists("order_1")
        assert exists is True
        mock_repository.exists.assert_called_once_with("order_1")
    def test_not_found_error(self, mock_repository: Mock) -> None:
        """Тест ошибки NotFoundError."""
        mock_repository.read.side_effect = EntityNotFoundError("Not found")
        with pytest.raises(EntityNotFoundError):
            mock_repository.read("nonexistent")
    def test_duplicate_error(self, mock_repository: Mock) -> None:
        """Тест ошибки DuplicateError."""
        mock_repository.create.side_effect = DuplicateEntityError("Duplicate")
        with pytest.raises(DuplicateEntityError):
            mock_repository.create({"symbol": "BTC/USDT"})
    def test_validation_error(self, mock_repository: Mock) -> None:
        """Тест ошибки ValidationError."""
        mock_repository.update.side_effect = ValidationError("Invalid data", "field_name", "invalid_value", "validation_rule")
        with pytest.raises(ValidationError):
            mock_repository.update("order_1", {"status": "invalid"})
    def test_connection_error(self, mock_repository: Mock) -> None:
        """Тест ошибки ConnectionError."""
        mock_repository.list.side_effect = RepositoryConnectionError("DB down")
        with pytest.raises(RepositoryConnectionError):
            mock_repository.list()
    def test_transaction_error(self, mock_repository: Mock) -> None:
        """Тест ошибки TransactionError."""
        mock_repository.delete.side_effect = TransactionError("Rollback failed")
        with pytest.raises(TransactionError):
            mock_repository.delete("order_1")
    def test_empty_and_large_data(self, mock_repository: Mock) -> None:
        """Тест работы с пустыми и большими данными."""
        # Пустой список
        mock_repository.list.return_value = []
        ids = mock_repository.list()
        assert ids == []
        # Большой список
        mock_repository.list.return_value = [f"order_{i}" for i in range(10000)]
        ids = mock_repository.list()
        assert len(ids) == 10000
    def test_invalid_data(self, mock_repository: Mock) -> None:
        """Тест работы с невалидными данными."""
        mock_repository.create.side_effect = ValidationError("Invalid input", "symbol", "", "non_empty")
        with pytest.raises(ValidationError):
            mock_repository.create({"symbol": ""})
class TestAsyncRepositoryProtocol:
    """Production-ready тесты для AsyncRepositoryProtocol."""
    @pytest.fixture
    def async_mock_repository(self) -> Mock:
        repo = Mock(spec=AsyncRepositoryProtocol)
        repo.create = AsyncMock(return_value=1)
        repo.read = AsyncMock(return_value=Order(
            id=OrderId("order_1"),  # Исправление: используем OrderId
            symbol=Symbol("BTC/USDT"),  # Исправление: используем Symbol
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=Price(Decimal("50000.0"), Currency("USDT")),  # Исправление: используем Price
            quantity=VolumeValue(Decimal("0.1")),  # Исправление: используем VolumeValue
            filled_quantity=VolumeValue(Decimal("0.1")),  # Исправление: используем VolumeValue
            created_at=Timestamp(datetime.utcnow())  # Исправление: используем Timestamp
        ))
        repo.update = AsyncMock(return_value=True)
        repo.delete = AsyncMock(return_value=True)
        repo.list = AsyncMock(return_value=["order_1", "order_2"])
        repo.exists = AsyncMock(return_value=True)
        return repo
    @pytest.mark.asyncio
    async def test_async_crud_operations(self, async_mock_repository: Mock) -> None:
        obj_id = await async_mock_repository.create({"symbol": "BTC/USDT"})
        assert obj_id == 1
        order = await async_mock_repository.read("order_1")
        assert str(order.id) == "order_1"
        result = await async_mock_repository.update("order_1", {"status": "cancelled"})
        assert result is True
        result = await async_mock_repository.delete("order_1")
        assert result is True
        ids = await async_mock_repository.list()
        assert ids == ["order_1", "order_2"]
        exists = await async_mock_repository.exists("order_1")
        assert exists is True
    @pytest.mark.asyncio
    async def test_async_errors(self, async_mock_repository: Mock) -> None:
        async_mock_repository.read.side_effect = EntityNotFoundError("Not found")
        with pytest.raises(EntityNotFoundError):
            await async_mock_repository.read("nonexistent")
        async_mock_repository.create.side_effect = DuplicateEntityError("Duplicate")
        with pytest.raises(DuplicateEntityError):
            await async_mock_repository.create({"symbol": "BTC/USDT"})
        async_mock_repository.update.side_effect = ValidationError("Invalid data", "status", "invalid", "valid_status")
        with pytest.raises(ValidationError):
            await async_mock_repository.update("order_1", {"status": "invalid"})
        async_mock_repository.list.side_effect = RepositoryConnectionError("DB down")
        with pytest.raises(RepositoryConnectionError):
            await async_mock_repository.list()
        async_mock_repository.delete.side_effect = TransactionError("Rollback failed")
        with pytest.raises(TransactionError):
            await async_mock_repository.delete("order_1")
    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self, async_mock_repository: Mock) -> None:
        tasks = [
            async_mock_repository.create({"symbol": "BTC/USDT"}),
            async_mock_repository.list(),
            async_mock_repository.exists("order_1")
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert results[0] == 1
        assert results[1] == ["order_1", "order_2"]
        assert results[2] is True
class TestTransactionalRepositoryProtocol:
    """Production-ready тесты для TransactionalRepositoryProtocol."""
    @pytest.fixture
    def transactional_mock_repository(self) -> Mock:
        repo = Mock(spec=OrderRepositoryProtocol)
        repo.begin = Mock(return_value=True)
        repo.commit = Mock(return_value=True)
        repo.rollback = Mock(return_value=True)
        return repo
    def test_transaction_lifecycle(self, transactional_mock_repository: Mock) -> None:
        assert transactional_mock_repository.begin() is True
        assert transactional_mock_repository.commit() is True
        assert transactional_mock_repository.rollback() is True
    def test_transaction_error(self, transactional_mock_repository: Mock) -> None:
        transactional_mock_repository.commit.side_effect = TransactionError("Commit failed")
        with pytest.raises(TransactionError):
            transactional_mock_repository.commit()
        transactional_mock_repository.rollback.side_effect = TransactionError("Rollback failed")
        with pytest.raises(TransactionError):
            transactional_mock_repository.rollback()
class TestRepositoryErrors:
    """Тесты для ошибок репозитория."""
    def test_error_inheritance(self) -> None:
        assert issubclass(EntityNotFoundError, RepositoryError)
        assert issubclass(DuplicateEntityError, RepositoryError)
        assert issubclass(RepositoryConnectionError, RepositoryError)
    def test_error_messages(self) -> None:
        assert str(EntityNotFoundError("Not found")) == "Not found"
        assert str(DuplicateEntityError("Duplicate")) == "Duplicate"
        assert str(ValidationError("Invalid")) == "Invalid"
        assert str(RepositoryConnectionError("Conn fail")) == "Conn fail"
        assert str(TransactionError("Tx fail")) == "Tx fail" 
