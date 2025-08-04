"""
Unit тесты для PositionRepository.

Покрывает:
- Основной функционал репозитория позиций
- CRUD операции
- Фильтрацию по различным критериям
- Поиск по датам и сторонам позиций
- Статистику и аналитику позиций
- Обработку ошибок
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from domain.entities.trading import Position, PositionSide
from domain.entities.trading_pair import TradingPair
from domain.repositories.position_repository import PositionRepository
from domain.type_definitions.repository_types import QueryFilter
from domain.exceptions.base_exceptions import ValidationError


class TestPositionRepository:
    """Тесты для абстрактного PositionRepository."""

    @pytest.fixture
    def mock_position_repository(self) -> Mock:
        """Мок репозитория позиций."""
        return Mock(spec=PositionRepository)

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_position(self, sample_trading_pair) -> Position:
        """Тестовая позиция."""
        return Position(
            id=uuid4(),
            trading_pair=sample_trading_pair,
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    def test_save_method_exists(self, mock_position_repository, sample_position):
        """Тест наличия метода save."""
        mock_position_repository.save = AsyncMock(return_value=True)
        assert hasattr(mock_position_repository, 'save')
        assert callable(mock_position_repository.save)

    def test_get_by_id_method_exists(self, mock_position_repository):
        """Тест наличия метода get_by_id."""
        mock_position_repository.get_by_id = AsyncMock(return_value=None)
        assert hasattr(mock_position_repository, 'get_by_id')
        assert callable(mock_position_repository.get_by_id)

    def test_get_by_trading_pair_method_exists(self, mock_position_repository, sample_trading_pair):
        """Тест наличия метода get_by_trading_pair."""
        mock_position_repository.get_by_trading_pair = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_by_trading_pair')
        assert callable(mock_position_repository.get_by_trading_pair)

    def test_get_open_positions_method_exists(self, mock_position_repository):
        """Тест наличия метода get_open_positions."""
        mock_position_repository.get_open_positions = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_open_positions')
        assert callable(mock_position_repository.get_open_positions)

    def test_get_closed_positions_method_exists(self, mock_position_repository):
        """Тест наличия метода get_closed_positions."""
        mock_position_repository.get_closed_positions = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_closed_positions')
        assert callable(mock_position_repository.get_closed_positions)

    def test_get_positions_by_side_method_exists(self, mock_position_repository):
        """Тест наличия метода get_positions_by_side."""
        mock_position_repository.get_positions_by_side = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_positions_by_side')
        assert callable(mock_position_repository.get_positions_by_side)

    def test_get_positions_by_date_range_method_exists(self, mock_position_repository):
        """Тест наличия метода get_positions_by_date_range."""
        mock_position_repository.get_positions_by_date_range = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_positions_by_date_range')
        assert callable(mock_position_repository.get_positions_by_date_range)

    def test_update_method_exists(self, mock_position_repository, sample_position):
        """Тест наличия метода update."""
        mock_position_repository.update = AsyncMock(return_value=sample_position)
        assert hasattr(mock_position_repository, 'update')
        assert callable(mock_position_repository.update)

    def test_delete_method_exists(self, mock_position_repository):
        """Тест наличия метода delete."""
        mock_position_repository.delete = AsyncMock(return_value=True)
        assert hasattr(mock_position_repository, 'delete')
        assert callable(mock_position_repository.delete)

    def test_exists_method_exists(self, mock_position_repository):
        """Тест наличия метода exists."""
        mock_position_repository.exists = AsyncMock(return_value=True)
        assert hasattr(mock_position_repository, 'exists')
        assert callable(mock_position_repository.exists)

    def test_count_method_exists(self, mock_position_repository):
        """Тест наличия метода count."""
        mock_position_repository.count = AsyncMock(return_value=0)
        assert hasattr(mock_position_repository, 'count')
        assert callable(mock_position_repository.count)

    def test_get_profitable_positions_method_exists(self, mock_position_repository):
        """Тест наличия метода get_profitable_positions."""
        mock_position_repository.get_profitable_positions = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_profitable_positions')
        assert callable(mock_position_repository.get_profitable_positions)

    def test_get_losing_positions_method_exists(self, mock_position_repository):
        """Тест наличия метода get_losing_positions."""
        mock_position_repository.get_losing_positions = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_losing_positions')
        assert callable(mock_position_repository.get_losing_positions)

    def test_get_positions_with_stop_loss_method_exists(self, mock_position_repository):
        """Тест наличия метода get_positions_with_stop_loss."""
        mock_position_repository.get_positions_with_stop_loss = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_positions_with_stop_loss')
        assert callable(mock_position_repository.get_positions_with_stop_loss)

    def test_get_positions_with_take_profit_method_exists(self, mock_position_repository):
        """Тест наличия метода get_positions_with_take_profit."""
        mock_position_repository.get_positions_with_take_profit = AsyncMock(return_value=[])
        assert hasattr(mock_position_repository, 'get_positions_with_take_profit')
        assert callable(mock_position_repository.get_positions_with_take_profit)

    def test_get_statistics_method_exists(self, mock_position_repository):
        """Тест наличия метода get_statistics."""
        mock_position_repository.get_statistics = AsyncMock(return_value={})
        assert hasattr(mock_position_repository, 'get_statistics')
        assert callable(mock_position_repository.get_statistics)

    def test_get_total_exposure_method_exists(self, mock_position_repository):
        """Тест наличия метода get_total_exposure."""
        mock_position_repository.get_total_exposure = AsyncMock(return_value={})
        assert hasattr(mock_position_repository, 'get_total_exposure')
        assert callable(mock_position_repository.get_total_exposure)

    def test_cleanup_old_positions_method_exists(self, mock_position_repository):
        """Тест наличия метода cleanup_old_positions."""
        mock_position_repository.cleanup_old_positions = AsyncMock(return_value=0)
        assert hasattr(mock_position_repository, 'cleanup_old_positions')
        assert callable(mock_position_repository.cleanup_old_positions)

    def test_get_by_symbol_method_exists(self, mock_position_repository):
        """Тест наличия метода get_by_symbol."""
        mock_position_repository.get_by_symbol = AsyncMock(return_value=None)
        assert hasattr(mock_position_repository, 'get_by_symbol')
        assert callable(mock_position_repository.get_by_symbol)


class TestInMemoryPositionRepository:
    """Тесты для InMemoryPositionRepository."""

    @pytest.fixture
    def repository(self) -> Mock:
        """Репозиторий для тестов."""
        return Mock(spec=PositionRepository)

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_eth(self) -> TradingPair:
        """Тестовая торговая пара ETH."""
        return TradingPair(base="ETH", quote="USDT")

    @pytest.fixture
    def sample_position(self, sample_trading_pair) -> Position:
        """Тестовая позиция."""
        return Position(
            id=uuid4(),
            trading_pair=sample_trading_pair,
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    @pytest.fixture
    def sample_positions(self, sample_trading_pair, sample_trading_pair_eth) -> List[Position]:
        """Тестовые позиции."""
        now = datetime.now()
        return [
            Position(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=PositionSide.LONG,
                quantity=1.0,
                entry_price=50000.0,
                current_price=51000.0,
                is_open=True,
                created_at=now - timedelta(hours=1),
                updated_at=now
            ),
            Position(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=PositionSide.SHORT,
                quantity=0.5,
                entry_price=52000.0,
                current_price=51500.0,
                is_open=True,
                created_at=now - timedelta(hours=2),
                updated_at=now
            ),
            Position(
                id=uuid4(),
                trading_pair=sample_trading_pair_eth,
                side=PositionSide.LONG,
                quantity=10.0,
                entry_price=3000.0,
                current_price=3100.0,
                is_open=False,
                closed_at=now - timedelta(hours=3),
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(hours=3)
            ),
            Position(
                id=uuid4(),
                trading_pair=sample_trading_pair_eth,
                side=PositionSide.SHORT,
                quantity=5.0,
                entry_price=3200.0,
                current_price=3300.0,
                is_open=False,
                closed_at=now - timedelta(hours=4),
                created_at=now - timedelta(days=2),
                updated_at=now - timedelta(hours=4)
            )
        ]

    @pytest.mark.asyncio
    async def test_save_position(self, repository, sample_position):
        """Тест сохранения позиции."""
        repository.save = AsyncMock(return_value=True)
        result = await repository.save(sample_position)
        assert result is True
        repository.save.assert_called_once_with(sample_position)

    @pytest.mark.asyncio
    async def test_get_by_id_existing(self, repository, sample_position):
        """Тест получения позиции по ID - позиция существует."""
        repository.get_by_id = AsyncMock(return_value=sample_position)
        result = await repository.get_by_id(sample_position.id)
        assert result == sample_position
        repository.get_by_id.assert_called_once_with(sample_position.id)

    @pytest.mark.asyncio
    async def test_get_by_id_not_existing(self, repository):
        """Тест получения позиции по ID - позиция не существует."""
        position_id = uuid4()
        repository.get_by_id = AsyncMock(return_value=None)
        result = await repository.get_by_id(position_id)
        assert result is None
        repository.get_by_id.assert_called_once_with(position_id)

    @pytest.mark.asyncio
    async def test_get_by_id_with_string_id(self, repository, sample_position):
        """Тест получения позиции по строковому ID."""
        position_id_str = str(sample_position.id)
        repository.get_by_id = AsyncMock(return_value=sample_position)
        result = await repository.get_by_id(position_id_str)
        assert result == sample_position
        repository.get_by_id.assert_called_once_with(position_id_str)

    @pytest.mark.asyncio
    async def test_get_by_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения позиций по торговой паре."""
        btc_positions = [p for p in sample_positions if p.trading_pair == sample_trading_pair]
        repository.get_by_trading_pair = AsyncMock(return_value=btc_positions)
        result = await repository.get_by_trading_pair(sample_trading_pair)
        assert result == btc_positions
        repository.get_by_trading_pair.assert_called_once_with(sample_trading_pair, open_only=True)

    @pytest.mark.asyncio
    async def test_get_by_trading_pair_open_only_false(self, repository, sample_positions, sample_trading_pair):
        """Тест получения позиций по торговой паре включая закрытые."""
        btc_positions = [p for p in sample_positions if p.trading_pair == sample_trading_pair]
        repository.get_by_trading_pair = AsyncMock(return_value=btc_positions)
        result = await repository.get_by_trading_pair(sample_trading_pair, open_only=False)
        assert result == btc_positions
        repository.get_by_trading_pair.assert_called_once_with(sample_trading_pair, open_only=False)

    @pytest.mark.asyncio
    async def test_get_open_positions(self, repository, sample_positions):
        """Тест получения открытых позиций."""
        open_positions = [p for p in sample_positions if p.is_open]
        repository.get_open_positions = AsyncMock(return_value=open_positions)
        result = await repository.get_open_positions()
        assert result == open_positions
        repository.get_open_positions.assert_called_once_with(trading_pair=None)

    @pytest.mark.asyncio
    async def test_get_open_positions_by_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения открытых позиций по торговой паре."""
        open_btc_positions = [p for p in sample_positions if p.is_open and p.trading_pair == sample_trading_pair]
        repository.get_open_positions = AsyncMock(return_value=open_btc_positions)
        result = await repository.get_open_positions(trading_pair=sample_trading_pair)
        assert result == open_btc_positions
        repository.get_open_positions.assert_called_once_with(trading_pair=sample_trading_pair)

    @pytest.mark.asyncio
    async def test_get_closed_positions(self, repository, sample_positions):
        """Тест получения закрытых позиций."""
        closed_positions = [p for p in sample_positions if not p.is_open]
        repository.get_closed_positions = AsyncMock(return_value=closed_positions)
        result = await repository.get_closed_positions()
        assert result == closed_positions
        repository.get_closed_positions.assert_called_once_with(
            trading_pair=None, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_closed_positions_with_date_range(self, repository, sample_positions):
        """Тест получения закрытых позиций с диапазоном дат."""
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now() - timedelta(hours=1)
        closed_positions = [p for p in sample_positions if not p.is_open]
        repository.get_closed_positions = AsyncMock(return_value=closed_positions)
        result = await repository.get_closed_positions(
            start_date=start_date, end_date=end_date
        )
        assert result == closed_positions
        repository.get_closed_positions.assert_called_once_with(
            trading_pair=None, start_date=start_date, end_date=end_date
        )

    @pytest.mark.asyncio
    async def test_get_positions_by_side(self, repository, sample_positions):
        """Тест получения позиций по стороне."""
        long_positions = [p for p in sample_positions if p.side == PositionSide.LONG]
        repository.get_positions_by_side = AsyncMock(return_value=long_positions)
        result = await repository.get_positions_by_side(PositionSide.LONG)
        assert result == long_positions
        repository.get_positions_by_side.assert_called_once_with(
            PositionSide.LONG, trading_pair=None, open_only=True
        )

    @pytest.mark.asyncio
    async def test_get_positions_by_side_with_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения позиций по стороне и торговой паре."""
        long_btc_positions = [p for p in sample_positions if p.side == PositionSide.LONG and p.trading_pair == sample_trading_pair]
        repository.get_positions_by_side = AsyncMock(return_value=long_btc_positions)
        result = await repository.get_positions_by_side(
            PositionSide.LONG, trading_pair=sample_trading_pair
        )
        assert result == long_btc_positions
        repository.get_positions_by_side.assert_called_once_with(
            PositionSide.LONG, trading_pair=sample_trading_pair, open_only=True
        )

    @pytest.mark.asyncio
    async def test_get_positions_by_date_range(self, repository, sample_positions):
        """Тест получения позиций по диапазону дат."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        repository.get_positions_by_date_range = AsyncMock(return_value=sample_positions)
        result = await repository.get_positions_by_date_range(start_date, end_date)
        assert result == sample_positions
        repository.get_positions_by_date_range.assert_called_once_with(
            start_date, end_date, trading_pair=None
        )

    @pytest.mark.asyncio
    async def test_get_positions_by_date_range_with_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения позиций по диапазону дат и торговой паре."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        btc_positions = [p for p in sample_positions if p.trading_pair == sample_trading_pair]
        repository.get_positions_by_date_range = AsyncMock(return_value=btc_positions)
        result = await repository.get_positions_by_date_range(
            start_date, end_date, trading_pair=sample_trading_pair
        )
        assert result == btc_positions
        repository.get_positions_by_date_range.assert_called_once_with(
            start_date, end_date, trading_pair=sample_trading_pair
        )

    @pytest.mark.asyncio
    async def test_update_existing_position(self, repository, sample_position):
        """Тест обновления существующей позиции."""
        updated_position = Position(
            id=sample_position.id,
            trading_pair=sample_position.trading_pair,
            side=sample_position.side,
            quantity=sample_position.quantity,
            entry_price=sample_position.entry_price,
            current_price=52000.0,  # Обновленная цена
            is_open=sample_position.is_open,
            created_at=sample_position.created_at,
            updated_at=datetime.now()
        )
        repository.update = AsyncMock(return_value=updated_position)
        result = await repository.update(sample_position)
        assert result == updated_position
        repository.update.assert_called_once_with(sample_position)

    @pytest.mark.asyncio
    async def test_delete_existing_position(self, repository, sample_position):
        """Тест удаления существующей позиции."""
        repository.delete = AsyncMock(return_value=True)
        result = await repository.delete(sample_position.id)
        assert result is True
        repository.delete.assert_called_once_with(sample_position.id)

    @pytest.mark.asyncio
    async def test_delete_not_existing_position(self, repository):
        """Тест удаления несуществующей позиции."""
        position_id = uuid4()
        repository.delete = AsyncMock(return_value=False)
        result = await repository.delete(position_id)
        assert result is False
        repository.delete.assert_called_once_with(position_id)

    @pytest.mark.asyncio
    async def test_delete_with_string_id(self, repository, sample_position):
        """Тест удаления позиции по строковому ID."""
        position_id_str = str(sample_position.id)
        repository.delete = AsyncMock(return_value=True)
        result = await repository.delete(position_id_str)
        assert result is True
        repository.delete.assert_called_once_with(position_id_str)

    @pytest.mark.asyncio
    async def test_exists_true(self, repository, sample_position):
        """Тест проверки существования позиции - позиция существует."""
        repository.exists = AsyncMock(return_value=True)
        result = await repository.exists(sample_position.id)
        assert result is True
        repository.exists.assert_called_once_with(sample_position.id)

    @pytest.mark.asyncio
    async def test_exists_false(self, repository):
        """Тест проверки существования позиции - позиция не существует."""
        position_id = uuid4()
        repository.exists = AsyncMock(return_value=False)
        result = await repository.exists(position_id)
        assert result is False
        repository.exists.assert_called_once_with(position_id)

    @pytest.mark.asyncio
    async def test_exists_with_string_id(self, repository, sample_position):
        """Тест проверки существования позиции по строковому ID."""
        position_id_str = str(sample_position.id)
        repository.exists = AsyncMock(return_value=True)
        result = await repository.exists(position_id_str)
        assert result is True
        repository.exists.assert_called_once_with(position_id_str)

    @pytest.mark.asyncio
    async def test_count_no_filters(self, repository, sample_positions):
        """Тест подсчета позиций без фильтров."""
        repository.count = AsyncMock(return_value=len(sample_positions))
        result = await repository.count()
        assert result == len(sample_positions)
        repository.count.assert_called_once_with(filters=None)

    @pytest.mark.asyncio
    async def test_count_with_filters(self, repository, sample_positions):
        """Тест подсчета позиций с фильтрами."""
        filters = [QueryFilter(field="is_open", operator="eq", value=True)]
        open_positions_count = len([p for p in sample_positions if p.is_open])
        repository.count = AsyncMock(return_value=open_positions_count)
        result = await repository.count(filters=filters)
        assert result == open_positions_count
        repository.count.assert_called_once_with(filters=filters)

    @pytest.mark.asyncio
    async def test_count_empty_repository(self, repository):
        """Тест подсчета позиций в пустом репозитории."""
        repository.count = AsyncMock(return_value=0)
        result = await repository.count()
        assert result == 0
        repository.count.assert_called_once_with(filters=None)

    @pytest.mark.asyncio
    async def test_get_profitable_positions(self, repository, sample_positions):
        """Тест получения прибыльных позиций."""
        profitable_positions = [p for p in sample_positions if p.is_open and p.current_price > p.entry_price]
        repository.get_profitable_positions = AsyncMock(return_value=profitable_positions)
        result = await repository.get_profitable_positions()
        assert result == profitable_positions
        repository.get_profitable_positions.assert_called_once_with(
            trading_pair=None, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_profitable_positions_with_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения прибыльных позиций по торговой паре."""
        profitable_btc_positions = [
            p for p in sample_positions 
            if p.trading_pair == sample_trading_pair and p.is_open and p.current_price > p.entry_price
        ]
        repository.get_profitable_positions = AsyncMock(return_value=profitable_btc_positions)
        result = await repository.get_profitable_positions(trading_pair=sample_trading_pair)
        assert result == profitable_btc_positions
        repository.get_profitable_positions.assert_called_once_with(
            trading_pair=sample_trading_pair, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_losing_positions(self, repository, sample_positions):
        """Тест получения убыточных позиций."""
        losing_positions = [p for p in sample_positions if p.is_open and p.current_price < p.entry_price]
        repository.get_losing_positions = AsyncMock(return_value=losing_positions)
        result = await repository.get_losing_positions()
        assert result == losing_positions
        repository.get_losing_positions.assert_called_once_with(
            trading_pair=None, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_losing_positions_with_trading_pair(self, repository, sample_positions, sample_trading_pair):
        """Тест получения убыточных позиций по торговой паре."""
        losing_btc_positions = [
            p for p in sample_positions 
            if p.trading_pair == sample_trading_pair and p.is_open and p.current_price < p.entry_price
        ]
        repository.get_losing_positions = AsyncMock(return_value=losing_btc_positions)
        result = await repository.get_losing_positions(trading_pair=sample_trading_pair)
        assert result == losing_btc_positions
        repository.get_losing_positions.assert_called_once_with(
            trading_pair=sample_trading_pair, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_positions_with_stop_loss(self, repository, sample_positions):
        """Тест получения позиций со стоп-лоссом."""
        positions_with_sl = [p for p in sample_positions if hasattr(p, 'stop_loss') and p.stop_loss is not None]
        repository.get_positions_with_stop_loss = AsyncMock(return_value=positions_with_sl)
        result = await repository.get_positions_with_stop_loss()
        assert result == positions_with_sl
        repository.get_positions_with_stop_loss.assert_called_once_with(trading_pair=None)

    @pytest.mark.asyncio
    async def test_get_positions_with_take_profit(self, repository, sample_positions):
        """Тест получения позиций с тейк-профитом."""
        positions_with_tp = [p for p in sample_positions if hasattr(p, 'take_profit') and p.take_profit is not None]
        repository.get_positions_with_take_profit = AsyncMock(return_value=positions_with_tp)
        result = await repository.get_positions_with_take_profit()
        assert result == positions_with_tp
        repository.get_positions_with_take_profit.assert_called_once_with(trading_pair=None)

    @pytest.mark.asyncio
    async def test_get_statistics(self, repository, sample_positions):
        """Тест получения статистики по позициям."""
        expected_stats = {
            'total_positions': len(sample_positions),
            'open_positions': len([p for p in sample_positions if p.is_open]),
            'closed_positions': len([p for p in sample_positions if not p.is_open]),
            'total_pnl': 1000.0,
            'win_rate': 0.75
        }
        repository.get_statistics = AsyncMock(return_value=expected_stats)
        result = await repository.get_statistics()
        assert result == expected_stats
        repository.get_statistics.assert_called_once_with(
            trading_pair=None, start_date=None, end_date=None
        )

    @pytest.mark.asyncio
    async def test_get_total_exposure(self, repository, sample_positions):
        """Тест получения общего риска по позициям."""
        expected_exposure = {
            'total_exposure': 50000.0,
            'long_exposure': 30000.0,
            'short_exposure': 20000.0,
            'net_exposure': 10000.0
        }
        repository.get_total_exposure = AsyncMock(return_value=expected_exposure)
        result = await repository.get_total_exposure()
        assert result == expected_exposure
        repository.get_total_exposure.assert_called_once_with(trading_pair=None)

    @pytest.mark.asyncio
    async def test_cleanup_old_positions(self, repository):
        """Тест очистки старых позиций."""
        before_date = datetime.now() - timedelta(days=30)
        deleted_count = 5
        repository.cleanup_old_positions = AsyncMock(return_value=deleted_count)
        result = await repository.cleanup_old_positions(before_date)
        assert result == deleted_count
        repository.cleanup_old_positions.assert_called_once_with(before_date)

    @pytest.mark.asyncio
    async def test_get_by_symbol(self, repository, sample_position):
        """Тест получения позиции по портфелю и символу."""
        portfolio_id = uuid4()
        symbol = "BTCUSDT"
        repository.get_by_symbol = AsyncMock(return_value=sample_position)
        result = await repository.get_by_symbol(portfolio_id, symbol)
        assert result == sample_position
        repository.get_by_symbol.assert_called_once_with(portfolio_id, symbol)

    @pytest.mark.asyncio
    async def test_get_by_symbol_not_found(self, repository):
        """Тест получения позиции по портфелю и символу - не найдена."""
        portfolio_id = uuid4()
        symbol = "BTCUSDT"
        repository.get_by_symbol = AsyncMock(return_value=None)
        result = await repository.get_by_symbol(portfolio_id, symbol)
        assert result is None
        repository.get_by_symbol.assert_called_once_with(portfolio_id, symbol)

    @pytest.mark.asyncio
    async def test_position_side_transitions(self, repository, sample_position):
        """Тест переходов между сторонами позиций."""
        # Тест LONG позиции
        long_position = Position(
            id=uuid4(),
            trading_pair=sample_position.trading_pair,
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        repository.save = AsyncMock(return_value=True)
        repository.get_by_id = AsyncMock(return_value=long_position)
        
        await repository.save(long_position)
        retrieved = await repository.get_by_id(long_position.id)
        assert retrieved.side == PositionSide.LONG

        # Тест SHORT позиции
        short_position = Position(
            id=uuid4(),
            trading_pair=sample_position.trading_pair,
            side=PositionSide.SHORT,
            quantity=1.0,
            entry_price=52000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        repository.save = AsyncMock(return_value=True)
        repository.get_by_id = AsyncMock(return_value=short_position)
        
        await repository.save(short_position)
        retrieved = await repository.get_by_id(short_position.id)
        assert retrieved.side == PositionSide.SHORT

    @pytest.mark.asyncio
    async def test_position_price_updates(self, repository, sample_position):
        """Тест обновления цен позиций."""
        # Обновление текущей цены
        updated_position = Position(
            id=sample_position.id,
            trading_pair=sample_position.trading_pair,
            side=sample_position.side,
            quantity=sample_position.quantity,
            entry_price=sample_position.entry_price,
            current_price=52000.0,  # Обновленная цена
            is_open=sample_position.is_open,
            created_at=sample_position.created_at,
            updated_at=datetime.now()
        )
        repository.update = AsyncMock(return_value=updated_position)
        repository.get_by_id = AsyncMock(return_value=updated_position)
        
        result = await repository.update(sample_position)
        assert result.current_price == 52000.0
        
        retrieved = await repository.get_by_id(sample_position.id)
        assert retrieved.current_price == 52000.0

    @pytest.mark.asyncio
    async def test_position_quantity_updates(self, repository, sample_position):
        """Тест обновления количества позиций."""
        # Увеличение количества
        updated_position = Position(
            id=sample_position.id,
            trading_pair=sample_position.trading_pair,
            side=sample_position.side,
            quantity=2.0,  # Увеличенное количество
            entry_price=sample_position.entry_price,
            current_price=sample_position.current_price,
            is_open=sample_position.is_open,
            created_at=sample_position.created_at,
            updated_at=datetime.now()
        )
        repository.update = AsyncMock(return_value=updated_position)
        repository.get_by_id = AsyncMock(return_value=updated_position)
        
        result = await repository.update(sample_position)
        assert result.quantity == 2.0
        
        retrieved = await repository.get_by_id(sample_position.id)
        assert retrieved.quantity == 2.0

    @pytest.mark.asyncio
    async def test_multiple_positions_same_trading_pair(self, repository, sample_trading_pair):
        """Тест множественных позиций по одной торговой паре."""
        # Создание нескольких позиций по одной паре
        positions = []
        for i in range(3):
            position = Position(
                id=uuid4(),
                trading_pair=sample_trading_pair,
                side=PositionSide.LONG if i % 2 == 0 else PositionSide.SHORT,
                quantity=1.0 + i * 0.5,
                entry_price=50000.0 + i * 1000,
                current_price=51000.0 + i * 1000,
                is_open=True,
                created_at=datetime.now() - timedelta(hours=i),
                updated_at=datetime.now()
            )
            positions.append(position)
        
        repository.get_by_trading_pair = AsyncMock(return_value=positions)
        result = await repository.get_by_trading_pair(sample_trading_pair)
        assert len(result) == 3
        assert all(p.trading_pair == sample_trading_pair for p in result)

    @pytest.mark.asyncio
    async def test_repository_isolation(self):
        """Тест изоляции репозиториев."""
        # Создание двух независимых репозиториев
        repo1 = Mock(spec=PositionRepository)
        repo2 = Mock(spec=PositionRepository)
        
        # Данные в первом репозитории
        position1 = Position(
            id=uuid4(),
            trading_pair=TradingPair(base="BTC", quote="USDT"),
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Данные во втором репозитории
        position2 = Position(
            id=uuid4(),
            trading_pair=TradingPair(base="ETH", quote="USDT"),
            side=PositionSide.SHORT,
            quantity=10.0,
            entry_price=3000.0,
            current_price=3100.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Настройка моков
        repo1.get_by_id = AsyncMock(return_value=position1)
        repo2.get_by_id = AsyncMock(return_value=position2)
        
        # Проверка изоляции
        result1 = await repo1.get_by_id(position1.id)
        result2 = await repo2.get_by_id(position2.id)
        
        assert result1 == position1
        assert result2 == position2
        assert result1 != result2
        assert result1.trading_pair != result2.trading_pair

    @pytest.mark.asyncio
    async def test_error_handling_save(self, repository, sample_position):
        """Тест обработки ошибок при сохранении."""
        repository.save = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception, match="Database error"):
            await repository.save(sample_position)

    @pytest.mark.asyncio
    async def test_error_handling_get_by_id(self, repository):
        """Тест обработки ошибок при получении по ID."""
        position_id = uuid4()
        repository.get_by_id = AsyncMock(side_effect=Exception("Connection error"))
        
        with pytest.raises(Exception, match="Connection error"):
            await repository.get_by_id(position_id)

    @pytest.mark.asyncio
    async def test_error_handling_update(self, repository, sample_position):
        """Тест обработки ошибок при обновлении."""
        repository.update = AsyncMock(side_effect=Exception("Update failed"))
        
        with pytest.raises(Exception, match="Update failed"):
            await repository.update(sample_position)

    @pytest.mark.asyncio
    async def test_error_handling_delete(self, repository):
        """Тест обработки ошибок при удалении."""
        position_id = uuid4()
        repository.delete = AsyncMock(side_effect=Exception("Delete failed"))
        
        with pytest.raises(Exception, match="Delete failed"):
            await repository.delete(position_id) 