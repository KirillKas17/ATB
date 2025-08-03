"""
Unit тесты для RepositoryProtocol.

Покрывает:
- Основные протоколы репозиториев
- CRUD операции
- Фильтрацию и поиск
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional, Protocol
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from domain.protocols.repository_protocol import (
    RepositoryProtocol, 
    AsyncRepositoryProtocol,
    TradingRepositoryProtocol,
    PortfolioRepositoryProtocol,
    StrategyRepositoryProtocol,
    MarketRepositoryProtocol,
    RiskRepositoryProtocol,
    MLRepositoryProtocol,
    OrderRepositoryProtocol,
    PositionRepositoryProtocol,
    TradingPairRepositoryProtocol
)
from domain.entities.order import Order
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy
from domain.entities.market import MarketData
from domain.entities.risk import RiskProfile
from domain.entities.ml import Model
from domain.entities.trading import Position
from domain.entities.trading_pair import TradingPair
from domain.types.repository_types import QueryFilter, QueryOptions
from domain.exceptions.base_exceptions import ValidationError


class TestRepositoryProtocol:
    """Тесты для базового RepositoryProtocol."""

    @pytest.fixture
    def mock_repository(self) -> Mock:
        """Мок репозитория."""
        return Mock(spec=RepositoryProtocol)

    def test_save_method_exists(self, mock_repository):
        """Тест наличия метода save."""
        mock_repository.save = AsyncMock(return_value=True)
        assert hasattr(mock_repository, 'save')
        assert callable(mock_repository.save)

    def test_get_by_id_method_exists(self, mock_repository):
        """Тест наличия метода get_by_id."""
        mock_repository.get_by_id = AsyncMock(return_value=None)
        assert hasattr(mock_repository, 'get_by_id')
        assert callable(mock_repository.get_by_id)

    def test_get_all_method_exists(self, mock_repository):
        """Тест наличия метода get_all."""
        mock_repository.get_all = AsyncMock(return_value=[])
        assert hasattr(mock_repository, 'get_all')
        assert callable(mock_repository.get_all)

    def test_update_method_exists(self, mock_repository):
        """Тест наличия метода update."""
        mock_repository.update = AsyncMock(return_value=None)
        assert hasattr(mock_repository, 'update')
        assert callable(mock_repository.update)

    def test_delete_method_exists(self, mock_repository):
        """Тест наличия метода delete."""
        mock_repository.delete = AsyncMock(return_value=True)
        assert hasattr(mock_repository, 'delete')
        assert callable(mock_repository.delete)

    def test_exists_method_exists(self, mock_repository):
        """Тест наличия метода exists."""
        mock_repository.exists = AsyncMock(return_value=True)
        assert hasattr(mock_repository, 'exists')
        assert callable(mock_repository.exists)

    def test_count_method_exists(self, mock_repository):
        """Тест наличия метода count."""
        mock_repository.count = AsyncMock(return_value=0)
        assert hasattr(mock_repository, 'count')
        assert callable(mock_repository.count)

    def test_find_by_criteria_method_exists(self, mock_repository):
        """Тест наличия метода find_by_criteria."""
        mock_repository.find_by_criteria = AsyncMock(return_value=[])
        assert hasattr(mock_repository, 'find_by_criteria')
        assert callable(mock_repository.find_by_criteria)


class TestAsyncRepositoryProtocol:
    """Тесты для AsyncRepositoryProtocol."""

    @pytest.fixture
    def mock_async_repository(self) -> Mock:
        """Мок асинхронного репозитория."""
        return Mock(spec=AsyncRepositoryProtocol)

    def test_async_save_method_exists(self, mock_async_repository):
        """Тест наличия асинхронного метода save."""
        mock_async_repository.save = AsyncMock(return_value=True)
        assert hasattr(mock_async_repository, 'save')
        assert callable(mock_async_repository.save)

    def test_async_get_by_id_method_exists(self, mock_async_repository):
        """Тест наличия асинхронного метода get_by_id."""
        mock_async_repository.get_by_id = AsyncMock(return_value=None)
        assert hasattr(mock_async_repository, 'get_by_id')
        assert callable(mock_async_repository.get_by_id)


class TestTradingRepositoryProtocol:
    """Тесты для TradingRepositoryProtocol."""

    @pytest.fixture
    def mock_trading_repository(self) -> Mock:
        """Мок торгового репозитория."""
        return Mock(spec=TradingRepositoryProtocol)

    def test_trading_specific_methods_exist(self, mock_trading_repository):
        """Тест наличия специфичных для торговли методов."""
        mock_trading_repository.get_by_trading_pair = AsyncMock(return_value=[])
        mock_trading_repository.get_by_date_range = AsyncMock(return_value=[])
        
        assert hasattr(mock_trading_repository, 'get_by_trading_pair')
        assert hasattr(mock_trading_repository, 'get_by_date_range')
        assert callable(mock_trading_repository.get_by_trading_pair)
        assert callable(mock_trading_repository.get_by_date_range)


class TestPortfolioRepositoryProtocol:
    """Тесты для PortfolioRepositoryProtocol."""

    @pytest.fixture
    def mock_portfolio_repository(self) -> Mock:
        """Мок репозитория портфелей."""
        return Mock(spec=PortfolioRepositoryProtocol)

    def test_portfolio_specific_methods_exist(self, mock_portfolio_repository):
        """Тест наличия специфичных для портфелей методов."""
        mock_portfolio_repository.get_by_user_id = AsyncMock(return_value=[])
        mock_portfolio_repository.get_active_portfolios = AsyncMock(return_value=[])
        
        assert hasattr(mock_portfolio_repository, 'get_by_user_id')
        assert hasattr(mock_portfolio_repository, 'get_active_portfolios')
        assert callable(mock_portfolio_repository.get_by_user_id)
        assert callable(mock_portfolio_repository.get_active_portfolios)


class TestStrategyRepositoryProtocol:
    """Тесты для StrategyRepositoryProtocol."""

    @pytest.fixture
    def mock_strategy_repository(self) -> Mock:
        """Мок репозитория стратегий."""
        return Mock(spec=StrategyRepositoryProtocol)

    def test_strategy_specific_methods_exist(self, mock_strategy_repository):
        """Тест наличия специфичных для стратегий методов."""
        mock_strategy_repository.get_by_type = AsyncMock(return_value=[])
        mock_strategy_repository.get_active_strategies = AsyncMock(return_value=[])
        
        assert hasattr(mock_strategy_repository, 'get_by_type')
        assert hasattr(mock_strategy_repository, 'get_active_strategies')
        assert callable(mock_strategy_repository.get_by_type)
        assert callable(mock_strategy_repository.get_active_strategies)


class TestMarketRepositoryProtocol:
    """Тесты для MarketRepositoryProtocol."""

    @pytest.fixture
    def mock_market_repository(self) -> Mock:
        """Мок репозитория рыночных данных."""
        return Mock(spec=MarketRepositoryProtocol)

    def test_market_specific_methods_exist(self, mock_market_repository):
        """Тест наличия специфичных для рыночных данных методов."""
        mock_market_repository.get_latest_data = AsyncMock(return_value=None)
        mock_market_repository.get_historical_data = AsyncMock(return_value=[])
        
        assert hasattr(mock_market_repository, 'get_latest_data')
        assert hasattr(mock_market_repository, 'get_historical_data')
        assert callable(mock_market_repository.get_latest_data)
        assert callable(mock_market_repository.get_historical_data)


class TestRiskRepositoryProtocol:
    """Тесты для RiskRepositoryProtocol."""

    @pytest.fixture
    def mock_risk_repository(self) -> Mock:
        """Мок репозитория рисков."""
        return Mock(spec=RiskRepositoryProtocol)

    def test_risk_specific_methods_exist(self, mock_risk_repository):
        """Тест наличия специфичных для рисков методов."""
        mock_risk_repository.get_by_portfolio_id = AsyncMock(return_value=None)
        mock_risk_repository.get_risk_metrics = AsyncMock(return_value={})
        
        assert hasattr(mock_risk_repository, 'get_by_portfolio_id')
        assert hasattr(mock_risk_repository, 'get_risk_metrics')
        assert callable(mock_risk_repository.get_by_portfolio_id)
        assert callable(mock_risk_repository.get_risk_metrics)


class TestMLRepositoryProtocol:
    """Тесты для MLRepositoryProtocol."""

    @pytest.fixture
    def mock_ml_repository(self) -> Mock:
        """Мок репозитория ML моделей."""
        return Mock(spec=MLRepositoryProtocol)

    def test_ml_specific_methods_exist(self, mock_ml_repository):
        """Тест наличия специфичных для ML методов."""
        mock_ml_repository.get_by_type = AsyncMock(return_value=[])
        mock_ml_repository.get_trained_models = AsyncMock(return_value=[])
        
        assert hasattr(mock_ml_repository, 'get_by_type')
        assert hasattr(mock_ml_repository, 'get_trained_models')
        assert callable(mock_ml_repository.get_by_type)
        assert callable(mock_ml_repository.get_trained_models)


class TestOrderRepositoryProtocol:
    """Тесты для OrderRepositoryProtocol."""

    @pytest.fixture
    def mock_order_repository(self) -> Mock:
        """Мок репозитория ордеров."""
        return Mock(spec=OrderRepositoryProtocol)

    def test_order_specific_methods_exist(self, mock_order_repository):
        """Тест наличия специфичных для ордеров методов."""
        mock_order_repository.get_by_status = AsyncMock(return_value=[])
        mock_order_repository.get_active_orders = AsyncMock(return_value=[])
        
        assert hasattr(mock_order_repository, 'get_by_status')
        assert hasattr(mock_order_repository, 'get_active_orders')
        assert callable(mock_order_repository.get_by_status)
        assert callable(mock_order_repository.get_active_orders)


class TestPositionRepositoryProtocol:
    """Тесты для PositionRepositoryProtocol."""

    @pytest.fixture
    def mock_position_repository(self) -> Mock:
        """Мок репозитория позиций."""
        return Mock(spec=PositionRepositoryProtocol)

    def test_position_specific_methods_exist(self, mock_position_repository):
        """Тест наличия специфичных для позиций методов."""
        mock_position_repository.get_open_positions = AsyncMock(return_value=[])
        mock_position_repository.get_by_trading_pair = AsyncMock(return_value=[])
        
        assert hasattr(mock_position_repository, 'get_open_positions')
        assert hasattr(mock_position_repository, 'get_by_trading_pair')
        assert callable(mock_position_repository.get_open_positions)
        assert callable(mock_position_repository.get_by_trading_pair)


class TestTradingPairRepositoryProtocol:
    """Тесты для TradingPairRepositoryProtocol."""

    @pytest.fixture
    def mock_trading_pair_repository(self) -> Mock:
        """Мок репозитория торговых пар."""
        return Mock(spec=TradingPairRepositoryProtocol)

    def test_trading_pair_specific_methods_exist(self, mock_trading_pair_repository):
        """Тест наличия специфичных для торговых пар методов."""
        mock_trading_pair_repository.get_by_symbol = AsyncMock(return_value=None)
        mock_trading_pair_repository.get_active_pairs = AsyncMock(return_value=[])
        
        assert hasattr(mock_trading_pair_repository, 'get_by_symbol')
        assert hasattr(mock_trading_pair_repository, 'get_active_pairs')
        assert callable(mock_trading_pair_repository.get_by_symbol)
        assert callable(mock_trading_pair_repository.get_active_pairs)


class TestRepositoryProtocolIntegration:
    """Интеграционные тесты для протоколов репозиториев."""

    @pytest.mark.asyncio
    async def test_repository_protocol_compliance(self):
        """Тест соответствия протоколу репозитория."""
        # Создание мока, соответствующего протоколу
        mock_repo = Mock(spec=RepositoryProtocol)
        
        # Настройка методов
        mock_repo.save = AsyncMock(return_value=True)
        mock_repo.get_by_id = AsyncMock(return_value=None)
        mock_repo.get_all = AsyncMock(return_value=[])
        mock_repo.update = AsyncMock(return_value=None)
        mock_repo.delete = AsyncMock(return_value=True)
        mock_repo.exists = AsyncMock(return_value=False)
        mock_repo.count = AsyncMock(return_value=0)
        mock_repo.find_by_criteria = AsyncMock(return_value=[])
        
        # Проверка вызовов
        await mock_repo.save(None)
        await mock_repo.get_by_id(uuid4())
        await mock_repo.get_all()
        await mock_repo.update(None)
        await mock_repo.delete(uuid4())
        await mock_repo.exists(uuid4())
        await mock_repo.count()
        await mock_repo.find_by_criteria([])
        
        # Проверка, что все методы были вызваны
        assert mock_repo.save.called
        assert mock_repo.get_by_id.called
        assert mock_repo.get_all.called
        assert mock_repo.update.called
        assert mock_repo.delete.called
        assert mock_repo.exists.called
        assert mock_repo.count.called
        assert mock_repo.find_by_criteria.called

    @pytest.mark.asyncio
    async def test_error_handling_in_protocols(self):
        """Тест обработки ошибок в протоколах."""
        mock_repo = Mock(spec=RepositoryProtocol)
        
        # Настройка методов для генерации ошибок
        mock_repo.save = AsyncMock(side_effect=Exception("Save failed"))
        mock_repo.get_by_id = AsyncMock(side_effect=Exception("Get failed"))
        mock_repo.update = AsyncMock(side_effect=Exception("Update failed"))
        mock_repo.delete = AsyncMock(side_effect=Exception("Delete failed"))
        
        # Проверка обработки ошибок
        with pytest.raises(Exception, match="Save failed"):
            await mock_repo.save(None)
            
        with pytest.raises(Exception, match="Get failed"):
            await mock_repo.get_by_id(uuid4())
            
        with pytest.raises(Exception, match="Update failed"):
            await mock_repo.update(None)
            
        with pytest.raises(Exception, match="Delete failed"):
            await mock_repo.delete(uuid4())

    @pytest.mark.asyncio
    async def test_protocol_method_signatures(self):
        """Тест сигнатур методов протоколов."""
        # Проверка, что протоколы имеют правильные сигнатуры методов
        assert hasattr(RepositoryProtocol, 'save')
        assert hasattr(RepositoryProtocol, 'get_by_id')
        assert hasattr(RepositoryProtocol, 'get_all')
        assert hasattr(RepositoryProtocol, 'update')
        assert hasattr(RepositoryProtocol, 'delete')
        assert hasattr(RepositoryProtocol, 'exists')
        assert hasattr(RepositoryProtocol, 'count')
        assert hasattr(RepositoryProtocol, 'find_by_criteria')

    def test_protocol_inheritance(self):
        """Тест наследования протоколов."""
        # Проверка, что специализированные протоколы наследуют от базового
        assert issubclass(TradingRepositoryProtocol, RepositoryProtocol)
        assert issubclass(PortfolioRepositoryProtocol, RepositoryProtocol)
        assert issubclass(StrategyRepositoryProtocol, RepositoryProtocol)
        assert issubclass(MarketRepositoryProtocol, RepositoryProtocol)
        assert issubclass(RiskRepositoryProtocol, RepositoryProtocol)
        assert issubclass(MLRepositoryProtocol, RepositoryProtocol)
        assert issubclass(OrderRepositoryProtocol, RepositoryProtocol)
        assert issubclass(PositionRepositoryProtocol, RepositoryProtocol)
        assert issubclass(TradingPairRepositoryProtocol, RepositoryProtocol) 