"""
Unit тесты для domain/repositories/portfolio_repository.py.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from domain.repositories.portfolio_repository import PortfolioRepository
from domain.entities.portfolio import Portfolio
from domain.entities.position import Position
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions.base_exceptions import EntityNotFoundError, ValidationError


class TestPortfolioRepository:
    """Тесты для PortfolioRepository."""
    
    @pytest.fixture
    def repository(self):
        """Создание репозитория."""
        return PortfolioRepository()
    
    @pytest.fixture
    def sample_position(self) -> Position:
        """Тестовая позиция."""
        return Position(
            id="pos_001",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("2.0"),
            average_price=Decimal("45000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("10000.00"),
            realized_pnl=Decimal("5000.00"),
            timestamp=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_portfolio_data(self) -> Dict[str, Any]:
        """Тестовые данные портфеля."""
        return {
            "id": "portfolio_001",
            "name": "Test Portfolio",
            "currency": "USD",
            "total_value": Decimal("100000.00"),
            "cash_balance": Decimal("20000.00"),
            "positions": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
    
    @pytest.fixture
    def sample_portfolios(self, sample_position) -> List[Portfolio]:
        """Тестовые портфели."""
        return [
            Portfolio(
                id="portfolio_001",
                name="Conservative Portfolio",
                currency="USD",
                total_value=Decimal("100000.00"),
                cash_balance=Decimal("20000.00"),
                positions=[sample_position],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            Portfolio(
                id="portfolio_002",
                name="Aggressive Portfolio",
                currency="USD",
                total_value=Decimal("50000.00"),
                cash_balance=Decimal("5000.00"),
                positions=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            Portfolio(
                id="portfolio_003",
                name="EUR Portfolio",
                currency="EUR",
                total_value=Decimal("80000.00"),
                cash_balance=Decimal("15000.00"),
                positions=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        ]
    
    def test_add_portfolio(self, repository, sample_portfolios):
        """Тест добавления портфеля."""
        portfolio = sample_portfolios[0]
        
        result = repository.add(portfolio)
        
        assert result == portfolio
        assert repository.get_by_id("portfolio_001") == portfolio
        assert len(repository.get_all()) == 1
    
    def test_get_by_id_existing(self, repository, sample_portfolios):
        """Тест получения существующего портфеля по ID."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        result = repository.get_by_id("portfolio_001")
        
        assert result == portfolio
        assert result.id == "portfolio_001"
        assert result.name == "Conservative Portfolio"
        assert result.currency == "USD"
    
    def test_get_by_id_not_found(self, repository):
        """Тест получения несуществующего портфеля по ID."""
        with pytest.raises(EntityNotFoundError, match="Portfolio with id portfolio_999 not found"):
            repository.get_by_id("portfolio_999")
    
    def test_get_by_name_existing(self, repository, sample_portfolios):
        """Тест получения портфеля по названию."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        result = repository.get_by_name("Conservative Portfolio")
        
        assert result == portfolio
        assert result.name == "Conservative Portfolio"
    
    def test_get_by_name_not_found(self, repository):
        """Тест получения портфеля по несуществующему названию."""
        with pytest.raises(EntityNotFoundError, match="Portfolio with name Non-existent not found"):
            repository.get_by_name("Non-existent")
    
    def test_get_portfolios_by_currency(self, repository, sample_portfolios):
        """Тест получения портфелей по валюте."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        usd_portfolios = repository.get_portfolios_by_currency("USD")
        eur_portfolios = repository.get_portfolios_by_currency("EUR")
        
        assert len(usd_portfolios) == 2
        assert len(eur_portfolios) == 1
        assert all(p.currency == "USD" for p in usd_portfolios)
        assert all(p.currency == "EUR" for p in eur_portfolios)
    
    def test_get_portfolios_by_currency_not_found(self, repository, sample_portfolios):
        """Тест получения портфелей по несуществующей валюте."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        invalid_portfolios = repository.get_portfolios_by_currency("INVALID")
        assert len(invalid_portfolios) == 0
    
    def test_get_portfolios_by_value_range(self, repository, sample_portfolios):
        """Тест получения портфелей по диапазону стоимости."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        # Портфели со стоимостью от 50000 до 150000
        portfolios_in_range = repository.get_portfolios_by_value_range(
            min_value=Decimal("50000.00"),
            max_value=Decimal("150000.00")
        )
        
        assert len(portfolios_in_range) == 2
        assert any(p.id == "portfolio_001" for p in portfolios_in_range)
        assert any(p.id == "portfolio_002" for p in portfolios_in_range)
    
    def test_get_portfolios_by_cash_balance_range(self, repository, sample_portfolios):
        """Тест получения портфелей по диапазону наличных средств."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        # Портфели с наличными от 10000 до 25000
        portfolios_in_range = repository.get_portfolios_by_cash_balance_range(
            min_balance=Decimal("10000.00"),
            max_balance=Decimal("25000.00")
        )
        
        assert len(portfolios_in_range) == 2
        assert any(p.id == "portfolio_001" for p in portfolios_in_range)
        assert any(p.id == "portfolio_003" for p in portfolios_in_range)
    
    def test_get_portfolios_with_positions(self, repository, sample_portfolios):
        """Тест получения портфелей с позициями."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        portfolios_with_positions = repository.get_portfolios_with_positions()
        
        assert len(portfolios_with_positions) == 1
        assert portfolios_with_positions[0].id == "portfolio_001"
        assert len(portfolios_with_positions[0].positions) > 0
    
    def test_get_portfolios_without_positions(self, repository, sample_portfolios):
        """Тест получения портфелей без позиций."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        portfolios_without_positions = repository.get_portfolios_without_positions()
        
        assert len(portfolios_without_positions) == 2
        assert all(len(p.positions) == 0 for p in portfolios_without_positions)
    
    def test_update_portfolio(self, repository, sample_portfolios):
        """Тест обновления портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        updated_portfolio = Portfolio(
            id="portfolio_001",
            name="Updated Conservative Portfolio",
            currency="USD",
            total_value=Decimal("120000.00"),
            cash_balance=Decimal("25000.00"),
            positions=portfolio.positions,
            created_at=portfolio.created_at,
            updated_at=datetime.now(timezone.utc)
        )
        
        result = repository.update(updated_portfolio)
        
        assert result == updated_portfolio
        stored = repository.get_by_id("portfolio_001")
        assert stored.name == "Updated Conservative Portfolio"
        assert stored.total_value == Decimal("120000.00")
        assert stored.cash_balance == Decimal("25000.00")
    
    def test_update_portfolio_not_found(self, repository, sample_portfolios):
        """Тест обновления несуществующего портфеля."""
        portfolio = Portfolio(
            id="portfolio_999",
            name="Non-existent Portfolio",
            currency="USD",
            total_value=Decimal("10000.00"),
            cash_balance=Decimal("1000.00"),
            positions=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        with pytest.raises(EntityNotFoundError, match="Portfolio with id portfolio_999 not found"):
            repository.update(portfolio)
    
    def test_delete_portfolio(self, repository, sample_portfolios):
        """Тест удаления портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        result = repository.delete("portfolio_001")
        
        assert result == portfolio
        with pytest.raises(EntityNotFoundError):
            repository.get_by_id("portfolio_001")
        assert len(repository.get_all()) == 0
    
    def test_delete_portfolio_not_found(self, repository):
        """Тест удаления несуществующего портфеля."""
        with pytest.raises(EntityNotFoundError, match="Portfolio with id portfolio_999 not found"):
            repository.delete("portfolio_999")
    
    def test_add_position_to_portfolio(self, repository, sample_portfolios, sample_position):
        """Тест добавления позиции к портфелю."""
        portfolio = sample_portfolios[1]  # Aggressive Portfolio без позиций
        repository.add(portfolio)
        
        new_position = Position(
            id="pos_002",
            symbol="ETHUSD",
            side="LONG",
            quantity=Decimal("10.0"),
            average_price=Decimal("3000.00"),
            current_price=Decimal("3500.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        repository.add_position_to_portfolio("portfolio_002", new_position)
        
        updated_portfolio = repository.get_by_id("portfolio_002")
        assert len(updated_portfolio.positions) == 1
        assert updated_portfolio.positions[0] == new_position
    
    def test_add_position_to_portfolio_not_found(self, repository, sample_position):
        """Тест добавления позиции к несуществующему портфелю."""
        with pytest.raises(EntityNotFoundError, match="Portfolio with id portfolio_999 not found"):
            repository.add_position_to_portfolio("portfolio_999", sample_position)
    
    def test_remove_position_from_portfolio(self, repository, sample_portfolios):
        """Тест удаления позиции с портфеля."""
        portfolio = sample_portfolios[0]  # Conservative Portfolio с позициями
        repository.add(portfolio)
        
        initial_position_count = len(portfolio.positions)
        repository.remove_position_from_portfolio("portfolio_001", portfolio.positions[0])
        
        updated_portfolio = repository.get_by_id("portfolio_001")
        assert len(updated_portfolio.positions) == initial_position_count - 1
    
    def test_update_portfolio_value(self, repository, sample_portfolios):
        """Тест обновления стоимости портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        new_value = Decimal("120000.00")
        repository.update_portfolio_value("portfolio_001", new_value)
        
        updated_portfolio = repository.get_by_id("portfolio_001")
        assert updated_portfolio.total_value == new_value
    
    def test_update_portfolio_cash_balance(self, repository, sample_portfolios):
        """Тест обновления наличных средств портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        new_balance = Decimal("25000.00")
        repository.update_portfolio_cash_balance("portfolio_001", new_balance)
        
        updated_portfolio = repository.get_by_id("portfolio_001")
        assert updated_portfolio.cash_balance == new_balance
    
    def test_get_portfolios_by_creation_date_range(self, repository, sample_portfolios):
        """Тест получения портфелей по диапазону дат создания."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        # Получаем портфели за последний день
        end_date = datetime.now(timezone.utc)
        start_date = end_date.replace(day=end_date.day - 1)
        
        portfolios_in_range = repository.get_portfolios_by_creation_date_range(start_date, end_date)
        
        assert len(portfolios_in_range) == 3  # Все портфели созданы в последний день
    
    def test_get_portfolios_by_update_date_range(self, repository, sample_portfolios):
        """Тест получения портфелей по диапазону дат обновления."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        # Получаем портфели за последний день
        end_date = datetime.now(timezone.utc)
        start_date = end_date.replace(day=end_date.day - 1)
        
        portfolios_in_range = repository.get_portfolios_by_update_date_range(start_date, end_date)
        
        assert len(portfolios_in_range) == 3  # Все портфели обновлены в последний день
    
    def test_search_portfolios_by_name(self, repository, sample_portfolios):
        """Тест поиска портфелей по названию."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        conservative_portfolios = repository.search_portfolios_by_name("Conservative")
        aggressive_portfolios = repository.search_portfolios_by_name("Aggressive")
        portfolio_portfolios = repository.search_portfolios_by_name("Portfolio")
        
        assert len(conservative_portfolios) == 1
        assert len(aggressive_portfolios) == 1
        assert len(portfolio_portfolios) == 3  # Все портфели содержат "Portfolio"
    
    def test_get_portfolio_statistics(self, repository, sample_portfolios):
        """Тест получения статистики портфелей."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        stats = repository.get_portfolio_statistics()
        
        assert isinstance(stats, dict)
        assert "total_portfolios" in stats
        assert "total_value" in stats
        assert "average_value" in stats
        assert "total_cash_balance" in stats
        
        assert stats["total_portfolios"] == 3
        assert stats["total_value"] == Decimal("230000.00")  # 100000 + 50000 + 80000
    
    def test_get_top_portfolios_by_value(self, repository, sample_portfolios):
        """Тест получения топ портфелей по стоимости."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        top_portfolios = repository.get_top_portfolios_by_value(limit=2)
        
        assert len(top_portfolios) == 2
        # Conservative Portfolio должен быть первым (наибольшая стоимость)
        assert top_portfolios[0].id == "portfolio_001"
        assert top_portfolios[1].id == "portfolio_003"
    
    def test_get_top_portfolios_by_cash_balance(self, repository, sample_portfolios):
        """Тест получения топ портфелей по наличным средствам."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        top_portfolios = repository.get_top_portfolios_by_cash_balance(limit=2)
        
        assert len(top_portfolios) == 2
        # Conservative Portfolio должен быть первым (наибольшие наличные)
        assert top_portfolios[0].id == "portfolio_001"
        assert top_portfolios[1].id == "portfolio_003"
    
    def test_exists_by_name(self, repository, sample_portfolios):
        """Тест проверки существования портфеля по названию."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        assert repository.exists_by_name("Conservative Portfolio") is True
        assert repository.exists_by_name("Non-existent") is False
    
    def test_get_portfolios_by_multiple_criteria(self, repository, sample_portfolios):
        """Тест получения портфелей по множественным критериям."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        # USD портфели со стоимостью более 50000
        criteria_portfolios = repository.get_portfolios_by_multiple_criteria(
            currency="USD",
            min_value=Decimal("50000.00")
        )
        
        assert len(criteria_portfolios) == 2
        assert all(p.currency == "USD" and p.total_value >= Decimal("50000.00") 
                  for p in criteria_portfolios)
    
    def test_bulk_update_portfolio_values(self, repository, sample_portfolios):
        """Тест массового обновления стоимости портфелей."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        value_updates = {
            "portfolio_001": Decimal("120000.00"),
            "portfolio_002": Decimal("60000.00"),
            "portfolio_003": Decimal("90000.00")
        }
        
        repository.bulk_update_portfolio_values(value_updates)
        
        for portfolio_id, new_value in value_updates.items():
            portfolio = repository.get_by_id(portfolio_id)
            assert portfolio.total_value == new_value
    
    def test_get_portfolios_by_position_symbol(self, repository, sample_portfolios):
        """Тест получения портфелей по символу позиции."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        btc_portfolios = repository.get_portfolios_by_position_symbol("BTCUSD")
        eth_portfolios = repository.get_portfolios_by_position_symbol("ETHUSD")
        
        assert len(btc_portfolios) == 1
        assert len(eth_portfolios) == 0
        assert btc_portfolios[0].id == "portfolio_001"
    
    def test_get_portfolios_by_position_side(self, repository, sample_portfolios):
        """Тест получения портфелей по стороне позиции."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        long_portfolios = repository.get_portfolios_by_position_side("LONG")
        short_portfolios = repository.get_portfolios_by_position_side("SHORT")
        
        assert len(long_portfolios) == 1
        assert len(short_portfolios) == 0
        assert long_portfolios[0].id == "portfolio_001"
    
    def test_get_portfolios_with_profitable_positions(self, repository, sample_portfolios):
        """Тест получения портфелей с прибыльными позициями."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        profitable_portfolios = repository.get_portfolios_with_profitable_positions()
        
        assert len(profitable_portfolios) == 1
        assert profitable_portfolios[0].id == "portfolio_001"
    
    def test_get_portfolios_with_losing_positions(self, repository, sample_portfolios):
        """Тест получения портфелей с убыточными позициями."""
        for portfolio in sample_portfolios:
            repository.add(portfolio)
        
        losing_portfolios = repository.get_portfolios_with_losing_positions()
        
        assert len(losing_portfolios) == 0  # Все позиции прибыльные
    
    def test_get_portfolio_performance_metrics(self, repository, sample_portfolios):
        """Тест получения метрик производительности портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        metrics = repository.get_portfolio_performance_metrics("portfolio_001")
        
        assert isinstance(metrics, dict)
        assert "total_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "realized_pnl" in metrics
        assert "return_percentage" in metrics
    
    def test_get_portfolio_risk_metrics(self, repository, sample_portfolios):
        """Тест получения метрик риска портфеля."""
        portfolio = sample_portfolios[0]
        repository.add(portfolio)
        
        risk_metrics = repository.get_portfolio_risk_metrics("portfolio_001")
        
        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert "var_95" in risk_metrics 