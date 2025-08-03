"""
Unit тесты для domain/repositories/market_repository.py.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from domain.repositories.market_repository import MarketRepository
from domain.entities.market import Market
from domain.value_objects.trading_pair import TradingPair
from domain.exceptions.base_exceptions import EntityNotFoundError, ValidationError


class TestMarketRepository:
    """Тесты для MarketRepository."""
    
    @pytest.fixture
    def repository(self):
        """Создание репозитория."""
        return MarketRepository()
    
    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(
            base_currency="BTC",
            quote_currency="USD",
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100.0"),
            price_precision=2,
            quantity_precision=6,
            min_notional=Decimal("10.00"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.000001")
        )
    
    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые данные рынка."""
        return {
            "id": "market_001",
            "name": "Bitcoin Market",
            "symbol": "BTCUSD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "is_active": True,
            "trading_pairs": [],
            "current_price": Decimal("50000.00"),
            "volume_24h": Decimal("1000000.00"),
            "price_change_24h": Decimal("5.5"),
            "high_24h": Decimal("52000.00"),
            "low_24h": Decimal("48000.00"),
            "timestamp": datetime.now(timezone.utc)
        }
    
    @pytest.fixture
    def sample_markets(self, sample_trading_pair) -> List[Market]:
        """Тестовые рынки."""
        return [
            Market(
                id="market_001",
                name="Bitcoin Market",
                symbol="BTCUSD",
                base_currency="BTC",
                quote_currency="USD",
                is_active=True,
                trading_pairs=[sample_trading_pair],
                current_price=Decimal("50000.00"),
                volume_24h=Decimal("1000000.00"),
                price_change_24h=Decimal("5.5"),
                high_24h=Decimal("52000.00"),
                low_24h=Decimal("48000.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            Market(
                id="market_002",
                name="Ethereum Market",
                symbol="ETHUSD",
                base_currency="ETH",
                quote_currency="USD",
                is_active=True,
                trading_pairs=[],
                current_price=Decimal("3500.00"),
                volume_24h=Decimal("500000.00"),
                price_change_24h=Decimal("-2.3"),
                high_24h=Decimal("3600.00"),
                low_24h=Decimal("3400.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            Market(
                id="market_003",
                name="Litecoin Market",
                symbol="LTCUSD",
                base_currency="LTC",
                quote_currency="USD",
                is_active=False,
                trading_pairs=[],
                current_price=Decimal("150.00"),
                volume_24h=Decimal("100000.00"),
                price_change_24h=Decimal("1.2"),
                high_24h=Decimal("155.00"),
                low_24h=Decimal("145.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]
    
    def test_add_market(self, repository, sample_markets):
        """Тест добавления рынка."""
        market = sample_markets[0]
        
        result = repository.add(market)
        
        assert result == market
        assert repository.get_by_id("market_001") == market
        assert len(repository.get_all()) == 1
    
    def test_get_by_id_existing(self, repository, sample_markets):
        """Тест получения существующего рынка по ID."""
        market = sample_markets[0]
        repository.add(market)
        
        result = repository.get_by_id("market_001")
        
        assert result == market
        assert result.id == "market_001"
        assert result.name == "Bitcoin Market"
        assert result.symbol == "BTCUSD"
    
    def test_get_by_id_not_found(self, repository):
        """Тест получения несуществующего рынка по ID."""
        with pytest.raises(EntityNotFoundError, match="Market with id market_999 not found"):
            repository.get_by_id("market_999")
    
    def test_get_by_symbol_existing(self, repository, sample_markets):
        """Тест получения рынка по символу."""
        market = sample_markets[0]
        repository.add(market)
        
        result = repository.get_by_symbol("BTCUSD")
        
        assert result == market
        assert result.symbol == "BTCUSD"
    
    def test_get_by_symbol_not_found(self, repository):
        """Тест получения рынка по несуществующему символу."""
        with pytest.raises(EntityNotFoundError, match="Market with symbol INVALID not found"):
            repository.get_by_symbol("INVALID")
    
    def test_get_active_markets(self, repository, sample_markets):
        """Тест получения активных рынков."""
        for market in sample_markets:
            repository.add(market)
        
        active_markets = repository.get_active_markets()
        
        assert len(active_markets) == 2
        assert all(market.is_active for market in active_markets)
        assert any(market.symbol == "BTCUSD" for market in active_markets)
        assert any(market.symbol == "ETHUSD" for market in active_markets)
    
    def test_get_markets_by_currency(self, repository, sample_markets):
        """Тест получения рынков по валюте."""
        for market in sample_markets:
            repository.add(market)
        
        # Рынки с базовой валютой BTC
        btc_markets = repository.get_markets_by_base_currency("BTC")
        assert len(btc_markets) == 1
        assert btc_markets[0].base_currency == "BTC"
        
        # Рынки с котируемой валютой USD
        usd_markets = repository.get_markets_by_quote_currency("USD")
        assert len(usd_markets) == 3
        assert all(market.quote_currency == "USD" for market in usd_markets)
    
    def test_get_markets_by_currency_not_found(self, repository, sample_markets):
        """Тест получения рынков по несуществующей валюте."""
        for market in sample_markets:
            repository.add(market)
        
        btc_markets = repository.get_markets_by_base_currency("INVALID")
        assert len(btc_markets) == 0
        
        usd_markets = repository.get_markets_by_quote_currency("INVALID")
        assert len(usd_markets) == 0
    
    def test_update_market(self, repository, sample_markets):
        """Тест обновления рынка."""
        market = sample_markets[0]
        repository.add(market)
        
        updated_market = Market(
            id="market_001",
            name="Updated Bitcoin Market",
            symbol="BTCUSD",
            base_currency="BTC",
            quote_currency="USD",
            is_active=True,
            trading_pairs=market.trading_pairs,
            current_price=Decimal("55000.00"),
            volume_24h=Decimal("1200000.00"),
            price_change_24h=Decimal("10.0"),
            high_24h=Decimal("56000.00"),
            low_24h=Decimal("54000.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        result = repository.update(updated_market)
        
        assert result == updated_market
        stored = repository.get_by_id("market_001")
        assert stored.name == "Updated Bitcoin Market"
        assert stored.current_price == Decimal("55000.00")
    
    def test_update_market_not_found(self, repository, sample_markets):
        """Тест обновления несуществующего рынка."""
        market = Market(
            id="market_999",
            name="Non-existent Market",
            symbol="INVALID",
            base_currency="INV",
            quote_currency="USD",
            is_active=True,
            trading_pairs=[],
            current_price=Decimal("100.00"),
            volume_24h=Decimal("0.00"),
            price_change_24h=Decimal("0.0"),
            high_24h=Decimal("100.00"),
            low_24h=Decimal("100.00"),
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(EntityNotFoundError, match="Market with id market_999 not found"):
            repository.update(market)
    
    def test_delete_market(self, repository, sample_markets):
        """Тест удаления рынка."""
        market = sample_markets[0]
        repository.add(market)
        
        result = repository.delete("market_001")
        
        assert result == market
        with pytest.raises(EntityNotFoundError):
            repository.get_by_id("market_001")
        assert len(repository.get_all()) == 0
    
    def test_delete_market_not_found(self, repository):
        """Тест удаления несуществующего рынка."""
        with pytest.raises(EntityNotFoundError, match="Market with id market_999 not found"):
            repository.delete("market_999")
    
    def test_get_markets_by_price_range(self, repository, sample_markets):
        """Тест получения рынков по диапазону цен."""
        for market in sample_markets:
            repository.add(market)
        
        # Рынки с ценой от 1000 до 10000
        markets_in_range = repository.get_markets_by_price_range(
            min_price=Decimal("1000.00"),
            max_price=Decimal("10000.00")
        )
        
        assert len(markets_in_range) == 1
        assert markets_in_range[0].symbol == "ETHUSD"
    
    def test_get_markets_by_volume_range(self, repository, sample_markets):
        """Тест получения рынков по диапазону объема."""
        for market in sample_markets:
            repository.add(market)
        
        # Рынки с объемом более 500000
        high_volume_markets = repository.get_markets_by_volume_range(
            min_volume=Decimal("500000.00"),
            max_volume=None
        )
        
        assert len(high_volume_markets) == 2
        assert any(market.symbol == "BTCUSD" for market in high_volume_markets)
        assert any(market.symbol == "ETHUSD" for market in high_volume_markets)
    
    def test_get_top_markets_by_volume(self, repository, sample_markets):
        """Тест получения топ рынков по объему."""
        for market in sample_markets:
            repository.add(market)
        
        top_markets = repository.get_top_markets_by_volume(limit=2)
        
        assert len(top_markets) == 2
        # BTCUSD должен быть первым (наибольший объем)
        assert top_markets[0].symbol == "BTCUSD"
        assert top_markets[1].symbol == "ETHUSD"
    
    def test_get_top_markets_by_price_change(self, repository, sample_markets):
        """Тест получения топ рынков по изменению цены."""
        for market in sample_markets:
            repository.add(market)
        
        top_gainers = repository.get_top_markets_by_price_change(limit=2, positive_only=True)
        top_losers = repository.get_top_markets_by_price_change(limit=2, positive_only=False)
        
        assert len(top_gainers) == 2
        assert len(top_losers) == 1  # Только ETHUSD имеет отрицательное изменение
        
        # BTCUSD должен быть первым (наибольший рост)
        assert top_gainers[0].symbol == "BTCUSD"
        assert top_losers[0].symbol == "ETHUSD"
    
    def test_search_markets_by_name(self, repository, sample_markets):
        """Тест поиска рынков по названию."""
        for market in sample_markets:
            repository.add(market)
        
        bitcoin_markets = repository.search_markets_by_name("Bitcoin")
        ethereum_markets = repository.search_markets_by_name("Ethereum")
        market_markets = repository.search_markets_by_name("Market")
        
        assert len(bitcoin_markets) == 1
        assert len(ethereum_markets) == 1
        assert len(market_markets) == 3  # Все рынки содержат "Market"
    
    def test_get_markets_with_trading_pairs(self, repository, sample_markets):
        """Тест получения рынков с торговыми парами."""
        for market in sample_markets:
            repository.add(market)
        
        markets_with_pairs = repository.get_markets_with_trading_pairs()
        
        assert len(markets_with_pairs) == 1
        assert markets_with_pairs[0].symbol == "BTCUSD"
        assert len(markets_with_pairs[0].trading_pairs) > 0
    
    def test_add_trading_pair_to_market(self, repository, sample_markets, sample_trading_pair):
        """Тест добавления торговой пары к рынку."""
        market = sample_markets[1]  # ETHUSD без торговых пар
        repository.add(market)
        
        new_trading_pair = TradingPair(
            base_currency="ETH",
            quote_currency="USD",
            min_order_size=Decimal("0.01"),
            max_order_size=Decimal("1000.0"),
            price_precision=2,
            quantity_precision=6,
            min_notional=Decimal("10.00"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.000001")
        )
        
        repository.add_trading_pair_to_market("market_002", new_trading_pair)
        
        updated_market = repository.get_by_id("market_002")
        assert len(updated_market.trading_pairs) == 1
        assert updated_market.trading_pairs[0] == new_trading_pair
    
    def test_add_trading_pair_to_market_not_found(self, repository, sample_trading_pair):
        """Тест добавления торговой пары к несуществующему рынку."""
        with pytest.raises(EntityNotFoundError, match="Market with id market_999 not found"):
            repository.add_trading_pair_to_market("market_999", sample_trading_pair)
    
    def test_remove_trading_pair_from_market(self, repository, sample_markets):
        """Тест удаления торговой пары с рынка."""
        market = sample_markets[0]  # BTCUSD с торговыми парами
        repository.add(market)
        
        initial_pair_count = len(market.trading_pairs)
        repository.remove_trading_pair_from_market("market_001", market.trading_pairs[0])
        
        updated_market = repository.get_by_id("market_001")
        assert len(updated_market.trading_pairs) == initial_pair_count - 1
    
    def test_update_market_price(self, repository, sample_markets):
        """Тест обновления цены рынка."""
        market = sample_markets[0]
        repository.add(market)
        
        new_price = Decimal("55000.00")
        repository.update_market_price("market_001", new_price)
        
        updated_market = repository.get_by_id("market_001")
        assert updated_market.current_price == new_price
    
    def test_update_market_volume(self, repository, sample_markets):
        """Тест обновления объема рынка."""
        market = sample_markets[0]
        repository.add(market)
        
        new_volume = Decimal("1500000.00")
        repository.update_market_volume("market_001", new_volume)
        
        updated_market = repository.get_by_id("market_001")
        assert updated_market.volume_24h == new_volume
    
    def test_get_markets_by_activity_status(self, repository, sample_markets):
        """Тест получения рынков по статусу активности."""
        for market in sample_markets:
            repository.add(market)
        
        active_markets = repository.get_markets_by_activity_status(True)
        inactive_markets = repository.get_markets_by_activity_status(False)
        
        assert len(active_markets) == 2
        assert len(inactive_markets) == 1
        assert all(market.is_active for market in active_markets)
        assert all(not market.is_active for market in inactive_markets)
    
    def test_get_markets_by_timestamp_range(self, repository, sample_markets):
        """Тест получения рынков по диапазону времени."""
        for market in sample_markets:
            repository.add(market)
        
        # Получаем рынки за последний час
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(hour=end_time.hour - 1)
        
        markets_in_range = repository.get_markets_by_timestamp_range(start_time, end_time)
        
        assert len(markets_in_range) == 3  # Все рынки созданы в последний час
    
    def test_get_market_statistics(self, repository, sample_markets):
        """Тест получения статистики рынков."""
        for market in sample_markets:
            repository.add(market)
        
        stats = repository.get_market_statistics()
        
        assert isinstance(stats, dict)
        assert "total_markets" in stats
        assert "active_markets" in stats
        assert "total_volume" in stats
        assert "average_price" in stats
        
        assert stats["total_markets"] == 3
        assert stats["active_markets"] == 2
    
    def test_exists_by_symbol(self, repository, sample_markets):
        """Тест проверки существования рынка по символу."""
        market = sample_markets[0]
        repository.add(market)
        
        assert repository.exists_by_symbol("BTCUSD") is True
        assert repository.exists_by_symbol("INVALID") is False
    
    def test_get_markets_by_multiple_criteria(self, repository, sample_markets):
        """Тест получения рынков по множественным критериям."""
        for market in sample_markets:
            repository.add(market)
        
        # Активные рынки с объемом более 500000
        criteria_markets = repository.get_markets_by_multiple_criteria(
            is_active=True,
            min_volume=Decimal("500000.00")
        )
        
        assert len(criteria_markets) == 2
        assert all(market.is_active and market.volume_24h >= Decimal("500000.00") 
                  for market in criteria_markets)
    
    def test_bulk_update_market_prices(self, repository, sample_markets):
        """Тест массового обновления цен рынков."""
        for market in sample_markets:
            repository.add(market)
        
        price_updates = {
            "market_001": Decimal("55000.00"),
            "market_002": Decimal("3600.00"),
            "market_003": Decimal("160.00")
        }
        
        repository.bulk_update_market_prices(price_updates)
        
        for market_id, new_price in price_updates.items():
            market = repository.get_by_id(market_id)
            assert market.current_price == new_price
    
    def test_get_markets_by_currency_pair(self, repository, sample_markets):
        """Тест получения рынков по паре валют."""
        for market in sample_markets:
            repository.add(market)
        
        btc_usd_markets = repository.get_markets_by_currency_pair("BTC", "USD")
        eth_usd_markets = repository.get_markets_by_currency_pair("ETH", "USD")
        
        assert len(btc_usd_markets) == 1
        assert len(eth_usd_markets) == 1
        assert btc_usd_markets[0].symbol == "BTCUSD"
        assert eth_usd_markets[0].symbol == "ETHUSD" 