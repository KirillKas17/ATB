"""
Unit тесты для market_entity.

Покрывает:
- Класс Market
- Все методы и свойства
- Сериализацию и десериализацию
- Соответствие протоколу MarketProtocol
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from domain.market.market_entity import Market


class TestMarket:
    """Тесты для класса Market."""

    @pytest.fixture
    def sample_market(self) -> Market:
        """Тестовый рынок."""
        return Market(
            id="test-market-1",
            symbol="BTC/USD",
            name="Bitcoin/US Dollar",
            is_active=True,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"source": "binance", "exchange": "binance", "extra": {"test": "value"}}
        )

    @pytest.fixture
    def inactive_market(self) -> Market:
        """Неактивный рынок."""
        return Market(
            symbol="ETH/USD",
            name="Ethereum/US Dollar",
            is_active=False
        )

    def test_market_creation(self, sample_market):
        """Тест создания рынка."""
        assert sample_market.id == "test-market-1"
        assert sample_market.symbol == "BTC/USD"
        assert sample_market.name == "Bitcoin/US Dollar"
        assert sample_market.is_active is True
        assert sample_market.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert sample_market.updated_at == datetime(2024, 1, 1, 12, 0, 0)
        assert sample_market.metadata["source"] == "binance"
        assert sample_market.metadata["exchange"] == "binance"
        assert sample_market.metadata["extra"]["test"] == "value"

    def test_market_defaults(self):
        """Тест значений по умолчанию."""
        market = Market()
        
        assert market.symbol == ""
        assert market.name == ""
        assert market.is_active is True
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
        assert market.metadata == {"source": "", "exchange": "", "extra": {}}

    def test_market_with_custom_id(self):
        """Тест создания рынка с пользовательским ID."""
        market = Market(id="custom-id", symbol="BTC/USD")
        
        assert market.id == "custom-id"
        assert market.symbol == "BTC/USD"

    def test_market_inactive(self, inactive_market):
        """Тест неактивного рынка."""
        assert inactive_market.symbol == "ETH/USD"
        assert inactive_market.name == "Ethereum/US Dollar"
        assert inactive_market.is_active is False

    def test_to_dict(self, sample_market):
        """Тест преобразования в словарь."""
        result = sample_market.to_dict()
        
        assert result["id"] == "test-market-1"
        assert result["symbol"] == "BTC/USD"
        assert result["name"] == "Bitcoin/US Dollar"
        assert result["is_active"] is True
        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["updated_at"] == "2024-01-01T12:00:00"
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["exchange"] == "binance"
        assert result["metadata"]["extra"]["test"] == "value"

    def test_from_dict(self):
        """Тест создания из словаря."""
        data = {
            "id": "test-market-2",
            "symbol": "ETH/USD",
            "name": "Ethereum/US Dollar",
            "is_active": False,
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "metadata": {"source": "coinbase", "exchange": "coinbase", "extra": {"test": "value2"}}
        }
        
        market = Market.from_dict(data)
        
        assert market.id == "test-market-2"
        assert market.symbol == "ETH/USD"
        assert market.name == "Ethereum/US Dollar"
        assert market.is_active is False
        assert market.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert market.updated_at == datetime(2024, 1, 1, 12, 0, 0)
        assert market.metadata["source"] == "coinbase"
        assert market.metadata["exchange"] == "coinbase"
        assert market.metadata["extra"]["test"] == "value2"

    def test_from_dict_with_optional_fields(self):
        """Тест создания из словаря с опциональными полями."""
        data = {
            "symbol": "ADA/USD",
            "name": "Cardano/US Dollar"
        }
        
        market = Market.from_dict(data)
        
        assert market.symbol == "ADA/USD"
        assert market.name == "Cardano/US Dollar"
        assert market.is_active is True  # значение по умолчанию
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
        assert market.metadata == {"source": "", "exchange": "", "extra": {}}

    def test_from_dict_with_missing_timestamps(self):
        """Тест создания из словаря с отсутствующими временными метками."""
        data = {
            "symbol": "DOT/USD",
            "name": "Polkadot/US Dollar"
        }
        
        market = Market.from_dict(data)
        
        assert market.symbol == "DOT/USD"
        assert market.name == "Polkadot/US Dollar"
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)

    def test_from_dict_with_string_boolean(self):
        """Тест создания из словаря со строковым булевым значением."""
        data = {
            "symbol": "LINK/USD",
            "name": "Chainlink/US Dollar",
            "is_active": "false"
        }
        
        market = Market.from_dict(data)
        
        assert market.symbol == "LINK/USD"
        assert market.name == "Chainlink/US Dollar"
        assert market.is_active == "false"  # остается строкой

    def test_unique_id_generation(self):
        """Тест генерации уникальных ID."""
        market1 = Market()
        market2 = Market()
        
        assert market1.id != market2.id
        assert isinstance(market1.id, str)
        assert isinstance(market2.id, str)

    def test_market_equality(self):
        """Тест равенства рынков."""
        market1 = Market(id="same-id", symbol="BTC/USD")
        market2 = Market(id="same-id", symbol="BTC/USD")
        market3 = Market(id="different-id", symbol="BTC/USD")
        
        # Рынки с одинаковым ID должны быть равны
        assert market1.id == market2.id
        # Рынки с разным ID должны быть разными
        assert market1.id != market3.id

    def test_market_metadata_immutability(self):
        """Тест неизменяемости метаданных."""
        market = Market(symbol="BTC/USD")
        
        # Метаданные должны быть словарем
        assert isinstance(market.metadata, dict)
        assert "source" in market.metadata
        assert "exchange" in market.metadata
        assert "extra" in market.metadata

    def test_market_timestamp_consistency(self):
        """Тест консистентности временных меток."""
        market = Market(symbol="BTC/USD")
        
        # Временные метки должны быть datetime объектами
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
        
        # updated_at не должен быть раньше created_at
        assert market.updated_at >= market.created_at

    def test_protocol_compliance(self, sample_market):
        """Тест соответствия протоколу MarketProtocol."""
        from domain.market.market_protocols import MarketProtocol
        assert isinstance(sample_market, MarketProtocol)

    def test_market_serialization_roundtrip(self, sample_market):
        """Тест полного цикла сериализации и десериализации."""
        # Преобразуем в словарь
        market_dict = sample_market.to_dict()
        
        # Создаем новый объект из словаря
        restored_market = Market.from_dict(market_dict)
        
        # Проверяем что все поля восстановлены
        assert restored_market.id == sample_market.id
        assert restored_market.symbol == sample_market.symbol
        assert restored_market.name == sample_market.name
        assert restored_market.is_active == sample_market.is_active
        assert restored_market.created_at == sample_market.created_at
        assert restored_market.updated_at == sample_market.updated_at
        assert restored_market.metadata == sample_market.metadata

    def test_market_with_minimal_data(self):
        """Тест создания рынка с минимальными данными."""
        market = Market(symbol="XRP/USD")
        
        assert market.symbol == "XRP/USD"
        assert market.name == ""
        assert market.is_active is True
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)

    def test_market_with_full_data(self):
        """Тест создания рынка с полными данными."""
        created_at = datetime(2024, 1, 1, 10, 0, 0)
        updated_at = datetime(2024, 1, 1, 12, 0, 0)
        
        market = Market(
            id="full-market",
            symbol="SOL/USD",
            name="Solana/US Dollar",
            is_active=True,
            created_at=created_at,
            updated_at=updated_at,
            metadata={
                "source": "kraken",
                "exchange": "kraken",
                "extra": {
                    "market_cap": 1000000000,
                    "volume_24h": 50000000
                }
            }
        )
        
        assert market.id == "full-market"
        assert market.symbol == "SOL/USD"
        assert market.name == "Solana/US Dollar"
        assert market.is_active is True
        assert market.created_at == created_at
        assert market.updated_at == updated_at
        assert market.metadata["source"] == "kraken"
        assert market.metadata["extra"]["market_cap"] == 1000000000
        assert market.metadata["extra"]["volume_24h"] == 50000000


class TestMarketIntegration:
    """Интеграционные тесты для Market."""

    def test_multiple_markets_creation(self):
        """Тест создания множественных рынков."""
        markets = [
            Market(symbol="BTC/USD", name="Bitcoin/US Dollar"),
            Market(symbol="ETH/USD", name="Ethereum/US Dollar"),
            Market(symbol="ADA/USD", name="Cardano/US Dollar"),
            Market(symbol="DOT/USD", name="Polkadot/US Dollar")
        ]
        
        assert len(markets) == 4
        assert all(isinstance(market, Market) for market in markets)
        assert all(market.is_active is True for market in markets)
        assert all(isinstance(market.created_at, datetime) for market in markets)
        
        # Проверяем уникальность ID
        ids = [market.id for market in markets]
        assert len(set(ids)) == len(ids)

    def test_market_lifecycle(self):
        """Тест жизненного цикла рынка."""
        # Создание активного рынка
        market = Market(
            symbol="MATIC/USD",
            name="Polygon/US Dollar",
            is_active=True
        )
        
        assert market.is_active is True
        
        # Деактивация рынка
        market.is_active = False
        assert market.is_active is False
        
        # Обновление временной метки
        old_updated_at = market.updated_at
        market.updated_at = datetime.now()
        assert market.updated_at > old_updated_at

    def test_market_data_consistency(self):
        """Тест консистентности данных рынка."""
        # Создаем рынок с определенными данными
        market = Market(
            symbol="AVAX/USD",
            name="Avalanche/US Dollar",
            is_active=True
        )
        
        # Проверяем консистентность
        assert market.symbol == "AVAX/USD"
        assert market.name == "Avalanche/US Dollar"
        assert market.is_active is True
        
        # Сериализуем и десериализуем
        market_dict = market.to_dict()
        restored_market = Market.from_dict(market_dict)
        
        # Проверяем что данные сохранились
        assert restored_market.symbol == market.symbol
        assert restored_market.name == market.name
        assert restored_market.is_active == market.is_active

    def test_market_metadata_operations(self):
        """Тест операций с метаданными рынка."""
        market = Market(symbol="FTM/USD")
        
        # Проверяем структуру метаданных по умолчанию
        assert "source" in market.metadata
        assert "exchange" in market.metadata
        assert "extra" in market.metadata
        
        # Добавляем пользовательские метаданные
        market.metadata["extra"]["custom_field"] = "custom_value"
        market.metadata["source"] = "custom_source"
        
        # Проверяем что метаданные сохранились
        assert market.metadata["extra"]["custom_field"] == "custom_value"
        assert market.metadata["source"] == "custom_source"
        
        # Сериализуем и проверяем
        market_dict = market.to_dict()
        assert market_dict["metadata"]["extra"]["custom_field"] == "custom_value"
        assert market_dict["metadata"]["source"] == "custom_source" 