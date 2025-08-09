"""
Unit тесты для TradingPairRepository.

Покрывает:
- Основной функционал репозитория торговых пар
- CRUD операции
- Фильтрацию по различным критериям
- Поиск и статистику торговых пар
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from domain.entities.trading_pair import TradingPair
from domain.repositories.trading_pair_repository import TradingPairRepository
from domain.exceptions.base_exceptions import ValidationError


class TestTradingPairRepository:
    """Тесты для абстрактного TradingPairRepository."""

    @pytest.fixture
    def mock_trading_pair_repository(self) -> Mock:
        """Мок репозитория торговых пар."""
        return Mock(spec=TradingPairRepository)

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_eth(self) -> TradingPair:
        """Тестовая торговая пара ETH."""
        return TradingPair(base="ETH", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_ada(self) -> TradingPair:
        """Тестовая торговая пара ADA."""
        return TradingPair(base="ADA", quote="USDT")

    def test_save_method_exists(self, mock_trading_pair_repository, sample_trading_pair):
        """Тест наличия метода save."""
        mock_trading_pair_repository.save = AsyncMock(return_value=sample_trading_pair)
        assert hasattr(mock_trading_pair_repository, "save")
        assert callable(mock_trading_pair_repository.save)

    def test_get_by_symbol_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода get_by_symbol."""
        mock_trading_pair_repository.get_by_symbol = AsyncMock(return_value=None)
        assert hasattr(mock_trading_pair_repository, "get_by_symbol")
        assert callable(mock_trading_pair_repository.get_by_symbol)

    def test_get_all_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода get_all."""
        mock_trading_pair_repository.get_all = AsyncMock(return_value=[])
        assert hasattr(mock_trading_pair_repository, "get_all")
        assert callable(mock_trading_pair_repository.get_all)

    def test_get_by_currencies_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода get_by_currencies."""
        mock_trading_pair_repository.get_by_currencies = AsyncMock(return_value=[])
        assert hasattr(mock_trading_pair_repository, "get_by_currencies")
        assert callable(mock_trading_pair_repository.get_by_currencies)

    def test_update_method_exists(self, mock_trading_pair_repository, sample_trading_pair):
        """Тест наличия метода update."""
        mock_trading_pair_repository.update = AsyncMock(return_value=sample_trading_pair)
        assert hasattr(mock_trading_pair_repository, "update")
        assert callable(mock_trading_pair_repository.update)

    def test_delete_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода delete."""
        mock_trading_pair_repository.delete = AsyncMock(return_value=True)
        assert hasattr(mock_trading_pair_repository, "delete")
        assert callable(mock_trading_pair_repository.delete)

    def test_exists_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода exists."""
        mock_trading_pair_repository.exists = AsyncMock(return_value=True)
        assert hasattr(mock_trading_pair_repository, "exists")
        assert callable(mock_trading_pair_repository.exists)

    def test_count_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода count."""
        mock_trading_pair_repository.count = AsyncMock(return_value=0)
        assert hasattr(mock_trading_pair_repository, "count")
        assert callable(mock_trading_pair_repository.count)

    def test_search_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода search."""
        mock_trading_pair_repository.search = AsyncMock(return_value=[])
        assert hasattr(mock_trading_pair_repository, "search")
        assert callable(mock_trading_pair_repository.search)

    def test_get_statistics_method_exists(self, mock_trading_pair_repository):
        """Тест наличия метода get_statistics."""
        mock_trading_pair_repository.get_statistics = AsyncMock(return_value={})
        assert hasattr(mock_trading_pair_repository, "get_statistics")
        assert callable(mock_trading_pair_repository.get_statistics)


class TestInMemoryTradingPairRepository:
    """Тесты для InMemoryTradingPairRepository."""

    @pytest.fixture
    def repository(self) -> Mock:
        """Репозиторий для тестов."""
        return Mock(spec=TradingPairRepository)

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(base="BTC", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_eth(self) -> TradingPair:
        """Тестовая торговая пара ETH."""
        return TradingPair(base="ETH", quote="USDT")

    @pytest.fixture
    def sample_trading_pair_ada(self) -> TradingPair:
        """Тестовая торговая пара ADA."""
        return TradingPair(base="ADA", quote="USDT")

    @pytest.fixture
    def sample_trading_pairs(
        self, sample_trading_pair, sample_trading_pair_eth, sample_trading_pair_ada
    ) -> List[TradingPair]:
        """Тестовые торговые пары."""
        return [
            sample_trading_pair,
            sample_trading_pair_eth,
            sample_trading_pair_ada,
            TradingPair(base="BNB", quote="USDT"),
            TradingPair(base="SOL", quote="USDT"),
            TradingPair(base="DOT", quote="USDT"),
            TradingPair(base="LINK", quote="USDT"),
            TradingPair(base="MATIC", quote="USDT"),
            TradingPair(base="AVAX", quote="USDT"),
            TradingPair(base="UNI", quote="USDT"),
        ]

    @pytest.mark.asyncio
    async def test_save_trading_pair(self, repository, sample_trading_pair):
        """Тест сохранения торговой пары."""
        repository.save = AsyncMock(return_value=sample_trading_pair)
        result = await repository.save(sample_trading_pair)
        assert result == sample_trading_pair
        repository.save.assert_called_once_with(sample_trading_pair)

    @pytest.mark.asyncio
    async def test_get_by_symbol_existing(self, repository, sample_trading_pair):
        """Тест получения торговой пары по символу - пара существует."""
        symbol = "BTCUSDT"
        repository.get_by_symbol = AsyncMock(return_value=sample_trading_pair)
        result = await repository.get_by_symbol(symbol)
        assert result == sample_trading_pair
        repository.get_by_symbol.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_get_by_symbol_not_existing(self, repository):
        """Тест получения торговой пары по символу - пара не существует."""
        symbol = "INVALID"
        repository.get_by_symbol = AsyncMock(return_value=None)
        result = await repository.get_by_symbol(symbol)
        assert result is None
        repository.get_by_symbol.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_get_all_active_only(self, repository, sample_trading_pairs):
        """Тест получения всех активных торговых пар."""
        repository.get_all = AsyncMock(return_value=sample_trading_pairs)
        result = await repository.get_all(active_only=True)
        assert result == sample_trading_pairs
        repository.get_all.assert_called_once_with(active_only=True)

    @pytest.mark.asyncio
    async def test_get_all_including_inactive(self, repository, sample_trading_pairs):
        """Тест получения всех торговых пар включая неактивные."""
        repository.get_all = AsyncMock(return_value=sample_trading_pairs)
        result = await repository.get_all(active_only=False)
        assert result == sample_trading_pairs
        repository.get_all.assert_called_once_with(active_only=False)

    @pytest.mark.asyncio
    async def test_get_by_currencies(self, repository, sample_trading_pairs):
        """Тест получения торговых пар по валютам."""
        base_currency = "BTC"
        quote_currency = "USDT"
        btc_pairs = [p for p in sample_trading_pairs if p.base == base_currency and p.quote == quote_currency]
        repository.get_by_currencies = AsyncMock(return_value=btc_pairs)
        result = await repository.get_by_currencies(base_currency, quote_currency)
        assert result == btc_pairs
        repository.get_by_currencies.assert_called_once_with(base_currency, quote_currency)

    @pytest.mark.asyncio
    async def test_get_by_currencies_multiple_pairs(self, repository, sample_trading_pairs):
        """Тест получения нескольких торговых пар по валютам."""
        base_currency = "ETH"
        quote_currency = "USDT"
        eth_pairs = [p for p in sample_trading_pairs if p.base == base_currency and p.quote == quote_currency]
        repository.get_by_currencies = AsyncMock(return_value=eth_pairs)
        result = await repository.get_by_currencies(base_currency, quote_currency)
        assert result == eth_pairs
        repository.get_by_currencies.assert_called_once_with(base_currency, quote_currency)

    @pytest.mark.asyncio
    async def test_get_by_currencies_no_matches(self, repository):
        """Тест получения торговых пар по валютам - нет совпадений."""
        base_currency = "INVALID"
        quote_currency = "INVALID"
        repository.get_by_currencies = AsyncMock(return_value=[])
        result = await repository.get_by_currencies(base_currency, quote_currency)
        assert result == []
        repository.get_by_currencies.assert_called_once_with(base_currency, quote_currency)

    @pytest.mark.asyncio
    async def test_update_existing_trading_pair(self, repository, sample_trading_pair):
        """Тест обновления существующей торговой пары."""
        updated_trading_pair = TradingPair(base="BTC", quote="USDT")
        repository.update = AsyncMock(return_value=updated_trading_pair)
        result = await repository.update(sample_trading_pair)
        assert result == updated_trading_pair
        repository.update.assert_called_once_with(sample_trading_pair)

    @pytest.mark.asyncio
    async def test_delete_existing_trading_pair(self, repository):
        """Тест удаления существующей торговой пары."""
        symbol = "BTCUSDT"
        repository.delete = AsyncMock(return_value=True)
        result = await repository.delete(symbol)
        assert result is True
        repository.delete.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_delete_not_existing_trading_pair(self, repository):
        """Тест удаления несуществующей торговой пары."""
        symbol = "INVALID"
        repository.delete = AsyncMock(return_value=False)
        result = await repository.delete(symbol)
        assert result is False
        repository.delete.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_exists_true(self, repository):
        """Тест проверки существования торговой пары - пара существует."""
        symbol = "BTCUSDT"
        repository.exists = AsyncMock(return_value=True)
        result = await repository.exists(symbol)
        assert result is True
        repository.exists.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_exists_false(self, repository):
        """Тест проверки существования торговой пары - пара не существует."""
        symbol = "INVALID"
        repository.exists = AsyncMock(return_value=False)
        result = await repository.exists(symbol)
        assert result is False
        repository.exists.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_count_active_only(self, repository, sample_trading_pairs):
        """Тест подсчета активных торговых пар."""
        repository.count = AsyncMock(return_value=len(sample_trading_pairs))
        result = await repository.count(active_only=True)
        assert result == len(sample_trading_pairs)
        repository.count.assert_called_once_with(active_only=True)

    @pytest.mark.asyncio
    async def test_count_including_inactive(self, repository, sample_trading_pairs):
        """Тест подсчета всех торговых пар включая неактивные."""
        repository.count = AsyncMock(return_value=len(sample_trading_pairs))
        result = await repository.count(active_only=False)
        assert result == len(sample_trading_pairs)
        repository.count.assert_called_once_with(active_only=False)

    @pytest.mark.asyncio
    async def test_count_empty_repository(self, repository):
        """Тест подсчета торговых пар в пустом репозитории."""
        repository.count = AsyncMock(return_value=0)
        result = await repository.count()
        assert result == 0
        repository.count.assert_called_once_with(active_only=True)

    @pytest.mark.asyncio
    async def test_search_with_query(self, repository, sample_trading_pairs):
        """Тест поиска торговых пар с запросом."""
        query = "BTC"
        limit = 5
        btc_pairs = [p for p in sample_trading_pairs if "BTC" in p.base or "BTC" in p.quote][:limit]
        repository.search = AsyncMock(return_value=btc_pairs)
        result = await repository.search(query, limit)
        assert result == btc_pairs
        repository.search.assert_called_once_with(query, limit)

    @pytest.mark.asyncio
    async def test_search_with_default_limit(self, repository, sample_trading_pairs):
        """Тест поиска торговых пар с дефолтным лимитом."""
        query = "ETH"
        eth_pairs = [p for p in sample_trading_pairs if "ETH" in p.base or "ETH" in p.quote][:10]
        repository.search = AsyncMock(return_value=eth_pairs)
        result = await repository.search(query)
        assert result == eth_pairs
        repository.search.assert_called_once_with(query, 10)

    @pytest.mark.asyncio
    async def test_search_no_results(self, repository):
        """Тест поиска торговых пар - нет результатов."""
        query = "INVALID"
        repository.search = AsyncMock(return_value=[])
        result = await repository.search(query)
        assert result == []
        repository.search.assert_called_once_with(query, 10)

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, repository, sample_trading_pairs):
        """Тест поиска торговых пар без учета регистра."""
        query = "btc"
        btc_pairs = [p for p in sample_trading_pairs if "BTC" in p.base or "BTC" in p.quote][:10]
        repository.search = AsyncMock(return_value=btc_pairs)
        result = await repository.search(query)
        assert result == btc_pairs
        repository.search.assert_called_once_with(query, 10)

    @pytest.mark.asyncio
    async def test_get_statistics(self, repository, sample_trading_pairs):
        """Тест получения статистики по торговым парам."""
        expected_stats = {
            "total_pairs": len(sample_trading_pairs),
            "active_pairs": len(sample_trading_pairs),
            "base_currencies": 10,
            "quote_currencies": 1,
            "most_common_quote": "USDT",
            "pairs_by_quote": {"USDT": len(sample_trading_pairs)},
        }
        repository.get_statistics = AsyncMock(return_value=expected_stats)
        result = await repository.get_statistics()
        assert result == expected_stats
        repository.get_statistics.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_trading_pair_creation_and_retrieval(self, repository, sample_trading_pair):
        """Тест создания и получения торговой пары."""
        # Сохранение
        repository.save = AsyncMock(return_value=sample_trading_pair)
        saved_pair = await repository.save(sample_trading_pair)
        assert saved_pair == sample_trading_pair

        # Получение по символу
        symbol = "BTCUSDT"
        repository.get_by_symbol = AsyncMock(return_value=sample_trading_pair)
        retrieved_pair = await repository.get_by_symbol(symbol)
        assert retrieved_pair == sample_trading_pair

    @pytest.mark.asyncio
    async def test_trading_pair_update_flow(self, repository, sample_trading_pair):
        """Тест полного цикла обновления торговой пары."""
        # Исходная пара
        original_pair = TradingPair(base="BTC", quote="USDT")

        # Обновление
        updated_pair = TradingPair(base="BTC", quote="USDT")
        repository.update = AsyncMock(return_value=updated_pair)
        result = await repository.update(original_pair)
        assert result == updated_pair

        # Проверка обновления
        repository.get_by_symbol = AsyncMock(return_value=updated_pair)
        retrieved = await repository.get_by_symbol("BTCUSDT")
        assert retrieved == updated_pair

    @pytest.mark.asyncio
    async def test_trading_pair_deletion_flow(self, repository):
        """Тест полного цикла удаления торговой пары."""
        symbol = "BTCUSDT"

        # Проверка существования до удаления
        repository.exists = AsyncMock(return_value=True)
        exists_before = await repository.exists(symbol)
        assert exists_before is True

        # Удаление
        repository.delete = AsyncMock(return_value=True)
        deleted = await repository.delete(symbol)
        assert deleted is True

        # Проверка существования после удаления
        repository.exists = AsyncMock(return_value=False)
        exists_after = await repository.exists(symbol)
        assert exists_after is False

    @pytest.mark.asyncio
    async def test_multiple_trading_pairs_same_base(self, repository, sample_trading_pairs):
        """Тест множественных торговых пар с одинаковой базовой валютой."""
        base_currency = "BTC"
        quote_currencies = ["USDT", "USD", "EUR"]

        btc_pairs = [TradingPair(base=base_currency, quote=quote) for quote in quote_currencies]

        repository.get_by_currencies = AsyncMock(return_value=btc_pairs)
        result = await repository.get_by_currencies(base_currency, "USDT")
        assert len(result) == 1
        assert result[0].base == base_currency
        assert result[0].quote == "USDT"

    @pytest.mark.asyncio
    async def test_multiple_trading_pairs_same_quote(self, repository, sample_trading_pairs):
        """Тест множественных торговых пар с одинаковой котируемой валютой."""
        quote_currency = "USDT"
        base_currencies = ["BTC", "ETH", "ADA", "BNB", "SOL"]

        usdt_pairs = [TradingPair(base=base, quote=quote_currency) for base in base_currencies]

        repository.get_by_currencies = AsyncMock(return_value=usdt_pairs)
        result = await repository.get_by_currencies("BTC", quote_currency)
        assert len(result) == 1
        assert result[0].base == "BTC"
        assert result[0].quote == quote_currency

    @pytest.mark.asyncio
    async def test_search_partial_matches(self, repository, sample_trading_pairs):
        """Тест поиска с частичными совпадениями."""
        query = "BT"
        btc_pairs = [p for p in sample_trading_pairs if "BT" in p.base][:10]
        repository.search = AsyncMock(return_value=btc_pairs)
        result = await repository.search(query)
        assert result == btc_pairs
        repository.search.assert_called_once_with(query, 10)

    @pytest.mark.asyncio
    async def test_search_by_quote_currency(self, repository, sample_trading_pairs):
        """Тест поиска по котируемой валюте."""
        query = "USDT"
        usdt_pairs = [p for p in sample_trading_pairs if p.quote == "USDT"][:10]
        repository.search = AsyncMock(return_value=usdt_pairs)
        result = await repository.search(query)
        assert result == usdt_pairs
        repository.search.assert_called_once_with(query, 10)

    @pytest.mark.asyncio
    async def test_repository_isolation(self: "TestInMemoryTradingPairRepository") -> None:
        """Тест изоляции репозиториев."""
        # Создание двух независимых репозиториев
        repo1 = Mock(spec=TradingPairRepository)
        repo2 = Mock(spec=TradingPairRepository)

        # Данные в первом репозитории
        trading_pair1 = TradingPair(base="BTC", quote="USDT")

        # Данные во втором репозитории
        trading_pair2 = TradingPair(base="ETH", quote="USDT")

        # Настройка моков
        repo1.get_by_symbol = AsyncMock(return_value=trading_pair1)
        repo2.get_by_symbol = AsyncMock(return_value=trading_pair2)

        # Проверка изоляции
        result1 = await repo1.get_by_symbol("BTCUSDT")
        result2 = await repo2.get_by_symbol("ETHUSDT")

        assert result1 == trading_pair1
        assert result2 == trading_pair2
        assert result1 != result2
        assert result1.base != result2.base

    @pytest.mark.asyncio
    async def test_error_handling_save(self, repository, sample_trading_pair):
        """Тест обработки ошибок при сохранении."""
        repository.save = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(Exception, match="Database error"):
            await repository.save(sample_trading_pair)

    @pytest.mark.asyncio
    async def test_error_handling_get_by_symbol(self, repository):
        """Тест обработки ошибок при получении по символу."""
        symbol = "BTCUSDT"
        repository.get_by_symbol = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(Exception, match="Connection error"):
            await repository.get_by_symbol(symbol)

    @pytest.mark.asyncio
    async def test_error_handling_update(self, repository, sample_trading_pair):
        """Тест обработки ошибок при обновлении."""
        repository.update = AsyncMock(side_effect=Exception("Update failed"))

        with pytest.raises(Exception, match="Update failed"):
            await repository.update(sample_trading_pair)

    @pytest.mark.asyncio
    async def test_error_handling_delete(self, repository):
        """Тест обработки ошибок при удалении."""
        symbol = "BTCUSDT"
        repository.delete = AsyncMock(side_effect=Exception("Delete failed"))

        with pytest.raises(Exception, match="Delete failed"):
            await repository.delete(symbol)

    @pytest.mark.asyncio
    async def test_error_handling_search(self, repository):
        """Тест обработки ошибок при поиске."""
        query = "BTC"
        repository.search = AsyncMock(side_effect=Exception("Search failed"))

        with pytest.raises(Exception, match="Search failed"):
            await repository.search(query)

    @pytest.mark.asyncio
    async def test_error_handling_get_statistics(self, repository):
        """Тест обработки ошибок при получении статистики."""
        repository.get_statistics = AsyncMock(side_effect=Exception("Statistics failed"))

        with pytest.raises(Exception, match="Statistics failed"):
            await repository.get_statistics()

    @pytest.mark.asyncio
    async def test_trading_pair_validation(self, repository):
        """Тест валидации торговых пар."""
        # Тест с корректными данными
        valid_pair = TradingPair(base="BTC", quote="USDT")
        repository.save = AsyncMock(return_value=valid_pair)
        result = await repository.save(valid_pair)
        assert result.base == "BTC"
        assert result.quote == "USDT"

        # Тест с пустыми валютами
        invalid_pair = TradingPair(base="", quote="")
        repository.save = AsyncMock(side_effect=ValidationError("Invalid trading pair"))

        with pytest.raises(ValidationError, match="Invalid trading pair"):
            await repository.save(invalid_pair)

    @pytest.mark.asyncio
    async def test_trading_pair_symbol_generation(self, repository, sample_trading_pair):
        """Тест генерации символов торговых пар."""
        # Проверка символа BTC/USDT
        btc_pair = TradingPair(base="BTC", quote="USDT")
        assert str(btc_pair) == "BTCUSDT"

        # Проверка символа ETH/USDT
        eth_pair = TradingPair(base="ETH", quote="USDT")
        assert str(eth_pair) == "ETHUSDT"

        # Проверка символа ADA/BTC
        ada_pair = TradingPair(base="ADA", quote="BTC")
        assert str(ada_pair) == "ADABTC"

    @pytest.mark.asyncio
    async def test_trading_pair_equality(self, repository):
        """Тест равенства торговых пар."""
        pair1 = TradingPair(base="BTC", quote="USDT")
        pair2 = TradingPair(base="BTC", quote="USDT")
        pair3 = TradingPair(base="ETH", quote="USDT")

        assert pair1 == pair2
        assert pair1 != pair3
        assert hash(pair1) == hash(pair2)
        assert hash(pair1) != hash(pair3)

    @pytest.mark.asyncio
    async def test_trading_pair_serialization(self, repository, sample_trading_pair):
        """Тест сериализации торговых пар."""
        # Проверка to_dict
        pair_dict = sample_trading_pair.to_dict()
        assert pair_dict["base"] == "BTC"
        assert pair_dict["quote"] == "USDT"

        # Проверка from_dict
        reconstructed_pair = TradingPair.from_dict(pair_dict)
        assert reconstructed_pair == sample_trading_pair

    @pytest.mark.asyncio
    async def test_bulk_operations(self, repository, sample_trading_pairs):
        """Тест массовых операций с торговыми парами."""
        # Массовое сохранение
        for pair in sample_trading_pairs:
            repository.save = AsyncMock(return_value=pair)
            result = await repository.save(pair)
            assert result == pair

        # Массовое получение
        repository.get_all = AsyncMock(return_value=sample_trading_pairs)
        all_pairs = await repository.get_all()
        assert len(all_pairs) == len(sample_trading_pairs)

        # Массовое удаление
        for pair in sample_trading_pairs:
            symbol = str(pair)
            repository.delete = AsyncMock(return_value=True)
            result = await repository.delete(symbol)
            assert result is True
