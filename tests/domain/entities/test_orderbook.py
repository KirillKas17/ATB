"""
Тесты для сущности OrderBookSnapshot.
"""

import pytest
from decimal import Decimal
from typing import Any

from domain.entities.orderbook import OrderBookSnapshot
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.type_definitions import MetadataDict


class TestOrderBookSnapshot:
    """Тесты для снимка ордербука."""

    @pytest.fixture
    def sample_bids(self) -> list[tuple[Price, Volume]]:
        """Фикстура с примерами покупок."""
        return [
            (Price(Decimal("50000"), Currency.USD), Volume(Decimal("1.5"), Currency.USD)),
            (Price(Decimal("49900"), Currency.USD), Volume(Decimal("2.0"), Currency.USD)),
            (Price(Decimal("49800"), Currency.USD), Volume(Decimal("1.0"), Currency.USD)),
        ]

    @pytest.fixture
    def sample_asks(self) -> list[tuple[Price, Volume]]:
        """Фикстура с примерами продаж."""
        return [
            (Price(Decimal("50100"), Currency.USD), Volume(Decimal("1.0"), Currency.USD)),
            (Price(Decimal("50200"), Currency.USD), Volume(Decimal("2.5"), Currency.USD)),
            (Price(Decimal("50300"), Currency.USD), Volume(Decimal("1.8"), Currency.USD)),
        ]

    @pytest.fixture
    def orderbook_snapshot(self, sample_bids: list[tuple[Price, Volume]], sample_asks: list[tuple[Price, Volume]]) -> OrderBookSnapshot:
        """Фикстура с валидным снимком ордербука."""
        return OrderBookSnapshot(
            exchange="test_exchange",
            symbol="BTC/USD",
            bids=sample_bids,
            asks=sample_asks,
            timestamp=Timestamp(1640995200),
            sequence_id=12345,
            meta=MetadataDict({"filtered": False, "noise_analysis": {"level": "low"}}),
        )

    def test_orderbook_creation(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест создания снимка ордербука."""
        assert orderbook_snapshot.exchange == "test_exchange"
        assert orderbook_snapshot.symbol == "BTC/USD"
        assert len(orderbook_snapshot.bids) == 3
        assert len(orderbook_snapshot.asks) == 3
        assert orderbook_snapshot.sequence_id == 12345
        assert orderbook_snapshot.meta["filtered"] is False

    def test_bids_sorting(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест сортировки покупок по убыванию цены."""
        bid_prices = [bid[0].value for bid in orderbook_snapshot.bids]
        assert bid_prices == [Decimal("50000"), Decimal("49900"), Decimal("49800")]

    def test_asks_sorting(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест сортировки продаж по возрастанию цены."""
        ask_prices = [ask[0].value for ask in orderbook_snapshot.asks]
        assert ask_prices == [Decimal("50100"), Decimal("50200"), Decimal("50300")]

    def test_best_bid(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения лучшей цены покупки."""
        best_bid = orderbook_snapshot.best_bid
        assert best_bid is not None
        assert best_bid[0].value == Decimal("50000")
        assert best_bid[1].value == Decimal("1.5")

    def test_best_ask(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения лучшей цены продажи."""
        best_ask = orderbook_snapshot.best_ask
        assert best_ask is not None
        assert best_ask[0].value == Decimal("50100")
        assert best_ask[1].value == Decimal("1.0")

    def test_best_bid_empty_bids(self) -> None:
        """Тест лучшей цены покупки при пустых покупках."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[(Price(Decimal("50100"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.best_bid is None

    def test_best_ask_empty_asks(self) -> None:
        """Тест лучшей цены продажи при пустых продажах."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[(Price(Decimal("50000"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.best_ask is None

    def test_mid_price(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета средней цены."""
        mid_price = orderbook_snapshot.mid_price
        assert mid_price is not None
        expected_mid = (Decimal("50000") + Decimal("50100")) / 2
        assert mid_price.value == expected_mid
        assert mid_price.currency == Currency.USD

    def test_mid_price_no_bids(self) -> None:
        """Тест средней цены при отсутствии покупок."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[(Price(Decimal("50100"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.mid_price is None

    def test_mid_price_no_asks(self) -> None:
        """Тест средней цены при отсутствии продаж."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[(Price(Decimal("50000"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.mid_price is None

    def test_spread(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета спреда."""
        spread = orderbook_snapshot.spread
        assert spread is not None
        expected_spread = Decimal("50100") - Decimal("50000")
        assert spread == expected_spread

    def test_spread_percentage(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета спреда в процентах."""
        spread_percentage = orderbook_snapshot.spread_percentage
        assert spread_percentage is not None
        expected_percentage = (Decimal("100") / Decimal("50050")) * 100
        assert abs(spread_percentage - expected_percentage) < Decimal("0.01")

    def test_total_bid_volume(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета общего объема покупок."""
        total_bid_volume = orderbook_snapshot.total_bid_volume
        expected_total = Decimal("1.5") + Decimal("2.0") + Decimal("1.0")
        assert total_bid_volume.value == expected_total
        assert total_bid_volume.currency == Currency.USD

    def test_total_bid_volume_empty(self) -> None:
        """Тест общего объема покупок при пустых покупках."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[(Price(Decimal("50100"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            timestamp=Timestamp(1640995200),
        )
        total_bid_volume = orderbook.total_bid_volume
        assert total_bid_volume.value == Decimal("0")
        assert total_bid_volume.currency == Currency.USD

    def test_total_ask_volume(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета общего объема продаж."""
        total_ask_volume = orderbook_snapshot.total_ask_volume
        expected_total = Decimal("1.0") + Decimal("2.5") + Decimal("1.8")
        assert total_ask_volume.value == expected_total
        assert total_ask_volume.currency == Currency.USD

    def test_total_ask_volume_empty(self) -> None:
        """Тест общего объема продаж при пустых продажах."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[(Price(Decimal("50000"), Currency.USD), Volume(Decimal("1.0"), Currency.USD))],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        total_ask_volume = orderbook.total_ask_volume
        assert total_ask_volume.value == Decimal("0")
        assert total_ask_volume.currency == Currency.USD

    def test_volume_imbalance(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета дисбаланса объемов."""
        imbalance = orderbook_snapshot.volume_imbalance
        bid_total = Decimal("1.5") + Decimal("2.0") + Decimal("1.0")  # 4.5
        ask_total = Decimal("1.0") + Decimal("2.5") + Decimal("1.8")  # 5.3
        expected_imbalance = bid_total - ask_total
        assert imbalance == expected_imbalance

    def test_volume_imbalance_ratio(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест расчета отношения дисбаланса объемов."""
        ratio = orderbook_snapshot.volume_imbalance_ratio
        assert ratio is not None
        bid_total = Decimal("1.5") + Decimal("2.0") + Decimal("1.0")  # 4.5
        ask_total = Decimal("1.0") + Decimal("2.5") + Decimal("1.8")  # 5.3
        total_volume = bid_total + ask_total
        expected_ratio = (bid_total - ask_total) / total_volume
        assert abs(ratio - expected_ratio) < Decimal("0.001")

    def test_volume_imbalance_ratio_zero_total(self) -> None:
        """Тест отношения дисбаланса при нулевом общем объеме."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.volume_imbalance_ratio is None

    def test_get_bid_volume_at_price(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения объема покупок по цене."""
        # Цена выше всех покупок
        volume = orderbook_snapshot.get_bid_volume_at_price(Price(Decimal("51000"), Currency.USD))
        assert volume.value == Decimal("0")

        # Цена равна лучшей покупке
        volume = orderbook_snapshot.get_bid_volume_at_price(Price(Decimal("50000"), Currency.USD))
        assert volume.value == Decimal("1.5")

        # Цена между покупками
        volume = orderbook_snapshot.get_bid_volume_at_price(Price(Decimal("49950"), Currency.USD))
        assert volume.value == Decimal("1.5")

    def test_get_ask_volume_at_price(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения объема продаж по цене."""
        # Цена ниже всех продаж
        volume = orderbook_snapshot.get_ask_volume_at_price(Price(Decimal("49000"), Currency.USD))
        assert volume.value == Decimal("0")

        # Цена равна лучшей продаже
        volume = orderbook_snapshot.get_ask_volume_at_price(Price(Decimal("50100"), Currency.USD))
        assert volume.value == Decimal("1.0")

        # Цена между продажами
        volume = orderbook_snapshot.get_ask_volume_at_price(Price(Decimal("50150"), Currency.USD))
        assert volume.value == Decimal("1.0")

    def test_get_depth_at_price(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения глубины рынка по цене."""
        # Цена между лучшими bid и ask
        bid_vol, ask_vol = orderbook_snapshot.get_depth_at_price(Price(Decimal("50050"), Currency.USD))
        assert bid_vol.value == Decimal("0")  # Нет покупок по цене выше лучшей
        assert ask_vol.value == Decimal("0")  # Нет продаж по цене ниже лучшей

    def test_is_filtered_true(self) -> None:
        """Тест проверки фильтрации при включенной фильтрации."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[],
            timestamp=Timestamp(1640995200),
            meta=MetadataDict({"filtered": True}),
        )
        assert orderbook.is_filtered() is True

    def test_is_filtered_false(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест проверки фильтрации при выключенной фильтрации."""
        assert orderbook_snapshot.is_filtered() is False

    def test_is_filtered_no_meta(self) -> None:
        """Тест проверки фильтрации при отсутствии метаданных."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.is_filtered() is False

    def test_get_noise_analysis(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест получения анализа шума."""
        noise_analysis = orderbook_snapshot.get_noise_analysis()
        assert noise_analysis is not None
        assert noise_analysis["level"] == "low"

    def test_get_noise_analysis_no_data(self) -> None:
        """Тест получения анализа шума при отсутствии данных."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.get_noise_analysis() is None

    def test_to_dict(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест преобразования в словарь."""
        data = orderbook_snapshot.to_dict()
        assert data["exchange"] == "test_exchange"
        assert data["symbol"] == "BTC/USD"
        assert len(data["bids"]) == 3
        assert len(data["asks"]) == 3
        assert data["sequence_id"] == 12345
        assert data["meta"]["filtered"] is False
        assert data["meta"]["noise_analysis"]["level"] == "low"

    def test_from_dict(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест создания из словаря."""
        data = orderbook_snapshot.to_dict()
        restored_orderbook = OrderBookSnapshot.from_dict(data)
        
        assert restored_orderbook.exchange == orderbook_snapshot.exchange
        assert restored_orderbook.symbol == orderbook_snapshot.symbol
        assert len(restored_orderbook.bids) == len(orderbook_snapshot.bids)
        assert len(restored_orderbook.asks) == len(orderbook_snapshot.asks)
        assert restored_orderbook.sequence_id == orderbook_snapshot.sequence_id
        assert restored_orderbook.meta["filtered"] == orderbook_snapshot.meta["filtered"]

    def test_from_dict_without_optional_fields(self) -> None:
        """Тест создания из словаря без опциональных полей."""
        data = {
            "exchange": "test",
            "symbol": "BTC/USD",
            "bids": [("50000", "1.0")],
            "asks": [("50100", "1.0")],
            "timestamp": {"value": "2022-01-01T00:00:00+00:00"},
        }
        orderbook = OrderBookSnapshot.from_dict(data)
        assert orderbook.exchange == "test"
        assert orderbook.symbol == "BTC/USD"
        assert orderbook.sequence_id is None
        assert len(orderbook.meta) == 0

    def test_string_representation(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест строкового представления."""
        str_repr = str(orderbook_snapshot)
        assert "OrderBookSnapshot" in str_repr
        assert "test_exchange" in str_repr
        assert "BTC/USD" in str_repr
        assert "3 bids" in str_repr
        assert "3 asks" in str_repr

    def test_repr_representation(self, orderbook_snapshot: OrderBookSnapshot) -> None:
        """Тест представления для отладки."""
        repr_str = repr(orderbook_snapshot)
        assert "OrderBookSnapshot" in repr_str
        assert "exchange='test_exchange'" in repr_str
        assert "symbol='BTC/USD'" in repr_str
        assert "bids=3" in repr_str
        assert "asks=3" in repr_str
        assert "timestamp=" in repr_str

    def test_empty_orderbook(self) -> None:
        """Тест пустого ордербука."""
        orderbook = OrderBookSnapshot(
            exchange="test",
            symbol="BTC/USD",
            bids=[],
            asks=[],
            timestamp=Timestamp(1640995200),
        )
        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.mid_price is None
        assert orderbook.spread is None
        assert orderbook.spread_percentage is None
        assert orderbook.total_bid_volume.value == Decimal("0")
        assert orderbook.total_ask_volume.value == Decimal("0")
        assert orderbook.volume_imbalance == Decimal("0")
        assert orderbook.volume_imbalance_ratio is None 