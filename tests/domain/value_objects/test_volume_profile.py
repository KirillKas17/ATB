"""
Unit тесты для domain/value_objects/volume_profile.py.
"""

import pytest
from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime, timezone

from domain.value_objects.volume_profile import VolumeProfile, VolumeLevel
from domain.exceptions.base_exceptions import ValidationError


class TestVolumeLevel:
    """Тесты для VolumeLevel."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "price": Decimal("50000.00"),
            "volume": Decimal("1000.0"),
            "buy_volume": Decimal("600.0"),
            "sell_volume": Decimal("400.0"),
            "timestamp": datetime.now(timezone.utc),
        }

    def test_creation(self, sample_data):
        """Тест создания VolumeLevel."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        assert level.price == sample_data["price"]
        assert level.volume == sample_data["volume"]
        assert level.buy_volume == sample_data["buy_volume"]
        assert level.sell_volume == sample_data["sell_volume"]
        assert level.timestamp == sample_data["timestamp"]

    def test_validation_negative_price(self, sample_data):
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValidationError, match="Price cannot be negative"):
            VolumeLevel(
                price=Decimal("-100.00"),
                volume=sample_data["volume"],
                buy_volume=sample_data["buy_volume"],
                sell_volume=sample_data["sell_volume"],
                timestamp=sample_data["timestamp"],
            )

    def test_validation_negative_volume(self, sample_data):
        """Тест валидации отрицательного объема."""
        with pytest.raises(ValidationError, match="Volume cannot be negative"):
            VolumeLevel(
                price=sample_data["price"],
                volume=Decimal("-100.0"),
                buy_volume=sample_data["buy_volume"],
                sell_volume=sample_data["sell_volume"],
                timestamp=sample_data["timestamp"],
            )

    def test_validation_negative_buy_volume(self, sample_data):
        """Тест валидации отрицательного buy объема."""
        with pytest.raises(ValidationError, match="Buy volume cannot be negative"):
            VolumeLevel(
                price=sample_data["price"],
                volume=sample_data["volume"],
                buy_volume=Decimal("-100.0"),
                sell_volume=sample_data["sell_volume"],
                timestamp=sample_data["timestamp"],
            )

    def test_validation_negative_sell_volume(self, sample_data):
        """Тест валидации отрицательного sell объема."""
        with pytest.raises(ValidationError, match="Sell volume cannot be negative"):
            VolumeLevel(
                price=sample_data["price"],
                volume=sample_data["volume"],
                buy_volume=sample_data["buy_volume"],
                sell_volume=Decimal("-100.0"),
                timestamp=sample_data["timestamp"],
            )

    def test_validation_volume_mismatch(self, sample_data):
        """Тест валидации несоответствия объемов."""
        with pytest.raises(ValidationError, match="Total volume must equal buy volume plus sell volume"):
            VolumeLevel(
                price=sample_data["price"],
                volume=Decimal("1000.0"),
                buy_volume=Decimal("600.0"),
                sell_volume=Decimal("500.0"),  # 600 + 500 != 1000
                timestamp=sample_data["timestamp"],
            )

    def test_buy_ratio_property(self, sample_data):
        """Тест свойства buy_ratio."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        expected_ratio = Decimal("0.6")  # 600 / 1000
        assert level.buy_ratio == expected_ratio

    def test_sell_ratio_property(self, sample_data):
        """Тест свойства sell_ratio."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        expected_ratio = Decimal("0.4")  # 400 / 1000
        assert level.sell_ratio == expected_ratio

    def test_imbalance_property(self, sample_data):
        """Тест свойства imbalance."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        expected_imbalance = Decimal("0.2")  # (600 - 400) / 1000
        assert level.imbalance == expected_imbalance

    def test_is_buy_dominant(self, sample_data):
        """Тест проверки доминирования покупок."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        assert level.is_buy_dominant() is True

    def test_is_sell_dominant(self, sample_data):
        """Тест проверки доминирования продаж."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=Decimal("400.0"),
            sell_volume=Decimal("600.0"),
            timestamp=sample_data["timestamp"],
        )

        assert level.is_sell_dominant() is True

    def test_is_balanced(self, sample_data):
        """Тест проверки сбалансированности."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=Decimal("1000.0"),
            buy_volume=Decimal("500.0"),
            sell_volume=Decimal("500.0"),
            timestamp=sample_data["timestamp"],
        )

        assert level.is_balanced() is True

    def test_to_dict(self, sample_data):
        """Тест сериализации в словарь."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        result = level.to_dict()

        assert result["price"] == str(sample_data["price"])
        assert result["volume"] == str(sample_data["volume"])
        assert result["buy_volume"] == str(sample_data["buy_volume"])
        assert result["sell_volume"] == str(sample_data["sell_volume"])
        assert result["timestamp"] == sample_data["timestamp"].isoformat()
        assert result["buy_ratio"] == str(Decimal("0.6"))
        assert result["sell_ratio"] == str(Decimal("0.4"))
        assert result["imbalance"] == str(Decimal("0.2"))

    def test_from_dict(self, sample_data):
        """Тест десериализации из словаря."""
        data = {
            "price": str(sample_data["price"]),
            "volume": str(sample_data["volume"]),
            "buy_volume": str(sample_data["buy_volume"]),
            "sell_volume": str(sample_data["sell_volume"]),
            "timestamp": sample_data["timestamp"].isoformat(),
        }

        level = VolumeLevel.from_dict(data)

        assert level.price == sample_data["price"]
        assert level.volume == sample_data["volume"]
        assert level.buy_volume == sample_data["buy_volume"]
        assert level.sell_volume == sample_data["sell_volume"]
        assert level.timestamp == sample_data["timestamp"]

    def test_equality(self, sample_data):
        """Тест равенства объектов."""
        level1 = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        level2 = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        assert level1 == level2
        assert hash(level1) == hash(level2)

    def test_str_representation(self, sample_data):
        """Тест строкового представления."""
        level = VolumeLevel(
            price=sample_data["price"],
            volume=sample_data["volume"],
            buy_volume=sample_data["buy_volume"],
            sell_volume=sample_data["sell_volume"],
            timestamp=sample_data["timestamp"],
        )

        expected = "VolumeLevel(50000.00: 1000.0 total, 600.0 buy, 400.0 sell)"
        assert str(level) == expected


class TestVolumeProfile:
    """Тесты для VolumeProfile."""

    @pytest.fixture
    def sample_levels(self) -> List[VolumeLevel]:
        """Тестовые уровни объема."""
        return [
            VolumeLevel(
                price=Decimal("50000.00"),
                volume=Decimal("1000.0"),
                buy_volume=Decimal("600.0"),
                sell_volume=Decimal("400.0"),
                timestamp=datetime.now(timezone.utc),
            ),
            VolumeLevel(
                price=Decimal("50100.00"),
                volume=Decimal("800.0"),
                buy_volume=Decimal("400.0"),
                sell_volume=Decimal("400.0"),
                timestamp=datetime.now(timezone.utc),
            ),
            VolumeLevel(
                price=Decimal("50200.00"),
                volume=Decimal("1200.0"),
                buy_volume=Decimal("300.0"),
                sell_volume=Decimal("900.0"),
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "symbol": "BTCUSD",
            "timeframe": "1h",
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
            "levels": [],
        }

    def test_creation(self, sample_data, sample_levels):
        """Тест создания VolumeProfile."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        assert profile.symbol == sample_data["symbol"]
        assert profile.timeframe == sample_data["timeframe"]
        assert profile.start_time == sample_data["start_time"]
        assert profile.end_time == sample_data["end_time"]
        assert profile.levels == sample_data["levels"]

    def test_validation_empty_symbol(self, sample_data, sample_levels):
        """Тест валидации пустого символа."""
        sample_data["symbol"] = ""
        sample_data["levels"] = sample_levels

        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            VolumeProfile(
                symbol=sample_data["symbol"],
                timeframe=sample_data["timeframe"],
                start_time=sample_data["start_time"],
                end_time=sample_data["end_time"],
                levels=sample_data["levels"],
            )

    def test_validation_empty_timeframe(self, sample_data, sample_levels):
        """Тест валидации пустого таймфрейма."""
        sample_data["timeframe"] = ""
        sample_data["levels"] = sample_levels

        with pytest.raises(ValidationError, match="Timeframe cannot be empty"):
            VolumeProfile(
                symbol=sample_data["symbol"],
                timeframe=sample_data["timeframe"],
                start_time=sample_data["start_time"],
                end_time=sample_data["end_time"],
                levels=sample_data["levels"],
            )

    def test_validation_invalid_time_range(self, sample_data, sample_levels):
        """Тест валидации некорректного временного диапазона."""
        sample_data["levels"] = sample_levels
        sample_data["end_time"] = sample_data["start_time"].replace(year=sample_data["start_time"].year - 1)

        with pytest.raises(ValidationError, match="End time must be after start time"):
            VolumeProfile(
                symbol=sample_data["symbol"],
                timeframe=sample_data["timeframe"],
                start_time=sample_data["start_time"],
                end_time=sample_data["end_time"],
                levels=sample_data["levels"],
            )

    def test_total_volume_property(self, sample_data, sample_levels):
        """Тест свойства total_volume."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        expected_total = Decimal("3000.0")  # 1000 + 800 + 1200
        assert profile.total_volume == expected_total

    def test_total_buy_volume_property(self, sample_data, sample_levels):
        """Тест свойства total_buy_volume."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        expected_buy_total = Decimal("1300.0")  # 600 + 400 + 300
        assert profile.total_buy_volume == expected_buy_total

    def test_total_sell_volume_property(self, sample_data, sample_levels):
        """Тест свойства total_sell_volume."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        expected_sell_total = Decimal("1700.0")  # 400 + 400 + 900
        assert profile.total_sell_volume == expected_sell_total

    def test_overall_imbalance_property(self, sample_data, sample_levels):
        """Тест свойства overall_imbalance."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        expected_imbalance = Decimal("-0.133333")  # (1300 - 1700) / 3000
        assert profile.overall_imbalance == expected_imbalance

    def test_price_range_property(self, sample_data, sample_levels):
        """Тест свойства price_range."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        min_price, max_price = profile.price_range
        assert min_price == Decimal("50000.00")
        assert max_price == Decimal("50200.00")

    def test_volume_weighted_average_price_property(self, sample_data, sample_levels):
        """Тест свойства volume_weighted_average_price."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        vwap = profile.volume_weighted_average_price
        # (50000*1000 + 50100*800 + 50200*1200) / 3000
        expected_vwap = Decimal("50106.67")
        assert abs(vwap - expected_vwap) < Decimal("0.01")

    def test_poc_price_property(self, sample_data, sample_levels):
        """Тест свойства poc_price (Point of Control)."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        poc_price = profile.poc_price
        assert poc_price == Decimal("50200.00")  # Уровень с максимальным объемом

    def test_value_area_property(self, sample_data, sample_levels):
        """Тест свойства value_area."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        value_area = profile.value_area
        assert len(value_area) == 3  # Все уровни входят в value area

    def test_get_level_at_price(self, sample_data, sample_levels):
        """Тест получения уровня по цене."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        level = profile.get_level_at_price(Decimal("50000.00"))
        assert level is not None
        assert level.price == Decimal("50000.00")

    def test_get_level_at_price_not_found(self, sample_data, sample_levels):
        """Тест получения уровня по несуществующей цене."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        level = profile.get_level_at_price(Decimal("51000.00"))
        assert level is None

    def test_add_level(self, sample_data, sample_levels):
        """Тест добавления уровня."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        new_level = VolumeLevel(
            price=Decimal("50300.00"),
            volume=Decimal("500.0"),
            buy_volume=Decimal("250.0"),
            sell_volume=Decimal("250.0"),
            timestamp=datetime.now(timezone.utc),
        )

        profile.add_level(new_level)

        assert len(profile.levels) == 4
        assert profile.get_level_at_price(Decimal("50300.00")) is not None

    def test_remove_level(self, sample_data, sample_levels):
        """Тест удаления уровня."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        profile.remove_level(Decimal("50000.00"))

        assert len(profile.levels) == 2
        assert profile.get_level_at_price(Decimal("50000.00")) is None

    def test_to_dict(self, sample_data, sample_levels):
        """Тест сериализации в словарь."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        result = profile.to_dict()

        assert result["symbol"] == sample_data["symbol"]
        assert result["timeframe"] == sample_data["timeframe"]
        assert result["start_time"] == sample_data["start_time"].isoformat()
        assert result["end_time"] == sample_data["end_time"].isoformat()
        assert len(result["levels"]) == 3
        assert result["total_volume"] == str(Decimal("3000.0"))
        assert result["total_buy_volume"] == str(Decimal("1300.0"))
        assert result["total_sell_volume"] == str(Decimal("1700.0"))

    def test_from_dict(self, sample_data, sample_levels):
        """Тест десериализации из словаря."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        data = profile.to_dict()
        new_profile = VolumeProfile.from_dict(data)

        assert new_profile.symbol == profile.symbol
        assert new_profile.timeframe == profile.timeframe
        assert new_profile.start_time == profile.start_time
        assert new_profile.end_time == profile.end_time
        assert len(new_profile.levels) == len(profile.levels)

    def test_equality(self, sample_data, sample_levels):
        """Тест равенства объектов."""
        sample_data["levels"] = sample_levels

        profile1 = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        profile2 = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        assert profile1 == profile2
        assert hash(profile1) == hash(profile2)

    def test_str_representation(self, sample_data, sample_levels):
        """Тест строкового представления."""
        sample_data["levels"] = sample_levels

        profile = VolumeProfile(
            symbol=sample_data["symbol"],
            timeframe=sample_data["timeframe"],
            start_time=sample_data["start_time"],
            end_time=sample_data["end_time"],
            levels=sample_data["levels"],
        )

        expected = "VolumeProfile(BTCUSD 1h: 3 levels, 3000.0 total volume)"
        assert str(profile) == expected
