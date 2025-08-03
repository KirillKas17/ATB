# -*- coding: utf-8 -*-
"""Тесты для системы гравитации ликвидности."""

import time

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from application.risk.liquidity_gravity_monitor import (
    LiquidityGravityMonitor,
    MonitorConfig,
    RiskAssessmentResult,
)
from domain.market.liquidity_gravity import (LiquidityGravityConfig,
                                             LiquidityGravityModel,
                                             LiquidityGravityResult,
                                             OrderBookSnapshot,
                                             compute_liquidity_gravity)
from domain.value_objects.timestamp import Timestamp


class TestLiquidityGravityModel:
    """Тесты для модели гравитации ликвидности."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = LiquidityGravityConfig(
            gravitational_constant=1e-6,
            min_volume_threshold=0.001,
            max_price_distance=0.1,
            volume_weight=1.0,
            price_weight=1.0,
        )
        self.model = LiquidityGravityModel(self.config)

    def test_initialization(self) -> None:
        """Тест инициализации модели."""
        assert self.model.config.gravitational_constant == 1e-6
        assert self.model.config.min_volume_threshold == 0.001
        assert self.model.config.max_price_distance == 0.1
        assert self.model.config.volume_weight == 1.0
        assert self.model.config.price_weight == 1.0

    def test_compute_gravitational_force(self) -> None:
        """Тест вычисления силы гравитации."""
        # Тест с нормальными значениями
        force = self.model._compute_gravitational_force(1.0, 2.0, 0.01)
        expected_force = (1e-6 * 1.0 * 2.0) / (0.01**2)
        assert abs(force - expected_force) < 1e-10

        # Тест с нулевым расстоянием
        force = self.model._compute_gravitational_force(1.0, 2.0, 0.0)
        assert force > 0  # Должно обработать деление на ноль

    def test_compute_liquidity_gravity_normal(self) -> None:
        """Тест вычисления гравитации с нормальным ордербуком."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 1.0)],
            asks=[(50001, 1.0), (50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        gravity = self.model.compute_liquidity_gravity(order_book)
        assert isinstance(gravity, float)
        assert gravity >= 0.0

    def test_compute_liquidity_gravity_empty(self) -> None:
        """Тест вычисления гравитации с пустым ордербуком."""
        order_book = OrderBookSnapshot(
            bids=[], asks=[], timestamp=Timestamp(time.time()), symbol="BTC/USDT"
        )

        gravity = self.model.compute_liquidity_gravity(order_book)
        assert gravity == 0.0

    def test_compute_liquidity_gravity_bids_only(self) -> None:
        """Тест вычисления гравитации только с бидами."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 1.0)],
            asks=[],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        gravity = self.model.compute_liquidity_gravity(order_book)
        assert gravity == 0.0

    def test_compute_liquidity_gravity_asks_only(self) -> None:
        """Тест вычисления гравитации только с асками."""
        order_book = OrderBookSnapshot(
            bids=[],
            asks=[(50001, 1.0), (50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        gravity = self.model.compute_liquidity_gravity(order_book)
        assert gravity == 0.0

    def test_analyze_liquidity_gravity(self) -> None:
        """Тест полного анализа гравитации ликвидности."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 1.0)],
            asks=[(50001, 1.0), (50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        result = self.model.analyze_liquidity_gravity(order_book)

        assert isinstance(result, LiquidityGravityResult)
        assert isinstance(result.total_gravity, float)
        assert isinstance(result.bid_ask_forces, list)
        assert isinstance(result.gravity_distribution, dict)
        assert isinstance(result.risk_level, str)
        assert isinstance(result.timestamp, Timestamp)
        assert isinstance(result.metadata, dict)

    def test_determine_risk_level(self) -> None:
        """Тест определения уровня риска."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0)],
            asks=[(50001, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        # Тест с низкой гравитацией
        risk_level = self.model._determine_risk_level(0.05, order_book)
        assert risk_level == "low"

        # Тест со средней гравитацией
        risk_level = self.model._determine_risk_level(0.3, order_book)
        assert risk_level == "medium"

        # Тест с высокой гравитацией
        risk_level = self.model._determine_risk_level(0.8, order_book)
        assert risk_level == "high"

        # Тест с экстремальной гравитацией
        risk_level = self.model._determine_risk_level(2.5, order_book)
        assert risk_level == "extreme"

    def test_compute_gravity_gradient(self) -> None:
        """Тест вычисления градиента гравитации."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 1.0), (49998, 1.0)],
            asks=[(50001, 1.0), (50002, 1.0), (50003, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        gradient = self.model.compute_gravity_gradient(order_book)
        assert isinstance(gradient, dict)

    def test_get_model_statistics(self) -> None:
        """Тест получения статистики модели."""
        stats = self.model.get_model_statistics()
        assert isinstance(stats, dict)
        assert "gravitational_constant" in stats
        assert "min_volume_threshold" in stats
        assert "max_price_distance" in stats

    def test_update_config(self) -> None:
        """Тест обновления конфигурации."""
        new_config = LiquidityGravityConfig(
            gravitational_constant=2e-6, min_volume_threshold=0.002
        )

        self.model.update_config(new_config)
        assert self.model.config.gravitational_constant == 2e-6
        assert self.model.config.min_volume_threshold == 0.002


class TestOrderBookSnapshot:
    """Тесты для снимка ордербука."""

    def test_creation(self) -> None:
        """Тест создания снимка ордербука."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 1.0)],
            asks=[(50001, 1.0), (50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
        assert order_book.symbol == "BTC/USDT"

    def test_get_bid_volume(self) -> None:
        """Тест получения объема бидов."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0), (49999, 2.0)],
            asks=[(50001, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        assert order_book.get_bid_volume() == 3.0

    def test_get_ask_volume(self) -> None:
        """Тест получения объема асков."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0)],
            asks=[(50001, 1.0), (50002, 2.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        assert order_book.get_ask_volume() == 3.0

    def test_get_mid_price(self) -> None:
        """Тест получения средней цены."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0)],
            asks=[(50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        assert order_book.get_mid_price() == 50001.0

    def test_get_spread(self) -> None:
        """Тест получения спреда."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0)],
            asks=[(50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        assert order_book.get_spread() == 2.0

    def test_get_spread_percentage(self) -> None:
        """Тест получения спреда в процентах."""
        order_book = OrderBookSnapshot(
            bids=[(50000, 1.0)],
            asks=[(50002, 1.0)],
            timestamp=Timestamp(time.time()),
            symbol="BTC/USDT",
        )

        expected_percentage = (2.0 / 50001.0) * 100
        assert abs(order_book.get_spread_percentage() - expected_percentage) < 1e-6


def test_compute_liquidity_gravity_function() -> None:
    """Тест удобной функции compute_liquidity_gravity."""
    order_book = OrderBookSnapshot(
        bids=[(50000, 1.0), (49999, 1.0)],
        asks=[(50001, 1.0), (50002, 1.0)],
        timestamp=Timestamp(time.time()),
        symbol="BTC/USDT",
    )

    gravity = compute_liquidity_gravity(order_book)
    assert isinstance(gravity, float)
    assert gravity >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
