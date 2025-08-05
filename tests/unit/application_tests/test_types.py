"""
Тесты для application/types.py
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from application.types import (
    CreateOrderRequest, CreateOrderResponse, CancelOrderRequest, CancelOrderResponse,
    GetOrdersRequest, GetOrdersResponse, MarketPhase, SignalType, OrderStatus,
    RiskLevel, MarketSummary, TechnicalIndicators, VolumeProfile, MarketRegime,
    PriceLevel, VolumeLevel, MoneyAmount, Percentage, Timestamp
)


class TestApplicationTypes:
    """Тесты для типов application слоя."""

    def test_create_order_request(self: "TestApplicationTypes") -> None:
        """Тест создания CreateOrderRequest."""
        request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type="LIMIT",
            side="BUY",
            amount=VolumeLevel(Decimal("0.1")),
            price=PriceLevel(Decimal("50000")),
            stop_price=None
        )
        
        assert request.portfolio_id is not None
        assert request.symbol == "BTC/USD"
        assert request.order_type == "LIMIT"
        assert request.side == "BUY"
        assert request.amount.value == Decimal("0.1")
        assert request.price.value == Decimal("50000")
        assert request.stop_price is None

    def test_create_order_response_success(self: "TestApplicationTypes") -> None:
        """Тест успешного CreateOrderResponse."""
        response = CreateOrderResponse(
            success=True,
            order=None,  # В реальном случае здесь был бы Order
            estimated_cost=MoneyAmount(Decimal("5000")),
            warnings=["Low liquidity"],
            message="Order created successfully"
        )
        
        assert response.success is True
        assert response.estimated_cost.amount == Decimal("5000")
        assert "Low liquidity" in response.warnings
        assert response.message == "Order created successfully"

    def test_create_order_response_error(self: "TestApplicationTypes") -> None:
        """Тест CreateOrderResponse с ошибкой."""
        response = CreateOrderResponse(
            success=False,
            message="Insufficient funds",
            errors=["Insufficient balance"]
        )
        
        assert response.success is False
        assert response.message == "Insufficient funds"
        assert "Insufficient balance" in response.errors

    def test_cancel_order_request(self: "TestApplicationTypes") -> None:
        """Тест создания CancelOrderRequest."""
        order_id = uuid4()
        portfolio_id = uuid4()
        
        request = CancelOrderRequest(
            order_id=order_id,
            portfolio_id=portfolio_id
        )
        
        assert request.order_id == order_id
        assert request.portfolio_id == portfolio_id

    def test_cancel_order_response(self: "TestApplicationTypes") -> None:
        """Тест CancelOrderResponse."""
        response = CancelOrderResponse(
            cancelled=True,
            order=None,  # В реальном случае здесь был бы Order
            message="Order cancelled successfully"
        )
        
        assert response.cancelled is True
        assert response.message == "Order cancelled successfully"

    def test_get_orders_request(self: "TestApplicationTypes") -> None:
        """Тест создания GetOrdersRequest."""
        request = GetOrdersRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            status="PENDING",
            limit=50,
            offset=0
        )
        
        assert request.portfolio_id is not None
        assert request.symbol == "BTC/USD"
        assert request.status == "PENDING"
        assert request.limit == 50
        assert request.offset == 0

    def test_get_orders_response(self: "TestApplicationTypes") -> None:
        """Тест GetOrdersResponse."""
        response = GetOrdersResponse(
            orders=[],  # В реальном случае здесь были бы Order
            total_count=0,
            has_more=False,
            message="No orders found"
        )
        
        assert response.orders == []
        assert response.total_count == 0
        assert response.has_more is False
        assert response.message == "No orders found"

    def test_market_phase_enum(self: "TestApplicationTypes") -> None:
        """Тест enum MarketPhase."""
        assert MarketPhase.BULL_MARKET == "bull_market"
        assert MarketPhase.BEAR_MARKET == "bear_market"
        assert MarketPhase.SIDEWAYS == "sideways"
        assert MarketPhase.VOLATILE == "volatile"

    def test_signal_type_enum(self: "TestApplicationTypes") -> None:
        """Тест enum SignalType."""
        assert SignalType.BUY == "buy"
        assert SignalType.SELL == "sell"
        assert SignalType.HOLD == "hold"
        assert SignalType.STRONG_BUY == "strong_buy"
        assert SignalType.STRONG_SELL == "strong_sell"

    def test_order_status_enum(self: "TestApplicationTypes") -> None:
        """Тест enum OrderStatus."""
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.OPEN == "open"
        assert OrderStatus.PARTIALLY_FILLED == "partially_filled"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.CANCELLED == "cancelled"
        assert OrderStatus.REJECTED == "rejected"
        assert OrderStatus.EXPIRED == "expired"

    def test_risk_level_enum(self: "TestApplicationTypes") -> None:
        """Тест enum RiskLevel."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.EXTREME == "extreme"

    def test_market_summary(self: "TestApplicationTypes") -> None:
        """Тест MarketSummary."""
        summary = MarketSummary(
            symbol="BTC/USD",
            last_price=PriceLevel(Decimal("50000")),
            price_change=MoneyAmount(Decimal("1000")),
            price_change_percent=Percentage(Decimal("2.0")),
            volume=VolumeLevel(Decimal("1000")),
            high=PriceLevel(Decimal("51000")),
            low=PriceLevel(Decimal("49000")),
            market_phase=MarketPhase.BULL_MARKET,
            volatility=Percentage(Decimal("15.5")),
            support_levels=[PriceLevel(Decimal("48000"))],
            resistance_levels=[PriceLevel(Decimal("52000"))],
            timestamp=Timestamp(datetime.now())
        )
        
        assert summary.symbol == "BTC/USD"
        assert summary.last_price.value == Decimal("50000")
        assert summary.price_change.amount == Decimal("1000")
        assert summary.price_change_percent.value == Decimal("2.0")
        assert summary.volume.value == Decimal("1000")
        assert summary.high.value == Decimal("51000")
        assert summary.low.value == Decimal("49000")
        assert summary.market_phase == MarketPhase.BULL_MARKET
        assert summary.volatility.value == Decimal("15.5")
        assert len(summary.support_levels) == 1
        assert len(summary.resistance_levels) == 1

    def test_technical_indicators(self: "TestApplicationTypes") -> None:
        """Тест TechnicalIndicators."""
        indicators = TechnicalIndicators(
            symbol="BTC/USD",
            timeframe="1h",
            rsi=[Decimal("65.5")],
            macd=[Decimal("0.5")],
            macd_signal=[Decimal("0.3")],
            macd_histogram=[Decimal("0.2")],
            sma_20=[Decimal("49500")],
            sma_50=[Decimal("49000")],
            ema_12=[Decimal("49800")],
            ema_26=[Decimal("49200")],
            bollinger_upper=[Decimal("51000")],
            bollinger_middle=[Decimal("50000")],
            bollinger_lower=[Decimal("49000")],
            atr=[Decimal("500")],
            timestamp=Timestamp(datetime.now())
        )
        
        assert indicators.symbol == "BTC/USD"
        assert indicators.timeframe == "1h"
        assert indicators.rsi[0] == Decimal("65.5")
        assert indicators.macd[0] == Decimal("0.5")
        assert indicators.macd_signal[0] == Decimal("0.3")
        assert indicators.macd_histogram[0] == Decimal("0.2")
        assert indicators.sma_20[0] == Decimal("49500")
        assert indicators.sma_50[0] == Decimal("49000")
        assert indicators.ema_12[0] == Decimal("49800")
        assert indicators.ema_26[0] == Decimal("49200")
        assert indicators.bollinger_upper[0] == Decimal("51000")
        assert indicators.bollinger_middle[0] == Decimal("50000")
        assert indicators.bollinger_lower[0] == Decimal("49000")
        assert indicators.atr[0] == Decimal("500")

    def test_volume_profile(self: "TestApplicationTypes") -> None:
        """Тест VolumeProfile."""
        profile = VolumeProfile(
            symbol="BTC/USD",
            timeframe="1h",
            poc_price=PriceLevel(Decimal("50000")),
            total_volume=VolumeLevel(Decimal("10000")),
            volume_profile={"50000": Decimal("5000")},
            price_range={
                "min": PriceLevel(Decimal("48000")),
                "max": PriceLevel(Decimal("52000"))
            },
            timestamp=Timestamp(datetime.now())
        )
        
        assert profile.symbol == "BTC/USD"
        assert profile.timeframe == "1h"
        assert profile.poc_price.value == Decimal("50000")
        assert profile.total_volume.value == Decimal("10000")
        assert profile.volume_profile["50000"] == Decimal("5000")
        assert profile.price_range["min"].value == Decimal("48000")
        assert profile.price_range["max"].value == Decimal("52000")

    def test_market_regime(self: "TestApplicationTypes") -> None:
        """Тест MarketRegime."""
        regime = MarketRegime(
            symbol="BTC/USD",
            timeframe="1h",
            regime="trending",
            volatility=Percentage(Decimal("20.5")),
            trend_strength=Percentage(Decimal("75.0")),
            price_trend=Decimal("0.001"),
            volume_trend=Decimal("0.05"),
            confidence=Percentage(Decimal("85.0")),
            timestamp=Timestamp(datetime.now())
        )
        
        assert regime.symbol == "BTC/USD"
        assert regime.timeframe == "1h"
        assert regime.regime == "trending"
        assert regime.volatility.value == Decimal("20.5")
        assert regime.trend_strength.value == Decimal("75.0")
        assert regime.price_trend == Decimal("0.001")
        assert regime.volume_trend == Decimal("0.05")
        assert regime.confidence.value == Decimal("85.0")

    def test_value_objects_validation(self: "TestApplicationTypes") -> None:
        """Тест валидации value objects."""
        # Тест PriceLevel
        price = PriceLevel(Decimal("50000"))
        assert price.value == Decimal("50000")
        
        # Тест VolumeLevel
        volume = VolumeLevel(Decimal("0.1"))
        assert volume.value == Decimal("0.1")
        
        # Тест MoneyAmount
        money = MoneyAmount(Decimal("5000"))
        assert money.amount == Decimal("5000")
        
        # Тест Percentage
        percentage = Percentage(Decimal("2.5"))
        assert percentage.value == Decimal("2.5")
        
        # Тест Timestamp
        now = datetime.now()
        timestamp = Timestamp(now)
        assert timestamp.value == now

    def test_invalid_values(self: "TestApplicationTypes") -> None:
        """Тест обработки некорректных значений."""
        # Отрицательная цена должна вызывать ошибку
        with pytest.raises(ValueError):
            PriceLevel(Decimal("-100"))
        
        # Отрицательный объем должен вызывать ошибку
        with pytest.raises(ValueError):
            VolumeLevel(Decimal("-0.1"))
        
        # Процент больше 100 должен вызывать ошибку
        with pytest.raises(ValueError):
            Percentage(Decimal("150"))

    def test_serialization(self: "TestApplicationTypes") -> None:
        """Тест сериализации типов."""
        request = CreateOrderRequest(
            portfolio_id=uuid4(),
            symbol="BTC/USD",
            order_type="LIMIT",
            side="BUY",
            amount=VolumeLevel(Decimal("0.1")),
            price=PriceLevel(Decimal("50000"))
        )
        
        # Проверяем, что объект можно преобразовать в dict
        data = request.__dict__
        assert "portfolio_id" in data
        assert "symbol" in data
        assert "order_type" in data
        assert "side" in data
        assert "amount" in data
        assert "price" in data 
