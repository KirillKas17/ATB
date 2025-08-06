#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграционные тесты критических торговых потоков.
"""

import pytest
import asyncio
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from application.orchestration.trading_orchestrator import TradingOrchestrator
from infrastructure.external_services.bybit_client import BybitClient
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.position import Position
from domain.strategies.quantum_arbitrage_strategy import QuantumArbitrageStrategy
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions import TradingError, ValidationError


class TestTradingWorkflow:
    """Интеграционные тесты торгового workflow."""

    @pytest.fixture
    async def trading_orchestrator(self) -> TradingOrchestrator:
        """Фикстура торгового оркестратора."""
        config = {
            "risk_limit": Decimal("0.02"),  # 2% риск на сделку
            "max_open_positions": 5,
            "execution_timeout": 30,  # секунд
            "slippage_tolerance": Decimal("0.005")  # 0.5%
        }
        return TradingOrchestrator(**config)

    @pytest.fixture
    async def mock_exchange_client(self) -> BybitClient:
        """Мок биржевого клиента."""
        client = AsyncMock(spec=BybitClient)
        
        # Настраиваем методы клиента
        client.get_account_balance.return_value = {
            "USDT": {"total": Decimal("10000.00"), "available": Decimal("9500.00")},
            "BTC": {"total": Decimal("0.1"), "available": Decimal("0.1")}
        }
        
        client.get_ticker.return_value = {
            "symbol": "BTCUSDT",
            "price": Decimal("45000.00"),
            "volume": Decimal("1000.0"),
            "bid": Decimal("44999.50"),
            "ask": Decimal("45000.50")
        }
        
        client.place_limit_order.return_value = {
            "order_id": "test_order_123",
            "status": "pending"
        }
        
        return client

    @pytest.fixture
    def sample_strategy(self) -> QuantumArbitrageStrategy:
        """Фикстура торговой стратегии."""
        config = {
            "min_arbitrage_threshold": Decimal("0.001"),
            "max_position_size": Decimal("1.0"),
            "quantum_coherence_threshold": 0.85,
            "exchanges": ["binance", "bybit"],
            "symbols": ["BTCUSDT"],
            "quantum_states": 8
        }
        return QuantumArbitrageStrategy(**config)

    @pytest.mark.asyncio
    async def test_complete_buy_order_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient,
        sample_strategy: QuantumArbitrageStrategy
    ) -> None:
        """Тест полного workflow размещения buy ордера."""
        
        # Настраиваем стратегию в оркестраторе
        trading_orchestrator.add_strategy(sample_strategy)
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Создаем сигнал на покупку
        buy_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "strategy_id": sample_strategy.strategy_id,
            "confidence": 0.85
        }
        
        # Выполняем торговый workflow
        result = await trading_orchestrator.execute_trade_signal(buy_signal)
        
        # Проверяем результат
        assert result["status"] == "success"
        assert result["order_id"] is not None
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "BUY"
        
        # Проверяем, что был вызван метод размещения ордера
        mock_exchange_client.place_limit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_management_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow риск-менеджмента."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Создаем большой ордер, превышающий лимиты риска
        large_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("5.0"),  # Большое количество
            "price": Decimal("45000.00"),
            "strategy_id": "risk_test_strategy"
        }
        
        # Workflow должен отклонить ордер из-за превышения риска
        with pytest.raises(TradingError, match="Risk limits exceeded"):
            await trading_orchestrator.execute_trade_signal(large_signal)
        
        # Проверяем, что ордер не был размещен
        mock_exchange_client.place_limit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_position_management_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow управления позициями."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Настраиваем мок для получения позиций
        mock_exchange_client.get_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "size": Decimal("0.1"),
                "entry_price": Decimal("44000.00"),
                "mark_price": Decimal("45000.00"),
                "unrealized_pnl": Decimal("100.00")
            }
        ]
        
        # Получаем текущие позиции
        positions = await trading_orchestrator.get_current_positions()
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"
        assert positions[0]["side"] == "LONG"
        assert positions[0]["unrealized_pnl"] == Decimal("100.00")
        
        # Тестируем закрытие позиции
        close_result = await trading_orchestrator.close_position("BTCUSDT")
        
        assert close_result["status"] == "success"
        mock_exchange_client.place_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_status_monitoring_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow мониторинга статуса ордеров."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Настраиваем мок для статуса ордера
        mock_exchange_client.get_order_status.return_value = {
            "order_id": "test_order_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "status": "FILLED",
            "filled_quantity": Decimal("0.001"),
            "average_price": Decimal("45000.50")
        }
        
        # Размещаем ордер
        order_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "strategy_id": "monitor_test"
        }
        
        order_result = await trading_orchestrator.execute_trade_signal(order_signal)
        order_id = order_result["order_id"]
        
        # Мониторим статус ордера
        status = await trading_orchestrator.monitor_order_status(order_id)
        
        assert status["order_id"] == order_id
        assert status["status"] == "FILLED"
        assert status["filled_quantity"] == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_multi_strategy_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow с множественными стратегиями."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Создаем несколько стратегий
        strategy1 = QuantumArbitrageStrategy(
            min_arbitrage_threshold=Decimal("0.001"),
            exchanges=["binance", "bybit"],
            symbols=["BTCUSDT"]
        )
        
        strategy2 = QuantumArbitrageStrategy(
            min_arbitrage_threshold=Decimal("0.002"),
            exchanges=["bybit", "okx"],
            symbols=["ETHUSDT"]
        )
        
        trading_orchestrator.add_strategy(strategy1)
        trading_orchestrator.add_strategy(strategy2)
        
        # Генерируем сигналы от разных стратегий
        signals = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": Decimal("0.001"),
                "strategy_id": strategy1.strategy_id,
                "confidence": 0.8
            },
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": Decimal("0.1"),
                "strategy_id": strategy2.strategy_id,
                "confidence": 0.7
            }
        ]
        
        # Обрабатываем сигналы параллельно
        results = await trading_orchestrator.process_multiple_signals(signals)
        
        assert len(results) == 2
        assert all(result["status"] == "success" for result in results)

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow восстановления после ошибок."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Настраиваем мок для симуляции ошибки сети
        mock_exchange_client.place_limit_order.side_effect = [
            ConnectionError("Network error"),  # Первая попытка - ошибка
            {"order_id": "retry_order_456", "status": "pending"}  # Вторая попытка - успех
        ]
        
        # Размещаем ордер с retry логикой
        signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "strategy_id": "error_recovery_test"
        }
        
        result = await trading_orchestrator.execute_trade_signal_with_retry(signal)
        
        # Проверяем, что после retry ордер был размещен
        assert result["status"] == "success"
        assert result["order_id"] == "retry_order_456"
        assert mock_exchange_client.place_limit_order.call_count == 2

    @pytest.mark.asyncio
    async def test_real_time_market_data_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow real-time обработки рыночных данных."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Симулируем поток рыночных данных
        market_data_stream = [
            {
                "symbol": "BTCUSDT",
                "price": Decimal("45000.00"),
                "volume": Decimal("100.0"),
                "timestamp": 1640995200000
            },
            {
                "symbol": "BTCUSDT",
                "price": Decimal("45050.00"),
                "volume": Decimal("150.0"),
                "timestamp": 1640995201000
            },
            {
                "symbol": "BTCUSDT",
                "price": Decimal("45100.00"),
                "volume": Decimal("200.0"),
                "timestamp": 1640995202000
            }
        ]
        
        # Обрабатываем поток данных
        signals_generated = []
        async for market_data in trading_orchestrator.process_market_data_stream(market_data_stream):
            if market_data.get("signal"):
                signals_generated.append(market_data["signal"])
        
        # Проверяем, что были сгенерированы торговые сигналы
        assert len(signals_generated) > 0
        assert all("symbol" in signal for signal in signals_generated)

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow ребалансировки портфеля."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Настраиваем текущий портфель
        current_portfolio = {
            "BTCUSDT": {"position": Decimal("0.2"), "target": Decimal("0.1")},
            "ETHUSDT": {"position": Decimal("1.0"), "target": Decimal("2.0")},
            "ADAUSDT": {"position": Decimal("0.0"), "target": Decimal("100.0")}
        }
        
        # Выполняем ребалансировку
        rebalance_orders = await trading_orchestrator.rebalance_portfolio(current_portfolio)
        
        assert len(rebalance_orders) == 3
        
        # Проверяем направления ордеров
        btc_order = next(order for order in rebalance_orders if order["symbol"] == "BTCUSDT")
        eth_order = next(order for order in rebalance_orders if order["symbol"] == "ETHUSDT")
        ada_order = next(order for order in rebalance_orders if order["symbol"] == "ADAUSDT")
        
        assert btc_order["side"] == "SELL"  # Уменьшаем позицию
        assert eth_order["side"] == "BUY"   # Увеличиваем позицию
        assert ada_order["side"] == "BUY"   # Открываем новую позицию

    @pytest.mark.asyncio
    async def test_stop_loss_take_profit_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow стоп-лосса и тейк-профита."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Размещаем основной ордер
        main_order_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "stop_loss": Decimal("44000.00"),    # 2.2% стоп-лосс
            "take_profit": Decimal("46000.00"),  # 2.2% тейк-профит
            "strategy_id": "sl_tp_test"
        }
        
        # Выполняем размещение с автоматическими SL/TP
        result = await trading_orchestrator.execute_trade_with_risk_management(main_order_signal)
        
        assert result["main_order"]["status"] == "success"
        assert result["stop_loss_order"]["status"] == "success"
        assert result["take_profit_order"]["status"] == "success"
        
        # Проверяем, что были размещены 3 ордера
        assert mock_exchange_client.place_limit_order.call_count >= 1
        assert mock_exchange_client.place_stop_order.call_count >= 2

    @pytest.mark.asyncio
    async def test_market_volatility_response_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow реакции на волатильность рынка."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Симулируем высокую волатильность
        volatile_market_data = {
            "symbol": "BTCUSDT",
            "price": Decimal("45000.00"),
            "volatility": Decimal("0.15"),  # 15% волатильность
            "volume_spike": True,
            "price_change_1m": Decimal("0.05")  # 5% изменение за минуту
        }
        
        # Система должна адаптировать параметры торговли
        adapted_params = await trading_orchestrator.adapt_to_market_conditions(volatile_market_data)
        
        assert adapted_params["position_size_multiplier"] < 1.0  # Уменьшенные позиции
        assert adapted_params["stop_loss_tighter"] is True       # Более жесткие стоп-лоссы
        assert adapted_params["execution_timeout"] < 30         # Быстрое исполнение
        
        # Размещаем ордер в условиях высокой волатильности
        volatile_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "market_conditions": "high_volatility"
        }
        
        result = await trading_orchestrator.execute_trade_signal(volatile_signal)
        
        # Ордер должен быть адаптирован под условия
        assert result["status"] == "success"
        assert result["adapted_for_volatility"] is True

    @pytest.mark.asyncio
    async def test_cross_exchange_arbitrage_workflow(
        self,
        trading_orchestrator: TradingOrchestrator
    ) -> None:
        """Тест workflow кросс-биржевого арбитража."""
        
        # Создаем моки для нескольких бирж
        binance_client = AsyncMock(spec=BybitClient)
        bybit_client = AsyncMock(spec=BybitClient)
        
        # Настраиваем разные цены на биржах
        binance_client.get_ticker.return_value = {
            "symbol": "BTCUSDT",
            "price": Decimal("45000.00"),
            "bid": Decimal("44999.50"),
            "ask": Decimal("45000.50")
        }
        
        bybit_client.get_ticker.return_value = {
            "symbol": "BTCUSDT",
            "price": Decimal("45050.00"),
            "bid": Decimal("45049.50"),
            "ask": Decimal("45050.50")
        }
        
        trading_orchestrator.add_exchange_client("binance", binance_client)
        trading_orchestrator.add_exchange_client("bybit", bybit_client)
        
        # Обнаруживаем арбитражную возможность
        arbitrage_opportunity = await trading_orchestrator.detect_arbitrage_opportunity("BTCUSDT")
        
        assert arbitrage_opportunity["profit_percentage"] > Decimal("0.001")
        assert arbitrage_opportunity["buy_exchange"] == "binance"
        assert arbitrage_opportunity["sell_exchange"] == "bybit"
        
        # Выполняем арбитражную сделку
        arbitrage_result = await trading_orchestrator.execute_arbitrage_trade(arbitrage_opportunity)
        
        assert arbitrage_result["status"] == "success"
        assert arbitrage_result["buy_order"]["exchange"] == "binance"
        assert arbitrage_result["sell_order"]["exchange"] == "bybit"

    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(
        self,
        trading_orchestrator: TradingOrchestrator,
        mock_exchange_client: BybitClient
    ) -> None:
        """Тест workflow мониторинга производительности."""
        
        trading_orchestrator.add_exchange_client("bybit", mock_exchange_client)
        
        # Выполняем серию торговых операций
        signals = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": Decimal("0.001"),
                "strategy_id": "perf_test_1"
            },
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": Decimal("0.1"),
                "strategy_id": "perf_test_2"
            }
        ]
        
        # Измеряем производительность
        performance_metrics = await trading_orchestrator.execute_with_performance_monitoring(signals)
        
        assert "total_execution_time" in performance_metrics
        assert "average_order_latency" in performance_metrics
        assert "successful_orders_ratio" in performance_metrics
        assert "throughput_orders_per_second" in performance_metrics
        
        # Проверяем качество метрик
        assert performance_metrics["total_execution_time"] > 0
        assert 0 <= performance_metrics["successful_orders_ratio"] <= 1
        assert performance_metrics["throughput_orders_per_second"] > 0