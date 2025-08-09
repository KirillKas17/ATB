#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для DDD архитектуры Syntra
"""

from decimal import Decimal
from uuid import uuid4

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

# Импорты сервисов приложения
from application.services.trading_service import TradingService
from domain.entities.portfolio_fixed import Balance, Portfolio, Position
from domain.entities.risk import RiskLevel, RiskManager, RiskProfile, RiskType
from domain.entities.strategy import Signal, SignalStrength, SignalType, Strategy, StrategyType

# Импорты доменных сущностей
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.repositories.portfolio_repository import InMemoryPortfolioRepository

# Импорты репозиториев
from domain.repositories.trading_repository import InMemoryTradingRepository
from domain.value_objects.currency import Currency

# Импорты общих компонентов
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from shared.exceptions import InsufficientFundsError


class TestDomainEntities:
    """Тесты доменных сущностей"""

    def test_order_creation(self: "TestDomainEntities") -> None:
        """Тест создания ордера"""
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        assert order.trading_pair == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity.value == Decimal("0.001")
        assert order.price.value == Decimal("50000")
        assert order.status == OrderStatus.PENDING
        assert order.is_active is True

    def test_order_fill(self: "TestDomainEntities") -> None:
        """Тест заполнения ордера"""
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )

        # Заполнение частично
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50000")))

        assert order.filled_quantity.value == Decimal("0.0005")
        assert order.average_price.value == Decimal("50000")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity.value == Decimal("0.0005")

        # Заполнение полностью
        order.fill(Volume(Decimal("0.0005")), Price(Decimal("50100")))

        assert order.filled_quantity.value == Decimal("0.001")
        assert order.average_price.value == Decimal("50050")  # Средняя цена
        assert order.status == OrderStatus.FILLED
        assert order.remaining_quantity.value == Decimal("0")

    def test_order_cancel(self: "TestDomainEntities") -> None:
        """Тест отмены ордера"""
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        order.cancel()

        assert order.status == OrderStatus.CANCELLED
        assert order.is_active is False

    def test_position_creation(self: "TestDomainEntities") -> None:
        """Тест создания позиции"""
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )

        assert position.trading_pair == "BTCUSDT"
        assert position.side == "long"
        assert position.quantity.value == Decimal("0.001")
        assert position.average_price.value == Decimal("50000")
        assert position.current_price.value == Decimal("51000")
        assert position.is_open is True

    def test_position_pnl_calculation(self: "TestDomainEntities") -> None:
        """Тест расчета PnL позиции"""
        # Длинная позиция с прибылью
        long_position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )

        long_position._calculate_unrealized_pnl()
        assert long_position.unrealized_pnl.value == Decimal("10")  # (51000-50000)*0.001

        # Короткая позиция с убытком
        short_position = Position(
            trading_pair="BTCUSDT",
            side="short",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )

        short_position._calculate_unrealized_pnl()
        assert short_position.unrealized_pnl.value == Decimal("-10")  # (50000-51000)*0.001

    def test_portfolio_creation(self: "TestDomainEntities") -> None:
        """Тест создания портфеля"""
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
            free_margin=Money(Decimal("10000"), Currency.USD),
        )

        assert portfolio.account_id == "test_account"
        assert portfolio.total_equity.value == Decimal("10000")
        assert portfolio.free_margin.value == Decimal("10000")

    def test_risk_manager_creation(self: "TestDomainEntities") -> None:
        """Тест создания менеджера рисков"""
        risk_profile = RiskProfile(
            name="Test Profile",
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
            max_drawdown=Percentage(Decimal("20")),
            max_leverage=Decimal("3"),
        )

        risk_manager = RiskManager(risk_profile=risk_profile)

        assert risk_manager.risk_profile.name == "Test Profile"
        assert risk_manager.risk_profile.max_position_size.value == Decimal("1000")

    def test_strategy_creation(self: "TestDomainEntities") -> None:
        """Тест создания стратегии"""
        strategy = Strategy(
            name="Test Strategy",
            description="Test Description",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTCUSDT", "ETHUSDT"],
        )

        assert strategy.name == "Test Strategy"
        assert strategy.strategy_type == StrategyType.TREND_FOLLOWING
        assert "BTCUSDT" in strategy.trading_pairs
        assert strategy.status.value == "active"

    def test_signal_creation(self: "TestDomainEntities") -> None:
        """Тест создания сигнала"""
        signal = Signal(
            strategy_id=uuid4(),
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.8"),
            price=Price(Decimal("50000")),
            quantity=Decimal("0.001"),
        )

        assert signal.trading_pair == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == Decimal("0.8")
        assert signal.is_actionable is True


class TestRepositories:
    """Тесты репозиториев"""

    @pytest.mark.asyncio
    async def test_trading_repository(self: "TestRepositories") -> None:
        """Тест репозитория торговых операций"""
        repo = InMemoryTradingRepository()

        # Создание ордера
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )

        # Сохранение
        saved_order = await repo.save_order(order)
        assert saved_order.id == order.id

        # Получение
        retrieved_order = await repo.get_order(order.id)
        assert retrieved_order is not None
        assert retrieved_order.trading_pair == "BTCUSDT"

        # Получение активных ордеров
        active_orders = await repo.get_active_orders()
        assert len(active_orders) == 1
        assert active_orders[0].id == order.id

        # Обновление
        order.fill(Volume(Decimal("0.001")), Price(Decimal("50000")))
        updated_order = await repo.update_order(order)
        assert updated_order.status == OrderStatus.FILLED

        # Удаление
        deleted = await repo.delete_order(order.id)
        assert deleted is True

        retrieved_order = await repo.get_order(order.id)
        assert retrieved_order is None

    @pytest.mark.asyncio
    async def test_portfolio_repository(self: "TestRepositories") -> None:
        """Тест репозитория портфеля"""
        repo = InMemoryPortfolioRepository()

        # Создание портфеля
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
        )

        # Сохранение портфеля
        saved_portfolio = await repo.save_portfolio(portfolio)
        assert saved_portfolio.id == portfolio.id

        # Получение портфеля
        retrieved_portfolio = await repo.get_portfolio(portfolio.id)
        assert retrieved_portfolio is not None
        assert retrieved_portfolio.account_id == "test_account"

        # Получение по аккаунту
        account_portfolio = await repo.get_portfolio_by_account("test_account")
        assert account_portfolio is not None
        assert account_portfolio.id == portfolio.id

        # Создание позиции
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
        )

        # Сохранение позиции
        saved_position = await repo.save_position(portfolio.id, position)
        assert saved_position.trading_pair == "BTCUSDT"

        # Получение позиции
        retrieved_position = await repo.get_position(portfolio.id, "BTCUSDT")
        assert retrieved_position is not None
        assert retrieved_position.side == "long"

        # Получение всех позиций
        all_positions = await repo.get_all_positions(portfolio.id)
        assert len(all_positions) == 1

        # Получение открытых позиций
        open_positions = await repo.get_open_positions(portfolio.id)
        assert len(open_positions) == 1

        # Создание баланса
        balance = Balance(
            currency=Currency.USD,
            available=Money(Decimal("8000"), Currency.USD),
            total=Money(Decimal("10000"), Currency.USD),
        )

        # Сохранение баланса
        saved_balance = await repo.save_balance(portfolio.id, balance)
        assert saved_balance.currency == Currency.USD

        # Получение баланса
        retrieved_balance = await repo.get_balance(portfolio.id, Currency.USD)
        assert retrieved_balance is not None
        assert retrieved_balance.available.value == Decimal("8000")


class TestApplicationServices:
    """Тесты сервисов приложения"""

    @pytest.mark.asyncio
    async def test_trading_service_creation(self: "TestApplicationServices") -> None:
        """Тест создания торгового сервиса"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        assert service is not None
        assert service.trading_repository == trading_repo
        assert service.portfolio_repository == portfolio_repo
        assert service.risk_manager == risk_manager

    @pytest.mark.asyncio
    async def test_create_order_success(self: "TestApplicationServices") -> None:
        """Тест успешного создания ордера"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        # Создание портфеля с достаточными средствами
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
            free_margin=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_portfolio(portfolio)

        # Добавление баланса
        balance = Balance(
            currency=Currency.USD,
            available=Money(Decimal("10000"), Currency.USD),
            total=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_balance(portfolio.id, balance)

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        # Создание ордера
        order = await service.create_order(
            portfolio_id=portfolio.id,
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        assert order is not None
        assert order.trading_pair == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity.value == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_create_order_insufficient_funds(self: "TestApplicationServices") -> None:
        """Тест создания ордера с недостаточными средствами"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        # Создание портфеля с недостаточными средствами
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("100"), Currency.USD),
            free_margin=Money(Decimal("100"), Currency.USD),
        )
        await portfolio_repo.save_portfolio(portfolio)

        # Добавление баланса
        balance = Balance(
            currency=Currency.USD,
            available=Money(Decimal("100"), Currency.USD),
            total=Money(Decimal("100"), Currency.USD),
        )
        await portfolio_repo.save_balance(portfolio.id, balance)

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        # Попытка создать ордер с недостаточными средствами
        with pytest.raises(InsufficientFundsError):
            await service.create_order(
                portfolio_id=portfolio.id,
                trading_pair="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Volume(Decimal("0.001")),
                price=Price(Decimal("50000")),
            )

    @pytest.mark.asyncio
    async def test_execute_order(self: "TestApplicationServices") -> None:
        """Тест исполнения ордера"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        # Создание портфеля
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_portfolio(portfolio)

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        # Создание ордера
        order = await service.create_order(
            portfolio_id=portfolio.id,
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Исполнение ордера
        trade = await service.execute_order(
            order_id=order.id,
            execution_price=Price(Decimal("50000")),
            execution_quantity=Volume(Decimal("0.001")),
        )

        assert trade is not None
        assert trade.order_id == order.id
        assert trade.trading_pair == "BTCUSDT"
        assert trade.quantity.value == Decimal("0.001")
        assert trade.price.value == Decimal("50000")

        # Проверка статуса ордера
        updated_order = await trading_repo.get_order(order.id)
        assert updated_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_order(self: "TestApplicationServices") -> None:
        """Тест отмены ордера"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        # Создание портфеля
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_portfolio(portfolio)

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        # Создание ордера
        order = await service.create_order(
            portfolio_id=portfolio.id,
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Отмена ордера
        cancelled_order = await service.cancel_order(order.id)

        assert cancelled_order.status == OrderStatus.CANCELLED
        assert cancelled_order.is_active is False

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self: "TestApplicationServices") -> None:
        """Тест получения сводки портфеля"""
        trading_repo = InMemoryTradingRepository()
        portfolio_repo = InMemoryPortfolioRepository()

        # Создание портфеля
        portfolio = Portfolio(
            account_id="test_account",
            total_equity=Money(Decimal("10000"), Currency.USD),
            free_margin=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_portfolio(portfolio)

        # Добавление баланса
        balance = Balance(
            currency=Currency.USD,
            available=Money(Decimal("8000"), Currency.USD),
            total=Money(Decimal("10000"), Currency.USD),
        )
        await portfolio_repo.save_balance(portfolio.id, balance)

        # Добавление позиции
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )
        await portfolio_repo.save_position(portfolio.id, position)

        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )
        risk_manager = RiskManager(risk_profile=risk_profile)

        service = TradingService(
            trading_repository=trading_repo,
            portfolio_repository=portfolio_repo,
            risk_manager=risk_manager,
        )

        # Получение сводки
        summary = await service.get_portfolio_summary(portfolio.id)

        assert summary is not None
        assert summary["portfolio_id"] == str(portfolio.id)
        assert summary["total_equity"] == "10000"
        assert summary["open_positions_count"] == 1
        assert len(summary["balances"]) == 1
        assert len(summary["positions"]) == 1


class TestRiskManagement:
    """Тесты управления рисками"""

    def test_risk_limit_creation(self: "TestRiskManagement") -> None:
        """Тест создания лимита риска"""
        limit = RiskLimit(
            risk_type=RiskType.POSITION_SIZE,
            name="Position Size Limit",
            max_value=Decimal("1000"),
            warning_threshold=Decimal("800"),
            critical_threshold=Decimal("900"),
        )

        assert limit.risk_type == RiskType.POSITION_SIZE
        assert limit.max_value == Decimal("1000")
        assert limit.warning_threshold == Decimal("800")
        assert limit.critical_threshold == Decimal("900")

    def test_risk_limit_violation_check(self: "TestRiskManagement") -> None:
        """Тест проверки нарушения лимита риска"""
        limit = RiskLimit(
            risk_type=RiskType.POSITION_SIZE,
            name="Position Size Limit",
            max_value=Decimal("1000"),
            warning_threshold=Decimal("800"),
            critical_threshold=Decimal("900"),
        )

        # Нет нарушения
        limit.update_current_value(Decimal("500"))
        violation = limit.check_violation()
        assert violation is None

        # Предупреждение
        limit.update_current_value(Decimal("850"))
        violation = limit.check_violation()
        assert violation == RiskLevel.HIGH

        # Критическое нарушение
        limit.update_current_value(Decimal("950"))
        violation = limit.check_violation()
        assert violation == RiskLevel.CRITICAL

    def test_risk_manager_violations(self: "TestRiskManagement") -> None:
        """Тест нарушений в менеджере рисков"""
        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )

        risk_manager = RiskManager(risk_profile=risk_profile)

        # Добавление лимита
        limit = RiskLimit(
            risk_type=RiskType.POSITION_SIZE,
            name="Position Size Limit",
            max_value=Decimal("1000"),
            warning_threshold=Decimal("800"),
            critical_threshold=Decimal("900"),
        )
        risk_manager.add_limit(limit)

        # Проверка нарушений
        limit.update_current_value(Decimal("950"))
        violations = risk_manager.check_all_limits()

        assert len(violations) == 1
        assert violations[0]["risk_level"] == "critical"
        assert violations[0]["current_value"] == 950.0

    def test_risk_manager_trade_validation(self: "TestRiskManagement") -> None:
        """Тест валидации сделки в менеджере рисков"""
        risk_profile = RiskProfile(
            max_position_size=Money(Decimal("1000"), Currency.USD),
            max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        )

        risk_manager = RiskManager(risk_profile=risk_profile)

        # Добавление лимита
        limit = RiskLimit(
            risk_type=RiskType.POSITION_SIZE,
            name="Position Size Limit",
            max_value=Decimal("1000"),
            warning_threshold=Decimal("800"),
            critical_threshold=Decimal("900"),
        )
        risk_manager.add_limit(limit)

        # Установка текущего значения
        limit.update_current_value(Decimal("950"))

        # Проверка торговых параметров
        trade_params = {
            "position_size": 100,  # Добавит 100 к текущим 950
            "trading_pair": "BTCUSDT",
            "side": "buy",
        }

        result = risk_manager.should_allow_trade(trade_params)

        assert result["allowed"] is False
        assert "risk limit violation" in result["reasons"][0]
        assert result["risk_level"] == "critical"


class TestValueObjects:
    """Тесты объектов-значений"""

    def test_money_operations(self: "TestValueObjects") -> None:
        """Тест операций с деньгами"""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("200"), Currency.USD)

        # Сложение
        result = money1 + money2
        assert result.value == Decimal("300")
        assert result.currency == Currency.USD

        # Вычитание
        result = money2 - money1
        assert result.value == Decimal("100")

        # Умножение
        result = money1 * 2
        assert result.value == Decimal("200")

    def test_price_operations(self: "TestValueObjects") -> None:
        """Тест операций с ценой"""
        price = Price(Decimal("50000"))
        volume = Volume(Decimal("0.001"))

        # Умножение цены на объем
        result = price * volume
        assert result.value == Decimal("50")
        assert result.currency == Currency.USD

    def test_volume_operations(self: "TestValueObjects") -> None:
        """Тест операций с объемом"""
        volume1 = Volume(Decimal("0.001"))
        volume2 = Volume(Decimal("0.002"))

        # Сложение
        result = volume1 + volume2
        assert result.value == Decimal("0.003")

        # Вычитание
        result = volume2 - volume1
        assert result.value == Decimal("0.001")

    def test_percentage_operations(self: "TestValueObjects") -> None:
        """Тест операций с процентами"""
        percentage = Percentage(Decimal("20"))

        # Получение десятичного значения
        assert percentage.decimal_value == Decimal("0.2")

        # Создание из десятичного значения
        percentage2 = Percentage.from_decimal(Decimal("0.15"))
        assert percentage2.value == Decimal("15")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
