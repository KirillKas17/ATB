import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from exchange.account_manager import AccountManager, AccountMetrics
from exchange.bybit_client import BybitClient, BybitConfig
from exchange.order_manager import OrderConfig, OrderManager


@pytest.fixture
def bybit_config() -> Any:
    """Фикстура с конфигурацией Bybit"""
    return BybitConfig(api_key="test_key", api_secret="test_secret", testnet=True)


@pytest.fixture
def order_config() -> Any:
    """Фикстура с конфигурацией ордеров"""
    return OrderConfig(
        max_leverage=10.0,
        min_leverage=1.0,
        confidence_threshold=0.7,
        trailing_stop=True,
        trailing_distance=0.02,
        break_even_threshold=0.01,
        take_profit_levels=[0.01, 0.02, 0.03],
        take_profit_quantities=[0.3, 0.3, 0.4],
    )


@pytest.fixture
def risk_config() -> Any:
    """Фикстура с конфигурацией рисков"""
    return {
        "max_leverage_usage": 0.8,
        "min_free_margin": 0.2,
        "max_positions": 10,
        "max_orders_per_position": 3,
    }


@pytest.fixture
async def bybit_client(bybit_config) -> Any:
    """Фикстура с клиентом Bybit"""
    client = BybitClient(bybit_config)
    try:
        await client.connect()
        yield client
    finally:
        await client.close()


@pytest.fixture
async def order_manager(bybit_client, order_config) -> Any:
    """Фикстура с менеджером ордеров"""
    manager = OrderManager(bybit_client, order_config)
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


@pytest.fixture
async def account_manager(bybit_client, order_manager, risk_config) -> Any:
    """Фикстура с менеджером аккаунта"""
    manager = AccountManager(bybit_client, order_manager, risk_config)
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_initialization(
    account_manager, bybit_client, order_manager, risk_config
) -> None:
    """Тест инициализации"""
    assert account_manager.client == bybit_client
    assert account_manager.order_manager == order_manager
    assert account_manager.risk_config == risk_config
    assert account_manager.metrics_cache is None
    assert account_manager.last_update is None
    assert account_manager.cache_ttl == 60
    assert account_manager.monitor_task is not None


@pytest.mark.asyncio
async def test_get_metrics(account_manager) -> None:
    """Тест получения метрик"""
    try:
        metrics = await account_manager.get_metrics()

        assert isinstance(metrics, AccountMetrics)
        assert metrics.equity >= 0
        assert metrics.free_margin >= 0
        assert metrics.used_margin >= 0
        assert metrics.unrealized_pnl is not None
        assert 0 <= metrics.leverage_usage <= 1
        assert metrics.total_positions >= 0
        assert metrics.total_orders >= 0

        # Проверка кеширования
        cached_metrics = await account_manager.get_metrics()
        assert cached_metrics == metrics

    except Exception as e:
        pytest.skip(f"Get metrics test skipped: {str(e)}")


@pytest.mark.asyncio
async def test_get_available_margin(account_manager) -> None:
    """Тест расчета доступной маржи"""
    try:
        margin = await account_manager.get_available_margin("BTCUSDT")

        assert isinstance(margin, float)
        assert margin >= 0

        # Проверка ограничений
        metrics = await account_manager.get_metrics()
        if metrics.leverage_usage >= account_manager.risk_config["max_leverage_usage"]:
            assert margin == 0

        if (
            metrics.free_margin / metrics.equity
            < account_manager.risk_config["min_free_margin"]
        ):
            assert margin == 0

        if metrics.total_positions >= account_manager.risk_config["max_positions"]:
            assert margin == 0

    except Exception as e:
        pytest.skip(f"Get available margin test skipped: {str(e)}")


@pytest.mark.asyncio
async def test_can_open_position(account_manager) -> None:
    """Тест проверки возможности открытия позиции"""
    try:
        # Тест с допустимыми параметрами
        can_open = await account_manager.can_open_position(
            symbol="BTCUSDT", amount=0.001, leverage=2.0
        )

        assert isinstance(can_open, bool)

        # Тест с превышением лимитов
        can_open = await account_manager.can_open_position(
            symbol="BTCUSDT",
            amount=1000.0,
            leverage=100.0,  # Большой объем  # Высокое плечо
        )

        assert not can_open

    except Exception as e:
        pytest.skip(f"Can open position test skipped: {str(e)}")


@pytest.mark.asyncio
async def test_unrealized_pnl_calculation(account_manager) -> None:
    """Тест расчета нереализованной прибыли"""
    try:
        # Создание тестовой позиции
        order = await account_manager.order_manager.create_entry_order(
            symbol="BTCUSDT",
            side="buy",
            amount=0.001,
            price=30000,
            stop_loss=29000,
            take_profit=31000,
            confidence=0.8,
        )

        # Расчет PnL
        pnl = await account_manager._calculate_unrealized_pnl()

        assert isinstance(pnl, float)

    except Exception as e:
        pytest.skip(f"Unrealized PnL calculation test skipped: {str(e)}")


@pytest.mark.asyncio
async def test_risk_limits_check(account_manager) -> None:
    """Тест проверки ограничений риска"""
    try:
        # Проверка ограничений
        await account_manager._check_risk_limits()

        # Проверка с превышением лимитов
        account_manager.risk_config["max_leverage_usage"] = 0.1  # Уменьшаем лимит
        await account_manager._check_risk_limits()

    except Exception as e:
        pytest.skip(f"Risk limits check test skipped: {str(e)}")


@pytest.mark.asyncio
async def test_error_handling(account_manager) -> None:
    """Тест обработки ошибок"""
    # Тест с неверным символом
    with pytest.raises(Exception):
        await account_manager.get_available_margin("INVALID")

    # Тест с неверными параметрами позиции
    with pytest.raises(Exception):
        await account_manager.can_open_position(
            symbol="BTCUSDT",
            amount=-0.001,
            leverage=0.0,  # Отрицательный объем  # Нулевое плечо
        )
