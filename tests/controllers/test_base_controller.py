from datetime import datetime

import pytest

from core.controllers.base import BaseController
from core.models import SystemState


@pytest.fixture
def base_controller():
    return BaseController()


def test_init(base_controller):
    """Тест инициализации контроллера"""
    assert isinstance(base_controller.state, SystemState)
    assert isinstance(base_controller.config, dict)
    assert isinstance(base_controller.trading_pairs, dict)
    assert isinstance(base_controller.active_orders, dict)
    assert isinstance(base_controller.positions, dict)
    assert isinstance(base_controller.decision_history, list)


@pytest.mark.asyncio
async def test_start(base_controller):
    """Тест запуска контроллера"""
    await base_controller.start()
    assert base_controller.state.is_running is True
    assert isinstance(base_controller.state.last_update, datetime)


@pytest.mark.asyncio
async def test_stop(base_controller):
    """Тест остановки контроллера"""
    await base_controller.start()
    await base_controller.stop()
    assert base_controller.state.is_running is False
    assert isinstance(base_controller.state.last_update, datetime)


@pytest.mark.asyncio
async def test_update_state(base_controller):
    """Тест обновления состояния"""
    await base_controller.update_state()
    assert isinstance(base_controller.state.last_update, datetime)
