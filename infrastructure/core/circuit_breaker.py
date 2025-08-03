# -*- coding: utf-8 -*-
import asyncio
from enum import Enum
from typing import Any, Callable, Dict, Optional

from loguru import logger

from infrastructure.messaging.optimized_event_bus import (
    EventBus,
    Event,
    EventType,
    EventPriority,
    EventMetadata,
)


class CircuitState(Enum):
    CLOSED = "closed"  # Нормальная работа
    OPEN = "open"  # Блокировка операций
    HALF_OPEN = "half_open"  # Тестирование восстановления


class CircuitBreaker:
    """
    Circuit Breaker - защита системы от каскадных сбоев:
    - Автоматическое отключение при ошибках
    - Постепенное восстановление
    - Мониторинг состояния
    - Уведомления о сбоях
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # Состояния circuit breaker для разных компонентов
        self.circuits = {
            "exchange": {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None,
                "threshold": 5,  # Количество ошибок для открытия
                "timeout": 300,  # Время в открытом состоянии (секунды)
                "success_threshold": 3,  # Успешных операций для полуоткрытия
            },
            "database": {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None,
                "threshold": 3,
                "timeout": 180,
                "success_threshold": 2,
            },
            "api": {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None,
                "threshold": 10,
                "timeout": 60,
                "success_threshold": 5,
            },
            "general": {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None,
                "threshold": 20,
                "timeout": 600,
                "success_threshold": 10,
            },
        }

        # Callback функции для уведомлений
        self.callbacks: Dict[str, Callable] = {}

        # Запуск мониторинга
        self._start_monitoring()

    async def trigger(self, circuit_name: str, error: Optional[Exception] = None) -> None:
        """Активация circuit breaker"""
        if circuit_name not in self.circuits:
            logger.warning(f"Unknown circuit: {circuit_name}")
            return

        circuit = self.circuits[circuit_name]

        # Увеличение счетчика ошибок
        failure_count = circuit["failure_count"]
        if not isinstance(failure_count, int):
            try:
                if failure_count is not None and hasattr(failure_count, '__int__'):
                    circuit["failure_count"] = int(failure_count)
                else:
                    circuit["failure_count"] = 0
            except (ValueError, TypeError):
                circuit["failure_count"] = 0
        else:
            circuit["failure_count"] = failure_count + 1
        circuit["last_failure_time"] = asyncio.get_event_loop().time()

        logger.warning(
            f"Circuit {circuit_name} failure #{circuit['failure_count']}: {error}"
        )

        # Проверка необходимости открытия circuit breaker
        if (
            isinstance(circuit["failure_count"], int)
            and isinstance(circuit["threshold"], int)
            and circuit["failure_count"] >= circuit["threshold"]
        ) and circuit["state"] == CircuitState.CLOSED:

            await self._open_circuit(circuit_name)

    async def _open_circuit(self, circuit_name: str) -> None:
        """Открытие circuit breaker"""
        circuit = self.circuits[circuit_name]
        circuit["state"] = CircuitState.OPEN

        logger.critical(f"Circuit {circuit_name} OPENED - blocking operations")

        # Отправка уведомления
        await self.event_bus.publish(
            Event(
                type=EventType.SYSTEM,
                data={
                    "circuit": circuit_name,
                    "failure_count": circuit["failure_count"],
                    "threshold": circuit["threshold"],
                },
                metadata=EventMetadata(
                    event_id=f"circuit_{circuit_name}_opened",
                    timestamp=asyncio.get_event_loop().time(),
                    priority=EventPriority.CRITICAL,
                    source="circuit_breaker",
                ),
            )
        )

        # Вызов callback функции
        if circuit_name in self.callbacks:
            try:
                await self.callbacks[circuit_name]("opened")
            except Exception as e:
                logger.error(f"Error in circuit callback: {e}")

        # Планирование перехода в полуоткрытое состояние
        asyncio.create_task(self._schedule_half_open(circuit_name))

    async def _schedule_half_open(self, circuit_name: str) -> None:
        """Планирование перехода в полуоткрытое состояние"""
        circuit = self.circuits[circuit_name]
        timeout = circuit["timeout"]

        if not isinstance(timeout, (int, float)):
            timeout = 600.0  # Значение по умолчанию

        await asyncio.sleep(float(timeout))

        # Проверка, что circuit все еще открыт
        if circuit["state"] == CircuitState.OPEN:
            await self._half_open_circuit(circuit_name)

    async def _half_open_circuit(self, circuit_name: str) -> None:
        """Переход в полуоткрытое состояние"""
        circuit = self.circuits[circuit_name]
        circuit["state"] = CircuitState.HALF_OPEN

        logger.info(f"Circuit {circuit_name} HALF-OPEN - testing recovery")

        # Отправка уведомления
        await self.event_bus.publish(
            Event(
                type=EventType.SYSTEM,
                data={"circuit": circuit_name},
                metadata=EventMetadata(
                    event_id=f"circuit_{circuit_name}_half_open",
                    timestamp=asyncio.get_event_loop().time(),
                    priority=EventPriority.HIGH,
                    source="circuit_breaker",
                ),
            )
        )

    async def success(self, circuit_name: str) -> None:
        """Успешная операция"""
        if circuit_name not in self.circuits:
            return

        circuit = self.circuits[circuit_name]

        if circuit["state"] == CircuitState.HALF_OPEN:
            # В полуоткрытом состоянии - проверяем успешность
            success_count = circuit.get("success_count")
            if not isinstance(success_count, int):
                try:
                    if success_count is not None and hasattr(success_count, '__int__'):
                        circuit["success_count"] = int(success_count)
                    else:
                        circuit["success_count"] = 0
                except (ValueError, TypeError):
                    circuit["success_count"] = 0
            else:
                circuit["success_count"] = success_count + 1

            if (
                isinstance(circuit["success_count"], int)
                and isinstance(circuit["success_threshold"], int)
                and circuit["success_count"] >= circuit["success_threshold"]
            ):
                await self._close_circuit(circuit_name)
        elif circuit["state"] == CircuitState.CLOSED:
            # В закрытом состоянии - сбрасываем счетчик ошибок
            circuit["failure_count"] = 0

    async def _close_circuit(self, circuit_name: str) -> None:
        """Закрытие circuit breaker"""
        circuit = self.circuits[circuit_name]
        circuit["state"] = CircuitState.CLOSED
        circuit["failure_count"] = 0
        circuit["success_count"] = 0
        circuit["last_failure_time"] = None

        logger.info(f"Circuit {circuit_name} CLOSED - normal operation resumed")

        # Отправка уведомления
        await self.event_bus.publish(
            Event(
                type=EventType.SYSTEM,
                data={"circuit": circuit_name},
                metadata=EventMetadata(
                    event_id=f"circuit_{circuit_name}_closed",
                    timestamp=asyncio.get_event_loop().time(),
                    priority=EventPriority.NORMAL,
                    source="circuit_breaker",
                ),
            )
        )

    async def is_open(self, circuit_name: str) -> bool:
        """Проверка, открыт ли circuit breaker"""
        if circuit_name not in self.circuits:
            return False

        return self.circuits[circuit_name]["state"] == CircuitState.OPEN

    async def is_half_open(self, circuit_name: str) -> bool:
        """Проверка, полуоткрыт ли circuit breaker"""
        if circuit_name not in self.circuits:
            return False

        return self.circuits[circuit_name]["state"] == CircuitState.HALF_OPEN

    async def can_execute(self, circuit_name: str) -> bool:
        """Проверка возможности выполнения операции"""
        if circuit_name not in self.circuits:
            return True

        circuit = self.circuits[circuit_name]

        if circuit["state"] == CircuitState.CLOSED:
            return True
        elif circuit["state"] == CircuitState.HALF_OPEN:
            # В полуоткрытом состоянии разрешаем ограниченное количество операций
            return True
        else:  # OPEN
            return False

    def register_callback(self, circuit_name: str, callback: Callable) -> None:
        """Регистрация callback функции"""
        self.callbacks[circuit_name] = callback
        logger.info(f"Registered callback for circuit: {circuit_name}")

    async def _start_monitoring(self) -> None:
        """Запуск мониторинга circuit breakers"""
        logger.info("Starting circuit breaker monitoring")

        while True:
            try:
                await self._monitor_circuits()
                await asyncio.sleep(30.0)  # Проверка каждые 30 секунд
            except Exception as e:
                logger.error(f"Error in circuit monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_circuits(self) -> None:
        """Мониторинг состояния circuit breakers"""
        for circuit_name, circuit in self.circuits.items():
            if circuit["state"] == CircuitState.OPEN:
                # Проверяем, не пора ли переходить в полуоткрытое состояние
                last_failure_time = circuit["last_failure_time"]
                if last_failure_time is not None:
                    current_time = asyncio.get_event_loop().time()
                    timeout = circuit["timeout"]
                    if isinstance(timeout, (int, float)) and current_time - last_failure_time >= timeout:
                        await self._half_open_circuit(circuit_name)

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса всех circuit breakers"""
        status = {}
        for circuit_name, circuit in self.circuits.items():
            state_value = circuit["state"].value if hasattr(circuit["state"], "value") else str(circuit["state"])
            status[circuit_name] = {
                "state": state_value,
                "failure_count": circuit["failure_count"],
                "threshold": circuit["threshold"],
                "timeout": circuit["timeout"],
                "last_failure_time": circuit["last_failure_time"],
            }
        return status

    async def reset_circuit(self, circuit_name: str) -> None:
        """Сброс circuit breaker"""
        if circuit_name not in self.circuits:
            logger.warning(f"Unknown circuit: {circuit_name}")
            return

        circuit = self.circuits[circuit_name]
        circuit["state"] = CircuitState.CLOSED
        circuit["failure_count"] = 0
        circuit["success_count"] = 0
        circuit["last_failure_time"] = None

        logger.info(f"Circuit {circuit_name} RESET")

        # Отправка уведомления
        await self.event_bus.publish(
            Event(
                type=EventType.SYSTEM,
                data={"circuit": circuit_name},
                metadata=EventMetadata(
                    event_id=f"circuit_{circuit_name}_reset",
                    timestamp=asyncio.get_event_loop().time(),
                    priority=EventPriority.NORMAL,
                    source="circuit_breaker",
                ),
            )
        )

    async def emergency_stop(self) -> None:
        """Экстренная остановка всех circuit breakers"""
        logger.critical("EMERGENCY STOP - opening all circuit breakers")

        for circuit_name in self.circuits:
            circuit = self.circuits[circuit_name]
            circuit["state"] = CircuitState.OPEN
            circuit["failure_count"] = circuit["threshold"] + 1

        # Отправка уведомления об экстренной остановке
        await self.event_bus.publish(
            Event(
                type=EventType.SYSTEM,
                data={"message": "Emergency stop activated"},
                metadata=EventMetadata(
                    event_id="emergency_stop",
                    timestamp=asyncio.get_event_loop().time(),
                    priority=EventPriority.CRITICAL,
                    source="circuit_breaker",
                ),
            )
        )
